"""Offline review of general approval versus approval with own content.

Usage: uv run cli.py general-approval-review

Open question (docs/stance_policy_v2.md, "Open question 2026-10-07"): is a
blanket approval ("Stimmt", "Super Idee") neutral or in_favor per key
message, and does approval followed by own content change that? This module
assembles the evidence for the annotator from saved artifacts only:

  - a diagnostic table with one row per (comment, key message) pair for every
    comment the model treated as general approval, claimed general support
    for without evidence, or the annotator noted as a general approving
    reaction (including noted comments on contributions in no model run);
  - a safe summary with IDs and counts only (no texts, notes, key messages,
    explanations or evaluator names);
  - a German Excel review pack for the annotator, and a reader that parses
    and validates her answers (deriving rule outcomes is a later round).

No inference and no network; .env is never loaded. Every input is only read
(the annotation database is opened read-only) and hashed before and after
use; outputs are created exclusively in a new directory and never
overwritten. Reference labels are never changed here.

The comment signals and the heuristic cause are descriptive triage whose
patterns were derived after inspecting this cohort; they are not a validated
classifier. Folds 1-4 are held-out data: nothing here may be used to tune a
rule or prompt on them.
"""
from __future__ import annotations

import hashlib
import html
import json
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
from openpyxl.styles import Alignment, Font, PatternFill, Protection
from openpyxl.utils import get_column_letter
from openpyxl.workbook.defined_name import DefinedName
from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl.writer.excel import ExcelWriter

import evaluator_v2
from dataset_builder import DEFAULT_EVAL_DB, PRIMARY_EVALUATOR, compute_pair_id
from fold_builder import load_folds

DIAGNOSTIC_SCHEMA_VERSION = "ga_diagnostic_v1"
PACK_SCHEMA_VERSION = "ga_review_pack_v1"
SIGNALS_VERSION = "ga_signals_v1"

EXPERIMENTS_DIR = Path("dataset/experiments")
OUTPUT_DIR_PREFIX = "general_approval_review_"
DIAGNOSTIC_FILE = "diagnostic_pairs.parquet"
SUMMARY_FILE = "general_approval_review.safe_summary.json"
PACK_FILE_TEMPLATE = "general_approval_review_pack_v1_{created_utc}.xlsx"
HASHES_FILE = "artifact_hashes.json"
CREATED_UTC_FORMAT = "%Y%m%dT%H%M%S%fZ"


@dataclass(frozen=True)
class RunSpec:
    """One saved model run: its predictions and its expected-pair cohort.

    Attributes:
        name: Unique short name of the run, used as run_name in all outputs.
        predictions: Parquet file with the run's per-pair predictions.
        expected: Parquet file listing every pair the run was meant to score
            (pair_id, contribution_id), including pairs without a prediction.
    """

    name: str
    predictions: Path
    expected: Path


DEFAULT_RUNS: tuple[RunSpec, ...] = (
    RunSpec(
        "fold0_optionc_prompt2",
        EXPERIMENTS_DIR / "fold0_gptoss-120b_optionc_prompt2_20261006_live.parquet",
        EXPERIMENTS_DIR / "fold0_gptoss-120b_optionc_prompt2_20261006_live.expected.parquet",
    ),
    RunSpec(
        "heldout_folds1-4",
        EXPERIMENTS_DIR / "heldout_folds1-4_gptoss-120b_20261006_live.parquet",
        EXPERIMENTS_DIR / "heldout_folds1-4_gptoss-120b_20261006_live.expected.parquet",
    ),
)
DEFAULT_SOURCES: tuple[Path, ...] = (
    EXPERIMENTS_DIR / "screening_subset_fold0_source.parquet",
    EXPERIMENTS_DIR / "heldout_folds1-4_source.parquet",
)
DEFAULT_ANNOTATIONS_DB = DEFAULT_EVAL_DB

GENERAL_APPROVAL = "general_approval"
GA_CLAIM_REVIEW_REASONS = frozenset({"general_support_missing_evidence", "stage1_general_support_missing_evidence"})
# The frozen source parquets cut contributionContent at this many characters.
CONTRIBUTION_TEXT_LIMIT = 1000
EXCEL_CELL_LIMIT = 32767
SHORT_COMMENT_CHARS = 200

HELDOUT_NOTE = (
    "Folds 1-4 are held-out evaluation data. Any rule or prompt change tuned on these numbers "
    "turns them into development data; reference labels are never overwritten here."
)

# --------------------------------------------------------------------------
# Annotation-note matchers
# --------------------------------------------------------------------------

# Broad: both terms anywhere in the note, in either order ("zustimmende generelle
# Reaktion" as well as "generelle zustimmende Reaktion").
GA_NOTE_PATTERN = (re.compile(r"zustimm", re.IGNORECASE), re.compile(r"generell|allgemein", re.IGNORECASE))
# Strict: the matcher behind the earlier "84 pairs" sentence in docs/stance_policy_v2.md.
GA_NOTE_STRICT_PATTERN = re.compile(r"(?:generell|allgemein)\w*\s+zustimmend", re.IGNORECASE)
GENERAL_REJECTION_NOTE_PATTERN = (re.compile(r"ablehnend", re.IGNORECASE),
                                  re.compile(r"generell|allgemein", re.IGNORECASE))
NOT_NEUTRAL_NOTE_PATTERN = re.compile(r"nicht\s+neutral", re.IGNORECASE)


def note_flags(note: object) -> dict[str, bool]:
    """Which annotator-note categories a free-text note matches.

    Args:
        note: The annotator's free-text note; anything that is not a string
            (None, NaN) is treated as an empty note.

    Returns:
        A dict with the boolean flags note_general_approval,
        note_general_approval_strict, note_general_rejection and
        note_not_neutral.
    """
    text = note if isinstance(note, str) else ""
    return {
        "note_general_approval": all(p.search(text) for p in GA_NOTE_PATTERN),
        "note_general_approval_strict": bool(GA_NOTE_STRICT_PATTERN.search(text)),
        "note_general_rejection": all(p.search(text) for p in GENERAL_REJECTION_NOTE_PATTERN),
        "note_not_neutral": bool(NOT_NEUTRAL_NOTE_PATTERN.search(text)),
    }


# --------------------------------------------------------------------------
# Comment signals (derived after inspecting this cohort: descriptive only)
# --------------------------------------------------------------------------

_APPROVAL_PREFIX_SOURCES = (
    r"stimmt(?: genau| absolut| total)?",
    r"(?:(?:volle|100 ?%) )?zustimmung",
    r"in der tat",
    r"warum (?:eigentlich )?nicht",
    r"du hast (?:völlig |vollkommen |absolut |so )?recht",
    r"ganz meine meinung",
    r"(?:super|gute|tolle|klasse|coole) idee",
    r"(?:das ist )?(?:ein )?(?:sehr |absolut )?(?:guter|toller|exzellenter|hervorragender|super) vorschlag"
    r"(?: aus meiner sicht)?",
    r"(?:dieser|diese|dieses|der|die|das) (?:vorschlag|idee) ist (?:sehr |echt |wirklich )?(?:super|gut|toll|klasse|prima)",
    r"ich finde (?:die|den|diese|diesen) (?:idee|vorschlag) (?:sehr |echt |wirklich )?(?:gut|super|toll|klasse|sinnvoll)",
    r"(?:ich stimme|wir stimmen)(?: \w+){0,5}? zu",
    r"(?:dem|diesem) vorschlag (?:kann ich|können wir) (?:nur )?zustimmen",
    r"(?:diesen|den) vorschlag (?:begrüße ich|begrüßen wir)(?: sehr)?",
    r"ich unterstütze (?:das|dies|diese|diesen|dieses|den|die)"
    r"(?: (?:forderung|vorschlag|idee|anliegen|beitrag|initiative|vorhaben))?",
    r"ich schließe mich(?: \w+){0,3}? an",
    r"das geht in die richtige richtung",
    r"(?:sehr|absolut|völlig|genau|so) richtig",
    r"richtiger hinweis",
    r"so wahr",
    r"this",
    r"gut(?:gut)*",
    r"danke,? ja",
    r"auf jeden fall(?: ja)?",
    r"ja+",
    r"klingt (?:smart|gut|super|toll|sinnvoll|vernünftig)",
    r"tolle aktion",
    r"das wäre ein riesenfortschritt",
)
# Each formula must start the (already left-stripped) clause; (?!\w) stops it from matching the
# beginning of a longer word, e.g. "ja" must not match "Jahr" and "this" must not match "thistle".
APPROVAL_PREFIX_PATTERNS = tuple(
    re.compile(rf"^(?:{source})(?!\w)", re.IGNORECASE) for source in _APPROVAL_PREFIX_SOURCES
)
# "ne{2,}" catches colloquial "nee"/"neee"; "kein\w*" covers keine/keinen/keiner/...
NEGATION_CUES = tuple(re.compile(p, re.IGNORECASE) for p in (r"\bnicht\b", r"\bne{2,}\b", r"\bkein\w*", r"\begal\b"))
# Laughter ("haha", "hahaha"), winking smileys ";)" / ";-)", and the words ironisch/sarkastisch.
IRONY_CUES = tuple(re.compile(p, re.IGNORECASE) for p in (r"\bha(?:ha)+\b", r";-?\)", r"\bironisch", r"\bsarkas"))
BACK_REFERENCE_CUES = tuple(re.compile(p, re.IGNORECASE) for p in (
    r"\bsiehe oben\b", r"\bs\. oben\b", r"\b(?:obigen?|vorherigen?) kommentar",
    r"\bschließe mich(?: \w+){0,3}? an\b", r"\bin der beschreibung\b",
))
GERMAN_STOPWORDS = frozenset("""
aber alle allen als also auch auf aus bei bis dann das dass dem den denn der des die dies diese diesem
diesen dieser dieses doch dort durch ein eine einem einen einer eines eben euch für gegen ganz gerade
gibt hab habe haben hat hatte hier ich ihm ihn ihnen ihr ihre ist kann kein keine können man mehr mein
meine meinen meiner mich mit muss müssen nach nicht noch nur oder schon sehr sein seine sich sie sind
soll sollte sollten über uns und unter von war wäre was weil wenn werden wer wie wir wird wurde zum zur
""".split())
MAX_BLANKET_RESIDUAL_WORDS = 2
STEM_CHARS = 5

SIGNAL_CLASSES = ("blanket_no_content", "approval_plus_content", "content_without_approval_phrase",
                  "negation_or_irony_cue", "back_reference")
HEURISTIC_CAUSES = ("match", "A_label_inconsistency_candidate", "B_model_error_candidate", "rule_dependent",
                    "not_scorable", "model_other_basis")
_CAUSE_BY_SIGNAL_CLASS = {
    "blanket_no_content": "A_label_inconsistency_candidate",
    "approval_plus_content": "rule_dependent",
    "content_without_approval_phrase": "B_model_error_candidate",
    "negation_or_irony_cue": "B_model_error_candidate",
    "back_reference": "B_model_error_candidate",
}


@dataclass(frozen=True)
class CommentSignals:
    """Surface signals of one comment after stripping approval formulas.

    Attributes:
        approval_prefix: The first approval formula found at the start of the
            comment, a sentence or a comma clause (as written), or None.
        residual_text: The comment without any stripped approval formula.
        residual_word_count: Number of letters-only tokens of at least 3
            characters in residual_text.
        residual_content_terms: Sorted crude stems (first STEM_CHARS letters)
            of residual tokens that are not stopwords.
        negation_cue: Whether a negation cue occurs in the residual text.
        irony_cue: Whether an irony cue occurs anywhere in the comment.
        back_reference_cue: Whether a back-reference cue ("siehe oben", ...)
            occurs anywhere in the comment.
        char_length: Length of the unescaped, whitespace-normalized comment.
    """

    approval_prefix: str | None
    residual_text: str
    residual_word_count: int
    residual_content_terms: tuple[str, ...]
    negation_cue: bool
    irony_cue: bool
    back_reference_cue: bool
    char_length: int


def _normalize(text: str) -> str:
    """Decode HTML entities and collapse all whitespace runs to single spaces.

    Args:
        text: Raw stored text (may contain entities such as "&amp;").

    Returns:
        The decoded text with single spaces and no leading/trailing space.
    """
    return " ".join(html.unescape(text).split())


def _letter_tokens(text: str) -> list[str]:
    """Letters-only tokens of at least 3 characters (digits and "_" split tokens).

    Args:
        text: Text to tokenize.

    Returns:
        The tokens in order of appearance, case preserved.
    """
    # [^\W\d_] is "a word character that is neither a digit nor an underscore",
    # i.e. any Unicode letter (including umlauts and ß).
    return [token for token in re.findall(r"[^\W\d_]+", text) if len(token) >= 3]


def content_terms(text: object) -> frozenset[str]:
    """Crude stems of the non-stopword tokens of a text.

    Args:
        text: Text to analyse; anything that is not a string yields no terms.

    Returns:
        Lower-cased stems (first STEM_CHARS letters) of the letters-only
        tokens that are not in GERMAN_STOPWORDS; empty for non-strings.
    """
    if not isinstance(text, str):
        return frozenset()
    tokens = (token.lower() for token in _letter_tokens(_normalize(text)))
    return frozenset(token[:STEM_CHARS] for token in tokens if token not in GERMAN_STOPWORDS)


def _strip_prefixes(clause: str) -> tuple[str, str | None]:
    """Repeatedly strip approval formulas from the start of one clause.

    Leading punctuation and spaces are removed before each attempt, so
    chained formulas such as "Stimmt, super Idee!" are all stripped.

    Args:
        clause: One sentence or comma clause.

    Returns:
        A tuple of the remaining text and the first formula stripped (as
        written), or None when the clause did not start with a formula.
    """
    first = None
    while True:
        # Drop leading non-word characters (punctuation, spaces, emoji) and underscores.
        clause = re.sub(r"^[\W_]+", "", clause)
        # First pattern (in _APPROVAL_PREFIX_SOURCES order) that matches; ":=" binds the match inside the generator.
        match = next((m for p in APPROVAL_PREFIX_PATTERNS if (m := p.match(clause))), None)
        # An empty match would never advance the clause, so it ends the loop like no match.
        if match is None or not match.group(0):
            return clause, first
        first = first or match.group(0)
        clause = clause[match.end():]


def compute_comment_signals(comment_text: str) -> CommentSignals:
    """Compute the descriptive approval/content signals of one comment (pure).

    Approval formulas are stripped at the start of each sentence and then at
    the start of each comma clause. Negation is looked for in the residual
    text only (so "Warum nicht!" is not a negation); irony and back-reference
    cues anywhere in the comment.

    Args:
        comment_text: The raw comment text as stored.

    Returns:
        The CommentSignals of the comment.
    """
    text = _normalize(comment_text)
    residual_clauses, prefix = [], None
    # Sentence boundaries are approximated by runs of . ! ? ; and :.
    for sentence in re.split(r"[.!?;:]+", text):
        remainder, found = _strip_prefixes(sentence.strip())
        prefix = prefix or found
        for clause in remainder.split(","):
            residual, found = _strip_prefixes(clause.strip())
            prefix = prefix or found
            if residual.strip():
                residual_clauses.append(residual.strip())
    residual_text = " ".join(residual_clauses)
    return CommentSignals(
        approval_prefix=prefix,
        residual_text=residual_text,
        residual_word_count=len(_letter_tokens(residual_text)),
        residual_content_terms=tuple(sorted(content_terms(residual_text))),
        negation_cue=any(p.search(residual_text) for p in NEGATION_CUES),
        irony_cue=any(p.search(text) for p in IRONY_CUES),
        back_reference_cue=any(p.search(text) for p in BACK_REFERENCE_CUES),
        char_length=len(text),
    )


def classify_signals(signals: CommentSignals) -> str:
    """Map signals to one SIGNAL_CLASSES value, in fixed precedence order.

    Precedence: back reference, then irony (or negation without further
    content), then missing approval formula, then blanket versus content by
    MAX_BLANKET_RESIDUAL_WORDS.

    Args:
        signals: The comment's signals from compute_comment_signals.

    Returns:
        One of SIGNAL_CLASSES.
    """
    has_content = signals.residual_word_count > MAX_BLANKET_RESIDUAL_WORDS
    if signals.back_reference_cue:
        return "back_reference"
    if signals.irony_cue or (signals.negation_cue and not has_content):
        return "negation_or_irony_cue"
    if signals.approval_prefix is None:
        return "content_without_approval_phrase"
    return "approval_plus_content" if has_content else "blanket_no_content"


def heuristic_cause(*, scored: bool, match: bool | None, basis: object, signal_class: str) -> str:
    """Descriptive triage of one pair; see HEURISTIC_CAUSES.

    Args:
        scored: Whether the pair is scorable (accepted prediction and a
            resolved reference).
        match: Whether the model's final stance equals the reference label;
            None for unscored pairs.
        basis: The model's basis for the pair (None when there is none).
        signal_class: The comment's SIGNAL_CLASSES value.

    Returns:
        One of HEURISTIC_CAUSES. Only scored general-approval mismatches are
        mapped from the signal class; all other pairs get a fixed cause.
    """
    if not scored:
        return "not_scorable"
    if match:
        return "match"
    if basis != GENERAL_APPROVAL:
        return "model_other_basis"
    return _CAUSE_BY_SIGNAL_CLASS[signal_class]


# --------------------------------------------------------------------------
# Loading (read-only)
# --------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    """Hex SHA-256 of a file's bytes, read in 64 KiB chunks.

    Args:
        path: File to hash.

    Returns:
        The lower-case hex digest.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        # iter(callable, sentinel) keeps calling f.read until it returns b"" (end of file).
        for chunk in iter(lambda: f.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class LoadedRun:
    """A RunSpec's raw predictions and expected cohort, read from disk.

    Attributes:
        spec: The run's specification.
        predictions: The raw predictions parquet.
        expected: The expected-pair cohort parquet.
        run_id: The single run ID found in the predictions, or "" when the
            predictions carry none.
    """

    spec: RunSpec
    predictions: pd.DataFrame
    expected: pd.DataFrame
    run_id: str


def load_runs(specs: Sequence[RunSpec]) -> list[LoadedRun]:
    """Read each run; reject mixed run IDs within a run and pairs shared across runs.

    Args:
        specs: The runs to read.

    Returns:
        One LoadedRun per spec, in the given order.

    Raises:
        ValueError: If run names are not unique, a run's predictions mix run
            IDs or have null pair IDs (via evaluator_v2.unique_predictions),
            or a pair ID appears in more than one run.
    """
    if len({spec.name for spec in specs}) != len(specs):
        raise ValueError("Run names must be unique")
    loaded, seen = [], {}
    for spec in specs:
        predictions = pd.read_parquet(spec.predictions)
        expected = pd.read_parquet(spec.expected)
        evaluator_v2.unique_predictions(predictions)  # raises on mixed run IDs
        run_ids = predictions["run_id"].dropna().unique() if "run_id" in predictions else []
        pair_ids = set(expected.pair_id.astype(str)) | set(predictions.pair_id.astype(str))
        shared = sorted(pair_ids & set(seen))
        if shared:
            raise ValueError(f"Pair {shared[0]} appears in runs {seen[shared[0]]!r} and {spec.name!r}")
        seen.update(dict.fromkeys(pair_ids, spec.name))
        loaded.append(LoadedRun(spec, predictions, expected, str(run_ids[0]) if len(run_ids) else ""))
    return loaded


def load_annotations(db_path: Path) -> pd.DataFrame:
    """Read stance_evaluations read-only and attach pair_id and note flags.

    Args:
        db_path: The SQLite annotation database.

    Returns:
        One row per stance_evaluations row (ordered by id) with the selected
        columns plus pair_id, is_primary (evaluator is PRIMARY_EVALUATOR) and
        the note_flags columns. contribution_id is converted to str.

    Raises:
        sqlite3.OperationalError: If the database cannot be opened or lacks
            the expected table or columns.
    """
    # A file: URI with mode=ro makes SQLite refuse every write, and (unlike a plain
    # path) never creates an empty database when the file is missing.
    uri = f"{Path(db_path).resolve().as_uri()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        frame = pd.read_sql_query(
            "select id, contribution_id, key_message, comment_text, verdict, comment, evaluator, "
            "corrected_label from stance_evaluations order by id",
            conn,
        )
    finally:
        conn.close()
    frame["contribution_id"] = frame.contribution_id.astype(str)
    frame["pair_id"] = [compute_pair_id(r.contribution_id, r.key_message, r.comment_text) for r in frame.itertuples()]
    frame["is_primary"] = frame.evaluator.eq(PRIMARY_EVALUATOR)
    flags = pd.DataFrame([note_flags(note) for note in frame.comment], index=frame.index)
    return pd.concat([frame, flags], axis=1)


def load_sources(paths: Sequence[Path]) -> dict[tuple[str, str], tuple[str, ...]]:
    """Map (contribution_id, key_message) to the extraction's related sentences.

    Args:
        paths: Frozen extraction source parquets; sentences of the same key
            are merged across files.

    Returns:
        A dict from (contribution_id as str, key_message) to the sorted,
        deduplicated related sentences (non-strings dropped).
    """
    lookup: dict[tuple[str, str], set[str]] = {}
    for path in paths:
        source = pd.read_parquet(path, columns=["contribution_id", "key_message", "related_sentences"])
        for row in source.itertuples():
            sentences = row.related_sentences if row.related_sentences is not None else []
            lookup.setdefault((str(row.contribution_id), row.key_message), set()).update(
                s for s in sentences if isinstance(s, str))
    return {key: tuple(sorted(value)) for key, value in lookup.items()}


def load_fold_lookup(folds_path: Path) -> dict[str, int]:
    """Map each contribution to the outer fold in which it is test data.

    Args:
        folds_path: A persisted fold assignment (see fold_builder.load_folds).

    Returns:
        A dict from contribution_id (as str) to its 0-based fold index.
    """
    assignment = load_folds(folds_path)
    return {str(cid): index for index, fold in enumerate(assignment.folds) for cid in fold["test_contribution_ids"]}


# --------------------------------------------------------------------------
# Diagnostic table
# --------------------------------------------------------------------------

DIAGNOSTIC_COLUMNS = (
    "pair_id", "comment_id", "contribution_id", "run_name", "run_id", "fold",
    "key_message_id", "key_message", "comment_text", "contribution_text", "contribution_text_truncated",
    "status", "proposed_stance", "final_stance", "basis", "comment_evidence", "source_evidence",
    "explanation", "review_reasons",
    "resolution_status", "reference_label", "tier", "reference_basis", "reference_provenance",
    "established_original_label", "post_fix_original_label",
    "annotator_verdict", "annotator_corrected_label", "annotator_note", "note_general_approval",
    "note_general_approval_strict", "note_general_rejection", "note_not_neutral",
    "pop_model_ga", "pop_model_ga_claim_without_evidence", "pop_annotator_note", "pop_outside_runs",
    "scored", "match", "mismatch_direction",
    "approval_prefix", "residual_text", "residual_word_count", "residual_content_terms", "negation_cue",
    "irony_cue", "back_reference_cue", "char_length",
    "evidence_char_coverage", "evidence_beyond_approval_prefix", "comment_other_bases",
    "comment_reference_labels_disagree", "km_term_overlap", "related_sentence_term_overlap",
    "signal_class", "heuristic_cause",
)
_BOOLEAN_COLUMNS = ("contribution_text_truncated", "note_general_approval", "note_general_approval_strict",
                    "note_general_rejection", "note_not_neutral", "pop_model_ga",
                    "pop_model_ga_claim_without_evidence", "pop_annotator_note", "pop_outside_runs", "scored",
                    "match", "negation_cue", "irony_cue", "back_reference_cue", "evidence_beyond_approval_prefix",
                    "comment_reference_labels_disagree")
_INTEGER_COLUMNS = ("fold", "residual_word_count", "char_length", "km_term_overlap", "related_sentence_term_overlap")
_LIST_COLUMNS = ("comment_evidence", "source_evidence", "review_reasons", "residual_content_terms")
_PROVENANCE_BY_REFERENCE_BASIS = {
    "verdict_correct": "accepted_shown_label",
    "verdict_incorrect_corrected": "active_correction",
    "tier_b_parsed": "note_parsed",
}


@dataclass(frozen=True)
class DiagnosticTable:
    """The per-pair diagnostic rows plus the evaluator consistency check.

    Attributes:
        pairs: One row per pair, columns DIAGNOSTIC_COLUMNS, in stable order.
        consistency_checks: Per run name, the general-approval counts that
            were verified against evaluator_v2.
        run_ids: Run name to run ID.
    """

    pairs: pd.DataFrame
    consistency_checks: dict
    run_ids: dict[str, str]


def comment_id_for(contribution_id: object, comment_text: str) -> str:
    """Deterministic 16-hex ID of one comment on one contribution.

    Args:
        contribution_id: The contribution's ID (stringified).
        comment_text: The exact raw comment text.

    Returns:
        The first 16 hex characters of SHA-256 over "contribution_id|comment_text".
    """
    return hashlib.sha256(f"{contribution_id}|{comment_text}".encode("utf-8")).hexdigest()[:16]


def _clean(value: object) -> object:
    """Turn missing scalars into None.

    Args:
        value: Any cell value.

    Returns:
        None for None, NaN and pd.NA; the value unchanged otherwise.
    """
    # pd.isna on a list/array returns an array, so it is only applied to scalars;
    # list cells (e.g. comment_evidence) pass through unchanged.
    return None if pd.api.types.is_scalar(value) and pd.isna(value) else value


def _as_list(value: object) -> list[str] | None:
    """Convert a list/array cell to a plain list of strings.

    Args:
        value: A list, tuple or numpy array (parquet reads lists as arrays), or
            a missing scalar.

    Returns:
        The items as strings, or None when the cell is missing.
    """
    value = _clean(value)
    return None if value is None else [str(item) for item in value]


def _in_run(row) -> bool:
    """Whether a population row comes from a model run.

    Args:
        row: A row (Series or namedtuple) with a run_name field.

    Returns:
        False for comments outside every run, whose run_name is None or NaN.
    """
    return isinstance(row.run_name, str)


def _align_run(run: LoadedRun, reference: pd.DataFrame) -> pd.DataFrame:
    """Expected pairs of one run with their (deduplicated) predictions, as the evaluator sees them.

    Expected pairs without a prediction get status "failed"; missing
    key_message and comment_text values are filled in from the reference.

    Args:
        run: The loaded run.
        reference: The selected reference rows (pair_id as str).

    Returns:
        One row per expected pair with the prediction columns plus run_name
        and run_id.

    Raises:
        ValueError: If a pair has no key message or comment text in either
            the predictions or the reference, or if
            evaluator_v2.unique_predictions rejects the predictions.
    """
    expected = run.expected[["pair_id", "contribution_id"]].astype(str).drop_duplicates("pair_id")
    predictions = evaluator_v2.unique_predictions(run.predictions)
    # Internal evaluator columns start with "_"; contribution_id comes from the expected cohort instead.
    columns = [c for c in predictions.columns if c not in ("contribution_id",) and not c.startswith("_")]
    # indicator=True adds a "_merge" column telling which side each row came from.
    merged = expected.merge(predictions[columns], on="pair_id", how="left", indicator=True)
    missing = merged["_merge"] == "left_only"
    merged.loc[missing, "status"] = "failed"
    merged = merged.drop(columns="_merge")
    texts = reference.set_index("pair_id")[["key_message", "comment_text"]]
    for column in ("key_message", "comment_text"):
        merged[column] = merged[column].where(merged[column].notna(), merged.pair_id.map(texts[column]))
    if merged.comment_text.isna().any() or merged.key_message.isna().any():
        raise ValueError(f"Run {run.spec.name!r} has pairs without text in predictions or reference")
    merged["run_name"], merged["run_id"] = run.spec.name, run.run_id
    return merged


def _outside_rows(annotations: pd.DataFrame, keys: set[tuple[str, str]]) -> pd.DataFrame:
    """Population rows for noted comments on contributions that are in no model run.

    Args:
        annotations: Output of load_annotations.
        keys: (contribution_id, comment_text) keys of the outside comments.

    Returns:
        One row per primary annotation of those comments, with the run and
        prediction columns set to None.
    """
    primary = annotations[annotations.is_primary]
    in_keys = [(cid, text) in keys for cid, text in zip(primary.contribution_id, primary.comment_text)]
    rows = primary[in_keys][["pair_id", "contribution_id", "key_message", "comment_text"]].copy()
    rows["run_name"] = rows["run_id"] = rows["key_message_id"] = rows["status"] = None
    return rows


def _population(aligned: pd.DataFrame, annotations: pd.DataFrame) -> tuple[pd.DataFrame, set, set, set, set]:
    """All pairs of every comment in the review population, plus the per-flag comment keys.

    A comment (keyed by contribution_id and comment_text) is in the
    population if the model gave basis general_approval on any of its pairs,
    flagged a general-support claim without evidence (GA_CLAIM_REVIEW_REASONS),
    or the primary annotator's note matches the broad general-approval
    pattern. Every pair of such a comment is included, not only the flagged
    ones.

    Args:
        aligned: Output of _align_run for all runs, concatenated.
        annotations: Output of load_annotations.

    Returns:
        A tuple (rows, model_ga, claim, noted, outside): the population rows
        and the sets of comment keys per population flag; outside holds the
        noted comments that appear in no run.

    Raises:
        ValueError: If a pair ID occurs twice in the population (e.g.
            duplicate primary annotations of an outside comment).
    """
    key = list(zip(aligned.contribution_id, aligned.comment_text))
    aligned = aligned.assign(_key=key)
    reasons = aligned.review_reasons.map(lambda r: bool(set(_as_list(r) or ()) & GA_CLAIM_REVIEW_REASONS))
    model_ga = set(aligned.loc[aligned.basis.eq(GENERAL_APPROVAL), "_key"])
    claim = set(aligned.loc[reasons, "_key"])
    noted_rows = annotations[annotations.is_primary & annotations.note_general_approval]
    noted = set(zip(noted_rows.contribution_id, noted_rows.comment_text))
    outside = noted - set(aligned._key)
    in_runs = aligned[aligned._key.isin(model_ga | claim | noted)].drop(columns="_key")
    rows = pd.concat([in_runs, _outside_rows(annotations, outside)], ignore_index=True)
    if rows.pair_id.duplicated().any():
        raise ValueError("Duplicate pair IDs in the review population")
    return rows, model_ga, claim, noted, outside


def _pair_record(row, signals: CommentSignals, related: Mapping, fold_of: Mapping) -> dict:
    """Pair-level diagnostic fields of one population row.

    Args:
        row: A population row merged with its reference columns (a Series).
        signals: The signals of the row's comment.
        related: Output of load_sources.
        fold_of: Output of load_fold_lookup.

    Returns:
        The pair-level DIAGNOSTIC_COLUMNS values (without the comment-level,
        annotation and heuristic_cause fields). Model fields are None for
        rows outside every run; match is None for unscored pairs.
    """
    in_run = _in_run(row)
    # row.get: optional prediction columns may be absent altogether when no run provides them.
    evidence = _as_list(row.get("comment_evidence")) if in_run else None
    scored = bool(in_run and row.status == "accepted" and row.resolution_status == "resolved")
    match = (row.final_stance == row.reference_label) if scored else None
    contribution = _clean(row.get("contributionContent"))
    sentences = related.get((row.contribution_id, row.key_message))
    residual_terms = set(signals.residual_content_terms)
    coverage = None
    if in_run:
        # Share of the comment's characters quoted as evidence, capped at 1 (quotes may overlap).
        quoted = sum(len(_normalize(q)) for q in evidence or [])
        coverage = min(1.0, quoted / signals.char_length) if signals.char_length else 0.0
    return {
        "pair_id": row.pair_id,
        "contribution_id": row.contribution_id,
        "run_name": row.run_name,
        "run_id": row.run_id,
        "fold": fold_of.get(row.contribution_id),
        "key_message_id": _clean(row.get("key_message_id")),
        "key_message": row.key_message,
        "comment_text": row.comment_text,
        "contribution_text": contribution,
        # A text at the source's cut-off length is assumed to have been cut there.
        "contribution_text_truncated": (len(contribution) >= CONTRIBUTION_TEXT_LIMIT) if contribution else None,
        "status": _clean(row.status),
        "proposed_stance": _clean(row.get("proposed_stance")),
        "final_stance": _clean(row.get("final_stance")),
        "basis": _clean(row.get("basis")),
        "comment_evidence": evidence,
        "source_evidence": _as_list(row.get("source_evidence")) if in_run else None,
        "explanation": _clean(row.get("explanation")),
        "review_reasons": _as_list(row.get("review_reasons")) if in_run else None,
        # Pairs absent from the selected reference (role/tiers) have no resolution_status after the left merge.
        "resolution_status": _clean(row.resolution_status) or "no_reference",
        "reference_label": _clean(row.reference_label),
        "tier": _clean(row.tier),
        "reference_basis": _clean(row.reference_basis),
        "reference_provenance": (_PROVENANCE_BY_REFERENCE_BASIS.get(row.reference_basis, "unresolved")
                                 if row.resolution_status == "resolved" else "unresolved"),
        "established_original_label": _clean(row.established_original_label),
        "post_fix_original_label": _clean(row.post_fix_original_label),
        "scored": scored,
        "match": match,
        "mismatch_direction": f"{row.final_stance}_vs_{row.reference_label}" if scored and not match else None,
        "evidence_char_coverage": coverage,
        "evidence_beyond_approval_prefix": (
            compute_comment_signals(" ".join(evidence)).residual_word_count > MAX_BLANKET_RESIDUAL_WORDS
            if evidence else None),
        "km_term_overlap": len(residual_terms & content_terms(row.key_message)),
        "related_sentence_term_overlap": (
            len(residual_terms & content_terms(" ".join(sentences))) if sentences is not None else None),
    }


def _signal_fields(signals: CommentSignals) -> dict:
    """Comment-level diagnostic fields derived from the comment's signals.

    Args:
        signals: The comment's signals.

    Returns:
        The signal columns of DIAGNOSTIC_COLUMNS, including signal_class.
    """
    return {
        "approval_prefix": signals.approval_prefix,
        "residual_text": signals.residual_text,
        "residual_word_count": signals.residual_word_count,
        "residual_content_terms": list(signals.residual_content_terms),
        "negation_cue": signals.negation_cue,
        "irony_cue": signals.irony_cue,
        "back_reference_cue": signals.back_reference_cue,
        "char_length": signals.char_length,
        "signal_class": classify_signals(signals),
    }


def _annotation_fields(annotation) -> dict:
    """Primary-annotator fields of one pair.

    Args:
        annotation: The pair's primary annotation row (a Series from
            load_annotations), or None when the pair has none.

    Returns:
        annotator_verdict, annotator_corrected_label, annotator_note (None
        when missing) and the four note flags (False when there is no
        annotation).
    """
    if annotation is None:
        return {"annotator_verdict": None, "annotator_corrected_label": None, "annotator_note": None,
                **dict.fromkeys(("note_general_approval", "note_general_approval_strict",
                                 "note_general_rejection", "note_not_neutral"), False)}
    return {
        "annotator_verdict": _clean(annotation.verdict),
        "annotator_corrected_label": _clean(annotation.corrected_label),
        "annotator_note": _clean(annotation.comment),
        **{name: bool(getattr(annotation, name)) for name in (
            "note_general_approval", "note_general_approval_strict", "note_general_rejection", "note_not_neutral")},
    }


def _comment_fields(group: pd.DataFrame) -> dict:
    """Comment-level fields computed across all pairs of one comment.

    Args:
        group: All population rows of one comment, with reference columns.

    Returns:
        comment_other_bases: ";"-joined sorted model bases other than
        general_approval ("none" for pairs without a basis), or None.
        comment_reference_labels_disagree: Whether the comment's resolved
        reference labels differ across its key messages.
    """
    other = sorted({b if isinstance(b, str) else "none" for b in group.basis} - {GENERAL_APPROVAL})
    labels = group.loc[group.resolution_status == "resolved", "reference_label"].dropna()
    return {"comment_other_bases": ";".join(other) or None,
            "comment_reference_labels_disagree": labels.nunique() > 1}


def _finalize(table: pd.DataFrame) -> pd.DataFrame:
    """Sort the diagnostic rows deterministically and apply nullable dtypes.

    Rows are ordered by numeric contribution ID, then comment_id, then key
    messages with an ID (by ID) before those without (by text).

    Args:
        table: The raw diagnostic records.

    Returns:
        The table with exactly DIAGNOSTIC_COLUMNS and a fresh index.
    """
    table = table.assign(_cid=table.contribution_id.astype(int), _no_km=table.key_message_id.isna(),
                         _km=table.key_message_id.fillna(""))
    table = table.sort_values(["_cid", "comment_id", "_no_km", "_km", "key_message"], kind="stable")
    table = table.drop(columns=["_cid", "_no_km", "_km"]).reset_index(drop=True)[list(DIAGNOSTIC_COLUMNS)]
    # Nullable pandas dtypes ("boolean", "Int64", "Float64") keep missing values as <NA>
    # instead of turning bool/int columns into object or float columns with NaN.
    for column in _BOOLEAN_COLUMNS:
        table[column] = table[column].astype("boolean")
    for column in _INTEGER_COLUMNS:
        table[column] = table[column].astype("Int64")
    table["evidence_char_coverage"] = table.evidence_char_coverage.astype("Float64")
    return table


def verify_against_evaluator(table: pd.DataFrame, runs: Sequence[LoadedRun], reference: pd.DataFrame,
                             *, role: str, tiers: Sequence[str]) -> dict:
    """Check that scored general-approval pairs per run equal evaluator_v2's basis breakdown.

    This guards against the diagnostic table drifting from how the official
    evaluator selects and scores pairs.

    Args:
        table: The finalized diagnostic table.
        runs: The loaded runs.
        reference: The full (unselected) reference dataset.
        role: Reference role selection passed to evaluator_v2.
        tiers: Reference tiers passed to evaluator_v2.

    Returns:
        Per run name: general_approval_pairs, general_approval_correct and
        matches_evaluator_v2 (always True when returned).

    Raises:
        ValueError: If the (pairs, correct) counts of any run differ.
    """
    checks = {}
    for run in runs:
        by_basis = evaluator_v2.accepted_accuracy_by_basis(
            run.predictions, reference, run.expected, role=role, tiers=tuple(tiers))
        expected = by_basis.get(GENERAL_APPROVAL)
        rows = table[table.run_name.eq(run.spec.name) & table.scored.fillna(False) & table.basis.eq(GENERAL_APPROVAL)]
        observed = (len(rows), int(rows.match.fillna(False).sum()))
        wanted = (expected.n, expected.correct) if expected else (0, 0)
        if observed != wanted:
            raise ValueError(f"Run {run.spec.name!r}: diagnostic table scores {observed} general_approval pairs/"
                             f"correct, evaluator_v2 scores {wanted}")
        checks[run.spec.name] = {"general_approval_pairs": observed[0], "general_approval_correct": observed[1],
                                 "matches_evaluator_v2": True}
    return checks


def build_diagnostic_table(runs: Sequence[LoadedRun], reference: pd.DataFrame, annotations: pd.DataFrame, *,
                           related_sentences: Mapping[tuple[str, str], tuple[str, ...]],
                           fold_of: Mapping[str, int], role: str = "primary",
                           tiers: Sequence[str] = ("A", "B")) -> DiagnosticTable:
    """Build one row per pair of every comment in the review population (see module docstring).

    Args:
        runs: The loaded runs (pair IDs must be disjoint across runs).
        reference: The full reference dataset; it is filtered here with
            evaluator_v2.select_reference.
        annotations: Output of load_annotations.
        related_sentences: Output of load_sources.
        fold_of: Output of load_fold_lookup.
        role: Reference role selection. Defaults to "primary".
        tiers: Reference tiers. Defaults to ("A", "B").

    Returns:
        The DiagnosticTable with its consistency checks and run IDs.

    Raises:
        ValueError: If a run has pairs without text, the population has
            duplicate pair IDs, the selected reference has duplicate pair IDs
            (pandas MergeError), or the counts disagree with evaluator_v2.
    """
    selected = evaluator_v2.select_reference(reference, role=role, tiers=tuple(tiers))
    selected = selected.assign(pair_id=selected.pair_id.astype(str),
                               contribution_id=selected.contribution_id.astype(str))
    aligned = pd.concat([_align_run(run, selected) for run in runs], ignore_index=True)
    rows, model_ga, claim, noted, outside = _population(aligned, annotations)
    ref_columns = ["resolution_status", "reference_label", "tier", "basis", "established_original_label",
                   "post_fix_original_label"]
    rows = rows.merge(selected[["pair_id", *ref_columns]].rename(columns={"basis": "reference_basis"}),
                      on="pair_id", how="left", validate="one_to_one")
    # For in-run pairs the earliest primary annotation (lowest id) is shown; duplicates of
    # outside-run comments were already rejected in _population.
    primary = annotations[annotations.is_primary].drop_duplicates("pair_id").set_index("pair_id")
    records = []
    for (cid, text), group in rows.groupby(["contribution_id", "comment_text"], sort=False):
        signals = compute_comment_signals(text)
        key = (cid, text)
        comment = {
            "comment_id": comment_id_for(cid, text), **_signal_fields(signals), **_comment_fields(group),
            "pop_model_ga": key in model_ga, "pop_model_ga_claim_without_evidence": key in claim,
            "pop_annotator_note": key in noted, "pop_outside_runs": key in outside,
        }
        for _, row in group.iterrows():
            annotation = primary.loc[row.pair_id] if row.pair_id in primary.index else None
            record = {**comment, **_pair_record(row, signals, related_sentences, fold_of),
                      **_annotation_fields(annotation)}
            record["heuristic_cause"] = heuristic_cause(scored=record["scored"], match=record["match"],
                                                        basis=record["basis"], signal_class=record["signal_class"])
            records.append(record)
    table = _finalize(pd.DataFrame.from_records(records, columns=list(DIAGNOSTIC_COLUMNS)))
    checks = verify_against_evaluator(table, runs, reference, role=role, tiers=tiers)
    return DiagnosticTable(table, checks, {run.spec.name: run.run_id for run in runs})


# --------------------------------------------------------------------------
# Safe summary (IDs and counts only)
# --------------------------------------------------------------------------


def _counts(values: pd.Series) -> dict[str, int]:
    """Count values, with missing values counted as "unresolved".

    Args:
        values: The values to count.

    Returns:
        A dict from str(value) to its count, sorted by value.
    """
    return {str(k): int(v) for k, v in sorted(Counter(values.fillna("unresolved")).items())}


def _scored_block(frame: pd.DataFrame) -> dict:
    """Summary counts of a set of scored pairs.

    Args:
        frame: Diagnostic rows (normally scored general-approval pairs).

    Returns:
        pairs, correct (pairs with match True), reference_labels (counts per
        label), comments and contributions (distinct counts).
    """
    return {"pairs": len(frame), "correct": int(frame.match.fillna(False).sum()),
            "reference_labels": _counts(frame.reference_label), "comments": int(frame.comment_id.nunique()),
            "contributions": int(frame.contribution_id.nunique())}


def _pairs_comments(frame: pd.DataFrame, comment_key: Sequence[str]) -> dict:
    """Count rows and distinct comments of a frame.

    Args:
        frame: Diagnostic or annotation rows.
        comment_key: Columns that identify a comment (comment_id in the
            diagnostic table; contribution_id and comment_text in annotations).

    Returns:
        pairs (row count) and comments (distinct comment keys).
    """
    return {"pairs": len(frame), "comments": int(frame[list(comment_key)].drop_duplicates().shape[0])}


def diagnostic_content_sha256(table: pd.DataFrame) -> str:
    """Canonical hash of the table's content (sorted by pair_id, no timestamps).

    Args:
        table: The diagnostic table.

    Returns:
        Hex SHA-256 of the table serialized as JSON records; identical inputs
        give the same hash regardless of run time or row order.
    """
    # double_precision=10 fixes float formatting so the JSON (and the hash) is stable.
    canonical = table.sort_values("pair_id", kind="stable").to_json(
        orient="records", force_ascii=False, double_precision=10)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def count_illegal_characters(table: pd.DataFrame) -> int:
    """Count characters Excel cannot store, summed over the diagnostic rows' texts.

    The pack shows U+FFFD instead of each of them (see safe_text).

    Args:
        table: The diagnostic table.

    Returns:
        Number of ILLEGAL_CHARACTERS_RE matches in comment, contribution,
        key-message, note, explanation and comment-evidence texts.
    """
    columns = ("comment_text", "contribution_text", "key_message", "annotator_note", "explanation")
    texts = [v for column in columns for v in table[column] if isinstance(v, str)]
    texts += [q for value in table.comment_evidence if value is not None for q in value]
    return sum(len(ILLEGAL_CHARACTERS_RE.findall(text)) for text in texts)


def _notes_section(pairs: pd.DataFrame, annotations: pd.DataFrame) -> dict:
    """The annotator_notes block of the safe summary.

    Note-category counts come from all primary annotations; the
    reference-label breakdowns and the outside/in-run splits come from the
    diagnostic rows.

    Args:
        pairs: The diagnostic table.
        annotations: Output of load_annotations.

    Returns:
        Counts per note category, the broad general-approval notes outside
        any run (contributions_in_a_fold counts distinct contributions that
        have a fold), and the in-run noted pairs per "basis/status".
    """
    primary = annotations[annotations.is_primary]
    comment_key = ("contribution_id", "comment_text")
    section = {"primary_annotation_rows": len(primary)}
    for name, column in (("general_approval_broad", "note_general_approval"),
                         ("general_approval_strict", "note_general_approval_strict")):
        noted = pairs[pairs[column].fillna(False)]
        section[name] = {**_pairs_comments(primary[primary[column]], comment_key),
                         "reference_labels": _counts(noted.reference_label)}
    for name, column in (("general_rejection", "note_general_rejection"), ("not_neutral", "note_not_neutral")):
        section[name] = _pairs_comments(primary[primary[column]], comment_key)
    noted = pairs[pairs.note_general_approval.fillna(False)]
    outside = noted[noted.pop_outside_runs.fillna(False)]
    section["general_approval_broad_outside_runs"] = {
        **_pairs_comments(outside, ["comment_id"]), "contributions": int(outside.contribution_id.nunique()),
        "contributions_in_a_fold": int(outside.loc[outside.fold.notna(), "contribution_id"].nunique())}
    in_runs = noted[~noted.pop_outside_runs.fillna(False)]
    section["general_approval_broad_in_runs_by_model_basis_and_status"] = _counts(
        in_runs.basis.fillna("none") + "/" + in_runs.status.fillna("none"))
    return section


def _signal_section(ga: pd.DataFrame) -> dict:
    """Match/mismatch counts of scored general-approval pairs per signal class.

    Args:
        ga: The scored general-approval diagnostic rows.

    Returns:
        Per signal class (sorted): comments, pairs and, per reference label,
        the match and mismatch counts.
    """
    section = {}
    for signal_class, group in ga.groupby("signal_class"):
        section[signal_class] = {
            "comments": int(group.comment_id.nunique()), "pairs": len(group),
            "by_reference_label": {
                str(label): {"match": int(g.match.fillna(False).sum()), "mismatch": int((~g.match.fillna(False)).sum())}
                for label, g in sorted(group.groupby("reference_label"))}}
    return dict(sorted(section.items()))


def build_safe_summary(diagnostic: DiagnosticTable, annotations: pd.DataFrame, *, inputs: Mapping[str, Mapping],
                       created_utc: str, role: str, tiers: Sequence[str]) -> dict:
    """Build the safe summary: counts and IDs only.

    Never contains texts, notes, key messages, explanations or evaluator
    names, so it can be shared and committed.

    Args:
        diagnostic: The diagnostic table and its consistency checks.
        annotations: Output of load_annotations.
        inputs: Per input name, its path and sha256 (from _hash_inputs).
        created_utc: Creation timestamp in CREATED_UTC_FORMAT.
        role: Reference role selection that was used.
        tiers: Reference tiers that were used.

    Returns:
        The JSON-serializable summary dict, including module_sha256 (hash of
        this source file) and diagnostic_content_sha256.
    """
    pairs = diagnostic.pairs
    ga = pairs[pairs.scored.fillna(False) & pairs.basis.eq(GENERAL_APPROVAL)]
    accepted_ga = pairs[pairs.status.eq("accepted") & pairs.basis.eq(GENERAL_APPROVAL)]
    claim = pairs[pairs.review_reasons.map(lambda r: bool(set(_as_list(r) or ()) & GA_CLAIM_REVIEW_REASONS))]
    disagree = ga[ga.comment_reference_labels_disagree.fillna(False)]
    pack_comments = pairs.drop_duplicates("comment_id")
    pack_keys = set(zip(pack_comments.contribution_id, pack_comments.comment_text))
    secondary = annotations[~annotations.is_primary]
    on_pack = [key in pack_keys for key in zip(secondary.contribution_id, secondary.comment_text)]
    return {
        "created_utc": created_utc,
        "analysis_only": True,
        "new_inference_requests": 0,
        "network_calls": 0,
        "versions": {"diagnostic_schema": DIAGNOSTIC_SCHEMA_VERSION, "pack_schema": PACK_SCHEMA_VERSION,
                     "signals": SIGNALS_VERSION},
        "selection": {"role": role, "tiers": list(tiers)},
        "run_ids": dict(sorted(diagnostic.run_ids.items())),
        "module_sha256": sha256_file(Path(__file__).resolve()),
        "input_sha256": {name: dict(value) for name, value in sorted(inputs.items())},
        "scored_general_approval": {
            "definition": "status accepted, basis general_approval, resolved reference (selected role and tiers)",
            "total": _scored_block(ga),
            "by_run": {name: _scored_block(g) for name, g in sorted(ga.groupby("run_name"))},
            "by_fold": {str(int(fold)): _scored_block(g) for fold, g in sorted(ga.groupby("fold"))},
        },
        "accepted_general_approval_unresolved_reference": _pairs_comments(
            accepted_ga[~accepted_ga.scored.fillna(False)], ["comment_id"]),
        "general_approval_claim_without_evidence": {
            **_pairs_comments(claim, ["comment_id"]), "by_status": _counts(claim.status)},
        "annotator_notes": _notes_section(pairs, annotations),
        "scored_general_approval_reference_provenance": {
            str(p): _counts(g.reference_label) for p, g in sorted(ga.groupby("reference_provenance"))},
        "scored_general_approval_signal_classes": _signal_section(ga),
        "heuristic_causes": {"pack_pairs": _counts(pairs.heuristic_cause),
                             "scored_general_approval_pairs": _counts(ga.heuristic_cause)},
        "scored_general_approval_comments_with_disagreeing_reference_labels": int(disagree.comment_id.nunique()),
        "pack": {
            "comments": int(pairs.comment_id.nunique()), "key_message_rows": len(pairs),
            "comments_in_runs": int(pairs.loc[~pairs.pop_outside_runs.fillna(False), "comment_id"].nunique()),
            "comments_outside_runs": int(pairs.loc[pairs.pop_outside_runs.fillna(False), "comment_id"].nunique()),
            "comments_by_population_flag": {
                flag: int(pairs.loc[pairs[flag].fillna(False), "comment_id"].nunique())
                for flag in ("pop_model_ga", "pop_model_ga_claim_without_evidence", "pop_annotator_note",
                             "pop_outside_runs")},
            "illegal_characters_in_diagnostic_texts": count_illegal_characters(pairs),
        },
        "secondary_annotations": {"rows_total": len(secondary), "rows_on_pack_comments": int(sum(on_pack))},
        "consistency_checks": {"general_approval_vs_evaluator_v2": diagnostic.consistency_checks},
        "diagnostic_content_sha256": diagnostic_content_sha256(pairs),
        "heuristics_note": "Signals and heuristic causes were derived after inspecting this cohort; "
                           "they are descriptive triage, not validated.",
        "heldout_note": HELDOUT_NOTE,
    }


# --------------------------------------------------------------------------
# Review pack (German)
# --------------------------------------------------------------------------

ANLEITUNG_SHEET, COMMENTS_SHEET, KM_SHEET = "Anleitung", "Kommentare", "Kernbotschaften"
OVERVIEW_SHEET, LISTS_SHEET, META_SHEET = "Übersicht", "Listen", "Meta"
SHEET_ORDER = (ANLEITUNG_SHEET, COMMENTS_SHEET, KM_SHEET, OVERVIEW_SHEET, LISTS_SHEET, META_SHEET)
KEY_ROW, HEADER_ROW, FIRST_DATA_ROW = 1, 2, 3
GLOBAL_ANSWER_NAMES = ("g_regel_insgesamt", "g_begruendung")

OPTION_SETS: dict[str, tuple[tuple[str, str], ...]] = {
    "einordnung": (
        ("blanket_no_content", "Pauschale Zustimmung ohne eigenen Inhalt"),
        ("blanket_plus_content", "Pauschale Zustimmung mit eigenem Inhalt zu einzelnen Kernbotschaften"),
        ("content_only", "Eigener Inhalt zu einzelnen Kernbotschaften, keine pauschale Zustimmung"),
        ("no_approval", "Keine Zustimmung (z. B. Ablehnung, Ironie, Themenwechsel)"),
        ("reference_unclear", "Bezug unklar (z. B. Antwort auf anderen Kommentar)"),
        ("undecidable", "Nicht entscheidbar"),
    ),
    "regel": (
        ("rule_1_always_neutral", "Regel 1: immer neutral"),
        ("rule_2_always_in_favor", "Regel 2: immer dafür"),
        ("rule_3_neutral_unless_own_content", "Regel 3: neutral, außer mit eigenem Inhalt"),
        ("no_preference", "keine Präferenz"),
    ),
    "reichweite": (
        ("all_key_messages", "alle Kernbotschaften"),
        ("addressed_only", "nur inhaltlich angesprochene"),
        ("unclear", "unklar"),
    ),
    "angesprochen": (("yes", "ja"), ("no", "nein"), ("unclear", "unklar")),
    "ursache": (
        ("A_label_inconsistency", "A Label uneinheitlich"),
        ("B_model_error", "B Modellfehler"),
        ("rule_dependent", "regelabhängig"),
        ("no_error", "kein Fehler"),
        ("not_assessable", "nicht bewertbar"),
    ),
}

STANCE_DE = {"in_favor": "dafür", "against": "dagegen", "neutral": "neutral"}
STATUS_DE = {"accepted": "angenommen", "needs_review": "zur Prüfung", "failed": "fehlgeschlagen"}
BASIS_DE = {"general_approval": "pauschale Zustimmung", "explicit_target": "ausdrücklich angesprochen",
            "unaddressed": "nicht angesprochen", "no_position": "keine Position",
            "explicit_exclusion": "ausdrücklich ausgenommen", "unresolved": "ungeklärt"}
PROVENANCE_DE = {"accepted_shown_label": "angezeigtes Label bestätigt", "active_correction": "aktiv korrigiert",
                 "note_parsed": "aus Notiz abgeleitet", "unresolved": "nicht auflösbar"}
SIGNAL_CLASS_DE = {"blanket_no_content": "Zustimmungsformel ohne weiteren Inhalt",
                   "approval_plus_content": "Zustimmungsformel mit weiterem Inhalt",
                   "content_without_approval_phrase": "keine Zustimmungsformel erkannt",
                   "negation_or_irony_cue": "Hinweis auf Verneinung oder Ironie",
                   "back_reference": "Verweis auf anderen Text"}
CAUSE_DE = {"match": "Übereinstimmung", "A_label_inconsistency_candidate": "A Label uneinheitlich (Kandidat)",
            "B_model_error_candidate": "B Modellfehler (Kandidat)", "rule_dependent": "regelabhängig",
            "not_scorable": "nicht bewertbar", "model_other_basis": "andere Modell-Einordnung"}


@dataclass(frozen=True)
class ColumnSpec:
    """One pack column: machine key (row 1), German header (row 2), layout and editability.

    Attributes:
        key: Machine key written into the hidden key row; the reader locates
            columns by it, so columns may be reordered.
        header: German header shown in the header row.
        width: Excel column width.
        group: "main" (visible), "background" / "team" (collapsed outline
            groups), "separator" (narrow visible column right of a group,
            carrying its plus button, which keeps the two groups apart),
            "hidden" (IDs).
        editable: Whether the annotator or team fills the cell (unlocked,
            coloured, read back by the reader). Defaults to False.
        list_name: OPTION_SETS key of the drop-down list, or None for free
            text. Defaults to None.
    """

    key: str
    header: str
    width: float
    group: str
    editable: bool = False
    list_name: str | None = None


def _separator(key: str) -> ColumnSpec:
    """A narrow, header-less separator column (see ColumnSpec.group).

    Args:
        key: The column's machine key.

    Returns:
        The separator ColumnSpec.
    """
    return ColumnSpec(key, "", 3, "separator")


COMMENT_COLUMNS = (
    ColumnSpec("nr", "Nr.", 8, "main"),
    ColumnSpec("comment_text", "Kommentar", 60, "main"),
    ColumnSpec("contribution_text", "Beitrag (Auszug)", 50, "main"),
    ColumnSpec("key_messages_overview", "Kernbotschaften des Beitrags", 45, "main"),
    ColumnSpec("d_einordnung", "Einordnung des Kommentars", 36, "main", True, "einordnung"),
    ColumnSpec("d_regel", "Welche Regel passt für diesen Kommentar?", 28, "main", True, "regel"),
    ColumnSpec("d_reichweite", "Nur bei Regel 3: Reichweite", 26, "main", True, "reichweite"),
    ColumnSpec("d_anmerkung", "Ihre Anmerkung", 40, "main", True),
    ColumnSpec("h_notizen", "Hintergrund: Ihre damaligen Notizen", 40, "background"),
    ColumnSpec("h_modell", "Hintergrund: Modell je Kernbotschaft", 45, "background"),
    ColumnSpec("h_belege", "Hintergrund: vom Modell zitierte Stellen", 45, "background"),
    ColumnSpec("h_begruendung", "Hintergrund: Begründung des Modells", 50, "background"),
    ColumnSpec("h_ihre_labels", "Hintergrund: Ihr bisheriges Label je Kernbotschaft", 42, "background"),
    ColumnSpec("h_signale", "Hintergrund: Textsignale (automatisch)", 34, "background"),
    _separator("sep_hintergrund"),
    ColumnSpec("comment_id", "comment_id", 18, "hidden"),
    ColumnSpec("pair_ids", "pair_ids", 18, "hidden"),
    ColumnSpec("row_fingerprint", "row_fingerprint", 18, "hidden"),
    ColumnSpec("t_ursache", "Projektteam: Ursache", 24, "team", True, "ursache"),
    ColumnSpec("t_vorschlag", "Projektteam: Vorschlag (Heuristik)", 40, "team"),
    ColumnSpec("t_anmerkung", "Projektteam: Anmerkung", 40, "team", True),
    _separator("sep_projektteam"),
)
KM_COLUMNS = (
    ColumnSpec("nr", "Nr.", 9, "main"),
    ColumnSpec("comment_text", "Kommentar (gekürzt)", 45, "main"),
    ColumnSpec("key_message", "Kernbotschaft", 50, "main"),
    ColumnSpec("d_km_angesprochen", "Spricht der Kommentar diese Kernbotschaft inhaltlich an?", 30, "main", True,
               "angesprochen"),
    ColumnSpec("h_ihr_label", "Hintergrund: Ihr bisheriges Label", 22, "background"),
    ColumnSpec("h_herkunft", "Hintergrund: Herkunft des Labels", 26, "background"),
    ColumnSpec("h_modell_label", "Hintergrund: Modell-Label", 18, "background"),
    ColumnSpec("h_modell_basis", "Hintergrund: Modell-Einordnung", 26, "background"),
    ColumnSpec("h_wortueberschneidung", "Hintergrund: gemeinsame Wörter mit der Kernbotschaft", 20, "background"),
    _separator("sep_hintergrund"),
    ColumnSpec("pair_id", "pair_id", 18, "hidden"),
    ColumnSpec("comment_id", "comment_id", 18, "hidden"),
    ColumnSpec("row_fingerprint", "row_fingerprint", 18, "hidden"),
)
OUTLINED_GROUPS = ("background", "team")

FONT = Font(name="Arial", size=10)
BOLD = Font(name="Arial", size=10, bold=True)
TITLE = Font(name="Arial", size=12, bold=True)
ANSWER_FILL = PatternFill("solid", fgColor="FFFF99")
TEAM_FILL = PatternFill("solid", fgColor="DDEBF7")
HEADER_FILL = PatternFill("solid", fgColor="D9D9D9")
BACKGROUND_HEADER_FILL = PatternFill("solid", fgColor="EDEDED")
WRAP = Alignment(wrap_text=True, vertical="top")
UNLOCKED = Protection(locked=False)
# openpyxl stores a string starting with "=" as a formula, and Excel may evaluate cells starting
# with any of these characters (formula injection); such texts are forced to plain text in _write_cell.
_FORMULA_LEADS = ("=", "+", "-", "@")


def safe_text(value: object) -> tuple[str | None, int]:
    """Make a text Excel-safe.

    Newlines are normalized to "\\n" and the text is stripped, characters
    Excel cannot store are replaced with U+FFFD, and the result is capped at
    EXCEL_CELL_LIMIT characters.

    Args:
        value: Any value (stringified); None stays None.

    Returns:
        A tuple of the safe text (None for None) and the number of replaced
        illegal characters.
    """
    if value is None:
        return None, 0
    text = str(value).replace("\r\n", "\n").replace("\r", "\n").strip()
    # ILLEGAL_CHARACTERS_RE: the control characters openpyxl refuses to write (XML 1.0 cannot hold them).
    text, replaced = ILLEGAL_CHARACTERS_RE.subn("\ufffd", text)
    return text[:EXCEL_CELL_LIMIT], replaced


def row_fingerprint(*parts: str) -> str:
    """16-hex fingerprint of a row's identifying cells, as written.

    The reader recomputes it from the cells it reads back, so any edit to a
    text or ID cell of that row is detected.

    Args:
        *parts: The identifying cell values, in a fixed order.

    Returns:
        The first 16 hex characters of SHA-256 over the joined parts.
    """
    # "\x1f" (ASCII unit separator) cannot occur in Excel-safe texts (safe_text replaces it) or hex IDs,
    # so the join is unambiguous: ("ab", "c") and ("a", "bc") give different fingerprints.
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:16]


def _lines(values) -> str:
    """Join non-blank strings, one per line, without duplicates.

    Args:
        values: Iterable of values; non-strings and blank strings are skipped.

    Returns:
        The distinct strings in first-seen order, joined with newlines.
    """
    # dict.fromkeys deduplicates while keeping the first-seen order (a set would not).
    seen = list(dict.fromkeys(v for v in values if isinstance(v, str) and v.strip()))
    return "\n".join(seen)


def _short(text: str) -> str:
    """Shorten a text to SHORT_COMMENT_CHARS characters plus " ..." when longer.

    Args:
        text: The text to shorten.

    Returns:
        The text unchanged, or its truncated form.
    """
    return text if len(text) <= SHORT_COMMENT_CHARS else text[:SHORT_COMMENT_CHARS].rstrip() + " ..."


def _display(value: object) -> str | None:
    """Display form of a stored text: HTML entities decoded and Excel-safe.

    Args:
        value: A stored text, or a missing value.

    Returns:
        The display text, or None when the value is missing.
    """
    value = _clean(value)
    return None if value is None else safe_text(html.unescape(str(value)))[0]


def _stance_de(value: object) -> str:
    """German label of a stance code; "kein Label" when missing or unknown.

    Args:
        value: A stance code (in_favor, against, neutral) or a missing value.

    Returns:
        The German label.
    """
    return STANCE_DE.get(_clean(value), "kein Label")


def _basis_de(value: object, missing: str) -> str:
    """German label of a model basis code.

    Args:
        value: A basis code or a missing value.
        missing: Text to return when the basis is missing.

    Returns:
        The German label; unknown codes are returned as they are.
    """
    basis = _clean(value)
    return missing if basis is None else BASIS_DE.get(basis, str(basis))


def _model_line(index: int, row) -> str:
    """One numbered "stance - basis - status" line of the model background column.

    Args:
        index: 1-based key-message number within the comment.
        row: The pair's diagnostic row.

    Returns:
        The German line; a fixed note for pairs in no model run.
    """
    if not _in_run(row):
        return f"{index}. kein Modelllauf (Beitrag in keinem Lauf)"
    status = _clean(row.status)
    return (f"{index}. {_stance_de(row.proposed_stance)} - {_basis_de(row.basis, 'keine Einordnung')} - "
            f"{STATUS_DE.get(status, status or 'kein Status')}")


def _label_line(row) -> str:
    """The annotator's reference label of one pair with its provenance, in German.

    Args:
        row: The pair's diagnostic row.

    Returns:
        "label (provenance)".
    """
    return f"{_stance_de(row.reference_label)} ({PROVENANCE_DE[row.reference_provenance]})"


def _signals_text(first) -> str:
    """The German "Textsignale" background cell of one comment.

    Args:
        first: Any diagnostic row of the comment (signals are comment-level).

    Returns:
        Four lines: approval formula, residual word count, cues and class.
    """
    cues = [name for name, flag in (("Verneinung", first.negation_cue), ("Ironie", first.irony_cue),
                                    ("Verweis", first.back_reference_cue)) if flag]
    return "\n".join([
        f"Zustimmungsformel: {_display(first.approval_prefix) or 'keine'}",
        f"Wörter außerhalb der Formel: {first.residual_word_count}",
        f"Hinweise: {', '.join(cues) or 'keine'}",
        f"Klasse: {SIGNAL_CLASS_DE[first.signal_class]}",
    ])


def _contribution_cell(first) -> str:
    """The contribution excerpt cell of one comment.

    Args:
        first: Any diagnostic row of the comment.

    Returns:
        The display text, with "[gekürzt]" appended when the source cut it,
        or a fixed German note when no contribution text is available.
    """
    text = _display(first.contribution_text)
    if text is None:
        return "(Beitragstext liegt hier nicht vor: Beitrag in keinem Modelllauf)"
    # "is True" instead of a truth test: the column is nullable boolean and bool(pd.NA) raises TypeError.
    return text + ("\n[gekürzt]" if first.contribution_text_truncated is True else "")


def build_pack_rows(table: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    """Comment rows and key-message rows of the pack, in table order (pure).

    Texts are shown decoded (HTML entities) and Excel-safe; the row
    fingerprints are computed from exactly these displayed texts, which is
    what the reader compares. The parquet keeps the raw texts.

    Args:
        table: The diagnostic table (rows of one comment are contiguous).

    Returns:
        A tuple (comment_rows, km_rows) of dicts keyed by ColumnSpec.key:
        one comment row per comment (numbered G001, G002, ...) and one
        key-message row per pair (numbered G001.1, G001.2, ...).
    """
    comment_rows, km_rows = [], []
    for number, (comment_id, group) in enumerate(table.groupby("comment_id", sort=False), start=1):
        rows = list(group.itertuples())
        first, nr = rows[0], f"G{number:03d}"
        comment_text = _display(first.comment_text)
        key_messages = [_display(r.key_message) for r in rows]
        pair_ids = sorted(group.pair_id)
        evidence = [f'"{_display(q)}"' for r in rows for q in (_as_list(r.comment_evidence) or [])]
        comment_rows.append({
            "nr": nr, "comment_text": comment_text, "contribution_text": _contribution_cell(first),
            "key_messages_overview": "\n".join(f"{i}. {km}" for i, km in enumerate(key_messages, 1)),
            "h_notizen": _lines(_display(r.annotator_note) for r in rows),
            "h_modell": "\n".join(_model_line(i, r) for i, r in enumerate(rows, 1)),
            "h_belege": _lines(evidence), "h_begruendung": _lines(_display(r.explanation) for r in rows),
            "h_ihre_labels": "\n".join(f"{i}. {_label_line(r)}" for i, r in enumerate(rows, 1)),
            "h_signale": _signals_text(first),
            "t_vorschlag": "; ".join(f"{CAUSE_DE[c]}: {n}" for c, n in sorted(Counter(group.heuristic_cause).items())),
            "comment_id": comment_id, "pair_ids": ";".join(pair_ids),
            "row_fingerprint": row_fingerprint(comment_id, comment_text, *pair_ids),
        })
        for index, (row, key_message) in enumerate(zip(rows, key_messages), start=1):
            km_rows.append({
                "nr": f"{nr}.{index}", "comment_text": _short(comment_text), "key_message": key_message,
                "h_ihr_label": _stance_de(row.reference_label),
                "h_herkunft": PROVENANCE_DE[row.reference_provenance],
                "h_modell_label": _stance_de(row.proposed_stance) if _in_run(row) else "kein Lauf",
                "h_modell_basis": _basis_de(row.basis, "-") if _in_run(row) else "kein Lauf",
                "h_wortueberschneidung": int(row.km_term_overlap),
                "pair_id": row.pair_id, "comment_id": comment_id,
                "row_fingerprint": row_fingerprint(row.pair_id, comment_id, key_message),
            })
    return comment_rows, km_rows


def _write_cell(ws, row: int, column: int, value: object, font: Font = FONT) -> None:
    """Write one value through safe_text, with font and wrapping.

    Missing scalars (None, NaN, pd.NA) leave the cell empty, and strings
    that look like formulas are stored as plain text.

    Args:
        ws: The worksheet.
        row: 1-based row.
        column: 1-based column.
        value: The value to write.
        font: Cell font. Defaults to FONT.
    """
    value = _clean(value)
    if isinstance(value, str):
        value = safe_text(value)[0]
    cell = ws.cell(row=row, column=column, value=value)
    if isinstance(value, str) and value.startswith(_FORMULA_LEADS):
        # openpyxl infers data_type "f" (formula) for strings starting with "="; forcing "s" writes the
        # text as a string, and quotePrefix keeps Excel from re-parsing it as a formula when edited.
        cell.data_type = "s"
        cell.quotePrefix = True
    cell.font, cell.alignment = font, WRAP


def _protect(ws) -> None:
    """Protect a text-only sheet (Anleitung, Übersicht) without a password.

    The answer sheets stay unprotected so their column groups can be
    expanded; the reader detects edited texts and IDs by row fingerprints
    instead. Cells marked UNLOCKED stay editable on a protected sheet.

    Args:
        ws: The worksheet to protect.
    """
    ws.protection.sheet = True


def _write_table_sheet(ws, specs: Sequence[ColumnSpec], rows: list[dict], list_ranges: Mapping[str, str]) -> None:
    """Write one answer sheet (Kommentare or Kernbotschaften).

    Row KEY_ROW holds the machine keys (hidden), HEADER_ROW the German
    headers, data starts at FIRST_DATA_ROW. Background and team columns form
    collapsed outline groups; editable cells are unlocked, coloured and get
    a drop-down list when the column has one.

    Args:
        ws: The worksheet to fill.
        specs: The sheet's columns, in order.
        rows: Row dicts from build_pack_rows.
        list_ranges: OPTION_SETS name to its range formula on the Listen
            sheet (from _write_lists).
    """
    for column, spec in enumerate(specs, start=1):
        ws.cell(row=KEY_ROW, column=column, value=spec.key).font = FONT
        header = ws.cell(row=HEADER_ROW, column=column, value=spec.header or None)
        header.font, header.alignment = BOLD, WRAP
        header.fill = HEADER_FILL if spec.group == "main" else BACKGROUND_HEADER_FILL
        dimension = ws.column_dimensions[get_column_letter(column)]
        dimension.width = spec.width
        if spec.group in OUTLINED_GROUPS:
            dimension.outline_level, dimension.hidden = 1, True
        elif spec.group == "hidden":
            dimension.hidden = True
        elif spec.group == "separator":
            # The column right of a collapsed group carries its plus button (summaryRight).
            dimension.collapsed = True
        for offset, values in enumerate(rows):
            row = FIRST_DATA_ROW + offset
            _write_cell(ws, row, column, values.get(spec.key))
            if spec.editable:
                cell = ws.cell(row=row, column=column)
                cell.protection = UNLOCKED
                cell.fill = ANSWER_FILL if spec.group == "main" else TEAM_FILL
        if spec.list_name and rows:
            letter = get_column_letter(column)
            validation = _validation(list_ranges[spec.list_name])
            ws.add_data_validation(validation)
            validation.add(f"{letter}{FIRST_DATA_ROW}:{letter}{FIRST_DATA_ROW + len(rows) - 1}")
    ws.row_dimensions[KEY_ROW].hidden = True
    # Freezing at C3 keeps the key/header rows and the columns Nr. and Kommentar visible while scrolling.
    ws.freeze_panes = f"C{FIRST_DATA_ROW}"
    last = get_column_letter(len(specs))
    ws.auto_filter.ref = f"A{HEADER_ROW}:{last}{HEADER_ROW + max(len(rows), 1)}"
    # openpyxl writes sheetFormatPr.outlineLevelCol from this value before it computes it from the columns,
    # so without presetting it the attribute would be missing and Excel would not show the group controls.
    ws.column_dimensions.max_outline = 1
    # Place each group's plus/minus button on the column to the right of the group (the separator column).
    ws.sheet_properties.outlinePr.summaryRight = True


def _validation(formula: str) -> DataValidation:
    """A drop-down list validation that rejects values not in the list.

    Args:
        formula: Range formula of the allowed values (from _write_lists).

    Returns:
        The DataValidation; empty cells are allowed.
    """
    return DataValidation(type="list", formula1=formula, allow_blank=True, showErrorMessage=True,
                          errorTitle="Ungültiger Wert", error="Bitte einen Wert aus der Liste wählen.")


def _write_lists(ws) -> dict[str, str]:
    """Write the option labels of OPTION_SETS to the Listen sheet and hide it.

    Each option set gets one column: its name in row 1 and the German labels
    below.

    Args:
        ws: The Listen worksheet.

    Returns:
        OPTION_SETS name to an absolute range formula such as
        "=Listen!$A$2:$A$7", for use in data validations.
    """
    ranges = {}
    for column, (name, options) in enumerate(OPTION_SETS.items(), start=1):
        letter = get_column_letter(column)
        ws.cell(row=1, column=column, value=name).font = BOLD
        for row, (_, label) in enumerate(options, start=2):
            ws.cell(row=row, column=column, value=label).font = FONT
        ranges[name] = f"={LISTS_SHEET}!${letter}$2:${letter}${len(options) + 1}"
    ws.sheet_state = "hidden"
    return ranges


ANLEITUNG_TEXT: tuple[tuple[str, str], ...] = (
    ("title", "Rückfrage: Pauschale Zustimmung in Kommentaren"),
    ("heading", "Worum es geht"),
    ("text", "Viele Kommentare stimmen einem Beitrag pauschal zu, ohne auf einzelne Kernbotschaften einzugehen."),
    ("text", "Wir möchten mit Ihnen festlegen, wie solche Kommentare je Kernbotschaft gelabelt werden sollen."),
    ("text", "Ihre bisherigen Labels werden dabei nicht verändert."),
    ("text", "Eine spätere Neubewertung würde in einer neuen, getrennt versionierten Datei abgelegt."),
    ("heading", "Drei mögliche Regeln"),
    ("text", "Regel 1 (immer neutral): Eine pauschale Zustimmung erhält bei jeder Kernbotschaft das Label „neutral“."),
    ("text", "Regel 2 (immer dafür): Eine pauschale Zustimmung erhält bei jeder Kernbotschaft das Label „dafür“."),
    ("text", "Regel 3 (neutral, außer mit eigenem Inhalt): Eine pauschale Zustimmung ohne eigenen Inhalt erhält "
             "„neutral“, eine pauschale Zustimmung mit eigenem Inhalt erhält „dafür“."),
    ("text", "Bei Regel 3 fragen wir zusätzlich nach der Reichweite: Gilt „dafür“ dann für alle Kernbotschaften "
             "oder nur für die inhaltlich angesprochenen?"),
    ("text", "Unter jeder Regel gilt: Kernbotschaften, die der Kommentar inhaltlich anspricht, werden für sich bewertet."),
    ("text", "Es gibt keine richtige oder falsche Antwort; uns interessiert Ihre Einschätzung."),
    ("heading", "Zwei erfundene Beispiele"),
    ("text", "Beitrag mit drei Kernbotschaften: 1. Fahrradständer aufstellen, 2. Parkplätze entfernen, "
             "3. Bäume pflanzen."),
    ("text", "Kommentar A: „Sehr gut!“ (pauschale Zustimmung ohne eigenen Inhalt)"),
    ("text", "Regel 1: alle drei Kernbotschaften „neutral“. Regel 2: alle drei „dafür“. "
             "Regel 3: alle drei „neutral“, weil kein eigener Inhalt hinzukommt."),
    ("text", "Kommentar B: „Sehr gut! Am Bahnhof fehlen wirklich Fahrradständer.“ "
             "(pauschale Zustimmung mit eigenem Inhalt zur ersten Kernbotschaft)"),
    ("text", "Regel 1: Fahrradständer wird für sich bewertet (hier „dafür“), Parkplätze und Bäume „neutral“. "
             "Regel 2: alle drei „dafür“."),
    ("text", "Regel 3 mit Reichweite „alle Kernbotschaften“: alle drei „dafür“. "
             "Regel 3 mit Reichweite „nur inhaltlich angesprochene“: Fahrradständer „dafür“, "
             "Parkplätze und Bäume „neutral“."),
    ("text", "So würden Sie Kommentar B eintragen: Einordnung „Pauschale Zustimmung mit eigenem Inhalt zu einzelnen "
             "Kernbotschaften“; Regel und Reichweite nach Ihrer Einschätzung; im Blatt „Kernbotschaften“ "
             "Fahrradständer „ja“, Parkplätze „nein“, Bäume „nein“."),
    ("heading", "So gehen Sie vor"),
    ("text", "1. Blatt „Kommentare“: Lesen Sie zuerst nur den Kommentar, den Beitrag und die Kernbotschaften, "
             "und füllen Sie dann die gelben Zellen aus."),
    ("text", "2. Erst danach können Sie bei Bedarf die Spaltengruppe „Hintergrund“ einblenden: Plus-Zeichen über "
             "der schmalen Spalte rechts von „Ihre Anmerkung“. Dort stehen Ihre damaligen Notizen, die Einordnung "
             "des Modells und Ihre bisherigen Labels."),
    ("text", "3. Blatt „Kernbotschaften“: Geben Sie je Kernbotschaft an, ob der Kommentar sie inhaltlich anspricht."),
    ("text", "4. Beantworten Sie unten auf diesem Blatt die zwei allgemeinen Fragen."),
    ("text", "Bitte ändern Sie keine Kommentar- oder Kernbotschaftstexte; Sortieren und Filtern sind unproblematisch."),
    ("heading", "Legende"),
    ("answer_legend", "Gelb: bitte von Ihnen ausfüllen (Auswahlliste oder freier Text)."),
    ("team_legend", "Hellblau: Notizen des Projektteams, bitte leer lassen."),
    ("text", "Die Spaltengruppe „Projektteam“ ganz rechts ist ausgeblendet und für Sie nicht relevant."),
    ("heading", "Allgemeine Fragen"),
)


def _write_anleitung(ws, list_ranges: Mapping[str, str], workbook: Workbook) -> None:
    """Write the instruction sheet with the two global answer cells, then protect it.

    The answer cells are unlocked and registered as workbook defined names
    (GLOBAL_ANSWER_NAMES), so the reader finds them regardless of row.

    Args:
        ws: The Anleitung worksheet.
        list_ranges: Output of _write_lists (for the rule drop-down).
        workbook: The workbook that receives the defined names.
    """
    ws.column_dimensions["A"].width, ws.column_dimensions["B"].width = 30, 110
    row = 1
    for kind, text in ANLEITUNG_TEXT:
        if kind == "heading":
            row += 1
        font = {"title": TITLE, "heading": BOLD}.get(kind, FONT)
        _write_cell(ws, row, 1 if kind in ("title", "heading") else 2, text, font)
        if kind in ("answer_legend", "team_legend"):
            ws.cell(row=row, column=1).fill = ANSWER_FILL if kind == "answer_legend" else TEAM_FILL
        row += 1
    for name, label in zip(GLOBAL_ANSWER_NAMES, ("Bevorzugte Regel insgesamt", "Begründung")):
        row += 1
        _write_cell(ws, row, 1, label, BOLD)
        answer = ws.cell(row=row, column=2)
        answer.font, answer.alignment, answer.fill, answer.protection = FONT, WRAP, ANSWER_FILL, UNLOCKED
        workbook.defined_names[name] = DefinedName(name, attr_text=f"{ANLEITUNG_SHEET}!$B${row}")
        if name == "g_regel_insgesamt":
            validation = _validation(list_ranges["regel"])
            ws.add_data_validation(validation)
            validation.add(f"B{row}")
        else:
            ws.row_dimensions[row].height = 90
    _protect(ws)


def _overview_rows(summary: Mapping) -> list[tuple[str, object]]:
    """The (label, value) rows of the Übersicht sheet, taken from the safe summary.

    Args:
        summary: The safe summary.

    Returns:
        Rows in display order; rows whose label is "heading" are section
        headings with the heading text as value.
    """
    scored = summary["scored_general_approval"]
    notes = summary["annotator_notes"]
    pack = summary["pack"]

    def block(label: str, values: Mapping) -> list[tuple[str, object]]:
        """One German result line for a _scored_block (total, run or fold)."""
        labels = ", ".join(f"{STANCE_DE.get(k, k)} {v}" for k, v in values["reference_labels"].items())
        return [(label, f"{values['correct']} von {values['pairs']} Paaren stimmen mit Ihrem Label überein "
                        f"({values['comments']} Kommentare); Ihre Labels: {labels}")]

    rows: list[tuple[str, object]] = [
        ("heading", "Modell-Einordnung \"pauschale Zustimmung\", angenommen, mit Ihrem Label"),
        *block("Insgesamt", scored["total"]),
        *[row for name, values in scored["by_run"].items() for row in block(f"Lauf {name}", values)],
        *[row for fold, values in scored["by_fold"].items() for row in block(f"Fold {fold}", values)],
        ("heading", "Ihre Notizen \"generell/allgemein zustimmend\""),
        ("Paare mit Notiz (breite Suche)", notes["general_approval_broad"]["pairs"]),
        ("Paare mit Notiz (enge Suche, frühere Zählung)", notes["general_approval_strict"]["pairs"]),
        ("Davon Kommentare zu Beiträgen ohne Modelllauf", notes["general_approval_broad_outside_runs"]["comments"]),
        ("heading", "Dieses Paket"),
        ("Kommentare", pack["comments"]),
        ("Zeilen im Blatt Kernbotschaften", pack["key_message_rows"]),
        ("Kommentare ohne Modelllauf", pack["comments_outside_runs"]),
    ]
    return rows


def _write_overview(ws, summary: Mapping) -> None:
    """Write the Übersicht sheet (fixed values, no formulas), then protect it.

    Args:
        ws: The Übersicht worksheet.
        summary: The safe summary.
    """
    ws.column_dimensions["A"].width, ws.column_dimensions["B"].width = 48, 100
    _write_cell(ws, 1, 1, "Übersicht (feste Werte, keine Formeln)", TITLE)
    row = 3
    for label, value in _overview_rows(summary):
        if label == "heading":
            row += 1
            _write_cell(ws, row, 1, value, BOLD)
        else:
            _write_cell(ws, row, 1, label)
            _write_cell(ws, row, 2, value)
        row += 1
    _write_cell(ws, row + 1, 1, f"Quelle: {SUMMARY_FILE}, {summary['created_utc']}")
    _protect(ws)


def _write_meta(ws, summary: Mapping, comment_rows: list[dict], km_rows: list[dict]) -> None:
    """Write the Meta sheet of key/value pairs that the reader validates against.

    It holds the schema versions, provenance hashes, the expected comment
    and pair IDs, and the header of every column.

    Args:
        ws: The Meta worksheet.
        summary: The safe summary.
        comment_rows: Comment rows from build_pack_rows.
        km_rows: Key-message rows from build_pack_rows.
    """
    entries = [
        ("pack_schema_version", PACK_SCHEMA_VERSION),
        ("diagnostic_schema_version", DIAGNOSTIC_SCHEMA_VERSION),
        ("signals_version", SIGNALS_VERSION),
        ("created_utc", summary["created_utc"]),
        ("diagnostic_content_sha256", summary["diagnostic_content_sha256"]),
        ("run_ids", json.dumps(summary["run_ids"], sort_keys=True)),
        ("comment_ids", ";".join(r["comment_id"] for r in comment_rows)),
        ("pair_ids", ";".join(r["pair_id"] for r in km_rows)),
    ]
    entries += [(f"column:{sheet}:{spec.key}", spec.header)
                for sheet, specs in ((COMMENTS_SHEET, COMMENT_COLUMNS), (KM_SHEET, KM_COLUMNS)) for spec in specs]
    for row, (key, value) in enumerate(entries, start=1):
        _write_cell(ws, row, 1, key)
        _write_cell(ws, row, 2, value)
    # "veryHidden" sheets cannot be unhidden from Excel's UI (only via VBA), unlike "hidden".
    ws.sheet_state = "veryHidden"


def write_review_pack(table: pd.DataFrame, summary: Mapping, path: Path) -> None:
    """Write the German review pack to a new file (never overwrites an existing one).

    The workbook's created/modified properties are set to the summary's
    created_utc, so the file content does not depend on the wall clock.

    Args:
        table: The diagnostic table.
        summary: The safe summary (feeds the Übersicht and Meta sheets).
        path: Target .xlsx path; must not exist yet.

    Raises:
        FileExistsError: If path already exists.
    """
    comment_rows, km_rows = build_pack_rows(table)
    workbook = Workbook()
    sheets = {name: (workbook.active if i == 0 else workbook.create_sheet(name)) for i, name in enumerate(SHEET_ORDER)}
    sheets[ANLEITUNG_SHEET].title = ANLEITUNG_SHEET
    list_ranges = _write_lists(sheets[LISTS_SHEET])
    _write_anleitung(sheets[ANLEITUNG_SHEET], list_ranges, workbook)
    _write_table_sheet(sheets[COMMENTS_SHEET], COMMENT_COLUMNS, comment_rows, list_ranges)
    _write_table_sheet(sheets[KM_SHEET], KM_COLUMNS, km_rows, list_ranges)
    _write_overview(sheets[OVERVIEW_SHEET], summary)
    _write_meta(sheets[META_SHEET], summary, comment_rows, km_rows)
    workbook.active = 0
    created = datetime.strptime(summary["created_utc"], CREATED_UTC_FORMAT)
    properties = workbook.properties
    properties.creator = properties.lastModifiedBy = "stance-detection-app"
    properties.created = properties.modified = created
    properties.title = "Rückfrage pauschale Zustimmung"
    # Mode "xb" (exclusive create) fails if the file exists; Workbook.save() would silently overwrite it.
    with open(path, "xb") as f, ZipFile(f, "w", ZIP_DEFLATED, allowZip64=True) as archive:
        # ExcelWriter directly: openpyxl's save() would overwrite `modified` with the wall clock.
        # write_data() leaves closing the archive to the "with" block.
        ExcelWriter(workbook, archive).write_data()


# --------------------------------------------------------------------------
# Reading the annotator's answers (parse and validate only)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ReviewPackAnswers:
    """Parsed answers; option labels mapped back to OPTION_SETS codes, empty cells None.

    Attributes:
        meta: The Meta sheet's key/value pairs.
        comments: One row per comment in pack order: comment_id, nr,
            pair_ids (list) and the editable Kommentare columns.
        key_messages: One row per pair in pack order: pair_id, comment_id,
            nr and the editable Kernbotschaften columns.
        global_answers: GLOBAL_ANSWER_NAMES to the answer (a rule code for
            g_regel_insgesamt, free text for g_begruendung) or None.
    """

    meta: dict[str, str]
    comments: pd.DataFrame
    key_messages: pd.DataFrame
    global_answers: dict[str, str | None]


def _label_to_code(list_name: str, value: object, where: str) -> str | None:
    """Map a drop-down label back to its OPTION_SETS code.

    Args:
        list_name: The OPTION_SETS key.
        value: The cell value.
        where: Location used in the error message.

    Returns:
        The code, or None for an empty or blank cell.

    Raises:
        ValueError: If the value is not one of the list's labels.
    """
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    codes = {label: code for code, label in OPTION_SETS[list_name]}
    code = codes.get(str(value).strip())
    if code is None:
        raise ValueError(f"{where}: unknown value {value!r}")
    return code


def _cell_text(value: object) -> str | None:
    """A read cell value as text with "\\n" newlines (as safe_text wrote them).

    Args:
        value: The cell value (numbers are stringified).

    Returns:
        The text, or None for an empty cell.
    """
    if value is None:
        return None
    return str(value).replace("\r\n", "\n").replace("\r", "\n")


def _read_meta(workbook) -> dict[str, str]:
    """Read and check the Meta sheet.

    Args:
        workbook: The loaded pack.

    Returns:
        Key to value (empty cells as "").

    Raises:
        ValueError: If the Meta sheet is missing, the pack schema version is
            not PACK_SCHEMA_VERSION, or comment_ids / pair_ids are empty.
    """
    if META_SHEET not in workbook.sheetnames:
        raise ValueError("Not a review pack: Meta sheet missing")
    meta = {}
    for key, value in workbook[META_SHEET].iter_rows(min_col=1, max_col=2, values_only=True):
        if key is not None:
            meta[str(key)] = "" if value is None else str(value)
    if meta.get("pack_schema_version") != PACK_SCHEMA_VERSION:
        raise ValueError(f"Unsupported pack schema version {meta.get('pack_schema_version')!r}; "
                         f"expected {PACK_SCHEMA_VERSION!r}")
    missing = [key for key in ("comment_ids", "pair_ids") if not meta.get(key)]
    if missing:
        raise ValueError(f"Meta sheet lacks {missing}")
    return meta


def _read_rows(ws, specs: Sequence[ColumnSpec], required: Sequence[str]) -> list[dict]:
    """Read the data rows of an answer sheet, locating columns by their machine key.

    Because columns are found via the hidden key row, the annotator may
    reorder columns and rows; fully empty rows are skipped.

    Args:
        ws: The worksheet.
        specs: The sheet's ColumnSpecs; only their keys are read.
        required: Keys that must be present.

    Returns:
        One dict per non-empty data row, keyed by the spec keys found.

    Raises:
        ValueError: If a key appears twice or a required key is missing.
    """
    keys = [cell.value for cell in ws[KEY_ROW]]
    positions = {}
    for index, key in enumerate(keys):
        if key is None:
            continue
        if key in positions:
            raise ValueError(f"{ws.title}: column key {key!r} appears twice")
        positions[key] = index
    missing = [key for key in required if key not in positions]
    if missing:
        raise ValueError(f"{ws.title}: missing columns {missing}")
    wanted = [spec.key for spec in specs if spec.key in positions]
    rows = []
    for values in ws.iter_rows(min_row=FIRST_DATA_ROW, values_only=True):
        # Defensive: a value tuple shorter than the key row yields None for the cells beyond its end.
        record = {key: _cell_text(values[positions[key]]) if positions[key] < len(values) else None for key in wanted}
        if any(v is not None and v.strip() for v in record.values()):
            rows.append(record)
    return rows


def _decisions(record: dict, specs: Sequence[ColumnSpec], sheet: str) -> dict:
    """The editable cells of one row as answer values.

    Args:
        record: A row from _read_rows.
        specs: The sheet's ColumnSpecs.
        sheet: Sheet name, used in error messages.

    Returns:
        Per editable column key: the OPTION_SETS code for list columns, the
        stripped text for free-text columns, None when empty.

    Raises:
        ValueError: If a list cell holds a value not in its list.
    """
    decided = {}
    for spec in specs:
        if not spec.editable:
            continue
        value, where = record.get(spec.key), f"{sheet} row {record.get('nr')}, column {spec.header!r}"
        decided[spec.key] = (_label_to_code(spec.list_name, value, where) if spec.list_name
                             else (value.strip() or None) if isinstance(value, str) else None)
    return decided


def _answer_frame(records: list[dict]) -> pd.DataFrame:
    """Build an object-dtype frame, so unanswered cells stay None instead of becoming NaN.

    Args:
        records: Parsed answer rows.

    Returns:
        The frame with a fresh 0-based index.
    """
    return pd.DataFrame(records, dtype=object).reset_index(drop=True)


def _check_ids(rows: list[dict], id_key: str, expected: list[str], sheet: str) -> None:
    """Check that a sheet holds exactly the expected IDs, each once.

    Args:
        rows: Rows from _read_rows.
        id_key: The ID column key (comment_id or pair_id).
        expected: The IDs recorded on the Meta sheet.
        sheet: Sheet name, used in error messages.

    Raises:
        ValueError: If a row lacks its ID, an ID repeats, or the set of IDs
            differs from the expected set.
    """
    seen = set()
    for record in rows:
        identifier = record.get(id_key)
        if not identifier:
            raise ValueError(f"{sheet} row {record.get('nr')}: missing {id_key}")
        if identifier in seen:
            raise ValueError(f"{sheet} row {record.get('nr')}: duplicate {id_key} {identifier}")
        seen.add(identifier)
    if seen != set(expected):
        absent, extra = sorted(set(expected) - seen), sorted(seen - set(expected))
        raise ValueError(f"{sheet}: rows missing for {absent} or unexpected rows {extra}")


def _parse_comments(ws, expected: list[str]) -> pd.DataFrame:
    """Parse and validate the Kommentare sheet.

    Args:
        ws: The Kommentare worksheet.
        expected: Comment IDs from the Meta sheet, in pack order.

    Returns:
        One answer row per comment, in pack order (see ReviewPackAnswers).

    Raises:
        ValueError: If columns or rows are missing or duplicated, a row's
            fingerprint no longer matches its comment text and IDs, or an
            answer is not a valid list value.
    """
    editable = [s.key for s in COMMENT_COLUMNS if s.editable]
    rows = _read_rows(ws, COMMENT_COLUMNS, ["nr", "comment_text", "comment_id", "pair_ids", "row_fingerprint",
                                            *editable])
    _check_ids(rows, "comment_id", expected, COMMENTS_SHEET)
    parsed = []
    for record in rows:
        # filter(None, ...) drops empty strings, e.g. from an empty cell.
        pair_ids = sorted(filter(None, (record["pair_ids"] or "").split(";")))
        # Same inputs as in build_pack_rows; a mismatch means the text or IDs were edited.
        if row_fingerprint(record["comment_id"], record["comment_text"] or "", *pair_ids) != record["row_fingerprint"]:
            raise ValueError(f"{COMMENTS_SHEET} row {record['nr']}: fingerprint mismatch (text or IDs changed)")
        parsed.append({"comment_id": record["comment_id"], "nr": record["nr"], "pair_ids": pair_ids,
                       **_decisions(record, COMMENT_COLUMNS, COMMENTS_SHEET)})
    order = {cid: i for i, cid in enumerate(expected)}
    return _answer_frame(sorted(parsed, key=lambda r: order[r["comment_id"]]))


def _parse_key_messages(ws, expected: list[str]) -> pd.DataFrame:
    """Parse and validate the Kernbotschaften sheet.

    Args:
        ws: The Kernbotschaften worksheet.
        expected: Pair IDs from the Meta sheet, in pack order.

    Returns:
        One answer row per pair, in pack order (see ReviewPackAnswers).

    Raises:
        ValueError: If columns or rows are missing or duplicated, a row's
            fingerprint no longer matches its key message and IDs, or an
            answer is not a valid list value.
    """
    rows = _read_rows(ws, KM_COLUMNS, ["nr", "key_message", "pair_id", "comment_id", "row_fingerprint",
                                       "d_km_angesprochen"])
    _check_ids(rows, "pair_id", expected, KM_SHEET)
    parsed = []
    for record in rows:
        fingerprint = row_fingerprint(record["pair_id"], record["comment_id"] or "", record["key_message"] or "")
        if fingerprint != record["row_fingerprint"]:
            raise ValueError(f"{KM_SHEET} row {record['nr']}: fingerprint mismatch (text or IDs changed)")
        parsed.append({"pair_id": record["pair_id"], "comment_id": record["comment_id"], "nr": record["nr"],
                       **_decisions(record, KM_COLUMNS, KM_SHEET)})
    order = {pid: i for i, pid in enumerate(expected)}
    return _answer_frame(sorted(parsed, key=lambda r: order[r["pair_id"]]))


def _global_answers(workbook) -> dict[str, str | None]:
    """Read the two global answers via their workbook defined names.

    Args:
        workbook: The loaded pack.

    Returns:
        g_regel_insgesamt as a rule code and g_begruendung as stripped text;
        None when empty.

    Raises:
        ValueError: If a defined name is missing or the rule is not a valid
            list value.
    """
    answers = {}
    for name in GLOBAL_ANSWER_NAMES:
        if name not in workbook.defined_names:
            raise ValueError(f"Defined name {name!r} missing")
        # destinations yields (sheet title, "$B$40")-style pairs; cell lookup needs "B40".
        sheet, coordinate = next(iter(workbook.defined_names[name].destinations))
        value = _cell_text(workbook[sheet][coordinate.replace("$", "")].value)
        answers[name] = (_label_to_code("regel", value, f"{ANLEITUNG_SHEET}: Bevorzugte Regel insgesamt")
                         if name == "g_regel_insgesamt" else (value.strip() or None) if value else None)
    return answers


def read_review_pack(path: Path) -> ReviewPackAnswers:
    """Parse and validate a filled pack; the workbook is only read, never saved.

    Args:
        path: The filled .xlsx review pack.

    Returns:
        The parsed answers.

    Raises:
        ValueError: If the file is not a valid pack of this schema, IDs or
            protected texts were changed, a pair was moved to another
            comment, or an answer is not a valid list value.
        KeyError: If the Kommentare or Kernbotschaften sheet is missing.
    """
    workbook = load_workbook(path, data_only=False)
    try:
        meta = _read_meta(workbook)
        comments = _parse_comments(workbook[COMMENTS_SHEET], meta["comment_ids"].split(";"))
        key_messages = _parse_key_messages(workbook[KM_SHEET], meta["pair_ids"].split(";"))
        # Each pair must still sit under the comment whose pair_ids list it (catches rows moved between comments).
        owner = {pid: cid for cid, pids in zip(comments.comment_id, comments.pair_ids) for pid in pids}
        for record in key_messages.itertuples():
            if owner.get(record.pair_id) != record.comment_id:
                raise ValueError(f"{KM_SHEET} row {record.nr}: pair does not belong to comment {record.comment_id}")
        return ReviewPackAnswers(meta, comments, key_messages, _global_answers(workbook))
    finally:
        workbook.close()


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ReviewRunResult:
    """Where one review run wrote its artifacts, and its safe summary.

    Attributes:
        output_dir: The newly created output directory.
        files: Artifact role ("diagnostic", "summary", "pack", "hashes") to
            its path.
        summary: The safe summary that was written.
    """

    output_dir: Path
    files: dict[str, Path]
    summary: dict


def _input_paths(runs: Sequence[RunSpec], reference_path: Path, folds_path: Path, annotations_path: Path,
                 source_paths: Sequence[Path]) -> dict[str, Path]:
    """Name every input file, for hashing and for the summary's input_sha256.

    Args:
        runs: The runs to read.
        reference_path: Reference dataset parquet.
        folds_path: Fold assignment JSON.
        annotations_path: Annotation SQLite database.
        source_paths: Extraction source parquets.

    Returns:
        Input name ("reference", "run:<name>:predictions", "source:<i>", ...)
        to path.
    """
    paths = {"reference": Path(reference_path), "folds": Path(folds_path), "annotations": Path(annotations_path)}
    for spec in runs:
        paths[f"run:{spec.name}:predictions"] = Path(spec.predictions)
        paths[f"run:{spec.name}:expected"] = Path(spec.expected)
    for index, path in enumerate(source_paths):
        paths[f"source:{index}"] = Path(path)
    return paths


def _hash_inputs(paths: Mapping[str, Path]) -> dict[str, dict]:
    """Hash every input file.

    Args:
        paths: Output of _input_paths.

    Returns:
        Input name to {"path": ..., "sha256": ...}.

    Raises:
        FileNotFoundError: If any input is not an existing file.
    """
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing input files: {missing}")
    return {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in paths.items()}


def _write_json(path: Path, payload: Mapping) -> None:
    """Write pretty-printed UTF-8 JSON to a new file.

    Args:
        path: Target path; must not exist yet (mode "x").
        payload: JSON-serializable mapping; key order is preserved.

    Raises:
        FileExistsError: If path already exists.
    """
    with open(path, "x", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=False)
        f.write("\n")


def run_general_approval_review(*, runs: Sequence[RunSpec] = DEFAULT_RUNS, reference_path: Path,
                                folds_path: Path, annotations_path: Path = DEFAULT_ANNOTATIONS_DB,
                                source_paths: Sequence[Path] = DEFAULT_SOURCES, role: str = "primary",
                                tiers: Sequence[str] = ("A", "B"), output_dir: Path | None = None,
                                now: datetime | None = None) -> ReviewRunResult:
    """Build and write the diagnostic table, safe summary and review pack into a new directory.

    The directory must not exist (its parent must). artifact_hashes.json is
    written last as the completion marker; on failure the partial directory
    is left without it and nothing is deleted or overwritten.

    Args:
        runs: Saved runs to analyse. Defaults to DEFAULT_RUNS.
        reference_path: Reference dataset parquet.
        folds_path: Fold assignment JSON.
        annotations_path: Annotation SQLite database (opened read-only).
            Defaults to DEFAULT_ANNOTATIONS_DB.
        source_paths: Frozen extraction source parquets. Defaults to
            DEFAULT_SOURCES.
        role: Reference role selection. Defaults to "primary".
        tiers: Reference tiers. Defaults to ("A", "B").
        output_dir: Directory to create. Defaults to
            EXPERIMENTS_DIR / "general_approval_review_<created_utc>".
        now: Creation time (for tests); a naive datetime is read as local
            time. Defaults to the current UTC time.

    Returns:
        The output directory, the written files and the safe summary.

    Raises:
        FileNotFoundError: If an input file is missing or the output
            directory's parent does not exist.
        FileExistsError: If the output directory already exists.
        RuntimeError: If an input file changed while the review was built.
        ValueError: If the inputs are inconsistent (see
            build_diagnostic_table and load_runs).
    """
    created_utc = (now or datetime.now(timezone.utc)).astimezone(timezone.utc).strftime(CREATED_UTC_FORMAT)
    paths = _input_paths(runs, reference_path, folds_path, annotations_path, source_paths)
    inputs = _hash_inputs(paths)
    annotations = load_annotations(paths["annotations"])
    diagnostic = build_diagnostic_table(
        load_runs(runs), pd.read_parquet(paths["reference"]), annotations,
        related_sentences=load_sources(source_paths), fold_of=load_fold_lookup(paths["folds"]),
        role=role, tiers=tiers)
    summary = build_safe_summary(diagnostic, annotations, inputs=inputs, created_utc=created_utc,
                                 role=role, tiers=tiers)
    # Re-hash: the outputs must describe exactly the input bytes recorded in the summary.
    if _hash_inputs(paths) != inputs:
        raise RuntimeError("An input file changed while the review was being built")

    output_dir = Path(output_dir) if output_dir else EXPERIMENTS_DIR / f"{OUTPUT_DIR_PREFIX}{created_utc}"
    output_dir.mkdir(parents=False, exist_ok=False)
    files = {"diagnostic": output_dir / DIAGNOSTIC_FILE, "summary": output_dir / SUMMARY_FILE,
             "pack": output_dir / PACK_FILE_TEMPLATE.format(created_utc=created_utc)}
    with open(files["diagnostic"], "xb") as f:
        diagnostic.pairs.to_parquet(f, index=False)
    _write_json(files["summary"], summary)
    write_review_pack(diagnostic.pairs, summary, files["pack"])
    files["hashes"] = output_dir / HASHES_FILE
    _write_json(files["hashes"], {
        "created_utc": created_utc, "complete": True,
        "files": {path.name: sha256_file(path) for key, path in files.items() if key != "hashes"},
    })
    return ReviewRunResult(output_dir, files, summary)
