"""Annotation pack for the general-reaction split (docs/reaction_split_plan.md, phase 1).

Usage:
    uv run cli.py reaction-split-pack
    uv run cli.py reaction-split-read --pack <filled .xlsx>

The annotator splits each comment into a general reaction (a blanket
approval or rejection of the whole contribution, e.g. "Super Idee!",
"Bitte nicht.") and the rest, gives the reaction's polarity and the words
that signal it, and then judges the rest text's stance on every key
message. Her labels define the split stage of the pipeline (phase 3) and
are the reference it is evaluated against (phase 4).

The population has two parts:

  - the 57 comments of the 2026-10-07 general-approval review pack, read
    from that run's frozen diagnostic table, whose hash is checked against
    the run's artifact_hashes.json;
  - every other fold-0 comment of the frozen fold-0 extraction source, so
    the split prompt has more development data (fold 0 is the only fold it
    may be tuned on). Their key-message IDs, pair IDs and comment IDs are
    derived exactly as for the model runs; the derivation is checked on the
    fold-0 comments both parts share.

Only IDs and texts are read: the pack is blank and shows no model output,
no earlier labels and no heuristic signals, so her answers are not anchored
on them.

No inference and no network; .env is never loaded. Outputs are created
exclusively in a new directory and never overwritten; artifact_hashes.json
is written last as the completion marker. Reference labels are never
changed: returned answers go into their own versioned label files.

Folds 1-4 are held-out data: labels on their comments may be used to
evaluate the split stage, never to tune its prompt.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import pandas as pd
from openpyxl import Workbook, load_workbook

from benchmark import load_frozen_comments
from dataset_builder import compute_pair_id
from fold_builder import load_folds
from review_pack_common import (
    CREATED_UTC_FORMAT,
    HASHES_FILE,
    ColumnSpec,
    answer_frame,
    check_ids,
    clean,
    decisions,
    display,
    normalize_whitespace,
    read_meta,
    read_rows,
    row_fingerprint,
    save_workbook_exclusive,
    sha256_file,
    write_artifact_hashes,
    write_json,
    write_lists,
    write_meta,
    write_table_sheet,
    write_text_sheet,
)

PACK_SCHEMA_VERSION = "reaction_split_pack_v1"
LABELS_SCHEMA_VERSION = "reaction_split_labels_v1"

EXPERIMENTS_DIR = Path("dataset/experiments")
DEFAULT_POPULATION_DIR = EXPERIMENTS_DIR / "general_approval_review_20261007T133543733476Z"
POPULATION_FILE = "diagnostic_pairs.parquet"
DEFAULT_EXTRA_SOURCE = EXPERIMENTS_DIR / "screening_subset_fold0_source.parquet"
DEFAULT_FOLDS_PATH = Path("dataset/folds_v1.json")
EXTRA_FOLD = 0
# The frozen source parquets cut contributionContent at this many characters.
CONTRIBUTION_TEXT_LIMIT = 1000
PACK_DIR_PREFIX = "reaction_split_pack_"
PACK_FILE_TEMPLATE = "reaction_split_pack_v1_{created_utc}.xlsx"
PACK_SUMMARY_FILE = "reaction_split_pack.summary.json"
LABELS_DIR_PREFIX = "reaction_split_labels_"
LABELS_COMMENTS_FILE = "reaction_split_labels_comments.parquet"
LABELS_KM_FILE = "reaction_split_labels_key_messages.parquet"
LABELS_SUMMARY_FILE = "reaction_split_labels.summary.json"

# Only these columns are read from the frozen diagnostic table; its model,
# reference and heuristic columns never enter this module.
POPULATION_COLUMNS = (
    "pair_id", "comment_id", "contribution_id", "fold", "key_message_id", "key_message", "comment_text",
    "contribution_text", "contribution_text_truncated",
)

HELDOUT_NOTE = (
    "Folds 1-4 are held-out evaluation data. Labels on their comments may evaluate the split stage "
    "but must not be used to tune its prompt."
)

ANLEITUNG_SHEET, COMMENTS_SHEET, KM_SHEET = "Anleitung", "Kommentare", "Kernbotschaften"
LISTS_SHEET, META_SHEET = "Listen", "Meta"
SHEET_ORDER = (ANLEITUNG_SHEET, COMMENTS_SHEET, KM_SHEET, LISTS_SHEET, META_SHEET)

OPTION_SETS: dict[str, tuple[tuple[str, str], ...]] = {
    "reaktion": (("yes", "ja"), ("no", "nein"), ("unclear", "unklar")),
    "polaritaet": (("positive", "positiv"), ("negative", "negativ"), ("unclear", "unklar")),
    "trennbar": (("yes", "ja"), ("no", "nein")),
    "haltung": (
        ("in_favor", "dafür"),
        ("against", "dagegen"),
        ("neutral", "neutral (angesprochen, ohne Position)"),
        ("unaddressed", "nicht angesprochen"),
        ("unclear", "unklar"),
    ),
}
DIRECTIONAL_OR_NEUTRAL = frozenset({"in_favor", "against", "neutral"})

COMMENT_COLUMNS = (
    ColumnSpec("nr", "Nr.", 8),
    ColumnSpec("comment_text", "Kommentar", 60),
    ColumnSpec("contribution_text", "Beitrag (Auszug)", 50),
    ColumnSpec("key_messages_overview", "Kernbotschaften des Beitrags", 45),
    ColumnSpec("d_reaktion", "Enthält der Kommentar eine allgemeine Reaktion?", 20, editable=True,
               list_name="reaktion"),
    ColumnSpec("d_polaritaet", "Falls ja: positiv oder negativ?", 18, editable=True, list_name="polaritaet"),
    ColumnSpec("d_hinweis_zitat", "Hinweis: Wörter der allgemeinen Reaktion (wörtlich kopiert, je Stelle eine Zeile)",
               34, editable=True),
    ColumnSpec("d_hinweis_begruendung", "Hinweis: kurze Begründung", 34, editable=True),
    ColumnSpec("d_resttext", "Resttext ohne allgemeine Reaktion (wörtlich kopiert, je Teil eine Zeile)", 45,
               editable=True),
    ColumnSpec("d_trennbar", "Lässt sich der Kommentar sauber trennen?", 18, editable=True, list_name="trennbar"),
    ColumnSpec("d_anmerkung", "Ihre Anmerkung", 40, editable=True),
    ColumnSpec("comment_id", "comment_id", 18, hidden=True),
    ColumnSpec("pair_ids", "pair_ids", 18, hidden=True),
    ColumnSpec("row_fingerprint", "row_fingerprint", 18, hidden=True),
)
KM_COLUMNS = (
    ColumnSpec("nr", "Nr.", 9),
    ColumnSpec("comment_text", "Kommentar", 50),
    ColumnSpec("key_message", "Kernbotschaft", 50),
    ColumnSpec("d_haltung", "Haltung des Resttexts zu dieser Kernbotschaft", 28, editable=True, list_name="haltung"),
    ColumnSpec("d_anmerkung", "Ihre Anmerkung", 40, editable=True),
    ColumnSpec("pair_id", "pair_id", 18, hidden=True),
    ColumnSpec("comment_id", "comment_id", 18, hidden=True),
    ColumnSpec("row_fingerprint", "row_fingerprint", 18, hidden=True),
)

# Every example is invented; no real comment is tied to an answer.
ANLEITUNG_TEXT: tuple[tuple[str, str], ...] = (
    ("title", "Aufteilung von Kommentaren: allgemeine Reaktion und Resttext"),
    ("heading", "Worum es geht"),
    ("text", "Viele Kommentare beginnen mit einer allgemeinen Reaktion auf den ganzen Beitrag, z. B. „Super Idee!“ "
             "oder „Bitte nicht.“, und sagen danach etwas zu einzelnen Kernbotschaften."),
    ("text", "Wir möchten jeden Kommentar in diese zwei Teile aufteilen: die allgemeine Reaktion und den Resttext."),
    ("text", "Die allgemeine Reaktion wird einmal je Kommentar als positiv oder negativ erfasst."),
    ("text", "Die Haltung zu einzelnen Kernbotschaften wird nur aus dem Resttext bestimmt."),
    ("text", "Ihre bisherigen Labels werden dabei nicht verändert; Ihre Antworten werden in einer neuen, "
             "getrennt versionierten Datei abgelegt."),
    ("heading", "Begriffe"),
    ("text", "Allgemeine Reaktion: der Teil des Kommentars, der dem ganzen Beitrag pauschal zustimmt oder ihn "
             "pauschal ablehnt, ohne auf einen bestimmten Inhalt einzugehen."),
    ("text", "Positiv oder negativ richtet sich nach der gemeinten Bedeutung, nicht nach dem Wortlaut. "
             "Ironisches Lob ist eine negative Reaktion."),
    ("text", "Hinweis: die Wörter, an denen Sie die allgemeine Reaktion erkennen, und eine kurze Begründung."),
    ("text", "Resttext: alles, was nach Abzug der allgemeinen Reaktion übrig bleibt. Satzzeichen und Leerzeichen "
             "am Rand spielen keine Rolle."),
    ("text", "Sauber trennbar: „nein“, wenn beide Teile nur zusammen verständlich sind, z. B. weil der Resttext "
             "die allgemeine Reaktion einschränkt oder umkehrt. Füllen Sie die übrigen Felder trotzdem so gut "
             "wie möglich aus und erklären Sie den Fall in Ihrer Anmerkung."),
    ("heading", "Erfundene Beispiele"),
    ("text", "Beitrag mit drei Kernbotschaften: 1. Fahrradständer am Bahnhof aufstellen, 2. Parkplätze entfernen, "
             "3. Bäume pflanzen."),
    ("text", "A: „Super Idee!“ - allgemeine Reaktion: ja, positiv. Hinweis: „Super Idee!“, pauschales Lob ohne "
             "eigenen Inhalt. Resttext: leer. Trennbar: ja. Kernbotschaften: alle „nicht angesprochen“."),
    ("text", "B: „Super Idee! Am Bahnhof fehlen wirklich Fahrradständer.“ - allgemeine Reaktion: ja, positiv. "
             "Hinweis: „Super Idee!“. Resttext: „Am Bahnhof fehlen wirklich Fahrradständer.“ Trennbar: ja. "
             "Kernbotschaften: 1 „dafür“, 2 und 3 „nicht angesprochen“."),
    ("text", "C: „Bitte nicht.“ - allgemeine Reaktion: ja, negativ. Hinweis: „Bitte nicht.“ Resttext: leer. "
             "Kernbotschaften: alle „nicht angesprochen“."),
    ("text", "D: „Ja klar, tolle Idee. Als ob dann irgendwer Rad fährt.“ - allgemeine Reaktion: ja, negativ "
             "(ironisch). Hinweis: „Ja klar, tolle Idee.“ Resttext: „Als ob dann irgendwer Rad fährt.“ "
             "Kernbotschaften: 1 „dagegen“, 2 und 3 „nicht angesprochen“."),
    ("text", "E: „Die Bäume finde ich gut, den Rest weniger.“ - allgemeine Reaktion: nein. Resttext: darf leer "
             "bleiben, dann gilt der ganze Kommentar als Resttext. Kernbotschaften: 3 „dafür“, 1 und 2 nach Ihrer "
             "Einschätzung."),
    ("text", "F: „Gute Idee, nur halt nicht hier.“ - allgemeine Reaktion: ja, Polarität nach Ihrer Einschätzung. "
             "Trennbar: nein, weil „nur halt nicht hier“ die Zustimmung wieder aufhebt. Bitte in der Anmerkung "
             "kurz erklären."),
    ("heading", "So gehen Sie vor"),
    ("text", "1. Blatt „Kommentare“: Lesen Sie den Kommentar, den Beitrag und die Kernbotschaften und füllen Sie "
             "die gelben Zellen aus."),
    ("text", "2. Hinweis-Wörter und Resttext bitte per Kopieren und Einfügen wörtlich aus dem Kommentar übernehmen, "
             "damit wir sie automatisch zuordnen können. Mehrere Stellen jeweils in eine eigene Zeile "
             "(Alt+Eingabe bzw. Wahltaste+Eingabe)."),
    ("text", "3. Ohne allgemeine Reaktion bleiben Polarität und Hinweis leer, der Resttext darf leer bleiben."),
    ("text", "4. Blatt „Kernbotschaften“: Geben Sie je Kernbotschaft die Haltung des Resttexts an. "
             "„neutral“ heißt: der Resttext spricht die Kernbotschaft an, bezieht aber keine Position. "
             "Ohne Resttext ist jede Kernbotschaft „nicht angesprochen“."),
    ("text", "Bitte ändern Sie keine Kommentar- oder Kernbotschaftstexte; Sortieren und Filtern sind unproblematisch."),
    ("heading", "Legende"),
    ("answer_legend", "Gelb: bitte von Ihnen ausfüllen (Auswahlliste oder freier Text)."),
)

# Quote marks the annotator may put around a copied passage; stripped before matching.
_QUOTE_MARKS = "\"'„“”‚‘’«»"


# --------------------------------------------------------------------------
# Population
# --------------------------------------------------------------------------


def load_population(population_dir: Path) -> tuple[pd.DataFrame, str]:
    """Read the frozen population and verify it against its run's artifact hashes.

    Args:
        population_dir: Directory of a completed general-approval review run
            (holding POPULATION_FILE and HASHES_FILE).

    Returns:
        A tuple of the population (POPULATION_COLUMNS only, file order, one
        row per pair) and the file's SHA-256.

    Raises:
        FileNotFoundError: If the population file or the hashes file is
            missing.
        ValueError: If the hashes file does not list the population file,
            or pair IDs repeat, or a comment's rows are not contiguous.
        RuntimeError: If the file's hash differs from the recorded one.
    """
    population_dir = Path(population_dir)
    path, hashes_path = population_dir / POPULATION_FILE, population_dir / HASHES_FILE
    recorded = json.loads(hashes_path.read_text(encoding="utf-8"))
    expected = recorded.get("files", {}).get(POPULATION_FILE)
    if not recorded.get("complete") or expected is None:
        raise ValueError(f"{hashes_path} does not record a completed {POPULATION_FILE}")
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"{path} changed since it was written (sha256 {actual}, recorded {expected})")
    frame = pd.read_parquet(path, columns=list(POPULATION_COLUMNS))
    if frame.pair_id.duplicated().any():
        raise ValueError("Duplicate pair IDs in the population")
    # A comment's rows must form one block, so numbering by first appearance is unambiguous.
    # Each block starts where the ID differs from the row above; one block per comment means contiguous.
    if int(frame.comment_id.ne(frame.comment_id.shift()).sum()) != frame.comment_id.nunique():
        raise ValueError("A comment's rows are not contiguous in the population")
    return frame.reset_index(drop=True), actual


def comment_id_for(contribution_id: object, comment_text: str) -> str:
    """Deterministic 16-hex ID of one comment on one contribution.

    Same derivation as the 2026-10-07 diagnostic table's comment_id.

    Args:
        contribution_id: The contribution's ID (stringified).
        comment_text: The exact raw comment text.

    Returns:
        The first 16 hex characters of SHA-256 over "contribution_id|comment_text".
    """
    return hashlib.sha256(f"{contribution_id}|{comment_text}".encode("utf-8")).hexdigest()[:16]


def load_fold_comments(source_path: Path, folds_path: Path, fold: int = EXTRA_FOLD) -> pd.DataFrame:
    """Every comment of one fold in a frozen extraction source, shaped like the population.

    Key-message IDs ("<contribution>:km<n>") and pair IDs are derived as in
    benchmark runs over the same source (benchmark.load_frozen_comments and
    context_builder's numbering), so labels join to model predictions.

    Args:
        source_path: A frozen whole_process-shaped extraction source.
        folds_path: The persisted fold assignment.
        fold: The fold whose test contributions are kept. Defaults to
            EXTRA_FOLD (0, the development fold).

    Returns:
        POPULATION_COLUMNS rows, one per pair, rows of a comment contiguous.

    Raises:
        ValueError: If the source is inconsistent (see load_frozen_comments).
    """
    assignment = load_folds(folds_path)
    in_fold = {str(cid) for cid in assignment.folds[fold]["test_contribution_ids"]}
    rows = []
    for entry in load_frozen_comments(pd.read_parquet(source_path)):
        contribution_id, comment_text = str(entry["contribution_id"]), entry["comment_text"]
        if contribution_id not in in_fold:
            continue
        text = clean(entry["contribution_text"])
        for position, km in enumerate(entry["key_messages"], start=1):
            rows.append({
                "pair_id": compute_pair_id(contribution_id, km["key_message"], comment_text),
                "comment_id": comment_id_for(contribution_id, comment_text),
                "contribution_id": contribution_id, "fold": fold,
                "key_message_id": f"{contribution_id}:km{position}", "key_message": km["key_message"],
                "comment_text": comment_text, "contribution_text": text,
                # A text at the source's cut-off length is assumed to have been cut there.
                "contribution_text_truncated": (len(text) >= CONTRIBUTION_TEXT_LIMIT) if text else None,
            })
    frame = pd.DataFrame(rows, columns=list(POPULATION_COLUMNS))
    frame["fold"] = frame["fold"].astype("Int64")
    frame["contribution_text_truncated"] = frame["contribution_text_truncated"].astype("boolean")
    return frame


def combine_populations(base: pd.DataFrame, extra: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Add the extra comments that are not in base, ordered by contribution.

    Comments present in both must have the same pair IDs; this checks that
    the extra source's ID derivation matches the base table.

    Args:
        base: The frozen population (output of load_population).
        extra: Output of load_fold_comments.

    Returns:
        A tuple of the combined population (sorted by numeric contribution
        ID, stable, so each comment's rows stay contiguous) and the comment
        IDs that were added, in combined order.

    Raises:
        ValueError: If a shared comment has different pair IDs in the two
            parts, or the combined population repeats a pair ID.
    """
    shared = set(base.comment_id) & set(extra.comment_id)
    for comment_id in sorted(shared):
        if set(base.pair_id[base.comment_id == comment_id]) != set(extra.pair_id[extra.comment_id == comment_id]):
            raise ValueError(f"Comment {comment_id}: pair IDs differ between the population and the extra source")
    added = extra[~extra.comment_id.isin(shared)]
    combined = pd.concat([base, added], ignore_index=True)
    if combined.pair_id.duplicated().any():
        raise ValueError("Duplicate pair IDs in the combined population")
    combined = combined.sort_values("contribution_id", key=lambda ids: ids.astype(int), kind="stable")
    combined = combined.reset_index(drop=True)
    order = list(dict.fromkeys(combined.comment_id))
    added_ids = set(added.comment_id)
    return combined, [cid for cid in order if cid in added_ids]


# --------------------------------------------------------------------------
# Writing the pack
# --------------------------------------------------------------------------


def _contribution_cell(first) -> str:
    """The contribution excerpt cell of one comment.

    Args:
        first: Any population row of the comment.

    Returns:
        The display text, with "[gekürzt]" appended when the source cut it,
        or a fixed German note when no contribution text is available.
    """
    text = display(first.contribution_text)
    if text is None:
        return "(Beitragstext liegt hier nicht vor)"
    # "is True": the column is nullable boolean and bool(pd.NA) raises TypeError.
    return text + ("\n[gekürzt]" if clean(first.contribution_text_truncated) is True else "")


def build_pack_rows(population: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    """Comment rows and key-message rows of the pack, in population order (pure).

    Texts are shown decoded (HTML entities) and Excel-safe; the row
    fingerprints are computed from exactly these displayed texts, which is
    what the reader compares. Answer cells are left empty.

    Args:
        population: Output of load_population (rows of a comment contiguous).

    Returns:
        A tuple (comment_rows, km_rows) of dicts keyed by ColumnSpec.key:
        one comment row per comment (numbered R001, R002, ...) and one
        key-message row per pair (numbered R001.1, R001.2, ...).
    """
    comment_rows, km_rows = [], []
    for number, (comment_id, group) in enumerate(population.groupby("comment_id", sort=False), start=1):
        rows = list(group.itertuples())
        first, nr = rows[0], f"R{number:03d}"
        comment_text = display(first.comment_text)
        key_messages = [display(r.key_message) for r in rows]
        pair_ids = sorted(group.pair_id)
        comment_rows.append({
            "nr": nr, "comment_text": comment_text, "contribution_text": _contribution_cell(first),
            "key_messages_overview": "\n".join(f"{i}. {km}" for i, km in enumerate(key_messages, 1)),
            "comment_id": comment_id, "pair_ids": ";".join(pair_ids),
            "row_fingerprint": row_fingerprint(comment_id, comment_text, *pair_ids),
        })
        for index, (row, key_message) in enumerate(zip(rows, key_messages), start=1):
            km_rows.append({
                "nr": f"{nr}.{index}", "comment_text": comment_text, "key_message": f"{index}. {key_message}",
                "pair_id": row.pair_id, "comment_id": comment_id,
                "row_fingerprint": row_fingerprint(row.pair_id, comment_id, f"{index}. {key_message}"),
            })
    return comment_rows, km_rows


def write_pack(population: pd.DataFrame, path: Path, *, created_utc: str, population_sha256: str,
               extra_source_sha256: str | None = None) -> None:
    """Write the blank German annotation pack to a new file.

    Args:
        population: Output of load_population.
        path: Target .xlsx path; must not exist yet.
        created_utc: Creation timestamp in CREATED_UTC_FORMAT.
        population_sha256: SHA-256 of the population file, recorded on the
            Meta sheet.
        extra_source_sha256: SHA-256 of the extra fold-0 source, recorded on
            the Meta sheet when given. Defaults to None.

    Raises:
        FileExistsError: If path already exists.
    """
    comment_rows, km_rows = build_pack_rows(population)
    workbook = Workbook()
    sheets = {name: (workbook.active if i == 0 else workbook.create_sheet(name)) for i, name in enumerate(SHEET_ORDER)}
    sheets[ANLEITUNG_SHEET].title = ANLEITUNG_SHEET
    list_ranges = write_lists(sheets[LISTS_SHEET], OPTION_SETS)
    write_text_sheet(sheets[ANLEITUNG_SHEET], ANLEITUNG_TEXT)
    write_table_sheet(sheets[COMMENTS_SHEET], COMMENT_COLUMNS, comment_rows, list_ranges)
    write_table_sheet(sheets[KM_SHEET], KM_COLUMNS, km_rows, list_ranges)
    meta = [
        ("pack_schema_version", PACK_SCHEMA_VERSION),
        ("created_utc", created_utc),
        ("population_sha256", population_sha256),
        ("extra_source_sha256", extra_source_sha256 or ""),
        ("comment_ids", ";".join(r["comment_id"] for r in comment_rows)),
        ("pair_ids", ";".join(r["pair_id"] for r in km_rows)),
    ]
    meta += [(f"column:{sheet}:{spec.key}", spec.header)
             for sheet, specs in ((COMMENTS_SHEET, COMMENT_COLUMNS), (KM_SHEET, KM_COLUMNS)) for spec in specs]
    write_meta(sheets[META_SHEET], meta)
    workbook.active = 0
    save_workbook_exclusive(workbook, path, created_utc=created_utc,
                            title="Aufteilung allgemeine Reaktion und Resttext")


def _created_utc(now: datetime | None) -> str:
    """Creation timestamp in CREATED_UTC_FORMAT.

    Args:
        now: Creation time; a naive datetime is read as local time. Defaults
            to the current UTC time.

    Returns:
        The formatted UTC timestamp.
    """
    return (now or datetime.now(timezone.utc)).astimezone(timezone.utc).strftime(CREATED_UTC_FORMAT)


@dataclass(frozen=True)
class ArtifactRun:
    """Where one run wrote its artifacts, and its summary.

    Attributes:
        output_dir: The newly created output directory.
        files: Artifact role to its path.
        summary: The summary that was written.
    """

    output_dir: Path
    files: dict[str, Path]
    summary: dict


def run_reaction_split_pack(*, population_dir: Path = DEFAULT_POPULATION_DIR,
                            extra_source: Path | None = DEFAULT_EXTRA_SOURCE,
                            folds_path: Path = DEFAULT_FOLDS_PATH, output_dir: Path | None = None,
                            now: datetime | None = None) -> ArtifactRun:
    """Build the blank annotation pack and its summary in a new directory.

    Args:
        population_dir: Completed general-approval review run whose
            population is used. Defaults to DEFAULT_POPULATION_DIR.
        extra_source: Frozen extraction source whose remaining fold-0
            comments are added; None adds nothing. Defaults to
            DEFAULT_EXTRA_SOURCE.
        folds_path: Fold assignment used to select the extra comments.
            Defaults to DEFAULT_FOLDS_PATH.
        output_dir: Directory to create (its parent must exist). Defaults to
            EXPERIMENTS_DIR / "reaction_split_pack_<created_utc>".
        now: Creation time (for tests). Defaults to the current UTC time.

    Returns:
        The output directory, the written files and the summary.

    Raises:
        FileNotFoundError: If an input or the output directory's parent is
            missing.
        FileExistsError: If the output directory already exists.
        RuntimeError: If the population file changed since it was written.
        ValueError: If the population is inconsistent.
    """
    created_utc = _created_utc(now)
    population, population_sha256 = load_population(population_dir)
    base_comments = population.comment_id.nunique()
    inputs = {"population": {"path": str(Path(population_dir) / POPULATION_FILE), "sha256": population_sha256}}
    added: list[str] = []
    if extra_source is not None:
        inputs["extra_source"] = {"path": str(extra_source), "sha256": sha256_file(Path(extra_source)),
                                  "fold": EXTRA_FOLD}
        inputs["folds"] = {"path": str(folds_path), "sha256": sha256_file(Path(folds_path))}
        population, added = combine_populations(population, load_fold_comments(extra_source, folds_path))
    comments = population.drop_duplicates("comment_id")
    folds = Counter("none" if pd.isna(f) else str(int(f)) for f in comments.fold)
    summary = {
        "created_utc": created_utc,
        "pack_schema_version": PACK_SCHEMA_VERSION,
        "new_inference_requests": 0,
        "network_calls": 0,
        "module_sha256": sha256_file(Path(__file__).resolve()),
        "inputs": inputs,
        "comments": len(comments),
        "comments_from_review_2026_10_07": base_comments,
        "comments_added_from_fold_0": len(added),
        "key_message_rows": len(population),
        "comments_by_fold": dict(sorted(folds.items())),
        "comment_ids": comments.comment_id.tolist(),
        "added_comment_ids": added,
        "pair_ids": population.pair_id.tolist(),
        "heldout_note": HELDOUT_NOTE,
    }
    output_dir = Path(output_dir) if output_dir else EXPERIMENTS_DIR / f"{PACK_DIR_PREFIX}{created_utc}"
    output_dir.mkdir(parents=False, exist_ok=False)
    files = {"pack": output_dir / PACK_FILE_TEMPLATE.format(created_utc=created_utc),
             "summary": output_dir / PACK_SUMMARY_FILE}
    write_pack(population, files["pack"], created_utc=created_utc, population_sha256=population_sha256,
               extra_source_sha256=inputs.get("extra_source", {}).get("sha256"))
    write_json(files["summary"], summary)
    files["hashes"] = write_artifact_hashes(output_dir, created_utc, [files["pack"], files["summary"]])
    return ArtifactRun(output_dir, files, summary)


# --------------------------------------------------------------------------
# Reading the annotator's answers
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ReactionSplitAnswers:
    """Parsed and checked answers of a filled pack.

    Option labels are mapped back to OPTION_SETS codes; empty cells are None.
    Problems that the annotator can fix (a quote not found in the comment,
    contradictory answers, unanswered rows) do not raise: they are listed in
    each row's "issues" column, so nothing is accepted silently.

    Attributes:
        meta: The Meta sheet's key/value pairs.
        comments: One row per comment in pack order: comment_id, nr,
            comment_text, pair_ids (list), general_reaction, polarity,
            hint_quotes (list), hint_explanation, rest_quotes (list),
            rest_is_whole_comment, splittable, note and issues (list).
        key_messages: One row per pair in pack order: pair_id, comment_id,
            nr, stance, note and issues (list).
    """

    meta: dict[str, str]
    comments: pd.DataFrame
    key_messages: pd.DataFrame


def split_quotes(value: str | None) -> list[str]:
    """Split a quote cell into its passages, one per non-blank line.

    Surrounding whitespace and quote marks are removed from each passage.

    Args:
        value: The cell text, or None.

    Returns:
        The passages in order; empty for an empty cell.
    """
    if not value:
        return []
    passages = (line.strip().strip(_QUOTE_MARKS).strip() for line in value.split("\n"))
    return [p for p in passages if p]


def _quote_issues(quotes: Sequence[str], comment_text: str, name: str) -> list[str]:
    """Issues for passages that are not in the comment (whitespace-normalized).

    Args:
        quotes: The passages.
        comment_text: The comment as shown in the pack.
        name: Issue prefix, e.g. "hint_quote".

    Returns:
        One "<name>_not_in_comment:<line>" issue per passage not found,
        with 1-based line numbers.
    """
    normalized = normalize_whitespace(comment_text)
    return [f"{name}_not_in_comment:{i}" for i, quote in enumerate(quotes, 1)
            if normalize_whitespace(quote) not in normalized]


def _comment_answer(record: dict) -> dict:
    """Parse and check one Kommentare row (fingerprint already verified).

    Args:
        record: A row from read_rows.

    Returns:
        The answer row (see ReactionSplitAnswers.comments).
    """
    decided = decisions(record, COMMENT_COLUMNS, COMMENTS_SHEET, OPTION_SETS)
    comment_text = record["comment_text"] or ""
    reaction, polarity = decided["d_reaktion"], decided["d_polaritaet"]
    hints, rest = split_quotes(decided["d_hinweis_zitat"]), split_quotes(decided["d_resttext"])
    issues = []
    if reaction is None:
        issues.append("unanswered")
    if reaction == "yes":
        issues += [] if polarity else ["missing_polarity"]
        issues += [] if hints else ["missing_hint_quote"]
    else:
        issues += ["polarity_without_general_reaction"] if polarity else []
        issues += ["hint_quote_without_general_reaction"] if hints else []
    if reaction is not None and decided["d_trennbar"] is None:
        issues.append("missing_splittable")
    issues += _quote_issues(hints, comment_text, "hint_quote") + _quote_issues(rest, comment_text, "rest_quote")
    # Without a general reaction the whole comment is the rest (the instructions allow an empty cell).
    whole = reaction == "no" and not rest
    return {
        "comment_id": record["comment_id"], "nr": record["nr"], "comment_text": comment_text,
        "pair_ids": sorted(filter(None, (record["pair_ids"] or "").split(";"))),
        "general_reaction": reaction, "polarity": polarity, "hint_quotes": hints,
        "hint_explanation": decided["d_hinweis_begruendung"],
        "rest_quotes": [comment_text] if whole else rest, "rest_is_whole_comment": whole,
        "splittable": decided["d_trennbar"], "note": decided["d_anmerkung"], "issues": issues,
    }


def _parse_comments(ws, expected: list[str]) -> pd.DataFrame:
    """Parse and check the Kommentare sheet.

    Args:
        ws: The Kommentare worksheet.
        expected: Comment IDs from the Meta sheet, in pack order.

    Returns:
        One answer row per comment, in pack order.

    Raises:
        ValueError: If columns or rows are missing or duplicated, a row's
            fingerprint no longer matches its comment text and IDs, or an
            answer is not a valid list value.
    """
    editable = [s.key for s in COMMENT_COLUMNS if s.editable]
    rows = read_rows(ws, COMMENT_COLUMNS, ["nr", "comment_text", "comment_id", "pair_ids", "row_fingerprint",
                                           *editable])
    check_ids(rows, "comment_id", expected, COMMENTS_SHEET)
    parsed = []
    for record in rows:
        pair_ids = sorted(filter(None, (record["pair_ids"] or "").split(";")))
        # Same inputs as in build_pack_rows; a mismatch means the text or IDs were edited.
        if row_fingerprint(record["comment_id"], record["comment_text"] or "", *pair_ids) != record["row_fingerprint"]:
            raise ValueError(f"{COMMENTS_SHEET} row {record['nr']}: fingerprint mismatch (text or IDs changed)")
        parsed.append(_comment_answer(record))
    order = {cid: i for i, cid in enumerate(expected)}
    return answer_frame(sorted(parsed, key=lambda r: order[r["comment_id"]]))


def _parse_key_messages(ws, expected: list[str], comments: pd.DataFrame) -> pd.DataFrame:
    """Parse and check the Kernbotschaften sheet.

    Args:
        ws: The Kernbotschaften worksheet.
        expected: Pair IDs from the Meta sheet, in pack order.
        comments: The parsed comments (for ownership and rest-text checks).

    Returns:
        One answer row per pair, in pack order.

    Raises:
        ValueError: If columns or rows are missing or duplicated, a row's
            fingerprint no longer matches, a pair sits under another
            comment, or an answer is not a valid list value.
    """
    rows = read_rows(ws, KM_COLUMNS, ["nr", "key_message", "pair_id", "comment_id", "row_fingerprint", "d_haltung"])
    check_ids(rows, "pair_id", expected, KM_SHEET)
    owner = {pid: cid for cid, pids in zip(comments.comment_id, comments.pair_ids) for pid in pids}
    has_rest = dict(zip(comments.comment_id, comments.rest_quotes.map(bool)))
    parsed = []
    for record in rows:
        fingerprint = row_fingerprint(record["pair_id"], record["comment_id"] or "", record["key_message"] or "")
        if fingerprint != record["row_fingerprint"]:
            raise ValueError(f"{KM_SHEET} row {record['nr']}: fingerprint mismatch (text or IDs changed)")
        if owner.get(record["pair_id"]) != record["comment_id"]:
            raise ValueError(f"{KM_SHEET} row {record['nr']}: pair does not belong to comment {record['comment_id']}")
        decided = decisions(record, KM_COLUMNS, KM_SHEET, OPTION_SETS)
        stance, issues = decided["d_haltung"], []
        if stance is None:
            issues.append("unanswered")
        elif stance in DIRECTIONAL_OR_NEUTRAL and not has_rest[record["comment_id"]]:
            issues.append("stance_without_rest_text")
        parsed.append({"pair_id": record["pair_id"], "comment_id": record["comment_id"], "nr": record["nr"],
                       "stance": stance, "note": decided["d_anmerkung"], "issues": issues})
    order = {pid: i for i, pid in enumerate(expected)}
    return answer_frame(sorted(parsed, key=lambda r: order[r["pair_id"]]))


def read_reaction_split_pack(path: Path) -> ReactionSplitAnswers:
    """Parse and check a filled pack; the workbook is only read, never saved.

    Args:
        path: The filled .xlsx pack.

    Returns:
        The parsed answers with per-row issues.

    Raises:
        ValueError: If the file is not a pack of this schema, IDs or texts
            were changed, a pair was moved to another comment, or an answer
            is not a valid list value.
        KeyError: If the Kommentare or Kernbotschaften sheet is missing.
    """
    workbook = load_workbook(path, data_only=False)
    try:
        meta = read_meta(workbook, META_SHEET, PACK_SCHEMA_VERSION, ("comment_ids", "pair_ids"))
        comments = _parse_comments(workbook[COMMENTS_SHEET], meta["comment_ids"].split(";"))
        key_messages = _parse_key_messages(workbook[KM_SHEET], meta["pair_ids"].split(";"), comments)
        return ReactionSplitAnswers(meta, comments, key_messages)
    finally:
        workbook.close()


def _issue_counts(frame: pd.DataFrame) -> dict[str, int]:
    """Count issues by kind (the part before ":"), sorted.

    Args:
        frame: Answer rows with an "issues" list column.

    Returns:
        Issue kind to the number of rows that have it at least once.
    """
    counts = Counter(kind for issues in frame.issues for kind in {i.split(":")[0] for i in issues})
    return dict(sorted(counts.items()))


def _value_counts(values: pd.Series) -> dict[str, int]:
    """Count answer codes, with empty answers as "unanswered".

    Args:
        values: Answer codes or None.

    Returns:
        Code to count, sorted.
    """
    return dict(sorted(Counter("unanswered" if v is None else v for v in values).items()))


def export_reaction_split_labels(pack_path: Path, *, output_dir: Path | None = None,
                                 now: datetime | None = None) -> ArtifactRun:
    """Read a filled pack and write its labels as a new, versioned label set.

    Args:
        pack_path: The filled .xlsx pack (only read).
        output_dir: Directory to create (its parent must exist). Defaults to
            EXPERIMENTS_DIR / "reaction_split_labels_<created_utc>".
        now: Creation time (for tests). Defaults to the current UTC time.

    Returns:
        The output directory, the written files and the summary (counts,
        issue counts and the pack's hash; no texts).

    Raises:
        FileExistsError: If the output directory already exists.
        ValueError: See read_reaction_split_pack.
    """
    created_utc = _created_utc(now)
    pack_sha256 = sha256_file(Path(pack_path))
    answers = read_reaction_split_pack(pack_path)
    comments, key_messages = answers.comments, answers.key_messages
    summary = {
        "created_utc": created_utc,
        "labels_schema_version": LABELS_SCHEMA_VERSION,
        "pack": {"path": str(pack_path), "sha256": pack_sha256,
                 "schema_version": answers.meta["pack_schema_version"],
                 "created_utc": answers.meta.get("created_utc"),
                 "population_sha256": answers.meta.get("population_sha256"),
                 "extra_source_sha256": answers.meta.get("extra_source_sha256") or None},
        "module_sha256": sha256_file(Path(__file__).resolve()),
        "comments": len(comments),
        "key_message_rows": len(key_messages),
        "general_reaction": _value_counts(comments.general_reaction),
        "polarity": _value_counts(comments[comments.general_reaction.eq("yes")].polarity),
        "splittable": _value_counts(comments.splittable),
        "stance": _value_counts(key_messages.stance),
        "comment_issues": _issue_counts(comments),
        "key_message_issues": _issue_counts(key_messages),
        "comments_without_issues": int(comments.issues.map(len).eq(0).sum()),
        "key_message_rows_without_issues": int(key_messages.issues.map(len).eq(0).sum()),
        "heldout_note": HELDOUT_NOTE,
    }
    output_dir = Path(output_dir) if output_dir else EXPERIMENTS_DIR / f"{LABELS_DIR_PREFIX}{created_utc}"
    output_dir.mkdir(parents=False, exist_ok=False)
    files = {"comments": output_dir / LABELS_COMMENTS_FILE, "key_messages": output_dir / LABELS_KM_FILE,
             "summary": output_dir / LABELS_SUMMARY_FILE}
    for key, frame in (("comments", comments), ("key_messages", key_messages)):
        with open(files[key], "xb") as f:
            frame.assign(labels_schema_version=LABELS_SCHEMA_VERSION).to_parquet(f, index=False)
    write_json(files["summary"], summary)
    files["hashes"] = write_artifact_hashes(output_dir, created_utc, list(files.values()))
    return ArtifactRun(output_dir, files, summary)
