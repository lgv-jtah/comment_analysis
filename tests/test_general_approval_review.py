import hashlib
import json
import re
import socket
import sqlite3
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import pandas as pd
import pytest
from openpyxl import Workbook, load_workbook

import cli
import evaluator_v2
import general_approval_review as gar
from dataset_builder import PRIMARY_EVALUATOR, compute_pair_id
from fold_builder import FoldAssignment

ROOT = Path(__file__).resolve().parents[1]
SECONDARY_EVALUATOR = "Zweitperson"
KEY_MESSAGES = {
    "101": ("Mehr Fahrradständer am Bahnhof", "Parkplätze am Bahnhof entfernen"),
    "102": ("Tempo dreißig in der Schulstraße", "Zebrastreifen vor der Schule"),
    "103": ("Neue Sitzbänke im Stadtpark", "Mehr Mülleimer im Stadtpark"),
}
EXPLANATION = "Pauschale Zustimmung zum gesamten Beitrag erkannt."


def _p(stance=None, *, basis=None, status="accepted", reasons=(), evidence=None):
    return dict(status=status, proposed_stance=stance, final_stance=stance if status == "accepted" else None,
                basis=basis, review_reasons=list(reasons), comment_evidence=evidence)


def _r(label, basis="verdict_correct", tier="A", verdict="correct", corrected=None):
    if label is None:
        return dict(resolution_status="unresolved", reference_label=None, tier=None, basis=None,
                    verdict=verdict, corrected=corrected)
    return dict(resolution_status="resolved", reference_label=label, tier=tier, basis=basis,
                verdict=verdict, corrected=corrected)


GA = "general_approval"
# (contribution, comment, run, [(prediction or None, reference, note)] per key message)
# run None: the contribution is in no model run (only the annotator note brings it in);
# prediction None: the pair is in the run's expected cohort but has no prediction row.
COMMENTS = [
    ("101", "Stimmt", "fold0", [
        (_p("neutral", basis=GA, evidence=["Stimmt"]), _r("neutral"), "generelle zustimmende Reaktion"),
        (_p("neutral", basis=GA, evidence=["Stimmt"]), _r("in_favor"), None)]),
    ("101", "Gute Idee, endlich mehr Fahrradständer am Bahnhof.", "fold0", [
        (_p("neutral", basis=GA, evidence=["Gute Idee, endlich mehr Fahrradständer am Bahnhof."]),
         _r("in_favor", "verdict_incorrect_corrected", verdict="incorrect", corrected="in favor"), None),
        (_p("neutral", basis=GA, evidence=["Gute Idee"]), _r("neutral"), None)]),
    ("101", "Die Ampel am Bahnhof ist viel zu lang geschaltet.", "fold0", [
        (_p("neutral", basis=GA), _r("in_favor", "tier_b_parsed", tier="B", verdict="incorrect"), None),
        (_p("against", basis="explicit_target"), _r("neutral"), None)]),
    ("101", "Bäume pflanzen wäre schön.", "fold0", [
        (_p("neutral", basis="unaddressed"), _r("neutral"), "allgemeine ablehnende Reaktion"),
        (_p("neutral", basis="unaddressed"), _r("neutral"), None)]),
    ("102", "Hahaha, ok.", "heldout", [
        (_p("neutral", basis=GA), _r("against", "verdict_incorrect_corrected", verdict="incorrect",
                                     corrected="against"), None),
        (_p("neutral", basis=GA), _r("neutral"), None)]),
    ("102", "Warum nicht!", "heldout", [
        (_p(status="failed", reasons=["stage1_general_support_missing_evidence"]), _r("neutral"), None),
        (_p(status="failed", reasons=["stage1_general_support_missing_evidence"]), _r("neutral"), None)]),
    ("102", "Siehe oben", "heldout", [
        (_p("neutral", basis="unaddressed"), _r(None), "Eine allgemeine zustimmende Reaktion"),
        (_p("neutral", basis="unaddressed"), _r("neutral"), None)]),
    ("102", "Super Idee", "heldout", [
        (_p("neutral", basis=GA), _r(None), None),
        (None, _r("neutral"), None)]),
    ("103", "Danke, ja!", None, [
        (None, _r(None), "eigentlich eine zustimmende generelle Reaktion. Aber definitiv nicht neutral."),
        (None, _r("neutral"), None)]),
]
NOTES = [note for *_, kms in COMMENTS for _, _, note in kms if note]


def _pairs():
    for cid, text, run, kms in COMMENTS:
        for index, (prediction, reference, note) in enumerate(kms):
            km = KEY_MESSAGES[cid][index]
            yield SimpleNamespace(cid=cid, text=text, run=run, km=km, km_id=f"{cid}:km{index + 1}",
                                  pair_id=compute_pair_id(cid, km, text), prediction=prediction,
                                  reference=reference, note=note)


def _write_annotations(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.execute("create table stance_evaluations (id integer primary key, contribution_id integer, key_message text,"
                 " comment_text text, verdict text, comment text, evaluator text, created_at text,"
                 " comment_index integer, corrected_label text)")
    rows = [(int(p.cid), p.km, p.text, p.reference["verdict"], p.note, PRIMARY_EVALUATOR, "2026-09-01",
             None, p.reference["corrected"]) for p in _pairs()]
    first = next(_pairs())
    rows.append((int(first.cid), first.km, first.text, "correct", "Zweitmeinung", SECONDARY_EVALUATOR,
                 "2026-09-02", None, None))
    conn.executemany("insert into stance_evaluations (contribution_id, key_message, comment_text, verdict, comment,"
                     " evaluator, created_at, comment_index, corrected_label) values (?,?,?,?,?,?,?,?,?)", rows)
    conn.commit()
    conn.close()


def _reference() -> pd.DataFrame:
    rows = []
    for row_id, p in enumerate(_pairs(), start=1):
        r = p.reference
        rows.append(dict(pair_id=p.pair_id, contribution_id=int(p.cid), key_message=p.km, comment_text=p.text,
                         source_row_id=row_id, evaluator=PRIMARY_EVALUATOR, cohort_role="primary",
                         verdict=r["verdict"], resolution_status=r["resolution_status"], tier=r["tier"],
                         reference_label=r["reference_label"], basis=r["basis"],
                         established_original_label="in_favor", post_fix_original_label="neutral"))
    secondary = dict(rows[0], evaluator=SECONDARY_EVALUATOR, cohort_role="secondary", source_row_id=999)
    return pd.DataFrame(rows + [secondary])


def _write_run(directory: Path, run: str, run_id: str) -> gar.RunSpec:
    predictions, expected = [], []
    for p in _pairs():
        if p.run != run:
            continue
        expected.append(dict(pair_id=p.pair_id, contribution_id=p.cid))
        if p.prediction is None:
            continue
        evidence = p.prediction["comment_evidence"]
        predictions.append(dict(
            contribution_id=p.cid, contributionContent=f"Beitrag {p.cid}: Am Bahnhof fehlen Fahrradständer.",
            key_message_id=p.km_id, key_message=p.km, comment_text=p.text, pair_id=p.pair_id,
            status=p.prediction["status"], proposed_stance=p.prediction["proposed_stance"],
            final_stance=p.prediction["final_stance"], basis=p.prediction["basis"],
            comment_evidence=list(evidence or []), source_evidence=[], explanation=EXPLANATION,
            review_reasons=p.prediction["review_reasons"], policy_version="v2", run_id=run_id,
            extraction_model_name=None, stance_model_name="test-model"))
    spec = gar.RunSpec(run, directory / f"{run}.parquet", directory / f"{run}.expected.parquet")
    pd.DataFrame(predictions).to_parquet(spec.predictions)
    pd.DataFrame(expected).to_parquet(spec.expected)
    return spec


@pytest.fixture
def inputs(tmp_path):
    data = tmp_path / "inputs"
    data.mkdir()
    reference_path, annotations_path = data / "pairs.parquet", data / "eval.sq3"
    _reference().to_parquet(reference_path)
    _write_annotations(annotations_path)
    folds_path = data / "folds.json"
    folds = [dict(train_contribution_ids=[], test_contribution_ids=[cid]) for cid in ("101", "102", "201", "202", "203")]
    folds_path.write_text(json.dumps(asdict(FoldAssignment(42, 5, "contribution_id", "skipped", 0, folds))))
    source_path = data / "source.parquet"
    pd.DataFrame([dict(contribution_id=101, key_message=KEY_MESSAGES["101"][0],
                       related_sentences=["Am Bahnhof fehlen Fahrradständer."])]).to_parquet(source_path)
    runs = (_write_run(data, "fold0", "run_a"), _write_run(data, "heldout", "run_b"))
    return SimpleNamespace(runs=runs, reference=reference_path, folds=folds_path, annotations=annotations_path,
                           sources=(source_path,), root=tmp_path)


def _run(inputs, output_dir, now=datetime(2026, 10, 7, 12, 0, tzinfo=timezone.utc)):
    return gar.run_general_approval_review(
        runs=inputs.runs, reference_path=inputs.reference, folds_path=inputs.folds,
        annotations_path=inputs.annotations, source_paths=inputs.sources, output_dir=output_dir, now=now)


@pytest.fixture
def result(inputs):
    return _run(inputs, inputs.root / "out")


def _diagnostic(inputs):
    return gar.build_diagnostic_table(
        gar.load_runs(inputs.runs), pd.read_parquet(inputs.reference), gar.load_annotations(inputs.annotations),
        related_sentences=gar.load_sources(inputs.sources), fold_of=gar.load_fold_lookup(inputs.folds))


def _comment(table, text):
    return table[table.comment_text == text]


# --------------------------------------------------------------------------
# Signals
# --------------------------------------------------------------------------

BLANKET_APPROVALS = [
    "Stimmt", "Stimmt genau!", "Volle Zustimmung", "100% Zustimmung", "Zustimmung", "In der Tat.", "Warum nicht!",
    "Du hast Recht", "Ganz meine Meinung", "Super Idee", "Tolle Idee!", "Ein sehr guter Vorschlag.",
    "Das ist ein hervorragender Vorschlag!", "Ein exzellenter Vorschlag aus meiner Sicht.",
    "Dieser Vorschlag ist super.", "Ich finde die Idee sehr gut!", "Ich stimme voll und ganz zu.",
    "Wir stimmen diesem Vorschlag voll und ganz zu!! :)", "Dem Vorschlag können wir nur zustimmen!",
    "Diesen Vorschlag begrüßen wir sehr.", "Ich unterstütze diese Forderung!", "Das geht in die richtige Richtung!",
    "Sehr richtig.", "Richtiger Hinweis.", "so wahr, so richtig", "This!", "Gutgutgutgut", "Ja!", "Danke, ja!",
    "Auf jeden Fall ja", "Klingt smart!", "Tolle Aktion!", "Das wäre ein Riesenfortschritt!",
]


@pytest.mark.parametrize("text", BLANKET_APPROVALS)
def test_every_prefix_family_is_a_blanket_approval(text):
    signals = gar.compute_comment_signals(text)
    assert signals.approval_prefix is not None
    assert signals.residual_word_count == 0
    assert gar.classify_signals(signals) == "blanket_no_content"


def test_residual_boundary_between_blanket_and_content():
    two = gar.compute_comment_signals("Stimmt. Sehr schön")
    three = gar.compute_comment_signals("Stimmt. Sehr schön gemacht")
    assert (two.residual_word_count, gar.classify_signals(two)) == (2, "blanket_no_content")
    assert (three.residual_word_count, gar.classify_signals(three)) == (3, "approval_plus_content")
    assert gar.compute_comment_signals("Stimmt, ok so").residual_word_count == 0


def test_prefix_in_comma_clause_keeps_the_rest_as_content():
    signals = gar.compute_comment_signals("Gute Idee, so reduzieren wir das Baumsterben.")
    assert signals.approval_prefix == "Gute Idee"
    assert signals.residual_text == "so reduzieren wir das Baumsterben"
    assert gar.classify_signals(signals) == "approval_plus_content"


def test_html_entities_are_unescaped():
    signals = gar.compute_comment_signals("&quot;Super Idee&quot; &amp;")
    assert signals.approval_prefix == "Super Idee"
    assert signals.residual_word_count == 0
    assert signals.char_length == len('"Super Idee" &')


@pytest.mark.parametrize("text, expected", [
    ("Eigentlich egal, stimmt", "negation_or_irony_cue"),
    ("Nee, egal", "negation_or_irony_cue"),
    ("Nee, das finde ich nicht gut", "content_without_approval_phrase"),
    ("Hahaha, ok.", "negation_or_irony_cue"),
    ("Super Idee ;)", "negation_or_irony_cue"),
    ("Das ist ironisch gemeint", "negation_or_irony_cue"),
    ("Warum nicht!", "blanket_no_content"),
    ("Siehe oben", "back_reference"),
    ("Ich schließe mich den obigen Kommentaren an! Es gibt kaum Vergleichbares.", "back_reference"),
    ("Mein Kommentar ist in der Beschreibung.", "back_reference"),
    ("Die Ampel ist viel zu lang geschaltet", "content_without_approval_phrase"),
    ("Jahrelang warten wir schon darauf", "content_without_approval_phrase"),
    ("Volle Zustimmung. Hier fehlt kein einziger Radweg mehr.", "approval_plus_content"),
])
def test_signal_classes(text, expected):
    assert gar.classify_signals(gar.compute_comment_signals(text)) == expected


def test_negation_cue_is_read_from_the_residual_only():
    assert not gar.compute_comment_signals("Warum nicht!").negation_cue
    assert gar.compute_comment_signals("Warum nicht! Aber nicht hier.").negation_cue


def test_stopwords_and_stems():
    signals = gar.compute_comment_signals("Stimmt. Die Fahrradständer und Fahrradwege")
    assert signals.residual_content_terms == ("fahrr",)
    assert gar.content_terms("und der Baum") == frozenset({"baum"})
    assert gar.content_terms(None) == frozenset()


def test_note_flags():
    assert gar.note_flags("generelle zustimmende Reaktion") == dict(
        note_general_approval=True, note_general_approval_strict=True, note_general_rejection=False,
        note_not_neutral=False)
    reverse = gar.note_flags("eigentlich eine zustimmende generelle Reaktion. Aber definitiv nicht neutral.")
    assert reverse["note_general_approval"] and not reverse["note_general_approval_strict"]
    assert reverse["note_not_neutral"]
    rejection = gar.note_flags("allgemeine ablehnende Reaktion")
    assert rejection["note_general_rejection"] and not rejection["note_general_approval"]
    assert not any(gar.note_flags(None).values())


@pytest.mark.parametrize("kwargs, expected", [
    (dict(scored=False, match=None, basis=GA, signal_class="blanket_no_content"), "not_scorable"),
    (dict(scored=True, match=True, basis=GA, signal_class="blanket_no_content"), "match"),
    (dict(scored=True, match=False, basis="explicit_target", signal_class="blanket_no_content"), "model_other_basis"),
    (dict(scored=True, match=False, basis=GA, signal_class="blanket_no_content"), "A_label_inconsistency_candidate"),
    (dict(scored=True, match=False, basis=GA, signal_class="approval_plus_content"), "rule_dependent"),
    (dict(scored=True, match=False, basis=GA, signal_class="content_without_approval_phrase"),
     "B_model_error_candidate"),
    (dict(scored=True, match=False, basis=GA, signal_class="negation_or_irony_cue"), "B_model_error_candidate"),
    (dict(scored=True, match=False, basis=GA, signal_class="back_reference"), "B_model_error_candidate"),
])
def test_heuristic_cause(kwargs, expected):
    assert gar.heuristic_cause(**kwargs) == expected


# --------------------------------------------------------------------------
# Loaders and diagnostic table
# --------------------------------------------------------------------------


def test_population_includes_every_pair_of_flagged_comments(inputs):
    table = _diagnostic(inputs).pairs
    assert set(table.comment_text) == {text for _, text, _, _ in COMMENTS} - {"Bäume pflanzen wäre schön."}
    assert len(table) == 16 and table.pair_id.is_unique
    assert list(table.columns) == list(gar.DIAGNOSTIC_COLUMNS)
    outside = _comment(table, "Danke, ja!")
    assert outside.pop_outside_runs.all() and outside.run_name.isna().all() and outside.fold.isna().all()
    assert outside.contribution_text.isna().all() and outside.status.isna().all()
    assert _comment(table, "Warum nicht!").pop_model_ga_claim_without_evidence.all()
    siehe = _comment(table, "Siehe oben")
    assert siehe.pop_annotator_note.all() and not siehe.pop_model_ga.any()
    assert table.loc[table.comment_text == "Stimmt", "fold"].tolist() == [0, 0]
    assert table.contribution_id.tolist() == sorted(table.contribution_id, key=int)


def test_scoring_provenance_and_triage(inputs):
    table = _diagnostic(inputs).pairs
    stimmt = _comment(table, "Stimmt")
    assert stimmt.scored.all() and stimmt.comment_reference_labels_disagree.all()
    assert pd.isna(stimmt.mismatch_direction.iloc[0]) and stimmt.mismatch_direction.iloc[1] == "neutral_vs_in_favor"
    assert stimmt.heuristic_cause.tolist() == ["match", "A_label_inconsistency_candidate"]
    assert stimmt.reference_provenance.tolist() == ["accepted_shown_label"] * 2
    assert stimmt.annotator_note.tolist()[0] == "generelle zustimmende Reaktion"
    gute = _comment(table, "Gute Idee, endlich mehr Fahrradständer am Bahnhof.")
    assert gute.heuristic_cause.tolist() == ["rule_dependent", "match"]
    assert gute.reference_provenance.tolist()[0] == "active_correction"
    assert gute.evidence_beyond_approval_prefix.tolist() == [True, False]
    assert gute.evidence_char_coverage.tolist()[0] == 1.0
    assert gute.related_sentence_term_overlap.tolist()[0] >= 1 and pd.isna(gute.related_sentence_term_overlap.iloc[1])
    ampel = _comment(table, "Die Ampel am Bahnhof ist viel zu lang geschaltet.")
    assert ampel.heuristic_cause.tolist() == ["B_model_error_candidate", "model_other_basis"]
    assert ampel.reference_provenance.tolist()[0] == "note_parsed"
    assert ampel.comment_other_bases.tolist() == ["explicit_target"] * 2
    assert _comment(table, "Hahaha, ok.").heuristic_cause.tolist()[0] == "B_model_error_candidate"
    super_idee = _comment(table, "Super Idee")
    assert not super_idee.scored.any() and super_idee.heuristic_cause.tolist() == ["not_scorable"] * 2
    assert super_idee.status.tolist() == ["accepted", "failed"]  # missing prediction counts as failed
    assert super_idee.reference_provenance.tolist()[0] == "unresolved"


def test_self_check_raises_when_evaluator_disagrees(inputs, monkeypatch):
    monkeypatch.setattr(evaluator_v2, "accepted_accuracy_by_basis",
                        lambda *a, **k: {GA: evaluator_v2.BasisMetrics(GA, n=99, correct=0, accuracy=0.0)})
    with pytest.raises(ValueError, match="evaluator_v2 scores"):
        _diagnostic(inputs)


def test_self_check_matches_evaluator(inputs):
    checks = _diagnostic(inputs).consistency_checks
    assert checks["fold0"] == dict(general_approval_pairs=5, general_approval_correct=2, matches_evaluator_v2=True)
    assert checks["heldout"]["general_approval_pairs"] == 2


def test_mixed_run_ids_are_rejected(inputs):
    predictions = pd.read_parquet(inputs.runs[0].predictions)
    predictions.loc[0, "run_id"] = "other"
    predictions.to_parquet(inputs.runs[0].predictions)
    with pytest.raises(ValueError, match="Mixed prediction runs"):
        gar.load_runs(inputs.runs)


def test_pairs_shared_by_two_runs_and_duplicate_names_are_rejected(inputs):
    clash = gar.RunSpec("clash", inputs.runs[0].predictions, inputs.runs[0].expected)
    with pytest.raises(ValueError, match="appears in runs"):
        gar.load_runs([inputs.runs[0], clash])
    with pytest.raises(ValueError, match="unique"):
        gar.load_runs([inputs.runs[0], inputs.runs[0]])


def _execute(db: Path, sql: str, params=()) -> None:
    conn = sqlite3.connect(db)
    conn.execute(sql, params)
    conn.commit()
    conn.close()


def test_pair_without_primary_annotation_has_empty_annotator_fields(inputs):
    _execute(inputs.annotations, "delete from stance_evaluations where comment_text = ? and evaluator = ?",
             ("Hahaha, ok.", PRIMARY_EVALUATOR))
    hahaha = _comment(_diagnostic(inputs).pairs, "Hahaha, ok.")
    assert hahaha.annotator_verdict.isna().all() and not hahaha.note_general_approval.any()


def test_duplicate_primary_annotations_are_rejected(inputs):
    _execute(inputs.annotations, "insert into stance_evaluations (contribution_id, key_message, comment_text, verdict,"
             " comment, evaluator, created_at) select contribution_id, key_message, comment_text, verdict, comment,"
             " evaluator, created_at from stance_evaluations where comment_text = 'Danke, ja!'")
    with pytest.raises(ValueError, match="Duplicate pair IDs"):
        _diagnostic(inputs)


def test_expected_pair_without_any_text_is_rejected(inputs):
    expected = pd.read_parquet(inputs.runs[1].expected)
    pd.concat([expected, pd.DataFrame([dict(pair_id="ffffffffffffffff", contribution_id="102")])]).to_parquet(
        inputs.runs[1].expected)
    with pytest.raises(ValueError, match="without text"):
        _diagnostic(inputs)


def test_annotations_are_opened_read_only(inputs):
    before = gar.sha256_file(inputs.annotations)
    annotations = gar.load_annotations(inputs.annotations)
    assert annotations.is_primary.sum() == 18 and (~annotations.is_primary).sum() == 1
    assert gar.sha256_file(inputs.annotations) == before


# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------


def test_summary_counts(inputs, result):
    summary = result.summary
    scored = summary["scored_general_approval"]
    assert (scored["total"]["pairs"], scored["total"]["correct"]) == (7, 3)
    assert scored["by_fold"]["0"]["reference_labels"] == {"in_favor": 3, "neutral": 2}
    assert scored["by_run"]["heldout"]["pairs"] == 2
    notes = summary["annotator_notes"]
    assert notes["general_approval_broad"]["pairs"] == 3 and notes["general_approval_strict"]["pairs"] == 2
    assert notes["general_rejection"] == {"pairs": 1, "comments": 1}
    assert notes["not_neutral"] == {"pairs": 1, "comments": 1}
    assert notes["general_approval_broad_outside_runs"]["comments"] == 1
    assert summary["pack"]["comments"] == 8 and summary["pack"]["key_message_rows"] == 16
    assert summary["accepted_general_approval_unresolved_reference"] == {"pairs": 1, "comments": 1}
    assert summary["general_approval_claim_without_evidence"] == {"pairs": 2, "comments": 1, "by_status": {"failed": 2}}
    assert summary["secondary_annotations"] == {"rows_total": 1, "rows_on_pack_comments": 1}
    assert summary["scored_general_approval_comments_with_disagreeing_reference_labels"] == 4
    assert summary["analysis_only"] is True and summary["new_inference_requests"] == 0
    assert summary["input_sha256"]["annotations"]["sha256"] == gar.sha256_file(inputs.annotations)


def test_outside_runs_contributions_in_a_fold_counts_contributions_not_pairs(inputs):
    """Two noted pairs on one outside contribution that has a fold count as one contribution."""
    _execute(inputs.annotations, "update stance_evaluations set comment = 'generelle zustimmende Reaktion' "
             "where comment_text = 'Danke, ja!' and comment is null")
    assignment = json.loads(inputs.folds.read_text())
    assignment["folds"][4]["test_contribution_ids"] = ["103"]
    inputs.folds.write_text(json.dumps(assignment))
    outside = _run(inputs, inputs.root / "fold_out").summary["annotator_notes"]["general_approval_broad_outside_runs"]
    assert (outside["pairs"], outside["comments"], outside["contributions"]) == (2, 1, 1)
    assert outside["contributions_in_a_fold"] == 1


def test_summary_contains_no_texts_notes_or_names(result):
    serialized = (result.output_dir / gar.SUMMARY_FILE).read_text(encoding="utf-8")
    forbidden = [text for _, text, _, _ in COMMENTS] + [km for kms in KEY_MESSAGES.values() for km in kms]
    forbidden += NOTES + [EXPLANATION, PRIMARY_EVALUATOR, SECONDARY_EVALUATOR, "Zweitmeinung"]
    assert [text for text in forbidden if text in serialized] == []
    assert json.loads(serialized) == result.summary


def test_diagnostic_content_hash_is_deterministic(inputs):
    first = _run(inputs, inputs.root / "a")
    second = _run(inputs, inputs.root / "b", now=datetime(2026, 10, 8, tzinfo=timezone.utc))
    assert first.summary["created_utc"] != second.summary["created_utc"]
    assert first.summary["diagnostic_content_sha256"] == second.summary["diagnostic_content_sha256"]


# --------------------------------------------------------------------------
# Review pack
# --------------------------------------------------------------------------


def _pack(result):
    return result.files["pack"]


def test_pack_layout(result):
    workbook = load_workbook(_pack(result))
    assert workbook.sheetnames == list(gar.SHEET_ORDER)
    assert workbook.active.title == "Anleitung"
    assert workbook["Listen"].sheet_state == "hidden" and workbook["Meta"].sheet_state == "veryHidden"
    assert workbook.properties.creator == "stance-detection-app"
    assert workbook.properties.created == datetime(2026, 10, 7, 12, 0)
    for name, specs, rows in (("Kommentare", gar.COMMENT_COLUMNS, 8), ("Kernbotschaften", gar.KM_COLUMNS, 16)):
        ws = workbook[name]
        assert ws.freeze_panes == "C3" and ws.row_dimensions[1].hidden
        assert [c.value for c in ws[1]] == [s.key for s in specs]
        assert [c.value for c in ws[2]] == [s.header or None for s in specs]
        assert ws.max_row == 2 + rows
        assert not ws.protection.sheet  # protection would block expanding the column groups
        for column, spec in enumerate(specs, start=1):
            dimension = ws.column_dimensions[gar.get_column_letter(column)]
            assert dimension.width == spec.width
            assert bool(dimension.hidden) == (spec.group in ("background", "team", "hidden"))
            assert dimension.outline_level == (1 if spec.group in ("background", "team") else 0)
            assert bool(dimension.collapsed) == (spec.group == "separator")
            cell = ws.cell(row=3, column=column)
            assert cell.font.name == "Arial" and cell.alignment.wrap_text
            if spec.editable:
                assert cell.value is None and not cell.protection.locked
            else:
                assert cell.protection.locked
        validations = {str(dv.sqref): dv.formula1 for dv in ws.data_validations.dataValidation}
        expected = {}
        for column, spec in enumerate(specs, start=1):
            if spec.list_name:
                letter = gar.get_column_letter(column)
                expected[f"{letter}3:{letter}{2 + rows}"] = spec.list_name
        assert set(validations) == set(expected)
        lists = workbook["Listen"]
        for sqref, list_name in expected.items():
            letter = gar.get_column_letter([c.value for c in lists[1]].index(list_name) + 1)
            assert validations[sqref] == f"=Listen!${letter}$2:${letter}${len(gar.OPTION_SETS[list_name]) + 1}"
    comments = workbook["Kommentare"]
    assert comments["A3"].value == "G001" and workbook["Kernbotschaften"]["A4"].value == "G001.2"
    assert "Fahrradständer" not in str(comments["A3"].value)
    overview = [c.value for row in workbook["Übersicht"].iter_rows() for c in row if c.value]
    assert not any(isinstance(v, str) and v.startswith("=") for v in overview)
    assert any(isinstance(v, str) and v.startswith("Quelle: general_approval_review.safe_summary.json") for v in overview)
    assert workbook["Anleitung"].protection.sheet and workbook["Anleitung"].protection.password is None
    texts = [c.value for ws in workbook for row in ws.iter_rows() for c in row if isinstance(c.value, str)]
    assert not any("\u2014" in t for t in texts)
    assert not any(PRIMARY_EVALUATOR in t or SECONDARY_EVALUATOR in t for t in texts)


def _sheet_xml(path: Path, title: str) -> str:
    workbook = load_workbook(path)
    index = workbook.sheetnames.index(title) + 1
    with ZipFile(path) as archive:
        return archive.read(f"xl/worksheets/sheet{index}.xml").decode("utf-8")


@pytest.mark.parametrize("title, specs", [("Kommentare", gar.COMMENT_COLUMNS), ("Kernbotschaften", gar.KM_COLUMNS)])
def test_column_groups_are_persisted_collapsed_and_separate(result, title, specs):
    xml = _sheet_xml(_pack(result), title)
    assert re.search(r'<sheetFormatPr[^>]*outlineLevelCol="1"', xml)
    assert "<sheetProtection" not in xml
    columns = {int(m.group(1)): m.group(0) for m in re.finditer(r'<col [^>]*\bmin="(\d+)"[^>]*/>', xml)}
    for index, spec in enumerate(specs, start=1):
        attributes = columns[index]
        grouped = spec.group in ("background", "team")
        assert ('outlineLevel="1"' in attributes) == grouped
        assert ('hidden="1"' in attributes) == (grouped or spec.group == "hidden")
        assert ('collapsed="1"' in attributes) == (spec.group == "separator")
    groups = [spec.group for spec in specs]
    for index, group in enumerate(groups):
        if group in ("background", "team") and (index + 1 == len(groups) or groups[index + 1] != group):
            assert groups[index + 1] == "separator"  # plus button on a visible column, groups never touch


def _all_texts(path: Path) -> list[str]:
    workbook = load_workbook(path)
    return [c.value for ws in workbook for row in ws.iter_rows() for c in row if isinstance(c.value, str)]


def test_pack_shows_no_missing_value_markers(result):
    texts = _all_texts(_pack(result))
    assert [t for t in texts if re.search(r"\b(nan|NaN|None)\b|<NA>", t)] == []
    comments = load_workbook(_pack(result))["Kommentare"]
    columns = _columns(comments)
    failed = [comments.cell(r, columns["h_modell"]).value for r in range(3, comments.max_row + 1)
              if comments.cell(r, columns["comment_text"]).value == "Warum nicht!"]
    assert failed == ["1. kein Label - keine Einordnung - fehlgeschlagen\n2. kein Label - keine Einordnung - fehlgeschlagen"]
    kms = load_workbook(_pack(result))["Kernbotschaften"]
    km_columns = _columns(kms)
    bases = {kms.cell(r, km_columns["h_modell_basis"]).value for r in range(3, kms.max_row + 1)}
    assert None not in bases and "-" in bases


def test_pack_decodes_html_entities_for_display(inputs, tmp_path):
    diagnostic = _diagnostic(inputs)
    table = diagnostic.pairs.copy()
    first = table.comment_id.iloc[0]
    raw = 'Volle Zustimmung &gt;&gt;&quot;kein Platz&quot;&lt;&lt;'
    table.loc[table.comment_id == first, "comment_text"] = raw
    table.loc[table.comment_id == first, "key_message"] = "Fu&szlig;weg"
    table["comment_evidence"] = [[raw] if cid == first else ev for cid, ev in zip(table.comment_id,
                                                                                table.comment_evidence)]
    summary = gar.build_safe_summary(diagnostic, gar.load_annotations(inputs.annotations), inputs={},
                                     created_utc="20261007T120000000000Z", role="primary", tiers=("A", "B"))
    path = tmp_path / "pack.xlsx"
    gar.write_review_pack(table, summary, path)
    workbook = load_workbook(path)
    comments, kms = workbook["Kommentare"], workbook["Kernbotschaften"]
    shown = 'Volle Zustimmung >>"kein Platz"<<'
    assert comments.cell(3, _columns(comments)["comment_text"]).value == shown
    assert comments.cell(3, _columns(comments)["h_belege"]).value == f'"{shown}"'
    assert kms.cell(3, _columns(kms)["key_message"]).value == "Fußweg"
    assert not any("&gt;" in t or "&quot;" in t for t in _all_texts(path))
    assert gar.read_review_pack(path).comments.shape[0] == 8  # fingerprints use the displayed text


def test_anleitung_uses_the_same_invented_examples_for_every_rule(result):
    rows = [row[1] or row[0] for row in load_workbook(_pack(result))["Anleitung"].iter_rows(values_only=True)
            if row[0] or row[1]]
    text = "\n".join(rows)
    for real in ("Stimmt", "Warum nicht", "Super Idee", "Danke, ja", "Ampelphase", "Gerade weil"):
        assert real not in text
    assert "Blattschutz" not in text and '".' not in text
    example_rules = [r for r in rows if r.startswith("Regel ") and "„" in r and "alle drei" in r]
    assert len(example_rules) == 3  # one outcome line per example, every rule shown for both examples
    assert all("Regel 1" in r or "Regel 3" in r for r in example_rules)
    assert sum("Regel 2: alle drei „dafür“" in r for r in rows) == 2
    assert "Hintergrund: Ihre damaligen Notizen" in [s.header for s in gar.COMMENT_COLUMNS
                                                    if s.group == "background"]


def test_pack_guards_formulas_and_illegal_characters(inputs, tmp_path):
    diagnostic = _diagnostic(inputs)
    table = diagnostic.pairs.copy()
    formula = '=HYPERLINK("http://example.invalid","Super Idee")'
    first, second = table.comment_id.unique()[:2]
    table.loc[table.comment_id == first, "comment_text"] = formula
    table.loc[table.comment_id == second, "comment_text"] = "Stimmt\x07 genau"
    summary = gar.build_safe_summary(diagnostic, gar.load_annotations(inputs.annotations), inputs={},
                                     created_utc="20261007T120000000000Z", role="primary", tiers=("A", "B"))
    path = tmp_path / "pack.xlsx"
    gar.write_review_pack(table, summary, path)
    assert gar.count_illegal_characters(table) == 2  # one character, on both key-message rows
    ws = load_workbook(path)["Kommentare"]
    assert ws["B3"].value == formula and ws["B3"].data_type == "s"
    assert ws["B4"].value == "Stimmt\ufffd genau"
    assert gar.read_review_pack(path).comments.shape[0] == 8
    with pytest.raises(FileExistsError):
        gar.write_review_pack(table, summary, path)


def test_safe_text():
    assert gar.safe_text(None) == (None, 0)
    assert gar.safe_text(" a\r\nb\x01 ") == ("a\nb\ufffd", 1)
    assert len(gar.safe_text("x" * 40000)[0]) == gar.EXCEL_CELL_LIMIT


# --------------------------------------------------------------------------
# Reader
# --------------------------------------------------------------------------


def _columns(ws):
    return {cell.value: cell.column for cell in ws[1] if cell.value}


def _fill(path: Path, target: Path) -> None:
    workbook = load_workbook(path)
    comments, kms = workbook["Kommentare"], workbook["Kernbotschaften"]
    columns = _columns(comments)
    for row in range(3, comments.max_row + 1):
        comments.cell(row, columns["d_einordnung"]).value = "Pauschale Zustimmung ohne eigenen Inhalt"
        comments.cell(row, columns["d_regel"]).value = "Regel 3: neutral, außer mit eigenem Inhalt"
        comments.cell(row, columns["d_reichweite"]).value = "nur inhaltlich angesprochene"
        comments.cell(row, columns["d_anmerkung"]).value = f"  Notiz {row} "
    comments.cell(3, columns["t_ursache"]).value = "B Modellfehler"
    km_columns = _columns(kms)
    for row in range(3, kms.max_row + 1):
        kms.cell(row, km_columns["d_km_angesprochen"]).value = "ja" if row % 2 else "nein"
    _set_answer(workbook, "g_regel_insgesamt", "Regel 1: immer neutral")
    _set_answer(workbook, "g_begruendung", "Weil.")
    workbook.save(target)


@pytest.fixture
def filled(result, tmp_path):
    target = tmp_path / "filled.xlsx"
    _fill(_pack(result), target)
    return target


def test_reader_round_trip(filled, result):
    before = gar.sha256_file(filled)
    answers = gar.read_review_pack(filled)
    assert gar.sha256_file(filled) == before
    assert answers.meta["pack_schema_version"] == gar.PACK_SCHEMA_VERSION
    assert answers.global_answers == {"g_regel_insgesamt": "rule_1_always_neutral", "g_begruendung": "Weil."}
    comments = answers.comments
    assert comments.nr.tolist()[0] == "G001" and len(comments) == 8
    assert set(comments.d_einordnung) == {"blanket_no_content"}
    assert set(comments.d_regel) == {"rule_3_neutral_unless_own_content"}
    assert set(comments.d_reichweite) == {"addressed_only"}
    assert comments.d_anmerkung.tolist()[0] == "Notiz 3"
    assert comments.t_ursache.iloc[0] == "B_model_error" and comments.t_ursache.iloc[1:].isna().all()
    assert set(answers.key_messages.d_km_angesprochen) == {"yes", "no"}
    assert len(answers.key_messages) == 16


def test_unanswered_pack_reads_as_empty(result):
    answers = gar.read_review_pack(_pack(result))
    for frame in (answers.comments, answers.key_messages):
        for spec in (*gar.COMMENT_COLUMNS, *gar.KM_COLUMNS):
            if spec.editable and spec.key in frame:
                assert all(value is None for value in frame[spec.key])
    assert answers.global_answers == {"g_regel_insgesamt": None, "g_begruendung": None}


def test_reader_tolerates_reordered_rows_and_columns(filled, tmp_path):
    workbook = load_workbook(filled)
    for name in ("Kommentare", "Kernbotschaften"):
        ws = workbook[name]
        rows = [[c.value for c in row] for row in ws.iter_rows(min_row=3)]
        for offset, values in enumerate(reversed(rows)):
            for column, value in enumerate(values, start=1):
                ws.cell(3 + offset, column).value = value
        ws.insert_cols(1)
    workbook.save(tmp_path / "moved.xlsx")
    answers = gar.read_review_pack(tmp_path / "moved.xlsx")
    assert answers.comments.nr.tolist()[0] == "G001"
    assert set(answers.comments.d_einordnung) == {"blanket_no_content"}


def _mutate(filled, tmp_path, mutate):
    workbook = load_workbook(filled)
    mutate(workbook)
    target = tmp_path / "mutated.xlsx"
    workbook.save(target)
    return target


def _set(workbook, sheet, key, row, value):
    ws = workbook[sheet]
    ws.cell(row, _columns(ws)[key]).value = value


def _set_answer(workbook, name, value):
    sheet, coordinate = next(iter(workbook.defined_names[name].destinations))
    workbook[sheet][coordinate.replace("$", "")].value = value


def _duplicate_comment_id(workbook):
    ws = workbook["Kommentare"]
    _set(workbook, "Kommentare", "comment_id", 4, ws.cell(3, _columns(ws)["comment_id"]).value)


@pytest.mark.parametrize("mutate, message", [
    (lambda wb: _set(wb, "Kommentare", "d_einordnung", 4, "Quatsch"), "unknown value"),
    (lambda wb: _set(wb, "Kernbotschaften", "d_km_angesprochen", 4, "vielleicht"), "unknown value"),
    (lambda wb: _set(wb, "Kommentare", "comment_text", 3, "Geändert"), "fingerprint mismatch"),
    (lambda wb: _set(wb, "Kernbotschaften", "key_message", 3, "Geändert"), "fingerprint mismatch"),
    (lambda wb: wb["Kommentare"].delete_rows(3), "rows missing"),
    (lambda wb: _set(wb, "Kommentare", "comment_id", 4, None), "missing comment_id"),
    (_duplicate_comment_id, "duplicate comment_id"),
    (lambda wb: _set(wb, "Kommentare", "d_regel", 1, None), "missing columns"),
    (lambda wb: _set(wb, "Kommentare", "nr", 1, "comment_id"), "appears twice"),
    (lambda wb: wb["Meta"].cell(1, 2, "ga_review_pack_v0"), "Unsupported pack schema"),
    (lambda wb: setattr(wb["Meta"].cell(7, 2), "value", None), "Meta sheet lacks"),
    (lambda wb: wb.defined_names.pop("g_begruendung"), "Defined name"),
    (lambda wb: _set_answer(wb, "g_regel_insgesamt", "Regel 9"), "unknown value"),
])
def test_reader_rejects_invalid_packs(filled, tmp_path, mutate, message):
    with pytest.raises(ValueError, match=message):
        gar.read_review_pack(_mutate(filled, tmp_path, mutate))


def test_reader_rejects_pair_moved_to_another_comment(filled, tmp_path):
    def mutate(workbook):
        ws = workbook["Kernbotschaften"]
        columns = _columns(ws)
        other = ws.cell(ws.max_row, columns["comment_id"]).value
        pair_id, key_message = ws.cell(3, columns["pair_id"]).value, ws.cell(3, columns["key_message"]).value
        ws.cell(3, columns["comment_id"]).value = other
        ws.cell(3, columns["row_fingerprint"]).value = gar.row_fingerprint(pair_id, other, key_message)
    with pytest.raises(ValueError, match="does not belong"):
        gar.read_review_pack(_mutate(filled, tmp_path, mutate))


def test_reader_rejects_non_pack(tmp_path):
    Workbook().save(tmp_path / "empty.xlsx")
    with pytest.raises(ValueError, match="Meta sheet missing"):
        gar.read_review_pack(tmp_path / "empty.xlsx")


# --------------------------------------------------------------------------
# Output exclusivity
# --------------------------------------------------------------------------


def test_artifact_hashes_are_written_last_and_match(result):
    marker = json.loads((result.output_dir / gar.HASHES_FILE).read_text())
    assert marker["complete"] is True
    assert sorted(path.name for path in result.output_dir.iterdir()) == sorted(
        [gar.HASHES_FILE, *marker["files"]])
    for name, digest in marker["files"].items():
        assert gar.sha256_file(result.output_dir / name) == digest
    assert pd.read_parquet(result.files["diagnostic"]).shape == (16, len(gar.DIAGNOSTIC_COLUMNS))


def test_existing_output_dir_is_never_overwritten(inputs, result):
    before = {p.name: gar.sha256_file(p) for p in result.output_dir.iterdir()}
    with pytest.raises(FileExistsError):
        _run(inputs, result.output_dir)
    assert {p.name: gar.sha256_file(p) for p in result.output_dir.iterdir()} == before


def test_failure_leaves_partial_dir_without_marker(inputs, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("pack failed")
    monkeypatch.setattr(gar, "write_review_pack", boom)
    with pytest.raises(RuntimeError, match="pack failed"):
        _run(inputs, inputs.root / "partial")
    assert sorted(p.name for p in (inputs.root / "partial").iterdir()) == [gar.DIAGNOSTIC_FILE, gar.SUMMARY_FILE]


def test_missing_input_and_changed_input_are_rejected(inputs, monkeypatch):
    missing = SimpleNamespace(**{**vars(inputs), "folds": inputs.root / "nope.json"})
    with pytest.raises(FileNotFoundError, match="nope.json"):
        _run(missing, inputs.root / "x")
    original = gar.build_safe_summary

    def tamper(*args, **kwargs):
        inputs.folds.write_text(inputs.folds.read_text() + "\n")
        return original(*args, **kwargs)
    monkeypatch.setattr(gar, "build_safe_summary", tamper)
    with pytest.raises(RuntimeError, match="changed"):
        _run(inputs, inputs.root / "y")
    assert not (inputs.root / "y").exists()


def test_default_output_dir_is_timestamped(inputs, monkeypatch):
    monkeypatch.setattr(gar, "EXPERIMENTS_DIR", inputs.root)
    result = _run(inputs, None)
    assert result.output_dir == inputs.root / "general_approval_review_20261007T120000000000Z"
    assert result.files["pack"].name == "general_approval_review_pack_v1_20261007T120000000000Z.xlsx"


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def test_cli_parser_defaults():
    args = cli.build_parser().parse_args(["general-approval-review"])
    assert args.run is None and args.source is None and args.output_dir is None
    assert (args.reference, args.folds) == (str(cli.DEFAULT_REFERENCE_PATH), str(cli.DEFAULT_FOLDS_PATH))
    assert args.annotations == str(gar.DEFAULT_ANNOTATIONS_DB)
    assert (args.role, args.tiers) == ("primary", "A+B")
    assert args.func is cli._run_general_approval_review


def _offline(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("network or .env access attempted")
    monkeypatch.setattr(cli, "load_dotenv", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)


def test_cli_end_to_end_offline(inputs, monkeypatch, capsys):
    _offline(monkeypatch)
    output = inputs.root / "cli_out"
    command = ["general-approval-review", "--reference", str(inputs.reference), "--folds", str(inputs.folds),
               "--annotations", str(inputs.annotations), "--output-dir", str(output)]
    for spec in inputs.runs:
        command += ["--run", spec.name, str(spec.predictions), str(spec.expected)]
    for source in inputs.sources:
        command += ["--source", str(source)]
    cli.main(command)
    printed = capsys.readouterr().out
    assert "fold 0" in printed and "Review pack: 8 comments" in printed
    assert (output / gar.HASHES_FILE).exists()


def test_cli_uses_default_runs_and_sources(inputs, monkeypatch):
    _offline(monkeypatch)
    monkeypatch.setattr(cli, "DEFAULT_RUNS", inputs.runs)
    monkeypatch.setattr(cli, "DEFAULT_SOURCES", inputs.sources)
    output = inputs.root / "cli_defaults"
    cli.main(["general-approval-review", "--reference", str(inputs.reference), "--folds", str(inputs.folds),
              "--annotations", str(inputs.annotations), "--output-dir", str(output)])
    summary = json.loads((output / gar.SUMMARY_FILE).read_text())
    assert summary["run_ids"] == {"fold0": "run_a", "heldout": "run_b"}


def test_module_has_no_inference_imports():
    source = Path(gar.__file__).read_text(encoding="utf-8")
    for forbidden in ("llm_handler", "benchmark", "httpx", "dotenv", "openai", "requests"):
        assert f"import {forbidden}" not in source and f"from {forbidden}" not in source


# --------------------------------------------------------------------------
# Committed artifacts (guarded; writes only to tmp_path)
# --------------------------------------------------------------------------

COMMITTED = [ROOT / p for p in ("dataset/pairs.parquet", "dataset/folds_v1.json", str(gar.DEFAULT_ANNOTATIONS_DB),
                                *[str(s.predictions) for s in gar.DEFAULT_RUNS],
                                *[str(s.expected) for s in gar.DEFAULT_RUNS], *map(str, gar.DEFAULT_SOURCES))]


@pytest.mark.skipif(not all(p.exists() for p in COMMITTED), reason="committed inputs not available")
def test_committed_artifacts_reproduce_known_counts(tmp_path):
    runs = [gar.RunSpec(s.name, ROOT / s.predictions, ROOT / s.expected) for s in gar.DEFAULT_RUNS]
    before = {p: gar.sha256_file(p) for p in COMMITTED}
    result = gar.run_general_approval_review(
        runs=runs, reference_path=ROOT / "dataset/pairs.parquet", folds_path=ROOT / "dataset/folds_v1.json",
        annotations_path=ROOT / gar.DEFAULT_ANNOTATIONS_DB, source_paths=[ROOT / s for s in gar.DEFAULT_SOURCES],
        output_dir=tmp_path / "out")
    summary = result.summary
    scored = summary["scored_general_approval"]
    assert (scored["total"]["pairs"], scored["total"]["correct"], scored["total"]["comments"]) == (108, 65, 40)
    assert scored["total"]["reference_labels"] == {"against": 3, "in_favor": 40, "neutral": 65}
    assert [(b["pairs"], b["correct"]) for b in scored["by_run"].values()] == [(31, 24), (77, 41)]
    notes = summary["annotator_notes"]
    assert (notes["general_approval_broad"]["pairs"], notes["general_approval_strict"]["pairs"]) == (87, 84)
    assert (summary["pack"]["comments"], summary["pack"]["key_message_rows"]) == (57, 178)
    assert summary["pack"]["comments_outside_runs"] == 3
    assert gar.read_review_pack(result.files["pack"]).comments.shape[0] == 57
    assert {p: gar.sha256_file(p) for p in COMMITTED} == before
    assert hashlib.sha256(b"").hexdigest() not in before.values()
