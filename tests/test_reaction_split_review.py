"""Tests for reaction_split_review.py and review_pack_common.py (offline, no network)."""
import json
from datetime import datetime, timezone

import pandas as pd
import pytest
from openpyxl import load_workbook

import cli
import reaction_split_review as rsr
from review_pack_common import sha256_file

NOW = datetime(2026, 10, 12, 9, 0, tzinfo=timezone.utc)
SECRETS = ("SECRET_MODEL_EXPLANATION", "SECRET_ANNOTATOR_NOTE", "SECRET_SIGNAL")


def _pair(pair_id, comment_id, contribution_id, fold, km_index, key_message, comment_text, contribution_text):
    return {
        "pair_id": pair_id, "comment_id": comment_id, "contribution_id": contribution_id, "fold": fold,
        "key_message_id": f"{contribution_id}:km{km_index}" if fold is not None else None,
        "key_message": key_message, "comment_text": comment_text, "contribution_text": contribution_text,
        "contribution_text_truncated": False if contribution_text else None,
        # Columns the pack must never show.
        "explanation": "SECRET_MODEL_EXPLANATION", "annotator_note": "SECRET_ANNOTATOR_NOTE",
        "signal_class": "SECRET_SIGNAL", "final_stance": "neutral", "reference_label": "in_favor",
    }


def _population_rows():
    contribution = "Fahrradständer am Bahnhof &amp; mehr Bäume."
    return [
        _pair("p1a", "c1", "10", 0, 1, "Fahrradständer aufstellen", "Super Idee!", contribution),
        _pair("p1b", "c1", "10", 0, 2, "Bäume pflanzen", "Super Idee!", contribution),
        _pair("p2a", "c2", "10", 0, 1, "Fahrradständer aufstellen",
              "Super Idee! Am Bahnhof fehlen wirklich Fahrradständer.", contribution),
        _pair("p2b", "c2", "10", 0, 2, "Bäume pflanzen",
              "Super Idee! Am Bahnhof fehlen wirklich Fahrradständer.", contribution),
        _pair("p3a", "c3", "20", None, 1, "Parkplätze entfernen", "=Bitte nicht.", None),
    ]


def _write_population(directory, rows=None, *, tamper=False):
    directory.mkdir()
    frame = pd.DataFrame(rows or _population_rows())
    frame["fold"] = frame["fold"].astype("Int64")
    frame["contribution_text_truncated"] = frame["contribution_text_truncated"].astype("boolean")
    path = directory / rsr.POPULATION_FILE
    frame.to_parquet(path, index=False)
    digest = sha256_file(path)
    if tamper:
        digest = "0" * 64
    (directory / "artifact_hashes.json").write_text(
        json.dumps({"complete": True, "files": {rsr.POPULATION_FILE: digest}}), encoding="utf-8")
    return directory


@pytest.fixture
def pack(tmp_path):
    population = _write_population(tmp_path / "population")
    return rsr.run_reaction_split_pack(population_dir=population, extra_source=None, output_dir=tmp_path / "pack",
                                       now=NOW)


def _all_cell_values(path):
    workbook = load_workbook(path)
    values = [cell.value for ws in workbook.worksheets for row in ws.iter_rows() for cell in row]
    workbook.close()
    return [v for v in values if v is not None]


def _fill(path, target, comments=None, key_messages=None, edit=None):
    """Write answers into a copy of the pack, addressing rows by Nr. and columns by machine key."""
    workbook = load_workbook(path)
    for sheet, answers in ((rsr.COMMENTS_SHEET, comments or {}), (rsr.KM_SHEET, key_messages or {})):
        ws = workbook[sheet]
        column = {cell.value: cell.column for cell in ws[1] if cell.value}
        for row in range(3, ws.max_row + 1):
            nr = ws.cell(row=row, column=column["nr"]).value
            for key, value in answers.get(nr, {}).items():
                ws.cell(row=row, column=column[key], value=value)
    if edit:
        edit(workbook)
    workbook.save(target)
    return target


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------


def test_pack_run_writes_new_directory_with_hashes_last(pack):
    assert pack.output_dir.name == "pack"
    assert pack.files["pack"].name == "reaction_split_pack_v1_20261012T090000000000Z.xlsx"
    hashes = json.loads(pack.files["hashes"].read_text(encoding="utf-8"))
    assert hashes["complete"] is True
    assert hashes["files"] == {p.name: sha256_file(p) for k, p in pack.files.items() if k != "hashes"}


def test_pack_summary_counts_and_folds(pack):
    summary = pack.summary
    assert summary["comments"] == 3 and summary["key_message_rows"] == 5
    assert summary["comments_by_fold"] == {"0": 2, "none": 1}
    assert summary["comment_ids"] == ["c1", "c2", "c3"]
    assert summary["new_inference_requests"] == 0


def test_pack_is_blank_and_shows_no_model_or_label_columns(pack):
    values = [str(v) for v in _all_cell_values(pack.files["pack"])]
    assert not any(secret in value for secret in SECRETS for value in values)
    answers = rsr.read_reaction_split_pack(pack.files["pack"])
    assert all(issues == ["unanswered"] for issues in answers.comments.issues)
    assert all(issues == ["unanswered"] for issues in answers.key_messages.issues)


def test_pack_texts_are_decoded_and_formula_safe(pack):
    workbook = load_workbook(pack.files["pack"])
    ws = workbook[rsr.COMMENTS_SHEET]
    column = {cell.value: cell.column for cell in ws[1] if cell.value}
    contribution = ws.cell(row=3, column=column["contribution_text"]).value
    formula_like = ws.cell(row=5, column=column["comment_text"])
    assert contribution == "Fahrradständer am Bahnhof & mehr Bäume."
    assert formula_like.value == "=Bitte nicht." and formula_like.data_type == "s"
    assert ws.cell(row=5, column=column["contribution_text"]).value == "(Beitragstext liegt hier nicht vor)"
    assert workbook[rsr.META_SHEET].sheet_state == "veryHidden"
    assert workbook[rsr.ANLEITUNG_SHEET].protection.sheet is True
    workbook.close()


def test_anleitung_has_no_em_dash():
    assert not any("\u2014" in text for _, text in rsr.ANLEITUNG_TEXT)


def test_pack_refuses_existing_output_dir(tmp_path, pack):
    with pytest.raises(FileExistsError):
        rsr.run_reaction_split_pack(population_dir=tmp_path / "population", extra_source=None,
                                    output_dir=pack.output_dir, now=NOW)


def test_population_hash_mismatch_is_rejected(tmp_path):
    population = _write_population(tmp_path / "population", tamper=True)
    with pytest.raises(RuntimeError, match="changed"):
        rsr.run_reaction_split_pack(population_dir=population, extra_source=None, output_dir=tmp_path / "pack",
                                    now=NOW)
    assert not (tmp_path / "pack").exists()


def test_population_with_split_comment_rows_is_rejected(tmp_path):
    rows = _population_rows()
    rows[1], rows[2] = rows[2], rows[1]
    population = _write_population(tmp_path / "population", rows)
    with pytest.raises(ValueError, match="contiguous"):
        rsr.load_population(population)


# --------------------------------------------------------------------------
# Extra fold-0 comments
# --------------------------------------------------------------------------


def _source_row(contribution_id, key_message, comment_text, content="Fahrradständer am Bahnhof &amp; mehr Bäume."):
    return {"contribution_id": contribution_id, "contributionContent": content, "key_message": key_message,
            "key_message_type": "Wunsch", "related_sentences": ["s"], "related_sentence_spans": None,
            "key_message_explanation": "", "comment_text": comment_text, "stance": "SECRET_MODEL_EXPLANATION",
            "explanation": "SECRET_MODEL_EXPLANATION", "extraction_model_name": "x", "stance_model_name": "y"}


def _extra_inputs(tmp_path, *, conflicting=False):
    rows = [
        # Already in the population (comment c2 on contribution 10): must not be added twice.
        _source_row(10, "Fahrradständer aufstellen", "Super Idee! Am Bahnhof fehlen wirklich Fahrradständer."),
        _source_row(10, "Bäume pflanzen" if not conflicting else "Andere Kernbotschaft",
                    "Super Idee! Am Bahnhof fehlen wirklich Fahrradständer."),
        # New fold-0 comments, on contributions 10 and 5.
        _source_row(10, "Fahrradständer aufstellen", "Lieber mehr Bäume."),
        _source_row(10, "Bäume pflanzen", "Lieber mehr Bäume."),
        _source_row(5, "Spielplatz bauen", "Bitte nicht.", content="x" * 1000),
        # Contribution in fold 1: held out, never added.
        _source_row(30, "Parkplätze", "Gute Idee"),
    ]
    source = tmp_path / "source.parquet"
    pd.DataFrame(rows).to_parquet(source, index=False)
    folds = tmp_path / "folds.json"
    folds.write_text(json.dumps({
        "seed": 1, "n_splits": 2, "grouping_key": "contribution_id", "near_duplicate_check": "skipped",
        "dup_groups_merged": 0, "schema_version": "folds_v1",
        "folds": [{"train_contribution_ids": ["30"], "test_contribution_ids": ["10", "5"]},
                  {"train_contribution_ids": ["10", "5"], "test_contribution_ids": ["30"]}],
    }), encoding="utf-8")
    return source, folds


def _pair_id_for_c2(km):
    from dataset_builder import compute_pair_id
    return compute_pair_id("10", km, "Super Idee! Am Bahnhof fehlen wirklich Fahrradständer.")


def _population_with_real_c2_ids():
    rows = _population_rows()
    for row in rows:
        if row["comment_id"] == "c2":
            row["pair_id"] = _pair_id_for_c2(row["key_message"])
            row["comment_id"] = rsr.comment_id_for("10", row["comment_text"])
    return rows


def test_extra_fold0_comments_are_added_once_and_ordered_by_contribution(tmp_path):
    population = _write_population(tmp_path / "population", _population_with_real_c2_ids())
    source, folds = _extra_inputs(tmp_path)

    result = rsr.run_reaction_split_pack(population_dir=population, extra_source=source, folds_path=folds,
                                         output_dir=tmp_path / "pack", now=NOW)

    summary = result.summary
    assert summary["comments_from_review_2026_10_07"] == 3
    assert summary["comments_added_from_fold_0"] == 2 and summary["comments"] == 5
    assert summary["comments_by_fold"] == {"0": 4, "none": 1}
    assert summary["inputs"]["extra_source"]["sha256"] == sha256_file(source)
    answers = rsr.read_reaction_split_pack(result.files["pack"])
    texts = answers.comments.comment_text.tolist()
    # Contribution 5 first, then 10, then 20; the held-out contribution 30 is absent.
    assert texts[0] == "Bitte nicht." and "Gute Idee" not in texts
    assert texts[1:4] == ["Super Idee!", "Super Idee! Am Bahnhof fehlen wirklich Fahrradständer.", "Lieber mehr Bäume."]
    values = [str(v) for v in _all_cell_values(result.files["pack"])]
    assert not any(secret in value for secret in SECRETS for value in values)


def test_extra_comment_ids_match_the_model_run_derivation(tmp_path):
    source, folds = _extra_inputs(tmp_path)
    extra = rsr.load_fold_comments(source, folds)
    added = extra[extra.comment_text == "Lieber mehr Bäume."]
    assert added.key_message_id.tolist() == ["10:km1", "10:km2"]
    assert added.comment_id.iloc[0] == rsr.comment_id_for("10", "Lieber mehr Bäume.")
    truncated = extra[extra.contribution_id == "5"].contribution_text_truncated.iloc[0]
    assert bool(truncated) is True


def test_shared_comment_with_different_pairs_is_rejected(tmp_path):
    population = _write_population(tmp_path / "population", _population_with_real_c2_ids())
    source, folds = _extra_inputs(tmp_path, conflicting=True)
    with pytest.raises(ValueError, match="pair IDs differ"):
        rsr.run_reaction_split_pack(population_dir=population, extra_source=source, folds_path=folds,
                                    output_dir=tmp_path / "pack", now=NOW)


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------


def test_filled_pack_is_parsed_with_issues(tmp_path, pack):
    filled = _fill(pack.files["pack"], tmp_path / "filled.xlsx", comments={
        "R001": {"d_reaktion": "ja", "d_polaritaet": "positiv", "d_hinweis_zitat": "„Super Idee!“",
                 "d_hinweis_begruendung": "pauschales Lob", "d_trennbar": "ja"},
        "R002": {"d_reaktion": "ja", "d_polaritaet": "positiv", "d_hinweis_zitat": "Super Idee!",
                 "d_resttext": "Am Bahnhof fehlen wirklich\nFahrradständer.\nerfunden", "d_trennbar": "ja"},
        "R003": {"d_reaktion": "nein", "d_polaritaet": "negativ", "d_trennbar": "ja"},
    }, key_messages={
        "R001.1": {"d_haltung": "nicht angesprochen"}, "R001.2": {"d_haltung": "dafür"},
        "R002.1": {"d_haltung": "dafür"}, "R002.2": {"d_haltung": "nicht angesprochen"},
    })

    answers = rsr.read_reaction_split_pack(filled)
    comments = answers.comments.set_index("nr")
    key_messages = answers.key_messages.set_index("nr")

    assert comments.loc["R001", "hint_quotes"] == ["Super Idee!"]
    assert comments.loc["R001", "rest_quotes"] == [] and comments.loc["R001", "issues"] == []
    assert comments.loc["R002", "rest_quotes"] == ["Am Bahnhof fehlen wirklich", "Fahrradständer.", "erfunden"]
    assert comments.loc["R002", "issues"] == ["rest_quote_not_in_comment:3"]
    # Without a general reaction the whole comment is the rest; a polarity is contradictory.
    assert comments.loc["R003", "rest_quotes"] == ["=Bitte nicht."]
    assert comments.loc["R003", "rest_is_whole_comment"] is True
    assert comments.loc["R003", "issues"] == ["polarity_without_general_reaction"]
    assert key_messages.loc["R001.2", "issues"] == ["stance_without_rest_text"]
    assert key_messages.loc["R002.1", "stance"] == "in_favor" and key_messages.loc["R002.1", "issues"] == []
    assert key_messages.loc["R003.1", "issues"] == ["unanswered"]


def test_missing_polarity_and_hint_are_flagged(tmp_path, pack):
    filled = _fill(pack.files["pack"], tmp_path / "filled.xlsx",
                   comments={"R001": {"d_reaktion": "ja"}})
    issues = rsr.read_reaction_split_pack(filled).comments.set_index("nr").loc["R001", "issues"]
    assert issues == ["missing_polarity", "missing_hint_quote", "missing_splittable"]


def test_edited_comment_text_is_rejected(tmp_path, pack):
    filled = _fill(pack.files["pack"], tmp_path / "filled.xlsx",
                   comments={"R001": {"comment_text": "Super Idee! (geändert)"}})
    with pytest.raises(ValueError, match="fingerprint"):
        rsr.read_reaction_split_pack(filled)


def test_unknown_list_value_is_rejected(tmp_path, pack):
    filled = _fill(pack.files["pack"], tmp_path / "filled.xlsx", comments={"R001": {"d_reaktion": "vielleicht"}})
    with pytest.raises(ValueError, match="unknown value"):
        rsr.read_reaction_split_pack(filled)


def test_reordered_rows_and_columns_are_read(tmp_path, pack):
    def reorder(workbook):
        ws = workbook[rsr.COMMENTS_SHEET]
        ws.move_range("A3:N3", rows=10)
        ws.insert_cols(2)

    filled = _fill(pack.files["pack"], tmp_path / "filled.xlsx",
                   comments={"R001": {"d_reaktion": "nein", "d_trennbar": "ja"}}, edit=reorder)
    comments = rsr.read_reaction_split_pack(filled).comments
    assert comments.nr.tolist() == ["R001", "R002", "R003"]
    assert comments.set_index("nr").loc["R001", "general_reaction"] == "no"


def test_wrong_schema_version_is_rejected(tmp_path, pack):
    def downgrade(workbook):
        workbook[rsr.META_SHEET]["B1"] = "other_v0"

    filled = _fill(pack.files["pack"], tmp_path / "filled.xlsx", edit=downgrade)
    with pytest.raises(ValueError, match="schema version"):
        rsr.read_reaction_split_pack(filled)


def test_export_labels_writes_versioned_label_set(tmp_path, pack):
    filled = _fill(pack.files["pack"], tmp_path / "filled.xlsx", comments={
        "R001": {"d_reaktion": "ja", "d_polaritaet": "positiv", "d_hinweis_zitat": "Super Idee!",
                 "d_trennbar": "ja"},
    }, key_messages={"R001.1": {"d_haltung": "nicht angesprochen"}})

    result = rsr.export_reaction_split_labels(filled, output_dir=tmp_path / "labels", now=NOW)

    comments = pd.read_parquet(result.files["comments"])
    key_messages = pd.read_parquet(result.files["key_messages"])
    assert len(comments) == 3 and len(key_messages) == 5
    assert set(comments.labels_schema_version) == {rsr.LABELS_SCHEMA_VERSION}
    summary = result.summary
    assert summary["general_reaction"] == {"unanswered": 2, "yes": 1}
    assert summary["polarity"] == {"positive": 1}
    assert summary["comments_without_issues"] == 1
    assert summary["pack"]["sha256"] == sha256_file(filled)
    hashes = json.loads(result.files["hashes"].read_text(encoding="utf-8"))
    assert set(hashes["files"]) == {rsr.LABELS_COMMENTS_FILE, rsr.LABELS_KM_FILE, rsr.LABELS_SUMMARY_FILE}


def test_split_quotes_strips_quote_marks_and_blank_lines():
    assert rsr.split_quotes(" „Super!“ \n\n 'Danke' ") == ["Super!", "Danke"]
    assert rsr.split_quotes(None) == []


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def test_cli_pack_and_read(tmp_path, capsys):
    population = _write_population(tmp_path / "population")
    cli.main(["reaction-split-pack", "--population-dir", str(population), "--no-extra-fold0",
              "--output-dir", str(tmp_path / "pack")])
    pack_path = next((tmp_path / "pack").glob("*.xlsx"))
    cli.main(["reaction-split-read", "--pack", str(pack_path), "--output-dir", str(tmp_path / "labels")])
    out = capsys.readouterr().out
    assert "3 comments" in out and "Comments: 3" in out
    assert (tmp_path / "labels" / "artifact_hashes.json").is_file()


def test_old_general_approval_review_subcommand_is_gone():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["general-approval-review"])
