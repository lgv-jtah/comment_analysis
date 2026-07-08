"""Tests for the markdown report writer (no network, no LLM calls)."""
from datetime import datetime, timezone
from pathlib import Path

from evaluator import (
    StanceEvaluator,
    RunMetadata,
    write_markdown_report,
    format_markdown_report,
    JudgeComparisonRow,
    format_judge_comparison_report,
    write_judge_comparison_report,
)

FIXED_TS = datetime(2026, 6, 17, 14, 23, 0, tzinfo=timezone.utc)


def _gold_row(row_index: int, gold_stance: str) -> dict:
    return {"row_index": row_index, "gold_stance": gold_stance}


def _meta(**overrides):
    defaults = dict(
        model_name="TestModel/7B",
        base_url="http://localhost:5555/v1",
        prompt_key="base",
        fast_path_count=3,
        ensemble_count=5,
        llm_count=10,
        total_elapsed=12.34,
        notes=None,
    )
    return RunMetadata(**{**defaults, **overrides})


def _three_class_report():
    """Build a deterministic 3-class report.

    Gold:  a a a b b b c c c
    Pred:  a a b b b c c c a
    """
    gold_data = [
        _gold_row(10, "a"),
        _gold_row(11, "a"),
        _gold_row(12, "a"),
        _gold_row(13, "b"),
        _gold_row(14, "b"),
        _gold_row(15, "b"),
        _gold_row(16, "c"),
        _gold_row(17, "c"),
        _gold_row(18, "c"),
    ]
    predictions = {
        10: "a",
        11: "a",
        12: "b",
        13: "b",
        14: "b",
        15: "c",
        16: "c",
        17: "c",
        18: "a",
    }
    return StanceEvaluator(gold_data=gold_data).evaluate(predictions)


def test_filename_format(tmp_path):
    report = _three_class_report()
    path = write_markdown_report(
        report, _meta(prompt_key="base"), output_dir=tmp_path, timestamp=FIXED_TS
    )
    assert path.name == "2026-06-17T142300Z_base.md"


def test_slug_sanitizes_prompt_key(tmp_path):
    report = _three_class_report()
    path = write_markdown_report(
        report,
        _meta(prompt_key="My Prompt v2!"),
        output_dir=tmp_path,
        timestamp=FIXED_TS,
    )
    assert "my_prompt_v2" in path.name
    assert path.name.endswith(".md")


def test_creates_reports_dir_when_absent(tmp_path):
    report = _three_class_report()
    target = tmp_path / "nested" / "reports"
    assert not target.exists()
    path = write_markdown_report(
        report, _meta(), output_dir=target, timestamp=FIXED_TS
    )
    assert target.exists()
    assert path.exists()


def test_no_overwrite_on_same_timestamp(tmp_path):
    report = _three_class_report()
    first = write_markdown_report(
        report, _meta(notes="first"), output_dir=tmp_path, timestamp=FIXED_TS
    )
    first_content = first.read_text(encoding="utf-8")
    second = write_markdown_report(
        report, _meta(notes="second"), output_dir=tmp_path, timestamp=FIXED_TS
    )
    assert first != second
    # First file must be untouched by the second write.
    assert first.read_text(encoding="utf-8") == first_content
    assert "first" in first_content
    assert "second" in second.read_text(encoding="utf-8")


def test_all_sections_present(tmp_path):
    report = _three_class_report()
    path = write_markdown_report(
        report, _meta(notes=None), output_dir=tmp_path, timestamp=FIXED_TS
    )
    text = path.read_text(encoding="utf-8")
    assert "# Stance Evaluation Report" in text
    assert "## Run metadata" in text
    assert "## Overall" in text
    assert "## Per-class metrics" in text
    assert "## Confusion matrix (rows = gold, cols = predicted)" in text
    # Notes omitted when notes is None.
    assert "## Notes" not in text


def test_notes_section_rendered(tmp_path):
    report = _three_class_report()
    note = "tweaked system prompt wording"
    path = write_markdown_report(
        report, _meta(notes=note), output_dir=tmp_path, timestamp=FIXED_TS
    )
    text = path.read_text(encoding="utf-8")
    assert "## Notes" in text
    assert note in text


def test_metadata_values_appear(tmp_path):
    report = _three_class_report()
    path = write_markdown_report(
        report, _meta(), output_dir=tmp_path, timestamp=FIXED_TS
    )
    text = path.read_text(encoding="utf-8")
    assert "TestModel/7B" in text
    assert "http://localhost:5555/v1" in text
    assert "base" in text
    assert "| Fast-path count | 3 |" in text
    assert "| Ensemble count | 5 |" in text
    assert "| LLM calls | 10 |" in text
    assert "| Total routed | 18 |" in text
    assert "12.34s" in text


def test_overall_metrics_appear(tmp_path):
    report = _three_class_report()
    path = write_markdown_report(
        report, _meta(), output_dir=tmp_path, timestamp=FIXED_TS
    )
    text = path.read_text(encoding="utf-8")
    assert f"Accuracy: {report.accuracy:.4f}" in text
    assert f"Evaluated rows: {report.evaluated_rows}" in text
    assert f"Skipped rows: {report.skipped_rows}" in text
    assert f"Total rows: {report.total_rows}" in text


def test_per_class_table_values():
    report = _three_class_report()
    md = format_markdown_report(report, _meta(), FIXED_TS)
    # Label "a": tp=2, fp=1 (c->a), fn=1 (a->b) => p=2/3, r=2/3, support=3.
    metrics = report.per_class["a"]
    expected_row = (
        f"| a "
        f"| {metrics['precision']:.4f} "
        f"| {metrics['recall']:.4f} "
        f"| {metrics['f1']:.4f} "
        f"| 3 |"
    )
    assert expected_row in md
    assert "| --- | --- | --- | --- | --- |" in md


def test_confusion_matrix_rendered():
    # Include a stray predicted label ("d") not present among gold rows.
    gold_data = [_gold_row(1, "a"), _gold_row(2, "b")]
    predictions = {1: "a", 2: "d"}
    report = StanceEvaluator(gold_data=gold_data).evaluate(predictions)
    md = format_markdown_report(report, _meta(), FIXED_TS)
    # Columns derived from gold + predicted: a, b, d.
    assert "| gold \\ pred | a | b | d |" in md
    # Gold "a" predicted "a" once, nothing else.
    assert "| a | 1 | 0 | 0 |" in md
    # Gold "b" predicted "d" once.
    assert "| b | 0 | 0 | 1 |" in md


def test_empty_report_no_rows(tmp_path):
    report = StanceEvaluator(gold_data=[]).evaluate({})
    path = write_markdown_report(
        report, _meta(), output_dir=tmp_path, timestamp=FIXED_TS
    )
    text = path.read_text(encoding="utf-8")
    assert path.exists()
    assert "_No evaluated rows._" in text


def test_utf8_content(tmp_path):
    gold_data = [_gold_row(1, "ablehnung"), _gold_row(2, "ablehnung")]
    predictions = {1: "ablehnung", 2: "ablehnung"}
    report = StanceEvaluator(gold_data=gold_data).evaluate(predictions)
    path = write_markdown_report(
        report,
        _meta(notes="Überprüfung"),
        output_dir=tmp_path,
        timestamp=FIXED_TS,
    )
    text = path.read_text(encoding="utf-8")
    assert "ablehnung" in text
    assert "Überprüfung" in text


# --- judge comparison report ---


def _cmp_row(
    label,
    file_name="judged_x.parquet",
    total=1,
    judged=1,
    no_result=0,
    mean_overall=4.0,
    mean_completeness=3.0,
    mean_semantic_precision=5.0,
    mean_downstream_viability=2.0,
):
    """Build a JudgeComparisonRow with sensible defaults for report tests."""
    return JudgeComparisonRow(
        label=label,
        file_name=file_name,
        total=total,
        judged=judged,
        no_result=no_result,
        mean_overall=mean_overall,
        mean_completeness=mean_completeness,
        mean_semantic_precision=mean_semantic_precision,
        mean_downstream_viability=mean_downstream_viability,
    )


def test_judge_comparison_filename_format(tmp_path):
    rows = [_cmp_row("model-a")]
    path = write_judge_comparison_report(
        rows, output_dir=tmp_path, timestamp=FIXED_TS
    )
    assert path.name == "2026-06-17T142300Z_judge_comparison.md"


def test_judge_comparison_no_overwrite(tmp_path):
    rows = [_cmp_row("model-a")]
    first = write_judge_comparison_report(
        rows, output_dir=tmp_path, timestamp=FIXED_TS
    )
    first_content = first.read_text(encoding="utf-8")
    second = write_judge_comparison_report(
        rows, output_dir=tmp_path, timestamp=FIXED_TS
    )
    assert first != second
    assert second.name == "2026-06-17T142300Z_judge_comparison-1.md"
    # First file must be untouched by the second write.
    assert first.read_text(encoding="utf-8") == first_content


def test_judge_comparison_sections_and_rows():
    rows = [
        _cmp_row("model-a", mean_overall=4.5, mean_completeness=3.25),
        _cmp_row(
            "model-b",
            mean_overall=None,
            mean_completeness=None,
            mean_semantic_precision=None,
            mean_downstream_viability=None,
            judged=0,
            no_result=1,
        ),
    ]
    md = format_judge_comparison_report(rows, FIXED_TS)
    assert "# Key Message Judge Comparison" in md
    assert "## Summary" in md
    assert "## Per-file comparison" in md
    # 9-column table header and separator.
    assert (
        "| Rank | Model / file | Judged | No result | Total | Completeness | "
        "Semantic precision | Downstream viability | Overall |" in md
    )
    assert "| --- | --- | --- | --- | --- | --- | --- | --- | --- |" in md
    assert "model-a" in md
    assert "model-b" in md
    # None means render as N/A; numeric means are formatted to 2 decimals.
    assert "N/A" in md
    assert "4.50" in md
    assert "3.25" in md


def test_judge_comparison_ranking_and_ties():
    rows = [
        _cmp_row("first"),
        _cmp_row("second"),
        _cmp_row("third"),
    ]
    md = format_judge_comparison_report(rows, FIXED_TS)
    # Rank column matches input order 1, 2, 3.
    assert "| 1 | first |" in md
    assert "| 2 | second |" in md
    assert "| 3 | third |" in md


def test_judge_comparison_utf8_and_pipe_escaping(tmp_path):
    rows = [_cmp_row("mödel|pipe")]
    path = write_judge_comparison_report(
        rows, output_dir=tmp_path, timestamp=FIXED_TS
    )
    # Written file is valid UTF-8.
    text = path.read_text(encoding="utf-8")
    assert "mödel" in text
    # The literal pipe is escaped so the markdown table isn't broken.
    assert "mödel\\|pipe" in text
