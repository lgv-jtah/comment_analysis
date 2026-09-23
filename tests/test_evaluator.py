"""Pure-logic tests for the StanceEvaluator (no network, no LLM calls)."""
import json
from pathlib import Path

import pytest

from evaluator import StanceEvaluator, write_result_json

GOLD_PATH = Path(__file__).resolve().parent.parent / "gold_labels_corrected.json"


def _gold_row(row_index: int, gold_stance: str) -> dict:
    return {"row_index": row_index, "gold_stance": gold_stance}


def test_hand_computed_multiclass_metrics():
    """Exact metrics for a small 3-class set with known tp/fp/fn."""
    # Gold:   a a a b b b c c c
    # Pred:   a a b b b c c c a
    # row_index deliberately differs from list position.
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
        12: "b",  # a misclassified as b
        13: "b",
        14: "b",
        15: "c",  # b misclassified as c
        16: "c",
        17: "c",
        18: "a",  # c misclassified as a
    }
    evaluator = StanceEvaluator(gold_data=gold_data)
    report = evaluator.evaluate(predictions)

    assert report.total_rows == 9
    assert report.evaluated_rows == 9
    assert report.skipped_rows == 0
    assert report.labels == ["a", "b", "c"]
    # 6 correct of 9.
    assert report.accuracy == pytest.approx(6 / 9)

    # Class a: tp=2, fp=1 (row18), fn=1 (row12).
    assert report.per_class["a"]["precision"] == pytest.approx(2 / 3)
    assert report.per_class["a"]["recall"] == pytest.approx(2 / 3)
    assert report.per_class["a"]["f1"] == pytest.approx(2 / 3)
    assert report.per_class["a"]["support"] == 3

    # Class b: tp=2, fp=1 (row12), fn=1 (row15).
    assert report.per_class["b"]["precision"] == pytest.approx(2 / 3)
    assert report.per_class["b"]["recall"] == pytest.approx(2 / 3)
    assert report.per_class["b"]["f1"] == pytest.approx(2 / 3)
    assert report.per_class["b"]["support"] == 3

    # Class c: tp=2, fp=1 (row15), fn=1 (row18).
    assert report.per_class["c"]["precision"] == pytest.approx(2 / 3)
    assert report.per_class["c"]["recall"] == pytest.approx(2 / 3)
    assert report.per_class["c"]["f1"] == pytest.approx(2 / 3)
    assert report.per_class["c"]["support"] == 3

    assert report.confusion_matrix == {
        "a": {"a": 2, "b": 1, "c": 0},
        "b": {"a": 0, "b": 2, "c": 1},
        "c": {"a": 1, "b": 0, "c": 2},
    }


def test_missing_and_none_predictions_excluded_but_reported():
    """None values and absent keys are skipped; counts stay consistent."""
    gold_data = [
        _gold_row(0, "a"),
        _gold_row(1, "a"),
        _gold_row(2, "b"),
        _gold_row(3, "b"),
        _gold_row(4, "b"),
    ]
    predictions = {
        0: "a",       # evaluated, correct
        1: None,      # skipped (explicit None)
        2: "b",       # evaluated, correct
        # row 3 absent entirely -> skipped
        4: "a",       # evaluated, wrong
    }
    evaluator = StanceEvaluator(gold_data=gold_data)
    report = evaluator.evaluate(predictions)

    assert report.total_rows == 5
    assert report.evaluated_rows == 3
    assert report.skipped_rows == 2
    assert report.evaluated_rows + report.skipped_rows == report.total_rows

    # Only rows 0, 2, 4 count. 2 of 3 correct.
    assert report.accuracy == pytest.approx(2 / 3)
    # forced_choice_accuracy uses the FULL denominator (5 total_rows): the 2
    # skipped rows (1 explicit None, 1 absent) count as incorrect, so it's
    # 2 correct out of 5, not 2 out of 3.
    assert report.forced_choice_accuracy == pytest.approx(2 / 5)
    # Class a: gold rows among evaluated = {row0}; support 1.
    assert report.per_class["a"]["support"] == 1
    # Class b: gold rows among evaluated = {row2, row4}; support 2.
    assert report.per_class["b"]["support"] == 2
    # row4 (gold b) predicted a -> b recall = 1/2.
    assert report.per_class["b"]["recall"] == pytest.approx(1 / 2)


def test_zero_division_label_never_predicted():
    """A label predicted by nobody yields precision 0.0 without raising."""
    gold_data = [
        _gold_row(0, "a"),
        _gold_row(1, "a"),
        _gold_row(2, "b"),
    ]
    # Nobody ever predicts "b": its precision denominator (tp+fp) is 0.
    predictions = {0: "a", 1: "a", 2: "a"}
    evaluator = StanceEvaluator(gold_data=gold_data)
    report = evaluator.evaluate(predictions)

    # b: tp=0, fp=0 -> precision 0.0; tp=0, fn=1 -> recall 0.0; f1 0.0.
    assert report.per_class["b"]["precision"] == 0.0
    assert report.per_class["b"]["recall"] == 0.0
    assert report.per_class["b"]["f1"] == 0.0
    # a: tp=2, fp=1 -> precision 2/3, recall 2/2=1.0.
    assert report.per_class["a"]["precision"] == pytest.approx(2 / 3)
    assert report.per_class["a"]["recall"] == pytest.approx(1.0)


def test_evaluate_empty_predictions_all_skipped():
    """evaluate({}) returns a safe zeroed report without raising."""
    gold_data = [_gold_row(0, "a"), _gold_row(1, "b")]
    evaluator = StanceEvaluator(gold_data=gold_data)
    report = evaluator.evaluate({})

    assert report.accuracy == 0.0
    assert report.forced_choice_accuracy == 0.0
    assert report.evaluated_rows == 0
    assert report.skipped_rows == report.total_rows == 2
    assert report.per_class == {}
    assert report.confusion_matrix == {}
    assert report.labels == []


def test_malformed_gold_rows_excluded():
    """Rows missing row_index or gold_stance are excluded from total_rows."""
    gold_data = [
        _gold_row(0, "a"),
        {"row_index": 1},            # missing gold_stance
        {"gold_stance": "b"},        # missing row_index
        _gold_row(2, "b"),
    ]
    evaluator = StanceEvaluator(gold_data=gold_data)
    report = evaluator.evaluate({0: "a", 2: "b"})
    assert report.total_rows == 2
    assert report.evaluated_rows == 2


def test_stray_predicted_label_visible_in_confusion_matrix():
    """A predicted label not in the gold set appears as an extra column."""
    gold_data = [_gold_row(0, "a"), _gold_row(1, "a")]
    predictions = {0: "a", 1: "zzz"}
    evaluator = StanceEvaluator(gold_data=gold_data)
    report = evaluator.evaluate(predictions)

    assert report.labels == ["a"]  # per-class only over gold labels
    assert "zzz" in report.confusion_matrix["a"]
    assert report.confusion_matrix["a"]["zzz"] == 1


def test_format_report_smoke():
    """format_report produces a non-empty string with expected sections."""
    gold_data = [_gold_row(0, "a"), _gold_row(1, "b")]
    evaluator = StanceEvaluator(gold_data=gold_data)
    text = evaluator.format_report(evaluator.evaluate({0: "a", 1: "b"}))
    assert "Stance Evaluation Report" in text
    assert "Per-class metrics" in text
    assert "Confusion matrix" in text
    assert "accuracy=1.0000" in text
    assert "forced_choice_accuracy=1.0000" in text


def test_forced_choice_accuracy_equals_accuracy_when_nothing_skipped():
    gold_data = [_gold_row(0, "a"), _gold_row(1, "b")]
    evaluator = StanceEvaluator(gold_data=gold_data)
    report = evaluator.evaluate({0: "a", 1: "a"})  # row 1 wrong, none skipped

    assert report.evaluated_rows == report.total_rows
    assert report.forced_choice_accuracy == pytest.approx(report.accuracy)
    assert report.forced_choice_accuracy == pytest.approx(0.5)


def test_write_result_json(tmp_path):
    """write_result_json creates a JSON file with predicted_stance per row."""
    gold_data = [
        {"row_index": 0, "gold_stance": "in favor", "comment_text": "Great!"},
        {"row_index": 1, "gold_stance": "neutral", "comment_text": "Okay."},
    ]
    predictions = {0: "in favor", 1: "ablehnung"}
    path = write_result_json(
        gold_data,
        predictions,
        output_dir=str(tmp_path),
        timestamp="20260101T000000Z",
    )
    assert path.exists()
    data = json.loads(path.read_text())
    assert len(data) == 2
    assert data[0]["predicted_stance"] == "in favor"
    assert data[1]["predicted_stance"] == "ablehnung"
    # Original gold fields are preserved.
    assert data[0]["gold_stance"] == "in favor"
    assert data[1]["comment_text"] == "Okay."
    # Backward compatibility: with no stance_model_name kwarg no such key is added.
    assert "stance_model_name" not in data[0]
    assert "stance_model_name" not in data[1]


def test_write_result_json_includes_stance_model_name(tmp_path):
    """When stance_model_name is passed, every row records it for provenance."""
    gold_data = [
        {"row_index": 0, "gold_stance": "in favor", "comment_text": "Great!"},
        {"row_index": 1, "gold_stance": "neutral", "comment_text": "Okay."},
    ]
    predictions = {0: "in favor", 1: "neutral"}
    path = write_result_json(
        gold_data,
        predictions,
        output_dir=str(tmp_path),
        timestamp="20260101T000000Z",
        stance_model_name="Qwen/Qwen3-32B",
    )
    data = json.loads(path.read_text())
    assert len(data) == 2
    for entry in data:
        assert entry["stance_model_name"] == "Qwen/Qwen3-32B"


def _load_real_gold() -> list[dict]:
    with open(GOLD_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def test_real_data_perfect_prediction():
    """Predicting gold_stance for every row yields a perfect report."""
    gold_data = _load_real_gold()
    predictions = {row["row_index"]: row["gold_stance"] for row in gold_data}
    evaluator = StanceEvaluator(gold_data=gold_data)
    report = evaluator.evaluate(predictions)

    assert report.accuracy == 1.0
    assert report.evaluated_rows == 155
    assert report.skipped_rows == 0
    for label in report.labels:
        assert report.per_class[label]["precision"] == 1.0
        assert report.per_class[label]["recall"] == 1.0
        assert report.per_class[label]["f1"] == 1.0


def test_real_data_baseline_regression():
    """Baseline pipeline stance reproduces the Researcher's precomputed numbers.

    The hardcoded expected metrics below were computed independently against the
    real gold file (not derived from this evaluator), so this guards against
    silent regressions in the scoring logic rather than tautologically checking
    the code against itself.
    """
    gold_data = _load_real_gold()
    predictions = {
        row["row_index"]: row["pipeline_baseline_stance"] for row in gold_data
    }
    evaluator = StanceEvaluator(gold_data=gold_data)
    report = evaluator.evaluate(predictions)

    assert report.accuracy == pytest.approx(112 / 155)

    ablehnung = report.per_class["ablehnung"]
    assert ablehnung["precision"] == pytest.approx(0.11538461538461539)
    assert ablehnung["recall"] == pytest.approx(1.0)
    assert ablehnung["f1"] == pytest.approx(0.20689655172413793)
    assert ablehnung["support"] == 3

    in_favor = report.per_class["in favor"]
    assert in_favor["precision"] == pytest.approx(0.7741935483870968)
    assert in_favor["recall"] == pytest.approx(0.5333333333333333)
    assert in_favor["f1"] == pytest.approx(0.631578947368421)
    assert in_favor["support"] == 45

    neutral = report.per_class["neutral"]
    assert neutral["precision"] == pytest.approx(0.8673469387755102)
    assert neutral["recall"] == pytest.approx(0.794392523364486)
    assert neutral["f1"] == pytest.approx(0.8292682926829268)
    assert neutral["support"] == 107

    assert report.confusion_matrix["ablehnung"] == {
        "ablehnung": 3,
        "in favor": 0,
        "neutral": 0,
    }
    assert report.confusion_matrix["in favor"] == {
        "ablehnung": 8,
        "in favor": 24,
        "neutral": 13,
    }
    assert report.confusion_matrix["neutral"] == {
        "ablehnung": 15,
        "in favor": 7,
        "neutral": 85,
    }
