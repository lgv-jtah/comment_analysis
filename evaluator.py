"""Stance-prediction evaluator.

Scores stance predictions against the corrected gold labels. Predictions are
matched to gold rows by ``row_index`` (NOT list position), because in the real
dataset many rows have a list position that differs from their ``row_index``.

Zero-division convention: following scikit-learn's ``zero_division=0`` behaviour,
precision/recall/f1 return ``0.0`` (rather than raising) whenever the relevant
denominator is 0. ``accuracy`` is likewise ``0.0`` when no rows were evaluated.
"""
from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass
class EvaluationReport:
    """Container for the result of an evaluation run.

    Attributes:
        accuracy: Fraction of evaluated rows whose prediction matched gold.
        per_class: Gold label -> {"precision", "recall", "f1", "support"}.
        confusion_matrix: gold_label -> {predicted_label -> count}.
        total_rows: Count of valid gold rows.
        evaluated_rows: Rows with a non-None prediction present.
        skipped_rows: total_rows - evaluated_rows.
        labels: Sorted list of gold labels observed among evaluated rows.
    """

    accuracy: float
    per_class: dict[str, dict[str, float | int]]
    confusion_matrix: dict[str, dict[str, int]]
    total_rows: int
    evaluated_rows: int
    skipped_rows: int
    labels: list[str]


class StanceEvaluator:
    """Evaluate stance predictions against corrected gold labels."""

    def __init__(
        self,
        gold_data: list[dict] | None = None,
        gold_path: str = "gold_labels_corrected.json",
    ) -> None:
        """Build the evaluator from gold data.

        Args:
            gold_data: Pre-loaded gold rows. When provided it is used directly
                (handy for synthetic test fixtures). When omitted, the gold rows
                are loaded from ``gold_path``.
            gold_path: Path to the gold-labels JSON file. Loaded directly via
                ``json.load`` to avoid the noisy print side effect that
                ``StanceDetector.load_data()`` produces.
        """
        if gold_data is None:
            with open(gold_path, encoding="utf-8") as fh:
                gold_data = json.load(fh)

        # Map row_index -> gold_stance, defensively skipping malformed rows so a
        # missing key excludes the row rather than crashing the whole run.
        self._gold: dict[int, str] = {}
        for row in gold_data:
            row_index = row.get("row_index")
            gold_stance = row.get("gold_stance")
            if row_index is None or gold_stance is None:
                continue
            self._gold[row_index] = gold_stance

    def evaluate(self, predictions: dict[int, str | None]) -> EvaluationReport:
        """Score predictions against the gold labels.

        Args:
            predictions: Mapping of ``row_index`` -> predicted stance. A missing
                key or a ``None`` value marks the row as skipped (excluded from
                all metrics). Predictions whose ``row_index`` is not in the gold
                set are ignored.

        Returns:
            An :class:`EvaluationReport`. See the module docstring for the
            zero-division convention.
        """
        total_rows = len(self._gold)

        # Collect (gold, pred) pairs for rows with a usable prediction.
        pairs: list[tuple[str, str]] = []
        for row_index, gold_label in self._gold.items():
            pred = predictions.get(row_index)
            if pred is None:
                continue
            pairs.append((gold_label, pred))

        evaluated_rows = len(pairs)
        skipped_rows = total_rows - evaluated_rows

        # Gold labels observed among evaluated rows drive per-class metrics.
        gold_labels = sorted({gold for gold, _ in pairs})
        # Predicted-only labels become extra confusion-matrix columns so a stray
        # prediction is visible rather than silently dropped.
        all_predicted = {pred for _, pred in pairs}
        matrix_columns = sorted(set(gold_labels) | all_predicted)

        # Confusion matrix: gold rows only; columns include predicted-only ones.
        confusion_matrix: dict[str, dict[str, int]] = {
            gold: {col: 0 for col in matrix_columns} for gold in gold_labels
        }
        for gold, pred in pairs:
            confusion_matrix[gold][pred] += 1

        per_class: dict[str, dict[str, float | int]] = {}
        total_tp = 0
        for label in gold_labels:
            tp = sum(1 for g, p in pairs if g == label and p == label)
            fp = sum(1 for g, p in pairs if g != label and p == label)
            fn = sum(1 for g, p in pairs if g == label and p != label)
            support = sum(1 for g, _ in pairs if g == label)
            total_tp += tp

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
            per_class[label] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }

        accuracy = total_tp / evaluated_rows if evaluated_rows > 0 else 0.0

        return EvaluationReport(
            accuracy=accuracy,
            per_class=per_class,
            confusion_matrix=confusion_matrix,
            total_rows=total_rows,
            evaluated_rows=evaluated_rows,
            skipped_rows=skipped_rows,
            labels=gold_labels,
        )

    def format_report(self, report: EvaluationReport) -> str:
        """Render an :class:`EvaluationReport` as a plain-text block.

        Args:
            report: The report to format.

        Returns:
            A human-readable string with a header, a per-class table, and a
            confusion-matrix block. Uses no external formatting library.
        """
        lines: list[str] = []
        lines.append("=== Stance Evaluation Report ===")
        lines.append(
            f"accuracy={report.accuracy:.4f}  "
            f"evaluated={report.evaluated_rows}  "
            f"skipped={report.skipped_rows}  "
            f"total={report.total_rows}"
        )

        # Per-class table.
        lines.append("")
        lines.append("Per-class metrics:")
        header = f"{'label':<20}{'precision':>11}{'recall':>11}{'f1':>11}{'support':>9}"
        lines.append(header)
        lines.append("-" * len(header))
        for label in report.labels:
            metrics = report.per_class[label]
            lines.append(
                f"{label:<20}"
                f"{metrics['precision']:>11.4f}"
                f"{metrics['recall']:>11.4f}"
                f"{metrics['f1']:>11.4f}"
                f"{metrics['support']:>9}"
            )

        # Confusion matrix block (rows = gold, columns = predicted).
        lines.append("")
        lines.append("Confusion matrix (rows=gold, cols=predicted):")
        columns: list[str] = sorted(
            {col for row in report.confusion_matrix.values() for col in row}
        )
        col_header = f"{'gold\\pred':<20}" + "".join(f"{col:>14}" for col in columns)
        lines.append(col_header)
        lines.append("-" * len(col_header))
        for gold in report.labels:
            row = report.confusion_matrix.get(gold, {})
            lines.append(
                f"{gold:<20}" + "".join(f"{row.get(col, 0):>14}" for col in columns)
            )

        return "\n".join(lines)
