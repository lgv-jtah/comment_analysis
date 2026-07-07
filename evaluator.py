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
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


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


@dataclass
class RunMetadata:
    """Metadata describing a single stance-detection run.

    Attributes:
        model_name: Name of the model used for LLM calls.
        base_url: Base URL of the model server.
        prompt_key: Key identifying the prompt configuration used.
        fast_path_count: Number of rows routed through the fast path.
        ensemble_count: Number of rows routed through the ensemble path.
        llm_count: Number of rows routed through a plain LLM call.
        total_elapsed: Total wall-clock time for the run, in seconds.
        notes: Optional free-form notes about the run.
    """

    model_name: str
    base_url: str
    prompt_key: str
    fast_path_count: int
    ensemble_count: int
    llm_count: int
    total_elapsed: float
    notes: str | None = None


def format_markdown_report(
    report: EvaluationReport, metadata: RunMetadata, timestamp: datetime
) -> str:
    """Render an evaluation report and run metadata as a markdown document.

    Emits up to six sections: the header, the run-metadata table, the overall
    metrics, the per-class metrics table, the confusion matrix, and an optional
    notes section (rendered only when ``metadata.notes`` is non-empty). When no
    rows were evaluated the confusion matrix falls back to ``_No evaluated rows._``.

    Args:
        report: The evaluation results to render.
        metadata: Run metadata describing model, routing counts, and timing.
        timestamp: The generation time stamped into the header.

    Returns:
        A markdown string ending with a trailing newline.
    """
    stamp = timestamp.strftime("%Y-%m-%dT%H%M%SZ")
    total_routed = (
        metadata.fast_path_count + metadata.ensemble_count + metadata.llm_count
    )

    lines: list[str] = []
    lines.append("# Stance Evaluation Report")
    lines.append("")
    lines.append(f"Generated: {stamp}")

    # Run metadata table.
    lines.append("")
    lines.append("## Run metadata")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("| --- | --- |")
    lines.append(f"| Model | {metadata.model_name} |")
    lines.append(f"| Base URL | {metadata.base_url} |")
    lines.append(f"| Prompt key | {metadata.prompt_key} |")
    lines.append(f"| Fast-path count | {metadata.fast_path_count} |")
    lines.append(f"| Ensemble count | {metadata.ensemble_count} |")
    lines.append(f"| LLM calls | {metadata.llm_count} |")
    lines.append(f"| Total routed | {total_routed} |")
    lines.append(f"| Total elapsed | {metadata.total_elapsed:.2f}s |")

    # Overall metrics.
    lines.append("")
    lines.append("## Overall")
    lines.append("")
    lines.append(f"- Accuracy: {report.accuracy:.4f}")
    lines.append(f"- Evaluated rows: {report.evaluated_rows}")
    lines.append(f"- Skipped rows: {report.skipped_rows}")
    lines.append(f"- Total rows: {report.total_rows}")

    # Per-class metrics table.
    lines.append("")
    lines.append("## Per-class metrics")
    lines.append("")
    lines.append("| label | precision | recall | f1 | support |")
    lines.append("| --- | --- | --- | --- | --- |")
    for label in report.labels:
        metrics = report.per_class[label]
        lines.append(
            f"| {label} "
            f"| {metrics['precision']:.4f} "
            f"| {metrics['recall']:.4f} "
            f"| {metrics['f1']:.4f} "
            f"| {int(metrics['support'])} |"
        )

    # Confusion matrix table.
    lines.append("")
    lines.append("## Confusion matrix (rows = gold, cols = predicted)")
    lines.append("")
    columns = sorted(
        {col for row in report.confusion_matrix.values() for col in row}
    )
    if not report.labels:
        lines.append("_No evaluated rows._")
    else:
        header = "| gold \\ pred | " + " | ".join(columns) + " |"
        lines.append(header)
        lines.append("| --- |" + " --- |" * len(columns))
        for gold in report.labels:
            row = report.confusion_matrix.get(gold, {})
            cells = " | ".join(str(row.get(col, 0)) for col in columns)
            lines.append(f"| {gold} | {cells} |")

    # Optional notes section.
    if isinstance(metadata.notes, str) and metadata.notes:
        lines.append("")
        lines.append("## Notes")
        lines.append("")
        lines.append(metadata.notes)

    return "\n".join(lines) + "\n"


def write_markdown_report(
    report: EvaluationReport,
    metadata: RunMetadata,
    output_dir: str = "reports",
    timestamp: datetime | None = None,
) -> Path:
    """Write a markdown evaluation report to disk without overwriting.

    Args:
        report: The evaluation results to render.
        metadata: Run metadata; its ``prompt_key`` drives the filename slug.
        output_dir: Directory to write into; created if absent.
        timestamp: Generation time; defaults to the current UTC time.

    Returns:
        The :class:`Path` actually written.
    """
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)

    slug = re.sub(r"[^a-z0-9]+", "_", metadata.prompt_key.lower()).strip("_") or "run"
    stamp = timestamp.strftime("%Y-%m-%dT%H%M%SZ")

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    path = directory / f"{stamp}_{slug}.md"
    # No-overwrite: append -1, -2, ... before the suffix until a free path.
    counter = 1
    while path.exists():
        path = directory / f"{stamp}_{slug}-{counter}.md"
        counter += 1

    content = format_markdown_report(report, metadata, timestamp)
    path.write_text(content, encoding="utf-8")
    return path


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


def write_result_json(
    gold_data: list[dict],
    predictions: dict,
    output_dir: str = "results",
    timestamp: str | None = None,
    stance_model_name: str | None = None,
) -> Path:
    """Write per-row results (gold + predicted stance) to a JSON file.

    Each output entry is the original gold row with an added
    ``predicted_stance`` key. Rows whose ``row_index`` has no entry in
    *predictions* receive ``None`` for that key.

    Args:
        gold_data: The gold-label rows (each must contain ``row_index``).
        predictions: Mapping of ``row_index`` -> predicted stance string (or
            ``None`` for rows that were skipped / failed).
        output_dir: Directory to write into; created if absent.
        timestamp: UTC timestamp string used in the filename
            (``"%Y%m%dT%H%M%SZ"`` format). Defaults to the current UTC time.
        stance_model_name: When provided, the full (unsanitized) stance model
            name is recorded on every output row under ``stance_model_name`` for
            provenance. When ``None`` (the default) no such key is added, so the
            output is byte-for-byte identical to the pre-provenance behaviour.

    Returns:
        The :class:`Path` actually written.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if timestamp is None:
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    base = out_dir / f"{timestamp}_results"
    path = base.with_suffix(".json")
    counter = 1
    while path.exists():
        path = base.with_name(f"{base.name}_{counter}").with_suffix(".json")
        counter += 1
    rows = []
    for row in gold_data:
        entry = dict(row)
        entry["predicted_stance"] = predictions.get(row["row_index"])
        if stance_model_name is not None:
            entry["stance_model_name"] = stance_model_name
        rows.append(entry)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    return path
