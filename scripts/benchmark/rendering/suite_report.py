# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

"""Offline report reconstruction for all pure-rendering R-series experiments."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.benchmark.core.artifacts import ArtifactStore, read_json, write_json
from scripts.benchmark.core.records import validate_run_result
from scripts.benchmark.reporting.aggregation import aggregate_runs
from scripts.benchmark.reporting.comparison import compare_metric
from scripts.benchmark.reporting.figures import write_metric_csv, write_svg_bar_chart
from scripts.benchmark.reporting.tables import render_markdown_table

from .suite import metric_definitions

__all__ = ["rebuild_report"]


def _display(value: object, scale: float = 1.0) -> str:
    """Format a finite report cell in human units."""
    if value is None:
        return "N/A"
    if isinstance(value, (int, float)):
        return f"{float(value) * scale:.3f}"
    return str(value).replace("|", "/").replace("\n", " ")


def _comparison_payload(result: object) -> dict[str, object]:
    """Serialize a comparison result without depending on dataclasses."""
    return {
        "eligible": result.eligible,
        "reasons": list(result.reasons),
        "baseline": result.baseline,
        "candidate": result.candidate,
        "metric": result.metric,
        "pairs": result.pairs,
        "ratio": result.ratio,
        "delta": result.delta,
    }


def rebuild_report(root: Path) -> Path:
    """Rebuild summary, CSV, SVG and Markdown from stored run records only."""
    root = Path(root)
    rows = read_json(root / "runs.json")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("runs.json must contain a list of objects")
    for row in rows:
        validate_run_result(row)
    definition = (
        read_json(root / "definition.json")
        if (root / "definition.json").exists()
        else {}
    )
    experiment_id = str(definition.get("experiment_id", "rendering-suite"))
    summary = aggregate_runs(rows, metric_definitions())
    comparisons: dict[str, dict[str, object]] = {}
    for case_id in sorted(
        {row.get("case_id") for row in rows if row.get("case_id") is not None}
    ):
        case_rows = [row for row in rows if row.get("case_id") == case_id]
        result = compare_metric(
            case_rows,
            metric="observations_per_s",
            baseline="embodichain",
            candidate="isaaclab",
            invariant_paths=(
                "config_sha256",
                "hardware_id",
                "metrics.boundary",
                "metrics.completion",
            ),
        )
        comparisons[str(case_id)] = _comparison_payload(result)
    comparison_reasons = sorted(
        {
            reason
            for comparison in comparisons.values()
            for reason in comparison["reasons"]
        }
    )
    summary.update(
        experiment_id=experiment_id,
        comparison_status=(
            "eligible" if comparisons and not comparison_reasons else "not_qualified"
        ),
        comparison_reasons=comparison_reasons,
        comparisons=comparisons,
    )
    write_json(root / "summary.json", summary)
    ArtifactStore(root).write_metrics(summary)

    metric_rows = []
    for row in rows:
        metrics = row.get("metrics", {})
        metric_rows.append(
            {
                "experiment": experiment_id,
                "case": row.get("case_id"),
                "backend": row.get("backend"),
                "repeat": row.get("repeat"),
                "status": row.get("status"),
                "observations_per_s": metrics.get("observations_per_s"),
                "camera_frames_per_s": metrics.get("camera_frames_per_s"),
                "p50_ms": (
                    metrics.get("latency_p50_s") * 1000
                    if isinstance(metrics.get("latency_p50_s"), (int, float))
                    else None
                ),
                "p95_ms": (
                    metrics.get("latency_p95_s") * 1000
                    if isinstance(metrics.get("latency_p95_s"), (int, float))
                    else None
                ),
                "readback_bytes_per_s": metrics.get("host_readback_bytes_per_s"),
            }
        )
    columns = (
        "experiment",
        "case",
        "backend",
        "repeat",
        "status",
        "observations_per_s",
        "camera_frames_per_s",
        "p50_ms",
        "p95_ms",
        "readback_bytes_per_s",
    )
    write_metric_csv(root / "metrics.csv", metric_rows, columns=columns)

    group_rows = []
    plot_values = {}
    for group in summary["groups"]:
        metrics = group["metrics"]
        value = metrics["observations_per_s"]["median"]
        group_rows.append(
            {
                "Backend": group["backend"],
                "Case": group["case_id"],
                "Completed / scheduled": f"{group['completed']} / {group['scheduled']}",
                "Observation 1/s": _display(value),
                "P50 ms": _display(metrics["latency_p50_s"]["median"], 1000),
            }
        )
        if value is not None:
            plot_values[f"{group['backend']}:{group['case_id']}"] = value
    if plot_values:
        write_svg_bar_chart(
            root / "observations_per_s.svg",
            title=f"{experiment_id} observation throughput",
            values=plot_values,
        )

    validation_rows = []
    for row in rows:
        validation = row.get("validation", {})
        reason = row.get("error") or json.dumps(validation, ensure_ascii=False)
        validation_rows.append(
            {
                "Backend": row.get("backend"),
                "Case": row.get("case_id"),
                "Repeat": row.get("repeat"),
                "Status": row.get("status"),
                "Quality": row.get("quality_status", "missing"),
                "Evidence / reason": reason,
            }
        )
    lines = [
        f"# {experiment_id} pure-rendering benchmark",
        "",
        "The report is rebuilt from stored run records and does not initialize a simulator.",
        "Execution status, quality qualification and observation delivery evidence are separate.",
        "",
        "## Measured metrics",
        "",
        render_markdown_table(metric_rows, columns=columns),
        "",
        "## Validation and failures",
        "",
        render_markdown_table(
            validation_rows,
            columns=(
                "Backend",
                "Case",
                "Repeat",
                "Status",
                "Quality",
                "Evidence / reason",
            ),
        ),
        "",
        "## Aggregated groups",
        "",
        render_markdown_table(
            group_rows,
            columns=(
                "Backend",
                "Case",
                "Completed / scheduled",
                "Observation 1/s",
                "P50 ms",
            ),
        ),
        "",
        "## Comparison eligibility",
        "",
        f"Status: **{summary['comparison_status']}**",
        f"Reasons: {', '.join(comparison_reasons) if comparison_reasons else 'none'}",
        "",
        "Equal-quality speedup is emitted only when the domain quality protocol and all declared invariants pass.",
        "R-03/R-04/R-05/R-06/R-09 pilot runs remain `not_qualified` until image quality and treatment equivalence are calibrated.",
        "",
    ]
    output = root / "report.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    return output
