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

"""Camera report template over simulator-independent analysis primitives."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.benchmark.core.artifacts import ArtifactStore, read_json, write_json
from scripts.benchmark.core.records import validate_run_result
from scripts.benchmark.reporting.aggregation import MetricDefinition, aggregate_runs
from scripts.benchmark.reporting.comparison import comparison_reasons

__all__ = ["rebuild_report"]

METRICS = (
    MetricDefinition("camera_frames_per_s", "1/s", "camera_exposure"),
    MetricDefinition("latency_p50_s", "s", "capture_p50"),
    MetricDefinition("latency_p95_s", "s", "capture_p95"),
    MetricDefinition("cpu_peak_rss_bytes", "byte", "process_lifetime"),
)


def _display(value: float | None, scale: float = 1.0) -> str:
    return "N/A" if value is None else f"{value * scale:.3f}"


def rebuild_report(root: Path) -> Path:
    """Rebuild current and v0.1 camera records without importing simulators."""
    rows = read_json(root / "runs.json")
    if not isinstance(rows, list):
        raise ValueError("runs.json must contain a list")
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Each run must be an object")
        validate_run_result(row)
    summary = aggregate_runs(rows, METRICS)
    reasons = comparison_reasons(
        rows,
        invariant_paths=(
            "config_sha256",
            "hardware_id",
            "metrics.boundary",
            "metrics.completion",
        ),
    )
    summary.update(
        speedup=None,
        comparison_status="not_qualified" if reasons else "eligible",
        comparison_reasons=list(reasons),
        reason="Quality/condition qualification required; this pilot template reports raw results only.",
        platforms={},
    )
    text = [
        "# Camera pilot comparison",
        "",
        "One scene, one RGB camera. Timing includes rendering, output access,",
        "RGB normalization and blocking CPU readback; excludes physics, startup and file writes.",
        "Per-capture latency and full-loop throughput have separate timing windows.",
        "**Pilot only: no equal-quality speedup is calculated.**",
        "Comparison checks: "
        + (", ".join(reasons) if reasons else "declared conditions compatible"),
        "",
        "## Time & memory",
        "",
        "| Backend | Repeat | Status | Camera frames/s | P50 ms | P95 ms | CPU lifetime peak GiB |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        m = row.get("metrics", {})
        text.append(
            f"| {row['backend']} | {row['repeat']} | {row['status']} | "
            f"{_display(m.get('camera_frames_per_s'))} | {_display(m.get('latency_p50_s'), 1000)} | "
            f"{_display(m.get('latency_p95_s'), 1000)} | {_display(m.get('cpu_peak_rss_bytes'), 1 / 1024**3)} |"
        )
    text += [
        "",
        "## Validation & failures",
        "",
        "| Backend | Repeat | Quality | Evidence / reason |",
        "|---|---:|---|---|",
    ]
    for row in rows:
        reason = (
            str(
                row.get("error")
                or json.dumps(row.get("validation", {}), ensure_ascii=False)
            )
            .replace("|", "/")
            .replace("\n", " ")
        )
        directory = row.get("run_dir")
        evidence = (
            f"[result]({directory}/result.json); [image]({directory}/sample.png)"
            if directory and row["status"] == "completed"
            else reason
        )
        if (
            directory
            and row["status"] != "completed"
            and (root / directory / "worker.log").is_file()
        ):
            evidence += f"; [log]({directory}/worker.log)"
        text.append(
            f"| {row['backend']} | {row['repeat']} | {row.get('quality_status', 'not_qualified')} | {evidence} |"
        )
    text += [
        "",
        "## Medians by platform and workload",
        "",
        "| Backend | Workload | Completed / scheduled | Median camera frames/s | Median P50 ms |",
        "|---|---|---:|---:|---:|",
    ]
    for group in summary["groups"]:
        metrics = group["metrics"]
        fps = metrics["camera_frames_per_s"]["median"]
        latency = metrics["latency_p50_s"]["median"]
        fingerprint = group["config_sha256"] or "unknown"
        case_id = group["case_id"]
        identity = (
            f"{case_id[:12]}/{fingerprint[:12]}"
            if case_id and case_id != fingerprint
            else fingerprint[:12]
        )
        text.append(
            f"| {group['backend']} | {identity} | {group['completed']} / {group['scheduled']} | "
            f"{_display(fps)} | {_display(latency, 1000)} |"
        )
    # Preserve the single-workload v0.1 summary keys without mixing workloads.
    for backend in sorted({row["backend"] for row in rows}):
        groups = [g for g in summary["groups"] if g["backend"] == backend]
        completed_groups = [group for group in groups if group["completed"]]
        unique = completed_groups[0] if len(completed_groups) == 1 else None
        summary["platforms"][backend] = {
            "completed": sum(g["completed"] for g in groups),
            "scheduled": sum(g["scheduled"] for g in groups),
            "median_camera_frames_per_s": (
                unique["metrics"]["camera_frames_per_s"]["median"] if unique else None
            ),
            "median_latency_p50_s": (
                unique["metrics"]["latency_p50_s"]["median"] if unique else None
            ),
        }
    text += [
        "",
        "Medians use independent completed runs within the same workload; failures remain in scheduled counts.",
        "Raw seconds/bytes and metric populations are retained in summary.json.",
        "GPU device snapshots include other processes; CPU peak includes initialization.",
        "Inspect sample images and provenance before interpreting performance differences.",
        "",
    ]
    write_json(root / "summary.json", summary)
    ArtifactStore(root).write_metrics(summary)
    output = root / "report.md"
    output.write_text("\n".join(text), encoding="utf-8")
    return output
