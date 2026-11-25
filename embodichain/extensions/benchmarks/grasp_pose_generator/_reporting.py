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

"""Write physical grasp trial artifacts and a compact benchmark report."""

from __future__ import annotations

from collections import defaultdict
import os
from pathlib import Path
from statistics import median
import tempfile

from .metrics import _METRICS


def _export_trials(out, cases):
    """Export one typed, atomically replaceable Parquet part per frozen case."""
    import polars as pl

    dest = out / "trials"
    dest.mkdir(exist_ok=True)
    identity = ("physics", "algorithm", "object", "category", "gripper", "placement")
    schema = {
        k: pl.String
        for k in (
            *identity,
            "trial_id",
            "condition",
            "status",
            "failure_stage",
            "reason",
        )
    }
    schema.update(
        rank=pl.Int64,
        seed=pl.Int64,
        success=pl.Boolean,
        peak_acceleration_m_s2=pl.Float64,
        peak_axis_acceleration_m_s2=pl.Float64,
    )
    for exponent in ("0.5", "1.0", "3.0"):
        schema[f"retained_{exponent}s"] = pl.Boolean
    for phase in ("hold", "disturbance"):
        schema[f"{phase}_entered"] = pl.Boolean
        schema[f"{phase}_censored"] = pl.Boolean
        for key in (
            "drop_s",
            "slip_mm",
            "rotation_deg",
            "bilateral_fraction",
            "duration_s",
        ):
            schema[f"{phase}_{key}"] = pl.Float64
    for case in cases:
        if not case["trials"]:
            continue
        records = []
        for trial in case["trials"]:
            row = {k: case[k] for k in (*identity, "seed")}
            row.update(
                {
                    k: trial.get(k)
                    for k in (
                        "trial_id",
                        "condition",
                        "status",
                        "failure_stage",
                        "reason",
                        "rank",
                        "success",
                        "peak_acceleration_m_s2",
                        "peak_axis_acceleration_m_s2",
                    )
                }
            )
            row.update(
                {
                    f"retained_{t}s": value
                    for t, value in trial.get("retention", {}).items()
                }
            )
            for phase in ("hold", "disturbance"):
                for key, value in (trial.get(phase) or {}).items():
                    if f"{phase}_{key}" in schema:
                        row[f"{phase}_{key}"] = value
            records.append(row)
        path = dest / (case["trial_id_prefix"] + ".parquet")
        with tempfile.NamedTemporaryFile(dir=dest, delete=False) as tmp:
            pl.DataFrame(records, schema=schema).write_parquet(tmp)
        os.replace(tmp.name, path)


def _metric_groups(report):
    for hand, group in report["by_gripper"].items():
        yield hand, "All", group
    for category, group in report["by_category"].items():
        yield "All", category, group
    for hand, categories in report["by_gripper_category"].items():
        for category, group in categories.items():
            yield hand, category, group


def _write_report(out, report):
    protocol = report["protocol"]
    wait_steps = protocol["close_wait_steps"]
    lines = [
        "# Physical grasp benchmark",
        "",
        f"Physics backend {protocol['physics']} with solver {protocol['solver']}.",
        "",
        f"The closing command stayed at its final target for {wait_steps} extra steps "
        f"({wait_steps * protocol['physics_dt_s']:.2f} s) before lift.",
        "",
        "## Time & Memory",
        "",
        "| Gripper | Object category | Algorithm | Timed cases | Median time (ms) | Median CPU Δ (MiB) | Median GPU Δ (MiB) | Peak GPU (MiB) |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    timing = defaultdict(list)
    for case in report["case_inventory"]:
        if "performance" in case:
            timing[case["gripper"], case["category"], case["algorithm"]].append(
                case["performance"]
            )
    for (hand, category, method), samples in timing.items():
        values = [
            median(p[key] for p in samples)
            for key in ("cost_time_ms", "cpu_delta_mb", "gpu_delta_mb")
        ]
        values.append(max(p["peak_gpu_mb"] for p in samples))
        cells = " | ".join(f"{value:.2f}" for value in values)
        lines.append(f"| {hand} | {category} | {method} | {len(samples)} | {cells} |")
    lines.extend(
        [
            "",
            f"Run wall time {report['wall_s']:.2f} s. Recorded trials {report['attempts']}. "
            "Memory uses worker RSS and the Torch CUDA allocator.",
            "",
            "## Success & Other Metrics",
            "",
            "| Gripper | Object category | Algorithm | Objects | Completed cases | Missing cases | "
            + " | ".join(_METRICS)
            + " |",
            "|---|---|---|---:|---:|---:|" + "---:|" * len(_METRICS),
        ]
    )
    for hand, category, group in _metric_groups(report):
        for method, stats in group["algorithms"].items():
            metrics = stats["metrics"]
            cells = " | ".join(
                f"{metrics[key]:.1%}" if metrics else "—" for key in _METRICS
            )
            lines.append(
                f"| {hand} | {category} | {method} | {group['objects']} | {stats['completed_cases']} | {stats['missing_cases']} | {cells} |"
            )
    lines += [
        "",
        "## Leaderboard",
        "",
        "| Algorithm | Overall Success@1 | Overall Robust@1 |",
        "|---|---:|---:|",
    ]
    methods = report["overall"]["algorithms"]
    for method, stats in sorted(
        methods.items(),
        key=lambda item: (item[1]["metrics"] or {}).get("Success@1", -1),
        reverse=True,
    ):
        metrics = stats["metrics"]
        cells = " | ".join(
            f"{metrics[key]:.1%}" if metrics else "—"
            for key in ("Success@1", "Robust@1")
        )
        lines.append(f"| {method} | {cells} |")
    lines += [
        "",
        f"Missing cases {report['missing_cases']}. Each object instance and gripper pair has equal weight. "
        "Stage diagnostics and object-bootstrap intervals are in report.json.",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n")


def _visual_outcome(case: dict) -> tuple[str, str | None]:
    """Describe the observed top-ranked base trial or why it is unavailable."""
    if "error" in case:
        return "error", case["error"]
    trials = case["trials"]
    if not trials and case["count"] == 0 and case.get("finished"):
        return "no_candidate", None
    if len(trials) != 1 or (trials[0]["rank"], trials[0]["condition"]) != (1, "base"):
        return "missing", "Expected one rank-1 base trial"
    trial = trials[0]
    if trial["status"] != "ok":
        return "missing", trial.get("reason", trial["status"])
    if trial["success"]:
        return "success", None
    return "failed", trial["failure_stage"]


def _visual_report(cases: list[dict], protocol: dict) -> dict:
    """Summarize one visual check per gripper and object category."""
    counts = dict.fromkeys(("success", "failed", "no_candidate", "missing", "error"), 0)
    rows = []
    for case in cases:
        status, reason = _visual_outcome(case)
        counts[status] += 1
        rows.append(
            dict(
                gripper=case["gripper"],
                category=case["category"],
                object=case["object"],
                status=status,
                reason=reason,
                attempts=len(case["trials"]),
            )
        )
    return dict(
        protocol=protocol,
        complete=counts["missing"] == counts["error"] == 0,
        attempts=sum(row["attempts"] for row in rows),
        counts=counts,
        cases=rows,
    )


def _write_visual_report(out: Path, report: dict) -> None:
    """Write the per-pair outcomes of a single-grasp inspection."""
    counts = report["counts"]
    lines = [
        "# Visual grasp check",
        "",
        f"{len(report['cases'])} gripper/category pairs, {report['attempts']} recorded "
        f"attempts on {report['protocol']['physics']}.",
        "",
        f"{counts['success']} successful, {counts['failed']} failed, "
        f"{counts['no_candidate']} without candidates, {counts['missing']} missing "
        f"observations, {counts['error']} errors.",
        "",
        "| Gripper | Object category | Object | Outcome | Stage or reason |",
        "|---|---|---|---|---|",
    ]
    for row in report["cases"]:
        cells = [
            row["gripper"],
            row["category"],
            row["object"],
            row["status"].replace("_", " "),
            row["reason"] or "—",
        ]
        cells = [cell.replace("|", "\\|").replace("\n", " ") for cell in cells]
        lines.append("| " + " | ".join(cells) + " |")
    (out / "report.md").write_text("\n".join(lines) + "\n")
