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

import os
import tempfile

import numpy as np


def _export_trials(out, cases):
    """Export one typed, atomically replaceable Parquet part per frozen case."""
    import polars as pl

    dest = out / "trials"
    dest.mkdir(exist_ok=True)
    identity = ("algorithm", "object", "family", "gripper", "placement")
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
        object_holdout=pl.Boolean,
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
            row = {k: case[k] for k in (*identity, "seed", "object_holdout")}
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


def _write_report(out, report):
    lines = [
        "# Physical grasp benchmark",
        "",
        "## Time & Memory",
        "",
        "| Worker case | cost_time_ms | cpu_delta_mb | gpu_delta_mb | peak_gpu_mb |",
        "|---|---:|---:|---:|---:|",
        *[
            f"| {case['trial_id_prefix'][:12]} | "
            + " | ".join(f"{v:.2f}" for v in case["performance"].values())
            + " |"
            for case in report["case_inventory"]
            if "performance" in case
        ],
        "",
        f"Run wall time {report['wall_s']:.2f} s. Recorded trials {report['attempts']}. Memory uses worker RSS and the Torch CUDA allocator.",
        "",
        "## Success & Other Metrics",
        "",
        "| Held-out hand | Split | Algorithm | Success@1 | Hit@5 | Hit@10 | Precision@10 | Robust@1 |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    leaderboard = {}
    for hand, fold in report["folds"].items():
        for label in ("seen", "new_object", "new_gripper", "both_new"):
            if fold[label] is None:
                continue
            for method, stats in fold[label]["algorithms"].items():
                metrics = stats["metrics"]
                values = " | ".join(f"{x:.1%}" for x in metrics.values())
                lines.append(f"| {hand} | {label} | {method} | {values} |")
                if label == "both_new":
                    leaderboard.setdefault(method, []).append(
                        [metrics["Success@1"], metrics["Robust@1"]]
                    )
    lines += [
        "",
        "## Leaderboard",
        "",
        "| Algorithm | Double-held-out Success@1 | Robust@1 |",
        "|---|---:|---:|",
    ]
    means = {m: np.mean(v, axis=0) for m, v in leaderboard.items()}
    for method, scores in sorted(means.items(), key=lambda item: -item[1][0]):
        lines.append(f"| {method} | {scores[0]:.1%} | {scores[1]:.1%} |")
    lines += [
        "",
        f"Missing cases: {report['missing_cases']}. Intervals and stage diagnostics are in report.json.",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n")
