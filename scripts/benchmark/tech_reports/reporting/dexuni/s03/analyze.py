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
"""Offline S03 material-corner measurements from explicit run directories."""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np

__all__ = ["load_run", "task_metrics"]
PHASES = (
    "settle",
    "close",
    "lift",
    "hold",
    "transport",
    "lower",
    "release",
    "retreat",
    "observe",
    "done",
)


def load_run(path: Path) -> dict:
    """Read one run without discovery or replacement of earlier attempts.

    Args:
        path: Exact run directory.

    Returns:
        Manifest, config, timing and copied state arrays.
    """
    timing = np.genfromtxt(
        path / "timing.csv", delimiter=",", names=True, dtype=None, encoding=None
    )
    return {
        "path": path,
        "manifest": json.loads((path / "manifest.json").read_text()),
        "config": json.loads((path / "config.json").read_text()),
        "step_seconds": np.atleast_1d(timing["step_seconds"]),
        "states": dict(np.load(path / "states.npz")),
    }


def task_metrics(run: dict) -> dict:
    """Measure physical corner hold and post-release return, retaining failure.

    Args:
        run: Loaded run with physical corner node identities.

    Returns:
        Per-stage and combined acceptance plus task and timing measurements.
    """
    m, c, d = run["manifest"], run["config"], run["states"]
    t = d["time"]
    phase = d["phase"]
    p = d["particles"]
    ids = d["corner_ids"]
    steps = run["step_seconds"]
    corner = p[:, ids].mean(axis=1)
    ref = (phase == 0) & (t >= c["measurement"]["reference_window"][0])
    h0 = float(corner[ref, 2].mean())
    lift = corner[:, 2] - h0
    hold = phase == 3
    transport = phase == 4
    events = m["controller"]["events"]
    acc = c["acceptance"]
    row = {
        "run_id": m["run_id"],
        "platform": m["platform"],
        "repeat": m["repeat"],
        "state": m["state"]["id"],
        "variant": m["variant"],
        "kind": m["kind"],
        "frames": len(steps),
        "init_s": m.get("init_seconds"),
        "gpu_peak_mib": m.get("gpu_memory", {}).get("gpu_process_peak_mib"),
        "step_ms_mean": float(1000 * steps.mean()),
        "corner_nodes": len(ids),
        "corner_reference_height_mm": 1000 * h0,
        "finite": bool(np.isfinite(p).all()),
        "controller_finished": bool(m["controller"]["finished"]),
    }
    if not hold.any() or not transport.any():
        row.update(success=False, failed_checks="missing_hold_or_transport")
        return row
    holdids = np.flatnonzero(hold)
    tail = hold & (t >= t[holdids[-1]] - 1)
    tailids = np.flatnonzero(tail)
    table = c["scene"]["table"]
    inside_table = (
        np.abs(p[:, :, :2] - np.asarray(table["center"][:2]))
        <= np.asarray(table["size"][:2]) / 2
    ).all(axis=-1)
    lowest_on_table = (
        float(p[:, :, 2][inside_table].min())
        if inside_table.any()
        else c["scene"]["table_z"]
    )
    row.update(
        corner_hold_min_mm=float(1000 * lift[hold].min()),
        corner_hold_mean_mm=float(1000 * lift[hold].mean()),
        corner_hold_drop_mm=float(
            1000 * (corner[tailids[0], 2] - corner[tailids[-1], 2])
        ),
        hold_duration_s=float(events["transport"] - events["hold"]),
        transport_corner_min_mm=float(1000 * lift[transport].min()),
        release_corner_height_mm=float(1000 * (corner[-1, 2] - c["scene"]["table_z"])),
        table_penetration_mm=float(
            1000 * max(0.0, c["scene"]["table_z"] - lowest_on_table)
        ),
        global_below_table_plane_mm=float(
            1000 * max(0.0, c["scene"]["table_z"] - p[:, :, 2].min())
        ),
        table_footprint_node_samples=int(inside_table.sum()),
        grip_drift_mm=float(
            1000
            * np.linalg.norm(
                (corner[holdids[-1]] - d["tcp"][holdids[-1]])
                - (corner[holdids[0]] - d["tcp"][holdids[0]])
            )
        ),
        peak_corner_lift_mm=float(1000 * lift.max()),
    )
    row["final_sheet_max_height_mm"] = float(
        1000 * (p[-1, :, 2].max() - c["scene"]["table_z"])
    )
    if "qpos" in d and "joint_names" in d:
        finger_ids = [
            i for i, name in enumerate(d["joint_names"]) if "finger_joint" in str(name)
        ]
        if finger_ids:
            row["final_finger_min_mm"] = float(1000 * d["qpos"][-1, finger_ids].min())
    row["pass_lift"] = row["corner_hold_min_mm"] >= 1000 * acc["corner_lift_min"]
    row["pass_hold"] = (
        row["hold_duration_s"] >= c["trajectory"]["hold_duration"] - 1e-6
        and row["corner_hold_drop_mm"] <= 1000 * acc["hold_drop_max"]
    )
    row["pass_transport"] = (
        row["transport_corner_min_mm"] >= 1000 * acc["corner_lift_min"]
    )
    row["pass_release"] = (
        row["release_corner_height_mm"] <= 1000 * acc["release_corner_height_max"]
    )
    row["pass_table"] = (
        row["table_penetration_mm"] <= 1000 * acc["table_penetration_tolerance"]
    )
    # Check gross loss of the grasp while the TCP is still above the table.
    # Near touchdown the material corner can fold against the worktop.
    descent = (phase == 5) & (d["tcp"][:, 2] > c["scene"]["table_z"] + 0.03)
    relative = corner - d["tcp"]
    row["descent_grasp_drift_mm"] = (
        float(
            1000
            * np.linalg.norm(relative[descent] - relative[holdids[-1]], axis=1).max()
        )
        if descent.any()
        else 0.0
    )
    row["pass_descent"] = row["descent_grasp_drift_mm"] <= 1000 * acc.get(
        "descent_grasp_drift_max", float("inf")
    )
    checks = (
        "finite",
        "controller_finished",
        "pass_lift",
        "pass_hold",
        "pass_transport",
        "pass_descent",
        "pass_release",
        "pass_table",
    )
    row["success"] = all(row[k] for k in checks)
    row["failed_checks"] = ";".join(k for k in checks if not row[k])
    return row
