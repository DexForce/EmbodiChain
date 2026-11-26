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

"""Recompute the UR5 outcome from measured simulator samples."""

from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path
from typing import Any

import numpy as np

__all__: list[str] = []

_MIN_LIFT_M = 0.03
_CENTER_TOLERANCE_M = 0.01
_STABILITY_WINDOW_S = 0.5


def verify(trajectory: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Score grasp, lift, placement and stability independently of the runner."""
    with trajectory.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    if not rows:
        raise ValueError("Missing native pose observations")

    steps = np.asarray([row["physics_step"] for row in rows])
    times = np.asarray([row["simulation_time_s"] for row in rows])
    dt = manifest["physics_dt_s"]
    if not np.array_equal(steps, np.arange(len(rows))) or not np.allclose(
        times, steps * dt, rtol=0.0, atol=1e-8
    ):
        raise ValueError("Native pose observations must cover every physics step")

    def poses(field: str) -> np.ndarray:
        value = np.asarray([row[field] for row in rows], dtype=np.float64)
        if value.shape != (len(rows), 4, 4) or not np.isfinite(value).all():
            raise ValueError(f"Invalid pose evidence: {field}")
        return value

    cube = poses("cube_pose")
    platform = poses("platform_pose")
    tool = poses("tcp_pose")
    phase = np.asarray([row["phase"] for row in rows])
    grip = manifest["gripper"]
    open_m, close_m = grip["open_position_m"], grip["grasp_position_m"]
    if not np.isfinite([open_m, close_m]).all() or open_m == close_m:
        raise ValueError("Missing gripper calibration")
    qpos = np.asarray([row["qpos"] for row in rows])
    closed_fraction = (qpos[:, grip["joint_index"]] - open_m) / (close_m - open_m)
    if not np.isfinite(qpos).all():
        raise ValueError("Non-finite measured joints")
    lift = cube[:, 2, 3] - cube[0, 2, 3]
    in_pick = phase == "pick"
    lift_peak = float(np.max(lift[in_pick])) if in_pick.any() else 0.0
    lifted_rows = np.flatnonzero(
        in_pick & (lift >= _MIN_LIFT_M) & (closed_fraction >= 0.3)
    )[-12:]
    carried = False
    if len(lifted_rows) == 12 and np.all(np.diff(lifted_rows) == 1):
        cube_in_tool = np.linalg.solve(tool[lifted_rows], cube[lifted_rows])
        offsets = cube_in_tool[:, :3, 3]
        carried = bool(
            np.max(np.linalg.norm(offsets, axis=-1)) <= 0.12
            and np.max(np.linalg.norm(offsets - offsets[0], axis=-1)) <= 0.005
        )

    cube_in_platform = np.linalg.solve(platform, cube)
    cube_half = np.asarray(manifest["size_m"]["cube"]) / 2.0
    platform_half = np.asarray(manifest["size_m"]["platform"]) / 2.0
    corners = np.asarray(list(product((-1, 1), repeat=3))) * cube_half
    vertices = (
        corners @ np.swapaxes(cube_in_platform[:, :3, :3], 1, 2)
        + cube_in_platform[:, None, :3, 3]
    )
    inside = (np.abs(vertices[..., :2]) <= platform_half[:2] + 1e-6).all(axis=(1, 2))
    gaps = vertices[..., 2].min(axis=1) - platform_half[2]
    on_surface = (gaps >= -0.003) & (gaps < 0.012)
    centers = cube_in_platform[:, :3, 3]
    center_error = np.linalg.norm(centers[:, :2], axis=-1)
    window = times >= times[-1] - _STABILITY_WINDOW_S - 1e-8
    max_drift = float(np.max(np.linalg.norm(centers[window] - centers[-1], axis=-1)))
    released = bool(
        np.max(closed_fraction[window]) <= 0.1
        and np.min(np.linalg.norm(tool[window, :3, 3] - cube[window, :3, 3], axis=-1))
        >= 0.04
    )
    window_rotations = cube_in_platform[window, :3, :3]
    rotations_from_end = window_rotations @ cube_in_platform[-1, :3, :3].T
    max_rotation = float(
        np.rad2deg(
            np.arccos(
                np.clip(
                    (np.trace(rotations_from_end, axis1=1, axis2=2) - 1.0) / 2.0,
                    -1.0,
                    1.0,
                )
            )
        ).max()
    )
    stability = bool(
        times[-1] - times[window][0] >= _STABILITY_WINDOW_S - 1e-8
        and np.all(phase[window] == "settle")
        and np.all(inside[window] & on_surface[window])
        and max_drift <= 0.002
        and max_rotation <= 2.0
    )
    checks = {
        "lifted_at_least_30_mm": lift_peak >= _MIN_LIFT_M,
        "retained_during_lift": carried,
        "entire_cube_on_platform": bool(inside[-1] and on_surface[-1]),
        "center_within_10_mm": bool(center_error[-1] <= _CENTER_TOLERANCE_M),
        "gripper_released_and_retracted": released,
        "stable_for_0_5_s": stability,
    }
    return {
        "passed": manifest["program_passed"] and all(checks.values()),
        "program_passed": manifest["program_passed"],
        "checks": checks,
        "lift_m": lift_peak,
        "final_gripper_position_m": float(qpos[-1, grip["joint_index"]]),
        "center_error_m": float(center_error[-1]),
        "bottom_gap_m": float(gaps[-1]),
        "final_window_max_drift_m": max_drift,
        "final_window_max_rotation_deg": max_rotation,
        "verified_stability_window_s": _STABILITY_WINDOW_S,
        "control_steps": rows[-1]["control_step"],
        "physics_steps": rows[-1]["physics_step"],
        "simulation_time_s": float(times[-1]),
    }


def main() -> None:
    """Score an exported UR5 run without a simulator."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    folder = parser.parse_args().run_directory
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    report = verify(folder / "ur5-trajectory.jsonl", manifest)
    print(json.dumps(report, indent=2, allow_nan=False))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
