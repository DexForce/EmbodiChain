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
"""Analyze recorded states of the supplied paper-bag demo and lifting tail.

Run: .venv/bin/python -m scripts.benchmark.tech_reports.physics.experiments.dexuni.c02_analysis --run <recording>
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

__all__ = ["analyze", "main"]


def analyze(run: Path) -> dict[str, Any]:
    """Check the original primitive extents and actual loaded lift heights."""
    with np.load(run / "preview/states.npz") as a:
        s = dict(a)
    with np.load(run / "inputs/bag.npz") as a:
        mesh = dict(a)
    with np.load(run / "inputs/joint_commands.npz") as a:
        commands = dict(a)
    manifest = json.loads((run / "manifest.json").read_text())
    t = s["time"]
    inverse = np.swapaxes(s["bag_rotation"], -1, -2)
    local = np.einsum(
        "tij,tkj->tki", inverse, s["snack_pose"][:, :, 4:] - s["bag_position"][:, None]
    )
    rotation = (
        Rotation.from_quat(s["snack_pose"][:, :, :4].reshape(-1, 4))
        .as_matrix()
        .reshape(-1, 2, 3, 3)
    )
    relative = np.einsum("tij,tkjl->tkil", inverse, rotation)
    half = np.array(((0.0325, 0.0325, 0.059), (0.024, 0.030, 0.059)))
    extent = np.empty_like(local)
    extent[:, 0] = half[0, 0] * np.linalg.norm(relative[:, 0, :, :2], axis=-1) + half[
        0, 2
    ] * np.abs(relative[:, 0, :, 2])
    extent[:, 1] = np.abs(relative[:, 1]) @ half[1]
    loaded = (
        (
            np.abs(local[:, :, :2]) + extent[:, :, :2]
            < np.array((0.13 / 2 + 0.003, 0.32 / 2 + 0.003))
        ).all(-1)
        & (local[:, :, 2] - extent[:, :, 2] > -0.007)
        & (local[:, :, 2] + extent[:, :, 2] < 0.25)
    )
    assert np.array_equal(
        loaded, s["loaded"]
    ), "Independent extent formula disagrees with reference checks"
    world_extent = np.empty((len(t), 2))
    world_extent[:, 0] = half[0, 0] * np.linalg.norm(
        rotation[:, 0, 2, :2], axis=-1
    ) + half[0, 2] * np.abs(rotation[:, 0, 2, 2])
    world_extent[:, 1] = np.abs(rotation[:, 1, 2]) @ half[1]
    bottom = s["bag"][:, mesh["bottom"], 2].min(1)
    payload_bottom = s["snack_pose"][:, :, 6] - world_extent
    hold = (t >= 58 - 1e-8) & (t <= 60 + 1e-8)
    anchor = np.flatnonzero(t <= 56 + 1e-8)[-1]
    original = np.flatnonzero(t <= 50 + 1e-8)[-1]
    grip = {}
    for side, y in enumerate((0.16, -0.16)):
        ids = mesh["rim"][
            (np.abs(mesh["vertices"][mesh["rim"], 1] - y) < 0.001)
            & (np.abs(mesh["vertices"][mesh["rim"], 0]) < 0.025)
        ]
        delta = s["bag"][:, ids].mean(1) - s["tcp"][:, side]
        grip["left" if side == 0 else "right"] = (
            float(np.linalg.norm(delta[t >= 56 - 1e-8] - delta[anchor], axis=-1).max())
            if hold.any()
            else None
        )
    result = dict(
        task="C02",
        reference_scene=True,
        states_sha256=hashlib.sha256(
            (run / "preview/states.npz").read_bytes()
        ).hexdigest(),
        finite=all(np.isfinite(v).all() for v in s.values()),
        samples=len(t),
        actual_seconds=float(t[-1]),
        actual_steps=manifest.get("actual_steps"),
        original_final_checks_passed=manifest.get(
            "reference_final_checks_passed", False
        ),
        original_50s_loaded=loaded[original].tolist(),
        final_loaded=loaded[-1].tolist(),
        max_joint_replay_error=float(np.max(np.abs(s["qpos"] - commands["targets"]))),
        max_tcp_target_error_m=float(
            np.linalg.norm(s["tcp"] - s["tcp_targets"], axis=-1).max()
        ),
        max_clock_error_s=float(abs(t - commands["time"]).max()),
        max_shape_fit_error_m=float(s["shape_error"].max()),
        max_lip_bow_m=manifest.get("max_lip_bow_m"),
        self_contact_enabled=manifest["effective"]["vbd"][
            "particle_enable_self_contact"
        ],
        graph=manifest["effective"]["cuda_graph_status"],
        lift=dict(
            completed=bool(hold.sum() >= 120 and t[-1] >= 64 - 1e-8),
            hold_samples=int(hold.sum()),
            hold_seconds=[58, 60],
            mean_fitted_bag_lift_m=(
                float(s["bag_position"][hold, 2].mean() - s["bag_position"][anchor, 2])
                if hold.any()
                else None
            ),
            bag_bottom_min_clearance_m=(
                float(bottom[hold].min() - 0.92) if hold.any() else None
            ),
            payload_bottom_min_clearance_m=(
                (payload_bottom[hold].min(0) - 0.92).tolist() if hold.any() else None
            ),
            both_loaded_throughout_hold=bool(hold.any() and loaded[hold].all()),
            rim_to_tcp_relative_drift_after_grasp_m=grip,
        ),
        scope="Exact reference primitive extents and recorded outer steps; no continuous-collision or full cloth self-intersection guarantee",
    )
    output = run / "analysis"
    output.mkdir(exist_ok=True)
    (output / "current_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    np.savez_compressed(
        output / "plot_data.npz",
        time=t,
        bag_bottom_height=bottom - 0.92,
        payload_bottom_height=payload_bottom - 0.92,
        bag_position=s["bag_position"],
        loaded=loaded,
        shape_error=s["shape_error"],
        local_snack_centres=local,
        local_snack_extent=extent,
    )
    return result


def main() -> None:
    """Read existing output without executing simulation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    print(json.dumps(analyze(parser.parse_args().run), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
