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

"""Grounded interaction repair preserves gravity, phases and bounded evidence."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from embodichain.data_pipeline.real2sim import PhaseAnnotation
from embodichain.lab.sim.motion.refinement import (
    InteractionConstraintSet,
    KinematicMotionRefiner,
    MotionRefinementCfg,
)


def _pick() -> tuple[np.ndarray, tuple[PhaseAnnotation, ...]]:
    poses = np.repeat(np.eye(4)[None], 100, axis=0)
    poses[:40, 2, 3] = np.linspace(0.1, 0, 40)
    poses[60:, 2, 3] = np.linspace(0, 0.1, 40)
    phases = (
        PhaseAnnotation("approach", 0, 40, "free", "object"),
        PhaseAnnotation("close", 40, 60, "contact", "object"),
        PhaseAnnotation("lift", 60, 100, "hold", "gravity"),
    )
    return poses, phases


@pytest.mark.parametrize("lift_kind", ["hold", "contact"])
def test_corrupt_grasp_is_repaired_then_contact_baseline_is_frozen(
    lift_kind: str,
) -> None:
    oracle, phases = _pick()
    phases = (*phases[:2], replace(phases[2], kind=lift_kind))
    source = oracle.copy()
    source[:, 0, 3] += 0.008
    source[:, :3, :3] = Rotation.from_euler("x", 4, degrees=True).as_matrix()
    refiner = KinematicMotionRefiner(MotionRefinementCfg(smoothing_weight=0))
    result = refiner.refine(
        source,
        InteractionConstraintSet("pick_up", phases, 60, np.eye(4), "analytic-box-v1"),
    )
    np.testing.assert_allclose(result.fit.poses, oracle, atol=1e-9)
    np.testing.assert_array_equal(result.fit.poses[40:60], oracle[40:60])
    # A tool roll correction must not rotate the vertical lift into a lateral move.
    np.testing.assert_allclose(result.fit.poses[60:, :2, 3], 0, atol=1e-12)
    assert result.fit.poses[-1, 2, 3] == pytest.approx(0.1)
    assert result.report["time_grid_changed"] is False
    assert result.report["calibration_estimated"] is False
    assert result.report["collision_qualified"] is False
    assert source[60, 0, 3] == pytest.approx(0.008)
    repeated = refiner.refine(
        source,
        InteractionConstraintSet("pick_up", phases, 60, np.eye(4), "analytic-box-v1"),
    )
    assert result.report["baseline_id"] == repeated.report["baseline_id"]


def test_refinement_rejects_qpos_authority_and_over_budget_source() -> None:
    poses, phases = _pick()
    with pytest.raises(ValueError, match="qpos-authority"):
        InteractionConstraintSet(
            "pick_up",
            (replace(phases[0], authority="qpos"), *phases[1:]),
            60,
            np.eye(4),
            "box",
        )
    poses[:, 0, 3] = 0.03
    with pytest.raises(ValueError, match="budget"):
        KinematicMotionRefiner(MotionRefinementCfg()).refine(
            poses, InteractionConstraintSet("pick_up", phases, 60, np.eye(4), "box")
        )


def test_transport_goal_repair_keeps_sample_count_and_endpoints() -> None:
    poses = np.repeat(np.eye(4)[None], 60, axis=0)
    poses[:, 0, 3] = np.linspace(0, 0.1, 60)
    goal = poses[-1].copy()
    goal[1, 3] = 0.006
    phases = (PhaseAnnotation("transport", 0, 60, "free", "target"),)
    result = KinematicMotionRefiner(MotionRefinementCfg()).refine(
        poses,
        InteractionConstraintSet(
            "move_held_object", phases, 59, goal, "held-relation-v1"
        ),
    )
    assert result.fit.poses.shape == poses.shape
    np.testing.assert_allclose(result.fit.poses[:, 0, 3], poses[:, 0, 3], atol=1e-9)
    np.testing.assert_allclose(result.fit.poses[:, 1, 3], 0.006, atol=1e-9)
