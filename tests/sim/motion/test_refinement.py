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


def test_opt_in_pick_projection_consumes_budget_and_preserves_source() -> None:
    poses, phases = _pick()
    progress = np.linspace(0, 1, 40)
    poses[60:, 0, 3] = 0.012 * progress
    poses[60:, :3, :3] = Rotation.from_euler(
        "z", 15 * progress[:, None], degrees=True
    ).as_matrix()
    before = poses.copy()
    cfg = MotionRefinementCfg(
        project_pick_lift=True, max_rotation_rad=np.deg2rad(20), smoothing_weight=0
    )
    constraints = InteractionConstraintSet(
        "pick_up", phases, 60, np.eye(4), "box-side-faces", lift_height_m=0.1
    )
    result = KinematicMotionRefiner(cfg).refine(poses, constraints)
    expected = np.eye(4)
    expected[2, 3] = 0.1
    np.testing.assert_allclose(
        result.fit.poses[-2:], np.repeat(expected[None], 2, axis=0), atol=1e-10
    )
    np.testing.assert_allclose(
        result.fit.poses[60:62], np.repeat(np.eye(4)[None], 2, axis=0), atol=1e-10
    )
    np.testing.assert_array_equal(poses, before)
    assert result.report["pick_lift_projected"] is True
    assert result.fit.max_translation_m <= 0.02
    assert result.fit.max_rotation_rad <= np.deg2rad(20)
    legacy = KinematicMotionRefiner(
        MotionRefinementCfg(max_rotation_rad=np.deg2rad(20), smoothing_weight=0)
    ).refine(poses, constraints)
    np.testing.assert_allclose(legacy.fit.poses[-1], poses[-1], atol=1e-10)
    assert legacy.report["pick_lift_projected"] is False
    with pytest.raises(ValueError, match="budget"):
        KinematicMotionRefiner(replace(cfg, max_translation_m=0.005)).refine(
            poses, constraints
        )
    with pytest.raises(ValueError, match="explicit goal height"):
        KinematicMotionRefiner(cfg).refine(
            poses, replace(constraints, lift_height_m=None)
        )


def test_local_pick_approach_repairs_heading_without_rotating_distant_entry() -> None:
    poses, phases = _pick()
    poses[:40, 0, 3] = np.linspace(-0.25, 0, 40)
    goal = np.eye(4)
    goal[:3, :3] = Rotation.from_euler("z", 40, degrees=True).as_matrix()
    cfg = MotionRefinementCfg(
        project_pick_approach=True,
        pick_entry_projection_samples=24,
        max_translation_m=0.05,
        max_rotation_rad=np.deg2rad(50),
        smoothing_weight=0,
    )
    constraints = InteractionConstraintSet(
        "pick_up", phases, 60, goal, "directed-side-faces"
    )
    result = KinematicMotionRefiner(cfg).refine(poses, constraints)
    np.testing.assert_allclose(
        result.fit.poses[:2], np.repeat(poses[:1], 2, axis=0), atol=1e-10
    )
    np.testing.assert_allclose(
        result.fit.poses[38:60], np.repeat(goal[None], 22, axis=0), atol=1e-10
    )
    assert result.report["pick_approach_projected"] is True
    assert result.fit.max_translation_m < 0.05
    assert result.fit.max_rotation_rad <= np.deg2rad(50)
    # A full rigid yaw warp would move the far free entry by > 15 cm.
    with pytest.raises(ValueError, match="budget"):
        KinematicMotionRefiner(
            replace(cfg, project_pick_approach=False, pick_entry_projection_samples=0)
        ).refine(poses, constraints)


def test_free_boundaries_preserve_release_and_express_a_short_retract() -> None:
    poses = np.repeat(np.eye(4)[None], 35, axis=0)
    poses[:24, 0, 3] = np.linspace(0.05, 0.0, 24)
    poses[30:, 2, 3] = np.linspace(0.0, 0.03, 5)
    before = poses.copy()
    phases = (
        PhaseAnnotation("approach", 0, 24, "free", "target"),
        PhaseAnnotation("release", 24, 30, "contact", "target"),
        PhaseAnnotation("retract", 30, 35, "free", "target"),
    )
    result = KinematicMotionRefiner(
        MotionRefinementCfg(project_free_boundaries=True, smoothing_weight=0)
    ).refine(
        poses, InteractionConstraintSet("place", phases, 24, np.eye(4), "box-support")
    )
    assert len(result.fit.poses) == 35
    np.testing.assert_allclose(
        result.fit.poses[:2], np.repeat(poses[:1], 2, axis=0), atol=1e-10
    )
    np.testing.assert_allclose(
        result.fit.poses[22:31], np.repeat(np.eye(4)[None], 9, axis=0), atol=1e-10
    )
    np.testing.assert_allclose(
        result.fit.poses[30:, 2, 3], [0.0, 0.001875, 0.0075, 0.016875, 0.03], atol=1e-10
    )
    np.testing.assert_array_equal(poses, before)
    assert result.report["time_grid_changed"] is False
    assert result.fit.max_translation_m <= 0.02
