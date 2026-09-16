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

"""Tests for atomic_actions.affordance (Affordance, AntipodalAffordance, InteractionPoints)."""

from __future__ import annotations

from dataclasses import replace
import math

import pytest
import torch

from embodichain.lab.sim.atomic_actions.affordance_sampling import (
    AffordanceExpansionCfg,
    AffordanceSamplingContext,
    AffordancePoseCandidates,
)
from embodichain.lab.sim.atomic_actions.affordance import (
    Affordance,
    AssembleAffordance,
    InteractionPoints,
    PressAffordance,
    TwistAffordance,
)
from embodichain.lab.sim.atomic_actions.state import (
    PlanningContext,
    RobotObservation,
    TaskState,
    SceneSnapshot,
    ObservedArticulationJointState,
)
from embodichain.lab.sim.atomic_actions.primitives._helpers import _sample_joint_motion
from embodichain.utils.math import axis_angle_to_rotation_matrix

BRANCH_COUNT = 9


def _sampling() -> AffordanceSamplingContext:
    return AffordanceSamplingContext(count=BRANCH_COUNT, seed=42, episode_id=3)


def _poses(count: int = BRANCH_COUNT) -> torch.Tensor:
    return torch.eye(4).repeat(count, 1, 1)


@pytest.mark.parametrize("count", [0, -1, True, 1.5])
def test_expansion_rejects_nonpositive_or_noninteger_count(count):
    with pytest.raises(ValueError):
        AffordanceExpansionCfg(count=count)


def test_stratified_values_are_reproducible_and_independent_of_row_order():
    sampling = _sampling()
    ids = torch.arange(BRANCH_COUNT)
    original_state = torch.random.get_rng_state().clone()
    values = sampling.sample_range(0.0, (-math.pi, math.pi), env_ids=ids, key="press:1")
    reversed_values = sampling.sample_range(
        0.0, (-math.pi, math.pi), env_ids=ids.flip(0), key="press:1"
    )
    assert torch.equal(values, reversed_values.flip(0))
    assert torch.equal(torch.random.get_rng_state(), original_state)
    assert values[0] == 0.0
    assert len(torch.unique(values)) == BRANCH_COUNT
    assert not torch.equal(
        values,
        replace(sampling, attempt_id=1).sample_range(
            0.0, (-math.pi, math.pi), env_ids=ids, key="press:1"
        ),
    )
    assert not torch.equal(
        values,
        sampling.sample_range(0.0, (-math.pi, math.pi), env_ids=ids, key="press:2"),
    )


def test_fixed_target_and_disabled_expansion_preserve_nominal():
    ids = torch.arange(BRANCH_COUNT)
    for sampling, bounds in [
        (_sampling(), None),
        (AffordanceSamplingContext(), (0.0, 1.0)),
    ]:
        values = sampling.sample_range(0.4, bounds, env_ids=ids, key="fixed")
        torch.testing.assert_close(values, torch.full((BRANCH_COUNT,), 0.4))


def test_candidates_distribute_distinct_grasps_in_object_coordinates():
    local = _poses()
    local[:, 0, 3] = torch.linspace(-0.04, 0.04, BRANCH_COUNT)
    objects = _poses()
    objects[:, 1, 3] = torch.arange(BRANCH_COUNT) * 2.5
    poses = objects[:, None] @ local[None]
    costs = torch.arange(BRANCH_COUNT, dtype=torch.float32).repeat(BRANCH_COUNT, 1)
    batch = AffordancePoseCandidates(
        poses, costs, torch.ones_like(costs, dtype=torch.bool)
    )
    success, selected, metadata = batch.select(
        _sampling(),
        env_ids=torch.arange(BRANCH_COUNT),
        key="pick",
        reference_poses=objects,
    )
    assert success.all()
    assert len(set(metadata["candidate_ids"])) == BRANCH_COUNT
    assert metadata["candidate_ids"][0] == 0
    torch.testing.assert_close(selected[:, 1, 3], objects[:, 1, 3])
    # Slicing/permuting rows cannot change the branch selection.
    order = torch.tensor([7, 2, 8])
    subset = AffordancePoseCandidates(poses[order], costs[order], batch.valid[order])
    _, subset_poses, _ = subset.select(
        _sampling(), env_ids=order, key="pick", reference_poses=objects[order]
    )
    torch.testing.assert_close(subset_poses, selected[order])


def test_empty_nonfinite_duplicate_and_padded_candidates_are_not_new_samples():
    row = _poses(3)
    row[2, 0, 3] = 0.05
    batch = AffordancePoseCandidates.from_rows(
        [
            (row, torch.tensor([0.0, 0.1, float("inf")])),
            (torch.empty(0, 4, 4), torch.empty(0)),
            (row[:1], torch.zeros(1)),
        ],
        device="cpu",
    )
    success, poses, metadata = batch.select(
        _sampling(), env_ids=torch.tensor([0, 1, 8]), key="pick"
    )
    assert success.tolist() == [True, False, True]
    assert metadata["candidate_ids"] == [0, -1, 0]
    assert metadata["unique_candidate_counts"] == [1, 0, 1]
    assert metadata["reused"] == [False, False, True]
    assert torch.isfinite(poses).all()


def test_press_roll_preserves_contact_point_and_forward_axis_in_rotated_scene():
    target = _poses()
    target[:, :3, :3] = axis_angle_to_rotation_matrix(torch.tensor([0.4, 0.2, -0.3]))
    target[:, :3, 3] = torch.tensor([0.5, 0.1, 0.3])
    affordance = PressAffordance(
        press_axis=torch.tensor([1.0, 0.0, 0.0]), press_position=(0.01, 0.02, 0.03)
    )
    nominal = affordance.get_press_pose(target)
    expanded = affordance.get_press_pose(
        target, sampling=_sampling(), env_ids=torch.arange(BRANCH_COUNT)
    )
    torch.testing.assert_close(expanded[:, :3, 3], nominal[:, :3, 3])
    torch.testing.assert_close(expanded[:, :3, 2], nominal[:, :3, 2])
    torch.testing.assert_close(
        torch.linalg.det(expanded[:, :3, :3]), torch.ones(BRANCH_COUNT)
    )
    assert len(torch.unique(expanded[:, :3, 0], dim=0)) == BRANCH_COUNT


def test_twist_contact_roll_requires_declared_symmetry():
    affordance = TwistAffordance(
        grasp_position=(0.1, 0.0, 0.0), axis_origin=(0.0, 0.0, 0.0)
    )
    nominal = affordance.get_grasp_pose(_poses())
    fixed = affordance.get_grasp_pose(_poses(), sampling=_sampling())
    torch.testing.assert_close(fixed, nominal)
    affordance.grasp_roll_range = (-0.2, 0.2)
    expanded = affordance.get_grasp_pose(_poses(), sampling=_sampling())
    torch.testing.assert_close(expanded[:, :3, 3], nominal[:, :3, 3])
    torch.testing.assert_close(expanded[:, :3, 2], nominal[:, :3, 2])
    assert not torch.equal(expanded, nominal)


def test_assembly_samples_only_declared_equivalent_rotations():
    rotations = _poses(3)
    rotations[:, :3, :3] = axis_angle_to_rotation_matrix(
        torch.tensor(
            [[0.0, 0.0, math.pi / 2], [0.0, 0.0, math.pi], [0.0, 0.0, -math.pi / 2]]
        )
    )
    relation = torch.eye(4)
    relation[2, 3] = 0.03
    affordance = AssembleAffordance(
        assemble_to_base_pose=relation, symmetry_transforms=rotations
    )
    result = affordance.get_assemble_object_pose(_poses(), sampling=_sampling())
    torch.testing.assert_close(
        result[:, :3, 3], relation[:3, 3].expand(BRANCH_COUNT, -1)
    )
    assert len(torch.unique(result.flatten(1), dim=0)) == 4
    affordance.symmetry_transforms[0, 0, 3] = 0.01
    with pytest.raises(ValueError, match="mating center"):
        affordance.get_assemble_object_pose(_poses(), sampling=_sampling())


def test_interaction_points_sample_matching_surface_points_and_normals():
    affordance = InteractionPoints(
        points=torch.tensor([[0.0, 0.0, 0.0], [0.03, 0.0, 0.0], [9.0, 0.0, 0.0]]),
        normals=torch.tensor([[0.0, 0.0, -1.0]]).repeat(3, 1),
        point_types=["press", "press", "pull"],
    )
    result = affordance.sample_poses(
        _poses(),
        sampling=_sampling(),
        env_ids=torch.arange(BRANCH_COUNT),
        point_type="press",
    )
    assert (result[:, 0, 3] <= 0.03).all()
    assert len(torch.unique(result[:, :3, 3], dim=0)) == 2
    torch.testing.assert_close(
        result[:, :3, 2], torch.tensor([0.0, 0.0, 1.0]).expand(BRANCH_COUNT, -1)
    )


def test_base_affordance_preserves_exact_constraint_without_aliasing():
    target = _poses()
    result = Affordance().sample_poses(
        target, sampling=_sampling(), env_ids=torch.arange(BRANCH_COUNT)
    )
    torch.testing.assert_close(result, target)
    assert result.data_ptr() != target.data_ptr()


def _joint_context(position: torch.Tensor) -> PlanningContext:
    count = len(position)
    qpos = torch.zeros(count, 2)
    scene = SceneSnapshot(
        timestamp=0.0,
        version=0,
        entities={},
        articulation_joints={
            ("cabinet", "joint"): ObservedArticulationJointState(
                position=position[:, None]
            ),
        },
    )
    return PlanningContext(
        robot=RobotObservation(timestamp=0.0, qpos=qpos, qvel=qpos.clone()),
        task=TaskState.empty(batch_size=count, device="cpu"),
        scene=scene,
        env_ids=torch.arange(count),
        control_dt=0.01,
        affordance_sampling=_sampling(),
    )


def test_travel_intersects_task_interval_with_signed_joint_limits():
    context = _joint_context(torch.tensor([0.5, 0.9, 0.1]))
    # A pull along negative affordance axis maps to positive native joint travel.
    travel, valid = _sample_joint_motion(
        0.08,
        (0.05, 0.2),
        context,
        joint_name="joint",
        joint_limits=(0.0, 1.0),
        joint_axis_sign=-1,
        motion_sign=-1,
        key="pull",
    )
    assert valid.all()
    assert travel[0] == pytest.approx(0.08)
    assert 0.05 <= travel[1] <= 0.1
    assert 0.05 <= travel[2] <= 0.2
    assert (
        context.scene.articulation_joints[("cabinet", "joint")].position[:, 0] + travel
        <= 1.0
    ).all()


def test_travel_with_empty_legal_interval_fails_only_that_row():
    context = _joint_context(torch.tensor([0.5, 0.99]))
    travel, valid = _sample_joint_motion(
        0.08,
        (0.05, 0.2),
        context,
        joint_name="joint",
        joint_limits=(0.0, 1.0),
        joint_axis_sign=1,
        motion_sign=1,
        key="twist",
    )
    assert valid.tolist() == [True, False]
    assert torch.isfinite(travel).all()
    assert (
        context.project(qpos=context.robot.qpos, task=context.task).affordance_sampling
        == context.affordance_sampling
    )
