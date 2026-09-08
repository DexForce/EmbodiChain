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

from __future__ import annotations

from dataclasses import replace
import math

import pytest
import torch

from embodichain.lab.sim.motion.expansion.contracts import (
    TrajectoryPhase,
    TrajectoryTemplate,
)
from embodichain.lab.sim.motion.expansion.operators import (
    joint_residual,
    retime,
    rotate_grasp_about_object_axis,
    validate_motion_limits,
)


def test_grasp_rotation_uses_object_origin_and_preserves_inputs() -> None:
    object_pose = torch.eye(4)
    object_pose[:3, 3] = torch.tensor([2.0, 3.0, 4.0])
    grasp = object_pose.clone()
    grasp[0, 3] += 0.2
    grasp[2, 3] += 0.1
    before = grasp.clone()
    candidates = rotate_grasp_about_object_axis(
        object_pose,
        grasp,
        axis=torch.tensor([0.0, 0.0, 2.0]),
        angles=torch.tensor([0.0, math.pi / 2]),
    )
    torch.testing.assert_close(candidates[0], grasp)
    torch.testing.assert_close(candidates[1, :3, 3], torch.tensor([2.0, 3.2, 4.1]))
    torch.testing.assert_close(
        candidates[1, :3, 0], torch.tensor([0.0, 1.0, 0.0]), atol=1e-6, rtol=0
    )
    torch.testing.assert_close(grasp, before)
    candidates[0, 0, 3] = 100
    torch.testing.assert_close(grasp, before)


def test_grasp_variants_follow_rotated_object_frame() -> None:
    # The object's local Z points along scene X. Quarter-turn augmentation
    # must orbit that axis, even after translating the entire scene.
    object_pose = torch.tensor(
        [
            [0.0, 0.0, 1.0, 3.0],
            [0.0, 1.0, 0.0, -2.0],
            [-1.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    grasp = object_pose.clone()
    grasp[2, 3] -= 0.2
    candidate = rotate_grasp_about_object_axis(
        object_pose,
        grasp,
        axis=torch.tensor([0.0, 0.0, 1.0]),
        angles=torch.tensor([math.pi / 2]),
    )[0]
    torch.testing.assert_close(candidate[:3, 3], torch.tensor([3.0, -1.8, 1.0]))
    torch.testing.assert_close(candidate[:3, :3].T @ candidate[:3, :3], torch.eye(3))
    assert torch.linalg.det(candidate[:3, :3]) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "bad", ["reflection", "zero_axis", "axis_shape", "nan_angle", "empty_angles"]
)
def test_grasp_rotation_rejects_invalid_geometry(bad: str) -> None:
    pose = torch.eye(4)
    axis = torch.tensor([0.0, 0.0, 1.0])
    angles = torch.zeros(1)
    if bad == "reflection":
        pose[0, 0] = -1
    elif bad == "zero_axis":
        axis.zero_()
    elif bad == "axis_shape":
        axis = torch.ones(2)
    elif bad == "nan_angle":
        angles[0] = torch.nan
    elif bad == "empty_angles":
        angles = torch.zeros(0)
    with pytest.raises(ValueError):
        rotate_grasp_about_object_axis(pose, torch.eye(4), axis=axis, angles=angles)


def template() -> TrajectoryTemplate:
    return TrajectoryTemplate(
        source_id="reference",
        source_revision="one",
        template_id="pick",
        joint_names=("arm", "gripper"),
        positions=torch.tensor(
            [[0.0, 0.0], [0.25, 0.0], [0.5, 0.0], [0.5, 0.0], [0.5, 0.1], [0.5, 0.1]]
        ),
        dt=torch.tensor([0.0, 0.1, 0.1, 0.1, 0.1, 0.1]),
        phases=(
            TrajectoryPhase(
                "transit", 0, 3, allowed_operators=("retime", "joint_residual")
            ),
            TrajectoryPhase("grasp", 3, 6, kind="contact"),
        ),
        controlled_joint_indices=(0,),
        allowed_operators=("retime", "joint_residual"),
    )


def test_retime_changes_length_preserves_contact_duration_and_endpoints() -> None:
    source = template()
    result = retime(source, duration_scale=2.0, control_dt=0.1)
    assert result.positions.shape == (8, 2)
    assert result.phases[0].stop_index == 5
    assert result.phases[1].start_index == 5
    torch.testing.assert_close(result.positions[4], source.positions[2])
    torch.testing.assert_close(result.positions[5:], source.positions[3:])
    torch.testing.assert_close(result.dt[6:].sum(), source.dt[4:].sum())
    torch.testing.assert_close(source.positions, template().positions)


def test_spatial_residual_preserves_locked_samples_and_uncontrolled_joints() -> None:
    source = template()
    generator = torch.Generator().manual_seed(5)
    result = joint_residual(
        source,
        joint_limits=torch.tensor([[-1.0, 1.0], [-1.0, 1.0]]),
        normalized_scale=0.05,
        generator=generator,
    )
    assert result.positions[1, 0] != source.positions[1, 0]
    assert torch.equal(
        result.positions[[0, 2, 3, 4, 5]], source.positions[[0, 2, 3, 4, 5]]
    )
    assert torch.equal(result.positions[:, 1], source.positions[:, 1])


def test_explicit_generator_is_reproducible_without_consuming_global_rng() -> None:
    before = torch.random.get_rng_state().clone()
    kwargs = dict(
        joint_limits=torch.tensor([[-1.0, 1.0], [-1.0, 1.0]]), normalized_scale=0.05
    )
    first = joint_residual(
        template(), generator=torch.Generator().manual_seed(2), **kwargs
    )
    second = joint_residual(
        template(), generator=torch.Generator().manual_seed(2), **kwargs
    )
    assert torch.equal(first.positions, second.positions)
    assert torch.equal(before, torch.random.get_rng_state())


def test_unlabelled_qpos_cannot_be_augmented() -> None:
    with pytest.raises(ValueError, match="annotated"):
        retime(replace(template(), phases=()), duration_scale=2, control_dt=0.1)


def test_retime_does_not_silently_move_uncontrolled_schedule() -> None:
    source = template()
    source.positions[1, 1] = 0.01
    with pytest.raises(ValueError, match="uncontrolled"):
        retime(source, duration_scale=2, control_dt=0.1)


def test_retime_rejects_clock_mismatch_and_unbounded_allocation() -> None:
    with pytest.raises(ValueError, match="clock"):
        retime(template(), duration_scale=2, control_dt=0.07)
    with pytest.raises(ValueError, match="budget"):
        retime(template(), duration_scale=1000, control_dt=0.1, max_samples=10)


def test_dynamic_limits_reject_overfast_motion_but_do_not_claim_task_success() -> None:
    source = template()
    failed = validate_motion_limits(
        source,
        velocity_limits=torch.tensor([1.0, 2.0]),
        acceleration_limits=torch.full((2,), 100.0),
    )
    assert not failed.accepted
    slow = retime(source, duration_scale=4, control_dt=0.1)
    passed = validate_motion_limits(
        slow,
        velocity_limits=torch.tensor([1.0, 2.0]),
        acceleration_limits=torch.full((2,), 100.0),
    )
    assert passed.accepted
    assert passed.checks[0].check_id == "motion_limits"


def test_contact_internal_event_cannot_be_removed_by_host_clock() -> None:
    source = replace(template(), dt=torch.tensor([0.0, 0.1, 0.1, 0.1, 0.05, 0.15]))
    with pytest.raises(ValueError, match="clock"):
        retime(source, duration_scale=2, control_dt=0.1)


def test_large_derivatives_return_a_failed_check() -> None:
    source = replace(
        template(), positions=template().positions * 1e38, dt=template().dt * 1e-3
    )
    result = validate_motion_limits(
        source, velocity_limits=torch.ones(2), acceleration_limits=torch.ones(2)
    )
    assert not result.accepted
    assert result.checks[0].metrics["speed_ratio"] > 1e30


def test_half_precision_derivatives_are_accumulated_in_high_precision() -> None:
    source = replace(
        template(), positions=template().positions.half(), dt=template().dt.half()
    )
    result = validate_motion_limits(
        source,
        velocity_limits=torch.full((2,), 20.0),
        acceleration_limits=torch.full((2,), 0.0001),
    )
    assert not result.accepted
    assert result.checks[0].metrics["acceleration_ratio"] > 65504
