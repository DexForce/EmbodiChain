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

"""Tests for pure atomic-action trajectory operations."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from embodichain.lab.sim.atomic_actions.plans import normalize_success_mask
from embodichain.lab.sim.atomic_actions.primitives._helpers import (
    resolve_object_target,
)
from embodichain.lab.sim.atomic_actions.trajectory_ops import (
    axis_translation_keyframes,
    build_joint_plan_states,
    build_pose_plan_states,
    interpolate_hand_qpos,
    interpolate_joint_trajectory,
    resolve_joint_target,
    resolve_pose_target,
    split_three_segments,
    translate_pose_world,
)
from embodichain.lab.sim.planners import MoveType

CPU = torch.device("cpu")


class TestNormalizeSuccessMask:
    def test_python_bool_is_expanded_without_collapsing_batch(self):
        success = normalize_success_mask(
            True,
            num_envs=2,
            device=CPU,
            name="IK success",
        )

        assert success.tolist() == [True, True]

    def test_per_environment_tensor_is_preserved(self):
        success = normalize_success_mask(
            torch.tensor([True, False]),
            num_envs=2,
            device=CPU,
            name="IK success",
        )

        assert success.tolist() == [True, False]

    def test_binary_integer_success_is_normalized_at_planner_boundary(self):
        success = normalize_success_mask(
            torch.tensor([1, 0], dtype=torch.int32),
            num_envs=2,
            device=CPU,
            name="IK success",
        )

        assert success.dtype == torch.bool
        assert success.tolist() == [True, False]

    def test_non_binary_integer_success_is_rejected(self):
        with pytest.raises(TypeError, match="binary integer"):
            normalize_success_mask(
                torch.tensor([1, 2], dtype=torch.int32),
                num_envs=2,
                device=CPU,
                name="IK success",
            )

    def test_cuda_device_requires_available_runtime(self, monkeypatch):
        def unexpected_current_device():
            raise AssertionError("current_device must not be queried")

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(torch.cuda, "current_device", unexpected_current_device)

        with pytest.raises(ValueError, match="CUDA device requested"):
            normalize_success_mask(
                True,
                num_envs=2,
                device="cuda",
                name="IK success",
            )


class TestResolvePoseTarget:
    def test_unbatched_pose_broadcasts(self):
        pose = torch.eye(4)
        out = resolve_pose_target(pose, num_envs=2, device=CPU)
        assert out.shape == (2, 4, 4)

    def test_batched_pose_passes_through(self):
        pose = torch.eye(4).unsqueeze(0).repeat(2, 1, 1)
        out = resolve_pose_target(pose, num_envs=2, device=CPU)
        assert torch.equal(out, pose)

    def test_batched_pose_returns_owned_tensor(self):
        pose = torch.eye(4).unsqueeze(0).repeat(2, 1, 1)
        out = resolve_pose_target(pose, num_envs=2, device=CPU)
        out[:, 0, 0] = 2.0
        assert torch.equal(pose, torch.eye(4).unsqueeze(0).repeat(2, 1, 1))

    def test_pose_converts_to_float32_on_requested_device(self):
        pose = torch.eye(4, dtype=torch.float64)
        out = resolve_pose_target(pose, num_envs=2, device=CPU)
        assert out.dtype == torch.float32
        assert out.device == CPU

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError):
            resolve_pose_target(torch.eye(3), num_envs=2, device=CPU)

    def test_multi_waypoint_passes_through(self):
        pose = torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(2, 3, 1, 1)
        pose[0, 1, :3, 3] = torch.tensor([1.0, 0.0, 0.0])
        out = resolve_pose_target(pose, num_envs=2, device=CPU)
        assert out.shape == (2, 3, 4, 4)
        assert torch.equal(out, pose.to(torch.float32))

    def test_multi_waypoint_wrong_envs_raises(self):
        pose = torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(3, 2, 1, 1)
        with pytest.raises(ValueError):
            resolve_pose_target(pose, num_envs=2, device=CPU)

    def test_multi_waypoint_empty_raises(self):
        empty = torch.zeros((2, 0, 4, 4), dtype=torch.float32)
        with pytest.raises(ValueError, match="zero waypoints"):
            resolve_pose_target(empty, num_envs=2, device=CPU)


class TestResolveObjectTarget:
    def test_batched_pose_returns_owned_tensor(self):
        pose = torch.eye(4).unsqueeze(0).repeat(2, 1, 1)
        out = resolve_object_target(
            pose,
            num_envs=2,
            device=CPU,
        )

        out[:, 0, 0] = 2.0

        assert torch.equal(pose, torch.eye(4).unsqueeze(0).repeat(2, 1, 1))


class TestResolveJointTarget:
    def test_unbatched_qpos_broadcasts(self):
        qpos = torch.arange(6, dtype=torch.float32)
        out = resolve_joint_target(
            qpos,
            num_envs=2,
            joint_dof=6,
            control_part="arm",
            device=CPU,
        )
        assert out.shape == (2, 6)
        assert torch.allclose(out[0], qpos)
        assert torch.allclose(out[1], qpos)

    def test_batched_qpos_passes_through(self):
        qpos = torch.arange(12, dtype=torch.float32).reshape(2, 6)
        out = resolve_joint_target(
            qpos,
            num_envs=2,
            joint_dof=6,
            control_part="arm",
            device=CPU,
        )
        assert torch.equal(out, qpos)

    def test_batched_qpos_returns_owned_tensor(self):
        qpos = torch.arange(12, dtype=torch.float32).reshape(2, 6)
        expected = qpos.clone()
        out = resolve_joint_target(
            qpos,
            num_envs=2,
            joint_dof=6,
            control_part="arm",
            device=CPU,
        )
        out.zero_()
        assert torch.equal(qpos, expected)

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError):
            resolve_joint_target(
                torch.zeros(5),
                num_envs=2,
                joint_dof=6,
                control_part="arm",
                device=CPU,
            )

    def test_multi_waypoint_passes_through(self):
        qpos = torch.arange(24, dtype=torch.float32).reshape(2, 2, 6)
        out = resolve_joint_target(
            qpos,
            num_envs=2,
            joint_dof=6,
            control_part="arm",
            device=CPU,
        )
        assert out.shape == (2, 2, 6)
        assert torch.equal(out, qpos.to(torch.float32))

    def test_multi_waypoint_wrong_envs_raises(self):
        with pytest.raises(ValueError):
            resolve_joint_target(
                torch.zeros(3, 2, 6),
                num_envs=2,
                joint_dof=6,
                control_part="arm",
                device=CPU,
            )

    def test_multi_waypoint_wrong_dof_raises(self):
        with pytest.raises(ValueError):
            resolve_joint_target(
                torch.zeros(2, 2, 5),
                num_envs=2,
                joint_dof=6,
                control_part="arm",
                device=CPU,
            )

    def test_multi_waypoint_empty_raises(self):
        empty = torch.zeros((2, 0, 6), dtype=torch.float32)
        with pytest.raises(ValueError, match="zero waypoints"):
            resolve_joint_target(
                empty,
                num_envs=2,
                joint_dof=6,
                control_part="arm",
                device=CPU,
            )


class TestBuildPlanStates:
    def test_pose_waypoints_remain_batched(self):
        poses = torch.eye(4).repeat(2, 3, 1, 1)

        states = build_pose_plan_states(poses)

        assert len(states) == 3
        assert all(state.move_type is MoveType.EEF_MOVE for state in states)
        assert torch.equal(states[1].xpos, poses[:, 1])

    def test_single_pose_target_creates_one_waypoint(self):
        poses = torch.eye(4).repeat(2, 1, 1)

        states = build_pose_plan_states(poses)

        assert len(states) == 1
        assert states[0].xpos is not None
        assert states[0].xpos.shape == (2, 4, 4)

    def test_joint_waypoints_remain_batched(self):
        qpos = torch.arange(24, dtype=torch.float32).reshape(2, 2, 6)

        states = build_joint_plan_states(qpos)

        assert len(states) == 2
        assert all(state.move_type is MoveType.JOINT_MOVE for state in states)
        assert torch.equal(states[1].qpos, qpos[:, 1])

    def test_empty_pose_waypoints_are_rejected(self):
        with pytest.raises(ValueError, match="at least one waypoint"):
            build_pose_plan_states(torch.empty(2, 0, 4, 4))


class TestSplitThreeSegments:
    def test_default_ratio(self):
        first, hand, third = split_three_segments(80, 5)
        assert hand == 5
        assert first + hand + third == 80
        assert first == int(round((80 - 5) * 0.6))

    def test_raises_when_first_segment_too_small(self):
        with pytest.raises(ValueError):
            split_three_segments(6, 5)

    def test_ratio_is_rounded_after_multiplication(self):
        first, hand, third = split_three_segments(10, 2)

        assert (first, hand, third) == (5, 2, 3)


class TestTranslatePoseWorld:
    def test_offset_adds_to_translation(self):
        pose = torch.eye(4).unsqueeze(0).repeat(2, 1, 1)
        offset = torch.tensor([0.0, 0.0, 0.1])
        out = translate_pose_world(pose, offset)
        expected = torch.tensor([0.0, 0.0, 0.1]).expand(2, 3)
        assert torch.allclose(out[:, :3, 3], expected)

    def test_batched_offset(self):
        pose = torch.eye(4).unsqueeze(0).repeat(2, 1, 1)
        offset = torch.tensor([[0.1, 0.0, 0.0], [0.0, 0.2, 0.0]])
        out = translate_pose_world(pose, offset)
        assert torch.allclose(out[0, :3, 3], torch.tensor([0.1, 0.0, 0.0]))
        assert torch.allclose(out[1, :3, 3], torch.tensor([0.0, 0.2, 0.0]))

    def test_incompatible_offset_batch_raises(self):
        pose = torch.eye(4).unsqueeze(0).repeat(2, 1, 1)
        offset = torch.zeros(3, 3)
        with pytest.raises(ValueError, match="offset batch size"):
            translate_pose_world(pose, offset)


class TestAxisTranslationKeyframes:
    def test_excludes_start_includes_end_and_stays_on_axis(self):
        start = torch.eye(4).repeat(2, 1, 1)
        start[:, :3, 3] = torch.tensor([[-0.1, 0.2, 0.3], [0.4, -0.2, 0.1]])
        axis = torch.tensor([[1.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
        axis = torch.nn.functional.normalize(axis, dim=1)
        end = start.clone()
        end[:, :3, 3] += axis * torch.tensor([[0.5], [-0.3]])

        keyframes = axis_translation_keyframes(
            start,
            end,
            axis,
            n_waypoints=5,
        )

        displacement = keyframes[:, :, :3, 3] - start[:, None, :3, 3]
        orthogonal = (
            displacement
            - (displacement * axis[:, None]).sum(dim=-1, keepdim=True) * axis[:, None]
        )
        assert keyframes.shape == (2, 5, 4, 4)
        assert torch.allclose(keyframes[:, -1], end)
        assert torch.allclose(orthogonal, torch.zeros_like(orthogonal), atol=1.0e-6)

    def test_rejects_off_axis_displacement(self):
        start = torch.eye(4).unsqueeze(0)
        end = start.clone()
        end[:, 1, 3] = 0.1

        with pytest.raises(ValueError, match="parallel to axis"):
            axis_translation_keyframes(
                start,
                end,
                torch.tensor([1.0, 0.0, 0.0]),
                n_waypoints=2,
            )


def test_interpolate_hand_qpos_preserves_endpoints():
    start = torch.tensor([[0.0, 0.0]])
    end = torch.tensor([[1.0, 1.0]])
    out = interpolate_hand_qpos(start, end, n_waypoints=5)
    assert torch.allclose(out[:, 0], start)
    assert torch.allclose(out[:, -1], end)


class TestInterpolateJointTrajectory:
    def test_interpolates_start_to_target(self):
        start = torch.zeros(2, 6)
        target = torch.ones(2, 6)
        expected = torch.ones(2, 5, 6)
        with patch(
            "embodichain.lab.sim.atomic_actions.trajectory_ops.interpolate_with_distance",
            return_value=expected,
        ) as interpolate:
            out = interpolate_joint_trajectory(start, target, n_waypoints=5)

        assert out is expected
        _, kwargs = interpolate.call_args
        assert kwargs["interp_num"] == 5
        assert torch.equal(kwargs["trajectory"][:, 0, :], start)
        assert torch.equal(kwargs["trajectory"][:, 1, :], target)

    def test_interpolates_start_through_multi_waypoints(self):
        start = torch.zeros(2, 6)
        waypoints = torch.arange(24, dtype=torch.float32).reshape(2, 2, 6)
        expected = torch.ones(2, 5, 6)
        with patch(
            "embodichain.lab.sim.atomic_actions.trajectory_ops.interpolate_with_distance",
            return_value=expected,
        ) as interpolate:
            out = interpolate_joint_trajectory(start, waypoints, n_waypoints=5)

        assert out is expected
        _, kwargs = interpolate.call_args
        assert kwargs["interp_num"] == 5
        assert kwargs["trajectory"].shape == (2, 3, 6)
        assert torch.equal(kwargs["trajectory"][:, 0, :], start)
        assert torch.equal(kwargs["trajectory"][:, 1, :], waypoints[:, 0, :])
        assert torch.equal(kwargs["trajectory"][:, 2, :], waypoints[:, 1, :])

    def test_emits_every_multi_waypoint_as_exact_sample(self):
        start = torch.zeros(2, 6)
        waypoints = torch.stack(
            [torch.ones(2, 6), torch.full((2, 6), 3.0)],
            dim=1,
        )

        result = interpolate_joint_trajectory(start, waypoints, n_waypoints=5)

        keyframes = torch.cat([start.unsqueeze(1), waypoints], dim=1)
        matches = torch.all(result.unsqueeze(2) == keyframes.unsqueeze(1), dim=-1).any(
            dim=1
        )
        assert torch.all(matches)

    def test_rejects_sample_count_smaller_than_keyframe_count(self):
        start = torch.zeros(2, 6)
        waypoints = torch.stack(
            [torch.ones(2, 6), torch.full((2, 6), 3.0)],
            dim=1,
        )

        with pytest.raises(ValueError, match="at least the number of keyframes"):
            interpolate_joint_trajectory(start, waypoints, n_waypoints=2)
