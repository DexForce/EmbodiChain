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

"""Tests for atomic_actions.trajectory.TrajectoryBuilder."""

from __future__ import annotations

import pytest
import torch
from unittest.mock import Mock, patch

from embodichain.lab.sim.atomic_actions.trajectory import TrajectoryBuilder


def _make_mock_motion_generator(num_envs: int = 2, arm_dof: int = 6) -> Mock:
    robot = Mock()
    robot.device = torch.device("cpu")
    robot.dof = arm_dof

    def get_qpos(name=None):
        return torch.zeros(num_envs, arm_dof)

    robot.get_qpos = get_qpos

    def compute_ik(pose=None, qpos_seed=None, name=None, joint_seed=None):
        seed = joint_seed if joint_seed is not None else qpos_seed
        if seed is None:
            seed = torch.zeros(num_envs, arm_dof)
        return torch.ones(num_envs, dtype=torch.bool), seed.clone()

    robot.compute_ik = compute_ik

    def compute_fk(qpos=None, name=None, to_matrix=True):
        n = qpos.shape[0] if qpos is not None else num_envs
        return torch.eye(4).unsqueeze(0).repeat(n, 1, 1)

    robot.compute_fk = compute_fk

    mg = Mock()
    mg.robot = robot
    mg.device = torch.device("cpu")
    return mg


class TestAllEnvsSuccess:
    def setup_method(self):
        self.builder = TrajectoryBuilder(_make_mock_motion_generator())

    def test_python_bool_true(self):
        assert self.builder.all_envs_success(True) is True

    def test_python_bool_false(self):
        assert self.builder.all_envs_success(False) is False

    def test_tensor_all_true(self):
        assert self.builder.all_envs_success(torch.tensor([True, True])) is True

    def test_tensor_any_false(self):
        assert self.builder.all_envs_success(torch.tensor([True, False])) is False


class TestResolvePoseTarget:
    def setup_method(self):
        self.builder = TrajectoryBuilder(_make_mock_motion_generator())

    def test_unbatched_pose_broadcasts(self):
        pose = torch.eye(4)
        out = self.builder.resolve_pose_target(pose, n_envs=2)
        assert out.shape == (2, 4, 4)

    def test_batched_pose_passes_through(self):
        pose = torch.eye(4).unsqueeze(0).repeat(2, 1, 1)
        out = self.builder.resolve_pose_target(pose, n_envs=2)
        assert torch.equal(out, pose)

    def test_pose_converts_to_builder_dtype_and_device(self):
        pose = torch.eye(4, dtype=torch.float64)
        out = self.builder.resolve_pose_target(pose, n_envs=2)
        assert out.dtype == torch.float32
        assert out.device == self.builder.device

    def test_wrong_shape_raises(self):
        with pytest.raises(Exception):
            self.builder.resolve_pose_target(torch.eye(3), n_envs=2)


class TestResolveJointTarget:
    def setup_method(self):
        self.builder = TrajectoryBuilder(_make_mock_motion_generator())

    def test_unbatched_qpos_broadcasts(self):
        qpos = torch.arange(6, dtype=torch.float32)
        out = self.builder.resolve_joint_target(
            qpos, n_envs=2, joint_dof=6, control_part="arm"
        )
        assert out.shape == (2, 6)
        assert torch.allclose(out[0], qpos)
        assert torch.allclose(out[1], qpos)

    def test_batched_qpos_passes_through(self):
        qpos = torch.arange(12, dtype=torch.float32).reshape(2, 6)
        out = self.builder.resolve_joint_target(
            qpos, n_envs=2, joint_dof=6, control_part="arm"
        )
        assert torch.equal(out, qpos)

    def test_wrong_shape_raises(self):
        with pytest.raises(Exception):
            self.builder.resolve_joint_target(
                torch.zeros(5), n_envs=2, joint_dof=6, control_part="arm"
            )


class TestSplitThreePhase:
    def setup_method(self):
        self.builder = TrajectoryBuilder(_make_mock_motion_generator())

    def test_default_ratio(self):
        a, b, c = self.builder.split_three_phase(80, 5)
        assert b == 5
        assert a + b + c == 80
        # First-phase ratio is 0.6 of remaining waypoints
        assert a == int(round((80 - 5) * 0.6))

    def test_raises_when_first_phase_too_small(self):
        with pytest.raises(Exception):
            self.builder.split_three_phase(6, 5)


class TestApplyLocalOffset:
    def setup_method(self):
        self.builder = TrajectoryBuilder(_make_mock_motion_generator())

    def test_offset_adds_to_translation(self):
        pose = torch.eye(4).unsqueeze(0).repeat(2, 1, 1)
        offset = torch.tensor([0.0, 0.0, 0.1])
        out = self.builder.apply_local_offset(pose, offset)
        assert torch.allclose(out[:, :3, 3], torch.tensor([0.0, 0.0, 0.1]).expand(2, 3))

    def test_batched_offset(self):
        pose = torch.eye(4).unsqueeze(0).repeat(2, 1, 1)
        offset = torch.tensor([[0.1, 0.0, 0.0], [0.0, 0.2, 0.0]])
        out = self.builder.apply_local_offset(pose, offset)
        assert torch.allclose(out[0, :3, 3], torch.tensor([0.1, 0.0, 0.0]))
        assert torch.allclose(out[1, :3, 3], torch.tensor([0.0, 0.2, 0.0]))

    def test_incompatible_offset_batch_raises(self):
        pose = torch.eye(4).unsqueeze(0).repeat(2, 1, 1)
        offset = torch.zeros(3, 3)
        with pytest.raises(ValueError, match="offset batch size"):
            self.builder.apply_local_offset(pose, offset)


class TestExpandHandQpos:
    def setup_method(self):
        self.builder = TrajectoryBuilder(_make_mock_motion_generator())

    def test_unbatched_expanded(self):
        q = torch.tensor([0.1, 0.2])
        out = self.builder.expand_hand_qpos(q, n_envs=3, hand_dof=2)
        assert out.shape == (3, 2)
        assert torch.allclose(out[0], q)

    def test_batched_passes_through(self):
        q = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
        out = self.builder.expand_hand_qpos(q, n_envs=2, hand_dof=2)
        assert torch.equal(out, q)


class TestInterpolateHandQpos:
    def setup_method(self):
        self.builder = TrajectoryBuilder(_make_mock_motion_generator())

    def test_endpoints_match(self):
        a = torch.tensor([[0.0, 0.0]])
        b = torch.tensor([[1.0, 1.0]])
        out = self.builder.interpolate_hand_qpos(a, b, n_waypoints=5)
        assert torch.allclose(out[:, 0], a)
        assert torch.allclose(out[:, -1], b)


class TestPlanJointTraj:
    def setup_method(self):
        self.builder = TrajectoryBuilder(_make_mock_motion_generator())

    def test_interpolates_start_to_target(self):
        start = torch.zeros(2, 6)
        target = torch.ones(2, 6)
        expected = torch.ones(2, 5, 6)
        with patch(
            "embodichain.lab.sim.atomic_actions.trajectory.interpolate_with_distance",
            return_value=expected,
        ) as interpolate:
            out = self.builder.plan_joint_traj(start, target, n_waypoints=5)

        assert out is expected
        _, kwargs = interpolate.call_args
        assert kwargs["interp_num"] == 5
        assert torch.equal(kwargs["trajectory"][:, 0, :], start)
        assert torch.equal(kwargs["trajectory"][:, 1, :], target)


class TestIkSolve:
    def test_uses_first_env_seed_for_single_pose(self):
        mg = _make_mock_motion_generator(num_envs=3, arm_dof=6)
        builder = TrajectoryBuilder(mg)

        def compute_ik(pose=None, name=None, joint_seed=None, env_ids=None):
            return torch.ones(1, dtype=torch.bool), joint_seed + 1.0

        mg.robot.compute_ik = Mock(side_effect=compute_ik)

        out = builder.ik_solve(torch.eye(4), control_part="arm")

        _, kwargs = mg.robot.compute_ik.call_args
        assert kwargs["joint_seed"].shape == (1, 6)
        assert kwargs["env_ids"] == [0]
        assert out.shape == (6,)
