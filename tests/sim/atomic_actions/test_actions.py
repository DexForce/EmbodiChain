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

"""Tests for atomic action implementations (MoveAction, PickUpAction, PlaceAction)."""

from __future__ import annotations

import pytest
import torch
from unittest.mock import MagicMock, Mock

from embodichain.lab.sim.atomic_actions.core import (
    ActionCfg,
    Affordance,
    ObjectSemantics,
)
from embodichain.lab.sim.atomic_actions.actions import (
    MoveAction,
    MoveActionCfg,
    PickUpAction,
    PickUpActionCfg,
    PlaceAction,
    PlaceActionCfg,
)

# ---------------------------------------------------------------------------
# Mock Helpers
# ---------------------------------------------------------------------------

NUM_ENVS = 2  # number of parallel environments used in tests
ARM_DOF = 6  # typical arm joint count
HAND_DOF = 2  # typical hand joint count
TOTAL_DOF = ARM_DOF + HAND_DOF


def _make_mock_robot(
    num_envs: int = NUM_ENVS,
    arm_dof: int = ARM_DOF,
    hand_dof: int = HAND_DOF,
) -> Mock:
    """Create a mock Robot with arm and hand control parts."""
    robot = Mock()
    robot.device = torch.device("cpu")
    robot.dof = arm_dof + hand_dof

    def get_qpos(name=None):
        if name == "arm":
            return torch.zeros(num_envs, arm_dof)
        if name == "hand":
            return torch.zeros(num_envs, hand_dof)
        # Full qpos
        return torch.zeros(num_envs, arm_dof + hand_dof)

    robot.get_qpos = get_qpos

    def get_joint_ids(name=None):
        if name == "arm":
            return list(range(arm_dof))
        if name == "hand":
            return list(range(arm_dof, arm_dof + hand_dof))
        return list(range(arm_dof + hand_dof))

    robot.get_joint_ids = get_joint_ids

    # compute_ik: return success and identity-like qpos
    def compute_ik(pose=None, qpos_seed=None, name=None, joint_seed=None):
        seed = joint_seed if joint_seed is not None else qpos_seed
        if seed is None:
            seed = torch.zeros(num_envs, arm_dof)
        success = torch.ones(num_envs, dtype=torch.bool)
        return success, seed.clone()

    robot.compute_ik = compute_ik

    # compute_fk: return identity-like poses
    def compute_fk(qpos=None, name=None, to_matrix=True):
        n = qpos.shape[0] if qpos is not None else num_envs
        poses = torch.eye(4).unsqueeze(0).repeat(n, 1, 1)
        return poses

    robot.compute_fk = compute_fk

    return robot


def _make_mock_motion_generator(robot: Mock | None = None) -> Mock:
    """Create a mock MotionGenerator."""
    mg = Mock()
    mg.robot = robot or _make_mock_robot()
    mg.device = mg.robot.device
    return mg


# ---------------------------------------------------------------------------
# MoveAction
# ---------------------------------------------------------------------------


class TestMoveActionHelpers:
    """Tests for MoveAction helper methods that don't need simulation."""

    def setup_method(self):
        self.robot = _make_mock_robot()
        self.mg = _make_mock_motion_generator(self.robot)
        self.cfg = MoveActionCfg(sample_interval=50)
        self.action = MoveAction(self.mg, cfg=self.cfg)

    def test_init_sets_attributes(self):
        assert self.action.n_envs == NUM_ENVS
        assert self.action.dof == ARM_DOF
        assert self.action.device == torch.device("cpu")

    def test_resolve_pose_target_from_4x4(self):
        target = torch.eye(4)
        is_success, result = self.action._resolve_pose_target(
            target, action_name="TestAction"
        )
        assert is_success is True
        assert result.shape == (NUM_ENVS, 4, 4)
        # Single pose should be repeated for all envs
        for i in range(NUM_ENVS):
            assert torch.equal(result[i], torch.eye(4))

    def test_resolve_pose_target_from_batched(self):
        target = torch.eye(4).unsqueeze(0).repeat(NUM_ENVS, 1, 1)
        target[:, 2, 3] = 0.5  # offset z for each env
        is_success, result = self.action._resolve_pose_target(
            target, action_name="TestAction"
        )
        assert is_success is True
        assert result.shape == (NUM_ENVS, 4, 4)
        for i in range(NUM_ENVS):
            assert result[i, 2, 3].item() == pytest.approx(0.5)

    def test_resolve_start_qpos_defaults_to_current(self):
        result = self.action._resolve_start_qpos(None)
        assert result.shape == (NUM_ENVS, ARM_DOF)

    def test_resolve_start_qpos_broadcasts_single(self):
        single = torch.ones(ARM_DOF)
        result = self.action._resolve_start_qpos(single)
        assert result.shape == (NUM_ENVS, ARM_DOF)
        for i in range(NUM_ENVS):
            assert torch.equal(result[i], single)

    def test_compute_three_phase_waypoints_sums_to_sample_interval(self):
        hand_interp_steps = 5
        first, second, third = self.action._compute_three_phase_waypoints(
            hand_interp_steps,
            first_phase_name="approach",
            third_phase_name="lift",
        )
        assert first + second + third == self.cfg.sample_interval
        assert first >= 2
        assert third >= 2

    def test_interpolate_hand_qpos_shape(self):
        n_waypoints = 10
        start = torch.zeros(HAND_DOF)
        end = torch.ones(HAND_DOF)
        result = self.action._interpolate_hand_qpos(start, end, n_waypoints)
        assert result.shape == (n_waypoints, HAND_DOF)
        # First and last should match endpoints
        assert torch.allclose(result[0], start)
        assert torch.allclose(result[-1], end)

    def test_interpolate_hand_qpos_linear(self):
        """Verify linear interpolation between two hand configs."""
        n_waypoints = 3
        start = torch.tensor([0.0, 0.0])
        end = torch.tensor([1.0, 1.0])
        result = self.action._interpolate_hand_qpos(start, end, n_waypoints)
        expected_mid = torch.tensor([0.5, 0.5])
        assert torch.allclose(result[1], expected_mid, atol=1e-6)


# ---------------------------------------------------------------------------
# PickUpAction
# ---------------------------------------------------------------------------


class TestPickUpActionInit:
    """Tests for PickUpAction initialization and config validation."""

    def setup_method(self):
        self.robot = _make_mock_robot()
        self.mg = _make_mock_motion_generator(self.robot)

    def _make_cfg(self, **overrides):
        defaults = dict(
            hand_open_qpos=torch.tensor([0.0, 0.0]),
            hand_close_qpos=torch.tensor([0.025, 0.025]),
            control_part="arm",
            hand_control_part="hand",
            pre_grasp_distance=0.15,
            lift_height=0.15,
            approach_direction=torch.tensor([0.0, 0.0, -1.0]),
        )
        defaults.update(overrides)
        return PickUpActionCfg(**defaults)

    def test_init_sets_hand_joint_ids(self):
        cfg = self._make_cfg()
        action = PickUpAction(self.mg, cfg=cfg)
        assert action.hand_joint_ids == list(range(ARM_DOF, ARM_DOF + HAND_DOF))
        assert action.joint_ids == list(range(ARM_DOF)) + list(
            range(ARM_DOF, ARM_DOF + HAND_DOF)
        )
        assert action.dof == TOTAL_DOF


# ---------------------------------------------------------------------------
# PlaceAction
# ---------------------------------------------------------------------------


class TestPlaceActionInit:
    """Tests for PlaceAction initialization."""

    def setup_method(self):
        self.robot = _make_mock_robot()
        self.mg = _make_mock_motion_generator(self.robot)

    def _make_cfg(self, **overrides):
        defaults = dict(
            hand_open_qpos=torch.tensor([0.0, 0.0]),
            hand_close_qpos=torch.tensor([0.025, 0.025]),
            control_part="arm",
            hand_control_part="hand",
            lift_height=0.15,
        )
        defaults.update(overrides)
        return PlaceActionCfg(**defaults)

    def test_init_sets_hand_joint_ids(self):
        cfg = self._make_cfg()
        action = PlaceAction(self.mg, cfg=cfg)
        assert action.hand_joint_ids == list(range(ARM_DOF, ARM_DOF + HAND_DOF))
        assert action.dof == TOTAL_DOF


# ---------------------------------------------------------------------------
# AtomicAction._apply_offset
# ---------------------------------------------------------------------------


class TestAtomicActionApplyOffset:
    """Tests for the shared _apply_offset method inherited from AtomicAction."""

    def setup_method(self):
        self.robot = _make_mock_robot()
        self.mg = _make_mock_motion_generator(self.robot)
        self.cfg = MoveActionCfg()
        self.action = MoveAction(self.mg, cfg=self.cfg)

    def test_apply_offset_batched(self):
        # [N, 4, 4] poses, [N, 3] offsets
        poses = torch.eye(4).unsqueeze(0).repeat(3, 1, 1)
        offsets = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        result = self.action._apply_offset(poses, offsets)
        assert result.shape == (3, 4, 4)
        assert result[0, :3, 3].tolist() == pytest.approx([1.0, 0.0, 0.0])
        assert result[1, :3, 3].tolist() == pytest.approx([0.0, 1.0, 0.0])
        assert result[2, :3, 3].tolist() == pytest.approx([0.0, 0.0, 1.0])

    def test_apply_offset_broadcasts_single_offset(self):
        # [N, 4, 4] poses, [3] single offset broadcast to all
        poses = torch.eye(4).unsqueeze(0).repeat(2, 1, 1)
        offset = torch.tensor([0.1, 0.2, 0.3])
        result = self.action._apply_offset(poses, offset)
        assert result.shape == (2, 4, 4)
        for i in range(2):
            assert result[i, :3, 3].tolist() == pytest.approx([0.1, 0.2, 0.3])

    def test_apply_offset_preserves_rotation(self):
        """Offset only affects translation; rotation part stays unchanged."""
        poses = torch.eye(4).unsqueeze(0).repeat(1, 1, 1)
        # Set a non-trivial rotation
        poses[0, 0, 1] = -1.0
        poses[0, 1, 0] = 1.0
        offset = torch.tensor([1.0, 2.0, 3.0])
        result = self.action._apply_offset(poses, offset)
        # Rotation block should be unchanged
        assert torch.equal(result[0, :3, :3], poses[0, :3, :3])


if __name__ == "__main__":
    # For visual debugging
    test = TestMoveActionHelpers()
    test.setup_method()
    test.test_compute_three_phase_waypoints_sums_to_sample_interval()
