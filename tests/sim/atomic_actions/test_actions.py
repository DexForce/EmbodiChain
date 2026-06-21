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

"""Tests for the four concrete atomic action classes."""

from __future__ import annotations

import pytest
import torch
from unittest.mock import Mock, patch

from embodichain.lab.sim.atomic_actions.affordance import (
    AntipodalAffordance,
)
from embodichain.lab.sim.atomic_actions.core import (
    ActionResult,
    GraspTarget,
    HeldObjectState,
    HeldObjectTarget,
    ObjectSemantics,
    PoseTarget,
    WorldState,
)
from embodichain.lab.sim.atomic_actions.actions import (
    MoveAction,
    MoveActionCfg,
    MoveObjectAction,
    MoveObjectActionCfg,
    PickUpAction,
    PickUpActionCfg,
    PlaceAction,
    PlaceActionCfg,
)

NUM_ENVS = 2
ARM_DOF = 6
HAND_DOF = 2
TOTAL_DOF = ARM_DOF + HAND_DOF


def _make_mock_robot():
    robot = Mock()
    robot.device = torch.device("cpu")
    robot.dof = TOTAL_DOF

    def get_qpos(name=None):
        if name == "arm":
            return torch.zeros(NUM_ENVS, ARM_DOF)
        if name == "hand":
            return torch.zeros(NUM_ENVS, HAND_DOF)
        return torch.zeros(NUM_ENVS, TOTAL_DOF)

    robot.get_qpos = get_qpos

    def get_joint_ids(name=None):
        if name == "arm":
            return list(range(ARM_DOF))
        if name == "hand":
            return list(range(ARM_DOF, TOTAL_DOF))
        return list(range(TOTAL_DOF))

    robot.get_joint_ids = get_joint_ids

    def compute_ik(pose=None, qpos_seed=None, name=None, joint_seed=None):
        seed = joint_seed if joint_seed is not None else qpos_seed
        if seed is None:
            seed = torch.zeros(NUM_ENVS, ARM_DOF)
        return torch.ones(NUM_ENVS, dtype=torch.bool), seed.clone()

    robot.compute_ik = compute_ik

    def compute_batch_ik(pose=None, name=None, joint_seed=None):
        if joint_seed is not None:
            return (
                torch.ones(joint_seed.shape[:2], dtype=torch.bool),
                joint_seed.clone(),
            )
        return torch.ones(NUM_ENVS, dtype=torch.bool), torch.zeros(NUM_ENVS, ARM_DOF)

    robot.compute_batch_ik = compute_batch_ik

    def compute_fk(qpos=None, name=None, to_matrix=True):
        n = qpos.shape[0] if qpos is not None else NUM_ENVS
        return torch.eye(4).unsqueeze(0).repeat(n, 1, 1)

    robot.compute_fk = compute_fk
    return robot


def _make_mock_motion_generator():
    mg = Mock()
    mg.robot = _make_mock_robot()
    mg.device = torch.device("cpu")
    return mg


def _hand_open():
    return torch.zeros(HAND_DOF, dtype=torch.float32)


def _hand_close():
    return torch.full((HAND_DOF,), 0.025, dtype=torch.float32)


# ---------------------------------------------------------------------------
# MoveAction
# ---------------------------------------------------------------------------


class TestMoveAction:
    def setup_method(self):
        self.mg = _make_mock_motion_generator()

    def test_target_type_is_pose_target(self):
        assert MoveAction.TargetType is PoseTarget

    def test_execute_returns_full_dof_trajectory(self):
        action = MoveAction(self.mg, MoveActionCfg(sample_interval=10))
        with patch(
            "embodichain.lab.sim.atomic_actions.trajectory.interpolate_with_distance",
            return_value=torch.zeros(NUM_ENVS, 10, ARM_DOF),
        ):
            state = WorldState(last_qpos=torch.zeros(NUM_ENVS, TOTAL_DOF))
            result = action.execute(PoseTarget(xpos=torch.eye(4)), state)
        assert isinstance(result, ActionResult)
        assert result.success is True
        assert result.trajectory.shape == (NUM_ENVS, 10, TOTAL_DOF)
        # Move doesn't touch held_object
        assert result.next_state.held_object is None


# ---------------------------------------------------------------------------
# PickUpAction
# ---------------------------------------------------------------------------


class TestPickUpAction:
    def setup_method(self):
        self.mg = _make_mock_motion_generator()

    def test_target_type_is_grasp_target(self):
        assert PickUpAction.TargetType is GraspTarget

    def test_execute_populates_held_object_state(self):
        cfg = PickUpActionCfg(
            hand_open_qpos=_hand_open(),
            hand_close_qpos=_hand_close(),
            sample_interval=20,
            hand_interp_steps=4,
        )
        action = PickUpAction(self.mg, cfg)

        # Fake affordance that returns a single identity grasp pose.
        affordance = AntipodalAffordance()
        affordance.get_valid_grasp_poses = Mock(
            return_value=[
                (torch.eye(4).unsqueeze(0), torch.tensor([0.5]))
                for _ in range(NUM_ENVS)
            ]
        )

        entity = Mock()
        entity.get_local_pose = Mock(
            return_value=torch.eye(4).unsqueeze(0).repeat(NUM_ENVS, 1, 1)
        )

        sem = ObjectSemantics(
            affordance=affordance,
            geometry={},
            label="mug",
            entity=entity,
        )

        with patch(
            "embodichain.lab.sim.atomic_actions.trajectory.interpolate_with_distance",
            side_effect=lambda trajectory, interp_num, device: torch.zeros(
                NUM_ENVS, interp_num, ARM_DOF
            ),
        ):
            state = WorldState(last_qpos=torch.zeros(NUM_ENVS, TOTAL_DOF))
            result = action.execute(GraspTarget(semantics=sem), state)
        assert result.success is True
        assert result.trajectory.shape[0] == NUM_ENVS
        assert result.trajectory.shape[2] == TOTAL_DOF
        assert isinstance(result.next_state.held_object, HeldObjectState)
        assert result.next_state.held_object.semantics is sem


# ---------------------------------------------------------------------------
# MoveObjectAction
# ---------------------------------------------------------------------------


class TestMoveObjectAction:
    def setup_method(self):
        self.mg = _make_mock_motion_generator()

    def test_target_type_is_held_object_target(self):
        assert MoveObjectAction.TargetType is HeldObjectTarget

    def test_requires_held_object_in_state(self):
        cfg = MoveObjectActionCfg(
            hand_close_qpos=_hand_close(),
            sample_interval=10,
        )
        action = MoveObjectAction(self.mg, cfg)
        state = WorldState(last_qpos=torch.zeros(NUM_ENVS, TOTAL_DOF))
        with pytest.raises(Exception):
            action.execute(HeldObjectTarget(object_target_pose=torch.eye(4)), state)

    def test_preserves_held_object(self):
        cfg = MoveObjectActionCfg(
            hand_close_qpos=_hand_close(),
            sample_interval=10,
        )
        action = MoveObjectAction(self.mg, cfg)
        sem = ObjectSemantics(
            affordance=AntipodalAffordance(), geometry={}, label="mug"
        )
        held = HeldObjectState(
            semantics=sem,
            object_to_eef=torch.eye(4).unsqueeze(0).repeat(NUM_ENVS, 1, 1),
            grasp_xpos=torch.eye(4).unsqueeze(0).repeat(NUM_ENVS, 1, 1),
        )
        state = WorldState(last_qpos=torch.zeros(NUM_ENVS, TOTAL_DOF), held_object=held)
        with patch(
            "embodichain.lab.sim.atomic_actions.trajectory.interpolate_with_distance",
            return_value=torch.zeros(NUM_ENVS, 10, ARM_DOF),
        ):
            result = action.execute(
                HeldObjectTarget(object_target_pose=torch.eye(4)), state
            )
        assert result.success is True
        assert result.trajectory.shape == (NUM_ENVS, 10, TOTAL_DOF)
        assert result.next_state.held_object is held


# ---------------------------------------------------------------------------
# PlaceAction
# ---------------------------------------------------------------------------


class TestPlaceAction:
    def setup_method(self):
        self.mg = _make_mock_motion_generator()

    def test_target_type_is_pose_target(self):
        assert PlaceAction.TargetType is PoseTarget

    def test_execute_clears_held_object(self):
        cfg = PlaceActionCfg(
            hand_open_qpos=_hand_open(),
            hand_close_qpos=_hand_close(),
            sample_interval=20,
            hand_interp_steps=4,
        )
        action = PlaceAction(self.mg, cfg)
        sem = ObjectSemantics(
            affordance=AntipodalAffordance(), geometry={}, label="mug"
        )
        held = HeldObjectState(
            semantics=sem,
            object_to_eef=torch.eye(4).unsqueeze(0).repeat(NUM_ENVS, 1, 1),
            grasp_xpos=torch.eye(4).unsqueeze(0).repeat(NUM_ENVS, 1, 1),
        )
        state = WorldState(last_qpos=torch.zeros(NUM_ENVS, TOTAL_DOF), held_object=held)
        with patch(
            "embodichain.lab.sim.atomic_actions.trajectory.interpolate_with_distance",
            side_effect=lambda trajectory, interp_num, device: torch.zeros(
                NUM_ENVS, interp_num, ARM_DOF
            ),
        ):
            result = action.execute(PoseTarget(xpos=torch.eye(4)), state)
        assert result.success is True
        assert result.trajectory.shape[2] == TOTAL_DOF
        assert result.next_state.held_object is None
