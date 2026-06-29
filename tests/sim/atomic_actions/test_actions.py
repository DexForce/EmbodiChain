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

"""Tests for the concrete atomic action classes."""

from __future__ import annotations

import pytest
import torch
from unittest.mock import Mock, patch

from embodichain.lab.sim.atomic_actions.affordance import (
    AntipodalAffordance,
)
from embodichain.lab.sim.atomic_actions.core import (
    ActionResult,
    CoordinatedPlacementTarget,
    GraspTarget,
    HeldObjectState,
    HeldObjectPoseTarget,
    JointPositionTarget,
    NamedJointPositionTarget,
    ObjectSemantics,
    EndEffectorPoseTarget,
    WorldState,
)
from embodichain.lab.sim.atomic_actions.actions import (
    CoordinatedPlacement,
    CoordinatedPlacementCfg,
    MoveEndEffector,
    MoveEndEffectorCfg,
    MoveJoints,
    MoveJointsCfg,
    MoveHeldObject,
    MoveHeldObjectCfg,
    PickUp,
    PickUpCfg,
    Place,
    PlaceCfg,
    Press,
    PressCfg,
)

NUM_ENVS = 2
ARM_DOF = 6
HAND_DOF = 2
TOTAL_DOF = ARM_DOF + HAND_DOF
DUAL_TOTAL_DOF = ARM_DOF * 2 + 2


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


def _make_mock_dual_arm_robot():
    robot = Mock()
    robot.device = torch.device("cpu")
    robot.dof = DUAL_TOTAL_DOF

    def get_qpos(name=None):
        if name in ("left_arm", "right_arm"):
            return torch.zeros(NUM_ENVS, ARM_DOF)
        if name in ("left_hand", "right_hand"):
            return torch.zeros(NUM_ENVS, 1)
        if name == "dual_arm":
            return torch.zeros(NUM_ENVS, ARM_DOF * 2)
        return torch.zeros(NUM_ENVS, DUAL_TOTAL_DOF)

    robot.get_qpos = get_qpos

    def get_joint_ids(name=None):
        if name == "left_arm":
            return list(range(ARM_DOF))
        if name == "right_arm":
            return list(range(ARM_DOF, ARM_DOF * 2))
        if name == "dual_arm":
            return list(range(ARM_DOF * 2))
        if name == "left_hand":
            return [ARM_DOF * 2]
        if name == "right_hand":
            return [ARM_DOF * 2 + 1]
        return list(range(DUAL_TOTAL_DOF))

    robot.get_joint_ids = get_joint_ids

    def compute_ik(pose=None, qpos_seed=None, name=None, joint_seed=None):
        seed = joint_seed if joint_seed is not None else qpos_seed
        if seed is None:
            seed = torch.zeros(NUM_ENVS, ARM_DOF)
        return torch.ones(NUM_ENVS, dtype=torch.bool), seed.clone()

    robot.compute_ik = compute_ik
    return robot


def _make_mock_dual_arm_motion_generator():
    mg = Mock()
    mg.robot = _make_mock_dual_arm_robot()
    mg.device = torch.device("cpu")
    return mg


def _hand_open():
    return torch.zeros(HAND_DOF, dtype=torch.float32)


def _hand_close():
    return torch.full((HAND_DOF,), 0.025, dtype=torch.float32)


# ---------------------------------------------------------------------------
# MoveEndEffector
# ---------------------------------------------------------------------------


class TestMoveEndEffectorAction:
    def setup_method(self):
        self.mg = _make_mock_motion_generator()

    def test_target_type_is_pose_target(self):
        assert MoveEndEffector.TargetType is EndEffectorPoseTarget

    def test_default_name_is_explicit(self):
        assert MoveEndEffectorCfg().name == "move_end_effector"

    def test_execute_returns_full_dof_trajectory(self):
        action = MoveEndEffector(self.mg, MoveEndEffectorCfg(sample_interval=10))
        with patch(
            "embodichain.lab.sim.atomic_actions.trajectory.interpolate_with_distance",
            return_value=torch.zeros(NUM_ENVS, 10, ARM_DOF),
        ):
            state = WorldState(last_qpos=torch.zeros(NUM_ENVS, TOTAL_DOF))
            result = action.execute(EndEffectorPoseTarget(xpos=torch.eye(4)), state)
        assert isinstance(result, ActionResult)
        assert result.success is True
        assert result.trajectory.shape == (NUM_ENVS, 10, TOTAL_DOF)
        # MoveEndEffector preserves held_object.
        assert result.next_state.held_object is None


# ---------------------------------------------------------------------------
# MoveJoints
# ---------------------------------------------------------------------------


class TestMoveJointsAction:
    def setup_method(self):
        self.mg = _make_mock_motion_generator()

    def test_target_type_accepts_explicit_and_named_joint_targets(self):
        assert MoveJoints.TargetType == (JointPositionTarget, NamedJointPositionTarget)

    def test_default_name_is_explicit(self):
        assert MoveJointsCfg().name == "move_joints"

    def test_execute_with_explicit_qpos_returns_full_dof_trajectory(self):
        action = MoveJoints(self.mg, MoveJointsCfg(sample_interval=10))
        target_qpos = torch.full((ARM_DOF,), 0.5)
        hand_qpos = torch.full((NUM_ENVS, HAND_DOF), 0.25)
        last_qpos = torch.cat([torch.zeros(NUM_ENVS, ARM_DOF), hand_qpos], dim=1)
        sem = ObjectSemantics(
            affordance=AntipodalAffordance(), geometry={}, label="mug"
        )
        held = HeldObjectState(
            semantics=sem,
            object_to_eef=torch.eye(4).unsqueeze(0).repeat(NUM_ENVS, 1, 1),
            grasp_xpos=torch.eye(4).unsqueeze(0).repeat(NUM_ENVS, 1, 1),
        )

        def interpolate(trajectory, interp_num, device):
            assert trajectory.shape == (NUM_ENVS, 2, ARM_DOF)
            return trajectory[:, -1:, :].repeat(1, interp_num, 1)

        with patch(
            "embodichain.lab.sim.atomic_actions.trajectory.interpolate_with_distance",
            side_effect=interpolate,
        ):
            result = action.execute(
                JointPositionTarget(qpos=target_qpos),
                WorldState(last_qpos=last_qpos, held_object=held),
            )

        assert result.success is True
        assert result.trajectory.shape == (NUM_ENVS, 10, TOTAL_DOF)
        assert torch.allclose(result.trajectory[:, -1, :ARM_DOF], target_qpos)
        assert torch.allclose(result.trajectory[:, -1, ARM_DOF:], hand_qpos)
        assert result.next_state.held_object is held

    def test_execute_with_named_qpos_resolves_cfg_target(self):
        action = MoveJoints(
            self.mg,
            MoveJointsCfg(
                sample_interval=8,
                named_joint_positions={"home": torch.full((ARM_DOF,), 0.2)},
            ),
        )
        with patch(
            "embodichain.lab.sim.atomic_actions.trajectory.interpolate_with_distance",
            side_effect=lambda trajectory, interp_num, device: trajectory[
                :, -1:, :
            ].repeat(1, interp_num, 1),
        ):
            result = action.execute(
                NamedJointPositionTarget(name="home"),
                WorldState(last_qpos=torch.zeros(NUM_ENVS, TOTAL_DOF)),
            )
        assert result.success is True
        assert torch.allclose(
            result.next_state.last_qpos[:, :ARM_DOF],
            torch.full((NUM_ENVS, ARM_DOF), 0.2),
        )

    def test_unknown_named_qpos_raises(self):
        action = MoveJoints(self.mg, MoveJointsCfg())
        with pytest.raises(KeyError, match="missing"):
            action.execute(
                NamedJointPositionTarget(name="missing"),
                WorldState(last_qpos=torch.zeros(NUM_ENVS, TOTAL_DOF)),
            )


# ---------------------------------------------------------------------------
# PickUp
# ---------------------------------------------------------------------------


class TestPickUpAction:
    def setup_method(self):
        self.mg = _make_mock_motion_generator()

    def test_target_type_is_grasp_target(self):
        assert PickUp.TargetType is GraspTarget

    def test_execute_populates_held_object_state(self):
        cfg = PickUpCfg(
            hand_open_qpos=_hand_open(),
            hand_close_qpos=_hand_close(),
            sample_interval=20,
            hand_interp_steps=4,
        )
        action = PickUp(self.mg, cfg)

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
# MoveHeldObject
# ---------------------------------------------------------------------------


class TestMoveHeldObjectAction:
    def setup_method(self):
        self.mg = _make_mock_motion_generator()

    def test_target_type_is_held_object_target(self):
        assert MoveHeldObject.TargetType is HeldObjectPoseTarget

    def test_default_name_is_explicit(self):
        assert (
            MoveHeldObjectCfg(hand_close_qpos=_hand_close()).name == "move_held_object"
        )

    def test_requires_held_object_in_state(self):
        cfg = MoveHeldObjectCfg(
            hand_close_qpos=_hand_close(),
            sample_interval=10,
        )
        action = MoveHeldObject(self.mg, cfg)
        state = WorldState(last_qpos=torch.zeros(NUM_ENVS, TOTAL_DOF))
        with pytest.raises(Exception):
            action.execute(HeldObjectPoseTarget(object_target_pose=torch.eye(4)), state)

    def test_preserves_held_object(self):
        cfg = MoveHeldObjectCfg(
            hand_close_qpos=_hand_close(),
            sample_interval=10,
        )
        action = MoveHeldObject(self.mg, cfg)
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
                HeldObjectPoseTarget(object_target_pose=torch.eye(4)), state
            )
        assert result.success is True
        assert result.trajectory.shape == (NUM_ENVS, 10, TOTAL_DOF)
        assert result.next_state.held_object is held


# ---------------------------------------------------------------------------
# Place
# ---------------------------------------------------------------------------


class TestPlaceAction:
    def setup_method(self):
        self.mg = _make_mock_motion_generator()

    def test_target_type_is_pose_target(self):
        assert Place.TargetType is EndEffectorPoseTarget

    def test_execute_clears_held_object(self):
        cfg = PlaceCfg(
            hand_open_qpos=_hand_open(),
            hand_close_qpos=_hand_close(),
            sample_interval=20,
            hand_interp_steps=4,
        )
        action = Place(self.mg, cfg)
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
            result = action.execute(EndEffectorPoseTarget(xpos=torch.eye(4)), state)
        assert result.success is True
        assert result.trajectory.shape[2] == TOTAL_DOF
        assert result.next_state.held_object is None


# ---------------------------------------------------------------------------
# Press
# ---------------------------------------------------------------------------


class TestPressAction:
    def setup_method(self):
        self.mg = _make_mock_motion_generator()

    def test_target_type_is_pose_target(self):
        assert Press.TargetType is EndEffectorPoseTarget

    def test_default_name_is_explicit(self):
        assert PressCfg(hand_close_qpos=_hand_close()).name == "press"

    def test_execute_closes_hand_and_preserves_held_object(self):
        cfg = PressCfg(
            hand_close_qpos=_hand_close(),
            sample_interval=12,
            hand_interp_steps=4,
        )
        action = Press(self.mg, cfg)
        sem = ObjectSemantics(
            affordance=AntipodalAffordance(), geometry={}, label="mug"
        )
        held = HeldObjectState(
            semantics=sem,
            object_to_eef=torch.eye(4).unsqueeze(0).repeat(NUM_ENVS, 1, 1),
            grasp_xpos=torch.eye(4).unsqueeze(0).repeat(NUM_ENVS, 1, 1),
        )
        start_hand_qpos = torch.full((NUM_ENVS, HAND_DOF), 0.01)
        last_qpos = torch.cat([torch.zeros(NUM_ENVS, ARM_DOF), start_hand_qpos], dim=1)
        state = WorldState(last_qpos=last_qpos, held_object=held)

        def interpolate(trajectory, interp_num, device):
            return trajectory[:, -1:, :].repeat(1, interp_num, 1)

        with patch(
            "embodichain.lab.sim.atomic_actions.trajectory.interpolate_with_distance",
            side_effect=interpolate,
        ):
            result = action.execute(EndEffectorPoseTarget(xpos=torch.eye(4)), state)

        assert result.success is True
        assert result.trajectory.shape == (NUM_ENVS, 12, TOTAL_DOF)
        expected_hand_qpos = _hand_close().unsqueeze(0).repeat(NUM_ENVS, 1)
        assert torch.allclose(result.trajectory[:, -1, ARM_DOF:], expected_hand_qpos)
        assert torch.allclose(
            result.next_state.last_qpos[:, :ARM_DOF],
            last_qpos[:, :ARM_DOF],
        )
        assert result.next_state.held_object is held


# ---------------------------------------------------------------------------
# CoordinatedPlacement
# ---------------------------------------------------------------------------


class TestCoordinatedPlacementAction:
    def setup_method(self):
        self.mg = _make_mock_dual_arm_motion_generator()
        self.cfg = CoordinatedPlacementCfg(
            placing_hand_open_qpos=torch.tensor([0.0]),
            placing_hand_close_qpos=torch.tensor([0.03]),
            support_hand_close_qpos=torch.tensor([0.025]),
            sample_interval=30,
            hand_interp_steps=4,
            hold_steps=3,
            retreat_steps=5,
            lift_height=0.08,
        )
        self.action = CoordinatedPlacement(self.mg, cfg=self.cfg)

    def _make_target(self) -> CoordinatedPlacementTarget:
        placing_pose = torch.eye(4)
        placing_pose[0, 3] = 0.2
        support_pose = torch.eye(4)
        support_pose[0, 3] = 0.2
        support_pose[2, 3] = -0.05

        placing_object_to_eef = torch.eye(4)
        placing_object_to_eef[2, 3] = 0.12
        support_object_to_eef = torch.eye(4)
        support_object_to_eef[2, 3] = 0.10

        placing_semantics = ObjectSemantics(
            affordance=AntipodalAffordance(), geometry={}, label="placing"
        )
        support_semantics = ObjectSemantics(
            affordance=AntipodalAffordance(), geometry={}, label="support"
        )
        return CoordinatedPlacementTarget(
            placing_object_target_pose=placing_pose,
            support_object_target_pose=support_pose,
            placing_held_object=HeldObjectState(
                semantics=placing_semantics,
                object_to_eef=placing_object_to_eef,
                grasp_xpos=torch.eye(4),
            ),
            support_held_object=HeldObjectState(
                semantics=support_semantics,
                object_to_eef=support_object_to_eef,
                grasp_xpos=torch.eye(4),
            ),
        )

    def test_target_type_is_coordinated_placement_target(self):
        assert CoordinatedPlacement.TargetType is CoordinatedPlacementTarget

    def test_init_sets_dual_arm_and_hand_joint_ids(self):
        assert self.action.dual_arm_joint_ids == list(range(ARM_DOF * 2))
        assert self.action.placing_arm_joint_ids == list(range(ARM_DOF))
        assert self.action.support_arm_joint_ids == list(range(ARM_DOF, ARM_DOF * 2))
        assert self.action.placing_hand_joint_ids == [ARM_DOF * 2]
        assert self.action.support_hand_joint_ids == [ARM_DOF * 2 + 1]
        assert self.action.joint_ids == list(range(DUAL_TOTAL_DOF))

    def test_resolve_target_composes_object_and_tcp_poses(self):
        target = self._make_target()
        placing_xpos, support_xpos, release, held_state = self.action._resolve_target(
            target
        )
        assert placing_xpos.shape == (NUM_ENVS, 4, 4)
        assert support_xpos.shape == (NUM_ENVS, 4, 4)
        assert placing_xpos[0, 2, 3].item() == pytest.approx(0.12)
        assert support_xpos[0, 2, 3].item() == pytest.approx(0.05)
        assert release is True
        assert held_state.semantics is target.support_held_object.semantics
        assert held_state.object_to_eef.shape == (NUM_ENVS, 4, 4)
        assert held_state.grasp_xpos.shape == (NUM_ENVS, 4, 4)
        assert torch.allclose(
            held_state.object_to_eef,
            target.support_held_object.object_to_eef.unsqueeze(0).repeat(
                NUM_ENVS, 1, 1
            ),
        )

    def test_segment_lengths_sum_to_sample_interval(self):
        segments = self.action._compute_segment_lengths(self.cfg.release)
        assert sum(segments.values()) == self.cfg.sample_interval
        assert segments["approach"] >= 2
        assert segments["release"] == self.cfg.hand_interp_steps
        assert segments["retreat"] == self.cfg.retreat_steps

    def test_execute_returns_full_dof_and_final_hand_states(self):
        target = self._make_target()
        state = WorldState(last_qpos=torch.zeros(NUM_ENVS, DUAL_TOTAL_DOF))

        def interpolate(trajectory, interp_num, device):
            weights = torch.linspace(
                0.0,
                1.0,
                steps=interp_num,
                dtype=trajectory.dtype,
                device=trajectory.device,
            )
            return torch.lerp(
                trajectory[:, :1],
                trajectory[:, -1:],
                weights.view(1, -1, 1),
            )

        with patch(
            "embodichain.lab.sim.atomic_actions.trajectory.interpolate_with_distance",
            side_effect=interpolate,
        ):
            result = self.action.execute(target, state)

        assert result.success is True
        assert result.trajectory.shape == (
            NUM_ENVS,
            self.cfg.sample_interval,
            DUAL_TOTAL_DOF,
        )
        assert result.trajectory[0, -1, ARM_DOF * 2].item() == pytest.approx(0.0)
        assert result.trajectory[0, -1, ARM_DOF * 2 + 1].item() == pytest.approx(0.025)
        assert result.next_state.held_object is not None
        assert (
            result.next_state.held_object.semantics
            is target.support_held_object.semantics
        )
        assert result.next_state.held_object.object_to_eef.shape == (NUM_ENVS, 4, 4)
        assert result.next_state.held_object.grasp_xpos.shape == (NUM_ENVS, 4, 4)
