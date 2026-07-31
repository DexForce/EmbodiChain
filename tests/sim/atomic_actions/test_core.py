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

"""Tests for atomic-action target contracts and shared core state."""

from __future__ import annotations

import dataclasses
from typing import get_args

import pytest
import torch

import embodichain.lab.sim.atomic_actions.core as core_module
from embodichain.lab.sim.atomic_actions.affordance import Affordance
from embodichain.lab.sim.atomic_actions import (
    BuiltinTarget,
    CoordinatedPickTarget,
    CoordinatedPickmentTarget,
    CoordinatedPlacementTarget,
    EndEffectorPoseTarget,
    GraspTarget,
    HeldObjectPoseTarget,
    JointPositionTarget,
    NamedJointPositionTarget,
    ObjectActionTarget,
    PlaceTarget,
    PressTarget,
)
from embodichain.lab.sim.atomic_actions.core import (
    ActionTarget,
    ActionCfg,
    ActionResult,
    CoordinatedHeldObjectState,
    HeldObjectState,
    ObjectSemantics,
    WorldState,
)


class TestTypedTargets:
    def test_core_does_not_own_concrete_target_types(self):
        assert not hasattr(core_module, "GraspTarget")

    def test_builtin_target_contains_press_contract(self):
        assert PressTarget in get_args(BuiltinTarget)

    def test_object_action_target_owns_shared_semantics_contract(self):
        semantics = ObjectSemantics(
            affordance=Affordance(),
            geometry={},
            label="shared-object",
        )
        target = ObjectActionTarget(semantics=semantics)
        assert target.semantics is semantics
        assert not hasattr(target, "xpos")

    def test_object_action_target_rejects_non_semantics_value(self):
        with pytest.raises(TypeError, match="semantics"):
            ObjectActionTarget(semantics=object())  # type: ignore[arg-type]

    def test_object_action_target_lives_in_neutral_module(self):
        assert (
            ObjectActionTarget.__module__
            == "embodichain.lab.sim.atomic_actions.targets"
        )

    def test_object_action_target_is_not_a_builtin_executable_contract(self):
        assert ObjectActionTarget not in get_args(BuiltinTarget)

    @pytest.mark.parametrize(
        ("target_type", "owner_module"),
        [
            (
                EndEffectorPoseTarget,
                "embodichain.lab.sim.atomic_actions.primitives.move_end_effector",
            ),
            (
                JointPositionTarget,
                "embodichain.lab.sim.atomic_actions.primitives.move_joints",
            ),
            (
                NamedJointPositionTarget,
                "embodichain.lab.sim.atomic_actions.primitives.move_joints",
            ),
            (GraspTarget, "embodichain.lab.sim.atomic_actions.primitives.pick_up"),
            (
                HeldObjectPoseTarget,
                "embodichain.lab.sim.atomic_actions.primitives.move_held_object",
            ),
            (PlaceTarget, "embodichain.lab.sim.atomic_actions.primitives.place"),
            (PressTarget, "embodichain.lab.sim.atomic_actions.primitives.press"),
            (
                CoordinatedPickTarget,
                "embodichain.lab.sim.atomic_actions.primitives.coordinated_pickment",
            ),
            (
                CoordinatedPlacementTarget,
                "embodichain.lab.sim.atomic_actions.primitives.coordinated_placement",
            ),
        ],
    )
    def test_target_is_defined_by_owning_primitive(
        self,
        target_type: type[ActionTarget],
        owner_module: str,
    ):
        assert target_type.__module__ == owner_module

    def test_pose_target_holds_tensor(self):
        x = torch.eye(4)
        assert EndEffectorPoseTarget(xpos=x).xpos is x

    def test_place_target_can_declare_tcp_symmetry(self):
        target = PlaceTarget(xpos=torch.eye(4), tcp_symmetry="z_roll_180")
        assert target.tcp_symmetry == "z_roll_180"

    def test_place_target_rejects_unknown_tcp_symmetry(self):
        with pytest.raises(ValueError, match="tcp_symmetry"):
            PlaceTarget(
                xpos=torch.eye(4), tcp_symmetry="yaw_90"  # type: ignore[arg-type]
            )

    def test_press_target_rejects_multiple_waypoints(self):
        with pytest.raises(ValueError, match="xpos"):
            PressTarget(xpos=torch.eye(4).reshape(1, 1, 4, 4))

    def test_pose_target_rejects_invalid_shape(self):
        with pytest.raises(ValueError, match="xpos"):
            EndEffectorPoseTarget(xpos=torch.zeros(3, 3))

    def test_pose_targets_use_identity_equality(self):
        first = EndEffectorPoseTarget(xpos=torch.eye(4))
        second = EndEffectorPoseTarget(xpos=torch.eye(4))
        assert first == first
        assert first != second

    def test_pose_target_is_frozen(self):
        t = EndEffectorPoseTarget(xpos=torch.eye(4))
        with pytest.raises(dataclasses.FrozenInstanceError):
            t.xpos = torch.zeros(4, 4)  # type: ignore[misc]

    def test_joint_position_target_holds_qpos(self):
        qpos = torch.zeros(6)
        assert JointPositionTarget(qpos=qpos).qpos is qpos

    def test_joint_position_target_is_frozen(self):
        t = JointPositionTarget(qpos=torch.zeros(6))
        with pytest.raises(dataclasses.FrozenInstanceError):
            t.qpos = torch.ones(6)  # type: ignore[misc]

    def test_named_joint_position_target_holds_name(self):
        assert NamedJointPositionTarget(name="home").name == "home"

    def test_named_joint_position_target_is_frozen(self):
        t = NamedJointPositionTarget(name="home")
        with pytest.raises(dataclasses.FrozenInstanceError):
            t.name = "ready"  # type: ignore[misc]

    def test_grasp_target_holds_semantics(self):
        sem = ObjectSemantics(affordance=Affordance(), geometry={}, label="mug")
        target = GraspTarget(semantics=sem)
        assert target.semantics is sem
        assert isinstance(target, ObjectActionTarget)

    def test_grasp_target_is_frozen(self):
        sem = ObjectSemantics(affordance=Affordance(), geometry={}, label="mug")
        t = GraspTarget(semantics=sem)
        with pytest.raises(dataclasses.FrozenInstanceError):
            t.semantics = ObjectSemantics(  # type: ignore[misc]
                affordance=Affordance(), geometry={}, label="other"
            )

    def test_held_object_target_holds_pose(self):
        x = torch.eye(4)
        assert HeldObjectPoseTarget(object_target_pose=x).object_target_pose is x

    def test_held_object_target_is_frozen(self):
        t = HeldObjectPoseTarget(object_target_pose=torch.eye(4))
        with pytest.raises(dataclasses.FrozenInstanceError):
            t.object_target_pose = torch.zeros(4, 4)  # type: ignore[misc]

    def test_coordinated_pick_target_holds_object_offsets(self):
        sem = ObjectSemantics(affordance=Affordance(), geometry={}, label="pencil")
        target = CoordinatedPickTarget(
            semantics=sem,
            object_target_pose=torch.eye(4),
            left_object_to_eef=torch.eye(4),
            right_object_to_eef=torch.eye(4),
        )
        assert target.semantics is sem
        assert isinstance(target, ObjectActionTarget)
        assert target.left_object_to_eef.shape == (4, 4)
        assert CoordinatedPickmentTarget is CoordinatedPickTarget

    def test_coordinated_placement_target_only_holds_desired_state(self):
        target = CoordinatedPlacementTarget(
            placing_object_target_pose=torch.eye(4),
            support_object_target_pose=torch.eye(4),
        )
        assert isinstance(target, ActionTarget)
        assert not hasattr(target, "placing_held_object")
        assert target.support_object_target_pose.shape == (4, 4)


class TestObjectSemantics:
    def test_does_not_mutate_affordance_geometry(self):
        # The redesign removes the __post_init__ aliasing footgun.
        aff = Affordance()
        geometry = {"bounding_box": [0.1, 0.1, 0.1]}
        ObjectSemantics(affordance=aff, geometry=geometry, label="mug")
        # affordance should not have a geometry attribute, or if it does it should
        # NOT be the same object as the semantics' geometry dict.
        assert getattr(aff, "geometry", None) is not geometry

    def test_sets_object_label_on_affordance(self):
        aff = Affordance()
        ObjectSemantics(affordance=aff, geometry={}, label="mug")
        assert aff.object_label == "mug"

    def test_default_optional_fields(self):
        sem = ObjectSemantics(affordance=Affordance(), geometry={})
        assert sem.label == "none"
        assert sem.properties == {}
        assert sem.entity is None


class TestHeldObjectState:
    def test_required_fields(self):
        sem = ObjectSemantics(affordance=Affordance(), geometry={})
        s = HeldObjectState(
            semantics=sem,
            object_to_eef=torch.eye(4).unsqueeze(0),
            grasp_xpos=torch.eye(4).unsqueeze(0),
        )
        assert s.semantics is sem
        assert s.object_to_eef.shape == (1, 4, 4)
        assert s.grasp_xpos.shape == (1, 4, 4)


class TestCoordinatedHeldObjectState:
    def test_required_fields(self):
        sem = ObjectSemantics(affordance=Affordance(), geometry={})
        s = CoordinatedHeldObjectState(
            semantics=sem,
            left_object_to_eef=torch.eye(4).unsqueeze(0),
            right_object_to_eef=torch.eye(4).unsqueeze(0),
            left_grasp_xpos=torch.eye(4).unsqueeze(0),
            right_grasp_xpos=torch.eye(4).unsqueeze(0),
        )
        assert s.semantics is sem
        assert s.left_object_to_eef.shape == (1, 4, 4)
        assert s.right_grasp_xpos.shape == (1, 4, 4)


class TestWorldState:
    def test_constructs_with_last_qpos_only(self):
        qpos = torch.zeros(2, 6)
        ws = WorldState(last_qpos=qpos)
        assert ws.last_qpos is qpos
        assert ws.held_objects == {}
        assert ws.coordinated_held_objects == {}

    def test_carries_held_object(self):
        sem = ObjectSemantics(affordance=Affordance(), geometry={})
        held = HeldObjectState(
            semantics=sem,
            object_to_eef=torch.eye(4).unsqueeze(0),
            grasp_xpos=torch.eye(4).unsqueeze(0),
        )
        ws = WorldState(
            last_qpos=torch.zeros(1, 6),
            held_objects={"left_arm": held},
        )
        assert ws.get_held_object("left_arm") is held
        assert ws.get_held_object("right_arm") is None

    def test_carries_coordinated_held_object(self):
        sem = ObjectSemantics(affordance=Affordance(), geometry={})
        held = CoordinatedHeldObjectState(
            semantics=sem,
            left_object_to_eef=torch.eye(4).unsqueeze(0),
            right_object_to_eef=torch.eye(4).unsqueeze(0),
            left_grasp_xpos=torch.eye(4).unsqueeze(0),
            right_grasp_xpos=torch.eye(4).unsqueeze(0),
        )
        ws = WorldState(
            last_qpos=torch.zeros(1, 14),
            coordinated_held_objects={("left_arm", "right_arm"): held},
        )
        assert ws.get_coordinated_held_object("left_arm", "right_arm") is held

    def test_with_updates_does_not_alias_held_state_dictionaries(self):
        ws = WorldState(last_qpos=torch.zeros(1, 6))
        successor = ws.with_updates(last_qpos=torch.ones(1, 6))
        successor.held_objects["arm"] = HeldObjectState(
            semantics=ObjectSemantics(affordance=Affordance(), geometry={}),
            object_to_eef=torch.eye(4).unsqueeze(0),
            grasp_xpos=torch.eye(4).unsqueeze(0),
        )
        assert ws.held_objects == {}


class TestActionResult:
    def test_shape_contract(self):
        traj = torch.zeros(2, 10, 8)
        ws = WorldState(last_qpos=torch.zeros(2, 8))
        res = ActionResult(success=True, trajectory=traj, next_state=ws)
        assert res.success is True
        assert res.trajectory.shape == (2, 10, 8)
        assert res.next_state is ws


class TestActionCfg:
    def test_defaults(self):
        cfg = ActionCfg()
        assert cfg.name == "default"
        assert cfg.control_part == "arm"
        assert cfg.interpolation_type == "linear"
        assert cfg.velocity_limit is None
        assert cfg.acceleration_limit is None
