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

"""Tests for the semantic-skill tutorial declarations."""

from __future__ import annotations

from typing import cast, TYPE_CHECKING
from unittest.mock import Mock

import pytest
import torch

from embodichain.lab.sim import SimulationManager
from embodichain.lab.sim.atomic_actions import (
    AntipodalAffordance,
    EffectVerificationRequest,
    GraspGoal,
    HandOverGoal,
    HandOverOptions,
    PlanningContext,
    TrackingPolicy,
)
from embodichain.lab.sim.skills import (
    GRASP_AFFORDANCE_CAPABILITY,
    Pick,
    Place,
    RegisteredSemanticCall,
    SceneRegistry,
)
import scripts.tutorials.semantic_skill.hand_over as handover_tutorial
import scripts.tutorials.semantic_skill.place as place_tutorial
from scripts.tutorials.semantic_skill.hand_over import (
    FINAL_OBJECT_POSITION,
    HANDOVER_CALL_ID,
    TRACKING_ERROR_THRESHOLD as HANDOVER_TRACKING_ERROR_THRESHOLD,
    TutorialHandOverLowerer,
    create_handover_effect_verifier,
    create_handover_task,
    create_robot_profile as create_dual_arm_profile,
)
from scripts.tutorials.semantic_skill.place import (
    MINIMUM_PICK_LIFT,
    TARGET_OBJECT_POSITION,
    TRACKING_ERROR_THRESHOLD as PLACE_TRACKING_ERROR_THRESHOLD,
    create_place_effect_verifier,
    create_place_task,
    create_robot_profile as create_single_arm_profile,
)
from scripts.tutorials.semantic_skill.tutorial_utils import (
    create_graspable_object_registry,
    create_runtime_step_observer,
)

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import RigidObject, Robot
    from embodichain.lab.sim.skills.integration import BoundSemanticCall

_TEST_PHYSICS_DT = 0.01
_TEST_GRASP_SAMPLE_COUNT = 8


class _PhysicalObject:
    """Small mutable pose source used by verifier tests."""

    def __init__(self, pose: torch.Tensor) -> None:
        self.pose = pose
        self.clear_count = 0

    def get_local_pose(self, to_matrix: bool = False) -> torch.Tensor:
        assert to_matrix is True
        return self.pose.clone()

    def clear_dynamics(self) -> None:
        self.clear_count += 1


class _PhysicalRobot:
    """Expose only the joint and FK observations used by tutorial verifiers."""

    def __init__(self) -> None:
        self.qpos: dict[str, torch.Tensor] = {}
        self.eef_pose: dict[str, torch.Tensor] = {}

    def get_qpos(self, name: str) -> torch.Tensor:
        return self.qpos[name].clone()

    def compute_fk(
        self,
        *,
        qpos: torch.Tensor,
        name: str,
        to_matrix: bool,
    ) -> torch.Tensor:
        del qpos
        assert to_matrix is True
        return self.eef_pose[name].clone()


def _pose_at(position: tuple[float, float, float]) -> torch.Tensor:
    pose = torch.eye(4).unsqueeze(0)
    pose[:, :3, 3] = torch.tensor(position)
    return pose


def _request(
    skill_id: str,
    *,
    held_control_part: str | None = None,
) -> EffectVerificationRequest:
    request = Mock()
    request.skill_id = skill_id
    held = Mock()
    held.object_to_eef = torch.eye(4).unsqueeze(0)
    request.expected_effects.held_object_updates = (
        {} if held_control_part is None else {held_control_part: held}
    )
    return cast(EffectVerificationRequest, request)


def _verification_context() -> PlanningContext:
    context = Mock()
    context.robot.qpos = torch.zeros(1, 1)
    return cast(PlanningContext, context)


def _graspable_registry() -> SceneRegistry:
    entity = Mock()
    entity.get_local_pose.return_value = torch.eye(4)
    simulation = Mock()
    simulation.get_rigid_object.return_value = entity

    registry, _ = create_graspable_object_registry(
        cast(SimulationManager, simulation),
        object_id="workpiece",
        simulation_uid="sim_cube",
        semantic_type="cube",
        affordance=AntipodalAffordance(),
    )
    return registry


def test_graspable_registry_maps_simulation_identity_to_semantic_identity() -> None:
    registry = _graspable_registry()
    object_ref = registry.resolve("workpiece")

    assert registry.resolve("sim_cube") == object_ref
    grasp_ref = registry.resolve_affordance(
        object_ref,
        capability=GRASP_AFFORDANCE_CAPABILITY,
    )
    semantics = registry.object_semantics(object_ref, affordance=grasp_ref)
    assert semantics.entity_id == "workpiece"
    assert type(semantics.affordance) is AntipodalAffordance


def test_place_tutorial_task_contains_no_robot_resource_names() -> None:
    calls = create_place_task()

    assert tuple(type(call) for call in calls) == (Pick, Place)
    assert calls[0].object == calls[1].object
    assert dict(calls[0].resources) == {}
    assert dict(calls[1].resources) == {}
    assert calls[1].at is not None
    torch.testing.assert_close(
        calls[1].at.position,
        torch.tensor(TARGET_OBJECT_POSITION),
    )


def test_place_tutorial_profile_owns_single_arm_binding_and_policies() -> None:
    profile = create_single_arm_profile(
        torch.tensor([0.0, 0.0]),
        torch.tensor([0.5, 0.5]),
    )

    resource = profile.resources["primary_manipulator"]
    assert resource.endpoints["motion"].control_part == "arm"
    assert resource.endpoints["grasp"].control_part == "hand"
    assert dict(profile.defaults["pick_up"].resources) == {
        "primary": "primary_manipulator"
    }
    assert dict(profile.defaults["place"].resources) == {
        "primary": "primary_manipulator"
    }
    assert dict(profile.skill_presets) == {"pick_up": "pick", "place": "place"}
    assert profile.presets["pick"].tracking_policy == TrackingPolicy.joint_position(
        in_flight_max_abs_error=PLACE_TRACKING_ERROR_THRESHOLD,
        terminal_max_abs_error=PLACE_TRACKING_ERROR_THRESHOLD,
    )
    assert profile.presets["place"].recovery_policy.max_action_retries == 0
    assert profile.presets["place"].tracking_policy == TrackingPolicy.joint_position(
        in_flight_max_abs_error=PLACE_TRACKING_ERROR_THRESHOLD,
        terminal_max_abs_error=PLACE_TRACKING_ERROR_THRESHOLD,
    )


def test_place_application_installs_default_effect_verifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    simulation = Mock()
    simulation.sim_config.physics_dt = _TEST_PHYSICS_DT
    simulation.get_rigid_object.return_value = Mock()
    robot = Mock()
    obj = _PhysicalObject(_pose_at((0.0, 0.0, 0.0)))
    semantics = Mock(affordance=AntipodalAffordance())
    motion_generator = Mock()
    runtime = Mock()
    runtime_factory = Mock(return_value=runtime)

    monkeypatch.setattr(
        place_tutorial,
        "create_antipodal_semantics",
        Mock(return_value=semantics),
    )
    monkeypatch.setattr(
        place_tutorial,
        "create_curobo_motion_generator",
        Mock(return_value=motion_generator),
    )
    monkeypatch.setattr(
        place_tutorial.SkillRuntime,
        "from_simulation",
        runtime_factory,
    )

    result = place_tutorial.create_place_application(
        cast(SimulationManager, simulation),
        cast("Robot", robot),
        cast("RigidObject", obj),
        hand_open=torch.zeros(1),
        hand_grasp=torch.ones(1),
        n_sample=_TEST_GRASP_SAMPLE_COUNT,
        force_reannotate=False,
    )

    assert result is runtime
    assert type(runtime_factory.call_args.kwargs["scene_registry"]) is SceneRegistry
    assert (
        runtime_factory.call_args.kwargs["robot_profile"].profile_id
        == "tutorial.single_arm"
    )
    assert runtime_factory.call_args.kwargs["motion_generator"] is motion_generator
    assert callable(runtime_factory.call_args.kwargs["effect_verifier"])


def test_place_tutorial_verifies_observed_pick_lift_and_eef_proximity() -> None:
    physical_object = _PhysicalObject(_pose_at((0.0, 0.0, 0.0)))
    physical_robot = _PhysicalRobot()
    physical_robot.qpos["arm"] = torch.zeros(1, 1)
    physical_robot.qpos["hand"] = torch.zeros(1, 1)
    verifier = create_place_effect_verifier(
        cast("RigidObject", physical_object),
        cast("Robot", physical_robot),
        torch.zeros(1),
    )
    lifted_pose = _pose_at((0.0, 0.0, MINIMUM_PICK_LIFT + 0.01))
    physical_object.pose = lifted_pose
    physical_robot.eef_pose["arm"] = lifted_pose

    success = verifier(
        create_place_task()[0],
        _request("pick_up", held_control_part="arm"),
        _verification_context(),
    )

    assert success.tolist() == [True]


def test_place_tutorial_rejects_lift_with_wrong_grasp_relation() -> None:
    physical_object = _PhysicalObject(_pose_at((0.0, 0.0, 0.0)))
    physical_robot = _PhysicalRobot()
    physical_robot.qpos["arm"] = torch.zeros(1, 1)
    physical_robot.qpos["hand"] = torch.zeros(1, 1)
    verifier = create_place_effect_verifier(
        cast("RigidObject", physical_object),
        cast("Robot", physical_robot),
        torch.zeros(1),
    )
    physical_object.pose = _pose_at((0.0, 0.0, MINIMUM_PICK_LIFT + 0.01))
    physical_robot.eef_pose["arm"] = _pose_at((0.2, 0.0, MINIMUM_PICK_LIFT + 0.01))

    success = verifier(
        create_place_task()[0],
        _request("pick_up", held_control_part="arm"),
        _verification_context(),
    )

    assert success.tolist() == [False]


def test_place_tutorial_verifies_release_at_requested_position() -> None:
    physical_object = _PhysicalObject(_pose_at((0.0, 0.0, 0.0)))
    physical_robot = _PhysicalRobot()
    hand_open = torch.tensor([0.0, 0.0])
    physical_robot.qpos["hand"] = hand_open.unsqueeze(0)
    verifier = create_place_effect_verifier(
        cast("RigidObject", physical_object),
        cast("Robot", physical_robot),
        hand_open,
    )
    physical_object.pose = _pose_at(TARGET_OBJECT_POSITION)

    success = verifier(
        create_place_task()[1],
        _request("place"),
        _verification_context(),
    )

    assert success.tolist() == [True]


def test_handover_tutorial_registers_tuned_atomic_lowering() -> None:
    calls = create_handover_task()
    context = Mock()
    context.robot.qpos = torch.zeros(1, 1)
    lowerer = TutorialHandOverLowerer(_graspable_registry())

    assert tuple(type(call) for call in calls) == (RegisteredSemanticCall,)
    assert calls[0].call_id == HANDOVER_CALL_ID
    lowering = lowerer.lower(
        calls[0],
        context=cast(PlanningContext, context),
        bound=cast("BoundSemanticCall", object()),
    )
    assert type(lowering.goal) is HandOverGoal
    assert type(lowering.skill_options) is HandOverOptions
    assert lowering.goal.semantics.entity_id == "workpiece"
    torch.testing.assert_close(
        lowering.goal.target_pose[:3, 3],
        torch.tensor(FINAL_OBJECT_POSITION),
    )


def test_handover_tutorial_profile_binds_disjoint_arms() -> None:
    profile = create_dual_arm_profile(
        torch.tensor([0.0]),
        torch.tensor([0.5]),
        torch.tensor([0.0]),
        torch.tensor([0.5]),
    )

    assert profile.resources["left"].endpoints["motion"].control_part == "left_arm"
    assert profile.resources["right"].endpoints["motion"].control_part == "right_arm"
    assert dict(profile.defaults["hand_over"].resources) == {
        "source": "left",
        "destination": "right",
    }
    assert (
        profile.presets["hand_over"].recovery_policy.tracking_error_threshold
        == HANDOVER_TRACKING_ERROR_THRESHOLD
    )


def test_handover_application_installs_extension_and_default_verifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    simulation = Mock()
    simulation.sim_config.physics_dt = _TEST_PHYSICS_DT
    simulation.get_rigid_object.return_value = Mock()
    robot = Mock()
    obj = _PhysicalObject(_pose_at((0.0, 0.0, 0.0)))
    semantics = Mock(affordance=AntipodalAffordance())
    motion_generator = Mock()
    runtime = Mock()
    runtime_factory = Mock(return_value=runtime)

    monkeypatch.setattr(
        handover_tutorial,
        "create_antipodal_semantics",
        Mock(return_value=semantics),
    )
    monkeypatch.setattr(
        handover_tutorial,
        "create_toppra_motion_generator",
        Mock(return_value=motion_generator),
    )
    monkeypatch.setattr(
        handover_tutorial.SkillRuntime,
        "from_simulation",
        runtime_factory,
    )

    result = handover_tutorial.create_handover_application(
        cast(SimulationManager, simulation),
        cast("Robot", robot),
        cast("RigidObject", obj),
        left_open=torch.zeros(1),
        left_grasp=torch.ones(1),
        right_open=torch.zeros(1),
        right_grasp=torch.ones(1),
        n_sample=_TEST_GRASP_SAMPLE_COUNT,
        force_reannotate=False,
    )

    assert result is runtime
    assert type(runtime_factory.call_args.kwargs["scene_registry"]) is SceneRegistry
    assert (
        runtime_factory.call_args.kwargs["robot_profile"].profile_id
        == "tutorial.dual_arm"
    )
    assert runtime_factory.call_args.kwargs["motion_generator"] is motion_generator
    assert (
        HANDOVER_CALL_ID in runtime_factory.call_args.kwargs["call_catalog"].descriptors
    )
    assert callable(runtime_factory.call_args.kwargs["effect_verifier"])
    assert type(runtime_factory.call_args.kwargs["registered_lowerers"][0]) is (
        TutorialHandOverLowerer
    )


def test_handover_tutorial_verifies_release_at_final_target() -> None:
    physical_object = _PhysicalObject(_pose_at((0.0, 0.0, 0.0)))
    physical_robot = _PhysicalRobot()
    left_open = torch.tensor([0.0])
    right_open = torch.tensor([0.0])
    physical_robot.qpos["left_hand"] = left_open.unsqueeze(0)
    physical_robot.qpos["right_hand"] = right_open.unsqueeze(0)
    verifier = create_handover_effect_verifier(
        cast("RigidObject", physical_object),
        cast("Robot", physical_robot),
        left_open=left_open,
        right_open=right_open,
    )
    final_pose = _pose_at(FINAL_OBJECT_POSITION)
    physical_object.pose = final_pose
    physical_robot.qpos["right_arm"] = torch.zeros(1, 1)
    physical_robot.eef_pose["right_arm"] = final_pose

    success = verifier(
        create_handover_task()[0],
        _request("hand_over"),
        _verification_context(),
    )

    assert success.tolist() == [True]


def test_runtime_step_observer_stabilizes_initial_grasp_once() -> None:
    physical_object = _PhysicalObject(_pose_at((0.0, 0.0, 0.0)))
    physical_robot = _PhysicalRobot()
    physical_robot.qpos["hand"] = torch.zeros(1, 1)
    observer = create_runtime_step_observer(
        cast("RigidObject", physical_object),
        cast("Robot", physical_robot),
        grasp_control_part="hand",
        grasp_target=torch.ones(1),
    )
    runner_step = Mock(tick=None)

    observer(runner_step)
    physical_robot.qpos["hand"] = torch.ones(1, 1)
    observer(runner_step)
    observer(runner_step)

    assert physical_object.clear_count == 1
