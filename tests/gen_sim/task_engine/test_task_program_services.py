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

"""GenSim-owned service contracts migrated from shared integration tests."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from embodichain.gen_sim.task_engine._task_program.align_held import _AlignHeldLowerer

from embodichain.gen_sim.task_engine._task_program.configured import (
    decode_task_lowerer,
)
from embodichain.lab.task_program.integrations.configured import _decode_robot_profile
from embodichain.gen_sim.task_engine._task_program.services import (
    _AbsolutePoseTarget,
    _CoordinatedHoldLowerer,
    _CoordinatedTransportRoute,
    _MoveHeldObjectLowerer,
    _MoveHeldObjectRoute,
    _PickLowerer,
    _PickRoute,
)
from embodichain.lab.gym.utils._component_composition import _resolve_gym_components
from embodichain.lab.task_program.integrations._configured_composition import (
    _compose_integration_payload,
    _resolve_task_program_components,
)
from embodichain.lab.sim.atomic_actions import (
    AntipodalAffordance,
    CoordinatedPickmentOptions,
    EntityState,
    GraspGoal,
    HandOverOptions,
    HeldObjectPoseGoal,
    MoveHeldObjectOptions,
    ObjectSemantics,
    PickUpOptions,
    PlanningContext,
    RobotObservation,
    SceneSnapshot,
    TaskState,
)
from embodichain.lab.task_program.semantics import (
    HeldObjectRelation,
    RegisteredSemanticCall,
    SceneObjectRef,
    SemanticEffectKind,
)
from embodichain.utils.utility import load_config

__all__: list[str] = []


def test_current_pose_upright_binding_preserves_each_environments_position() -> None:
    poses = torch.eye(4).repeat(2, 1, 1)
    poses[:, :3, 3] = torch.tensor([[0.1, -0.2, 1.0], [0.3, 0.2, 1.2]])
    before = poses.clone()
    robot = Mock()
    lowerer = _AlignHeldLowerer(
        (("can", "current_object_pose", True, (1.0, 0.0, 0.0), None),), robot
    )
    context = SimpleNamespace(
        batch_size=2,
        task=SimpleNamespace(
            get_held_object=lambda key: SimpleNamespace(
                semantics=SimpleNamespace(entity_id="can")
            )
        ),
        scene=SimpleNamespace(
            entities={"can": SimpleNamespace(pose=poses, confidence=1.0)}
        ),
    )
    bound = SimpleNamespace(
        binding=SimpleNamespace(
            resources={
                "primary": SimpleNamespace(
                    endpoints={"motion": SimpleNamespace(task_state_key="left")}
                )
            }
        )
    )
    result = lowerer.lower(
        RegisteredSemanticCall(
            call_id="gen_sim.align_held",
            arguments={
                "object": "can",
                "target": "current_object_pose",
                "preserve_yaw": True,
            },
        ),
        context=context,
        bound=bound,
        option_template=MoveHeldObjectOptions(),
    )
    target = result.goal.object_target_pose
    torch.testing.assert_close(target[:, :3, 3], before[:, :3, 3])
    torch.testing.assert_close(
        target[:, :3, 0],
        torch.tensor([[0.0, 0.0, 1.0]]).repeat(2, 1),
        atol=1e-6,
        rtol=0,
    )
    torch.testing.assert_close(poses, before)
    assert robot.mock_calls == []


def test_coordinated_hold_lowerer_retains_both_verified_attachments() -> None:
    semantics = ObjectSemantics(
        affordance=AntipodalAffordance(),
        geometry={},
        label="tray",
        entity_id="tray",
    )
    lowerer = _CoordinatedHoldLowerer(
        (
            _CoordinatedTransportRoute(
                object_id="tray",
                target_id="tray_up",
                world_displacement=(0.0, 0.0, 0.14),
            ),
        ),
        (semantics,),
    )

    lowering = lowerer.lower(
        RegisteredSemanticCall(
            call_id="simulation.coordinated_hold",
            arguments={
                "object": "tray",
                "target": "tray_up",
                "world_displacement": [0.0, 0.0, 0.14],
            },
        ),
        context=PlanningContext(
            robot=RobotObservation(
                timestamp=1.0,
                qpos=torch.zeros((1, 1)),
                qvel=torch.zeros((1, 1)),
            ),
            task=TaskState(batch_size=1, device="cpu"),
            scene=SceneSnapshot(
                timestamp=1.0,
                version=1,
                entities={"tray": EntityState(torch.eye(4).unsqueeze(0))},
            ),
            env_ids=torch.tensor([0]),
        ),
        bound=None,  # type: ignore[arg-type]
        option_template=CoordinatedPickmentOptions(release=False),
    )

    assert lowering.registered_effect is not None
    assert lowering.registered_effect.effect_kind is SemanticEffectKind.ATTACH
    assert [effect.relation for effect in lowering.registered_effect.held_objects] == [
        HeldObjectRelation.ATTACHED,
        HeldObjectRelation.ATTACHED,
    ]
    factory = decode_task_lowerer(
        {
            "kind": "coordinated_hold",
            "routes": [
                {
                    "object_id": "tray",
                    "target_id": "tray_up",
                    "world_displacement": [0.0, 0.0, 0.14],
                }
            ],
        },
        path="integration.runtime_services.registered_semantic_lowerers[0]",
    )
    assert factory.call_id == "simulation.coordinated_hold"


def test_configured_handover_uses_baseline_timing_and_rejects_removed_wait_field() -> (
    None
):
    path = (
        Path(__file__).parents[3]
        / "embodichain_tasks/configs/tasks/manipulation/hand_over"
        / "task.dual_ur5_dh_pgi_140_80.yaml"
    )
    physical = _resolve_gym_components(load_config(path), base_dir=path.parent)
    _, task, policy = _resolve_task_program_components(
        physical.config["task_program"], base_dir=path.parent
    )
    payload = _compose_integration_payload(
        task=task,
        policy=policy,
        skill_profile=physical.embodiment_skill_profile,
        scene=task["scene_binding"],
    )["robot_profile"]
    payload["presets"][0]["action_options"]["hand_over"].update(
        retreat_distance=0.12, retreat_steps=28
    )
    before = deepcopy(payload)
    options = (
        _decode_robot_profile(payload).presets[0].action_option_templates["hand_over"]
    )

    assert payload == before
    assert type(options) is HandOverOptions
    assert options.retreat_distance == pytest.approx(0.12)
    assert options.retreat_steps == 28
    assert not hasattr(options, "source_release_settle_steps")
    payload["presets"][0]["action_options"]["hand_over"][
        "source_release_settle_steps"
    ] = 16
    with pytest.raises(ValueError, match="source_release_settle_steps"):
        _decode_robot_profile(payload)


def test_configured_transport_binds_one_baseline_pose_without_planning() -> None:
    pose = _AbsolutePoseTarget((0.1, 0.2, 0.8), (1.0, 0.0, 0.0, 0.0))
    lowerer = _MoveHeldObjectLowerer(
        (_MoveHeldObjectRoute("part", "inspection", pose),)
    )
    forbidden_robot = Mock()
    goal = lowerer.lower(
        RegisteredSemanticCall(
            call_id="simulation.move_held_object",
            arguments={"object": "part", "target": "inspection"},
        ),
        context=forbidden_robot,
        bound=forbidden_robot,
        option_template=MoveHeldObjectOptions(),
    ).goal
    forbidden_robot.assert_not_called()
    assert forbidden_robot.mock_calls == []
    assert type(goal) is HeldObjectPoseGoal
    torch.testing.assert_close(goal.object_target_pose, pose.to_matrix())
    assert not hasattr(goal, "alternative_object_target_poses")
    lookahead = lowerer.pick_lookahead_targets(
        RegisteredSemanticCall(
            call_id="simulation.move_held_object", arguments={"target": "inspection"}
        ),
        picked_object=SceneObjectRef("different_part"),
        bound=None,
        previous_target=None,
    )
    assert lookahead is None


def test_transport_decoder_rejects_old_alternative_pose_declarations() -> None:
    pose = {
        "kind": "pose",
        "position": [0.1, 0.2, 0.8],
        "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
    }
    with pytest.raises(ValueError, match="alternatives"):
        decode_task_lowerer(
            {
                "kind": "move_held_object",
                "routes": [
                    {
                        "object_id": "part",
                        "target_id": "inspection",
                        "pose": pose,
                        "alternatives": [pose],
                    }
                ],
            },
            path="lowerer",
        )


def test_configured_pick_keeps_baseline_goal_and_preset_ownership() -> None:
    semantics = ObjectSemantics(
        affordance=AntipodalAffordance(), geometry={}, entity_id="part"
    )
    route = _PickRoute("part", "inspection")
    options = PickUpOptions(
        pick_object_part="bottom", approach_direction=torch.tensor([1.0, 0.0, 0.0])
    )
    lowering = _PickLowerer((route,), (semantics,)).lower(
        RegisteredSemanticCall(
            call_id="simulation.pick",
            arguments={"object": "part", "target": "inspection"},
        ),
        context=None,
        bound=None,
        option_template=options,
    )
    assert type(lowering.goal) is GraspGoal
    assert lowering.goal.semantics is semantics
    assert lowering.skill_options is None
    assert options.pick_object_part == "bottom"
    assert options.downstream_object_target_poses == ()
    assert lowering.registered_effect.effect_kind is SemanticEffectKind.ATTACH
    assert lowering.phase_protection.gate_segment == "lift"
    assert lowering.phase_protection.active_segments == ("lift",)


@pytest.mark.parametrize("protected", (False, True))
def test_held_move_phase_protection_is_an_explicit_service_opt_in(
    protected: bool,
) -> None:
    config = {
        "kind": "move_held_object",
        "routes": [
            {
                "object_id": "part",
                "target_id": "inspection",
                "pose": {
                    "kind": "pose",
                    "position": [0.0, 0.0, 1.0],
                    "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                },
            }
        ],
    }
    if protected:
        config["phase_protection"] = "held_object_v1"
    factory = decode_task_lowerer(config, path="runtime_services")
    robot = object()
    registry = SimpleNamespace(resolve=lambda *args, **kwargs: SceneObjectRef("part"))
    lowerer = factory.create(
        simulation=None,
        robot=robot,
        scene_registry=registry,
        engine=SimpleNamespace(robot=robot),
    )
    lowered = lowerer.lower(
        RegisteredSemanticCall(
            call_id="simulation.move_held_object",
            arguments={"object": "part", "target": "inspection"},
        ),
        context=None,
        bound=None,
        option_template=MoveHeldObjectOptions(),
    )
    assert (lowered.phase_protection is not None) is protected
    if protected:
        assert lowered.phase_protection.active_segments == ("transport",)
        assert lowered.phase_protection.gate_segment is None
    assert lowered.registered_effect is None


def test_pick_decoder_rejects_removed_runtime_option_declarations() -> None:
    with pytest.raises(ValueError, match="required_object_target_poses"):
        decode_task_lowerer(
            {
                "kind": "pick",
                "routes": [
                    {
                        "object_id": "part",
                        "target_id": "inspection",
                        "required_object_target_poses": [],
                    }
                ],
            },
            path="pick",
        )


def test_task_pick_alias_factory_has_stable_immutable_identity() -> None:
    payload = {
        "kind": "pick",
        "call_id": "gen_sim.pick.step_01",
        "routes": [{"object_id": "part", "target_id": "release"}],
    }
    first = decode_task_lowerer(payload, path="pick")
    second = decode_task_lowerer(payload, path="pick")
    assert first.call_id == "gen_sim.pick.step_01"
    assert first.lowerer_type.call_id == first.call_id
    assert type(first).__qualname__ == type(second).__qualname__
    assert first.target_descriptor.skill_id == "pick_up"
    robot = object()
    registry = Mock()
    registry.resolve.return_value = SceneObjectRef("part")
    registry.object_semantics.return_value = ObjectSemantics(
        affordance=AntipodalAffordance(), geometry={}, entity_id="part"
    )
    lowerer = first.create(
        simulation=None,
        robot=robot,
        scene_registry=registry,
        engine=SimpleNamespace(robot=robot, grasp_pose_generators={}),
    )
    assert type(lowerer).call_id == first.call_id
    result = lowerer.lower(
        RegisteredSemanticCall(
            call_id=first.call_id, arguments={"object": "part", "target": "release"}
        ),
        context=None,
        bound=None,
        option_template=PickUpOptions(),
    )
    assert type(result.goal) is GraspGoal
    assert result.skill_options is None
    with pytest.raises(FrozenInstanceError):
        first.routes = ()


@pytest.mark.parametrize(
    "kind",
    [
        "articulation_link_slide",
        "articulation_link_press",
        "articulation_link_twist",
        "release_safe_pick",
        "move_held_object_upright",
    ],
)
def test_task_decoder_rejects_out_of_scope_services(kind: str) -> None:
    with pytest.raises(ValueError, match="unsupported Task Engine service"):
        decode_task_lowerer({"kind": kind}, path="lowerer")
