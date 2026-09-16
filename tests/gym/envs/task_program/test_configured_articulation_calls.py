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

"""Strict configured contracts for articulation Press, Twist, and OpenDoor."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from embodichain.lab.sim.atomic_actions import (
    ArticulationAffordanceGeometry,
    OpenDoor,
    OpenDoorGoal,
    OpenDoorOptions,
    Press,
    PressGoal,
    PressOptions,
    SceneEntityPose,
    SlideOptions,
    Twist,
    TwistGoal,
    TwistOptions,
)
from embodichain.lab.task_program.integrations.configured import (
    _decode_action_options,
    _decode_registered_lowerer,
    _decode_runtime_services,
)
from embodichain.lab.task_program.integrations.extensions import (
    RegisteredSemanticLowererFactory,
)
from embodichain.lab.task_program.compiler.lowering import RegisteredSemanticLowerer
from embodichain.lab.task_program.semantics import (
    RegisteredSemanticCall,
    SceneArticulationRef,
    SceneEntityRegistration,
    SceneLinkRef,
    SceneRegistry,
)

_ENTITY_ID = "appliance"
_LINK_ENTITY_ID = "interaction_target"
_NATIVE_LINK = "contact_link"
_JOINT_NAME = "interaction_joint"
_JOINT_LIMITS = (-1.5, 1.5)
_KINDS = ("press", "twist", "open_door")
_OPTION_TYPES = {
    "press": PressOptions,
    "twist": TwistOptions,
    "open_door": OpenDoorOptions,
}
_GOAL_TYPES = {"press": PressGoal, "twist": TwistGoal, "open_door": OpenDoorGoal}
_SKILLS = {"press": Press, "twist": Twist, "open_door": OpenDoor}


class _NeverObserveProvider:
    """Assembly tests must not observe or step a simulation."""

    def observe(self, *, timestamp: float, env_ids: torch.Tensor) -> object:
        raise AssertionError("Factory construction must not observe scene state.")


def _service(kind: str, **overrides: object) -> dict[str, object]:
    """Return one executable-free service declaration."""
    config = {
        "kind": f"articulation_link_{kind}",
        "articulation_id": _ENTITY_ID,
        "articulation_simulation_uid": _ENTITY_ID,
        "link_entity_id": _LINK_ENTITY_ID,
    }
    if kind == "twist":
        config["grasp_position"] = [0.0, 0.0, 0.0]
    return {**config, **overrides}


def _geometry() -> ArticulationAffordanceGeometry:
    """Return an exposed target whose inward direction is positive X."""
    target = torch.tensor([[0.0, -0.02, -0.02], [0.0, 0.02, -0.02], [0.0, 0.0, 0.02]])
    body = target + torch.tensor([0.03, 0.0, 0.0])
    return ArticulationAffordanceGeometry(
        target_link_point_cloud=target,
        articulation_point_cloud=torch.cat((target, body)),
        non_target_articulation_point_cloud=body,
        prismatic_joint_axis=torch.tensor([1.0, 0.0, 0.0]),
        revolute_joint_axis=torch.tensor([-1.0, 0.0, 0.0]),
        revolute_axis_origin=torch.tensor([0.01, 0.0, 0.0]),
    )


def _runtime() -> tuple[SimpleNamespace, SimpleNamespace, SceneRegistry]:
    """Build public articulation facts without a simulator or controller."""
    geometry = _geometry()
    joint = SimpleNamespace(
        name=_JOINT_NAME,
        joint_type="revolute",
        parent_link_name="base",
        joint_limits=_JOINT_LIMITS,
        origin_pose=torch.eye(4),
        axis=torch.tensor([0.0, 0.0, 1.0]),
    )
    articulation = SimpleNamespace(
        link_names=("base", _NATIVE_LINK),
        joint_names=(_JOINT_NAME,),
        cfg=SimpleNamespace(init_qpos=(0.0,), body_scale=(1.0, 1.0, 1.0)),
        get_parent_joint_chain=lambda link: (joint,),
        get_link_pose=lambda *args, **kwargs: torch.eye(4).unsqueeze(0),
        get_link_vert_face=lambda link: (
            geometry.target_link_point_cloud,
            torch.tensor([[0, 1, 2]]),
        ),
    )
    reference = SceneArticulationRef(_ENTITY_ID)
    registry = SceneRegistry(
        (
            SceneEntityRegistration(
                ref=reference, state_provider=_NeverObserveProvider()
            ),
            SceneEntityRegistration(
                ref=SceneLinkRef(_LINK_ENTITY_ID),
                parent=reference,
                native_name=_NATIVE_LINK,
                state_provider=_NeverObserveProvider(),
            ),
        )
    )
    simulation = SimpleNamespace(
        get_articulation=lambda uid: articulation if uid == _ENTITY_ID else None
    )
    return simulation, articulation, registry


def _lowerer(
    kind: str,
    factory: RegisteredSemanticLowererFactory | None = None,
    **overrides: object,
) -> RegisteredSemanticLowerer:
    """Construct the decoded production factory against public fake scene facts."""
    if factory is None:
        factory = _decode_registered_lowerer(
            _service(kind, **overrides), path="lowerer"
        )
    simulation, articulation, registry = _runtime()
    robot = object()
    engine = SimpleNamespace(robot=robot)
    with patch(
        "embodichain.lab.task_program.integrations._configured_services.sample_initial_articulation_geometry",
        return_value=_geometry(),
    ) as sampler:
        lowerer = factory.create(
            simulation=simulation, robot=robot, scene_registry=registry, engine=engine
        )
    if kind == "open_door":
        sampler.assert_not_called()
    else:
        sampler.assert_called_once_with(
            articulation,
            _NATIVE_LINK,
            initial_qpos=(0.0,),
            initial_qpos_joint_names=(_JOINT_NAME,),
            body_scale=(1.0, 1.0, 1.0),
        )
    return lowerer


def _arguments(kind: str) -> dict[str, object]:
    """Return the canonical task-owned call arguments."""
    if kind == "open_door":
        return {"handle": _LINK_ENTITY_ID, "open_fraction": 0.5}
    return {"target": _LINK_ENTITY_ID}


@pytest.mark.parametrize("kind", _KINDS)
def test_factory_matches_skill_and_constructs_fresh_lowerers(kind: str) -> None:
    """Registration metadata identifies the exact Atomic Skill and fresh state."""
    factory = _decode_registered_lowerer(_service(kind), path="lowerer")
    first, second = _lowerer(kind, factory), _lowerer(kind, factory)
    assert factory.call_id == f"simulation.articulation_link_{kind}"
    assert factory.revision == "1"
    assert factory.target_descriptor == _SKILLS[kind].descriptor()
    assert first.target_descriptor == factory.target_descriptor
    assert first is not second
    assert first._semantics is not second._semantics


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("mode", ("live", "snapshot"))
def test_lowering_uses_typed_goals_and_keeps_motion_options_in_preset(
    kind: str, mode: str
) -> None:
    """The pose mode controls grounding while the policy owns typed options."""
    lowerer = _lowerer(kind, target_pose_mode=mode)
    observed = torch.eye(4).repeat(2, 1, 1)
    context = SimpleNamespace(
        scene=SimpleNamespace(
            entities={_LINK_ENTITY_ID: SimpleNamespace(pose=observed)}
        )
    )
    call = RegisteredSemanticCall(call_id=lowerer.call_id, arguments=_arguments(kind))
    result = lowerer.lower(
        call, context=context, bound=None, option_template=_OPTION_TYPES[kind]()
    )
    assert type(result.goal) is _GOAL_TYPES[kind]
    assert result.skill_options is None
    assert result.goal.semantics.entity_id == _LINK_ENTITY_ID
    if mode == "live":
        assert type(result.goal.target_pose) is SceneEntityPose
        assert result.goal.target_pose.entity_id == _LINK_ENTITY_ID
    else:
        observed[:, 0, 3] = 1.0
        torch.testing.assert_close(
            result.goal.target_pose, torch.eye(4).repeat(2, 1, 1)
        )
    assert (
        lowerer.pick_lookahead_targets(
            call, picked_object=None, bound=None, previous_target=None
        )
        is None
    )


def test_twist_geometry_retains_joint_identity_limits_and_native_sign() -> None:
    """Joint-aware expansion can resolve live qpos and limit the sampled arc."""
    affordance = _lowerer("twist")._semantics.affordance
    assert affordance.joint_name == _JOINT_NAME
    assert affordance.joint_limits == _JOINT_LIMITS
    assert affordance.joint_axis_sign == -1
    assert affordance.axis_origin == pytest.approx((0.01, 0.0, 0.0))
    torch.testing.assert_close(affordance.twist_axis, torch.tensor([1.0, 0.0, 0.0]))


def test_twist_service_preserves_authored_nominal_grasp_roll() -> None:
    affordance = _lowerer(
        "twist", grasp_roll=torch.pi / 2, grasp_roll_range=[1.2, 1.8]
    )._semantics.affordance
    assert affordance.grasp_roll == pytest.approx(torch.pi / 2)
    assert affordance.grasp_roll_range == (1.2, 1.8)


@pytest.mark.parametrize("roll", (True, float("nan"), float("inf"), "1.5"))
def test_twist_service_rejects_invalid_nominal_grasp_roll(roll: object) -> None:
    with pytest.raises((TypeError, ValueError), match="grasp_roll"):
        _decode_registered_lowerer(_service("twist", grasp_roll=roll), path="lowerer")


def test_twist_service_requires_range_to_include_its_nominal_roll() -> None:
    with pytest.raises(ValueError, match="nominal grasp_roll"):
        _decode_registered_lowerer(
            _service("twist", grasp_roll=torch.pi / 2, grasp_roll_range=[-0.2, 0.2]),
            path="lowerer",
        )


def test_press_geometry_resolves_contact_and_inward_axis() -> None:
    """A configured press does not default silently to an unrelated world axis."""
    affordance = _lowerer("press")._semantics.affordance
    assert affordance.press_position is not None
    torch.testing.assert_close(affordance.press_axis, torch.tensor([1.0, 0.0, 0.0]))


@pytest.mark.parametrize("kind", _KINDS)
def test_factory_rejects_geometry_from_another_articulation(kind: str) -> None:
    """Native UIDs and semantic link parents must select the same body."""
    simulation, _, registry = _runtime()
    other_id = "other_appliance"
    mismatched_registry = SceneRegistry(
        (
            registry.lookup(_ENTITY_ID),
            registry.lookup(_LINK_ENTITY_ID),
            SceneEntityRegistration(
                ref=SceneArticulationRef(other_id),
                state_provider=_NeverObserveProvider(),
            ),
        )
    )
    factory = _decode_registered_lowerer(
        _service(kind, articulation_simulation_uid=other_id), path="lowerer"
    )
    robot = object()
    with pytest.raises(ValueError, match="parent articulation"):
        factory.create(
            simulation=simulation,
            robot=robot,
            scene_registry=mismatched_registry,
            engine=SimpleNamespace(robot=robot),
        )


def test_factory_accepts_native_uid_registered_as_articulation_alias() -> None:
    """Semantic IDs may differ from native UIDs when their binding declares it."""
    _, articulation, registry = _runtime()
    native_uid = "native_appliance"
    aliased_registry = SceneRegistry(
        (
            SceneEntityRegistration(
                ref=SceneArticulationRef(_ENTITY_ID),
                aliases=(native_uid,),
                state_provider=_NeverObserveProvider(),
            ),
            registry.lookup(_LINK_ENTITY_ID),
        )
    )
    simulation = SimpleNamespace(
        get_articulation=lambda uid: articulation if uid == native_uid else None
    )
    factory = _decode_registered_lowerer(
        _service("open_door", articulation_simulation_uid=native_uid), path="lowerer"
    )
    robot = object()
    lowerer = factory.create(
        simulation=simulation,
        robot=robot,
        scene_registry=aliased_registry,
        engine=SimpleNamespace(robot=robot),
    )
    assert lowerer._semantics.entity_id == _LINK_ENTITY_ID


def test_twist_factory_rejects_ambiguous_active_ancestor_chain() -> None:
    """A sampled arc must have one unambiguous live joint state to constrain."""
    simulation, articulation, registry = _runtime()
    joint = articulation.get_parent_joint_chain(_NATIVE_LINK)[0]
    articulation.get_parent_joint_chain = lambda link: (joint, joint)
    factory = _decode_registered_lowerer(_service("twist"), path="lowerer")
    robot = object()
    with pytest.raises(ValueError, match="unambiguous revolute"):
        factory.create(
            simulation=simulation,
            robot=robot,
            scene_registry=registry,
            engine=SimpleNamespace(robot=robot),
        )


def test_door_lowering_preserves_task_owned_opening_range() -> None:
    """Opening fractions belong to the goal rather than actuator policy."""
    lowerer = _lowerer("open_door", opening_direction=-1, hinge_joint_name=_JOINT_NAME)
    result = lowerer.lower(
        RegisteredSemanticCall(
            call_id=lowerer.call_id,
            arguments={**_arguments("open_door"), "open_fraction_range": [0.3, 0.7]},
        ),
        context=None,
        bound=None,
        option_template=OpenDoorOptions(),
    )
    assert result.goal.open_fraction == 0.5
    assert result.goal.open_fraction_range == (0.3, 0.7)
    assert result.goal.semantics.affordance.opening_direction == -1
    assert result.goal.semantics.affordance.joint_name == _JOINT_NAME


@pytest.mark.parametrize("kind", _KINDS)
def test_lowerer_rejects_unconfigured_target_and_motion_arguments(kind: str) -> None:
    """A call cannot replace the service target or override bound options."""
    lowerer = _lowerer(kind)
    target_key = "handle" if kind == "open_door" else "target"
    invalid = (
        {**_arguments(kind), target_key: "unknown"},
        {**_arguments(kind), "hand_interp_steps": 100},
        {},
    )
    for arguments in invalid:
        with pytest.raises(ValueError, match="arguments"):
            lowerer.lower(
                RegisteredSemanticCall(call_id=lowerer.call_id, arguments=arguments),
                context=None,
                bound=None,
                option_template=_OPTION_TYPES[kind](),
            )
    with pytest.raises(TypeError, match="exact"):
        lowerer.lower(
            RegisteredSemanticCall(call_id=lowerer.call_id, arguments=_arguments(kind)),
            context=None,
            bound=None,
            option_template=SlideOptions(),
        )


@pytest.mark.parametrize("fraction", (-0.1, 1.1, True, "0.5"))
def test_door_rejects_non_fraction_goal_values(fraction: object) -> None:
    lowerer = _lowerer("open_door")
    with pytest.raises((TypeError, ValueError), match="open_fraction"):
        lowerer.lower(
            RegisteredSemanticCall(
                call_id=lowerer.call_id,
                arguments={"handle": _LINK_ENTITY_ID, "open_fraction": fraction},
            ),
            context=None,
            bound=None,
            option_template=OpenDoorOptions(),
        )


@pytest.mark.parametrize("kind", _KINDS)
def test_services_reject_unknown_fields_and_invalid_pose_modes(kind: str) -> None:
    with pytest.raises(ValueError, match="unsupported fields"):
        _decode_registered_lowerer(
            _service(kind, class_type="arbitrary.Import"), path="lowerer"
        )
    with pytest.raises(ValueError, match="target_pose_mode"):
        _decode_registered_lowerer(
            _service(kind, target_pose_mode="future"), path="lowerer"
        )
    with pytest.raises(ValueError, match="duplicate call IDs"):
        _decode_runtime_services(
            {"registered_semantic_lowerers": [_service(kind), _service(kind)]}
        )


@pytest.mark.parametrize(
    "kind,values",
    (
        (
            "press",
            {
                "hand_interp_steps": 12,
                "press_distance": 0.03,
                "approach_distance": 0.12,
                "press_position": [0.0, 0.0, 0.01],
            },
        ),
        (
            "twist",
            {
                "twist_waypoint_count": 12,
                "twist_angle": -0.6,
                "twist_angle_range": [-0.8, -0.4],
                "pre_grasp_distance": 0.12,
            },
        ),
        (
            "open_door",
            {
                "door_waypoint_count": 30,
                "retract_distance": 0.1,
                "joint_position_tolerance": 0.001,
            },
        ),
    ),
)
def test_options_decoder_preserves_typed_values(
    kind: str, values: dict[str, object]
) -> None:
    options = _decode_action_options({"kind": kind, **values}, path="options")
    assert type(options) is _OPTION_TYPES[kind]
    for key, value in values.items():
        assert getattr(options, key) == pytest.approx(value)


@pytest.mark.parametrize(
    "kind,field,value",
    (
        ("press", "press_distance", 0.0),
        ("press", "press_position", [0.0, 0.0]),
        ("press", "twist_angle", 0.5),
        ("twist", "twist_waypoint_count", True),
        ("twist", "twist_angle_range", [-1.0, 1.0]),
        ("open_door", "door_waypoint_count", 0),
        ("open_door", "retract_distance", -0.1),
        ("open_door", "open_fraction", 0.5),
    ),
)
def test_options_decoder_rejects_invalid_or_other_skill_fields(
    kind: str, field: str, value: object
) -> None:
    with pytest.raises((TypeError, ValueError)):
        _decode_action_options({"kind": kind, field: value}, path="options")


def test_slide_sampling_range_is_accepted_by_outer_decoder_allowlist() -> None:
    """The common strict mapping must permit the existing Slide range field."""
    options = _decode_action_options(
        {
            "kind": "slide",
            "translation_distance": 0.18,
            "translation_distance_range": [0.15, 0.2],
        },
        path="options",
    )
    assert options.translation_distance_range == (0.15, 0.2)
