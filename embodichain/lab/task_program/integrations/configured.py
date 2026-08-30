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

"""Build composable Task Program integrations from executable-free config."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
import math
from typing import TYPE_CHECKING, TypeVar

from ._configured_services import (
    _AntipodalGraspPoseGeneratorFactory,
    _ArticulationLinkSlideLowererFactory,
    _JointPositionConstraintEvidenceProviderFactory,
    _MoveHeldObjectLowererFactory,
    _PourLowererFactory,
    _PushObjectLowererFactory,
)
from .catalog import (
    SimulationTaskProgramRegistration,
)
from .simulation import (
    AntipodalGraspAffordanceBinding,
    ContainerAffordanceBinding,
    ControlPartCommandPreset,
    ControlPartEndpointBinding,
    ControlPartResourceBinding,
    SimulationArticulationBinding,
    SimulationArticulationLinkBinding,
    SimulationRigidObjectBinding,
    SimulationRobotSkillProfileBinding,
    SimulationSceneBinding,
    SupportSurfaceAffordanceBinding,
)
from .simulation.environment import (
    SimulationTaskProgramAdapterFactory,
)
from .simulation.handover import (
    ConfiguredHandOverPoseProvider,
)
from embodichain.lab.sim.atomic_actions import (
    ActionOptions,
    DynamicCollisionMode,
    ExecutionRunnerCfg,
    HandOverOptions,
    MotionPolicy,
    MoveHeldObjectOptions,
    PickUpOptions,
    PlaceOptions,
    PourOptions,
    PushObjectOptions,
    PushObjectToolCalibration,
    RecoveryPolicy,
    SlideOptions,
    TrackingPolicy,
)
from embodichain.lab.task_program.semantics import (
    EffectAssurance,
    EffectMonitorRef,
    RegisteredSemanticCall,
    SceneCollisionRole,
    SceneCollisionWorldMode,
    SceneDynamics,
    SemanticCallDescriptor,
    SkillPolicyPreset,
    WorkflowRecoveryPolicy,
    builtin_semantic_call_catalog,
)
from embodichain.toolkits.graspkit import (
    ParallelJawGripperModelCfg,
    get_parallel_jaw_gripper_model,
)

if TYPE_CHECKING:
    from .environment import TaskProgramEnvironmentAdapter

__all__: list[str] = []

_CONFIGURED_ANTIPODAL_GRASP_REVISION = "1"
_EnumT = TypeVar("_EnumT", bound=Enum)


def _mapping(
    value: object,
    *,
    path: str,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
) -> Mapping[str, object]:
    """Return one strict string-keyed mapping with an exact field set."""
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a mapping.")
    if not all(type(key) is str for key in value):
        raise TypeError(f"{path} keys must be exact strings.")
    missing = sorted(required.difference(value))
    if missing:
        raise ValueError(f"{path} is missing required fields: {missing}.")
    unexpected = sorted(set(value).difference(required | optional))
    if unexpected:
        raise ValueError(f"{path} contains unsupported fields: {unexpected}.")
    return value


def _sequence(value: object, *, path: str) -> tuple[object, ...]:
    """Return one independently owned non-string sequence."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{path} must be a sequence.")
    return tuple(value)


def _identifier(value: object, *, path: str) -> str:
    """Return one non-empty exact string without outer whitespace."""
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{path} must be a non-empty string without outer whitespace.")
    return value


def _optional_identifier(value: object, *, path: str) -> str | None:
    """Return one optional exact identifier."""
    if value is None:
        return None
    return _identifier(value, path=path)


def _identifier_tuple(value: object, *, path: str) -> tuple[str, ...]:
    """Return one duplicate-free tuple of identifiers."""
    values = tuple(
        _identifier(item, path=f"{path}[{index}]")
        for index, item in enumerate(_sequence(value, path=path))
    )
    if len(set(values)) != len(values):
        raise ValueError(f"{path} must not contain duplicates.")
    return values


def _integer(
    value: object,
    *,
    path: str,
    minimum: int,
    maximum: int | None = None,
) -> int:
    """Return one exact integer within configured bounds."""
    if type(value) is not int or value < minimum:
        raise ValueError(f"{path} must be an integer of at least {minimum}.")
    if maximum is not None and value > maximum:
        raise ValueError(f"{path} must be an integer of at most {maximum}.")
    return value


def _real(
    value: object,
    *,
    path: str,
    minimum: float = -math.inf,
    maximum: float = math.inf,
    strict_minimum: bool = False,
) -> float:
    """Return one finite real value within configured bounds."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{path} must be a finite number.")
    normalized = float(value)
    below = normalized <= minimum if strict_minimum else normalized < minimum
    if not math.isfinite(normalized) or below or normalized > maximum:
        lower = "greater than" if strict_minimum else "at least"
        raise ValueError(
            f"{path} must be finite, {lower} {minimum}, and at most {maximum}."
        )
    return normalized


def _boolean(value: object, *, path: str) -> bool:
    """Return one exact boolean."""
    if type(value) is not bool:
        raise TypeError(f"{path} must be a bool.")
    return value


def _finite_tuple(
    value: object,
    *,
    path: str,
    expected_length: int | None = None,
) -> tuple[float, ...]:
    """Return one non-empty tuple of finite real values."""
    normalized = tuple(
        _real(item, path=f"{path}[{index}]")
        for index, item in enumerate(_sequence(value, path=path))
    )
    if not normalized:
        raise ValueError(f"{path} must not be empty.")
    if expected_length is not None and len(normalized) != expected_length:
        raise ValueError(f"{path} must contain exactly {expected_length} values.")
    return normalized


def _enum(value: object, enum_type: type[_EnumT], *, path: str) -> _EnumT:
    """Decode one exact string-backed enum value."""
    selected = _identifier(value, path=path)
    try:
        return enum_type(selected)
    except ValueError as exc:
        choices = [item.value for item in enum_type]
        raise ValueError(f"{path} must be one of {choices}, got {selected!r}.") from exc


def _nested_identifier_mapping(
    value: object,
    *,
    path: str,
) -> dict[str, dict[str, str]]:
    """Decode one two-level identifier mapping."""
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a mapping.")
    result: dict[str, dict[str, str]] = {}
    for key, nested_value in value.items():
        outer_key = _identifier(key, path=f"{path} keys")
        if not isinstance(nested_value, Mapping):
            raise TypeError(f"{path}.{outer_key} must be a mapping.")
        nested: dict[str, str] = {}
        for nested_key, item in nested_value.items():
            slot_id = _identifier(nested_key, path=f"{path}.{outer_key} keys")
            nested[slot_id] = _identifier(
                item,
                path=f"{path}.{outer_key}.{slot_id}",
            )
        result[outer_key] = nested
    return result


def _identifier_mapping(value: object, *, path: str) -> dict[str, str]:
    """Decode one identifier-to-identifier mapping."""
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a mapping.")
    return {
        _identifier(key, path=f"{path} keys"): _identifier(
            item,
            path=f"{path}.{key}",
        )
        for key, item in value.items()
    }


def _decode_scene_entity_common(
    config: Mapping[str, object],
    *,
    path: str,
) -> dict[str, object]:
    """Decode fields shared by simulation root entities."""
    entity_id = _identifier(config["entity_id"], path=f"{path}.entity_id")
    return {
        "entity_id": entity_id,
        "simulation_uid": _identifier(
            config.get("simulation_uid", entity_id),
            path=f"{path}.simulation_uid",
        ),
        "aliases": _identifier_tuple(config.get("aliases", ()), path=f"{path}.aliases"),
        "dynamics": _enum(
            config.get("dynamics", SceneDynamics.UNKNOWN.value),
            SceneDynamics,
            path=f"{path}.dynamics",
        ),
        "collision_role": _enum(
            config.get("collision_role", SceneCollisionRole.NONE.value),
            SceneCollisionRole,
            path=f"{path}.collision_role",
        ),
        "semantic_type": _optional_identifier(
            config.get("semantic_type"),
            path=f"{path}.semantic_type",
        ),
    }


def _decode_rigid_object(value: object, *, path: str) -> SimulationRigidObjectBinding:
    """Decode one configured rigid-object binding."""
    config = _mapping(
        value,
        path=path,
        required=frozenset({"entity_id"}),
        optional=frozenset(
            {
                "simulation_uid",
                "aliases",
                "dynamics",
                "collision_role",
                "semantic_type",
                "default_grasp_affordance",
                "affordances",
            }
        ),
    )
    return SimulationRigidObjectBinding(
        **_decode_scene_entity_common(config, path=path),
        default_grasp_affordance=_optional_identifier(
            config.get("default_grasp_affordance"),
            path=f"{path}.default_grasp_affordance",
        ),
    )


def _decode_articulation(
    value: object,
    *,
    path: str,
) -> SimulationArticulationBinding:
    """Decode one configured articulation binding."""
    config = _mapping(
        value,
        path=path,
        required=frozenset({"entity_id"}),
        optional=frozenset(
            {
                "simulation_uid",
                "aliases",
                "dynamics",
                "collision_role",
                "semantic_type",
                "affordances",
            }
        ),
    )
    return SimulationArticulationBinding(
        **_decode_scene_entity_common(config, path=path)
    )


def _decode_link(
    value: object,
    *,
    path: str,
) -> SimulationArticulationLinkBinding:
    """Decode one configured articulation-link binding."""
    config = _mapping(
        value,
        path=path,
        required=frozenset({"entity_id", "articulation_id", "native_link_name"}),
        optional=frozenset({"aliases", "dynamics", "semantic_type", "affordances"}),
    )
    return SimulationArticulationLinkBinding(
        entity_id=_identifier(config["entity_id"], path=f"{path}.entity_id"),
        articulation_id=_identifier(
            config["articulation_id"],
            path=f"{path}.articulation_id",
        ),
        native_link_name=_identifier(
            config["native_link_name"],
            path=f"{path}.native_link_name",
        ),
        aliases=_identifier_tuple(config.get("aliases", ()), path=f"{path}.aliases"),
        dynamics=_enum(
            config.get("dynamics", SceneDynamics.UNKNOWN.value),
            SceneDynamics,
            path=f"{path}.dynamics",
        ),
        semantic_type=_optional_identifier(
            config.get("semantic_type"),
            path=f"{path}.semantic_type",
        ),
    )


def _decode_antipodal_grasp(
    value: object,
    *,
    object_id: str,
    path: str,
) -> AntipodalGraspAffordanceBinding:
    """Decode one configured antipodal grasp affordance."""
    config = _mapping(
        value,
        path=path,
        required=frozenset({"kind", "entity_id"}),
        optional=frozenset(
            {
                "native_name",
                "revision",
                "aliases",
                "relative_pose",
                "mesh_env_id",
                "internal_axis",
            }
        ),
    )
    entity_id = _identifier(config["entity_id"], path=f"{path}.entity_id")
    identity_pose = (
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    )
    return AntipodalGraspAffordanceBinding(
        entity_id=entity_id,
        object_id=object_id,
        native_name=_identifier(
            config.get("native_name", entity_id),
            path=f"{path}.native_name",
        ),
        revision=_identifier(
            config.get("revision", _CONFIGURED_ANTIPODAL_GRASP_REVISION),
            path=f"{path}.revision",
        ),
        aliases=_identifier_tuple(config.get("aliases", ()), path=f"{path}.aliases"),
        relative_pose=(
            _finite_tuple(
                config["relative_pose"],
                path=f"{path}.relative_pose",
                expected_length=16,
            )
            if "relative_pose" in config
            else identity_pose
        ),
        mesh_env_id=_integer(
            config.get("mesh_env_id", 0),
            path=f"{path}.mesh_env_id",
            minimum=0,
        ),
        internal_axis=(
            _finite_tuple(
                config["internal_axis"],
                path=f"{path}.internal_axis",
                expected_length=3,
            )
            if "internal_axis" in config
            else None
        ),
    )


def _decode_placement_affordance(
    value: object,
    *,
    parent_id: str,
    expected_kind: str,
    path: str,
    binding_type: (
        type[SupportSurfaceAffordanceBinding] | type[ContainerAffordanceBinding]
    ),
) -> SupportSurfaceAffordanceBinding | ContainerAffordanceBinding:
    """Decode one configured placement affordance."""
    config = _mapping(
        value,
        path=path,
        required=frozenset({"kind", "entity_id", "native_name"}),
        optional=frozenset(
            {"aliases", "object_target_pose", "minimum_confidence", "is_default"}
        ),
    )
    kind = _identifier(config["kind"], path=f"{path}.kind")
    if kind != expected_kind:
        raise ValueError(f"{path}.kind must be {expected_kind!r}, got {kind!r}.")
    kwargs = {
        "entity_id": _identifier(config["entity_id"], path=f"{path}.entity_id"),
        "parent_id": parent_id,
        "native_name": _identifier(
            config["native_name"],
            path=f"{path}.native_name",
        ),
        "aliases": _identifier_tuple(
            config.get("aliases", ()),
            path=f"{path}.aliases",
        ),
        "minimum_confidence": _real(
            config.get("minimum_confidence", 0.0),
            path=f"{path}.minimum_confidence",
            minimum=0.0,
            maximum=1.0,
        ),
        "is_default": _boolean(
            config.get("is_default", False),
            path=f"{path}.is_default",
        ),
    }
    if "object_target_pose" in config:
        kwargs["object_target_pose"] = _finite_tuple(
            config["object_target_pose"],
            path=f"{path}.object_target_pose",
            expected_length=16,
        )
    return binding_type(**kwargs)


def _decode_entity_affordance(
    value: object,
    *,
    parent_id: str,
    parent_category: str,
    path: str,
) -> (
    AntipodalGraspAffordanceBinding
    | SupportSurfaceAffordanceBinding
    | ContainerAffordanceBinding
):
    """Decode one affordance nested under its owning scene entity."""
    common = _mapping(
        value,
        path=path,
        required=frozenset({"kind", "entity_id"}),
        optional=frozenset(
            {
                "native_name",
                "revision",
                "aliases",
                "relative_pose",
                "mesh_env_id",
                "internal_axis",
                "object_target_pose",
                "minimum_confidence",
                "is_default",
            }
        ),
    )
    kind = _identifier(common["kind"], path=f"{path}.kind")
    if kind == "antipodal_grasp":
        if parent_category != "rigid_objects":
            raise ValueError(
                f"{path}.kind {kind!r} is supported only under a rigid object."
            )
        return _decode_antipodal_grasp(
            value,
            object_id=parent_id,
            path=path,
        )
    if kind == "support_surface":
        return _decode_placement_affordance(
            value,
            parent_id=parent_id,
            expected_kind="support_surface",
            path=path,
            binding_type=SupportSurfaceAffordanceBinding,
        )
    if kind == "container":
        return _decode_placement_affordance(
            value,
            parent_id=parent_id,
            expected_kind="container",
            path=path,
            binding_type=ContainerAffordanceBinding,
        )
    raise ValueError(
        f"{path}.kind must be one of "
        "['antipodal_grasp', 'support_surface', 'container'], "
        f"got {kind!r}."
    )


def _decode_entity_affordances(
    value: object,
    *,
    parent_id: str,
    parent_category: str,
    path: str,
) -> tuple[
    AntipodalGraspAffordanceBinding
    | SupportSurfaceAffordanceBinding
    | ContainerAffordanceBinding,
    ...,
]:
    """Decode all affordances nested under one validated scene entity."""
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a mapping.")
    return tuple(
        _decode_entity_affordance(
            affordance_value,
            parent_id=parent_id,
            parent_category=parent_category,
            path=f"{path}.affordances[{index}]",
        )
        for index, affordance_value in enumerate(
            _sequence(
                value.get("affordances", ()),
                path=f"{path}.affordances",
            )
        )
    )


def _decode_scene(value: object) -> SimulationSceneBinding:
    """Decode one complete provider-free simulation scene binding."""
    path = "integration.scene"
    config = _mapping(
        value,
        path=path,
        required=frozenset({"registry_id"}),
        optional=frozenset(
            {
                "rigid_objects",
                "articulations",
                "links",
                "collision_world_mode",
            }
        ),
    )

    antipodal_grasps: list[AntipodalGraspAffordanceBinding] = []
    support_surfaces: list[SupportSurfaceAffordanceBinding] = []
    containers: list[ContainerAffordanceBinding] = []

    def decode_entities(field_name: str, decoder):
        bindings = []
        for index, item in enumerate(
            _sequence(config.get(field_name, ()), path=f"{path}.{field_name}")
        ):
            entity_path = f"{path}.{field_name}[{index}]"
            binding = decoder(item, path=entity_path)
            bindings.append(binding)
            for affordance in _decode_entity_affordances(
                item,
                parent_id=binding.entity_id,
                parent_category=field_name,
                path=entity_path,
            ):
                if isinstance(affordance, AntipodalGraspAffordanceBinding):
                    antipodal_grasps.append(affordance)
                elif isinstance(affordance, SupportSurfaceAffordanceBinding):
                    support_surfaces.append(affordance)
                elif isinstance(affordance, ContainerAffordanceBinding):
                    containers.append(affordance)
                else:
                    raise TypeError(
                        f"{entity_path}.affordances decoded an unsupported "
                        f"binding type {type(affordance).__name__}."
                    )
        return tuple(bindings)

    rigid_objects = decode_entities("rigid_objects", _decode_rigid_object)
    articulations = decode_entities("articulations", _decode_articulation)
    links = decode_entities("links", _decode_link)

    return SimulationSceneBinding(
        registry_id=_identifier(config["registry_id"], path=f"{path}.registry_id"),
        rigid_objects=rigid_objects,
        articulations=articulations,
        links=links,
        antipodal_grasps=tuple(antipodal_grasps),
        support_surfaces=tuple(support_surfaces),
        containers=tuple(containers),
        collision_world_mode=(
            None
            if "collision_world_mode" not in config
            else _enum(
                config["collision_world_mode"],
                SceneCollisionWorldMode,
                path=f"{path}.collision_world_mode",
            )
        ),
    )


def _decode_endpoint(value: object, *, path: str) -> ControlPartEndpointBinding:
    """Decode one control-part endpoint binding."""
    config = _mapping(
        value,
        path=path,
        required=frozenset({"endpoint_id", "control_part", "capabilities"}),
        optional=frozenset({"command_preset"}),
    )
    return ControlPartEndpointBinding(
        endpoint_id=_identifier(config["endpoint_id"], path=f"{path}.endpoint_id"),
        control_part=_identifier(config["control_part"], path=f"{path}.control_part"),
        capabilities=frozenset(
            _identifier_tuple(config["capabilities"], path=f"{path}.capabilities")
        ),
        command_preset=_optional_identifier(
            config.get("command_preset"),
            path=f"{path}.command_preset",
        ),
    )


def _decode_resource(value: object, *, path: str) -> ControlPartResourceBinding:
    """Decode one control-part resource binding."""
    config = _mapping(
        value,
        path=path,
        required=frozenset({"resource_id", "endpoints"}),
        optional=frozenset({"members"}),
    )
    return ControlPartResourceBinding(
        resource_id=_identifier(config["resource_id"], path=f"{path}.resource_id"),
        endpoints=tuple(
            _decode_endpoint(item, path=f"{path}.endpoints[{index}]")
            for index, item in enumerate(
                _sequence(config["endpoints"], path=f"{path}.endpoints")
            )
        ),
        members=_identifier_tuple(config.get("members", ()), path=f"{path}.members"),
    )


def _decode_command_preset(
    value: object,
    *,
    path: str,
) -> ControlPartCommandPreset:
    """Decode one named control-part command preset."""
    config = _mapping(
        value,
        path=path,
        required=frozenset({"preset_id", "control_part", "commands"}),
    )
    commands_value = config["commands"]
    if not isinstance(commands_value, Mapping):
        raise TypeError(f"{path}.commands must be a mapping.")
    commands = {
        _identifier(command_id, path=f"{path}.commands keys"): _finite_tuple(
            positions,
            path=f"{path}.commands.{command_id}",
        )
        for command_id, positions in commands_value.items()
    }
    return ControlPartCommandPreset(
        preset_id=_identifier(config["preset_id"], path=f"{path}.preset_id"),
        control_part=_identifier(config["control_part"], path=f"{path}.control_part"),
        commands=commands,
    )


def _decode_action_options(value: object, *, path: str) -> ActionOptions:
    """Decode one allowlisted atomic-action option template."""
    common = _mapping(
        value,
        path=path,
        required=frozenset({"kind"}),
        optional=frozenset(
            {
                "hand_interp_steps",
                "grasp_settle_steps",
                "release_settle_steps",
                "pick_object_part",
                "lift_height",
                "pre_grasp_distance",
                "approach_direction",
                "approach_alignment_max_angle",
                "obj_upright_direction",
                "rotate_upright",
                "grasp_frame_to_eef",
                "fixed_object_to_eef",
                "max_approach_retract_z",
                "cartesian_waypoint_count",
                "preserve_current_object_orientation",
                "direction",
                "approach_distance",
                "translation_distance",
                "rotate_angle",
                "approach_height",
                "retract_height",
                "contact_distance",
                "push_overshoot",
                "completion_tolerance",
                "object_contact_offset",
                "support_frame_planar_contact_offset",
                "contact_frame_to_eef",
                "tool_calibrations",
            }
        ),
    )
    kind = _identifier(common["kind"], path=f"{path}.kind")
    if kind == "pick_up":
        config = _mapping(
            value,
            path=path,
            required=frozenset({"kind"}),
            optional=frozenset(
                {
                    "hand_interp_steps",
                    "grasp_settle_steps",
                    "pick_object_part",
                    "lift_height",
                    "pre_grasp_distance",
                    "approach_direction",
                    "approach_alignment_max_angle",
                    "obj_upright_direction",
                    "rotate_upright",
                    "grasp_frame_to_eef",
                    "fixed_object_to_eef",
                }
            ),
        )
        kwargs: dict[str, object] = {}
        if "hand_interp_steps" in config:
            kwargs["hand_interp_steps"] = _integer(
                config["hand_interp_steps"],
                path=f"{path}.hand_interp_steps",
                minimum=1,
            )
        if "grasp_settle_steps" in config:
            kwargs["grasp_settle_steps"] = _integer(
                config["grasp_settle_steps"],
                path=f"{path}.grasp_settle_steps",
                minimum=0,
            )
        if "pick_object_part" in config:
            kwargs["pick_object_part"] = _identifier(
                config["pick_object_part"],
                path=f"{path}.pick_object_part",
            )
        for field_name in ("lift_height", "pre_grasp_distance"):
            if field_name in config:
                kwargs[field_name] = _real(
                    config[field_name],
                    path=f"{path}.{field_name}",
                    minimum=0.0,
                )
        if "approach_direction" in config:
            import torch

            kwargs["approach_direction"] = torch.tensor(
                _finite_tuple(
                    config["approach_direction"],
                    path=f"{path}.approach_direction",
                    expected_length=3,
                ),
                dtype=torch.float32,
            )
        if "approach_alignment_max_angle" in config:
            kwargs["approach_alignment_max_angle"] = _real(
                config["approach_alignment_max_angle"],
                path=f"{path}.approach_alignment_max_angle",
                minimum=0.0,
                maximum=math.pi / 2,
            )
        if "obj_upright_direction" in config:
            import torch

            kwargs["obj_upright_direction"] = torch.tensor(
                _finite_tuple(
                    config["obj_upright_direction"],
                    path=f"{path}.obj_upright_direction",
                    expected_length=3,
                ),
                dtype=torch.float32,
            )
        if "rotate_upright" in config:
            kwargs["rotate_upright"] = _real(
                config["rotate_upright"],
                path=f"{path}.rotate_upright",
            )
        if "grasp_frame_to_eef" in config:
            import torch

            kwargs["grasp_frame_to_eef"] = torch.tensor(
                _finite_tuple(
                    config["grasp_frame_to_eef"],
                    path=f"{path}.grasp_frame_to_eef",
                    expected_length=16,
                ),
                dtype=torch.float32,
            ).reshape(4, 4)
        if "fixed_object_to_eef" in config:
            import torch

            kwargs["fixed_object_to_eef"] = torch.tensor(
                _finite_tuple(
                    config["fixed_object_to_eef"],
                    path=f"{path}.fixed_object_to_eef",
                    expected_length=16,
                ),
                dtype=torch.float32,
            ).reshape(4, 4)
        return PickUpOptions(**kwargs)
    if kind == "place":
        config = _mapping(
            value,
            path=path,
            required=frozenset({"kind"}),
            optional=frozenset(
                {
                    "hand_interp_steps",
                    "release_settle_steps",
                    "lift_height",
                    "max_approach_retract_z",
                    "cartesian_waypoint_count",
                    "preserve_current_object_orientation",
                }
            ),
        )
        kwargs = {}
        if "hand_interp_steps" in config:
            kwargs["hand_interp_steps"] = _integer(
                config["hand_interp_steps"],
                path=f"{path}.hand_interp_steps",
                minimum=1,
            )
        if "release_settle_steps" in config:
            kwargs["release_settle_steps"] = _integer(
                config["release_settle_steps"],
                path=f"{path}.release_settle_steps",
                minimum=0,
            )
        if "lift_height" in config:
            kwargs["lift_height"] = _real(
                config["lift_height"],
                path=f"{path}.lift_height",
                minimum=0.0,
            )
        if "max_approach_retract_z" in config:
            kwargs["max_approach_retract_z"] = _real(
                config["max_approach_retract_z"],
                path=f"{path}.max_approach_retract_z",
            )
        if "cartesian_waypoint_count" in config:
            kwargs["cartesian_waypoint_count"] = _integer(
                config["cartesian_waypoint_count"],
                path=f"{path}.cartesian_waypoint_count",
                minimum=1,
            )
        if "preserve_current_object_orientation" in config:
            kwargs["preserve_current_object_orientation"] = _boolean(
                config["preserve_current_object_orientation"],
                path=f"{path}.preserve_current_object_orientation",
            )
        return PlaceOptions(**kwargs)
    if kind == "move_held_object":
        _mapping(
            value,
            path=path,
            required=frozenset({"kind"}),
        )
        return MoveHeldObjectOptions()
    if kind == "pour":
        config = _mapping(
            value,
            path=path,
            required=frozenset({"kind"}),
            optional=frozenset({"rotate_angle"}),
        )
        kwargs = {}
        if "rotate_angle" in config:
            kwargs["rotate_angle"] = _real(
                config["rotate_angle"],
                path=f"{path}.rotate_angle",
            )
        return PourOptions(**kwargs)
    if kind == "push_object":
        config = _mapping(
            value,
            path=path,
            required=frozenset({"kind"}),
            optional=frozenset(
                {
                    "hand_interp_steps",
                    "approach_height",
                    "retract_height",
                    "contact_distance",
                    "push_overshoot",
                    "completion_tolerance",
                    "object_contact_offset",
                    "support_frame_planar_contact_offset",
                    "contact_frame_to_eef",
                    "tool_calibrations",
                }
            ),
        )
        kwargs: dict[str, object] = {}
        if "hand_interp_steps" in config:
            kwargs["hand_interp_steps"] = _integer(
                config["hand_interp_steps"],
                path=f"{path}.hand_interp_steps",
                minimum=1,
            )
        for field_name in (
            "approach_height",
            "retract_height",
            "contact_distance",
            "push_overshoot",
            "completion_tolerance",
        ):
            if field_name in config:
                kwargs[field_name] = _real(
                    config[field_name],
                    path=f"{path}.{field_name}",
                    minimum=0.0,
                )
        if "object_contact_offset" in config:
            import torch

            kwargs["object_contact_offset"] = torch.tensor(
                _finite_tuple(
                    config["object_contact_offset"],
                    path=f"{path}.object_contact_offset",
                    expected_length=3,
                ),
                dtype=torch.float32,
            )
        if "support_frame_planar_contact_offset" in config:
            import torch

            kwargs["support_frame_planar_contact_offset"] = torch.tensor(
                _finite_tuple(
                    config["support_frame_planar_contact_offset"],
                    path=f"{path}.support_frame_planar_contact_offset",
                    expected_length=3,
                ),
                dtype=torch.float32,
            )
        if "contact_frame_to_eef" in config:
            import torch

            kwargs["contact_frame_to_eef"] = torch.tensor(
                _finite_tuple(
                    config["contact_frame_to_eef"],
                    path=f"{path}.contact_frame_to_eef",
                    expected_length=16,
                ),
                dtype=torch.float32,
            ).reshape(4, 4)
        if "tool_calibrations" in config:
            import torch

            calibrations: list[PushObjectToolCalibration] = []
            for index, item in enumerate(
                _sequence(
                    config["tool_calibrations"],
                    path=f"{path}.tool_calibrations",
                )
            ):
                calibration_path = f"{path}.tool_calibrations[{index}]"
                calibration = _mapping(
                    item,
                    path=calibration_path,
                    required=frozenset({"control_part", "contact_frame_to_eef"}),
                    optional=frozenset({"contact_distance"}),
                )
                calibrations.append(
                    PushObjectToolCalibration(
                        control_part=_identifier(
                            calibration["control_part"],
                            path=f"{calibration_path}.control_part",
                        ),
                        contact_frame_to_eef=torch.tensor(
                            _finite_tuple(
                                calibration["contact_frame_to_eef"],
                                path=f"{calibration_path}.contact_frame_to_eef",
                                expected_length=16,
                            ),
                            dtype=torch.float32,
                        ).reshape(4, 4),
                        contact_distance=(
                            None
                            if "contact_distance" not in calibration
                            else _real(
                                calibration["contact_distance"],
                                path=f"{calibration_path}.contact_distance",
                                minimum=0.0,
                            )
                        ),
                    )
                )
            kwargs["tool_calibrations"] = tuple(calibrations)
        return PushObjectOptions(**kwargs)
    if kind == "slide":
        config = _mapping(
            value,
            path=path,
            required=frozenset({"kind"}),
            optional=frozenset(
                {
                    "direction",
                    "hand_interp_steps",
                    "approach_distance",
                    "translation_distance",
                }
            ),
        )
        kwargs = {}
        if "direction" in config:
            kwargs["direction"] = _identifier(
                config["direction"],
                path=f"{path}.direction",
            )
        if "hand_interp_steps" in config:
            kwargs["hand_interp_steps"] = _integer(
                config["hand_interp_steps"],
                path=f"{path}.hand_interp_steps",
                minimum=1,
            )
        if "approach_distance" in config:
            kwargs["approach_distance"] = _real(
                config["approach_distance"],
                path=f"{path}.approach_distance",
                minimum=0.0,
            )
        if "translation_distance" in config:
            kwargs["translation_distance"] = _real(
                config["translation_distance"],
                path=f"{path}.translation_distance",
                minimum=0.0,
                strict_minimum=True,
            )
        return SlideOptions(**kwargs)
    if kind == "hand_over":
        config = _mapping(
            value,
            path=path,
            required=frozenset({"kind"}),
            optional=frozenset(
                {"pre_grasp_distance", "lift_height", "hand_interp_steps"}
            ),
        )
        kwargs = {}
        for field_name in ("pre_grasp_distance", "lift_height"):
            if field_name in config:
                kwargs[field_name] = _real(
                    config[field_name],
                    path=f"{path}.{field_name}",
                    minimum=0.0,
                )
        if "hand_interp_steps" in config:
            kwargs["hand_interp_steps"] = _integer(
                config["hand_interp_steps"],
                path=f"{path}.hand_interp_steps",
                minimum=1,
            )
        return HandOverOptions(**kwargs)
    raise ValueError(
        f"Unsupported {path}.kind {kind!r}; supported kinds are "
        "['hand_over', 'move_held_object', 'pick_up', 'place', 'pour', "
        "'push_object', 'slide']."
    )


def _decode_motion_policy(value: object, *, path: str) -> MotionPolicy:
    """Decode one motion-generation policy."""
    config = _mapping(
        value,
        path=path,
        required=frozenset({"sample_count"}),
        optional=frozenset({"strategy", "dynamic_collision_mode"}),
    )
    return MotionPolicy(
        strategy=_identifier(
            config.get("strategy", "ik_interp"),
            path=f"{path}.strategy",
        ),
        sample_count=_integer(
            config["sample_count"],
            path=f"{path}.sample_count",
            minimum=2,
        ),
        dynamic_collision_mode=_enum(
            config.get("dynamic_collision_mode", DynamicCollisionMode.AUTO.value),
            DynamicCollisionMode,
            path=f"{path}.dynamic_collision_mode",
        ),
    )


def _decode_tracking_policy(value: object, *, path: str) -> TrackingPolicy:
    """Decode one allowlisted tracking policy."""
    common = _mapping(
        value,
        path=path,
        required=frozenset({"kind"}),
        optional=frozenset(
            {
                "settle_duration",
                "in_flight_max_abs_error",
                "terminal_max_abs_error",
                "terminal_settle_timeout",
                "consecutive_violations",
                "consecutive_acceptances",
                "grace_period",
            }
        ),
    )
    kind = _identifier(common["kind"], path=f"{path}.kind")
    if kind == "timed":
        config = _mapping(
            value,
            path=path,
            required=frozenset({"kind"}),
            optional=frozenset({"settle_duration"}),
        )
        return TrackingPolicy.timed(
            settle_duration=_real(
                config.get("settle_duration", 0.0),
                path=f"{path}.settle_duration",
                minimum=0.0,
            )
        )
    if kind == "joint_position":
        config = _mapping(
            value,
            path=path,
            required=frozenset(
                {"kind", "in_flight_max_abs_error", "terminal_max_abs_error"}
            ),
            optional=frozenset(
                {
                    "terminal_settle_timeout",
                    "consecutive_violations",
                    "consecutive_acceptances",
                    "grace_period",
                }
            ),
        )
        return TrackingPolicy.joint_position(
            in_flight_max_abs_error=_real(
                config["in_flight_max_abs_error"],
                path=f"{path}.in_flight_max_abs_error",
                minimum=0.0,
                strict_minimum=True,
            ),
            terminal_max_abs_error=_real(
                config["terminal_max_abs_error"],
                path=f"{path}.terminal_max_abs_error",
                minimum=0.0,
                strict_minimum=True,
            ),
            terminal_settle_timeout=_real(
                config.get("terminal_settle_timeout", 0.5),
                path=f"{path}.terminal_settle_timeout",
                minimum=0.0,
            ),
            consecutive_violations=_integer(
                config.get("consecutive_violations", 1),
                path=f"{path}.consecutive_violations",
                minimum=1,
            ),
            consecutive_acceptances=_integer(
                config.get("consecutive_acceptances", 1),
                path=f"{path}.consecutive_acceptances",
                minimum=1,
            ),
            grace_period=_real(
                config.get("grace_period", 0.0),
                path=f"{path}.grace_period",
                minimum=0.0,
            ),
        )
    raise ValueError(
        f"Unsupported {path}.kind {kind!r}; supported kinds are "
        "['joint_position', 'timed']."
    )


def _decode_recovery_policy(value: object, *, path: str) -> RecoveryPolicy:
    """Decode one bounded local recovery policy."""
    config = _mapping(
        value,
        path=path,
        required=frozenset({"max_replans", "max_action_retries"}),
        optional=frozenset(
            {
                "goal_translation_threshold",
                "goal_rotation_threshold",
                "action_timeout",
            }
        ),
    )
    kwargs: dict[str, object] = {
        "max_replans": _integer(
            config["max_replans"],
            path=f"{path}.max_replans",
            minimum=0,
        ),
        "max_action_retries": _integer(
            config["max_action_retries"],
            path=f"{path}.max_action_retries",
            minimum=0,
        ),
    }
    for field_name in (
        "goal_translation_threshold",
        "goal_rotation_threshold",
        "action_timeout",
    ):
        if field_name in config:
            kwargs[field_name] = _real(
                config[field_name],
                path=f"{path}.{field_name}",
                minimum=0.0,
                strict_minimum=True,
            )
    return RecoveryPolicy(**kwargs)


def _decode_runner_cfg(value: object, *, path: str) -> ExecutionRunnerCfg:
    """Decode one execution-runner policy."""
    config = _mapping(
        value,
        path=path,
        required=frozenset({"hold_on_completion", "hold_during_effect_verification"}),
        optional=frozenset(
            {"command_timeout", "safe_stop_timeout", "minimum_cycle_time"}
        ),
    )
    kwargs: dict[str, object] = {
        "hold_on_completion": _boolean(
            config["hold_on_completion"],
            path=f"{path}.hold_on_completion",
        ),
        "hold_during_effect_verification": _boolean(
            config["hold_during_effect_verification"],
            path=f"{path}.hold_during_effect_verification",
        ),
    }
    for field_name in ("command_timeout", "safe_stop_timeout"):
        if field_name in config:
            kwargs[field_name] = _real(
                config[field_name],
                path=f"{path}.{field_name}",
                minimum=0.0,
                strict_minimum=True,
            )
    if "minimum_cycle_time" in config:
        kwargs["minimum_cycle_time"] = _real(
            config["minimum_cycle_time"],
            path=f"{path}.minimum_cycle_time",
            minimum=0.0,
        )
    return ExecutionRunnerCfg(**kwargs)


def _decode_effect_monitors(
    value: object,
    *,
    path: str,
) -> dict[str, EffectMonitorRef]:
    """Decode an explicit semantic-effect monitor selection."""
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a mapping.")
    monitors: dict[str, EffectMonitorRef] = {}
    for semantic_id, item in value.items():
        selected_id = _identifier(semantic_id, path=f"{path} keys")
        monitor_path = f"{path}.{selected_id}"
        config = _mapping(
            item,
            path=monitor_path,
            required=frozenset({"monitor_id", "revision"}),
            optional=frozenset({"params"}),
        )
        params = config.get("params", {})
        if not isinstance(params, Mapping):
            raise TypeError(f"{monitor_path}.params must be a mapping.")
        monitors[selected_id] = EffectMonitorRef(
            _identifier(config["monitor_id"], path=f"{monitor_path}.monitor_id"),
            _identifier(config["revision"], path=f"{monitor_path}.revision"),
            params,
        )
    return monitors


def _decode_policy_preset(value: object, *, path: str) -> SkillPolicyPreset:
    """Decode one complete skill policy preset."""
    config = _mapping(
        value,
        path=path,
        required=frozenset(
            {
                "preset_id",
                "action_options",
                "motion",
                "tracking",
                "recovery",
                "workflow_recovery",
                "runner",
                "effect_assurance",
                "effect_monitors",
            }
        ),
        optional=frozenset({"required_planner"}),
    )
    action_options_value = config["action_options"]
    if not isinstance(action_options_value, Mapping):
        raise TypeError(f"{path}.action_options must be a mapping.")
    action_options = {
        _identifier(semantic_id, path=f"{path}.action_options keys"): (
            _decode_action_options(
                item,
                path=f"{path}.action_options.{semantic_id}",
            )
        )
        for semantic_id, item in action_options_value.items()
    }
    workflow = _mapping(
        config["workflow_recovery"],
        path=f"{path}.workflow_recovery",
        required=frozenset({"max_recovery_attempts"}),
    )
    return SkillPolicyPreset(
        _identifier(config["preset_id"], path=f"{path}.preset_id"),
        action_option_templates=action_options,
        effect_assurance=_enum(
            config["effect_assurance"],
            EffectAssurance,
            path=f"{path}.effect_assurance",
        ),
        motion_policy=_decode_motion_policy(config["motion"], path=f"{path}.motion"),
        tracking_policy=_decode_tracking_policy(
            config["tracking"],
            path=f"{path}.tracking",
        ),
        recovery_policy=_decode_recovery_policy(
            config["recovery"],
            path=f"{path}.recovery",
        ),
        workflow_recovery_policy=WorkflowRecoveryPolicy(
            max_recovery_attempts=_integer(
                workflow["max_recovery_attempts"],
                path=f"{path}.workflow_recovery.max_recovery_attempts",
                minimum=0,
                maximum=100,
            )
        ),
        runner_cfg=_decode_runner_cfg(config["runner"], path=f"{path}.runner"),
        effect_monitors=_decode_effect_monitors(
            config["effect_monitors"],
            path=f"{path}.effect_monitors",
        ),
        required_planner=_optional_identifier(
            config.get("required_planner"),
            path=f"{path}.required_planner",
        ),
    )


def _decode_robot_profile(value: object) -> SimulationRobotSkillProfileBinding:
    """Decode one complete provider-free robot skill profile binding."""
    path = "integration.robot_profile"
    config = _mapping(
        value,
        path=path,
        required=frozenset(
            {
                "profile_id",
                "resources",
                "command_presets",
                "defaults",
                "presets",
                "default_preset",
            }
        ),
        optional=frozenset({"skill_presets", "grounding_providers"}),
    )
    return SimulationRobotSkillProfileBinding(
        profile_id=_identifier(config["profile_id"], path=f"{path}.profile_id"),
        resources=tuple(
            _decode_resource(item, path=f"{path}.resources[{index}]")
            for index, item in enumerate(
                _sequence(config["resources"], path=f"{path}.resources")
            )
        ),
        command_presets=tuple(
            _decode_command_preset(item, path=f"{path}.command_presets[{index}]")
            for index, item in enumerate(
                _sequence(
                    config["command_presets"],
                    path=f"{path}.command_presets",
                )
            )
        ),
        defaults=_nested_identifier_mapping(
            config["defaults"], path=f"{path}.defaults"
        ),
        presets=tuple(
            _decode_policy_preset(item, path=f"{path}.presets[{index}]")
            for index, item in enumerate(
                _sequence(config["presets"], path=f"{path}.presets")
            )
        ),
        default_preset=_optional_identifier(
            config["default_preset"],
            path=f"{path}.default_preset",
        ),
        skill_presets=_identifier_mapping(
            config.get("skill_presets", {}),
            path=f"{path}.skill_presets",
        ),
        grounding_providers=_identifier_mapping(
            config.get("grounding_providers", {}),
            path=f"{path}.grounding_providers",
        ),
    )


def _decode_parallel_jaw_gripper_model(
    value: object,
    *,
    path: str,
) -> ParallelJawGripperModelCfg:
    """Resolve one built-in model ID or validate one inline geometry model."""
    if type(value) is str:
        try:
            return get_parallel_jaw_gripper_model(value)
        except ValueError as exc:
            raise ValueError(f"{path}: {exc}") from exc

    model = _mapping(
        value,
        path=path,
        required=frozenset({"model_id"}),
        optional=frozenset(
            {
                "min_opening_width",
                "max_opening_width",
                "finger_length",
                "finger_width",
                "finger_thickness",
                "palm_depth",
            }
        ),
    )
    defaults = ParallelJawGripperModelCfg()
    min_opening_width = _real(
        model.get("min_opening_width", defaults.min_opening_width),
        path=f"{path}.min_opening_width",
        minimum=0.0,
        strict_minimum=True,
    )
    max_opening_width = _real(
        model.get("max_opening_width", defaults.max_opening_width),
        path=f"{path}.max_opening_width",
        minimum=0.0,
        strict_minimum=True,
    )
    if min_opening_width >= max_opening_width:
        raise ValueError(
            f"{path}.min_opening_width must be less than {path}.max_opening_width."
        )
    return ParallelJawGripperModelCfg(
        model_id=_identifier(model["model_id"], path=f"{path}.model_id"),
        min_opening_width=min_opening_width,
        max_opening_width=max_opening_width,
        finger_length=_real(
            model.get("finger_length", defaults.finger_length),
            path=f"{path}.finger_length",
            minimum=0.0,
            strict_minimum=True,
        ),
        finger_width=_real(
            model.get("finger_width", defaults.finger_width),
            path=f"{path}.finger_width",
            minimum=0.0,
            strict_minimum=True,
        ),
        finger_thickness=_real(
            model.get("finger_thickness", defaults.finger_thickness),
            path=f"{path}.finger_thickness",
            minimum=0.0,
            strict_minimum=True,
        ),
        palm_depth=_real(
            model.get("palm_depth", defaults.palm_depth),
            path=f"{path}.palm_depth",
            minimum=0.0,
            strict_minimum=True,
        ),
    )


def _decode_grasp_generator(
    value: object,
    *,
    path: str,
) -> _AntipodalGraspPoseGeneratorFactory:
    """Decode one allowlisted parallel-jaw antipodal generator declaration."""
    config = _mapping(
        value,
        path=path,
        required=frozenset({"kind", "model"}),
        optional=frozenset(
            {
                "sample_count",
                "approach_direction_samples",
                "opening_margin",
                "point_sample_density",
                "filter_ground_collision",
                "force_refresh",
            }
        ),
    )
    kind = _identifier(config["kind"], path=f"{path}.kind")
    if kind != "antipodal_parallel_jaw":
        raise ValueError(
            f"Unsupported {path}.kind {kind!r}; supported kinds are "
            "['antipodal_parallel_jaw']."
        )
    model = _decode_parallel_jaw_gripper_model(
        config["model"],
        path=f"{path}.model",
    )
    return _AntipodalGraspPoseGeneratorFactory(
        model_id=model.model_id,
        min_opening_width=model.min_opening_width,
        max_opening_width=model.max_opening_width,
        finger_length=model.finger_length,
        finger_width=model.finger_width,
        finger_thickness=model.finger_thickness,
        palm_depth=model.palm_depth,
        sample_count=(
            _integer(
                config["sample_count"],
                path=f"{path}.sample_count",
                minimum=1,
            )
            if "sample_count" in config
            else None
        ),
        approach_direction_samples=(
            _integer(
                config["approach_direction_samples"],
                path=f"{path}.approach_direction_samples",
                minimum=1,
            )
            if "approach_direction_samples" in config
            else None
        ),
        opening_margin=(
            _real(
                config["opening_margin"],
                path=f"{path}.opening_margin",
                minimum=0.0,
            )
            if "opening_margin" in config
            else None
        ),
        point_sample_density=(
            _real(
                config["point_sample_density"],
                path=f"{path}.point_sample_density",
                minimum=0.0,
                strict_minimum=True,
            )
            if "point_sample_density" in config
            else None
        ),
        filter_ground_collision=(
            _boolean(
                config["filter_ground_collision"],
                path=f"{path}.filter_ground_collision",
            )
            if "filter_ground_collision" in config
            else None
        ),
        force_refresh=(
            _boolean(config["force_refresh"], path=f"{path}.force_refresh")
            if "force_refresh" in config
            else None
        ),
    )


def _decode_handover_pose_provider(
    value: object,
    *,
    path: str,
) -> ConfiguredHandOverPoseProvider:
    """Decode one configured object-space hand-over pose provider."""
    config = _mapping(
        value,
        path=path,
        required=frozenset(
            {
                "kind",
                "final_position",
                "final_quaternion_wxyz",
            }
        ),
    )
    kind = _identifier(config["kind"], path=f"{path}.kind")
    if kind != "configured_pose":
        raise ValueError(
            f"Unsupported {path}.kind {kind!r}; supported kinds are "
            "['configured_pose']."
        )
    return ConfiguredHandOverPoseProvider(
        final_position=_finite_tuple(
            config["final_position"],
            path=f"{path}.final_position",
            expected_length=3,
        ),
        final_quaternion_wxyz=_finite_tuple(
            config["final_quaternion_wxyz"],
            path=f"{path}.final_quaternion_wxyz",
            expected_length=4,
        ),
    )


def _decode_registered_lowerer(
    value: object,
    *,
    path: str,
) -> (
    _ArticulationLinkSlideLowererFactory
    | _MoveHeldObjectLowererFactory
    | _PourLowererFactory
    | _PushObjectLowererFactory
):
    """Decode one allowlisted registered semantic lowerer factory."""
    common = _mapping(
        value,
        path=path,
        required=frozenset({"kind"}),
        optional=frozenset(
            {
                "articulation_id",
                "articulation_simulation_uid",
                "link_entity_id",
                "translation_axis",
                "target_pose_mode",
                "target_id",
                "reference_entity_id",
                "relative_pose",
                "object_id",
                "routes",
            }
        ),
    )
    kind = _identifier(common["kind"], path=f"{path}.kind")
    if kind == "articulation_link_slide":
        config = _mapping(
            value,
            path=path,
            required=frozenset(
                {
                    "kind",
                    "articulation_id",
                    "articulation_simulation_uid",
                    "link_entity_id",
                    "translation_axis",
                }
            ),
            optional=frozenset({"target_pose_mode"}),
        )
        target_pose_mode = _identifier(
            config.get("target_pose_mode", "live"),
            path=f"{path}.target_pose_mode",
        )
        if target_pose_mode not in {"live", "snapshot"}:
            raise ValueError(
                f"{path}.target_pose_mode must be one of ['live', 'snapshot']."
            )
        return _ArticulationLinkSlideLowererFactory(
            articulation_id=_identifier(
                config["articulation_id"],
                path=f"{path}.articulation_id",
            ),
            articulation_simulation_uid=_identifier(
                config["articulation_simulation_uid"],
                path=f"{path}.articulation_simulation_uid",
            ),
            link_entity_id=_identifier(
                config["link_entity_id"],
                path=f"{path}.link_entity_id",
            ),
            translation_axis=_finite_tuple(
                config["translation_axis"],
                path=f"{path}.translation_axis",
                expected_length=3,
            ),
            target_pose_mode=target_pose_mode,
        )
    if kind == "move_held_object":
        config = _mapping(
            value,
            path=path,
            required=frozenset(
                {
                    "kind",
                    "target_id",
                    "reference_entity_id",
                    "relative_pose",
                }
            ),
        )
        return _MoveHeldObjectLowererFactory(
            target_id=_identifier(config["target_id"], path=f"{path}.target_id"),
            reference_entity_id=_identifier(
                config["reference_entity_id"],
                path=f"{path}.reference_entity_id",
            ),
            relative_pose=_finite_tuple(
                config["relative_pose"],
                path=f"{path}.relative_pose",
                expected_length=16,
            ),
        )
    if kind == "pour":
        config = _mapping(
            value,
            path=path,
            required=frozenset({"kind", "object_id"}),
        )
        return _PourLowererFactory(
            object_id=_identifier(config["object_id"], path=f"{path}.object_id")
        )
    if kind == "push_object":
        config = _mapping(
            value,
            path=path,
            required=frozenset({"kind", "routes"}),
        )
        routes: list[tuple[str, str]] = []
        for index, route_value in enumerate(
            _sequence(config["routes"], path=f"{path}.routes")
        ):
            route_path = f"{path}.routes[{index}]"
            route = _mapping(
                route_value,
                path=route_path,
                required=frozenset({"object_id", "target_entity_id"}),
            )
            routes.append(
                (
                    _identifier(
                        route["object_id"],
                        path=f"{route_path}.object_id",
                    ),
                    _identifier(
                        route["target_entity_id"],
                        path=f"{route_path}.target_entity_id",
                    ),
                )
            )
        return _PushObjectLowererFactory(routes=tuple(routes))
    raise ValueError(
        f"Unsupported {path}.kind {kind!r}; supported kinds are "
        "['articulation_link_slide', 'move_held_object', 'pour', 'push_object']."
    )


def _decode_control_part_evidence(
    value: object,
    *,
    path: str,
) -> _JointPositionConstraintEvidenceProviderFactory:
    """Decode one allowlisted control-part evidence provider factory."""
    config = _mapping(
        value,
        path=path,
        required=frozenset(
            {
                "kind",
                "control_parts",
                "object_ids",
                "open_qpos",
                "minimum_displacement",
            }
        ),
    )
    kind = _identifier(config["kind"], path=f"{path}.kind")
    if kind != "joint_position_constraint":
        raise ValueError(
            f"Unsupported {path}.kind {kind!r}; supported kinds are "
            "['joint_position_constraint']."
        )
    return _JointPositionConstraintEvidenceProviderFactory(
        control_parts=_identifier_tuple(
            config["control_parts"],
            path=f"{path}.control_parts",
        ),
        object_ids=_identifier_tuple(
            config["object_ids"],
            path=f"{path}.object_ids",
        ),
        open_qpos=_finite_tuple(config["open_qpos"], path=f"{path}.open_qpos"),
        minimum_displacement=_real(
            config["minimum_displacement"],
            path=f"{path}.minimum_displacement",
            minimum=0.0,
            strict_minimum=True,
        ),
    )


@dataclass(frozen=True, slots=True)
class _DecodedRuntimeServices:
    """Typed configured services separated by registration ownership."""

    grasp_pose_generators: tuple[
        tuple[str, _AntipodalGraspPoseGeneratorFactory], ...
    ] = ()
    handover_pose_providers: tuple[ConfiguredHandOverPoseProvider, ...] = ()
    registered_semantic_lowerers: tuple[
        _ArticulationLinkSlideLowererFactory
        | _MoveHeldObjectLowererFactory
        | _PourLowererFactory
        | _PushObjectLowererFactory,
        ...,
    ] = ()
    control_part_evidence: _JointPositionConstraintEvidenceProviderFactory | None = None


def _decode_runtime_services(value: object) -> _DecodedRuntimeServices:
    """Decode the allowlisted live-service portion of an integration."""
    path = "integration.runtime_services"
    config = _mapping(
        value,
        path=path,
        required=frozenset(),
        optional=frozenset(
            {
                "grasp_pose_generators",
                "handover_pose_providers",
                "registered_semantic_lowerers",
                "control_part_evidence",
            }
        ),
    )
    generators_value = config.get("grasp_pose_generators", {})
    if not isinstance(generators_value, Mapping):
        raise TypeError(f"{path}.grasp_pose_generators must be a mapping.")
    generators = tuple(
        sorted(
            (
                _identifier(target_id, path=f"{path}.grasp_pose_generators keys"),
                _decode_grasp_generator(
                    item,
                    path=f"{path}.grasp_pose_generators.{target_id}",
                ),
            )
            for target_id, item in generators_value.items()
        )
    )
    handover = tuple(
        _decode_handover_pose_provider(
            item,
            path=f"{path}.handover_pose_providers[{index}]",
        )
        for index, item in enumerate(
            _sequence(
                config.get("handover_pose_providers", ()),
                path=f"{path}.handover_pose_providers",
            )
        )
    )
    lowerers = tuple(
        _decode_registered_lowerer(
            item,
            path=f"{path}.registered_semantic_lowerers[{index}]",
        )
        for index, item in enumerate(
            _sequence(
                config.get("registered_semantic_lowerers", ()),
                path=f"{path}.registered_semantic_lowerers",
            )
        )
    )
    lowerer_call_ids = tuple(factory.call_id for factory in lowerers)
    if len(set(lowerer_call_ids)) != len(lowerer_call_ids):
        raise ValueError(
            f"{path}.registered_semantic_lowerers contains duplicate call IDs."
        )
    evidence = (
        None
        if "control_part_evidence" not in config
        else _decode_control_part_evidence(
            config["control_part_evidence"],
            path=f"{path}.control_part_evidence",
        )
    )
    return _DecodedRuntimeServices(
        grasp_pose_generators=generators,
        handover_pose_providers=handover,
        registered_semantic_lowerers=lowerers,
        control_part_evidence=evidence,
    )


@dataclass(frozen=True, slots=True)
class _ConfiguredTaskProgramAdapterFactory:
    """Expose one config identity while delegating live adapter construction."""

    delegate: SimulationTaskProgramAdapterFactory
    integration_fingerprint: str

    @property
    def registration(self) -> SimulationTaskProgramRegistration:
        """Return the immutable registration owned by the delegate."""
        return self.delegate.registration

    def create_adapter(
        self,
        environment: object,
    ) -> TaskProgramEnvironmentAdapter:
        """Create a live adapter for an initialized environment."""
        return self.delegate.create_adapter(environment)


@dataclass(frozen=True, slots=True)
class _ConfiguredTaskProgramIntegration:
    """Decoded provider-free integration plus its lazy live factory."""

    registration: SimulationTaskProgramRegistration
    adapter_factory: _ConfiguredTaskProgramAdapterFactory
    integration_fingerprint: str


def _decode_configured_task_program_integration(
    value: object,
) -> _ConfiguredTaskProgramIntegration:
    """Decode one composable, callable-free Task Program integration."""
    path = "integration"
    config = _mapping(
        value,
        path=path,
        required=frozenset({"scene", "robot_profile"}),
        optional=frozenset({"runtime_services"}),
    )
    scene_binding = _decode_scene(config["scene"])
    robot_profile_binding = _decode_robot_profile(config["robot_profile"])
    services = _decode_runtime_services(config.get("runtime_services", {}))

    call_catalog = builtin_semantic_call_catalog()
    for lowerer_factory in services.registered_semantic_lowerers:
        call_catalog = call_catalog.with_descriptor(
            SemanticCallDescriptor(
                call_id=lowerer_factory.call_id,
                spec_type=RegisteredSemanticCall,
                target_descriptor=lowerer_factory.target_descriptor,
            )
        )
    registration = SimulationTaskProgramRegistration(
        scene_binding=scene_binding,
        robot_profile_binding=robot_profile_binding,
        call_catalog=call_catalog,
        handover_pose_providers=services.handover_pose_providers,
        control_part_evidence_factory=services.control_part_evidence,
        registered_semantic_lowerer_factories=(services.registered_semantic_lowerers),
    )
    grasp_fingerprint = tuple(
        {
            "target_id": target_id,
            "factory": asdict(factory),
        }
        for target_id, factory in services.grasp_pose_generators
    )
    fingerprint_payload = {
        "registration": registration.fingerprint,
        "grasp_pose_generators": grasp_fingerprint,
    }
    integration_fingerprint = hashlib.sha256(
        json.dumps(
            fingerprint_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    delegate = SimulationTaskProgramAdapterFactory(
        registration,
        grasp_pose_generator_factories=dict(services.grasp_pose_generators),
    )
    adapter_factory = _ConfiguredTaskProgramAdapterFactory(
        delegate=delegate,
        integration_fingerprint=integration_fingerprint,
    )
    return _ConfiguredTaskProgramIntegration(
        registration=registration,
        adapter_factory=adapter_factory,
        integration_fingerprint=integration_fingerprint,
    )
