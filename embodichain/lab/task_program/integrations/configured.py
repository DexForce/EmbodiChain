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
from copy import deepcopy
from dataclasses import asdict, dataclass, fields
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

from ._configured_services import (
    _AntipodalGraspPoseGeneratorFactory,
    _ArticulationLinkSlideLowererFactory,
    _AxisAlignLowererFactory,
    _CoordinatedTransportLowererFactory,
    _CoordinatedTransportRoute,
    _JointPositionConstraintEvidenceProviderFactory,
    _MoveHeldObjectLowererFactory,
    _ParkLowererFactory,
    _PourLowererFactory,
    _PourTargetBinding,
    _PushObjectLowererFactory,
    _RelativePlaceLowererFactory,
    _RelativePlaceRoute,
)
from ._configured_hand_geometry import (
    _RobotHandGeometryCfgFactory,
    _decode_hand_geometry,
)
from ._configured_controls import _decode_control_operation, _decode_control_lowerer
from .simulation.control_assembly import ControlOperationDeclaration
from .simulation.recorded_contacts import SimulationRecordedContactCfg
from embodichain.lab.sim.atomic_actions.grip_evidence import DriverOpeningBinding
from .simulation.cargo_observation import SimulationCargoEnvelopeCfg
from embodichain.lab.sim.sensors.contact.timeline import ContactTimelineCfg
from .simulation.control_policy import SimulationControlConditionCfg
from .simulation.control_lowering import PressPrepareFactory, TwistPrepareFactory
from .catalog import (
    SimulationTaskProgramRegistration,
)
from embodichain.lab.gym.envs.settling import DynamicSettleMonitorCfg
from .simulation.policies import SimulationSettleMonitorCfg
from .simulation.stability import (
    SimulationStabilityConditionCfg,
    SimulationJointStabilityConditionCfg,
    SupportSensorDeclarationCfg,
)
from .composition import (
    ConfiguredTaskProgramIntegration as _ConfiguredTaskProgramIntegration,
)
from .nominal import _ConfiguredPrototypeSource
from .simulation.targets import AbsolutePoseTarget, SceneEntityTarget
from .simulation.transport import (
    HeldObjectTransportRoute,
    RoutedMoveHeldObjectLowererFactory,
)
from .simulation import (
    AntipodalGraspAffordanceBinding,
    ContainerAffordanceBinding,
    ControlPartCommandPreset,
    ControlPartEndpointBinding,
    ControlPartResourceBinding,
    RigidizedArticulationAntipodalGraspBinding,
    SimulationArticulationBinding,
    SimulationArticulationLinkBinding,
    SimulationRigidizedArticulationObjectBinding,
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
    AxisAlignOptions,
    CoordinatedPickmentOptions,
    DynamicCollisionMode,
    ExecutionRunnerCfg,
    HandOverOptions,
    HandOverPlanningRecovery,
    MotionPolicy,
    MoveEndEffectorOptions,
    MoveHeldObjectOptions,
    MoveJointsOptions,
    PickUpOptions,
    PlaceOptions,
    PourOptions,
    PressOptions,
    PressPlanningRecovery,
    PushObjectOptions,
    PushObjectToolCalibration,
    RecoveryPolicy,
    SlideOptions,
    SlidePlanningRecovery,
    ArticulationWithdrawalPlanningRecovery,
    ArticulationParkPlanningRecovery,
    TrackingPolicy,
    TwistOptions,
    TwistPlanningRecovery,
)
from embodichain.lab.sim.motion.planners.trapezoidal_planner import (
    TrapezoidalPlanOptions,
)
from embodichain.lab.task_program.semantics import (
    EffectAssurance,
    EffectMonitorRef,
    ReleaseGateEvidence,
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
    resolve_parallel_jaw_gripper_model,
)

if TYPE_CHECKING:
    from .environment import TaskProgramEnvironmentAdapter

from .simulation.planning_services import (
    CollisionSceneRosterCfg,
    SimulationPlanningServicesCfg,
)
from .invocation_policy import InvocationPolicyBinding, PostPolicyPresetBinding
from .simulation.held_alignment import HeldAlignmentRoute, HeldAlignmentLowererFactory
from .simulation.release_clearance import (
    ReleasedHandRoute,
    NominalHandBinding,
    ClearReleasedLowererFactory,
)
from .simulation.idle_clearance import IdleClearanceRoute, IdleClearanceFactory
from embodichain.toolkits.graspkit.hand_envelope import NominalHandEnvelope

from ._configured_prismatic import _decode_prismatic_lowerer
from .simulation.prismatic_lowering import (
    ArticulationSlideFactory,
    ArticulationParkFactory,
)
from embodichain.lab.sim.atomic_actions.articulation_continuation import (
    SlidePlanningRecovery,
    ArticulationWithdrawalPlanningRecovery,
    ArticulationParkPlanningRecovery,
)
from .simulation.effect_contracts import (
    PlacementEffectContract,
    TransportEffectContract,
)
from .simulation.planning_services import ObservedCuroboServicesCfg
from .simulation.place_scene import PlaceSceneBinding
from .simulation.grip_evidence import NativeGripEvidenceCfg, NativeGripEvidenceFactory
from .simulation.coordinated_grasp import CoordinatedGraspServiceCfg
from embodichain.toolkits.graspkit.pg_grasp.coordinated import CoordinatedGraspCfg
from .simulation.held_planning import HeldStagingInvocationBinding
from .simulation.released_continuation import ReleasedHandContinuationBinding
from embodichain.lab.sim.atomic_actions.released_continuation import (
    ReleasedHandClearanceRecovery,
    ReleasedHandParkRecovery,
)
from .simulation.pick_closing import PickClosingBinding
from .simulation.pour_binding import PourProjectionBinding
from .simulation.slide_preshape import SlidePreshapeBinding
from .simulation.place_continuation import (
    PlaceContinuationBinding,
    HeldAlignmentContinuationBinding,
)
from .simulation.drawer_planning import DrawerInvocationBinding
from .simulation.measured_rigid import MeasuredRigidInvocationBinding
from .simulation.placement import (
    ObservedRelativePlaceBinding,
    ObservedRelativePlaceLowererFactory,
    UprightPlaceLowererFactory,
)
from .simulation.retained_pour import RetainedPourFactory
from .simulation.jaw_pour import JawPourFactory
from .simulation.pour_transport import (
    PourStageRoute,
    PourTransportFactory,
    PourRegionTransportFactory,
)
from .simulation.stack_placement import StackPlaceLowererFactory
from .simulation.pick_targets import (
    PickTargetRoute,
    ConfiguredPickLowererFactory,
    make_pick_alias_factory,
)
from .simulation.coordinated import (
    CoordinatedHoldLowererFactory,
    CoordinatedTargetRoute,
    CoordinatedPlacementLowererFactory,
    _CoordinatedRetainedTargetFactory,
)
from .simulation.transport import (
    TerminalAttachmentTransportLowererFactory,
    DrawerTransportLowererFactory,
)
from embodichain.lab.sim.atomic_actions.primitives.twist_recovery import (
    TwistPlanningRecovery,
)
from embodichain.lab.sim.atomic_actions import (
    PickPlanningRecovery,
    PlacePlanningRecovery,
    MoveHeldObjectPlanningRecovery,
)

from embodichain.lab.sim.atomic_actions.grip_evidence import DriverOpeningBinding

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


def _decode_rigidized_articulation(
    value: object,
    *,
    path: str,
) -> SimulationRigidizedArticulationObjectBinding:
    """Decode one locked native articulation exposed as a semantic object."""
    config = _mapping(
        value,
        path=path,
        required=frozenset({"entity_id", "locked_qpos"}),
        optional=frozenset(
            {
                "simulation_uid",
                "aliases",
                "dynamics",
                "collision_role",
                "semantic_type",
                "default_grasp_affordance",
                "joint_position_tolerance",
                "link_transform_tolerance",
                "affordances",
            }
        ),
    )
    raw_locked_qpos = config["locked_qpos"]
    if not isinstance(raw_locked_qpos, Mapping):
        raise TypeError(f"{path}.locked_qpos must be a mapping.")
    locked_qpos = {
        _identifier(name, path=f"{path}.locked_qpos keys"): _real(
            position,
            path=f"{path}.locked_qpos.{name}",
        )
        for name, position in raw_locked_qpos.items()
    }
    if not locked_qpos:
        raise ValueError(f"{path}.locked_qpos must not be empty.")
    common = _decode_scene_entity_common(config, path=path)
    if common["collision_role"] is not SceneCollisionRole.NONE:
        raise ValueError(
            f"{path} rigidized articulation collision_role must be 'none' in "
            "configured integrations; no planner collision geometry is available."
        )
    return SimulationRigidizedArticulationObjectBinding(
        **common,
        locked_qpos=locked_qpos,
        default_grasp_affordance=_optional_identifier(
            config.get("default_grasp_affordance"),
            path=f"{path}.default_grasp_affordance",
        ),
        joint_position_tolerance=_real(
            config.get("joint_position_tolerance", 1.0e-3),
            path=f"{path}.joint_position_tolerance",
            minimum=0.0,
            strict_minimum=True,
        ),
        link_transform_tolerance=_real(
            config.get("link_transform_tolerance", 1.0e-5),
            path=f"{path}.link_transform_tolerance",
            minimum=0.0,
            strict_minimum=True,
        ),
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


def _decode_rigidized_articulation_antipodal_grasp(
    value: object,
    *,
    object_id: str,
    path: str,
) -> RigidizedArticulationAntipodalGraspBinding:
    """Decode one link-backed grasp for a rigidized articulation object."""
    config = _mapping(
        value,
        path=path,
        required=frozenset({"kind", "entity_id", "grasp_link"}),
        optional=frozenset(
            {
                "native_name",
                "revision",
                "aliases",
                "relative_pose",
            }
        ),
    )
    grasp_link = _identifier(config["grasp_link"], path=f"{path}.grasp_link")
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
    return RigidizedArticulationAntipodalGraspBinding(
        entity_id=_identifier(config["entity_id"], path=f"{path}.entity_id"),
        object_id=object_id,
        grasp_link=grasp_link,
        native_name=_identifier(
            config.get("native_name", grasp_link),
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
    optional = {
        "aliases",
        "object_target_pose",
        "object_target_frame",
        "minimum_confidence",
        "is_default",
    }
    if binding_type is ContainerAffordanceBinding:
        optional.add("release_clearance")
    config = _mapping(
        value,
        path=path,
        required=frozenset({"kind", "entity_id", "native_name"}),
        optional=frozenset(optional),
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
        "object_target_frame": _identifier(
            config.get("object_target_frame", "parent"),
            path=f"{path}.object_target_frame",
        ),
    }
    if "object_target_pose" in config:
        kwargs["object_target_pose"] = _finite_tuple(
            config["object_target_pose"],
            path=f"{path}.object_target_pose",
            expected_length=16,
        )
    if binding_type is ContainerAffordanceBinding:
        kwargs["release_clearance"] = _real(
            config.get("release_clearance", 0.0),
            path=f"{path}.release_clearance",
            minimum=0.0,
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
    | RigidizedArticulationAntipodalGraspBinding
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
                "grasp_link",
                "object_target_pose",
                "object_target_frame",
                "minimum_confidence",
                "release_clearance",
                "is_default",
            }
        ),
    )
    kind = _identifier(common["kind"], path=f"{path}.kind")
    if kind == "antipodal_grasp":
        if parent_category == "rigid_objects":
            return _decode_antipodal_grasp(
                value,
                object_id=parent_id,
                path=path,
            )
        if parent_category == "rigidized_articulations":
            return _decode_rigidized_articulation_antipodal_grasp(
                value,
                object_id=parent_id,
                path=path,
            )
        raise ValueError(
            f"{path}.kind {kind!r} is supported only under a rigid object or "
            "rigidized articulation."
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
    | RigidizedArticulationAntipodalGraspBinding
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
                "rigidized_articulations",
                "articulations",
                "links",
                "collision_world_mode",
            }
        ),
    )

    antipodal_grasps: list[AntipodalGraspAffordanceBinding] = []
    rigidized_articulation_grasps: list[RigidizedArticulationAntipodalGraspBinding] = []
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
                elif isinstance(
                    affordance,
                    RigidizedArticulationAntipodalGraspBinding,
                ):
                    rigidized_articulation_grasps.append(affordance)
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
    rigidized_articulations = decode_entities(
        "rigidized_articulations",
        _decode_rigidized_articulation,
    )
    articulations = decode_entities("articulations", _decode_articulation)
    links = decode_entities("links", _decode_link)

    return SimulationSceneBinding(
        registry_id=_identifier(config["registry_id"], path=f"{path}.registry_id"),
        rigid_objects=rigid_objects,
        rigidized_articulations=rigidized_articulations,
        articulations=articulations,
        links=links,
        antipodal_grasps=tuple(antipodal_grasps),
        rigidized_articulation_grasps=tuple(rigidized_articulation_grasps),
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
                "require_free_motion",
                "hand_interp_steps",
                "hold_steps",
                "release",
                "release_at_target",
                "arm_selection",
                "release_steps",
                "retreat_distance",
                "retreat_steps",
                "receive_pick_object_part",
                "adaptive_receive",
                "require_receiver_scene",
                "receive_region",
                "existing_hold_hand_source",
                "source_retreat_mode",
                "object_motion_keyframes",
                "continue_verified_holds",
                "prefer_cargo_clearance",
                "grasp_settle_steps",
                "release_settle_steps",
                "pick_object_part",
                "adaptive_grasp",
                "full_opening_clearance",
                "native_closing_revision",
                "require_qualified_approach",
                "constrained_grasp",
                "qualify_fixed_grasp",
                "measured_preshape",
                "qualify_downstream_targets",
                "native_scene_trial",
                "lift_height",
                "pre_grasp_distance",
                "grasp_commit_fraction",
                "approach_direction",
                "left_to_right_arm_direction",
                "middle_empty_ratio",
                "grasp_seed",
                "approach_alignment_max_angle",
                "obj_upright_direction",
                "rotate_upright",
                "grasp_frame_to_eef",
                "fixed_object_to_eef",
                "max_approach_retract_z",
                "linear_segments",
                "lift_clearance",
                "hand_hold",
                "report_retained_attachment",
                "qualify_required_waypoints",
                "monitor_scene_until_approach",
                "drawer_placement",
                "measured_inside",
                "drawer_transport",
                "drawer_containment",
                "preserve_vertical",
                "cartesian_waypoint_count",
                "preserve_current_object_orientation",
                "direction",
                "approach_distance",
                "press_distance",
                "press_position",
                "contact_roll_alternatives",
                "translation_distance",
                "press_distance",
                "press_position",
                "contact_roll_alternatives",
                "contact_bite_depth",
                "contact_roll",
                "contact_settle_steps",
                "rotation_enabled",
                "twist_angle",
                "max_twist_angle",
                "twist_waypoint_count",
                "preshape_fraction",
                "qualify_hand_clearance",
                "approach_along_grasp_axis",
                "release_retreat_distance",
                "rotate_angle",
                "dwell_duration",
                "mouth_height",
                "max_mouth_height",
                "clearance",
                "max_rotation_step",
                "approach_height",
                "retract_height",
                "contact_distance",
                "push_overshoot",
                "completion_tolerance",
                "object_contact_offset",
                "support_frame_planar_contact_offset",
                "contact_frame_to_eef",
                "tool_calibrations",
                "target_axis",
            }
        ),
    )
    kind = _identifier(common["kind"], path=f"{path}.kind")
    if kind in {"axis_align", "pick_up"}:
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
                    "grasp_commit_fraction",
                    "approach_direction",
                    "approach_alignment_max_angle",
                    "obj_upright_direction",
                    "rotate_upright",
                    "grasp_frame_to_eef",
                    "fixed_object_to_eef",
                    *(
                        (
                            "approach_along_grasp_axis",
                            "adaptive_grasp",
                            "full_opening_clearance",
                            "native_closing_revision",
                            "require_qualified_approach",
                            "constrained_grasp",
                            "qualify_fixed_grasp",
                            "measured_preshape",
                            "qualify_downstream_targets",
                            "native_scene_trial",
                            "linear_segments",
                        )
                        if kind == "pick_up"
                        else ()
                    ),
                    *(("target_axis",) if kind == "axis_align" else ()),
                }
            ),
        )
        kwargs: dict[str, object] = {}
        if "approach_along_grasp_axis" in config:
            kwargs["approach_along_grasp_axis"] = _boolean(
                config["approach_along_grasp_axis"],
                path=f"{path}.approach_along_grasp_axis",
            )
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
        if "grasp_commit_fraction" in config:
            kwargs["grasp_commit_fraction"] = _real(
                config["grasp_commit_fraction"],
                path=f"{path}.grasp_commit_fraction",
                minimum=0.0,
                maximum=1.0,
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
        if kind == "axis_align":
            if "target_axis" in config:
                import torch

                kwargs["target_axis"] = torch.tensor(
                    _finite_tuple(
                        config["target_axis"],
                        path=f"{path}.target_axis",
                        expected_length=3,
                    ),
                    dtype=torch.float32,
                )
            return AxisAlignOptions(**kwargs)
        if "adaptive_grasp" in config:
            kwargs["adaptive_grasp"] = _identifier(
                config["adaptive_grasp"], path=f"{path}.adaptive_grasp"
            )
        for name in (
            "full_opening_clearance",
            "require_qualified_approach",
            "constrained_grasp",
            "qualify_fixed_grasp",
            "measured_preshape",
            "qualify_downstream_targets",
            "native_scene_trial",
            "linear_segments",
        ):
            if name in config:
                kwargs[name] = _boolean(config[name], path=f"{path}.{name}")
        if "native_closing_revision" in config:
            kwargs["native_closing_revision"] = _integer(
                config["native_closing_revision"],
                path=f"{path}.native_closing_revision",
                minimum=1,
            )
        return PickUpOptions(**kwargs)
    if kind == "place":
        config = _mapping(
            value,
            path=path,
            required=frozenset({"kind"}),
            optional=frozenset(
                {
                    "drawer_placement",
                    "measured_inside",
                    "preserve_vertical",
                    "hand_interp_steps",
                    "release_settle_steps",
                    "lift_height",
                    "max_approach_retract_z",
                    "cartesian_waypoint_count",
                    "preserve_current_object_orientation",
                    "linear_segments",
                    "qualify_required_waypoints",
                    "monitor_scene_until_approach",
                }
            ),
        )
        kwargs = {}
        for name in (
            "drawer_placement",
            "measured_inside",
            "preserve_vertical",
        ):
            if name in config:
                kwargs[name] = _boolean(config[name], path=f"{path}.{name}")

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
        for name in (
            "linear_segments",
            "qualify_required_waypoints",
            "monitor_scene_until_approach",
        ):
            if name in config:
                kwargs[name] = _boolean(config[name], path=f"{path}.{name}")
        return PlaceOptions(**kwargs)
    if kind == "move_held_object":
        config = _mapping(
            value,
            path=path,
            required=frozenset({"kind"}),
            optional=frozenset(
                {
                    "drawer_transport",
                    "preserve_vertical",
                    "linear_segments",
                    "lift_clearance",
                    "hand_hold",
                    "report_retained_attachment",
                }
            ),
        )
        kwargs = {}
        for name in (
            "drawer_transport",
            "preserve_vertical",
        ):
            if name in config:
                kwargs[name] = _boolean(config[name], path=f"{path}.{name}")

        for name in ("linear_segments", "report_retained_attachment"):
            if name in config:
                kwargs[name] = _boolean(config[name], path=f"{path}.{name}")
        if "lift_clearance" in config:
            kwargs["lift_clearance"] = (
                None
                if config["lift_clearance"] is None
                else _real(
                    config["lift_clearance"], path=f"{path}.lift_clearance", minimum=0.0
                )
            )
        if "hand_hold" in config:
            kwargs["hand_hold"] = _identifier(
                config["hand_hold"], path=f"{path}.hand_hold"
            )
        return MoveHeldObjectOptions(**kwargs)
    if kind == "move_joints":
        config = _mapping(
            value,
            path=path,
            required=frozenset({"kind"}),
            optional=frozenset({"require_free_motion"}),
        )
        return MoveJointsOptions(
            require_free_motion=config.get("require_free_motion", False)
        )
    if kind == "move_end_effector":
        config = _mapping(
            value,
            path=path,
            required=frozenset({"kind"}),
            optional=frozenset({"linear_segments"}),
        )
        return MoveEndEffectorOptions(
            linear_segments=_boolean(
                config.get("linear_segments", False), path=f"{path}.linear_segments"
            )
        )
    if kind == "coordinated_pickment":
        config = _mapping(
            value,
            path=path,
            required=frozenset({"kind"}),
            optional=frozenset(
                {
                    "object_motion_keyframes",
                    "continue_verified_holds",
                    "prefer_cargo_clearance",
                    "pre_grasp_distance",
                    "lift_height",
                    "hand_interp_steps",
                    "hold_steps",
                    "release",
                    "release_steps",
                    "retreat_distance",
                    "retreat_steps",
                    "approach_direction",
                    "left_to_right_arm_direction",
                    "middle_empty_ratio",
                    "grasp_seed",
                }
            ),
        )
        kwargs: dict[str, object] = {}
        for field_name in (
            "object_motion_keyframes",
            "hand_interp_steps",
            "hold_steps",
            "release_steps",
            "retreat_steps",
            "grasp_seed",
        ):
            if field_name in config:
                kwargs[field_name] = _integer(
                    config[field_name],
                    path=f"{path}.{field_name}",
                    minimum=(2 if field_name == "object_motion_keyframes" else 0),
                )
        for field_name in (
            "pre_grasp_distance",
            "lift_height",
            "retreat_distance",
            "middle_empty_ratio",
        ):
            if field_name in config:
                kwargs[field_name] = _real(
                    config[field_name],
                    path=f"{path}.{field_name}",
                    minimum=0.0,
                    maximum=(1.0 if field_name == "middle_empty_ratio" else math.inf),
                )
        if "prefer_cargo_clearance" in config:
            kwargs["prefer_cargo_clearance"] = _boolean(
                config["prefer_cargo_clearance"], path=f"{path}.prefer_cargo_clearance"
            )
        if "continue_verified_holds" in config:
            kwargs["continue_verified_holds"] = _boolean(
                config["continue_verified_holds"],
                path=f"{path}.continue_verified_holds",
            )
        if "release" in config:
            kwargs["release"] = _boolean(
                config["release"],
                path=f"{path}.release",
            )
        import torch

        for field_name in ("approach_direction", "left_to_right_arm_direction"):
            if field_name in config:
                kwargs[field_name] = torch.tensor(
                    _finite_tuple(
                        config[field_name],
                        path=f"{path}.{field_name}",
                        expected_length=3,
                    ),
                    dtype=torch.float32,
                )
        return CoordinatedPickmentOptions(**kwargs)
    if kind == "pour":
        config = _mapping(
            value,
            path=path,
            required=frozenset({"kind"}),
            optional=frozenset(
                {
                    "rotate_angle",
                    "dwell_duration",
                    "mouth_height",
                    "max_mouth_height",
                    "clearance",
                    "approach_distance",
                    "max_rotation_step",
                }
            ),
        )
        kwargs = {}
        if "rotate_angle" in config:
            kwargs["rotate_angle"] = _real(
                config["rotate_angle"],
                path=f"{path}.rotate_angle",
            )
        if "dwell_duration" in config:
            kwargs["dwell_duration"] = _real(
                config["dwell_duration"],
                path=f"{path}.dwell_duration",
                minimum=0.0,
            )
        for field_name in (
            "mouth_height",
            "max_mouth_height",
            "clearance",
            "approach_distance",
            "max_rotation_step",
        ):
            if field_name in config:
                kwargs[field_name] = _real(
                    config[field_name], path=f"{path}.{field_name}"
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
    if kind == "twist":
        config = _mapping(
            value,
            path=path,
            required=frozenset({"kind"}),
            optional=frozenset(
                {
                    "hand_interp_steps",
                    "twist_waypoint_count",
                    "pre_grasp_distance",
                    "twist_angle",
                    "max_twist_angle",
                    "contact_bite_depth",
                    "contact_roll",
                    "contact_settle_steps",
                    "rotation_enabled",
                }
            ),
        )
        kwargs = {}
        for name in (
            "hand_interp_steps",
            "twist_waypoint_count",
            "contact_settle_steps",
        ):
            if name in config:
                kwargs[name] = _integer(
                    config[name],
                    path=f"{path}.{name}",
                    minimum=0 if name == "contact_settle_steps" else 1,
                )
        for name in (
            "pre_grasp_distance",
            "twist_angle",
            "max_twist_angle",
            "contact_bite_depth",
            "contact_roll",
        ):
            if name in config:
                kwargs[name] = (
                    None
                    if config[name] is None
                    and name in {"contact_bite_depth", "contact_roll"}
                    else _real(config[name], path=f"{path}.{name}")
                )
        if "rotation_enabled" in config:
            kwargs["rotation_enabled"] = _boolean(
                config["rotation_enabled"], path=f"{path}.rotation_enabled"
            )
        return TwistOptions(**kwargs)
    if kind == "press":
        config = _mapping(
            value,
            path=path,
            required=frozenset({"kind"}),
            optional=frozenset(
                {
                    "hand_interp_steps",
                    "approach_distance",
                    "press_distance",
                    "press_position",
                    "contact_roll_alternatives",
                }
            ),
        )
        kwargs = {}
        if "hand_interp_steps" in config:
            kwargs["hand_interp_steps"] = _integer(
                config["hand_interp_steps"], path=f"{path}.hand_interp_steps", minimum=1
            )
        for name in ("approach_distance", "press_distance"):
            if name in config:
                kwargs[name] = _real(
                    config[name],
                    path=f"{path}.{name}",
                    minimum=0.0,
                    strict_minimum=name == "press_distance",
                )
        if "press_position" in config:
            kwargs["press_position"] = (
                None
                if config["press_position"] is None
                else _finite_tuple(
                    config["press_position"],
                    path=f"{path}.press_position",
                    expected_length=3,
                )
            )
        if "contact_roll_alternatives" in config:
            kwargs["contact_roll_alternatives"] = tuple(
                _real(
                    item,
                    path=f"{path}.contact_roll_alternatives[{index}]",
                    minimum=-math.pi,
                    maximum=math.pi,
                )
                for index, item in enumerate(
                    _sequence(
                        config["contact_roll_alternatives"],
                        path=f"{path}.contact_roll_alternatives",
                    )
                )
            )
        return PressOptions(**kwargs)
    if kind == "slide":
        config = _mapping(
            value,
            path=path,
            required=frozenset({"kind"}),
            optional=frozenset(
                {
                    "drawer_containment",
                    "direction",
                    "hand_interp_steps",
                    "approach_distance",
                    "translation_distance",
                    "preshape_fraction",
                    "qualify_hand_clearance",
                    "approach_along_grasp_axis",
                    "release_retreat_distance",
                }
            ),
        )
        kwargs = {}
        for name in ("drawer_containment",):
            if name in config:
                kwargs[name] = _boolean(config[name], path=f"{path}.{name}")

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
        for name in ("preshape_fraction", "release_retreat_distance"):
            if name in config:
                kwargs[name] = _real(config[name], path=f"{path}.{name}", minimum=0.0)
        if "qualify_hand_clearance" in config:
            kwargs["qualify_hand_clearance"] = _boolean(
                config["qualify_hand_clearance"], path=f"{path}.qualify_hand_clearance"
            )
        if "approach_along_grasp_axis" in config:
            kwargs["approach_along_grasp_axis"] = _boolean(
                config["approach_along_grasp_axis"],
                path=f"{path}.approach_along_grasp_axis",
            )
        return SlideOptions(**kwargs)
    if kind == "hand_over":
        config = _mapping(
            value,
            path=path,
            required=frozenset({"kind"}),
            optional=frozenset(
                {
                    "pre_grasp_distance",
                    "lift_height",
                    "hand_interp_steps",
                    "hold_steps",
                    "retreat_distance",
                    "retreat_steps",
                    "receive_pick_object_part",
                    "release_at_target",
                    "arm_selection",
                    "adaptive_receive",
                    "linear_segments",
                    "require_receiver_scene",
                    "receive_region",
                    "existing_hold_hand_source",
                    "source_retreat_mode",
                }
            ),
        )
        kwargs = {}
        if "linear_segments" in config:
            kwargs["linear_segments"] = _boolean(
                config["linear_segments"], path=f"{path}.linear_segments"
            )
        for field_name in (
            "pre_grasp_distance",
            "lift_height",
            "retreat_distance",
        ):
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
        for field_name in ("hold_steps", "retreat_steps"):
            if field_name in config:
                kwargs[field_name] = _integer(
                    config[field_name],
                    path=f"{path}.{field_name}",
                    minimum=0 if field_name == "hold_steps" else 2,
                )
        if "receive_pick_object_part" in config:
            kwargs["receive_pick_object_part"] = _identifier(
                config["receive_pick_object_part"],
                path=f"{path}.receive_pick_object_part",
            )
        if "release_at_target" in config:
            kwargs["release_at_target"] = _boolean(
                config["release_at_target"],
                path=f"{path}.release_at_target",
            )
        if "arm_selection" in config:
            kwargs["arm_selection"] = _identifier(
                config["arm_selection"],
                path=f"{path}.arm_selection",
            )
        if "require_receiver_scene" in config:
            kwargs["require_receiver_scene"] = _boolean(
                config["require_receiver_scene"], path=f"{path}.require_receiver_scene"
            )
        if "adaptive_receive" in config:
            kwargs["adaptive_receive"] = _boolean(
                config["adaptive_receive"], path=f"{path}.adaptive_receive"
            )
        for field_name in (
            "receive_region",
            "existing_hold_hand_source",
            "source_retreat_mode",
        ):
            if field_name in config:
                kwargs[field_name] = _identifier(
                    config[field_name], path=f"{path}.{field_name}"
                )
        return HandOverOptions(**kwargs)
    raise ValueError(
        f"Unsupported {path}.kind {kind!r}; supported kinds are "
        "['coordinated_pickment', 'hand_over', 'move_end_effector', 'move_held_object', "
        "'move_joints', "
        "'pick_up', 'place', 'pour', 'press', 'push_object', 'slide', 'twist']."
    )


def _decode_motion_policy(value: object, *, path: str) -> MotionPolicy:
    """Decode one motion-generation policy."""
    config = _mapping(
        value,
        path=path,
        required=frozenset({"sample_count"}),
        optional=frozenset(
            {
                "strategy",
                "velocity_limit",
                "acceleration_limit",
                "dynamic_collision_mode",
                "velocity_targets",
                "plan_opts",
                "interpolation_mode",
                "ik_seed_mode",
                "sample_count_retries",
                "segment_sample_count_retries",
                "segment_continuous_ik_recovery",
                "require_final_velocity",
                "final_velocity_rejection",
            }
        ),
    )
    sample_count = _integer(
        config["sample_count"], path=f"{path}.sample_count", minimum=2
    )
    kwargs: dict[str, object] = {
        "strategy": _identifier(
            config.get("strategy", "ik_interp"),
            path=f"{path}.strategy",
        ),
        "velocity_targets": _identifier(
            config.get("velocity_targets", "auto"),
            path=f"{path}.velocity_targets",
        ),
        "sample_count": sample_count,
        "final_velocity_rejection": _identifier(
            config.get("final_velocity_rejection", "hold"),
            path=f"{path}.final_velocity_rejection",
        ),
        "require_final_velocity": _boolean(
            config.get("require_final_velocity", False),
            path=f"{path}.require_final_velocity",
        ),
        "sample_count_retries": tuple(
            _integer(item, path=f"{path}.sample_count_retries[{index}]", minimum=2)
            for index, item in enumerate(
                _sequence(
                    config.get("sample_count_retries", ()),
                    path=f"{path}.sample_count_retries",
                )
            )
        ),
        "segment_sample_count_retries": tuple(
            _integer(
                item, path=f"{path}.segment_sample_count_retries[{index}]", minimum=2
            )
            for index, item in enumerate(
                _sequence(
                    config.get("segment_sample_count_retries", ()),
                    path=f"{path}.segment_sample_count_retries",
                )
            )
        ),
        "segment_continuous_ik_recovery": _boolean(
            config.get("segment_continuous_ik_recovery", False),
            path=f"{path}.segment_continuous_ik_recovery",
        ),
        "interpolation_mode": _identifier(
            config.get("interpolation_mode", "adaptive"),
            path=f"{path}.interpolation_mode",
        ),
        "ik_seed_mode": _identifier(
            config.get("ik_seed_mode", "batched"), path=f"{path}.ik_seed_mode"
        ),
        "dynamic_collision_mode": _enum(
            config.get("dynamic_collision_mode", DynamicCollisionMode.AUTO.value),
            DynamicCollisionMode,
            path=f"{path}.dynamic_collision_mode",
        ),
    }
    for field_name in ("velocity_limit", "acceleration_limit"):
        if field_name in config:
            kwargs[field_name] = (
                None
                if config[field_name] is None
                else _real(
                    config[field_name],
                    path=f"{path}.{field_name}",
                    minimum=0.0,
                    strict_minimum=True,
                )
            )
    if "plan_opts" in config:
        option_path = f"{path}.plan_opts"
        options = _mapping(
            config["plan_opts"],
            path=option_path,
            required=frozenset({"kind"}),
            optional=frozenset(
                {"stop_at_waypoints", "blend_tolerance", "collinearity_tolerance"}
            ),
        )
        if _identifier(options["kind"], path=f"{option_path}.kind") != "trapezoidal":
            raise ValueError(f"{option_path}.kind must be 'trapezoidal'.")
        plan_opts = TrapezoidalPlanOptions(
            sample_interval=sample_count,
            stop_at_waypoints=_boolean(
                options.get("stop_at_waypoints", True),
                path=f"{option_path}.stop_at_waypoints",
            ),
            blend_tolerance=_real(
                options.get("blend_tolerance", 0.0),
                path=f"{option_path}.blend_tolerance",
                minimum=0.0,
            ),
            collinearity_tolerance=_real(
                options.get("collinearity_tolerance", 1e-5),
                path=f"{option_path}.collinearity_tolerance",
                minimum=0.0,
                maximum=1.0,
            ),
        )
        # Explicit backend options own their constraints in MotionGenerator.
        # Carry YAML policy limits into those options rather than losing them.
        for field_name, constraint in (
            ("velocity_limit", "velocity"),
            ("acceleration_limit", "acceleration"),
        ):
            if field_name in kwargs:
                plan_opts.constraints[constraint] = kwargs[field_name]
        kwargs["plan_opts"] = plan_opts
    return MotionPolicy(**kwargs)


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
                "planning_recovery",
                "planning_budget",
                "retry_all_failed_plans",
                "planning_recovery",
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
    if "retry_all_failed_plans" in config:
        kwargs["retry_all_failed_plans"] = _boolean(
            config["retry_all_failed_plans"], path=f"{path}.retry_all_failed_plans"
        )
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
    if config.get("planning_budget") is not None:
        from embodichain.lab.sim.atomic_actions.policies import PlanningBudget

        budget_path = f"{path}.planning_budget"
        budget = _mapping(
            config["planning_budget"],
            path=budget_path,
            required=frozenset(),
            optional=frozenset({"wall_seconds"}),
        )
        wall_seconds = budget.get("wall_seconds")
        kwargs["planning_budget"] = PlanningBudget(
            wall_seconds=(
                None
                if wall_seconds is None
                else _real(
                    wall_seconds,
                    path=f"{budget_path}.wall_seconds",
                    minimum=0.0,
                    strict_minimum=True,
                )
            )
        )
    if config.get("planning_recovery") is not None:
        declaration_path = f"{path}.planning_recovery"
        declaration = _mapping(
            config["planning_recovery"],
            path=declaration_path,
            required=frozenset({"kind", "revision"}),
            optional=frozenset(
                {
                    "variant",
                    "inside_world_yaw",
                    "staging",
                    "constrained_roll",
                    "preshape_aperture_recovery",
                    "place_continuation",
                }
            ),
        )
        kind = _identifier(declaration["kind"], path=f"{declaration_path}.kind")
        declaration_types = {
            "press_bounded": PressPlanningRecovery,
            "twist_bounded": TwistPlanningRecovery,
            "slide_bounded": SlidePlanningRecovery,
            "articulation_withdrawal": ArticulationWithdrawalPlanningRecovery,
            "articulation_park": ArticulationParkPlanningRecovery,
            "released_clearance": ReleasedHandClearanceRecovery,
            "released_park": ReleasedHandParkRecovery,
            "pick_bounded": PickPlanningRecovery,
            "place_bounded": PlacePlanningRecovery,
            "handover_bounded": HandOverPlanningRecovery,
            "held_alignment_bounded": MoveHeldObjectPlanningRecovery,
        }
        if kind not in declaration_types:
            raise ValueError(
                f"{declaration_path}.kind must be one of {sorted(declaration_types)}."
            )
        if (
            _integer(
                declaration["revision"], path=f"{declaration_path}.revision", minimum=1
            )
            != 1
        ):
            raise ValueError(f"Unsupported {declaration_path}.revision.")
        if "constrained_roll" in declaration and kind != "pick_bounded":
            raise ValueError("This selection only belongs to Pick recovery.")
        if kind != "slide_bounded" and "preshape_aperture_recovery" in declaration:
            raise ValueError("Preshape aperture recovery belongs only to Slide.")

        if "place_continuation" in declaration and kind not in {
            "pick_bounded",
            "held_alignment_bounded",
        }:
            raise ValueError(
                "Place continuation only belongs to Pick or held alignment recovery."
            )
        if kind == "pick_bounded":
            if set(declaration) - {
                "kind",
                "revision",
                "constrained_roll",
                "place_continuation",
            }:
                raise ValueError(
                    "Pick recovery supports only constrained_roll and place_continuation selections."
                )
            kwargs["planning_recovery"] = PickPlanningRecovery(
                place_continuation=_boolean(
                    declaration.get("place_continuation", False),
                    path=f"{declaration_path}.place_continuation",
                ),
                constrained_roll=_boolean(
                    declaration.get("constrained_roll", False),
                    path=f"{declaration_path}.constrained_roll",
                ),
            )
        elif kind == "slide_bounded":
            if set(declaration) - {"kind", "revision", "preshape_aperture_recovery"}:
                raise ValueError("Slide recovery has unsupported fields.")
            kwargs["planning_recovery"] = SlidePlanningRecovery(
                preshape_aperture_recovery=_boolean(
                    declaration.get("preshape_aperture_recovery", False),
                    path=f"{declaration_path}.preshape_aperture_recovery",
                )
            )
        elif kind == "held_alignment_bounded":
            if "variant" in declaration or "inside_world_yaw" in declaration:
                raise ValueError(
                    f"{declaration_path} does not support variant or inside_world_yaw."
                )
            kwargs["planning_recovery"] = MoveHeldObjectPlanningRecovery(
                place_continuation=_boolean(
                    declaration.get("place_continuation", False),
                    path=f"{declaration_path}.place_continuation",
                ),
                staging=_boolean(
                    declaration.get("staging", False),
                    path=f"{declaration_path}.staging",
                ),
            )
        elif kind == "place_bounded":
            if "staging" in declaration:
                raise ValueError(f"{declaration_path} does not support staging.")
            kwargs["planning_recovery"] = PlacePlanningRecovery(
                variant=_identifier(
                    declaration.get("variant", "waypoint"),
                    path=f"{declaration_path}.variant",
                ),
                inside_world_yaw=_boolean(
                    declaration.get("inside_world_yaw", False),
                    path=f"{declaration_path}.inside_world_yaw",
                ),
            )
        else:
            if (
                "variant" in declaration
                or "inside_world_yaw" in declaration
                or "staging" in declaration
            ):
                raise ValueError(
                    f"{declaration_path} does not support variant, inside_world_yaw or staging."
                )
            kwargs["planning_recovery"] = declaration_types[kind]()
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
        optional=frozenset({"required_planner", "release_gate_evidence"}),
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
    motion_policy = _decode_motion_policy(config["motion"], path=f"{path}.motion")
    required_planner = _optional_identifier(
        config.get("required_planner"), path=f"{path}.required_planner"
    )
    if isinstance(motion_policy.plan_opts, TrapezoidalPlanOptions):
        if required_planner not in (None, "trapezoidal"):
            raise ValueError(
                f"{path}.motion.plan_opts requires the trapezoidal planner."
            )
        required_planner = "trapezoidal"
    recovery_policy = _decode_recovery_policy(
        config["recovery"], path=f"{path}.recovery"
    )
    handover_recovery = isinstance(
        recovery_policy.planning_recovery, HandOverPlanningRecovery
    )
    handover_options = [
        value for value in action_options.values() if type(value) is HandOverOptions
    ]
    if (handover_recovery and not handover_options) or any(
        value.adaptive_receive != handover_recovery for value in handover_options
    ):
        raise ValueError(
            f"{path}: adaptive receiving and handover_bounded recovery must be declared together."
        )
    return SkillPolicyPreset(
        _identifier(config["preset_id"], path=f"{path}.preset_id"),
        action_option_templates=action_options,
        effect_assurance=_enum(
            config["effect_assurance"],
            EffectAssurance,
            path=f"{path}.effect_assurance",
        ),
        motion_policy=motion_policy,
        tracking_policy=_decode_tracking_policy(
            config["tracking"],
            path=f"{path}.tracking",
        ),
        recovery_policy=recovery_policy,
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
        release_gate_evidence=_enum(
            config.get(
                "release_gate_evidence", ReleaseGateEvidence.TERMINAL_CLAUSES.value
            ),
            ReleaseGateEvidence,
            path=f"{path}.release_gate_evidence",
        ),
        required_planner=required_planner,
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
    """Resolve a registered aperture model, with optional inline bounds."""
    declaration = (
        value
        if type(value) is str
        else _mapping(
            value,
            path=path,
            required=frozenset({"model_id"}),
            optional=frozenset({"min_opening_width", "max_opening_width"}),
        )
    )
    try:
        return resolve_parallel_jaw_gripper_model(declaration)
    except (TypeError, ValueError) as error:
        raise type(error)(f"{path}: {error}") from error


def _geometry_pose(value: object, *, path: str) -> tuple[float, ...]:
    """Freeze and validate one explicit rigid geometry/TCP transform."""
    import numpy as np

    rows = _sequence(value, path=path)
    if len(rows) != 4:
        raise ValueError(f"{path} must have shape (4, 4).")
    values = []
    for index, row in enumerate(rows):
        row_values = _sequence(row, path=f"{path}[{index}]")
        if len(row_values) != 4:
            raise ValueError(f"{path} must have shape (4, 4).")
        values.extend(_real(item, path=f"{path}[{index}]") for item in row_values)
    matrix = np.asarray(values).reshape(4, 4)
    rotation = matrix[:3, :3]
    if (
        not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-7, rtol=0)
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0)
        or not np.isclose(np.linalg.det(rotation), 1, atol=1e-6, rtol=0)
    ):
        raise ValueError(f"{path} must be a proper rigid transform.")
    return tuple(values)


def _decode_grasp_generator(
    value: object, *, path: str
) -> _AntipodalGraspPoseGeneratorFactory:
    """Decode explicit physical geometry; removed box fields are rejected."""
    config = _mapping(
        value,
        path=path,
        required=frozenset({"kind", "model"}),
        optional=frozenset(
            {
                "sample_count",
                "approach_direction_samples",
                "approach_deviation_angle",
                "max_candidates",
                "center_mode",
                "filter_ground_collision",
                "force_refresh",
                "point_spacing",
                "side",
                "root_from_tcp",
                "root_from_contact_frame",
            }
        ),
    )
    kind = _identifier(config["kind"], path=f"{path}.kind")
    if kind != "antipodal_parallel_jaw":
        raise ValueError(
            f"Unsupported {path}.kind {kind!r}; supported kinds are ['antipodal_parallel_jaw']."
        )
    model = _decode_parallel_jaw_gripper_model(config["model"], path=f"{path}.model")
    side = (
        _identifier(config["side"], path=f"{path}.side") if "side" in config else None
    )
    root_from_tcp = (
        _geometry_pose(config["root_from_tcp"], path=f"{path}.root_from_tcp")
        if "root_from_tcp" in config
        else None
    )
    if model.model_id == "tianji_marvin_gripper":
        if side not in ("left", "right"):
            raise ValueError(
                f"{path}: Marvin geometry requires explicit left/right side."
            )
    elif side is not None:
        raise ValueError(f"{path}: side is only supported by Marvin geometry.")
    center_mode = config.get("center_mode")
    if "center_mode" in config and center_mode not in ("bounds", "centroid"):
        raise ValueError(f"{path}.center_mode must be 'bounds' or 'centroid'.")
    return _AntipodalGraspPoseGeneratorFactory(
        model_id=model.model_id,
        min_opening_width=model.min_opening_width,
        max_opening_width=model.max_opening_width,
        point_spacing=_real(
            config.get("point_spacing", 0.012),
            path=f"{path}.point_spacing",
            minimum=0,
            strict_minimum=True,
        ),
        side=side,
        root_from_tcp=root_from_tcp,
        root_from_contact_frame=(
            _geometry_pose(
                config["root_from_contact_frame"],
                path=f"{path}.root_from_contact_frame",
            )
            if "root_from_contact_frame" in config
            else None
        ),
        sample_count=(
            _integer(config["sample_count"], path=f"{path}.sample_count", minimum=1)
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
        approach_deviation_angle=(
            _real(
                config["approach_deviation_angle"],
                path=f"{path}.approach_deviation_angle",
                minimum=0,
            )
            if "approach_deviation_angle" in config
            else None
        ),
        max_candidates=(
            _integer(config["max_candidates"], path=f"{path}.max_candidates", minimum=1)
            if "max_candidates" in config
            else None
        ),
        center_mode=center_mode,
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
                "final_quaternion_xyzw",
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
        final_quaternion_xyzw=_finite_tuple(
            config["final_quaternion_xyzw"],
            path=f"{path}.final_quaternion_xyzw",
            expected_length=4,
        ),
    )


def _decode_goal_pose(
    value: object, *, path: str
) -> AbsolutePoseTarget | SceneEntityTarget:
    """Decode one closed absolute or scene-relative target declaration."""
    common = _mapping(
        value,
        path=path,
        required=frozenset({"kind"}),
        optional=frozenset(
            {
                "position",
                "quaternion_xyzw",
                "entity_id",
                "relative_pose",
                "world_orientation",
            }
        ),
    )
    kind = _identifier(common["kind"], path=f"{path}.kind")
    if kind == "pose":
        config = _mapping(
            value,
            path=path,
            required=frozenset({"kind", "position", "quaternion_xyzw"}),
        )
        return AbsolutePoseTarget(
            _finite_tuple(
                config["position"], path=f"{path}.position", expected_length=3
            ),
            _finite_tuple(
                config["quaternion_xyzw"],
                path=f"{path}.quaternion_xyzw",
                expected_length=4,
            ),
        )
    if kind == "scene_entity":
        config = _mapping(
            value,
            path=path,
            required=frozenset({"kind", "entity_id"}),
            optional=frozenset({"relative_pose", "world_orientation"}),
        )
        return SceneEntityTarget(
            _identifier(config["entity_id"], path=f"{path}.entity_id"),
            world_orientation=(
                None
                if "world_orientation" not in config
                else _finite_tuple(
                    config["world_orientation"],
                    path=f"{path}.world_orientation",
                    expected_length=9,
                )
            ),
            relative_pose=(
                None
                if "relative_pose" not in config
                else _finite_tuple(
                    config["relative_pose"],
                    path=f"{path}.relative_pose",
                    expected_length=16,
                )
            ),
        )
    raise ValueError(
        f"Unsupported {path}.kind {kind!r}; expected pose or scene_entity."
    )


def _decode_registered_lowerer(
    value: object,
    *,
    path: str,
    control_operations: tuple[ControlOperationDeclaration, ...] = (),
) -> (
    _ArticulationLinkSlideLowererFactory
    | _AxisAlignLowererFactory
    | _CoordinatedTransportLowererFactory
    | CoordinatedHoldLowererFactory
    | CoordinatedPlacementLowererFactory
    | _MoveHeldObjectLowererFactory
    | _ParkLowererFactory
    | RoutedMoveHeldObjectLowererFactory
    | _PourLowererFactory
    | _PushObjectLowererFactory
    | _RelativePlaceLowererFactory
    | ObservedRelativePlaceLowererFactory
    | UprightPlaceLowererFactory
    | StackPlaceLowererFactory
    | RetainedPourFactory
    | JawPourFactory
    | PourTransportFactory
    | PourRegionTransportFactory
    | HeldAlignmentLowererFactory
    | ClearReleasedLowererFactory
    | ConfiguredPickLowererFactory
    | PressPrepareFactory
    | TwistPrepareFactory
    | ArticulationSlideFactory
    | ArticulationParkFactory
):
    """Decode one allowlisted registered semantic lowerer factory."""
    if isinstance(value, Mapping) and value.get("kind") == "idle_arm_clearance":
        from .simulation.idle_clearance import IdleClearanceFactory, IdleClearanceRoute

        config = _mapping(value, path=path, required=frozenset({"kind", "routes"}))
        routes = tuple(
            IdleClearanceRoute(
                **_mapping(
                    item,
                    path=f"{path}.routes[{index}]",
                    required=frozenset(
                        {
                            "route_id",
                            "clear_segment",
                            "restore_segment",
                            "resource",
                            "joint_name",
                            "offset_rad",
                        }
                    ),
                    optional=frozenset({"search", "protected_control_part"}),
                )
            )
            for index, item in enumerate(
                _sequence(config["routes"], path=f"{path}.routes")
            )
        )
        return IdleClearanceFactory(routes)

    if isinstance(value, Mapping) and value.get("kind") in (
        "prismatic_slide",
        "prismatic_withdraw",
        "articulation_park",
    ):
        return _decode_prismatic_lowerer(value, path=path)
    if isinstance(value, Mapping) and value.get("kind") in (
        "press_prepare",
        "press",
        "twist_prepare",
        "twist",
    ):
        return _decode_control_lowerer(value, operations=control_operations, path=path)
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
                "call_id",
                "verify_phases",
                "object_ids",
                "receivers",
                "routes",
                "staging_routes",
                "proxy_position_tolerance_m",
                "stage_id",
                "grasp_transform_source",
                "verify_retention",
                "hands",
                "retreat_distance",
                "approach_clearance",
                "release_clearance",
                "nominal_clearance",
                "effect_contract",
                "target",
            }
        ),
    )
    kind = _identifier(common["kind"], path=f"{path}.kind")
    if kind in {"pour_transport", "pour_region_transport"}:
        config = _mapping(
            value,
            path=path,
            required=frozenset(
                {"kind", "routes", "staging_routes"}
                | (
                    {"proxy_position_tolerance_m"}
                    if kind == "pour_region_transport"
                    else set()
                )
            ),
        )
        routes = []
        for index, raw in enumerate(_sequence(config["routes"], path=f"{path}.routes")):
            route_path = f"{path}.routes[{index}]"
            route = _mapping(
                raw,
                path=route_path,
                required=frozenset({"object_id", "target_id", "pose"}),
            )
            routes.append(
                HeldObjectTransportRoute(
                    object_id=_identifier(
                        route["object_id"], path=f"{route_path}.object_id"
                    ),
                    target_id=_identifier(
                        route["target_id"], path=f"{route_path}.target_id"
                    ),
                    pose=_decode_goal_pose(route["pose"], path=f"{route_path}.pose"),
                )
            )
        stages = tuple(
            PourStageRoute(
                **_mapping(
                    raw,
                    path=f"{path}.staging_routes[{index}]",
                    required=frozenset(
                        {"object_id", "target_id", "receiver_id", "clearance_m"}
                    ),
                )
            )
            for index, raw in enumerate(
                _sequence(config["staging_routes"], path=f"{path}.staging_routes")
            )
        )
        if kind == "pour_region_transport":
            return PourRegionTransportFactory(
                tuple(routes), stages, config["proxy_position_tolerance_m"]
            )
        return PourTransportFactory(tuple(routes), stages)
    if kind == "pick":
        config = _mapping(
            value,
            path=path,
            required=frozenset({"kind", "routes"}),
            optional=frozenset({"call_id", "verify_phases", "hands"}),
        )
        routes = []
        for index, raw in enumerate(_sequence(config["routes"], path=f"{path}.routes")):
            route_path = f"{path}.routes[{index}]"
            route = _mapping(
                raw,
                path=route_path,
                required=frozenset({"object_id", "target_id"}),
                optional=frozenset(
                    {
                        "release_clearance_object_pose",
                        "release_clearance_plane_z",
                        "release_clearance_safety_margin",
                        "grasp_region",
                    }
                ),
            )
            if route.get("release_clearance_object_pose") is not None:
                route["release_clearance_object_pose"] = _decode_goal_pose(
                    route["release_clearance_object_pose"],
                    path=f"{route_path}.release_clearance_object_pose",
                )
            routes.append(PickTargetRoute(**route))
        hands = []
        for index, raw in enumerate(
            _sequence(config.get("hands", ()), path=f"{path}.hands")
        ):
            hand = _mapping(
                raw,
                path=f"{path}.hands[{index}]",
                required=frozenset({"grasp_target_id", "model_id", "envelope"}),
            )
            envelope = NominalHandEnvelope(
                **_mapping(
                    hand["envelope"],
                    path=f"{path}.hands[{index}].envelope",
                    required=frozenset(
                        {
                            "finger_length",
                            "finger_width",
                            "finger_thickness",
                            "palm_depth",
                            "opening_margin",
                        }
                    ),
                )
            )
            hands.append(
                NominalHandBinding(hand["grasp_target_id"], hand["model_id"], envelope)
            )
        return make_pick_alias_factory(
            tuple(routes),
            config.get("call_id", "simulation.pick"),
            verify_phases=config.get("verify_phases", False),
            hands=tuple(hands),
        )
    if kind == "clear_released":
        config = _mapping(
            value,
            path=path,
            required=frozenset({"kind", "routes", "hands"}),
            optional=frozenset({"retreat_distance"}),
        )
        routes = tuple(
            ReleasedHandRoute(
                **_mapping(
                    raw,
                    path=f"{path}.routes[{index}]",
                    required=frozenset({"object_id", "target_id", "minimum_height"}),
                )
            )
            for index, raw in enumerate(
                _sequence(config["routes"], path=f"{path}.routes")
            )
        )
        hands = []
        for index, raw in enumerate(_sequence(config["hands"], path=f"{path}.hands")):
            hand = _mapping(
                raw,
                path=f"{path}.hands[{index}]",
                required=frozenset({"grasp_target_id", "model_id", "envelope"}),
            )
            envelope = NominalHandEnvelope(
                **_mapping(
                    hand["envelope"],
                    path=f"{path}.hands[{index}].envelope",
                    required=frozenset(
                        {
                            "finger_length",
                            "finger_width",
                            "finger_thickness",
                            "palm_depth",
                            "opening_margin",
                        }
                    ),
                )
            )
            hands.append(
                NominalHandBinding(hand["grasp_target_id"], hand["model_id"], envelope)
            )
        return ClearReleasedLowererFactory(
            routes, tuple(hands), retreat_distance=config.get("retreat_distance", 0.15)
        )
    if kind == "align_held":
        config = _mapping(
            value,
            path=path,
            required=frozenset({"kind", "routes"}),
            optional=frozenset({"verify_retention"}),
        )
        routes = []
        for index, raw in enumerate(_sequence(config["routes"], path=f"{path}.routes")):
            route = _mapping(
                raw,
                path=f"{path}.routes[{index}]",
                required=frozenset(
                    {
                        "object_id",
                        "target_id",
                        "preserve_yaw",
                        "axis_direction_free",
                        "local_axis",
                    }
                ),
                optional=frozenset({"position"}),
            )
            routes.append(HeldAlignmentRoute(**route))
        return HeldAlignmentLowererFactory(
            tuple(routes), verify_retention=config.get("verify_retention", False)
        )
    if "effect_contract" in value and kind not in {
        "move_held_object",
        "place_relative",
        "place_upright",
        "stack_place",
    }:
        raise ValueError(f"{path}.effect_contract is unsupported for {kind!r}.")
    if kind == "axis_align":
        config = _mapping(
            value,
            path=path,
            required=frozenset({"kind", "object_ids"}),
        )
        return _AxisAlignLowererFactory(
            object_ids=_identifier_tuple(
                config["object_ids"],
                path=f"{path}.object_ids",
            )
        )
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
                }
            ),
            optional=frozenset({"target_pose_mode", "translation_axis"}),
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
            translation_axis=(
                _finite_tuple(
                    config["translation_axis"],
                    path=f"{path}.translation_axis",
                    expected_length=3,
                )
                if "translation_axis" in config
                else None
            ),
            target_pose_mode=target_pose_mode,
        )
    if kind in {"move_held_object", "drawer_transport"} and "routes" in value:
        config = _mapping(
            value,
            path=path,
            required=frozenset({"kind", "routes"}),
            optional=(
                frozenset({"effect_contract"})
                if kind == "move_held_object"
                else frozenset()
            ),
        )
        routes = []
        for index, item in enumerate(
            _sequence(config["routes"], path=f"{path}.routes")
        ):
            route_path = f"{path}.routes[{index}]"
            route = _mapping(
                item,
                path=route_path,
                required=frozenset({"object_id", "target_id", "pose"}),
            )
            routes.append(
                HeldObjectTransportRoute(
                    object_id=_identifier(
                        route["object_id"], path=f"{route_path}.object_id"
                    ),
                    target_id=_identifier(
                        route["target_id"], path=f"{route_path}.target_id"
                    ),
                    pose=_decode_goal_pose(route["pose"], path=f"{route_path}.pose"),
                )
            )
        if kind == "drawer_transport":
            return DrawerTransportLowererFactory(routes=tuple(routes))
        try:
            contract = TransportEffectContract(
                config.get(
                    "effect_contract", TransportEffectContract.GUARDED_ATTACHMENT.value
                )
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{path}.effect_contract is not a supported transport contract."
            ) from error
        return (
            TerminalAttachmentTransportLowererFactory
            if contract is TransportEffectContract.TERMINAL_ATTACHMENT
            else RoutedMoveHeldObjectLowererFactory
        )(routes=tuple(routes))
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
    if kind in {"coordinated_transport", "coordinated_hold"}:
        config = _mapping(
            value,
            path=path,
            required=frozenset({"kind", "routes"}),
        )
        routes: list[_CoordinatedTransportRoute] = []
        for index, route_value in enumerate(
            _sequence(config["routes"], path=f"{path}.routes")
        ):
            route_path = f"{path}.routes[{index}]"
            route = _mapping(
                route_value,
                path=route_path,
                required=frozenset({"object_id", "target_id"}),
                optional=frozenset(
                    {
                        "reference_entity_id",
                        "relative_pose",
                        "world_displacement",
                        "relation",
                    }
                ),
            )
            has_reference = "reference_entity_id" in route
            has_pose = "relative_pose" in route
            has_displacement = "world_displacement" in route
            if "relation" in route:
                if (
                    kind != "coordinated_transport"
                    or route["relation"] != "on"
                    or not has_reference
                    or not has_displacement
                    or has_pose
                ):
                    raise ValueError(
                        "Coordinated on placement requires release, a distinct rigid reference and world offset."
                    )
            elif has_reference != has_pose or has_reference == has_displacement:
                raise ValueError(
                    f"{route_path} must declare exactly one of "
                    "reference_entity_id with relative_pose or "
                    "world_displacement."
                )
            special = "relation" in route or (
                has_displacement
                and all(value == 0 for value in route["world_displacement"])
            )
            route_type = (
                CoordinatedTargetRoute if special else _CoordinatedTransportRoute
            )
            routes.append(
                route_type(
                    **({"relation": route.get("relation")} if special else {}),
                    object_id=_identifier(
                        route["object_id"], path=f"{route_path}.object_id"
                    ),
                    target_id=_identifier(
                        route["target_id"], path=f"{route_path}.target_id"
                    ),
                    reference_entity_id=(
                        _identifier(
                            route["reference_entity_id"],
                            path=f"{route_path}.reference_entity_id",
                        )
                        if has_reference
                        else None
                    ),
                    relative_pose=(
                        _finite_tuple(
                            route["relative_pose"],
                            path=f"{route_path}.relative_pose",
                            expected_length=16,
                        )
                        if has_pose
                        else None
                    ),
                    world_displacement=(
                        _finite_tuple(
                            route["world_displacement"],
                            path=f"{route_path}.world_displacement",
                            expected_length=3,
                        )
                        if has_displacement
                        else None
                    ),
                )
            )
        if any(type(route) is CoordinatedTargetRoute for route in routes):
            routes = [
                (
                    CoordinatedTargetRoute(**asdict(route))
                    if type(route) is _CoordinatedTransportRoute
                    else route
                )
                for route in routes
            ]
            factory_type = (
                _CoordinatedRetainedTargetFactory
                if kind == "coordinated_hold"
                else CoordinatedPlacementLowererFactory
            )
            return factory_type(routes=tuple(routes))
        factory_type = (
            CoordinatedHoldLowererFactory
            if kind == "coordinated_hold"
            else _CoordinatedTransportLowererFactory
        )
        return factory_type(routes=tuple(routes))
    if kind in {"place_relative", "place_upright", "stack_place"}:
        config = _mapping(
            value,
            path=path,
            required=frozenset({"kind", "routes"}),
            optional=frozenset(
                (
                    {"approach_clearance", "release_clearance", "nominal_clearance"}
                    if kind == "stack_place"
                    else {"grasp_transform_source"}
                )
                | {"effect_contract"}
            ),
        )
        source = config.get(
            "grasp_transform_source",
            "observed" if kind in {"place_upright", "stack_place"} else "verified",
        )
        if kind == "place_upright" and source != "observed":
            raise ValueError("Upright placement requires observed grasp geometry.")
        if source not in {"verified", "observed"}:
            raise ValueError("grasp_transform_source must be verified or observed.")
        observed = source == "observed"
        expected_contract = (
            PlacementEffectContract.TERMINAL_RELEASE
            if observed
            else PlacementEffectContract.PHASED_RELEASE
        )
        try:
            contract = PlacementEffectContract(
                config.get("effect_contract", expected_contract.value)
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{path}.effect_contract is not a supported release contract."
            ) from error
        if contract is not expected_contract:
            raise ValueError(
                f"{path}: {source} placement requires effect_contract={expected_contract.value!r}."
            )
        route_type = ObservedRelativePlaceBinding if observed else _RelativePlaceRoute
        routes = []
        for index, route_value in enumerate(
            _sequence(config["routes"], path=f"{path}.routes")
        ):
            route_path = f"{path}.routes[{index}]"
            route = _mapping(
                route_value,
                path=route_path,
                required=frozenset(
                    {
                        "object_id",
                        "reference_entity_id",
                        "relation",
                        "world_displacement",
                    }
                ),
                optional=frozenset(
                    {"stage_id"}
                    | (
                        {"world_yaw_offset", "reference_link_id", "reference_part"}
                        if observed
                        else set()
                    )
                ),
            )
            routes.append(
                route_type(
                    object_id=_identifier(
                        route["object_id"],
                        path=f"{route_path}.object_id",
                    ),
                    reference_entity_id=_identifier(
                        route["reference_entity_id"],
                        path=f"{route_path}.reference_entity_id",
                    ),
                    relation=_identifier(
                        route["relation"],
                        path=f"{route_path}.relation",
                    ),
                    world_displacement=_finite_tuple(
                        route["world_displacement"],
                        path=f"{route_path}.world_displacement",
                        expected_length=3,
                    ),
                    **(
                        {
                            "world_yaw_offset": _real(
                                route.get("world_yaw_offset", 0.0),
                                path=f"{route_path}.world_yaw_offset",
                            ),
                            "reference_link_id": (
                                None
                                if route.get("reference_link_id") is None
                                else _identifier(
                                    route["reference_link_id"],
                                    path=f"{route_path}.reference_link_id",
                                )
                            ),
                            "reference_part": (
                                None
                                if route.get("reference_part") is None
                                else _identifier(
                                    route["reference_part"],
                                    path=f"{route_path}.reference_part",
                                )
                            ),
                        }
                        if observed
                        else {}
                    ),
                    stage_id=(
                        _identifier(route["stage_id"], path=f"{route_path}.stage_id")
                        if str(route.get("stage_id", "")).strip()
                        else ""
                    ),
                )
            )
        if kind == "stack_place":
            return StackPlaceLowererFactory(
                routes=tuple(routes),
                **{
                    name: config[name]
                    for name in (
                        "approach_clearance",
                        "release_clearance",
                        "nominal_clearance",
                    )
                    if name in config
                },
            )
        return (
            UprightPlaceLowererFactory
            if kind == "place_upright"
            else (
                ObservedRelativePlaceLowererFactory
                if observed
                else _RelativePlaceLowererFactory
            )
        )(routes=tuple(routes))
    if kind == "park":
        _mapping(
            value,
            path=path,
            required=frozenset({"kind"}),
        )
        return _ParkLowererFactory()
    if kind == "jaw_pour":
        config = _mapping(
            value, path=path, required=frozenset({"kind", "object_id", "receivers"})
        )
        receivers = []
        for index, value in enumerate(
            _sequence(config["receivers"], path=f"{path}.receivers")
        ):
            pair = _sequence(value, path=f"{path}.receivers[{index}]")
            if len(pair) != 2:
                raise ValueError("Jaw Pour receivers require occurrence/entity pairs.")
            receivers.append(
                tuple(
                    _identifier(item, path=f"{path}.receivers[{index}]")
                    for item in pair
                )
            )
        return JawPourFactory(
            object_id=_identifier(config["object_id"], path=f"{path}.object_id"),
            receivers=tuple(receivers),
        )
    if kind == "retained_pour":
        config = _mapping(value, path=path, required=frozenset({"kind", "object_id"}))
        return RetainedPourFactory(
            _identifier(config["object_id"], path=f"{path}.object_id")
        )
    if kind == "pour":
        config = _mapping(
            value,
            path=path,
            required=frozenset({"kind", "object_id"}),
            optional=frozenset({"target"}),
        )
        target = None
        if "target" in config:
            target_config = _mapping(
                config["target"],
                path=f"{path}.target",
                required=frozenset(
                    {
                        "receiver_id",
                        "source_mouth_pose",
                        "receiver_mouth_pose",
                        "receiver_radius",
                        "mouth_radius",
                    }
                ),
            )
            target = _PourTargetBinding(
                receiver_id=_identifier(
                    target_config["receiver_id"], path=f"{path}.target.receiver_id"
                ),
                source_mouth_pose=_finite_tuple(
                    target_config["source_mouth_pose"],
                    path=f"{path}.target.source_mouth_pose",
                    expected_length=16,
                ),
                receiver_mouth_pose=_finite_tuple(
                    target_config["receiver_mouth_pose"],
                    path=f"{path}.target.receiver_mouth_pose",
                    expected_length=16,
                ),
                receiver_radius=_real(
                    target_config["receiver_radius"],
                    path=f"{path}.target.receiver_radius",
                ),
                mouth_radius=_real(
                    target_config["mouth_radius"], path=f"{path}.target.mouth_radius"
                ),
            )
        return _PourLowererFactory(
            object_id=_identifier(config["object_id"], path=f"{path}.object_id"),
            target=target,
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
        "['articulation_link_slide', 'axis_align', 'coordinated_transport', 'coordinated_hold', "
        "'move_held_object', 'park', 'place_relative', 'pour', 'push_object']."
    )


def _decode_control_part_evidence(
    value: object,
    *,
    path: str,
) -> _JointPositionConstraintEvidenceProviderFactory | NativeGripEvidenceFactory:
    """Decode one allowlisted control-part evidence provider factory."""
    if isinstance(value, Mapping) and value.get("kind") == "native_grip":
        return _decode_native_grip_evidence(value, path=path)
    config = _mapping(
        value,
        path=path,
        required=frozenset(
            {
                "kind",
                "control_parts",
                "open_qpos",
                "minimum_displacement",
            }
        ),
        optional=frozenset({"object_ids"}),
    )
    kind = _identifier(config["kind"], path=f"{path}.kind")
    if kind != "joint_position_constraint":
        raise ValueError(
            f"Unsupported {path}.kind {kind!r}; supported kinds are "
            "['joint_position_constraint', 'native_grip']."
        )
    return _JointPositionConstraintEvidenceProviderFactory(
        control_parts=_identifier_tuple(
            config["control_parts"],
            path=f"{path}.control_parts",
        ),
        object_ids=(
            None
            if "object_ids" not in config
            else _identifier_tuple(
                config["object_ids"],
                path=f"{path}.object_ids",
            )
        ),
        open_qpos=_finite_tuple(config["open_qpos"], path=f"{path}.open_qpos"),
        minimum_displacement=_real(
            config["minimum_displacement"],
            path=f"{path}.minimum_displacement",
            minimum=0.0,
            strict_minimum=True,
        ),
    )


def _decode_driver_openings(
    value: object, *, path: str
) -> tuple[DriverOpeningBinding, ...]:
    openings = []
    for index, raw in enumerate(_sequence(value, path=f"{path}")):
        item = _mapping(
            raw,
            path=f"{path}[{index}]",
            required=frozenset(
                {
                    "control_part",
                    "joint_names",
                    "driver_indices",
                    "open_qpos",
                    "tolerance",
                }
            ),
        )
        openings.append(
            DriverOpeningBinding(
                control_part=_identifier(
                    item["control_part"],
                    path=f"{path}[{index}].control_part",
                ),
                joint_names=_identifier_tuple(
                    item["joint_names"],
                    path=f"{path}[{index}].joint_names",
                ),
                driver_indices=tuple(
                    _sequence(
                        item["driver_indices"],
                        path=f"{path}[{index}].driver_indices",
                    )
                ),
                open_qpos=_finite_tuple(
                    item["open_qpos"],
                    path=f"{path}[{index}].open_qpos",
                ),
                tolerance=_real(
                    item["tolerance"],
                    path=f"{path}[{index}].tolerance",
                    minimum=0.0,
                    strict_minimum=True,
                ),
            )
        )
    return tuple(openings)


def _decode_recorded_contacts(
    value: object, *, path: str
) -> SimulationRecordedContactCfg:
    config = _mapping(
        value,
        path=path,
        required=frozenset({"sensor_uid", "retention", "pad_bindings"}),
        optional=frozenset({"opening_bindings", "observer_name"}),
    )
    retained = _mapping(
        config["retention"],
        path=f"{path}.retention",
        required=frozenset({"step_dt"}),
        optional=frozenset(
            {
                "max_control_steps",
                "max_contact_rows",
                "max_substeps_per_control",
                "max_errors_per_env",
            }
        ),
    )
    pads = []
    for index, value in enumerate(
        _sequence(config["pad_bindings"], path=f"{path}.pad_bindings")
    ):
        pair = _sequence(value, path=f"{path}.pad_bindings[{index}]")
        if len(pair) != 2:
            raise ValueError("Recorded pad bindings require part/link-list pairs.")
        pads.append(
            (
                _identifier(pair[0], path=f"{path}.pad_bindings[{index}].part"),
                _identifier_tuple(pair[1], path=f"{path}.pad_bindings[{index}].links"),
            )
        )
    return SimulationRecordedContactCfg(
        sensor_uid=_identifier(config["sensor_uid"], path=f"{path}.sensor_uid"),
        retention=ContactTimelineCfg(**retained),
        pad_bindings=tuple(pads),
        opening_bindings=_decode_driver_openings(
            config.get("opening_bindings", ()), path=f"{path}.opening_bindings"
        ),
        observer_name=config.get("observer_name", "task_program.recorded_contacts"),
    ).snapshot()


@dataclass(frozen=True, slots=True)
class _DecodedRuntimeServices:
    """Typed configured services separated by registration ownership."""

    recorded_contacts_cfg: SimulationRecordedContactCfg | None = None
    cargo_envelope_cfg: SimulationCargoEnvelopeCfg | None = None

    planning_services: SimulationPlanningServicesCfg | None = None
    invocation_policy_bindings: tuple[InvocationPolicyBinding, ...] = ()
    grasp_pose_generators: tuple[
        tuple[str, _AntipodalGraspPoseGeneratorFactory], ...
    ] = ()
    hand_geometries: tuple[tuple[str, _RobotHandGeometryCfgFactory], ...] = ()
    handover_pose_providers: tuple[ConfiguredHandOverPoseProvider, ...] = ()
    registered_semantic_lowerers: tuple[
        _ArticulationLinkSlideLowererFactory
        | _AxisAlignLowererFactory
        | _CoordinatedTransportLowererFactory
        | CoordinatedHoldLowererFactory
        | CoordinatedPlacementLowererFactory
        | _MoveHeldObjectLowererFactory
        | _ParkLowererFactory
        | RoutedMoveHeldObjectLowererFactory
        | _PourLowererFactory
        | _PushObjectLowererFactory
        | _RelativePlaceLowererFactory
        | ObservedRelativePlaceLowererFactory
        | UprightPlaceLowererFactory
        | StackPlaceLowererFactory
        | RetainedPourFactory
        | JawPourFactory
        | PourTransportFactory
        | PourRegionTransportFactory
        | HeldAlignmentLowererFactory
        | ClearReleasedLowererFactory
        | ConfiguredPickLowererFactory
        | PressPrepareFactory
        | TwistPrepareFactory
        | ArticulationSlideFactory
        | ArticulationParkFactory,
        ...,
    ] = ()
    control_part_evidence: (
        _JointPositionConstraintEvidenceProviderFactory
        | NativeGripEvidenceFactory
        | None
    ) = None
    nominal_motion_sources: tuple[_ConfiguredPrototypeSource, ...] = ()
    control_operations: tuple[ControlOperationDeclaration, ...] = ()
    post_policy_bindings: tuple[PostPolicyPresetBinding, ...] = ()
    settle_presets: (
        tuple[
            tuple[
                str,
                DynamicSettleMonitorCfg
                | SimulationStabilityConditionCfg
                | SimulationJointStabilityConditionCfg
                | SimulationControlConditionCfg,
            ],
            ...,
        ]
        | None
    ) = None


def _decode_settle_presets(value: object, *, path: str) -> tuple[
    tuple[
        str,
        DynamicSettleMonitorCfg
        | SimulationStabilityConditionCfg
        | SimulationJointStabilityConditionCfg
        | SimulationControlConditionCfg,
    ],
    ...,
]:
    """Decode a complete closed named table for the existing registration owner."""
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{path} must be a nonempty mapping.")
    result = []
    for name, item in value.items():
        name = _identifier(name, path=f"{path} keys")
        item_path = f"{path}.{name}"
        if not isinstance(item, Mapping):
            raise TypeError(f"{item_path} must be a mapping.")
        if item.get("kind") in {"press_control", "twist_control"}:
            config = _mapping(
                item,
                path=item_path,
                required=frozenset({"kind", "entity", "operation_id", "phase"}),
                optional=frozenset({"terminal"}),
            )
            preset = SimulationControlConditionCfg(
                entity=config["entity"],
                operation_id=config["operation_id"],
                family="press" if config["kind"] == "press_control" else "twist",
                phase=config["phase"],
                terminal=config.get("terminal", False),
            ).snapshot()
            result.append((name, preset))
            continue
        if item.get("kind") == "joint_position_and_stability":
            config = _mapping(
                item,
                path=item_path,
                required=frozenset({"kind", "entity", "joint", "target", "tolerance"}),
                optional=frozenset(
                    {
                        "duration",
                        "timeout",
                        "position_drift",
                        "invalid_observation_policy",
                        "max_observation_samples",
                    }
                ),
            )
            preset = SimulationJointStabilityConditionCfg(
                **{
                    name: deepcopy(value)
                    for name, value in config.items()
                    if name != "kind"
                }
            ).snapshot()
        elif "kind" in item:
            config = _mapping(
                item,
                path=item_path,
                required=frozenset({"kind", "entity"}),
                optional=frozenset(
                    field.name for field in fields(SimulationStabilityConditionCfg)
                )
                - {"kind", "entity"},
            )
            kwargs = deepcopy(dict(config))
            sensor = kwargs.get("support_sensor")
            if sensor is not None:
                sensor = _mapping(
                    sensor,
                    path=f"{item_path}.support_sensor",
                    required=frozenset({"uid"}),
                    optional=frozenset({"max_contacts_per_env"}),
                )
                kwargs["support_sensor"] = SupportSensorDeclarationCfg(
                    **sensor
                ).snapshot()
            preset = SimulationStabilityConditionCfg(**kwargs).snapshot()
        else:
            config = _mapping(
                item,
                path=item_path,
                required=frozenset(),
                optional=frozenset(
                    field.name for field in fields(DynamicSettleMonitorCfg)
                )
                | {"measurement"},
            )
            cfg_type = (
                SimulationSettleMonitorCfg
                if "measurement" in config
                else DynamicSettleMonitorCfg
            )
            preset = cfg_type(**deepcopy(dict(config))).snapshot()
        result.append((name, preset))
    return tuple(result)


def _decode_runtime_services(
    value: object,
    *,
    robot_profile_binding: SimulationRobotSkillProfileBinding | None = None,
) -> _DecodedRuntimeServices:
    """Decode the allowlisted live-service portion of an integration."""
    path = "integration.runtime_services"
    config = _mapping(
        value,
        path=path,
        required=frozenset(),
        optional=frozenset(
            {
                "grasp_pose_generators",
                "hand_geometries",
                "handover_pose_providers",
                "registered_semantic_lowerers",
                "control_operations",
                "cargo_envelope",
                "recorded_contacts",
                "control_part_evidence",
                "nominal_motion_sources",
                "settle_presets",
                "planning_services",
                "invocation_policy_bindings",
                "post_policy_bindings",
            }
        ),
    )
    cargo_cfg = None
    if "cargo_envelope" in config:
        cargo = _mapping(
            config["cargo_envelope"],
            path=f"{path}.cargo_envelope",
            required=frozenset({"carrier_uids"}),
            optional=frozenset({"excluded_uids"}),
        )
        cargo_cfg = SimulationCargoEnvelopeCfg(
            carrier_uids=tuple(
                _identifier(uid, path=f"{path}.cargo_envelope.carrier_uids")
                for uid in _sequence(
                    cargo["carrier_uids"], path=f"{path}.cargo_envelope.carrier_uids"
                )
            ),
            excluded_uids=tuple(
                _identifier(uid, path=f"{path}.cargo_envelope.excluded_uids")
                for uid in _sequence(
                    cargo.get("excluded_uids", ()),
                    path=f"{path}.cargo_envelope.excluded_uids",
                )
            ),
        ).snapshot()
    generators_value = config.get("grasp_pose_generators", {})
    if not isinstance(generators_value, Mapping):
        raise TypeError(f"{path}.grasp_pose_generators must be a mapping.")
    generators = tuple(
        sorted(
            (
                (
                    _identifier(target_id, path=f"{path}.grasp_pose_generators keys"),
                    _decode_grasp_generator(
                        item, path=f"{path}.grasp_pose_generators.{target_id}"
                    ),
                )
                for target_id, item in generators_value.items()
            )
        )
    )
    hand_values = config.get("hand_geometries", {})
    if not isinstance(hand_values, Mapping):
        raise TypeError(f"{path}.hand_geometries must be a mapping.")
    if hand_values and robot_profile_binding is None:
        raise ValueError("Hand geometry declarations require a decoded robot profile.")
    hand_geometries = tuple(
        sorted(
            (
                (
                    _identifier(target_id, path=f"{path}.hand_geometries keys"),
                    _decode_hand_geometry(
                        item,
                        path=f"{path}.hand_geometries.{target_id}",
                        target_id=target_id,
                        profile=robot_profile_binding,
                    ),
                )
                for target_id, item in hand_values.items()
            )
        )
    )
    handover = tuple(
        (
            _decode_handover_pose_provider(
                item, path=f"{path}.handover_pose_providers[{index}]"
            )
            for index, item in enumerate(
                _sequence(
                    config.get("handover_pose_providers", ()),
                    path=f"{path}.handover_pose_providers",
                )
            )
        )
    )
    operations = tuple(
        (
            _decode_control_operation(item, path=f"{path}.control_operations[{index}]")
            for index, item in enumerate(
                _sequence(
                    config.get("control_operations", ()),
                    path=f"{path}.control_operations",
                )
            )
        )
    )
    if len({item.call.operation_id for item in operations}) != len(operations):
        raise ValueError("Control operation IDs must be unique.")
    if not {item.grasp_target_id for item in operations} <= {
        key for key, _ in hand_geometries
    }:
        raise ValueError(
            "Control operations require explicitly configured hand geometry."
        )
    lowerers = tuple(
        (
            _decode_registered_lowerer(
                item,
                path=f"{path}.registered_semantic_lowerers[{index}]",
                control_operations=operations,
            )
            for index, item in enumerate(
                _sequence(
                    config.get("registered_semantic_lowerers", ()),
                    path=f"{path}.registered_semantic_lowerers",
                )
            )
        )
    )
    lowerer_call_ids = tuple((factory.call_id for factory in lowerers))
    if len(set(lowerer_call_ids)) != len(lowerer_call_ids):
        raise ValueError(
            f"{path}.registered_semantic_lowerers contains duplicate call IDs."
        )
    evidence = (
        None
        if "control_part_evidence" not in config
        else _decode_control_part_evidence(
            config["control_part_evidence"], path=f"{path}.control_part_evidence"
        )
    )
    nominal = []
    for index, value in enumerate(
        _sequence(
            config.get("nominal_motion_sources", ()),
            path=f"{path}.nominal_motion_sources",
        )
    ):
        item_path = f"{path}.nominal_motion_sources[{index}]"
        item = _mapping(
            value,
            path=item_path,
            required=frozenset(
                {
                    "call_index",
                    "program_fingerprint",
                    "provider_id",
                    "revision",
                    "prototype_manifest",
                    "prototype_sha256",
                    "object_entity_id",
                }
            ),
            optional=frozenset(
                {
                    "target_entity_id",
                    "target_entity_to_reference",
                    "authority",
                    "gripper_mode",
                    "require_collision_validation",
                    "refinement",
                }
            ),
        )
        if item["provider_id"] != "real2sim.prototype" or item["revision"] != "1":
            raise ValueError(f"{item_path} names an unknown provider or revision")
        refinement = None
        if item.get("refinement") is not None:
            from embodichain.lab.sim.motion.refinement import MotionRefinementCfg

            options = _mapping(
                item["refinement"],
                path=f"{item_path}.refinement",
                required=frozenset(),
                optional=frozenset(
                    {
                        "max_translation_m",
                        "max_rotation_rad",
                        "smoothing_weight",
                        "project_pick_lift",
                        "project_pick_approach",
                        "project_free_boundaries",
                        "pick_entry_projection_samples",
                        "max_candidates",
                        "seed",
                        "support_margin_m",
                        "max_release_gap_m",
                        "max_iterations",
                    }
                ),
            )
            refinement = MotionRefinementCfg(**options)
        declaration = _ConfiguredPrototypeSource(
            call_index=_integer(
                item["call_index"], path=f"{item_path}.call_index", minimum=0
            ),
            program_fingerprint=_identifier(
                item["program_fingerprint"], path=f"{item_path}.program_fingerprint"
            ),
            prototype_manifest=_identifier(
                item["prototype_manifest"], path=f"{item_path}.prototype_manifest"
            ),
            prototype_sha256=_identifier(
                item["prototype_sha256"], path=f"{item_path}.prototype_sha256"
            ),
            object_entity_id=_identifier(
                item["object_entity_id"], path=f"{item_path}.object_entity_id"
            ),
            target_entity_id=_optional_identifier(
                item.get("target_entity_id"), path=f"{item_path}.target_entity_id"
            ),
            target_entity_to_reference=(
                None
                if item.get("target_entity_to_reference") is None
                else _finite_tuple(
                    item["target_entity_to_reference"],
                    path=f"{item_path}.target_entity_to_reference",
                    expected_length=16,
                )
            ),
            authority=_identifier(
                item.get("authority", "qpos"), path=f"{item_path}.authority"
            ),
            gripper_mode=_identifier(
                item.get("gripper_mode", "recorded"), path=f"{item_path}.gripper_mode"
            ),
            require_collision_validation=_boolean(
                item.get("require_collision_validation", True),
                path=f"{item_path}.require_collision_validation",
            ),
            refinement=refinement,
        )
        declaration.validate_source()
        nominal.append(declaration)
    if len({item.call_index for item in nominal}) != len(nominal):
        raise ValueError(f"{path}.nominal_motion_sources has duplicate call indices")
    planning = None
    if "planning_services" in config:
        planning = _mapping(
            config["planning_services"],
            path=f"{path}.planning_services",
            required=frozenset({"kind"}),
            optional=frozenset(
                {
                    "revision",
                    "observed_curobo",
                    "collision_roster",
                    "place_scene_bindings",
                    "held_staging_bindings",
                    "pick_closing_bindings",
                    "released_hand_continuations",
                    "pour_projection_bindings",
                    "slide_preshape_bindings",
                    "place_continuation_bindings",
                    "drawer_bindings",
                    "measured_rigid_bindings",
                    "free_motion_query",
                    "capture_root_pose",
                    "capture_native_revision",
                    "native_handover_scene",
                    "capture_coordinated_cargo",
                    "coordinated_grasps",
                    "hold_qpos_source",
                }
            ),
        )
        planning = dict(planning)
        if "collision_roster" in planning:
            roster = _mapping(
                planning["collision_roster"],
                path=f"{path}.planning_services.collision_roster",
                required=frozenset({"rigid_objects", "articulation_links"}),
            )
            planning["collision_roster"] = CollisionSceneRosterCfg(**roster).snapshot()
        if "measured_rigid_bindings" in planning:
            planning["measured_rigid_bindings"] = tuple(
                (
                    MeasuredRigidInvocationBinding(
                        **_mapping(
                            value,
                            path=f"{path}.planning_services.measured_rigid_bindings[{index}]",
                            required=frozenset({"invocation_id", "object_id"}),
                            optional=frozenset({"container_id"}),
                        )
                    )
                    for index, value in enumerate(
                        _sequence(
                            planning["measured_rigid_bindings"],
                            path=f"{path}.planning_services.measured_rigid_bindings",
                        )
                    )
                )
            )
        if "drawer_bindings" in planning:
            from embodichain.lab.sim.scene.articulation_inputs import PrismaticBinding

            rows = []
            for index, item in enumerate(
                _sequence(
                    planning["drawer_bindings"],
                    path=f"{path}.planning_services.drawer_bindings",
                )
            ):
                row = dict(
                    _mapping(
                        item,
                        path=f"{path}.planning_services.drawer_bindings[{index}]",
                        required=frozenset(
                            {
                                "invocation_id",
                                "operation",
                                "target_id",
                                "link_entity_id",
                                "object_id",
                                "source",
                                "lower",
                                "upper",
                            }
                        ),
                    )
                )
                row["source"] = PrismaticBinding.decode(row["source"])
                rows.append(DrawerInvocationBinding(**row))
            planning["drawer_bindings"] = tuple(rows)
        if "pour_projection_bindings" in planning:
            planning["pour_projection_bindings"] = tuple(
                (
                    PourProjectionBinding(
                        **_mapping(
                            value,
                            path=f"{path}.planning_services.pour_projection_bindings[{index}]",
                            required=frozenset(
                                {
                                    "pick_invocation_id",
                                    "stage_invocation_id",
                                    "pour_invocation_id",
                                    "return_invocation_id",
                                    "place_invocation_id",
                                }
                            ),
                        )
                    )
                    for index, value in enumerate(
                        _sequence(
                            planning["pour_projection_bindings"],
                            path=f"{path}.planning_services.pour_projection_bindings",
                        )
                    )
                )
            )
        if "coordinated_grasps" in planning:
            pairs = []
            for index, raw in enumerate(
                _sequence(
                    planning["coordinated_grasps"],
                    path=f"{path}.planning_services.coordinated_grasps",
                )
            ):
                pair = _mapping(
                    raw,
                    path=f"{path}.planning_services.coordinated_grasps[{index}]",
                    required=frozenset(
                        {"left_grasp_target_id", "right_grasp_target_id", "contact"}
                    ),
                )
                contact = _mapping(
                    pair["contact"],
                    path=f"{path}.planning_services.coordinated_grasps[{index}].contact",
                    required=frozenset(
                        {
                            "contact_tolerance",
                            "minimum_normal_cosine",
                            "normal_probe_distance",
                            "max_candidates",
                            "candidate_chunk_size",
                            "cargo_clearance_reserve",
                        }
                    ),
                )
                pairs.append(
                    CoordinatedGraspServiceCfg(
                        left_grasp_target_id=pair["left_grasp_target_id"],
                        right_grasp_target_id=pair["right_grasp_target_id"],
                        contact=CoordinatedGraspCfg(**contact),
                    ).snapshot()
                )
            planning["coordinated_grasps"] = tuple(pairs)
        if "place_continuation_bindings" in planning:
            selected = []
            for index, value in enumerate(
                _sequence(
                    planning["place_continuation_bindings"],
                    path=f"{path}.planning_services.place_continuation_bindings",
                )
            ):
                item = _mapping(
                    value,
                    path=f"{path}.planning_services.place_continuation_bindings[{index}]",
                    required=frozenset(
                        {
                            "program_fingerprint",
                            "source_invocation_id",
                            "place_invocation_id",
                        }
                    ),
                    optional=frozenset({"transport_invocation_id", "kind"}),
                )
                declaration = dict(item)
                kind = declaration.pop("kind", "pick")
                if kind not in {"pick", "held_alignment"}:
                    raise ValueError("Unknown Place continuation binding kind.")
                selected.append(
                    (
                        PlaceContinuationBinding
                        if kind == "pick"
                        else HeldAlignmentContinuationBinding
                    )(**declaration)
                )
            planning["place_continuation_bindings"] = tuple(selected)
        if "released_hand_continuations" in planning:
            values = []
            for index, value in enumerate(
                _sequence(
                    planning["released_hand_continuations"],
                    path=f"{path}.planning_services.released_hand_continuations",
                )
            ):
                row = dict(
                    _mapping(
                        value,
                        path=f"{path}.planning_services.released_hand_continuations[{index}]",
                        required=frozenset(
                            {
                                "program_fingerprint",
                                "release_invocation_id",
                                "clearance_invocation_id",
                                "park_invocation_ids",
                            }
                        ),
                    )
                )
                row["park_invocation_ids"] = tuple(
                    _sequence(
                        row["park_invocation_ids"],
                        path=f"{path}.planning_services.released_hand_continuations[{index}].park_invocation_ids",
                    )
                )
                values.append(ReleasedHandContinuationBinding(**row))
            planning["released_hand_continuations"] = tuple(values)
        if "pick_closing_bindings" in planning:
            selected = []
            for index, value in enumerate(
                _sequence(
                    planning["pick_closing_bindings"],
                    path=f"{path}.planning_services.pick_closing_bindings",
                )
            ):
                entry_path = f"{path}.planning_services.pick_closing_bindings[{index}]"
                item = dict(
                    _mapping(
                        value,
                        path=entry_path,
                        required=frozenset({"invocation_id", "object_id"}),
                        optional=frozenset(
                            {
                                "opening_reference_id",
                                "opening_object_target",
                                "opening_place_invocation_id",
                            }
                        ),
                    )
                )
                if item.get("opening_object_target") is not None:
                    item["opening_object_target"] = _decode_goal_pose(
                        item["opening_object_target"],
                        path=f"{entry_path}.opening_object_target",
                    )
                selected.append(PickClosingBinding(**item))
            planning["pick_closing_bindings"] = tuple(selected)
        if "slide_preshape_bindings" in planning:
            planning["slide_preshape_bindings"] = tuple(
                (
                    SlidePreshapeBinding(
                        **_mapping(
                            item,
                            path=f"{path}.planning_services.slide_preshape_bindings[{index}]",
                            required=frozenset(
                                {"invocation_id", "grasp_target_id", "table_uid"}
                            ),
                        )
                    )
                    for index, item in enumerate(
                        _sequence(
                            planning["slide_preshape_bindings"],
                            path=f"{path}.planning_services.slide_preshape_bindings",
                        )
                    )
                )
            )
            if any(
                (
                    binding.grasp_target_id not in dict(hand_geometries)
                    for binding in planning["slide_preshape_bindings"]
                )
            ):
                raise ValueError(
                    "Slide preshape requires declared hand geometry calibration."
                )
        if "held_staging_bindings" in planning:
            planning["held_staging_bindings"] = tuple(
                (
                    HeldStagingInvocationBinding(
                        **_mapping(
                            item,
                            path=f"{path}.planning_services.held_staging_bindings[{index}]",
                            required=frozenset({"invocation_id", "rigid_entity_id"}),
                            optional=frozenset(
                                {
                                    "articulation_scene",
                                    "predecessor_invocation_id",
                                    "predecessor_call_id",
                                    "resource_id",
                                }
                            ),
                        )
                    )
                    for index, item in enumerate(
                        _sequence(
                            planning["held_staging_bindings"],
                            path=f"{path}.planning_services.held_staging_bindings",
                        )
                    )
                )
            )
        if "place_scene_bindings" in planning:
            planning["place_scene_bindings"] = tuple(
                (
                    PlaceSceneBinding(
                        **_mapping(
                            item,
                            path=f"{path}.planning_services.place_scene_bindings[{index}]",
                            required=frozenset({"invocation_id", "floor_entity_id"}),
                            optional=frozenset(
                                {
                                    "support_owner_id",
                                    "inside_subject_id",
                                    "inside_container_id",
                                }
                            ),
                        )
                    )
                    for index, item in enumerate(
                        _sequence(
                            planning["place_scene_bindings"],
                            path=f"{path}.planning_services.place_scene_bindings",
                        )
                    )
                )
            )
        if "observed_curobo" in planning:
            cfg = _mapping(
                planning["observed_curobo"],
                path=f"{path}.planning_services.observed_curobo",
                required=frozenset(),
                optional=frozenset({"cuda_device", "max_voxel_count"}),
            )
            planning["observed_curobo"] = ObservedCuroboServicesCfg(**cfg).snapshot()
        planning = SimulationPlanningServicesCfg(**planning).snapshot()
        if planning.measured_rigid_bindings and (
            not hand_geometries
            or any((factory.sample_count != 65 for _, factory in hand_geometries))
        ):
            raise ValueError(
                "Measured rigid planning requires explicit 65-sample hand geometry declarations."
            )
        if planning.pour_projection_bindings and (not hand_geometries):
            raise ValueError(
                "Configured Pour projection requires explicit hand geometry declarations."
            )
        if planning.pick_closing_bindings and (
            not hand_geometries
            or any((factory.sample_count != 65 for _, factory in hand_geometries))
        ):
            raise ValueError(
                "Configured native Pick requires explicit 65-sample hand geometry declarations."
            )
    if planning is not None and planning.coordinated_grasps:
        declared_generators = dict(generators)
        declared_geometry = dict(hand_geometries)
        for pair in planning.coordinated_grasps:
            for target in (pair.left_grasp_target_id, pair.right_grasp_target_id):
                factory = declared_generators.get(target)
                geometry = declared_geometry.get(target)
                if (
                    type(factory) is not _AntipodalGraspPoseGeneratorFactory
                    or factory.filter_ground_collision is not False
                    or geometry is None
                    or (geometry.hand_part != target)
                ):
                    raise ValueError(
                        "Configured coordinated grasp targets require explicit antipodal generators without ground filtering and matching hand geometry."
                    )
        for pair in planning.coordinated_grasps:
            left, right = (
                declared_generators[name]
                for name in (pair.left_grasp_target_id, pair.right_grasp_target_id)
            )
            if (left.model_id, left.min_opening_width, left.max_opening_width) != (
                right.model_id,
                right.min_opening_width,
                right.max_opening_width,
            ):
                raise ValueError(
                    "Configured coordinated grasp pairs require matching declared gripper models and openings."
                )
    bindings = []
    for index, raw in enumerate(
        _sequence(
            config.get("invocation_policy_bindings", ()),
            path=f"{path}.invocation_policy_bindings",
        )
    ):
        item = _mapping(
            raw,
            path=f"{path}.invocation_policy_bindings[{index}]",
            required=frozenset({"program_fingerprint", "invocation_id", "preset_id"}),
        )
        bindings.append(InvocationPolicyBinding(**item))
    post_bindings = []
    for index, raw in enumerate(
        _sequence(
            config.get("post_policy_bindings", ()), path=f"{path}.post_policy_bindings"
        )
    ):
        item = _mapping(
            raw,
            path=f"{path}.post_policy_bindings[{index}]",
            required=frozenset({"program_fingerprint", "occurrence_id", "preset_id"}),
        )
        post_bindings.append(PostPolicyPresetBinding(**item))
    return _DecodedRuntimeServices(
        recorded_contacts_cfg=(
            None
            if "recorded_contacts" not in config
            else _decode_recorded_contacts(
                config["recorded_contacts"], path=f"{path}.recorded_contacts"
            )
        ),
        cargo_envelope_cfg=cargo_cfg,
        planning_services=planning,
        invocation_policy_bindings=tuple(bindings),
        post_policy_bindings=tuple(post_bindings),
        grasp_pose_generators=generators,
        hand_geometries=hand_geometries,
        handover_pose_providers=handover,
        registered_semantic_lowerers=lowerers,
        control_operations=operations,
        control_part_evidence=evidence,
        nominal_motion_sources=tuple(nominal),
        settle_presets=(
            None
            if "settle_presets" not in config
            else _decode_settle_presets(
                config["settle_presets"], path=f"{path}.settle_presets"
            )
        ),
    )


@dataclass(frozen=True, slots=True)
class _ConfiguredTaskProgramAdapterFactory:
    """Expose one config identity while delegating live adapter construction."""

    delegate: SimulationTaskProgramAdapterFactory
    integration_fingerprint: str
    nominal_motion_sources: tuple[_ConfiguredPrototypeSource, ...] = ()

    @property
    def registration(self) -> SimulationTaskProgramRegistration:
        """Return the immutable registration owned by the delegate."""
        return self.delegate.registration

    def declared_asset_files(self) -> tuple[Path, ...]:
        """Return effective service asset files through the public delegate API."""
        return self.delegate.declared_asset_files()

    def create_adapter(
        self,
        environment: object,
    ) -> TaskProgramEnvironmentAdapter:
        """Create a live adapter for an initialized environment."""
        adapter = self.delegate.create_adapter(environment)
        adapter._set_configured_nominal_sources(self.nominal_motion_sources)
        return adapter


def _decode_configured_task_program_integration(
    value: object, *, planner_config: Mapping[str, object] | None = None
) -> _ConfiguredTaskProgramIntegration:
    """Decode one composable, callable-free Task Program integration.

    Args:
        value: Provider-free scene, robot-profile, and runtime-service payload.
        planner_config: Optional execution-policy planner declaration retained
            for the live simulation adapter. It is deliberately kept outside
            the semantic integration payload.
    """
    path = "integration"
    config = _mapping(
        value,
        path=path,
        required=frozenset({"scene", "robot_profile"}),
        optional=frozenset({"runtime_services"}),
    )
    scene_binding = _decode_scene(config["scene"])
    robot_profile_binding = _decode_robot_profile(config["robot_profile"])
    services = _decode_runtime_services(
        config.get("runtime_services", {}), robot_profile_binding=robot_profile_binding
    )
    call_catalog = builtin_semantic_call_catalog()
    for lowerer_factory in services.registered_semantic_lowerers:
        call_catalog = call_catalog.with_descriptor(
            SemanticCallDescriptor(
                call_id=lowerer_factory.call_id,
                spec_type=RegisteredSemanticCall,
                target_descriptor=lowerer_factory.target_descriptor,
            )
        )
    if type(services.control_part_evidence) is NativeGripEvidenceFactory:
        geometry_by_target = dict(services.hand_geometries)
        for part, pads in services.control_part_evidence.pad_bindings:
            geometry = geometry_by_target.get(part)
            if geometry is not None and (
                geometry.hand_part != part or tuple(geometry.pad_link_names) != pads
            ):
                raise ValueError(
                    "Native grip pads differ from the declared hand geometry roles."
                )
    if any(
        (
            type(factory) is JawPourFactory
            for factory in services.registered_semantic_lowerers
        )
    ):
        if (
            not services.hand_geometries
            or services.planning_services is None
            or (not services.planning_services.capture_root_pose)
            or (not services.planning_services.capture_native_revision)
        ):
            raise ValueError(
                "Source jaw Pour requires explicit hand geometry and root/native capture services."
            )
    registration = SimulationTaskProgramRegistration(
        planning_services=services.planning_services,
        invocation_policy_bindings=services.invocation_policy_bindings,
        post_policy_bindings=services.post_policy_bindings,
        scene_binding=scene_binding,
        robot_profile_binding=robot_profile_binding,
        call_catalog=call_catalog,
        handover_pose_providers=services.handover_pose_providers,
        control_part_evidence_factory=services.control_part_evidence,
        registered_semantic_lowerer_factories=services.registered_semantic_lowerers,
        control_operations=services.control_operations,
        cargo_envelope_cfg=services.cargo_envelope_cfg,
        recorded_contacts_cfg=services.recorded_contacts_cfg,
        **(
            {}
            if services.settle_presets is None
            else {"settle_presets": dict(services.settle_presets)}
        ),
    )
    grasp_fingerprint = tuple(
        (
            {
                "target_id": target_id,
                "factory": {
                    key: value
                    for key, value in asdict(factory).items()
                    if not (
                        key
                        in {"approach_deviation_angle", "max_candidates", "center_mode"}
                        and value is None
                    )
                },
            }
            for target_id, factory in services.grasp_pose_generators
        )
    )
    fingerprint_payload = {
        "registration": registration.fingerprint,
        "grasp_pose_generators": grasp_fingerprint,
        "planner_config": deepcopy(planner_config),
    }
    if services.hand_geometries:
        fingerprint_payload["hand_geometries"] = tuple(
            (
                {"target_id": target_id, "factory": asdict(factory)}
                for target_id, factory in services.hand_geometries
            )
        )
    if services.nominal_motion_sources:
        fingerprint_payload["nominal_motion_sources"] = [
            {
                key: value
                for key, value in asdict(item).items()
                if key != "prototype_manifest"
            }
            for item in services.nominal_motion_sources
        ]
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
        hand_geometry_cfg_factories=dict(services.hand_geometries),
        planner_config=planner_config,
    )
    adapter_factory = _ConfiguredTaskProgramAdapterFactory(
        delegate=delegate,
        integration_fingerprint=integration_fingerprint,
        nominal_motion_sources=services.nominal_motion_sources,
    )
    return _ConfiguredTaskProgramIntegration(
        registration=registration,
        adapter_factory=adapter_factory,
        integration_fingerprint=integration_fingerprint,
    )


def _decode_native_grip_evidence(
    value: object, *, path: str
) -> NativeGripEvidenceFactory:
    config = _mapping(
        value,
        path=path,
        required=frozenset({"kind", "sensor_uid", "object_bindings", "pad_bindings"}),
        optional=frozenset(
            {
                "algorithm_revision",
                "opening_bindings",
                "max_contacts_per_env",
                "trace_capacity",
            }
        ),
    )
    objects = []
    for index, raw in enumerate(
        _sequence(config["object_bindings"], path=f"{path}.object_bindings")
    ):
        pair = _sequence(raw, path=f"{path}.object_bindings[{index}]")
        if len(pair) != 2:
            raise ValueError("Native grip object bindings require identity/UID pairs.")
        objects.append(
            tuple(
                _identifier(item, path=f"{path}.object_bindings[{index}]")
                for item in pair
            )
        )
    pads = []
    for index, raw in enumerate(
        _sequence(config["pad_bindings"], path=f"{path}.pad_bindings")
    ):
        pair = _sequence(raw, path=f"{path}.pad_bindings[{index}]")
        if len(pair) != 2:
            raise ValueError(
                "Native grip pad bindings require control-part/link-list pairs."
            )
        pads.append(
            (
                _identifier(pair[0], path=f"{path}.pad_bindings[{index}]"),
                _identifier_tuple(pair[1], path=f"{path}.pad_bindings[{index}].links"),
            )
        )
    openings = _decode_driver_openings(
        config.get("opening_bindings", ()), path=f"{path}.opening_bindings"
    )
    return NativeGripEvidenceFactory(
        NativeGripEvidenceCfg(
            sensor_uid=_identifier(config["sensor_uid"], path=f"{path}.sensor_uid"),
            object_bindings=tuple(objects),
            pad_bindings=tuple(pads),
            opening_bindings=tuple(openings),
            **{
                name: config[name]
                for name in (
                    "algorithm_revision",
                    "max_contacts_per_env",
                    "trace_capacity",
                )
                if name in config
            },
        )
    )
