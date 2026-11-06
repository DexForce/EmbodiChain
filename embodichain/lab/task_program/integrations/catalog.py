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

"""Immutable task-registration catalog for declarative Task Programs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
import hashlib
import json
import math
from _thread import LockType
from threading import Lock
from types import MappingProxyType
from typing import TYPE_CHECKING, cast
import torch

from embodichain.lab.gym.envs.settling import DynamicSettleMonitorCfg
from embodichain.lab.sim.atomic_actions import (
    Affordance,
    AtomicActionEngine,
    EndpointTrackingFeedbackAddress,
    GRASP_CAPABILITY,
    JOINT_POSITION_CHANNEL,
    SceneProvider,
    SkillDescriptor,
)
from embodichain.lab.sim.atomic_actions.primitives import BUILTIN_ACTION_TYPES
from embodichain.lab.sim.atomic_actions.tracking import (
    FeedbackTerminalAcceptance,
    TrackingRuntime,
)
from embodichain.lab.task_program.semantics import (
    CONTACT_EFFECT_CHANNEL,
    CONSTRAINT_EFFECT_CHANNEL,
    ControlPartEndpoint,
    ControlPartEvidenceAddress,
    FORCE_EFFECT_CHANNEL,
    JOINT_STATE_EFFECT_CHANNEL,
    PLACE_IN_AFFORDANCE_CAPABILITY,
    PLACE_ON_AFFORDANCE_CAPABILITY,
    POSE_RELATION_EFFECT_CHANNEL,
    BoundRobotSkillProfile,
    EffectEvidenceProvider,
    Place,
    RobotSkillProfile,
    RegisteredSemanticCall,
    ResourceEndpoint,
    ResourceEndpointAdapter,
    SceneAffordanceRef,
    SceneArticulationRef,
    SceneEntityRef,
    SceneManifest,
    SceneObjectRef,
    SceneRegistry,
    SemanticCallCatalog,
    SemanticIntegrationManifest,
    builtin_semantic_call_catalog,
)
from embodichain.lab.task_program.compiler.lowering import (
    ContainerRelationTargetGrounder,
    HandOverPoseProvider,
    RegisteredSemanticLowerer,
    RelationTargetGrounder,
    SupportSurfaceRelationTargetGrounder,
)
from embodichain.lab.task_program.semantics.effects import (
    COMPOSITE_EFFECT_MONITOR_ID,
    COMPOSITE_EFFECT_MONITOR_REVISION,
    CompositeEffectMonitorFactory,
    EffectMonitorRegistry,
)
from embodichain.lab.task_program.runtime.parallel_executor import (
    ParallelCommandSafetyValidator,
)

from embodichain.lab.task_program.language.schema import (
    TaskProgramCfg,
    TaskProgramIntegrationCfg,
    PostPolicyCfg,
    RegisteredSemanticCallCfg,
    SemanticCallCfg,
    ValidatorCfg,
)
from embodichain.lab.task_program.compiler import (
    CompiledTaskProgram,
    TaskProgramCompiler,
)
from embodichain.lab.task_program.language.decoder import (
    ConfigPath,
    TaskProgramValidationError,
    SceneReferenceRole,
)
from embodichain.lab.gym.envs.task_program.bridge import RuntimeTransportActionEncoder
from .extensions import (
    ControlPartEvidenceProviderDeclaration,
    ControlPartEvidenceProviderFactory,
    EndpointAdapterDeclaration,
    ParallelCommandSafetyValidatorFactory,
    ParallelSafetyDeclaration,
    RegisteredSemanticLowererDeclaration,
    RegisteredSemanticLowererFactory,
    RegisteredSemanticLowererSceneValidator,
    RuntimeTransportDeclaration,
    RuntimeBoundSemanticLowererFactory,
    StandardExtensionDeclarations,
    build_standard_extension_declarations,
    validate_immutable_extension_declaration,
)
from .simulation import SimulationRobotSkillProfileBinding, SimulationSceneBinding
from .simulation.policies import (
    SimulationSettleMonitorCfg,
    default_simulation_settle_presets,
)
from .simulation.planning_services import SimulationPlanningServicesCfg
from .simulation.idle_clearance import IdleClearanceFactory
from .simulation.planning_services import _effective_planning_service_fields
from .simulation.held_planning import (
    HeldStagingInvocationBinding,
    _validate_program_bindings,
)
from .simulation.pick_closing import PickClosingBinding, _validate_pick_program
from .simulation.place_continuation import (
    PlaceContinuationBinding,
    HeldAlignmentContinuationBinding,
    _bind_place_continuation_declarations,
)
from .simulation.place_scene import PlaceSceneBinding
from .simulation.pour_binding import (
    PourProjectionBinding,
    resolve_pour_projection_routes,
)
from .simulation.pour_projection import ResolvedPourProjectionRoute
from .simulation.released_continuation import (
    ReleasedHandContinuationBinding,
    _bind_released_continuations,
)
from .simulation.slide_preshape import SlidePreshapeBinding
from .simulation.drawer_planning import DrawerInvocationBinding
from .simulation.measured_rigid import (
    MeasuredRigidInvocationBinding,
    _validate_measured_program,
)
from .simulation.transport import DrawerTransportLowererFactory


from .simulation.recorded_contacts import SimulationRecordedContactCfg
from .simulation.cargo_observation import SimulationCargoEnvelopeCfg
from .runtime_services import TaskProgramRuntimeServiceContext
from .simulation.control_policy import SimulationControlConditionCfg
from .simulation.control_lowering import TwistCallDeclaration
from .simulation.control_assembly import (
    ControlOperationDeclaration,
    _validate_control_registration,
)

if TYPE_CHECKING:
    from .invocation_policy import InvocationPolicyBinding, PostPolicyPresetBinding

from .simulation.stability import (
    SimulationStabilityConditionCfg,
    SimulationJointStabilityConditionCfg,
)

_CATALOG_FINGERPRINT_SCHEMA_VERSION = 4
_POST_POLICY_KINDS = frozenset({"wait_stable"})
_VALIDATOR_KINDS = frozenset(
    {
        "articulation_joint_position",
        "object_near_relative_target",
        "object_near_target",
        "object_lifted",
        "object_relative_position",
        "object_upright",
    }
)


class IntegrationFingerprintMismatch(RuntimeError):
    """Raised when a live integration no longer matches its registration."""


def _qualified_name(value: type[object] | object) -> str:
    """Return a stable fully-qualified type name."""
    value_type = value if isinstance(value, type) else type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _canonical_value(value: object) -> object:
    """Convert provider-free declarations to deterministic JSON values."""
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("Fingerprint metadata cannot contain non-finite floats.")
        return value
    if isinstance(value, Enum):
        return {
            "type": _qualified_name(value),
            "value": _canonical_value(value.value),
        }
    if isinstance(value, type):
        return {"type": _qualified_name(value)}
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        return {
            "tensor_dtype": str(tensor.dtype),
            "tensor_shape": list(tensor.shape),
            "tensor_value": tensor.tolist(),
        }
    if isinstance(value, Mapping):
        normalized: dict[str, object] = {}
        for key, nested in value.items():
            if type(key) is not str:
                raise TypeError("Fingerprint mapping keys must be exact strings.")
            normalized[key] = _canonical_value(nested)
        return {key: normalized[key] for key in sorted(normalized)}
    if isinstance(value, (tuple, list)):
        return [_canonical_value(nested) for nested in value]
    if isinstance(value, (set, frozenset)):
        normalized = [_canonical_value(nested) for nested in value]
        return sorted(
            normalized,
            key=lambda item: json.dumps(
                item,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ),
        )
    if is_dataclass(value):
        from embodichain.lab.sim.atomic_actions.policies import RecoveryPolicy

        metadata = {
            data_field.name: _canonical_value(getattr(value, data_field.name))
            for data_field in fields(value)
            if not (
                type(value) is RecoveryPolicy
                and data_field.name == "planning_budget"
                and getattr(value, data_field.name) is None
            )
        }
        from embodichain.lab.sim.atomic_actions.policies import MotionPolicy

        from embodichain.lab.sim.atomic_actions.core import SkillDescriptor

        if (
            type(value) is SkillDescriptor
            and not value.supports_segment_motion_recovery
        ):
            metadata.pop("supports_segment_motion_recovery")
        if type(value) is MotionPolicy and value.final_velocity_rejection == "hold":
            metadata.pop("final_velocity_rejection")
        if type(value) is RecoveryPolicy and value.retry_all_failed_plans is True:
            metadata.pop("retry_all_failed_plans")
        if type(value) is SimulationPlanningServicesCfg:
            effective = _effective_planning_service_fields(value, validate=False)
            metadata = {
                name: item for name, item in metadata.items() if name in effective
            }
        if type(value) is PlaceSceneBinding:
            for name in ("inside_subject_id", "inside_container_id"):
                if getattr(value, name) is None:
                    metadata.pop(name, None)
        if (
            type(value) is PickClosingBinding
            and value.opening_place_invocation_id is None
        ):
            metadata.pop("opening_place_invocation_id", None)
        from embodichain.lab.sim.atomic_actions.primitives.pick_up import (
            PickUpOptions,
            PickPlanningRecovery,
        )
        from embodichain.lab.sim.atomic_actions.primitives.place import (
            PlaceOptions,
            PlacePlanningRecovery,
        )
        from embodichain.lab.sim.atomic_actions.primitives.hand_over import (
            HandOverOptions,
        )

        # Preserve pre-extension defaults for these opt-ins only. Nondefault
        # values and fields from the existing action schemas remain complete.
        if type(value) is PickUpOptions:
            for name in (
                "constrained_grasp",
                "linear_segments",
                "measured_preshape",
                "native_scene_trial",
                "qualify_fixed_grasp",
            ):
                if getattr(value, name) is False:
                    metadata.pop(name)
            if value.qualify_downstream_targets is True:
                metadata.pop("qualify_downstream_targets")
        if type(value) is PlaceOptions:
            for name in ("drawer_placement", "measured_inside", "preserve_vertical"):
                if getattr(value, name) is False:
                    metadata.pop(name)
        if type(value) is HandOverOptions:
            for name in (
                "adaptive_receive",
                "linear_segments",
                "require_receiver_scene",
            ):
                if getattr(value, name) is False:
                    metadata.pop(name)
            for name, default in {
                "existing_hold_hand_source": "observed",
                "receive_region": "mesh_extent",
                "source_retreat_mode": "approach",
            }.items():
                if (
                    type(getattr(value, name)) is str
                    and getattr(value, name) == default
                ):
                    metadata.pop(name)
        if type(value) is PickPlanningRecovery:
            for name in ("constrained_roll", "place_continuation"):
                if getattr(value, name) is False:
                    metadata.pop(name)
        if type(value) is PlacePlanningRecovery and value.inside_world_yaw is False:
            metadata.pop("inside_world_yaw")
        return {"type": _qualified_name(value), "fields": metadata}
    raise TypeError(
        "Registration fingerprint metadata contains unsupported value type "
        f"{_qualified_name(value)!r}. Values must be complete declarative data; "
        "live or opaque objects cannot be fingerprinted by type alone."
    )


def _provider_fingerprint_declaration(provider: object) -> object:
    """Return the complete canonical declaration for one validated provider."""
    if is_dataclass(provider):
        return provider
    return {"provider_type": _qualified_name(provider)}


def _canonical_json(value: object) -> str:
    """Encode one declaration using the versioned canonical JSON form."""
    return json.dumps(
        _canonical_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _digest(payload: object) -> str:
    """Return the SHA-256 digest for one canonical declaration payload."""
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _snapshot_settle_presets(
    values: Mapping[
        str,
        DynamicSettleMonitorCfg
        | SimulationStabilityConditionCfg
        | SimulationJointStabilityConditionCfg
        | SimulationControlConditionCfg,
    ],
) -> Mapping[
    str,
    DynamicSettleMonitorCfg
    | SimulationStabilityConditionCfg
    | SimulationJointStabilityConditionCfg
    | SimulationControlConditionCfg,
]:
    """Own one strict named settle-preset table."""
    if not isinstance(values, Mapping) or not values:
        raise ValueError("settle_presets must be a non-empty mapping.")
    normalized: dict[
        str,
        DynamicSettleMonitorCfg
        | SimulationStabilityConditionCfg
        | SimulationJointStabilityConditionCfg
        | SimulationControlConditionCfg,
    ] = {}
    for preset_id, preset in values.items():
        if (
            type(preset_id) is not str
            or not preset_id
            or preset_id != preset_id.strip()
        ):
            raise ValueError(
                "Settle preset IDs must be non-empty strings without outer "
                "whitespace."
            )
        if not isinstance(preset, DynamicSettleMonitorCfg) and type(preset) not in (
            SimulationStabilityConditionCfg,
            SimulationJointStabilityConditionCfg,
            SimulationControlConditionCfg,
        ):
            raise TypeError(
                "settle_presets require dynamic monitors or typed stability conditions."
            )
        normalized[preset_id] = preset.snapshot()
    return MappingProxyType(normalized)


def _exact_identifier(value: object, *, field_name: str) -> str:
    """Validate one exact catalog identifier."""
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(
            f"{field_name} must be a non-empty string without outer whitespace."
        )
    return value


def _relation_grounder_key(
    grounder: RelationTargetGrounder,
) -> tuple[str, type[Affordance], str]:
    """Return the compiler-compatible exact key for one relation grounder."""
    grounder_type = type(grounder)
    capability = _exact_identifier(
        getattr(grounder_type, "capability", None),
        field_name="RelationTargetGrounder.capability",
    )
    affordance_type = getattr(grounder_type, "affordance_type", None)
    if not isinstance(affordance_type, type) or not issubclass(
        affordance_type,
        Affordance,
    ):
        raise TypeError(
            "RelationTargetGrounder.affordance_type must be an Affordance subclass."
        )
    revision = _exact_identifier(
        getattr(grounder_type, "affordance_revision", None),
        field_name="RelationTargetGrounder.affordance_revision",
    )
    return capability, affordance_type, revision


def _relation_grounder_order_key(
    grounder: RelationTargetGrounder,
) -> tuple[str, str, str]:
    """Return one totally ordered rendering of a relation-grounder key."""
    capability, affordance_type, revision = _relation_grounder_key(grounder)
    return capability, _qualified_name(affordance_type), revision


def _snapshot_relation_grounders(
    values: tuple[RelationTargetGrounder, ...],
) -> tuple[RelationTargetGrounder, ...]:
    """Validate and own one immutable relation-grounder tuple."""
    if type(values) is not tuple:
        raise TypeError("relation_grounders must be an exact tuple.")
    seen: set[tuple[str, type[Affordance], str]] = set()
    for grounder in values:
        if not isinstance(grounder, RelationTargetGrounder):
            raise TypeError(
                "relation_grounders must contain RelationTargetGrounder instances."
            )
        validate_immutable_extension_declaration(
            grounder,
            field_name="relation_grounders",
        )
        key = _relation_grounder_key(grounder)
        if key in seen:
            raise ValueError(f"Duplicate relation grounder key {key!r}.")
        seen.add(key)
    return tuple(values)


def _builtin_relation_grounders(
    scene_binding: SimulationSceneBinding,
) -> tuple[RelationTargetGrounder, ...]:
    """Install standard grounders for declared production relation bindings."""
    values: list[RelationTargetGrounder] = []
    if scene_binding.support_surfaces:
        values.append(SupportSurfaceRelationTargetGrounder())
    if scene_binding.containers:
        values.append(ContainerRelationTargetGrounder())
    return tuple(values)


def _configured_relation_grounders(
    scene_binding: SimulationSceneBinding,
    values: tuple[RelationTargetGrounder, ...],
) -> tuple[RelationTargetGrounder, ...]:
    """Remove standard grounders already materialized on a registration.

    ``SimulationTaskProgramRegistration`` exposes the complete grounder tuple,
    including the scene-derived standard entries.  Immutable registration
    transforms commonly use :func:`dataclasses.replace`, which feeds that
    exposed tuple back through the constructor.  Treat one exact instance of a
    standard grounder as already installed so those transforms remain
    idempotent while duplicate custom declarations still fail below.
    """
    builtins = _builtin_relation_grounders(scene_binding)
    builtin_by_key = {
        _relation_grounder_key(grounder): grounder for grounder in builtins
    }
    consumed: set[tuple[str, type[Affordance], str]] = set()
    configured: list[RelationTargetGrounder] = []
    for grounder in values:
        key = _relation_grounder_key(grounder)
        builtin = builtin_by_key.get(key)
        if builtin is not None and type(grounder) is type(builtin):
            if key not in consumed:
                consumed.add(key)
                continue
        configured.append(grounder)
    return tuple(configured)


def _snapshot_relation_grounder_keys(
    values: frozenset[tuple[str, type[Affordance], str]],
) -> frozenset[tuple[str, type[Affordance], str]]:
    """Validate immutable provider-free relation-grounder lookup keys."""
    if type(values) is not frozenset:
        raise TypeError("relation_grounder_keys must be an exact frozenset.")
    normalized: set[tuple[str, type[Affordance], str]] = set()
    for key in values:
        if type(key) is not tuple or len(key) != 3:
            raise TypeError("relation_grounder_keys must contain exact 3-tuple values.")
        capability, affordance_type, revision = key
        _exact_identifier(capability, field_name="relation grounder capability")
        if not isinstance(affordance_type, type) or not issubclass(
            affordance_type,
            Affordance,
        ):
            raise TypeError(
                "relation grounder affordance types must be Affordance subclasses."
            )
        _exact_identifier(revision, field_name="relation grounder revision")
        normalized.add((capability, affordance_type, revision))
    return frozenset(normalized)


def _handover_pose_provider_id(provider: HandOverPoseProvider) -> str:
    """Return the compiler-compatible class ID for one hand-over provider."""
    return _exact_identifier(
        getattr(type(provider), "provider_id", None),
        field_name="HandOverPoseProvider.provider_id",
    )


def _snapshot_handover_pose_providers(
    values: tuple[HandOverPoseProvider, ...],
) -> tuple[HandOverPoseProvider, ...]:
    """Validate and own one immutable hand-over-provider tuple."""
    if type(values) is not tuple:
        raise TypeError("handover_pose_providers must be an exact tuple.")
    seen: set[str] = set()
    for provider in values:
        if not isinstance(provider, HandOverPoseProvider):
            raise TypeError(
                "handover_pose_providers must contain HandOverPoseProvider instances."
            )
        validate_immutable_extension_declaration(
            provider,
            field_name="handover_pose_providers",
        )
        provider_id = _handover_pose_provider_id(provider)
        if provider_id in seen:
            raise ValueError(f"Duplicate handover pose provider {provider_id!r}.")
        seen.add(provider_id)
    return tuple(values)


def _validate_standard_call_catalog(
    call_catalog: SemanticCallCatalog,
    lowerer_declarations: Mapping[str, RegisteredSemanticLowererDeclaration],
) -> None:
    """Require exact built-ins and factory coverage for every registered call."""
    builtins = builtin_semantic_call_catalog().descriptors
    registered_call_ids: set[str] = set()
    for descriptor in call_catalog.descriptors.values():
        if descriptor.spec_type is RegisteredSemanticCall:
            registered_call_ids.add(descriptor.call_id)
            continue
        expected = builtins.get(descriptor.call_id)
        if expected != descriptor:
            raise ValueError(
                f"Semantic call {descriptor.call_id!r} does not match its exact "
                "curated descriptor."
            )
    declared_call_ids = set(lowerer_declarations)
    if registered_call_ids != declared_call_ids:
        raise ValueError(
            "Registered semantic call descriptors and lowerer factories must "
            "cover exactly the same call IDs; "
            f"missing={sorted(registered_call_ids - declared_call_ids)}, "
            f"unused={sorted(declared_call_ids - registered_call_ids)}."
        )


def _validate_standard_effect_monitors(profile: RobotSkillProfile) -> None:
    """Require every preset to use the exact built-in effect-monitor factory."""
    registry = EffectMonitorRegistry((CompositeEffectMonitorFactory(),))
    builtin_key = (
        COMPOSITE_EFFECT_MONITOR_ID,
        COMPOSITE_EFFECT_MONITOR_REVISION,
    )
    for preset_id, preset in profile.presets.items():
        for semantic_id, monitor_ref in preset.effect_monitors.items():
            key = monitor_ref.monitor_id, monitor_ref.revision
            if key != builtin_key:
                raise ValueError(
                    f"Preset {preset_id!r} semantic call {semantic_id!r} selects "
                    f"non-built-in effect monitor {key!r}; the standard "
                    "simulation registration supports only {builtin_key!r}."
                )
            try:
                registry.validate_ref(monitor_ref)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Preset {preset_id!r} semantic call {semantic_id!r} has an "
                    "invalid built-in effect-monitor declaration."
                ) from exc


def _validate_standard_tracking_metrics(profile: RobotSkillProfile) -> None:
    """Resolve every reachable metric through the exact built-in evaluator table."""
    evaluators = TrackingRuntime.with_builtins().evaluators
    for preset_id, preset in profile.presets.items():
        policy = preset.tracking_policy
        metric_groups = []
        if policy.in_flight is not None:
            metric_groups.append(("in_flight", policy.in_flight.metrics))
        if isinstance(policy.terminal, FeedbackTerminalAcceptance):
            metric_groups.append(("terminal", policy.terminal.metrics))
        for phase, metrics in metric_groups:
            for metric in metrics:
                try:
                    evaluators.resolve(metric)
                except (KeyError, TypeError, ValueError) as exc:
                    key = metric.metric_id, metric.revision, _qualified_name(metric)
                    raise ValueError(
                        f"Preset {preset_id!r} {phase} tracking metric {key!r} "
                        "has no exact built-in evaluator in the standard "
                        "simulation registration."
                    ) from exc


@dataclass(frozen=True, slots=True)
class TaskProgramIntegrationCatalog:
    """Provider-free integration directory owned by one task registration."""

    scene_registry_id: str
    robot_profile_id: str
    scene: SceneManifest
    robot_profile: RobotSkillProfile
    call_catalog: SemanticCallCatalog
    relation_grounder_keys: frozenset[tuple[str, type[Affordance], str]]
    settle_preset_ids: frozenset[str]
    endpoint_adapter_declarations: Mapping[
        type[ResourceEndpoint], EndpointAdapterDeclaration
    ]
    runtime_transport_declarations: tuple[RuntimeTransportDeclaration, ...]
    parallel_safety_declaration: ParallelSafetyDeclaration | None
    control_part_evidence_declaration: ControlPartEvidenceProviderDeclaration | None
    registered_semantic_lowerer_declarations: Mapping[
        str, RegisteredSemanticLowererDeclaration
    ]
    fingerprint: str
    _required_skills: Mapping[str, SkillDescriptor] = field(
        repr=False,
        compare=False,
    )
    pose_delta_settle_preset_ids: frozenset[str] = field(
        default_factory=frozenset, kw_only=True
    )
    """Presets explicitly selecting consecutive rigid-pose measurements."""
    rigid_settle_entity_ids: frozenset[str] = field(
        default_factory=frozenset, kw_only=True
    )
    """Native rigid-object identities and affordances resolving to those objects."""
    condition_settle_entities: Mapping[str, str] = field(
        default_factory=dict, kw_only=True
    )
    """Immutable preset-to-entity binding for declared geometric conditions."""

    invocation_policy_bindings: tuple[InvocationPolicyBinding, ...] = field(
        default=(), kw_only=True
    )
    post_policy_bindings: tuple[PostPolicyPresetBinding, ...] = field(
        default=(), kw_only=True
    )
    """Exact program-bound named presets for sequential execution occurrences."""

    idle_factories: tuple[IdleClearanceFactory, ...] = field(default=(), kw_only=True)

    control_calls: tuple[tuple[str, ControlOperationDeclaration, bool], ...] = field(
        default=(), kw_only=True
    )
    control_presets: Mapping[str, tuple[str, str]] = field(
        default_factory=dict, kw_only=True
    )
    control_conditions: Mapping[str, SimulationControlConditionCfg] = field(
        default_factory=dict, kw_only=True, repr=False
    )
    control_cleanup_call_ids: frozenset[str] = field(
        default_factory=frozenset, kw_only=True
    )

    inside_landing_targets: Mapping[str, tuple[str, str]] = field(
        default_factory=dict, kw_only=True
    )

    measured_rigid_bindings: tuple[MeasuredRigidInvocationBinding, ...] = field(
        default=(), kw_only=True
    )

    drawer_bindings: tuple[DrawerInvocationBinding, ...] = field(
        default=(), kw_only=True
    )

    slide_preshape_bindings: tuple[SlidePreshapeBinding, ...] = field(
        default=(), kw_only=True
    )

    coordinated_grasp_pairs: tuple[tuple[str, str], ...] = field(
        default=(), kw_only=True
    )

    held_staging_bindings: tuple[HeldStagingInvocationBinding, ...] = field(
        default=(), kw_only=True
    )

    place_continuation_bindings: tuple[PlaceContinuationBinding, ...] = field(
        default=(), kw_only=True
    )

    continuation_scene_invocations: frozenset[str] = field(
        default_factory=frozenset, kw_only=True
    )

    native_pick_free_motion: bool = field(default=False, kw_only=True)

    pour_projection_bindings: tuple[PourProjectionBinding, ...] = field(
        default=(), kw_only=True
    )

    pour_projection_factories: tuple[RegisteredSemanticLowererFactory, ...] = field(
        default=(), kw_only=True, repr=False
    )

    released_hand_continuations: tuple[ReleasedHandContinuationBinding, ...] = field(
        default=(), kw_only=True
    )

    pick_closing_bindings: tuple[PickClosingBinding, ...] = field(
        default=(), kw_only=True
    )

    def __post_init__(self) -> None:
        from .invocation_policy import InvocationPolicyBinding, PostPolicyPresetBinding

        if type(self.measured_rigid_bindings) is not tuple or any(
            type(value) is not MeasuredRigidInvocationBinding
            for value in self.measured_rigid_bindings
        ):
            raise TypeError(
                "Measured catalog bindings require immutable typed declarations."
            )
        if type(self.held_staging_bindings) is not tuple or any(
            type(value) is not HeldStagingInvocationBinding
            for value in self.held_staging_bindings
        ):
            raise TypeError(
                "Held staging declarations must be an immutable typed tuple."
            )
        if (
            type(self.native_pick_free_motion) is not bool
            or type(self.pick_closing_bindings) is not tuple
            or any(
                type(value) is not PickClosingBinding
                for value in self.pick_closing_bindings
            )
        ):
            raise TypeError(
                "Native Pick catalog services require immutable typed declarations."
            )
        if type(self.drawer_bindings) is not tuple or any(
            type(value) is not DrawerInvocationBinding for value in self.drawer_bindings
        ):
            raise TypeError("Drawer bindings must be an immutable typed tuple.")
        if type(self.slide_preshape_bindings) is not tuple or any(
            type(value) is not SlidePreshapeBinding
            for value in self.slide_preshape_bindings
        ):
            raise TypeError("Slide preshape bindings must be an immutable typed tuple.")
        if (
            type(self.pour_projection_bindings) is not tuple
            or any(
                type(value) is not PourProjectionBinding
                for value in self.pour_projection_bindings
            )
            or type(self.pour_projection_factories) is not tuple
        ):
            raise TypeError(
                "Pour projection catalog data requires immutable typed declarations."
            )
        if type(self.coordinated_grasp_pairs) is not tuple or any(
            type(pair) is not tuple
            or len(pair) != 2
            or any(
                type(name) is not str or not name or name != name.strip()
                for name in pair
            )
            for pair in self.coordinated_grasp_pairs
        ):
            raise TypeError(
                "Coordinated grasp pairs require immutable exact target pairs."
            )
        targets = dict(self.inside_landing_targets)
        if any(
            type(key) is not str
            or not key
            or key != key.strip()
            or type(value) is not tuple
            or len(value) != 2
            or any(
                type(item) is not str or not item or item != item.strip()
                for item in value
            )
            for key, value in targets.items()
        ):
            raise TypeError(
                "Inside landing targets require exact immutable subject/container identities."
            )
        object.__setattr__(self, "inside_landing_targets", MappingProxyType(targets))
        if type(self.post_policy_bindings) is not tuple or any(
            type(value) is not PostPolicyPresetBinding
            for value in self.post_policy_bindings
        ):
            raise TypeError(
                "post_policy_bindings must be a tuple of PostPolicyPresetBinding."
            )
        if type(self.invocation_policy_bindings) is not tuple or not all(
            type(item) is InvocationPolicyBinding
            for item in self.invocation_policy_bindings
        ):
            raise TypeError(
                "invocation_policy_bindings must be an immutable tuple of InvocationPolicyBinding."
            )
        if (
            len({item.program_fingerprint for item in self.invocation_policy_bindings})
            > 1
        ):
            raise ValueError(
                "Configured invocation bindings must select one program fingerprint."
            )
        if len({item.invocation_id for item in self.invocation_policy_bindings}) != len(
            self.invocation_policy_bindings
        ):
            raise ValueError("Duplicate invocation policy occurrence.")
        object.__setattr__(
            self,
            "control_conditions",
            MappingProxyType(
                {
                    key: value.snapshot()
                    for key, value in self.control_conditions.items()
                }
            ),
        )
        for field_name in ("scene_registry_id", "robot_profile_id"):
            value = getattr(self, field_name)
            if type(value) is not str or not value or value != value.strip():
                raise ValueError(f"{field_name} must be an exact identifier.")
        if type(self.scene) is not SceneManifest:
            raise TypeError("scene must be exactly SceneManifest.")
        if type(self.robot_profile) is not RobotSkillProfile:
            raise TypeError("robot_profile must be exactly RobotSkillProfile.")
        if type(self.call_catalog) is not SemanticCallCatalog:
            raise TypeError("call_catalog must be exactly SemanticCallCatalog.")
        object.__setattr__(
            self,
            "relation_grounder_keys",
            _snapshot_relation_grounder_keys(self.relation_grounder_keys),
        )
        extensions = StandardExtensionDeclarations(
            endpoint_adapters=self.endpoint_adapter_declarations,
            runtime_transports=self.runtime_transport_declarations,
            parallel_safety=self.parallel_safety_declaration,
            control_part_evidence=self.control_part_evidence_declaration,
            registered_semantic_lowerers=(
                self.registered_semantic_lowerer_declarations
            ),
        )
        profile_endpoint_types = frozenset(
            type(endpoint)
            for resource in self.robot_profile.resources.values()
            for endpoint in resource.endpoints.values()
        )
        if profile_endpoint_types != frozenset(extensions.endpoint_adapters):
            raise ValueError(
                "endpoint_adapter_declarations must cover every exact robot "
                "profile endpoint type and no others."
            )
        object.__setattr__(
            self,
            "endpoint_adapter_declarations",
            extensions.endpoint_adapters,
        )
        object.__setattr__(
            self,
            "runtime_transport_declarations",
            extensions.runtime_transports,
        )
        object.__setattr__(
            self,
            "parallel_safety_declaration",
            extensions.parallel_safety,
        )
        object.__setattr__(
            self,
            "control_part_evidence_declaration",
            extensions.control_part_evidence,
        )
        object.__setattr__(
            self,
            "registered_semantic_lowerer_declarations",
            extensions.registered_semantic_lowerers,
        )
        if self.robot_profile.profile_id != self.robot_profile_id:
            raise ValueError("robot_profile_id must match robot_profile.profile_id.")
        preset_ids = frozenset(self.settle_preset_ids)
        if not preset_ids:
            raise ValueError("settle_preset_ids must not be empty.")
        object.__setattr__(self, "settle_preset_ids", preset_ids)
        for name in ("pose_delta_settle_preset_ids", "rigid_settle_entity_ids"):
            values = frozenset(getattr(self, name))
            for value in values:
                _exact_identifier(value, field_name=name)
            object.__setattr__(self, name, values)
        if not self.pose_delta_settle_preset_ids <= preset_ids:
            raise ValueError(
                "Pose-delta presets must belong to the settle-preset table."
            )
        conditions = dict(self.condition_settle_entities)
        if not set(conditions) <= preset_ids:
            raise ValueError(
                "Condition presets must belong to the settle-preset table."
            )
        for key, value in conditions.items():
            _exact_identifier(key, field_name="condition preset")
            _exact_identifier(value, field_name="condition entity")
        object.__setattr__(
            self, "condition_settle_entities", MappingProxyType(conditions)
        )

        if (
            type(self.fingerprint) is not str
            or len(self.fingerprint) != 64
            or any(
                character not in "0123456789abcdef" for character in self.fingerprint
            )
        ):
            raise ValueError("fingerprint must be a lowercase SHA-256 digest.")
        object.__setattr__(
            self,
            "_required_skills",
            MappingProxyType(dict(self._required_skills)),
        )

    def validate_integration(
        self,
        integration: TaskProgramIntegrationCfg,
        *,
        path: ConfigPath,
    ) -> None:
        """Validate exact scene, profile, and runtime-preset selection."""
        del path
        if integration.scene_registry != self.scene_registry_id:
            raise ValueError(
                f"Expected scene_registry {self.scene_registry_id!r}, got "
                f"{integration.scene_registry!r}."
            )
        if integration.robot_profile != self.robot_profile_id:
            raise ValueError(
                f"Expected robot_profile {self.robot_profile_id!r}, got "
                f"{integration.robot_profile!r}."
            )
        if integration.runtime_preset not in self.robot_profile.presets:
            raise KeyError(
                f"Unknown runtime preset {integration.runtime_preset!r}; available "
                f"presets are {sorted(self.robot_profile.presets)}."
            )

    def validate_semantic_call(
        self,
        call: SemanticCallCfg,
        *,
        path: ConfigPath,
    ) -> None:
        """Validate that the semantic call is present in the integration catalog."""
        call_id = call.call_id if type(call) is RegisteredSemanticCallCfg else call.kind
        self.call_catalog.discover(call_id)

    def _validate_place_relation_grounder(
        self,
        call: Place,
        *,
        affordance: SceneAffordanceRef,
        path: ConfigPath,
    ) -> None:
        """Require the exact linked relation-affordance grounder pre-sim."""
        if call.on is not None:
            capability = PLACE_ON_AFFORDANCE_CAPABILITY
            relation_field = "on"
        elif call.inside is not None:
            capability = PLACE_IN_AFFORDANCE_CAPABILITY
            relation_field = "inside"
        else:
            return
        entry = self.scene.lookup(
            affordance,
            expected_type=SceneAffordanceRef,
            path=(*path, relation_field),
        )
        payload_type = entry.affordance_payload_type
        revision = entry.affordance_revision
        if payload_type is None or revision is None:
            raise TaskProgramValidationError(
                "incomplete_relation_affordance_declaration",
                (*path, relation_field),
                f"Relation affordance {affordance.entity_id!r} must declare an "
                "exact payload type and revision.",
            )
        key = (capability, payload_type, revision)
        if key not in self.relation_grounder_keys:
            rendered_key = (
                capability,
                _qualified_name(payload_type),
                revision,
            )
            raise TaskProgramValidationError(
                "relation_grounder_not_registered",
                (*path, relation_field),
                f"No task-registration relation grounder matches linked "
                f"affordance {affordance.entity_id!r} with key {rendered_key!r}.",
            )

    def validate_scene_reference(
        self,
        reference: str,
        *,
        role: SceneReferenceRole,
        path: ConfigPath,
    ) -> None:
        """Validate one typed scene reference against its declared role."""
        expected: dict[str, tuple[type[SceneEntityRef], ...]] = {
            "entity": (SceneEntityRef,),
            "object": (SceneObjectRef,),
            "articulation": (SceneArticulationRef,),
            "affordance": (SceneAffordanceRef,),
            "object_or_affordance": (SceneObjectRef, SceneAffordanceRef),
        }
        expected_types = expected.get(role)
        if expected_types is None:
            raise ValueError(f"Unsupported scene reference role {role!r}.")
        resolved = self.scene.resolve(reference, path=path)
        if not isinstance(resolved, expected_types):
            raise TypeError(
                f"Scene reference {reference!r} is {type(resolved).__name__}, "
                f"not one of {tuple(value.__name__ for value in expected_types)}."
            )

    def validate_post_policy(
        self,
        policy: PostPolicyCfg,
        *,
        path: ConfigPath,
    ) -> None:
        """Validate one registered post-policy kind and named preset."""
        if policy.kind not in _POST_POLICY_KINDS:
            raise KeyError(
                f"Unknown post-policy kind {policy.kind!r}; available kinds are "
                f"{sorted(_POST_POLICY_KINDS)}."
            )
        if policy.preset not in self.settle_preset_ids:
            raise KeyError(
                f"Unknown settle preset {policy.preset!r}; available presets are "
                f"{sorted(self.settle_preset_ids)}."
            )
        if policy.preset in self.pose_delta_settle_preset_ids:
            entity = self.scene.resolve(policy.entity, path=(*path, "entity"))
            if entity.entity_id not in self.rigid_settle_entity_ids:
                raise TaskProgramValidationError(
                    "unsupported_pose_delta_settle_target",
                    path,
                    "Pose-delta settling requires an explicitly bound native rigid object.",
                )
        if policy.preset in self.condition_settle_entities and not (
            self.post_policy_bindings and policy.preset in self.control_presets
        ):
            entity = self.scene.resolve(policy.entity, path=(*path, "entity"))
            if entity.entity_id != self.condition_settle_entities[policy.preset]:
                raise TaskProgramValidationError(
                    "stability_condition_target_mismatch",
                    path,
                    "The wait-stable entity differs from its declared condition.",
                )

    def validate_validator(
        self,
        validator: ValidatorCfg,
        *,
        path: ConfigPath,
    ) -> None:
        """Validate one registered segment-validator kind."""
        del path
        if validator.kind not in _VALIDATOR_KINDS:
            raise KeyError(
                f"Unknown validator kind {validator.kind!r}; available kinds are "
                f"{sorted(_VALIDATOR_KINDS)}."
            )

    def preflight(self, program: TaskProgramCfg) -> CompiledTaskProgram:
        """Compile and statically link every expanded semantic call."""
        self.validate_integration(program.integration, path=("integration",))
        compiled = TaskProgramCompiler(self.scene).compile(program)
        manifest = SemanticIntegrationManifest(
            scene=self.scene,
            robot_profile=self.robot_profile,
            call_catalog=self.call_catalog,
            runtime_preset=program.integration.runtime_preset,
        )
        from embodichain.lab.sim.atomic_actions.primitives.place import (
            PlacePlanningRecovery,
        )

        inside_presets = {
            name
            for name, preset in self.robot_profile.presets.items()
            if isinstance(
                preset.recovery_policy.planning_recovery, PlacePlanningRecovery
            )
            and preset.recovery_policy.planning_recovery.inside_world_yaw
        }
        selected_presets = {
            binding.invocation_id: binding.preset_id
            for binding in self.invocation_policy_bindings
        }
        for segment in compiled.iter_segments():
            for policy in segment.post_policies:
                if (
                    policy.cfg.preset in self.pose_delta_settle_preset_ids
                    or policy.cfg.preset in self.condition_settle_entities
                ):
                    self.validate_post_policy(policy.cfg, path=policy.source_path)
            if (
                segment.parallel_block is not None
                and self.parallel_safety_declaration is None
            ):
                raise TaskProgramValidationError(
                    "parallel_safety_factory_not_registered",
                    segment.parallel_block.source_path,
                    "Parallel execution requires a task-registration-owned "
                    "physical safety-validator factory.",
                )
            for call in segment.calls:
                occurrence = f"{compiled.program_id}/{segment.segment_id}:{call.segment_call_index}"
                linked = manifest.link_call(
                    call.call,
                    path=call.source_path,
                    preset_id=selected_presets.get(occurrence),
                )
                if linked.descriptor.target_descriptor.skill_id == "hand_over":
                    occurrence = f"{compiled.program_id}/{segment.segment_id}:{call.segment_call_index}"
                    preset = self.robot_profile.presets[
                        selected_presets.get(occurrence, linked.preset_id)
                    ]
                    options = preset.action_option_template(call.call.semantic_id)
                    if options.linear_segments:
                        policy = preset.motion_policy
                        if (
                            policy.strategy != "ik_interp"
                            or policy.interpolation_mode != "fixed_grid"
                            or policy.ik_seed_mode != "sequential"
                            or policy.dynamic_collision_mode != "off"
                            or policy.segment_sample_count_retries != (260, 320)
                            or policy.segment_continuous_ik_recovery
                        ):
                            raise ValueError(
                                "Cartesian HandOver requires its declared fixed-grid sequential IK source policy."
                            )
                if linked.descriptor.skill_id in {
                    "move_held_object",
                    "move_end_effector",
                }:
                    preset = self.robot_profile.presets[linked.preset_id]
                    options = preset.action_option_template(call.call.semantic_id)
                    if options.linear_segments:
                        policy = preset.motion_policy
                        if (
                            policy.strategy != "ik_interp"
                            or policy.interpolation_mode != "fixed_grid"
                            or policy.ik_seed_mode != "sequential"
                            or policy.dynamic_collision_mode != "off"
                            or policy.segment_continuous_ik_recovery
                            or (
                                linked.descriptor.skill_id == "move_end_effector"
                                and policy.segment_sample_count_retries != (260, 320)
                            )
                        ):
                            raise ValueError(
                                "Cartesian motion aliases require their declared fixed-grid sequential IK source policy."
                            )
                declaration = self.registered_semantic_lowerer_declarations.get(
                    call.call.semantic_id
                )
                if declaration is not None:
                    from .simulation.transport import (
                        TerminalAttachmentTransportLowererFactory,
                    )

                    if (
                        declaration.factory_type
                        is TerminalAttachmentTransportLowererFactory
                    ):
                        preset = self.robot_profile.presets[linked.preset_id]
                        TerminalAttachmentTransportLowererFactory.validate_preset(
                            preset
                        )
                if (
                    self.coordinated_grasp_pairs
                    and linked.descriptor.target_descriptor.skill_id
                    == "coordinated_pickment"
                ):
                    defaults = self.robot_profile.defaults.get("coordinated_pickment")
                    selections = {} if defaults is None else dict(defaults.resources)
                    selections.update(call.call.resources)
                    targets = []
                    for slot in ("left", "right"):
                        resource = self.robot_profile.resources.get(
                            selections.get(slot)
                        )
                        endpoint = (
                            None
                            if resource is None
                            else resource.endpoints.get("grasp")
                        )
                        targets.append(getattr(endpoint, "control_part", None))
                    if tuple(targets) not in self.coordinated_grasp_pairs:
                        raise ValueError(
                            "Coordinated call grasp slots differ from the declared geometry pair."
                        )
                if inside_presets:
                    occurrence = f"{compiled.program_id}/{segment.segment_id}:{call.segment_call_index}"
                    if (
                        selected_presets.get(occurrence, linked.preset_id)
                        in inside_presets
                        and occurrence not in self.inside_landing_targets
                    ):
                        raise ValueError(
                            "Inside yaw recovery requires an exact occurrence-bound landing declaration."
                        )
                    if (
                        selected_presets.get(occurrence, linked.preset_id)
                        in inside_presets
                    ):
                        destination = linked.affordances.get("destination")
                        if (
                            type(linked.call) is not Place
                            or linked.call.inside is None
                            or destination is None
                            or self.inside_landing_targets[occurrence]
                            != (linked.call.object.entity_id, destination.entity_id)
                        ):
                            raise ValueError(
                                "Inside yaw binding must match the selected inside Place subject and container."
                            )
                if type(linked.call) is Place and linked.call.at is None:
                    destination = linked.affordances.get("destination")
                    if destination is None:
                        raise AssertionError(
                            "Linked relation Place call lacks a destination "
                            "affordance."
                        )
                    self._validate_place_relation_grounder(
                        linked.call,
                        affordance=destination,
                        path=call.source_path,
                    )
        if self.invocation_policy_bindings:
            from .invocation_policy import BoundInvocationPolicies

            BoundInvocationPolicies(
                compiled, self.invocation_policy_bindings
            ).validate_manifest(compiled, manifest)
        if self.idle_factories:
            from .simulation.idle_clearance import _validate_idle_program

            _validate_idle_program(
                compiled, manifest, self.invocation_policy_bindings, self.idle_factories
            )

        from embodichain.lab.sim.atomic_actions.primitives.move_held_object import (
            MoveHeldObjectPlanningRecovery,
        )

        if self.held_staging_bindings or any(
            (
                isinstance(
                    preset.recovery_policy.planning_recovery,
                    MoveHeldObjectPlanningRecovery,
                )
                and preset.recovery_policy.planning_recovery.staging
                for preset in self.robot_profile.presets.values()
            )
        ):
            _validate_program_bindings(
                compiled,
                manifest,
                self.invocation_policy_bindings,
                self.held_staging_bindings,
            )
        continuation_routes = _bind_place_continuation_declarations(
            compiled,
            manifest,
            self.invocation_policy_bindings,
            self.place_continuation_bindings,
            held_staging_bindings=self.held_staging_bindings,
            rigid_entity_ids=self.rigid_settle_entity_ids,
        )
        if any(
            (
                entry.route.requires_full_preview
                and entry.route.place_invocation_id
                not in self.continuation_scene_invocations
                for entry in continuation_routes
            )
        ):
            raise ValueError(
                "Full Place continuation requires its exact declared Place scene service."
            )
        _bind_released_continuations(
            compiled,
            manifest,
            self.invocation_policy_bindings,
            self.released_hand_continuations,
        )
        _validate_pick_program(
            compiled,
            manifest,
            self.invocation_policy_bindings,
            self.pick_closing_bindings,
            free_motion_query=self.native_pick_free_motion,
            continuation_routes=continuation_routes,
        )
        from .simulation.slide_preshape import _validate_slide_preshape_program

        _validate_slide_preshape_program(
            compiled,
            manifest,
            self.invocation_policy_bindings,
            self.slide_preshape_bindings,
        )
        from .simulation.jaw_pour import JawPourFactory

        for factory in self.pour_projection_factories:
            if type(factory) is JawPourFactory:
                factory.validate_program(compiled, self.scene)
        self.resolve_pour_projection_routes(compiled)
        _validate_measured_program(
            compiled,
            manifest,
            self.invocation_policy_bindings,
            self.measured_rigid_bindings,
        )
        from .simulation.drawer_planning import _validate_drawer_program

        _validate_drawer_program(
            compiled, manifest, self.invocation_policy_bindings, self.drawer_bindings
        )
        self.validate_control_program(compiled)
        return compiled

    def resolve_pour_projection_routes(
        self, program: CompiledTaskProgram
    ) -> tuple[ResolvedPourProjectionRoute, ...]:
        """Resolve already-declared runtime planning operands without live assembly.

        Args:
            program: Exact compiled program owning the selected occurrences.

        Returns:
            Typed immutable source projection routes, or an empty tuple when
            this capability is not selected. No Robot or simulator is constructed.
        """
        return resolve_pour_projection_routes(
            program,
            profile=self.robot_profile,
            scene=self.scene,
            call_catalog=self.call_catalog,
            factories=self.pour_projection_factories,
            invocation_policies=self.invocation_policy_bindings,
            bindings=self.pour_projection_bindings,
        )

    def validate_control_program(self, program: CompiledTaskProgram) -> None:
        """Reject foreign targets, budgets or policy phases in a compiled control program."""
        if type(program) is not CompiledTaskProgram:
            raise TypeError("Control validation requires a compiled Task Program.")
        control_calls = {
            call_id: (declaration, prepare)
            for call_id, declaration, prepare in self.control_calls
        }
        counts = {call_id: 0 for call_id in control_calls}
        for segment in program.iter_segments():
            for call in segment.calls:
                call_id = getattr(call.call, "call_id", None)
                if call_id not in control_calls:
                    continue
                declaration, prepare = control_calls[call_id]
                expected = {"object": declaration.binding.object_id}
                if type(declaration.call) is TwistCallDeclaration:
                    expected["setting"] = declaration.binding.target_setting
                if (
                    segment.parallel_block is not None
                    or dict(call.call.arguments) != expected
                ):
                    raise ValueError(
                        "Control occurrences require exact sequential declared targets."
                    )
                required_phase = (
                    "ready"
                    if prepare
                    else "chunk" if declaration.chunk_count is not None else "released"
                )
                allowed_phases = {required_phase}
                if (
                    not prepare
                    and declaration.chunk_count is not None
                    and counts[call_id] == declaration.chunk_count - 1
                ):
                    # The original final Twist owns one release window instead
                    # of a chunk-feedback window. It must not acquire an extra
                    # hold merely to satisfy declaration validation.
                    allowed_phases.add("released")
                matching = [
                    self.control_presets.get(policy.cfg.preset)
                    for policy in segment.post_policies
                ]
                if not any(
                    cfg is not None
                    and cfg[0] == declaration.call.operation_id
                    and cfg[1] in allowed_phases
                    for cfg in matching
                ):
                    raise ValueError(
                        "Control call occurrence lacks its matching physical post-policy."
                    )
                counts[call_id] += 1
        for call_id, (declaration, prepare) in control_calls.items():
            expected_count = (
                1
                if prepare or declaration.chunk_count is None
                else declaration.chunk_count
            )
            if counts[call_id] != expected_count:
                raise ValueError(
                    "Program control occurrences differ from the declared operation budget."
                )
        if self.post_policy_bindings:
            from .simulation.control_assembly import _bind_control_post_policies

            _bind_control_post_policies(
                program,
                self.post_policy_bindings,
                self.control_calls,
                self.control_conditions,
                self.control_cleanup_call_ids,
            )

    def validate_engine(self, engine: AtomicActionEngine) -> None:
        """Require the live engine to expose every statically selected skill."""
        if not isinstance(engine, AtomicActionEngine):
            raise TypeError("engine must be an AtomicActionEngine.")
        for skill_id, expected in self._required_skills.items():
            actual = engine.skills.get(skill_id)
            if actual != expected:
                raise IntegrationFingerprintMismatch(
                    f"Live skill {skill_id!r} differs from the registered "
                    "semantic target descriptor."
                )

    def validate_bound_endpoint_extensions(
        self,
        bound_profile: BoundRobotSkillProfile,
    ) -> None:
        """Match every live resolved endpoint to its fingerprinted declaration."""
        if type(bound_profile) is not BoundRobotSkillProfile:
            raise TypeError("bound_profile must be exactly BoundRobotSkillProfile.")
        bound_profile.assert_current()
        if bound_profile.profile_id != self.robot_profile_id:
            raise IntegrationFingerprintMismatch(
                "The bound robot profile ID differs from the registered profile."
            )

        transport_owner_by_target_type = {
            target_type: transport
            for transport in self.runtime_transport_declarations
            for target_type in transport.target_types
        }
        expected_resource_ids = frozenset(self.robot_profile.resources)
        live_resource_ids = frozenset(bound_profile.resources)
        if live_resource_ids != expected_resource_ids:
            raise IntegrationFingerprintMismatch(
                "Bound robot resource IDs differ from the registered profile; "
                f"expected {sorted(expected_resource_ids)}, "
                f"got {sorted(live_resource_ids)}."
            )
        for resource_id, resource in bound_profile.resources.items():
            expected_resource = self.robot_profile.resources[resource_id]
            if resource.resource_id != expected_resource.resource_id:
                raise IntegrationFingerprintMismatch(
                    f"Bound resource {resource_id!r} declaration ID differs from "
                    "the registered profile."
                )
            if resource.members != expected_resource.members:
                raise IntegrationFingerprintMismatch(
                    f"Bound resource {resource_id!r} members differ from the "
                    "registered profile."
                )
            expected_endpoint_ids = frozenset(expected_resource.endpoints)
            live_endpoint_ids = frozenset(resource.endpoints)
            if live_endpoint_ids != expected_endpoint_ids:
                raise IntegrationFingerprintMismatch(
                    f"Bound resource {resource_id!r} endpoint IDs differ from the "
                    f"registered profile; expected {sorted(expected_endpoint_ids)}, "
                    f"got {sorted(live_endpoint_ids)}."
                )
            for endpoint_id, endpoint in resource.endpoints.items():
                location = f"{resource_id}.{endpoint_id}"
                expected_endpoint = expected_resource.endpoints[endpoint_id]
                if type(endpoint.endpoint) is not type(expected_endpoint) or (
                    _canonical_json(endpoint.endpoint)
                    != _canonical_json(expected_endpoint)
                ):
                    raise IntegrationFingerprintMismatch(
                        f"Bound endpoint {location!r} declaration differs from the "
                        "registered robot profile."
                    )
                endpoint_type = type(endpoint.endpoint)
                declaration = self.endpoint_adapter_declarations.get(endpoint_type)
                if declaration is None:
                    raise IntegrationFingerprintMismatch(
                        f"Bound endpoint {location!r} has undeclared exact type "
                        f"{_qualified_name(endpoint_type)!r}."
                    )
                if endpoint.adapter_id != declaration.adapter_id:
                    raise IntegrationFingerprintMismatch(
                        f"Bound endpoint {location!r} adapter ID "
                        f"{endpoint.adapter_id!r} differs from registered "
                        f"{declaration.adapter_id!r}."
                    )

                target = endpoint.runtime_target
                target_type = type(target)
                if target_type not in declaration.runtime_target_types:
                    raise IntegrationFingerprintMismatch(
                        f"Bound endpoint {location!r} resolved undeclared exact "
                        f"runtime target type {_qualified_name(target_type)!r}."
                    )
                owner = transport_owner_by_target_type.get(target_type)
                if owner is None or owner.transport_id not in (
                    declaration.runtime_transport_ids
                ):
                    raise IntegrationFingerprintMismatch(
                        f"Bound endpoint {location!r} target type has no registered "
                        "adapter transport owner."
                    )
                if target.transport_id != owner.transport_id:
                    raise IntegrationFingerprintMismatch(
                        f"Bound endpoint {location!r} live transport "
                        f"{target.transport_id!r} differs from target type owner "
                        f"{owner.transport_id!r}."
                    )

                feedback_keys = frozenset(
                    (binding.source.provider_id, binding.source.revision)
                    for binding in endpoint.tracking_channels.values()
                )
                if feedback_keys != declaration.tracking_feedback_source_keys:
                    raise IntegrationFingerprintMismatch(
                        f"Bound endpoint {location!r} tracking-feedback routes "
                        "differ from its registered adapter declaration."
                    )
                projector_keys = frozenset(
                    (binding.projector.projector_id, binding.projector.revision)
                    for binding in endpoint.tracking_channels.values()
                )
                if projector_keys != declaration.tracking_projector_keys:
                    raise IntegrationFingerprintMismatch(
                        f"Bound endpoint {location!r} tracking-projector routes "
                        "differ from its registered adapter declaration."
                    )
                evidence_keys = frozenset(
                    (source.provider_id, source.revision)
                    for source in endpoint.effect_sources.values()
                )
                if evidence_keys != declaration.effect_evidence_source_keys:
                    raise IntegrationFingerprintMismatch(
                        f"Bound endpoint {location!r} effect-evidence routes "
                        "differ from its registered adapter declaration."
                    )
                if endpoint_type is ControlPartEndpoint:
                    control_part = endpoint.endpoint.control_part
                    if getattr(target, "control_part", None) != control_part:
                        raise IntegrationFingerprintMismatch(
                            f"Bound endpoint {location!r} runtime target addresses "
                            "a different control part."
                        )
                    if frozenset(endpoint.tracking_channels) != frozenset(
                        {JOINT_POSITION_CHANNEL}
                    ):
                        raise IntegrationFingerprintMismatch(
                            f"Bound endpoint {location!r} must expose exactly the "
                            "built-in joint-position tracking channel."
                        )
                    tracking = endpoint.tracking_channels[JOINT_POSITION_CHANNEL]
                    feedback_address = tracking.source.address
                    if type(feedback_address) is not EndpointTrackingFeedbackAddress:
                        raise IntegrationFingerprintMismatch(
                            f"Bound endpoint {location!r} must use the exact "
                            "built-in endpoint tracking address."
                        )
                    if (
                        feedback_address.channel_id != JOINT_POSITION_CHANNEL
                        or type(feedback_address.target) is not target_type
                        or _canonical_json(feedback_address.target)
                        != _canonical_json(target)
                    ):
                        raise IntegrationFingerprintMismatch(
                            f"Bound endpoint {location!r} tracking address differs "
                            "from its runtime target or channel."
                        )

                    expected_effect_channels = {
                        POSE_RELATION_EFFECT_CHANNEL,
                        JOINT_STATE_EFFECT_CHANNEL,
                    }
                    if GRASP_CAPABILITY in endpoint.endpoint.capabilities:
                        expected_effect_channels.update(
                            {
                                CONTACT_EFFECT_CHANNEL,
                                CONSTRAINT_EFFECT_CHANNEL,
                                FORCE_EFFECT_CHANNEL,
                            }
                        )
                    if frozenset(endpoint.effect_sources) != frozenset(
                        expected_effect_channels
                    ):
                        raise IntegrationFingerprintMismatch(
                            f"Bound endpoint {location!r} effect-evidence channels "
                            "differ from the exact built-in control-part routes."
                        )
                    for channel, source in endpoint.effect_sources.items():
                        address = source.address
                        if (
                            type(address) is not ControlPartEvidenceAddress
                            or address.control_part != control_part
                            or address.channel != channel
                        ):
                            raise IntegrationFingerprintMismatch(
                                f"Bound endpoint {location!r} effect-evidence "
                                f"address for channel {channel!r} differs from its "
                                "control part or channel."
                            )
                elif endpoint.tracking_channels or endpoint.effect_sources:
                    raise IntegrationFingerprintMismatch(
                        f"Bound custom endpoint {location!r} exposes closed-loop "
                        "routes forbidden by the C1 standard runtime."
                    )


def _registration_payload(
    *,
    scene_binding: SimulationSceneBinding,
    scene: SceneManifest,
    robot_profile_binding: SimulationRobotSkillProfileBinding,
    robot_profile: RobotSkillProfile,
    call_catalog: SemanticCallCatalog,
    settle_presets: Mapping[
        str,
        DynamicSettleMonitorCfg
        | SimulationStabilityConditionCfg
        | SimulationJointStabilityConditionCfg
        | SimulationControlConditionCfg,
    ],
    relation_grounder_keys: frozenset[tuple[str, type[Affordance], str]],
    relation_grounders: tuple[RelationTargetGrounder, ...],
    handover_pose_providers: tuple[HandOverPoseProvider, ...],
    extensions: StandardExtensionDeclarations,
    endpoint_adapters: tuple[ResourceEndpointAdapter, ...],
    runtime_transports: tuple[RuntimeTransportActionEncoder, ...],
    parallel_safety_factory: ParallelCommandSafetyValidatorFactory | None,
    control_part_evidence_factory: ControlPartEvidenceProviderFactory | None,
    registered_semantic_lowerer_factories: tuple[RegisteredSemanticLowererFactory, ...],
    planning_services: SimulationPlanningServicesCfg | None = None,
    invocation_policy_bindings: tuple[InvocationPolicyBinding, ...] = (),
    post_policy_bindings: tuple[PostPolicyPresetBinding, ...] = (),
    control_operations: tuple[ControlOperationDeclaration, ...] = (),
    cargo_envelope_cfg: SimulationCargoEnvelopeCfg | None = None,
    recorded_contacts_cfg: SimulationRecordedContactCfg | None = None,
) -> dict[str, object]:
    """Build the versioned canonical fingerprint payload."""
    payload = {
        **({"control_operations": control_operations} if control_operations else {}),
        "schema_version": _CATALOG_FINGERPRINT_SCHEMA_VERSION,
        "scene_binding": scene_binding,
        "scene_manifest": scene.entries,
        "robot_profile_binding": robot_profile_binding,
        "robot_profile": robot_profile,
        "call_descriptors": tuple(
            sorted(
                call_catalog.descriptors.values(),
                key=lambda descriptor: descriptor.call_id,
            )
        ),
        "relation_grounder_keys": relation_grounder_keys,
        "relation_grounders": tuple(
            {
                "key": _relation_grounder_key(grounder),
                "provider": _provider_fingerprint_declaration(grounder),
            }
            for grounder in sorted(
                relation_grounders,
                key=_relation_grounder_order_key,
            )
        ),
        "handover_pose_providers": tuple(
            {
                "provider_id": _handover_pose_provider_id(provider),
                "provider": _provider_fingerprint_declaration(provider),
            }
            for provider in sorted(
                handover_pose_providers,
                key=_handover_pose_provider_id,
            )
        ),
        "standard_extensions": {
            "endpoint_adapters": tuple(
                sorted(
                    extensions.endpoint_adapters.values(),
                    key=lambda declaration: declaration.adapter_id,
                )
            ),
            "runtime_transports": extensions.runtime_transports,
            "parallel_safety": extensions.parallel_safety,
            "control_part_evidence": extensions.control_part_evidence,
            "registered_semantic_lowerers": tuple(
                sorted(
                    extensions.registered_semantic_lowerers.values(),
                    key=lambda declaration: declaration.call_id,
                )
            ),
        },
        "endpoint_adapters": tuple(
            {
                "declaration": extensions.endpoint_adapters[
                    getattr(type(adapter), "endpoint_type")
                ],
                "provider": _provider_fingerprint_declaration(adapter),
            }
            for adapter in sorted(
                endpoint_adapters,
                key=lambda value: getattr(type(value), "adapter_id"),
            )
        ),
        "runtime_transports": tuple(
            {
                "declaration": next(
                    declaration
                    for declaration in extensions.runtime_transports
                    if declaration.transport_id
                    == getattr(type(transport), "transport_id")
                ),
                "provider": _provider_fingerprint_declaration(transport),
            }
            for transport in runtime_transports
        ),
        "parallel_safety_factory": (
            None
            if parallel_safety_factory is None
            else {
                "declaration": extensions.parallel_safety,
                "provider": _provider_fingerprint_declaration(parallel_safety_factory),
            }
        ),
        "control_part_evidence_factory": (
            None
            if control_part_evidence_factory is None
            else {
                "declaration": extensions.control_part_evidence,
                "provider": _provider_fingerprint_declaration(
                    control_part_evidence_factory
                ),
            }
        ),
        "registered_semantic_lowerer_factories": tuple(
            {
                "declaration": extensions.registered_semantic_lowerers[
                    getattr(type(factory), "call_id")
                ],
                "provider": _provider_fingerprint_declaration(factory),
            }
            for factory in sorted(
                registered_semantic_lowerer_factories,
                key=lambda value: getattr(type(value), "call_id"),
            )
        ),
        "post_policy_kinds": _POST_POLICY_KINDS,
        "settle_presets": settle_presets,
        "validator_kinds": _VALIDATOR_KINDS,
    }

    if planning_services is not None:
        payload["planning_services"] = _effective_planning_service_fields(
            planning_services
        )
    if invocation_policy_bindings:
        payload["invocation_policy_bindings"] = invocation_policy_bindings
    if post_policy_bindings:
        payload["post_policy_bindings"] = post_policy_bindings
    if cargo_envelope_cfg is not None:
        payload["cargo_envelope"] = cargo_envelope_cfg
    if recorded_contacts_cfg is not None:
        payload["recorded_contacts"] = recorded_contacts_cfg
    return payload


def _validate_native_planning_registration(
    self: SimulationTaskProgramRegistration,
    profile: RobotSkillProfile,
    scene: SceneManifest,
) -> None:
    """Cross-check immutable planning declarations before constructing a live host."""
    from .simulation.place_scene import _validate_inside_binding
    from .simulation.targets import SceneEntityTarget
    from embodichain.lab.sim.atomic_actions.primitives.hand_over import HandOverOptions

    if (
        self.planning_services is not None
        and self.planning_services.measured_rigid_bindings
    ):
        policies = {value.invocation_id for value in self.invocation_policy_bindings}
        rigid = {value.entity_id for value in self.scene_binding.rigid_objects}
        containers = {value.entity_id: value for value in self.scene_binding.containers}
        for value in self.planning_services.measured_rigid_bindings:
            if (
                value.invocation_id not in policies
                or value.object_id not in rigid
                or (
                    value.container_id is not None
                    and (
                        value.container_id not in containers
                        or containers[value.container_id].parent_id not in rigid
                    )
                )
            ):
                raise ValueError(
                    "Measured bindings require program-bound policies and exact rigid/container owners."
                )
    if self.planning_services is not None:
        occurrences = {item.invocation_id for item in self.invocation_policy_bindings}
        if any(
            item.invocation_id not in occurrences
            for item in self.planning_services.place_scene_bindings
        ):
            raise ValueError(
                "Place scene roles require a program-bound invocation policy declaration."
            )
    if self.planning_services is not None:
        from .simulation.place_scene import _validate_inside_binding

        containers = {item.entity_id: item for item in self.scene_binding.containers}
        rigids = {
            item.entity_id: item.simulation_uid
            for item in self.scene_binding.rigid_objects
        }
        roster = (
            dict(self.planning_services.collision_roster.rigid_objects)
            if self.planning_services.collision_roster is not None
            else {}
        )
        for binding in self.planning_services.place_scene_bindings:
            if binding.inside_subject_id is None:
                continue
            _validate_inside_binding(binding, roster, containers)
            parent = containers[binding.inside_container_id].parent_id
            if any(
                rigids.get(entity) != roster[entity]
                for entity in (binding.inside_subject_id, parent)
            ):
                raise ValueError(
                    "Inside landing roster must retain exact registered native rigid owners."
                )
    if self.planning_services is not None and self.planning_services.coordinated_grasps:
        grasp_targets = {
            endpoint.control_part
            for resource in self.robot_profile_binding.resources
            for endpoint in resource.endpoints
            if endpoint.endpoint_id == "grasp"
        }
        if any(
            name not in grasp_targets
            for pair in self.planning_services.coordinated_grasps
            for name in (pair.left_grasp_target_id, pair.right_grasp_target_id)
        ):
            raise ValueError(
                "Configured coordinated grasp pairs must select declared grasp endpoints."
            )
    if self.planning_services is not None:
        selected_ids = {item.invocation_id for item in self.invocation_policy_bindings}
        rigid_ids = {item.entity_id for item in self.scene_binding.rigid_objects}
        for binding in self.planning_services.held_staging_bindings:
            if (
                binding.invocation_id not in selected_ids
                or binding.rigid_entity_id not in rigid_ids
                or (
                    binding.predecessor_invocation_id is not None
                    and binding.predecessor_invocation_id not in selected_ids
                )
            ):
                raise ValueError(
                    "Held staging requires registered rigid identities and program-bound current/predecessor policies."
                )
    if self.planning_services is not None:
        occurrences = {value.invocation_id for value in self.invocation_policy_bindings}
        rigid_ids = {value.entity_id for value in self.scene_binding.rigid_objects}
        for value in self.planning_services.pick_closing_bindings:
            if (
                value.invocation_id not in occurrences
                or value.object_id not in rigid_ids
                or (
                    value.opening_reference_id is not None
                    and value.opening_reference_id not in rigid_ids
                )
            ):
                raise ValueError(
                    "Native Pick bindings require program-bound policies and explicit rigid subjects/references."
                )
    if self.planning_services is not None:
        policy_ids = {value.invocation_id for value in self.invocation_policy_bindings}
        if any(
            identity not in policy_ids
            for value in self.planning_services.pour_projection_bindings
            for identity in value.invocation_ids
        ):
            raise ValueError(
                "All Pour projection roles require program-bound invocation policies."
            )
    if self.planning_services is not None:
        occurrences = {value.invocation_id for value in self.invocation_policy_bindings}
        native_rigids = {
            value.simulation_uid for value in self.scene_binding.rigid_objects
        }
        native_articulations = {
            value.simulation_uid for value in self.scene_binding.articulations
        }
        rigid_owners = {
            value.entity_id: value.simulation_uid
            for value in self.scene_binding.rigid_objects
        }
        articulation_owners = {
            value.entity_id: value.simulation_uid
            for value in self.scene_binding.articulations
        }
        link_owners = {
            value.entity_id: (
                articulation_owners.get(value.articulation_id),
                value.native_link_name,
            )
            for value in self.scene_binding.links
        }
        drawer_target_parents = {
            value.entity_id: value.parent_id for value in self.scene_binding.containers
        }
        for value in self.planning_services.drawer_bindings:
            if (
                value.invocation_id not in occurrences
                or rigid_owners.get(value.object_id) != value.object_id
                or value.source.object_id not in native_articulations
                or link_owners.get(value.link_entity_id)
                != (value.source.object_id, value.source.link)
            ):
                raise ValueError(
                    "Drawer bindings require policy-bound occurrences and exact native entity owners."
                )
            if (
                value.operation in {"place", "transport"}
                and drawer_target_parents.get(value.target_id) != value.link_entity_id
            ) or (
                value.operation == "close" and value.target_id != value.link_entity_id
            ):
                raise ValueError(
                    "Drawer targets must belong to the declared physical drawer link."
                )
        for value in self.planning_services.slide_preshape_bindings:
            if (
                value.invocation_id not in occurrences
                or value.table_uid not in native_rigids
            ):
                raise ValueError(
                    "Slide preshape requires program-bound occurrences and declared native support."
                )
    from embodichain.lab.sim.atomic_actions.primitives.coordinated_pickment import (
        CoordinatedPickmentOptions,
    )

    if (
        self.planning_services is not None
        and self.planning_services.capture_coordinated_cargo
        and not self.scene_binding.rigid_objects
    ):
        raise ValueError(
            "Coordinated cargo capture requires declared native rigid bindings."
        )
    for preset in profile.presets.values():
        for options in preset.action_option_templates.values():
            if (
                isinstance(options, CoordinatedPickmentOptions)
                and options.prefer_cargo_clearance
            ):
                if (
                    self.planning_services is None
                    or not self.planning_services.capture_coordinated_cargo
                ):
                    raise ValueError(
                        "Cargo preference requires explicit capture_coordinated_cargo services."
                    )
            if isinstance(options, HandOverOptions) and options.require_receiver_scene:
                if (
                    self.planning_services is None
                    or not self.planning_services.native_handover_scene
                ):
                    raise ValueError(
                        "Native receiver qualification requires declared native_handover_scene services."
                    )
    from .simulation.transport import DrawerTransportLowererFactory

    drawer_factories = tuple(
        factory
        for factory in self.registered_semantic_lowerer_factories
        if type(factory) is DrawerTransportLowererFactory
    )
    if drawer_factories:
        from .simulation.targets import SceneEntityTarget

        transports = (
            ()
            if self.planning_services is None
            else tuple(
                value
                for value in self.planning_services.drawer_bindings
                if value.operation == "transport"
            )
        )
        if not transports:
            raise ValueError(
                "Configured drawer transport requires its drawer-specific planning service; generic MoveHeldObject is not a compatible fallback."
            )
        routes = {
            route.target_id: route
            for factory in drawer_factories
            for route in factory.routes
        }
        for value in transports:
            route = routes.get(value.target_id)
            if (
                route is None
                or route.object_id != value.object_id
                or type(route.pose) is not SceneEntityTarget
                or route.pose.entity_id != value.target_id
            ):
                raise ValueError(
                    "Drawer transport route must retain its declared payload and scene target."
                )

    from .simulation.grip_evidence import NativeGripEvidenceFactory

    if type(self.control_part_evidence_factory) is NativeGripEvidenceFactory:
        evidence = self.control_part_evidence_factory
        objects = {
            item.entity_id: item.simulation_uid
            for item in self.scene_binding.rigid_objects
        }
        resolved = tuple(
            (scene.resolve(entity, expected_type=SceneObjectRef).entity_id, uid)
            for entity, uid in evidence.object_bindings
        )
        if len({entity for entity, _ in resolved}) != len(resolved):
            raise ValueError(
                "Native grip object bindings require distinct canonical identities."
            )
        if any(objects.get(entity) != uid for entity, uid in resolved):
            raise ValueError(
                "Native grip object bindings must match exact registered rigid owners."
            )
        if any(
            (
                binding.control_part not in dict(evidence.pad_bindings)
                for binding in evidence.opening_bindings
            )
        ):
            raise ValueError(
                "Native grip opening bindings must select declared pad hands."
            )


@dataclass(frozen=True, slots=True)
class SimulationTaskProgramRegistration:
    """Exact immutable task-owned simulation integration registration."""

    scene_binding: SimulationSceneBinding
    robot_profile_binding: SimulationRobotSkillProfileBinding
    call_catalog: SemanticCallCatalog = field(
        default_factory=builtin_semantic_call_catalog
    )
    settle_presets: Mapping[
        str,
        DynamicSettleMonitorCfg
        | SimulationStabilityConditionCfg
        | SimulationJointStabilityConditionCfg
        | SimulationControlConditionCfg,
    ] = field(default_factory=default_simulation_settle_presets)
    relation_grounders: tuple[RelationTargetGrounder, ...] = ()
    handover_pose_providers: tuple[HandOverPoseProvider, ...] = ()
    endpoint_adapters: tuple[ResourceEndpointAdapter, ...] = ()
    runtime_transports: tuple[RuntimeTransportActionEncoder, ...] = ()
    parallel_safety_factory: ParallelCommandSafetyValidatorFactory | None = None
    control_part_evidence_factory: ControlPartEvidenceProviderFactory | None = None
    registered_semantic_lowerer_factories: tuple[
        RegisteredSemanticLowererFactory, ...
    ] = ()
    planning_services: SimulationPlanningServicesCfg | None = field(
        default=None, kw_only=True
    )
    invocation_policy_bindings: tuple[InvocationPolicyBinding, ...] = field(
        default=(), kw_only=True
    )
    post_policy_bindings: tuple[PostPolicyPresetBinding, ...] = field(
        default=(), kw_only=True
    )
    control_operations: tuple[ControlOperationDeclaration, ...] = ()
    recorded_contacts_cfg: SimulationRecordedContactCfg | None = field(
        default=None, kw_only=True
    )
    cargo_envelope_cfg: SimulationCargoEnvelopeCfg | None = field(
        default=None, kw_only=True
    )
    catalog: TaskProgramIntegrationCatalog = field(init=False)
    _parallel_safety_validator_history: list[ParallelCommandSafetyValidator] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _parallel_safety_validator_lock: LockType = field(
        init=False,
        repr=False,
        compare=False,
    )
    _control_part_evidence_provider_history: list[EffectEvidenceProvider] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _control_part_evidence_provider_lock: LockType = field(
        init=False,
        repr=False,
        compare=False,
    )
    _registered_semantic_lowerer_history: list[RegisteredSemanticLowerer] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _registered_semantic_lowerer_lock: LockType = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self.recorded_contacts_cfg is not None:
            if type(self.recorded_contacts_cfg) is not SimulationRecordedContactCfg:
                raise TypeError("recorded_contacts_cfg requires its typed declaration.")
            cfg = self.recorded_contacts_cfg.snapshot()
            parts = {
                endpoint.control_part
                for resource in self.robot_profile_binding.resources
                for endpoint in resource.endpoints
                if endpoint.endpoint_id == "grasp"
            }
            if not {part for part, _ in cfg.pad_bindings} <= parts:
                raise ValueError("Recorded pads require declared grasp control parts.")
            object.__setattr__(self, "recorded_contacts_cfg", cfg)
        if self.cargo_envelope_cfg is not None:
            if type(self.cargo_envelope_cfg) is not SimulationCargoEnvelopeCfg:
                raise TypeError("cargo_envelope_cfg requires its typed declaration.")
            cargo = self.cargo_envelope_cfg.snapshot()
            rigid = {value.simulation_uid for value in self.scene_binding.rigid_objects}
            owners = rigid | {
                value.simulation_uid for value in self.scene_binding.articulations
            }
            if (
                not set(cargo.carrier_uids) <= rigid
                or not set(cargo.excluded_uids) <= owners
            ):
                raise ValueError(
                    "Cargo selections differ from registered native physical owners."
                )
            object.__setattr__(self, "cargo_envelope_cfg", cargo)
        from .invocation_policy import InvocationPolicyBinding, PostPolicyPresetBinding

        if type(self.post_policy_bindings) is not tuple or any(
            type(value) is not PostPolicyPresetBinding
            for value in self.post_policy_bindings
        ):
            raise TypeError(
                "post_policy_bindings must be a tuple of PostPolicyPresetBinding."
            )
        if type(self.invocation_policy_bindings) is not tuple or not all(
            type(item) is InvocationPolicyBinding
            for item in self.invocation_policy_bindings
        ):
            raise TypeError(
                "invocation_policy_bindings must be an immutable tuple of InvocationPolicyBinding."
            )
        if (
            len({item.program_fingerprint for item in self.invocation_policy_bindings})
            > 1
        ):
            raise ValueError(
                "Configured invocation bindings must select one program fingerprint."
            )
        if len({item.invocation_id for item in self.invocation_policy_bindings}) != len(
            self.invocation_policy_bindings
        ):
            raise ValueError("Duplicate invocation policy occurrence.")
        if self.planning_services is not None:
            if type(self.planning_services) is not SimulationPlanningServicesCfg:
                raise TypeError(
                    "planning_services must be SimulationPlanningServicesCfg or None."
                )
            object.__setattr__(
                self, "planning_services", self.planning_services.snapshot()
            )
        if type(self.scene_binding) is not SimulationSceneBinding:
            raise TypeError("scene_binding must be exactly SimulationSceneBinding.")
        if type(self.robot_profile_binding) is not SimulationRobotSkillProfileBinding:
            raise TypeError(
                "robot_profile_binding must be exactly "
                "SimulationRobotSkillProfileBinding."
            )
        if type(self.call_catalog) is not SemanticCallCatalog:
            raise TypeError("call_catalog must be exactly SemanticCallCatalog.")
        settle_presets = _snapshot_settle_presets(self.settle_presets)
        control_calls = _validate_control_registration(
            self.control_operations,
            self.scene_binding,
            settle_presets,
            self.registered_semantic_lowerer_factories,
        )
        rigid_ids = {binding.entity_id for binding in self.scene_binding.rigid_objects}
        reference_ids = (
            rigid_ids
            | {binding.entity_id for binding in self.scene_binding.articulations}
            | {binding.entity_id for binding in self.scene_binding.links}
        )
        articulation_ids = {
            binding.entity_id for binding in self.scene_binding.articulations
        }
        for cfg in settle_presets.values():
            if type(cfg) is SimulationJointStabilityConditionCfg:
                if cfg.entity not in articulation_ids:
                    raise ValueError(
                        "Joint stability requires an explicit articulation binding."
                    )
                continue
            if type(cfg) is not SimulationStabilityConditionCfg:
                continue
            if cfg.entity not in rigid_ids:
                raise ValueError(
                    "Stability condition entities require native rigid bindings."
                )
            if cfg.reference is not None and cfg.reference not in reference_ids:
                raise ValueError(
                    "Stability condition reference is not explicitly bound."
                )
            if (
                cfg.kind in {"stack", "supported_placement", "supported_region"}
                and cfg.reference not in rigid_ids
            ):
                raise ValueError(
                    "Contact support conditions require a native rigid reference."
                )
        object.__setattr__(self, "settle_presets", settle_presets)
        configured_relation_grounders = _configured_relation_grounders(
            self.scene_binding,
            _snapshot_relation_grounders(self.relation_grounders),
        )
        relation_grounders = _snapshot_relation_grounders(
            (
                *_builtin_relation_grounders(self.scene_binding),
                *configured_relation_grounders,
            )
        )
        object.__setattr__(self, "relation_grounders", relation_grounders)
        relation_grounder_keys = frozenset(
            _relation_grounder_key(grounder) for grounder in relation_grounders
        )
        handover_pose_providers = _snapshot_handover_pose_providers(
            self.handover_pose_providers
        )
        object.__setattr__(
            self,
            "handover_pose_providers",
            handover_pose_providers,
        )

        from .simulation.prismatic_lowering import _validate_prismatic_source_bindings

        _validate_prismatic_source_bindings(
            self.scene_binding, self.registered_semantic_lowerer_factories
        )
        scene = self.scene_binding.declare()
        profile = self.robot_profile_binding.declare()
        if (
            self.planning_services is not None
            and self.planning_services.native_handover_scene
        ):
            native_rigids = {
                uid for _, uid in self.planning_services.collision_roster.rigid_objects
            }
            if not native_rigids or any(
                item.simulation_uid not in native_rigids
                for item in self.scene_binding.rigid_objects
            ):
                raise ValueError(
                    "Native HandOver collision_roster must cover declared rigid owners."
                )
        from embodichain.lab.sim.atomic_actions.primitives.hand_over import (
            HandOverOptions,
            HandOverPlanningRecovery,
        )

        for preset in profile.presets.values():
            for options in preset.action_option_templates.values():
                if type(options) is HandOverOptions:
                    adaptive = (
                        type(preset.recovery_policy.planning_recovery)
                        is HandOverPlanningRecovery
                    )
                    if options.adaptive_receive != adaptive:
                        raise ValueError(
                            "Adaptive receiving and its bounded recovery must be declared together."
                        )
                    if options.require_receiver_scene and (
                        self.planning_services is None
                        or not self.planning_services.native_handover_scene
                    ):
                        raise ValueError(
                            "Native receiver qualification requires declared native_handover_scene services."
                        )
        from embodichain.lab.sim.atomic_actions.primitives.move_joints import (
            MoveJointsOptions,
        )

        if any(
            type(options) is MoveJointsOptions and options.require_free_motion
            for preset in profile.presets.values()
            for options in preset.action_option_templates.values()
        ) and (
            self.planning_services is None
            or self.planning_services.free_motion_query != "discrete_mesh"
        ):
            raise ValueError(
                "Qualified joint motion requires explicit discrete_mesh planning services."
            )
        idle_factories = tuple(
            factory
            for factory in self.registered_semantic_lowerer_factories
            if type(factory) is IdleClearanceFactory
        )
        if idle_factories and (
            self.planning_services is None
            or self.planning_services.free_motion_query != "discrete_mesh"
        ):
            raise ValueError(
                "Idle clear/restore requires explicit discrete_mesh planning services."
            )
        _validate_native_planning_registration(self, profile, scene)
        _validate_standard_effect_monitors(profile)
        _validate_standard_tracking_metrics(profile)
        extensions = build_standard_extension_declarations(
            profile=profile,
            endpoint_adapters=self.endpoint_adapters,
            runtime_transports=self.runtime_transports,
            parallel_safety_factory=self.parallel_safety_factory,
            control_part_evidence_factory=self.control_part_evidence_factory,
            registered_semantic_lowerer_factories=(
                self.registered_semantic_lowerer_factories
            ),
        )
        _validate_standard_call_catalog(
            self.call_catalog,
            extensions.registered_semantic_lowerers,
        )
        for factory in self.registered_semantic_lowerer_factories:
            if isinstance(factory, RegisteredSemanticLowererSceneValidator):
                factory.validate_scene_declaration(scene)
        selected_handover_provider = profile.grounding_providers.get("hand_over")
        registered_handover_provider_ids = {
            _handover_pose_provider_id(provider) for provider in handover_pose_providers
        }
        if (
            selected_handover_provider is not None
            and selected_handover_provider not in registered_handover_provider_ids
        ):
            raise ValueError(
                "Robot profile selects handover pose provider "
                f"{selected_handover_provider!r}, but the task registration did "
                "not install it."
            )
        builtin_skills = {
            descriptor.skill_id: descriptor
            for action_type in BUILTIN_ACTION_TYPES
            if (descriptor := action_type.descriptor()).agent_visible
            and descriptor.binding_contract is not None
        }
        required_skills: dict[str, SkillDescriptor] = {}
        for descriptor in self.call_catalog.descriptors.values():
            target = descriptor.target_descriptor
            installed = builtin_skills.get(descriptor.skill_id)
            if target is None or installed != target:
                raise ValueError(
                    f"Semantic call {descriptor.call_id!r} targets skill "
                    f"{descriptor.skill_id!r}, which is not installed by the "
                    "standard simulation factory."
                )
            required_skills[descriptor.skill_id] = target

        fingerprint = _digest(
            _registration_payload(
                scene_binding=self.scene_binding,
                scene=scene,
                robot_profile_binding=self.robot_profile_binding,
                robot_profile=profile,
                call_catalog=self.call_catalog,
                settle_presets=settle_presets,
                relation_grounder_keys=relation_grounder_keys,
                relation_grounders=relation_grounders,
                handover_pose_providers=handover_pose_providers,
                extensions=extensions,
                endpoint_adapters=self.endpoint_adapters,
                runtime_transports=self.runtime_transports,
                parallel_safety_factory=self.parallel_safety_factory,
                control_part_evidence_factory=self.control_part_evidence_factory,
                registered_semantic_lowerer_factories=(
                    self.registered_semantic_lowerer_factories
                ),
                planning_services=self.planning_services,
                invocation_policy_bindings=self.invocation_policy_bindings,
                post_policy_bindings=self.post_policy_bindings,
                control_operations=self.control_operations,
                cargo_envelope_cfg=self.cargo_envelope_cfg,
                recorded_contacts_cfg=self.recorded_contacts_cfg,
            )
        )
        from .simulation.prismatic_lowering import ArticulationParkFactory

        object.__setattr__(
            self,
            "catalog",
            TaskProgramIntegrationCatalog(
                invocation_policy_bindings=self.invocation_policy_bindings,
                post_policy_bindings=self.post_policy_bindings,
                inside_landing_targets={
                    binding.invocation_id: (
                        binding.inside_subject_id,
                        binding.inside_container_id,
                    )
                    for binding in (
                        ()
                        if self.planning_services is None
                        else self.planning_services.place_scene_bindings
                    )
                    if binding.inside_subject_id is not None
                },
                coordinated_grasp_pairs=tuple(
                    (pair.left_grasp_target_id, pair.right_grasp_target_id)
                    for pair in (
                        ()
                        if self.planning_services is None
                        else self.planning_services.coordinated_grasps
                    )
                ),
                held_staging_bindings=(
                    ()
                    if self.planning_services is None
                    else self.planning_services.held_staging_bindings
                ),
                pour_projection_bindings=(
                    ()
                    if self.planning_services is None
                    else self.planning_services.pour_projection_bindings
                ),
                pour_projection_factories=self.registered_semantic_lowerer_factories,
                place_continuation_bindings=(
                    ()
                    if self.planning_services is None
                    else self.planning_services.place_continuation_bindings
                ),
                continuation_scene_invocations=frozenset(
                    ()
                    if self.planning_services is None
                    else (
                        value.invocation_id
                        for value in self.planning_services.place_scene_bindings
                    )
                ),
                native_pick_free_motion=self.planning_services is not None
                and self.planning_services.free_motion_query == "discrete_mesh",
                released_hand_continuations=(
                    ()
                    if self.planning_services is None
                    else self.planning_services.released_hand_continuations
                ),
                pick_closing_bindings=(
                    ()
                    if self.planning_services is None
                    else self.planning_services.pick_closing_bindings
                ),
                measured_rigid_bindings=(
                    ()
                    if self.planning_services is None
                    else self.planning_services.measured_rigid_bindings
                ),
                drawer_bindings=(
                    ()
                    if self.planning_services is None
                    else self.planning_services.drawer_bindings
                ),
                slide_preshape_bindings=(
                    ()
                    if self.planning_services is None
                    else self.planning_services.slide_preshape_bindings
                ),
                idle_factories=idle_factories,
                scene_registry_id=self.scene_binding.registry_id,
                robot_profile_id=self.robot_profile_binding.profile_id,
                scene=scene,
                robot_profile=profile,
                call_catalog=self.call_catalog,
                relation_grounder_keys=relation_grounder_keys,
                settle_preset_ids=frozenset(settle_presets),
                condition_settle_entities={
                    key: cfg.entity
                    for key, cfg in settle_presets.items()
                    if type(cfg)
                    in (
                        SimulationStabilityConditionCfg,
                        SimulationJointStabilityConditionCfg,
                        SimulationControlConditionCfg,
                    )
                },
                pose_delta_settle_preset_ids=frozenset(
                    key
                    for key, cfg in settle_presets.items()
                    if isinstance(cfg, SimulationSettleMonitorCfg)
                    and cfg.measurement == "pose_delta"
                ),
                rigid_settle_entity_ids=frozenset(
                    binding.entity_id for binding in self.scene_binding.rigid_objects
                )
                | frozenset(
                    binding.entity_id for binding in self.scene_binding.antipodal_grasps
                ),
                endpoint_adapter_declarations=extensions.endpoint_adapters,
                runtime_transport_declarations=extensions.runtime_transports,
                parallel_safety_declaration=extensions.parallel_safety,
                control_part_evidence_declaration=extensions.control_part_evidence,
                registered_semantic_lowerer_declarations=(
                    extensions.registered_semantic_lowerers
                ),
                fingerprint=fingerprint,
                control_calls=control_calls,
                control_conditions={
                    key: cfg
                    for key, cfg in settle_presets.items()
                    if type(cfg) is SimulationControlConditionCfg
                },
                control_cleanup_call_ids=frozenset(
                    value.call_id
                    for value in self.registered_semantic_lowerer_factories
                    if isinstance(value, ArticulationParkFactory)
                ),
                control_presets={
                    key: (cfg.operation_id, cfg.phase)
                    for key, cfg in settle_presets.items()
                    if type(cfg) is SimulationControlConditionCfg
                },
                _required_skills=required_skills,
            ),
        )
        object.__setattr__(self, "_parallel_safety_validator_history", [])
        object.__setattr__(self, "_parallel_safety_validator_lock", Lock())
        object.__setattr__(self, "_control_part_evidence_provider_history", [])
        object.__setattr__(self, "_control_part_evidence_provider_lock", Lock())
        object.__setattr__(self, "_registered_semantic_lowerer_history", [])
        object.__setattr__(self, "_registered_semantic_lowerer_lock", Lock())

    def validate_num_envs(self, num_envs: object) -> None:
        """Preflight the unchanged single-environment scope of explicit services.

        Args:
            num_envs: Positive environment count declared by the host.
        """
        self.assert_unchanged()
        if self.recorded_contacts_cfg is not None:
            self.recorded_contacts_cfg.validate_num_envs(num_envs)
        if self.planning_services is not None:
            self.planning_services.validate_num_envs(num_envs)
        from embodichain.lab.sim.atomic_actions.primitives.hand_over import (
            HandOverOptions,
        )

        from embodichain.lab.sim.atomic_actions.primitives.move_held_object import (
            MoveHeldObjectOptions,
        )
        from embodichain.lab.sim.atomic_actions.primitives.move_end_effector import (
            MoveEndEffectorOptions,
        )

        adaptive = any(
            type(options) is HandOverOptions and options.adaptive_receive
            for preset in self.catalog.robot_profile.presets.values()
            for options in preset.action_option_templates.values()
        )
        linear = any(
            type(options)
            in (HandOverOptions, MoveHeldObjectOptions, MoveEndEffectorOptions)
            and options.linear_segments
            for preset in self.catalog.robot_profile.presets.values()
            for options in preset.action_option_templates.values()
        )
        if (adaptive or linear) and (type(num_envs) is not int or num_envs != 1):
            raise ValueError("Declared Cartesian manipulation modes require B=1.")

    def validate_physics(self, physics: object) -> None:
        """Preflight declared native observation capabilities without starting Sim.

        Args:
            physics: Explicit environment-owned backend name.
        """
        self.assert_unchanged()
        if self.recorded_contacts_cfg is not None and physics != "default":
            raise ValueError("Recorded source contacts require Default physics.")
        if self.planning_services is not None:
            self.planning_services.validate_physics(physics)

    @property
    def requires_scoped_control_services(self) -> bool:
        """Whether declared source controls require a real compiled bridge scope."""
        from .simulation.prismatic_lowering import (
            ArticulationSlideFactory,
            ArticulationParkFactory,
        )

        return bool(self.control_operations) or any(
            isinstance(value, (ArticulationSlideFactory, ArticulationParkFactory))
            for value in self.registered_semantic_lowerer_factories
        )

    @property
    def fingerprint(self) -> str:
        """Return the canonical registration fingerprint."""
        return self.catalog.fingerprint

    def _checked_fingerprint_payload(self) -> dict[str, object]:
        scene = self.scene_binding.declare()
        profile = self.robot_profile_binding.declare()
        try:
            from .simulation.prismatic_lowering import (
                _validate_prismatic_source_bindings,
            )

            _validate_prismatic_source_bindings(
                self.scene_binding, self.registered_semantic_lowerer_factories
            )
            _validate_control_registration(
                self.control_operations,
                self.scene_binding,
                self.settle_presets,
                self.registered_semantic_lowerer_factories,
            )
            _validate_standard_effect_monitors(profile)
            _validate_standard_tracking_metrics(profile)
            extensions = build_standard_extension_declarations(
                profile=profile,
                endpoint_adapters=self.endpoint_adapters,
                runtime_transports=self.runtime_transports,
                parallel_safety_factory=self.parallel_safety_factory,
                control_part_evidence_factory=self.control_part_evidence_factory,
                registered_semantic_lowerer_factories=(
                    self.registered_semantic_lowerer_factories
                ),
            )
            _validate_standard_call_catalog(
                self.call_catalog,
                extensions.registered_semantic_lowerers,
            )
            for factory in self.registered_semantic_lowerer_factories:
                if isinstance(factory, RegisteredSemanticLowererSceneValidator):
                    factory.validate_scene_declaration(scene)
            relation_grounders = _snapshot_relation_grounders(self.relation_grounders)
            relation_grounder_keys = frozenset(
                _relation_grounder_key(grounder) for grounder in relation_grounders
            )
            handover_pose_providers = _snapshot_handover_pose_providers(
                self.handover_pose_providers
            )
            payload = _registration_payload(
                scene_binding=self.scene_binding,
                scene=scene,
                robot_profile_binding=self.robot_profile_binding,
                robot_profile=profile,
                call_catalog=self.call_catalog,
                settle_presets=self.settle_presets,
                relation_grounder_keys=relation_grounder_keys,
                relation_grounders=relation_grounders,
                handover_pose_providers=handover_pose_providers,
                extensions=extensions,
                endpoint_adapters=self.endpoint_adapters,
                runtime_transports=self.runtime_transports,
                parallel_safety_factory=self.parallel_safety_factory,
                control_part_evidence_factory=self.control_part_evidence_factory,
                registered_semantic_lowerer_factories=(
                    self.registered_semantic_lowerer_factories
                ),
                planning_services=self.planning_services,
                invocation_policy_bindings=self.invocation_policy_bindings,
                post_policy_bindings=self.post_policy_bindings,
                control_operations=self.control_operations,
                cargo_envelope_cfg=self.cargo_envelope_cfg,
                recorded_contacts_cfg=self.recorded_contacts_cfg,
            )
            current = _digest(payload)
        except (TypeError, ValueError) as exc:
            raise IntegrationFingerprintMismatch(
                "Task Program integration provider declaration changed after "
                "task registration."
            ) from exc
        if current != self.fingerprint:
            raise IntegrationFingerprintMismatch(
                "Task Program integration declaration changed after task "
                "registration."
            )
        return payload

    def assert_unchanged(self) -> None:
        """Reject nested declaration drift before live component creation."""
        self._checked_fingerprint_payload()

    def fingerprint_payload(self) -> dict[str, object]:
        """Snapshot the complete validated registration declaration as JSON data.

        Returns:
            Independently owned canonical data whose SHA-256 equals
            :attr:`fingerprint`. Changing the returned data does not change the
            registration or its runtime identity.

        Raises:
            IntegrationFingerprintMismatch: A registered declaration has drifted.
        """
        try:
            payload = cast(
                dict[str, object], _canonical_value(self._checked_fingerprint_payload())
            )
            current = _digest(payload)
        except (TypeError, ValueError) as exc:
            raise IntegrationFingerprintMismatch(
                "Task Program integration declaration changed during snapshot."
            ) from exc
        if current != self.fingerprint:
            raise IntegrationFingerprintMismatch(
                "Task Program integration declaration changed during snapshot."
            )
        return payload

    @property
    def endpoint_adapter_map(
        self,
    ) -> Mapping[type[ResourceEndpoint], ResourceEndpointAdapter]:
        """Return custom live adapters keyed by their exact endpoint type."""
        return MappingProxyType(
            {
                getattr(type(adapter), "endpoint_type"): adapter
                for adapter in self.endpoint_adapters
            }
        )

    def create_registered_semantic_lowerers(
        self,
        *,
        simulation: object,
        robot: object,
        scene_registry: SceneRegistry,
        engine: AtomicActionEngine,
        runtime_services: TaskProgramRuntimeServiceContext | None = None,
    ) -> tuple[RegisteredSemanticLowerer, ...]:
        """Create fresh live lowerers declared by this registration."""
        self.assert_unchanged()
        if type(scene_registry) is not SceneRegistry:
            raise TypeError("scene_registry must be exactly SceneRegistry.")
        if not isinstance(engine, AtomicActionEngine):
            raise TypeError("engine must be an AtomicActionEngine.")
        if engine.robot is not robot:
            raise ValueError("engine and factory must reference the exact same robot.")

        if (
            runtime_services is not None
            and type(runtime_services) is not TaskProgramRuntimeServiceContext
        ):
            raise TypeError(
                "runtime_services must be exactly TaskProgramRuntimeServiceContext."
            )
        created: list[RegisteredSemanticLowerer] = []
        with self._registered_semantic_lowerer_lock:
            for factory in self.registered_semantic_lowerer_factories:
                call_id = _exact_identifier(
                    getattr(type(factory), "call_id", None),
                    field_name="RegisteredSemanticLowererFactory.call_id",
                )
                if isinstance(factory, RuntimeBoundSemanticLowererFactory):
                    if runtime_services is None:
                        raise ValueError(
                            "This lowerer requires the actual bridge-owned runtime service context."
                        )
                    lowerer = factory.create_with_runtime_services(
                        simulation=simulation,
                        robot=robot,
                        scene_registry=scene_registry,
                        engine=engine,
                        runtime_services=runtime_services,
                    )
                else:
                    lowerer = factory.create(
                        simulation=simulation,
                        robot=robot,
                        scene_registry=scene_registry,
                        engine=engine,
                    )
                if not isinstance(lowerer, RegisteredSemanticLowerer):
                    raise TypeError(
                        "RegisteredSemanticLowererFactory.create() must return a "
                        "RegisteredSemanticLowerer."
                    )
                if any(
                    lowerer is previous
                    for previous in self._registered_semantic_lowerer_history
                ):
                    raise ValueError(
                        "RegisteredSemanticLowererFactory.create() must return a "
                        "fresh lowerer for every runtime assembly owned by this "
                        "registration."
                    )
                descriptor = self.call_catalog.discover(call_id)
                lowerer_type = type(lowerer)
                if getattr(lowerer_type, "call_id", None) != call_id:
                    raise ValueError(
                        "The live registered lowerer call_id must match its "
                        "factory declaration."
                    )
                if (
                    getattr(lowerer_type, "target_descriptor", None)
                    != descriptor.target_descriptor
                ):
                    raise ValueError(
                        f"Live lowerer {call_id!r} target_descriptor must exactly "
                        "match the registered catalog target."
                    )
                self._registered_semantic_lowerer_history.append(lowerer)
                created.append(lowerer)
        return tuple(created)

    def create_control_part_evidence_provider(
        self,
        *,
        simulation: object,
        robot: object,
        scene_registry: SceneRegistry,
        engine: AtomicActionEngine,
        scene_provider: SceneProvider,
    ) -> EffectEvidenceProvider | None:
        """Create the registration-owned live control-part evidence provider.

        Args:
            simulation: Simulation that owns the selected robot and sensors.
            robot: Exact robot selected by the environment factory.
            scene_registry: Exact live semantic scene registry.
            engine: Atomic-action engine assembled for the robot.
            scene_provider: Shared synchronized live scene provider.

        Returns:
            Fresh registered provider, or ``None`` when no factory is declared.
        """
        self.assert_unchanged()
        factory = self.control_part_evidence_factory
        if factory is None:
            return None
        if type(scene_registry) is not SceneRegistry:
            raise TypeError("scene_registry must be exactly SceneRegistry.")
        if not isinstance(engine, AtomicActionEngine):
            raise TypeError("engine must be an AtomicActionEngine.")
        if engine.robot is not robot:
            raise ValueError("engine and factory must reference the exact same robot.")
        if not isinstance(scene_provider, SceneProvider):
            raise TypeError("scene_provider must implement SceneProvider.")
        declaration = self.catalog.control_part_evidence_declaration
        if declaration is None:
            raise AssertionError(
                "A registered control-part evidence factory lost its declaration."
            )
        with self._control_part_evidence_provider_lock:
            provider = factory.create(
                simulation=simulation,
                robot=robot,
                scene_registry=scene_registry,
                engine=engine,
                scene_provider=scene_provider,
            )
            if not isinstance(provider, EffectEvidenceProvider):
                raise TypeError(
                    "control_part_evidence_factory.create() must return an "
                    "EffectEvidenceProvider."
                )
            if (provider.provider_id, provider.revision) != (
                declaration.provider_id,
                declaration.revision,
            ):
                raise ValueError(
                    "The live control-part evidence provider identity must match "
                    "its exact registration declaration."
                )
            if any(
                provider is previous
                for previous in self._control_part_evidence_provider_history
            ):
                raise ValueError(
                    "ControlPartEvidenceProviderFactory.create() must return a "
                    "fresh provider for every runtime assembly owned by this "
                    "registration."
                )
            self._control_part_evidence_provider_history.append(provider)
        return provider

    def create_parallel_safety_validator(
        self,
        *,
        simulation: object,
        robot: object,
        scene_registry: SceneRegistry,
        engine: AtomicActionEngine,
    ) -> ParallelCommandSafetyValidator | None:
        """Create and strictly validate the registration-owned live safety gate."""
        self.assert_unchanged()
        factory = self.parallel_safety_factory
        if factory is None:
            return None
        if type(scene_registry) is not SceneRegistry:
            raise TypeError("scene_registry must be exactly SceneRegistry.")
        if not isinstance(engine, AtomicActionEngine):
            raise TypeError("engine must be an AtomicActionEngine.")
        if engine.robot is not robot:
            raise ValueError("engine and factory must reference the exact same robot.")
        with self._parallel_safety_validator_lock:
            validator = factory.create(
                simulation=simulation,
                robot=robot,
                scene_registry=scene_registry,
                engine=engine,
            )
            if not isinstance(validator, ParallelCommandSafetyValidator):
                raise TypeError(
                    "parallel_safety_factory.create() must return a "
                    "ParallelCommandSafetyValidator."
                )
            if any(
                validator is previous
                for previous in self._parallel_safety_validator_history
            ):
                raise ValueError(
                    "ParallelCommandSafetyValidatorFactory.create() must return a "
                    "fresh validator for every runtime assembly owned by this "
                    "registration."
                )
            self._parallel_safety_validator_history.append(validator)
        return validator

    def validate_scene_registry(self, registry: SceneRegistry) -> None:
        """Validate a live registry against the registered scene declaration."""
        self.assert_unchanged()
        self.catalog.scene.validate_registry(registry)

    def validate_engine(self, engine: AtomicActionEngine) -> None:
        """Validate live atomic skills against this registration."""
        self.assert_unchanged()
        self.catalog.validate_engine(engine)

    def validate_bound_profile(
        self,
        bound_profile: BoundRobotSkillProfile,
    ) -> None:
        """Validate resolved semantic endpoints against this registration."""
        self.assert_unchanged()
        self.catalog.validate_bound_endpoint_extensions(bound_profile)

    def validate_robot_profile(
        self,
        profile: RobotSkillProfile,
    ) -> None:
        """Validate a live profile against its registered declaration."""
        self.assert_unchanged()
        if type(profile) is not RobotSkillProfile:
            raise TypeError("profile must be exactly RobotSkillProfile.")
        if _canonical_json(profile) != _canonical_json(self.catalog.robot_profile):
            raise IntegrationFingerprintMismatch(
                "Live robot skill profile differs from the registered declaration."
            )


__all__ = [
    "TaskProgramIntegrationCatalog",
    "IntegrationFingerprintMismatch",
    "SimulationTaskProgramRegistration",
]
