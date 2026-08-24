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

"""Immutable task-registration catalog for declarative Expert Programs."""

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
import torch

from embodichain.lab.gym.envs.settling import DynamicSettleMonitorCfg
from embodichain.lab.sim.atomic_actions import (
    Affordance,
    AtomicActionEngine,
    EndpointTrackingFeedbackAddress,
    GRASP_CAPABILITY,
    JOINT_POSITION_CHANNEL,
    SkillDescriptor,
)
from embodichain.lab.sim.atomic_actions.primitives import BUILTIN_ACTION_TYPES
from embodichain.lab.sim.atomic_actions.tracking import (
    FeedbackTerminalAcceptance,
    TrackingRuntime,
)
from embodichain.lab.sim.skills import (
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
    ContainerRelationTargetGrounder,
    HandOverPoseProvider,
    Place,
    RelationTargetGrounder,
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
    SupportSurfaceRelationTargetGrounder,
    builtin_semantic_call_catalog,
)
from embodichain.lab.sim.skills.effects import (
    COMPOSITE_EFFECT_MONITOR_ID,
    COMPOSITE_EFFECT_MONITOR_REVISION,
    CompositeEffectMonitorFactory,
    EffectMonitorRegistry,
)
from embodichain.lab.sim.skills.parallel_runtime import (
    ParallelCommandSafetyValidator,
)

from .cfg import (
    ExpertProgramCfg,
    ExpertProgramIntegrationCfg,
    PostPolicyCfg,
    RegisteredSemanticCallCfg,
    SemanticCallCfg,
    ValidatorCfg,
)
from .compiler import (
    CompiledProgram,
    ExpertProgramCompiler,
)
from .decoder import (
    ConfigPath,
    ExpertProgramValidationError,
    SceneReferenceRole,
)
from .bridge import RuntimeTransportActionEncoder
from .extensions import (
    EndpointAdapterDeclaration,
    ParallelCommandSafetyValidatorFactory,
    ParallelSafetyDeclaration,
    RuntimeTransportDeclaration,
    StandardExtensionDeclarations,
    build_standard_extension_declarations,
    validate_immutable_extension_declaration,
)
from .simulation import SimulationRobotSkillProfileBinding, SimulationSceneBinding
from .simulation_policies import default_simulation_settle_presets

_CATALOG_FINGERPRINT_SCHEMA_VERSION = 2
_POST_POLICY_KINDS = frozenset({"wait_stable"})
_VALIDATOR_KINDS = frozenset({"object_near_target"})


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
        metadata = {
            data_field.name: _canonical_value(getattr(value, data_field.name))
            for data_field in fields(value)
        }
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
    values: Mapping[str, DynamicSettleMonitorCfg],
) -> Mapping[str, DynamicSettleMonitorCfg]:
    """Own one strict named settle-preset table."""
    if not isinstance(values, Mapping) or not values:
        raise ValueError("settle_presets must be a non-empty mapping.")
    normalized: dict[str, DynamicSettleMonitorCfg] = {}
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
        if not isinstance(preset, DynamicSettleMonitorCfg):
            raise TypeError(
                "settle_presets values must be DynamicSettleMonitorCfg values."
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


def _validate_standard_call_catalog(call_catalog: SemanticCallCatalog) -> None:
    """Reject semantic lowerer extensions from the standard registration path."""
    builtins = builtin_semantic_call_catalog().descriptors
    for descriptor in call_catalog.descriptors.values():
        if descriptor.spec_type is RegisteredSemanticCall:
            raise ValueError(
                f"Registered semantic call {descriptor.call_id!r} is not "
                "supported by the standard simulation registration; only "
                "curated semantic calls may be registered."
            )
        expected = builtins.get(descriptor.call_id)
        if expected != descriptor:
            raise ValueError(
                f"Semantic call {descriptor.call_id!r} does not match its exact "
                "curated descriptor."
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
class ExpertProgramIntegrationCatalog:
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
    fingerprint: str
    _required_skills: Mapping[str, SkillDescriptor] = field(
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
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
        if self.robot_profile.profile_id != self.robot_profile_id:
            raise ValueError("robot_profile_id must match robot_profile.profile_id.")
        preset_ids = frozenset(self.settle_preset_ids)
        if not preset_ids:
            raise ValueError("settle_preset_ids must not be empty.")
        object.__setattr__(self, "settle_preset_ids", preset_ids)
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
        integration: ExpertProgramIntegrationCfg,
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
        """Validate semantic-call catalog and payload revision references."""
        call_id = call.call_id if type(call) is RegisteredSemanticCallCfg else call.kind
        descriptor = self.call_catalog.discover(call_id)
        if type(call) is RegisteredSemanticCallCfg and (
            call.schema_version != descriptor.schema_version
        ):
            raise ValueError(
                f"Semantic call {call_id!r} requires schema_version "
                f"{descriptor.schema_version}, got {call.schema_version}."
            )

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
            raise ExpertProgramValidationError(
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
            raise ExpertProgramValidationError(
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
        del path
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

    def preflight(self, program: ExpertProgramCfg) -> CompiledProgram:
        """Compile and statically link every expanded semantic call."""
        self.validate_integration(program.integration, path=("integration",))
        compiled = ExpertProgramCompiler(self.scene).compile(program)
        manifest = SemanticIntegrationManifest(
            scene=self.scene,
            robot_profile=self.robot_profile,
            call_catalog=self.call_catalog,
            runtime_preset=program.integration.runtime_preset,
        )
        for segment in compiled.iter_segments():
            if (
                segment.parallel_block is not None
                and self.parallel_safety_declaration is None
            ):
                raise ExpertProgramValidationError(
                    "parallel_safety_factory_not_registered",
                    segment.parallel_block.source_path,
                    "Parallel execution requires a task-registration-owned "
                    "physical safety-validator factory.",
                )
            for call in segment.calls:
                linked = manifest.link_call(call.call, path=call.source_path)
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
        return compiled

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
        bound_profile = engine.skill_profile
        if type(bound_profile) is not BoundRobotSkillProfile:
            raise IntegrationFingerprintMismatch(
                "The standard live engine must own one exact bound robot profile."
            )
        self.validate_bound_endpoint_extensions(bound_profile)

    def validate_bound_endpoint_extensions(
        self,
        bound_profile: BoundRobotSkillProfile,
    ) -> None:
        """Match every live resolved endpoint to its fingerprinted declaration."""
        if type(bound_profile) is not BoundRobotSkillProfile:
            raise TypeError("bound_profile must be exactly BoundRobotSkillProfile.")
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
    settle_presets: Mapping[str, DynamicSettleMonitorCfg],
    relation_grounder_keys: frozenset[tuple[str, type[Affordance], str]],
    relation_grounders: tuple[RelationTargetGrounder, ...],
    handover_pose_providers: tuple[HandOverPoseProvider, ...],
    extensions: StandardExtensionDeclarations,
    endpoint_adapters: tuple[ResourceEndpointAdapter, ...],
    runtime_transports: tuple[RuntimeTransportActionEncoder, ...],
    parallel_safety_factory: ParallelCommandSafetyValidatorFactory | None,
) -> dict[str, object]:
    """Build the versioned canonical fingerprint payload."""
    return {
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
        "post_policy_kinds": _POST_POLICY_KINDS,
        "settle_presets": settle_presets,
        "validator_kinds": _VALIDATOR_KINDS,
    }


@dataclass(frozen=True, slots=True)
class SimulationExpertProgramRegistration:
    """Exact immutable task-owned simulation integration registration."""

    scene_binding: SimulationSceneBinding
    robot_profile_binding: SimulationRobotSkillProfileBinding
    call_catalog: SemanticCallCatalog = field(
        default_factory=builtin_semantic_call_catalog
    )
    settle_presets: Mapping[str, DynamicSettleMonitorCfg] = field(
        default_factory=default_simulation_settle_presets
    )
    relation_grounders: tuple[RelationTargetGrounder, ...] = ()
    handover_pose_providers: tuple[HandOverPoseProvider, ...] = ()
    endpoint_adapters: tuple[ResourceEndpointAdapter, ...] = ()
    runtime_transports: tuple[RuntimeTransportActionEncoder, ...] = ()
    parallel_safety_factory: ParallelCommandSafetyValidatorFactory | None = None
    catalog: ExpertProgramIntegrationCatalog = field(init=False)
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

    def __post_init__(self) -> None:
        if type(self.scene_binding) is not SimulationSceneBinding:
            raise TypeError("scene_binding must be exactly SimulationSceneBinding.")
        if type(self.robot_profile_binding) is not SimulationRobotSkillProfileBinding:
            raise TypeError(
                "robot_profile_binding must be exactly "
                "SimulationRobotSkillProfileBinding."
            )
        if type(self.call_catalog) is not SemanticCallCatalog:
            raise TypeError("call_catalog must be exactly SemanticCallCatalog.")
        _validate_standard_call_catalog(self.call_catalog)
        settle_presets = _snapshot_settle_presets(self.settle_presets)
        object.__setattr__(self, "settle_presets", settle_presets)
        configured_relation_grounders = _snapshot_relation_grounders(
            self.relation_grounders
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

        scene = self.scene_binding.declare()
        profile = self.robot_profile_binding.declare()
        _validate_standard_effect_monitors(profile)
        _validate_standard_tracking_metrics(profile)
        extensions = build_standard_extension_declarations(
            profile=profile,
            endpoint_adapters=self.endpoint_adapters,
            runtime_transports=self.runtime_transports,
            parallel_safety_factory=self.parallel_safety_factory,
        )
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
            )
        )
        object.__setattr__(
            self,
            "catalog",
            ExpertProgramIntegrationCatalog(
                scene_registry_id=self.scene_binding.registry_id,
                robot_profile_id=self.robot_profile_binding.profile_id,
                scene=scene,
                robot_profile=profile,
                call_catalog=self.call_catalog,
                relation_grounder_keys=relation_grounder_keys,
                settle_preset_ids=frozenset(settle_presets),
                endpoint_adapter_declarations=extensions.endpoint_adapters,
                runtime_transport_declarations=extensions.runtime_transports,
                parallel_safety_declaration=extensions.parallel_safety,
                fingerprint=fingerprint,
                _required_skills=required_skills,
            ),
        )
        object.__setattr__(self, "_parallel_safety_validator_history", [])
        object.__setattr__(self, "_parallel_safety_validator_lock", Lock())

    @property
    def fingerprint(self) -> str:
        """Return the canonical registration fingerprint."""
        return self.catalog.fingerprint

    def assert_unchanged(self) -> None:
        """Reject nested declaration drift before live component creation."""
        scene = self.scene_binding.declare()
        profile = self.robot_profile_binding.declare()
        try:
            _validate_standard_call_catalog(self.call_catalog)
            _validate_standard_effect_monitors(profile)
            _validate_standard_tracking_metrics(profile)
            extensions = build_standard_extension_declarations(
                profile=profile,
                endpoint_adapters=self.endpoint_adapters,
                runtime_transports=self.runtime_transports,
                parallel_safety_factory=self.parallel_safety_factory,
            )
            relation_grounders = _snapshot_relation_grounders(self.relation_grounders)
            relation_grounder_keys = frozenset(
                _relation_grounder_key(grounder) for grounder in relation_grounders
            )
            handover_pose_providers = _snapshot_handover_pose_providers(
                self.handover_pose_providers
            )
            current = _digest(
                _registration_payload(
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
                )
            )
        except (TypeError, ValueError) as exc:
            raise IntegrationFingerprintMismatch(
                "Expert Program integration provider declaration changed after "
                "task registration."
            ) from exc
        if current != self.fingerprint:
            raise IntegrationFingerprintMismatch(
                "Expert Program integration declaration changed after task "
                "registration."
            )

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

    def create_parallel_safety_validator(
        self,
        *,
        simulation: object,
        robot: object,
    ) -> ParallelCommandSafetyValidator | None:
        """Create and strictly validate the registration-owned live safety gate."""
        self.assert_unchanged()
        factory = self.parallel_safety_factory
        if factory is None:
            return None
        with self._parallel_safety_validator_lock:
            validator = factory.create(simulation=simulation, robot=robot)
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
        """Validate live skills and resolved endpoints against this registration."""
        self.assert_unchanged()
        self.catalog.validate_engine(engine)

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
    "ExpertProgramIntegrationCatalog",
    "IntegrationFingerprintMismatch",
    "SimulationExpertProgramRegistration",
]
