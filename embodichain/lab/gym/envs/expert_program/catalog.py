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
from dataclasses import dataclass, field, fields, is_dataclass, replace
from enum import Enum
import hashlib
import json
import math
from types import MappingProxyType
import torch

from embodichain.lab.gym.envs.settling import DynamicSettleMonitorCfg
from embodichain.lab.sim.atomic_actions import (
    Affordance,
    ArticulationOperationAffordance,
    AtomicActionEngine,
    SkillDescriptor,
)
from embodichain.lab.sim.atomic_actions.primitives import BUILTIN_ACTION_TYPES
from embodichain.lab.sim.skills import (
    ARTICULATION_OPERATION_AFFORDANCE_CAPABILITY,
    PLACE_IN_AFFORDANCE_CAPABILITY,
    PLACE_ON_AFFORDANCE_CAPABILITY,
    HandOverPoseProvider,
    OperateArticulation,
    Place,
    RelationTargetGrounder,
    RobotSkillProfile,
    SceneAffordanceRef,
    SceneArticulationRef,
    SceneEntityRef,
    SceneManifest,
    SceneObjectRef,
    SceneRegistry,
    SemanticCallCatalog,
    SemanticIntegrationManifest,
    SemanticValidationError,
    SkillPolicyPreset,
    builtin_semantic_call_catalog,
)

from .cfg import (
    ExpertProgramCfg,
    ExpertProgramIntegrationCfg,
    OperateArticulationCfg,
    PostPolicyCfg,
    RegisteredSemanticCallCfg,
    SemanticCallCfg,
    ValidatorCfg,
)
from .compiler import (
    CompiledProgram,
    ExpertProgramCompileError,
    ExpertProgramCompiler,
    ExpertProgramSceneResolver,
)
from .decoder import (
    ConfigPath,
    ExpertProgramValidationError,
    SceneReferenceRole,
)
from .simulation import SimulationRobotSkillProfileBinding, SimulationSceneBinding
from .simulation_policies import default_simulation_settle_presets

_CATALOG_FINGERPRINT_SCHEMA_VERSION = 1
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


def _validate_provider_declaration(provider: object, *, field_name: str) -> None:
    """Accept only frozen dataclass declarations or stateless providers."""
    dataclass_declaration = is_dataclass(provider)
    dataclass_field_names: set[str] = set()
    if dataclass_declaration:
        params = getattr(type(provider), "__dataclass_params__", None)
        if params is None or not params.frozen:
            raise TypeError(
                f"{field_name} stateful declarations must be frozen dataclasses "
                "so every configuration field enters the registration fingerprint."
            )
        dataclass_field_names.update(
            declaration_field.name for declaration_field in fields(provider)
        )

    state_names: set[str] = set()
    instance_state = getattr(provider, "__dict__", None)
    if isinstance(instance_state, Mapping):
        state_names.update(instance_state)
    for owner in type(provider).__mro__:
        declared_slots = getattr(owner, "__slots__", ())
        slots = (declared_slots,) if isinstance(declared_slots, str) else declared_slots
        for slot_name in slots:
            if slot_name in {"__dict__", "__weakref__"}:
                continue
            storage_name = (
                f"_{owner.__name__.lstrip('_')}{slot_name}"
                if slot_name.startswith("__") and not slot_name.endswith("__")
                else slot_name
            )
            if hasattr(provider, storage_name):
                state_names.add(storage_name)
    undeclared_state = (
        state_names.difference(dataclass_field_names)
        if dataclass_declaration
        else state_names
    )
    if undeclared_state:
        raise TypeError(
            f"{field_name} providers contain unfingerprinted state "
            f"{sorted(undeclared_state)}. Use a frozen dataclass declaration with "
            "every state field declared; non-dataclass providers must be stateless."
        )


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
        _validate_provider_declaration(
            grounder,
            field_name="relation_grounders",
        )
        key = _relation_grounder_key(grounder)
        if key in seen:
            raise ValueError(f"Duplicate relation grounder key {key!r}.")
        seen.add(key)
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
        _validate_provider_declaration(
            provider,
            field_name="handover_pose_providers",
        )
        provider_id = _handover_pose_provider_id(provider)
        if provider_id in seen:
            raise ValueError(f"Duplicate handover pose provider {provider_id!r}.")
        seen.add(provider_id)
    return tuple(values)


def _declared_articulation_operation_targets(
    scene_binding: SimulationSceneBinding,
) -> dict[str, frozenset[str]]:
    """Derive named operation-target IDs from the task-owned scene binding."""
    return {
        binding.entity_id: frozenset(binding.semantic_targets)
        for binding in scene_binding.articulation_operations
    }


def _snapshot_articulation_operation_targets(
    values: Mapping[str, frozenset[str]],
    *,
    scene: SceneManifest,
) -> Mapping[str, frozenset[str]]:
    """Own and cross-check provider-free named articulation targets."""
    if not isinstance(values, Mapping):
        raise TypeError("articulation_operation_targets must be a mapping.")
    normalized: dict[str, frozenset[str]] = {}
    for affordance_id, target_ids in values.items():
        _exact_identifier(
            affordance_id,
            field_name="articulation operation affordance IDs",
        )
        if type(target_ids) is not frozenset:
            raise TypeError(
                "articulation_operation_targets values must be exact frozensets."
            )
        for target_id in target_ids:
            _exact_identifier(
                target_id,
                field_name="articulation operation target IDs",
            )
        entry = scene.lookup(
            affordance_id,
            expected_type=SceneAffordanceRef,
            path=("articulation_operation_targets", affordance_id),
        )
        if entry.ref.entity_id != affordance_id:
            raise ValueError(
                "articulation_operation_targets keys must use canonical "
                "affordance IDs."
            )
        if (
            ARTICULATION_OPERATION_AFFORDANCE_CAPABILITY
            not in entry.affordance_capabilities
            or entry.affordance_payload_type is not ArticulationOperationAffordance
        ):
            raise TypeError(
                f"Scene affordance {affordance_id!r} is not an articulation "
                "operation affordance."
            )
        normalized[affordance_id] = frozenset(target_ids)

    declared_affordance_ids = {
        entry.ref.entity_id
        for entry in scene.entries
        if type(entry.ref) is SceneAffordanceRef
        and ARTICULATION_OPERATION_AFFORDANCE_CAPABILITY
        in entry.affordance_capabilities
    }
    if set(normalized) != declared_affordance_ids:
        raise ValueError(
            "articulation_operation_targets must cover every declared operation "
            f"affordance exactly; expected {sorted(declared_affordance_ids)}, got "
            f"{sorted(normalized)}."
        )
    return MappingProxyType(normalized)


class _SceneManifestProgramResolver:
    """Resolve compiler references from an immutable :class:`SceneManifest`."""

    def __init__(self, scene: SceneManifest) -> None:
        if type(scene) is not SceneManifest:
            raise TypeError("scene must be exactly SceneManifest.")
        self._scene = scene

    def resolve(
        self,
        reference: str,
        *,
        expected_types: tuple[type[SceneEntityRef], ...],
        path: ConfigPath,
    ) -> SceneEntityRef:
        """Resolve one reference without retaining a live registry."""
        if (
            type(expected_types) is not tuple
            or not expected_types
            or not all(
                isinstance(expected_type, type)
                and issubclass(expected_type, SceneEntityRef)
                for expected_type in expected_types
            )
        ):
            raise TypeError(
                "expected_types must be a non-empty tuple of scene-ref types."
            )
        try:
            resolved = self._scene.resolve(reference, path=path)
        except (KeyError, TypeError, ValueError) as exc:
            raise ExpertProgramCompileError(
                "unknown_scene_reference",
                path,
                str(exc),
            ) from exc
        if type(resolved) not in expected_types:
            raise ExpertProgramCompileError(
                "scene_reference_type_mismatch",
                path,
                f"Scene reference {reference!r} resolves to "
                f"{type(resolved).__name__}, expected one of "
                f"{tuple(value.__name__ for value in expected_types)}.",
            )
        return type(resolved)(resolved.entity_id)


@dataclass(frozen=True, slots=True)
class ExpertProgramIntegrationCatalog:
    """Provider-free integration directory owned by one task registration."""

    scene_registry_id: str
    robot_profile_id: str
    scene: SceneManifest
    robot_profile: RobotSkillProfile
    call_catalog: SemanticCallCatalog
    relation_grounder_keys: frozenset[tuple[str, type[Affordance], str]]
    articulation_operation_targets: Mapping[str, frozenset[str]]
    settle_preset_ids: frozenset[str]
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
        object.__setattr__(
            self,
            "articulation_operation_targets",
            _snapshot_articulation_operation_targets(
                self.articulation_operation_targets,
                scene=self.scene,
            ),
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
        if type(call) is OperateArticulationCfg and call.target is not None:
            self._validate_articulation_operation_target(
                articulation=call.articulation,
                handle=call.handle,
                target=call.target,
                path=path,
            )

    def _validate_articulation_operation_target(
        self,
        *,
        articulation: str | SceneArticulationRef,
        handle: str | SceneAffordanceRef | None,
        target: str,
        path: ConfigPath,
    ) -> None:
        """Resolve one operation affordance and validate its named target."""
        try:
            articulation_ref = self.scene.resolve(
                articulation,
                expected_type=SceneArticulationRef,
                path=(*path, "articulation"),
            )
        except SemanticValidationError as exc:
            raise ExpertProgramValidationError(
                exc.diagnostic.code,
                exc.diagnostic.path,
                exc.diagnostic.message,
            ) from exc
        try:
            affordance = self.scene.resolve_affordance(
                articulation_ref,
                capability=ARTICULATION_OPERATION_AFFORDANCE_CAPABILITY,
                explicit=handle,
                path=(*path, "handle"),
            )
        except SemanticValidationError as exc:
            raise ExpertProgramValidationError(
                exc.diagnostic.code,
                exc.diagnostic.path,
                exc.diagnostic.message,
            ) from exc
        target_ids = self.articulation_operation_targets.get(affordance.entity_id)
        if target_ids is None:
            raise ExpertProgramValidationError(
                "missing_articulation_operation_targets",
                (*path, "handle"),
                f"Operation affordance {affordance.entity_id!r} has no static "
                "named-target declaration.",
            )
        if target not in target_ids:
            raise ExpertProgramValidationError(
                "unknown_articulation_operation_target",
                (*path, "target"),
                f"Unknown target {target!r} for operation affordance "
                f"{affordance.entity_id!r}; available targets are "
                f"{sorted(target_ids)}.",
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
        resolver: ExpertProgramSceneResolver = _SceneManifestProgramResolver(self.scene)
        compiled = ExpertProgramCompiler(resolver).compile(program)
        manifest = SemanticIntegrationManifest(
            scene=self.scene,
            robot_profile=self.robot_profile,
            call_catalog=self.call_catalog,
            runtime_preset=program.integration.runtime_preset,
        )
        for segment in compiled.iter_segments():
            for call in segment.calls:
                if (
                    type(call.call) is OperateArticulation
                    and call.call.target is not None
                ):
                    self._validate_articulation_operation_target(
                        articulation=call.call.articulation,
                        handle=call.call.handle,
                        target=call.call.target,
                        path=call.source_path,
                    )
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


def _profile_with_control_dt(
    profile: RobotSkillProfile,
    *,
    control_dt: float,
) -> RobotSkillProfile:
    """Return the registration profile aligned to one Gym control cadence."""
    return replace(
        profile,
        presets={
            preset_id: SkillPolicyPreset(
                preset_id=preset.preset_id,
                schema_version=preset.schema_version,
                motion_policy=replace(preset.motion_policy, control_dt=control_dt),
                tracking_policy=preset.tracking_policy,
                recovery_policy=preset.recovery_policy,
                runner_cfg=preset.runner_cfg,
                effect_monitors=preset.effect_monitors,
            )
            for preset_id, preset in profile.presets.items()
        },
    )


def _registration_payload(
    *,
    scene_binding: SimulationSceneBinding,
    scene: SceneManifest,
    articulation_operation_targets: Mapping[str, frozenset[str]],
    robot_profile_binding: SimulationRobotSkillProfileBinding,
    robot_profile: RobotSkillProfile,
    call_catalog: SemanticCallCatalog,
    settle_presets: Mapping[str, DynamicSettleMonitorCfg],
    relation_grounder_keys: frozenset[tuple[str, type[Affordance], str]],
    relation_grounders: tuple[RelationTargetGrounder, ...],
    handover_pose_providers: tuple[HandOverPoseProvider, ...],
) -> dict[str, object]:
    """Build the versioned canonical fingerprint payload."""
    return {
        "schema_version": _CATALOG_FINGERPRINT_SCHEMA_VERSION,
        "scene_binding": scene_binding,
        "scene_manifest": scene.entries,
        "articulation_operation_targets": articulation_operation_targets,
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
    catalog: ExpertProgramIntegrationCatalog = field(init=False)

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
        settle_presets = _snapshot_settle_presets(self.settle_presets)
        object.__setattr__(self, "settle_presets", settle_presets)
        relation_grounders = _snapshot_relation_grounders(self.relation_grounders)
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
        articulation_operation_targets = _declared_articulation_operation_targets(
            self.scene_binding
        )
        profile = self.robot_profile_binding.declare()
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
                articulation_operation_targets=articulation_operation_targets,
                robot_profile_binding=self.robot_profile_binding,
                robot_profile=profile,
                call_catalog=self.call_catalog,
                settle_presets=settle_presets,
                relation_grounder_keys=relation_grounder_keys,
                relation_grounders=relation_grounders,
                handover_pose_providers=handover_pose_providers,
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
                articulation_operation_targets=articulation_operation_targets,
                settle_preset_ids=frozenset(settle_presets),
                fingerprint=fingerprint,
                _required_skills=required_skills,
            ),
        )

    @property
    def fingerprint(self) -> str:
        """Return the canonical registration fingerprint."""
        return self.catalog.fingerprint

    def assert_unchanged(self) -> None:
        """Reject nested declaration drift before live component creation."""
        scene = self.scene_binding.declare()
        articulation_operation_targets = _declared_articulation_operation_targets(
            self.scene_binding
        )
        profile = self.robot_profile_binding.declare()
        try:
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
                    articulation_operation_targets=(articulation_operation_targets),
                    robot_profile_binding=self.robot_profile_binding,
                    robot_profile=profile,
                    call_catalog=self.call_catalog,
                    settle_presets=self.settle_presets,
                    relation_grounder_keys=relation_grounder_keys,
                    relation_grounders=relation_grounders,
                    handover_pose_providers=handover_pose_providers,
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

    def validate_scene_registry(self, registry: SceneRegistry) -> None:
        """Validate a live registry against the registered scene declaration."""
        self.assert_unchanged()
        self.catalog.scene.validate_registry(registry)

    def validate_robot_profile(
        self,
        profile: RobotSkillProfile,
        *,
        step_dt: float,
    ) -> None:
        """Validate a cadence-aligned live profile against its declaration."""
        self.assert_unchanged()
        if type(profile) is not RobotSkillProfile:
            raise TypeError("profile must be exactly RobotSkillProfile.")
        expected = _profile_with_control_dt(
            self.catalog.robot_profile,
            control_dt=step_dt,
        )
        if _canonical_json(profile) != _canonical_json(expected):
            raise IntegrationFingerprintMismatch(
                "Live robot skill profile differs from the registered declaration."
            )


__all__ = [
    "ExpertProgramIntegrationCatalog",
    "IntegrationFingerprintMismatch",
    "SimulationExpertProgramRegistration",
]
