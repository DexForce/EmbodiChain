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

"""Two-phase static and live semantic integration validation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import TypeVar

from embodichain.lab.sim.atomic_actions import (
    Affordance,
    AtomicActionEngine,
    DynamicCollisionMode,
    DisjointResourceSlots,
    DisjointSlotEndpoints,
    SkillResourceSlot,
)

from .calls import (
    HandOver,
    OperateArticulation,
    Pick,
    Place,
    RegisteredSemanticCall,
    SemanticCallCatalog,
    SemanticCallDescriptor,
    SemanticCallSpec,
)
from .profiles import (
    BoundRobotSkillProfile,
    ControlPartEndpoint,
    ResolvedSkillBinding,
    ResourceEndpoint,
    ResourceEndpointAdapter,
    RobotResource,
    RobotSkillProfile,
    SkillPolicyPreset,
)
from .scene import (
    ARTICULATION_OPERATION_AFFORDANCE_CAPABILITY,
    GRASP_AFFORDANCE_CAPABILITY,
    PLACE_IN_AFFORDANCE_CAPABILITY,
    PLACE_ON_AFFORDANCE_CAPABILITY,
    SceneAffordanceRef,
    SceneArticulationRef,
    SceneCollisionRole,
    SceneDynamics,
    SceneEntityMetadata,
    SceneEntityRef,
    SceneEntityRegistration,
    SceneLinkRef,
    SceneObjectRef,
    SceneRegistry,
)

PathPart = str | int
RefT = TypeVar("RefT", bound=SceneEntityRef)


def _validate_identifier(value: str, *, field_name: str) -> str:
    """Return one exact, non-empty identifier."""
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(
            f"{field_name} must be a non-empty string without outer whitespace."
        )
    return value


def _render_path(path: tuple[PathPart, ...]) -> str:
    """Render tuple path components in configuration notation."""
    output = ""
    for part in path:
        if isinstance(part, int):
            output += f"[{part}]"
        elif not output:
            output = part
        else:
            output += f".{part}"
    return output or "<root>"


@dataclass(frozen=True, slots=True)
class SemanticDiagnostic:
    """Structured deterministic semantic-integration diagnostic.

    Args:
        code: Stable machine-readable failure code.
        path: Complete configuration or program path.
        message: Human-readable explanation.
        candidates: Canonical candidate IDs, sorted when applicable.
    """

    code: str
    path: tuple[PathPart, ...]
    message: str
    candidates: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_identifier(self.code, field_name="SemanticDiagnostic.code")
        if isinstance(self.path, (str, bytes)):
            raise TypeError("SemanticDiagnostic.path must be a tuple of components.")
        path = tuple(self.path)
        if not all(
            (isinstance(part, str) and part)
            or (isinstance(part, int) and not isinstance(part, bool))
            for part in path
        ):
            raise ValueError("SemanticDiagnostic.path contains an invalid component.")
        object.__setattr__(self, "path", path)
        if not isinstance(self.message, str) or not self.message:
            raise ValueError("SemanticDiagnostic.message must be non-empty.")
        candidates = tuple(self.candidates)
        if not all(isinstance(candidate, str) for candidate in candidates):
            raise TypeError("SemanticDiagnostic.candidates must contain strings.")
        object.__setattr__(self, "candidates", tuple(sorted(candidates)))

    @property
    def rendered_path(self) -> str:
        """Return the path in dotted/indexed notation."""
        return _render_path(self.path)


class SemanticValidationError(ValueError):
    """Raise one structured error at a static or live integration boundary."""

    def __init__(self, diagnostic: SemanticDiagnostic) -> None:
        if not isinstance(diagnostic, SemanticDiagnostic):
            raise TypeError("diagnostic must be a SemanticDiagnostic.")
        self.diagnostic = diagnostic
        super().__init__(f"{diagnostic.rendered_path}: {diagnostic.message}")


@dataclass(frozen=True, slots=True)
class SceneEntityManifest:
    """Provider-free static scene-entity declaration.

    Args:
        ref: Canonical typed entity reference.
        aliases: Boundary aliases accepted during static linking.
        parent: Canonical parent for links and affordances.
        native_name: Backend-local child name.
        dynamics: Physical mobility classification.
        collision_role: Planner collision classification.
        semantic_type: Optional application classification.
        affordance_capabilities: Semantic operations supplied by an affordance.
        default_affordances: Capability-scoped direct-child defaults.
        affordance_payload_type: Exact registered affordance payload type.
        affordance_revision: Stable payload revision or fingerprint.
        relative_pose: Flattened parent-relative homogeneous transform.
    """

    ref: SceneEntityRef
    aliases: tuple[str, ...] = ()
    parent: SceneEntityRef | None = None
    native_name: str | None = None
    dynamics: SceneDynamics = SceneDynamics.UNKNOWN
    collision_role: SceneCollisionRole = SceneCollisionRole.NONE
    semantic_type: str | None = None
    affordance_capabilities: frozenset[str] = frozenset()
    default_affordances: Mapping[str, SceneAffordanceRef] = field(default_factory=dict)
    affordance_payload_type: type[Affordance] | None = None
    affordance_revision: str | None = None
    relative_pose: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.ref, SceneEntityRef):
            raise TypeError("SceneEntityManifest.ref must be a SceneEntityRef.")
        if isinstance(self.aliases, (str, bytes)):
            raise TypeError("aliases must be an iterable of identifiers.")
        aliases = tuple(self.aliases)
        for alias in aliases:
            _validate_identifier(alias, field_name="scene aliases")
        aliases = tuple(alias for alias in aliases if alias != self.ref.entity_id)
        if len(set(aliases)) != len(aliases):
            raise ValueError("SceneEntityManifest.aliases must be unique.")
        object.__setattr__(self, "aliases", aliases)
        if self.parent is not None and not isinstance(self.parent, SceneEntityRef):
            raise TypeError("parent must be a SceneEntityRef or None.")
        if self.semantic_type is not None:
            _validate_identifier(self.semantic_type, field_name="semantic_type")
        if isinstance(self.affordance_capabilities, (str, bytes)):
            raise TypeError("affordance_capabilities must be an iterable.")
        capabilities = frozenset(self.affordance_capabilities)
        for capability in capabilities:
            _validate_identifier(capability, field_name="affordance capabilities")
        object.__setattr__(self, "affordance_capabilities", capabilities)
        if not isinstance(self.default_affordances, Mapping):
            raise TypeError("default_affordances must be a mapping.")
        defaults: dict[str, SceneAffordanceRef] = {}
        for capability, affordance in self.default_affordances.items():
            _validate_identifier(capability, field_name="default capabilities")
            if type(affordance) is not SceneAffordanceRef:
                raise TypeError(
                    "default_affordances values must be SceneAffordanceRef values."
                )
            defaults[capability] = affordance
        object.__setattr__(
            self,
            "default_affordances",
            MappingProxyType(defaults),
        )
        metadata = SceneEntityMetadata(
            ref=self.ref,
            aliases=self.aliases,
            parent=self.parent,
            native_name=self.native_name,
            dynamics=self.dynamics,
            collision_role=self.collision_role,
            semantic_type=self.semantic_type,
            affordance_capabilities=self.affordance_capabilities,
            default_affordances=self.default_affordances,
            affordance_payload_type=self.affordance_payload_type,
            affordance_revision=self.affordance_revision,
            relative_pose=self.relative_pose,
        )
        object.__setattr__(self, "aliases", metadata.aliases)
        object.__setattr__(self, "default_affordances", metadata.default_affordances)
        object.__setattr__(self, "relative_pose", metadata.relative_pose)

    @classmethod
    def from_registration(
        cls,
        registration: SceneEntityRegistration,
    ) -> SceneEntityManifest:
        """Project live registration metadata without reading providers."""
        if not isinstance(registration, SceneEntityRegistration):
            raise TypeError("registration must be a SceneEntityRegistration.")
        return cls.from_metadata(SceneEntityMetadata.from_registration(registration))

    @classmethod
    def from_metadata(cls, metadata: SceneEntityMetadata) -> SceneEntityManifest:
        """Copy one provider-free registry metadata value."""
        if not isinstance(metadata, SceneEntityMetadata):
            raise TypeError("metadata must be a SceneEntityMetadata.")
        return cls(
            ref=metadata.ref,
            aliases=metadata.aliases,
            parent=metadata.parent,
            native_name=metadata.native_name,
            dynamics=metadata.dynamics,
            collision_role=metadata.collision_role,
            semantic_type=metadata.semantic_type,
            affordance_capabilities=metadata.affordance_capabilities,
            default_affordances=metadata.default_affordances,
            affordance_payload_type=metadata.affordance_payload_type,
            affordance_revision=metadata.affordance_revision,
            relative_pose=metadata.relative_pose,
        )


@dataclass(frozen=True, slots=True, init=False)
class SceneManifest:
    """Immutable provider-free scene catalog used before simulation starts."""

    _entries: tuple[SceneEntityManifest, ...]
    _by_id: Mapping[str, SceneEntityManifest]
    _aliases: Mapping[str, str]
    _affordances: Mapping[tuple[str, str], tuple[SceneAffordanceRef, ...]]

    def __init__(self, entries: Iterable[SceneEntityManifest] = ()) -> None:
        if isinstance(entries, (str, bytes)):
            raise TypeError("entries must be an iterable of scene manifests.")
        try:
            supplied = tuple(entries)
        except TypeError as exc:
            raise TypeError("entries must be an iterable of scene manifests.") from exc
        if not all(type(entry) is SceneEntityManifest for entry in supplied):
            raise TypeError("entries must contain exact SceneEntityManifest values.")
        by_id: dict[str, SceneEntityManifest] = {}
        for entry in supplied:
            if entry.ref.entity_id in by_id:
                raise ValueError(
                    f"Duplicate scene manifest ID {entry.ref.entity_id!r}."
                )
            by_id[entry.ref.entity_id] = entry
        aliases: dict[str, str] = {}
        for entry in supplied:
            for alias in entry.aliases:
                if alias in by_id:
                    raise ValueError(
                        f"Scene manifest alias {alias!r} collides with a canonical ID."
                    )
                previous = aliases.get(alias)
                if previous is not None:
                    raise ValueError(
                        f"Scene manifest alias {alias!r} is ambiguous between "
                        f"{previous!r} and {entry.ref.entity_id!r}."
                    )
                aliases[alias] = entry.ref.entity_id
        affordances: dict[tuple[str, str], list[SceneAffordanceRef]] = {}
        native_members: dict[tuple[type[SceneEntityRef], str, str], str] = {}
        for entry in supplied:
            if entry.parent is not None:
                parent_entry = by_id.get(entry.parent.entity_id)
                if parent_entry is None:
                    raise ValueError(
                        f"Scene manifest entity {entry.ref.entity_id!r} references "
                        f"unknown parent {entry.parent.entity_id!r}."
                    )
                if type(parent_entry.ref) is not type(entry.parent):
                    raise TypeError(
                        f"Scene manifest parent {entry.parent.entity_id!r} has "
                        "the wrong reference type."
                    )
                if entry.native_name is not None and isinstance(
                    entry.ref, (SceneLinkRef, SceneAffordanceRef)
                ):
                    native_key = (
                        type(entry.ref),
                        entry.parent.entity_id,
                        entry.native_name,
                    )
                    previous = native_members.get(native_key)
                    if previous is not None:
                        raise ValueError(
                            f"Scene manifest parent {entry.parent.entity_id!r} "
                            f"and native_name {entry.native_name!r} are already "
                            f"registered as {previous!r}."
                        )
                    native_members[native_key] = entry.ref.entity_id
            if isinstance(entry.ref, SceneAffordanceRef):
                if entry.parent is None:
                    raise ValueError(
                        f"Affordance {entry.ref.entity_id!r} requires a parent."
                    )
                for capability in entry.affordance_capabilities:
                    affordances.setdefault(
                        (entry.parent.entity_id, capability), []
                    ).append(entry.ref)
            elif entry.affordance_capabilities:
                raise ValueError(
                    "Only SceneAffordanceRef entries may declare "
                    "affordance_capabilities."
                )
        for entry in supplied:
            if isinstance(entry.ref, SceneAffordanceRef) and entry.default_affordances:
                raise ValueError(
                    "Scene affordance entries cannot declare default_affordances."
                )
            for capability, default in entry.default_affordances.items():
                default_entry = by_id.get(default.entity_id)
                if default_entry is None or not isinstance(
                    default_entry.ref, SceneAffordanceRef
                ):
                    raise ValueError(
                        f"Default affordance {default.entity_id!r} is not a "
                        "registered affordance entry."
                    )
                if default_entry.parent != entry.ref:
                    raise ValueError(
                        f"Default affordance {default.entity_id!r} is not a direct "
                        f"child of {entry.ref.entity_id!r}."
                    )
                if capability not in default_entry.affordance_capabilities:
                    raise ValueError(
                        f"Default affordance {default.entity_id!r} does not support "
                        f"capability {capability!r}."
                    )
        object.__setattr__(self, "_entries", supplied)
        object.__setattr__(self, "_by_id", MappingProxyType(by_id))
        object.__setattr__(self, "_aliases", MappingProxyType(aliases))
        object.__setattr__(
            self,
            "_affordances",
            MappingProxyType(
                {
                    key: tuple(sorted(refs, key=lambda ref: ref.entity_id))
                    for key, refs in affordances.items()
                }
            ),
        )

    @property
    def entries(self) -> tuple[SceneEntityManifest, ...]:
        """Return immutable provider-free entries in declaration order."""
        return self._entries

    @classmethod
    def from_registry(cls, registry: SceneRegistry) -> SceneManifest:
        """Project a live registry without observing any dynamic provider."""
        if not isinstance(registry, SceneRegistry):
            raise TypeError("registry must be a SceneRegistry.")
        return cls(
            SceneEntityManifest.from_metadata(metadata)
            for metadata in registry.entity_metadata
        )

    def resolve(
        self,
        identifier: str | SceneEntityRef,
        *,
        expected_type: type[RefT] = SceneEntityRef,
        path: tuple[PathPart, ...] = (),
    ) -> RefT:
        """Resolve one canonical or alias reference with pathful diagnostics."""
        if isinstance(identifier, SceneEntityRef):
            candidate_id = identifier.entity_id
            supplied_type: type[SceneEntityRef] | None = type(identifier)
        elif isinstance(identifier, str):
            _validate_identifier(identifier, field_name="scene identifier")
            candidate_id = self._aliases.get(identifier, identifier)
            supplied_type = None
        else:
            raise SemanticValidationError(
                SemanticDiagnostic(
                    "invalid_entity_reference",
                    path,
                    "Expected a scene identifier or typed scene reference.",
                )
            )
        entry = self._by_id.get(candidate_id)
        if entry is None:
            raise SemanticValidationError(
                SemanticDiagnostic(
                    "unknown_entity",
                    path,
                    f"Unknown scene entity {candidate_id!r}.",
                    tuple(self._by_id),
                )
            )
        if supplied_type is not None and supplied_type is not type(entry.ref):
            raise SemanticValidationError(
                SemanticDiagnostic(
                    "entity_type_mismatch",
                    path,
                    f"Scene entity {candidate_id!r} is "
                    f"{type(entry.ref).__name__}, not {supplied_type.__name__}.",
                )
            )
        if not isinstance(entry.ref, expected_type):
            raise SemanticValidationError(
                SemanticDiagnostic(
                    "entity_type_mismatch",
                    path,
                    f"Scene entity {candidate_id!r} is "
                    f"{type(entry.ref).__name__}, not {expected_type.__name__}.",
                )
            )
        return entry.ref  # type: ignore[return-value]

    def lookup(
        self,
        identifier: str | SceneEntityRef,
        *,
        expected_type: type[RefT] = SceneEntityRef,
        path: tuple[PathPart, ...] = (),
    ) -> SceneEntityManifest:
        """Return one static entry after canonical typed resolution."""
        ref = self.resolve(identifier, expected_type=expected_type, path=path)
        return self._by_id[ref.entity_id]

    def resolve_affordance(
        self,
        parent: str | SceneEntityRef,
        *,
        capability: str,
        explicit: str | SceneAffordanceRef | None = None,
        path: tuple[PathPart, ...] = (),
    ) -> SceneAffordanceRef:
        """Resolve one affordance using the same strict rule as SceneRegistry."""
        parent_ref = self.resolve(parent, path=path)
        _validate_identifier(capability, field_name="affordance capability")
        candidates = self._affordances.get((parent_ref.entity_id, capability), ())
        if explicit is not None:
            selected = self.resolve(
                explicit,
                expected_type=SceneAffordanceRef,
                path=path,
            )
            entry = self._by_id[selected.entity_id]
            if entry.parent != parent_ref:
                raise SemanticValidationError(
                    SemanticDiagnostic(
                        "affordance_parent_mismatch",
                        path,
                        f"Affordance {selected.entity_id!r} is not a direct child "
                        f"of {parent_ref.entity_id!r}.",
                        tuple(candidate.entity_id for candidate in candidates),
                    )
                )
            if capability not in entry.affordance_capabilities:
                raise SemanticValidationError(
                    SemanticDiagnostic(
                        "unsupported_affordance",
                        path,
                        f"Affordance {selected.entity_id!r} does not support "
                        f"{capability!r}.",
                        tuple(candidate.entity_id for candidate in candidates),
                    )
                )
            return selected
        if not candidates:
            raise SemanticValidationError(
                SemanticDiagnostic(
                    "missing_affordance",
                    path,
                    f"Scene entity {parent_ref.entity_id!r} has no affordance for "
                    f"{capability!r}.",
                )
            )
        if len(candidates) == 1:
            return candidates[0]
        parent_entry = self._by_id[parent_ref.entity_id]
        default = parent_entry.default_affordances.get(capability)
        if default is not None:
            return default
        raise SemanticValidationError(
            SemanticDiagnostic(
                "ambiguous_affordance",
                path,
                f"Multiple affordances support {capability!r}; configure a "
                "scoped default or select one explicitly.",
                tuple(candidate.entity_id for candidate in candidates),
            )
        )

    def validate_registry(
        self,
        registry: SceneRegistry,
        *,
        path: tuple[PathPart, ...] = ("integration", "scene_registry"),
    ) -> None:
        """Require a live registry to match this provider-free declaration."""
        try:
            live = SceneManifest.from_registry(registry)
        except Exception as exc:  # noqa: BLE001 - normalize integration failures
            raise SemanticValidationError(
                SemanticDiagnostic(
                    "invalid_scene_registry",
                    path,
                    f"Could not project the live scene registry: {exc}",
                )
            ) from exc
        static_ids = set(self._by_id)
        live_ids = set(live._by_id)
        if static_ids != live_ids:
            raise SemanticValidationError(
                SemanticDiagnostic(
                    "scene_manifest_mismatch",
                    path,
                    "Live scene IDs differ from the static manifest; "
                    f"missing={sorted(static_ids - live_ids)}, "
                    f"extra={sorted(live_ids - static_ids)}.",
                )
            )
        for entity_id in sorted(static_ids):
            if self._by_id[entity_id] != live._by_id[entity_id]:
                raise SemanticValidationError(
                    SemanticDiagnostic(
                        "scene_manifest_mismatch",
                        (*path, entity_id),
                        "Live scene metadata differs from the static manifest.",
                    )
                )


@dataclass(frozen=True, slots=True)
class LinkedSemanticCall:
    """Provider-free static link result for one semantic call."""

    call: SemanticCallSpec
    descriptor: SemanticCallDescriptor
    preset_id: str
    affordances: Mapping[str, SceneAffordanceRef] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if type(self.call) not in (
            Pick,
            Place,
            HandOver,
            OperateArticulation,
            RegisteredSemanticCall,
        ):
            raise TypeError("call must be an exact supported semantic call value.")
        if type(self.descriptor) is not SemanticCallDescriptor:
            raise TypeError("descriptor must be exactly SemanticCallDescriptor.")
        if type(self.call) is not self.descriptor.spec_type or (
            self.call.semantic_id != self.descriptor.call_id
        ):
            raise ValueError(
                "call type and semantic ID must match the linked descriptor."
            )
        _validate_identifier(self.preset_id, field_name="LinkedSemanticCall.preset_id")
        if not isinstance(self.affordances, Mapping):
            raise TypeError("affordances must be a mapping.")
        normalized: dict[str, SceneAffordanceRef] = {}
        for role, affordance in self.affordances.items():
            _validate_identifier(role, field_name="affordance roles")
            if type(affordance) is not SceneAffordanceRef:
                raise TypeError("affordances values must be SceneAffordanceRef values.")
            normalized[role] = affordance
        object.__setattr__(self, "affordances", MappingProxyType(normalized))


@dataclass(frozen=True, slots=True, init=False)
class BoundSemanticCall:
    """Factory-owned call linked to one installed engine/profile combination."""

    linked: LinkedSemanticCall
    binding: ResolvedSkillBinding
    preset: SkillPolicyPreset
    _robot_profile: BoundRobotSkillProfile = field(repr=False, compare=False)

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Reject construction outside :class:`BoundSemanticIntegration`."""
        del args, kwargs
        raise TypeError(
            "BoundSemanticCall values are created by "
            "BoundSemanticIntegration.link_call()."
        )

    @classmethod
    def _create(
        cls,
        *,
        linked: LinkedSemanticCall,
        binding: ResolvedSkillBinding,
        preset: SkillPolicyPreset,
        robot_profile: BoundRobotSkillProfile,
    ) -> BoundSemanticCall:
        """Create and validate one engine/profile-owned result."""
        instance = object.__new__(cls)
        object.__setattr__(instance, "linked", linked)
        object.__setattr__(instance, "binding", binding)
        object.__setattr__(instance, "preset", preset)
        object.__setattr__(instance, "_robot_profile", robot_profile)
        instance._validate()
        return instance

    def _validate(self) -> None:
        """Validate the static and live ownership links."""
        if not isinstance(self.linked, LinkedSemanticCall):
            raise TypeError("linked must be a LinkedSemanticCall.")
        if not isinstance(self.binding, ResolvedSkillBinding):
            raise TypeError("binding must be a ResolvedSkillBinding.")
        if not isinstance(self.preset, SkillPolicyPreset):
            raise TypeError("preset must be a SkillPolicyPreset.")
        if self.binding.skill_id != self.linked.descriptor.skill_id:
            raise ValueError(
                "binding skill_id must match the linked semantic descriptor."
            )
        if self.preset.preset_id != self.linked.preset_id:
            raise ValueError("preset ID must match the statically linked preset.")
        if not isinstance(self._robot_profile, BoundRobotSkillProfile):
            raise TypeError("robot_profile must be a BoundRobotSkillProfile.")
        if (
            self.binding.action_binding.owner_id
            != self._robot_profile.engine.binding_owner_id
        ):
            raise ValueError("binding belongs to a different action engine.")

    @property
    def robot_profile(self) -> BoundRobotSkillProfile:
        """Return the exact bound profile that produced this call."""
        return self._robot_profile


@dataclass(frozen=True, slots=True)
class SemanticIntegrationManifest:
    """Static scene/profile/catalog declaration validated before execution.

    Args:
        scene: Provider-free scene manifest.
        robot_profile: Declarative robot resource/profile snapshot.
        call_catalog: Discoverable semantic call descriptors.
        runtime_preset: Optional integration-wide policy preset override.
    """

    scene: SceneManifest
    robot_profile: RobotSkillProfile
    call_catalog: SemanticCallCatalog
    runtime_preset: str | None = None

    def __post_init__(self) -> None:
        if type(self.scene) is not SceneManifest:
            raise TypeError("scene must be exactly SceneManifest.")
        if type(self.robot_profile) is not RobotSkillProfile:
            raise TypeError("robot_profile must be exactly RobotSkillProfile.")
        if type(self.call_catalog) is not SemanticCallCatalog:
            raise TypeError("call_catalog must be exactly SemanticCallCatalog.")
        known_semantic_ids = set(self.call_catalog.descriptors)
        for preset_id, preset in self.robot_profile.presets.items():
            unknown_monitor_ids = sorted(
                set(preset.effect_monitors).difference(known_semantic_ids)
            )
            if unknown_monitor_ids:
                semantic_id = unknown_monitor_ids[0]
                raise SemanticValidationError(
                    SemanticDiagnostic(
                        "unknown_effect_monitor_call",
                        (
                            "integration",
                            "robot_profile",
                            "presets",
                            preset_id,
                            "effect_monitors",
                            semantic_id,
                        ),
                        f"Effect monitor configuration references unknown semantic "
                        f"call {semantic_id!r}.",
                        tuple(self.call_catalog.descriptors),
                    )
                )
        if self.runtime_preset is not None:
            _validate_identifier(
                self.runtime_preset,
                field_name="runtime_preset",
            )
            if self.runtime_preset not in self.robot_profile.presets:
                raise SemanticValidationError(
                    SemanticDiagnostic(
                        "unknown_preset",
                        ("integration", "runtime_preset"),
                        f"Unknown runtime preset {self.runtime_preset!r}.",
                        tuple(self.robot_profile.presets),
                    )
                )

    def link_call(
        self,
        call: SemanticCallSpec,
        *,
        path: tuple[PathPart, ...] = ("call",),
    ) -> LinkedSemanticCall:
        """Resolve static refs, affordances, and declared resource structure.

        This method never observes scene providers, constructs an engine,
        samples a grasp, or runs a planner.
        """
        if not isinstance(call, SemanticCallSpec):
            raise TypeError("call must be a SemanticCallSpec.")
        try:
            descriptor = self.call_catalog.discover(call)
        except (KeyError, TypeError, ValueError) as exc:
            raise SemanticValidationError(
                SemanticDiagnostic(
                    "unknown_call",
                    (*path, "kind"),
                    str(exc),
                    tuple(self.call_catalog.descriptors),
                )
            ) from exc

        affordances: dict[str, SceneAffordanceRef] = {}
        if isinstance(call, Pick):
            object_ref = self.scene.resolve(
                call.object,
                expected_type=SceneObjectRef,
                path=(*path, "object"),
            )
            grasp = self.scene.resolve_affordance(
                object_ref,
                capability=GRASP_AFFORDANCE_CAPABILITY,
                explicit=call.grasp,
                path=(*path, "grasp"),
            )
            normalized_call: SemanticCallSpec = replace(
                call,
                object=object_ref,
                grasp=grasp,
            )
            affordances["grasp"] = grasp
        elif isinstance(call, Place):
            object_ref = self.scene.resolve(
                call.object,
                expected_type=SceneObjectRef,
                path=(*path, "object"),
            )
            replacements: dict[str, object] = {"object": object_ref}
            if call.on is not None:
                destination, affordance = self._link_relation(
                    call.on,
                    capability=PLACE_ON_AFFORDANCE_CAPABILITY,
                    path=(*path, "on"),
                )
                replacements["on"] = destination
                affordances["destination"] = affordance
            elif call.inside is not None:
                destination, affordance = self._link_relation(
                    call.inside,
                    capability=PLACE_IN_AFFORDANCE_CAPABILITY,
                    path=(*path, "inside"),
                )
                replacements["inside"] = destination
                affordances["destination"] = affordance
            normalized_call = replace(call, **replacements)
        elif isinstance(call, HandOver):
            object_ref = self.scene.resolve(
                call.object,
                expected_type=SceneObjectRef,
                path=(*path, "object"),
            )
            grasp = self.scene.resolve_affordance(
                object_ref,
                capability=GRASP_AFFORDANCE_CAPABILITY,
                path=(*path, "object", "handover_grasp"),
            )
            normalized_call = replace(call, object=object_ref)
            affordances["receiver_grasp"] = grasp
        elif isinstance(call, OperateArticulation):
            articulation_ref = self.scene.resolve(
                call.articulation,
                expected_type=SceneArticulationRef,
                path=(*path, "articulation"),
            )
            handle = self.scene.resolve_affordance(
                articulation_ref,
                capability=ARTICULATION_OPERATION_AFFORDANCE_CAPABILITY,
                explicit=call.handle,
                path=(*path, "handle"),
            )
            normalized_call = replace(
                call,
                articulation=articulation_ref,
                handle=handle,
            )
            affordances["handle"] = handle
        elif isinstance(call, RegisteredSemanticCall):
            normalized_call = replace(
                call,
                arguments=self._normalize_registered_arguments(
                    call.arguments,
                    path=(*path, "arguments"),
                ),
            )
        else:  # defensive for future subclasses not represented by the catalog
            raise SemanticValidationError(
                SemanticDiagnostic(
                    "unsupported_call_type",
                    path,
                    f"No static linker exists for {type(call).__name__}.",
                )
            )
        self._validate_declared_resources(
            descriptor,
            normalized_call.resources,
            path=(*path, "resources"),
        )
        preset_id = self._resolve_declared_preset(
            descriptor,
            path=(*path, "preset"),
        )
        return LinkedSemanticCall(
            call=normalized_call,
            descriptor=descriptor,
            preset_id=preset_id,
            affordances=affordances,
        )

    def _selects_preset(self, preset_id: str) -> bool:
        """Return whether one preset is reachable through this integration.

        Args:
            preset_id: Stable policy preset identifier.

        Returns:
            ``True`` when the integration-wide override or at least one
            catalogued target skill can resolve to ``preset_id`` through its
            per-skill or profile-default selection. This is intentionally a
            conservative integration-level check, not a concrete-program
            reachability analysis.
        """
        _validate_identifier(preset_id, field_name="preset_id")
        if self.runtime_preset is not None:
            return self.runtime_preset == preset_id
        skill_ids = {
            descriptor.skill_id for descriptor in self.call_catalog.descriptors.values()
        }
        return any(
            self.robot_profile.skill_presets.get(
                skill_id,
                self.robot_profile.default_preset,
            )
            == preset_id
            for skill_id in skill_ids
        )

    def _resolve_declared_preset(
        self,
        descriptor: SemanticCallDescriptor,
        *,
        path: tuple[PathPart, ...],
    ) -> str:
        """Resolve the static integration/per-skill/profile preset ID."""
        preset_id = self.runtime_preset
        if preset_id is None:
            preset_id = self.robot_profile.skill_presets.get(descriptor.skill_id)
        if preset_id is None:
            preset_id = self.robot_profile.default_preset
        if preset_id is None:
            raise SemanticValidationError(
                SemanticDiagnostic(
                    "missing_preset",
                    path,
                    f"No policy preset is configured for skill "
                    f"{descriptor.skill_id!r}.",
                    tuple(self.robot_profile.presets),
                )
            )
        if preset_id not in self.robot_profile.presets:
            raise SemanticValidationError(
                SemanticDiagnostic(
                    "unknown_preset",
                    path,
                    f"Unknown policy preset {preset_id!r}.",
                    tuple(self.robot_profile.presets),
                )
            )
        return preset_id

    def _normalize_registered_arguments(
        self,
        value: object,
        *,
        path: tuple[PathPart, ...],
    ) -> object:
        """Canonicalize every typed scene ref in a registered payload."""
        if type(value) in (
            SceneEntityRef,
            SceneObjectRef,
            SceneArticulationRef,
            SceneLinkRef,
            SceneAffordanceRef,
        ):
            return self.scene.resolve(
                value,
                expected_type=type(value),
                path=path,
            )
        # Other exact scene-ref variants are admitted by the call value
        # contract and resolved through their exact runtime type here.
        if isinstance(value, SceneEntityRef):
            return self.scene.resolve(
                value,
                expected_type=type(value),
                path=path,
            )
        if isinstance(value, Mapping):
            return MappingProxyType(
                {
                    key: self._normalize_registered_arguments(
                        nested,
                        path=(*path, key),
                    )
                    for key, nested in value.items()
                }
            )
        if isinstance(value, tuple):
            return tuple(
                self._normalize_registered_arguments(
                    nested,
                    path=(*path, index),
                )
                for index, nested in enumerate(value)
            )
        return value

    def _link_relation(
        self,
        target: SceneObjectRef | SceneAffordanceRef,
        *,
        capability: str,
        path: tuple[PathPart, ...],
    ) -> tuple[SceneObjectRef | SceneAffordanceRef, SceneAffordanceRef]:
        """Normalize one placement relation and select its affordance."""
        if isinstance(target, SceneObjectRef):
            parent = self.scene.resolve(
                target,
                expected_type=SceneObjectRef,
                path=path,
            )
            affordance = self.scene.resolve_affordance(
                parent,
                capability=capability,
                path=path,
            )
            return parent, affordance
        explicit = self.scene.resolve(
            target,
            expected_type=SceneAffordanceRef,
            path=path,
        )
        entry = self.scene.lookup(explicit, path=path)
        assert entry.parent is not None
        affordance = self.scene.resolve_affordance(
            entry.parent,
            capability=capability,
            explicit=explicit,
            path=path,
        )
        return explicit, affordance

    def _validate_declared_resources(
        self,
        descriptor: SemanticCallDescriptor,
        selections: Mapping[str, str],
        *,
        path: tuple[PathPart, ...],
    ) -> None:
        """Validate resource IDs and obvious capability mismatches statically."""
        contract = descriptor.binding_contract
        default = self.robot_profile.defaults.get(descriptor.skill_id)
        if default is not None:
            expected_slots = set(contract.slot_ids)
            default_slots = set(default.resources)
            unknown_default_resources = sorted(
                set(default.resources.values()) - set(self.robot_profile.resources)
            )
            if default_slots != expected_slots or unknown_default_resources:
                raise SemanticValidationError(
                    SemanticDiagnostic(
                        "invalid_default_binding",
                        (
                            "integration",
                            "robot_profile",
                            "defaults",
                            descriptor.skill_id,
                        ),
                        "Default resource binding must cover the exact skill slots "
                        "and reference known resources.",
                        contract.slot_ids,
                    )
                )
        unknown_slots = sorted(set(selections) - set(contract.slot_ids))
        if unknown_slots:
            slot = unknown_slots[0]
            raise SemanticValidationError(
                SemanticDiagnostic(
                    "unknown_resource_slot",
                    (*path, slot),
                    f"Skill {descriptor.skill_id!r} has no resource slot {slot!r}.",
                    contract.slot_ids,
                )
            )
        unknown_resources = sorted(
            set(selections.values()) - set(self.robot_profile.resources)
        )
        if unknown_resources:
            unknown = unknown_resources[0]
            slot = next(key for key, value in selections.items() if value == unknown)
            raise SemanticValidationError(
                SemanticDiagnostic(
                    "unknown_resource",
                    (*path, slot),
                    f"Unknown robot resource {unknown!r}.",
                    tuple(self.robot_profile.resources),
                )
            )
        for slot in contract.slots:
            selected = selections.get(slot.slot_id)
            if default is not None:
                default_resource = self.robot_profile.resources[
                    default.resources[slot.slot_id]
                ]
                if not self._resource_declares_requirements(default_resource, slot):
                    raise SemanticValidationError(
                        SemanticDiagnostic(
                            "invalid_default_binding",
                            (
                                "integration",
                                "robot_profile",
                                "defaults",
                                descriptor.skill_id,
                                slot.slot_id,
                            ),
                            f"Default resource {default_resource.resource_id!r} "
                            f"does not satisfy slot {slot.slot_id!r}.",
                        )
                    )
            candidates = tuple(
                resource
                for resource in self.robot_profile.resources.values()
                if (selected is None or resource.resource_id == selected)
                and self._resource_declares_requirements(resource, slot)
            )
            if not candidates:
                code = (
                    "unsupported_resource"
                    if selected is not None
                    else "unsupported_skill"
                )
                raise SemanticValidationError(
                    SemanticDiagnostic(
                        code,
                        (*path, slot.slot_id),
                        f"No declared robot resource satisfies slot "
                        f"{slot.slot_id!r} for skill {descriptor.skill_id!r}.",
                        tuple(self.robot_profile.resources),
                    )
                )
        effective_selections: dict[str, str] = {}
        if default is not None:
            effective_selections.update(default.resources)
        effective_selections.update(selections)
        for constraint in contract.constraints:
            if not isinstance(constraint, DisjointResourceSlots) or not all(
                slot_id in effective_selections for slot_id in constraint.slots
            ):
                continue
            resources = [
                self.robot_profile.resources[effective_selections[slot_id]]
                for slot_id in constraint.slots
            ]
            leaf_sets = [
                self._declared_resource_leaves(resource) for resource in resources
            ]
            for index, left in enumerate(leaf_sets):
                if any(left & right for right in leaf_sets[index + 1 :]):
                    raise SemanticValidationError(
                        SemanticDiagnostic(
                            "resource_claim_conflict",
                            path,
                            f"Selected resources for slots {list(constraint.slots)} "
                            "share declared physical leaves.",
                            tuple(resource.resource_id for resource in resources),
                        )
                    )

    def _declared_resource_leaves(self, resource: RobotResource) -> frozenset[str]:
        """Return transitive leaves from the static profile resource DAG."""
        if not resource.members:
            return frozenset({resource.resource_id})
        leaves: set[str] = set()
        for member_id in resource.members:
            leaves.update(
                self._declared_resource_leaves(self.robot_profile.resources[member_id])
            )
        return frozenset(leaves)

    def _resource_declares_requirements(
        self,
        resource: RobotResource,
        slot: SkillResourceSlot,
    ) -> bool:
        """Check provider-free endpoint declarations without physical binding."""
        endpoints: dict[str, ResourceEndpoint] = {}
        for requirement in slot.endpoints:
            endpoint = resource.endpoints.get(requirement.endpoint_id)
            if endpoint is None or not requirement.capabilities.issubset(
                endpoint.capabilities
            ):
                return False
            if requirement.required_commands and isinstance(
                endpoint, ControlPartEndpoint
            ):
                profile_id = endpoint.command_profile or endpoint.control_part
                command_profile = self.robot_profile.command_profiles.get(profile_id)
                if command_profile is None:
                    return False
                if any(
                    not isinstance(command_profile.commands.get(name), command_type)
                    for name, command_type in requirement.required_commands.items()
                ):
                    return False
            endpoints[requirement.endpoint_id] = endpoint
        # Adapter claims are unavailable before live binding. For the built-in
        # endpoint, equal control parts are an exact static conflict.
        for constraint in slot.constraints:
            if not isinstance(constraint, DisjointSlotEndpoints):
                continue
            constrained = [endpoints[name] for name in constraint.endpoint_ids]
            for index, left in enumerate(constrained):
                if not isinstance(left, ControlPartEndpoint):
                    continue
                if any(
                    isinstance(right, ControlPartEndpoint)
                    and left.control_part == right.control_part
                    for right in constrained[index + 1 :]
                ):
                    return False
        return True

    def bind(
        self,
        scene_registry: SceneRegistry,
        engine: AtomicActionEngine,
        *,
        endpoint_adapters: (
            Mapping[type[ResourceEndpoint], ResourceEndpointAdapter] | None
        ) = None,
    ) -> BoundSemanticIntegration:
        """Validate live scene and robot bindings without observing or planning."""
        self.scene.validate_registry(scene_registry)
        if not isinstance(engine, AtomicActionEngine):
            raise TypeError("engine must be an AtomicActionEngine.")
        self._validate_safe_dynamic_collision_policy(
            scene_registry=scene_registry,
            engine=engine,
        )
        try:
            bound_profile = engine.bind_skill_profile(
                self.robot_profile,
                endpoint_adapters=endpoint_adapters,
            )
        except Exception as exc:  # noqa: BLE001 - add semantic integration path
            raise SemanticValidationError(
                SemanticDiagnostic(
                    "robot_profile_binding_failed",
                    ("integration", "robot_profile"),
                    str(exc),
                )
            ) from exc
        return BoundSemanticIntegration(
            manifest=self,
            scene_registry=scene_registry,
            robot_profile=bound_profile,
            engine=engine,
        )

    def _validate_safe_dynamic_collision_policy(
        self,
        *,
        scene_registry: SceneRegistry,
        engine: AtomicActionEngine,
    ) -> None:
        """Fail before observation when selected safe planning cannot be strict."""
        if not scene_registry.dynamic_collision_entity_ids or not self._selects_preset(
            "safe"
        ):
            return
        preset = self.robot_profile.presets["safe"]
        policy_path: tuple[PathPart, ...] = (
            "integration",
            "robot_profile",
            "presets",
            "safe",
            "motion_policy",
        )
        if preset.motion_policy.strategy != "motion_gen":
            raise SemanticValidationError(
                SemanticDiagnostic(
                    "safe_dynamic_collision_unsupported",
                    (*policy_path, "strategy"),
                    "The 'safe' preset requires strategy='motion_gen' when the "
                    "scene registry declares dynamic collision entities.",
                    ("motion_gen",),
                )
            )
        if (
            getattr(
                engine.motion_generator,
                "supports_dynamic_collision_world",
                False,
            )
            is not True
        ):
            raise SemanticValidationError(
                SemanticDiagnostic(
                    "safe_dynamic_collision_unsupported",
                    (*policy_path, "dynamic_collision_mode"),
                    "The 'safe' preset requires an active planner with dynamic "
                    "collision-world support for the registered dynamic entities "
                    f"{scene_registry.dynamic_collision_entity_ids!r}.",
                )
            )


class BoundSemanticIntegration:
    """Live-installed, still side-effect-free semantic integration link."""

    def __init__(
        self,
        *,
        manifest: SemanticIntegrationManifest,
        scene_registry: SceneRegistry,
        robot_profile: BoundRobotSkillProfile,
        engine: AtomicActionEngine,
    ) -> None:
        if type(manifest) is not SemanticIntegrationManifest:
            raise TypeError("manifest must be exactly SemanticIntegrationManifest.")
        if not isinstance(scene_registry, SceneRegistry):
            raise TypeError("scene_registry must be a SceneRegistry.")
        if type(robot_profile) is not BoundRobotSkillProfile:
            raise TypeError("robot_profile must be exactly BoundRobotSkillProfile.")
        if not isinstance(engine, AtomicActionEngine):
            raise TypeError("engine must be an AtomicActionEngine.")
        manifest.scene.validate_registry(scene_registry)
        manifest._validate_safe_dynamic_collision_policy(
            scene_registry=scene_registry,
            engine=engine,
        )
        if robot_profile.engine is not engine:
            raise ValueError("robot_profile belongs to a different engine.")
        if engine.skill_profile is not robot_profile:
            raise ValueError(
                "robot_profile must be the canonical profile installed on engine."
            )
        if robot_profile.source_profile is not manifest.robot_profile:
            raise ValueError(
                "robot_profile does not match the semantic integration manifest."
            )
        self._manifest = manifest
        self._scene_registry = scene_registry
        self._robot_profile = robot_profile
        self._engine = engine

    @property
    def manifest(self) -> SemanticIntegrationManifest:
        """Return the static integration declaration."""
        return self._manifest

    @property
    def scene_registry(self) -> SceneRegistry:
        """Return the validated live scene registry."""
        return self._scene_registry

    @property
    def robot_profile(self) -> BoundRobotSkillProfile:
        """Return the validated live robot profile."""
        return self._robot_profile

    @property
    def engine(self) -> AtomicActionEngine:
        """Return the engine whose used call targets are validated at link time."""
        return self._engine

    def link_call(
        self,
        call: SemanticCallSpec,
        *,
        path: tuple[PathPart, ...] = ("call",),
    ) -> BoundSemanticCall:
        """Resolve one call against exact installed skills, resources, and preset."""
        if self._engine.skill_profile is not self._robot_profile:
            raise SemanticValidationError(
                SemanticDiagnostic(
                    "semantic_profile_stale",
                    ("integration", "robot_profile"),
                    "The engine's canonical robot profile changed after this "
                    "semantic integration was bound.",
                )
            )
        linked = self._manifest.link_call(call, path=path)
        installed = self._engine.skills.get(linked.descriptor.skill_id)
        if installed is None or installed != linked.descriptor.target_descriptor:
            raise SemanticValidationError(
                SemanticDiagnostic(
                    "semantic_skill_not_installed",
                    (*path, "kind"),
                    f"Installed engine skill {linked.descriptor.skill_id!r} is "
                    "missing or has a different goal/options/resource contract.",
                    tuple(self._engine.skills),
                )
            )
        try:
            binding = self._robot_profile.resolve(
                linked.descriptor.skill_id,
                linked.call.resources,
            )
            preset = self._robot_profile.preset(
                linked.preset_id,
                skill_id=linked.descriptor.skill_id,
            )
        except Exception as exc:  # noqa: BLE001 - add complete call path
            raise SemanticValidationError(
                SemanticDiagnostic(
                    "semantic_binding_failed",
                    (*path, "resources"),
                    str(exc),
                )
            ) from exc
        if (
            linked.preset_id == "safe"
            and self._scene_registry.dynamic_collision_entity_ids
        ):
            preset = SkillPolicyPreset(
                preset_id=preset.preset_id,
                schema_version=preset.schema_version,
                motion_policy=replace(
                    preset.motion_policy,
                    dynamic_collision_mode=DynamicCollisionMode.REQUIRED,
                ),
                tracking_policy=preset.tracking_policy,
                recovery_policy=preset.recovery_policy,
                runner_cfg=preset.runner_cfg,
                effect_monitors=preset.effect_monitors,
            )
        return BoundSemanticCall._create(
            linked=linked,
            binding=binding,
            preset=preset,
            robot_profile=self._robot_profile,
        )


__all__ = [
    "BoundSemanticCall",
    "BoundSemanticIntegration",
    "LinkedSemanticCall",
    "PathPart",
    "SceneEntityManifest",
    "SceneManifest",
    "SemanticDiagnostic",
    "SemanticIntegrationManifest",
    "SemanticValidationError",
]
