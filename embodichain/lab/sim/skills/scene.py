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

"""Authoritative scene identity and registration value contracts."""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Iterator, Mapping
from copy import deepcopy
from dataclasses import dataclass, field, fields, is_dataclass, replace
from enum import Enum
import math
from types import MappingProxyType
from typing import Any, Protocol, TYPE_CHECKING, TypeVar, runtime_checkable

import torch

from embodichain.lab.sim.common import BatchEntity
from embodichain.lab.sim.atomic_actions import (
    Affordance,
    AntipodalAffordance,
    ArticulationOperationAffordance,
    EntityState,
    ObjectSemantics,
    ObservedArticulationJointState,
    SceneProvider,
    SceneSnapshot,
)
from .effects import EffectEvidenceAddress

if TYPE_CHECKING:
    from embodichain.lab.sim.planners import MotionGenerator
    from embodichain.lab.sim.sim_manager import SimulationManager


RefT = TypeVar("RefT", bound="SceneEntityRef")

GRASP_AFFORDANCE_CAPABILITY = "affordance.grasp"
"""Capability for an affordance usable by object pickup or handover."""

ARTICULATION_OPERATION_AFFORDANCE_CAPABILITY = "affordance.articulation.operation"
"""Capability for a typed handle-driven articulation operation."""

PLACE_ON_AFFORDANCE_CAPABILITY = "affordance.place.on"
"""Capability for an affordance that defines an ``on`` placement relation."""

PLACE_IN_AFFORDANCE_CAPABILITY = "affordance.place.in"
"""Capability for an affordance that defines an ``inside`` placement relation."""

SCENE_ARTICULATION_EVIDENCE_PROVIDER_ID = "builtin.scene_articulation"
"""Stable route for explicitly injected articulation-joint observations."""

SCENE_ARTICULATION_EVIDENCE_PROVIDER_REVISION = "1"
"""Exact contract revision for articulation-joint evidence addresses."""


@dataclass(frozen=True, slots=True)
class ArticulationJointEvidenceAddress(EffectEvidenceAddress):
    """Canonical scene articulation and joint observation address."""

    articulation_id: str
    joint_id: str

    def __post_init__(self) -> None:
        _validate_identifier(self.articulation_id, "articulation_id")
        _validate_identifier(self.joint_id, "joint_id")

    @property
    def address_fingerprint(self) -> Hashable:
        """Return the exact provider-independent joint address."""
        return type(self), self.articulation_id, self.joint_id


class UnsupportedSceneAffordanceError(ValueError):
    """Raised when a parent has no affordance for a required capability."""


class AmbiguousSceneAffordanceError(ValueError):
    """Raised when compatible affordances lack one explicitly scoped default."""


def _validate_identifier(value: str, name: str) -> None:
    """Validate an exact, non-empty identifier without normalizing it."""
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty string without outer whitespace.")


def _normalize_affordance_capabilities(
    values: Iterable[str],
) -> frozenset[str]:
    """Validate one open set of namespaced affordance capabilities."""
    if isinstance(values, (str, bytes)):
        raise TypeError(
            "affordance_capabilities must be an iterable of strings, not a string."
        )
    try:
        capabilities = frozenset(values)
    except TypeError as exc:
        raise TypeError(
            "affordance_capabilities must be an iterable of strings."
        ) from exc
    for capability in capabilities:
        _validate_identifier(capability, "affordance capability")
    return capabilities


def _normalize_default_affordances(
    values: Mapping[str, SceneAffordanceRef],
) -> Mapping[str, SceneAffordanceRef]:
    """Validate and own a capability-scoped default-affordance mapping."""
    if not isinstance(values, Mapping):
        raise TypeError("default_affordances must be a mapping.")
    defaults: dict[str, SceneAffordanceRef] = {}
    for capability, affordance_ref in values.items():
        _validate_identifier(capability, "default affordance capability")
        if type(affordance_ref) is not SceneAffordanceRef:
            raise TypeError(
                "default_affordances values must be SceneAffordanceRef instances."
            )
        defaults[capability] = affordance_ref
    return MappingProxyType(defaults)


@dataclass(frozen=True, slots=True)
class SceneEntityRef:
    """Typed reference to one authoritative scene-registry entity.

    Args:
        entity_id: Globally stable canonical registry identifier.
    """

    entity_id: str
    """Globally stable authoritative registry identifier."""

    def __post_init__(self) -> None:
        _validate_identifier(self.entity_id, "entity_id")


@dataclass(frozen=True, slots=True)
class SceneObjectRef(SceneEntityRef):
    """Reference to one object registered in the semantic scene."""


@dataclass(frozen=True, slots=True)
class SceneArticulationRef(SceneEntityRef):
    """Reference to one articulation registered in the semantic scene."""


@dataclass(frozen=True, slots=True)
class SceneLinkRef(SceneEntityRef):
    """Reference to one registered articulation link."""


@dataclass(frozen=True, slots=True)
class SceneAffordanceRef(SceneEntityRef):
    """Reference to one registered interaction affordance."""


class SceneDynamics(str, Enum):
    """Physical mobility classification owned by a scene registration."""

    UNKNOWN = "unknown"
    STATIC = "static"
    KINEMATIC = "kinematic"
    DYNAMIC = "dynamic"


class SceneCollisionRole(str, Enum):
    """How an entity participates in the planner collision world."""

    NONE = "none"
    STATIC = "static"
    DYNAMIC = "dynamic"


class SceneCollisionWorldMode(str, Enum):
    """Batch-sharing policy for a dynamic planner collision world."""

    SHARED = "shared"
    PER_ENV = "per_env"


@dataclass(frozen=True, slots=True)
class SceneEntityMetadata:
    """Provider-free semantic metadata projected from one registration.

    Args:
        ref: Canonical typed entity reference.
        aliases: Boundary aliases, compared as an order-independent set.
        parent: Canonical parent for links and affordances.
        native_name: Backend-local child name.
        dynamics: Physical mobility classification.
        collision_role: Planner collision classification.
        semantic_type: Optional application semantic type.
        affordance_capabilities: Open capabilities of an affordance.
        default_affordances: Capability-scoped direct-child defaults.
        affordance_payload_type: Exact registered affordance value type.
        affordance_revision: Integrator-owned payload revision or fingerprint.
        relative_pose: Flattened parent-relative 4x4 pose, when declared.
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
        allowed_ref_types = {
            SceneEntityRef,
            SceneObjectRef,
            SceneArticulationRef,
            SceneLinkRef,
            SceneAffordanceRef,
        }
        if type(self.ref) not in allowed_ref_types:
            raise TypeError("SceneEntityMetadata.ref must be a SceneEntityRef.")
        if isinstance(self.aliases, (str, bytes)):
            raise TypeError("SceneEntityMetadata.aliases must be an iterable.")
        aliases = tuple(sorted(set(self.aliases)))
        for alias in aliases:
            _validate_identifier(alias, "scene alias")
        object.__setattr__(self, "aliases", aliases)
        if self.parent is not None and type(self.parent) not in allowed_ref_types:
            raise TypeError("SceneEntityMetadata.parent must be a SceneEntityRef.")
        if self.native_name is not None:
            _validate_identifier(self.native_name, "native_name")
        if not isinstance(self.dynamics, SceneDynamics):
            raise TypeError("SceneEntityMetadata.dynamics must be SceneDynamics.")
        if not isinstance(self.collision_role, SceneCollisionRole):
            raise TypeError(
                "SceneEntityMetadata.collision_role must be SceneCollisionRole."
            )
        if self.semantic_type is not None:
            _validate_identifier(self.semantic_type, "semantic_type")
        object.__setattr__(
            self,
            "affordance_capabilities",
            _normalize_affordance_capabilities(self.affordance_capabilities),
        )
        object.__setattr__(
            self,
            "default_affordances",
            _normalize_default_affordances(self.default_affordances),
        )
        if self.affordance_payload_type is not None and (
            not isinstance(self.affordance_payload_type, type)
            or not issubclass(self.affordance_payload_type, Affordance)
        ):
            raise TypeError(
                "affordance_payload_type must be an Affordance subclass or None."
            )
        if self.affordance_revision is not None:
            _validate_identifier(self.affordance_revision, "affordance_revision")
        if self.relative_pose is not None:
            relative_pose = tuple(float(value) for value in self.relative_pose)
            if len(relative_pose) != 16 or not all(
                math.isfinite(value) for value in relative_pose
            ):
                raise ValueError(
                    "SceneEntityMetadata.relative_pose must contain 16 finite values."
                )
            object.__setattr__(self, "relative_pose", relative_pose)
        self._validate_topology()

    def _validate_topology(self) -> None:
        """Apply the typed topology contract without requiring live providers."""
        if isinstance(self.ref, (SceneObjectRef, SceneArticulationRef)):
            if self.parent is not None or self.native_name is not None:
                raise ValueError(
                    "Object and articulation metadata cannot declare a parent "
                    "or native_name."
                )
            if self.affordance_capabilities or self.affordance_payload_type is not None:
                raise ValueError(
                    "Object and articulation metadata cannot declare affordance "
                    "payload capabilities."
                )
            if self.affordance_revision is not None or self.relative_pose is not None:
                raise ValueError(
                    "Object and articulation metadata cannot declare affordance "
                    "revision or relative_pose."
                )
            return
        if isinstance(self.ref, SceneLinkRef):
            if not isinstance(self.parent, SceneArticulationRef) or (
                self.native_name is None
            ):
                raise ValueError(
                    "Link metadata requires an articulation parent and native_name."
                )
            if self.affordance_capabilities or self.affordance_payload_type is not None:
                raise ValueError(
                    "Link metadata cannot declare affordance payload capabilities."
                )
            if self.affordance_revision is not None or self.relative_pose is not None:
                raise ValueError(
                    "Link metadata cannot declare affordance revision or relative_pose."
                )
            return
        if isinstance(self.ref, SceneAffordanceRef):
            if (
                not isinstance(
                    self.parent,
                    (SceneObjectRef, SceneArticulationRef, SceneLinkRef),
                )
                or self.native_name is None
            ):
                raise ValueError(
                    "Affordance metadata requires an object, articulation, or link "
                    "parent and native_name."
                )
            if self.affordance_payload_type is None:
                raise ValueError(
                    "Affordance metadata requires affordance_payload_type."
                )
            if self.default_affordances:
                raise ValueError(
                    "Affordance metadata cannot declare default_affordances."
                )
            if self.affordance_capabilities and self.affordance_revision is None:
                raise ValueError(
                    "Capability-bearing affordance metadata requires an explicit "
                    "affordance_revision."
                )
            if (
                GRASP_AFFORDANCE_CAPABILITY in self.affordance_capabilities
                and not issubclass(self.affordance_payload_type, AntipodalAffordance)
            ):
                raise TypeError(
                    f"{GRASP_AFFORDANCE_CAPABILITY!r} requires an "
                    "AntipodalAffordance payload."
                )
            if (
                ARTICULATION_OPERATION_AFFORDANCE_CAPABILITY
                in self.affordance_capabilities
                and not issubclass(
                    self.affordance_payload_type,
                    ArticulationOperationAffordance,
                )
            ):
                raise TypeError(
                    f"{ARTICULATION_OPERATION_AFFORDANCE_CAPABILITY!r} requires "
                    "an ArticulationOperationAffordance payload."
                )
            return
        if self.parent is not None or self.native_name is not None:
            raise ValueError("Generic scene metadata cannot declare a parent.")
        if self.affordance_capabilities or self.affordance_payload_type is not None:
            raise ValueError(
                "Generic scene metadata cannot declare affordance capabilities."
            )
        if self.default_affordances:
            raise ValueError(
                "Generic scene metadata cannot declare default_affordances."
            )
        if self.affordance_revision is not None or self.relative_pose is not None:
            raise ValueError(
                "Generic scene metadata cannot declare affordance revision or pose."
            )

    @classmethod
    def from_registration(
        cls,
        registration: SceneEntityRegistration,
    ) -> SceneEntityMetadata:
        """Project semantic metadata without copying a live payload/provider."""
        relative_pose = registration.relative_pose
        return cls(
            ref=registration.ref,
            aliases=registration.aliases,
            parent=registration.parent,
            native_name=registration.native_name,
            dynamics=registration.dynamics,
            collision_role=registration.collision_role,
            semantic_type=registration.semantic_type,
            affordance_capabilities=registration.affordance_capabilities,
            default_affordances=registration.default_affordances,
            affordance_payload_type=(
                None
                if registration.affordance is None
                else type(registration.affordance)
            ),
            affordance_revision=registration.affordance_revision,
            relative_pose=(
                None
                if relative_pose is None
                else tuple(relative_pose.detach().cpu().reshape(-1).tolist())
            ),
        )


@dataclass(frozen=True, slots=True, init=False)
class _SceneMetadataIndex:
    """Shared identity and affordance index for static and live scene catalogs."""

    entries: tuple[SceneEntityMetadata, ...]
    by_id: Mapping[str, SceneEntityMetadata]
    aliases: Mapping[str, str]
    affordances_by_parent_capability: Mapping[
        tuple[str, str], tuple[SceneAffordanceRef, ...]
    ]

    def __init__(self, entries: Iterable[SceneEntityMetadata] = ()) -> None:
        try:
            supplied = tuple(entries)
        except TypeError as exc:
            raise TypeError("entries must be an iterable of scene metadata.") from exc
        if not all(isinstance(entry, SceneEntityMetadata) for entry in supplied):
            raise TypeError("entries must contain SceneEntityMetadata values.")

        by_id: dict[str, SceneEntityMetadata] = {}
        for entry in supplied:
            entity_id = entry.ref.entity_id
            if entity_id in by_id:
                raise ValueError(f"Duplicate canonical scene entity ID {entity_id!r}.")
            by_id[entity_id] = entry

        aliases: dict[str, str] = {}
        canonical_ids = set(by_id)
        for entry in supplied:
            canonical_id = entry.ref.entity_id
            for alias in entry.aliases:
                if alias in canonical_ids:
                    raise ValueError(
                        f"Scene alias {alias!r} collides with canonical entity ID "
                        f"{alias!r}."
                    )
                previous = aliases.get(alias)
                if previous is not None:
                    raise ValueError(
                        f"Scene alias {alias!r} is ambiguous between canonical "
                        f"IDs {previous!r} and {canonical_id!r}."
                    )
                aliases[alias] = canonical_id

        self._validate_relationships(supplied, by_id)
        affordances = self._index_affordances(supplied)
        object.__setattr__(self, "entries", supplied)
        object.__setattr__(self, "by_id", MappingProxyType(by_id))
        object.__setattr__(self, "aliases", MappingProxyType(aliases))
        object.__setattr__(
            self,
            "affordances_by_parent_capability",
            MappingProxyType(affordances),
        )

    @staticmethod
    def _validate_relationships(
        entries: tuple[SceneEntityMetadata, ...],
        by_id: Mapping[str, SceneEntityMetadata],
    ) -> None:
        """Validate canonical parents, native members, and scoped defaults."""
        native_members: dict[tuple[type[SceneEntityRef], str, str], str] = {}
        for entry in entries:
            parent = entry.parent
            if parent is None:
                continue
            if parent.entity_id == entry.ref.entity_id:
                raise ValueError(
                    f"Scene entity {entry.ref.entity_id!r} cannot parent itself."
                )
            parent_entry = by_id.get(parent.entity_id)
            if parent_entry is None:
                raise ValueError(
                    f"Scene entity {entry.ref.entity_id!r} references "
                    f"unregistered parent {parent.entity_id!r}."
                )
            if type(parent_entry.ref) is not type(parent):
                raise TypeError(
                    f"Parent {parent.entity_id!r} is registered as "
                    f"{type(parent_entry.ref).__name__}, not "
                    f"{type(parent).__name__}."
                )
            if isinstance(entry.ref, (SceneLinkRef, SceneAffordanceRef)):
                assert entry.native_name is not None
                member_key = (
                    type(entry.ref),
                    parent.entity_id,
                    entry.native_name,
                )
                previous = native_members.get(member_key)
                if previous is not None:
                    raise ValueError(
                        f"{type(entry.ref).__name__} parent {parent.entity_id!r} "
                        f"and native_name {entry.native_name!r} are already "
                        f"registered as canonical ID {previous!r}."
                    )
                native_members[member_key] = entry.ref.entity_id

        for entry in entries:
            for capability, default_ref in entry.default_affordances.items():
                default_entry = by_id.get(default_ref.entity_id)
                if default_entry is None:
                    raise ValueError(
                        f"Scene entity {entry.ref.entity_id!r} declares unknown "
                        f"default affordance {default_ref.entity_id!r} for "
                        f"capability {capability!r}."
                    )
                if not isinstance(default_entry.ref, SceneAffordanceRef):
                    raise TypeError(
                        f"Default affordance {default_ref.entity_id!r} is "
                        f"registered as {type(default_entry.ref).__name__}, not "
                        "SceneAffordanceRef."
                    )
                if default_entry.parent != entry.ref:
                    actual_parent = default_entry.parent
                    raise ValueError(
                        f"Default affordance {default_ref.entity_id!r} is not a "
                        f"direct child of {entry.ref.entity_id!r}; its parent is "
                        f"{None if actual_parent is None else actual_parent.entity_id!r}."
                    )
                if capability not in default_entry.affordance_capabilities:
                    raise ValueError(
                        f"Default affordance {default_ref.entity_id!r} does not "
                        f"declare capability {capability!r}."
                    )

    @staticmethod
    def _index_affordances(
        entries: tuple[SceneEntityMetadata, ...],
    ) -> dict[tuple[str, str], tuple[SceneAffordanceRef, ...]]:
        """Build deterministic parent/capability reverse lookup entries."""
        mutable: dict[tuple[str, str], list[SceneAffordanceRef]] = {}
        for entry in entries:
            if not isinstance(entry.ref, SceneAffordanceRef):
                continue
            assert entry.parent is not None
            for capability in entry.affordance_capabilities:
                mutable.setdefault((entry.parent.entity_id, capability), []).append(
                    entry.ref
                )
        return {
            key: tuple(sorted(refs, key=lambda ref: ref.entity_id))
            for key, refs in mutable.items()
        }

    def resolve(
        self,
        identifier: str | SceneEntityRef,
        *,
        expected_type: type[RefT] = SceneEntityRef,
    ) -> RefT:
        """Resolve one canonical ID, alias, or typed reference."""
        if not isinstance(expected_type, type) or not issubclass(
            expected_type,
            SceneEntityRef,
        ):
            raise TypeError("expected_type must be a SceneEntityRef subclass.")
        if isinstance(identifier, SceneEntityRef):
            canonical_id = identifier.entity_id
            supplied_ref: SceneEntityRef | None = identifier
        elif isinstance(identifier, str):
            _validate_identifier(identifier, "identifier")
            canonical_id = self.aliases.get(identifier, identifier)
            supplied_ref = None
        else:
            raise TypeError("identifier must be a string or SceneEntityRef.")

        entry = self.by_id.get(canonical_id)
        if entry is None:
            raise KeyError(f"Unknown scene entity {identifier!r}.")
        canonical_ref = entry.ref
        if supplied_ref is not None and type(supplied_ref) is not type(canonical_ref):
            raise TypeError(
                f"Scene entity {canonical_id!r} is registered as "
                f"{type(canonical_ref).__name__}, not "
                f"{type(supplied_ref).__name__}."
            )
        if not isinstance(canonical_ref, expected_type):
            raise TypeError(
                f"Scene entity {canonical_id!r} is "
                f"{type(canonical_ref).__name__}, not {expected_type.__name__}."
            )
        return canonical_ref  # type: ignore[return-value]

    def affordances(
        self,
        parent: str | SceneEntityRef,
        *,
        capability: str,
    ) -> tuple[SceneAffordanceRef, ...]:
        """Return compatible direct-child affordances without selecting one."""
        parent_ref = self.resolve(parent)
        _validate_identifier(capability, "affordance capability")
        return self.affordances_by_parent_capability.get(
            (parent_ref.entity_id, capability),
            (),
        )

    def resolve_affordance(
        self,
        parent: str | SceneEntityRef,
        *,
        capability: str,
        explicit: str | SceneAffordanceRef | None = None,
    ) -> SceneAffordanceRef:
        """Select one compatible affordance using strict scoped defaults."""
        parent_ref = self.resolve(parent)
        candidates = self.affordances(parent_ref, capability=capability)
        if explicit is not None:
            try:
                selected = self.resolve(explicit, expected_type=SceneAffordanceRef)
            except (KeyError, TypeError, ValueError) as exc:
                raise UnsupportedSceneAffordanceError(
                    f"Explicit affordance {explicit!r} is not a registered "
                    "SceneAffordanceRef."
                ) from exc
            entry = self.by_id[selected.entity_id]
            if entry.parent != parent_ref:
                raise UnsupportedSceneAffordanceError(
                    f"Affordance {selected.entity_id!r} is not a direct child of "
                    f"{parent_ref.entity_id!r}."
                )
            if capability not in entry.affordance_capabilities:
                raise UnsupportedSceneAffordanceError(
                    f"Affordance {selected.entity_id!r} does not support "
                    f"capability {capability!r}."
                )
            return selected
        if not candidates:
            raise UnsupportedSceneAffordanceError(
                f"Scene entity {parent_ref.entity_id!r} has no affordance for "
                f"capability {capability!r}."
            )
        if len(candidates) == 1:
            return candidates[0]
        default = self.by_id[parent_ref.entity_id].default_affordances.get(capability)
        if default is not None:
            return self.resolve(default, expected_type=SceneAffordanceRef)
        raise AmbiguousSceneAffordanceError(
            f"Scene entity {parent_ref.entity_id!r} has multiple affordances for "
            f"capability {capability!r}: "
            f"{[candidate.entity_id for candidate in candidates]}. Configure "
            "default_affordances for this parent and capability or select one "
            "explicitly."
        )


@runtime_checkable
class SceneEntityStateProvider(Protocol):
    """Observe one registered entity for an ordered environment batch."""

    def observe(
        self,
        *,
        timestamp: float,
        env_ids: torch.Tensor,
    ) -> EntityState:
        """Return the entity state whose rows follow ``env_ids``.

        Args:
            timestamp: Observation timestamp supplied by the integration.
            env_ids: Stable ordered environment correlation IDs.

        Returns:
            Current pose and confidence for the registered entity.
        """


@runtime_checkable
class SceneArticulationJointStateProvider(Protocol):
    """Observe canonical joints for one registered scene articulation."""

    def observe_joints(
        self,
        *,
        timestamp: float,
        env_ids: torch.Tensor,
    ) -> Mapping[str, ObservedArticulationJointState]:
        """Return live joint observations whose rows follow ``env_ids``."""


@runtime_checkable
class SceneGeometryProvider(Protocol):
    """Provide one entity's planner-facing collision geometry descriptor."""

    def get_geometry(self) -> object:
        """Return the planner-facing geometry descriptor.

        Returns:
            Backend-consumable geometry or a live simulation entity.
        """


@dataclass(frozen=True, slots=True, eq=False)
class SceneEntityRegistration:
    """Immutable integration metadata for one authoritative scene entity.

    Parent relationships, simulator-native names, pose sources, geometry, and
    affordances belong to the registry registration rather than the lightweight
    reference copied into semantic calls.

    Args:
        ref: Canonical typed reference.
        state_provider: Optional dynamic pose/confidence source.
        joint_state_provider: Optional live articulation-joint source.
        aliases: External names normalized at the registry boundary.
        parent: Canonical parent for a link or affordance.
        native_name: Backend-local member name under ``parent``.
        dynamics: Physical mobility classification.
        geometry_provider: Planner-facing collision geometry source.
        collision_role: Static, dynamic, or no planner collision role.
        semantic_type: Optional application semantic type.
        affordance: Affordance value for an affordance registration.
        affordance_capabilities: Open semantic operations supported by an
            affordance registration.
        default_affordances: Capability-to-child mapping owned by a parent
            object, articulation, or link registration.
        affordance_revision: Stable integrator-owned revision or fingerprint for
            capability-bearing affordance payload data.
        relative_pose: Optional parent-relative affordance transform.
    """

    ref: SceneEntityRef
    """Canonical typed reference owned by the registry."""

    state_provider: SceneEntityStateProvider | None = None
    """Explicit dynamic pose/confidence source."""

    aliases: tuple[str, ...] = ()
    """External or legacy names normalized once at the registry boundary."""

    parent: SceneEntityRef | None = None
    """Canonical parent reference for a link or affordance."""

    native_name: str | None = None
    """Backend-local link or affordance name under ``parent``."""

    dynamics: SceneDynamics = SceneDynamics.UNKNOWN
    """Static, kinematic, dynamic, or unknown mobility classification."""

    geometry_provider: SceneGeometryProvider | None = None
    """Collision geometry source required for planner collision roles."""

    collision_role: SceneCollisionRole = SceneCollisionRole.NONE
    """Static/dynamic planner-obstacle role, or ``none``."""

    semantic_type: str | None = None
    """Optional application semantic type such as ``container`` or ``tool``."""

    affordance: Affordance | None = None
    """Affordance value owned by a :class:`SceneAffordanceRef` registration."""

    affordance_capabilities: frozenset[str] = frozenset()
    """Open semantic capabilities declared by an affordance registration."""

    default_affordances: Mapping[str, SceneAffordanceRef] = field(default_factory=dict)
    """Capability-scoped child affordances selected when multiple are valid."""

    affordance_revision: str | None = None
    """Stable payload revision required by capability-bearing affordances."""

    relative_pose: torch.Tensor | None = None
    """Optional parent-relative pose when no explicit state provider exists."""

    joint_state_provider: SceneArticulationJointStateProvider | None = None
    """Explicit live joint source for an articulation registration."""

    def __post_init__(self) -> None:
        if type(self.ref) not in {
            SceneEntityRef,
            SceneObjectRef,
            SceneArticulationRef,
            SceneLinkRef,
            SceneAffordanceRef,
        }:
            raise TypeError("ref must be a SceneEntityRef.")
        if self.state_provider is not None and not isinstance(
            self.state_provider,
            SceneEntityStateProvider,
        ):
            raise TypeError("state_provider must implement SceneEntityStateProvider.")
        if self.joint_state_provider is not None and not isinstance(
            self.joint_state_provider,
            SceneArticulationJointStateProvider,
        ):
            raise TypeError(
                "joint_state_provider must implement "
                "SceneArticulationJointStateProvider."
            )

        if isinstance(self.aliases, (str, bytes)):
            raise TypeError("aliases must be an iterable of identifiers, not a string.")
        try:
            aliases = tuple(self.aliases)
        except TypeError as exc:
            raise TypeError("aliases must be an iterable of identifiers.") from exc
        for alias in aliases:
            _validate_identifier(alias, "alias")
        aliases = tuple(alias for alias in aliases if alias != self.ref.entity_id)
        if len(set(aliases)) != len(aliases):
            raise ValueError("aliases must be unique.")
        object.__setattr__(self, "aliases", aliases)

        if self.parent is not None and type(self.parent) not in {
            SceneEntityRef,
            SceneObjectRef,
            SceneArticulationRef,
            SceneLinkRef,
            SceneAffordanceRef,
        }:
            raise TypeError("parent must be a SceneEntityRef or None.")
        if self.native_name is not None:
            _validate_identifier(self.native_name, "native_name")
        if not isinstance(self.dynamics, SceneDynamics):
            raise TypeError("dynamics must be a SceneDynamics value.")
        if not isinstance(self.collision_role, SceneCollisionRole):
            raise TypeError("collision_role must be a SceneCollisionRole value.")
        if self.geometry_provider is not None and not isinstance(
            self.geometry_provider,
            SceneGeometryProvider,
        ):
            raise TypeError("geometry_provider must implement SceneGeometryProvider.")
        if self.semantic_type is not None:
            _validate_identifier(self.semantic_type, "semantic_type")
        if self.affordance is not None and not isinstance(self.affordance, Affordance):
            raise TypeError("affordance must be an Affordance or None.")
        object.__setattr__(
            self,
            "affordance_capabilities",
            _normalize_affordance_capabilities(self.affordance_capabilities),
        )
        object.__setattr__(
            self,
            "default_affordances",
            _normalize_default_affordances(self.default_affordances),
        )
        if self.affordance_revision is not None:
            _validate_identifier(self.affordance_revision, "affordance_revision")
        if self.relative_pose is not None:
            if not isinstance(self.relative_pose, torch.Tensor):
                raise TypeError("relative_pose must be a torch.Tensor or None.")
            if self.relative_pose.shape != (4, 4):
                raise ValueError("relative_pose must have shape (4, 4).")
            object.__setattr__(self, "relative_pose", self.relative_pose.clone())
        if self.state_provider is not None and self.relative_pose is not None:
            raise ValueError(
                "state_provider and relative_pose are mutually exclusive pose sources."
            )

        self._validate_reference_contract()
        SceneEntityMetadata.from_registration(self)
        if (
            self.collision_role is not SceneCollisionRole.NONE
            and self.geometry_provider is None
        ):
            raise ValueError(
                f"Collision entity {self.ref.entity_id!r} requires geometry_provider."
            )

    def _validate_reference_contract(self) -> None:
        """Validate fields whose meaning follows from the typed ref."""
        if isinstance(self.ref, (SceneObjectRef, SceneArticulationRef)):
            if self.parent is not None:
                raise ValueError(
                    "Object and articulation registrations cannot have a parent."
                )
            if self.native_name is not None:
                raise ValueError(
                    "Object and articulation registrations cannot have native_name."
                )
            if self.state_provider is None:
                raise ValueError(
                    "Object and articulation registrations require state_provider."
                )
            if self.relative_pose is not None:
                raise ValueError(
                    "Object and articulation registrations cannot use relative_pose."
                )
            if self.affordance is not None:
                raise ValueError(
                    "Affordance values require a SceneAffordanceRef registration."
                )
            if self.affordance_capabilities:
                raise ValueError(
                    "affordance_capabilities require a SceneAffordanceRef "
                    "registration."
                )
            if (
                isinstance(self.ref, SceneObjectRef)
                and self.joint_state_provider is not None
            ):
                raise ValueError(
                    "joint_state_provider requires a SceneArticulationRef "
                    "registration."
                )
            return

        if isinstance(self.ref, SceneLinkRef):
            if self.joint_state_provider is not None:
                raise ValueError(
                    "joint_state_provider requires a SceneArticulationRef "
                    "registration."
                )
            if (
                not isinstance(self.parent, SceneArticulationRef)
                or self.native_name is None
            ):
                raise ValueError("Link registrations require parent and native_name.")
            if self.state_provider is None:
                raise ValueError("Link registrations require state_provider.")
            if self.relative_pose is not None:
                raise ValueError("Link registrations cannot use relative_pose.")
            if self.affordance is not None:
                raise ValueError(
                    "Affordance values require a SceneAffordanceRef registration."
                )
            if self.affordance_capabilities:
                raise ValueError(
                    "affordance_capabilities require a SceneAffordanceRef "
                    "registration."
                )
            return

        if isinstance(self.ref, SceneAffordanceRef):
            if self.joint_state_provider is not None:
                raise ValueError(
                    "joint_state_provider requires a SceneArticulationRef "
                    "registration."
                )
            if (
                not isinstance(
                    self.parent,
                    (SceneObjectRef, SceneArticulationRef, SceneLinkRef),
                )
                or self.native_name is None
            ):
                raise ValueError(
                    "Affordance registrations require parent and native_name."
                )
            if self.affordance is None:
                raise ValueError("Affordance registrations require affordance.")
            if self.state_provider is None and self.relative_pose is None:
                raise ValueError(
                    "Affordance registrations require state_provider or relative_pose."
                )
            if self.default_affordances:
                raise ValueError(
                    "An affordance registration cannot declare default_affordances."
                )
            return

        if self.parent is not None or self.native_name is not None:
            raise ValueError("Generic entity registrations cannot declare a parent.")
        if self.joint_state_provider is not None:
            raise ValueError(
                "joint_state_provider requires a SceneArticulationRef registration."
            )
        if self.state_provider is None:
            raise ValueError("Generic entity registrations require state_provider.")
        if self.affordance_capabilities:
            raise ValueError(
                "affordance_capabilities require a SceneAffordanceRef registration."
            )
        if self.default_affordances:
            raise ValueError(
                "Only object, articulation, or link registrations may declare "
                "default_affordances."
            )


def _copy_registration(
    registration: SceneEntityRegistration,
) -> SceneEntityRegistration:
    """Copy registry metadata without cloning live providers or entities."""
    relative_pose = registration.relative_pose
    return replace(
        registration,
        affordance=_copy_affordance(registration.affordance),
        relative_pose=relative_pose.clone() if relative_pose is not None else None,
    )


def _copy_affordance(affordance: Affordance | None) -> Affordance | None:
    """Own mutable affordance metadata while preserving live entity handles."""
    if affordance is None:
        return None
    memo: dict[int, object] = {}
    visited: set[int] = set()

    def visit(value: object) -> None:
        value_id = id(value)
        if value_id in visited:
            return
        visited.add(value_id)
        if isinstance(value, BatchEntity):
            memo[value_id] = value
            return
        if is_dataclass(value) and not isinstance(value, type):
            for data_field in fields(value):
                nested = getattr(value, data_field.name)
                if data_field.name == "_generator" and nested is not None:
                    memo[id(nested)] = None
                else:
                    visit(nested)
            return
        if isinstance(value, Mapping):
            for key, nested in value.items():
                visit(key)
                visit(nested)
            return
        if isinstance(value, (list, tuple, set, frozenset)):
            for nested in value:
                visit(nested)

    visit(affordance)
    try:
        copied = deepcopy(affordance, memo)
    except Exception as exc:  # noqa: BLE001 - normalize opaque metadata failures
        raise TypeError(
            f"Affordance {type(affordance).__name__} must contain copyable "
            "registry metadata."
        ) from exc
    if copied is affordance or type(copied) is not type(affordance):
        raise TypeError(
            f"Affordance {type(affordance).__name__} must deepcopy to a distinct "
            "value of the exact same type."
        )
    return copied


@dataclass(frozen=True, slots=True, eq=False, init=False)
class SceneRegistry:
    """Immutable authoritative catalog of semantic scene entities.

    Canonical identifiers occupy one flat, globally unique namespace. Aliases
    are accepted only at lookup and integration boundaries and always resolve
    to a canonical typed reference before they leave the registry.

    Args:
        registrations: Complete scene registrations. The iterable is copied and
            cannot be extended after construction.
        collision_world_mode: Explicit dynamic-collision batch policy. It may be
            omitted for a single environment, which resolves to ``shared``. A
            multi-environment dynamic world must select a mode explicitly.
    """

    _registrations: tuple[SceneEntityRegistration, ...] = field(repr=False)
    _registrations_by_id: Mapping[str, SceneEntityRegistration] = field(repr=False)
    _metadata_index: _SceneMetadataIndex = field(repr=False)
    _collision_world_entity_ids: tuple[str, ...] = field(repr=False)
    _dynamic_collision_entity_ids: tuple[str, ...] = field(repr=False)
    _static_collision_entity_ids: tuple[str, ...] = field(repr=False)
    collision_world_mode: SceneCollisionWorldMode | None

    def __init__(
        self,
        registrations: Iterable[SceneEntityRegistration] = (),
        *,
        collision_world_mode: SceneCollisionWorldMode | None = None,
    ) -> None:
        if collision_world_mode is not None and not isinstance(
            collision_world_mode,
            SceneCollisionWorldMode,
        ):
            raise TypeError(
                "collision_world_mode must be a SceneCollisionWorldMode or None."
            )
        try:
            supplied = tuple(registrations)
        except TypeError as exc:
            raise TypeError("registrations must be an iterable.") from exc
        if not all(isinstance(item, SceneEntityRegistration) for item in supplied):
            raise TypeError(
                "registrations must contain SceneEntityRegistration values."
            )
        owned = tuple(_copy_registration(item) for item in supplied)
        entity_metadata = tuple(
            SceneEntityMetadata.from_registration(item) for item in owned
        )
        metadata_index = _SceneMetadataIndex(entity_metadata)
        by_id = {registration.ref.entity_id: registration for registration in owned}
        object.__setattr__(self, "_registrations", owned)
        object.__setattr__(
            self,
            "_registrations_by_id",
            MappingProxyType(by_id),
        )
        object.__setattr__(self, "_metadata_index", metadata_index)
        object.__setattr__(
            self,
            "_collision_world_entity_ids",
            tuple(
                item.ref.entity_id
                for item in owned
                if item.collision_role is not SceneCollisionRole.NONE
            ),
        )
        object.__setattr__(
            self,
            "_dynamic_collision_entity_ids",
            tuple(
                item.ref.entity_id
                for item in owned
                if item.collision_role is SceneCollisionRole.DYNAMIC
            ),
        )
        object.__setattr__(
            self,
            "_static_collision_entity_ids",
            tuple(
                item.ref.entity_id
                for item in owned
                if item.collision_role is SceneCollisionRole.STATIC
            ),
        )
        object.__setattr__(self, "collision_world_mode", collision_world_mode)

    @property
    def registrations(self) -> tuple[SceneEntityRegistration, ...]:
        """Return structurally independent registration values."""
        return tuple(_copy_registration(item) for item in self._registrations)

    @property
    def entity_metadata(self) -> tuple[SceneEntityMetadata, ...]:
        """Return provider-free metadata without copying affordance payloads."""
        return self._metadata_index.entries

    @property
    def entity_refs(self) -> tuple[SceneEntityRef, ...]:
        """Return canonical typed references in registration order."""
        return tuple(item.ref for item in self._registrations)

    @property
    def aliases(self) -> Mapping[str, str]:
        """Return the immutable alias-to-canonical-ID index."""
        return self._metadata_index.aliases

    @property
    def collision_world_entity_ids(self) -> tuple[str, ...]:
        """Return every canonical ID represented in the planner world."""
        return self._collision_world_entity_ids

    @property
    def dynamic_collision_entity_ids(self) -> tuple[str, ...]:
        """Return canonical IDs whose planner poses update dynamically."""
        return self._dynamic_collision_entity_ids

    @property
    def static_collision_entity_ids(self) -> tuple[str, ...]:
        """Return canonical IDs baked into the static planner world."""
        return self._static_collision_entity_ids

    def __len__(self) -> int:
        return len(self._registrations)

    def __iter__(self) -> Iterator[SceneEntityRef]:
        return iter(self.entity_refs)

    def __getitem__(
        self,
        identifier: str | SceneEntityRef,
    ) -> SceneEntityRegistration:
        return self.lookup(identifier)

    def resolve(
        self,
        identifier: str | SceneEntityRef,
        *,
        expected_type: type[RefT] = SceneEntityRef,
    ) -> RefT:
        """Resolve a canonical ID or alias to a typed canonical reference.

        Args:
            identifier: Canonical ID, alias, or already typed canonical ref.
            expected_type: Required reference class for typed lookup.

        Returns:
            Registry-owned canonical reference.

        Raises:
            KeyError: If the canonical ID or alias is unknown.
            TypeError: If the supplied or resolved reference has the wrong type.
        """
        return self._metadata_index.resolve(
            identifier,
            expected_type=expected_type,
        )

    def lookup(
        self,
        identifier: str | SceneEntityRef,
        *,
        expected_type: type[RefT] = SceneEntityRef,
    ) -> SceneEntityRegistration:
        """Return an owned registration after canonical typed resolution.

        Args:
            identifier: Canonical ID, alias, or typed canonical reference.
            expected_type: Required reference class.

        Returns:
            A structurally independent copy of the matching registration.
        """
        ref = self.resolve(identifier, expected_type=expected_type)
        return _copy_registration(self._registrations_by_id[ref.entity_id])

    def affordances(
        self,
        parent: str | SceneEntityRef,
        *,
        capability: str,
    ) -> tuple[SceneAffordanceRef, ...]:
        """Return compatible direct-child affordances without selecting one.

        Args:
            parent: Canonical ID, alias, or typed parent reference.
            capability: Required open affordance capability.

        Returns:
            Compatible canonical references sorted by canonical ID.
        """
        return self._metadata_index.affordances(
            parent,
            capability=capability,
        )

    def resolve_affordance(
        self,
        parent: str | SceneEntityRef,
        *,
        capability: str,
        explicit: str | SceneAffordanceRef | None = None,
    ) -> SceneAffordanceRef:
        """Select one compatible affordance with strict scoped-default rules.

        Args:
            parent: Entity that directly owns the affordance.
            capability: Required semantic affordance capability.
            explicit: Optional explicit affordance ID or typed reference.

        Returns:
            One canonical compatible affordance reference.

        Raises:
            UnsupportedSceneAffordanceError: If no compatible affordance exists
                or an explicit affordance has the wrong parent/capability.
            AmbiguousSceneAffordanceError: If multiple candidates exist without
                a scoped default.
        """
        return self._metadata_index.resolve_affordance(
            parent,
            capability=capability,
            explicit=explicit,
        )

    def object_semantics(
        self,
        object_ref: str | SceneObjectRef,
        *,
        affordance: str | SceneAffordanceRef,
    ) -> ObjectSemantics:
        """Build one owned atomic-action semantic snapshot.

        Args:
            object_ref: Canonical object ID, alias, or typed reference.
            affordance: Registered direct-child affordance for the object.

        Returns:
            Object semantics with an owned affordance payload and canonical ID.

        Raises:
            ValueError: If the affordance does not belong to the object.
        """
        canonical_object = self.resolve(
            object_ref,
            expected_type=SceneObjectRef,
        )
        object_registration = self._registrations_by_id[canonical_object.entity_id]
        affordance_registration = self.lookup(
            affordance,
            expected_type=SceneAffordanceRef,
        )
        if affordance_registration.parent != canonical_object:
            raise ValueError(
                f"Affordance {affordance_registration.ref.entity_id!r} is not a "
                f"direct child of object {canonical_object.entity_id!r}."
            )
        payload = affordance_registration.affordance
        if payload is None:
            raise AssertionError("Affordance registration lost its payload.")
        return ObjectSemantics(
            affordance=payload,
            geometry={},
            properties={},
            label=object_registration.semantic_type or "none",
            entity_id=canonical_object.entity_id,
        )

    def make_scene_provider(
        self,
        *,
        translation_threshold: float = 1.0e-4,
        rotation_threshold: float = 1.0e-3,
        batch_size: int | None = None,
    ) -> RegistrySceneProvider:
        """Create an independent provider without planner cross-validation.

        This factory is intended for perception and direct-core consumers. The
        canonical planning path must use :meth:`make_planning_scene_provider`
        so planner IDs, capabilities, and collision-world mode cannot drift.

        Args:
            translation_threshold: Accumulated translation needed to publish a
                material scene change.
            rotation_threshold: Accumulated rotation needed to publish a
                material scene change.
            batch_size: Optional fixed integration batch size. Supplying it
                validates the collision-world mode immediately and binds the
                provider to that row count.

        Returns:
            A new provider with independent revisions and published baselines.
        """
        return RegistrySceneProvider(
            self,
            translation_threshold=translation_threshold,
            rotation_threshold=rotation_threshold,
            batch_size=batch_size,
        )

    def make_planning_scene_provider(
        self,
        motion_generator: MotionGenerator,
        *,
        batch_size: int,
        translation_threshold: float = 1.0e-4,
        rotation_threshold: float = 1.0e-3,
    ) -> RegistrySceneProvider:
        """Create a provider after complete planner/registry validation.

        Args:
            motion_generator: Motion generator that will consume dynamic poses.
            batch_size: Number of execution environments.
            translation_threshold: Accumulated translation needed to publish a
                material scene change.
            rotation_threshold: Accumulated rotation needed to publish a
                material scene change.

        Returns:
            A new independently stateful, planner-validated scene provider.
        """
        provider = self.make_scene_provider(
            translation_threshold=translation_threshold,
            rotation_threshold=rotation_threshold,
            batch_size=batch_size,
        )
        self.validate_collision_integration(
            motion_generator,
            batch_size=batch_size,
            scene_provider=provider,
        )
        return provider

    def collision_geometry_by_id(
        self,
        role: SceneCollisionRole | None = None,
    ) -> Mapping[str, object]:
        """Materialize planner geometry under canonical registry IDs.

        Args:
            role: Optional exact collision-role filter. Without a filter, all
                static and dynamic collision registrations are included.
                Registrations whose role is :attr:`SceneCollisionRole.NONE`
                never enter the planner collision world.

        Returns:
            Fresh immutable canonical-ID-to-geometry mapping.
        """
        if role is not None and not isinstance(role, SceneCollisionRole):
            raise TypeError("role must be a SceneCollisionRole or None.")
        geometry: dict[str, object] = {}
        for registration in self._registrations:
            provider = registration.geometry_provider
            if provider is None:
                continue
            if role is None:
                if registration.collision_role is SceneCollisionRole.NONE:
                    continue
            elif registration.collision_role is not role:
                continue
            entity_id = registration.ref.entity_id
            descriptor = provider.get_geometry()
            if descriptor is None:
                raise ValueError(
                    f"Collision geometry provider for scene entity "
                    f"{entity_id!r} returned None."
                )
            geometry[entity_id] = descriptor
        return MappingProxyType(geometry)

    def validate_collision_integration(
        self,
        motion_generator: MotionGenerator,
        *,
        batch_size: int,
        scene_provider: SceneProvider | None = None,
    ) -> SceneCollisionWorldMode | None:
        """Validate registry/planner agreement before dynamic planning.

        Args:
            motion_generator: Motion generator whose planner consumes obstacles.
            batch_size: Number of execution environments.
            scene_provider: Optional external perception or hardware provider.
                Its concrete ``collision_entity_ids`` must agree exactly with
                the registry and planner declarations.

        Returns:
            Effective dynamic collision mode, or ``None`` without dynamic IDs.
        """
        effective_mode = self.resolve_collision_world_mode(batch_size=batch_size)
        try:
            planner_info = motion_generator.collision_world_info
            if planner_info is None:
                planner_dynamic_ids = ()
                planner_world_ids = ()
                supports_updates = False
                planner_mode = None
            else:
                planner_dynamic_ids = planner_info.dynamic_entity_ids
                planner_world_ids = planner_info.entity_ids
                supports_updates = planner_info.supports_updates
                planner_mode = planner_info.batch_mode
        except AttributeError as exc:
            raise TypeError(
                "motion_generator must expose collision_world_info."
            ) from exc
        planner_dynamic_ids = self._validate_integration_ids(
            planner_dynamic_ids,
            field_name="motion_generator.dynamic_collision_entity_ids",
        )
        planner_world_ids = self._validate_integration_ids(
            planner_world_ids,
            field_name="motion_generator.collision_world_entity_ids",
        )
        registry_dynamic_ids = set(self.dynamic_collision_entity_ids)
        planner_dynamic_id_set = set(planner_dynamic_ids)
        if registry_dynamic_ids != planner_dynamic_id_set:
            raise ValueError(
                "Dynamic collision entity mismatch: registry missing from planner "
                f"{sorted(registry_dynamic_ids - planner_dynamic_id_set)}, planner "
                "missing from registry "
                f"{sorted(planner_dynamic_id_set - registry_dynamic_ids)}. Planner IDs "
                "must use authoritative registry IDs, not aliases."
            )
        registry_world_ids = set(self.collision_world_entity_ids)
        planner_world_id_set = set(planner_world_ids)
        if registry_world_ids != planner_world_id_set:
            raise ValueError(
                "Collision world entity mismatch: registry missing from planner "
                f"{sorted(registry_world_ids - planner_world_id_set)}, planner "
                "missing from registry "
                f"{sorted(planner_world_id_set - registry_world_ids)}. Planner IDs "
                "must use authoritative registry IDs, not aliases."
            )
        if scene_provider is not None:
            if not isinstance(scene_provider, SceneProvider):
                raise TypeError("scene_provider must implement SceneProvider.")
            provider_ids = getattr(scene_provider, "collision_entity_ids", None)
            provider_ids = self._validate_integration_ids(
                provider_ids,
                field_name="scene_provider.collision_entity_ids",
            )
            provider_id_set = set(provider_ids)
            if registry_dynamic_ids != provider_id_set:
                raise ValueError(
                    "Dynamic collision entity mismatch: registry missing from "
                    "provider "
                    f"{sorted(registry_dynamic_ids - provider_id_set)}, provider "
                    "missing from registry "
                    f"{sorted(provider_id_set - registry_dynamic_ids)}. Provider IDs "
                    "must use authoritative registry IDs, not aliases."
                )
        collision_geometry = self.collision_geometry_by_id()
        if set(collision_geometry) != registry_world_ids:
            raise ValueError(
                "Collision geometry IDs do not match authoritative registry "
                f"world IDs {sorted(registry_world_ids)}."
            )
        if not registry_dynamic_ids:
            return None
        if supports_updates is not True:
            raise ValueError(
                "The selected motion generator does not support dynamic collision "
                f"updates required by {sorted(registry_dynamic_ids)}."
            )
        assert effective_mode is not None
        if planner_mode != effective_mode.value:
            raise ValueError(
                "Dynamic collision world mode mismatch: registry requires "
                f"{effective_mode.value!r}, planner declares {planner_mode!r}."
            )
        return effective_mode

    @staticmethod
    def _validate_integration_ids(
        value: object,
        *,
        field_name: str,
    ) -> tuple[str, ...]:
        """Validate one canonical collision-ID declaration at a boundary."""
        if not isinstance(value, tuple) or not all(
            isinstance(entity_id, str) and entity_id and entity_id == entity_id.strip()
            for entity_id in value
        ):
            raise TypeError(
                f"{field_name} must be a tuple of non-empty canonical IDs "
                "without outer whitespace."
            )
        if len(set(value)) != len(value):
            raise ValueError(f"{field_name} must contain unique IDs.")
        return value

    def resolve_collision_world_mode(
        self,
        *,
        batch_size: int,
    ) -> SceneCollisionWorldMode | None:
        """Resolve the configured collision mode for an execution batch.

        Args:
            batch_size: Number of execution environments.

        Returns:
            The effective mode, or ``None`` when no dynamic collision entity is
            registered.
        """
        return self._effective_collision_world_mode(batch_size)

    def _effective_collision_world_mode(
        self,
        batch_size: int,
    ) -> SceneCollisionWorldMode | None:
        """Resolve E without reading any live state or planner integration."""
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError("batch_size must be an integer.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if not self.dynamic_collision_entity_ids:
            return None
        if self.collision_world_mode is not None:
            return self.collision_world_mode
        if batch_size == 1:
            return SceneCollisionWorldMode.SHARED
        raise ValueError(
            "Multi-environment dynamic collision requires an explicit "
            "collision_world_mode of 'shared' or 'per_env'."
        )

    @classmethod
    def from_simulation(
        cls,
        simulation: SimulationManager,
        *,
        rigid_objects: Mapping[str, str] | None = None,
        articulations: Mapping[str, str] | None = None,
        collision_roles: Mapping[str, SceneCollisionRole] | None = None,
        geometry_providers: Mapping[str, SceneGeometryProvider] | None = None,
        collision_world_mode: SceneCollisionWorldMode | None = None,
    ) -> SceneRegistry:
        """Opt explicitly selected simulation entities into a registry.

        ``rigid_objects`` and ``articulations`` map authoritative registry IDs
        to simulation UIDs. UIDs become aliases automatically; unlisted
        simulation entities are never imported. Collision participation
        defaults to :attr:`SceneCollisionRole.NONE`.

        Args:
            simulation: Simulation manager used only for explicit UID lookup.
            rigid_objects: Canonical object IDs mapped to simulation UIDs.
            articulations: Canonical articulation IDs mapped to simulation UIDs.
            collision_roles: Optional collision roles keyed by canonical ID.
            geometry_providers: Optional geometry overrides keyed by canonical
                ID. Selected rigid objects otherwise expose their live handles.
            collision_world_mode: Optional dynamic collision batch-sharing mode.

        Returns:
            Immutable registry containing only the explicitly selected entities.
        """
        object_ids = cls._normalize_simulation_mapping(
            rigid_objects,
            name="rigid_objects",
        )
        articulation_ids = cls._normalize_simulation_mapping(
            articulations,
            name="articulations",
        )
        duplicate_ids = set(object_ids).intersection(articulation_ids)
        if duplicate_ids:
            raise ValueError(
                "Simulation registry IDs must be globally unique across entity "
                f"types: {sorted(duplicate_ids)}."
            )
        all_ids = set(object_ids).union(articulation_ids)
        roles = dict(collision_roles or {})
        geometry = dict(geometry_providers or {})
        for mapping_name, values in (
            ("collision_roles", roles),
            ("geometry_providers", geometry),
        ):
            unknown = set(values).difference(all_ids)
            if unknown:
                raise KeyError(
                    f"{mapping_name} reference unselected registry IDs: "
                    f"{sorted(unknown)}."
                )

        registrations: list[SceneEntityRegistration] = []
        for registry_id, uid in object_ids.items():
            entity = cls._get_simulation_entity(
                simulation,
                getter_name="get_rigid_object",
                registry_id=registry_id,
                uid=uid,
            )
            registrations.append(
                SceneEntityRegistration(
                    ref=SceneObjectRef(registry_id),
                    state_provider=_SimulationEntityStateProvider(entity),
                    aliases=(() if uid == registry_id else (uid,)),
                    geometry_provider=geometry.get(
                        registry_id,
                        _SimulationEntityGeometryProvider(entity),
                    ),
                    collision_role=roles.get(
                        registry_id,
                        SceneCollisionRole.NONE,
                    ),
                )
            )
        for registry_id, uid in articulation_ids.items():
            entity = cls._get_simulation_entity(
                simulation,
                getter_name="get_articulation",
                registry_id=registry_id,
                uid=uid,
            )
            registrations.append(
                SceneEntityRegistration(
                    ref=SceneArticulationRef(registry_id),
                    state_provider=_SimulationEntityStateProvider(entity),
                    joint_state_provider=(
                        _SimulationArticulationJointStateProvider(entity)
                    ),
                    aliases=(() if uid == registry_id else (uid,)),
                    geometry_provider=geometry.get(registry_id),
                    collision_role=roles.get(
                        registry_id,
                        SceneCollisionRole.NONE,
                    ),
                )
            )
        return cls(
            registrations,
            collision_world_mode=collision_world_mode,
        )

    @staticmethod
    def _normalize_simulation_mapping(
        mapping: Mapping[str, str] | None,
        *,
        name: str,
    ) -> dict[str, str]:
        if mapping is None:
            return {}
        if not isinstance(mapping, Mapping):
            raise TypeError(f"{name} must be a mapping from registry ID to UID.")
        normalized = dict(mapping)
        for registry_id, uid in normalized.items():
            _validate_identifier(registry_id, f"{name} registry ID")
            _validate_identifier(uid, f"{name} UID")
        return normalized

    @staticmethod
    def _get_simulation_entity(
        simulation: SimulationManager,
        *,
        getter_name: str,
        registry_id: str,
        uid: str,
    ) -> Any:
        getter = getattr(simulation, getter_name, None)
        if not callable(getter):
            raise TypeError(f"simulation must provide {getter_name}().")
        entity = getter(uid)
        if entity is None:
            raise KeyError(
                f"Simulation UID {uid!r} selected for registry entity "
                f"{registry_id!r} was not found."
            )
        return entity


@dataclass(frozen=True, slots=True)
class _SimulationEntityStateProvider:
    """Read poses from one explicitly selected simulation entity."""

    entity: Any

    def observe(
        self,
        *,
        timestamp: float,
        env_ids: torch.Tensor,
    ) -> EntityState:
        del timestamp, env_ids
        pose = self.entity.get_local_pose(to_matrix=True)
        if not isinstance(pose, torch.Tensor):
            raise TypeError("Simulation entity get_local_pose() must return a tensor.")
        return EntityState(pose)


@dataclass(frozen=True, slots=True)
class _SimulationArticulationJointStateProvider:
    """Read named measured qpos from one selected simulation articulation."""

    entity: Any

    def observe_joints(
        self,
        *,
        timestamp: float,
        env_ids: torch.Tensor,
    ) -> Mapping[str, ObservedArticulationJointState]:
        del timestamp
        qpos = self.entity.get_qpos(target=False)
        if not isinstance(qpos, torch.Tensor):
            raise TypeError("Simulation articulation get_qpos() must return a tensor.")
        if qpos.dim() != 2 or qpos.shape[0] == 0 or qpos.shape[1] == 0:
            raise ValueError(
                "Simulation articulation qpos must have non-empty shape (N, J)."
            )
        joint_names = tuple(self.entity.joint_names)
        if len(joint_names) != qpos.shape[1]:
            raise ValueError(
                "Simulation articulation joint_names must match qpos width."
            )
        for joint_name in joint_names:
            _validate_identifier(joint_name, "simulation articulation joint name")
        if len(set(joint_names)) != len(joint_names):
            raise ValueError("Simulation articulation joint_names must be unique.")
        indices = env_ids.to(device=qpos.device)
        if bool((indices < 0).any()) or int(indices.max().item()) >= qpos.shape[0]:
            raise ValueError(
                "Simulation scene env_ids must address valid articulation rows."
            )
        selected = qpos.index_select(0, indices)
        return MappingProxyType(
            {
                joint_name: ObservedArticulationJointState(
                    selected[:, index : index + 1]
                )
                for index, joint_name in enumerate(joint_names)
            }
        )


@dataclass(frozen=True, slots=True)
class _SimulationEntityGeometryProvider:
    """Expose a selected live rigid object as planner geometry input."""

    entity: Any

    def get_geometry(self) -> object:
        return self.entity


class RegistrySceneProvider(SceneProvider):
    """Stateful scene provider derived from an immutable registry.

    Instances are created by :meth:`SceneRegistry.make_scene_provider`; each
    instance owns its revision counters and material-pose baselines.

    Args:
        registry: Immutable catalog that owns entity registrations.
        translation_threshold: Accumulated translation needed to publish a
            material scene change.
        rotation_threshold: Accumulated rotation needed to publish a material
            scene change.
        batch_size: Optional fixed execution batch size. Factory-created
            planning providers bind this value before their first observation.
    """

    def __init__(
        self,
        registry: SceneRegistry,
        *,
        translation_threshold: float,
        rotation_threshold: float,
        batch_size: int | None = None,
    ) -> None:
        if not isinstance(registry, SceneRegistry):
            raise TypeError("registry must be a SceneRegistry.")
        for name, value in (
            ("translation_threshold", translation_threshold),
            ("rotation_threshold", rotation_threshold),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value < 0.0
            ):
                raise ValueError(f"{name} must be finite and non-negative.")
        self.registry = registry
        self.translation_threshold = float(translation_threshold)
        self.rotation_threshold = float(rotation_threshold)
        self.collision_entity_ids = registry.dynamic_collision_entity_ids
        self._expected_batch_size = batch_size
        self._last_timestamp: float | None = None
        self._env_ids: torch.Tensor | None = None
        self._published_poses: dict[str, torch.Tensor] = {}
        self._published_confidences: dict[str, float] = {}
        self._published_joint_positions: dict[tuple[str, str], torch.Tensor] = {}
        self._published_joint_validity: dict[tuple[str, str], torch.Tensor] = {}
        self._scene_version = 0
        self._collision_revisions: list[int] = []
        self._effective_collision_world_mode = (
            registry.resolve_collision_world_mode(batch_size=batch_size)
            if batch_size is not None
            else None
        )

    @property
    def collision_world_mode(self) -> SceneCollisionWorldMode | None:
        """Return the configured or first-snapshot-resolved collision mode."""
        return (
            self._effective_collision_world_mode
            if self._effective_collision_world_mode is not None
            else self.registry.collision_world_mode
        )

    def snapshot(
        self,
        *,
        timestamp: float,
        env_ids: torch.Tensor,
    ) -> SceneSnapshot:
        """Observe all canonical entities and advance material revisions.

        Args:
            timestamp: Non-negative monotonic observation timestamp.
            env_ids: Stable ordered correlation IDs for every environment row.

        Returns:
            An immutable snapshot keyed only by canonical registry IDs.
        """
        if (
            isinstance(timestamp, bool)
            or not isinstance(timestamp, (int, float))
            or not math.isfinite(float(timestamp))
            or timestamp < 0.0
        ):
            raise ValueError("timestamp must be finite and non-negative.")
        if self._last_timestamp is not None and timestamp < self._last_timestamp:
            raise ValueError("Scene provider timestamps must be monotonic.")
        if (
            not isinstance(env_ids, torch.Tensor)
            or env_ids.dtype != torch.long
            or env_ids.dim() != 1
            or env_ids.numel() == 0
        ):
            raise ValueError("env_ids must be a non-empty 1D int64 tensor.")
        if torch.unique(env_ids).numel() != env_ids.numel():
            raise ValueError("env_ids must be unique.")

        batch_size = int(env_ids.numel())
        if (
            self._expected_batch_size is not None
            and batch_size != self._expected_batch_size
        ):
            raise ValueError(
                "Scene provider batch size must remain equal to its configured "
                f"batch_size={self._expected_batch_size}; got {batch_size}."
            )
        effective_mode = self.registry.resolve_collision_world_mode(
            batch_size=batch_size
        )
        stable_ids = env_ids.detach().to("cpu")
        if self._env_ids is None:
            self._env_ids = stable_ids.clone()
            self._collision_revisions = [0] * batch_size
            self._effective_collision_world_mode = effective_mode
        elif not torch.equal(stable_ids, self._env_ids):
            raise ValueError("Scene provider env_ids must remain stable and ordered.")

        states = self._observe_states(
            timestamp=float(timestamp),
            env_ids=env_ids,
        )
        articulation_joints = self._observe_articulation_joints(
            timestamp=float(timestamp),
            env_ids=env_ids,
        )
        poses = {entity_id: state.pose for entity_id, state in states.items()}
        confidences = {
            entity_id: state.confidence for entity_id, state in states.items()
        }
        if self._published_poses:
            changed_by_entity = {
                entity_id: self._pose_change_mask(
                    self._published_poses[entity_id],
                    current_pose,
                )
                for entity_id, current_pose in poses.items()
            }
            confidence_changed = any(
                confidences[entity_id] != self._published_confidences[entity_id]
                for entity_id in confidences
            )
            joint_changed = self._joint_observations_changed(articulation_joints)
            if (
                confidence_changed
                or joint_changed
                or any(changed.any().item() for changed in changed_by_entity.values())
            ):
                self._scene_version += 1
            collision_changed = torch.zeros(batch_size, dtype=torch.bool)
            for entity_id in self.collision_entity_ids:
                collision_changed |= changed_by_entity[entity_id]
            for row in collision_changed.nonzero(as_tuple=False).flatten().tolist():
                self._collision_revisions[row] += 1

            for entity_id, changed in changed_by_entity.items():
                if changed.any():
                    published_pose = self._published_poses[entity_id]
                    changed_on_published_device = changed.to(published_pose.device)
                    current_pose = poses[entity_id].to(
                        device=published_pose.device,
                        dtype=published_pose.dtype,
                    )
                    published_pose[changed_on_published_device] = current_pose[
                        changed_on_published_device
                    ]
            self._published_confidences = confidences.copy()
        else:
            self._published_poses = {
                entity_id: pose.clone() for entity_id, pose in poses.items()
            }
            self._published_confidences = confidences.copy()
            self._store_joint_baseline(articulation_joints)

        self._last_timestamp = float(timestamp)
        return SceneSnapshot(
            timestamp=float(timestamp),
            version=self._scene_version,
            entities=states,
            collision_world_revision=tuple(self._collision_revisions),
            collision_entity_ids=self.collision_entity_ids,
            articulation_joints=articulation_joints,
        )

    def _observe_articulation_joints(
        self,
        *,
        timestamp: float,
        env_ids: torch.Tensor,
    ) -> dict[tuple[str, str], ObservedArticulationJointState]:
        """Observe every explicitly registered articulation-joint provider."""
        batch_size = int(env_ids.numel())
        observed: dict[tuple[str, str], ObservedArticulationJointState] = {}
        for registration in self.registry._registrations:
            provider = registration.joint_state_provider
            if provider is None:
                continue
            assert isinstance(registration.ref, SceneArticulationRef)
            supplied = provider.observe_joints(
                timestamp=timestamp,
                env_ids=env_ids.clone(),
            )
            if not isinstance(supplied, Mapping):
                raise TypeError(
                    f"Joint provider for {registration.ref.entity_id!r} must "
                    "return a mapping."
                )
            for joint_id, state in supplied.items():
                _validate_identifier(joint_id, "joint provider joint_id")
                if not isinstance(state, ObservedArticulationJointState):
                    raise TypeError(
                        f"Joint provider for {registration.ref.entity_id!r} must "
                        "return ObservedArticulationJointState values."
                    )
                key = registration.ref.entity_id, joint_id
                observed[key] = self._normalize_joint_observation(
                    state,
                    batch_size=batch_size,
                    address=key,
                )
        return observed

    @staticmethod
    def _normalize_joint_observation(
        state: ObservedArticulationJointState,
        *,
        batch_size: int,
        address: tuple[str, str],
    ) -> ObservedArticulationJointState:
        """Broadcast one live joint observation to the scene batch."""
        position = state.position
        if position.dim() == 1:
            position = position.unsqueeze(0).expand(batch_size, -1).clone()
        elif position.shape[0] != batch_size:
            raise ValueError(
                f"Articulation joint {address!r} observation must have {batch_size} "
                "rows."
            )
        valid = state.valid_mask
        if valid is None:
            valid = torch.ones(
                batch_size,
                dtype=torch.bool,
                device=position.device,
            )
        return ObservedArticulationJointState(position, valid)

    def _joint_observations_changed(
        self,
        states: Mapping[tuple[str, str], ObservedArticulationJointState],
    ) -> bool:
        """Update live joint baselines and report any material value change."""
        changed = set(states) != set(self._published_joint_positions)
        if not changed:
            for key, state in states.items():
                previous_position = self._published_joint_positions[key]
                previous_validity = self._published_joint_validity[key]
                current_position = state.position.to(
                    device=previous_position.device,
                    dtype=previous_position.dtype,
                )
                assert state.valid_mask is not None
                current_validity = state.valid_mask.to(previous_validity.device)
                if not torch.equal(
                    current_position, previous_position
                ) or not torch.equal(
                    current_validity,
                    previous_validity,
                ):
                    changed = True
                    break
        self._store_joint_baseline(states)
        return changed

    def _store_joint_baseline(
        self,
        states: Mapping[tuple[str, str], ObservedArticulationJointState],
    ) -> None:
        """Own the current live joint values used for scene revisioning."""
        self._published_joint_positions = {
            key: state.position.clone() for key, state in states.items()
        }
        self._published_joint_validity = {}
        for key, state in states.items():
            assert state.valid_mask is not None
            self._published_joint_validity[key] = state.valid_mask.clone()

    def _observe_states(
        self,
        *,
        timestamp: float,
        env_ids: torch.Tensor,
    ) -> dict[str, EntityState]:
        """Observe explicit sources before deriving relative affordance poses."""
        batch_size = int(env_ids.numel())
        states: dict[str, EntityState] = {}
        relative_registrations: list[SceneEntityRegistration] = []
        for registration in self.registry._registrations:
            entity_id = registration.ref.entity_id
            if registration.state_provider is None:
                relative_registrations.append(registration)
                continue
            state = registration.state_provider.observe(
                timestamp=timestamp,
                env_ids=env_ids.clone(),
            )
            if not isinstance(state, EntityState):
                raise TypeError(
                    f"State provider for {entity_id!r} must return EntityState."
                )
            states[entity_id] = EntityState(
                self._normalize_pose(state.pose, batch_size, entity_id),
                confidence=state.confidence,
            )

        for registration in relative_registrations:
            entity_id = registration.ref.entity_id
            assert registration.parent is not None
            assert registration.relative_pose is not None
            parent_state = states[registration.parent.entity_id]
            relative_pose = registration.relative_pose.to(
                device=parent_state.pose.device,
                dtype=parent_state.pose.dtype,
            )
            pose = torch.matmul(parent_state.pose, relative_pose)
            states[entity_id] = EntityState(
                pose,
                confidence=parent_state.confidence,
            )
        return states

    @staticmethod
    def _normalize_pose(
        pose: torch.Tensor,
        batch_size: int,
        entity_id: str,
    ) -> torch.Tensor:
        if pose.shape == (4, 4):
            return pose.unsqueeze(0).expand(batch_size, -1, -1).clone()
        if pose.shape != (batch_size, 4, 4):
            raise ValueError(
                f"Scene entity {entity_id!r} pose must have shape (4, 4) or "
                f"({batch_size}, 4, 4)."
            )
        return pose.clone()

    def _pose_change_mask(
        self,
        previous: torch.Tensor,
        current: torch.Tensor,
    ) -> torch.Tensor:
        """Return CPU rows changed against the last material publication."""
        current = current.to(device=previous.device, dtype=previous.dtype)
        translation = torch.linalg.vector_norm(
            current[:, :3, 3] - previous[:, :3, 3],
            dim=1,
        )
        relative_rotation = torch.bmm(
            previous[:, :3, :3].transpose(1, 2),
            current[:, :3, :3],
        )
        cosine = (
            (relative_rotation.diagonal(dim1=1, dim2=2).sum(dim=1) - 1.0) / 2.0
        ).clamp(-1.0, 1.0)
        rotation = torch.acos(cosine)
        return (
            (
                (translation > self.translation_threshold)
                | (rotation > self.rotation_threshold)
            )
            .detach()
            .to("cpu")
        )


__all__ = [
    "ARTICULATION_OPERATION_AFFORDANCE_CAPABILITY",
    "ArticulationJointEvidenceAddress",
    "AmbiguousSceneAffordanceError",
    "GRASP_AFFORDANCE_CAPABILITY",
    "PLACE_IN_AFFORDANCE_CAPABILITY",
    "PLACE_ON_AFFORDANCE_CAPABILITY",
    "RegistrySceneProvider",
    "SCENE_ARTICULATION_EVIDENCE_PROVIDER_ID",
    "SCENE_ARTICULATION_EVIDENCE_PROVIDER_REVISION",
    "SceneArticulationJointStateProvider",
    "SceneAffordanceRef",
    "SceneArticulationRef",
    "SceneCollisionRole",
    "SceneCollisionWorldMode",
    "SceneDynamics",
    "SceneEntityRef",
    "SceneEntityMetadata",
    "SceneEntityRegistration",
    "SceneEntityStateProvider",
    "SceneGeometryProvider",
    "SceneLinkRef",
    "SceneObjectRef",
    "SceneRegistry",
    "UnsupportedSceneAffordanceError",
]
