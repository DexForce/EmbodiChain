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

"""Immutable, robot-independent semantic call specifications."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
import math
import re
from types import MappingProxyType
from typing import ClassVar, TypeAlias

import torch

from embodichain.lab.sim.atomic_actions import (
    DisjointResourceSlots,
    DisjointSlotEndpoints,
    SkillBindingContract,
    SkillDescriptor,
    SkillEndpointRequirement,
    SkillResourceSlot,
)

from .scene import (
    SceneAffordanceRef,
    SceneArticulationRef,
    SceneEntityRef,
    SceneLinkRef,
    SceneObjectRef,
)


def _validate_identifier(value: str, *, field_name: str) -> str:
    """Return one exact, non-empty identifier.

    Args:
        value: Candidate identifier.
        field_name: Diagnostic field name.

    Returns:
        The validated input value.

    Raises:
        ValueError: If the value is empty or has outer whitespace.
    """
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(
            f"{field_name} must be a non-empty string without outer whitespace."
        )
    return value


def _validate_registered_call_id(value: str) -> str:
    """Validate one lowercase, multi-segment extension identifier."""
    _validate_identifier(value, field_name="registered semantic call ID")
    if re.fullmatch(r"[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+", value) is None:
        raise ValueError(
            "Registered semantic call IDs must contain two or more lowercase "
            "identifier segments separated by single dots."
        )
    return value


def _snapshot_resources(values: Mapping[str, str]) -> Mapping[str, str]:
    """Validate and own a generic slot-to-resource mapping."""
    if not isinstance(values, Mapping):
        raise TypeError("resources must be a mapping from slot IDs to resource IDs.")
    resources: dict[str, str] = {}
    for slot_id, resource_id in values.items():
        _validate_identifier(slot_id, field_name="resource slot IDs")
        _validate_identifier(resource_id, field_name="resource IDs")
        resources[slot_id] = resource_id
    return MappingProxyType(resources)


def _validate_static_binding_contract(
    contract: SkillBindingContract,
    *,
    field_name: str,
) -> None:
    """Reject runtime-bearing subclasses anywhere in a binding contract."""
    if type(contract) is not SkillBindingContract:
        raise TypeError(f"{field_name} must be exactly SkillBindingContract.")
    if type(contract.slots) is not tuple or type(contract.constraints) is not tuple:
        raise TypeError(f"{field_name} must contain exact immutable tuples.")
    for slot in contract.slots:
        if type(slot) is not SkillResourceSlot:
            raise TypeError(
                f"{field_name}.slots must contain exact SkillResourceSlot values."
            )
        _validate_identifier(slot.slot_id, field_name=f"{field_name} slot IDs")
        if type(slot.endpoints) is not tuple or type(slot.constraints) is not tuple:
            raise TypeError(f"{field_name}.slots must contain exact immutable tuples.")
        for endpoint in slot.endpoints:
            if type(endpoint) is not SkillEndpointRequirement:
                raise TypeError(
                    f"{field_name}.slots.endpoints must contain exact "
                    "SkillEndpointRequirement values."
                )
            _validate_identifier(
                endpoint.endpoint_id,
                field_name=f"{field_name} endpoint IDs",
            )
            if type(endpoint.capabilities) is not frozenset:
                raise TypeError(
                    f"{field_name} endpoint capabilities must be exact frozensets."
                )
            for capability in endpoint.capabilities:
                _validate_identifier(
                    capability,
                    field_name=f"{field_name} endpoint capabilities",
                )
            if type(endpoint.required_commands) is not MappingProxyType:
                raise TypeError(
                    f"{field_name} required commands must be an immutable snapshot."
                )
            for command_name, command_type in endpoint.required_commands.items():
                _validate_identifier(
                    command_name,
                    field_name=f"{field_name} required command names",
                )
                if not isinstance(command_type, type):
                    raise TypeError(
                        f"{field_name} required command contracts must be class "
                        "objects."
                    )
        for constraint in slot.constraints:
            if type(constraint) is not DisjointSlotEndpoints:
                raise TypeError(
                    f"{field_name}.slots.constraints must contain exact "
                    "DisjointSlotEndpoints values."
                )
            if type(constraint.endpoint_ids) is not tuple:
                raise TypeError(
                    f"{field_name} endpoint constraints must contain exact tuples."
                )
            for endpoint_id in constraint.endpoint_ids:
                _validate_identifier(
                    endpoint_id,
                    field_name=f"{field_name} constrained endpoint IDs",
                )
    for constraint in contract.constraints:
        if type(constraint) is not DisjointResourceSlots:
            raise TypeError(
                f"{field_name}.constraints must contain exact "
                "DisjointResourceSlots values."
            )
        if type(constraint.slots) is not tuple:
            raise TypeError(
                f"{field_name} resource constraints must contain exact tuples."
            )
        for slot_id in constraint.slots:
            _validate_identifier(
                slot_id,
                field_name=f"{field_name} constrained slot IDs",
            )


def _validate_static_skill_descriptor(
    descriptor: SkillDescriptor,
    *,
    field_name: str,
) -> None:
    """Validate one exact, provider-free atomic target descriptor."""
    if type(descriptor) is not SkillDescriptor:
        raise TypeError(f"{field_name} must be exactly SkillDescriptor.")
    _validate_identifier(descriptor.skill_id, field_name=f"{field_name}.skill_id")
    if type(descriptor.agent_visible) is not bool:
        raise TypeError(f"{field_name}.agent_visible must be exactly bool.")
    if type(descriptor.goal_type) is tuple:
        if not descriptor.goal_type or not all(
            type(goal_type) is type for goal_type in descriptor.goal_type
        ):
            raise TypeError(f"{field_name}.goal_type must contain exact class objects.")
    elif type(descriptor.goal_type) is not type:
        raise TypeError(
            f"{field_name}.goal_type must be an exact class or tuple of classes."
        )
    if type(descriptor.options_type) is not type:
        raise TypeError(f"{field_name}.options_type must be an exact class object.")
    if descriptor.binding_contract is None:
        raise TypeError(f"{field_name}.binding_contract must be declared.")
    _validate_static_binding_contract(
        descriptor.binding_contract,
        field_name=f"{field_name}.binding_contract",
    )


@dataclass(frozen=True, slots=True, init=False, eq=False)
class SemanticPose:
    """Object-space pose expressed as position and a WXYZ quaternion.

    The value owns normalized tensor snapshots and never exposes its internal
    tensors directly. A single pose or an environment batch is accepted.

    Args:
        position: Shape ``(3,)`` or ``(B, 3)``.
        quaternion_wxyz: Shape ``(4,)`` or ``(B, 4)``. Finite, non-zero
            quaternions are normalized at construction.
    """

    _position: torch.Tensor = field(repr=False)
    _quaternion_wxyz: torch.Tensor = field(repr=False)

    def __init__(
        self,
        position: torch.Tensor | tuple[float, float, float] | list[float],
        quaternion_wxyz: torch.Tensor | tuple[float, float, float, float] | list[float],
    ) -> None:
        position_tensor = torch.as_tensor(position, dtype=torch.float32)
        quaternion_tensor = torch.as_tensor(quaternion_wxyz, dtype=torch.float32)
        if position_tensor.dim() not in (1, 2) or position_tensor.shape[-1] != 3:
            raise ValueError("position must have shape (3,) or (B, 3).")
        if quaternion_tensor.dim() not in (1, 2) or quaternion_tensor.shape[-1] != 4:
            raise ValueError("quaternion_wxyz must have shape (4,) or (B, 4).")
        if position_tensor.dim() != quaternion_tensor.dim():
            raise ValueError(
                "position and quaternion_wxyz must both be unbatched or batched."
            )
        if position_tensor.dim() == 2 and (
            position_tensor.shape[0] != quaternion_tensor.shape[0]
        ):
            raise ValueError("position and quaternion_wxyz batch sizes must match.")
        if position_tensor.dim() == 2 and position_tensor.shape[0] == 0:
            raise ValueError("SemanticPose batches must contain at least one pose.")
        if not torch.isfinite(position_tensor).all():
            raise ValueError("position must contain only finite values.")
        if not torch.isfinite(quaternion_tensor).all():
            raise ValueError("quaternion_wxyz must contain only finite values.")
        norms = torch.linalg.vector_norm(quaternion_tensor, dim=-1, keepdim=True)
        if torch.any(norms <= torch.finfo(torch.float32).eps):
            raise ValueError("quaternion_wxyz must be non-zero.")
        object.__setattr__(self, "_position", position_tensor.clone())
        object.__setattr__(
            self,
            "_quaternion_wxyz",
            (quaternion_tensor / norms).clone(),
        )

    @property
    def position(self) -> torch.Tensor:
        """Return an independent position tensor."""
        return self._position.clone()

    @property
    def quaternion_wxyz(self) -> torch.Tensor:
        """Return an independent normalized quaternion tensor."""
        return self._quaternion_wxyz.clone()

    @property
    def batch_size(self) -> int | None:
        """Return the explicit batch size, or ``None`` for one broadcast pose."""
        return None if self._position.dim() == 1 else self._position.shape[0]

    def snapshot(self) -> SemanticPose:
        """Return an independently owned pose value."""
        return SemanticPose(self._position, self._quaternion_wxyz)

    def to_matrix(self) -> torch.Tensor:
        """Convert the semantic pose to a homogeneous transform.

        Returns:
            Shape ``(4, 4)`` for an unbatched pose or ``(B, 4, 4)`` for a
            batched pose.
        """
        quaternion = self._quaternion_wxyz
        was_unbatched = quaternion.dim() == 1
        if was_unbatched:
            quaternion = quaternion.unsqueeze(0)
            position = self._position.unsqueeze(0)
        else:
            position = self._position
        w, x, y, z = quaternion.unbind(dim=-1)
        output = torch.zeros(
            quaternion.shape[0],
            4,
            4,
            dtype=quaternion.dtype,
            device=quaternion.device,
        )
        output[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
        output[:, 0, 1] = 2.0 * (x * y - z * w)
        output[:, 0, 2] = 2.0 * (x * z + y * w)
        output[:, 1, 0] = 2.0 * (x * y + z * w)
        output[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
        output[:, 1, 2] = 2.0 * (y * z - x * w)
        output[:, 2, 0] = 2.0 * (x * z - y * w)
        output[:, 2, 1] = 2.0 * (y * z + x * w)
        output[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
        output[:, :3, 3] = position
        output[:, 3, 3] = 1.0
        return output[0] if was_unbatched else output


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class SemanticCallSpec:
    """Base value contract shared by every declarative semantic call.

    Args:
        resources: Optional skill-local slot to robot-resource overrides.
    """

    call_kind: ClassVar[str] = "semantic"

    resources: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "resources", _snapshot_resources(self.resources))

    @property
    def semantic_id(self) -> str:
        """Return the stable catalog identifier for this call."""
        return self.call_kind


@dataclass(frozen=True, slots=True, eq=False)
class Pick(SemanticCallSpec):
    """Pick one registered object using an optional explicit grasp affordance.

    Args:
        object: Authoritative semantic object reference.
        grasp: Optional explicit grasp affordance. Omission requests deterministic
            registry selection.
        resources: Optional skill-local resource overrides.
    """

    call_kind: ClassVar[str] = "pick"

    object: SceneObjectRef
    grasp: SceneAffordanceRef | None = None

    def __post_init__(self) -> None:
        SemanticCallSpec.__post_init__(self)
        if type(self.object) is not SceneObjectRef:
            raise TypeError("Pick.object must be a SceneObjectRef.")
        if self.grasp is not None and type(self.grasp) is not SceneAffordanceRef:
            raise TypeError("Pick.grasp must be a SceneAffordanceRef or None.")


PlaceRelationTarget: TypeAlias = SceneObjectRef | SceneAffordanceRef


@dataclass(frozen=True, slots=True, eq=False)
class Place(SemanticCallSpec):
    """Place a held object at exactly one semantic destination.

    Args:
        object: Authoritative held-object reference.
        at: Absolute object-space pose.
        on: Object or affordance supporting an ``on`` relation.
        inside: Object or affordance supporting an ``inside`` relation.
        resources: Optional skill-local resource overrides.
    """

    call_kind: ClassVar[str] = "place"

    object: SceneObjectRef
    at: SemanticPose | None = None
    on: PlaceRelationTarget | None = None
    inside: PlaceRelationTarget | None = None

    def __post_init__(self) -> None:
        SemanticCallSpec.__post_init__(self)
        if type(self.object) is not SceneObjectRef:
            raise TypeError("Place.object must be a SceneObjectRef.")
        destinations = {
            "at": self.at,
            "on": self.on,
            "inside": self.inside,
        }
        selected = [name for name, value in destinations.items() if value is not None]
        if len(selected) != 1:
            raise ValueError(
                "Place requires exactly one of at, on, or inside; selected "
                f"{selected}."
            )
        if self.at is not None:
            if type(self.at) is not SemanticPose:
                raise TypeError("Place.at must be a SemanticPose or None.")
            object.__setattr__(self, "at", self.at.snapshot())
        for field_name in ("on", "inside"):
            target = getattr(self, field_name)
            if target is not None and type(target) not in (
                SceneObjectRef,
                SceneAffordanceRef,
            ):
                raise TypeError(
                    f"Place.{field_name} must be a SceneObjectRef, "
                    "SceneAffordanceRef, or None."
                )


@dataclass(frozen=True, slots=True, eq=False)
class HandOver(SemanticCallSpec):
    """Transfer a held object to another robot resource.

    Args:
        object: Authoritative held-object reference.
        receiver: Optional destination resource ID. It is equivalent to the
            ``destination`` resource slot and must agree with an explicit map.
        final_target: Optional final object-space delivery pose.
        resources: Optional skill-local resource overrides.
    """

    call_kind: ClassVar[str] = "hand_over"

    object: SceneObjectRef
    receiver: str | None = None
    final_target: SemanticPose | None = None

    def __post_init__(self) -> None:
        SemanticCallSpec.__post_init__(self)
        if type(self.object) is not SceneObjectRef:
            raise TypeError("HandOver.object must be a SceneObjectRef.")
        resources = dict(self.resources)
        if self.receiver is not None:
            _validate_identifier(self.receiver, field_name="HandOver.receiver")
            selected = resources.get("destination")
            if selected is not None and selected != self.receiver:
                raise ValueError(
                    "HandOver.receiver conflicts with resources['destination']."
                )
            resources["destination"] = self.receiver
        object.__setattr__(self, "resources", _snapshot_resources(resources))
        if self.final_target is not None:
            if type(self.final_target) is not SemanticPose:
                raise TypeError("HandOver.final_target must be a SemanticPose or None.")
            object.__setattr__(
                self,
                "final_target",
                self.final_target.snapshot(),
            )


DeclarativeValue: TypeAlias = (
    None
    | bool
    | int
    | float
    | str
    | SceneEntityRef
    | SemanticPose
    | tuple["DeclarativeValue", ...]
    | Mapping[str, "DeclarativeValue"]
)


def _snapshot_declarative_value(
    value: object,
    *,
    path: str,
    _active: set[int] | None = None,
    _budget: list[int] | None = None,
    _depth: int = 0,
) -> DeclarativeValue:
    """Recursively own a bounded, acyclic, non-executable payload."""
    if _active is None:
        _active = set()
    if _budget is None:
        _budget = [4096]
    if _depth > 32:
        raise ValueError(f"{path} exceeds the maximum declarative depth of 32.")
    _budget[0] -= 1
    if _budget[0] < 0:
        raise ValueError(f"{path} exceeds the maximum declarative node count.")
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{path} must be finite.")
        return value
    if type(value) in (
        SceneEntityRef,
        SceneObjectRef,
        SceneArticulationRef,
        SceneLinkRef,
        SceneAffordanceRef,
    ):
        return value
    if type(value) is SemanticPose:
        snapshot = value.snapshot()
        if type(snapshot) is not SemanticPose or snapshot is value:
            raise TypeError(
                f"{path}.snapshot() must return an independent SemanticPose."
            )
        return snapshot
    if type(value) in (dict, MappingProxyType):
        container_id = id(value)
        if container_id in _active:
            raise ValueError(f"{path} contains a cyclic declarative mapping.")
        _active.add(container_id)
        try:
            snapshot: dict[str, DeclarativeValue] = {}
            for key, nested in value.items():
                _validate_identifier(key, field_name=f"{path} keys")
                snapshot[key] = _snapshot_declarative_value(
                    nested,
                    path=f"{path}.{key}",
                    _active=_active,
                    _budget=_budget,
                    _depth=_depth + 1,
                )
            return MappingProxyType(snapshot)
        finally:
            _active.remove(container_id)
    if type(value) in (tuple, list):
        container_id = id(value)
        if container_id in _active:
            raise ValueError(f"{path} contains a cyclic declarative sequence.")
        _active.add(container_id)
        try:
            return tuple(
                _snapshot_declarative_value(
                    nested,
                    path=f"{path}[{index}]",
                    _active=_active,
                    _budget=_budget,
                    _depth=_depth + 1,
                )
                for index, nested in enumerate(value)
            )
        finally:
            _active.remove(container_id)
    raise TypeError(
        f"{path} contains non-declarative {type(value).__name__}; callables, "
        "classes, modules, tensors, and live objects are not allowed."
    )


@dataclass(frozen=True, slots=True, eq=False)
class RegisteredSemanticCall(SemanticCallSpec):
    """Safe value payload for a catalog-registered semantic extension.

    Args:
        call_id: Stable extension identifier discovered in a semantic catalog.
        arguments: Nested declarative data. Executable or live values are
            rejected at construction.
        resources: Optional skill-local resource overrides.
    """

    call_kind: ClassVar[str] = "registered"

    call_id: str
    arguments: Mapping[str, DeclarativeValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        SemanticCallSpec.__post_init__(self)
        _validate_registered_call_id(self.call_id)
        if type(self.arguments) not in (dict, MappingProxyType):
            raise TypeError(
                "RegisteredSemanticCall.arguments must be an exact dict or "
                "immutable mapping proxy."
            )
        object.__setattr__(
            self,
            "arguments",
            _snapshot_declarative_value(
                self.arguments,
                path="RegisteredSemanticCall.arguments",
            ),
        )

    @property
    def semantic_id(self) -> str:
        """Return the registered extension identifier."""
        return self.call_id


@dataclass(frozen=True, slots=True)
class SemanticCallDescriptor:
    """Static catalog metadata for one semantic call kind.

    Args:
        call_id: Stable semantic call identifier.
        spec_type: Exact public call value type.
        skill_id: Atomic skill identifier installed separately on an engine.
        binding_contract: Robot-independent resource requirements.
        schema_version: Explicit configuration payload schema version.
        target_descriptor: Exact atomic goal/options/resource contract. It is
            inferred and non-overridable for curated calls and required for
            registered extensions.
    """

    call_id: str
    spec_type: type[SemanticCallSpec]
    skill_id: str
    binding_contract: SkillBindingContract
    schema_version: int = 1
    target_descriptor: SkillDescriptor | None = None

    def __post_init__(self) -> None:
        _validate_identifier(self.call_id, field_name="SemanticCallDescriptor.call_id")
        _validate_identifier(
            self.skill_id, field_name="SemanticCallDescriptor.skill_id"
        )
        if self.spec_type not in (Pick, Place, HandOver, RegisteredSemanticCall):
            raise TypeError(
                "spec_type must be exactly Pick, Place, HandOver, or "
                "RegisteredSemanticCall; extensions use the registered payload "
                "contract rather than executable call subclasses."
            )
        _validate_static_binding_contract(
            self.binding_contract,
            field_name="SemanticCallDescriptor.binding_contract",
        )
        if not isinstance(self.schema_version, int) or isinstance(
            self.schema_version, bool
        ):
            raise TypeError("schema_version must be an integer.")
        if self.schema_version != 1:
            raise ValueError(
                "Unsupported semantic call schema_version "
                f"{self.schema_version}; supported versions are [1]."
            )
        if self.spec_type is not RegisteredSemanticCall and (
            self.call_id != self.spec_type.call_kind
        ):
            raise ValueError(
                f"Descriptor ID {self.call_id!r} must match "
                f"{self.spec_type.__name__}.call_kind "
                f"{self.spec_type.call_kind!r}."
            )
        if self.spec_type is not RegisteredSemanticCall:
            expected = _builtin_call_target(self.spec_type)
            if (
                self.skill_id != expected.skill_id
                or (self.binding_contract != expected.binding_contract)
                or (
                    self.target_descriptor is not None
                    and self.target_descriptor != expected
                )
            ):
                raise ValueError(
                    f"Built-in semantic call {self.call_id!r} must target skill "
                    f"{expected.skill_id!r} with its exact curated descriptor. "
                    "Use RegisteredSemanticCall for extensions."
                )
            object.__setattr__(self, "target_descriptor", expected)
        else:
            if self.target_descriptor is None:
                raise TypeError(
                    "Registered semantic descriptors require target_descriptor."
                )
            _validate_static_skill_descriptor(
                self.target_descriptor,
                field_name="SemanticCallDescriptor.target_descriptor",
            )
            if (
                self.target_descriptor.skill_id != self.skill_id
                or self.target_descriptor.binding_contract != self.binding_contract
                or not self.target_descriptor.agent_visible
                or self.target_descriptor.binding_contract is None
            ):
                raise ValueError(
                    "Registered target_descriptor must be agent-visible and match "
                    "skill_id plus binding_contract exactly."
                )
        if self.spec_type is RegisteredSemanticCall and self.call_id in {
            Pick.call_kind,
            Place.call_kind,
            HandOver.call_kind,
            RegisteredSemanticCall.call_kind,
        }:
            raise ValueError(
                f"Registered semantic call ID {self.call_id!r} is reserved."
            )
        if self.spec_type is RegisteredSemanticCall:
            _validate_registered_call_id(self.call_id)


@dataclass(frozen=True, slots=True, init=False)
class SemanticCallCatalog:
    """Immutable discovery catalog separated from engine installation."""

    _descriptors: Mapping[str, SemanticCallDescriptor]

    def __init__(
        self,
        descriptors: Iterable[SemanticCallDescriptor],
    ) -> None:
        if isinstance(descriptors, (str, bytes)):
            raise TypeError("descriptors must be an iterable of descriptors.")
        try:
            supplied = tuple(descriptors)
        except TypeError as exc:
            raise TypeError("descriptors must be an iterable of descriptors.") from exc
        normalized: dict[str, SemanticCallDescriptor] = {}
        for descriptor in supplied:
            if type(descriptor) is not SemanticCallDescriptor:
                raise TypeError(
                    "descriptors must contain exact SemanticCallDescriptor values."
                )
            if descriptor.call_id in normalized:
                raise ValueError(f"Duplicate semantic call ID {descriptor.call_id!r}.")
            normalized[descriptor.call_id] = descriptor
        object.__setattr__(
            self,
            "_descriptors",
            MappingProxyType(normalized),
        )

    @property
    def descriptors(self) -> Mapping[str, SemanticCallDescriptor]:
        """Return immutable descriptors keyed by exact semantic ID."""
        return self._descriptors

    def discover(
        self,
        call: str | SemanticCallSpec,
    ) -> SemanticCallDescriptor:
        """Discover metadata without installing or executing an implementation.

        Args:
            call: Exact semantic ID or a call value.

        Returns:
            Matching immutable descriptor.

        Raises:
            KeyError: If the exact call ID is unknown.
            TypeError: If the call type disagrees with its descriptor.
        """
        if type(call) is str:
            call_id = _validate_identifier(call, field_name="semantic call ID")
            call_value = None
        elif type(call) in (Pick, Place, HandOver, RegisteredSemanticCall):
            call_id = call.semantic_id
            call_value = call
        else:
            raise TypeError(
                "call must be an exact semantic call ID or supported call value."
            )
        descriptor = self._descriptors.get(call_id)
        if descriptor is None:
            raise KeyError(
                f"Unknown semantic call {call_id!r}; available calls are "
                f"{sorted(self._descriptors)}."
            )
        if call_value is not None and type(call_value) is not descriptor.spec_type:
            raise TypeError(
                f"Semantic call {call_id!r} expects "
                f"{descriptor.spec_type.__name__}, got "
                f"{type(call_value).__name__}."
            )
        return descriptor

    def with_descriptor(
        self,
        descriptor: SemanticCallDescriptor,
    ) -> SemanticCallCatalog:
        """Return a new catalog containing one additional descriptor."""
        return SemanticCallCatalog((*self._descriptors.values(), descriptor))


def _builtin_call_target(
    spec_type: type[SemanticCallSpec],
) -> SkillDescriptor:
    """Return the non-overridable atomic target for one curated call type."""
    from embodichain.lab.sim.atomic_actions.primitives.hand_over import (
        HandOver as HandOverAction,
    )
    from embodichain.lab.sim.atomic_actions.primitives.pick_up import PickUp
    from embodichain.lab.sim.atomic_actions.primitives.place import Place as PlaceAction

    targets = {
        Pick: PickUp.descriptor(),
        Place: PlaceAction.descriptor(),
        HandOver: HandOverAction.descriptor(),
    }
    try:
        return targets[spec_type]
    except KeyError as exc:
        raise TypeError(f"Unsupported curated call type {spec_type!r}.") from exc


def builtin_semantic_call_catalog() -> SemanticCallCatalog:
    """Build the curated catalog for installed manipulation primitives.

    Returns:
        A fresh immutable catalog. Atomic implementations remain uninstalled;
        callers bind them to an engine through the separate runtime path.
    """
    descriptors = tuple(
        SemanticCallDescriptor(
            call_id=spec_type.call_kind,
            spec_type=spec_type,
            skill_id=_builtin_call_target(spec_type).skill_id,
            binding_contract=_builtin_call_target(spec_type).binding_contract,
        )
        for spec_type in (Pick, Place, HandOver)
    )
    return SemanticCallCatalog(descriptors)


__all__ = [
    "DeclarativeValue",
    "HandOver",
    "Pick",
    "Place",
    "PlaceRelationTarget",
    "RegisteredSemanticCall",
    "SemanticCallCatalog",
    "SemanticCallDescriptor",
    "SemanticCallSpec",
    "SemanticPose",
    "builtin_semantic_call_catalog",
]
