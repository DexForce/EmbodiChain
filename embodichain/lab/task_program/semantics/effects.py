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

"""Backend-neutral semantic-effect contracts, evidence, and monitors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Hashable, Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
import math
from types import MappingProxyType
from typing import ClassVar, TypeAlias

import torch

from embodichain.lab.sim.atomic_actions.verification import EffectVerificationRequest
from embodichain.lab.sim.atomic_actions.state import (
    ArticulationJointState,
    HeldObjectState,
)

from ._validation import validate_identifier as _validate_identifier

EffectMonitorParam: TypeAlias = (
    None
    | bool
    | int
    | float
    | str
    | tuple["EffectMonitorParam", ...]
    | Mapping[str, "EffectMonitorParam"]
)
"""Recursively immutable, non-executable monitor configuration value."""

COMPOSITE_EFFECT_MONITOR_ID = "builtin.composite_effect"
"""Stable ID of the built-in typed-clause monitor."""

COMPOSITE_EFFECT_MONITOR_REVISION = "1"
"""Exact behavior/configuration revision of the built-in monitor."""

CONTROL_PART_EVIDENCE_PROVIDER_ID = "builtin.control_part"
"""Stable provider ID used by generic control-part evidence addresses."""

CONTROL_PART_EVIDENCE_PROVIDER_REVISION = "1"
"""Exact contract revision of control-part evidence addresses."""

POSE_RELATION_EFFECT_CHANNEL = "pose_relation"
CONTACT_EFFECT_CHANNEL = "contact"
CONSTRAINT_EFFECT_CHANNEL = "constraint"
FORCE_EFFECT_CHANNEL = "force"
JOINT_STATE_EFFECT_CHANNEL = "joint_state"

_EFFECT_CHANNELS = frozenset(
    {
        POSE_RELATION_EFFECT_CHANNEL,
        CONTACT_EFFECT_CHANNEL,
        CONSTRAINT_EFFECT_CHANNEL,
        FORCE_EFFECT_CHANNEL,
        JOINT_STATE_EFFECT_CHANNEL,
    }
)
_SE3_BASE_ATOL = 1.0e-5
_SE3_EPS_MULTIPLIER = 10.0


def _metadata_value(value: object) -> object:
    """Convert one typed effect value to deterministic JSON-safe data."""
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        return value if math.isfinite(value) else None
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, torch.Tensor):
        return _metadata_value(value.detach().cpu().tolist())
    if isinstance(value, Mapping):
        return {
            str(key): _metadata_value(nested)
            for key, nested in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_metadata_value(nested) for nested in value]
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
            **{
                data_field.name: _metadata_value(getattr(value, data_field.name))
                for data_field in fields(value)
            },
        }
    return {"type": f"{type(value).__module__}.{type(value).__qualname__}"}


def _snapshot_declarative_value(
    value: object,
    *,
    path: str,
    active: set[int] | None = None,
    budget: list[int] | None = None,
    depth: int = 0,
) -> EffectMonitorParam:
    """Own one bounded, acyclic, non-executable declarative value."""
    if active is None:
        active = set()
    if budget is None:
        budget = [4096]
    if depth > 32:
        raise ValueError(f"{path} exceeds the maximum declarative depth of 32.")
    budget[0] -= 1
    if budget[0] < 0:
        raise ValueError(f"{path} exceeds the maximum declarative node count.")
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{path} must be finite.")
        return value
    if type(value) in (dict, MappingProxyType):
        container_id = id(value)
        if container_id in active:
            raise ValueError(f"{path} contains a cyclic mapping.")
        active.add(container_id)
        try:
            snapshot: dict[str, EffectMonitorParam] = {}
            for key, nested in value.items():
                _validate_identifier(key, field_name=f"{path} keys")
                snapshot[key] = _snapshot_declarative_value(
                    nested,
                    path=f"{path}.{key}",
                    active=active,
                    budget=budget,
                    depth=depth + 1,
                )
            return MappingProxyType(snapshot)
        finally:
            active.remove(container_id)
    if type(value) in (tuple, list):
        container_id = id(value)
        if container_id in active:
            raise ValueError(f"{path} contains a cyclic sequence.")
        active.add(container_id)
        try:
            return tuple(
                _snapshot_declarative_value(
                    nested,
                    path=f"{path}[{index}]",
                    active=active,
                    budget=budget,
                    depth=depth + 1,
                )
                for index, nested in enumerate(value)
            )
        finally:
            active.remove(container_id)
    raise TypeError(
        f"{path} contains non-declarative {type(value).__name__}; callables, "
        "classes, tensors, and live objects are not allowed."
    )


def _snapshot_monitor_params(
    values: Mapping[str, EffectMonitorParam],
) -> Mapping[str, EffectMonitorParam]:
    """Validate and own a monitor-parameter mapping."""
    if type(values) not in (dict, MappingProxyType):
        raise TypeError(
            "EffectMonitorRef.params must be an exact dict or mapping proxy."
        )
    snapshot = _snapshot_declarative_value(values, path="EffectMonitorRef.params")
    assert isinstance(snapshot, Mapping)
    return snapshot


def _validate_pose_batch(
    value: torch.Tensor,
    *,
    field_name: str,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Validate and own unbatched or batched proper SE(3) transforms."""
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{field_name} must be a torch.Tensor.")
    if value.shape != (4, 4) and (
        value.dim() != 3 or value.shape[0] == 0 or value.shape[-2:] != (4, 4)
    ):
        raise ValueError(f"{field_name} must have shape (4, 4) or (B, 4, 4).")
    if not value.is_floating_point():
        raise TypeError(f"{field_name} must use a floating-point dtype.")
    poses = value.unsqueeze(0) if value.dim() == 2 else value
    if valid_mask is not None:
        if not isinstance(valid_mask, torch.Tensor):
            raise TypeError("valid_mask must be a torch.Tensor.")
        if valid_mask.dtype != torch.bool or valid_mask.shape != (poses.shape[0],):
            raise ValueError("valid_mask must be a bool tensor with shape (B,).")
        if valid_mask.device != poses.device:
            raise ValueError("valid_mask and poses must share a device.")
        poses = poses[valid_mask]
    if poses.numel() == 0:
        return value.clone()
    if not torch.isfinite(poses).all():
        raise ValueError(f"{field_name} must contain only finite values.")
    tolerance = max(
        _SE3_BASE_ATOL,
        _SE3_EPS_MULTIPLIER * float(torch.finfo(value.dtype).eps),
    )
    checked = poses.to(dtype=torch.float64)
    expected_bottom = checked.new_tensor((0.0, 0.0, 0.0, 1.0))
    if not torch.isclose(
        checked[:, 3, :],
        expected_bottom.expand(checked.shape[0], -1),
        atol=tolerance,
        rtol=0.0,
    ).all():
        raise ValueError(
            f"{field_name} must contain SE(3) transforms with homogeneous "
            "bottom row [0, 0, 0, 1]."
        )
    rotations = checked[:, :3, :3]
    gram = rotations.transpose(-1, -2) @ rotations
    identity = torch.eye(3, dtype=checked.dtype, device=checked.device).expand_as(gram)
    if not torch.isclose(gram, identity, atol=tolerance, rtol=0.0).all():
        raise ValueError(
            f"{field_name} must contain SE(3) transforms with orthonormal rotations."
        )
    determinants = torch.linalg.det(rotations)
    if not torch.isclose(
        determinants,
        torch.ones_like(determinants),
        atol=tolerance,
        rtol=0.0,
    ).all():
        raise ValueError(
            f"{field_name} must contain SE(3) transforms with rotation "
            "determinant +1."
        )
    return value.clone()


class SemanticEffectKind(str, Enum):
    """Trace-level semantic effect category; clause types define behavior."""

    ATTACH = "attach"
    RELEASE = "release"
    TRANSFER = "transfer"
    ARTICULATION = "articulation"
    REGISTERED = "registered"


class SymbolicStateDomain(str, Enum):
    """Typed mapping domains owned by :class:`~atomic_actions.TaskState`."""

    HELD_OBJECT = "held_object"
    COORDINATED_HELD_OBJECT = "coordinated_held_object"
    ARTICULATION_JOINT = "articulation_joint"


@dataclass(frozen=True, slots=True)
class SymbolicStateKey:
    """Provider-free key for one exact symbolic ``TaskState`` write.

    The domain makes otherwise similar string and pair addresses impossible to
    conflate during static parallel analysis.  This contract intentionally
    describes only exact keys; dynamic or opaque effects must not manufacture
    a guessed key.
    """

    domain: SymbolicStateDomain
    address: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.domain, SymbolicStateDomain):
            raise TypeError("domain must be a SymbolicStateDomain.")
        address = tuple(self.address)
        expected_size = 1 if self.domain is SymbolicStateDomain.HELD_OBJECT else 2
        if len(address) != expected_size:
            raise ValueError(
                f"{self.domain.value} symbolic keys require exactly "
                f"{expected_size} address component(s)."
            )
        for component in address:
            _validate_identifier(
                component,
                field_name=f"{self.domain.value} symbolic key components",
            )
        object.__setattr__(self, "address", address)

    @classmethod
    def held_object(cls, task_state_key: str) -> SymbolicStateKey:
        """Build one held-object mapping key."""
        return cls(SymbolicStateDomain.HELD_OBJECT, (task_state_key,))

    @classmethod
    def coordinated_held_object(
        cls,
        first_task_state_key: str,
        second_task_state_key: str,
    ) -> SymbolicStateKey:
        """Build one ordered coordinated-held-object mapping key."""
        return cls(
            SymbolicStateDomain.COORDINATED_HELD_OBJECT,
            (first_task_state_key, second_task_state_key),
        )

    @classmethod
    def articulation_joint(
        cls,
        articulation_id: str,
        joint_id: str,
    ) -> SymbolicStateKey:
        """Build one articulation-joint mapping key."""
        return cls(
            SymbolicStateDomain.ARTICULATION_JOINT,
            (articulation_id, joint_id),
        )

    @property
    def rendered(self) -> str:
        """Return a deterministic domain-qualified diagnostic form."""
        return f"{self.domain.value}[{', '.join(repr(item) for item in self.address)}]"


@dataclass(frozen=True, slots=True)
class EffectMonitorRef:
    """Versioned, declarative reference to an effect-monitor factory."""

    monitor_id: str
    revision: str
    params: Mapping[str, EffectMonitorParam] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_identifier(self.monitor_id, field_name="EffectMonitorRef.monitor_id")
        _validate_identifier(self.revision, field_name="EffectMonitorRef.revision")
        object.__setattr__(self, "params", _snapshot_monitor_params(self.params))

    def snapshot(self) -> EffectMonitorRef:
        """Return an independently owned declarative reference."""
        return EffectMonitorRef(self.monitor_id, self.revision, self.params)

    def to_metadata(self) -> dict[str, object]:
        """Return a deterministic JSON-safe monitor selection."""
        return {
            "monitor_id": self.monitor_id,
            "revision": self.revision,
            "params": _metadata_value(self.params),
        }


class EffectEvidenceAddress(ABC):
    """Immutable observation address, deliberately separate from command targets."""

    @property
    @abstractmethod
    def address_fingerprint(self) -> Hashable:
        """Return a stable, hashable physical observation address."""

    def snapshot(self) -> EffectEvidenceAddress:
        """Return an independently owned address of the exact same type."""
        return deepcopy(self)


@dataclass(frozen=True, slots=True)
class ControlPartEvidenceAddress(EffectEvidenceAddress):
    """Provider-neutral robot control-part observation address."""

    control_part: str
    channel: str

    def __post_init__(self) -> None:
        _validate_identifier(
            self.control_part,
            field_name="ControlPartEvidenceAddress.control_part",
        )
        _validate_identifier(
            self.channel, field_name="ControlPartEvidenceAddress.channel"
        )
        if self.channel not in _EFFECT_CHANNELS:
            raise ValueError(
                f"Unknown control-part effect channel {self.channel!r}; expected "
                f"one of {sorted(_EFFECT_CHANNELS)}."
            )

    @property
    def address_fingerprint(self) -> Hashable:
        """Return the channel-scoped control-part observation address."""
        return type(self), self.control_part, self.channel


@dataclass(frozen=True, slots=True)
class EffectEvidenceSourceRef:
    """Versioned provider route plus one immutable observation address."""

    provider_id: str
    revision: str
    address: EffectEvidenceAddress

    def __post_init__(self) -> None:
        _validate_identifier(
            self.provider_id,
            field_name="EffectEvidenceSourceRef.provider_id",
        )
        _validate_identifier(
            self.revision,
            field_name="EffectEvidenceSourceRef.revision",
        )
        if not isinstance(self.address, EffectEvidenceAddress):
            raise TypeError(
                "EffectEvidenceSourceRef.address must be an EffectEvidenceAddress."
            )
        snapshot = self.address.snapshot()
        if type(snapshot) is not type(self.address) or snapshot is self.address:
            raise TypeError(
                "EffectEvidenceAddress.snapshot() must return an independently "
                "owned address of the same exact type."
            )
        try:
            source_fingerprint = self.address.address_fingerprint
            snapshot_fingerprint = snapshot.address_fingerprint
            hash(source_fingerprint)
            hash(snapshot_fingerprint)
        except TypeError as exc:
            raise TypeError(
                "EffectEvidenceAddress.address_fingerprint must be hashable."
            ) from exc
        if snapshot_fingerprint != source_fingerprint:
            raise ValueError(
                "EffectEvidenceAddress.snapshot() must preserve its fingerprint."
            )
        object.__setattr__(self, "address", snapshot)

    @property
    def source_fingerprint(self) -> Hashable:
        """Return the provider-scoped source address fingerprint."""
        return (
            self.provider_id,
            self.revision,
            type(self.address),
            self.address.address_fingerprint,
        )

    def snapshot(self) -> EffectEvidenceSourceRef:
        """Return an independently owned source reference."""
        return EffectEvidenceSourceRef(
            self.provider_id,
            self.revision,
            self.address,
        )

    def to_metadata(self) -> dict[str, object]:
        """Return the versioned physical observation address as JSON-safe data."""
        return {
            "provider_id": self.provider_id,
            "revision": self.revision,
            "address": _metadata_value(self.address),
        }


class HeldObjectRelation(str, Enum):
    """Expected symbolic held-object state at an effect boundary."""

    ATTACHED = "attached"
    DETACHED = "detached"


@dataclass(frozen=True, slots=True)
class HeldObjectStateExpectation:
    """Typed individual held-object postcondition."""

    expectation_id: str
    relation: HeldObjectRelation
    object_id: str
    slot_id: str
    resource_id: str
    task_state_key: str

    def __post_init__(self) -> None:
        for field_name in (
            "expectation_id",
            "object_id",
            "slot_id",
            "resource_id",
            "task_state_key",
        ):
            _validate_identifier(
                getattr(self, field_name),
                field_name=f"HeldObjectStateExpectation.{field_name}",
            )
        if not isinstance(self.relation, HeldObjectRelation):
            raise TypeError("relation must be a HeldObjectRelation.")

    def snapshot(self) -> HeldObjectStateExpectation:
        """Return an independently constructed state expectation."""
        return HeldObjectStateExpectation(
            self.expectation_id,
            self.relation,
            self.object_id,
            self.slot_id,
            self.resource_id,
            self.task_state_key,
        )


@dataclass(frozen=True, slots=True)
class CoordinatedHeldObjectCleanupExpectation:
    """Typed removal of one coordinated held-object relation."""

    expectation_id: str
    task_state_keys: tuple[str, str]

    def __post_init__(self) -> None:
        _validate_identifier(
            self.expectation_id,
            field_name="CoordinatedHeldObjectCleanupExpectation.expectation_id",
        )
        keys = tuple(self.task_state_keys)
        if len(keys) != 2:
            raise ValueError("task_state_keys must contain exactly two keys.")
        for key in keys:
            _validate_identifier(key, field_name="coordinated task-state keys")
        if keys[0] == keys[1]:
            raise ValueError("Coordinated task-state keys must be distinct.")
        object.__setattr__(self, "task_state_keys", keys)

    def snapshot(self) -> CoordinatedHeldObjectCleanupExpectation:
        """Return an independently constructed cleanup expectation."""
        return CoordinatedHeldObjectCleanupExpectation(
            self.expectation_id,
            self.task_state_keys,
        )


@dataclass(frozen=True, slots=True, eq=False)
class ArticulationJointStateExpectation:
    """Symbolic articulation-joint postcondition."""

    expectation_id: str
    articulation_id: str
    joint_id: str
    target_position: torch.Tensor

    def __post_init__(self) -> None:
        for field_name in ("expectation_id", "articulation_id", "joint_id"):
            _validate_identifier(
                getattr(self, field_name),
                field_name=f"ArticulationJointStateExpectation.{field_name}",
            )
        target = self.target_position
        if not isinstance(target, torch.Tensor) or target.dim() not in (1, 2):
            raise ValueError("target_position must have shape (J,) or (B, J).")
        if target.numel() == 0 or not target.is_floating_point():
            raise TypeError("target_position must be a non-empty floating tensor.")
        if not torch.isfinite(target).all():
            raise ValueError("target_position must be finite.")
        object.__setattr__(self, "target_position", target.clone())

    def snapshot(self) -> ArticulationJointStateExpectation:
        """Return an independently owned articulation expectation."""
        return ArticulationJointStateExpectation(
            self.expectation_id,
            self.articulation_id,
            self.joint_id,
            self.target_position,
        )


EffectStateExpectation: TypeAlias = (
    HeldObjectStateExpectation
    | CoordinatedHeldObjectCleanupExpectation
    | ArticulationJointStateExpectation
)


class PoseRelationExpectation(str, Enum):
    """Expected relationship to a grounded pose baseline."""

    MATCHED = "matched"
    SEPARATED = "separated"


class BinaryEvidenceKind(str, Enum):
    """Raw boolean evidence channel."""

    CONTACT = "contact"
    CONSTRAINT = "constraint"


class ScalarEvidenceKind(str, Enum):
    """Raw scalar physical evidence channel."""

    FORCE = "force"
    WRENCH = "wrench"


class ScalarExpectation(str, Enum):
    """Expected high/low magnitude band for scalar evidence."""

    PRESENT = "present"
    ABSENT = "absent"


def _validate_clause_identity(
    clause_id: str,
    expectation_id: str,
    source: EffectEvidenceSourceRef,
) -> EffectEvidenceSourceRef:
    """Validate common clause identity and own its source."""
    _validate_identifier(clause_id, field_name="effect clause_id")
    _validate_identifier(expectation_id, field_name="effect expectation_id")
    if not isinstance(source, EffectEvidenceSourceRef):
        raise TypeError("effect clause source must be an EffectEvidenceSourceRef.")
    return source.snapshot()


@dataclass(frozen=True, slots=True, eq=False)
class PoseRelationClause:
    """Object-to-endpoint pose condition with monitor-owned tolerances."""

    clause_id: str
    expectation_id: str
    source: EffectEvidenceSourceRef
    expectation: PoseRelationExpectation
    baseline_object_to_endpoint: torch.Tensor | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source",
            _validate_clause_identity(
                self.clause_id,
                self.expectation_id,
                self.source,
            ),
        )
        if not isinstance(self.expectation, PoseRelationExpectation):
            raise TypeError("expectation must be a PoseRelationExpectation.")
        baseline = self.baseline_object_to_endpoint
        if self.expectation is PoseRelationExpectation.SEPARATED:
            if baseline is None:
                raise ValueError("A separated pose clause requires a baseline.")
            object.__setattr__(
                self,
                "baseline_object_to_endpoint",
                _validate_pose_batch(
                    baseline,
                    field_name="PoseRelationClause.baseline_object_to_endpoint",
                ),
            )
        elif baseline is not None:
            raise ValueError(
                "A matched pose clause obtains its baseline from the expected "
                "held-object StateDelta and must not embed one."
            )

    def snapshot(self) -> PoseRelationClause:
        """Return an independently owned pose clause."""
        return PoseRelationClause(
            self.clause_id,
            self.expectation_id,
            self.source,
            self.expectation,
            self.baseline_object_to_endpoint,
        )


@dataclass(frozen=True, slots=True)
class BinaryEffectClause:
    """Raw contact or constraint-state condition."""

    clause_id: str
    expectation_id: str
    source: EffectEvidenceSourceRef
    evidence_kind: BinaryEvidenceKind
    expected: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source",
            _validate_clause_identity(
                self.clause_id,
                self.expectation_id,
                self.source,
            ),
        )
        if not isinstance(self.evidence_kind, BinaryEvidenceKind):
            raise TypeError("evidence_kind must be a BinaryEvidenceKind.")
        if type(self.expected) is not bool:
            raise TypeError("expected must be a bool.")

    def snapshot(self) -> BinaryEffectClause:
        """Return an independently owned binary clause."""
        return BinaryEffectClause(
            self.clause_id,
            self.expectation_id,
            self.source,
            self.evidence_kind,
            self.expected,
        )


@dataclass(frozen=True, slots=True)
class ScalarEffectClause:
    """Raw force/wrench magnitude condition with monitor-owned thresholds."""

    clause_id: str
    expectation_id: str
    source: EffectEvidenceSourceRef
    evidence_kind: ScalarEvidenceKind
    expectation: ScalarExpectation

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source",
            _validate_clause_identity(
                self.clause_id,
                self.expectation_id,
                self.source,
            ),
        )
        if not isinstance(self.evidence_kind, ScalarEvidenceKind):
            raise TypeError("evidence_kind must be a ScalarEvidenceKind.")
        if not isinstance(self.expectation, ScalarExpectation):
            raise TypeError("expectation must be a ScalarExpectation.")

    def snapshot(self) -> ScalarEffectClause:
        """Return an independently owned scalar clause."""
        return ScalarEffectClause(
            self.clause_id,
            self.expectation_id,
            self.source,
            self.evidence_kind,
            self.expectation,
        )


@dataclass(frozen=True, slots=True, eq=False)
class JointStateEffectClause:
    """Raw articulation/robot joint-position target condition."""

    clause_id: str
    expectation_id: str
    source: EffectEvidenceSourceRef
    target_position: torch.Tensor

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source",
            _validate_clause_identity(
                self.clause_id,
                self.expectation_id,
                self.source,
            ),
        )
        target = self.target_position
        if not isinstance(target, torch.Tensor) or target.dim() not in (1, 2):
            raise ValueError("target_position must have shape (J,) or (B, J).")
        if target.numel() == 0 or not target.is_floating_point():
            raise TypeError("target_position must be a non-empty floating tensor.")
        if not torch.isfinite(target).all():
            raise ValueError("target_position must be finite.")
        object.__setattr__(self, "target_position", target.clone())

    def snapshot(self) -> JointStateEffectClause:
        """Return an independently owned joint-state clause."""
        return JointStateEffectClause(
            self.clause_id,
            self.expectation_id,
            self.source,
            self.target_position,
        )


EffectClause: TypeAlias = (
    PoseRelationClause
    | BinaryEffectClause
    | ScalarEffectClause
    | JointStateEffectClause
)
_STATE_EXPECTATION_TYPES = (
    HeldObjectStateExpectation,
    CoordinatedHeldObjectCleanupExpectation,
    ArticulationJointStateExpectation,
)
_CLAUSE_TYPES = (
    PoseRelationClause,
    BinaryEffectClause,
    ScalarEffectClause,
    JointStateEffectClause,
)


@dataclass(frozen=True, slots=True, eq=False)
class SemanticEffectSpec:
    """Grounded typed physical clauses and symbolic postconditions for one call."""

    semantic_id: str
    effect_kind: SemanticEffectKind
    skill_id: str
    invocation_id: str | None
    invocation_revision: int
    env_ids: torch.Tensor
    state_expectations: tuple[EffectStateExpectation, ...]
    clauses: tuple[EffectClause, ...]

    def __post_init__(self) -> None:
        _validate_identifier(
            self.semantic_id,
            field_name="SemanticEffectSpec.semantic_id",
        )
        _validate_identifier(self.skill_id, field_name="SemanticEffectSpec.skill_id")
        if not isinstance(self.effect_kind, SemanticEffectKind):
            raise TypeError("effect_kind must be a SemanticEffectKind.")
        if self.invocation_id is not None:
            _validate_identifier(
                self.invocation_id,
                field_name="SemanticEffectSpec.invocation_id",
            )
        if type(self.invocation_revision) is not int or self.invocation_revision < 0:
            raise ValueError("invocation_revision must be a non-negative integer.")
        if not isinstance(self.env_ids, torch.Tensor):
            raise TypeError("env_ids must be a torch.Tensor.")
        if self.env_ids.dtype != torch.long or self.env_ids.dim() != 1:
            raise ValueError("env_ids must be a one-dimensional torch.long tensor.")
        if self.env_ids.numel() == 0:
            raise ValueError("env_ids must contain at least one environment ID.")
        if torch.unique(self.env_ids).numel() != self.env_ids.numel():
            raise ValueError("env_ids must be unique.")
        object.__setattr__(self, "env_ids", self.env_ids.clone())

        expectations = tuple(self.state_expectations)
        if not expectations or not all(
            type(value) in _STATE_EXPECTATION_TYPES for value in expectations
        ):
            raise TypeError(
                "state_expectations must contain exact typed state expectations."
            )
        expectation_ids = [value.expectation_id for value in expectations]
        if len(set(expectation_ids)) != len(expectation_ids):
            raise ValueError("State expectation IDs must be unique.")
        held_keys = [
            value.task_state_key
            for value in expectations
            if type(value) is HeldObjectStateExpectation
        ]
        if len(set(held_keys)) != len(held_keys):
            raise ValueError("Held-object task-state keys must be unique.")
        cleanup_keys = [
            value.task_state_keys
            for value in expectations
            if type(value) is CoordinatedHeldObjectCleanupExpectation
        ]
        if len(set(cleanup_keys)) != len(cleanup_keys):
            raise ValueError("Coordinated cleanup keys must be unique.")

        clauses = tuple(self.clauses)
        if not clauses or not all(type(value) in _CLAUSE_TYPES for value in clauses):
            raise TypeError("clauses must contain exact typed effect clauses.")
        clause_ids = [value.clause_id for value in clauses]
        if len(set(clause_ids)) != len(clause_ids):
            raise ValueError("Effect clause IDs must be unique.")
        unknown_expectations = {value.expectation_id for value in clauses}.difference(
            expectation_ids
        )
        if unknown_expectations:
            raise ValueError(
                "Effect clauses reference unknown state expectations: "
                f"{sorted(unknown_expectations)}."
            )
        uncovered = set(expectation_ids).difference(
            value.expectation_id for value in clauses
        )
        uncovered.difference_update(
            value.expectation_id
            for value in expectations
            if type(value) is CoordinatedHeldObjectCleanupExpectation
        )
        if uncovered:
            raise ValueError(
                "Every physical state expectation needs at least one clause; "
                f"missing {sorted(uncovered)}."
            )
        for value in expectations:
            if (
                type(value) is ArticulationJointStateExpectation
                and value.target_position.dim() == 2
                and value.target_position.shape[0] != self.env_ids.numel()
            ):
                raise ValueError(
                    "Batched articulation targets must match env_ids length."
                )
        for value in clauses:
            if type(value) is PoseRelationClause:
                baseline = value.baseline_object_to_endpoint
                if baseline is not None and baseline.dim() == 3:
                    if baseline.shape[0] != self.env_ids.numel():
                        raise ValueError(
                            "Batched pose baselines must match env_ids length."
                        )
                    if baseline.device != self.env_ids.device:
                        raise ValueError(
                            "Batched pose baselines and env_ids must share a device."
                        )
            elif (
                type(value) is JointStateEffectClause
                and value.target_position.dim() == 2
                and value.target_position.shape[0] != self.env_ids.numel()
            ):
                raise ValueError("Batched joint targets must match env_ids length.")

        held_relations = {
            value.relation
            for value in expectations
            if type(value) is HeldObjectStateExpectation
        }
        if self.effect_kind is SemanticEffectKind.ATTACH and held_relations != {
            HeldObjectRelation.ATTACHED
        }:
            raise ValueError("An attach effect requires only attached state.")
        if self.effect_kind is SemanticEffectKind.RELEASE and held_relations != {
            HeldObjectRelation.DETACHED
        }:
            raise ValueError("A release effect requires only detached state.")
        if self.effect_kind is SemanticEffectKind.TRANSFER and held_relations != {
            HeldObjectRelation.ATTACHED,
            HeldObjectRelation.DETACHED,
        }:
            raise ValueError("A transfer effect requires attached and detached state.")
        if self.effect_kind is SemanticEffectKind.ARTICULATION and not any(
            type(value) is ArticulationJointStateExpectation for value in expectations
        ):
            raise ValueError(
                "An articulation effect requires an articulation-joint expectation."
            )

        object.__setattr__(
            self,
            "state_expectations",
            tuple(value.snapshot() for value in expectations),
        )
        object.__setattr__(
            self,
            "clauses",
            tuple(value.snapshot() for value in clauses),
        )

    def snapshot(self) -> SemanticEffectSpec:
        """Return an independently owned grounded effect contract."""
        return SemanticEffectSpec(
            semantic_id=self.semantic_id,
            effect_kind=self.effect_kind,
            skill_id=self.skill_id,
            invocation_id=self.invocation_id,
            invocation_revision=self.invocation_revision,
            env_ids=self.env_ids,
            state_expectations=self.state_expectations,
            clauses=self.clauses,
        )

    def to_metadata(self) -> dict[str, object]:
        """Return this grounded effect contract as deterministic JSON-safe data."""
        return {
            "semantic_id": self.semantic_id,
            "effect_kind": self.effect_kind.value,
            "skill_id": self.skill_id,
            "invocation_id": self.invocation_id,
            "invocation_revision": self.invocation_revision,
            "env_ids": _metadata_value(self.env_ids),
            "state_expectations": [
                _metadata_value(value) for value in self.state_expectations
            ],
            "clauses": [_metadata_value(value) for value in self.clauses],
        }

    def state_expectation(self, expectation_id: str) -> EffectStateExpectation:
        """Return an owned state expectation by effect-local ID."""
        for value in self.state_expectations:
            if value.expectation_id == expectation_id:
                return value.snapshot()
        raise KeyError(f"Unknown effect state expectation {expectation_id!r}.")

    def validate_request(self, request: EffectVerificationRequest) -> None:
        """Validate execution identity and typed symbolic postconditions."""
        if not isinstance(request, EffectVerificationRequest):
            raise TypeError("request must be an EffectVerificationRequest.")
        if request.skill_id != self.skill_id:
            raise ValueError("Effect request skill_id does not match the spec.")
        if request.invocation_id != self.invocation_id:
            raise ValueError("Effect request invocation_id does not match the spec.")
        if request.invocation_revision != self.invocation_revision:
            raise ValueError(
                "Effect request invocation_revision does not match the spec."
            )
        if request.env_mask.shape != self.env_ids.shape:
            raise ValueError("Effect request row count does not match spec env_ids.")
        if request.env_mask.device != self.env_ids.device:
            raise ValueError(
                "Effect request mask and spec env_ids must share a device."
            )

        held_expectations = {
            value.task_state_key: value
            for value in self.state_expectations
            if type(value) is HeldObjectStateExpectation
        }
        expected_held = request.expected_effects.held_object_updates
        if set(expected_held) != set(held_expectations):
            raise ValueError(
                "Effect request held-object updates must exactly match typed "
                "state expectation keys."
            )
        for task_state_key, expectation in held_expectations.items():
            candidate = expected_held[task_state_key]
            if expectation.relation is HeldObjectRelation.DETACHED:
                if candidate is not None:
                    raise ValueError(
                        f"Detached expectation {expectation.expectation_id!r} must "
                        "remove its held-object state."
                    )
                continue
            if not isinstance(candidate, HeldObjectState):
                raise ValueError(
                    f"Attached expectation {expectation.expectation_id!r} requires "
                    "a HeldObjectState postcondition."
                )
            if candidate.semantics.entity_id != expectation.object_id:
                raise ValueError(
                    f"Attached expectation {expectation.expectation_id!r} targets "
                    "the wrong canonical object."
                )
            if candidate.object_to_eef.device != request.env_mask.device:
                raise ValueError(
                    "Attached postcondition poses and request rows must share a device."
                )
            if (
                candidate.object_to_eef.dim() == 3
                and candidate.object_to_eef.shape[0] != self.env_ids.numel()
            ):
                raise ValueError(
                    "Batched attached postcondition poses must match spec env_ids."
                )
            _validate_pose_batch(
                candidate.object_to_eef,
                field_name=(
                    f"Attached expectation {expectation.expectation_id!r} pose"
                ),
            )
            if candidate.env_mask is not None:
                if (
                    candidate.env_mask.shape != request.env_mask.shape
                    or candidate.env_mask.device != request.env_mask.device
                ):
                    raise ValueError(
                        "Attached postcondition masks must match request rows and device."
                    )
                if (request.env_mask & ~candidate.env_mask).any():
                    raise ValueError(
                        "Attached postconditions must cover every requested row."
                    )

        cleanup_expectations = {
            value.task_state_keys
            for value in self.state_expectations
            if type(value) is CoordinatedHeldObjectCleanupExpectation
        }
        expected_cleanup = request.expected_effects.coordinated_held_object_updates
        if set(expected_cleanup) != cleanup_expectations:
            raise ValueError(
                "Effect request coordinated updates must exactly match typed "
                "cleanup expectations."
            )
        if any(value is not None for value in expected_cleanup.values()):
            raise ValueError(
                "Coordinated held-object cleanup expectations may only remove state."
            )

        articulation_expectations = {
            (value.articulation_id, value.joint_id): value
            for value in self.state_expectations
            if type(value) is ArticulationJointStateExpectation
        }
        articulation_updates = request.expected_effects.articulation_joint_updates
        if set(articulation_updates) != set(articulation_expectations):
            raise ValueError(
                "Articulation-joint updates must exactly match typed state "
                "expectations."
            )
        for key, expectation in articulation_expectations.items():
            candidate = articulation_updates[key]
            if not isinstance(candidate, ArticulationJointState):
                raise ValueError(
                    f"Articulation expectation {expectation.expectation_id!r} "
                    "requires an ArticulationJointState postcondition."
                )
            if candidate.position.device != request.env_mask.device:
                raise ValueError(
                    "Articulation postconditions and request rows must share a device."
                )
            if candidate.position.dim() == 2:
                if candidate.position.shape[0] != self.env_ids.numel():
                    raise ValueError(
                        "Batched articulation postconditions must match spec env_ids."
                    )
                positions = candidate.position
            else:
                positions = candidate.position.unsqueeze(0).expand(
                    self.env_ids.numel(), -1
                )
            target = expectation.target_position
            if target.device != positions.device or target.dtype != positions.dtype:
                raise ValueError(
                    "Articulation postconditions must match target device and dtype."
                )
            if target.dim() == 1:
                target = target.unsqueeze(0).expand(self.env_ids.numel(), -1)
            if positions.shape != target.shape or not torch.equal(
                positions[request.env_mask],
                target[request.env_mask],
            ):
                raise ValueError(
                    f"Articulation expectation {expectation.expectation_id!r} "
                    "postcondition does not match its target position."
                )
            if candidate.env_mask is not None:
                if (
                    candidate.env_mask.shape != request.env_mask.shape
                    or candidate.env_mask.device != request.env_mask.device
                ):
                    raise ValueError(
                        "Articulation postcondition masks must match request rows "
                        "and device."
                    )
                if (request.env_mask & ~candidate.env_mask).any():
                    raise ValueError(
                        "Articulation postconditions must cover every requested row."
                    )


def _validate_evidence_common(
    *,
    evidence_id: str,
    valid: torch.Tensor,
    acquisition_errors: tuple[str | None, ...],
    timestamp: float,
    env_ids: torch.Tensor,
    observation_revision: int,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, tuple[str | None, ...], float, torch.Tensor]:
    """Validate and own fields shared by every raw evidence batch."""
    _validate_identifier(evidence_id, field_name="effect evidence_id")
    if not isinstance(valid, torch.Tensor):
        raise TypeError("valid must be a torch.Tensor.")
    if valid.dtype != torch.bool or valid.shape != (batch_size,):
        raise ValueError("valid must be a bool tensor with shape (B,).")
    if valid.device != device:
        raise ValueError("valid and evidence payload must share a device.")
    if not isinstance(env_ids, torch.Tensor):
        raise TypeError("env_ids must be a torch.Tensor.")
    if env_ids.dtype != torch.long or env_ids.shape != (batch_size,):
        raise ValueError("env_ids must be a torch.long tensor with shape (B,).")
    if env_ids.device != device:
        raise ValueError("env_ids and evidence payload must share a device.")
    if torch.unique(env_ids).numel() != env_ids.numel():
        raise ValueError("Evidence env_ids must be unique.")
    errors = tuple(acquisition_errors)
    if len(errors) != batch_size:
        raise ValueError("acquisition_errors must contain one entry per row.")
    for row, (row_valid, error) in enumerate(zip(valid.tolist(), errors)):
        if row_valid and error is not None:
            raise ValueError(f"Valid evidence row {row} must not carry an error.")
        if not row_valid and (
            type(error) is not str or not error or error != error.strip()
        ):
            raise ValueError(f"Invalid evidence row {row} requires a non-empty error.")
    if not isinstance(timestamp, (int, float)) or isinstance(timestamp, bool):
        raise TypeError("timestamp must be a number.")
    normalized_timestamp = float(timestamp)
    if not math.isfinite(normalized_timestamp) or normalized_timestamp < 0.0:
        raise ValueError("timestamp must be finite and non-negative.")
    if type(observation_revision) is not int or observation_revision < 0:
        raise ValueError("observation_revision must be a non-negative integer.")
    return valid.clone(), errors, normalized_timestamp, env_ids.clone()


@dataclass(frozen=True, slots=True, eq=False)
class PoseRelationEvidenceBatch:
    """Raw object-to-endpoint transform observations."""

    evidence_id: str
    object_to_endpoint: torch.Tensor
    valid: torch.Tensor
    acquisition_errors: tuple[str | None, ...]
    timestamp: float
    env_ids: torch.Tensor
    observation_revision: int

    def __post_init__(self) -> None:
        poses = self.object_to_endpoint
        if not isinstance(poses, torch.Tensor) or (
            poses.dim() != 3 or poses.shape[0] == 0 or poses.shape[-2:] != (4, 4)
        ):
            raise ValueError("object_to_endpoint must have shape (B, 4, 4).")
        if not poses.is_floating_point():
            raise TypeError("object_to_endpoint must use a floating-point dtype.")
        valid, errors, timestamp, env_ids = _validate_evidence_common(
            evidence_id=self.evidence_id,
            valid=self.valid,
            acquisition_errors=self.acquisition_errors,
            timestamp=self.timestamp,
            env_ids=self.env_ids,
            observation_revision=self.observation_revision,
            batch_size=poses.shape[0],
            device=poses.device,
        )
        object.__setattr__(
            self,
            "object_to_endpoint",
            _validate_pose_batch(
                poses,
                field_name="Valid pose-relation evidence",
                valid_mask=valid,
            ),
        )
        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "acquisition_errors", errors)
        object.__setattr__(self, "timestamp", timestamp)
        object.__setattr__(self, "env_ids", env_ids)

    def snapshot(self) -> PoseRelationEvidenceBatch:
        """Return an independently owned evidence batch."""
        return PoseRelationEvidenceBatch(
            self.evidence_id,
            self.object_to_endpoint,
            self.valid,
            self.acquisition_errors,
            self.timestamp,
            self.env_ids,
            self.observation_revision,
        )

    def to_metadata(self) -> dict[str, object]:
        """Return raw pose evidence as JSON-safe trace metadata."""
        return _evidence_metadata(
            self,
            payload={"object_to_endpoint": self.object_to_endpoint},
        )


@dataclass(frozen=True, slots=True, eq=False)
class BinaryEffectEvidenceBatch:
    """Raw per-row contact or constraint-state observations."""

    evidence_id: str
    evidence_kind: BinaryEvidenceKind
    values: torch.Tensor
    valid: torch.Tensor
    acquisition_errors: tuple[str | None, ...]
    timestamp: float
    env_ids: torch.Tensor
    observation_revision: int

    def __post_init__(self) -> None:
        if not isinstance(self.evidence_kind, BinaryEvidenceKind):
            raise TypeError("evidence_kind must be a BinaryEvidenceKind.")
        values = self.values
        if not isinstance(values, torch.Tensor):
            raise TypeError("values must be a torch.Tensor.")
        if values.dtype != torch.bool or values.dim() != 1 or values.numel() == 0:
            raise ValueError("binary evidence values must have bool shape (B,).")
        valid, errors, timestamp, env_ids = _validate_evidence_common(
            evidence_id=self.evidence_id,
            valid=self.valid,
            acquisition_errors=self.acquisition_errors,
            timestamp=self.timestamp,
            env_ids=self.env_ids,
            observation_revision=self.observation_revision,
            batch_size=values.shape[0],
            device=values.device,
        )
        object.__setattr__(self, "values", values.clone())
        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "acquisition_errors", errors)
        object.__setattr__(self, "timestamp", timestamp)
        object.__setattr__(self, "env_ids", env_ids)

    def snapshot(self) -> BinaryEffectEvidenceBatch:
        """Return an independently owned evidence batch."""
        return BinaryEffectEvidenceBatch(
            self.evidence_id,
            self.evidence_kind,
            self.values,
            self.valid,
            self.acquisition_errors,
            self.timestamp,
            self.env_ids,
            self.observation_revision,
        )

    def to_metadata(self) -> dict[str, object]:
        """Return raw binary evidence as JSON-safe trace metadata."""
        return _evidence_metadata(
            self,
            payload={
                "evidence_kind": self.evidence_kind.value,
                "values": self.values,
            },
        )


@dataclass(frozen=True, slots=True, eq=False)
class ScalarEffectEvidenceBatch:
    """Raw per-row force or wrench-magnitude observations."""

    evidence_id: str
    evidence_kind: ScalarEvidenceKind
    values: torch.Tensor
    valid: torch.Tensor
    acquisition_errors: tuple[str | None, ...]
    timestamp: float
    env_ids: torch.Tensor
    observation_revision: int

    def __post_init__(self) -> None:
        if not isinstance(self.evidence_kind, ScalarEvidenceKind):
            raise TypeError("evidence_kind must be a ScalarEvidenceKind.")
        values = self.values
        if not isinstance(values, torch.Tensor):
            raise TypeError("values must be a torch.Tensor.")
        if values.dim() != 1 or values.numel() == 0 or not values.is_floating_point():
            raise ValueError("scalar evidence values must have floating shape (B,).")
        valid, errors, timestamp, env_ids = _validate_evidence_common(
            evidence_id=self.evidence_id,
            valid=self.valid,
            acquisition_errors=self.acquisition_errors,
            timestamp=self.timestamp,
            env_ids=self.env_ids,
            observation_revision=self.observation_revision,
            batch_size=values.shape[0],
            device=values.device,
        )
        if not torch.isfinite(values[valid]).all():
            raise ValueError("Valid scalar evidence values must be finite.")
        object.__setattr__(self, "values", values.clone())
        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "acquisition_errors", errors)
        object.__setattr__(self, "timestamp", timestamp)
        object.__setattr__(self, "env_ids", env_ids)

    def snapshot(self) -> ScalarEffectEvidenceBatch:
        """Return an independently owned evidence batch."""
        return ScalarEffectEvidenceBatch(
            self.evidence_id,
            self.evidence_kind,
            self.values,
            self.valid,
            self.acquisition_errors,
            self.timestamp,
            self.env_ids,
            self.observation_revision,
        )

    def to_metadata(self) -> dict[str, object]:
        """Return raw scalar evidence as JSON-safe trace metadata."""
        return _evidence_metadata(
            self,
            payload={
                "evidence_kind": self.evidence_kind.value,
                "values": self.values,
            },
        )


@dataclass(frozen=True, slots=True, eq=False)
class JointStateEvidenceBatch:
    """Raw per-row joint position/velocity observations."""

    evidence_id: str
    positions: torch.Tensor
    velocities: torch.Tensor | None
    valid: torch.Tensor
    acquisition_errors: tuple[str | None, ...]
    timestamp: float
    env_ids: torch.Tensor
    observation_revision: int

    def __post_init__(self) -> None:
        positions = self.positions
        if not isinstance(positions, torch.Tensor):
            raise TypeError("positions must be a torch.Tensor.")
        if (
            positions.dim() != 2
            or positions.shape[0] == 0
            or positions.shape[1] == 0
            or not positions.is_floating_point()
        ):
            raise ValueError("positions must have non-empty floating shape (B, J).")
        valid, errors, timestamp, env_ids = _validate_evidence_common(
            evidence_id=self.evidence_id,
            valid=self.valid,
            acquisition_errors=self.acquisition_errors,
            timestamp=self.timestamp,
            env_ids=self.env_ids,
            observation_revision=self.observation_revision,
            batch_size=positions.shape[0],
            device=positions.device,
        )
        if not torch.isfinite(positions[valid]).all():
            raise ValueError("Valid joint positions must be finite.")
        velocities = self.velocities
        if velocities is not None:
            if not isinstance(velocities, torch.Tensor):
                raise TypeError("velocities must be a torch.Tensor or None.")
            if (
                velocities.shape != positions.shape
                or velocities.device != positions.device
            ):
                raise ValueError("velocities must match positions shape and device.")
            if not velocities.is_floating_point():
                raise TypeError("velocities must use a floating-point dtype.")
            if not torch.isfinite(velocities[valid]).all():
                raise ValueError("Valid joint velocities must be finite.")
        object.__setattr__(self, "positions", positions.clone())
        object.__setattr__(
            self,
            "velocities",
            None if velocities is None else velocities.clone(),
        )
        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "acquisition_errors", errors)
        object.__setattr__(self, "timestamp", timestamp)
        object.__setattr__(self, "env_ids", env_ids)

    def snapshot(self) -> JointStateEvidenceBatch:
        """Return an independently owned evidence batch."""
        return JointStateEvidenceBatch(
            self.evidence_id,
            self.positions,
            self.velocities,
            self.valid,
            self.acquisition_errors,
            self.timestamp,
            self.env_ids,
            self.observation_revision,
        )

    def to_metadata(self) -> dict[str, object]:
        """Return raw joint-state evidence as JSON-safe trace metadata."""
        return _evidence_metadata(
            self,
            payload={
                "positions": self.positions,
                "velocities": self.velocities,
            },
        )


def _evidence_metadata(
    batch: EffectEvidenceBatch,
    *,
    payload: Mapping[str, object],
) -> dict[str, object]:
    """Serialize fields shared by all raw physical-evidence batches."""
    return {
        "evidence_id": batch.evidence_id,
        **{key: _metadata_value(value) for key, value in payload.items()},
        "valid_mask": _metadata_value(batch.valid),
        "acquisition_errors": list(batch.acquisition_errors),
        "timestamp": batch.timestamp,
        "env_ids": _metadata_value(batch.env_ids),
        "observation_revision": batch.observation_revision,
    }


EffectEvidenceBatch: TypeAlias = (
    PoseRelationEvidenceBatch
    | BinaryEffectEvidenceBatch
    | ScalarEffectEvidenceBatch
    | JointStateEvidenceBatch
)
_EVIDENCE_TYPES = (
    PoseRelationEvidenceBatch,
    BinaryEffectEvidenceBatch,
    ScalarEffectEvidenceBatch,
    JointStateEvidenceBatch,
)


@dataclass(frozen=True, slots=True, eq=False)
class EffectExpectationDecision:
    """Per-row outcome for one physical state expectation.

    Rows absent from both ``satisfied_mask`` and ``contradicted_mask`` remain
    unresolved.  ``inverse_satisfied_mask`` is deliberately stronger than
    contradiction: it requires every clause in the expectation group to have
    reached its explicit inverse band for the configured consecutive-sample
    window.  This distinction lets failure reconciliation retain a relation
    only from complete inverse evidence rather than from one contradictory
    clause.
    """

    expectation_id: str
    satisfied_mask: torch.Tensor
    contradicted_mask: torch.Tensor
    inverse_satisfied_mask: torch.Tensor

    def __post_init__(self) -> None:
        _validate_identifier(
            self.expectation_id,
            field_name="EffectExpectationDecision.expectation_id",
        )
        for field_name in (
            "satisfied_mask",
            "contradicted_mask",
            "inverse_satisfied_mask",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{field_name} must be a torch.Tensor.")
            if value.dtype != torch.bool or value.dim() != 1:
                raise ValueError(f"{field_name} must be a one-dimensional bool tensor.")
        masks = (
            self.satisfied_mask,
            self.contradicted_mask,
            self.inverse_satisfied_mask,
        )
        if any(value.shape != masks[0].shape for value in masks[1:]):
            raise ValueError("Expectation decision masks must have equal shapes.")
        if any(value.device != masks[0].device for value in masks[1:]):
            raise ValueError("Expectation decision masks must use the same device.")
        if (self.satisfied_mask & self.contradicted_mask).any():
            raise ValueError("satisfied_mask and contradicted_mask must not overlap.")
        if (self.inverse_satisfied_mask & ~self.contradicted_mask).any():
            raise ValueError(
                "inverse_satisfied_mask must be a subset of contradicted_mask."
            )
        object.__setattr__(self, "satisfied_mask", self.satisfied_mask.clone())
        object.__setattr__(
            self,
            "contradicted_mask",
            self.contradicted_mask.clone(),
        )
        object.__setattr__(
            self,
            "inverse_satisfied_mask",
            self.inverse_satisfied_mask.clone(),
        )

    def snapshot(self) -> EffectExpectationDecision:
        """Return an independently owned expectation outcome."""
        return EffectExpectationDecision(
            expectation_id=self.expectation_id,
            satisfied_mask=self.satisfied_mask,
            contradicted_mask=self.contradicted_mask,
            inverse_satisfied_mask=self.inverse_satisfied_mask,
        )


@dataclass(frozen=True, slots=True, eq=False)
class EffectMonitorDecision:
    """Uncorrelated aggregate and per-expectation monitor decision.

    When ``expectation_decisions`` is non-empty, the aggregate masks are
    authoritative reductions of that current observation: success is the
    conjunction of every satisfied mask and failure is the union of every
    contradicted mask.  This prevents callers from combining expectation
    outcomes observed on different ticks.
    """

    success_mask: torch.Tensor
    failure_mask: torch.Tensor
    expectation_decisions: tuple[EffectExpectationDecision, ...] = ()

    def __post_init__(self) -> None:
        for field_name in ("success_mask", "failure_mask"):
            value = getattr(self, field_name)
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{field_name} must be a torch.Tensor.")
            if value.dtype != torch.bool or value.dim() != 1:
                raise ValueError(f"{field_name} must be a one-dimensional bool tensor.")
        if self.success_mask.shape != self.failure_mask.shape:
            raise ValueError("Decision masks must have equal shapes.")
        if self.success_mask.device != self.failure_mask.device:
            raise ValueError("Decision masks must use the same device.")
        if (self.success_mask & self.failure_mask).any():
            raise ValueError("Decision masks must not overlap.")
        expectation_decisions = tuple(self.expectation_decisions)
        if not all(
            type(value) is EffectExpectationDecision for value in expectation_decisions
        ):
            raise TypeError(
                "expectation_decisions must contain exact "
                "EffectExpectationDecision values."
            )
        expectation_ids = [value.expectation_id for value in expectation_decisions]
        if len(set(expectation_ids)) != len(expectation_ids):
            raise ValueError("Expectation decision IDs must be unique.")
        if expectation_decisions:
            for value in expectation_decisions:
                if value.satisfied_mask.shape != self.success_mask.shape:
                    raise ValueError(
                        "Expectation and aggregate decision masks must have "
                        "equal shapes."
                    )
                if value.satisfied_mask.device != self.success_mask.device:
                    raise ValueError(
                        "Expectation and aggregate decision masks must use the "
                        "same device."
                    )
            expected_success = torch.ones_like(self.success_mask)
            expected_failure = torch.zeros_like(self.failure_mask)
            for value in expectation_decisions:
                expected_success &= value.satisfied_mask
                expected_failure |= value.contradicted_mask
            if not torch.equal(self.success_mask, expected_success):
                raise ValueError(
                    "success_mask must equal the conjunction of expectation "
                    "satisfied masks."
                )
            if not torch.equal(self.failure_mask, expected_failure):
                raise ValueError(
                    "failure_mask must equal the union of expectation "
                    "contradicted masks."
                )
        object.__setattr__(self, "success_mask", self.success_mask.clone())
        object.__setattr__(self, "failure_mask", self.failure_mask.clone())
        object.__setattr__(
            self,
            "expectation_decisions",
            tuple(value.snapshot() for value in expectation_decisions),
        )


class EffectMonitor(ABC):
    """Stateful verifier owned by one grounded semantic call."""

    @property
    @abstractmethod
    def spec(self) -> SemanticEffectSpec:
        """Return an independently owned effect contract."""

    @property
    @abstractmethod
    def resolved_params(self) -> Mapping[str, EffectMonitorParam]:
        """Return all resolved monitor thresholds for trace metadata."""

    @abstractmethod
    def observe(
        self,
        request: EffectVerificationRequest,
        evidence: Mapping[str, EffectEvidenceBatch],
    ) -> EffectMonitorDecision:
        """Consume one synchronized raw observation and decide requested rows."""


class EffectMonitorFactory(ABC):
    """Versioned constructor for independent semantic-effect monitors."""

    monitor_id: ClassVar[str]
    revision: ClassVar[str]

    @abstractmethod
    def validate_ref(self, ref: EffectMonitorRef) -> None:
        """Validate one reference without providers or state creation."""

    @abstractmethod
    def create(
        self,
        spec: SemanticEffectSpec,
        ref: EffectMonitorRef,
    ) -> EffectMonitor:
        """Create one independent monitor for ``spec`` and ``ref``."""


def _same_tensor_value(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Return whether tensors have identical placement, type, shape, and value."""
    return (
        left.device == right.device
        and left.dtype == right.dtype
        and left.shape == right.shape
        and torch.equal(left, right)
    )


def _same_source(
    left: EffectEvidenceSourceRef,
    right: EffectEvidenceSourceRef,
) -> bool:
    return left.source_fingerprint == right.source_fingerprint


def _same_state_expectation(
    left: EffectStateExpectation,
    right: EffectStateExpectation,
) -> bool:
    if type(left) is not type(right):
        return False
    if type(left) is HeldObjectStateExpectation:
        assert type(right) is HeldObjectStateExpectation
        return left == right
    if type(left) is CoordinatedHeldObjectCleanupExpectation:
        assert type(right) is CoordinatedHeldObjectCleanupExpectation
        return left == right
    assert type(left) is ArticulationJointStateExpectation
    assert type(right) is ArticulationJointStateExpectation
    return (
        left.expectation_id == right.expectation_id
        and left.articulation_id == right.articulation_id
        and left.joint_id == right.joint_id
        and _same_tensor_value(left.target_position, right.target_position)
    )


def _same_clause(left: EffectClause, right: EffectClause) -> bool:
    if type(left) is not type(right):
        return False
    if (
        left.clause_id != right.clause_id
        or left.expectation_id != right.expectation_id
        or not _same_source(left.source, right.source)
    ):
        return False
    if type(left) is PoseRelationClause:
        assert type(right) is PoseRelationClause
        if left.expectation is not right.expectation:
            return False
        left_baseline = left.baseline_object_to_endpoint
        right_baseline = right.baseline_object_to_endpoint
        if left_baseline is None or right_baseline is None:
            return left_baseline is None and right_baseline is None
        return _same_tensor_value(left_baseline, right_baseline)
    if type(left) is BinaryEffectClause:
        assert type(right) is BinaryEffectClause
        return (
            left.evidence_kind is right.evidence_kind
            and left.expected is right.expected
        )
    if type(left) is ScalarEffectClause:
        assert type(right) is ScalarEffectClause
        return (
            left.evidence_kind is right.evidence_kind
            and left.expectation is right.expectation
        )
    assert type(left) is JointStateEffectClause
    assert type(right) is JointStateEffectClause
    return _same_tensor_value(left.target_position, right.target_position)


def _same_effect_spec(left: SemanticEffectSpec, right: SemanticEffectSpec) -> bool:
    """Return whether grounded typed effect specs are exactly equivalent."""
    return (
        left.semantic_id == right.semantic_id
        and left.effect_kind is right.effect_kind
        and left.skill_id == right.skill_id
        and left.invocation_id == right.invocation_id
        and left.invocation_revision == right.invocation_revision
        and _same_tensor_value(left.env_ids, right.env_ids)
        and len(left.state_expectations) == len(right.state_expectations)
        and all(
            _same_state_expectation(left_value, right_value)
            for left_value, right_value in zip(
                left.state_expectations,
                right.state_expectations,
                strict=True,
            )
        )
        and len(left.clauses) == len(right.clauses)
        and all(
            _same_clause(left_value, right_value)
            for left_value, right_value in zip(
                left.clauses,
                right.clauses,
                strict=True,
            )
        )
    )


class EffectMonitorRegistry:
    """Immutable exact-ID/revision registry of monitor factories."""

    __slots__ = ("_factories",)

    def __init__(self, factories: Iterable[EffectMonitorFactory] = ()) -> None:
        normalized: dict[tuple[str, str], EffectMonitorFactory] = {}
        for factory in factories:
            if not isinstance(factory, EffectMonitorFactory):
                raise TypeError("factories must contain EffectMonitorFactory objects.")
            monitor_id = _validate_identifier(
                factory.monitor_id,
                field_name="EffectMonitorFactory.monitor_id",
            )
            revision = _validate_identifier(
                factory.revision,
                field_name="EffectMonitorFactory.revision",
            )
            key = monitor_id, revision
            if key in normalized:
                raise ValueError(f"Duplicate effect-monitor factory {key!r}.")
            normalized[key] = factory
        self._factories = MappingProxyType(normalized)

    @property
    def factories(self) -> Mapping[tuple[str, str], EffectMonitorFactory]:
        """Return the immutable exact-key factory mapping."""
        return self._factories

    def resolve(self, ref: EffectMonitorRef) -> EffectMonitorFactory:
        """Resolve the exact factory named by a declarative reference."""
        if not isinstance(ref, EffectMonitorRef):
            raise TypeError("ref must be an EffectMonitorRef.")
        key = ref.monitor_id, ref.revision
        try:
            return self._factories[key]
        except KeyError as exc:
            raise KeyError(f"Unknown effect-monitor factory {key!r}.") from exc

    def validate_ref(self, ref: EffectMonitorRef) -> None:
        """Validate a reference provider-free through its exact factory."""
        self.resolve(ref).validate_ref(ref)

    def create(
        self,
        spec: SemanticEffectSpec,
        ref: EffectMonitorRef,
    ) -> EffectMonitor:
        """Create one independent monitor through exact factory lookup."""
        if not isinstance(spec, SemanticEffectSpec):
            raise TypeError("spec must be a SemanticEffectSpec.")
        factory = self.resolve(ref)
        factory.validate_ref(ref)
        monitor = factory.create(spec, ref)
        if not isinstance(monitor, EffectMonitor):
            raise TypeError(
                "EffectMonitorFactory.create() must return an EffectMonitor."
            )
        monitor_spec = monitor.spec
        if not isinstance(monitor_spec, SemanticEffectSpec):
            raise TypeError("EffectMonitor.spec must be a SemanticEffectSpec.")
        if monitor_spec is spec:
            raise TypeError(
                "EffectMonitor.spec must return an independently owned contract."
            )
        if not _same_effect_spec(monitor_spec, spec):
            raise ValueError(
                "EffectMonitorFactory created a monitor for a different effect spec."
            )
        return monitor


@dataclass(frozen=True, slots=True)
class CompositeEffectMonitorCfg:
    """Strict hysteresis policy for typed pose/binary/scalar/joint clauses."""

    attached_translation_threshold: float = 0.02
    attached_rotation_threshold: float = 0.20
    detached_translation_threshold: float = 0.05
    detached_rotation_threshold: float = 0.50
    force_absent_threshold: float = 0.20
    force_present_threshold: float = 1.00
    joint_success_tolerance: float = 0.02
    joint_failure_tolerance: float = 0.10
    consecutive_samples: int = 2

    def __post_init__(self) -> None:
        for field_name in (
            "attached_translation_threshold",
            "attached_rotation_threshold",
            "detached_translation_threshold",
            "detached_rotation_threshold",
            "force_absent_threshold",
            "force_present_threshold",
            "joint_success_tolerance",
            "joint_failure_tolerance",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"{field_name} must be a number.")
            if not math.isfinite(float(value)) or value < 0.0:
                raise ValueError(f"{field_name} must be finite and non-negative.")
            object.__setattr__(self, field_name, float(value))
        if self.attached_translation_threshold >= self.detached_translation_threshold:
            raise ValueError(
                "attached_translation_threshold must be less than "
                "detached_translation_threshold."
            )
        if self.attached_rotation_threshold >= self.detached_rotation_threshold:
            raise ValueError(
                "attached_rotation_threshold must be less than "
                "detached_rotation_threshold."
            )
        if self.detached_rotation_threshold > math.pi:
            raise ValueError("detached_rotation_threshold must not exceed pi.")
        if self.force_absent_threshold >= self.force_present_threshold:
            raise ValueError(
                "force_absent_threshold must be less than force_present_threshold."
            )
        if self.joint_success_tolerance >= self.joint_failure_tolerance:
            raise ValueError(
                "joint_success_tolerance must be less than joint_failure_tolerance."
            )
        if type(self.consecutive_samples) is not int or self.consecutive_samples <= 0:
            raise ValueError("consecutive_samples must be a positive integer.")

    @classmethod
    def from_params(
        cls,
        params: Mapping[str, EffectMonitorParam],
    ) -> CompositeEffectMonitorCfg:
        """Decode strict declarative factory parameters."""
        allowed = {
            "attached_translation_threshold",
            "attached_rotation_threshold",
            "detached_translation_threshold",
            "detached_rotation_threshold",
            "force_absent_threshold",
            "force_present_threshold",
            "joint_success_tolerance",
            "joint_failure_tolerance",
            "consecutive_samples",
        }
        unknown = set(params).difference(allowed)
        if unknown:
            raise ValueError(
                f"Unknown composite effect monitor parameters: {sorted(unknown)}."
            )
        return cls(**dict(params))  # type: ignore[arg-type]

    def to_metadata(self) -> dict[str, object]:
        """Return every resolved hysteresis threshold as JSON-safe data."""
        return {
            "attached_translation_threshold": self.attached_translation_threshold,
            "attached_rotation_threshold": self.attached_rotation_threshold,
            "detached_translation_threshold": self.detached_translation_threshold,
            "detached_rotation_threshold": self.detached_rotation_threshold,
            "force_absent_threshold": self.force_absent_threshold,
            "force_present_threshold": self.force_present_threshold,
            "joint_success_tolerance": self.joint_success_tolerance,
            "joint_failure_tolerance": self.joint_failure_tolerance,
            "consecutive_samples": self.consecutive_samples,
        }


def _pose_errors(
    observed: torch.Tensor,
    baseline: torch.Tensor,
) -> tuple[float, float]:
    baseline = baseline.to(device=observed.device, dtype=observed.dtype)
    translation = torch.linalg.vector_norm(observed[:3, 3] - baseline[:3, 3])
    relative_rotation = baseline[:3, :3].transpose(0, 1) @ observed[:3, :3]
    cosine = torch.clamp((torch.trace(relative_rotation) - 1.0) * 0.5, -1.0, 1.0)
    rotation = torch.acos(cosine)
    return float(translation.item()), float(rotation.item())


class CompositeEffectMonitor(EffectMonitor):
    """Stateful conjunction monitor over typed physical evidence clauses."""

    def __init__(
        self,
        spec: SemanticEffectSpec,
        cfg: CompositeEffectMonitorCfg,
    ) -> None:
        if not isinstance(spec, SemanticEffectSpec):
            raise TypeError("spec must be a SemanticEffectSpec.")
        if not isinstance(cfg, CompositeEffectMonitorCfg):
            raise TypeError("cfg must be a CompositeEffectMonitorCfg.")
        self._spec = spec.snapshot()
        self._cfg = cfg
        self._attempt_generation: int | None = None
        self._active_env_ids: frozenset[int] = frozenset()
        self._success_counts: dict[tuple[str, int], int] = {}
        self._failure_counts: dict[tuple[str, int], int] = {}
        self._inverse_success_counts: dict[tuple[str, int], int] = {}
        self._last_observations: dict[int, tuple[float, int]] = {}

    @property
    def spec(self) -> SemanticEffectSpec:
        """Return an independently owned effect contract."""
        return self._spec.snapshot()

    @property
    def resolved_params(self) -> Mapping[str, EffectMonitorParam]:
        """Return all effective typed-clause thresholds, including defaults."""
        return MappingProxyType(self._cfg.to_metadata())

    def _prepare_request(self, request: EffectVerificationRequest) -> None:
        self._spec.validate_request(request)
        active_env_ids = frozenset(
            int(value)
            for value in self._spec.env_ids[request.env_mask].detach().cpu().tolist()
        )
        if self._attempt_generation != request.attempt_generation:
            self._attempt_generation = request.attempt_generation
            self._active_env_ids = active_env_ids
            self._success_counts.clear()
            self._failure_counts.clear()
            self._inverse_success_counts.clear()
            self._last_observations.clear()
            return
        if not active_env_ids.issubset(self._active_env_ids):
            raise ValueError(
                "An effect-verification request may only shrink within one "
                "attempt_generation."
            )
        self._active_env_ids = active_env_ids
        self._success_counts = {
            key: count
            for key, count in self._success_counts.items()
            if key[1] in active_env_ids
        }
        self._failure_counts = {
            key: count
            for key, count in self._failure_counts.items()
            if key[1] in active_env_ids
        }
        self._inverse_success_counts = {
            key: count
            for key, count in self._inverse_success_counts.items()
            if key[1] in active_env_ids
        }
        self._last_observations = {
            env_id: observation
            for env_id, observation in self._last_observations.items()
            if env_id in active_env_ids
        }

    @staticmethod
    def _validate_evidence_type(
        clause: EffectClause,
        batch: EffectEvidenceBatch,
    ) -> None:
        if type(clause) is PoseRelationClause:
            if type(batch) is not PoseRelationEvidenceBatch:
                raise TypeError("PoseRelationClause requires pose evidence.")
            return
        if type(clause) is BinaryEffectClause:
            if type(batch) is not BinaryEffectEvidenceBatch:
                raise TypeError("BinaryEffectClause requires binary evidence.")
            if batch.evidence_kind is not clause.evidence_kind:
                raise ValueError("Binary evidence kind does not match its clause.")
            return
        if type(clause) is ScalarEffectClause:
            if type(batch) is not ScalarEffectEvidenceBatch:
                raise TypeError("ScalarEffectClause requires scalar evidence.")
            if batch.evidence_kind is not clause.evidence_kind:
                raise ValueError("Scalar evidence kind does not match its clause.")
            return
        if type(batch) is not JointStateEvidenceBatch:
            raise TypeError("JointStateEffectClause requires joint-state evidence.")

    def _normalize_evidence(
        self,
        evidence: Mapping[str, EffectEvidenceBatch],
        *,
        requested_at: float,
        deadline: float,
    ) -> tuple[Mapping[str, EffectEvidenceBatch], tuple[int, ...]]:
        if not isinstance(evidence, Mapping):
            raise TypeError("evidence must be a mapping.")
        clause_by_id = {value.clause_id: value for value in self._spec.clauses}
        if set(evidence) != set(clause_by_id):
            raise ValueError("Evidence keys must exactly match effect clause IDs.")
        normalized: dict[str, EffectEvidenceBatch] = {}
        first: EffectEvidenceBatch | None = None
        for clause_id, batch in evidence.items():
            if type(batch) not in _EVIDENCE_TYPES:
                raise TypeError(
                    "evidence values must be typed effect evidence batches."
                )
            if batch.evidence_id != clause_id:
                raise ValueError("Evidence keys must match batch evidence_id values.")
            self._validate_evidence_type(clause_by_id[clause_id], batch)
            if batch.timestamp < requested_at:
                raise ValueError("Effect evidence must not predate the request.")
            if batch.timestamp > deadline:
                raise ValueError(
                    "Effect evidence must not exceed the request deadline."
                )
            if first is None:
                first = batch
            elif (
                batch.timestamp != first.timestamp
                or batch.observation_revision != first.observation_revision
                or not torch.equal(batch.env_ids, first.env_ids)
            ):
                raise ValueError(
                    "All effect evidence must share timestamp, observation_revision, "
                    "and env_ids."
                )
            normalized[clause_id] = batch.snapshot()
        assert first is not None
        known_env_ids = set(self._spec.env_ids.detach().cpu().tolist())
        observed_env_ids = tuple(int(value) for value in first.env_ids.cpu().tolist())
        if not set(observed_env_ids).issubset(known_env_ids):
            raise ValueError("Evidence contains env_ids outside the effect spec.")
        missing = self._active_env_ids.difference(observed_env_ids)
        if missing:
            expectation_ids = {clause.expectation_id for clause in self._spec.clauses}
            for env_id in missing:
                for expectation_id in expectation_ids:
                    key = (expectation_id, env_id)
                    self._success_counts[key] = 0
                    self._failure_counts[key] = 0
                    self._inverse_success_counts[key] = 0
            raise ValueError(
                "Evidence must cover every active request env_id exactly once; "
                f"missing {sorted(missing)}. Acquisition failures must be explicit "
                "valid=False rows."
            )
        return MappingProxyType(normalized), observed_env_ids

    def _pose_baseline(
        self,
        clause: PoseRelationClause,
        request: EffectVerificationRequest,
        spec_row: int,
    ) -> torch.Tensor:
        baseline = clause.baseline_object_to_endpoint
        if baseline is None:
            expectation = self._spec.state_expectation(clause.expectation_id)
            if type(expectation) is not HeldObjectStateExpectation:
                raise ValueError(
                    "A request-derived pose baseline requires a held-object "
                    "state expectation."
                )
            candidate = request.expected_effects.held_object_updates[
                expectation.task_state_key
            ]
            assert isinstance(candidate, HeldObjectState)
            baseline = candidate.object_to_eef
        return baseline if baseline.dim() == 2 else baseline[spec_row]

    def _classify_clause(
        self,
        clause: EffectClause,
        batch: EffectEvidenceBatch,
        *,
        evidence_row: int,
        spec_row: int,
        request: EffectVerificationRequest,
    ) -> int:
        """Return 1 expected, -1 contradicted, or 0 unresolved."""
        if not bool(batch.valid[evidence_row].item()):
            return 0
        if type(clause) is PoseRelationClause:
            assert type(batch) is PoseRelationEvidenceBatch
            observed = batch.object_to_endpoint[evidence_row]
            baseline = self._pose_baseline(clause, request, spec_row)
            translation_error, rotation_error = _pose_errors(observed, baseline)
            matched = (
                translation_error <= self._cfg.attached_translation_threshold
                and rotation_error <= self._cfg.attached_rotation_threshold
            )
            separated = (
                translation_error >= self._cfg.detached_translation_threshold
                or rotation_error >= self._cfg.detached_rotation_threshold
            )
            if clause.expectation is PoseRelationExpectation.MATCHED:
                return 1 if matched else (-1 if separated else 0)
            return 1 if separated else (-1 if matched else 0)
        if type(clause) is BinaryEffectClause:
            assert type(batch) is BinaryEffectEvidenceBatch
            return (
                1 if bool(batch.values[evidence_row].item()) is clause.expected else -1
            )
        if type(clause) is ScalarEffectClause:
            assert type(batch) is ScalarEffectEvidenceBatch
            magnitude = abs(float(batch.values[evidence_row].item()))
            present = magnitude >= self._cfg.force_present_threshold
            absent = magnitude <= self._cfg.force_absent_threshold
            if clause.expectation is ScalarExpectation.PRESENT:
                return 1 if present else (-1 if absent else 0)
            return 1 if absent else (-1 if present else 0)
        assert type(clause) is JointStateEffectClause
        assert type(batch) is JointStateEvidenceBatch
        target = clause.target_position
        if target.dim() == 2:
            target = target[spec_row]
        observed = batch.positions[evidence_row]
        if target.shape != observed.shape:
            raise ValueError("Joint evidence width does not match its clause target.")
        error = float(torch.max(torch.abs(observed - target)).item())
        if error <= self._cfg.joint_success_tolerance:
            return 1
        if error >= self._cfg.joint_failure_tolerance:
            return -1
        return 0

    def observe(
        self,
        request: EffectVerificationRequest,
        evidence: Mapping[str, EffectEvidenceBatch],
    ) -> EffectMonitorDecision:
        """Update typed-clause hysteresis and decide current request rows."""
        self._prepare_request(request)
        batches, observed_env_ids = self._normalize_evidence(
            evidence,
            requested_at=request.requested_at,
            deadline=request.deadline,
        )
        spec_rows = {
            int(env_id): row
            for row, env_id in enumerate(self._spec.env_ids.detach().cpu().tolist())
        }
        request_rows = {
            int(env_id): row
            for row, env_id in enumerate(self._spec.env_ids.detach().cpu().tolist())
            if bool(request.env_mask[row].item())
        }
        first_batch = next(iter(batches.values()))
        observation_token = (
            first_batch.timestamp,
            first_batch.observation_revision,
        )
        for env_id in self._active_env_ids:
            previous = self._last_observations.get(env_id)
            if previous is None:
                continue
            if observation_token[0] < previous[0]:
                raise ValueError(
                    "Evidence timestamps must be monotonic for every active env_id."
                )
            if observation_token[1] < previous[1]:
                raise ValueError(
                    "Evidence observation_revision values must be monotonic for "
                    "every active env_id."
                )

        clauses_by_expectation: dict[str, list[EffectClause]] = {}
        for clause in self._spec.clauses:
            clauses_by_expectation.setdefault(clause.expectation_id, []).append(clause)
        physical_expectation_ids = tuple(
            expectation.expectation_id
            for expectation in self._spec.state_expectations
            if expectation.expectation_id in clauses_by_expectation
        )
        satisfied_masks = {
            expectation_id: torch.zeros_like(request.env_mask)
            for expectation_id in physical_expectation_ids
        }
        contradicted_masks = {
            expectation_id: torch.zeros_like(request.env_mask)
            for expectation_id in physical_expectation_ids
        }
        inverse_satisfied_masks = {
            expectation_id: torch.zeros_like(request.env_mask)
            for expectation_id in physical_expectation_ids
        }

        for evidence_row, env_id in enumerate(observed_env_ids):
            request_row = request_rows.get(env_id)
            if request_row is None:
                continue
            if self._last_observations.get(env_id) == observation_token:
                continue
            self._last_observations[env_id] = observation_token
            spec_row = spec_rows[env_id]
            for expectation_id in physical_expectation_ids:
                classifications = [
                    self._classify_clause(
                        clause,
                        batches[clause.clause_id],
                        evidence_row=evidence_row,
                        spec_row=spec_row,
                        request=request,
                    )
                    for clause in clauses_by_expectation[expectation_id]
                ]
                group_expected = all(value == 1 for value in classifications)
                group_contradicted = any(value == -1 for value in classifications)
                group_inverse_satisfied = all(value == -1 for value in classifications)
                key = (expectation_id, env_id)
                if group_expected:
                    self._success_counts[key] = self._success_counts.get(key, 0) + 1
                else:
                    self._success_counts[key] = 0
                if group_contradicted:
                    self._failure_counts[key] = self._failure_counts.get(key, 0) + 1
                else:
                    self._failure_counts[key] = 0
                if group_inverse_satisfied:
                    self._inverse_success_counts[key] = (
                        self._inverse_success_counts.get(key, 0) + 1
                    )
                else:
                    self._inverse_success_counts[key] = 0
                if self._success_counts.get(key, 0) >= self._cfg.consecutive_samples:
                    satisfied_masks[expectation_id][request_row] = True
                if self._failure_counts.get(key, 0) >= self._cfg.consecutive_samples:
                    contradicted_masks[expectation_id][request_row] = True
                if (
                    self._inverse_success_counts.get(key, 0)
                    >= self._cfg.consecutive_samples
                ):
                    inverse_satisfied_masks[expectation_id][request_row] = True

        expectation_decisions = tuple(
            EffectExpectationDecision(
                expectation_id=expectation_id,
                satisfied_mask=satisfied_masks[expectation_id] & request.env_mask,
                contradicted_mask=(
                    contradicted_masks[expectation_id] & request.env_mask
                ),
                inverse_satisfied_mask=(
                    inverse_satisfied_masks[expectation_id] & request.env_mask
                ),
            )
            for expectation_id in physical_expectation_ids
        )
        success_mask = request.env_mask.clone()
        failure_mask = torch.zeros_like(request.env_mask)
        for decision in expectation_decisions:
            success_mask &= decision.satisfied_mask
            failure_mask |= decision.contradicted_mask
        return EffectMonitorDecision(
            success_mask,
            failure_mask,
            expectation_decisions,
        )


class CompositeEffectMonitorFactory(EffectMonitorFactory):
    """Factory for the built-in typed-clause monitor."""

    monitor_id = COMPOSITE_EFFECT_MONITOR_ID
    revision = COMPOSITE_EFFECT_MONITOR_REVISION

    def validate_ref(self, ref: EffectMonitorRef) -> None:
        """Validate exact built-in selection and typed thresholds."""
        if not isinstance(ref, EffectMonitorRef):
            raise TypeError("ref must be an EffectMonitorRef.")
        if (ref.monitor_id, ref.revision) != (self.monitor_id, self.revision):
            raise ValueError("EffectMonitorRef does not select this exact factory.")
        CompositeEffectMonitorCfg.from_params(ref.params)

    def create(
        self,
        spec: SemanticEffectSpec,
        ref: EffectMonitorRef,
    ) -> CompositeEffectMonitor:
        """Create one independently stateful typed-clause monitor."""
        if not isinstance(spec, SemanticEffectSpec):
            raise TypeError("spec must be a SemanticEffectSpec.")
        self.validate_ref(ref)
        return CompositeEffectMonitor(
            spec.snapshot(),
            CompositeEffectMonitorCfg.from_params(ref.params),
        )


__all__ = [
    "ArticulationJointStateExpectation",
    "BinaryEffectClause",
    "BinaryEffectEvidenceBatch",
    "BinaryEvidenceKind",
    "COMPOSITE_EFFECT_MONITOR_ID",
    "COMPOSITE_EFFECT_MONITOR_REVISION",
    "CONTACT_EFFECT_CHANNEL",
    "CONSTRAINT_EFFECT_CHANNEL",
    "CONTROL_PART_EVIDENCE_PROVIDER_ID",
    "CONTROL_PART_EVIDENCE_PROVIDER_REVISION",
    "CompositeEffectMonitor",
    "CompositeEffectMonitorCfg",
    "CompositeEffectMonitorFactory",
    "ControlPartEvidenceAddress",
    "CoordinatedHeldObjectCleanupExpectation",
    "EffectClause",
    "EffectEvidenceAddress",
    "EffectEvidenceBatch",
    "EffectEvidenceSourceRef",
    "EffectExpectationDecision",
    "EffectMonitor",
    "EffectMonitorDecision",
    "EffectMonitorFactory",
    "EffectMonitorParam",
    "EffectMonitorRef",
    "EffectMonitorRegistry",
    "EffectStateExpectation",
    "FORCE_EFFECT_CHANNEL",
    "HeldObjectRelation",
    "HeldObjectStateExpectation",
    "JOINT_STATE_EFFECT_CHANNEL",
    "JointStateEffectClause",
    "JointStateEvidenceBatch",
    "POSE_RELATION_EFFECT_CHANNEL",
    "PoseRelationClause",
    "PoseRelationEvidenceBatch",
    "PoseRelationExpectation",
    "ScalarEffectClause",
    "ScalarEffectEvidenceBatch",
    "ScalarEvidenceKind",
    "ScalarExpectation",
    "SemanticEffectKind",
    "SemanticEffectSpec",
    "SymbolicStateDomain",
    "SymbolicStateKey",
]
