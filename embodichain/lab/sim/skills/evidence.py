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

"""Backend-neutral acquisition ports for typed semantic-effect evidence."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import ClassVar, Protocol, TypeAlias, runtime_checkable

import torch

from embodichain.utils.math import pose_inv

from ..atomic_actions import SceneProvider, SceneSnapshot
from .effects import (
    ArticulationJointStateExpectation,
    BinaryEffectClause,
    BinaryEffectEvidenceBatch,
    BinaryEvidenceKind,
    CONTACT_EFFECT_CHANNEL,
    CONSTRAINT_EFFECT_CHANNEL,
    CONTROL_PART_EVIDENCE_PROVIDER_ID,
    CONTROL_PART_EVIDENCE_PROVIDER_REVISION,
    ControlPartEvidenceAddress,
    CoordinatedHeldObjectCleanupExpectation,
    EffectClause,
    EffectEvidenceBatch,
    EffectEvidenceSourceRef,
    EffectStateExpectation,
    FORCE_EFFECT_CHANNEL,
    HeldObjectStateExpectation,
    JOINT_STATE_EFFECT_CHANNEL,
    JointStateEffectClause,
    JointStateEvidenceBatch,
    POSE_RELATION_EFFECT_CHANNEL,
    PoseRelationClause,
    PoseRelationEvidenceBatch,
    ScalarEffectClause,
    ScalarEffectEvidenceBatch,
    ScalarEvidenceKind,
    SemanticEffectSpec,
)
from .scene import (
    SCENE_ARTICULATION_EVIDENCE_PROVIDER_ID,
    SCENE_ARTICULATION_EVIDENCE_PROVIDER_REVISION,
    ArticulationJointEvidenceAddress,
)
from ._validation import validate_identifier as _validate_identifier

_EFFECT_EXPECTATION_TYPES = (
    HeldObjectStateExpectation,
    CoordinatedHeldObjectCleanupExpectation,
    ArticulationJointStateExpectation,
)
_EFFECT_BATCH_TYPES = (
    PoseRelationEvidenceBatch,
    BinaryEffectEvidenceBatch,
    ScalarEffectEvidenceBatch,
    JointStateEvidenceBatch,
)


@dataclass(frozen=True, slots=True, eq=False)
class EffectEvidenceCollectionContext:
    """One synchronized acquisition tick shared by all effect clauses.

    Args:
        timestamp: Non-negative backend observation time.
        observation_revision: Monotonic revision chosen by the runtime port.
        env_ids: Ordered environment correlation IDs to observe.
    """

    timestamp: float
    observation_revision: int
    env_ids: torch.Tensor

    def __post_init__(self) -> None:
        if isinstance(self.timestamp, bool) or not isinstance(
            self.timestamp, (int, float)
        ):
            raise TypeError("timestamp must be a number.")
        timestamp = float(self.timestamp)
        if not math.isfinite(timestamp) or timestamp < 0.0:
            raise ValueError("timestamp must be finite and non-negative.")
        if type(self.observation_revision) is not int or self.observation_revision < 0:
            raise ValueError("observation_revision must be a non-negative integer.")
        env_ids = self.env_ids
        if not isinstance(env_ids, torch.Tensor):
            raise TypeError("env_ids must be a torch.Tensor.")
        if env_ids.dtype != torch.long or env_ids.dim() != 1 or env_ids.numel() == 0:
            raise ValueError(
                "env_ids must be a non-empty one-dimensional int64 tensor."
            )
        if torch.unique(env_ids).numel() != env_ids.numel():
            raise ValueError("env_ids must be unique.")
        object.__setattr__(self, "timestamp", timestamp)
        object.__setattr__(self, "env_ids", env_ids.clone())

    def snapshot(self) -> EffectEvidenceCollectionContext:
        """Return an independently owned acquisition context."""
        return EffectEvidenceCollectionContext(
            self.timestamp,
            self.observation_revision,
            self.env_ids,
        )


def _snapshot_expectation(
    expectation: EffectStateExpectation,
) -> EffectStateExpectation:
    """Validate and own one exact typed effect expectation."""
    if type(expectation) not in _EFFECT_EXPECTATION_TYPES:
        raise TypeError("expectation must be an exact typed effect expectation.")
    return expectation.snapshot()


class EffectEvidenceQuery(ABC):
    """Typed request for the raw evidence of exactly one effect clause."""

    @property
    @abstractmethod
    def evidence_id(self) -> str:
        """Return the clause-local evidence identifier."""

    @property
    @abstractmethod
    def source(self) -> EffectEvidenceSourceRef:
        """Return an owned exact provider route and physical address."""

    @property
    @abstractmethod
    def expectation(self) -> EffectStateExpectation:
        """Return an owned symbolic expectation related to this query."""

    @abstractmethod
    def snapshot(self) -> EffectEvidenceQuery:
        """Return an independently owned query of the exact same type."""


def _validate_query(
    clause: EffectClause,
    expectation: EffectStateExpectation,
) -> EffectStateExpectation:
    """Validate common clause/expectation correlation."""
    owned = _snapshot_expectation(expectation)
    if clause.expectation_id != owned.expectation_id:
        raise ValueError("Query clause and expectation IDs must match.")
    return owned


@dataclass(frozen=True, slots=True, eq=False)
class PoseRelationEvidenceQuery(EffectEvidenceQuery):
    """Query for an object's pose relative to a resource endpoint."""

    clause: PoseRelationClause
    _expectation: EffectStateExpectation

    def __post_init__(self) -> None:
        if type(self.clause) is not PoseRelationClause:
            raise TypeError("clause must be a PoseRelationClause.")
        object.__setattr__(self, "clause", self.clause.snapshot())
        object.__setattr__(
            self,
            "_expectation",
            _validate_query(self.clause, self._expectation),
        )

    @property
    def evidence_id(self) -> str:
        """Return the source clause ID."""
        return self.clause.clause_id

    @property
    def source(self) -> EffectEvidenceSourceRef:
        """Return an owned source route."""
        return self.clause.source.snapshot()

    @property
    def expectation(self) -> EffectStateExpectation:
        """Return an owned correlated expectation."""
        return self._expectation.snapshot()

    def snapshot(self) -> PoseRelationEvidenceQuery:
        """Return an independently owned pose query."""
        return PoseRelationEvidenceQuery(self.clause, self._expectation)


@dataclass(frozen=True, slots=True, eq=False)
class BinaryEffectEvidenceQuery(EffectEvidenceQuery):
    """Query for one raw contact or constraint boolean."""

    clause: BinaryEffectClause
    _expectation: EffectStateExpectation

    def __post_init__(self) -> None:
        if type(self.clause) is not BinaryEffectClause:
            raise TypeError("clause must be a BinaryEffectClause.")
        object.__setattr__(self, "clause", self.clause.snapshot())
        object.__setattr__(
            self,
            "_expectation",
            _validate_query(self.clause, self._expectation),
        )

    @property
    def evidence_id(self) -> str:
        """Return the source clause ID."""
        return self.clause.clause_id

    @property
    def source(self) -> EffectEvidenceSourceRef:
        """Return an owned source route."""
        return self.clause.source.snapshot()

    @property
    def expectation(self) -> EffectStateExpectation:
        """Return an owned correlated expectation."""
        return self._expectation.snapshot()

    def snapshot(self) -> BinaryEffectEvidenceQuery:
        """Return an independently owned binary query."""
        return BinaryEffectEvidenceQuery(self.clause, self._expectation)


@dataclass(frozen=True, slots=True, eq=False)
class ScalarEffectEvidenceQuery(EffectEvidenceQuery):
    """Query for one raw force or wrench magnitude."""

    clause: ScalarEffectClause
    _expectation: EffectStateExpectation

    def __post_init__(self) -> None:
        if type(self.clause) is not ScalarEffectClause:
            raise TypeError("clause must be a ScalarEffectClause.")
        object.__setattr__(self, "clause", self.clause.snapshot())
        object.__setattr__(
            self,
            "_expectation",
            _validate_query(self.clause, self._expectation),
        )

    @property
    def evidence_id(self) -> str:
        """Return the source clause ID."""
        return self.clause.clause_id

    @property
    def source(self) -> EffectEvidenceSourceRef:
        """Return an owned source route."""
        return self.clause.source.snapshot()

    @property
    def expectation(self) -> EffectStateExpectation:
        """Return an owned correlated expectation."""
        return self._expectation.snapshot()

    def snapshot(self) -> ScalarEffectEvidenceQuery:
        """Return an independently owned scalar query."""
        return ScalarEffectEvidenceQuery(self.clause, self._expectation)


@dataclass(frozen=True, slots=True, eq=False)
class JointStateEvidenceQuery(EffectEvidenceQuery):
    """Query for current joint positions and optional velocities."""

    clause: JointStateEffectClause
    _expectation: EffectStateExpectation

    def __post_init__(self) -> None:
        if type(self.clause) is not JointStateEffectClause:
            raise TypeError("clause must be a JointStateEffectClause.")
        object.__setattr__(self, "clause", self.clause.snapshot())
        object.__setattr__(
            self,
            "_expectation",
            _validate_query(self.clause, self._expectation),
        )

    @property
    def evidence_id(self) -> str:
        """Return the source clause ID."""
        return self.clause.clause_id

    @property
    def source(self) -> EffectEvidenceSourceRef:
        """Return an owned source route."""
        return self.clause.source.snapshot()

    @property
    def expectation(self) -> EffectStateExpectation:
        """Return an owned correlated expectation."""
        return self._expectation.snapshot()

    def snapshot(self) -> JointStateEvidenceQuery:
        """Return an independently owned joint-state query."""
        return JointStateEvidenceQuery(self.clause, self._expectation)


EffectEvidenceQueryValue: TypeAlias = (
    PoseRelationEvidenceQuery
    | BinaryEffectEvidenceQuery
    | ScalarEffectEvidenceQuery
    | JointStateEvidenceQuery
)
"""Closed set of typed clause queries accepted by evidence providers."""


def build_effect_evidence_queries(
    spec: SemanticEffectSpec,
) -> tuple[EffectEvidenceQueryValue, ...]:
    """Build one independently owned typed query per effect clause.

    Args:
        spec: Grounded semantic effect contract.

    Returns:
        Queries in the contract's deterministic clause order.
    """
    if not isinstance(spec, SemanticEffectSpec):
        raise TypeError("spec must be a SemanticEffectSpec.")
    queries: list[EffectEvidenceQueryValue] = []
    for clause in spec.clauses:
        expectation = spec.state_expectation(clause.expectation_id)
        if type(clause) is PoseRelationClause:
            queries.append(PoseRelationEvidenceQuery(clause, expectation))
        elif type(clause) is BinaryEffectClause:
            queries.append(BinaryEffectEvidenceQuery(clause, expectation))
        elif type(clause) is ScalarEffectClause:
            queries.append(ScalarEffectEvidenceQuery(clause, expectation))
        elif type(clause) is JointStateEffectClause:
            queries.append(JointStateEvidenceQuery(clause, expectation))
        else:
            raise TypeError(f"Unsupported effect clause type {type(clause).__name__}.")
    return tuple(queries)


class EffectEvidenceProvider(ABC):
    """Versioned backend port that acquires a group of exact-source queries."""

    provider_id: ClassVar[str]
    revision: ClassVar[str]

    @abstractmethod
    def collect(
        self,
        queries: tuple[EffectEvidenceQueryValue, ...],
        context: EffectEvidenceCollectionContext,
    ) -> Mapping[str, EffectEvidenceBatch]:
        """Acquire one synchronized batch for every supplied query."""


class EffectEvidenceProviderRegistry:
    """Immutable exact-ID/revision registry of live evidence providers."""

    __slots__ = ("_providers",)

    def __init__(self, providers: Iterable[EffectEvidenceProvider] = ()) -> None:
        normalized: dict[tuple[str, str], EffectEvidenceProvider] = {}
        for provider in providers:
            if not isinstance(provider, EffectEvidenceProvider):
                raise TypeError(
                    "providers must contain EffectEvidenceProvider instances."
                )
            provider_id = _validate_identifier(
                provider.provider_id,
                field_name="EffectEvidenceProvider.provider_id",
            )
            revision = _validate_identifier(
                provider.revision,
                field_name="EffectEvidenceProvider.revision",
            )
            key = provider_id, revision
            if key in normalized:
                raise ValueError(f"Duplicate effect-evidence provider {key!r}.")
            normalized[key] = provider
        self._providers = MappingProxyType(normalized)

    @property
    def providers(self) -> Mapping[tuple[str, str], EffectEvidenceProvider]:
        """Return the immutable exact-key provider mapping."""
        return self._providers

    def resolve(self, source: EffectEvidenceSourceRef) -> EffectEvidenceProvider:
        """Resolve the exact provider selected by ``source``.

        Args:
            source: Versioned evidence route from one effect clause.

        Returns:
            Registered provider with the exact ID and revision.

        Raises:
            KeyError: If no exact provider version is installed.
        """
        if not isinstance(source, EffectEvidenceSourceRef):
            raise TypeError("source must be an EffectEvidenceSourceRef.")
        key = source.provider_id, source.revision
        try:
            return self._providers[key]
        except KeyError as exc:
            raise KeyError(
                f"Unknown effect-evidence provider {key!r}; exact versions are "
                "required."
            ) from exc


def _expected_batch_type(query: EffectEvidenceQueryValue) -> type[EffectEvidenceBatch]:
    """Return the exact evidence batch type required by one query."""
    if type(query) is PoseRelationEvidenceQuery:
        return PoseRelationEvidenceBatch
    if type(query) is BinaryEffectEvidenceQuery:
        return BinaryEffectEvidenceBatch
    if type(query) is ScalarEffectEvidenceQuery:
        return ScalarEffectEvidenceBatch
    if type(query) is JointStateEvidenceQuery:
        return JointStateEvidenceBatch
    raise TypeError(f"Unsupported effect evidence query {type(query).__name__}.")


class EffectEvidenceCollector:
    """Dispatch and normalize a synchronized observation for one effect spec."""

    __slots__ = ("_registry",)

    def __init__(self, registry: EffectEvidenceProviderRegistry) -> None:
        if not isinstance(registry, EffectEvidenceProviderRegistry):
            raise TypeError("registry must be an EffectEvidenceProviderRegistry.")
        self._registry = registry

    @property
    def registry(self) -> EffectEvidenceProviderRegistry:
        """Return the immutable provider registry."""
        return self._registry

    def collect(
        self,
        spec: SemanticEffectSpec,
        *,
        timestamp: float,
        observation_revision: int,
        env_ids: torch.Tensor | None = None,
    ) -> Mapping[str, EffectEvidenceBatch]:
        """Acquire and strictly synchronize evidence for every effect clause.

        Args:
            spec: Grounded semantic effect contract.
            timestamp: Backend observation time for this acquisition tick.
            observation_revision: Runtime-owned observation revision.
            env_ids: Optional ordered subset of ``spec.env_ids``. Acquisition
                failures must remain present as rows with ``valid=False``.

        Returns:
            Immutable mapping keyed exactly by effect clause ID.
        """
        if not isinstance(spec, SemanticEffectSpec):
            raise TypeError("spec must be a SemanticEffectSpec.")
        selected_env_ids = spec.env_ids if env_ids is None else env_ids
        context = EffectEvidenceCollectionContext(
            timestamp,
            observation_revision,
            selected_env_ids,
        )
        known_ids = set(spec.env_ids.detach().cpu().tolist())
        selected_ids = set(context.env_ids.detach().cpu().tolist())
        if not selected_ids.issubset(known_ids):
            raise ValueError("env_ids must be a subset of the effect spec env_ids.")

        queries = build_effect_evidence_queries(spec)
        groups: dict[
            tuple[str, str],
            list[EffectEvidenceQueryValue],
        ] = {}
        for query in queries:
            source = query.source
            self._registry.resolve(source)
            groups.setdefault((source.provider_id, source.revision), []).append(query)

        batches: dict[str, EffectEvidenceBatch] = {}
        for key, grouped_queries in groups.items():
            provider = self._registry.providers[key]
            owned_queries = tuple(query.snapshot() for query in grouped_queries)
            supplied = provider.collect(owned_queries, context.snapshot())
            if not isinstance(supplied, Mapping):
                raise TypeError(
                    f"Effect-evidence provider {key!r} must return a mapping."
                )
            expected_ids = {query.evidence_id for query in grouped_queries}
            if set(supplied) != expected_ids:
                raise ValueError(
                    f"Effect-evidence provider {key!r} must return exactly query "
                    f"IDs {sorted(expected_ids)}; got {sorted(supplied)}."
                )
            for query in grouped_queries:
                batch = supplied[query.evidence_id]
                expected_type = _expected_batch_type(query)
                if type(batch) is not expected_type:
                    raise TypeError(
                        f"Evidence {query.evidence_id!r} must be "
                        f"{expected_type.__name__}."
                    )
                if batch.evidence_id != query.evidence_id:
                    raise ValueError(
                        "Evidence mapping keys must match batch evidence_id values."
                    )
                if batch.timestamp != context.timestamp:
                    raise ValueError(
                        "Every evidence batch must use the collection timestamp."
                    )
                if batch.observation_revision != context.observation_revision:
                    raise ValueError(
                        "Every evidence batch must use the collection revision."
                    )
                if batch.env_ids.device != context.env_ids.device or not torch.equal(
                    batch.env_ids, context.env_ids
                ):
                    raise ValueError(
                        "Every evidence batch must use the ordered collection env_ids."
                    )
                if type(query) is BinaryEffectEvidenceQuery:
                    assert type(batch) is BinaryEffectEvidenceBatch
                    if batch.evidence_kind is not query.clause.evidence_kind:
                        raise ValueError("Binary evidence kind must match its query.")
                if type(query) is ScalarEffectEvidenceQuery:
                    assert type(batch) is ScalarEffectEvidenceBatch
                    if batch.evidence_kind is not query.clause.evidence_kind:
                        raise ValueError("Scalar evidence kind must match its query.")
                batches[query.evidence_id] = batch.snapshot()

        expected_all = {query.evidence_id for query in queries}
        if set(batches) != expected_all:
            raise AssertionError("Evidence dispatch lost one or more effect clauses.")
        return MappingProxyType(batches)


@dataclass(frozen=True, slots=True, eq=False)
class BinaryEffectObservation:
    """Callback-owned raw binary values with explicit row validity."""

    values: torch.Tensor
    valid: torch.Tensor | None = None
    acquisition_errors: tuple[str | None, ...] = ()

    def __post_init__(self) -> None:
        values = self.values
        if not isinstance(values, torch.Tensor):
            raise TypeError("values must be a torch.Tensor.")
        if values.dtype != torch.bool or values.dim() != 1 or values.numel() == 0:
            raise ValueError("values must have non-empty bool shape (B,).")
        valid = torch.ones_like(values) if self.valid is None else self.valid
        if (
            not isinstance(valid, torch.Tensor)
            or valid.dtype != torch.bool
            or valid.shape != values.shape
            or valid.device != values.device
        ):
            raise ValueError("valid must match values shape, bool dtype, and device.")
        errors = self.acquisition_errors or (None,) * values.shape[0]
        _validate_observation_errors(valid, errors)
        object.__setattr__(self, "values", values.clone())
        object.__setattr__(self, "valid", valid.clone())
        object.__setattr__(self, "acquisition_errors", tuple(errors))


@dataclass(frozen=True, slots=True, eq=False)
class ScalarEffectObservation:
    """Callback-owned raw scalar values with explicit row validity."""

    values: torch.Tensor
    valid: torch.Tensor | None = None
    acquisition_errors: tuple[str | None, ...] = ()

    def __post_init__(self) -> None:
        values = self.values
        if not isinstance(values, torch.Tensor):
            raise TypeError("values must be a torch.Tensor.")
        if not values.is_floating_point() or values.dim() != 1 or values.numel() == 0:
            raise ValueError("values must have non-empty floating shape (B,).")
        valid = (
            torch.ones_like(values, dtype=torch.bool)
            if self.valid is None
            else self.valid
        )
        if (
            not isinstance(valid, torch.Tensor)
            or valid.dtype != torch.bool
            or valid.shape != values.shape
            or valid.device != values.device
        ):
            raise ValueError("valid must match values shape, bool dtype, and device.")
        if not torch.isfinite(values[valid]).all():
            raise ValueError("Valid scalar observations must be finite.")
        errors = self.acquisition_errors or (None,) * values.shape[0]
        _validate_observation_errors(valid, errors)
        object.__setattr__(self, "values", values.clone())
        object.__setattr__(self, "valid", valid.clone())
        object.__setattr__(self, "acquisition_errors", tuple(errors))


@dataclass(frozen=True, slots=True, eq=False)
class JointStateObservation:
    """Callback-owned raw joint state with explicit row validity."""

    positions: torch.Tensor
    velocities: torch.Tensor | None = None
    valid: torch.Tensor | None = None
    acquisition_errors: tuple[str | None, ...] = ()

    def __post_init__(self) -> None:
        positions = self.positions
        if not isinstance(positions, torch.Tensor):
            raise TypeError("positions must be a torch.Tensor.")
        if (
            not positions.is_floating_point()
            or positions.dim() != 2
            or positions.shape[0] == 0
            or positions.shape[1] == 0
        ):
            raise ValueError("positions must have non-empty floating shape (B, J).")
        valid = (
            torch.ones(positions.shape[0], dtype=torch.bool, device=positions.device)
            if self.valid is None
            else self.valid
        )
        if (
            not isinstance(valid, torch.Tensor)
            or valid.dtype != torch.bool
            or valid.shape != (positions.shape[0],)
            or valid.device != positions.device
        ):
            raise ValueError("valid must have bool shape (B,) on the positions device.")
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
        errors = self.acquisition_errors or (None,) * positions.shape[0]
        _validate_observation_errors(valid, errors)
        object.__setattr__(self, "positions", positions.clone())
        object.__setattr__(
            self,
            "velocities",
            None if velocities is None else velocities.clone(),
        )
        object.__setattr__(self, "valid", valid.clone())
        object.__setattr__(self, "acquisition_errors", tuple(errors))


def _validate_observation_errors(
    valid: torch.Tensor,
    errors: Sequence[str | None],
) -> None:
    """Validate explicit per-row acquisition errors."""
    if len(errors) != valid.shape[0]:
        raise ValueError("acquisition_errors must contain one entry per row.")
    for row, (row_valid, error) in enumerate(zip(valid.tolist(), errors)):
        if row_valid and error is not None:
            raise ValueError(f"Valid observation row {row} must not carry an error.")
        if not row_valid and (
            type(error) is not str or not error or error != error.strip()
        ):
            raise ValueError(
                f"Invalid observation row {row} requires a non-empty error."
            )


BinaryObservationCallback: TypeAlias = Callable[
    [BinaryEffectEvidenceQuery, EffectEvidenceCollectionContext],
    BinaryEffectObservation,
]
ScalarObservationCallback: TypeAlias = Callable[
    [ScalarEffectEvidenceQuery, EffectEvidenceCollectionContext],
    ScalarEffectObservation,
]
ArticulationJointObservationCallback: TypeAlias = Callable[
    [JointStateEvidenceQuery, EffectEvidenceCollectionContext],
    JointStateObservation,
]


class SceneArticulationEvidenceProvider(EffectEvidenceProvider):
    """Typed adapter for scene-articulation joint-state observations.

    Integrations inject either a direct observer or a :class:`SceneProvider`
    whose snapshot contains ``ObservedArticulationJointState`` values.
    The adapter never discovers live simulator objects from an environment.
    Repeated clauses share one synchronized snapshot and one sample per exact
    physical address.
    """

    provider_id = SCENE_ARTICULATION_EVIDENCE_PROVIDER_ID
    revision = SCENE_ARTICULATION_EVIDENCE_PROVIDER_REVISION

    def __init__(
        self,
        observer: ArticulationJointObservationCallback | None = None,
        *,
        scene_provider: SceneProvider | None = None,
    ) -> None:
        if (observer is None) == (scene_provider is None):
            raise ValueError(
                "Exactly one of observer or scene_provider must be supplied."
            )
        if observer is not None and not callable(observer):
            raise TypeError("observer must be callable or None.")
        if scene_provider is not None and not isinstance(scene_provider, SceneProvider):
            raise TypeError("scene_provider must implement SceneProvider or be None.")
        self._observer = observer
        self._scene_provider = scene_provider

    def collect(
        self,
        queries: tuple[EffectEvidenceQueryValue, ...],
        context: EffectEvidenceCollectionContext,
    ) -> Mapping[str, EffectEvidenceBatch]:
        """Collect synchronized joint state for exact scene addresses."""
        if not isinstance(context, EffectEvidenceCollectionContext):
            raise TypeError("context must be an EffectEvidenceCollectionContext.")
        if not isinstance(queries, tuple) or not queries:
            raise ValueError("queries must be a non-empty tuple.")
        owned_queries = tuple(self._validate_query(query) for query in queries)
        if len({query.evidence_id for query in owned_queries}) != len(owned_queries):
            raise ValueError("queries must have unique evidence IDs.")

        observations: dict[object, JointStateObservation] = {}
        batches: dict[str, JointStateEvidenceBatch] = {}
        scene_snapshot: SceneSnapshot | None = None
        if self._scene_provider is not None:
            scene_snapshot = self._scene_provider.snapshot(
                timestamp=context.timestamp,
                env_ids=context.env_ids.clone(),
            )
            if not isinstance(scene_snapshot, SceneSnapshot):
                raise TypeError("scene_provider.snapshot() must return SceneSnapshot.")
            if scene_snapshot.timestamp != context.timestamp:
                raise ValueError(
                    "Scene snapshot timestamp must match the evidence tick."
                )
        for query in owned_queries:
            address = query.source.address
            assert type(address) is ArticulationJointEvidenceAddress
            fingerprint = address.address_fingerprint
            observation = observations.get(fingerprint)
            if observation is None:
                supplied = (
                    self._observe_scene_snapshot(query, context, scene_snapshot)
                    if scene_snapshot is not None
                    else self._observer(query.snapshot(), context.snapshot())
                )
                if not isinstance(supplied, JointStateObservation):
                    raise TypeError(
                        "Articulation observers must return JointStateObservation."
                    )
                observation = supplied
                observations[fingerprint] = observation
            self._validate_observation(query, observation, context)
            assert observation.valid is not None
            batches[query.evidence_id] = JointStateEvidenceBatch(
                query.evidence_id,
                observation.positions,
                observation.velocities,
                observation.valid,
                observation.acquisition_errors,
                context.timestamp,
                context.env_ids,
                context.observation_revision,
            )
        return MappingProxyType(batches)

    @staticmethod
    def _observe_scene_snapshot(
        query: JointStateEvidenceQuery,
        context: EffectEvidenceCollectionContext,
        snapshot: SceneSnapshot,
    ) -> JointStateObservation:
        """Adapt one typed live scene joint into raw effect evidence."""
        address = query.source.address
        assert type(address) is ArticulationJointEvidenceAddress
        state = snapshot.get_articulation_joint_state(
            address.articulation_id,
            address.joint_id,
        )
        batch_size = int(context.env_ids.numel())
        if state is None:
            width = int(query.clause.target_position.shape[-1])
            return JointStateObservation(
                positions=torch.zeros(
                    (batch_size, width),
                    dtype=query.clause.target_position.dtype,
                    device=context.env_ids.device,
                ),
                valid=torch.zeros(
                    batch_size,
                    dtype=torch.bool,
                    device=context.env_ids.device,
                ),
                acquisition_errors=(
                    f"Scene snapshot has no live articulation joint "
                    f"{(address.articulation_id, address.joint_id)!r}.",
                )
                * batch_size,
            )
        positions = state.position
        if positions.dim() == 1:
            positions = positions.unsqueeze(0).expand(batch_size, -1)
        if positions.shape[0] != batch_size:
            raise ValueError(
                "Scene articulation observation rows must match context env_ids."
            )
        positions = positions.to(device=context.env_ids.device)
        valid = state.valid_mask
        if valid is None:
            valid = torch.ones(
                batch_size,
                dtype=torch.bool,
                device=context.env_ids.device,
            )
        else:
            valid = valid.to(device=context.env_ids.device)
        errors = tuple(
            None if bool(row_valid) else "Scene articulation joint row is invalid."
            for row_valid in valid.tolist()
        )
        return JointStateObservation(
            positions=positions,
            valid=valid,
            acquisition_errors=errors,
        )

    def _validate_query(
        self,
        query: EffectEvidenceQueryValue,
    ) -> JointStateEvidenceQuery:
        """Require one exact joint query and matching canonical address."""
        if type(query) is not JointStateEvidenceQuery:
            raise TypeError(
                "SceneArticulationEvidenceProvider accepts only "
                "JointStateEvidenceQuery values."
            )
        source = query.source
        if (source.provider_id, source.revision) != (
            self.provider_id,
            self.revision,
        ):
            raise ValueError("Query does not select this exact provider version.")
        if type(source.address) is not ArticulationJointEvidenceAddress:
            raise TypeError(
                "Scene articulation evidence requires "
                "ArticulationJointEvidenceAddress."
            )
        expectation = query.expectation
        if type(expectation) is not ArticulationJointStateExpectation:
            raise TypeError(
                "Scene articulation evidence requires an "
                "ArticulationJointStateExpectation."
            )
        if (
            expectation.articulation_id != source.address.articulation_id
            or expectation.joint_id != source.address.joint_id
        ):
            raise ValueError(
                "Articulation evidence address must exactly match its typed "
                "state expectation."
            )
        return query.snapshot()

    @staticmethod
    def _validate_observation(
        query: JointStateEvidenceQuery,
        observation: JointStateObservation,
        context: EffectEvidenceCollectionContext,
    ) -> None:
        """Require callback rows/device/width to match the synchronized query."""
        if observation.positions.shape[0] != context.env_ids.numel():
            raise ValueError(
                "Articulation observation rows must match context env_ids."
            )
        if observation.positions.device != context.env_ids.device:
            raise ValueError(
                "Articulation observations and context env_ids must share a device."
            )
        target_width = int(query.clause.target_position.shape[-1])
        if observation.positions.shape[1] != target_width:
            raise ValueError(
                f"Joint observation width {observation.positions.shape[1]} does "
                f"not match query target width {target_width}."
            )


@runtime_checkable
class ControlPartRobotEvidenceSource(Protocol):
    """Minimal simulation robot API used by the built-in provider."""

    def get_qpos(self, name: str | None = None, target: bool = False) -> torch.Tensor:
        """Return current robot or control-part joint positions."""

    def get_qvel(self, name: str | None = None, target: bool = False) -> torch.Tensor:
        """Return current robot or control-part joint velocities."""

    def compute_fk(
        self,
        qpos: torch.Tensor,
        name: str | None = None,
        env_ids: Sequence[int] | None = None,
        to_matrix: bool = False,
    ) -> torch.Tensor:
        """Return the selected endpoint pose for current joint positions."""


class ControlPartSimulationEvidenceProvider(EffectEvidenceProvider):
    """Built-in simulation acquisition for control-part evidence addresses.

    Pose evidence is computed as ``inverse(object_pose) @ endpoint_pose`` from
    one scene snapshot and :meth:`Robot.compute_fk`. Joint evidence reads the
    control part's measured positions and velocities. Contact, constraint,
    force, and wrench signals are backend-specific, so callers inject raw
    observation callbacks. An omitted callback yields explicit invalid rows;
    the effect monitor can then retry until its normal deadline.
    """

    provider_id = CONTROL_PART_EVIDENCE_PROVIDER_ID
    revision = CONTROL_PART_EVIDENCE_PROVIDER_REVISION

    def __init__(
        self,
        robot: ControlPartRobotEvidenceSource,
        *,
        scene_provider: SceneProvider | None = None,
        contact_observer: BinaryObservationCallback | None = None,
        constraint_observer: BinaryObservationCallback | None = None,
        force_observer: ScalarObservationCallback | None = None,
        wrench_observer: ScalarObservationCallback | None = None,
    ) -> None:
        if not isinstance(robot, ControlPartRobotEvidenceSource):
            raise TypeError("robot must implement ControlPartRobotEvidenceSource.")
        if scene_provider is not None and not isinstance(scene_provider, SceneProvider):
            raise TypeError("scene_provider must implement SceneProvider or be None.")
        for name, callback in (
            ("contact_observer", contact_observer),
            ("constraint_observer", constraint_observer),
            ("force_observer", force_observer),
            ("wrench_observer", wrench_observer),
        ):
            if callback is not None and not callable(callback):
                raise TypeError(f"{name} must be callable or None.")
        self._robot = robot
        self._scene_provider = scene_provider
        self._binary_observers = {
            BinaryEvidenceKind.CONTACT: contact_observer,
            BinaryEvidenceKind.CONSTRAINT: constraint_observer,
        }
        self._scalar_observers = {
            ScalarEvidenceKind.FORCE: force_observer,
            ScalarEvidenceKind.WRENCH: wrench_observer,
        }

    def collect(
        self,
        queries: tuple[EffectEvidenceQueryValue, ...],
        context: EffectEvidenceCollectionContext,
    ) -> Mapping[str, EffectEvidenceBatch]:
        """Acquire all supplied control-part queries at one observation tick."""
        if not isinstance(context, EffectEvidenceCollectionContext):
            raise TypeError("context must be an EffectEvidenceCollectionContext.")
        if not isinstance(queries, tuple) or not queries:
            raise ValueError("queries must be a non-empty tuple.")
        owned_queries = tuple(
            self._validate_and_snapshot_query(query) for query in queries
        )
        if len({query.evidence_id for query in owned_queries}) != len(owned_queries):
            raise ValueError("queries must have unique evidence IDs.")

        pose_queries = tuple(
            query for query in owned_queries if type(query) is PoseRelationEvidenceQuery
        )
        scene_snapshot = self._capture_scene(pose_queries, context)
        joint_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        endpoint_cache: dict[str, torch.Tensor] = {}
        results: dict[str, EffectEvidenceBatch] = {}
        for query in owned_queries:
            address = query.source.address
            assert type(address) is ControlPartEvidenceAddress
            if type(query) is PoseRelationEvidenceQuery:
                results[query.evidence_id] = self._collect_pose(
                    query,
                    address,
                    context,
                    scene_snapshot,
                    joint_cache,
                    endpoint_cache,
                )
            elif type(query) is BinaryEffectEvidenceQuery:
                results[query.evidence_id] = self._collect_binary(
                    query,
                    address,
                    context,
                )
            elif type(query) is ScalarEffectEvidenceQuery:
                results[query.evidence_id] = self._collect_scalar(
                    query,
                    address,
                    context,
                )
            elif type(query) is JointStateEvidenceQuery:
                results[query.evidence_id] = self._collect_joint_state(
                    query,
                    address,
                    context,
                    joint_cache,
                )
            else:
                raise TypeError(f"Unsupported query type {type(query).__name__}.")
        return MappingProxyType(results)

    def _validate_and_snapshot_query(
        self,
        query: EffectEvidenceQueryValue,
    ) -> EffectEvidenceQueryValue:
        """Require the exact built-in route and a control-part address."""
        if type(query) not in {
            PoseRelationEvidenceQuery,
            BinaryEffectEvidenceQuery,
            ScalarEffectEvidenceQuery,
            JointStateEvidenceQuery,
        }:
            raise TypeError("queries must contain exact typed evidence queries.")
        source = query.source
        if (source.provider_id, source.revision) != (
            self.provider_id,
            self.revision,
        ):
            raise ValueError("Query does not select this exact provider version.")
        if type(source.address) is not ControlPartEvidenceAddress:
            raise TypeError(
                "ControlPartSimulationEvidenceProvider requires "
                "ControlPartEvidenceAddress values."
            )
        return query.snapshot()

    def _capture_scene(
        self,
        queries: tuple[PoseRelationEvidenceQuery, ...],
        context: EffectEvidenceCollectionContext,
    ) -> SceneSnapshot | None:
        """Capture one shared scene snapshot if pose queries need it."""
        if not queries or self._scene_provider is None:
            return None
        snapshot = self._scene_provider.snapshot(
            timestamp=context.timestamp,
            env_ids=context.env_ids.clone(),
        )
        if not isinstance(snapshot, SceneSnapshot):
            raise TypeError("scene_provider.snapshot() must return SceneSnapshot.")
        if snapshot.timestamp != context.timestamp:
            raise ValueError("Scene snapshot timestamp must match the evidence tick.")
        return snapshot

    @staticmethod
    def _require_channel(
        address: ControlPartEvidenceAddress,
        expected: str,
        *,
        evidence_id: str,
    ) -> None:
        """Reject clause/address channel mismatches before acquisition."""
        if address.channel != expected:
            raise ValueError(
                f"Evidence query {evidence_id!r} requires channel {expected!r}, "
                f"not {address.channel!r}."
            )

    @staticmethod
    def _select_rows(
        value: torch.Tensor, context: EffectEvidenceCollectionContext
    ) -> torch.Tensor:
        """Select simulator rows addressed by the context's integer env IDs."""
        if not isinstance(value, torch.Tensor):
            raise TypeError("Robot state accessors must return torch.Tensor values.")
        if value.dim() != 2 or value.shape[0] == 0 or value.shape[1] == 0:
            raise ValueError("Robot joint state must have non-empty shape (N, J).")
        indices = context.env_ids.to(device=value.device)
        if bool((indices < 0).any()) or int(indices.max().item()) >= value.shape[0]:
            raise ValueError(
                "The built-in simulation provider requires env_ids to address "
                "valid simulator batch rows."
            )
        selected = value.index_select(0, indices)
        if selected.device != context.env_ids.device:
            raise ValueError(
                "Robot evidence and collection env_ids must share a device."
            )
        return selected.clone()

    def _joint_state(
        self,
        control_part: str,
        context: EffectEvidenceCollectionContext,
        cache: dict[str, tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Read and cache measured positions and velocities for one part."""
        cached = cache.get(control_part)
        if cached is not None:
            return cached[0].clone(), cached[1].clone()
        qpos = self._select_rows(
            self._robot.get_qpos(name=control_part, target=False),
            context,
        )
        qvel = self._select_rows(
            self._robot.get_qvel(name=control_part, target=False),
            context,
        )
        if qvel.shape != qpos.shape or qvel.device != qpos.device:
            raise ValueError("Robot qvel must match qpos shape and device.")
        if not qpos.is_floating_point() or not qvel.is_floating_point():
            raise TypeError("Robot qpos and qvel must use floating-point dtypes.")
        cache[control_part] = qpos.clone(), qvel.clone()
        return qpos, qvel

    def _endpoint_pose(
        self,
        control_part: str,
        context: EffectEvidenceCollectionContext,
        joint_cache: dict[str, tuple[torch.Tensor, torch.Tensor]],
        endpoint_cache: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute and cache one control-part endpoint pose."""
        cached = endpoint_cache.get(control_part)
        if cached is not None:
            return cached.clone()
        qpos, _ = self._joint_state(control_part, context, joint_cache)
        pose = self._robot.compute_fk(
            qpos=qpos,
            name=control_part,
            env_ids=context.env_ids.detach().cpu().tolist(),
            to_matrix=True,
        )
        if not isinstance(pose, torch.Tensor):
            raise TypeError("robot.compute_fk() must return a torch.Tensor.")
        if pose.shape != (context.env_ids.numel(), 4, 4):
            raise ValueError("robot.compute_fk() must return shape (B, 4, 4).")
        if pose.device != context.env_ids.device:
            raise ValueError(
                "Endpoint poses and collection env_ids must share a device."
            )
        endpoint_cache[control_part] = pose.clone()
        return pose

    @staticmethod
    def _pose_entity_id(query: PoseRelationEvidenceQuery) -> str:
        """Resolve the canonical scene entity observed by a pose relation."""
        expectation = query.expectation
        if type(expectation) is HeldObjectStateExpectation:
            return expectation.object_id
        if type(expectation) is ArticulationJointStateExpectation:
            return expectation.articulation_id
        raise ValueError(
            "Pose relation evidence requires an expectation with one canonical "
            "scene entity."
        )

    def _collect_pose(
        self,
        query: PoseRelationEvidenceQuery,
        address: ControlPartEvidenceAddress,
        context: EffectEvidenceCollectionContext,
        scene_snapshot: SceneSnapshot | None,
        joint_cache: dict[str, tuple[torch.Tensor, torch.Tensor]],
        endpoint_cache: dict[str, torch.Tensor],
    ) -> PoseRelationEvidenceBatch:
        """Collect object-to-endpoint transforms from scene and FK state."""
        self._require_channel(
            address,
            POSE_RELATION_EFFECT_CHANNEL,
            evidence_id=query.evidence_id,
        )
        if scene_snapshot is None:
            return self._invalid_pose(
                query.evidence_id,
                context,
                "No scene provider is configured for pose-relation evidence.",
            )
        entity_id = self._pose_entity_id(query)
        try:
            state = scene_snapshot.entities[entity_id]
        except KeyError as exc:
            raise KeyError(
                f"Pose evidence references missing scene entity {entity_id!r}."
            ) from exc
        object_pose = state.pose
        batch_size = int(context.env_ids.numel())
        if object_pose.shape == (4, 4):
            object_pose = object_pose.unsqueeze(0).expand(batch_size, -1, -1)
        if object_pose.shape != (batch_size, 4, 4):
            raise ValueError(
                f"Scene entity {entity_id!r} pose must have shape (B, 4, 4)."
            )
        endpoint_pose = self._endpoint_pose(
            address.control_part,
            context,
            joint_cache,
            endpoint_cache,
        )
        object_pose = object_pose.to(
            device=endpoint_pose.device,
            dtype=endpoint_pose.dtype,
        )
        relative = torch.bmm(pose_inv(object_pose), endpoint_pose)
        valid = torch.full(
            (batch_size,),
            state.confidence > 0.0,
            dtype=torch.bool,
            device=relative.device,
        )
        errors: tuple[str | None, ...]
        if bool(valid.all()):
            errors = (None,) * batch_size
        else:
            errors = (
                f"Scene entity {entity_id!r} has zero observation confidence.",
            ) * batch_size
        return PoseRelationEvidenceBatch(
            query.evidence_id,
            relative,
            valid,
            errors,
            context.timestamp,
            context.env_ids,
            context.observation_revision,
        )

    def _collect_binary(
        self,
        query: BinaryEffectEvidenceQuery,
        address: ControlPartEvidenceAddress,
        context: EffectEvidenceCollectionContext,
    ) -> BinaryEffectEvidenceBatch:
        """Collect callback-provided contact or constraint state."""
        expected_channel = (
            CONTACT_EFFECT_CHANNEL
            if query.clause.evidence_kind is BinaryEvidenceKind.CONTACT
            else CONSTRAINT_EFFECT_CHANNEL
        )
        self._require_channel(address, expected_channel, evidence_id=query.evidence_id)
        callback = self._binary_observers[query.clause.evidence_kind]
        if callback is None:
            return self._invalid_binary(
                query,
                context,
                f"No {query.clause.evidence_kind.value} observation callback is configured.",
            )
        observation = callback(query.snapshot(), context.snapshot())
        if not isinstance(observation, BinaryEffectObservation):
            raise TypeError(
                "Binary observation callbacks must return BinaryEffectObservation."
            )
        self._validate_callback_rows(observation.values, context)
        assert observation.valid is not None
        return BinaryEffectEvidenceBatch(
            query.evidence_id,
            query.clause.evidence_kind,
            observation.values,
            observation.valid,
            observation.acquisition_errors,
            context.timestamp,
            context.env_ids,
            context.observation_revision,
        )

    def _collect_scalar(
        self,
        query: ScalarEffectEvidenceQuery,
        address: ControlPartEvidenceAddress,
        context: EffectEvidenceCollectionContext,
    ) -> ScalarEffectEvidenceBatch:
        """Collect callback-provided force or wrench magnitude."""
        self._require_channel(
            address, FORCE_EFFECT_CHANNEL, evidence_id=query.evidence_id
        )
        callback = self._scalar_observers[query.clause.evidence_kind]
        if callback is None:
            return self._invalid_scalar(
                query,
                context,
                f"No {query.clause.evidence_kind.value} observation callback is configured.",
            )
        observation = callback(query.snapshot(), context.snapshot())
        if not isinstance(observation, ScalarEffectObservation):
            raise TypeError(
                "Scalar observation callbacks must return ScalarEffectObservation."
            )
        self._validate_callback_rows(observation.values, context)
        assert observation.valid is not None
        return ScalarEffectEvidenceBatch(
            query.evidence_id,
            query.clause.evidence_kind,
            observation.values,
            observation.valid,
            observation.acquisition_errors,
            context.timestamp,
            context.env_ids,
            context.observation_revision,
        )

    def _collect_joint_state(
        self,
        query: JointStateEvidenceQuery,
        address: ControlPartEvidenceAddress,
        context: EffectEvidenceCollectionContext,
        joint_cache: dict[str, tuple[torch.Tensor, torch.Tensor]],
    ) -> JointStateEvidenceBatch:
        """Collect measured control-part joint positions and velocities."""
        self._require_channel(
            address,
            JOINT_STATE_EFFECT_CHANNEL,
            evidence_id=query.evidence_id,
        )
        qpos, qvel = self._joint_state(address.control_part, context, joint_cache)
        target_width = int(query.clause.target_position.shape[-1])
        if qpos.shape[1] != target_width:
            raise ValueError(
                f"Joint evidence width {qpos.shape[1]} does not match query target "
                f"width {target_width}."
            )
        batch_size = int(context.env_ids.numel())
        return JointStateEvidenceBatch(
            query.evidence_id,
            qpos,
            qvel,
            torch.ones(batch_size, dtype=torch.bool, device=qpos.device),
            (None,) * batch_size,
            context.timestamp,
            context.env_ids,
            context.observation_revision,
        )

    @staticmethod
    def _validate_callback_rows(
        values: torch.Tensor,
        context: EffectEvidenceCollectionContext,
    ) -> None:
        """Require callback values to follow the synchronized context rows."""
        if values.shape != context.env_ids.shape:
            raise ValueError("Observation callback rows must match context env_ids.")
        if values.device != context.env_ids.device:
            raise ValueError(
                "Observation callback values and context env_ids must share a device."
            )

    @staticmethod
    def _invalid_pose(
        evidence_id: str,
        context: EffectEvidenceCollectionContext,
        message: str,
    ) -> PoseRelationEvidenceBatch:
        """Create explicit invalid rows for unavailable pose acquisition."""
        batch_size = int(context.env_ids.numel())
        poses = torch.eye(
            4,
            dtype=torch.float32,
            device=context.env_ids.device,
        ).expand(batch_size, -1, -1)
        return PoseRelationEvidenceBatch(
            evidence_id,
            poses,
            torch.zeros(batch_size, dtype=torch.bool, device=context.env_ids.device),
            (message,) * batch_size,
            context.timestamp,
            context.env_ids,
            context.observation_revision,
        )

    @staticmethod
    def _invalid_binary(
        query: BinaryEffectEvidenceQuery,
        context: EffectEvidenceCollectionContext,
        message: str,
    ) -> BinaryEffectEvidenceBatch:
        """Create explicit invalid rows for an unavailable binary channel."""
        batch_size = int(context.env_ids.numel())
        return BinaryEffectEvidenceBatch(
            query.evidence_id,
            query.clause.evidence_kind,
            torch.zeros(batch_size, dtype=torch.bool, device=context.env_ids.device),
            torch.zeros(batch_size, dtype=torch.bool, device=context.env_ids.device),
            (message,) * batch_size,
            context.timestamp,
            context.env_ids,
            context.observation_revision,
        )

    @staticmethod
    def _invalid_scalar(
        query: ScalarEffectEvidenceQuery,
        context: EffectEvidenceCollectionContext,
        message: str,
    ) -> ScalarEffectEvidenceBatch:
        """Create explicit invalid rows for an unavailable scalar channel."""
        batch_size = int(context.env_ids.numel())
        return ScalarEffectEvidenceBatch(
            query.evidence_id,
            query.clause.evidence_kind,
            torch.zeros(batch_size, dtype=torch.float32, device=context.env_ids.device),
            torch.zeros(batch_size, dtype=torch.bool, device=context.env_ids.device),
            (message,) * batch_size,
            context.timestamp,
            context.env_ids,
            context.observation_revision,
        )


__all__ = [
    "ArticulationJointObservationCallback",
    "BinaryEffectEvidenceQuery",
    "BinaryEffectObservation",
    "BinaryObservationCallback",
    "ControlPartRobotEvidenceSource",
    "ControlPartSimulationEvidenceProvider",
    "EffectEvidenceCollectionContext",
    "EffectEvidenceCollector",
    "EffectEvidenceProvider",
    "EffectEvidenceProviderRegistry",
    "EffectEvidenceQuery",
    "EffectEvidenceQueryValue",
    "JointStateEvidenceQuery",
    "JointStateObservation",
    "PoseRelationEvidenceQuery",
    "ScalarEffectEvidenceQuery",
    "ScalarEffectObservation",
    "ScalarObservationCallback",
    "SceneArticulationEvidenceProvider",
    "build_effect_evidence_queries",
]
