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

"""Tests for typed semantic-effect contracts and raw evidence monitors."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
import json
import math
from types import MappingProxyType

import pytest
import torch

from embodichain.lab.sim.atomic_actions import (
    Affordance,
    ArticulationJointState,
    EffectVerificationRequest,
    HeldObjectState,
    ObjectSemantics,
    StateDelta,
)
from embodichain.lab.sim.skills.effects import (
    ArticulationJointStateExpectation,
    BinaryEffectClause,
    BinaryEffectEvidenceBatch,
    BinaryEvidenceKind,
    COMPOSITE_EFFECT_MONITOR_ID,
    COMPOSITE_EFFECT_MONITOR_REVISION,
    CompositeEffectMonitor,
    CompositeEffectMonitorCfg,
    CompositeEffectMonitorFactory,
    CoordinatedHeldObjectCleanupExpectation,
    EffectEvidenceAddress,
    EffectEvidenceBatch,
    EffectEvidenceSourceRef,
    EffectMonitor,
    EffectMonitorDecision,
    EffectMonitorFactory,
    EffectMonitorRef,
    EffectMonitorRegistry,
    HeldObjectRelation,
    HeldObjectStateExpectation,
    JointStateEffectClause,
    JointStateEvidenceBatch,
    PoseRelationClause,
    PoseRelationEvidenceBatch,
    PoseRelationExpectation,
    ScalarEffectClause,
    ScalarEffectEvidenceBatch,
    ScalarEvidenceKind,
    ScalarExpectation,
    SemanticEffectKind,
    SemanticEffectSpec,
)

_ENV_IDS = torch.tensor([101, 205, 309], dtype=torch.long)
_OBJECT_ID = "scene/cube"
_STATE_KEY = "left_actor"
_SKILL_ID = "pick_up"
_INVOCATION_ID = "call-7"


@dataclass(frozen=True, slots=True)
class _EvidenceAddress(EffectEvidenceAddress):
    """Minimal custom observation address used by contract tests."""

    endpoint: str
    channel: str

    @property
    def address_fingerprint(self) -> tuple[type, str, str]:
        return type(self), self.endpoint, self.channel


class _AliasingAddress(_EvidenceAddress):
    """Address intentionally violating snapshot ownership."""

    def snapshot(self) -> EffectEvidenceAddress:
        return self


def _source(channel: str) -> EffectEvidenceSourceRef:
    return EffectEvidenceSourceRef(
        "test.raw_evidence",
        "1",
        _EvidenceAddress("left_actor", channel),
    )


def _poses(*x_offsets: float) -> torch.Tensor:
    poses = torch.eye(4).repeat(len(x_offsets), 1, 1)
    poses[:, 0, 3] = torch.tensor(x_offsets)
    return poses


def _semantics(object_id: str = _OBJECT_ID) -> ObjectSemantics:
    return ObjectSemantics(
        affordance=Affordance(),
        geometry={},
        label="object",
        entity_id=object_id,
    )


def _held(
    *,
    object_id: str = _OBJECT_ID,
    baseline: torch.Tensor | None = None,
    env_mask: torch.Tensor | None = None,
) -> HeldObjectState:
    poses = _poses(0.0, 0.0, 0.0) if baseline is None else baseline
    if env_mask is None:
        env_mask = torch.ones(3, dtype=torch.bool)
    return HeldObjectState(
        semantics=_semantics(object_id),
        object_to_eef=poses,
        grasp_xpos=_poses(0.0, 0.0, 0.0),
        env_mask=env_mask,
    )


def _expectation(
    relation: HeldObjectRelation = HeldObjectRelation.ATTACHED,
    *,
    expectation_id: str = "destination",
    state_key: str = _STATE_KEY,
) -> HeldObjectStateExpectation:
    return HeldObjectStateExpectation(
        expectation_id=expectation_id,
        relation=relation,
        object_id=_OBJECT_ID,
        slot_id="primary",
        resource_id="left_actor",
        task_state_key=state_key,
    )


def _attach_spec() -> SemanticEffectSpec:
    return SemanticEffectSpec(
        semantic_id="pick",
        effect_kind=SemanticEffectKind.ATTACH,
        skill_id=_SKILL_ID,
        invocation_id=_INVOCATION_ID,
        invocation_revision=2,
        env_ids=_ENV_IDS,
        state_expectations=(_expectation(),),
        clauses=(
            PoseRelationClause(
                "destination.pose",
                "destination",
                _source("pose_relation"),
                PoseRelationExpectation.MATCHED,
            ),
            BinaryEffectClause(
                "destination.constraint",
                "destination",
                _source("constraint"),
                BinaryEvidenceKind.CONSTRAINT,
                True,
            ),
        ),
    )


def _request(
    *,
    env_mask: torch.Tensor | None = None,
    attempt_generation: int = 0,
    verification_id: int = 1,
    effects: StateDelta | None = None,
) -> EffectVerificationRequest:
    if env_mask is None:
        env_mask = torch.ones(3, dtype=torch.bool)
    if effects is None:
        effects = StateDelta(held_object_updates={_STATE_KEY: _held()})
    return EffectVerificationRequest(
        verification_id=verification_id,
        skill_id=_SKILL_ID,
        invocation_id=_INVOCATION_ID,
        invocation_revision=2,
        invocation_index=0,
        attempt_generation=attempt_generation,
        terminal_segment="close",
        requested_at=1.0,
        deadline=10.0,
        env_mask=env_mask,
        expected_effects=effects,
    )


def _pose_evidence(
    offsets: tuple[float, ...],
    *,
    timestamp: float,
    env_ids: torch.Tensor = _ENV_IDS,
    valid: torch.Tensor | None = None,
    revision: int = 4,
) -> PoseRelationEvidenceBatch:
    if valid is None:
        valid = torch.ones(len(offsets), dtype=torch.bool)
    return PoseRelationEvidenceBatch(
        evidence_id="destination.pose",
        object_to_endpoint=_poses(*offsets),
        valid=valid,
        acquisition_errors=tuple(
            None if row_valid else "pose unavailable" for row_valid in valid
        ),
        timestamp=timestamp,
        env_ids=env_ids,
        observation_revision=revision,
    )


def _binary_evidence(
    values: tuple[bool, ...],
    *,
    timestamp: float,
    env_ids: torch.Tensor = _ENV_IDS,
    valid: torch.Tensor | None = None,
    revision: int = 4,
) -> BinaryEffectEvidenceBatch:
    if valid is None:
        valid = torch.ones(len(values), dtype=torch.bool)
    return BinaryEffectEvidenceBatch(
        evidence_id="destination.constraint",
        evidence_kind=BinaryEvidenceKind.CONSTRAINT,
        values=torch.tensor(values, dtype=torch.bool),
        valid=valid,
        acquisition_errors=tuple(
            None if row_valid else "constraint unavailable" for row_valid in valid
        ),
        timestamp=timestamp,
        env_ids=env_ids,
        observation_revision=revision,
    )


def _evidence(
    offsets: tuple[float, ...],
    constraints: tuple[bool, ...],
    *,
    timestamp: float,
    env_ids: torch.Tensor = _ENV_IDS,
    valid: torch.Tensor | None = None,
    revision: int = 4,
) -> Mapping[str, object]:
    return {
        "destination.pose": _pose_evidence(
            offsets,
            timestamp=timestamp,
            env_ids=env_ids,
            valid=valid,
            revision=revision,
        ),
        "destination.constraint": _binary_evidence(
            constraints,
            timestamp=timestamp,
            env_ids=env_ids,
            valid=valid,
            revision=revision,
        ),
    }


def test_monitor_ref_owns_bounded_non_executable_params() -> None:
    params = {"limits": [1, {"enabled": True}]}
    ref = EffectMonitorRef("monitor", "v1", params)
    params["limits"][1]["enabled"] = False  # type: ignore[index]

    assert isinstance(ref.params, MappingProxyType)
    assert ref.params["limits"] == (1, MappingProxyType({"enabled": True}))
    assert ref.snapshot().params is not ref.params


@pytest.mark.parametrize("value", [torch.tensor(1.0), lambda: None, math.inf])
def test_monitor_ref_rejects_live_or_nonfinite_params(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        EffectMonitorRef("monitor", "v1", {"bad": value})


def test_monitor_ref_rejects_cyclic_params() -> None:
    params: dict[str, object] = {}
    params["cycle"] = params

    with pytest.raises(ValueError, match="cyclic"):
        EffectMonitorRef("monitor", "v1", params)


def test_evidence_source_is_independent_from_runtime_command_addresses() -> None:
    address = _EvidenceAddress("left_actor", "pose_relation")
    source = EffectEvidenceSourceRef("provider", "2", address)

    assert source.address is not address
    assert source.source_fingerprint == (
        "provider",
        "2",
        _EvidenceAddress,
        (_EvidenceAddress, "left_actor", "pose_relation"),
    )
    assert not hasattr(source, "transport_id")


def test_evidence_source_enforces_snapshot_ownership() -> None:
    with pytest.raises(TypeError, match="independently owned"):
        EffectEvidenceSourceRef(
            "provider",
            "1",
            _AliasingAddress("left_actor", "pose_relation"),
        )


def test_semantic_spec_owns_typed_state_and_heterogeneous_clauses() -> None:
    env_ids = _ENV_IDS.clone()
    spec = _attach_spec()
    env_ids[0] = -1

    assert torch.equal(spec.env_ids, _ENV_IDS)
    assert type(spec.state_expectations[0]) is HeldObjectStateExpectation
    assert tuple(type(clause) for clause in spec.clauses) == (
        PoseRelationClause,
        BinaryEffectClause,
    )
    assert spec.snapshot().clauses[0] is not spec.clauses[0]


def test_spec_rejects_clause_without_typed_state_expectation() -> None:
    with pytest.raises(ValueError, match="unknown state expectations"):
        replace(
            _attach_spec(),
            clauses=(replace(_attach_spec().clauses[0], expectation_id="missing"),),
        )


def test_articulation_and_joint_clause_are_first_class_typed_contracts() -> None:
    target = torch.tensor([0.42])
    spec = SemanticEffectSpec(
        semantic_id="test_articulation",
        effect_kind=SemanticEffectKind.ARTICULATION,
        skill_id="test_articulation",
        invocation_id="drawer-1",
        invocation_revision=0,
        env_ids=_ENV_IDS,
        state_expectations=(
            ArticulationJointStateExpectation(
                "drawer_joint",
                "drawer",
                "slide",
                target,
            ),
        ),
        clauses=(
            JointStateEffectClause(
                "drawer_joint.position",
                "drawer_joint",
                _source("joint_state"),
                target,
            ),
        ),
    )
    target.fill_(9.0)

    expectation = spec.state_expectations[0]
    clause = spec.clauses[0]
    assert isinstance(expectation, ArticulationJointStateExpectation)
    assert isinstance(clause, JointStateEffectClause)
    torch.testing.assert_close(expectation.target_position, torch.tensor([0.42]))
    torch.testing.assert_close(clause.target_position, torch.tensor([0.42]))

    request = EffectVerificationRequest(
        verification_id=1,
        skill_id="test_articulation",
        invocation_id="drawer-1",
        invocation_revision=0,
        invocation_index=0,
        attempt_generation=0,
        terminal_segment="operate",
        requested_at=1.0,
        deadline=10.0,
        env_mask=torch.ones(3, dtype=torch.bool),
        expected_effects=StateDelta(
            articulation_joint_updates={
                ("drawer", "slide"): ArticulationJointState(torch.tensor([0.42]))
            }
        ),
    )
    spec.validate_request(request)

    wrong = replace(
        request,
        expected_effects=StateDelta(
            articulation_joint_updates={
                ("drawer", "slide"): ArticulationJointState(torch.tensor([0.7]))
            }
        ),
    )
    with pytest.raises(ValueError, match="target position"):
        spec.validate_request(wrong)


def test_request_validation_uses_logical_state_key() -> None:
    _attach_spec().validate_request(_request())

    wrong_key = _request(
        effects=StateDelta(held_object_updates={"arm_control_part": _held()})
    )
    with pytest.raises(ValueError, match="exactly match"):
        _attach_spec().validate_request(wrong_key)


def test_request_validation_declares_coordinated_cleanup_explicitly() -> None:
    cleanup = CoordinatedHeldObjectCleanupExpectation(
        "cleanup:left_actor:support",
        (_STATE_KEY, "support"),
    )
    spec = replace(
        _attach_spec(),
        state_expectations=(*_attach_spec().state_expectations, cleanup),
    )
    request = _request(
        effects=StateDelta(
            held_object_updates={_STATE_KEY: _held()},
            coordinated_held_object_updates={(_STATE_KEY, "support"): None},
        )
    )

    spec.validate_request(request)


def test_pose_evidence_owns_rows_and_allows_invalid_nonfinite_payload() -> None:
    poses = _poses(0.0, 0.1)
    poses[1].fill_(math.nan)
    valid = torch.tensor([True, False])
    batch = PoseRelationEvidenceBatch(
        "pose",
        poses,
        valid,
        (None, "occluded"),
        2.0,
        torch.tensor([101, 205]),
        3,
    )
    poses.zero_()
    valid.fill_(True)

    assert torch.isnan(batch.object_to_endpoint[1]).all()
    assert batch.valid.tolist() == [True, False]


def test_effect_contract_evidence_and_resolved_thresholds_are_json_safe() -> None:
    poses = _poses(0.0, 0.1)
    poses[1].fill_(math.nan)
    batch = PoseRelationEvidenceBatch(
        "pose",
        poses,
        torch.tensor([True, False]),
        (None, "occluded"),
        2.0,
        torch.tensor([101, 205]),
        3,
    )
    monitor = CompositeEffectMonitor(
        _attach_spec(),
        CompositeEffectMonitorCfg(consecutive_samples=3),
    )

    metadata = {
        "spec": _attach_spec().to_metadata(),
        "evidence": batch.to_metadata(),
        "thresholds": dict(monitor.resolved_params),
    }

    json.dumps(metadata, allow_nan=False, sort_keys=True)
    assert metadata["evidence"]["object_to_endpoint"][1][0][0] is None
    assert metadata["thresholds"]["attached_translation_threshold"] == 0.02
    assert metadata["thresholds"]["consecutive_samples"] == 3


def test_binary_scalar_and_joint_evidence_are_distinct_raw_batches() -> None:
    valid = torch.tensor([True, True])
    env_ids = torch.tensor([101, 205])
    binary = BinaryEffectEvidenceBatch(
        "contact",
        BinaryEvidenceKind.CONTACT,
        torch.tensor([True, False]),
        valid,
        (None, None),
        2.0,
        env_ids,
        3,
    )
    scalar = ScalarEffectEvidenceBatch(
        "force",
        ScalarEvidenceKind.FORCE,
        torch.tensor([2.0, 0.0]),
        valid,
        (None, None),
        2.0,
        env_ids,
        3,
    )
    joint = JointStateEvidenceBatch(
        "joint",
        torch.tensor([[0.4], [0.5]]),
        torch.zeros(2, 1),
        valid,
        (None, None),
        2.0,
        env_ids,
        3,
    )

    assert binary.values.dtype == torch.bool
    assert scalar.values.tolist() == [2.0, 0.0]
    assert joint.positions.shape == (2, 1)


def test_valid_raw_evidence_rejects_nonfinite_payload() -> None:
    with pytest.raises(ValueError, match="finite"):
        ScalarEffectEvidenceBatch(
            "force",
            ScalarEvidenceKind.FORCE,
            torch.tensor([math.nan]),
            torch.tensor([True]),
            (None,),
            2.0,
            torch.tensor([101]),
            3,
        )


def test_monitor_requires_pose_and_binary_physical_evidence() -> None:
    monitor = CompositeEffectMonitor(
        _attach_spec(),
        CompositeEffectMonitorCfg(consecutive_samples=1),
    )
    request = _request()
    pose_only = monitor.observe(
        request,
        _evidence(
            (0.0, 0.0, 0.0),
            (False, False, False),
            timestamp=2.0,
        ),  # type: ignore[arg-type]
    )

    assert not pose_only.success_mask.any()
    assert pose_only.failure_mask.all()


def test_monitor_reports_success_only_for_complete_consecutive_evidence() -> None:
    monitor = CompositeEffectMonitor(
        _attach_spec(),
        CompositeEffectMonitorCfg(consecutive_samples=2),
    )
    request = _request()
    first = monitor.observe(
        request,
        _evidence(
            (0.0, 0.01, 0.019),
            (True, True, True),
            timestamp=2.0,
        ),  # type: ignore[arg-type]
    )
    second = monitor.observe(
        request,
        _evidence(
            (0.0, 0.01, 0.019),
            (True, True, True),
            timestamp=3.0,
            revision=5,
        ),  # type: ignore[arg-type]
    )

    assert not first.success_mask.any()
    assert second.success_mask.all()
    assert not second.failure_mask.any()


def test_invalid_evidence_is_unresolved_and_resets_hysteresis() -> None:
    monitor = CompositeEffectMonitor(
        _attach_spec(),
        CompositeEffectMonitorCfg(consecutive_samples=2),
    )
    request = _request()
    monitor.observe(
        request,
        _evidence(
            (0.0, 0.0, 0.0),
            (True, True, True),
            timestamp=2.0,
        ),  # type: ignore[arg-type]
    )
    invalid = monitor.observe(
        request,
        _evidence(
            (0.0, 0.0, 0.0),
            (True, True, True),
            timestamp=3.0,
            valid=torch.tensor([False, True, True]),
            revision=5,
        ),  # type: ignore[arg-type]
    )
    after_reset = monitor.observe(
        request,
        _evidence(
            (0.0, 0.0, 0.0),
            (True, True, True),
            timestamp=4.0,
            revision=6,
        ),  # type: ignore[arg-type]
    )

    assert invalid.success_mask.tolist() == [False, True, True]
    assert after_reset.success_mask.tolist() == [False, True, True]


def test_request_shrink_preserves_counts_and_generation_change_resets() -> None:
    monitor = CompositeEffectMonitor(
        _attach_spec(),
        CompositeEffectMonitorCfg(consecutive_samples=2),
    )
    monitor.observe(
        _request(),
        _evidence(
            (0.0, 0.0, 0.0),
            (True, True, True),
            timestamp=2.0,
        ),  # type: ignore[arg-type]
    )
    shrunk = _request(
        env_mask=torch.tensor([False, True, True]),
        verification_id=2,
    )
    preserved = monitor.observe(
        shrunk,
        _evidence(
            (0.0, 0.0),
            (True, True),
            timestamp=3.0,
            env_ids=torch.tensor([205, 309]),
            revision=5,
        ),  # type: ignore[arg-type]
    )
    reset = monitor.observe(
        _request(attempt_generation=1, verification_id=3),
        _evidence(
            (0.0, 0.0, 0.0),
            (True, True, True),
            timestamp=3.0,
            revision=5,
        ),  # type: ignore[arg-type]
    )

    assert preserved.success_mask.tolist() == [False, True, True]
    assert not reset.success_mask.any()


def test_monitor_rejects_expansion_duplicate_counting_and_late_evidence() -> None:
    monitor = CompositeEffectMonitor(
        _attach_spec(),
        CompositeEffectMonitorCfg(consecutive_samples=2),
    )
    shrunk = _request(env_mask=torch.tensor([False, True, True]))
    sample = _evidence(
        (0.0, 0.0),
        (True, True),
        timestamp=2.0,
        env_ids=torch.tensor([205, 309]),
    )
    monitor.observe(shrunk, sample)  # type: ignore[arg-type]
    repeated = monitor.observe(shrunk, sample)  # type: ignore[arg-type]

    assert not repeated.success_mask.any()
    with pytest.raises(ValueError, match="only shrink"):
        monitor.observe(
            _request(verification_id=2),
            _evidence(
                (0.0, 0.0, 0.0),
                (True, True, True),
                timestamp=3.0,
                revision=5,
            ),  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="deadline"):
        CompositeEffectMonitor(
            _attach_spec(),
            CompositeEffectMonitorCfg(consecutive_samples=1),
        ).observe(
            _request(),
            _evidence(
                (0.0, 0.0, 0.0),
                (True, True, True),
                timestamp=10.01,
            ),  # type: ignore[arg-type]
        )


def test_scalar_and_joint_clauses_use_monitor_owned_policy() -> None:
    spec = replace(
        _attach_spec(),
        clauses=(
            ScalarEffectClause(
                "destination.force",
                "destination",
                _source("force"),
                ScalarEvidenceKind.FORCE,
                ScalarExpectation.PRESENT,
            ),
            JointStateEffectClause(
                "destination.joint",
                "destination",
                _source("joint_state"),
                torch.tensor([0.5]),
            ),
        ),
    )
    monitor = CompositeEffectMonitor(
        spec,
        CompositeEffectMonitorCfg(consecutive_samples=1),
    )
    valid = torch.ones(3, dtype=torch.bool)
    errors = (None, None, None)
    evidence = {
        "destination.force": ScalarEffectEvidenceBatch(
            "destination.force",
            ScalarEvidenceKind.FORCE,
            torch.tensor([2.0, 0.0, 0.5]),
            valid,
            errors,
            2.0,
            _ENV_IDS,
            4,
        ),
        "destination.joint": JointStateEvidenceBatch(
            "destination.joint",
            torch.tensor([[0.5], [0.5], [0.7]]),
            None,
            valid,
            errors,
            2.0,
            _ENV_IDS,
            4,
        ),
    }

    decision = monitor.observe(_request(), evidence)

    assert decision.success_mask.tolist() == [True, False, False]
    assert decision.failure_mask.tolist() == [False, True, True]


class _BoundMonitor(EffectMonitor):
    def __init__(self, spec: SemanticEffectSpec, *, alias: bool = False) -> None:
        self._spec = spec if alias else spec.snapshot()
        self._alias = alias

    @property
    def spec(self) -> SemanticEffectSpec:
        return self._spec if self._alias else self._spec.snapshot()

    def observe(
        self,
        request: EffectVerificationRequest,
        evidence: Mapping[str, EffectEvidenceBatch],
    ) -> EffectMonitorDecision:
        del evidence
        return EffectMonitorDecision(
            torch.zeros_like(request.env_mask),
            torch.zeros_like(request.env_mask),
        )


class _BoundFactory(EffectMonitorFactory):
    monitor_id = "test.bound"
    revision = "1"

    def __init__(self, spec: SemanticEffectSpec, *, alias: bool = False) -> None:
        self._spec = spec if alias else spec.snapshot()
        self._alias = alias

    def validate_ref(self, ref: EffectMonitorRef) -> None:
        if (ref.monitor_id, ref.revision) != (self.monitor_id, self.revision):
            raise ValueError("wrong key")

    def create(
        self,
        spec: SemanticEffectSpec,
        ref: EffectMonitorRef,
    ) -> EffectMonitor:
        del spec, ref
        return _BoundMonitor(self._spec, alias=self._alias)


def test_registry_is_exact_versioned_and_enforces_bound_spec() -> None:
    factory = CompositeEffectMonitorFactory()
    registry = EffectMonitorRegistry((factory,))
    ref = EffectMonitorRef(
        COMPOSITE_EFFECT_MONITOR_ID,
        COMPOSITE_EFFECT_MONITOR_REVISION,
        {"consecutive_samples": 1},
    )

    first = registry.create(_attach_spec(), ref)
    second = registry.create(_attach_spec(), ref)

    assert isinstance(first, CompositeEffectMonitor)
    assert first is not second
    with pytest.raises(KeyError):
        registry.resolve(EffectMonitorRef(factory.monitor_id, "unknown"))
    with pytest.raises(ValueError, match="Duplicate"):
        EffectMonitorRegistry((factory, CompositeEffectMonitorFactory()))


def test_registry_rejects_factory_spec_drift_or_aliasing() -> None:
    requested = _attach_spec()
    changed = replace(requested, semantic_id="other")
    drift = _BoundFactory(changed)
    with pytest.raises(ValueError, match="different effect spec"):
        EffectMonitorRegistry((drift,)).create(
            requested,
            EffectMonitorRef(drift.monitor_id, drift.revision),
        )

    alias = _BoundFactory(requested, alias=True)
    with pytest.raises(TypeError, match="independently owned"):
        EffectMonitorRegistry((alias,)).create(
            requested,
            EffectMonitorRef(alias.monitor_id, alias.revision),
        )


def test_composite_config_requires_real_hysteresis_gaps() -> None:
    with pytest.raises(ValueError, match="less than"):
        CompositeEffectMonitorCfg(
            attached_translation_threshold=0.05,
            detached_translation_threshold=0.05,
        )
    with pytest.raises(ValueError, match="positive integer"):
        CompositeEffectMonitorCfg(consecutive_samples=True)
    with pytest.raises(ValueError, match="Unknown"):
        CompositeEffectMonitorFactory().validate_ref(
            EffectMonitorRef(
                COMPOSITE_EFFECT_MONITOR_ID,
                COMPOSITE_EFFECT_MONITOR_REVISION,
                {"typo": 1},
            )
        )
