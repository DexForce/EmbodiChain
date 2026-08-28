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

"""Tests for synchronized semantic-effect evidence acquisition."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pytest
import torch

from embodichain.lab.sim.atomic_actions import (
    EntityState,
    ObservedArticulationJointState,
    SceneSnapshot,
)
from embodichain.lab.semantic_skills.effects import (
    ArticulationJointStateExpectation,
    BinaryEffectClause,
    BinaryEffectEvidenceBatch,
    BinaryEvidenceKind,
    CONTACT_EFFECT_CHANNEL,
    CONSTRAINT_EFFECT_CHANNEL,
    CONTROL_PART_EVIDENCE_PROVIDER_ID,
    CONTROL_PART_EVIDENCE_PROVIDER_REVISION,
    ControlPartEvidenceAddress,
    EffectEvidenceBatch,
    EffectEvidenceSourceRef,
    FORCE_EFFECT_CHANNEL,
    HeldObjectRelation,
    HeldObjectStateExpectation,
    JOINT_STATE_EFFECT_CHANNEL,
    JointStateEffectClause,
    POSE_RELATION_EFFECT_CHANNEL,
    PoseRelationClause,
    PoseRelationExpectation,
    ScalarEffectClause,
    ScalarEvidenceKind,
    ScalarExpectation,
    SemanticEffectKind,
    SemanticEffectSpec,
)
from embodichain.lab.semantic_skills.evidence import (
    BinaryEffectEvidenceQuery,
    BinaryEffectObservation,
    ControlPartSimulationEvidenceProvider,
    EffectEvidenceCollectionContext,
    EffectEvidenceCollector,
    EffectEvidenceProvider,
    EffectEvidenceProviderRegistry,
    JointStateEvidenceQuery,
    JointStateObservation,
    PoseRelationEvidenceQuery,
    ScalarEffectEvidenceQuery,
    ScalarEffectObservation,
    SceneArticulationEvidenceProvider,
    build_effect_evidence_queries,
)
from embodichain.lab.semantic_skills.scene import (
    SCENE_ARTICULATION_EVIDENCE_PROVIDER_ID,
    SCENE_ARTICULATION_EVIDENCE_PROVIDER_REVISION,
    ArticulationJointEvidenceAddress,
)


def _source(channel: str, *, provider_id: str | None = None) -> EffectEvidenceSourceRef:
    return EffectEvidenceSourceRef(
        provider_id or CONTROL_PART_EVIDENCE_PROVIDER_ID,
        CONTROL_PART_EVIDENCE_PROVIDER_REVISION,
        ControlPartEvidenceAddress("arm", channel),
    )


def _held_expectation() -> HeldObjectStateExpectation:
    return HeldObjectStateExpectation(
        "held",
        HeldObjectRelation.ATTACHED,
        "cube",
        "actor",
        "arm_resource",
        "arm_resource",
    )


def _attach_spec(
    *clauses: object,
    env_ids: torch.Tensor | None = None,
) -> SemanticEffectSpec:
    return SemanticEffectSpec(
        semantic_id="pick:cube",
        effect_kind=SemanticEffectKind.ATTACH,
        skill_id="PickUp",
        invocation_id="pick-1",
        invocation_revision=0,
        env_ids=torch.tensor([0, 1], dtype=torch.long) if env_ids is None else env_ids,
        state_expectations=(_held_expectation(),),
        clauses=clauses,
    )


def _pose_clause(clause_id: str = "pose") -> PoseRelationClause:
    return PoseRelationClause(
        clause_id,
        "held",
        _source(POSE_RELATION_EFFECT_CHANNEL),
        PoseRelationExpectation.MATCHED,
    )


def _binary_clause(
    clause_id: str = "contact",
    *,
    provider_id: str | None = None,
) -> BinaryEffectClause:
    return BinaryEffectClause(
        clause_id,
        "held",
        _source(CONTACT_EFFECT_CHANNEL, provider_id=provider_id),
        BinaryEvidenceKind.CONTACT,
        True,
    )


def _scalar_clause(clause_id: str = "force") -> ScalarEffectClause:
    return ScalarEffectClause(
        clause_id,
        "held",
        _source(FORCE_EFFECT_CHANNEL),
        ScalarEvidenceKind.FORCE,
        ScalarExpectation.PRESENT,
    )


class _FakeSceneProvider:
    def __init__(self, poses: torch.Tensor, *, confidence: float = 1.0) -> None:
        self.poses = poses
        self.confidence = confidence
        self.calls = 0
        self.received_env_ids: torch.Tensor | None = None

    def snapshot(self, *, timestamp: float, env_ids: torch.Tensor) -> SceneSnapshot:
        self.calls += 1
        self.received_env_ids = env_ids.clone()
        poses = self.poses.index_select(0, env_ids.to(device=self.poses.device))
        return SceneSnapshot(
            timestamp=timestamp,
            version=self.calls,
            entities={"cube": EntityState(poses, confidence=self.confidence)},
        )


class _FakeRobot:
    def __init__(self, qpos: torch.Tensor, qvel: torch.Tensor | None = None) -> None:
        self.qpos = qpos
        self.qvel = torch.zeros_like(qpos) if qvel is None else qvel
        self.fk_calls = 0
        self.qpos_calls = 0
        self.qvel_calls = 0

    def get_qpos(self, name: str | None = None, target: bool = False) -> torch.Tensor:
        assert name == "arm"
        assert target is False
        self.qpos_calls += 1
        return self.qpos

    def get_qvel(self, name: str | None = None, target: bool = False) -> torch.Tensor:
        assert name == "arm"
        assert target is False
        self.qvel_calls += 1
        return self.qvel

    def compute_fk(
        self,
        qpos: torch.Tensor,
        name: str | None = None,
        env_ids: Sequence[int] | None = None,
        to_matrix: bool = False,
    ) -> torch.Tensor:
        assert name == "arm"
        assert env_ids is not None
        assert to_matrix is True
        self.fk_calls += 1
        poses = torch.eye(4, dtype=qpos.dtype, device=qpos.device).repeat(
            qpos.shape[0], 1, 1
        )
        poses[:, 0, 3] = qpos[:, 0]
        return poses


class _WrongTimestampProvider(EffectEvidenceProvider):
    provider_id = "test.provider"
    revision = "1"

    def collect(
        self,
        queries: tuple[object, ...],
        context: EffectEvidenceCollectionContext,
    ) -> Mapping[str, EffectEvidenceBatch]:
        query = queries[0]
        assert isinstance(query, BinaryEffectEvidenceQuery)
        batch_size = int(context.env_ids.numel())
        return {
            query.evidence_id: BinaryEffectEvidenceBatch(
                query.evidence_id,
                BinaryEvidenceKind.CONTACT,
                torch.ones(batch_size, dtype=torch.bool),
                torch.ones(batch_size, dtype=torch.bool),
                (None,) * batch_size,
                context.timestamp + 1.0,
                context.env_ids,
                context.observation_revision,
            )
        }


class _SecondRevisionProvider(_WrongTimestampProvider):
    revision = "2"


def test_collection_context_validates_and_owns_env_ids() -> None:
    env_ids = torch.tensor([3, 1], dtype=torch.long)
    context = EffectEvidenceCollectionContext(1.25, 7, env_ids)
    env_ids[0] = 99

    assert context.timestamp == 1.25
    assert context.observation_revision == 7
    assert context.env_ids.tolist() == [3, 1]
    assert context.snapshot().env_ids.data_ptr() != context.env_ids.data_ptr()

    with pytest.raises(ValueError, match="unique"):
        EffectEvidenceCollectionContext(0.0, 0, torch.tensor([1, 1]))
    with pytest.raises(ValueError, match="non-negative"):
        EffectEvidenceCollectionContext(-0.1, 0, torch.tensor([0]))


def test_build_queries_preserves_clause_order_and_exact_types() -> None:
    spec = _attach_spec(_pose_clause(), _binary_clause(), _scalar_clause())

    queries = build_effect_evidence_queries(spec)

    assert tuple(type(query) for query in queries) == (
        PoseRelationEvidenceQuery,
        BinaryEffectEvidenceQuery,
        ScalarEffectEvidenceQuery,
    )
    assert tuple(query.evidence_id for query in queries) == (
        "pose",
        "contact",
        "force",
    )
    assert all(query.expectation.expectation_id == "held" for query in queries)


def test_provider_registry_requires_exact_unique_versions() -> None:
    first = _WrongTimestampProvider()
    second = _SecondRevisionProvider()
    registry = EffectEvidenceProviderRegistry((first, second))

    source_v1 = _source(CONTACT_EFFECT_CHANNEL, provider_id="test.provider")
    assert registry.resolve(source_v1) is first
    assert registry.providers[("test.provider", "2")] is second

    with pytest.raises(ValueError, match="Duplicate"):
        EffectEvidenceProviderRegistry((first, _WrongTimestampProvider()))
    with pytest.raises(KeyError, match="exact versions"):
        registry.resolve(
            EffectEvidenceSourceRef(
                "test.provider",
                "missing",
                ControlPartEvidenceAddress("arm", CONTACT_EFFECT_CHANNEL),
            )
        )


def test_collector_rejects_provider_metadata_drift() -> None:
    spec = _attach_spec(_binary_clause(provider_id="test.provider"))
    collector = EffectEvidenceCollector(
        EffectEvidenceProviderRegistry((_WrongTimestampProvider(),))
    )

    with pytest.raises(ValueError, match="collection timestamp"):
        collector.collect(spec, timestamp=2.0, observation_revision=4)


def test_control_part_provider_collects_pose_and_joint_state_once() -> None:
    object_poses = torch.eye(4).repeat(2, 1, 1)
    object_poses[:, 0, 3] = torch.tensor([0.25, 0.5])
    robot = _FakeRobot(torch.tensor([[0.75, 1.0], [1.5, 2.0]]))
    scene = _FakeSceneProvider(object_poses)
    joint_clause = JointStateEffectClause(
        "joints",
        "held",
        _source(JOINT_STATE_EFFECT_CHANNEL),
        torch.tensor([0.0, 0.0]),
    )
    spec = _attach_spec(_pose_clause(), joint_clause)
    collector = EffectEvidenceCollector(
        EffectEvidenceProviderRegistry(
            (ControlPartSimulationEvidenceProvider(robot, scene_provider=scene),)
        )
    )

    evidence = collector.collect(spec, timestamp=3.5, observation_revision=11)

    assert list(evidence) == ["pose", "joints"]
    assert evidence["pose"].timestamp == 3.5
    assert evidence["joints"].observation_revision == 11
    assert torch.allclose(
        evidence["pose"].object_to_endpoint[:, 0, 3],
        torch.tensor([0.5, 1.0]),
    )
    assert torch.equal(evidence["joints"].positions, robot.qpos)
    assert torch.equal(evidence["joints"].velocities, robot.qvel)
    assert scene.calls == 1
    assert robot.qpos_calls == 1
    assert robot.qvel_calls == 1
    assert robot.fk_calls == 1


def test_control_part_provider_selects_requested_simulator_rows() -> None:
    env_ids = torch.tensor([2, 0], dtype=torch.long)
    object_poses = torch.eye(4).repeat(3, 1, 1)
    robot = _FakeRobot(torch.tensor([[1.0], [2.0], [3.0]]))
    scene = _FakeSceneProvider(object_poses)
    spec = _attach_spec(_pose_clause(), env_ids=env_ids)
    collector = EffectEvidenceCollector(
        EffectEvidenceProviderRegistry(
            (ControlPartSimulationEvidenceProvider(robot, scene_provider=scene),)
        )
    )

    evidence = collector.collect(spec, timestamp=0.0, observation_revision=0)

    assert evidence["pose"].env_ids.tolist() == [2, 0]
    assert evidence["pose"].object_to_endpoint[:, 0, 3].tolist() == [3.0, 1.0]
    assert scene.received_env_ids is not None
    assert scene.received_env_ids.tolist() == [2, 0]


def test_pose_queries_share_one_scene_and_fk_snapshot() -> None:
    robot = _FakeRobot(torch.tensor([[0.0], [0.0]]))
    scene = _FakeSceneProvider(torch.eye(4).repeat(2, 1, 1))
    spec = _attach_spec(_pose_clause("first"), _pose_clause("second"))
    collector = EffectEvidenceCollector(
        EffectEvidenceProviderRegistry(
            (ControlPartSimulationEvidenceProvider(robot, scene_provider=scene),)
        )
    )

    evidence = collector.collect(spec, timestamp=1.0, observation_revision=1)

    assert set(evidence) == {"first", "second"}
    assert scene.calls == 1
    assert robot.fk_calls == 1
    assert robot.qpos_calls == 1


def test_missing_backend_specific_callbacks_return_explicit_invalid_rows() -> None:
    robot = _FakeRobot(torch.zeros((2, 1)))
    spec = _attach_spec(_binary_clause(), _scalar_clause())
    collector = EffectEvidenceCollector(
        EffectEvidenceProviderRegistry((ControlPartSimulationEvidenceProvider(robot),))
    )

    evidence = collector.collect(spec, timestamp=1.0, observation_revision=2)

    assert not evidence["contact"].valid.any()
    assert not evidence["force"].valid.any()
    assert all("callback" in error for error in evidence["contact"].acquisition_errors)
    assert all("callback" in error for error in evidence["force"].acquisition_errors)


def test_callbacks_receive_owned_queries_and_propagate_row_validity() -> None:
    robot = _FakeRobot(torch.zeros((2, 1)))
    binary_values = torch.tensor([True, False])
    scalar_values = torch.tensor([3.0, 0.0])
    received_query: BinaryEffectEvidenceQuery | None = None

    def observe_contact(
        query: BinaryEffectEvidenceQuery,
        context: EffectEvidenceCollectionContext,
    ) -> BinaryEffectObservation:
        nonlocal received_query
        received_query = query
        assert context.env_ids.tolist() == [0, 1]
        return BinaryEffectObservation(
            binary_values,
            torch.tensor([True, False]),
            (None, "contact sensor unavailable"),
        )

    def observe_force(
        query: ScalarEffectEvidenceQuery,
        context: EffectEvidenceCollectionContext,
    ) -> ScalarEffectObservation:
        del query, context
        return ScalarEffectObservation(scalar_values)

    provider = ControlPartSimulationEvidenceProvider(
        robot,
        contact_observer=observe_contact,
        force_observer=observe_force,
    )
    collector = EffectEvidenceCollector(EffectEvidenceProviderRegistry((provider,)))

    evidence = collector.collect(
        _attach_spec(_binary_clause(), _scalar_clause()),
        timestamp=1.0,
        observation_revision=2,
    )
    binary_values[:] = False
    scalar_values[:] = 99.0

    assert received_query is not None
    assert received_query.evidence_id == "contact"
    assert evidence["contact"].values.tolist() == [True, False]
    assert evidence["contact"].valid.tolist() == [True, False]
    assert evidence["force"].values.tolist() == [3.0, 0.0]


def test_pose_without_scene_provider_is_invalid_not_fabricated() -> None:
    robot = _FakeRobot(torch.zeros((2, 1)))
    collector = EffectEvidenceCollector(
        EffectEvidenceProviderRegistry((ControlPartSimulationEvidenceProvider(robot),))
    )

    evidence = collector.collect(
        _attach_spec(_pose_clause()),
        timestamp=0.0,
        observation_revision=0,
    )

    assert not evidence["pose"].valid.any()
    assert all(
        "scene provider" in error for error in evidence["pose"].acquisition_errors
    )
    assert robot.fk_calls == 0


def test_channel_mismatch_fails_before_callback() -> None:
    wrong_clause = BinaryEffectClause(
        "contact",
        "held",
        _source(CONSTRAINT_EFFECT_CHANNEL),
        BinaryEvidenceKind.CONTACT,
        True,
    )
    robot = _FakeRobot(torch.zeros((2, 1)))
    collector = EffectEvidenceCollector(
        EffectEvidenceProviderRegistry((ControlPartSimulationEvidenceProvider(robot),))
    )

    with pytest.raises(ValueError, match="requires channel"):
        collector.collect(
            _attach_spec(wrong_clause),
            timestamp=0.0,
            observation_revision=0,
        )


def test_joint_query_type_is_built_for_articulation_expectation() -> None:
    expectation = ArticulationJointStateExpectation(
        "drawer_joint",
        "drawer",
        "slide",
        torch.tensor([0.4]),
    )
    clause = JointStateEffectClause(
        "joint",
        "drawer_joint",
        _source(JOINT_STATE_EFFECT_CHANNEL),
        torch.tensor([0.4]),
    )
    spec = SemanticEffectSpec(
        semantic_id="open:drawer",
        effect_kind=SemanticEffectKind.ARTICULATION,
        skill_id="test_articulation",
        invocation_id="open-1",
        invocation_revision=0,
        env_ids=torch.tensor([0]),
        state_expectations=(expectation,),
        clauses=(clause,),
    )

    query = build_effect_evidence_queries(spec)[0]

    assert isinstance(query, JointStateEvidenceQuery)
    assert query.expectation.articulation_id == "drawer"


def _articulation_spec(
    *clauses: JointStateEffectClause,
    expectation_joint: str = "slide",
) -> SemanticEffectSpec:
    expectation = ArticulationJointStateExpectation(
        "drawer_joint",
        "drawer",
        expectation_joint,
        torch.tensor([[0.4], [0.4]]),
    )
    return SemanticEffectSpec(
        semantic_id="open:drawer",
        effect_kind=SemanticEffectKind.ARTICULATION,
        skill_id="test_articulation",
        invocation_id="open-1",
        invocation_revision=0,
        env_ids=torch.tensor([0, 1]),
        state_expectations=(expectation,),
        clauses=clauses,
    )


def _articulation_clause(clause_id: str = "joint") -> JointStateEffectClause:
    return JointStateEffectClause(
        clause_id,
        "drawer_joint",
        EffectEvidenceSourceRef(
            SCENE_ARTICULATION_EVIDENCE_PROVIDER_ID,
            SCENE_ARTICULATION_EVIDENCE_PROVIDER_REVISION,
            ArticulationJointEvidenceAddress("drawer", "slide"),
        ),
        torch.tensor([[0.4], [0.4]]),
    )


def test_scene_articulation_provider_uses_explicit_typed_observer() -> None:
    calls = 0

    def observe_joint(
        query: JointStateEvidenceQuery,
        context: EffectEvidenceCollectionContext,
    ) -> JointStateObservation:
        nonlocal calls
        calls += 1
        address = query.source.address
        assert isinstance(address, ArticulationJointEvidenceAddress)
        assert (address.articulation_id, address.joint_id) == ("drawer", "slide")
        assert context.observation_revision == 8
        return JointStateObservation(
            positions=torch.tensor([[0.4], [0.3]]),
            velocities=torch.tensor([[0.0], [0.1]]),
            valid=torch.tensor([True, False]),
            acquisition_errors=(None, "joint sensor unavailable"),
        )

    provider = SceneArticulationEvidenceProvider(observe_joint)
    collector = EffectEvidenceCollector(EffectEvidenceProviderRegistry((provider,)))

    evidence = collector.collect(
        _articulation_spec(_articulation_clause()),
        timestamp=4.0,
        observation_revision=8,
    )

    assert calls == 1
    assert torch.allclose(
        evidence["joint"].positions,
        torch.tensor([[0.4], [0.3]]),
    )
    assert evidence["joint"].valid.tolist() == [True, False]
    assert evidence["joint"].acquisition_errors == (
        None,
        "joint sensor unavailable",
    )


def test_scene_articulation_provider_reads_typed_scene_snapshot_once() -> None:
    class _JointSceneProvider:
        calls = 0

        def snapshot(
            self,
            *,
            timestamp: float,
            env_ids: torch.Tensor,
        ) -> SceneSnapshot:
            self.calls += 1
            return SceneSnapshot(
                timestamp=timestamp,
                version=self.calls,
                articulation_joints={
                    ("drawer", "slide"): ObservedArticulationJointState(
                        torch.tensor([[0.4], [0.3]]),
                        torch.tensor([True, False]),
                    )
                },
            )

    scene_provider = _JointSceneProvider()
    collector = EffectEvidenceCollector(
        EffectEvidenceProviderRegistry(
            (SceneArticulationEvidenceProvider(scene_provider=scene_provider),)
        )
    )

    evidence = collector.collect(
        _articulation_spec(
            _articulation_clause("position"),
            _articulation_clause("settled_position"),
        ),
        timestamp=2.0,
        observation_revision=5,
    )

    assert scene_provider.calls == 1
    assert torch.equal(evidence["position"].positions, torch.tensor([[0.4], [0.3]]))
    assert evidence["settled_position"].valid.tolist() == [True, False]


def test_scene_articulation_provider_samples_same_address_once() -> None:
    calls = 0

    def observe_joint(
        query: JointStateEvidenceQuery,
        context: EffectEvidenceCollectionContext,
    ) -> JointStateObservation:
        nonlocal calls
        del query, context
        calls += 1
        return JointStateObservation(torch.tensor([[0.4], [0.4]]))

    collector = EffectEvidenceCollector(
        EffectEvidenceProviderRegistry(
            (SceneArticulationEvidenceProvider(observe_joint),)
        )
    )
    spec = _articulation_spec(
        _articulation_clause("position"),
        _articulation_clause("settled_position"),
    )

    evidence = collector.collect(spec, timestamp=1.0, observation_revision=1)

    assert set(evidence) == {"position", "settled_position"}
    assert calls == 1


def test_scene_articulation_provider_rejects_address_expectation_drift() -> None:
    provider = SceneArticulationEvidenceProvider(
        lambda query, context: JointStateObservation(torch.tensor([[0.4], [0.4]]))
    )
    collector = EffectEvidenceCollector(EffectEvidenceProviderRegistry((provider,)))

    with pytest.raises(ValueError, match="exactly match"):
        collector.collect(
            _articulation_spec(
                _articulation_clause(),
                expectation_joint="other_joint",
            ),
            timestamp=1.0,
            observation_revision=1,
        )
