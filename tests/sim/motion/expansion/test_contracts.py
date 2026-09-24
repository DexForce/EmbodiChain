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

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from embodichain.lab.sim.motion.expansion import (
    CandidateCoordinator,
    CandidateIdentity,
    CandidateSpec,
    CandidateTrajectoryBatch,
    CommitReceipt,
    ExpertEpisode,
    MotionSnapshot,
    SceneCase,
    TrajectoryPhase,
    TrajectoryTemplate,
    ValidationCheck,
    ValidationResult,
)
from embodichain.lab.sim.motion.expansion.source import (
    PlanResultSourceAdapter,
    SourceContext,
    TemplateSourceAdapter,
)
from embodichain.lab.sim.motion.expansion.cfg import TrajectoryGenerationJobCfg
from embodichain.lab.sim.motion.expansion.session import GenerationSession
from embodichain.lab.sim.motion.planners import PlanResult


def _case() -> SceneCase:
    return SceneCase("case", "initial", "signature", "task", "robot")


def _source_context() -> SourceContext:
    return SourceContext("motion", "revision", "unit", _case(), 0.1)


def _identity(candidate_id: str = "candidate") -> CandidateIdentity:
    return CandidateIdentity(
        "case", "initial", candidate_id, "geometry", "source", "revision", "template"
    )


def test_candidate_spec_keeps_physical_identity_separate_from_observation_fanout():
    spec = CandidateSpec(
        identity=_identity(),
        affordance_selection={"geometry_family": "grasp_a", "ordinal": 2},
        trajectory_variant={"operator": "via_points", "ordinal": 1},
        compatibility_key="case:initial:dt:validator",
        estimated_cost=1.25,
        observation_profiles=("rgb_train_aug_v1", "depth_eval_v1"),
    )

    assert spec.identity.candidate_id == "candidate"
    assert spec.affordance_selection["geometry_family"] == "grasp_a"
    assert spec.trajectory_variant["operator"] == "via_points"
    assert spec.observation_profiles == ("rgb_train_aug_v1", "depth_eval_v1")
    with pytest.raises(TypeError):
        spec.affordance_selection["new"] = "mutation"  # type: ignore[index]


def test_source_context_keeps_provider_and_physical_clock_explicit():
    context = SourceContext(
        source_id="atomic_pick",
        source_revision="git:abc",
        unit_id="call_0",
        scene_case=_case(),
        control_dt=0.02,
    )
    assert context.scene_case.scene_case_id == "case"
    assert context.control_dt == pytest.approx(0.02)


def test_source_adapters_share_one_template_contract():
    context = SourceContext(
        source_id="motion",
        source_revision="git:abc",
        unit_id="unit_0",
        scene_case=_case(),
        control_dt=0.1,
    )
    source_template = _template()
    handwritten = TemplateSourceAdapter().export_template(
        source_template, context=context
    )
    generated = PlanResultSourceAdapter(
        joint_names=("arm", "tool"),
    ).export_template(
        PlanResult(
            success=True,
            positions=source_template.positions.unsqueeze(0),
            dt=source_template.dt.unsqueeze(0),
        ),
        context=context,
    )

    assert handwritten.template_id == generated.template_id == "unit_0"
    torch.testing.assert_close(handwritten.positions, generated.positions)
    torch.testing.assert_close(handwritten.dt, generated.dt)


def test_plan_result_source_normalizes_representable_arrival_intervals() -> None:
    result = PlanResult(
        success=True,
        positions=torch.tensor([[[0.0], [0.0], [1.0], [1.0]]]),
        dt=torch.tensor([[0.2, 0.0, 0.3, 0.0]]),
    )
    phase = TrajectoryPhase("free", 0, 4, "free", ("joint_residual",))

    template = PlanResultSourceAdapter(
        ("joint",),
        phases=(phase,),
    ).export_template(result, context=_source_context())

    assert template.dt.tolist() == pytest.approx([0.0, 0.2, 0.3])
    assert template.positions[:, 0].tolist() == [0.0, 0.0, 1.0]
    assert template.phases == (
        TrajectoryPhase("free", 0, 3, "free", ("joint_residual",)),
    )
    assert template.allowed_operators == ("joint_residual",)


def test_plan_result_source_rejects_zero_duration_motion() -> None:
    result = PlanResult(
        success=True,
        positions=torch.tensor([[[0.0], [1.0]]]),
        dt=torch.tensor([[0.0, 0.0]]),
    )

    with pytest.raises(ValueError, match="zero-duration position change"):
        PlanResultSourceAdapter(("joint",)).export_template(
            result,
            context=_source_context(),
        )


def test_plan_result_source_allows_explicit_template_permissions() -> None:
    result = PlanResult(
        success=True,
        positions=torch.tensor([[[0.0], [0.5], [1.0]]]),
        dt=torch.tensor([[0.0, 0.1, 0.1]]),
    )
    phase = TrajectoryPhase("free", 0, 3, "free", ("joint_residual",))

    template = PlanResultSourceAdapter(
        ("joint",),
        phases=(phase,),
        allowed_operators=("via_points",),
    ).export_template(result, context=_source_context())

    assert template.allowed_operators == ("via_points",)


def test_plan_result_source_rejects_duplicate_template_permissions() -> None:
    result = PlanResult(
        success=True,
        positions=torch.tensor([[[0.0], [0.5], [1.0]]]),
        dt=torch.tensor([[0.0, 0.1, 0.1]]),
    )

    with pytest.raises(ValueError, match="allowed_operators must be unique"):
        PlanResultSourceAdapter(
            ("joint",),
            allowed_operators=("via_points", "via_points"),
        ).export_template(result, context=_source_context())


def test_candidate_coordinator_queues_source_variants_with_session_identity():
    cfg = TrajectoryGenerationJobCfg.from_mapping(
        {
            "augmentation": {
                "factors": {
                    "spatial": {
                        "enabled": True,
                        "method": ["joint_residual"],
                        "joint_offset_scale": 0.1,
                    }
                },
                "coverage": {"joint_dedup_normalized_tol": 0.001},
            },
            "scheduling": {"candidate_budget": 4},
        }
    )
    case = _case()
    session = GenerationSession(cfg)
    limits = torch.tensor([[-2.0, 2.0], [-2.0, 2.0]])
    session.register_case(case, limits, joint_names=("arm", "tool"))
    context = SourceContext(
        source_id="handwritten",
        source_revision="test",
        unit_id="unit_0",
        scene_case=case,
        control_dt=0.1,
    )
    coordinator = CandidateCoordinator(
        cfg,
        session=session,
        source_adapter=TemplateSourceAdapter(),
        source_context=context,
        joint_limits=limits,
    )

    source_template = _template(
        phases=(TrajectoryPhase("free", 0, 3, "free", ("joint_residual",)),),
        allowed_operators=("joint_residual",),
        controlled_joint_indices=(0, 1),
    )
    items = coordinator.enqueue_source(source_template, count=2)

    assert len(items) == 2
    assert coordinator.pending_count == 2
    assert len({item.spec.identity.candidate_id for item in items}) == 2
    assert all(item.spec.identity.source_id == "handwritten" for item in items)
    first = coordinator.take_next()
    assert first == items[0]
    assert first.to_batch().identities[0] == first.spec.identity
    coordinator.admit_planned(
        first,
        ValidationResult((ValidationCheck("path_collision", "passed"),)),
    )
    ready = session.take_ready("case", "initial", episode_byte_budget=1024)
    assert ready is not None
    assert ready.identities[0] == first.spec.identity
    assert coordinator.pending_count == 1


def _template(**changes: object) -> TrajectoryTemplate:
    fields = dict(
        source_id="source",
        source_revision="revision",
        template_id="template",
        joint_names=("arm", "tool"),
        positions=torch.tensor([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]]),
        dt=torch.tensor([0.0, 0.1, 0.2]),
    )
    fields.update(changes)
    return TrajectoryTemplate(**fields)


def _coordinator_fixture(
    *,
    max_proposals: int = 100,
) -> tuple[
    GenerationSession,
    CandidateCoordinator,
    TrajectoryTemplate,
]:
    cfg = TrajectoryGenerationJobCfg.from_mapping(
        {
            "augmentation": {
                "factors": {
                    "spatial": {
                        "enabled": True,
                        "method": ["joint_residual"],
                        "joint_offset_scale": 0.1,
                    }
                },
                "coverage": {"joint_dedup_normalized_tol": 0.001},
            },
            "scheduling": {"candidate_budget": 8},
            "collection": {"max_proposals": max_proposals},
        }
    )
    case = _case()
    limits = torch.tensor([[-2.0, 2.0], [-2.0, 2.0]])
    session = GenerationSession(cfg)
    session.register_case(case, limits, joint_names=("arm", "tool"))
    coordinator = CandidateCoordinator(
        cfg,
        session=session,
        source_adapter=TemplateSourceAdapter(),
        source_context=SourceContext(
            "handwritten",
            "test",
            "unit_0",
            case,
            0.1,
        ),
        joint_limits=limits,
    )
    template = _template(
        phases=(TrajectoryPhase("free", 0, 3, "free", ("joint_residual",)),),
        allowed_operators=("joint_residual",),
        controlled_joint_indices=(0, 1),
    )
    return session, coordinator, template


def test_candidate_coordinator_refills_with_novel_variants() -> None:
    _, coordinator, template = _coordinator_fixture()

    first = coordinator.enqueue_source(template, count=2)
    second = coordinator.enqueue_source(template, count=2)

    assert len(first) == 2
    assert len(second) == 1
    assert not torch.equal(first[1].template.positions, second[0].template.positions)


def test_candidate_coordinator_namespaces_affordance_geometry() -> None:
    _, coordinator, template = _coordinator_fixture()
    first_template = replace(
        template,
        positions=torch.tensor([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]]),
    )
    second_template = replace(
        template,
        positions=torch.tensor([[0.0, 0.0], [-0.5, 0.0], [1.0, 0.0]]),
    )

    first = coordinator.enqueue_source(
        first_template,
        count=1,
        affordance_selection={"grasp": "left"},
    )[0]
    second = coordinator.enqueue_source(
        second_template,
        count=1,
        affordance_selection={"grasp": "right"},
    )[0]

    assert first.spec.identity.candidate_id != second.spec.identity.candidate_id
    assert (
        first.spec.identity.geometry_family_id
        != second.spec.identity.geometry_family_id
    )


def test_candidate_coordinator_enqueue_is_atomic() -> None:
    session, coordinator, template = _coordinator_fixture(max_proposals=1)

    with pytest.raises(RuntimeError, match="Proposal budget"):
        coordinator.enqueue_source(template, count=2)

    assert coordinator.pending_count == 0
    assert session.snapshot()["counts"]["proposed"] == 0
    assert session.snapshot()["audit"] == ()


def _batch(**changes: object) -> CandidateTrajectoryBatch:
    fields = dict(
        positions=torch.tensor(
            [[[0.0, 0.0], [1.0, 0.0], [1.0, 0.0]], [[2.0, 0.0], [3.0, 0.0], [4.0, 0.0]]]
        ),
        dt=torch.tensor([[0.0, 0.1, 0.0], [0.0, 0.1, 0.2]]),
        valid_length=torch.tensor([2, 3]),
        identities=(_identity("candidate_0"), _identity("candidate_1")),
        joint_names=("arm", "tool"),
        source_row_indices=torch.tensor([0, 0]),
        factors=({"duration_scale": 1.0}, {"duration_scale": 2.0}),
        phases=(
            (TrajectoryPhase("transit", 0, 2),),
            (TrajectoryPhase("transit", 0, 3),),
        ),
    )
    fields.update(changes)
    return CandidateTrajectoryBatch(**fields)


def _episode(**changes: object) -> ExpertEpisode:
    fields = dict(
        identity=_identity(),
        observations={
            "joint_positions": torch.tensor([[0.0], [0.1], [0.2]]),
            "image": torch.zeros((3, 2, 2, 3), dtype=torch.uint8),
        },
        actions=torch.tensor([[0.1], [0.2]]),
        timestamps=torch.tensor([0.0, 0.1, 0.2]),
        action_representation="joint_position_target",
        validation=ValidationResult((ValidationCheck("task_success", "passed"),)),
        episode_id="episode",
        commit_id="commit",
    )
    fields.update(changes)
    return ExpertEpisode(**fields)


def test_snapshot_copies_tensor_and_revision_inputs() -> None:
    positions = torch.tensor([0.0, 1.0], requires_grad=True)
    pose = torch.eye(4)
    revisions = {"object": 1}
    snapshot = MotionSnapshot(
        _case(),
        ("arm", "tool"),
        positions,
        torch.zeros(2),
        pose,
        {"object": pose},
        revisions,
    )
    with torch.no_grad():
        positions[:] = 9
    pose[0, 3] = 5
    revisions["object"] = 2
    assert snapshot.joint_positions.tolist() == [0.0, 1.0]
    assert not snapshot.joint_positions.requires_grad
    assert snapshot.root_pose[0, 3] == 0
    assert snapshot.entity_poses["object"][0, 3] == 0
    assert snapshot.dependency_revisions["object"] == 1
    with pytest.raises(TypeError):
        snapshot.dependency_revisions["object"] = 3


@pytest.mark.parametrize(
    "pose",
    [
        torch.zeros((4, 4)),
        torch.diag(torch.tensor([-1.0, 1.0, 1.0, 1.0])),
        torch.eye(3),
        torch.full((4, 4), float("nan")),
    ],
)
def test_snapshot_rejects_invalid_or_reflected_se3(pose: torch.Tensor) -> None:
    with pytest.raises(ValueError):
        MotionSnapshot(_case(), ("arm",), torch.zeros(1), torch.zeros(1), pose)


def test_snapshot_rejects_incomplete_joint_state() -> None:
    with pytest.raises(ValueError, match="complete joint_names"):
        MotionSnapshot(
            _case(), ("arm", "tool"), torch.zeros(1), torch.zeros(1), torch.eye(4)
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"dt": torch.tensor([0.1, 0.1, 0.1])},
        {"dt": torch.tensor([0.0, 0.0, 0.1])},
        {"positions": torch.full((3, 2), float("inf"))},
        {"positions": torch.ones((3, 2), dtype=torch.int64)},
        {"joint_names": ("arm", "arm")},
        {"controlled_joint_indices": (2,)},
        {"controlled_joint_indices": (True,)},
        {"controlled_joint_indices": (0, 0)},
        {"representation": "eef_delta"},
        {"phases": (TrajectoryPhase("transit", 0, 4),)},
        {"phases": (TrajectoryPhase("a", 0, 2), TrajectoryPhase("b", 1, 3))},
    ],
)
def test_template_rejects_invalid_representation_timing_and_indices(
    changes: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        _template(**changes)


def test_template_owns_inputs_and_leaves_unannotated_path_replay_only() -> None:
    positions = torch.zeros((3, 2))
    dt = torch.tensor([0.0, 0.1, 0.1])
    template = _template(positions=positions, dt=dt)
    positions.fill_(1)
    dt.fill_(1)
    assert template.positions.count_nonzero() == 0
    assert template.dt.tolist() == pytest.approx([0.0, 0.1, 0.1])
    assert template.phases == ()
    assert template.allowed_operators == ()


def test_batch_preserves_repeated_source_rows_and_metadata_on_extraction() -> None:
    batch = _batch()
    assert batch.valid_mask.tolist() == [[True, True, False], [True, True, True]]
    row = batch.row(1)
    assert row.source_row_indices.tolist() == [0]
    assert row.identities == (batch.identities[1],)
    assert row.phases[0][0].stop_index == 3
    assert row.factors[0]["duration_scale"] == 2.0
    row.positions.fill_(9)
    assert batch.positions[1, 0, 0] == 2


def test_batch_accepts_empty_candidate_set() -> None:
    batch = CandidateTrajectoryBatch(
        torch.empty((0, 0, 2)),
        torch.empty((0, 0)),
        torch.empty((0,), dtype=torch.int64),
        (),
        ("arm", "tool"),
    )
    assert batch.valid_mask.shape == (0, 0)
    with pytest.raises(IndexError):
        batch.row(0)


@pytest.mark.parametrize(
    "changes",
    [
        {"valid_length": torch.tensor([0, 3])},
        {"valid_length": torch.tensor([2, 4])},
        {"valid_length": torch.tensor([2.0, 3.0])},
        {"dt": torch.tensor([[0.0, 0.1, 0.1], [0.0, 0.1, 0.1]])},
        {"positions": torch.zeros((2, 3, 3))},
        {"identities": (_identity(), _identity())},
        {"factors": ({"scale": float("nan")}, {})},
        {"source_row_indices": torch.tensor([0, -1])},
        {"phases": ((TrajectoryPhase("transit", 0, 3),), ())},
    ],
)
def test_batch_rejects_invalid_lengths_padding_and_alignment(
    changes: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        _batch(**changes)


def test_batch_rejects_padding_that_moves_the_robot() -> None:
    positions = _batch().positions
    positions[0, 2, 0] = 5
    with pytest.raises(ValueError, match="padded positions"):
        _batch(positions=positions)


@pytest.mark.parametrize(
    "status,accepted",
    [("not_run", False), ("passed", True), ("failed", False), ("unavailable", False)],
)
def test_validation_accepts_only_completed_passing_checks(
    status: str, accepted: bool
) -> None:
    result = ValidationResult(
        (ValidationCheck("collision", status, metrics={"clearance": 0.02}),)
    )
    assert result.accepted is accepted


def test_validation_does_not_accept_empty_or_duplicate_checks() -> None:
    assert not ValidationResult(()).accepted
    with pytest.raises(ValueError, match="unique"):
        ValidationResult(
            (
                ValidationCheck("collision", "passed"),
                ValidationCheck("collision", "passed"),
            )
        )
    with pytest.raises(ValueError, match="finite"):
        ValidationCheck("collision", "passed", metrics={"clearance": float("nan")})


def test_episode_preserves_terminal_observation_and_copies_evidence() -> None:
    observations = {"joint_positions": torch.tensor([[0.0], [0.1], [0.2]])}
    metadata = {"lineage": ["parent"], "control_dt": 0.1}
    episode = _episode(observations=observations, metadata=metadata)
    observations["joint_positions"].fill_(9)
    metadata["lineage"].append("unexpected")
    assert episode.observations["joint_positions"][-1, 0] == pytest.approx(0.2)
    assert episode.actions.shape[0] + 1 == episode.timestamps.shape[0]
    assert episode.metadata["lineage"] == ("parent",)


@pytest.mark.parametrize(
    "changes",
    [
        {"observations": {"joint_positions": torch.zeros((2, 1))}},
        {"observations": {}},
        {"timestamps": torch.tensor([0.0, 0.1, 0.1])},
        {"actions": torch.zeros((0, 1))},
        {"metadata": {"host": object()}},
        {"metadata": {"metric": float("nan")}},
        {"phases": (TrajectoryPhase("transit", 0, 4),)},
    ],
)
def test_episode_rejects_noncausal_shapes_or_live_metadata(
    changes: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        _episode(**changes)


def test_receipts_use_stable_ids_and_distinct_write_attempts() -> None:
    receipt = CommitReceipt(
        "episode", "candidate", 0, "dataset/episode", "commit", "case"
    )
    retry = replace(receipt, submission_id=1, confirmed=False, error="storage offline")
    assert receipt.commit_id == retry.commit_id
    assert receipt.attempt_id == retry.attempt_id
    assert receipt.submission_id != retry.submission_id
    with pytest.raises(ValueError, match="only unconfirmed"):
        replace(receipt, error="inconsistent")
    with pytest.raises(ValueError, match="submission_id"):
        replace(receipt, submission_id=-1)
