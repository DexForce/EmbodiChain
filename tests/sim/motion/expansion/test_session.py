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

"""CPU lifecycle checks for fixed-scene collection accounting."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from embodichain.lab.sim.motion.expansion.cfg import (
    TrajectoryGenerationJobCfg,
)
from embodichain.lab.sim.motion.expansion.contracts import (
    CandidateIdentity,
    CandidateTrajectoryBatch,
    CommitReceipt,
    ExpertEpisode,
    SceneCase,
    TrajectoryPhase,
    ValidationCheck,
    ValidationResult,
)
from embodichain.lab.sim.motion.expansion.session import GenerationSession

CASE = SceneCase("case", "initial", "scene_v1", "lift", "robot")
JOINT_NAMES = ("joint_a", "joint_b")
LIMITS = torch.tensor([[-1.0, 1.0], [-1.0, 1.0]])
PLAN_VALID = ValidationResult((ValidationCheck("path_collision", "passed"),))
ROLLOUT_VALID = ValidationResult(
    (
        ValidationCheck("path_collision", "passed"),
        ValidationCheck("task_success", "passed"),
    )
)


def _session(**overrides: object) -> GenerationSession:
    cfg = TrajectoryGenerationJobCfg.from_mapping(overrides)
    session = GenerationSession(cfg)
    session.register_case(CASE, LIMITS, joint_names=JOINT_NAMES)
    return session


def _propose(session: GenerationSession, case: SceneCase = CASE, **kwargs: object):
    return session.propose(
        case.scene_case_id,
        case.initial_state_id,
        source_id="handwritten_qpos",
        source_revision="v1",
        template_id="reference_0",
        operator_id="joint_residual",
        **kwargs,
    )


def _batch(*identities: CandidateIdentity) -> CandidateTrajectoryBatch:
    # Two controlled joints and two waypoints; the host owns collision evidence.
    count = len(identities)
    return CandidateTrajectoryBatch(
        torch.zeros(count, 2, 2),
        torch.tensor([[0.0, 1.0]]).repeat(count, 1),
        torch.full((count,), 2, dtype=torch.int64),
        identities,
        ("joint_a", "joint_b"),
        source_row_indices=torch.arange(count),
    )


def _episode(
    session: GenerationSession,
    identity: CandidateIdentity,
    *,
    position: float = 0.5,
    duration: float = 1.0,
    validation: ValidationResult = ROLLOUT_VALID,
) -> ExpertEpisode:
    episode_id, commit_id = session.episode_ids(identity)
    return ExpertEpisode(
        identity,
        {"joint_positions": torch.tensor([[0.0, 0.0], [position, 0.0]])},
        torch.tensor([[position, 0.0]]),
        torch.tensor([0.0, duration]),
        "qpos",
        validation,
        episode_id,
        commit_id,
        phases=(TrajectoryPhase("move", 0, 2),),
    )


def _receipt(episode: ExpertEpisode, **kwargs: object) -> CommitReceipt:
    return CommitReceipt(
        episode.episode_id,
        episode.identity.candidate_id,
        episode.identity.attempt_id,
        "memory:episode",
        episode.commit_id,
        episode.identity.scene_case_id,
        **kwargs,
    )


def _start(
    session: GenerationSession,
    case: SceneCase = CASE,
    *,
    episode_byte_budget: int | None = 4096,
    **kwargs: object,
):
    identity, _ = _propose(session, case, **kwargs)
    session.add_planned(_batch(identity), PLAN_VALID)
    row = session.take_ready(
        case.scene_case_id,
        case.initial_state_id,
        episode_byte_budget=episode_byte_budget,
    )
    assert row.identities == (identity,)
    session.mark_rollout_started(identity)
    return identity


def test_proposals_use_case_local_ordinals_and_do_not_consume_global_rng() -> None:
    first, second = _session(), _session()
    other = replace(CASE, scene_case_id="other")
    second.register_case(other, LIMITS, joint_names=JOINT_NAMES)
    rng_before = torch.random.get_rng_state().clone()
    expected = [_propose(first) for _ in range(3)]
    actual = []
    for _ in range(3):
        _propose(second, other)
        actual.append(_propose(second))
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    for (left, left_rng), (right, right_rng) in zip(expected, actual):
        assert left == right
        assert torch.equal(
            torch.rand(5, generator=left_rng), torch.rand(5, generator=right_rng)
        )


@pytest.mark.parametrize(
    "limit", [{"ready_high_watermark": 1}, {"ready_max_bytes": 47}]
)
def test_ready_pool_rejects_whole_batch_when_count_or_bytes_would_overflow(
    limit,
) -> None:
    session = _session(execution={"ready_low_watermark": 0, **limit})
    identities = tuple(_propose(session)[0] for _ in range(2))
    with pytest.raises(BufferError, match="pool is full"):
        session.add_planned(_batch(*identities), PLAN_VALID)
    counts = session.snapshot()["counts"]
    assert counts["planned_valid"] == counts["ready"] == 0
    assert counts["proposed"] == 2


def test_planning_failure_and_assigned_cancellation_do_not_count_rollouts() -> None:
    session = _session()
    failed, _ = _propose(session)
    with pytest.raises(ValueError, match="mandatory planning checks"):
        session.add_planned(_batch(failed), ValidationResult(()))
    session.release(failed)
    ready, _ = _propose(session)
    session.add_planned(_batch(ready), PLAN_VALID)
    assert session.take_ready("case", "initial") is not None
    session.release(ready)
    assert session.snapshot()["counts"]["rollout_attempted"] == 0


def test_failed_validation_statuses_and_release_reason_survive_payload_discard() -> (
    None
):
    session = _session()
    planning, _ = _propose(session)
    unavailable = ValidationResult(
        (ValidationCheck("path_collision", "unavailable", "large backend trace"),)
    )
    with pytest.raises(ValueError, match="mandatory planning checks"):
        session.add_planned(_batch(planning), unavailable)
    session.release(planning, reason="collision provider missing")
    rollout = _start(session)
    failed = ValidationResult(
        (
            ValidationCheck("path_collision", "passed"),
            ValidationCheck("task_success", "failed"),
        )
    )
    assert not session.accept_episode(_episode(session, rollout, validation=failed))
    diagnostics = session.snapshot()["diagnostics"]
    assert diagnostics[planning.candidate_id]["reason"] == "collision provider missing"
    assert diagnostics[planning.candidate_id]["planning_validation"] == unavailable
    assert diagnostics[rollout.candidate_id]["reason"] == "rollout_validation_failed"
    assert diagnostics[rollout.candidate_id]["rollout_validation"] == failed
    with pytest.raises(TypeError):
        diagnostics[planning.candidate_id]["reason"] = "changed"


def test_tail_batch_reserves_global_target_before_first_commands() -> None:
    session = _session(collection={"target_committed_episodes": 2})
    identities = tuple(_propose(session)[0] for _ in range(3))
    session.add_planned(_batch(*identities), PLAN_VALID)
    first = session.take_ready("case", "initial", episode_byte_budget=4096)
    second = session.take_ready("case", "initial", episode_byte_budget=4096)
    assert first.identities == (identities[0],)
    assert second.identities == (identities[1],)
    assert session.take_ready("case", "initial") is None
    assert session.snapshot()["counts"]["rollout_attempted"] == 0
    session.release(identities[0])
    assert session.take_ready(
        "case", "initial", episode_byte_budget=4096
    ).identities == (identities[2],)


def test_assignment_also_reserves_remaining_rollout_attempts() -> None:
    session = _session(
        collection={"target_committed_episodes": 2, "max_rollout_attempts": 2}
    )
    failed = _start(session)
    session.release(failed)
    identities = tuple(_propose(session)[0] for _ in range(2))
    session.add_planned(_batch(*identities), PLAN_VALID)
    assert session.take_ready("case", "initial") is not None
    assert session.take_ready("case", "initial") is None
    session.mark_rollout_started(identities[0])
    session.release(identities[0])
    assert session.stop_reason == "rollout_budget_exhausted"


def test_candidate_rows_are_owned_and_keep_source_mapping() -> None:
    session = _session()
    identity, _ = _propose(session)
    batch = _batch(identity)
    session.add_planned(batch, PLAN_VALID)
    batch.positions.fill_(99)
    ready = session.take_ready("case", "initial")
    assert torch.equal(ready.positions, torch.zeros(1, 2, 2))
    assert ready.source_row_indices.tolist() == [0]


def test_candidate_joint_order_must_match_registered_limit_order() -> None:
    session = _session()
    identity, _ = _propose(session)
    swapped = replace(_batch(identity), joint_names=tuple(reversed(JOINT_NAMES)))
    with pytest.raises(ValueError, match="joint layout"):
        session.add_planned(swapped, PLAN_VALID)
    assert session.snapshot()["counts"]["planned_valid"] == 0


@pytest.mark.parametrize(
    "changed",
    [{"action_representation": "qvel"}, {"actions": torch.zeros(1, 1)}],
)
def test_unsupported_action_labels_cannot_enter_expert_dataset(changed) -> None:
    session = _session()
    episode = _episode(session, _start(session))
    with pytest.raises(ValueError, match="full-joint qpos"):
        session.accept_episode(replace(episode, **changed))
    assert session.snapshot()["counts"]["validated_accepted"] == 0


def test_large_factor_strings_cannot_bypass_ready_byte_limit() -> None:
    session = _session(execution={"ready_max_bytes": 4096})
    identity, _ = _propose(session)
    oversized = replace(_batch(identity), factors=({"description": "x" * 5000},))
    with pytest.raises(BufferError, match="pool is full"):
        session.add_planned(oversized, PLAN_VALID)
    assert session.snapshot()["counts"]["ready"] == 0


def test_large_episode_metadata_cannot_bypass_pending_byte_limit() -> None:
    session = _session(persistence={"pending_max_bytes": 4096})
    episode = _episode(session, _start(session))
    oversized = replace(episode, metadata={"nested": {"description": "x" * 5000}})
    with pytest.raises(BufferError, match="reserved episode byte budget"):
        session.accept_episode(oversized)
    assert session.snapshot()["counts"]["pending_write"] == 0


def test_only_confirmed_receipts_add_committed_count_and_formal_coverage() -> None:
    session = _session()
    identity = _start(session)
    episode = _episode(session, identity)
    assert session.accept_episode(episode)
    before = session.snapshot()
    assert before["counts"]["rollout_attempted"] == 1
    assert before["counts"]["validated_accepted"] == 1
    assert before["counts"]["pending_write"] == 1
    assert before["counts"]["committed"] == before["coverage"]["case"] == 0
    receipt = _receipt(episode)
    assert session.apply_receipt(receipt)
    assert not session.apply_receipt(receipt)
    after = session.snapshot()
    assert after["counts"]["committed"] == after["coverage"]["case"] == 1
    assert after["counts"]["pending_write"] == 0
    assert session.stop_reason == "target_reached"
    assert before["counts"]["committed"] == 0
    with pytest.raises(TypeError):
        after["counts"]["committed"] = 9


@pytest.mark.parametrize(
    "checks",
    [
        (),
        (ValidationCheck("task_success", "passed"),),
        (ValidationCheck("path_collision", "passed"),),
        (
            ValidationCheck("path_collision", "unavailable"),
            ValidationCheck("task_success", "passed"),
        ),
    ],
)
def test_missing_or_unavailable_required_validation_cannot_accept(checks) -> None:
    session = _session()
    identity = _start(session)
    assert not session.accept_episode(
        _episode(session, identity, validation=ValidationResult(checks))
    )
    assert session.snapshot()["counts"]["validated_accepted"] == 0
    assert session.snapshot()["counts"]["running"] == 0


def test_episode_and_receipt_must_match_original_candidate_identity() -> None:
    session = _session()
    identity = _start(session)
    episode = _episode(session, identity)
    with pytest.raises(ValueError, match="Episode and commit IDs"):
        session.accept_episode(replace(episode, episode_id="foreign"))
    assert session.accept_episode(episode)
    with pytest.raises(ValueError, match="original candidate"):
        session.apply_receipt(replace(_receipt(episode), scene_case_id="other"))
    assert session.snapshot()["counts"]["pending_write"] == 1


def test_failed_submission_retries_preserve_counts_and_ignore_old_duplicate_receipts() -> (
    None
):
    session = _session()
    identity = _start(session)
    episode = _episode(session, identity)
    session.accept_episode(episode)
    failure = _receipt(episode, confirmed=False, error="disk unavailable")
    session.apply_receipt(failure)
    assert session.snapshot()["counts"]["pending_write"] == 0
    payload, submission_id = session.retry_write(episode.commit_id)
    assert payload.commit_id == episode.commit_id
    assert submission_id == 1
    assert not session.apply_receipt(failure)
    assert session.snapshot()["counts"]["pending_write"] == 1
    with pytest.raises(ValueError, match="Contradictory final"):
        session.apply_receipt(_receipt(episode))
    assert session.apply_receipt(_receipt(episode, submission_id=submission_id))
    counts = session.snapshot()["counts"]
    assert (
        counts["rollout_attempted"]
        == counts["validated_accepted"]
        == counts["committed"]
        == 1
    )


def test_failed_write_cannot_retake_target_reserved_by_another_candidate() -> None:
    session = _session()
    episode = _episode(session, _start(session))
    session.accept_episode(episode)
    session.apply_receipt(_receipt(episode, confirmed=False))
    _start(session)
    with pytest.raises(RuntimeError, match="No collection quota"):
        session.retry_write(episode.commit_id)


def test_actual_geometry_controls_duplicates_and_timing_variants_share_coverage() -> (
    None
):
    session = _session(
        collection={"target_committed_episodes": 3},
        augmentation={"coverage": {"target_per_cell": 2}},
    )
    first = _start(session)
    first_episode = _episode(session, first)
    assert session.accept_episode(first_episode)
    # Proposed trajectories are identical zeros; real evidence decides coverage.
    distinct = _start(session)
    distinct_episode = _episode(session, distinct, position=-0.5)
    assert session.accept_episode(distinct_episode)
    session.apply_receipt(_receipt(distinct_episode))
    duplicate = _start(session)
    assert not session.accept_episode(_episode(session, duplicate))
    timing = _start(
        session,
        geometry_family_id=first.geometry_family_id,
        parent_id=first.candidate_id,
    )
    timing_episode = _episode(session, timing, duration=2.0)
    assert session.accept_episode(timing_episode)
    session.apply_receipt(_receipt(first_episode))
    session.apply_receipt(_receipt(timing_episode))
    assert session.snapshot()["counts"]["committed"] == 3
    assert session.snapshot()["coverage"]["case"] == 2


def test_same_measured_trajectory_has_independent_coverage_in_different_cases() -> None:
    session = _session(collection={"target_committed_episodes": 2})
    other = replace(CASE, scene_case_id="other")
    session.register_case(other, LIMITS, joint_names=JOINT_NAMES)
    first = _episode(session, _start(session))
    second = _episode(session, _start(session, other))
    assert session.accept_episode(first)
    assert session.accept_episode(second)
    # Confirmation arrives after another case was assigned, in reverse order.
    session.apply_receipt(_receipt(second))
    session.apply_receipt(_receipt(first))
    assert session.snapshot()["coverage"] == {"case": 1, "other": 1}


def test_pending_count_backpressure_prevents_rollout_and_retains_failed_writes() -> (
    None
):
    session = _session(
        collection={"target_committed_episodes": 2},
        persistence={"pending_episode_limit": 1},
    )
    identities = tuple(_propose(session)[0] for _ in range(2))
    session.add_planned(_batch(*identities), PLAN_VALID)
    session.take_ready("case", "initial", episode_byte_budget=4096)
    assert session.take_ready("case", "initial", episode_byte_budget=4096) is None
    session.mark_rollout_started(identities[0])
    assert session.take_ready("case", "initial", episode_byte_budget=4096) is None
    first = _episode(session, identities[0])
    assert session.accept_episode(first)
    session.apply_receipt(_receipt(first, confirmed=False))
    assert session.take_ready("case", "initial", episode_byte_budget=4096) is None
    assert session.snapshot()["counts"]["rollout_attempted"] == 1
    session.release(identities[0], reason="write_aborted")
    assert session.take_ready(
        "case", "initial", episode_byte_budget=4096
    ).identities == (identities[1],)


def test_assigned_byte_reservations_block_extra_rows_and_release_without_commands() -> (
    None
):
    session = _session(
        collection={"target_committed_episodes": 3},
        persistence={"pending_max_bytes": 8192},
    )
    identities = tuple(_propose(session)[0] for _ in range(3))
    session.add_planned(_batch(*identities), PLAN_VALID)
    session.take_ready("case", "initial", episode_byte_budget=4096)
    session.take_ready("case", "initial", episode_byte_budget=4096)
    assert session.snapshot()["pending_reserved_bytes"] == 8192
    assert session.take_ready("case", "initial", episode_byte_budget=1) is None
    session.release(identities[0])
    assert session.snapshot()["pending_reserved_bytes"] == 4096
    assert session.take_ready(
        "case", "initial", episode_byte_budget=4096
    ).identities == (identities[2],)
    assert session.snapshot()["counts"]["rollout_attempted"] == 0


def test_unspecified_episode_byte_budget_conservatively_serializes_until_confirmation() -> (
    None
):
    session = _session(
        collection={"target_committed_episodes": 2},
        persistence={"pending_max_bytes": 8192},
    )
    identities = tuple(_propose(session)[0] for _ in range(2))
    session.add_planned(_batch(*identities), PLAN_VALID)
    session.take_ready("case", "initial")
    assert session.snapshot()["pending_reserved_bytes"] == 8192
    assert session.take_ready("case", "initial") is None
    session.mark_rollout_started(identities[0])
    first = _episode(session, identities[0])
    assert session.accept_episode(first)
    assert 0 < session.snapshot()["pending_reserved_bytes"] < 8192
    assert session.take_ready("case", "initial") is None
    session.apply_receipt(_receipt(first))
    assert session.snapshot()["pending_reserved_bytes"] == 0
    assert session.take_ready("case", "initial").identities == (identities[1],)


def test_episode_exceeding_declared_byte_budget_releases_reserved_capacity() -> None:
    session = _session()
    identity = _start(session, episode_byte_budget=31)
    assert session.snapshot()["pending_reserved_bytes"] == 31
    with pytest.raises(BufferError, match="reserved episode byte budget"):
        session.accept_episode(_episode(session, identity))
    assert session.snapshot()["pending_reserved_bytes"] == 0
    assert session.snapshot()["counts"]["pending_write"] == 0
    assert (
        session.snapshot()["diagnostics"][identity.candidate_id]["reason"]
        == "episode_byte_budget_exceeded"
    )


@pytest.mark.parametrize("budget", [0, -1, True, 1.5, 8193])
def test_invalid_episode_byte_budget_does_not_assign_a_row(budget) -> None:
    session = _session(persistence={"pending_max_bytes": 8192})
    identity, _ = _propose(session)
    session.add_planned(_batch(identity), PLAN_VALID)
    with pytest.raises(ValueError, match="episode_byte_budget"):
        session.take_ready("case", "initial", episode_byte_budget=budget)
    assert session.snapshot()["counts"]["ready"] == 1
    assert session.snapshot()["pending_reserved_bytes"] == 0


def test_diagnostics_keep_bounded_failure_metrics_and_detail() -> None:
    session = _session()
    identity, _ = _propose(session)
    checks = ValidationResult(
        (
            ValidationCheck(
                "path_collision",
                "failed",
                "x" * 1024,
                {f"metric_{index}": float(index) for index in range(20)},
            ),
        )
    )
    with pytest.raises(ValueError, match="mandatory planning checks"):
        session.add_planned(_batch(identity), checks)
    summary = session.snapshot()["diagnostics"][identity.candidate_id][
        "planning_validation"
    ].checks[0]
    assert summary.status == "failed"
    assert summary.detail == "x" * 256
    assert len(summary.metrics) == 16
    assert summary.metrics["metric_15"] == 15.0


def test_wall_time_uses_injected_clock_and_still_allows_final_receipt() -> None:
    now = [0.0]
    session = GenerationSession(TrajectoryGenerationJobCfg(), clock=lambda: now[0])
    session.register_case(CASE, LIMITS, joint_names=JOINT_NAMES)
    episode = _episode(session, _start(session))
    assert session.accept_episode(episode)
    now[0] = 61.0
    assert session.stop_reason == "wall_time_exhausted"
    assert session.take_ready("case", "initial") is None
    with pytest.raises(RuntimeError, match="budget"):
        _propose(session)
    session.apply_receipt(_receipt(episode))
    assert session.snapshot()["counts"]["committed"] == 1


def test_proposal_budget_exhaustion_and_case_conditions_do_not_reset_history() -> None:
    session = _session(collection={"max_proposals": 1})
    identity, _ = _propose(session)
    session.release(identity)
    session.register_case(CASE, LIMITS, joint_names=JOINT_NAMES)
    assert session.stop_reason == "proposal_budget_exhausted"
    with pytest.raises(RuntimeError, match="budget"):
        _propose(session)
    with pytest.raises(ValueError, match="fixed conditions"):
        session.register_case(
            replace(CASE, scene_signature="changed"), LIMITS, joint_names=JOINT_NAMES
        )
