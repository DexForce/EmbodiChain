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

import pytest
import torch

import embodichain.lab.sim.motion.expansion as expansion
from embodichain.lab.sim.motion.execution import (
    EpisodeSink,
    FixedSceneInitialStatePort,
    InitialStatePort,
    MeasuredExecutor,
    SingleSlotOutcome,
    SingleSlotRunner,
)
from embodichain.lab.sim.motion.expansion import (
    CandidateCoordinator,
    CandidateSpec,
    CandidateWorkItem,
    CommitReceipt,
    ExpertEpisode,
    GenerationSession,
    SceneCase,
    SourceContext,
    TemplateSourceAdapter,
    TrajectoryGenerationJobCfg,
    TrajectoryPhase,
    TrajectoryTemplate,
    ValidationCheck,
    ValidationResult,
)


def _cfg(**collection: int) -> TrajectoryGenerationJobCfg:
    payload: dict[str, object] = {
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
    if collection:
        payload["collection"] = collection
    return TrajectoryGenerationJobCfg.from_mapping(payload)


def _template() -> TrajectoryTemplate:
    return TrajectoryTemplate(
        source_id="source",
        source_revision="source",
        template_id="template",
        joint_names=("arm", "tool"),
        positions=torch.tensor([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]]),
        dt=torch.tensor([0.0, 0.1, 0.1]),
        phases=(TrajectoryPhase("free", 0, 3, "free", ("joint_residual",)),),
        allowed_operators=("joint_residual",),
        controlled_joint_indices=(0, 1),
    )


def _coordinator(
    *,
    count: int = 1,
    **collection: int,
) -> tuple[GenerationSession, CandidateCoordinator, tuple[CandidateWorkItem, ...]]:
    cfg = _cfg(**collection)
    case = SceneCase("case", "initial", "signature", "task", "robot")
    limits = torch.tensor([[-2.0, 2.0], [-2.0, 2.0]])
    session = GenerationSession(cfg)
    session.register_case(case, limits, joint_names=("arm", "tool"))
    coordinator = CandidateCoordinator(
        cfg,
        session=session,
        source_adapter=TemplateSourceAdapter(),
        source_context=SourceContext("handwritten", "test", "unit", case, 0.1),
        joint_limits=limits,
    )
    items = coordinator.enqueue_source(_template(), count=count)
    return session, coordinator, items


class _Restorer:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.items: list[str] = []

    def restore(self, item: CandidateWorkItem) -> object:
        if self.fail:
            raise RuntimeError("restore failed")
        self.items.append(item.spec.identity.candidate_id)
        return {"epoch": len(self.items)}


class _Executor:
    def __init__(
        self,
        *,
        fail_validation: bool = False,
        fail_execution: bool = False,
        malformed_evidence: bool = False,
        start_rollout: bool = True,
    ) -> None:
        self.fail_validation = fail_validation
        self.fail_execution = fail_execution
        self.malformed_evidence = malformed_evidence
        self.start_rollout = start_rollout
        self.execution_count = 0

    def validate_plan(self, item: CandidateWorkItem) -> ValidationResult:
        del item
        if self.fail_validation:
            raise RuntimeError("planning failed")
        return ValidationResult((ValidationCheck("path_collision", "passed"),))

    def execute(
        self,
        item: CandidateWorkItem,
        prepared: object,
        *,
        episode_id: str,
        commit_id: str,
        on_rollout_started,
    ) -> ExpertEpisode:
        self.execution_count += 1
        assert prepared == {"epoch": self.execution_count}
        if self.start_rollout:
            on_rollout_started()
        if self.fail_execution:
            raise RuntimeError("execution failed")
        positions = item.template.positions
        observations = positions[:, :1] if self.malformed_evidence else positions
        return ExpertEpisode(
            identity=item.spec.identity,
            observations={"joint_positions": observations},
            actions=positions[:-1],
            timestamps=item.template.dt.cumsum(0),
            action_representation="qpos",
            validation=ValidationResult(
                (
                    ValidationCheck("path_collision", "passed"),
                    ValidationCheck("task_success", "passed"),
                )
            ),
            episode_id=episode_id,
            commit_id=commit_id,
            phases=item.template.phases,
        )


class _Sink:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail

    def submit(
        self,
        episode: ExpertEpisode,
        *,
        submission_id: int,
    ) -> CommitReceipt:
        if self.fail:
            raise RuntimeError("ambiguous sink failure")
        return CommitReceipt(
            episode_id=episode.episode_id,
            candidate_id=episode.identity.candidate_id,
            attempt_id=episode.identity.attempt_id,
            storage_id="memory://episode",
            commit_id=episode.commit_id,
            scene_case_id=episode.identity.scene_case_id,
            submission_id=submission_id,
        )


class _ReplayedReceiptSink(_Sink):
    def __init__(self) -> None:
        super().__init__()
        self.receipt: CommitReceipt | None = None

    def submit(
        self,
        episode: ExpertEpisode,
        *,
        submission_id: int,
    ) -> CommitReceipt:
        if self.receipt is None:
            self.receipt = super().submit(
                episode,
                submission_id=submission_id,
            )
        return self.receipt


class _FixedHost:
    def __init__(self) -> None:
        self.calls = 0

    def restore_initial(self) -> int:
        self.calls += 1
        return self.calls

    def verify_initial(self, binding: int) -> ValidationResult:
        assert binding == self.calls
        return ValidationResult((ValidationCheck("initial", "passed"),))


def _runner(
    coordinator: CandidateCoordinator,
    *,
    restorer: InitialStatePort | None = None,
    executor: MeasuredExecutor | None = None,
    sink: EpisodeSink | None = None,
) -> SingleSlotRunner:
    return SingleSlotRunner(
        coordinator,
        restorer or _Restorer(),
        executor or _Executor(),
        sink or _Sink(),
        episode_byte_budget=1024 * 1024,
    )


def test_generation_host_types_are_not_exported_from_expansion() -> None:
    for name in (
        "InitialStatePort",
        "MeasuredExecutor",
        "EpisodeSink",
        "FixedSceneInitialStatePort",
        "SingleSlotOutcome",
        "SingleSlotRunner",
    ):
        assert not hasattr(expansion, name)


def test_fixed_scene_port_adapts_restore_and_verify_contract() -> None:
    host = _FixedHost()
    port = FixedSceneInitialStatePort(host)
    assert port.restore(object()) == 1
    assert host.calls == 1


def test_single_slot_runner_completes_measured_receipt_lifecycle() -> None:
    session, coordinator, _ = _coordinator()
    restorer = _Restorer()

    outcome = _runner(coordinator, restorer=restorer).run_next()

    assert outcome.status == "committed"
    assert outcome.receipt is not None and outcome.receipt.confirmed
    assert len(restorer.items) == 1
    assert session.snapshot()["counts"]["committed"] == 1


def test_single_slot_runner_releases_planning_failure() -> None:
    session, coordinator, _ = _coordinator()

    outcome = _runner(
        coordinator,
        executor=_Executor(fail_validation=True),
    ).run_next()

    assert outcome.status == "planning_rejected"
    assert session.snapshot()["pending_reserved_episodes"] == 0


def test_single_slot_runner_releases_restore_failure() -> None:
    session, coordinator, _ = _coordinator()

    with pytest.raises(RuntimeError, match="restore failed"):
        _runner(coordinator, restorer=_Restorer(fail=True)).run_next()

    assert session.snapshot()["pending_reserved_episodes"] == 0


def test_single_slot_runner_releases_started_execution_failure() -> None:
    session, coordinator, _ = _coordinator()

    with pytest.raises(RuntimeError, match="execution failed"):
        _runner(
            coordinator,
            executor=_Executor(fail_execution=True),
        ).run_next()

    assert session.snapshot()["pending_reserved_episodes"] == 0


def test_single_slot_runner_releases_measured_admission_failure() -> None:
    session, coordinator, _ = _coordinator()

    with pytest.raises(ValueError, match="joint_positions"):
        _runner(
            coordinator,
            executor=_Executor(malformed_evidence=True),
        ).run_next()

    snapshot = session.snapshot()
    assert snapshot["pending_reserved_episodes"] == 0
    assert snapshot["counts"]["running"] == 0


def test_single_slot_runner_requires_executor_to_start_rollout() -> None:
    session, coordinator, _ = _coordinator()

    with pytest.raises(RuntimeError, match="before starting the rollout"):
        _runner(
            coordinator,
            executor=_Executor(start_rollout=False),
        ).run_next()

    assert session.snapshot()["pending_reserved_episodes"] == 0


def test_single_slot_runner_reports_ambiguous_write_as_pending() -> None:
    session, coordinator, items = _coordinator()
    episode_id, commit_id = session.episode_ids(items[0].spec.identity)

    outcome = _runner(coordinator, sink=_Sink(fail=True)).run_next()

    assert outcome == SingleSlotOutcome(
        "write_pending",
        candidate_id=items[0].spec.identity.candidate_id,
        episode_id=episode_id,
        commit_id=commit_id,
        submission_id=0,
        reason="ambiguous sink failure",
    )
    snapshot = session.snapshot()
    assert snapshot["pending_reserved_episodes"] == 1
    assert snapshot["counts"]["pending_write"] == 1


def test_single_slot_runner_rejects_budget_before_dequeue() -> None:
    session, coordinator, items = _coordinator()

    with pytest.raises(ValueError, match="pending_max_bytes"):
        SingleSlotRunner(
            coordinator,
            _Restorer(),
            _Executor(),
            _Sink(),
            episode_byte_budget=session.pending_max_bytes + 1,
        )

    assert coordinator.pending_count == 1
    assert session.snapshot()["audit"] == ((items[0].spec.identity, "proposed", 0),)


def test_single_slot_runner_releases_assignment_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, coordinator, items = _coordinator()
    take_ready = session.take_ready

    def fail_after_assignment(*args, **kwargs):
        assert take_ready(*args, **kwargs) is not None
        raise RuntimeError("assignment failed")

    monkeypatch.setattr(session, "take_ready", fail_after_assignment)

    with pytest.raises(RuntimeError, match="assignment failed"):
        _runner(coordinator).run_next()

    snapshot = session.snapshot()
    assert snapshot["pending_reserved_episodes"] == 0
    assert snapshot["audit"] == ((items[0].spec.identity, "released", 0),)


def test_single_slot_runner_rejects_unrelated_receipt() -> None:
    session, coordinator, items = _coordinator(
        count=2,
        target_committed_episodes=2,
    )

    outcomes = _runner(coordinator, sink=_ReplayedReceiptSink()).run_until_empty()

    assert tuple(outcome.status for outcome in outcomes) == (
        "committed",
        "write_pending",
    )
    assert outcomes[1].candidate_id == items[1].spec.identity.candidate_id
    assert outcomes[1].reason == "sink receipt does not match submitted episode"
    assert tuple(state for _, state, _ in session.snapshot()["audit"]) == (
        "committed",
        "pending_write",
    )


def test_single_slot_runner_reserves_its_requested_candidate() -> None:
    session, coordinator, items = _coordinator()
    item = items[0]
    external_identity, _ = session.propose(
        "case",
        "initial",
        source_id="external",
        source_revision="test",
        template_id="external",
        operator_id="nominal",
    )
    external = CandidateWorkItem(
        CandidateSpec(external_identity),
        _template(),
    )
    session.add_planned(
        external.to_batch(),
        ValidationResult((ValidationCheck("path_collision", "passed"),)),
    )

    outcome = _runner(coordinator).run_next()

    assert outcome.candidate_id == item.spec.identity.candidate_id
    audit = {
        identity.candidate_id: state
        for identity, state, _ in session.snapshot()["audit"]
    }
    assert audit[external_identity.candidate_id] == "ready"
    assert audit[item.spec.identity.candidate_id] == "committed"
