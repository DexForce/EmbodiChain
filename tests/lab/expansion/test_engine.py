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

"""Behavioral coverage for real cohort calls and canonical session accounting."""

from __future__ import annotations
from dataclasses import replace
import pytest
import torch
from embodichain.lab.expansion import ExpansionEngine, ExpansionJobCfg
from embodichain.lab.sim.motion.expansion import (
    ExpansionSession,
    TrajectoryExpansionJobCfg,
    SceneCase,
    CandidateSpec,
    CandidateWorkItem,
    TrajectoryTemplate,
    ValidationCheck,
    ValidationResult,
    ExpertEpisode,
    CommitReceipt,
)


def setup(count=3, batch_size=2, sink=None, host=None):
    session = ExpansionSession(
        TrajectoryExpansionJobCfg.from_mapping(
            {
                "collection": {
                    "target_committed_episodes": 10,
                    "max_rollout_attempts": 20,
                },
                "persistence": {"pending_episode_limit": 8},
            }
        )
    )
    items = []
    for i in range(count):
        # Independent initial states prevent trajectory coverage deduplication.
        case = SceneCase(f"case{i}", "initial", "scene", "task", "robot")
        session.register_case(case, torch.tensor([[-2.0, 2.0]]), joint_names=("joint",))
        identity, _ = session.propose(
            case.scene_case_id,
            "initial",
            source_id="source",
            source_revision="v1",
            template_id="reference",
            operator_id="nominal",
        )
        template = TrajectoryTemplate(
            "source",
            "v1",
            "reference",
            ("joint",),
            torch.tensor([[0.0], [0.1], [0.2]]),
            torch.tensor([0.0, 0.1, 0.1]),
        )
        items.append(CandidateWorkItem(CandidateSpec(identity), template))
    host = host or Host()
    sink = sink or Sink()
    engine = ExpansionEngine(
        ExpansionJobCfg(batch_size=batch_size, episode_byte_budget=10000),
        session=session,
        host=host,
        sink=sink,
    )
    engine.enqueue(items)
    return engine, host, sink, items


class Host:
    capacity = 32
    reject = None
    bad_identity = False
    fail = False

    def __init__(self):
        self.calls = []

    def preflight(self, items):
        pass

    def validate_plan(self, item):
        return ValidationResult((ValidationCheck("path_collision", "passed"),))

    def execute_batch(self, assignments, *, on_started):
        self.calls.append(assignments)
        result = []
        for a in assignments:
            on_started(a.item.spec.identity)
            if self.fail:
                raise RuntimeError("native failure")
            if a.item.spec.identity.candidate_id == self.reject:
                result.append(None)
                continue
            t = a.item.template
            result.append(
                ExpertEpisode(
                    a.item.spec.identity,
                    {"joint_positions": t.positions},
                    t.positions[:-1],
                    t.dt.cumsum(0),
                    "qpos",
                    ValidationResult(
                        (
                            ValidationCheck("path_collision", "passed"),
                            ValidationCheck("task_success", "passed"),
                        )
                    ),
                    "wrong" if self.bad_identity else a.episode_id,
                    a.commit_id,
                )
            )
        return result


class Sink:
    def __init__(self):
        self.calls = []
        self.fail = False

    def submit(self, episode, *, submission_id):
        self.calls.append(episode)
        if self.fail:
            raise RuntimeError("ambiguous write")
        return self.receipt(episode, submission_id)

    def receipt(self, episode, submission_id=0):
        return CommitReceipt(
            episode_id=episode.episode_id,
            candidate_id=episode.identity.candidate_id,
            attempt_id=episode.identity.attempt_id,
            commit_id=episode.commit_id,
            scene_case_id=episode.identity.scene_case_id,
            submission_id=submission_id,
            storage_id="test://episode",
        )


def test_true_cohort_call_and_tail_have_independent_receipts():
    engine, host, sink, items = setup()
    before = engine.session.snapshot()["counts"].copy()
    assert list(map(len, engine.preflight())) == [2, 1]
    assert engine.session.snapshot()["counts"] == before
    result = engine.run()
    assert [len(a) for a in host.calls] == [2, 1]
    assert [[a.env_id for a in batch] for batch in host.calls] == [[0, 1], [0]]
    assert result["counts"]["committed"] == 3
    assert result["counts"]["rollout_attempted"] == 3
    assert len(sink.calls) == 3


def test_row_rejection_preserves_peer_success():
    engine, host, sink, items = setup()
    host.reject = items[0].spec.identity.candidate_id
    counts = engine.run()["counts"]
    assert counts["committed"] == 2
    assert counts["rollout_attempted"] == 3


def test_ambiguous_write_preserves_whole_cohort_until_reconciliation():
    engine, host, sink, items = setup(count=2)
    sink.fail = True
    with pytest.raises(RuntimeError, match="ambiguous"):
        engine.run()
    assert len(engine.pending_commit_ids) == 2
    assert engine.session.snapshot()["counts"]["committed"] == 0
    with pytest.raises(RuntimeError, match="reconcile"):
        engine.run()
    engine.resolve_receipt(sink.receipt(sink.calls[0]))
    sink.fail = False
    assert engine.run()["counts"]["committed"] == 2
    assert len(host.calls) == 1


@pytest.mark.parametrize("fault", ["fail", "bad_identity"])
def test_host_fault_releases_reservations_and_stops_world(fault):
    engine, host, sink, _ = setup(count=2)
    setattr(host, fault, True)
    with pytest.raises((RuntimeError, ValueError)):
        engine.run()
    assert not sink.calls
    counts = engine.session.snapshot()["counts"]
    assert counts["assigned"] == counts["running"] == 0
    with pytest.raises(RuntimeError, match="host failed"):
        engine.run()


def test_duplicate_admission_is_atomic():
    engine, host, sink, items = setup()
    before = engine.preflight()
    with pytest.raises(ValueError, match="once"):
        engine.enqueue(items)
    assert engine.preflight() == before


def test_insufficient_host_capacity_rejects_without_execution():
    host = Host()
    host.capacity = 1
    with pytest.raises(ValueError, match="capacity"):
        setup(host=host)
    assert not host.calls


@pytest.mark.parametrize("batch_size", [1, 8, 32])
def test_recipe_identity_is_independent_of_cohort_capacity(batch_size):
    engine, host, sink, items = setup(count=3, batch_size=batch_size)
    assert [identity for cohort in engine.preflight() for identity in cohort] == [
        item.spec.identity.candidate_id for item in items
    ]
    assert engine.run()["counts"]["committed"] == 3


def test_compatibility_groups_and_plan_rejections_do_not_drop_later_groups():
    engine, host, sink, _ = setup(count=0)
    other, _, _, items = setup(count=3)
    # Retain canonical registration and use new compatibility descriptions.
    engine.session = other.session
    items = [
        replace(item, spec=replace(item.spec, compatibility_key=str(i % 2)))
        for i, item in enumerate(items)
    ]
    engine.enqueue(items)
    assert engine.preflight() == (
        (items[0].spec.identity.candidate_id, items[2].spec.identity.candidate_id),
        (items[1].spec.identity.candidate_id,),
    )
    host.validate_plan = lambda item: ValidationResult(
        (
            ValidationCheck(
                "path_collision",
                "failed" if item.spec.compatibility_key == "0" else "passed",
            ),
        )
    )
    assert engine.run()["counts"]["committed"] == 1
    assert len(host.calls) == 1


def test_invalid_later_evidence_retains_earlier_staged_payload():
    class InvalidHost(Host):
        def execute_batch(self, assignments, *, on_started):
            episodes = list(super().execute_batch(assignments, on_started=on_started))
            episodes[1] = replace(
                episodes[1], observations={"joint_positions": torch.zeros(3, 2)}
            )
            return episodes

    engine, host, sink, _ = setup(count=3, batch_size=3, host=InvalidHost())
    with pytest.raises(ValueError, match="joint_positions"):
        engine.run()
    assert not sink.calls
    assert len(engine.pending_commit_ids) == 1
    counts = engine.session.snapshot()["counts"]
    assert counts["assigned"] == counts["running"] == 0
    engine.flush()
    assert engine.session.snapshot()["counts"]["committed"] == 1


def test_cohort_larger_than_ready_watermark_makes_progress():
    engine, host, sink, _ = setup(count=32, batch_size=32)
    result = engine.run()
    assert result["counts"]["committed"] == 10
    assert sum(len(batch) for batch in host.calls) == 10


def test_admission_exception_releases_prior_reservations():
    engine, host, _, items = setup(count=3, batch_size=3)
    validate = host.validate_plan

    def checked(item):
        if item is items[1]:
            raise ValueError("invalid plan")
        return validate(item)

    host.validate_plan = checked
    with pytest.raises(ValueError, match="invalid plan"):
        engine.run()
    assert not host.calls
    assert all(
        state != "assigned" for _, state, _ in engine.session.snapshot()["audit"]
    )
