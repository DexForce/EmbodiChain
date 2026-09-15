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

"""Runner scheduling/accounting with tensor hosts and real local LeRobot writes.

These tests isolate orchestration; the tensor executor does not certify physics.
"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from embodichain.lab.sim.motion.expansion import (
    ExpertEpisode,
    MotionSnapshot,
    SceneCase,
    TrajectoryGenerationJobCfg,
    TrajectoryPhase,
    TrajectoryTemplate,
    ValidationCheck,
    ValidationResult,
)
from embodichain.lab.trajectory_generation.runner import (
    GenerationRunner,
    MotionLimitsProfile,
)
from embodichain.lab.trajectory_generation.sinks import LeRobotEpisodeSink


def _checks(name="task_success", status="passed"):
    return ValidationResult((ValidationCheck(name, status),))


class _Host:
    def __init__(self, rows):
        self.robot = SimpleNamespace(
            joint_names=("a", "b"),
            get_qpos_limits=lambda: torch.tensor([[-2.0, 2.0], [-2.0, 2.0]]).repeat(
                rows, 1, 1
            ),
        )
        self.adapter = SimpleNamespace(
            robot=self.robot,
            sim=SimpleNamespace(num_envs=rows),
            tolerances_profile_id="fixed_scene_tolerances",
        )
        self.profile = SimpleNamespace(profile_id="fixed_scene_initial_state")
        self.prepared = 0
        self.closed = False
        self.cases = ()
        self.fail_restore = False

    def acquire_case(self, cases):
        self.cases = cases
        self.prepared += 1
        return self.prepared

    def restore_initial(self):
        self.prepared += 1
        if self.fail_restore:
            raise RuntimeError("restoration mismatch")
        return self.prepared

    def snapshots(self, binding):
        return tuple(
            MotionSnapshot(
                case,
                self.robot.joint_names,
                torch.zeros(2),
                torch.zeros(2),
                torch.eye(4),
            )
            for case in self.cases
        )

    def verify_initial(self, binding):
        return _checks("initial_state")

    def close(self):
        self.closed = True


class _Planner:
    def __init__(self, host):
        self.robot = host.robot
        self.motion_generator = SimpleNamespace(
            collision_world_entity_ids=(), dynamic_collision_entity_ids=()
        )
        self.calls = []
        self.planning_status = "passed"
        self.actual_status = "passed"
        self.executed = False
        self.reject_first = 0

    def validate_qpos(self, batch, snapshots):
        self.calls.append(batch)
        status = self.actual_status if self.executed else self.planning_status
        if len(self.calls) <= self.reject_first:
            status = "failed"
        return (_checks("path_collision", status),)


class _Executor:
    control_dt = 0.05
    validator_id = "task_success"
    validation_profile_id = "verified_motion"
    max_episode_bytes = 16384

    def __init__(self, host, planner):
        self.host, self.planner = host, planner
        self.calls = []
        self.episodes = []
        self.fail_before = False
        self.fail_after = False
        self.status = "passed"
        self.actual_middle = None
        self.unsupported_observation = False
        self.after_first = lambda: None
        self.last_failures = {}
        self.no_transition_error = None

    def execute(self, binding, candidates, episode_ids, *, on_started, should_stop):
        self.calls.append(tuple(candidate is not None for candidate in candidates))
        if self.fail_before:
            raise RuntimeError("before command")
        episodes = []
        for candidate in candidates:
            if candidate is None or should_stop():
                episodes.append(None)
                continue
            identity = candidate.identities[0]
            on_started(identity)
            if self.no_transition_error:
                self.last_failures[identity.candidate_id] = self.no_transition_error
                episodes.append(None)
                continue
            if self.fail_after:
                raise RuntimeError("after command")
            count = int(candidate.valid_length[0])
            actual = candidate.positions[0, :count].clone()
            if self.actual_middle is not None:
                actual[1] = self.actual_middle
            observations = {"joint_positions": actual}
            if self.unsupported_observation:
                observations["unsupported"] = torch.ones(count, 2, dtype=torch.bool)
            episode = ExpertEpisode(
                identity,
                observations,
                candidate.positions[0, 1:count],
                torch.arange(count, dtype=torch.float64) * self.control_dt,
                "qpos",
                ValidationResult(
                    (
                        ValidationCheck("execution", "passed"),
                        ValidationCheck("fixed_collision_world", "passed"),
                        ValidationCheck("task_success", self.status),
                    )
                ),
                *episode_ids[identity.candidate_id],
                phases=candidate.phases[0],
            )
            episodes.append(episode)
            self.episodes.append(episode)
            self.after_first()
        self.planner.executed = True
        return tuple(episodes)


def _job(tmp_path, *, rows=1, configuration=None):
    cfg = TrajectoryGenerationJobCfg.from_mapping(configuration or {})
    host = _Host(rows)
    planner = _Planner(host)
    executor = _Executor(host, planner)
    sink = LeRobotEpisodeSink(tmp_path / "collection", fps=20, max_episode_bytes=16384)
    runner = GenerationRunner(
        cfg,
        host,
        planner,
        executor,
        sink,
        motion_limits=MotionLimitsProfile(
            torch.full((2,), 100.0), torch.full((2,), 10000.0)
        ),
    )
    cases = tuple(
        SceneCase(f"case-{row}", f"initial-{row}", "scene", "move", "robot")
        for row in range(rows)
    )
    templates = tuple(
        TrajectoryTemplate(
            "handwritten_qpos",
            "v1",
            "reference_0",
            host.robot.joint_names,
            torch.tensor([[0.0, 0.0], [0.2, 0.1], [0.4, 0.0]]),
            torch.tensor([0.0, 0.05, 0.05], dtype=torch.float64),
            (
                TrajectoryPhase(
                    "free", 0, 3, allowed_operators=("joint_residual", "retime")
                ),
            ),
            ("joint_residual", "retime"),
            validator_id="task_success",
            controlled_joint_indices=(0, 1),
        )
        for _ in range(rows)
    )
    return runner, cases, templates


def _report(runner):
    return json.loads((runner.sink.root / "generation_report.json").read_text())


def test_multiple_rounds_and_tail_commit_only_reserved_actual_evidence(tmp_path):
    runner, cases, templates = _job(
        tmp_path,
        rows=2,
        configuration={
            "augmentation": {
                "factors": {"spatial": {"enabled": True, "joint_offset_scale": 0.01}},
                "coverage": {"joint_dedup_normalized_tol": 0.000001},
            },
            "collection": {"target_committed_episodes": 3, "max_proposals": 8},
        },
    )
    report = runner.run(cases, templates)
    assert report["target_reached"]
    assert report["counts"]["committed"] == report["counts"]["rollout_attempted"] == 3
    assert runner.executor.calls == [(True, True), (False, True)]
    assert runner.host.prepared == 2
    assert runner.host.closed and not (runner.sink.root / ".writer.lock").exists()
    assert report["pending_reserved_bytes"] == 0
    manifest = json.loads((runner.sink.root / "manifest.json").read_text())
    assert len(manifest["episodes"]) == 3
    for record in manifest["episodes"]:
        evidence = json.loads(
            (runner.sink.root / record["shard"] / "episode.json").read_text()
        )
        checks = {check["check_id"] for check in evidence["validation"]}
        assert {
            "planned_path_collision",
            "path_collision",
            "actual_motion_limits",
            "motion_quality",
            "task_success",
        } <= checks
    assert _report(runner)["counts"]["committed"] == 3
    with pytest.raises(RuntimeError, match="single-use"):
        runner.run(cases, templates)


def test_planning_failures_do_not_restore_or_count_rollouts(tmp_path):
    runner, cases, templates = _job(tmp_path)
    runner.planner.reject_first = 2
    report = runner.run(cases, templates)
    assert report["counts"]["proposed"] == 3
    assert report["counts"]["rollout_attempted"] == 1
    assert report["counts"]["committed"] == 1
    assert runner.host.prepared == 1


def test_required_backend_unavailable_fails_immediately_with_audit(tmp_path):
    runner, cases, templates = _job(tmp_path)
    runner.planner.planning_status = "unavailable"
    with pytest.raises(RuntimeError, match="capability is unavailable"):
        runner.run(cases, templates)
    report = _report(runner)
    assert report["counts"]["proposed"] == 1
    assert report["counts"]["rollout_attempted"] == 0
    assert (
        report["diagnostics"][report["audit"][0][0]["candidate_id"]][
            "planning_validation"
        ]["checks"][0]["status"]
        == "unavailable"
    )
    assert runner.host.closed


@pytest.mark.parametrize("after", [False, True])
def test_failure_counts_only_commands_actually_started_and_releases_capacity(
    tmp_path, after
):
    runner, cases, templates = _job(tmp_path)
    runner.executor.fail_after, runner.executor.fail_before = after, not after
    with pytest.raises(RuntimeError, match="command"):
        runner.run(cases, templates)
    report = _report(runner)
    assert report["counts"]["rollout_attempted"] == int(after)
    assert report["counts"]["committed"] == 0
    assert report["pending_reserved_bytes"] == 0
    assert runner.host.closed


@pytest.mark.parametrize("reason", ["task", "collision", "quality", "dynamics"])
def test_actual_motion_gates_reject_before_expert_write(tmp_path, reason):
    runner, cases, templates = _job(
        tmp_path,
        configuration={"collection": {"max_proposals": 1, "max_rollout_attempts": 1}},
    )
    if reason == "task":
        runner.executor.status = "failed"
    elif reason == "collision":
        runner.planner.actual_status = "failed"
    elif reason == "quality":
        runner.executor.actual_middle = torch.tensor([-1.0, -1.0])
    else:
        # Planned peak speed is 4; the measured excursion has speed 8.
        runner.motion_limits.velocity_limits.fill_(5)
        runner.executor.actual_middle = torch.tensor([0.4, 0.1])
    report = runner.run(cases, templates)
    assert report["counts"]["rollout_attempted"] == 1
    assert report["counts"]["committed"] == 0
    assert report["stop_reason"] == "rollout_budget_exhausted"
    assert not (runner.sink.root / "manifest.json").exists()
    if reason == "collision":
        assert len(runner.planner.calls) == 2
        assert torch.equal(
            runner.planner.calls[-1].positions[0],
            runner.executor.episodes[0].observations["joint_positions"],
        )


def test_real_persistence_failure_retries_without_another_rollout(
    tmp_path, monkeypatch
):
    runner, cases, templates = _job(tmp_path)
    write = runner.sink._write_manifest
    calls = []

    def fail_once(record):
        calls.append(record["commit_id"])
        if len(calls) == 1:
            raise OSError("injected writer failure")
        write(record)

    monkeypatch.setattr(runner.sink, "_write_manifest", fail_once)
    report = runner.run(cases, templates)
    assert calls[0] == calls[1]
    assert report["counts"]["rollout_attempted"] == report["counts"]["committed"] == 1
    assert report["audit"][0][2] == 1
    assert report["pending_reserved_bytes"] == 0


def test_sink_input_error_releases_pending_payload_and_preserves_error(tmp_path):
    runner, cases, templates = _job(tmp_path)
    runner.executor.unsupported_observation = True
    with pytest.raises(ValueError, match="Unsupported observation"):
        runner.run(cases, templates)
    report = _report(runner)
    assert report["counts"]["committed"] == 0
    assert report["pending_reserved_bytes"] == 0
    assert not (runner.sink.root / "manifest.json").exists()


def test_cancellation_stops_new_commands_and_commits_completed_row(tmp_path):
    runner, cases, templates = _job(
        tmp_path, rows=2, configuration={"collection": {"target_committed_episodes": 2}}
    )
    stop = {"value": False}
    runner.executor.after_first = lambda: stop.update(value=True)
    report = runner.run(cases, templates, should_stop=lambda: stop["value"])
    assert report["cancelled"]
    assert report["counts"]["rollout_attempted"] == report["counts"]["committed"] == 1
    assert report["pending_reserved_bytes"] == 0


def test_restore_mismatch_stops_before_next_rollout(tmp_path):
    runner, cases, templates = _job(
        tmp_path, configuration={"collection": {"target_committed_episodes": 2}}
    )
    runner.host.fail_restore = True
    with pytest.raises(RuntimeError, match="restoration mismatch"):
        runner.run(cases, templates)
    report = _report(runner)
    assert report["counts"]["rollout_attempted"] == report["counts"]["committed"] == 1
    assert report["pending_reserved_bytes"] == 0


def test_template_validator_cannot_be_changed_by_job(tmp_path):
    runner, cases, templates = _job(tmp_path)
    with pytest.raises(ValueError, match="validator IDs"):
        runner.run(cases, (replace(templates[0], validator_id="weaker"),))
    assert runner.host.prepared == 0
    assert runner.host.closed


@pytest.mark.parametrize("static_only", [False, True])
def test_missing_rigid_collision_geometry_is_rejected_before_proposal(
    tmp_path, static_only
):
    runner, cases, templates = _job(tmp_path)
    if static_only:
        runner.planner.motion_generator.collision_world_entity_ids = (
            "unmodelled_cube",
        )
    snapshots = runner.host.snapshots
    runner.host.snapshots = lambda binding: tuple(
        replace(value, entity_poses={"unmodelled_cube": torch.eye(4)})
        for value in snapshots(binding)
    )
    with pytest.raises(ValueError, match="every rigid object"):
        runner.run(cases, templates)
    assert _report(runner)["counts"]["proposed"] == 0
    assert not runner.executor.calls


def test_wall_budget_expiring_during_planning_does_not_start_commands(tmp_path):
    runner, cases, templates = _job(tmp_path)
    clock = {"time": 0.0}
    runner.clock = lambda: clock["time"]
    validate = runner.planner.validate_qpos

    def slow_plan(*args):
        clock["time"] = 61.0
        return validate(*args)

    runner.planner.validate_qpos = slow_plan
    report = runner.run(cases, templates)
    assert report["stop_reason"] == "wall_time_exhausted"
    assert report["counts"]["rollout_attempted"] == 0
    assert report["pending_reserved_bytes"] == 0


def test_report_failure_does_not_replace_original_execution_error(
    tmp_path, monkeypatch
):
    runner, cases, templates = _job(tmp_path)
    runner.executor.fail_before = True

    def fail_report(report):
        raise OSError("report disk failure")

    monkeypatch.setattr(runner, "_write_report", fail_report)
    with pytest.raises(RuntimeError, match="before command") as caught:
        runner.run(cases, templates)
    assert "report disk failure" in caught.value.__notes__[0]
    assert runner.host.closed


def test_cancel_during_second_planning_round_does_not_restore_or_execute(tmp_path):
    runner, cases, templates = _job(
        tmp_path, configuration={"collection": {"target_committed_episodes": 2}}
    )
    stop = {"value": False}
    validate = runner.planner.validate_qpos

    def cancel_on_next_plan(*args):
        result = validate(*args)
        if len(runner.planner.calls) == 3:  # initial plan, actual path, second plan
            stop["value"] = True
        return result

    runner.planner.validate_qpos = cancel_on_next_plan
    report = runner.run(cases, templates, should_stop=lambda: stop["value"])
    assert report["cancelled"]
    assert runner.host.prepared == 1
    assert len(runner.executor.calls) == 1
    assert report["counts"]["committed"] == 1
    assert report["pending_reserved_bytes"] == 0


def test_clock_mismatch_fails_before_initial_preparation(tmp_path):
    runner, cases, templates = _job(tmp_path)
    template = replace(
        templates[0],
        dt=torch.tensor([0.0, 0.05000005, 0.05000005], dtype=torch.float64),
    )
    with pytest.raises(ValueError, match="timing must match"):
        runner.run(cases, (template,))
    assert runner.host.prepared == 0
    assert not runner.executor.calls


def test_incomplete_transition_keeps_actual_backend_failure_in_audit(tmp_path):
    runner, cases, templates = _job(
        tmp_path, configuration={"collection": {"max_rollout_attempts": 1}}
    )
    runner.executor.no_transition_error = "RuntimeError: physics integration failed"
    report = runner.run(cases, templates)
    assert report["counts"]["rollout_attempted"] == 1
    assert report["counts"]["committed"] == 0
    candidate_id = report["audit"][0][0].candidate_id
    assert (
        report["diagnostics"][candidate_id]["reason"]
        == runner.executor.no_transition_error
    )
