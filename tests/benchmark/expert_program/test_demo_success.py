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

"""Pure-Python tests for the no-retry demo-success benchmark."""

from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path

import pytest

from embodichain.lab.gym.envs.demo import DemoEpisodeResult, DemoSegmentResult
from scripts.benchmark.expert_program import demo_success as demo_success_module
from scripts.benchmark.expert_program.demo_success import (
    DemoSuccessCase,
    DemoSuccessRow,
    DemoSuccessTrial,
    MemorySnapshot,
    aggregate_demo_success_trials,
    collect_demo_success_trials,
    load_raw_trials,
    main,
    run_all_benchmarks,
    run_gym_demo_success_benchmark,
    write_markdown_report,
    write_raw_trials,
)


class _FakeEnv:
    """Record benchmark reset calls without creating a simulation."""

    def __init__(self, num_envs: int = 1) -> None:
        self.num_envs = num_envs
        self.reset_calls: list[dict[str, object]] = []
        self.seed: int | None = None

    def reset(self, **kwargs: object) -> None:
        self.reset_calls.append(dict(kwargs))
        if "seed" in kwargs:
            self.seed = int(kwargs["seed"])


class _PostEpisodeDiscardFailureEnv(_FakeEnv):
    """Fail the discard reset after allowing the non-committing seed reset."""

    def __init__(self) -> None:
        super().__init__()
        self.non_committing_resets = 0

    def reset(self, **kwargs: object) -> None:
        super().reset(**kwargs)
        if kwargs.get("options") == {"save_data": False}:
            self.non_committing_resets += 1
            if self.non_committing_resets == 2:
                raise RuntimeError("synthetic discard failure")


class _EpisodeExecutor:
    """Return queued demo results and record one call per seed."""

    def __init__(self, results: list[DemoEpisodeResult]) -> None:
        self.results = deque(results)
        self.calls: list[tuple[int | None, int]] = []

    def __call__(self, env: _FakeEnv, *, episode_index: int) -> DemoEpisodeResult:
        self.calls.append((env.seed, episode_index))
        return self.results.popleft()


def _result(
    successes: tuple[bool, ...],
    *,
    lengths: tuple[int, ...] | None = None,
    reasons: tuple[str, ...] | None = None,
    segments: tuple[DemoSegmentResult, ...] = (),
) -> DemoEpisodeResult:
    """Build a compact batched result with consistent vector metadata."""
    row_count = len(successes)
    row_lengths = lengths or tuple(1 for _ in successes)
    row_reasons = reasons or tuple(
        "success" if success else "task_incomplete" for success in successes
    )
    return DemoEpisodeResult(
        episode_index=0,
        length=max(row_lengths),
        completed=all(successes),
        success=successes,
        terminated=tuple(successes),
        truncated=tuple(False for _ in successes),
        terminal_reason="success" if all(successes) else "task_incomplete",
        segments=segments,
        lengths=row_lengths,
        completed_by_env=successes,
        terminal_reasons=row_reasons,
    )


def _clock(values: list[float]):
    """Return a deterministic clock backed by the supplied readings."""
    readings = iter(values)
    return lambda: next(readings)


def _memory_sampler(values: list[MemorySnapshot]):
    """Return a deterministic memory sampler backed by supplied snapshots."""
    snapshots = iter(values)

    def sample(*, reset_gpu_peak: bool = False) -> MemorySnapshot:  # noqa: ARG001
        return next(snapshots)

    return sample


def test_public_case_and_row_types_validate_and_snapshot_inputs() -> None:
    seeds = [3, 5]
    segment_failures = ["place:timeout"]
    call_failures = ["place:place:failed"]

    case = DemoSuccessCase("cube", seeds)  # type: ignore[arg-type]
    row = DemoSuccessRow(
        env_index=0,
        success=False,
        terminal_reason="timeout",
        length=4,
        segment_failure_reasons=segment_failures,  # type: ignore[arg-type]
        call_failure_keys=call_failures,  # type: ignore[arg-type]
    )
    seeds.append(7)
    segment_failures.append("mutated")
    call_failures.append("mutated")

    assert case.seeds == (3, 5)
    assert row.segment_failure_reasons == ("place:timeout",)
    assert row.call_failure_keys == ("place:place:failed",)
    with pytest.raises(TypeError, match="case_id must be a string"):
        DemoSuccessCase(7, (1,))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="evaluation seed"):
        DemoSuccessCase("cube", (True,))
    with pytest.raises(ValueError, match="env_index must be non-negative"):
        DemoSuccessRow(-1, False, "timeout", 0)
    with pytest.raises(TypeError, match="success must be a boolean"):
        DemoSuccessRow(0, 1, "timeout", 0)  # type: ignore[arg-type]


def test_public_trial_validates_rows_and_owns_nested_inputs() -> None:
    row = DemoSuccessRow(0, True, "success", 2)
    rows = [row]
    episode_result: dict[str, object] = {"success": [True]}

    trial = DemoSuccessTrial(
        case_id="cube",
        seed=3,
        cost_time_ms=1,
        cpu_delta_mb=0,
        gpu_delta_mb=0,
        peak_gpu_mb=0,
        rows=rows,  # type: ignore[arg-type]
        episode_result=episode_result,
    )
    rows.clear()
    episode_result["success"] = [False]

    assert trial.rows == (row,)
    assert trial.cost_time_ms == 1.0
    assert trial.episode_result == {"success": [True]}
    with pytest.raises(ValueError, match="unique contiguous env_index"):
        DemoSuccessTrial(
            "cube",
            3,
            1.0,
            0.0,
            0.0,
            0.0,
            (DemoSuccessRow(1, True, "success", 1),),
            {},
        )
    with pytest.raises(TypeError, match="exactly DemoSuccessRow"):
        DemoSuccessTrial(
            "cube",
            3,
            1.0,
            0.0,
            0.0,
            0.0,
            (object(),),  # type: ignore[arg-type]
            {},
        )


def test_each_seed_executes_once_without_retry_and_discards_data() -> None:
    env = _FakeEnv()
    executor = _EpisodeExecutor(
        [_result((False,)), _result((True,)), _result((False,))]
    )
    case = DemoSuccessCase(case_id="drawer", seeds=(11, 22, 33))
    memory_values = [MemorySnapshot(100.0, 10.0, 10.0)] * 6

    trials = collect_demo_success_trials(
        [case],
        lambda requested: env,
        episode_executor=executor,
        clock=_clock([0.0, 0.1, 1.0, 1.2, 2.0, 2.3]),
        memory_sampler=_memory_sampler(memory_values),
    )

    assert [call[0] for call in executor.calls] == [11, 22, 33]
    assert len(trials) == len(case.seeds)
    assert env.reset_calls == [
        {"seed": 11, "options": {"save_data": False}},
        {"options": {"save_data": False}},
        {"seed": 22, "options": {"save_data": False}},
        {"options": {"save_data": False}},
        {"seed": 33, "options": {"save_data": False}},
        {"options": {"save_data": False}},
    ]


def test_executor_error_is_counted_and_next_seed_still_executes() -> None:
    env = _FakeEnv(num_envs=2)
    calls: list[int | None] = []

    def execute(
        env: _FakeEnv, *, episode_index: int
    ) -> DemoEpisodeResult:  # noqa: ARG001
        calls.append(env.seed)
        if env.seed == 7:
            raise RuntimeError("synthetic execution failure")
        return _result((True, True))

    trials = collect_demo_success_trials(
        [DemoSuccessCase("drawer", (7, 8))],
        lambda requested: env,
        episode_executor=execute,
        clock=_clock([0.0, 0.1, 1.0, 1.1]),
        memory_sampler=_memory_sampler([MemorySnapshot(100.0, 0.0, 0.0)] * 4),
    )

    assert calls == [7, 8]
    assert [row.terminal_reason for row in trials[0].rows] == [
        "executor_error:RuntimeError",
        "executor_error:RuntimeError",
    ]
    assert [row.length for row in trials[0].rows] == [0, 0]
    assert trials[0].episode_result["executor_error"] == {
        "type": "RuntimeError",
        "message": "synthetic execution failure",
    }
    assert all(row.success for row in trials[1].rows)
    metric = aggregate_demo_success_trials(trials).success_and_metrics[0]
    assert metric["attempted"] == 4
    assert metric["successes"] == 2
    assert metric["success_rate"] == pytest.approx(0.5)
    assert env.reset_calls[-1] == {"options": {"save_data": False}}


def test_executor_error_remains_primary_when_discard_also_fails() -> None:
    env = _PostEpisodeDiscardFailureEnv()

    def execute(env: _FakeEnv, *, episode_index: int) -> DemoEpisodeResult:
        del env, episode_index
        raise ValueError("synthetic executor failure")

    with pytest.raises(ValueError, match="synthetic executor failure") as error:
        collect_demo_success_trials(
            [DemoSuccessCase("drawer", (7,))],
            lambda requested: env,
            episode_executor=execute,
            clock=_clock([0.0, 0.1]),
            memory_sampler=_memory_sampler([MemorySnapshot(100.0, 0.0, 0.0)] * 2),
        )

    assert error.value.__notes__ == [
        "Episode discard also failed: RuntimeError: synthetic discard failure"
    ]
    assert env.reset_calls == [
        {"seed": 7, "options": {"save_data": False}},
        {"options": {"save_data": False}},
    ]


def test_measurement_error_remains_primary_when_discard_also_fails() -> None:
    env = _PostEpisodeDiscardFailureEnv()
    clock_calls = 0

    def failing_clock() -> float:
        nonlocal clock_calls
        clock_calls += 1
        if clock_calls == 2:
            raise LookupError("synthetic clock failure")
        return 0.0

    with pytest.raises(LookupError, match="synthetic clock failure") as error:
        collect_demo_success_trials(
            [DemoSuccessCase("drawer", (7,))],
            lambda requested: env,
            episode_executor=_EpisodeExecutor([_result((True,))]),
            clock=failing_clock,
            memory_sampler=_memory_sampler([MemorySnapshot(100.0, 0.0, 0.0)]),
        )

    assert error.value.__notes__ == [
        "Episode discard also failed: RuntimeError: synthetic discard failure"
    ]


def test_batched_rows_aggregate_success_reasons_failures_and_lengths() -> None:
    env = _FakeEnv()
    segment = DemoSegmentResult(
        segment_id=0,
        name="place",
        start_step=0,
        end_step=5,
        success=False,
        failure_reason="segment_validation_failed",
        active=(True, True),
        start_steps=(0, 0),
        end_steps=(3, 5),
        successes=(True, False),
        failure_reasons=(None, "segment_validation_failed"),
    )
    executor = _EpisodeExecutor(
        [
            _result(
                (True, False),
                lengths=(3, 5),
                reasons=("success", "segment_validation_failed"),
                segments=(segment,),
            )
        ]
    )
    trials = collect_demo_success_trials(
        [DemoSuccessCase("batched", (5,))],
        lambda requested: env,
        episode_executor=executor,
        clock=_clock([1.0, 1.25]),
        memory_sampler=_memory_sampler(
            [
                MemorySnapshot(100.0, 20.0, 20.0),
                MemorySnapshot(104.0, 22.0, 25.0),
            ]
        ),
    )

    metric = aggregate_demo_success_trials(trials).success_and_metrics[0]

    assert metric["attempted"] == 2
    assert metric["successes"] == 1
    assert metric["success_rate"] == pytest.approx(0.5)
    assert json.loads(str(metric["terminal_reasons"])) == {
        "segment_validation_failed": 1,
        "success": 1,
    }
    assert metric["segment_failures"] == 1
    assert json.loads(str(metric["segment_failure_breakdown"])) == {
        "place:segment_validation_failed": 1
    }
    assert metric["length_mean"] == pytest.approx(4.0)


def test_runtime_call_failures_are_attributed_by_env_and_segment() -> None:
    env = _FakeEnv(num_envs=3)
    sequential = DemoSegmentResult(
        segment_id=0,
        name="prepare",
        start_step=0,
        end_step=1,
        success=False,
        metadata={
            "runtime": {
                "kind": "skill_result",
                "env_ids": [0, 1, 2],
                "calls": [
                    {
                        "semantic_id": "open",
                        "status": "failed",
                        "masks": {"failed": [True, False, False]},
                    }
                ],
            }
        },
        active=(True, True, True),
        start_steps=(0, 0, 0),
        end_steps=(1, 1, 1),
        successes=(False, True, True),
        failure_reasons=("timeout", None, None),
    )
    parallel = DemoSegmentResult(
        segment_id=1,
        name="transfer",
        start_step=1,
        end_step=2,
        success=False,
        metadata={
            "runtime": {
                "kind": "parallel_skill_result",
                "branches": {
                    "left": {
                        "kind": "skill_result",
                        "env_ids": [0, 2],
                        "calls": [
                            {
                                "semantic_id": "pick",
                                "status": "completed",
                                "masks": {"failed": [False, True]},
                            }
                        ],
                    },
                    "right": {
                        "kind": "skill_result",
                        "env_ids": [1],
                        "calls": [
                            {
                                "semantic_id": "place",
                                "status": "failed",
                                "masks": {"failed": [True]},
                            }
                        ],
                    },
                },
            }
        },
        active=(True, True, True),
        start_steps=(1, 1, 1),
        end_steps=(2, 2, 2),
        successes=(True, False, False),
        failure_reasons=(None, "collision", "batch_aborted"),
    )
    trials = collect_demo_success_trials(
        [DemoSuccessCase("runtime", (3,))],
        lambda requested: env,
        episode_executor=_EpisodeExecutor(
            [_result((False, False, False), segments=(sequential, parallel))]
        ),
        clock=_clock([0.0, 0.1]),
        memory_sampler=_memory_sampler([MemorySnapshot(100.0, 0.0, 0.0)] * 2),
    )

    metric = aggregate_demo_success_trials(trials).success_and_metrics[0]

    assert metric["call_failures"] == 3
    assert json.loads(str(metric["call_failure_breakdown"])) == {
        "prepare:open:failed": 1,
        "transfer:left:pick:completed": 1,
        "transfer:right:place:failed": 1,
    }
    assert json.loads(str(metric["segment_failure_breakdown"])) == {
        "prepare:timeout": 1,
        "transfer:batch_aborted": 1,
        "transfer:collision": 1,
    }


def _single_trial(case_id: str, successes: tuple[bool, ...]):
    """Collect one deterministic trial for ranking/report tests."""
    env = _FakeEnv()
    return collect_demo_success_trials(
        [DemoSuccessCase(case_id, (1,))],
        lambda requested: env,
        episode_executor=_EpisodeExecutor([_result(successes)]),
        clock=_clock([0.0, 0.01]),
        memory_sampler=_memory_sampler(
            [
                MemorySnapshot(100.0, 0.0, 0.0),
                MemorySnapshot(100.0, 0.0, 0.0),
            ]
        ),
    )[0]


def test_leaderboard_contains_every_case_with_deterministic_tie_break() -> None:
    trials = (
        _single_trial("zeta", (True, False)),
        _single_trial("alpha", (True, False)),
        _single_trial("winner", (True, True)),
    )

    leaderboard = aggregate_demo_success_trials(trials).leaderboard

    assert [row["case"] for row in leaderboard] == ["winner", "alpha", "zeta"]
    assert [row["rank"] for row in leaderboard] == [1, 2, 3]


def test_report_contains_exactly_three_tables(tmp_path: Path) -> None:
    trials = (_single_trial("case-a", (True,)),)
    report = write_markdown_report(
        tmp_path / "report.md", aggregate_demo_success_trials(trials)
    )

    text = report.read_text(encoding="utf-8")

    assert text.count("\n## ") == 3
    assert text.count("\n| ---") == 3
    assert "## Time & Memory" in text
    assert "## Success & Other Metrics" in text
    assert "## Leaderboard" in text


def test_raw_json_round_trip_preserves_trials(tmp_path: Path) -> None:
    trials = (_single_trial("case-a", (True, False)),)
    raw_path = write_raw_trials(tmp_path / "raw.json", trials)

    loaded = load_raw_trials(raw_path)

    assert [trial.to_dict() for trial in loaded] == [
        trial.to_dict() for trial in trials
    ]


def test_duplicate_case_seed_is_rejected_by_aggregate_write_and_load(
    tmp_path: Path,
) -> None:
    trial = _single_trial("case-a", (True,))
    duplicates = (trial, trial)

    with pytest.raises(ValueError, match="Duplicate demo success trial"):
        aggregate_demo_success_trials(duplicates)
    with pytest.raises(ValueError, match="Duplicate demo success trial"):
        write_raw_trials(tmp_path / "duplicates.json", duplicates)

    raw_path = write_raw_trials(tmp_path / "raw.json", (trial,))
    payload = json.loads(raw_path.read_text(encoding="utf-8"))
    payload["trials"].append(payload["trials"][0])
    raw_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate demo success trial"):
        load_raw_trials(raw_path)


def test_zero_case_and_zero_trial_benchmarks_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one benchmark case"):
        collect_demo_success_trials((), lambda requested: _FakeEnv())
    with pytest.raises(ValueError, match="at least one demo success trial"):
        aggregate_demo_success_trials(())
    with pytest.raises(ValueError, match="at least one demo success trial"):
        write_raw_trials(tmp_path / "empty.json", ())

    empty_raw = tmp_path / "empty-input.json"
    empty_raw.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "benchmark": "expert_program_demo_success",
                "trials": [],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="at least one demo success trial"):
        load_raw_trials(empty_raw)


def test_cli_offline_mode_aggregates_existing_raw_json(tmp_path: Path) -> None:
    raw_path = write_raw_trials(
        tmp_path / "raw.json", (_single_trial("case-a", (True,)),)
    )
    report_path = tmp_path / "offline-report.md"

    exit_code = main(["--raw-json", str(raw_path), "--report", str(report_path)])

    assert exit_code == 0
    assert report_path.is_file()
    assert len(list(tmp_path.glob("*.md"))) == 1


@pytest.mark.parametrize(
    "live_args",
    (
        ("--preview",),
        ("--action_config", "actions.json"),
        ("--headless",),
        ("--device", "cpu"),
        ("--num_envs", "1"),
        ("--renderer", "auto"),
    ),
)
def test_cli_offline_mode_rejects_explicit_live_options(
    tmp_path: Path,
    live_args: tuple[str, ...],
) -> None:
    with pytest.raises(SystemExit) as error:
        main([*live_args, "--raw-json", str(tmp_path / "raw.json")])

    assert error.value.code == 2


def test_gym_runner_reuses_one_environment_and_shared_no_retry_harness(
    tmp_path: Path,
) -> None:
    env = _FakeEnv()
    launcher_args = argparse.Namespace(gym_config="gym.json", action_config=None)
    factory_calls: list[tuple[object, Path]] = []
    closed: list[object] = []

    def environment_factory(args: object, program_path: str | Path) -> _FakeEnv:
        factory_calls.append((args, Path(program_path)))
        return env

    artifacts = run_gym_demo_success_benchmark(
        DemoSuccessCase("cube", (3, 5)),
        launcher_args=launcher_args,
        expert_program_path=tmp_path / "program.yaml",
        raw_json_path=tmp_path / "raw.json",
        report_path=tmp_path / "report.md",
        episode_executor=_EpisodeExecutor([_result((False,)), _result((True,))]),
        clock=_clock([0.0, 0.1, 1.0, 1.2]),
        memory_sampler=_memory_sampler([MemorySnapshot(100.0, 0.0, 0.0)] * 4),
        environment_factory=environment_factory,
        environment_closer=closed.append,
    )

    assert factory_calls == [(launcher_args, tmp_path / "program.yaml")]
    assert closed == [env]
    assert env.reset_calls == [
        {"seed": 3, "options": {"save_data": False}},
        {"options": {"save_data": False}},
        {"seed": 5, "options": {"save_data": False}},
        {"options": {"save_data": False}},
    ]
    assert [trial.seed for trial in artifacts.trials] == [3, 5]
    assert artifacts.raw_json_path.is_file()
    assert artifacts.report_path.is_file()


def test_gym_runner_closes_environment_when_seed_reset_fails(tmp_path: Path) -> None:
    class _ResetFailureEnv(_FakeEnv):
        def reset(self, **kwargs: object) -> None:
            super().reset(**kwargs)
            if "seed" in kwargs:
                raise RuntimeError("synthetic reset failure")

    env = _ResetFailureEnv()
    closed: list[object] = []

    with pytest.raises(RuntimeError, match="synthetic reset failure"):
        run_gym_demo_success_benchmark(
            DemoSuccessCase("cube", (3,)),
            launcher_args=argparse.Namespace(
                gym_config="gym.json",
                action_config=None,
            ),
            expert_program_path=tmp_path / "program.yaml",
            raw_json_path=tmp_path / "raw.json",
            report_path=tmp_path / "report.md",
            environment_factory=lambda args, path: env,
            environment_closer=closed.append,
        )

    assert closed == [env]


def test_gym_runner_flushes_cleanup_without_closing_when_factory_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory_error = LookupError("synthetic factory failure")
    cleanup_calls: list[str] = []
    close_calls: list[object] = []

    def fail_factory(
        launcher_args: argparse.Namespace,
        expert_program_path: str | Path,
    ) -> _FakeEnv:
        raise factory_error

    monkeypatch.setattr(
        "embodichain.lab.sim.sim_manager.SimulationManager.flush_cleanup_queue",
        lambda: cleanup_calls.append("flush_cleanup_queue"),
    )

    with pytest.raises(LookupError, match="synthetic factory failure") as error:
        run_gym_demo_success_benchmark(
            DemoSuccessCase("cube", (3,)),
            launcher_args=argparse.Namespace(
                gym_config="gym.json",
                action_config=None,
            ),
            expert_program_path=tmp_path / "program.yaml",
            raw_json_path=tmp_path / "raw.json",
            report_path=tmp_path / "report.md",
            environment_factory=fail_factory,
            environment_closer=close_calls.append,
        )

    assert error.value is factory_error
    assert cleanup_calls == ["flush_cleanup_queue"]
    assert close_calls == []


def test_gym_runner_preserves_factory_error_when_cleanup_flush_also_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory_error = LookupError("synthetic factory failure")
    cleanup_calls: list[str] = []
    close_calls: list[object] = []

    def fail_factory(
        launcher_args: argparse.Namespace,
        expert_program_path: str | Path,
    ) -> _FakeEnv:
        raise factory_error

    def fail_cleanup() -> None:
        cleanup_calls.append("flush_cleanup_queue")
        raise RuntimeError("synthetic cleanup failure")

    monkeypatch.setattr(
        "embodichain.lab.sim.sim_manager.SimulationManager.flush_cleanup_queue",
        fail_cleanup,
    )

    with pytest.raises(LookupError, match="synthetic factory failure") as error:
        run_gym_demo_success_benchmark(
            DemoSuccessCase("cube", (3,)),
            launcher_args=argparse.Namespace(
                gym_config="gym.json",
                action_config=None,
            ),
            expert_program_path=tmp_path / "program.yaml",
            raw_json_path=tmp_path / "raw.json",
            report_path=tmp_path / "report.md",
            environment_factory=fail_factory,
            environment_closer=close_calls.append,
        )

    assert error.value is factory_error
    assert error.value.__notes__ == [
        "Benchmark environment construction cleanup also failed: "
        "RuntimeError: synthetic cleanup failure"
    ]
    assert cleanup_calls == ["flush_cleanup_queue"]
    assert close_calls == []


def test_default_gym_environment_closer_uses_unwrapped_target_and_flushes_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []

    class _UnwrappedEnv:
        def close(self, *, exit_process: bool) -> None:
            calls.append(("close", exit_process))

    env = argparse.Namespace(unwrapped=_UnwrappedEnv())
    monkeypatch.setattr(
        "embodichain.lab.sim.sim_manager.SimulationManager.flush_cleanup_queue",
        lambda: calls.append("flush_cleanup_queue"),
    )

    demo_success_module._close_gym_demo_success_environment(env)

    assert calls == [("close", False), "flush_cleanup_queue"]


def test_gym_runner_preserves_body_error_when_default_close_also_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    close_calls: list[bool] = []
    cleanup_calls: list[str] = []

    class _CloseFailureTarget:
        def close(self, *, exit_process: bool) -> None:
            close_calls.append(exit_process)
            raise RuntimeError("synthetic close failure")

    class _BodyFailureEnv(_FakeEnv):
        def __init__(self) -> None:
            super().__init__()
            self.unwrapped = _CloseFailureTarget()

        def reset(self, **kwargs: object) -> None:
            super().reset(**kwargs)
            if "seed" in kwargs:
                raise LookupError("synthetic benchmark body failure")

    env = _BodyFailureEnv()
    monkeypatch.setattr(
        "embodichain.lab.sim.sim_manager.SimulationManager.flush_cleanup_queue",
        lambda: cleanup_calls.append("flush_cleanup_queue"),
    )

    with pytest.raises(LookupError, match="synthetic benchmark body failure") as error:
        run_gym_demo_success_benchmark(
            DemoSuccessCase("cube", (3,)),
            launcher_args=argparse.Namespace(
                gym_config="gym.json",
                action_config=None,
            ),
            expert_program_path=tmp_path / "program.yaml",
            raw_json_path=tmp_path / "raw.json",
            report_path=tmp_path / "report.md",
            environment_factory=lambda args, path: env,
        )

    assert error.value.__notes__ == [
        "Benchmark environment cleanup also failed: "
        "RuntimeError: synthetic close failure"
    ]
    assert close_calls == [False]
    assert cleanup_calls == ["flush_cleanup_queue"]


def test_gym_environment_builder_uses_standard_public_config_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    launcher_args = argparse.Namespace(
        gym_config="gym.json",
        action_config=None,
    )
    env_cfg = argparse.Namespace(expert_program=None)
    program = object()
    env = object()

    monkeypatch.setattr(
        demo_success_module,
        "discover_task_packages",
        lambda: calls.append("discover"),
    )
    monkeypatch.setattr(
        demo_success_module,
        "execute_init_hooks",
        lambda: calls.append("hooks"),
    )

    def build(args: argparse.Namespace):
        calls.append(("build", args))
        return env_cfg, {"id": "ExpertTask-v1"}, {}

    monkeypatch.setattr(demo_success_module, "build_env_cfg_from_args", build)
    monkeypatch.setattr(
        demo_success_module,
        "load_expert_program",
        lambda path: calls.append(("load", path)) or program,
    )
    monkeypatch.setattr(
        demo_success_module.gymnasium,
        "make",
        lambda **kwargs: calls.append(("make", kwargs)) or env,
    )

    created = demo_success_module._create_gym_demo_success_environment(
        launcher_args,
        "program.yaml",
    )

    assert created is env
    assert env_cfg.expert_program is program
    assert calls == [
        "discover",
        "hooks",
        ("build", launcher_args),
        ("load", "program.yaml"),
        ("make", {"id": "ExpertTask-v1", "cfg": env_cfg}),
    ]


@pytest.mark.parametrize(
    "unsupported",
    (
        ("--preview",),
        ("--action_config", "actions.json"),
    ),
)
def test_cli_live_mode_rejects_unsupported_launcher_options(
    tmp_path: Path,
    unsupported: tuple[str, ...],
) -> None:
    with pytest.raises(SystemExit) as error:
        main(
            [
                "--run-simulation",
                "--gym_config",
                "gym.json",
                "--expert-program",
                "program.yaml",
                "--case-id",
                "cube",
                "--seeds",
                "7",
                "--raw-json",
                str(tmp_path / "raw.json"),
                *unsupported,
            ]
        )

    assert error.value.code == 2


def test_cli_live_mode_dispatches_fixed_seed_case(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def run(case: DemoSuccessCase, **kwargs: object) -> object:
        captured["case"] = case
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        demo_success_module,
        "run_gym_demo_success_benchmark",
        run,
    )
    raw_path = tmp_path / "raw.json"

    exit_code = main(
        [
            "--run-simulation",
            "--gym_config",
            "gym.json",
            "--expert-program",
            "program.yaml",
            "--case-id",
            "cube",
            "--seeds",
            "7",
            "11",
            "--raw-json",
            str(raw_path),
        ]
    )

    assert exit_code == 0
    assert captured["case"] == DemoSuccessCase("cube", (7, 11))
    assert captured["expert_program_path"] == Path("program.yaml")
    assert captured["raw_json_path"] == raw_path
    assert captured["report_path"] == raw_path.with_suffix(".md")
    launcher_args = captured["launcher_args"]
    assert isinstance(launcher_args, argparse.Namespace)
    assert launcher_args.gym_config == "gym.json"
    assert launcher_args.num_envs is None
    assert launcher_args.renderer is None


def test_run_all_benchmarks_prints_report_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    env = _FakeEnv()
    report_path = tmp_path / "report.md"

    artifacts = run_all_benchmarks(
        [DemoSuccessCase("case-a", (1,))],
        lambda requested: env,
        raw_json_path=tmp_path / "raw.json",
        report_path=report_path,
        episode_executor=_EpisodeExecutor([_result((True,))]),
        clock=_clock([0.0, 0.1]),
        memory_sampler=_memory_sampler([MemorySnapshot(100.0, 0.0, 0.0)] * 2),
    )

    assert artifacts.report_path == report_path
    assert f"Markdown report saved: {report_path}" in capsys.readouterr().out
