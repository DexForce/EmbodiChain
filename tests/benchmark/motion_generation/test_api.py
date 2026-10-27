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

"""Public evaluation API, run-local adapters and caller-owned configuration."""

from __future__ import annotations

import json
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
import yaml
from embodichain_benchmarks.motion_generation import (
    BenchmarkCase,
    PlannerAdapter,
    PlannerSpecCfg,
    SuiteCfg,
    load_suite,
    run_benchmark,
)
from embodichain_benchmarks.motion_generation import runner as runner_module
from embodichain_benchmarks.motion_generation.config import TrackCfg, suite_to_dict
from embodichain_benchmarks.motion_generation.metrics.trajectory import (
    make_failure_outcomes,
)
from embodichain_benchmarks.motion_generation.registry import planner_adapter_names
from embodichain_benchmarks.motion_generation.scenarios.base import (
    ScenarioEvaluation,
    ScenarioProvider,
)

from embodichain.lab.sim.motion.planners.utils import PlanResult

pytestmark = pytest.mark.no_sim


def test_suite_rejects_retired_top_level_free_space_configuration() -> None:
    with pytest.raises(ValueError, match=r"tracks\[\]\.config"):
        SuiteCfg.from_dict({"free_space": {"batch_sizes": [8]}})


@pytest.mark.parametrize("batch_sizes", [[8], [1, 8]])
def test_direct_suite_preserves_explicit_track_population(
    monkeypatch: pytest.MonkeyPatch, batch_sizes: list[int]
) -> None:
    requested = {
        "batch_sizes": batch_sizes,
        "seeds": [37],
        "path_shapes": ["direct"],
        "waypoint_counts": [1],
    }
    suite = SuiteCfg(
        planners=[
            PlannerSpecCfg(id="baseline", adapter="ik_interpolate", enabled=True)
        ],
        tracks=[TrackCfg(id="custom", scenario="free_space", config=requested)],
    )
    monkeypatch.setattr(
        runner_module.BenchmarkRunner, "run", lambda runner: runner.suite
    )

    resolved = run_benchmark(suite, device="cpu")

    for field, value in requested.items():
        assert resolved.tracks[0].config[field] == value
    # Later per-track edits also survive API validation.
    suite.tracks[0].config["seeds"] = [43]
    assert run_benchmark(suite, device="cpu").tracks[0].config["seeds"] == [43]
    assert suite.tracks[0].config["seeds"] == [43]
    assert requested["seeds"] == [37]


@pytest.fixture
def fake_runtime(monkeypatch: pytest.MonkeyPatch) -> tuple[Mock, Mock, Mock]:
    """Replace the simulator boundary while executing the real benchmark runner."""
    robot = Mock(uid="test_robot")
    provider = Mock(control_part="arm")
    provider.add_robot.return_value = robot
    simulation = Mock()
    factory = Mock(return_value=simulation)
    monkeypatch.setattr(runner_module, "SimulationManager", factory)
    monkeypatch.setattr(runner_module, "create_robot_provider", lambda spec: provider)

    class Scenario(ScenarioProvider):
        def batch_sizes(self, suite, track):
            return [1]

        def generate_cases(self, suite, track, robot, control_part, batch_size):
            return [
                BenchmarkCase(
                    suite_version=suite.suite_version,
                    track=track.id,
                    scenario_id="free_space",
                    case_id="test_case",
                    seed=11,
                    batch_size=1,
                    num_waypoints=1,
                    path_shape="direct",
                    start_state_bin="nominal",
                    start_qpos=torch.zeros(1, 2),
                    target_waypoints=torch.eye(4).reshape(1, 1, 4, 4),
                    reference_qpos=torch.zeros(1, 1, 2),
                    robot_id=suite.robot.id,
                )
            ]

        def evaluate_case(self, result, case, robot, control_part, suite, **kwargs):
            # This fixture validates orchestration and serialization, not physics.
            return ScenarioEvaluation(outcomes=make_failure_outcomes(1, "fixture"))

    scenario = Scenario()
    monkeypatch.setattr(
        runner_module, "create_scenario_provider", lambda name: scenario
    )
    return factory, provider, scenario


class ExternalAdapter(PlannerAdapter):
    """External policy with observable construction and cleanup."""

    capabilities = frozenset({"eef_waypoint", "batched", "empty_world"})
    events: list[str] = []

    def build(self) -> None:
        self.events.append("build")

    def plan(self, case: BenchmarkCase) -> PlanResult:
        self.events.append("plan")
        return PlanResult(
            success=torch.ones(case.batch_size, dtype=torch.bool),
            positions=case.start_qpos[:, None].repeat(1, 2, 1),
            dt=torch.tensor([[0.0, 0.1]]).repeat(case.batch_size, 1),
        )

    def close(self) -> None:
        self.events.append("close")


@pytest.mark.parametrize("suite_input", ["object", "yaml_path"])
def test_external_adapter_runs_without_global_registration_or_input_mutation(
    fake_runtime, tmp_path: Path, suite_input: str
) -> None:
    suite = load_suite()
    suite.protocol.warmup_trials = 0
    suite.protocol.measured_trials = 1
    original = suite_to_dict(suite)
    spec = PlannerSpecCfg(id="external", adapter="external", role="candidate")
    registered = planner_adapter_names()
    ExternalAdapter.events = []
    supplied_suite = suite
    if suite_input == "yaml_path":
        supplied_suite = tmp_path / "suite.yaml"
        supplied_suite.write_text(yaml.safe_dump(original))

    result = run_benchmark(
        supplied_suite,
        planners=[spec],
        adapters={"external": ExternalAdapter},
        device="cpu",
        output_root=tmp_path,
    )

    assert ExternalAdapter.events == ["build", "plan", "plan", "close"]
    assert suite_to_dict(suite) == original
    assert not spec.enabled
    assert planner_adapter_names() == registered
    assert result.report_path.is_file()
    assert (
        json.loads((result.run_dir / "aggregates.json").read_text())[
            "report_schema_version"
        ]
        == "3"
    )
    records = [json.loads(line) for line in result.trials_path.read_text().splitlines()]
    measured = [row for row in records if row["phase"] == "measured"]
    assert [row["algorithm_id"] for row in measured] == ["external"]
    assert (
        result.aggregates["motion_planning_leaderboard"][0]["algorithm"] == "external"
    )
    resolved = load_suite(result.run_dir / "resolved_suite.yaml")
    assert [(row.id, row.enabled) for row in resolved.planners] == [("external", True)]
    fake_runtime[0].return_value.destroy.assert_called_once_with(exit_process=False)
    fake_runtime[0].flush_cleanup_queue.assert_called_once()


def test_adapter_mapping_is_local_to_each_run(fake_runtime, tmp_path: Path) -> None:
    class First(ExternalAdapter):
        events = []

    class Second(ExternalAdapter):
        events = []

    suite = load_suite()
    suite.protocol.warmup_trials = 0
    suite.protocol.measured_trials = 1
    spec = PlannerSpecCfg(id="external", adapter="same_name")
    for adapter in (First, Second):
        run_benchmark(
            suite,
            planners=[spec],
            adapters={"same_name": adapter},
            device="cpu",
            output_root=tmp_path,
        )
    assert First.events == Second.events == ["build", "plan", "plan", "close"]
    assert "same_name" not in planner_adapter_names()


def test_default_selection_uses_enabled_suite_planners(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    observed: list[runner_module.BenchmarkRunner] = []
    result_value = object()

    def record_run(runner: runner_module.BenchmarkRunner) -> object:
        observed.append(runner)
        return result_value

    monkeypatch.setattr(runner_module.BenchmarkRunner, "run", record_run)
    suite = load_suite()
    original = suite_to_dict(suite)
    result = run_benchmark(suite, output_root=tmp_path)
    assert result is result_value
    runner = observed[0]
    selected = runner.planner_specs
    assert [spec.id for spec in selected] == [
        spec.id for spec in suite.planners if spec.enabled
    ]
    assert suite_to_dict(runner.suite) == original
    runner.suite.protocol.measured_trials += 1
    selected[0].config["caller_isolation_probe"] = True
    assert suite_to_dict(suite) == original


def test_suite_with_no_enabled_planners_fails_before_simulation(fake_runtime) -> None:
    suite = load_suite()
    for spec in suite.planners:
        spec.enabled = False
    with pytest.raises(ValueError, match="No algorithms"):
        run_benchmark(suite, device="cpu")
    fake_runtime[0].assert_not_called()


def test_framework_revision_is_independent_of_caller_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from embodichain_benchmarks.motion_generation.artifacts import _git_commit

    framework_revision = _git_commit()
    assert framework_revision is not None
    monkeypatch.chdir(tmp_path)
    assert _git_commit() == framework_revision


@pytest.mark.parametrize(
    ("planners", "adapters", "error"),
    [
        ([], None, ValueError),
        (["wrong_type"], None, TypeError),
        ([PlannerSpecCfg(id="missing", adapter="unknown")], None, ValueError),
        ([PlannerSpecCfg(id="bad", adapter="bad")], {"bad": object}, TypeError),
        ([PlannerSpecCfg(id="bad", adapter="bad")], {"": ExternalAdapter}, ValueError),
        (
            [PlannerSpecCfg(id="repeat", adapter="toppra")] * 2,
            None,
            ValueError,
        ),
    ],
)
def test_invalid_inputs_fail_before_simulation(fake_runtime, planners, adapters, error):
    with pytest.raises(error):
        run_benchmark(planners=planners, adapters=adapters, device="cpu")
    fake_runtime[0].assert_not_called()


@pytest.mark.parametrize("hook", ["close_planner", "close_runtime"])
def test_failing_teardown_still_releases_simulation_and_adapter(
    fake_runtime, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, hook: str
) -> None:
    factory, _, scenario = fake_runtime
    monkeypatch.setattr(scenario, hook, Mock(side_effect=RuntimeError("teardown")))
    ExternalAdapter.events = []
    suite = load_suite()
    suite.protocol.warmup_trials = 0
    suite.protocol.measured_trials = 1
    with pytest.raises(RuntimeError, match="teardown"):
        run_benchmark(
            suite,
            planners=[PlannerSpecCfg(id="external", adapter="external")],
            adapters={"external": ExternalAdapter},
            device="cpu",
            output_root=tmp_path,
        )
    assert ExternalAdapter.events[-1] == "close"
    factory.return_value.destroy.assert_called_once_with(exit_process=False)
    factory.flush_cleanup_queue.assert_called_once()


def test_robot_creation_failure_cleans_up_simulation(
    fake_runtime, tmp_path: Path
) -> None:
    factory, provider, _ = fake_runtime
    provider.add_robot.side_effect = RuntimeError("robot construction")
    with pytest.raises(RuntimeError, match="robot construction"):
        run_benchmark(device="cpu", output_root=tmp_path)
    factory.return_value.destroy.assert_called_once_with(exit_process=False)
    factory.flush_cleanup_queue.assert_called_once()


def test_partial_scene_creation_failure_releases_scenario_and_simulation(
    fake_runtime, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    factory, _, scenario = fake_runtime
    close = Mock()
    monkeypatch.setattr(scenario, "close_runtime", close)
    monkeypatch.setattr(
        scenario,
        "create_runtime_entities",
        Mock(side_effect=RuntimeError("scene setup")),
    )
    with pytest.raises(RuntimeError, match="scene setup"):
        run_benchmark(device="cpu", output_root=tmp_path)
    close.assert_called_once()
    factory.return_value.destroy.assert_called_once_with(exit_process=False)
    factory.flush_cleanup_queue.assert_called_once()


def test_in_place_adapter_changes_cannot_modify_canonical_cases(fake_runtime, tmp_path):
    class MutatingAdapter(ExternalAdapter):
        separate_prepare = True

        def supports_case(self, case):
            case.start_qpos.add_(17)
            case.case_parameters["mutated"] = True
            return True, None

        def prepare_cases(self, cases):
            for case in cases:
                assert not case.start_qpos.any()
                case.target_waypoints.zero_()
                case.case_parameters["mutated"] = True
            return None

        def plan(self, case):
            assert not case.start_qpos.any()
            assert "mutated" not in case.case_parameters
            result = super().plan(case)
            case.start_qpos.add_(42)
            case.target_waypoints.zero_()
            case.case_parameters["mutated"] = True
            return result

    class ObservingAdapter(ExternalAdapter):
        def plan(self, case):
            assert not case.start_qpos.any()
            assert "mutated" not in case.case_parameters
            torch.testing.assert_close(case.target_waypoints[0, 0], torch.eye(4))
            return super().plan(case)

    suite = load_suite()
    suite.protocol.warmup_trials = 0
    suite.protocol.measured_trials = 2
    result = run_benchmark(
        suite,
        planners=[
            PlannerSpecCfg(id="mutator", adapter="mutator"),
            PlannerSpecCfg(id="observer", adapter="observer"),
        ],
        adapters={"mutator": MutatingAdapter, "observer": ObservingAdapter},
        device="cpu",
        output_root=tmp_path,
    )
    assert all(record.status == "ok" for record in result.records)
    manifest = json.loads((result.run_dir / "case_manifest.json").read_text())
    case = manifest["cases"][0]
    assert case["start_qpos"] == [[0.0, 0.0]]
    assert "mutated" not in case["case_parameters"]
    assert case["target_waypoints"][0][0] == torch.eye(4).tolist()


def test_frozen_manifest_replays_without_generating_new_inputs(
    fake_runtime, tmp_path, monkeypatch
):
    suite = load_suite()
    suite.protocol.warmup_trials = 0
    suite.protocol.measured_trials = 1
    spec = PlannerSpecCfg(id="external", adapter="external")
    kwargs = dict(
        planners=[spec],
        adapters={"external": ExternalAdapter},
        device="cpu",
        output_root=tmp_path,
    )
    first = run_benchmark(suite, **kwargs)
    path = first.run_dir / "case_manifest.json"
    monkeypatch.setattr(
        fake_runtime[2],
        "generate_cases",
        lambda *a, **k: pytest.fail("regenerated frozen inputs"),
    )
    second = run_benchmark(suite, case_manifest=path, **kwargs)
    before = json.loads(path.read_text())
    after = json.loads((second.run_dir / "case_manifest.json").read_text())
    assert before["cases_sha256"] == after["cases_sha256"]
    assert before["cases"] == after["cases"]
    before["cases"][0]["start_qpos"][0][0] = 1.0
    path.write_text(json.dumps(before))
    with pytest.raises(ValueError, match="hash mismatch"):
        run_benchmark(suite, case_manifest=path, **kwargs)


@pytest.mark.parametrize("change", ["robot_config", "binding_hash", "missing_binding"])
def test_frozen_manifest_rejects_changed_embodiment_before_simulation(
    fake_runtime, tmp_path: Path, change: str
) -> None:
    suite = load_suite()
    suite.protocol.warmup_trials = 0
    suite.protocol.measured_trials = 1
    kwargs = dict(
        planners=[PlannerSpecCfg(id="external", adapter="external")],
        adapters={"external": ExternalAdapter},
        device="cpu",
        output_root=tmp_path,
    )
    first = run_benchmark(suite, **kwargs)
    path = first.run_dir / "case_manifest.json"
    if change == "robot_config":
        # The same robot ID and DOF can represent a changed TCP or control-part order.
        suite.robot.config["tcp_offset"] = [0, 0, 0.1]
    else:
        payload = json.loads(path.read_text())
        if change == "binding_hash":
            payload["resolved_embodiment"]["config_hash"] = "different-geometry"
        else:
            payload.pop("resolved_embodiment")
        path.write_text(json.dumps(payload))
    fake_runtime[0].reset_mock()
    with pytest.raises(ValueError, match="embodiment"):
        run_benchmark(suite, case_manifest=path, **kwargs)
    fake_runtime[0].assert_not_called()


def test_prepare_case_manifest_is_planner_independent_and_never_overwrites(
    fake_runtime, tmp_path, monkeypatch
):
    from embodichain_benchmarks.motion_generation import prepare_case_manifest

    suite = load_suite()
    monkeypatch.setattr(
        runner_module,
        "create_planner_adapter",
        lambda *args, **kwargs: pytest.fail("case preparation constructed a planner"),
    )
    output = tmp_path / "frozen.json"
    assert prepare_case_manifest(suite, output=output, device="cpu") == output
    payload = json.loads(output.read_text())
    assert len(payload["cases"]) == 1
    assert payload["resolved_embodiment"]["embodiment_id"]
    fake_runtime[0].return_value.destroy.assert_called_once_with(exit_process=False)
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        prepare_case_manifest(suite, output=output, device="cpu")
    assert output.read_bytes() == original


class _InlineExecutor:
    """Exercise orchestration independently of process startup and hardware."""

    def __init__(self, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def submit(self, function, job):
        future = Future()
        try:
            future.set_result(function(job))
        except Exception as error:  # noqa: BLE001 - emulate a Future
            future.set_exception(error)
        return future


def _spawned_fake_shard(job):
    """Replace native simulation inside a real spawned worker process."""
    from dataclasses import replace
    from unittest.mock import patch

    from embodichain_benchmarks.motion_generation.parallel import _run_shard

    robot = Mock(uid="test_robot")
    provider = Mock(control_part="arm")
    provider.add_robot.return_value = robot
    simulation = Mock()
    factory = Mock(return_value=simulation)

    class Scenario(ScenarioProvider):
        def batch_sizes(self, suite, track):
            return [1]

        def generate_cases(self, *args):
            raise AssertionError("worker regenerated frozen inputs")

        def evaluate_case(self, result, case, *args, **kwargs):
            outcome = make_failure_outcomes(1, "fixture")[0]
            if case.case_id == "case_0":
                outcome = replace(
                    outcome,
                    finite=True,
                    ordered_waypoints_reached=True,
                    joint_limit_violation=False,
                    motion_valid=True,
                    planning_success=True,
                    completed_waypoint_ratio=1.0,
                    failure_code=None,
                )
            return ScenarioEvaluation(outcomes=(outcome,))

    with (
        patch.object(runner_module, "SimulationManager", factory),
        patch.object(runner_module, "create_robot_provider", return_value=provider),
        patch.object(
            runner_module, "create_scenario_provider", return_value=Scenario()
        ),
        patch.object(
            runner_module,
            "create_planner_adapter",
            side_effect=lambda spec, context: ExternalAdapter(spec, context),
        ),
    ):
        return _run_shard(job)


@pytest.mark.parametrize("execution", ["inline", "spawn"])
def test_parallel_quality_merges_full_population_and_preserves_inputs(
    fake_runtime, tmp_path, monkeypatch, execution
):
    from dataclasses import replace

    from embodichain_benchmarks.motion_generation import parallel, prepare_case_manifest
    from embodichain_benchmarks.motion_generation.models import TrialPhase

    suite = load_suite()
    suite.protocol.warmup_trials = 0
    suite.protocol.measured_trials = 1
    suite.planners = [
        PlannerSpecCfg(id="fixture", adapter="ik_interpolate", enabled=True)
    ]
    original_generate = fake_runtime[2].generate_cases

    def generate(*args):
        base = original_generate(*args)[0]
        return [
            replace(base, case_id=f"case_{index}", seed=11 + index)
            for index in range(3)
        ]

    def evaluate(result, case, *args, **kwargs):
        outcome = make_failure_outcomes(1, "fixture")[0]
        if case.case_id == "case_0":
            outcome = replace(
                outcome,
                finite=True,
                ordered_waypoints_reached=True,
                joint_limit_violation=False,
                motion_valid=True,
                planning_success=True,
                completed_waypoint_ratio=1.0,
                failure_code=None,
            )
        return ScenarioEvaluation(outcomes=(outcome,))

    monkeypatch.setattr(fake_runtime[2], "generate_cases", generate)
    monkeypatch.setattr(fake_runtime[2], "evaluate_case", evaluate)
    corpus = prepare_case_manifest(suite, tmp_path / "frozen.json", device="cpu")
    original = corpus.read_bytes()
    monkeypatch.setattr(
        fake_runtime[2], "generate_cases", lambda *a: pytest.fail("regenerated")
    )
    monkeypatch.setattr(
        runner_module,
        "create_planner_adapter",
        lambda spec, context: ExternalAdapter(spec, context),
    )
    if execution == "inline":
        monkeypatch.setattr(parallel, "ProcessPoolExecutor", _InlineExecutor)
    else:
        monkeypatch.setattr(parallel, "_run_shard", _spawned_fake_shard)
    serial = run_benchmark(
        suite, case_manifest=corpus, output_root=tmp_path / "serial", device="cpu"
    )
    result = run_benchmark(
        suite,
        case_manifest=corpus,
        output_root=tmp_path / "parallel",
        device="cpu",
        workers=2,
    )
    measured = [row for row in result.records if row.phase is TrialPhase.MEASURED]
    assert [row.case_id for row in measured] == ["case_0", "case_1", "case_2"]
    assert all(row.cost_time_ms is not None for row in measured)
    assert all(
        row.metadata["benchmark_execution"]["timing_comparable"] is False
        for row in measured
    )
    ranking = result.aggregates["motion_planning_leaderboard"][0]
    assert ranking["overall_success_rate"] == pytest.approx(1 / 3)
    assert (
        ranking["overall_success_rate"]
        == serial.aggregates["motion_planning_leaderboard"][0]["overall_success_rate"]
    )
    assert ranking["latency_p95_ms"] is None
    assert corpus.read_bytes() == original
    assert (
        json.loads((result.run_dir / "case_manifest.json").read_text())["cases_sha256"]
        == json.loads(original)["cases_sha256"]
    )
    assert json.loads((result.run_dir / "execution.json").read_text())["workers"] == 2
    assert result.report_path.is_file()

    frozen_rows = json.loads(original)["cases"]
    omitted = replace(
        result,
        records=tuple(
            row
            for row in result.records
            if not (row.case_id == "case_0" and row.phase is TrialPhase.MEASURED)
        ),
    )
    with pytest.raises(ValueError, match="omitted measured trials"):
        parallel._merge_records([omitted], frozen_rows, suite.planners, 1)
    failed = replace(
        measured[0],
        phase=TrialPhase.CONSTRUCT,
        repeat=-1,
        status="error",
        failure_code="planner_exception",
        outcomes=(),
    )
    lifecycle_failure = replace(
        result,
        records=(failed,)
        + tuple(row for row in result.records if row.case_id != "case_0"),
    )
    preserved, _ = parallel._merge_records(
        [lifecycle_failure], frozen_rows, suite.planners, 1
    )
    assert failed in preserved

    if execution == "spawn":
        state = json.loads((result.run_dir / "execution.json").read_text())
        import os

        assert len(state["shards"]) == 2
        assert all(shard["pid"] != os.getpid() for shard in state["shards"])
        assert all(shard["torch_threads"] == 1 for shard in state["shards"])
        return
    actual_worker = parallel._run_shard

    def failed_worker(job):
        if job["index"] == 1:
            raise RuntimeError("worker fixture failure")
        return actual_worker(job)

    monkeypatch.setattr(parallel, "_run_shard", failed_worker)
    with pytest.raises(RuntimeError, match="worker fixture failure"):
        run_benchmark(
            suite,
            case_manifest=corpus,
            output_root=tmp_path / "failed",
            device="cpu",
            workers=2,
        )
    execution = list((tmp_path / "failed").glob("*/*/execution.json"))
    assert len(execution) == 1
    assert json.loads(execution[0].read_text())["status"] == "error"


def test_parallel_merges_initialized_and_unsupported_only_nmg_workers(
    fake_runtime, tmp_path, monkeypatch
):
    from copy import deepcopy
    from dataclasses import replace

    from embodichain_benchmarks.motion_generation import parallel, prepare_case_manifest
    from embodichain_benchmarks.motion_generation.models import TrialPhase
    from embodichain_benchmarks.motion_generation.planners.nmg_onnx import (
        NmgOnnxAdapter,
    )

    fake_runtime[1].add_robot.return_value.get_qvel_limits.return_value = torch.ones(
        1, 2
    )
    original_generate = fake_runtime[2].generate_cases

    def generate(*args):
        base = original_generate(*args)[0]
        return [
            replace(
                base,
                case_id=f"k{count}",
                num_waypoints=count,
                target_waypoints=base.target_waypoints.repeat(1, count, 1, 1),
                reference_qpos=base.reference_qpos.repeat(1, count, 1),
            )
            for count in (1, 6)
        ]

    monkeypatch.setattr(fake_runtime[2], "generate_cases", generate)
    monkeypatch.setattr(NmgOnnxAdapter, "availability", lambda self: (True, None))
    monkeypatch.setattr(
        NmgOnnxAdapter, "build", lambda self: self.resolve_motion_limits()
    )
    monkeypatch.setattr(NmgOnnxAdapter, "plan", ExternalAdapter.plan)
    monkeypatch.setattr(NmgOnnxAdapter, "events", [], raising=False)
    monkeypatch.setattr(NmgOnnxAdapter, "close", lambda self: None)
    monkeypatch.setattr(parallel, "ProcessPoolExecutor", _InlineExecutor)
    worker_results = []
    actual_worker = parallel._run_shard

    def capture_worker(job):
        result, state = actual_worker(job)
        worker_results.append(result)
        return result, state

    monkeypatch.setattr(parallel, "_run_shard", capture_worker)
    suite = load_suite()
    suite.protocol.warmup_trials = 0
    suite.protocol.measured_trials = 1
    suite.planners = [
        PlannerSpecCfg(
            id="nmg", adapter="nmg_onnx", enabled=True, config={"num_waypoints": 5}
        )
    ]
    corpus = prepare_case_manifest(suite, tmp_path / "frozen.json", device="cpu")
    serial = run_benchmark(
        suite, case_manifest=corpus, output_root=tmp_path / "serial", device="cpu"
    )
    result = run_benchmark(
        suite,
        case_manifest=corpus,
        output_root=tmp_path / "parallel",
        device="cpu",
        workers=2,
    )
    for run in (serial, result):
        measured = [row for row in run.records if row.phase is TrialPhase.MEASURED]
        unsupported = [row for row in run.records if row.status == "unsupported"]
        assert [row.case_id for row in measured] == ["k1"]
        assert [(row.case_id, row.failure_code) for row in unsupported] == [
            ("k6", "unsupported_capacity")
        ]
    assert result.planner_metadata == serial.planner_metadata

    missing, resolved = sorted(
        worker_results,
        key=lambda run: "resolved_motion_limits" in run.planner_metadata[0].parameters,
    )
    assert "resolved_motion_limits" not in missing.planner_metadata[0].parameters
    assert "resolved_motion_limits" in resolved.planner_metadata[0].parameters
    before = deepcopy([run.planner_metadata for run in worker_results])
    rows = json.loads(corpus.read_text())["cases"]
    for shards in ([missing, resolved], [resolved, missing]):
        _, metadata = parallel._merge_records(shards, rows, suite.planners, 1)
        assert tuple(metadata) == serial.planner_metadata

    info = resolved.planner_metadata[0]
    altered_limits = deepcopy(info.parameters)
    altered_limits["resolved_motion_limits"]["velocity"][0] += 0.1
    for conflict in (
        replace(info, model_revision="different_revision"),
        replace(info, parameters={**info.parameters, "num_waypoints": 6}),
        replace(info, parameters=altered_limits),
    ):
        conflicting = replace(missing, planner_metadata=(conflict,))
        with pytest.raises(ValueError, match="planner metadata differ"):
            parallel._merge_records([resolved, conflicting], rows, suite.planners, 1)
    assert [run.planner_metadata for run in worker_results] == before


def test_parallel_partition_preserves_shared_geometry_and_rejects_timing(
    fake_runtime, tmp_path, monkeypatch
):
    from embodichain_benchmarks.motion_generation import parallel, prepare_case_manifest
    from embodichain_benchmarks.motion_generation.config import stable_hash

    rows = [
        {
            "case_id": str(index),
            "track": "motion",
            "num_waypoints": index + 1,
            "batch_size": 1,
            "case_parameters": {"base_case_id": str(index // 2)},
        }
        for index in range(6)
    ]
    shards = parallel._partition_rows(rows, 4)
    assert sorted(index for shard in shards for index in shard) == list(range(6))
    assert len(shards) == 3
    for shard in shards:
        assert (
            len({rows[index]["case_parameters"]["base_case_id"] for index in shard})
            == 1
        )
    suite = load_suite()
    corpus = prepare_case_manifest(suite, tmp_path / "timing.json", device="cpu")
    payload = json.loads(corpus.read_text())
    payload["cases"][0]["case_parameters"]["measurement_purpose"] = "performance_replay"
    payload["cases_sha256"] = stable_hash(payload["cases"])
    corpus.write_text(json.dumps(payload))
    monkeypatch.setattr(
        parallel,
        "ProcessPoolExecutor",
        lambda **kwargs: pytest.fail("launched timing workers"),
    )
    with pytest.raises(ValueError, match="Performance replay must run serially"):
        run_benchmark(
            suite, case_manifest=corpus, output_root=tmp_path, device="cpu", workers=2
        )


@pytest.mark.parametrize(
    ("requested", "expected", "uses_current"),
    [
        ("cuda", "cuda:5", True),
        ("auto", "cuda:5", True),
        ("cuda:2", "cuda:2", False),
        ("cpu", "cpu", False),
    ],
    ids=["implicit_accelerator", "automatic", "explicit_accelerator", "host"],
)
def test_parallel_resolves_device_index_in_parent(
    monkeypatch, requested, expected, uses_current
):
    from embodichain_benchmarks.motion_generation.parallel import _worker_device

    current = Mock(return_value=5)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", current)
    resolved = _worker_device(requested)
    assert resolved == expected
    assert current.call_count == int(uses_current)
    if torch.device(resolved).type == "cuda":
        assert torch.device(resolved).index is not None
