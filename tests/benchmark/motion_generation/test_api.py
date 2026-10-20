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
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
import yaml

from embodichain.benchmarks.motion_generation import (
    BenchmarkCase,
    PlannerAdapter,
    PlannerSpecCfg,
    load_suite,
    run_benchmark,
)
from embodichain.benchmarks.motion_generation import runner as runner_module
from embodichain.benchmarks.motion_generation.config import suite_to_dict
from embodichain.benchmarks.motion_generation.metrics.trajectory import (
    make_failure_outcomes,
)
from embodichain.benchmarks.motion_generation.registry import planner_adapter_names
from embodichain.benchmarks.motion_generation.scenarios.base import (
    ScenarioEvaluation,
    ScenarioProvider,
)
from embodichain.lab.sim.motion.planners.utils import PlanResult

pytestmark = pytest.mark.no_sim


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
    factory = Mock()
    monkeypatch.setattr(runner_module, "BenchmarkRunner", factory)
    suite = load_suite()
    result = run_benchmark(suite, output_root=tmp_path)
    assert result is factory.return_value.run.return_value
    selected = factory.call_args.args[1]
    assert [spec.id for spec in selected] == [
        spec.id for spec in suite.planners if spec.enabled
    ]
    assert factory.call_args.args[0] is not suite


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
    from embodichain.benchmarks.motion_generation.artifacts import _git_commit

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
