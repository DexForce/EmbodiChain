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

"""Independent selection, process isolation and success accounting contracts."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts.benchmark.atomic_action import reference_benchmark as benchmark

pytestmark = pytest.mark.no_sim


def _options(*arguments: str) -> dict:
    return vars(benchmark._parser().parse_args(list(arguments)))


def _case(skill: str, planner: str = "trapezoidal", robot: str = "ur5") -> dict:
    args = benchmark._parser().parse_args(
        ["--skill", skill, "--planner", planner, "--robot", robot]
    )
    return benchmark._cases(args)[0]


@pytest.mark.parametrize("skill", benchmark.SCENES)
@pytest.mark.parametrize("planner", benchmark.PLANNERS)
def test_independent_defaults_select_requested_skill_and_planner(
    skill: str, planner: str
) -> None:
    config, indices = benchmark._configuration(_case(skill, planner), _options())
    assert (config["skill"], config["robot"], config["planner"]) == (
        skill,
        "ur5",
        planner,
    )
    assert len(indices) == (4 if skill == "move_joint" else 16)
    assert "gym_config" not in config
    assert "task_program" not in config


@pytest.mark.parametrize(
    "arguments,expected",
    [
        ([], tuple(range(16))),
        (["--smoke"], (0, 1, 4, 5)),
        (["--no-expansion"], (0,)),
        (["--affordance-branches", "2", "--trajectory-variants", "2"], (0, 1, 2, 3)),
        (["--recipe-indices", "5", "0"], (5, 0)),
        (["--smoke", "--trajectory-variants", "1"], (0, 1, 2)),
    ],
)
def test_recipe_selection_independent_of_batch_capacity(
    arguments: list[str], expected: tuple[int, ...]
) -> None:
    config, indices = benchmark._configuration(_case("press"), _options(*arguments))
    assert indices == expected
    if "--no-expansion" in arguments:
        assert not config["expansion"]["trajectory"]["spatial"]["enabled"]
        assert not config["expansion"]["affordance"]["enabled"]


@pytest.mark.parametrize("indices", [[16], [-1], [0, 0]])
def test_invalid_recipes_rejected_before_simulation(indices: list[int]) -> None:
    options = _options()
    options["recipe_indices"] = indices
    with pytest.raises(ValueError, match="Recipe indices"):
        benchmark._configuration(_case("press"), options)


def test_robot_capabilities_match_independent_sample_presets() -> None:
    args = benchmark._parser().parse_args(["--robot", "all", "--skill", "all"])
    cases = benchmark._cases(args)
    assert len(cases) == 30
    assert sum(case["supported"] for case in cases) == 18
    assert _case("handover", robot="franka")["supported"]
    assert not _case("slide", robot="franka")["supported"]


@pytest.mark.parametrize(
    "skill", [skill for skill in benchmark.SCENES if skill != "handover"]
)
def test_benchmark_retains_each_single_arm_sample_tcp(
    skill: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.benchmark.atomic_action import scenes

    monkeypatch.setattr(
        scenes.support, "add_tutorial_robot", lambda sim, robot, **kwargs: kwargs
    )
    config = scenes._create_robot(
        object(), SimpleNamespace(robot="ur5", _config={"skill": skill})
    )
    assert config["tcp_point"] == (
        "default" if skill in {"move_joint", "press", "twist"} else "grasp"
    )


@pytest.mark.parametrize("robot", benchmark.ROBOTS)
def test_handover_benchmark_mounts_the_run_env_robotiq_asset(robot: str) -> None:
    """Exercise the benchmark robot factory without constructing native simulation."""
    from scripts.benchmark.atomic_action import scenes

    sim = SimpleNamespace(
        is_newton_backend=False,
        add_robot=lambda *, cfg: SimpleNamespace(cfg=cfg),
    )
    args = SimpleNamespace(robot=robot, _config={"skill": "handover"})
    cfg = scenes._create_robot(sim, args).cfg
    expected = "Robotiq/robotiq_arg2f_140/robotiq_arg2f_140.urdf"
    for side in ("left", "right"):
        assert cfg.urdf_cfg.components[f"{side}_hand"]["urdf_path"].endswith(expected)


def test_independent_modules_import_without_gym_or_task_launchers() -> None:
    """Import the actual scene/execution modules under a forbidden-host import hook."""
    source = """
import importlib.abc
import sys
class RejectHosts(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if (fullname.startswith("embodichain.lab.gym") or
            fullname in {"embodichain.lab.scripts.run_env", "embodichain.lab.scripts.run_task", "scripts.benchmark.task_program.demo_success"}):
            raise AssertionError("Benchmark imported a task host: " + fullname)
sys.meta_path.insert(0, RejectHosts())
from scripts.benchmark.atomic_action import scenes, execution, evaluation, reference_benchmark
"""
    subprocess.run(
        [sys.executable, "-c", source],
        cwd=benchmark._REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_worker_uses_fresh_unpadded_batches_without_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """All seeds/repeats run; failures remain rows; the last batch has one real slot."""
    import torch
    from scripts.benchmark.atomic_action import execution

    calls = []

    def run(
        config: dict, indices: tuple, options: dict, directory: Path, batch_id: int
    ) -> dict:
        calls.append((config["seed"], indices, batch_id))
        trial = benchmark._failure_trial(
            config["seed"], 0, indices, "object_not_grasped"
        )
        if len(calls) > 1:
            for row in trial["rows"]:
                row.update(
                    planning_success=True,
                    success=True,
                    failure_stage=None,
                    failure_reason=None,
                )
        return trial

    monkeypatch.setattr(execution, "run_batch", run)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    options = _options(
        "--num-envs",
        "2",
        "--repeat",
        "2",
        "--seeds",
        "0",
        "7",
        "--recipe-indices",
        "0",
        "1",
        "4",
    )
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"case": _case("press"), "options": options}))
    benchmark._worker(request)
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["status"] == "completed"
    assert len(calls) == 8
    assert [indices for _, indices, _ in calls] == [(0, 1), (4,)] * 4
    assert sum(len(trial["rows"]) for trial in result["trials"]) == 12
    assert [batch for _, _, batch in calls] == list(range(8))


def test_failed_worker_counts_unfinished_rows_and_continues_cases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    def run(command: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append(command)
        request = Path(command[-1])
        spec = json.loads(request.read_text())
        if len(calls) == 1:
            return SimpleNamespace(returncode=-11)
        (request.parent / "result.json").write_text(
            json.dumps(dict(spec["case"], status="validated", trials=[]))
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(benchmark.subprocess, "run", run)
    args = benchmark._parser().parse_args(
        ["--skill", "press", "slide", "--smoke", "--output-dir", str(tmp_path)]
    )
    with pytest.raises(RuntimeError, match="Some benchmark cases failed"):
        benchmark.run_all_benchmarks(args)
    results = json.loads((tmp_path / "results.json").read_text())["cases"]
    assert [case["status"] for case in results] == ["failed", "validated"]
    assert sum(len(trial["rows"]) for trial in results[0]["trials"]) == 4


def test_partial_failure_does_not_double_count_finished_rows() -> None:
    options = _options("--repeat", "2", "--seeds", "0", "7")
    result = {
        "recipe_indices": [0, 1, 4],
        "trials": [benchmark._failure_trial(0, 0, (0, 1), "object_dropped")],
    }
    benchmark._fill_missing(result, options, "timeout")
    identities = [
        (trial["seed"], trial["repeat"], index)
        for trial in result["trials"]
        for index in trial["recipe_indices"]
    ]
    assert len(identities) == len(set(identities)) == 12


def test_report_ranks_physical_outcomes_with_weighted_rows_and_all_planners(
    tmp_path: Path,
) -> None:
    def result(skill: str, planner: str, successes: tuple[bool, ...]) -> dict:
        trial = benchmark._failure_trial(0, 0, range(len(successes)), "task_goal_miss")
        for row, success in zip(trial["rows"], successes):
            row.update(planning_success=True, success=success)
        return dict(_case(skill, planner), status="completed", trials=[trial])

    content = benchmark._report(
        [
            result("press", "trapezoidal", (True,)),
            result("pickup", "trapezoidal", (False, False, False)),
            result("press", "toppra", (True, False)),
            dict(_case("press", "curobo"), status="skipped", trials=[]),
        ],
        tmp_path,
    ).read_text()
    assert content.count("| rank |") == 1
    assert content.count("| case |") == 2
    assert "| 1 | toppra | 2 | 0.5 |" in content
    assert "| 2 | trapezoidal | 4 | 0.25 |" in content
    assert "| 3 | curobo | 0 | n/a |" in content
    assert "`success_rate` is physical skill success" in content


@pytest.mark.parametrize(
    "cuda_available,device,installed,reason",
    [
        (False, "cuda", True, "CUDA device"),
        (True, "cpu", True, "--device cuda"),
        (True, "cuda", False, "not installed"),
    ],
    ids=("missing-hardware", "cpu-runtime", "missing-runtime"),
)
def test_unavailable_runtime_skips_without_scene_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cuda_available: bool,
    device: str,
    installed: bool,
    reason: str,
) -> None:
    import torch

    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(
            {"case": _case("press", "curobo"), "options": _options("--device", device)}
        )
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)
    monkeypatch.setattr(
        benchmark.importlib.util, "find_spec", lambda _: object() if installed else None
    )
    benchmark._worker(request)
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["status"] == "skipped"
    assert reason in result["reason"]


@pytest.mark.parametrize("legacy", [False, True])
def test_cli_dispatch_preserves_explicit_legacy_action(
    monkeypatch: pytest.MonkeyPatch, legacy: bool
) -> None:
    from scripts.benchmark import __main__ as benchmark_cli
    from scripts.benchmark.atomic_action import run_benchmark

    called = []
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "atomic-action",
            *(["--action", "move_joints"] if legacy else ["--skill", "press"]),
        ],
    )
    monkeypatch.setattr(benchmark, "main", lambda: called.append("independent"))
    monkeypatch.setattr(
        run_benchmark, "run_all_benchmarks", lambda: called.append("legacy")
    )
    benchmark_cli._run_atomic_action_cli(SimpleNamespace())
    assert called == ["legacy" if legacy else "independent"]
