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

"""Validate runnable locomotion configurations without patching their fields."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import faulthandler
import gc
import importlib
import json
import os
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

_NO_RENDER_TIMEOUT_SECONDS = 180
_WORKER_STARTED = time.perf_counter() if __name__ == "__main__" else None
if _WORKER_STARTED is not None:
    # Cover heavy imports as part of the first deployment's original budget.
    faulthandler.enable()
    faulthandler.dump_traceback_later(_NO_RENDER_TIMEOUT_SECONDS, exit=True)

import pytest

from embodichain.lab.gym.utils.gym_utils import config_to_cfg
from embodichain.lab.sim import cfg as sim_cfg
from embodichain.utils.utility import load_config

_TASKS = (
    "locomotion/velocity/g1_flat",
    "locomotion/velocity/h1_2_flat",
    "locomotion/velocity/go1_flat",
    "locomotion/velocity/go2_flat",
    "locomotion/velocity/anymal_c_flat",
    "locomotion/velocity/microduck_flat",
    "classic_control/humanoid",
)


@pytest.fixture(scope="module")
def _native_no_render_reports(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> Callable[[str], dict[str, dict[str, object]]]:
    """Run selected deployments with one child process per physics backend."""
    return _make_no_render_report_cache(request.session.items, tmp_path_factory)


def _make_no_render_report_cache(
    items: list[pytest.Item], tmp_path_factory: pytest.TempPathFactory
) -> Callable[[str], dict[str, dict[str, object]]]:
    """Cache results and infrastructure failure text without retrying workers."""
    reports: dict[str, dict[str, dict[str, object]]] = {}
    failures: dict[str, str] = {}

    def for_backend(backend: str) -> dict[str, dict[str, object]]:
        if backend not in reports and backend not in failures:
            try:
                tasks = tuple(
                    dict.fromkeys(
                        item.callspec.params["task"]
                        for item in items
                        if getattr(item, "obj", None)
                        is test_locomotion_runtime_uses_native_no_render
                        and item.callspec.params["backend"] == backend
                    )
                )
                assert tasks, f"No selected NoRender deployments for {backend}."
                report_path = (
                    tmp_path_factory.mktemp(f"no-render-{backend}") / "report.json"
                )
                reports[backend] = _run_native_no_render_batch(
                    tasks, backend, report_path
                )
            except Exception:
                failures[backend] = traceback.format_exc()
        assert backend not in failures, failures[backend]
        return reports[backend]

    return for_backend


@pytest.mark.no_sim
@pytest.mark.parametrize("task", _TASKS)
@pytest.mark.parametrize("backend", ("default", "newton"))
def test_packaged_deployment_and_agent_config_agree(
    task: str, backend: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real config decoding must work without smoke-script backend overrides."""
    monkeypatch.setattr(sim_cfg, "get_data_path", lambda value: value)
    directory = Path("extensions/embodichain_tasks/configs/tasks") / task
    suffix = "" if backend == "default" else ".newton"
    path = directory / f"env{suffix}.yaml"
    raw = load_config(str(path))
    agent = load_config(str(directory / f"agents/ppo{suffix}.yaml"))
    assert raw["physics"] == backend
    assert "physics_config_by_backend" not in raw
    cfg = config_to_cfg(
        raw,
        source_path=path,
        manager_modules=[
            f"embodichain_tasks.locomotion.managers.{module}"
            for module in ("observations", "rewards")
        ],
    )
    physics_type = (
        sim_cfg.DefaultPhysicsCfg if backend == "default" else sim_cfg.NewtonPhysicsCfg
    )
    assert isinstance(cfg.sim_cfg.physics_cfg, physics_type)
    assert cfg.sim_cfg.render_cfg.renderer in {
        "auto",
        "hybrid",
        "fast-rt",
        "rt",
        "no-render",
    }
    if task.startswith("locomotion/"):
        assert "renderer" not in raw
        assert "renderer" not in raw.get("render_cfg", {})
        assert "renderer" not in agent["trainer"]
        assert cfg.sim_cfg.render_cfg.renderer == "auto"
        expected_capacity = (
            {"g1_flat": 262144, "h1_2_flat": 262144, "microduck_flat": 524288}.get(
                task.rsplit("/", 1)[-1], 131071
            )
            if backend == "newton"
            else 131071
        )
        assert cfg.sim_cfg.scene_node_capacity == expected_capacity
        algorithm = agent["algorithm"]["cfg"]
        assert algorithm["batch_size"] == (
            agent["trainer"]["num_envs"]
            * agent["trainer"]["buffer_size"]
            // algorithm["num_mini_batches"]
        )
    else:
        assert agent["trainer"]["renderer"] == cfg.sim_cfg.render_cfg.renderer
    assert Path(agent["trainer"]["gym_config"]).resolve() == path.resolve()
    assert set(agent) == {"trainer", "policy", "algorithm"}
    assert agent["policy"]["initial_action_std"] > 0


@pytest.mark.subprocess_sim
@pytest.mark.gpu
@pytest.mark.parametrize(
    "task", [task for task in _TASKS if task.startswith("locomotion/")]
)
@pytest.mark.parametrize("backend", ("default", "newton"))
def test_locomotion_runtime_uses_native_no_render(
    task: str,
    backend: str,
    _native_no_render_reports: Callable[[str], dict[str, dict[str, object]]],
) -> None:
    """Packaged PPO runtime reaches the real NoRender engine without overrides."""
    report = _native_no_render_reports(backend)[task]
    _assert_native_no_render_report(report, task, backend)
    print("NORENDER_CHECK " + json.dumps(report))


def _run_native_no_render_batch(
    tasks: tuple[str, ...], backend: str, report_path: Path
) -> dict[str, dict[str, object]]:
    """Read complete structured results, including independently failed cases."""
    started = time.perf_counter()
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            backend,
            json.dumps(tasks),
            str(report_path),
        ],
        cwd=Path(__file__).resolve().parents[4],
        env={**os.environ, "EMBODICHAIN_SIM_EXIT_PROCESS": "0"},
        text=True,
        capture_output=True,
        timeout=_NO_RENDER_TIMEOUT_SECONDS * len(tasks),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert report_path.is_file(), (
        f"{backend}: worker returned without a verification report.\n"
        + result.stdout
        + result.stderr
    )
    batch = json.loads(report_path.read_text(encoding="utf-8"))
    assert batch["backend"] == backend
    assert batch["tasks"] == list(tasks)
    assert batch["process_starts"] == 1
    results = batch["results"]
    assert isinstance(results, list)
    assert all(isinstance(report, dict) for report in results)
    assert [report["task"] for report in results] == list(tasks)
    assert all(report["cleanup_passed"] is True for report in results)
    print(
        f"NORENDER_BATCH {backend}: process_starts=1, deployments={len(tasks)}, "
        f"elapsed_seconds={time.perf_counter() - started:.3f}"
    )
    return {report["task"]: report for report in results}


def _assert_native_no_render_report(
    report: dict[str, object], task: str, backend: str
) -> None:
    """Associate each original pytest node with its own measured result."""
    assert report["task"] == task
    assert report["physics"] == backend
    assert report["status"] == "passed", report.get("error", report)
    assert report["native_renderer"] == "NORENDER"


@pytest.mark.no_sim
@pytest.mark.parametrize(
    "native_stdout", ["", "\x1b[37mNative log\x00NORENDER_CHECK hidden\x00"]
)
def test_no_render_report_is_independent_of_native_stdout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, native_stdout: str
) -> None:
    task = "locomotion/velocity/g1_flat"

    def run_worker(args: list[str], **kwargs) -> subprocess.CompletedProcess:
        Path(args[-1]).write_text(
            json.dumps(
                {
                    "backend": "default",
                    "tasks": [task],
                    "process_starts": 1,
                    "results": [
                        {
                            "task": task,
                            "physics": "default",
                            "native_renderer": "NORENDER",
                            "status": "passed",
                            "cleanup_passed": True,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(args, 0, stdout=native_stdout, stderr="")

    monkeypatch.setattr(subprocess, "run", run_worker)
    reports = _run_native_no_render_batch((task,), "default", tmp_path / "report.json")
    _assert_native_no_render_report(reports[task], task, "default")


@pytest.mark.no_sim
def test_no_render_worker_must_produce_a_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args, 0, "native output", "native error"
        ),
    )
    with pytest.raises(AssertionError, match="native outputnative error"):
        _run_native_no_render_batch(
            ("locomotion/velocity/g1_flat",), "default", tmp_path / "report.json"
        )


@pytest.mark.no_sim
def test_failed_no_render_worker_is_not_started_again(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> None:
    tasks = ("locomotion/velocity/g1_flat", "locomotion/velocity/go1_flat")
    items = [
        SimpleNamespace(
            obj=test_locomotion_runtime_uses_native_no_render,
            callspec=SimpleNamespace(params={"task": task, "backend": "default"}),
        )
        for task in tasks
    ]
    launches: list[object] = []

    def run_worker(args: list[str], **kwargs) -> subprocess.CompletedProcess:
        launches.append(args)
        raise RuntimeError("native worker failed")

    monkeypatch.setattr(subprocess, "run", run_worker)
    for_backend = _make_no_render_report_cache(items, tmp_path_factory)
    for _ in tasks:
        with pytest.raises(AssertionError, match="native worker failed"):
            for_backend("default")
    assert len(launches) == 1


@pytest.mark.no_sim
@pytest.mark.parametrize(
    "failure", ["wrong-task", "duplicate", "cleanup-failure", "case-failure"]
)
def test_no_render_batch_results_fail_closed_per_deployment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure: str
) -> None:
    tasks = ("locomotion/velocity/g1_flat", "locomotion/velocity/go1_flat")
    records = [
        {
            "task": task,
            "physics": "default",
            "native_renderer": "NORENDER",
            "status": "passed",
            "cleanup_passed": True,
        }
        for task in tasks
    ]
    if failure == "wrong-task":
        records[1]["task"] = "unexpected-task"
    elif failure == "duplicate":
        records[1] = records[0].copy()
    elif failure == "cleanup-failure":
        records[0]["cleanup_passed"] = False
    else:
        records[0].update(status="failed", error="first deployment failed")

    def run_worker(args: list[str], **kwargs) -> subprocess.CompletedProcess:
        assert json.loads(args[-2]) == list(tasks)
        assert kwargs["timeout"] == 2 * _NO_RENDER_TIMEOUT_SECONDS
        assert kwargs["env"]["EMBODICHAIN_SIM_EXIT_PROCESS"] == "0"
        Path(args[-1]).write_text(
            json.dumps(
                {
                    "backend": "default",
                    "tasks": list(tasks),
                    "process_starts": 1,
                    "results": records,
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(subprocess, "run", run_worker)
    if failure == "case-failure":
        reports = _run_native_no_render_batch(
            tasks, "default", tmp_path / "report.json"
        )
        with pytest.raises(AssertionError, match="first deployment failed"):
            _assert_native_no_render_report(reports[tasks[0]], tasks[0], "default")
        _assert_native_no_render_report(reports[tasks[1]], tasks[1], "default")
    else:
        with pytest.raises(AssertionError):
            _run_native_no_render_batch(tasks, "default", tmp_path / "report.json")


def _verify_native_no_render(task: str, backend: str) -> dict[str, object]:
    """Build and fully release one deployment before returning its result."""
    import dexsim
    import torch
    from dexsim.types import Renderer
    from embodichain.lab.sim import SimulationManager
    from embodichain.learning.rl.runtime import _build_gym_environment

    started = time.perf_counter()
    report: dict[str, object] = {"task": task, "physics": backend}
    errors: list[str] = []
    cleanup_passed = True
    runtime = env = sim = observation = None
    try:
        importlib.import_module(
            f"embodichain_tasks.locomotion.velocity.{task.rsplit('/', 1)[-1]}"
        )
        directory = Path("extensions/embodichain_tasks/configs/tasks") / task
        suffix = "" if backend == "default" else ".newton"
        agent_path = directory / f"agents/ppo{suffix}.yaml"
        agent = load_config(agent_path)
        assert "renderer" not in agent["trainer"]
        sim_cfg.DEFAULT_RENDERER = "auto"
        runtime = _build_gym_environment(
            agent,
            simulation_device=torch.device(agent["trainer"]["device"]),
            num_envs=2 if backend == "default" else 1,
            headless=agent["trainer"]["headless"],
            renderer=agent["trainer"].get("renderer"),
            gpu_id=0,
            seed=agent["trainer"]["seed"],
            config_dir=agent_path.parent,
        )
        env = runtime.env.unwrapped
        sim = env.sim
        assert sim._requested_renderer == "auto"
        assert sim.sim_config.render_cfg.renderer == "no-render"
        assert sim.get_world().get_renderer_type() == Renderer.NORENDER
        assert not sim.get_world().is_window_initialized()
        assert not sim.has_native_renderer
        if backend == "default":
            # Measure real native allocations on a small batch, then budget the
            # authored PPO batch. Rounding up also budgets process-owned roots.
            import math

            nodes = sim.get_world().get_env().get_active_scene_node_count()
            nodes_per_env = math.ceil(nodes / env.num_envs)
            required = nodes_per_env * agent["trainer"]["num_envs"]
            capacity = sim.get_world().get_env().get_scene_node_capacity()
            assert required <= capacity <= 131071, (task, nodes, required, capacity)
        assert sim._sensors and all(
            sensor.cfg.sensor_type == "ContactSensor"
            for sensor in sim._sensors.values()
        )
        observation, _ = env.reset()
        for _ in range(2):
            observation, reward, _, _, _ = env.step(
                torch.zeros(env.action_space.shape, device=env.device)
            )
            assert torch.isfinite(reward).all()
            assert torch.isfinite(observation["policy"]).all()
        env.reset()
        report.update(
            {
                "physics": sim.physics.name,
                "requested_renderer": sim._requested_renderer,
                "native_renderer": sim.get_world().get_renderer_type().name,
                "contact_sensors": len(sim._sensors),
                "scene_nodes": sim.get_world().get_env().get_active_scene_node_count(),
                "scene_node_capacity": sim.get_world()
                .get_env()
                .get_scene_node_capacity(),
                "steps": 2,
                "reset": "passed",
            }
        )
    except Exception:
        # Store text, not exception tracebacks that could retain a live World.
        errors.append(traceback.format_exc())
    finally:
        try:
            if env is not None:
                env.close(exit_process=False)
        except Exception:
            cleanup_passed = False
            errors.append(traceback.format_exc())
        runtime = env = sim = observation = None
        gc.collect()
        try:
            SimulationManager.flush_cleanup_queue()
            assert (
                not SimulationManager._instances
            ), "SimulationManager instances remain."
            assert dexsim.get_world_num() == 0, "Native Worlds remain after cleanup."
        except Exception:
            cleanup_passed = False
            errors.append(traceback.format_exc())
    report.update(
        status="failed" if errors else "passed",
        cleanup_passed=cleanup_passed,
        elapsed_seconds=time.perf_counter() - started,
    )
    if errors:
        report["error"] = "\n".join(errors)
    return report


def _verify_no_render_backend(
    tasks: tuple[str, ...],
    backend: str,
    report_path: Path,
    *,
    startup_started: float | None = None,
) -> int:
    """Reuse process startup while retaining fresh topology and per-case results."""
    results: list[dict[str, object]] = []
    for index, task in enumerate(tasks):
        # A native stall must retain the original per-deployment time budget.
        budget = float(_NO_RENDER_TIMEOUT_SECONDS)
        if index == 0 and startup_started is not None:
            budget = max(0.001, budget - (time.perf_counter() - startup_started))
        faulthandler.dump_traceback_later(budget, exit=True)
        try:
            report = _verify_native_no_render(task, backend)
            results.append(report)
        finally:
            faulthandler.cancel_dump_traceback_later()
        if report["cleanup_passed"] is not True:
            break
    report_path.write_text(
        json.dumps(
            {
                "backend": backend,
                "tasks": list(tasks),
                "process_starts": 1,
                "results": results,
            }
        ),
        encoding="utf-8",
    )
    return (
        0
        if len(results) == len(tasks)
        and all(report["cleanup_passed"] is True for report in results)
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(
        _verify_no_render_backend(
            tuple(json.loads(sys.argv[2])),
            sys.argv[1],
            Path(sys.argv[3]),
            startup_started=_WORKER_STARTED,
        )
    )
