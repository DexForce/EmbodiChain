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

"""Run a generated Action Engine configuration."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
from types import SimpleNamespace
from typing import Any

import gymnasium
import numpy as np
import torch

from embodichain.gen_sim.action_engine.config import generation_defaults
from embodichain.gen_sim.action_engine.environment import (  # noqa: F401
    ACTION_ENGINE_ENV_ID,
)
from embodichain.gen_sim.action_engine.runtime import load_agent_execution_program
from embodichain.lab.gym.utils.gym_utils import (
    add_env_launcher_args_to_parser,
    build_env_cfg_from_args,
)
from embodichain.utils import set_seed
from embodichain.utils.logger import log_info, log_warning
from embodichain.utils.utility import load_config

__all__ = ["build_parser", "cli"]

_DEFAULT_MAX_EPISODES = int(generation_defaults()["task"]["max_episodes"])


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser used by generated demo commands."""
    parser = argparse.ArgumentParser(description="Execute an Action Engine task agent.")
    add_env_launcher_args_to_parser(parser)
    parser.add_argument("--task_name", required=True, help="Generated task name.")
    parser.add_argument(
        "--agent_config",
        required=True,
        help="Path to action_engine_config_v2 JSON.",
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Rebuild SeedGraph from TaskSpec in memory before execution.",
    )
    parser.add_argument(
        "--show-physical-collision",
        action="store_true",
        help="Show physical collision geometry after every reset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base random seed; episode N uses seed + N.",
    )
    parser.add_argument(
        "--runtime-backend",
        choices=("independent",),
        default="independent",
        help="Execution backend. Action Engine owns the production runtime.",
    )
    parser.add_argument(
        "--vlm-model",
        default=None,
        help="Optional runtime override for A/B visual facts and online planning.",
    )
    parser.add_argument(
        "--collaboration-report",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser


def _validate_gym_id(config: dict[str, Any]) -> None:
    if config.get("id") != ACTION_ENGINE_ENV_ID:
        raise ValueError(
            f"Gym config id must be {ACTION_ENGINE_ENV_ID!r}, "
            f"got {config.get('id')!r}."
        )


def _validate_run_contract(
    gym_config: dict[str, Any],
    agent_config: dict[str, Any],
    task_name: str,
) -> None:
    """Validate the small cross-artifact contract before simulator startup."""
    configured_task = agent_config.get("task_name")
    if configured_task != task_name:
        raise ValueError(
            f"--task_name {task_name!r} does not match agent_config task "
            f"{configured_task!r}."
        )
    extension = gym_config.get("env", {}).get("extensions", {}).get("action_engine", {})
    if extension.get("task_name") != task_name:
        raise ValueError("Gym and agent configs describe different tasks.")
    gym_hash = extension.get("seed_task_graph_hash")
    agent_hash = agent_config.get("seed_task_graph_hash")
    if not isinstance(agent_hash, str) or not agent_hash or gym_hash != agent_hash:
        raise ValueError("Gym and agent configs have different program hashes.")
    agent_mode = str(agent_config.get("planning_mode", "offline"))
    gym_mode = str(extension.get("planning_mode", "offline"))
    if agent_mode != gym_mode:
        raise ValueError(
            f"Gym and agent configs have different planning modes: "
            f"gym={gym_mode!r}, agent={agent_mode!r}."
        )


def cli() -> int | None:
    """Launch the environment and execute all configured episodes."""
    np.set_printoptions(precision=5, suppress=True)
    torch.set_printoptions(precision=5, sci_mode=False)
    args = build_parser().parse_args()
    if args.seed is not None:
        set_seed(args.seed)
    env_cfg, gym_config, _ = build_env_cfg_from_args(args)
    if args.seed is not None:
        env_cfg.seed = args.seed
    _validate_gym_id(gym_config)
    agent_config = load_config(args.agent_config)
    if not isinstance(agent_config, dict):
        raise ValueError("agent_config must contain a JSON object.")
    _validate_run_contract(gym_config, agent_config, args.task_name)
    planning_mode = str(agent_config.get("planning_mode", "offline"))
    if planning_mode == "ab":
        _run_ab(
            args,
            env_cfg=env_cfg,
            gym_config=gym_config,
            agent_config=agent_config,
        )
        return 0 if args.collaboration_report else None
    if planning_mode != "offline":
        raise ValueError(f"Unsupported Action Engine planning_mode {planning_mode!r}.")
    execution_program = load_agent_execution_program(
        agent_config,
        agent_config_path=args.agent_config,
        regenerate=bool(args.regenerate),
    )
    grounded_plan = _load_grounded_task_plan(args.agent_config)
    action_reporter = None
    if grounded_plan is not None:
        from embodichain.gen_sim.action_engine.agent import ActionAgent

        action_reporter = ActionAgent()

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    episodes = int(gym_config.get("max_episodes", _DEFAULT_MAX_EPISODES))
    runtime_arguments = {
        "agent_config": str(Path(args.agent_config).expanduser().resolve()),
        "base_seed": args.seed,
        "gym_config": str(Path(args.gym_config).expanduser().resolve()),
        "max_episodes": episodes,
        "planning_mode": planning_mode,
        "regenerate": bool(args.regenerate),
        "runtime_backend": str(args.runtime_backend),
        "task_name": str(args.task_name),
    }
    any_failed = False
    episode_index = 0
    episode_seed = None
    seed_graph = getattr(execution_program, "seed_graph", None)
    env = None
    try:
        env = gymnasium.make(
            id=gym_config["id"],
            cfg=env_cfg,
            agent_config=agent_config,
            agent_config_path=args.agent_config,
            task_name=args.task_name,
            runtime_backend=args.runtime_backend,
        )
        for episode_index in range(episodes):
            episode_seed = None if args.seed is None else int(args.seed) + episode_index
            env.reset(seed=episode_seed)
            if args.show_physical_collision:
                _show_physical_collision(env)
            execute = env.get_wrapper_attr("create_demo_action_list")
            result = execute(
                regenerate=bool(args.regenerate),
                runtime_run_id=run_id,
                episode_index=episode_index,
            )
            if not getattr(result, "already_executed", False):
                raise RuntimeError(
                    "Action Engine env returned an offline action sequence."
                )
            success = torch.as_tensor(
                getattr(result, "runtime_success"),
                dtype=torch.bool,
            )
            any_failed = any_failed or not bool(success.all())
            log_info(
                "Action Engine episode "
                f"{episode_index}: {int(success.sum())}/{success.numel()} "
                "environments succeeded.",
                color="green",
            )
            record_dir = getattr(result, "runtime_graph_output_dir", None)
            if record_dir:
                log_info(f"Runtime records: {record_dir}", color="green")
            if action_reporter is not None and isinstance(seed_graph, Mapping):
                report = action_reporter.report_execution_result(
                    result,
                    action_graph=seed_graph,
                    grounded_plan=grounded_plan,
                    run_id=run_id,
                    episode_index=episode_index,
                    episode_seed=episode_seed,
                    runtime_arguments=runtime_arguments,
                )
                log_info(
                    "Execution report: "
                    f"status={report.status}, actions={report.action_count}",
                    color="green" if report.status == "succeeded" else "yellow",
                )
        # EmbodiedEnv publishes the just-finished rollout during reset. Flush
        # the final episode as well; otherwise only episodes followed by a next
        # iteration reach the configured dataset recorder.
        env.reset(options={"final": True})
    except KeyboardInterrupt:
        log_warning("Action Engine run interrupted by user.")
        return 130 if args.collaboration_report else None
    except Exception as exc:
        if action_reporter is not None and isinstance(seed_graph, Mapping):
            report = action_reporter.abortion_report(
                seed_graph,
                exc,
                grounded_plan=grounded_plan,
                environment_count=_runtime_environment_count(env),
                run_id=run_id,
                episode_index=episode_index,
                episode_seed=episode_seed,
                runtime_arguments=runtime_arguments,
            )
            from embodichain.gen_sim.action_engine.runtime import (
                write_execution_report,
            )

            write_execution_report(Path(args.agent_config).resolve().parent, report)
        if args.collaboration_report:
            log_warning(f"Action Engine execution aborted: {type(exc).__name__}: {exc}")
            return 3
        raise
    finally:
        close = getattr(env, "close", None) if env is not None else None
        if callable(close):
            close()
    return int(any_failed) if args.collaboration_report else None


def _load_grounded_task_plan(agent_config_path: str | Path) -> dict[str, Any] | None:
    """Load the optional collaboration hand-off beside a legacy agent config."""
    path = (
        Path(agent_config_path).expanduser().resolve().parent
        / "grounded_task_plan.json"
    )
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read GroundedTaskPlan at {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ValueError("grounded_task_plan.json must contain a JSON object.")
    from embodichain.gen_sim.collaboration.contracts import (
        validate_grounded_task_plan,
    )

    return validate_grounded_task_plan(value)


def _runtime_environment_count(env: Any) -> int:
    value = getattr(getattr(env, "unwrapped", env), "num_envs", 1)
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return 1


class _BranchExecutor:
    def __init__(
        self,
        graph: dict[str, Any],
        env: gymnasium.Env,
        *,
        record_root: Path,
    ) -> None:
        self.graph = graph
        self.env = env
        self.record_root = record_root

    def preflight(self) -> bool:
        """Compile and capability-check the branch without sending motion."""
        route = getattr(self.env.unwrapped, "action_engine_ab_route", None)
        if route in {"offline", "online"} and self.graph.get("planner_route") != route:
            raise ValueError(
                f"A/B branch route {route!r} cannot execute graph route "
                f"{self.graph.get('planner_route')!r}."
            )
        try:
            preflight = self.env.get_wrapper_attr("preflight_seed_graph")
        except AttributeError:
            preflight = None
        if callable(preflight):
            value = preflight(self.graph)
            return value is not False
        # Older generated environments expose only execute_seed_graph.  The
        # loader is still a useful structural/capability preflight and does
        # not step the simulator.
        from embodichain.gen_sim.action_engine.runtime import load_execution_program

        source = self.env.unwrapped.agent_config.get("source")
        if source is None:
            source = {}
        if not isinstance(source, dict):
            raise ValueError("agent_config.source must be a mapping when provided.")
        uid_map = source.get("uid_map", {})
        if not isinstance(uid_map, dict):
            raise ValueError(
                "agent_config.source.uid_map must be a mapping when provided."
            )
        known_objects = {str(uid) for uid in uid_map.values()}
        load_execution_program(self.graph, known_objects=known_objects or None)
        return True

    def run(self, *, run_id: str, episode_index: int) -> Any:
        execute = self.env.get_wrapper_attr("execute_seed_graph")
        return execute(
            self.graph,
            runtime_run_id=run_id,
            episode_index=episode_index,
            record_root=self.record_root.as_posix(),
        )


@dataclass(frozen=True)
class _ABWorkerConfig:
    """Serializable startup contract for one process-isolated A/B branch."""

    route: str
    gym_config: dict[str, Any]
    env_options: dict[str, Any]
    gym_id: str
    agent_config: dict[str, Any]
    agent_config_path: str
    task_name: str
    runtime_backend: str
    seed: int
    camera_uids: tuple[str, ...]
    staging_dir: str


class _ABBranchWorker:
    """Small RPC proxy for one simulator process.

    DexSim entities resolve through a process-global default world.  Keeping
    each branch in a separate process is therefore a correctness requirement,
    not merely a way to parallelize A/B execution.
    """

    _STARTUP_TIMEOUT_SECONDS = 300.0
    _COMMAND_TIMEOUT_SECONDS = 1800.0
    _SHUTDOWN_TIMEOUT_SECONDS = 30.0

    def __init__(self, config: _ABWorkerConfig) -> None:
        self.action_engine_ab_route = config.route
        self._config = config
        self._closed = False
        self._context = mp.get_context("spawn")
        self._connection, child_connection = self._context.Pipe(duplex=True)
        self._process = self._context.Process(
            target=_ab_worker_main,
            args=(child_connection, config),
            name=f"action-engine-ab-{config.route}",
        )
        try:
            self._process.start()
        except BaseException:
            child_connection.close()
            self._connection.close()
            raise
        child_connection.close()
        try:
            startup = self._receive(
                "startup", timeout_seconds=self._STARTUP_TIMEOUT_SECONDS
            )
        except Exception:
            self.close()
            raise
        if not isinstance(startup, dict):
            self.close()
            raise RuntimeError(
                f"A/B {config.route} worker returned an invalid startup payload."
            )
        snapshot = startup.get("snapshot")
        if not isinstance(snapshot, dict):
            self.close()
            raise RuntimeError(
                f"A/B {config.route} worker did not return its reset snapshot."
            )
        self.startup_snapshot = snapshot
        self.startup_observation = startup.get("observation")

    def snapshot(self) -> dict[str, Any]:
        value = self._request("snapshot")
        if not isinstance(value, dict):
            raise RuntimeError(
                f"A/B {self.action_engine_ab_route} worker returned an invalid snapshot."
            )
        return value

    def preflight(self, graph: dict[str, Any]) -> bool:
        value = self._request("preflight", graph=graph)
        return value is not False

    def run(
        self,
        graph: dict[str, Any],
        *,
        run_id: str,
        episode_index: int,
        record_root: Path,
    ) -> Any:
        value = self._request(
            "run",
            graph=graph,
            run_id=run_id,
            episode_index=int(episode_index),
            record_root=record_root.as_posix(),
        )
        if not isinstance(value, dict):
            raise RuntimeError(
                f"A/B {self.action_engine_ab_route} worker returned an invalid result."
            )
        return _execution_result_from_wire(value)

    def finalize(self, branch_dir: Path, *, episode_index: int) -> list[str]:
        value = self._request(
            "finalize",
            branch_dir=branch_dir.as_posix(),
            episode_index=int(episode_index),
        )
        if not isinstance(value, list) or not all(
            isinstance(path, str) and path for path in value
        ):
            raise RuntimeError(
                f"A/B {self.action_engine_ab_route} worker returned invalid video paths."
            )
        return value

    def close(self) -> None:
        """Ask the worker to clean up, then force-stop only if it is stuck."""
        if self._closed:
            return
        self._closed = True
        try:
            if self._process.is_alive():
                try:
                    self._connection.send({"op": "shutdown"})
                    self._receive(
                        "shutdown", timeout_seconds=self._SHUTDOWN_TIMEOUT_SECONDS
                    )
                except Exception:
                    # The process is still joined/terminated below.  Cleanup
                    # errors cannot justify leaking a simulator child.
                    pass
        finally:
            try:
                self._connection.close()
            finally:
                self._process.join(timeout=self._SHUTDOWN_TIMEOUT_SECONDS)
                if self._process.is_alive():
                    self._process.terminate()
                    self._process.join(timeout=self._SHUTDOWN_TIMEOUT_SECONDS)

    def _request(self, operation: str, **payload: Any) -> Any:
        if self._closed:
            raise RuntimeError(
                f"A/B {self.action_engine_ab_route} worker is already closed."
            )
        try:
            self._connection.send({"op": operation, **payload})
        except (BrokenPipeError, EOFError, OSError) as exc:
            raise RuntimeError(
                f"A/B {self.action_engine_ab_route} worker could not receive "
                f"{operation!r}."
            ) from exc
        return self._receive(operation, timeout_seconds=self._COMMAND_TIMEOUT_SECONDS)

    def _receive(self, operation: str, *, timeout_seconds: float) -> Any:
        try:
            ready = self._connection.poll(timeout_seconds)
        except (EOFError, OSError) as exc:
            raise RuntimeError(
                f"A/B {self.action_engine_ab_route} worker closed during {operation}."
            ) from exc
        if not ready:
            exit_code = self._process.exitcode
            if exit_code is not None:
                raise RuntimeError(
                    f"A/B {self.action_engine_ab_route} worker exited with code "
                    f"{exit_code} during {operation}."
                )
            raise TimeoutError(
                f"A/B {self.action_engine_ab_route} worker timed out during "
                f"{operation} after {timeout_seconds:.0f}s."
            )
        try:
            response = self._connection.recv()
        except (EOFError, OSError) as exc:
            raise RuntimeError(
                f"A/B {self.action_engine_ab_route} worker closed during {operation}."
            ) from exc
        if not isinstance(response, dict) or "ok" not in response:
            raise RuntimeError(
                f"A/B {self.action_engine_ab_route} worker returned a malformed "
                f"response during {operation}."
            )
        if response["ok"] is True:
            return response.get("value")
        message = response.get("error")
        if not isinstance(message, str) or not message:
            message = "unknown worker error"
        raise RuntimeError(
            f"A/B {self.action_engine_ab_route} worker failed during {operation}: "
            f"{message}"
        )


class _SerializedABBranch:
    """Run one branch in fresh isolated workers when two worlds do not fit.

    Each worker still owns a separate DexSim process and is reset from the
    exact same seed.  The proxy only serializes their GPU residency: it probes
    a reset for planning, starts a fresh worker for preflight, then starts one
    more fresh worker for execution.  Every startup digest must match the
    planning reset before an RPC is allowed to progress.
    """

    def __init__(
        self,
        config: _ABWorkerConfig,
        *,
        startup_snapshot: Mapping[str, Any],
        startup_observation: Any,
        expected_initial_state_digest: str,
        worker_factory: Callable[[_ABWorkerConfig], Any] | None = None,
    ) -> None:
        self.action_engine_ab_route = config.route
        self.startup_snapshot = deepcopy(dict(startup_snapshot))
        self.startup_observation = startup_observation
        self._config = config
        self._expected_initial_state_digest = expected_initial_state_digest
        self._worker_factory = worker_factory or _ABBranchWorker
        self._active_worker: Any | None = None
        self._closed = False

    def snapshot(self) -> dict[str, Any]:
        """Return the verified reset snapshot without rehydrating a GPU world."""
        return deepcopy(self.startup_snapshot)

    def preflight(self, graph: dict[str, Any]) -> bool:
        worker = self._start_worker("preflight")
        try:
            return worker.preflight(graph)
        finally:
            worker.close()

    def run(
        self,
        graph: dict[str, Any],
        *,
        run_id: str,
        episode_index: int,
        record_root: Path,
    ) -> Any:
        if self._active_worker is not None:
            raise RuntimeError(
                f"A/B {self.action_engine_ab_route} execution worker is already active."
            )
        worker = self._start_worker("execute")
        self._active_worker = worker
        return worker.run(
            graph,
            run_id=run_id,
            episode_index=episode_index,
            record_root=record_root,
        )

    def finalize(self, branch_dir: Path, *, episode_index: int) -> list[str]:
        worker = self._active_worker
        if worker is None:
            raise RuntimeError(
                f"A/B {self.action_engine_ab_route} has no execution worker to finalize."
            )
        try:
            return worker.finalize(branch_dir, episode_index=episode_index)
        finally:
            self._active_worker = None
            worker.close()

    def close(self) -> None:
        self._closed = True
        worker = self._active_worker
        self._active_worker = None
        if worker is not None:
            worker.close()

    def _start_worker(self, phase: str) -> Any:
        if self._closed:
            raise RuntimeError(
                f"A/B {self.action_engine_ab_route} serialized branch is closed."
            )
        worker = self._worker_factory(_ab_phase_worker_config(self._config, phase))
        try:
            from embodichain.gen_sim.action_engine.evaluation import state_digest

            snapshot = worker.startup_snapshot
            actual_digest = state_digest(snapshot)
            if actual_digest != self._expected_initial_state_digest:
                raise RuntimeError(
                    "Strict A/B serialized reset mismatch before "
                    f"{phase}: route={self.action_engine_ab_route}, "
                    f"expected={self._expected_initial_state_digest}, "
                    f"actual={actual_digest}."
                )
            return worker
        except BaseException:
            worker.close()
            raise


def _ab_phase_worker_config(config: _ABWorkerConfig, phase: str) -> _ABWorkerConfig:
    """Give serial lifecycle phases distinct recorder and dataset roots."""
    if not phase:
        return config
    staging_dir = Path(config.staging_dir)
    return replace(
        config,
        staging_dir=(staging_dir.parent / phase / staging_dir.name).as_posix(),
    )


def _prepare_ab_branches(
    configs: Mapping[str, _ABWorkerConfig],
    *,
    worker_factory: Callable[[_ABWorkerConfig], Any] = _ABBranchWorker,
    prefer_serial: bool | None = None,
    gpu_id: int | None = None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Start concurrent worlds, with a digest-checked serialized fallback.

    A renderer can consume several GiB per DexSim process.  On smaller GPUs,
    starting the second isolated branch may fail before any action is sent. In
    that case keeping one world resident is not a semantic requirement, while
    the reset digest is; use fresh one-at-a-time workers instead.
    """
    if prefer_serial is None:
        prefer_serial = _prefer_serial_ab_startup(gpu_id=gpu_id)
    if prefer_serial:
        log_warning(
            "A/B GPU capacity is below the concurrent-world budget; "
            "using serialized isolated workers with reset-digest checks."
        )
        return _prepare_serial_ab_branches(configs, worker_factory=worker_factory)

    workers: dict[str, Any] = {}
    try:
        for route in ("offline", "online"):
            workers[route] = worker_factory(configs[route])
    except Exception as error:
        for worker in workers.values():
            worker.close()
        if not _is_gpu_memory_error(error):
            raise
        log_warning(
            "A/B concurrent simulator startup exhausted GPU memory; "
            "using serialized isolated workers with reset-digest checks."
        )
        return _prepare_serial_ab_branches(configs, worker_factory=worker_factory)
    return (
        workers,
        {route: worker.startup_snapshot for route, worker in workers.items()},
    )


def _prepare_serial_ab_branches(
    configs: Mapping[str, _ABWorkerConfig],
    *,
    worker_factory: Callable[[_ABWorkerConfig], Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Probe one branch at a time and return lazy serialized branch proxies."""

    snapshots: dict[str, dict[str, Any]] = {}
    observations: dict[str, Any] = {}
    for route in ("offline", "online"):
        worker = worker_factory(_ab_phase_worker_config(configs[route], "probe"))
        try:
            snapshots[route] = worker.startup_snapshot
            observations[route] = worker.startup_observation
        finally:
            worker.close()
    from embodichain.gen_sim.action_engine.evaluation import state_digest

    expected_digest = state_digest(snapshots["offline"])
    return (
        {
            route: _SerializedABBranch(
                configs[route],
                startup_snapshot=snapshots[route],
                startup_observation=observations[route],
                expected_initial_state_digest=expected_digest,
                worker_factory=worker_factory,
            )
            for route in ("offline", "online")
        },
        snapshots,
    )


def _is_gpu_memory_error(error: BaseException) -> bool:
    """Recognize the process-startup failures where serialization is safe."""
    message = str(error).lower()
    memory_markers = (
        "out of memory",
        "out_of_memory",
        "out_of_device_memory",
        "outofmemory",
        "resource exhausted",
    )
    return any(marker in message for marker in memory_markers) and (
        "cuda" in message
        or "gpu" in message
        or "vulkan" in message
        or "device" in message
    )


def _prefer_serial_ab_startup(*, gpu_id: int | None = None) -> bool:
    """Avoid a known OOM trial on GPUs too small for two renderer worlds."""
    if not torch.cuda.is_available():
        return False
    try:
        device = torch.device(f"cuda:{int(gpu_id)}" if gpu_id is not None else "cuda")
        free, _ = torch.cuda.mem_get_info(device=device)
    except (RuntimeError, ValueError):
        return False
    # One hybrid DexSim world with the four VLM cameras can occupy roughly
    # 11--13 GiB on the supported RTX setup.  Reserve 24 GiB for two worlds;
    # larger cards still attempt concurrent startup and retain the OOM fallback
    # for unusually heavy scenes.
    return int(free) < 24 * 1024**3


class _RemoteBranchExecutor:
    """Executor adapter which keeps simulator calls inside the branch worker."""

    def __init__(
        self,
        graph: dict[str, Any],
        worker: Any,
        *,
        record_root: Path,
    ) -> None:
        self.graph = graph
        self.worker = worker
        self.record_root = record_root

    def preflight(self) -> bool:
        return self.worker.preflight(self.graph)

    def run(self, *, run_id: str, episode_index: int) -> Any:
        return self.worker.run(
            self.graph,
            run_id=run_id,
            episode_index=episode_index,
            record_root=self.record_root,
        )


def _ab_worker_main(connection: Any, config: _ABWorkerConfig) -> None:
    """Create and drive exactly one real environment in a child process."""
    # SimulationManager otherwise exits the whole worker with os._exit(0)
    # during environment cleanup, bypassing the artifact/RPC shutdown contract.
    os.environ["EMBODICHAIN_SIM_EXIT_PROCESS"] = "0"
    env: gymnasium.Env | None = None
    try:
        from embodichain.lab.gym.utils.gym_utils import (
            config_to_cfg,
            get_manager_modules,
        )

        # ``config_to_cfg`` creates local component-config classes which are
        # intentionally not picklable.  Send the merged JSON contract over IPC
        # and reconstruct it inside each worker instead of pickling ``env_cfg``.
        branch_cfg = config_to_cfg(
            deepcopy(config.gym_config),
            manager_modules=get_manager_modules(),
        )
        _apply_ab_env_options(branch_cfg, config.env_options)
        branch_cfg.seed = int(config.seed)
        _configure_ab_branch_cfg(
            branch_cfg,
            staging_dir=Path(config.staging_dir),
            dataset_dir=Path(config.staging_dir).parent / ".dataset",
        )
        set_seed(int(config.seed))
        env = gymnasium.make(
            id=config.gym_id,
            cfg=branch_cfg,
            agent_config=deepcopy(config.agent_config),
            agent_config_path=config.agent_config_path,
            task_name=config.task_name,
            runtime_backend=config.runtime_backend,
        )
        setattr(env.unwrapped, "action_engine_ab_route", config.route)
        env.reset(seed=int(config.seed))
        startup: dict[str, Any] = {
            "snapshot": _snapshot_environment(env, list(config.camera_uids)),
        }
        if config.route == "online":
            from embodichain.gen_sim.action_engine.planning import (
                collect_scene_observation,
            )

            startup["observation"] = collect_scene_observation(
                env.unwrapped,
                camera_uids=config.camera_uids,
                env_id=0,
            )
        # The recorder normally receives its first frame from an interval
        # event during ``env.step``.  Capture one reset-time, no-motion frame so
        # an execution branch that fails before its first action still has a
        # valid video artifact after the mandatory final reset.
        _capture_ab_initial_frame(env)
        _worker_send(connection, ok=True, value=startup)
        while True:
            try:
                request = connection.recv()
            except EOFError:
                break
            if not isinstance(request, dict):
                raise ValueError("A/B worker request must be a mapping.")
            operation = request.get("op")
            try:
                if operation == "snapshot":
                    value = _snapshot_environment(env, list(config.camera_uids))
                elif operation == "preflight":
                    graph = _worker_graph(request.get("graph"))
                    value = _BranchExecutor(
                        graph,
                        env,
                        record_root=Path(config.staging_dir).parent / "runtime",
                    ).preflight()
                elif operation == "run":
                    graph = _worker_graph(request.get("graph"))
                    run_id = request.get("run_id")
                    record_root = request.get("record_root")
                    if not isinstance(run_id, str) or not run_id:
                        raise ValueError(
                            "A/B worker run_id must be a non-empty string."
                        )
                    if not isinstance(record_root, str) or not record_root:
                        raise ValueError(
                            "A/B worker record_root must be a non-empty path string."
                        )
                    result = _BranchExecutor(
                        graph,
                        env,
                        record_root=Path(record_root).expanduser().resolve(),
                    ).run(
                        run_id=run_id,
                        episode_index=int(request.get("episode_index", 0)),
                    )
                    value = _execution_result_to_wire(result)
                elif operation == "finalize":
                    branch_dir = request.get("branch_dir")
                    if not isinstance(branch_dir, str) or not branch_dir:
                        raise ValueError(
                            "A/B worker branch_dir must be a non-empty path string."
                        )
                    value = _finalize_ab_branch_video(
                        env,
                        staging_dir=Path(config.staging_dir),
                        branch_dir=Path(branch_dir).expanduser().resolve(),
                    )
                elif operation == "shutdown":
                    _worker_send(connection, ok=True, value=True)
                    break
                else:
                    raise ValueError(f"Unknown A/B worker operation {operation!r}.")
            except BaseException as exc:
                _worker_send(connection, ok=False, error=_worker_error(exc))
    except BaseException as exc:
        _worker_send(connection, ok=False, error=_worker_error(exc))
    finally:
        if env is not None:
            try:
                env.close()
            except BaseException:
                pass
            try:
                from embodichain.lab.sim import SimulationManager

                SimulationManager.flush_cleanup_queue()
            except BaseException:
                pass
        try:
            connection.close()
        except OSError:
            pass


def _worker_graph(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("A/B worker SeedGraph must be a JSON object.")
    return value


def _worker_send(
    connection: Any,
    *,
    ok: bool,
    value: Any = None,
    error: str | None = None,
) -> None:
    try:
        payload: dict[str, Any] = {"ok": bool(ok)}
        if ok:
            payload["value"] = value
        else:
            payload["error"] = error or "unknown worker error"
        connection.send(payload)
    except (BrokenPipeError, EOFError, OSError):
        pass


def _worker_error(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"


def _capture_ab_initial_frame(env: gymnasium.Env) -> None:
    """Append one audience-camera frame without advancing simulation state."""
    base = env.unwrapped
    manager = getattr(base, "event_manager", None)
    mode_cfgs = getattr(manager, "_mode_functor_cfgs", {})
    candidates: list[tuple[Any, dict[str, Any]]] = []
    for configured in mode_cfgs.values():
        for functor_cfg in configured:
            functor = _config_member(functor_cfg, "func")
            class_name = getattr(type(functor), "__name__", "")
            if not callable(functor) or class_name not in {
                "record_camera_data",
                "record_camera_data_async",
            }:
                continue
            params = _config_member(functor_cfg, "params") or {}
            if not isinstance(params, Mapping):
                raise ValueError("A/B record_camera params must be a mapping.")
            params = dict(params)
            if params.get("name") == "record_cam_audience_view":
                candidates.insert(0, (functor, params))
            else:
                candidates.append((functor, params))
    if candidates:
        # Prefer the explicitly generated audience recorder.  A single
        # unnamed legacy recorder remains a compatible fallback; selecting
        # among multiple non-audience recorders would silently produce the
        # wrong camera view, so fail instead.
        if len(candidates) > 1 and candidates[0][1].get("name") != (
            "record_cam_audience_view"
        ):
            raise RuntimeError(
                "A/B environment has multiple camera recorders but none is "
                "named 'record_cam_audience_view'."
            )
        functor, params = candidates[0]
        functor(base, None, **params)
        return
    raise RuntimeError(
        "A/B environment must define a record_camera_data audience recorder."
    )


def _execution_result_to_wire(result: Any) -> dict[str, Any]:
    """Strip simulator-owned state from an execution result before IPC."""

    actions = [_wire_tensor(action) for action in list(getattr(result, "actions", ()))]
    success = _wire_tensor(getattr(result, "success", False))
    return {
        "actions": actions,
        "success": success,
        "record_dir": getattr(result, "record_dir", None),
        "already_executed": bool(getattr(result, "already_executed", True)),
        "retry_count": int(getattr(result, "retry_count", 0)),
        "recovery_count": int(getattr(result, "recovery_count", 0)),
        "revision_count": int(getattr(result, "revision_count", 0)),
        "failure_events": list(getattr(result, "failure_events", ())),
        "runtime_revisions": list(getattr(result, "runtime_revisions", ())),
    }


def _wire_tensor(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    return value


def _execution_result_from_wire(value: dict[str, Any]) -> SimpleNamespace:
    required = {
        "actions",
        "success",
        "retry_count",
        "recovery_count",
        "revision_count",
        "failure_events",
        "runtime_revisions",
    }
    missing = sorted(required - set(value))
    if missing:
        raise RuntimeError(f"A/B worker result is missing fields: {missing}.")
    return SimpleNamespace(**value)


def _run_ab(
    args: argparse.Namespace,
    *,
    env_cfg: Any,
    gym_config: dict[str, Any],
    agent_config: dict[str, Any],
) -> None:
    """Plan and execute strict offline/online branches for every episode."""
    from embodichain.gen_sim.action_engine.evaluation import run_strict_ab, state_digest
    from embodichain.gen_sim.action_engine.planning import (
        plan_candidates_parallel,
        plan_online_seed_graph,
    )
    from embodichain.gen_sim.action_engine.generation import VLM_CAMERA_UIDS

    config_path = Path(args.agent_config).expanduser().resolve()
    task_path = _resolve_artifact_path(
        agent_config,
        config_path,
        "task_spec",
        "task_spec_path",
    )
    task_spec = _read_json(task_path, "TaskSpec")
    reference_program = load_agent_execution_program(
        agent_config,
        agent_config_path=config_path,
        regenerate=bool(getattr(args, "regenerate", False)),
        require_executable=False,
    )
    if reference_program.seed_graph is None:
        raise ValueError("A/B execution requires an immutable offline SeedGraph.")
    reference_graph = reference_program.seed_graph
    source = agent_config.get("source")
    if not isinstance(source, dict):
        source = {}
    uid_map = source.get("uid_map", {})
    if not isinstance(uid_map, dict):
        raise ValueError("agent_config.source.uid_map must be a mapping when provided.")
    known_objects = {str(uid) for uid in uid_map.values() if str(uid)}
    online_config = agent_config.get("online_planning", {})
    if online_config is None:
        online_config = {}
    if not isinstance(online_config, dict):
        raise ValueError("agent_config.online_planning must be a mapping.")
    camera_uids = online_config.get("camera_uids") or agent_config.get(
        "vlm_camera_uids", []
    )
    if camera_uids != list(VLM_CAMERA_UIDS):
        raise ValueError(
            "A/B execution requires the canonical VLM cameras "
            f"{list(VLM_CAMERA_UIDS)}."
        )
    vlm_model = (
        getattr(args, "vlm_model", None)
        or online_config.get("vlm_model")
        or agent_config.get("vlm_model")
    )
    robot_profile = str(agent_config.get("robot_profile", "dual_ur10"))
    base_seed = 0 if args.seed is None else int(args.seed)
    if args.seed is None:
        set_seed(base_seed)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output_root = config_path.parent / "ab_runs" / run_id
    episodes = int(gym_config.get("max_episodes", _DEFAULT_MAX_EPISODES))
    summaries = []
    env_options = _ab_env_options(env_cfg)

    for episode_index in range(episodes):
        episode_seed = base_seed + episode_index
        episode_root = output_root / f"episode_{episode_index:04d}"
        branch_envs: dict[str, Any] = {}
        ownership_transferred = False
        try:
            # The online VLM observes the same reset that its branch executes.
            # Each branch owns one simulator process because DexSim entities
            # resolve through a process-global default world.
            worker_gym_config = _ab_runtime_gym_config(gym_config, env_cfg)
            worker_configs = {
                route: _ABWorkerConfig(
                    route=route,
                    gym_config=worker_gym_config,
                    env_options=deepcopy(env_options),
                    gym_id=str(gym_config["id"]),
                    agent_config=agent_config,
                    agent_config_path=config_path.as_posix(),
                    task_name=args.task_name,
                    runtime_backend=getattr(args, "runtime_backend", "independent"),
                    seed=episode_seed,
                    camera_uids=tuple(str(uid) for uid in camera_uids),
                    staging_dir=(episode_root / ".work" / route / "video").as_posix(),
                )
                for route in ("offline", "online")
            }
            branch_envs, snapshots = _prepare_ab_branches(
                worker_configs,
                gpu_id=getattr(getattr(env_cfg, "sim_cfg", None), "gpu_id", None),
            )
            planning_digest = state_digest(snapshots["offline"])
            if planning_digest != state_digest(snapshots["online"]):
                raise RuntimeError(
                    "Strict A/B initial state mismatch before planning: "
                    f"offline={planning_digest}, "
                    f"online={state_digest(snapshots['online'])}."
                )
            observation = branch_envs["online"].startup_observation
            if observation is None:
                raise RuntimeError(
                    "Online A/B worker did not return scene observation."
                )
            visual_facts: dict[str, Any] = {}

            def offline_planner(*, task_spec: dict[str, Any]) -> dict[str, Any]:
                # Generation has already materialized the fixed recipe from the
                # shared TaskSpec. Reuse that immutable artifact verbatim so A/B
                # also supports legacy v2 bundles whose graph metadata predates
                # ``role_bindings``.
                del task_spec
                return deepcopy(reference_graph)

            def online_planner(*, task_spec: dict[str, Any]) -> dict[str, Any]:
                graph, facts = plan_online_seed_graph(
                    task_spec,
                    observation,
                    vlm_model=vlm_model,
                    robot_profile=robot_profile,
                )
                visual_facts.update(facts)
                return graph

            candidates = plan_candidates_parallel(
                task_spec,
                offline_planner=offline_planner,
                online_planner=online_planner,
                known_objects=known_objects or None,
                robot_profile=robot_profile,
            )
            if candidates.offline != reference_graph:
                from embodichain.gen_sim.action_engine.domain import seed_graph_hash

                if seed_graph_hash(candidates.offline) != seed_graph_hash(
                    reference_graph
                ):
                    raise RuntimeError(
                        "A/B offline recipe no longer matches the generated "
                        "reference graph."
                    )

            # Visual evidence is an online-planning artifact, not an execution
            # artifact. Persist it as soon as both candidates have passed their
            # static checks so it remains auditable even if a later preflight,
            # runtime action, or video flush fails.
            _write_json(episode_root / "online" / "visual_facts.json", visual_facts)

            # Rendering and external planning must be side-effect free.  Take
            # a second full snapshot immediately before executor preflight so
            # an accidental simulation advance cannot be hidden behind the
            # reset-time digest used for visual planning.
            snapshots = {
                route: worker.snapshot() for route, worker in branch_envs.items()
            }
            execution_digests = {
                route: state_digest(snapshot) for route, snapshot in snapshots.items()
            }
            if (
                execution_digests["offline"] != planning_digest
                or execution_digests["online"] != planning_digest
            ):
                raise RuntimeError(
                    "Strict A/B initial state changed during visual planning: "
                    f"offline={execution_digests['offline']}, "
                    f"online={execution_digests['online']}, "
                    f"expected={planning_digest}."
                )

            def executor_factory(graph: dict[str, Any], worker: Any) -> Any:
                route = worker.action_engine_ab_route
                if route not in branch_envs:
                    raise ValueError(f"Unknown A/B worker route {route!r}.")
                return _RemoteBranchExecutor(
                    graph,
                    worker,
                    record_root=episode_root / ".work" / route / "runtime",
                )

            def branch_finalizer(**kwargs: Any) -> list[str]:
                worker = kwargs["env"]
                branch_dir = Path(kwargs["branch_dir"])
                return worker.finalize(
                    branch_dir,
                    episode_index=int(kwargs.get("episode_index", episode_index)),
                )

            result = run_strict_ab(
                task_spec,
                candidates.offline,
                candidates.online,
                executor_factory=executor_factory,
                snapshot_reader=lambda env: _snapshot_environment(env, camera_uids),
                output_dir=episode_root,
                seed=episode_seed,
                shared_config={
                    "robot_profile": robot_profile,
                    "camera_uids": camera_uids,
                    "vlm_model": vlm_model,
                    "strict_state_digest": True,
                },
                planning_metrics=candidates.planning_metrics,
                known_objects=known_objects or None,
                expected_initial_state_digest=planning_digest,
                branch_finalizer=branch_finalizer,
                episode_index=episode_index,
                strict_state_digest=True,
                prepared_environments=branch_envs,
                prepared_snapshots=snapshots,
                require_branch_videos=True,
            )
            # run_strict_ab owns and closes prepared workers on both its normal
            # and exceptional execution paths.  Do not claim ownership until
            # it has entered/returned from that cleanup boundary; this also
            # closes workers when graph validation fails before its try/finally.
            ownership_transferred = True
        finally:
            if not ownership_transferred:
                for worker in branch_envs.values():
                    worker.close()
        summaries.append(
            {
                "episode_index": episode_index,
                "seed": episode_seed,
                "comparison": result.comparison_path.as_posix(),
                "initial_state_digest": result.initial_state_digest,
            }
        )
        log_info(
            "Action Engine A/B episode "
            f"{episode_index}: offline="
            f"{result.comparison['branches']['offline']['success_rate']:.3f}, "
            f"online={result.comparison['branches']['online']['success_rate']:.3f}.",
            color="green",
        )

    summary_path = output_root / "run_summary.json"
    _write_json(
        summary_path,
        {
            "schema_version": "action_engine_ab_run_v1",
            "task_id": args.task_name,
            "run_id": run_id,
            "episodes": summaries,
        },
    )
    log_info(f"A/B comparison artifacts: {output_root}", color="green")


def _configure_ab_branch_cfg(
    env_cfg: Any,
    *,
    staging_dir: Path,
    dataset_dir: Path,
) -> None:
    """Give one worker exclusive recorder paths before env construction."""
    staging_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    events = _config_member(env_cfg, "events")
    recorder = _config_member(events, "record_camera")
    if recorder is None:
        raise ValueError("A/B environment config must define record_camera.")
    _set_config_param(recorder, "save_path", staging_dir.as_posix())

    # Dataset output is not part of the A/B contract, but leaving the
    # generated path shared would still let two workers overwrite each other.
    dataset = _config_member(env_cfg, "dataset")
    if dataset is not None:
        for name in ("lerobot", "record", "dataset"):
            term = _config_member(dataset, name)
            if term is not None:
                _set_config_param(term, "save_path", dataset_dir.as_posix())


def _ab_runtime_gym_config(
    gym_config: Mapping[str, Any], env_cfg: Any
) -> dict[str, Any]:
    """Carry launcher-resolved simulation settings into spawned workers."""
    result = deepcopy(dict(gym_config))
    sim_cfg = getattr(env_cfg, "sim_cfg", None)
    if sim_cfg is None:
        return result
    result.update(
        {
            "device": str(getattr(sim_cfg, "sim_device", "cpu")),
            "gpu_id": int(getattr(sim_cfg, "gpu_id", 0)),
            "headless": bool(getattr(sim_cfg, "headless", False)),
            "arena_space": float(getattr(sim_cfg, "arena_space", 5.0)),
            "num_envs": int(getattr(sim_cfg, "num_envs", result.get("num_envs", 1))),
        }
    )
    render_cfg = getattr(sim_cfg, "render_cfg", None)
    renderer = getattr(render_cfg, "renderer", None)
    if renderer is not None:
        result["renderer"] = str(renderer)
    return result


def _ab_env_options(env_cfg: Any) -> dict[str, Any]:
    """Extract the non-JSON flags applied after gym config parsing."""
    profiler = getattr(env_cfg, "profiler", None)
    return {
        "filter_visual_rand": bool(getattr(env_cfg, "filter_visual_rand", False)),
        "filter_dataset_saving": bool(getattr(env_cfg, "filter_dataset_saving", False)),
        "record_trajectory": bool(getattr(env_cfg, "record_trajectory", False)),
        "trajectory_save_dir": getattr(env_cfg, "trajectory_save_dir", None),
        "profile": bool(getattr(profiler, "enable_time", False)),
        "profile_output": getattr(profiler, "output_path", None),
    }


def _apply_ab_env_options(env_cfg: Any, options: Mapping[str, Any]) -> None:
    """Apply launcher flags after reconstructing a worker's config."""
    env_cfg.filter_visual_rand = bool(options.get("filter_visual_rand", False))
    env_cfg.filter_dataset_saving = bool(options.get("filter_dataset_saving", False))
    env_cfg.record_trajectory = bool(options.get("record_trajectory", False))
    trajectory_dir = options.get("trajectory_save_dir")
    if trajectory_dir:
        env_cfg.trajectory_save_dir = str(trajectory_dir)
    if bool(options.get("profile", False)):
        from embodichain.lab.gym.utils.profiler import EnvProfilerCfg

        env_cfg.profiler = EnvProfilerCfg(
            enable_time=True,
            output_path=options.get("profile_output"),
        )


def _config_member(value: Any, name: str) -> Any:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None) if value is not None else None


def _set_config_param(term: Any, name: str, value: Any) -> None:
    params = _config_member(term, "params")
    if params is None:
        params = {}
        if isinstance(term, dict):
            term["params"] = params
        else:
            setattr(term, "params", params)
    if not isinstance(params, dict):
        raise ValueError(
            f"A/B recorder params must be a mapping, got {type(params)!r}."
        )
    params[name] = value


def _finalize_ab_branch_video(
    env: gymnasium.Env,
    *,
    staging_dir: Path,
    branch_dir: Path,
) -> list[str]:
    """Flush the final episode and publish exactly this worker's video."""
    before = {
        path: (path.stat().st_mtime_ns, path.stat().st_size)
        for path in staging_dir.glob("episode_*_record_cam_audience_view.mp4")
        if path.is_file()
    }
    env.reset(options={"final": True})
    video = _publish_branch_video(staging_dir, branch_dir, before=before)
    return [video.as_posix()]


def _snapshot_environment(
    env: gymnasium.Env,
    camera_uids: list[str],
) -> dict[str, Any]:
    base = env.unwrapped
    sim = base.sim
    object_poses = {}
    for uid in sim.get_rigid_object_uid_list():
        entity = sim.get_rigid_object(uid)
        if entity is not None:
            object_poses[str(uid)] = _snapshot_tensor(
                entity.get_local_pose(to_matrix=True)
            )
    articulation_state = {}
    for uid in getattr(sim, "get_articulation_uid_list", lambda: [])():
        entity = sim.get_articulation(uid)
        if entity is None:
            continue
        articulation_state[str(uid)] = {
            "pose": _snapshot_tensor(entity.get_local_pose(to_matrix=True)),
            "qpos": _snapshot_tensor(entity.get_qpos()),
        }
    camera_calibration = {}
    available_sensor_uids = getattr(sim, "get_sensor_uid_list", lambda: [])()
    snapshot_camera_uids = sorted(
        {str(uid) for uid in [*camera_uids, *available_sensor_uids] if str(uid)}
    )
    for uid in snapshot_camera_uids:
        sensor = sim.get_sensor(uid)
        if sensor is None:
            raise ValueError(f"A/B snapshot cannot find camera {uid!r}.")
        camera_calibration[uid] = {
            "intrinsics": _snapshot_sensor_value(sensor, "get_intrinsics"),
            "extrinsics": _snapshot_sensor_value(
                sensor, "get_arena_pose", to_matrix=True
            ),
        }
    return {
        "robot_qpos": _snapshot_tensor(base.robot.get_qpos()),
        "object_poses": object_poses,
        "articulation_state": articulation_state,
        "camera_calibration": camera_calibration,
    }


def _snapshot_tensor(value: Any) -> torch.Tensor:
    """Normalize simulator values for deterministic digesting."""
    tensor = torch.as_tensor(value)
    return tensor.detach().cpu().contiguous()


def _snapshot_sensor_value(
    sensor: Any,
    method_name: str,
    **kwargs: Any,
) -> torch.Tensor:
    method = getattr(sensor, method_name, None)
    if not callable(method):
        raise ValueError(f"A/B snapshot sensor lacks {method_name}().")
    try:
        value = method(**kwargs)
    except TypeError:
        value = method()
    return _snapshot_tensor(value)


def _publish_branch_video(
    staging_dir: Path,
    branch_dir: Path,
    *,
    before: dict[Path, tuple[int, int]] | None = None,
) -> Path:
    candidates = sorted(
        (
            path
            for path in staging_dir.glob("episode_*_record_cam_audience_view.mp4")
            if path.is_file()
            and (
                before is None
                or path not in before
                or (path.stat().st_mtime_ns, path.stat().st_size) != before[path]
            )
        ),
        key=lambda path: path.stat().st_mtime_ns,
    )
    if not candidates:
        raise RuntimeError(f"No completed A/B audience video found in {staging_dir}.")
    source = candidates[-1]
    destination = branch_dir / "video.mp4"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if destination.stat().st_size == 0:
        raise RuntimeError(f"A/B audience video is empty: {destination}.")
    return destination


def _resolve_artifact_path(
    config: dict[str, Any],
    config_path: Path,
    *keys: str,
) -> Path:
    for key in keys:
        value = config.get(key)
        if value is None:
            continue
        if not isinstance(value, str) or not value:
            raise ValueError(f"agent_config.{key} must be a non-empty path string.")
        path = Path(value).expanduser()
        return (
            path.resolve()
            if path.is_absolute()
            else (config_path.parent / path).resolve()
        )
    joined = " or ".join(f"agent_config.{key}" for key in keys)
    raise ValueError(f"A/B execution requires {joined}.")


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read {label} at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object.")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _show_physical_collision(env: gymnasium.Env) -> None:
    """Enable physical-shape visualization for all supported scene assets."""
    sim = env.get_wrapper_attr("sim")
    uids: list[str] = []
    for getter_name in (
        "get_rigid_object_uid_list",
        "get_rigid_object_group_uid_list",
        "get_articulation_uid_list",
    ):
        getter = getattr(sim, getter_name, None)
        if callable(getter):
            uids.extend(getter())
    visible = 0
    for uid in uids:
        asset = sim.get_asset(uid)
        if asset is None or not hasattr(asset, "set_physical_visible"):
            continue
        try:
            asset.set_physical_visible(
                visible=True,
                rgba=[1.0, 0.15, 0.1, 0.35],
            )
            visible += 1
        except Exception as exc:
            log_warning(f"Unable to show collision geometry for {uid!r}: {exc}")
    log_info(f"Physical collision geometry visible for {visible} assets.")


if __name__ == "__main__":
    raise SystemExit(cli())
