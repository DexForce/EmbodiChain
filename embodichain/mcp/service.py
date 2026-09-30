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

"""Provider-neutral MCP service facade for EmbodiChain capabilities."""

from __future__ import annotations

import json
import os
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
from time import time
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlsplit

from .backend import InMemorySimulationBackend, SimulationBackend

__all__ = ["EmbodiChainMCPService"]


TaskProvider = Callable[[], Sequence[Mapping[str, Any]]]


@dataclass
class _Run:
    run_id: str
    world_id: str
    steps: int
    status: str = "queued"
    metrics: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event)
    future: Future[dict[str, Any]] | None = None
    created_at: float = field(default_factory=time)


def _default_task_provider() -> Sequence[Mapping[str, Any]]:
    """Read the installed task catalog without requiring a simulator startup."""
    try:
        from embodichain.cli._task_catalog import _discover_catalog, _local_path

        tasks = _discover_catalog()
    except (ImportError, OSError, RuntimeError, TypeError, ValueError):
        return ()
    result: list[Mapping[str, Any]] = []
    for task in tasks:
        deployments = []
        for deployment in task.deployments:
            deployments.append(
                {
                    "name": deployment.name,
                    "env_id": deployment.env_id,
                    "config_ref": deployment.config_ref,
                    "config_path": (
                        str(path)
                        if (path := _local_path(deployment.resource))
                        else None
                    ),
                    "physics": deployment.physics,
                    "embodiments": list(deployment.embodiments),
                    "capabilities": sorted(deployment.capabilities),
                }
            )
        result.append(
            {
                "task_id": task.qualified_key,
                "title": task.title,
                "summary": task.summary,
                "task_path": list(task.task_path),
                "tags": list(task.tags),
                "default_deployment": task.default_deployment,
                "deployments": deployments,
            }
        )
    return result


class EmbodiChainMCPService:
    """Expose validated, coarse-grained EmbodiChain operations to an MCP server.

    Args:
        backend: Simulator adapter. Defaults to a deterministic CPU backend so
            the protocol can be smoke-tested without starting DexSim.
        task_provider: Callable returning JSON-compatible task records. The
            default reads the installed EmbodiChain task catalog.
        max_workers: Maximum number of concurrent rollout jobs.
    """

    protocol_version = "phase1"

    def __init__(
        self,
        backend: SimulationBackend | None = None,
        *,
        task_provider: TaskProvider | None = None,
        max_workers: int = 2,
    ) -> None:
        if type(max_workers) is not int or max_workers < 1:
            raise ValueError("max_workers must be a positive integer")
        self.backend = backend or InMemorySimulationBackend()
        self._task_provider = task_provider or _default_task_provider
        self._runs: dict[str, _Run] = {}
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="embodichain-mcp-rollout",
        )

    def close(self) -> None:
        """Cancel pending rollouts and release the rollout executor."""
        with self._lock:
            for run in self._runs.values():
                run.cancel_event.set()
        self._executor.shutdown(wait=True, cancel_futures=True)

    def server_info(self) -> dict[str, Any]:
        """Return protocol and backend metadata."""
        return {
            "name": "embodichain-mcp",
            "phase": self.protocol_version,
            "backend": getattr(self.backend, "name", type(self.backend).__name__),
            "transports": ["stdio", "streamable-http"],
            "scope": "simulation_and_read_only_analysis",
        }

    def capabilities(self) -> list[str]:
        """Return the first-phase operation names exposed by the service."""
        return [
            "list_robots",
            "get_robot_info",
            "list_tasks",
            "get_task_info",
            "list_physics_backends",
            "create_world",
            "load_task_or_scene",
            "reset_world",
            "destroy_world",
            "get_world_state",
            "snapshot_world",
            "restore_world",
            "step_simulation",
            "forward_kinematics",
            "solve_ik",
            "check_reachability",
            "check_collision",
            "plan_motion",
            "validate_trajectory",
            "start_rollout",
            "get_run_status",
            "get_run_metrics",
            "cancel_run",
            "compare_runs",
        ]

    def list_robots(self) -> list[dict[str, Any]]:
        """List robots available from the injected backend."""
        return deepcopy(self.backend.list_robots())

    def get_robot_info(self, robot_id: str) -> dict[str, Any]:
        """Return one robot record.

        Args:
            robot_id: Backend-specific robot identifier.

        Returns:
            The matching robot metadata.

        Raises:
            ValueError: If the robot is not available.
        """
        for robot in self.list_robots():
            if robot.get("robot_id") == robot_id:
                return robot
        raise ValueError(f"Unknown robot_id: {robot_id}")

    def list_tasks(self) -> list[dict[str, Any]]:
        """List task catalog records without starting a simulator."""
        return [dict(task) for task in deepcopy(list(self._task_provider()))]

    def get_task_info(self, task_id: str) -> dict[str, Any]:
        """Return one task catalog record."""
        for task in self.list_tasks():
            if task.get("task_id") == task_id:
                return task
        raise ValueError(f"Unknown task_id: {task_id}")

    def list_physics_backends(self) -> list[dict[str, Any]]:
        """List physics backends available to the injected backend."""
        return deepcopy(self.backend.list_physics_backends())

    def create_world(
        self, *, backend: str | None = None, seed: int | None = None
    ) -> dict[str, Any]:
        """Create a world handle with explicit backend and seed metadata."""
        selected_backend = backend or getattr(
            self.backend,
            "default_backend",
            getattr(self.backend, "name", "default"),
        )
        state = self.backend.create_world(backend=selected_backend, seed=seed)
        return self._envelope(state, world_id=state["world_id"])

    def load_task_or_scene(
        self,
        world_id: str,
        *,
        task_id: str | None = None,
        scene: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Load an existing task reference or JSON-compatible scene manifest."""
        if task_id is not None:
            task = self.get_task_info(task_id)
            if scene is None:
                scene = self._load_default_task_scene(task)
        state = self.backend.load_scene(
            world_id,
            task_id=task_id,
            scene=dict(scene) if scene is not None else None,
        )
        return self._envelope(state, world_id=world_id)

    @staticmethod
    def _load_default_task_scene(task: Mapping[str, Any]) -> dict[str, Any] | None:
        """Load a local default deployment config when the catalog exposes one."""
        deployments = task.get("deployments", ())
        if not isinstance(deployments, Sequence) or isinstance(
            deployments, (str, bytes)
        ):
            return None
        selected = task.get("default_deployment")
        records = [record for record in deployments if isinstance(record, Mapping)]
        record = next(
            (item for item in records if item.get("name") == selected),
            records[0] if records else None,
        )
        config_ref = record.get("config_path") if record else None
        if not isinstance(config_ref, str) or not os.path.isfile(config_ref):
            return None
        with open(config_ref, encoding="utf-8") as config_file:
            if config_ref.lower().endswith(".json"):
                value = json.load(config_file)
            else:
                import yaml

                value = yaml.safe_load(config_file)
        if not isinstance(value, Mapping):
            raise ValueError(f"Task config must contain a mapping: {config_ref}")
        return dict(value)

    def reset_world(self, world_id: str) -> dict[str, Any]:
        """Reset dynamic world state."""
        return self._envelope(self.backend.reset_world(world_id), world_id=world_id)

    def destroy_world(self, world_id: str) -> dict[str, Any]:
        """Destroy a world handle."""
        with self._lock:
            active = [
                run.run_id
                for run in self._runs.values()
                if run.world_id == world_id and run.status in {"queued", "running"}
            ]
        if active:
            raise RuntimeError(
                "Cannot destroy a world with active rollout jobs: " + ", ".join(active)
            )
        self.backend.destroy_world(world_id)
        return {"status": "succeeded", "world_id": world_id}

    def get_world_state(self, world_id: str) -> dict[str, Any]:
        """Return a detached world state with protocol metadata."""
        return self._envelope(self.backend.get_world_state(world_id), world_id=world_id)

    def snapshot_world(self, world_id: str) -> dict[str, Any]:
        """Create a snapshot handle for a world."""
        return self._envelope(self.backend.snapshot_world(world_id), world_id=world_id)

    def restore_world(self, world_id: str, snapshot_id: str) -> dict[str, Any]:
        """Restore a snapshot belonging to the requested world."""
        return self._envelope(
            self.backend.restore_world(world_id, snapshot_id), world_id=world_id
        )

    def step_simulation(self, world_id: str, *, steps: int = 1) -> dict[str, Any]:
        """Advance a world by a positive number of backend steps."""
        return self._envelope(
            self.backend.step_simulation(world_id, steps=steps), world_id=world_id
        )

    def forward_kinematics(
        self, robot_id: str, qpos: Sequence[float]
    ) -> dict[str, Any]:
        """Compute FK through the backend adapter."""
        return self._envelope(self.backend.forward_kinematics(robot_id, qpos))

    def solve_ik(self, robot_id: str, target_pose: Mapping[str, Any]) -> dict[str, Any]:
        """Compute IK through the backend adapter."""
        return self._envelope(self.backend.solve_ik(robot_id, dict(target_pose)))

    def check_reachability(
        self, robot_id: str, target_pose: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Check reachability without implying collision-free execution."""
        return self._envelope(
            self.backend.check_reachability(robot_id, dict(target_pose))
        )

    def check_collision(
        self, world_id: str, robot_id: str, qpos: Sequence[float]
    ) -> dict[str, Any]:
        """Run backend collision validation for a robot configuration."""
        return self._envelope(
            self.backend.check_collision(world_id, robot_id, qpos), world_id=world_id
        )

    def plan_motion(
        self,
        robot_id: str,
        start_qpos: Sequence[float],
        goal_qpos: Sequence[float],
        *,
        samples: int = 32,
    ) -> dict[str, Any]:
        """Generate a backend trajectory between two joint configurations."""
        return self._envelope(
            self.backend.plan_motion(robot_id, start_qpos, goal_qpos, samples=samples)
        )

    def validate_trajectory(
        self,
        world_id: str,
        robot_id: str,
        positions: Sequence[Sequence[float]],
    ) -> dict[str, Any]:
        """Validate a trajectory through the backend adapter."""
        return self._envelope(
            self.backend.validate_trajectory(world_id, robot_id, positions),
            world_id=world_id,
        )

    def start_rollout(self, world_id: str, *, steps: int = 1) -> dict[str, Any]:
        """Start a bounded asynchronous rollout job."""
        if type(steps) is not int or not 1 <= steps <= 1_000_000:
            raise ValueError("steps must be an integer between 1 and 1000000")
        self.backend.get_world_state(world_id)
        run = _Run(
            run_id=f"run-{uuid.uuid4().hex[:12]}", world_id=world_id, steps=steps
        )
        with self._lock:
            self._runs[run.run_id] = run
            run.future = self._executor.submit(self._execute_rollout, run)
        return self._run_status(run)

    def _execute_rollout(self, run: _Run) -> dict[str, Any]:
        with self._lock:
            run.status = "running"
        try:
            metrics = self.backend.run_rollout(
                run.world_id, steps=run.steps, cancel_event=run.cancel_event
            )
            with self._lock:
                run.metrics = deepcopy(metrics)
                run.status = str(metrics.get("status", "succeeded"))
        except Exception as error:  # noqa: BLE001 - preserve job failure details.
            with self._lock:
                run.status = "failed"
                run.error = f"{type(error).__name__}: {error}"
        return self._run_status(run)

    def get_run_status(self, run_id: str) -> dict[str, Any]:
        """Return status and metadata for a rollout job."""
        return self._run_status(self._get_run(run_id))

    def get_run_metrics(self, run_id: str) -> dict[str, Any]:
        """Return metrics for a completed or partially completed rollout."""
        run = self._get_run(run_id)
        with self._lock:
            return {"run_id": run_id, "status": run.status, **deepcopy(run.metrics)}

    def cancel_run(self, run_id: str) -> dict[str, Any]:
        """Request cancellation of a rollout job."""
        run = self._get_run(run_id)
        run.cancel_event.set()
        with self._lock:
            if run.status == "queued":
                run.status = "cancelled"
        return self._run_status(run)

    def compare_runs(self, run_ids: Sequence[str]) -> dict[str, Any]:
        """Compare numeric metrics from two or more rollout jobs."""
        if len(run_ids) < 2:
            raise ValueError("run_ids must contain at least two runs")
        metrics = [self.get_run_metrics(run_id) for run_id in run_ids]
        return {"run_ids": list(run_ids), "runs": metrics}

    def read_resource(self, uri: str) -> str:
        """Read a JSON resource exposed by the service.

        Args:
            uri: One of the ``embodichain://`` resource templates.

        Returns:
            A JSON document suitable for MCP ``resources/read``.

        Raises:
            ValueError: If the URI is unknown or malformed.
        """
        parsed = urlsplit(uri)
        if parsed.scheme != "embodichain":
            raise ValueError("Resource URI must use the embodichain scheme")
        parts = [part for part in parsed.path.split("/") if part]
        if parsed.netloc == "tasks" and parts == ["catalog"]:
            value: Any = self.list_tasks()
        elif parsed.netloc == "robots" and len(parts) == 2 and parts[1] == "config":
            value = self.get_robot_info(parts[0])
        elif parsed.netloc == "worlds" and len(parts) == 2 and parts[1] == "manifest":
            value = self.get_world_state(parts[0])
        elif parsed.netloc == "runs" and len(parts) == 2 and parts[1] == "status":
            value = self.get_run_status(parts[0])
        elif parsed.netloc == "runs" and len(parts) == 2 and parts[1] == "metrics":
            value = self.get_run_metrics(parts[0])
        else:
            raise ValueError(f"Unknown resource URI: {uri}")
        return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)

    def _get_run(self, run_id: str) -> _Run:
        with self._lock:
            try:
                return self._runs[run_id]
            except KeyError as exc:
                raise ValueError(f"Unknown run_id: {run_id}") from exc

    def _run_status(self, run: _Run) -> dict[str, Any]:
        with self._lock:
            result = {
                "run_id": run.run_id,
                "world_id": run.world_id,
                "steps": run.steps,
                "status": run.status,
                "created_at": run.created_at,
            }
            if run.error is not None:
                result["error"] = run.error
            if run.metrics:
                result["metrics"] = deepcopy(run.metrics)
            return result

    def _envelope(
        self, value: Mapping[str, Any], *, world_id: str | None = None
    ) -> dict[str, Any]:
        """Add common result metadata to a backend response."""
        result = {
            "status": "succeeded",
            "result": deepcopy(dict(value)),
            "backend": value.get(
                "backend", getattr(self.backend, "name", type(self.backend).__name__)
            ),
            "frame": "arena",
            "units": {"length": "m", "angle": "rad", "time": "s"},
            "diagnostics": [],
            "artifacts": [],
        }
        if world_id is not None:
            result["world_id"] = world_id
            result["scene_revision"] = value.get("scene_revision")
            result["seed"] = value.get("seed")
        return result
