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
from contextlib import contextmanager
from time import time
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlsplit

from typing_extensions import TypedDict

from .backend import (
    MAX_SIMULATION_STEPS,
    MAX_TRAJECTORY_SAMPLES,
    InMemorySimulationBackend,
    SimulationBackend,
)

__all__ = ["EmbodiChainMCPService"]


class ToolEnvelope(TypedDict, total=False):
    """Common structured result schema published by MCP tools."""

    status: str
    result: dict[str, Any]
    backend: str
    frame: str
    units: dict[str, str]
    diagnostics: list[dict[str, Any]]
    artifacts: list[dict[str, Any]]
    world_id: str
    scene_revision: int | None
    seed: int | None


class ServerInfoResult(TypedDict):
    """Schema for the server information tool."""

    name: str
    phase: str
    backend: str
    transports: list[str]
    scope: str


class HealthResult(TypedDict):
    """Schema for the health probe tool."""

    status: str
    server: str
    phase: str
    backend: str
    active_rollouts: int
    known_runs: int
    scope: str


TaskProvider = Callable[[], Sequence[Mapping[str, Any]]]
MAX_WAYPOINTS = 256
MAX_ACTIVE_RUNS_FACTOR = 4


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
                    "_config_resource": deployment.resource,
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
        self._trajectories: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._world_locks: dict[str, threading.RLock] = {}
        self._max_active_runs = max_workers * MAX_ACTIVE_RUNS_FACTOR
        self._closed = False
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="embodichain-mcp-rollout",
        )

    def close(self) -> None:
        """Cancel pending rollouts and release the rollout executor."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for run in self._runs.values():
                run.cancel_event.set()
        self._executor.shutdown(wait=True, cancel_futures=True)
        close_backend = getattr(self.backend, "close", None)
        if close_backend is not None:
            close_backend()

    def _world_lock(self, world_id: str) -> threading.RLock:
        """Return the service serialization lock for a world handle."""
        with self._lock:
            return self._world_locks.setdefault(world_id, threading.RLock())

    @contextmanager
    def _world_guard(self, world_id: str, expected_scene_revision: int | None = None):
        """Serialize a complete revision-checked operation for one world."""
        with self._world_lock(world_id):
            state = self._check_scene_revision(world_id, expected_scene_revision)
            yield state

    def server_info(self) -> ServerInfoResult:
        """Return protocol and backend metadata."""
        return {
            "name": "embodichain-mcp",
            "phase": self.protocol_version,
            "backend": getattr(self.backend, "name", type(self.backend).__name__),
            "transports": ["stdio"],
            "scope": "simulation_and_stateful_analysis",
        }

    def health(self) -> HealthResult:
        """Return a lightweight service health snapshot.

        The probe does not start a simulator or execute a rollout. It only
        reports service-owned state and the injected backend identity so an
        MCP Host can distinguish a live server from an unavailable backend.
        """
        with self._lock:
            active_rollouts = sum(
                run.status in {"queued", "running"} for run in self._runs.values()
            )
            known_runs = len(self._runs)
        return {
            "status": "ok",
            "server": "embodichain-mcp",
            "phase": self.protocol_version,
            "backend": getattr(self.backend, "name", type(self.backend).__name__),
            "active_rollouts": active_rollouts,
            "known_runs": known_runs,
            "scope": "simulation_and_stateful_analysis",
        }

    def capabilities(self) -> list[str]:
        """Return the first-phase operation names exposed by the service."""
        return [
            "server_info",
            "health",
            "list_robots",
            "get_robot_info",
            "list_tasks",
            "get_task_info",
            "list_physics_backends",
            "list_cameras",
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
            "generate_robot_trajectory",
            "execute_trajectory",
            "record_trajectory",
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

    @staticmethod
    def _public_task_record(task: Mapping[str, Any]) -> dict[str, Any]:
        """Remove internal resource handles from one catalog record."""
        public_task = {
            key: deepcopy(value)
            for key, value in task.items()
            if not key.startswith("_") and key != "deployments"
        }
        public_task["deployments"] = [
            {
                key: deepcopy(value)
                for key, value in deployment.items()
                if not key.startswith("_")
            }
            for deployment in task.get("deployments", ())
            if isinstance(deployment, Mapping)
        ]
        return public_task

    def _task_record(self, task_id: str) -> Mapping[str, Any]:
        """Return the private catalog record used for config resolution."""
        for task in self._task_provider():
            if task.get("task_id") == task_id:
                return task
        raise ValueError(f"Unknown task_id: {task_id}")

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
        records = list(self._task_provider())
        return [self._public_task_record(task) for task in records]

    def get_task_info(self, task_id: str) -> dict[str, Any]:
        """Return one task catalog record."""
        return self._public_task_record(self._task_record(task_id))

    def list_physics_backends(self) -> list[dict[str, Any]]:
        """List physics backends available to the injected backend."""
        return deepcopy(self.backend.list_physics_backends())

    def list_cameras(self, world_id: str) -> list[dict[str, Any]]:
        """List offscreen cameras registered in a World."""
        return deepcopy(self.backend.list_cameras(world_id))

    def create_world(
        self, *, backend: str | None = None, seed: int | None = None
    ) -> ToolEnvelope:
        """Create a world handle with explicit backend and seed metadata."""
        selected_backend = backend or getattr(
            self.backend,
            "default_backend",
            getattr(self.backend, "name", "default"),
        )
        state = self.backend.create_world(backend=selected_backend, seed=seed)
        with self._lock:
            self._world_locks[state["world_id"]] = threading.RLock()
        return self._envelope(state, world_id=state["world_id"])

    def load_task_or_scene(
        self,
        world_id: str,
        *,
        task_id: str | None = None,
        scene: Mapping[str, Any] | None = None,
        expected_scene_revision: int | None = None,
    ) -> ToolEnvelope:
        """Load an existing task reference or JSON-compatible scene manifest."""
        with self._world_guard(world_id, expected_scene_revision) as current:
            if current.get("scene_revision", 0):
                raise RuntimeError(
                    "A world can load only one scene; create a new world to load another."
                )
            if task_id is not None:
                task = self._task_record(task_id)
                if scene is None:
                    scene = self._load_default_task_scene(
                        task, selected_backend=current.get("backend")
                    )
            state = self.backend.load_scene(
                world_id,
                task_id=task_id,
                scene=dict(scene) if scene is not None else None,
            )
            return self._envelope(state, world_id=world_id)

    @staticmethod
    def _load_default_task_scene(
        task: Mapping[str, Any],
        *,
        selected_backend: str | None = None,
    ) -> dict[str, Any] | None:
        """Load and resolve a default deployment through Gym components."""
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
        if record is None:
            return None
        config_ref = record.get("config_path") if record else None
        resource = record.get("_config_resource") if record else None

        def load_config(config_path: str) -> dict[str, Any]:
            with open(config_path, encoding="utf-8") as config_file:
                if config_path.lower().endswith(".json"):
                    value = json.load(config_file)
                else:
                    import yaml

                    value = yaml.safe_load(config_file)
            if not isinstance(value, Mapping):
                raise ValueError(f"Task config must contain a mapping: {config_path}")
            from embodichain.lab.gym.utils._component_composition import (
                _resolve_gym_components,
            )

            resolved = _resolve_gym_components(
                value,
                base_dir=os.path.dirname(config_path),
                selected_backend=selected_backend,
            )
            return dict(resolved.config)

        if isinstance(config_ref, str) and os.path.isfile(config_ref):
            return load_config(config_ref)
        if resource is not None:
            from importlib.resources import as_file

            with as_file(resource.parent) as resource_dir:
                return load_config(str(resource_dir / resource.name))
        raise ValueError(
            "The selected task deployment has no readable config resource; "
            "pass an explicit scene or install the task package."
        )

    def reset_world(
        self, world_id: str, *, expected_scene_revision: int | None = None
    ) -> ToolEnvelope:
        """Reset dynamic world state."""
        with self._world_guard(world_id, expected_scene_revision):
            return self._envelope(self.backend.reset_world(world_id), world_id=world_id)

    def destroy_world(
        self, world_id: str, *, expected_scene_revision: int | None = None
    ) -> ToolEnvelope:
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
        with self._world_guard(world_id, expected_scene_revision):
            with self._lock:
                active = [
                    run.run_id
                    for run in self._runs.values()
                    if run.world_id == world_id and run.status in {"queued", "running"}
                ]
            if active:
                raise RuntimeError(
                    "Cannot destroy a world with active rollout jobs: "
                    + ", ".join(active)
                )
            self.backend.destroy_world(world_id)
            with self._lock:
                for trajectory_id, trajectory in list(self._trajectories.items()):
                    if trajectory["world_id"] == world_id:
                        self._trajectories.pop(trajectory_id)
                self._world_locks.pop(world_id, None)
            return {"status": "succeeded", "world_id": world_id}

    def get_world_state(
        self, world_id: str, *, expected_scene_revision: int | None = None
    ) -> ToolEnvelope:
        """Return a detached world state with protocol metadata."""
        with self._world_guard(world_id, expected_scene_revision):
            return self._envelope(
                self.backend.get_world_state(world_id), world_id=world_id
            )

    def snapshot_world(
        self, world_id: str, *, expected_scene_revision: int | None = None
    ) -> ToolEnvelope:
        """Create a snapshot handle for a world."""
        with self._world_guard(world_id, expected_scene_revision):
            return self._envelope(
                self.backend.snapshot_world(world_id), world_id=world_id
            )

    def restore_world(
        self,
        world_id: str,
        snapshot_id: str,
        *,
        expected_scene_revision: int | None = None,
    ) -> ToolEnvelope:
        """Restore a snapshot belonging to the requested world."""
        with self._world_guard(world_id, expected_scene_revision):
            return self._envelope(
                self.backend.restore_world(world_id, snapshot_id), world_id=world_id
            )

    def step_simulation(
        self,
        world_id: str,
        *,
        steps: int = 1,
        expected_scene_revision: int | None = None,
    ) -> ToolEnvelope:
        """Advance a world by a positive number of backend steps."""
        if type(steps) is not int or not 1 <= steps <= MAX_SIMULATION_STEPS:
            raise ValueError(
                f"steps must be an integer between 1 and {MAX_SIMULATION_STEPS}"
            )
        with self._world_guard(world_id, expected_scene_revision):
            return self._envelope(
                self.backend.step_simulation(world_id, steps=steps), world_id=world_id
            )

    def forward_kinematics(
        self,
        world_id: str,
        robot_id: str,
        qpos: Sequence[float],
        *,
        expected_scene_revision: int | None = None,
    ) -> ToolEnvelope:
        """Compute FK through the backend adapter."""
        with self._world_guard(world_id, expected_scene_revision):
            return self._envelope(
                self.backend.forward_kinematics(world_id, robot_id, qpos),
                world_id=world_id,
            )

    def solve_ik(
        self,
        world_id: str,
        robot_id: str,
        target_pose: Mapping[str, Any],
        *,
        expected_scene_revision: int | None = None,
    ) -> ToolEnvelope:
        """Compute IK through the backend adapter."""
        with self._world_guard(world_id, expected_scene_revision):
            return self._envelope(
                self.backend.solve_ik(world_id, robot_id, dict(target_pose)),
                world_id=world_id,
            )

    def check_reachability(
        self,
        world_id: str,
        robot_id: str,
        target_pose: Mapping[str, Any],
        *,
        expected_scene_revision: int | None = None,
    ) -> ToolEnvelope:
        """Check reachability without implying collision-free execution."""
        with self._world_guard(world_id, expected_scene_revision):
            return self._envelope(
                self.backend.check_reachability(world_id, robot_id, dict(target_pose)),
                world_id=world_id,
            )

    def check_collision(
        self,
        world_id: str,
        robot_id: str,
        qpos: Sequence[float],
        *,
        expected_scene_revision: int | None = None,
    ) -> ToolEnvelope:
        """Run backend collision validation for a robot configuration."""
        with self._world_guard(world_id, expected_scene_revision):
            return self._envelope(
                self.backend.check_collision(world_id, robot_id, qpos),
                world_id=world_id,
            )

    def plan_motion(
        self,
        world_id: str,
        robot_id: str,
        start_qpos: Sequence[float],
        goal_qpos: Sequence[float],
        *,
        samples: int = 32,
        expected_scene_revision: int | None = None,
    ) -> ToolEnvelope:
        """Generate a backend trajectory between two joint configurations."""
        if type(samples) is not int or not 2 <= samples <= MAX_TRAJECTORY_SAMPLES:
            raise ValueError(
                f"samples must be an integer between 2 and {MAX_TRAJECTORY_SAMPLES}"
            )
        with self._world_guard(world_id, expected_scene_revision):
            return self._envelope(
                self.backend.plan_motion(
                    world_id, robot_id, start_qpos, goal_qpos, samples=samples
                ),
                world_id=world_id,
            )

    def generate_robot_trajectory(
        self,
        world_id: str,
        robot_id: str,
        waypoints: Sequence[Sequence[float]],
        *,
        samples_per_segment: int = 32,
        expected_scene_revision: int | None = None,
    ) -> ToolEnvelope:
        """Generate and validate a multi-waypoint joint trajectory.

        Args:
            world_id: World containing the selected robot.
            robot_id: Robot UID registered in that world.
            waypoints: Ordered joint-position waypoints. The first waypoint is
                the starting configuration and each following waypoint is a
                requested target.
            samples_per_segment: Samples for each interpolated segment,
                including segment endpoints.

        Returns:
            A trajectory handle, samples, validation report, and metadata.
        """
        if len(waypoints) < 2:
            raise ValueError("waypoints must contain a start and at least one goal")
        if len(waypoints) > MAX_WAYPOINTS:
            raise ValueError(f"waypoints cannot contain more than {MAX_WAYPOINTS}")
        if (
            type(samples_per_segment) is not int
            or not 2 <= samples_per_segment <= MAX_TRAJECTORY_SAMPLES
        ):
            raise ValueError(
                "samples_per_segment must be between 2 and " f"{MAX_TRAJECTORY_SAMPLES}"
            )
        estimated_samples = 1 + (len(waypoints) - 1) * (samples_per_segment - 1)
        if estimated_samples > MAX_TRAJECTORY_SAMPLES:
            raise ValueError(
                f"expanded trajectory cannot contain more than "
                f"{MAX_TRAJECTORY_SAMPLES} samples"
            )
        with self._world_guard(world_id, expected_scene_revision) as world_state:
            segments: list[list[list[float]]] = []
            for start, goal in zip(waypoints, waypoints[1:]):
                result = self.backend.plan_motion(
                    world_id,
                    robot_id,
                    start,
                    goal,
                    samples=samples_per_segment,
                )
                positions = result["positions"]
                segments.append(positions if not segments else positions[1:])
            positions = [sample for segment in segments for sample in segment]
            validation = self.backend.validate_trajectory(world_id, robot_id, positions)
            trajectory_id = f"trajectory-{uuid.uuid4().hex[:12]}"
            record = {
                "trajectory_id": trajectory_id,
                "world_id": world_id,
                "scene_revision": world_state.get("scene_revision"),
                "robot_id": robot_id,
                "positions": positions,
                "waypoints": [list(waypoint) for waypoint in waypoints],
                "samples_per_segment": samples_per_segment,
                "validation": validation,
                "planner": "linear_interpolation",
            }
            with self._lock:
                self._trajectories[trajectory_id] = deepcopy(record)
            return self._envelope(record, world_id=world_id)

    def execute_trajectory(self, trajectory_id: str) -> ToolEnvelope:
        """Execute a stored trajectory as one coarse-grained backend operation.

        The backend owns the high-frequency loop. Codex sees one operation and
        receives a terminal summary instead of sending one MCP request per
        physics step.
        """
        with self._lock:
            try:
                trajectory = deepcopy(self._trajectories[trajectory_id])
            except KeyError as exc:
                raise ValueError(f"Unknown trajectory_id: {trajectory_id}") from exc
        if not trajectory["validation"]["valid"]:
            raise ValueError("Cannot execute an invalid trajectory")
        with self._world_guard(trajectory["world_id"], trajectory["scene_revision"]):
            result = self.backend.execute_trajectory(
                trajectory["world_id"],
                trajectory["robot_id"],
                trajectory["positions"],
                cancel_event=threading.Event(),
            )
            return self._envelope(result, world_id=trajectory["world_id"])

    def record_trajectory(
        self,
        trajectory_id: str,
        camera_id: str,
        output_path: str,
        *,
        fps: int = 30,
        include_depth: bool = True,
    ) -> ToolEnvelope:
        """Execute a stored trajectory while recording an offscreen camera."""
        with self._lock:
            try:
                trajectory = deepcopy(self._trajectories[trajectory_id])
            except KeyError as exc:
                raise ValueError(f"Unknown trajectory_id: {trajectory_id}") from exc
        if not trajectory["validation"]["valid"]:
            raise ValueError("Cannot record an invalid trajectory")
        with self._world_guard(trajectory["world_id"], trajectory["scene_revision"]):
            result = self.backend.record_trajectory(
                trajectory["world_id"],
                trajectory["robot_id"],
                trajectory["positions"],
                camera_id=camera_id,
                output_path=output_path,
                fps=fps,
                include_depth=include_depth,
                cancel_event=threading.Event(),
            )
            return self._envelope(result, world_id=trajectory["world_id"])

    def validate_trajectory(
        self,
        world_id: str,
        robot_id: str,
        positions: Sequence[Sequence[float]],
        *,
        expected_scene_revision: int | None = None,
    ) -> ToolEnvelope:
        """Validate a trajectory through the backend adapter."""
        if len(positions) > MAX_TRAJECTORY_SAMPLES:
            raise ValueError(
                f"positions cannot contain more than {MAX_TRAJECTORY_SAMPLES} samples"
            )
        with self._world_guard(world_id, expected_scene_revision):
            return self._envelope(
                self.backend.validate_trajectory(world_id, robot_id, positions),
                world_id=world_id,
            )

    def start_rollout(
        self,
        world_id: str,
        *,
        steps: int = 1,
        expected_scene_revision: int | None = None,
    ) -> ToolEnvelope:
        """Start a bounded asynchronous rollout job."""
        if type(steps) is not int or not 1 <= steps <= MAX_SIMULATION_STEPS:
            raise ValueError(
                f"steps must be an integer between 1 and {MAX_SIMULATION_STEPS}"
            )
        with self._world_guard(world_id, expected_scene_revision):
            run = _Run(
                run_id=f"run-{uuid.uuid4().hex[:12]}", world_id=world_id, steps=steps
            )
            with self._lock:
                active = sum(
                    item.status in {"queued", "running"} for item in self._runs.values()
                )
                if active >= self._max_active_runs:
                    raise RuntimeError(
                        "rollout capacity is full; wait for an existing run to finish"
                    )
                self._runs[run.run_id] = run
                run.future = self._executor.submit(self._execute_rollout, run)
            return self._run_status(run)

    def _execute_rollout(self, run: _Run) -> dict[str, Any]:
        with self._lock:
            if run.status == "cancelled":
                return self._run_status(run)
            run.status = "running"
        try:
            with self._world_guard(run.world_id):
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

    @staticmethod
    def codex_trajectory_demo_scene() -> dict[str, Any]:
        """Return the built-in Franka scene used by the Codex demo."""
        return {
            "robot": {
                "robot_class": "FrankaPandaCfg",
                "robot_type": "panda",
                "init_pos": [0.0, 0.0, 0.0],
                "init_rot": [0.0, 0.0, 0.0],
            },
            "sensor": [
                {
                    "sensor_type": "Camera",
                    "uid": "trajectory_camera",
                    "width": 320,
                    "height": 240,
                    "enable_color": True,
                    "enable_depth": True,
                    "intrinsics": [260.0, 260.0, 160.0, 120.0],
                    "extrinsics": {
                        "eye": [1.5, -1.5, 1.2],
                        "target": [0.0, 0.0, 0.6],
                        "up": [0.0, 0.0, 1.0],
                    },
                }
            ],
        }

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
        elif parsed.netloc == "demos" and parts == ["codex-trajectory", "scene"]:
            value = self.codex_trajectory_demo_scene()
        elif parsed.netloc == "robots" and len(parts) == 2 and parts[1] == "config":
            value = self.get_robot_info(parts[0])
        elif parsed.netloc == "worlds" and len(parts) == 2 and parts[1] == "manifest":
            value = self.get_world_state(parts[0])
        elif parsed.netloc == "runs" and len(parts) == 2 and parts[1] == "status":
            value = self.get_run_status(parts[0])
        elif parsed.netloc == "runs" and len(parts) == 2 and parts[1] == "metrics":
            value = self.get_run_metrics(parts[0])
        elif (
            parsed.netloc == "trajectories"
            and len(parts) == 1
            and parts[0] in self._trajectories
        ):
            with self._lock:
                value = deepcopy(self._trajectories[parts[0]])
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
    ) -> ToolEnvelope:
        """Add common result metadata to a backend response."""
        world_state = (
            self.backend.get_world_state(world_id) if world_id is not None else None
        )
        artifacts: list[dict[str, Any]] = []
        for key, media_type in (
            ("artifact_path", "video/mp4"),
            ("depth_artifact_path", "application/x-npz"),
        ):
            artifact_path = value.get(key)
            if isinstance(artifact_path, str):
                artifacts.append({"path": artifact_path, "media_type": media_type})
        result = {
            "status": "succeeded",
            "result": deepcopy(dict(value)),
            "backend": value.get(
                "backend", getattr(self.backend, "name", type(self.backend).__name__)
            ),
            "frame": "arena",
            "units": {"length": "m", "angle": "rad", "time": "s"},
            "diagnostics": [],
            "artifacts": artifacts,
        }
        if world_id is not None:
            result["world_id"] = world_id
            result["scene_revision"] = value.get(
                "scene_revision",
                None if world_state is None else world_state.get("scene_revision"),
            )
            result["seed"] = value.get(
                "seed", None if world_state is None else world_state.get("seed")
            )
        return result

    def _check_scene_revision(
        self, world_id: str, expected_scene_revision: int | None
    ) -> dict[str, Any]:
        """Validate an optional optimistic-concurrency scene revision."""
        state = self.backend.get_world_state(world_id)
        actual = state.get("scene_revision")
        if expected_scene_revision is not None and expected_scene_revision != actual:
            raise ValueError(
                f"scene_revision mismatch for {world_id}: expected "
                f"{expected_scene_revision}, current {actual}"
            )
        return state
