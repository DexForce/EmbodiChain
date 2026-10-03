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

"""Backend contracts and a deterministic CPU backend for MCP integration tests."""

from __future__ import annotations

import math
import threading
import uuid
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

__all__ = [
    "InMemorySimulationBackend",
    "SimulationBackend",
    "SimulationManagerBackend",
]


# These limits protect the protocol boundary.  Domain callers can still use
# the simulation APIs directly when they need a larger resource budget.
MAX_SIMULATION_STEPS = 100_000
MAX_TRAJECTORY_SAMPLES = 100_000
MAX_SCENE_SENSORS = 32
MAX_CAMERA_WIDTH = 4096
MAX_CAMERA_HEIGHT = 4096
MAX_CAMERA_PIXELS = 16_777_216
MAX_RECORD_PIXELS = 64_000_000
MAX_SCENE_ROBOTS = 16
MAX_SCENE_OBJECTS = 256
MAX_SCENE_ARTICULATIONS = 64
MAX_SCENE_LIGHTS = 64


def _interpolate_joint_path(
    start: Sequence[float], goal: Sequence[float], samples: int
) -> list[list[float]]:
    """Interpolate a joint path through the shared trajectory API."""
    import torch

    from embodichain.compute.trajectory import interpolate_with_nums

    keyframes = torch.as_tensor([list(start), list(goal)], dtype=torch.float64)
    return interpolate_with_nums(keyframes.unsqueeze(0), [samples - 1], device="cpu")[
        0
    ].tolist()


class SimulationBackend(Protocol):
    """Protocol implemented by a backend exposed through the MCP service.

    A backend owns simulator-specific state and algorithms.  The MCP service
    only validates arguments, assigns handles, and normalizes result metadata.
    """

    name: str

    def list_robots(self) -> list[dict[str, Any]]: ...

    def list_physics_backends(self) -> list[dict[str, Any]]: ...

    def list_cameras(self, world_id: str) -> list[dict[str, Any]]: ...

    def create_world(self, *, backend: str, seed: int | None) -> dict[str, Any]: ...

    def load_scene(
        self,
        world_id: str,
        *,
        task_id: str | None,
        scene: dict[str, Any] | None,
    ) -> dict[str, Any]: ...

    def reset_world(self, world_id: str) -> dict[str, Any]: ...

    def destroy_world(self, world_id: str) -> None: ...

    def get_world_state(self, world_id: str) -> dict[str, Any]: ...

    def snapshot_world(self, world_id: str) -> dict[str, Any]: ...

    def restore_world(self, world_id: str, snapshot_id: str) -> dict[str, Any]: ...

    def step_simulation(self, world_id: str, *, steps: int) -> dict[str, Any]: ...

    def forward_kinematics(
        self, world_id: str, robot_id: str, qpos: Sequence[float]
    ) -> dict[str, Any]: ...

    def solve_ik(
        self, world_id: str, robot_id: str, target_pose: dict[str, Any]
    ) -> dict[str, Any]: ...

    def check_reachability(
        self, world_id: str, robot_id: str, target_pose: dict[str, Any]
    ) -> dict[str, Any]: ...

    def check_collision(
        self, world_id: str, robot_id: str, qpos: Sequence[float]
    ) -> dict[str, Any]: ...

    def plan_motion(
        self,
        world_id: str,
        robot_id: str,
        start_qpos: Sequence[float],
        goal_qpos: Sequence[float],
        *,
        samples: int,
    ) -> dict[str, Any]: ...

    def execute_trajectory(
        self,
        world_id: str,
        robot_id: str,
        positions: Sequence[Sequence[float]],
        *,
        cancel_event: threading.Event,
    ) -> dict[str, Any]: ...

    def record_trajectory(
        self,
        world_id: str,
        robot_id: str,
        positions: Sequence[Sequence[float]],
        *,
        camera_id: str,
        output_path: str,
        fps: int,
        include_depth: bool,
        cancel_event: threading.Event,
    ) -> dict[str, Any]: ...

    def validate_trajectory(
        self, world_id: str, robot_id: str, positions: Sequence[Sequence[float]]
    ) -> dict[str, Any]: ...

    def run_rollout(
        self,
        world_id: str,
        *,
        steps: int,
        cancel_event: threading.Event,
    ) -> dict[str, Any]: ...


class InMemorySimulationBackend:
    """Deterministic CPU backend used for protocol smoke tests and examples.

    The backend models a two-link planar arm and a minimal world lifecycle.  It
    deliberately reports its limited collision scope so callers cannot mistake
    this backend for a full physics engine.  Production integrations can inject
    an implementation backed by :class:`SimulationManager` without changing
    the MCP service or its schemas.
    """

    name = "in-memory"
    _LINK_LENGTHS = (0.5, 0.4)
    _JOINT_LIMITS = ((-math.pi, math.pi), (-math.pi, math.pi))

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._worlds: dict[str, dict[str, Any]] = {}
        self._snapshots: dict[str, dict[str, Any]] = {}

    def list_robots(self) -> list[dict[str, Any]]:
        """Return the deterministic robot catalog."""
        return [
            {
                "robot_id": "demo_planar_arm",
                "title": "Demo planar two-link arm",
                "dof": 2,
                "joint_names": ["shoulder", "elbow"],
                "joint_limits": [list(limit) for limit in self._JOINT_LIMITS],
                "end_effector": "ee",
                "capabilities": ["fk", "ik", "reachability", "planning"],
                "scope": "in-memory planar kinematics; no physical collision model",
            }
        ]

    def list_physics_backends(self) -> list[dict[str, Any]]:
        """Return backends available to this process."""
        return [
            {
                "name": self.name,
                "available": True,
                "supports": ["world_lifecycle", "kinematics", "rollout"],
                "scope": "deterministic protocol backend",
            }
        ]

    def list_cameras(self, world_id: str) -> list[dict[str, Any]]:
        """Return no cameras because this backend has no renderer."""
        self._world(world_id)
        return []

    def create_world(self, *, backend: str, seed: int | None) -> dict[str, Any]:
        """Create a world handle and return its initial state."""
        if backend != self.name:
            raise ValueError(f"Unsupported backend for in-memory adapter: {backend}")
        world_id = f"world-{uuid.uuid4().hex[:12]}"
        with self._lock:
            self._worlds[world_id] = {
                "world_id": world_id,
                "backend": self.name,
                "seed": seed,
                "scene_revision": 0,
                "time_s": 0.0,
                "scene": {},
                "state": {"qpos": [0.0, 0.0], "qvel": [0.0, 0.0]},
            }
            return self.get_world_state(world_id)

    def _world(self, world_id: str) -> dict[str, Any]:
        try:
            return self._worlds[world_id]
        except KeyError as exc:
            raise ValueError(f"Unknown world_id: {world_id}") from exc

    def load_scene(
        self,
        world_id: str,
        *,
        task_id: str | None,
        scene: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Attach a task or opaque scene manifest to a world."""
        if task_id is None and scene is None:
            raise ValueError("Either task_id or scene must be provided")
        with self._lock:
            world = self._world(world_id)
            if world["scene_revision"]:
                raise RuntimeError(
                    "A world can load only one scene; create a new world to load another."
                )
            world["scene"] = {"task_id": task_id, "scene": deepcopy(scene or {})}
            world["scene_revision"] += 1
            return self.get_world_state(world_id)

    def reset_world(self, world_id: str) -> dict[str, Any]:
        """Reset dynamic state while preserving the scene manifest."""
        with self._lock:
            world = self._world(world_id)
            world["time_s"] = 0.0
            world["state"] = {"qpos": [0.0, 0.0], "qvel": [0.0, 0.0]}
            return self.get_world_state(world_id)

    def destroy_world(self, world_id: str) -> None:
        """Destroy a world and its in-memory snapshots."""
        with self._lock:
            self._world(world_id)
            self._worlds.pop(world_id)
            for snapshot_id, snapshot in list(self._snapshots.items()):
                if snapshot["world_id"] == world_id:
                    self._snapshots.pop(snapshot_id)

    def get_world_state(self, world_id: str) -> dict[str, Any]:
        """Return a detached world state snapshot."""
        with self._lock:
            return deepcopy(self._world(world_id))

    def snapshot_world(self, world_id: str) -> dict[str, Any]:
        """Save a world state under an explicit snapshot handle."""
        with self._lock:
            world = self._world(world_id)
            snapshot_id = f"snapshot-{uuid.uuid4().hex[:12]}"
            self._snapshots[snapshot_id] = {
                "snapshot_id": snapshot_id,
                "world_id": world_id,
                "scene_revision": world["scene_revision"],
                "state": {
                    "time_s": world["time_s"],
                    "state": deepcopy(world["state"]),
                },
            }
            return {"snapshot_id": snapshot_id, "world_id": world_id}

    def restore_world(self, world_id: str, snapshot_id: str) -> dict[str, Any]:
        """Restore a snapshot created for the same world."""
        with self._lock:
            world = self._world(world_id)
            try:
                snapshot = self._snapshots[snapshot_id]
            except KeyError as exc:
                raise ValueError(f"Unknown snapshot_id: {snapshot_id}") from exc
            if snapshot["world_id"] != world_id:
                raise ValueError("snapshot_id belongs to another world")
            if snapshot["scene_revision"] != world["scene_revision"]:
                raise ValueError(
                    "snapshot belongs to scene_revision "
                    f"{snapshot['scene_revision']}, current "
                    f"{world['scene_revision']}"
                )
            restored = deepcopy(snapshot["state"])
            world["time_s"] = restored["time_s"]
            world["state"] = restored["state"]
            return self.get_world_state(world_id)

    def step_simulation(self, world_id: str, *, steps: int) -> dict[str, Any]:
        """Advance a world by a fixed 10 ms timestep."""
        if type(steps) is not int or not 1 <= steps <= MAX_SIMULATION_STEPS:
            raise ValueError(
                f"steps must be an integer between 1 and {MAX_SIMULATION_STEPS}"
            )
        with self._lock:
            world = self._world(world_id)
            world["time_s"] += steps * 0.01
            return self.get_world_state(world_id)

    def _check_robot(self, robot_id: str) -> None:
        if robot_id != "demo_planar_arm":
            raise ValueError(f"Unknown robot_id: {robot_id}")

    @staticmethod
    def _finite_values(values: Sequence[float], label: str) -> list[float]:
        result = [float(value) for value in values]
        if not all(math.isfinite(value) for value in result):
            raise ValueError(f"{label} must contain finite numbers")
        return result

    def forward_kinematics(
        self, world_id: str, robot_id: str, qpos: Sequence[float]
    ) -> dict[str, Any]:
        """Compute planar end-effector position for the demo arm."""
        self._world(world_id)
        self._check_robot(robot_id)
        values = self._finite_values(qpos, "qpos")
        if len(values) != 2:
            raise ValueError("demo_planar_arm expects exactly two joint values")
        shoulder, elbow = values
        x = self._LINK_LENGTHS[0] * math.cos(shoulder) + self._LINK_LENGTHS[
            1
        ] * math.cos(shoulder + elbow)
        y = self._LINK_LENGTHS[0] * math.sin(shoulder) + self._LINK_LENGTHS[
            1
        ] * math.sin(shoulder + elbow)
        return {
            "robot_id": robot_id,
            "qpos": values,
            "end_effector": "ee",
            "position_m": [round(x, 9), round(y, 9), 0.0],
            "yaw_rad": round(shoulder + elbow, 9),
        }

    def solve_ik(
        self, world_id: str, robot_id: str, target_pose: dict[str, Any]
    ) -> dict[str, Any]:
        """Solve position-only planar IK for the demo arm."""
        self._world(world_id)
        self._check_robot(robot_id)
        position = target_pose.get("position_m")
        if not isinstance(position, Sequence) or isinstance(position, (str, bytes)):
            raise ValueError("target_pose.position_m must be a sequence")
        target = self._finite_values(position, "target_pose.position_m")
        if len(target) < 2:
            raise ValueError("target_pose.position_m must contain x and y")
        x, y = target[:2]
        distance = math.hypot(x, y)
        r_min = abs(self._LINK_LENGTHS[0] - self._LINK_LENGTHS[1])
        r_max = sum(self._LINK_LENGTHS)
        if distance < r_min - 1e-9 or distance > r_max + 1e-9:
            return {
                "reachable": False,
                "reason": "outside_geometric_workspace",
                "distance_m": distance,
                "workspace_m": [r_min, r_max],
            }
        cos_elbow = (
            distance**2 - self._LINK_LENGTHS[0] ** 2 - self._LINK_LENGTHS[1] ** 2
        ) / (2 * self._LINK_LENGTHS[0] * self._LINK_LENGTHS[1])
        elbow = math.acos(max(-1.0, min(1.0, cos_elbow)))
        shoulder = math.atan2(y, x) - math.atan2(
            self._LINK_LENGTHS[1] * math.sin(elbow),
            self._LINK_LENGTHS[0] + self._LINK_LENGTHS[1] * math.cos(elbow),
        )
        shoulder = (shoulder + math.pi) % (2.0 * math.pi) - math.pi
        qpos = [shoulder, elbow]
        joint_limits_valid = all(
            lower <= value <= upper
            for value, (lower, upper) in zip(qpos, self._JOINT_LIMITS)
        )
        return {
            "reachable": joint_limits_valid,
            "qpos": qpos,
            "target_pose": deepcopy(target_pose),
            "position_only": True,
            "joint_limits_checked": joint_limits_valid,
            "collision_checked": False,
        }

    def check_reachability(
        self, world_id: str, robot_id: str, target_pose: dict[str, Any]
    ) -> dict[str, Any]:
        """Check geometric reachability without claiming collision freedom."""
        result = self.solve_ik(world_id, robot_id, target_pose)
        return {
            "reachable": bool(result["reachable"]),
            "reason": result.get("reason", "geometrically_reachable"),
            "scope": "position_only_no_collision",
            **{
                key: result[key]
                for key in ("distance_m", "workspace_m")
                if key in result
            },
        }

    def check_collision(
        self, world_id: str, robot_id: str, qpos: Sequence[float]
    ) -> dict[str, Any]:
        """Check joint limits and report the absence of a geometry model."""
        self._world(world_id)
        self._check_robot(robot_id)
        values = self._finite_values(qpos, "qpos")
        if len(values) != 2:
            raise ValueError("demo_planar_arm expects exactly two joint values")
        joint_limits_valid = all(
            lower <= value <= upper
            for value, (lower, upper) in zip(values, self._JOINT_LIMITS)
        )
        return {
            "collision": False,
            "checked": True,
            "joint_limits_valid": joint_limits_valid,
            "collision_checked": False,
            "scope": "joint_limits_only_no_geometry_registered",
            "qpos": values,
        }

    def plan_motion(
        self,
        world_id: str,
        robot_id: str,
        start_qpos: Sequence[float],
        goal_qpos: Sequence[float],
        *,
        samples: int,
    ) -> dict[str, Any]:
        """Create a linearly interpolated joint path for the demo arm."""
        self._world(world_id)
        self._check_robot(robot_id)
        start = self._finite_values(start_qpos, "start_qpos")
        goal = self._finite_values(goal_qpos, "goal_qpos")
        if len(start) != 2 or len(goal) != 2:
            raise ValueError("demo_planar_arm expects two joint values")
        if type(samples) is not int or not 2 <= samples <= MAX_TRAJECTORY_SAMPLES:
            raise ValueError(
                f"samples must be an integer between 2 and {MAX_TRAJECTORY_SAMPLES}"
            )
        positions = _interpolate_joint_path(start, goal, samples)
        return {
            "robot_id": robot_id,
            "positions": positions,
            "dt_s": [0.01] * samples,
            "collision_checked": False,
            "planner": "linear_interpolation",
        }

    def validate_trajectory(
        self, world_id: str, robot_id: str, positions: Sequence[Sequence[float]]
    ) -> dict[str, Any]:
        """Check finite values and joint limits for a trajectory."""
        self._world(world_id)
        self._check_robot(robot_id)
        if not positions:
            raise ValueError("positions must contain at least one sample")
        if len(positions) > MAX_TRAJECTORY_SAMPLES:
            raise ValueError(
                f"positions cannot contain more than {MAX_TRAJECTORY_SAMPLES} samples"
            )
        violations: list[dict[str, Any]] = []
        normalized: list[list[float]] = []
        for index, position in enumerate(positions):
            values = self._finite_values(position, f"positions[{index}]")
            if len(values) != 2:
                raise ValueError("each trajectory sample must contain two values")
            normalized.append(values)
            for joint, (value, limits) in enumerate(zip(values, self._JOINT_LIMITS)):
                if not limits[0] <= value <= limits[1]:
                    violations.append(
                        {
                            "sample": index,
                            "joint": joint,
                            "value": value,
                            "limits": list(limits),
                        }
                    )
        return {
            "valid": not violations,
            "samples": len(normalized),
            "violations": violations,
            "collision_checked": False,
            "scope": "finite_values_and_joint_limits",
        }

    def execute_trajectory(
        self,
        world_id: str,
        robot_id: str,
        positions: Sequence[Sequence[float]],
        *,
        cancel_event: threading.Event,
    ) -> dict[str, Any]:
        """Execute a validated path inside the deterministic world state."""
        world = self._world(world_id)
        self._check_robot(robot_id)
        if not positions:
            raise ValueError("positions must contain at least one sample")
        if len(positions) > MAX_TRAJECTORY_SAMPLES:
            raise ValueError(
                f"positions cannot contain more than {MAX_TRAJECTORY_SAMPLES} samples"
            )
        normalized = [
            self._finite_values(row, "trajectory sample") for row in positions
        ]
        if any(len(row) != 2 for row in normalized):
            raise ValueError("demo_planar_arm expects two joint values")
        completed = 0
        for row in normalized:
            if cancel_event.is_set():
                return {"status": "cancelled", "steps": completed}
            world["state"]["qpos"] = row
            world["time_s"] += 0.01
            completed += 1
        return {
            "status": "succeeded",
            "steps": len(normalized),
            "final_qpos": normalized[-1],
            "final_time_s": world["time_s"],
        }

    def record_trajectory(
        self,
        world_id: str,
        robot_id: str,
        positions: Sequence[Sequence[float]],
        *,
        camera_id: str,
        output_path: str,
        fps: int,
        include_depth: bool,
        cancel_event: threading.Event,
    ) -> dict[str, Any]:
        """Reject camera recording because the deterministic backend is headless."""
        del positions, camera_id, output_path, fps, include_depth, cancel_event
        self._world(world_id)
        self._check_robot(robot_id)
        raise NotImplementedError("The in-memory backend has no offscreen camera")

    def run_rollout(
        self,
        world_id: str,
        *,
        steps: int,
        cancel_event: threading.Event,
    ) -> dict[str, Any]:
        """Run a deterministic no-op rollout and return aggregate metrics."""
        if type(steps) is not int or not 1 <= steps <= MAX_SIMULATION_STEPS:
            raise ValueError(
                f"steps must be an integer between 1 and {MAX_SIMULATION_STEPS}"
            )
        completed = 0
        for _ in range(steps):
            if cancel_event.is_set():
                return {"status": "cancelled", "steps": completed}
            self.step_simulation(world_id, steps=1)
            completed += 1
        state = self.get_world_state(world_id)
        return {
            "status": "succeeded",
            "steps": completed,
            "final_time_s": state["time_s"],
        }


class SimulationManagerBackend:
    """Adapt selected EmbodiChain ``SimulationManager`` APIs to MCP.

    Args:
        manager_factory: Optional callable receiving ``(backend, seed)`` and
            returning a prepared or unprepared ``SimulationManager``. The
            default lazily constructs a headless manager, which keeps importing
            the MCP package free of simulator startup side effects.

    This adapter intentionally exposes only backend-neutral lifecycle, state,
    FK, IK, and joint-limit validation. Scene builders and collision-aware
    planners remain injectable backend concerns until their complete runtime
    contracts can be represented in MCP schemas.
    """

    name = "simulation-manager"
    default_backend = "default"

    def __init__(
        self,
        manager_factory: Callable[[str, int | None], Any] | None = None,
    ) -> None:
        self._manager_factory = manager_factory or self._default_manager_factory
        self._managers: dict[str, Any] = {}
        self._world_metadata: dict[str, dict[str, Any]] = {}
        self._snapshots: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._world_locks: dict[str, threading.RLock] = {}
        self._prepared_worlds: set[str] = set()

    def _world_lock(self, world_id: str) -> threading.RLock:
        """Return the serialization lock owned by one world handle."""
        with self._lock:
            try:
                return self._world_locks[world_id]
            except KeyError as error:
                raise ValueError(f"Unknown world_id: {world_id}") from error

    def close(self) -> None:
        """Destroy all manager worlds and drain deferred native cleanup."""
        with self._lock:
            world_ids = tuple(self._managers)
        errors: list[Exception] = []
        for world_id in world_ids:
            try:
                self.destroy_world(world_id)
            except Exception as error:
                # Continue closing remaining worlds so one failed native cleanup
                # does not strand every other world owned by this backend.
                errors.append(error)
        if errors:
            raise RuntimeError(
                "Failed to close one or more simulation worlds: "
                + "; ".join(f"{type(error).__name__}: {error}" for error in errors)
            ) from errors[0]

    @staticmethod
    def _default_manager_factory(backend: str, seed: int | None) -> Any:
        """Construct a headless manager without importing DexSim at module load."""
        from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
        from embodichain.lab.sim.cfg import physics_cfg_for_backend
        from embodichain.utils import set_seed

        if seed is not None:
            set_seed(seed)
        cfg = SimulationManagerCfg(
            headless=True,
            enable_entity_gizmo=False,
            startup_summary="off",
            physics_config=physics_cfg_for_backend(backend),
        )
        return SimulationManager(cfg)

    def list_robots(self) -> list[dict[str, Any]]:
        """List robots currently registered in all manager worlds."""
        result: dict[str, dict[str, Any]] = {}
        with self._lock:
            managers = tuple(self._managers.values())
        for manager in managers:
            for robot_id in manager.get_robot_uid_list():
                robot = manager.get_robot(robot_id)
                if robot is None:
                    continue
                qpos = robot.get_qpos()
                result[robot_id] = {
                    "robot_id": robot_id,
                    "dof": int(qpos.shape[-1]),
                    "joint_names": list(getattr(robot, "joint_names", ())),
                    "control_parts": deepcopy(getattr(robot, "control_parts", None)),
                    "capabilities": ["fk", "ik", "joint_limits"],
                    "backend": "simulation-manager",
                }
        return list(result.values())

    def list_physics_backends(self) -> list[dict[str, Any]]:
        """Return the physics backends supported by SimulationManager."""
        return [
            {"name": "default", "available": True},
            {"name": "newton", "available": True},
        ]

    def list_cameras(self, world_id: str) -> list[dict[str, Any]]:
        """List cameras registered in a SimulationManager world."""
        with self._world_lock(world_id):
            manager = self._manager(world_id)
            result = []
            for camera_id in manager.get_sensor_uid_list():
                sensor = manager.get_sensor(camera_id)
                if (
                    sensor is None
                    or getattr(sensor.cfg, "sensor_type", None) != "Camera"
                ):
                    continue
                result.append(
                    {
                        "camera_id": camera_id,
                        "width": sensor.cfg.width,
                        "height": sensor.cfg.height,
                        "data_types": sensor.cfg.get_data_types(),
                        "role": sensor.cfg.visualization_role,
                    }
                )
            return result

    def create_world(self, *, backend: str, seed: int | None) -> dict[str, Any]:
        """Create a headless SimulationManager world."""
        if backend not in {"default", "newton"}:
            raise ValueError("backend must be 'default' or 'newton'")
        manager = self._manager_factory(backend, seed)
        world_id = f"sim-{uuid.uuid4().hex[:12]}"
        with self._lock:
            self._managers[world_id] = manager
            self._world_locks[world_id] = threading.RLock()
            self._world_metadata[world_id] = {
                "world_id": world_id,
                "backend": backend,
                "seed": seed,
                "scene_revision": 0,
                "scene": {},
            }
        try:
            return self.get_world_state(world_id)
        except Exception:
            try:
                manager.destroy(exit_process=False)
            except Exception:
                pass
            flush = getattr(type(manager), "flush_cleanup_queue", None)
            if flush is not None:
                try:
                    flush()
                except Exception:
                    pass
            with self._lock:
                self._managers.pop(world_id, None)
                self._world_metadata.pop(world_id, None)
                self._world_locks.pop(world_id, None)
                self._prepared_worlds.discard(world_id)
            raise

    def _manager(self, world_id: str) -> Any:
        try:
            return self._managers[world_id]
        except KeyError as exc:
            raise ValueError(f"Unknown world_id: {world_id}") from exc

    def load_scene(
        self,
        world_id: str,
        *,
        task_id: str | None,
        scene: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Load one validated scene and roll back partial native setup."""
        if task_id is None and scene is None:
            raise ValueError("Either task_id or scene must be provided")
        with self._world_lock(world_id):
            manager = self._manager(world_id)
            scene = dict(scene or {})
            if world_id in self._prepared_worlds:
                raise RuntimeError(
                    "A world can load only one scene; create a new world to load another."
                )
            self._validate_scene_resources(scene)
            self._validate_scene_configuration(world_id, scene)
            try:
                return self._load_scene_locked(world_id, task_id=task_id, scene=scene)
            except Exception:
                self._discard_world_after_failure(world_id, manager)
                raise

    @staticmethod
    def _scene_entries(scene: Mapping[str, Any], *names: str) -> list[Any]:
        """Combine scene collections while preserving separate object fields."""
        entries: list[Any] = []
        for name in names:
            value = scene.get(name, [])
            if value is None:
                continue
            if isinstance(value, dict):
                if name in {"robot", "robots", "articulation"} or (
                    name in {"background", "rigid_object"}
                    and any(key in value for key in ("uid", "shape", "fpath"))
                ):
                    value = [value]
                elif name == "light" and "light_type" in value:
                    value = [value]
                elif name == "light":
                    value = [
                        item
                        for group in value.values()
                        if isinstance(group, list)
                        for item in group
                    ]
                else:
                    value = list(value.values())
            if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
                raise ValueError(f"scene.{name} must be a sequence of mappings")
            entries.extend(value)
        return entries

    @staticmethod
    def _validate_scene_resources(scene: dict[str, Any]) -> None:
        """Reject unbounded sensor resource requests before native allocation."""
        collection_limits = (
            (("robot", "robots"), MAX_SCENE_ROBOTS, "robot"),
            (
                ("background", "rigid_object", "rigid_object_group"),
                MAX_SCENE_OBJECTS,
                "object",
            ),
            (("articulation",), MAX_SCENE_ARTICULATIONS, "articulation"),
            (("light",), MAX_SCENE_LIGHTS, "light"),
        )
        for names, limit, label in collection_limits:
            entries = SimulationManagerBackend._scene_entries(scene, *names)
            if len(entries) > limit:
                raise ValueError(f"scene contains more than {limit} {label} entries")
        sensor_values = scene.get("sensor", [])
        if isinstance(sensor_values, dict):
            sensor_values = [sensor_values]
        if sensor_values is None:
            return
        if isinstance(sensor_values, (str, bytes)) or not isinstance(
            sensor_values, Sequence
        ):
            raise ValueError("scene.sensor must be a sequence of mappings")
        if len(sensor_values) > MAX_SCENE_SENSORS:
            raise ValueError(
                f"scene.sensor cannot contain more than {MAX_SCENE_SENSORS} entries"
            )
        total_pixels = 0
        for index, value in enumerate(sensor_values):
            if not isinstance(value, dict):
                raise ValueError(f"scene.sensor[{index}] must be a mapping")
            if value.get("sensor_type", "") != "Camera":
                continue
            width = value.get("width", 640)
            height = value.get("height", 480)
            if (
                type(width) is not int
                or type(height) is not int
                or not 1 <= width <= MAX_CAMERA_WIDTH
                or not 1 <= height <= MAX_CAMERA_HEIGHT
            ):
                raise ValueError(
                    "Camera width and height must be positive integers no larger "
                    f"than {MAX_CAMERA_WIDTH}x{MAX_CAMERA_HEIGHT}"
                )
            pixels = width * height
            if pixels > MAX_CAMERA_PIXELS:
                raise ValueError(
                    f"Camera resolution cannot exceed {MAX_CAMERA_PIXELS} pixels"
                )
            total_pixels += pixels
        if total_pixels > MAX_CAMERA_PIXELS:
            raise ValueError(
                f"Total camera resolution cannot exceed {MAX_CAMERA_PIXELS} pixels"
            )

    def _validate_scene_configuration(
        self, world_id: str, scene: dict[str, Any]
    ) -> None:
        """Validate settings that can fail before native scene mutation."""
        with self._lock:
            world_backend = self._world_metadata[world_id]["backend"]
        declared_backend = scene.get("physics")
        if declared_backend is not None and declared_backend != world_backend:
            raise ValueError(
                f"Scene physics {declared_backend!r} does not match world backend "
                f"{world_backend!r}"
            )
        physics_config = scene.get("physics_config")
        if isinstance(physics_config, Mapping) and physics_config:
            raise ValueError(
                "Scene physics_config must be applied when the world is created; "
                "create a matching world and pass the settings through its backend."
            )

    def _discard_world_after_failure(self, world_id: str, manager: Any) -> None:
        """Best-effort cleanup for a scene that failed during materialization."""
        try:
            try:
                manager.destroy(exit_process=False)
            except Exception:
                pass
            flush = getattr(type(manager), "flush_cleanup_queue", None)
            if flush is not None:
                try:
                    flush()
                except Exception:
                    pass
        finally:
            with self._lock:
                self._managers.pop(world_id, None)
                self._world_metadata.pop(world_id, None)
                self._world_locks.pop(world_id, None)
                self._prepared_worlds.discard(world_id)
                for snapshot_id, snapshot in list(self._snapshots.items()):
                    if snapshot["world_id"] == world_id:
                        self._snapshots.pop(snapshot_id)

    def _load_scene_locked(
        self,
        world_id: str,
        *,
        task_id: str | None,
        scene: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Load the supported JSON scene fields into a manager world.

        The phase-one scene schema intentionally accepts the same top-level
        mappings used by the existing config decoder: ``robot``/``robots``,
        ``rigid_object``/``background``, ``articulation`` and ``light``.  A
        task reference without a resolved config is retained as metadata; the
        task catalog remains the owner of deployment and component resolution.
        """
        manager = self._manager(world_id)
        scene = dict(scene or {})
        if world_id in self._prepared_worlds:
            raise RuntimeError("A prepared world cannot load a second scene")
        from embodichain.lab.sim.cfg import (
            ArticulationCfg,
            LightCfg,
            RigidObjectCfg,
            RobotCfg,
        )
        from embodichain.lab.sim.robots import FrankaPandaCfg
        from embodichain.utils.utility import get_class_instance

        robot_values = self._scene_entries(scene, "robot", "robots")
        for robot_value in robot_values:
            if not isinstance(robot_value, dict):
                raise ValueError("scene.robots entries must be mappings")
            robot_data = dict(robot_value)
            robot_class = robot_data.pop(
                "robot_class", robot_data.pop("class_type", None)
            )
            if robot_class in {"FrankaPanda", "FrankaPandaCfg"} or (
                robot_class is None and robot_data.get("robot_type") == "panda"
            ):
                manager.add_robot(FrankaPandaCfg.from_dict(robot_data))
            elif robot_class is not None:
                class_name = (
                    robot_class if robot_class.endswith("Cfg") else f"{robot_class}Cfg"
                )
                try:
                    robot_cfg_type = get_class_instance(
                        "embodichain.lab.sim.robots", class_name
                    )
                except (AttributeError, ImportError, KeyError) as error:
                    raise ValueError(
                        f"Unsupported robot class_type: {robot_class}"
                    ) from error
                manager.add_robot(robot_cfg_type.from_dict(robot_data))
            else:
                manager.add_robot(RobotCfg.from_dict(robot_data))

        object_values = self._scene_entries(scene, "background", "rigid_object")
        for object_value in object_values:
            if not isinstance(object_value, dict):
                raise ValueError("scene.rigid_object entries must be mappings")
            manager.add_rigid_object(RigidObjectCfg.from_dict(dict(object_value)))

        articulation_values = self._scene_entries(scene, "articulation")
        for articulation_value in articulation_values:
            if not isinstance(articulation_value, dict):
                raise ValueError("scene.articulation entries must be mappings")
            manager.add_articulation(
                ArticulationCfg.from_dict(dict(articulation_value))
            )

        light_values = self._scene_entries(scene, "light")
        for light_value in light_values:
            if not isinstance(light_value, dict):
                raise ValueError("scene.light entries must be mappings")
            manager.add_light(LightCfg(**dict(light_value)))
        manager.prepare()
        sensor_values = scene.get("sensor", [])
        if isinstance(sensor_values, dict):
            sensor_values = [sensor_values]
        for sensor_value in sensor_values or []:
            if not isinstance(sensor_value, dict):
                raise ValueError("scene.sensor entries must be mappings")
            from embodichain.lab.sim.sensors import SensorCfg

            manager.add_sensor(SensorCfg.from_dict(dict(sensor_value)))
        self._prepared_worlds.add(world_id)
        with self._lock:
            metadata = self._world_metadata[world_id]
            metadata["scene"] = {"task_id": task_id, "scene": deepcopy(scene or {})}
            metadata["scene_revision"] += 1
        return self.get_world_state(world_id)

    def reset_world(self, world_id: str) -> dict[str, Any]:
        """Reset all registered manager objects."""
        with self._world_lock(world_id):
            self._manager(world_id).reset_objects_state()
            return self.get_world_state(world_id)

    def destroy_world(self, world_id: str) -> None:
        """Destroy a manager and flush its deferred native cleanup."""
        with self._world_lock(world_id):
            manager = self._manager(world_id)
            manager.destroy(exit_process=False)
            manager_type = type(manager)
            flush = getattr(manager_type, "flush_cleanup_queue", None)
            if flush is not None:
                flush()
            with self._lock:
                self._managers.pop(world_id, None)
                self._world_metadata.pop(world_id, None)
                self._world_locks.pop(world_id, None)
                self._prepared_worlds.discard(world_id)
                for snapshot_id, snapshot in list(self._snapshots.items()):
                    if snapshot["world_id"] == world_id:
                        self._snapshots.pop(snapshot_id)

    def get_world_state(self, world_id: str) -> dict[str, Any]:
        """Return manager metadata and detached robot states."""
        with self._world_lock(world_id):
            manager = self._manager(world_id)
            with self._lock:
                metadata = deepcopy(self._world_metadata[world_id])
            robots: dict[str, Any] = {}
            for robot_id in manager.get_robot_uid_list():
                robot = manager.get_robot(robot_id)
                if robot is None:
                    continue
                robots[robot_id] = {
                    "qpos": robot.get_qpos().detach().cpu().tolist(),
                    "qvel": robot.get_qvel().detach().cpu().tolist(),
                }
            metadata["robots"] = robots
            metadata["backend"] = getattr(manager.physics, "name", metadata["backend"])
            return metadata

    def snapshot_world(self, world_id: str) -> dict[str, Any]:
        """Capture manager robot joint state in a snapshot handle."""
        with self._world_lock(world_id):
            snapshot_id = f"snapshot-{uuid.uuid4().hex[:12]}"
            state = self.get_world_state(world_id)
            with self._lock:
                self._snapshots[snapshot_id] = {
                    "snapshot_id": snapshot_id,
                    "world_id": world_id,
                    "scene_revision": state["scene_revision"],
                    "state": state,
                }
            return {"snapshot_id": snapshot_id, "world_id": world_id}

    def restore_world(self, world_id: str, snapshot_id: str) -> dict[str, Any]:
        """Restore joint positions and velocities from a snapshot."""
        with self._world_lock(world_id):
            manager = self._manager(world_id)
            try:
                snapshot = self._snapshots[snapshot_id]
            except KeyError as exc:
                raise ValueError(f"Unknown snapshot_id: {snapshot_id}") from exc
            if snapshot["world_id"] != world_id:
                raise ValueError("snapshot_id belongs to another world")
            current = self.get_world_state(world_id)
            if snapshot["scene_revision"] != current["scene_revision"]:
                raise ValueError(
                    "snapshot belongs to scene_revision "
                    f"{snapshot['scene_revision']}, current "
                    f"{current['scene_revision']}"
                )
            import torch

            for robot_id, robot_state in snapshot["state"]["robots"].items():
                robot = manager.get_robot(robot_id)
                if robot is None:
                    continue
                qpos = torch.as_tensor(robot_state["qpos"], dtype=torch.float32)
                qvel = torch.as_tensor(robot_state["qvel"], dtype=torch.float32)
                robot.set_qpos(qpos, target=False)
                robot.set_qpos(qpos, target=True)
                robot.set_qvel(qvel, target=False)
                clear_dynamics = getattr(robot, "clear_dynamics", None)
                if clear_dynamics is not None:
                    clear_dynamics()
            sync_render_state = getattr(manager, "sync_render_state", None)
            if sync_render_state is not None:
                sync_render_state()
            return self.get_world_state(world_id)

    def step_simulation(self, world_id: str, *, steps: int) -> dict[str, Any]:
        """Advance a manager using its explicit update boundary."""
        if type(steps) is not int or not 1 <= steps <= MAX_SIMULATION_STEPS:
            raise ValueError(
                f"steps must be an integer between 1 and {MAX_SIMULATION_STEPS}"
            )
        with self._world_lock(world_id):
            manager = self._manager(world_id)
            manager.update(step=steps, render_final_step=False)
            return self.get_world_state(world_id)

    def _robot(self, world_id: str, robot_id: str) -> Any:
        manager = self._manager(world_id)
        robot = manager.get_robot(robot_id)
        if robot is None:
            raise ValueError(f"Unknown robot_id in world {world_id}: {robot_id}")
        return robot

    @staticmethod
    def _kinematic_part(robot: Any) -> str | None:
        """Select a configured control part that owns a kinematic solver."""
        control_parts = getattr(robot, "control_parts", None)
        if not isinstance(control_parts, Mapping):
            return None
        solver_names = getattr(robot, "_solvers", {})
        if isinstance(solver_names, Mapping):
            candidates = [
                name
                for name in ("arm", "manipulator", "default")
                if name in control_parts
            ]
            candidates.extend(name for name in control_parts if name not in candidates)
            for name in candidates:
                if name in solver_names:
                    return name
        return next(iter(control_parts), None)

    @staticmethod
    def _part_qpos(robot: Any, qpos: Any, part: str | None) -> Any:
        """Adapt full-articulation qpos to a selected solver's joint order."""
        if part is None:
            return qpos
        joint_ids = robot.get_joint_ids(name=part)
        full_dof = int(robot.get_qpos().shape[-1])
        if qpos.shape[-1] == full_dof:
            return qpos[:, joint_ids]
        if qpos.shape[-1] != len(joint_ids):
            raise ValueError(
                f"qpos width {qpos.shape[-1]} does not match control part "
                f"{part!r} width {len(joint_ids)}"
            )
        return qpos

    def forward_kinematics(
        self, world_id: str, robot_id: str, qpos: Sequence[float]
    ) -> dict[str, Any]:
        """Compute FK through the registered robot's configured solver."""
        import torch

        robot = self._robot(world_id, robot_id)
        qpos_tensor = torch.as_tensor(qpos, dtype=torch.float32)
        if qpos_tensor.ndim == 1:
            qpos_tensor = qpos_tensor.unsqueeze(0)
        part = self._kinematic_part(robot)
        if part is None:
            result = robot.compute_fk(qpos_tensor)
        else:
            result = robot.compute_fk(
                self._part_qpos(robot, qpos_tensor, part), name=part
            )
        if result.ndim == 3 and result.shape[-2:] == (4, 4):
            from embodichain.utils.math import quat_from_matrix

            result = torch.cat(
                (result[:, :3, 3], quat_from_matrix(result[:, :3, :3])), dim=-1
            )
        return {"robot_id": robot_id, "pose_xyz_xyzw": result.detach().cpu().tolist()}

    def solve_ik(
        self, world_id: str, robot_id: str, target_pose: dict[str, Any]
    ) -> dict[str, Any]:
        """Compute IK through the registered robot's configured solver."""
        import torch

        position = target_pose.get("position_m")
        if not isinstance(position, Sequence) or len(position) != 3:
            raise ValueError("target_pose.position_m must contain three values")
        quaternion = target_pose.get("quaternion_xyzw", [0.0, 0.0, 0.0, 1.0])
        if not isinstance(quaternion, Sequence) or len(quaternion) != 4:
            raise ValueError("target_pose.quaternion_xyzw must contain four values")
        robot = self._robot(world_id, robot_id)
        pose = torch.as_tensor([list(position) + list(quaternion)], dtype=torch.float32)
        part = self._kinematic_part(robot)
        if part is None:
            success, qpos = robot.compute_ik(pose)
        else:
            success, qpos = robot.compute_ik(pose, name=part)
            joint_ids = robot.get_joint_ids(name=part)
            qpos = qpos.reshape(qpos.shape[0], -1)
            full_qpos = robot.get_qpos().detach().clone().to(qpos)
            full_qpos[:, joint_ids] = qpos
            qpos = full_qpos
        return {
            "robot_id": robot_id,
            "reachable": bool(success.reshape(-1)[0].item()),
            "qpos": qpos.reshape(-1).detach().cpu().tolist(),
            "target_pose": deepcopy(target_pose),
            "collision_checked": False,
        }

    def check_reachability(
        self, world_id: str, robot_id: str, target_pose: dict[str, Any]
    ) -> dict[str, Any]:
        """Check configured IK reachability without collision certification."""
        result = self.solve_ik(world_id, robot_id, target_pose)
        return {
            "reachable": result["reachable"],
            "scope": "configured_ik_no_collision",
        }

    def check_collision(
        self, world_id: str, robot_id: str, qpos: Sequence[float]
    ) -> dict[str, Any]:
        """Check joint limits and report the absence of a generic geometry query."""
        self._manager(world_id)
        robot = self._robot(world_id, robot_id)
        import torch

        values = torch.as_tensor(qpos, dtype=torch.float32)
        if values.ndim != 1:
            raise ValueError("qpos must have shape (dof,)")
        limits = robot.get_qpos_limits()[0].detach().cpu()
        if values.shape[0] != limits.shape[0]:
            raise ValueError("qpos width does not match robot qpos limits")
        finite = bool(torch.isfinite(values).all().item())
        joint_limits_valid = finite and bool(
            ((limits[:, 0] <= values) & (values <= limits[:, 1])).all().item()
        )
        return {
            "collision": False,
            "checked": True,
            "joint_limits_valid": joint_limits_valid,
            "collision_checked": False,
            "scope": "joint_limits_only_no_generic_geometry_query",
        }

    def plan_motion(
        self,
        world_id: str,
        robot_id: str,
        start_qpos: Sequence[float],
        goal_qpos: Sequence[float],
        *,
        samples: int,
    ) -> dict[str, Any]:
        """Generate a deterministic joint interpolation for manager state.

        The current adapter uses the backend-neutral trajectory contract. A
        planner-specific adapter can replace this method with MotionGenerator
        or cuRobo while retaining the same MCP output schema.
        """
        self._robot(world_id, robot_id)
        start = [float(value) for value in start_qpos]
        goal = [float(value) for value in goal_qpos]
        if len(start) != len(goal) or not start:
            raise ValueError("start_qpos and goal_qpos must have equal non-zero width")
        if type(samples) is not int or not 2 <= samples <= MAX_TRAJECTORY_SAMPLES:
            raise ValueError(
                f"samples must be an integer between 2 and {MAX_TRAJECTORY_SAMPLES}"
            )
        positions = _interpolate_joint_path(start, goal, samples)
        return {
            "robot_id": robot_id,
            "positions": positions,
            "planner": "linear_interpolation",
        }

    def execute_trajectory(
        self,
        world_id: str,
        robot_id: str,
        positions: Sequence[Sequence[float]],
        *,
        cancel_event: threading.Event,
    ) -> dict[str, Any]:
        """Apply a joint trajectory inside the manager's explicit update loop."""
        manager = self._manager(world_id)
        robot = self._robot(world_id, robot_id)
        import torch

        trajectory = torch.as_tensor(positions, dtype=torch.float32)
        if trajectory.ndim != 2 or trajectory.shape[0] == 0:
            raise ValueError("positions must have shape (samples, dof)")
        if trajectory.shape[0] > MAX_TRAJECTORY_SAMPLES:
            raise ValueError(
                f"positions cannot contain more than {MAX_TRAJECTORY_SAMPLES} samples"
            )
        if not bool(torch.isfinite(trajectory).all().item()):
            raise ValueError("positions must contain finite values")
        expected_dof = int(robot.get_qpos().shape[-1])
        if trajectory.shape[1] != expected_dof:
            raise ValueError(
                f"trajectory width {trajectory.shape[1]} does not match robot "
                f"width {expected_dof}"
            )
        for position in trajectory:
            if cancel_event.is_set():
                return {"status": "cancelled", "steps": 0}
            robot.set_qpos(position.unsqueeze(0), target=False)
            robot.set_qpos(position.unsqueeze(0), target=True)
            manager.update(step=1, render_final_step=False)
        measured = robot.get_qpos().detach().cpu().tolist()
        return {
            "status": "succeeded",
            "steps": int(trajectory.shape[0]),
            "final_qpos": measured[0] if measured else [],
        }

    def record_trajectory(
        self,
        world_id: str,
        robot_id: str,
        positions: Sequence[Sequence[float]],
        *,
        camera_id: str,
        output_path: str,
        fps: int,
        include_depth: bool,
        cancel_event: threading.Event,
    ) -> dict[str, Any]:
        """Execute a trajectory while recording an offscreen camera."""
        if type(fps) is not int or not 1 <= fps <= 240:
            raise ValueError("fps must be an integer between 1 and 240")
        manager = self._manager(world_id)
        robot = self._robot(world_id, robot_id)
        camera = manager.get_sensor(camera_id)
        if camera is None or getattr(camera.cfg, "sensor_type", None) != "Camera":
            raise ValueError(f"Unknown camera_id in world {world_id}: {camera_id}")
        try:
            import imageio.v2 as imageio
        except ModuleNotFoundError as error:
            raise ModuleNotFoundError(
                "Camera recording requires the optional dependency: "
                "pip install 'embodichain[mcp-demo]'"
            ) from error
        import numpy as np
        import torch

        trajectory = torch.as_tensor(positions, dtype=torch.float32)
        if trajectory.ndim != 2 or trajectory.shape[0] == 0:
            raise ValueError("positions must have shape (samples, dof)")
        if trajectory.shape[0] > MAX_TRAJECTORY_SAMPLES:
            raise ValueError(
                f"positions cannot contain more than {MAX_TRAJECTORY_SAMPLES} samples"
            )
        if not bool(torch.isfinite(trajectory).all().item()):
            raise ValueError("positions must contain finite values")
        expected_dof = int(robot.get_qpos().shape[-1])
        if trajectory.shape[1] != expected_dof:
            raise ValueError(
                f"trajectory width {trajectory.shape[1]} does not match robot "
                f"width {expected_dof}"
            )
        frame_pixels = int(camera.cfg.width) * int(camera.cfg.height)
        if frame_pixels * int(trajectory.shape[0]) > MAX_RECORD_PIXELS:
            raise ValueError(
                "recording exceeds the MCP frame budget of "
                f"{MAX_RECORD_PIXELS} pixels"
            )
        frames: list[np.ndarray] = []
        depth_frames: list[np.ndarray] = []
        for position in trajectory:
            if cancel_event.is_set():
                return {"status": "cancelled", "frames": len(frames)}
            robot.set_qpos(position.unsqueeze(0), target=False)
            robot.set_qpos(position.unsqueeze(0), target=True)
            manager.update(step=1, render_final_step=False)
            manager.sync_render_state()
            camera.update()
            data = camera.get_data()
            color = data["color"][0].detach().cpu().numpy()
            frames.append(color[..., :3])
            if include_depth and "depth" in data:
                depth_frames.append(data["depth"][0].detach().cpu().numpy())
        if not frames:
            raise ValueError("trajectory produced no camera frames")
        output = Path(output_path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(output, frames, fps=fps, macro_block_size=1)
        result: dict[str, Any] = {
            "status": "succeeded",
            "artifact_path": str(output),
            "frames": len(frames),
            "fps": fps,
            "width": int(frames[0].shape[1]),
            "height": int(frames[0].shape[0]),
        }
        if depth_frames:
            depth_path = output.with_suffix(".depth.npz")
            np.savez_compressed(depth_path, depth=np.stack(depth_frames, axis=0))
            result["depth_artifact_path"] = str(depth_path)
        return result

    def validate_trajectory(
        self, world_id: str, robot_id: str, positions: Sequence[Sequence[float]]
    ) -> dict[str, Any]:
        """Validate finite samples against the robot's configured qpos limits."""
        self._manager(world_id)
        robot = self._robot(world_id, robot_id)
        import torch

        trajectory = torch.as_tensor(positions, dtype=torch.float32)
        if trajectory.ndim != 2 or trajectory.shape[0] == 0:
            raise ValueError("positions must have shape (samples, dof)")
        if trajectory.shape[0] > MAX_TRAJECTORY_SAMPLES:
            raise ValueError(
                f"positions cannot contain more than {MAX_TRAJECTORY_SAMPLES} samples"
            )
        if not bool(torch.isfinite(trajectory).all().item()):
            raise ValueError("positions must contain finite values")
        limits = robot.get_qpos_limits()[0].detach().cpu()
        if trajectory.shape[1] != limits.shape[0]:
            raise ValueError("trajectory width does not match robot qpos limits")
        violations = []
        tolerance = 1.0e-6
        for sample, row in enumerate(trajectory):
            for joint, value in enumerate(row):
                if not bool(
                    (
                        (limits[joint, 0] - tolerance <= value)
                        & (value <= limits[joint, 1] + tolerance)
                    ).item()
                ):
                    violations.append({"sample": sample, "joint": joint})
        return {
            "valid": not violations,
            "samples": int(trajectory.shape[0]),
            "violations": violations,
            "collision_checked": False,
        }

    def run_rollout(
        self,
        world_id: str,
        *,
        steps: int,
        cancel_event: threading.Event,
    ) -> dict[str, Any]:
        """Run explicit manager updates until completion or cancellation."""
        if type(steps) is not int or not 1 <= steps <= MAX_SIMULATION_STEPS:
            raise ValueError(
                f"steps must be an integer between 1 and {MAX_SIMULATION_STEPS}"
            )
        completed = 0
        for _ in range(steps):
            if cancel_event.is_set():
                return {"status": "cancelled", "steps": completed}
            self.step_simulation(world_id, steps=1)
            completed += 1
        return {"status": "succeeded", "steps": completed}
