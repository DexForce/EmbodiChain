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
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

__all__ = [
    "InMemorySimulationBackend",
    "SimulationBackend",
    "SimulationManagerBackend",
]


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
            self._world(world_id)
            snapshot_id = f"snapshot-{uuid.uuid4().hex[:12]}"
            self._snapshots[snapshot_id] = {
                "snapshot_id": snapshot_id,
                "world_id": world_id,
                "state": self.get_world_state(world_id),
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
            restored = deepcopy(snapshot["state"])
            world.update(restored)
            return self.get_world_state(world_id)

    def step_simulation(self, world_id: str, *, steps: int) -> dict[str, Any]:
        """Advance a world by a fixed 10 ms timestep."""
        if type(steps) is not int or steps <= 0:
            raise ValueError("steps must be a positive integer")
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
        qpos = [shoulder, elbow]
        return {
            "reachable": True,
            "qpos": qpos,
            "target_pose": deepcopy(target_pose),
            "position_only": True,
            "joint_limits_checked": True,
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
        """Validate inputs and report that this backend has no collision model."""
        self._world(world_id)
        self._check_robot(robot_id)
        values = self._finite_values(qpos, "qpos")
        if len(values) != 2:
            raise ValueError("demo_planar_arm expects exactly two joint values")
        return {
            "collision": False,
            "checked": False,
            "scope": "no_geometry_registered",
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
        if type(samples) is not int or not 2 <= samples <= 10000:
            raise ValueError("samples must be an integer between 2 and 10000")
        positions = [
            [start[j] + (goal[j] - start[j]) * i / (samples - 1) for j in range(2)]
            for i in range(samples)
        ]
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
        normalized = [
            self._finite_values(row, "trajectory sample") for row in positions
        ]
        if any(len(row) != 2 for row in normalized):
            raise ValueError("demo_planar_arm expects two joint values")
        for row in normalized:
            if cancel_event.is_set():
                return {"status": "cancelled", "steps": 0}
            world["state"]["qpos"] = row
            world["time_s"] += 0.01
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
        if type(steps) is not int or not 1 <= steps <= 1_000_000:
            raise ValueError("steps must be an integer between 1 and 1000000")
        for _ in range(steps):
            if cancel_event.is_set():
                return {"status": "cancelled", "steps": 0}
            self.step_simulation(world_id, steps=1)
        state = self.get_world_state(world_id)
        return {"status": "succeeded", "steps": steps, "final_time_s": state["time_s"]}


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
        self._prepared_worlds: set[str] = set()

    @staticmethod
    def _default_manager_factory(backend: str, seed: int | None) -> Any:
        """Construct a headless manager without importing DexSim at module load."""
        from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
        from embodichain.lab.sim.cfg import physics_cfg_for_backend

        del seed  # Seed application belongs to the environment/task owner.
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
        manager = self._manager(world_id)
        result = []
        for camera_id in manager.get_sensor_uid_list():
            sensor = manager.get_sensor(camera_id)
            if sensor is None or getattr(sensor.cfg, "sensor_type", None) != "Camera":
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
            self._world_metadata[world_id] = {
                "world_id": world_id,
                "backend": backend,
                "seed": seed,
                "scene_revision": 0,
                "scene": {},
            }
        return self.get_world_state(world_id)

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
        """Load the supported JSON scene fields into a manager world.

        The phase-one scene schema intentionally accepts the same top-level
        mappings used by the existing config decoder: ``robot``/``robots``,
        ``rigid_object``/``background``, ``articulation`` and ``light``.  A
        task reference without a resolved config is retained as metadata; the
        task catalog remains the owner of deployment and component resolution.
        """
        if task_id is None and scene is None:
            raise ValueError("Either task_id or scene must be provided")
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

        robot_values = scene.get("robots", scene.get("robot", []))
        if isinstance(robot_values, dict):
            robot_values = [robot_values]
        if robot_values is not None:
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
                else:
                    manager.add_robot(RobotCfg.from_dict(robot_data))

        object_values = scene.get("rigid_object", scene.get("background", []))
        if isinstance(object_values, dict):
            object_values = list(object_values.values())
        if object_values is not None:
            for object_value in object_values:
                if not isinstance(object_value, dict):
                    raise ValueError("scene.rigid_object entries must be mappings")
                manager.add_rigid_object(RigidObjectCfg.from_dict(dict(object_value)))

        articulation_values = scene.get("articulation", [])
        if isinstance(articulation_values, dict):
            articulation_values = [articulation_values]
        if articulation_values is not None:
            for articulation_value in articulation_values:
                if not isinstance(articulation_value, dict):
                    raise ValueError("scene.articulation entries must be mappings")
                manager.add_articulation(
                    ArticulationCfg.from_dict(dict(articulation_value))
                )

        light_values = scene.get("light", [])
        if isinstance(light_values, dict):
            if "light_type" in light_values:
                light_values = [light_values]
            else:
                light_values = [
                    item
                    for group in light_values.values()
                    if isinstance(group, list)
                    for item in group
                ]
        if light_values is not None:
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
        self._manager(world_id).reset_objects_state()
        return self.get_world_state(world_id)

    def destroy_world(self, world_id: str) -> None:
        """Destroy a manager and flush its deferred native cleanup."""
        manager = self._manager(world_id)
        manager.destroy(exit_process=False)
        manager_type = type(manager)
        flush = getattr(manager_type, "flush_cleanup_queue", None)
        if flush is not None:
            flush()
        with self._lock:
            self._managers.pop(world_id, None)
            self._world_metadata.pop(world_id, None)
            self._prepared_worlds.discard(world_id)

    def get_world_state(self, world_id: str) -> dict[str, Any]:
        """Return manager metadata and detached robot states."""
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
        snapshot_id = f"snapshot-{uuid.uuid4().hex[:12]}"
        state = self.get_world_state(world_id)
        with self._lock:
            self._snapshots[snapshot_id] = {
                "snapshot_id": snapshot_id,
                "world_id": world_id,
                "state": state,
            }
        return {"snapshot_id": snapshot_id, "world_id": world_id}

    def restore_world(self, world_id: str, snapshot_id: str) -> dict[str, Any]:
        """Restore joint positions and velocities from a snapshot."""
        manager = self._manager(world_id)
        try:
            snapshot = self._snapshots[snapshot_id]
        except KeyError as exc:
            raise ValueError(f"Unknown snapshot_id: {snapshot_id}") from exc
        if snapshot["world_id"] != world_id:
            raise ValueError("snapshot_id belongs to another world")
        import torch

        for robot_id, robot_state in snapshot["state"]["robots"].items():
            robot = manager.get_robot(robot_id)
            if robot is None:
                continue
            robot.set_qpos(torch.as_tensor(robot_state["qpos"], dtype=torch.float32))
            robot.set_qvel(torch.as_tensor(robot_state["qvel"], dtype=torch.float32))
        return self.get_world_state(world_id)

    def step_simulation(self, world_id: str, *, steps: int) -> dict[str, Any]:
        """Advance a manager using its explicit update boundary."""
        if type(steps) is not int or steps <= 0:
            raise ValueError("steps must be a positive integer")
        manager = self._manager(world_id)
        with self._lock:
            manager.update(step=steps, render_final_step=False)
        return self.get_world_state(world_id)

    def _robot(self, world_id: str, robot_id: str) -> Any:
        manager = self._manager(world_id)
        robot = manager.get_robot(robot_id)
        if robot is None:
            raise ValueError(f"Unknown robot_id in world {world_id}: {robot_id}")
        return robot

    def forward_kinematics(
        self, world_id: str, robot_id: str, qpos: Sequence[float]
    ) -> dict[str, Any]:
        """Compute FK through the registered robot's configured solver."""
        import torch

        robot = self._robot(world_id, robot_id)
        qpos_tensor = torch.as_tensor(qpos, dtype=torch.float32)
        if qpos_tensor.ndim == 1:
            qpos_tensor = qpos_tensor.unsqueeze(0)
        result = robot.compute_fk(qpos_tensor)
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
        success, qpos = robot.compute_ik(pose)
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
        """Report the absence of a generic manager collision query."""
        self._manager(world_id)
        self._robot(world_id, robot_id)
        return {
            "collision": False,
            "checked": False,
            "scope": "no_generic_simulation_manager_collision_query",
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
        if type(samples) is not int or not 2 <= samples <= 10000:
            raise ValueError("samples must be an integer between 2 and 10000")
        positions = [
            [
                start[j] + (goal[j] - start[j]) * i / (samples - 1)
                for j in range(len(start))
            ]
            for i in range(samples)
        ]
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
            robot.set_qpos(position.unsqueeze(0))
            manager.update(step=1, render_final_step=False)
        return {
            "status": "succeeded",
            "steps": int(trajectory.shape[0]),
            "final_qpos": trajectory[-1].detach().cpu().tolist(),
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
        if not bool(torch.isfinite(trajectory).all().item()):
            raise ValueError("positions must contain finite values")
        expected_dof = int(robot.get_qpos().shape[-1])
        if trajectory.shape[1] != expected_dof:
            raise ValueError(
                f"trajectory width {trajectory.shape[1]} does not match robot "
                f"width {expected_dof}"
            )
        frames: list[np.ndarray] = []
        depth_frames: list[np.ndarray] = []
        for position in trajectory:
            if cancel_event.is_set():
                return {"status": "cancelled", "frames": len(frames)}
            robot.set_qpos(position.unsqueeze(0))
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
        for _ in range(steps):
            if cancel_event.is_set():
                return {"status": "cancelled", "steps": 0}
            self.step_simulation(world_id, steps=1)
        return {"status": "succeeded", "steps": steps}
