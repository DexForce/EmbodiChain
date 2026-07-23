# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
"""Persistent cuRobo V2 worker process for the subprocess-isolated planner backend.

This module runs entirely inside a child process spawned (``spawn`` start method)
by :class:`~embodichain.lab.sim.planners.curobo.curobo_planner.CuroboPlanner`. The child
owns a private CUDA context, fully decoupled from DexSim's Vulkan/CUDA interop
semaphores, so cuRobo may capture and replay CUDA graphs - the path that crashes
DexSim's ``DFGpuSemaphore`` stream synchronization when run in-process.

The worker is a thin cuRobo executor: it builds (and caches, per
``(batch_size, multi_env)``) ``MotionPlanner`` / ``BatchMotionPlanner`` instances
with ``use_cuda_graph=True``, warms them up once, and serves ``plan`` /
``update_obstacle`` requests over two :class:`multiprocessing.Queue` instances.
All tensor payloads cross the process boundary as CPU tensors; the parent moves
results back onto its own device. The cuRobo result is returned as a
:dataclass:`_PlanResultMsg` whose fields mirror the attributes the parent's
existing ``_extract_segment`` / ``_map_curobo_to_sim`` post-processing reads, so
no EmbodiChain-side extraction logic is duplicated here.

This module intentionally imports only cuRobo / torch / the standard library - no
DexSim, no ``curobo_planner`` - so spawning the child never pulls the simulator
into the worker's context.
"""

from __future__ import annotations

import atexit
import ctypes
import importlib
import os
import traceback
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import torch
import yaml

__all__ = [
    "InitMsg",
    "ConfigureMsg",
    "PlanMsg",
    "UpdateObstacleMsg",
    "CloseMsg",
    "PlanResultMsg",
    "worker_main",
]


# =============================================================================
# Worker lifecycle helpers (mirrors toppra_planner._worker_init)
# =============================================================================


def _set_parent_death_signal() -> None:
    """Ask the kernel to SIGKILL this worker the instant its parent dies.

    This is what guarantees no residual worker processes when the parent exits
    via ``os._exit(0)`` (the path ``SimulationManager.destroy()`` takes, which
    skips every Python finalizer).
    """
    try:
        PR_SET_PDEATHSIG = 1
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl.argtypes = [
            ctypes.c_int,
            ctypes.c_ulong,
            ctypes.c_ulong,
            ctypes.c_ulong,
            ctypes.c_ulong,
        ]
        libc.prctl.restype = ctypes.c_int
        libc.prctl(PR_SET_PDEATHSIG, 9, 0, 0, 0)  # 9 == SIGKILL
    except Exception:
        pass


def _worker_init() -> None:
    """Run once at worker start: clear inherited atexit and arm parent-death reap."""
    import multiprocessing as mp

    if mp.get_start_method() == "fork":
        atexit._clear()
    _set_parent_death_signal()


# =============================================================================
# IPC message protocol (all picklable; tensors travel as CPU)
# =============================================================================


@dataclass
class InitMsg:
    """Light worker initialization (sent at spawn, before anything else is known).

    Only carries the CUDA device index - just enough to import cuRobo and pick a
    device. Everything else (cuRobo config params + robot profile + world) arrives
    via :class:`ConfigureMsg`, so the spawn + cuRobo import can overlap with the
    simulator build (see :meth:`CuroboPlanner.prewarm`).
    """

    device_index: int


@dataclass
class ConfigureMsg:
    """Send the cuRobo config + robot profile + collision world to a spawned worker.

    The worker stores these and builds/warms its planner lazily on the first
    plan. Sent once per control part (the profile is control-part-specific).
    """

    robot_config_path: str
    world_config_path: str | None
    tool_frame: str
    sim_joint_names: list[str]
    sim_to_curobo: dict[str, str]
    interpolation_dt: float
    collision_activation_distance: float
    collision_cache: dict | None
    multi_env: bool
    warmup_iterations: int


@dataclass
class PlanMsg:
    """A single plan request. Tensors are CPU float32.

    For ``move_type == "eef"``: ``goal_position`` / ``goal_quaternion`` carry the
    target already expressed in the cuRobo base frame and tool frame (the parent
    performs the sim-world -> curobo-base and tcp -> tool conversions). For
    ``move_type == "joint"``: ``goal_qpos`` carries the target in simulator
    control-part joint order; the worker reorders to cuRobo order.
    """

    batch_size: int
    move_type: str  # "eef" | "joint"
    start_qpos: torch.Tensor  # (B, D_sim) CPU
    max_attempts: int
    goal_position: torch.Tensor | None = None  # (B, 3) CPU, eef only
    goal_quaternion: torch.Tensor | None = None  # (B, 4) CPU, eef only
    goal_qpos: torch.Tensor | None = None  # (B, D_sim) CPU, joint only


@dataclass
class UpdateObstacleMsg:
    """Update one named obstacle's pose across the worker's cached planners.

    ``position`` / ``quaternion`` are already in the cuRobo base frame (the parent
    converts), batched as ``(B, 3)`` / ``(B, 4)``.
    """

    name: str
    position: torch.Tensor  # (B, 3) CPU
    quaternion: torch.Tensor  # (B, 4) CPU


@dataclass
class CloseMsg:
    """Sentinel asking the worker to exit its request loop cleanly."""


@dataclass
class PlanResultMsg:
    """A cuRobo plan result, with all tensors on CPU.

    Fields mirror the attributes the parent reads off the V2 ``v2_result``:
    ``success``, ``interpolated_trajectory.{position,joint_names,dt}``,
    ``interpolated_last_tstep``, ``total_time``.
    """

    success: torch.Tensor | float
    position: torch.Tensor
    joint_names: list[str]
    last_tstep: torch.Tensor
    dt: torch.Tensor | float | None
    total_time: torch.Tensor | float


# =============================================================================
# cuRobo bindings (lazily bound on first use inside the worker)
# =============================================================================


def _load_bindings() -> SimpleNamespace:
    """Import the cuRobo V2 facade types used by the worker."""
    planner_mod = importlib.import_module("curobo.motion_planner")
    batch_mod = importlib.import_module("curobo.batch_motion_planner")
    types_mod = importlib.import_module("curobo.types")
    return SimpleNamespace(
        MotionPlanner=planner_mod.MotionPlanner,
        MotionPlannerCfg=planner_mod.MotionPlannerCfg,
        BatchMotionPlanner=batch_mod.BatchMotionPlanner,
        JointState=types_mod.JointState,
        Pose=types_mod.Pose,
        GoalToolPose=types_mod.GoalToolPose,
        DeviceCfg=types_mod.DeviceCfg,
    )


# =============================================================================
# Scene model materialization (self-contained; the parent delegates scene
# cloning to the worker)
# =============================================================================


def _materialize_multi_env_scene_model(
    world_config_path: str | None, batch_size: int
) -> list[dict] | str | None:
    """Return the cuRobo scene model for one batch size.

    When ``multi_env`` is in use, cuRobo V2 infers the collision-world count from
    the length of a scene-model list, so a single mapping YAML is cloned for every
    batch row. When multi-env is off, the world path (or ``None``) is returned
    unchanged and cuRobo uses a single shared world.
    """
    if batch_size < 1:
        raise ValueError(
            f"multi-env cuRobo batch_size must be positive, got {batch_size}."
        )

    if world_config_path is None:
        # An initially empty collision world still needs one scene mapping per row.
        return [{} for _ in range(batch_size)]

    scene_path = Path(world_config_path)
    if not scene_path.is_absolute():
        content_mod = importlib.import_module("curobo.content")
        scene_path = Path(content_mod.get_scene_configs_path()) / scene_path
    with scene_path.open(encoding="utf-8") as scene_file:
        scene_model = yaml.safe_load(scene_file)

    if isinstance(scene_model, dict):
        return [deepcopy(scene_model) for _ in range(batch_size)]
    if isinstance(scene_model, list):
        if not scene_model or not all(isinstance(s, dict) for s in scene_model):
            raise ValueError(
                "A multi-env cuRobo scene YAML list must contain mapping worlds."
            )
        if len(scene_model) == 1:
            return [deepcopy(scene_model[0]) for _ in range(batch_size)]
        if len(scene_model) == batch_size:
            return [deepcopy(s) for s in scene_model]
        raise ValueError(
            "A multi-env cuRobo scene YAML list must have one world to clone or exactly "
            f"batch_size={batch_size} worlds; got {len(scene_model)}."
        )
    raise ValueError(
        f"A cuRobo scene YAML must be a mapping or list of mappings, got {type(scene_model).__name__}."
    )


# =============================================================================
# Worker executor
# =============================================================================


@dataclass
class _CuroboWorkerExecutor:
    """Stateful cuRobo executor running inside the worker process.

    Holds the fixed profile/world configuration received at init and a cache of
    built planners keyed by ``(batch_size, multi_env)`` (one planner per key, so
    batch and multi-env planning share one process). Each planner is built with
    ``use_cuda_graph=True`` and warmed up once so CUDA graphs are captured.
    """

    init_msg: InitMsg
    _bindings: SimpleNamespace = field(init=False)
    _device: torch.device = field(init=False)
    _planners: dict = field(init=False, default_factory=dict)
    _configure_msg: ConfigureMsg | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self._bindings = _load_bindings()
        self._device = torch.device("cuda", int(self.init_msg.device_index))

    def configure(self, msg: ConfigureMsg) -> None:
        """Store the robot profile + world (sent after the light init)."""
        self._configure_msg = msg

    @property
    def _cfg(self) -> ConfigureMsg:
        """The configured profile + world (must be set before any plan/build)."""
        if self._configure_msg is None:
            raise RuntimeError("cuRobo worker received a plan before ConfigureMsg.")
        return self._configure_msg

    # -- planner construction / caching ------------------------------------

    def _get_planner(self, batch_size: int):  # noqa: ANN202
        """Return a warmed planner for ``batch_size``, building it on first use."""
        cfg = self._cfg
        key = (int(batch_size), bool(cfg.multi_env))
        if key in self._planners:
            return self._planners[key]

        scene_model = (
            _materialize_multi_env_scene_model(cfg.world_config_path, int(batch_size))
            if cfg.multi_env
            else cfg.world_config_path
        )
        planner_cfg = self._bindings.MotionPlannerCfg.create(
            robot=cfg.robot_config_path,
            scene_model=scene_model,
            collision_cache=cfg.collision_cache,
            device_cfg=self._bindings.DeviceCfg(device=self._device),
            max_batch_size=int(batch_size),
            multi_env=bool(cfg.multi_env),
            optimizer_collision_activation_distance=cfg.collision_activation_distance,
            use_cuda_graph=True,
            interpolation_dt=float(cfg.interpolation_dt),
        )
        planner = (
            self._bindings.MotionPlanner(planner_cfg)
            if batch_size == 1
            else self._bindings.BatchMotionPlanner(planner_cfg)
        )
        self._validate_planner(planner)
        planner.warmup(
            enable_graph=True, num_warmup_iterations=int(cfg.warmup_iterations)
        )
        self._planners[key] = planner
        return planner

    def _validate_planner(self, planner) -> None:  # noqa: ANN001
        """Reject profiles whose joint/tool-frame mapping disagrees with the planner."""
        curobo_names = list(planner.joint_names)
        sim_to_curobo = self._cfg.sim_to_curobo
        mapped = [sim_to_curobo[name] for name in self._cfg.sim_joint_names]
        if len(mapped) != len(set(mapped)):
            raise ValueError(
                f"sim_to_curobo maps multiple sim joints to the same cuRobo joint: {mapped}."
            )
        missing = [n for n in mapped if n not in curobo_names]
        if missing:
            raise ValueError(
                f"cuRobo planner is missing mapped active joints {missing}; "
                f"planner joints are {curobo_names}."
            )
        unmapped = [n for n in curobo_names if n not in set(mapped)]
        if unmapped:
            raise ValueError(
                "cuRobo planner exposes joints outside the control part: "
                f"{unmapped}."
            )
        tool_frames = list(getattr(planner, "tool_frames", []))
        if tool_frames and self._cfg.tool_frame not in tool_frames:
            raise ValueError(
                f"tool_frame {self._cfg.tool_frame!r} is not available in the "
                f"cuRobo planner tool frames {tool_frames}."
            )

    # -- state / goal construction (runs entirely in the worker) --

    def _build_joint_state(self, sim_qpos: torch.Tensor, planner):  # noqa: ANN001
        """Reorder sim-order qpos to cuRobo order and wrap as a JointState."""
        curobo_names = list(planner.joint_names)
        sim_to_curobo = self._cfg.sim_to_curobo
        curobo_to_sim_idx = {
            cu_name: idx
            for idx, sim_name in enumerate(self._cfg.sim_joint_names)
            for cu_name in [sim_to_curobo[sim_name]]
        }
        if sim_qpos.dim() != 2 or sim_qpos.shape[1] != len(self._cfg.sim_joint_names):
            raise ValueError(
                "cuRobo start/goal qpos must have shape "
                f"(B, {len(self._cfg.sim_joint_names)}), got {tuple(sim_qpos.shape)}."
            )
        state = torch.zeros(
            sim_qpos.shape[0],
            len(curobo_names),
            device=self._device,
            dtype=torch.float32,
        )
        for i, cu_name in enumerate(curobo_names):
            state[:, i] = sim_qpos[:, curobo_to_sim_idx[cu_name]]
        return self._bindings.JointState.from_position(state, joint_names=curobo_names)

    def _build_pose_goal(
        self, position: torch.Tensor, quaternion: torch.Tensor
    ):  # noqa: ANN202
        """Build a single-tool-frame GoalToolPose from batched position/quaternion."""
        pose = self._bindings.Pose(position=position, quaternion=quaternion)
        return self._bindings.GoalToolPose.from_poses(
            {self._cfg.tool_frame: pose},
            ordered_tool_frames=[self._cfg.tool_frame],
            num_goalset=1,
        )

    # -- request handlers --------------------------------------------------

    def handle_plan(self, msg: PlanMsg) -> PlanResultMsg | None:
        """Run one cuRobo plan and return its result on CPU (or ``None``)."""
        planner = self._get_planner(msg.batch_size)
        start = msg.start_qpos.to(self._device, dtype=torch.float32)
        current_state = self._build_joint_state(start, planner)

        if msg.move_type == "eef":
            if msg.goal_position is None or msg.goal_quaternion is None:
                raise ValueError("EEF plan requires goal_position and goal_quaternion.")
            position = msg.goal_position.to(self._device, dtype=torch.float32)
            quaternion = msg.goal_quaternion.to(self._device, dtype=torch.float32)
            goal = self._build_pose_goal(position, quaternion)
            result = planner.plan_pose(
                goal, current_state, max_attempts=int(msg.max_attempts)
            )
        elif msg.move_type == "joint":
            if msg.goal_qpos is None:
                raise ValueError("JOINT plan requires goal_qpos.")
            goal_state = self._build_joint_state(
                msg.goal_qpos.to(self._device, dtype=torch.float32), planner
            )
            result = planner.plan_cspace(
                goal_state, current_state, max_attempts=int(msg.max_attempts)
            )
        else:
            raise ValueError(
                f"cuRobo worker does not support move_type {msg.move_type!r}."
            )

        if result is None:
            return None
        traj = result.interpolated_trajectory
        return PlanResultMsg(
            success=_to_cpu(result.success),
            position=_to_cpu(traj.position),
            joint_names=list(traj.joint_names),
            last_tstep=_to_cpu(result.interpolated_last_tstep),
            dt=_to_cpu(traj.dt),
            total_time=_to_cpu(result.total_time),
        )

    def handle_update_obstacle(self, msg: UpdateObstacleMsg) -> None:
        """Update one obstacle pose on every cached planner (shared or per-env)."""
        positions = msg.position.to(self._device, dtype=torch.float32)
        quaternions = msg.quaternion.to(self._device, dtype=torch.float32)
        batch_size = positions.shape[0]
        for planner in self._planners.values():
            if self._cfg.multi_env:
                planner_batch = _planner_batch_size(planner)
                if batch_size != planner_batch:
                    raise ValueError(
                        f"dynamic obstacle {msg.name!r} batch {batch_size} does not match "
                        f"this multi-env cuRobo planner's batch {planner_batch}."
                    )
                for env_idx in range(planner_batch):
                    pose = self._bindings.Pose(
                        position=positions[env_idx], quaternion=quaternions[env_idx]
                    )
                    planner.scene_collision_checker.update_obstacle_pose(
                        msg.name, pose, env_idx=env_idx
                    )
            else:
                if batch_size > 1 and not torch.allclose(
                    positions, positions[:1].expand_as(positions)
                ):
                    raise ValueError(
                        f"dynamic obstacle {msg.name!r} has different poses across a shared "
                        "cuRobo world. Enable world.multi_env for per-env worlds."
                    )
                pose = self._bindings.Pose(
                    position=positions[0], quaternion=quaternions[0]
                )
                planner.scene_collision_checker.update_obstacle_pose(
                    msg.name, pose, env_idx=0
                )

    def close(self) -> None:
        """Destroy every cached planner (best-effort)."""
        for planner in list(self._planners.values()):
            close_fn = getattr(planner, "close", None) or getattr(
                planner, "destroy", None
            )
            if close_fn is not None:
                try:
                    close_fn()
                except Exception:
                    pass
        self._planners.clear()


def _to_cpu(value):  # noqa: ANN001, ANN202
    """Move a tensor to CPU, leave non-tensors untouched (for IPC pickling)."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    return value


def _planner_batch_size(planner) -> int:  # noqa: ANN001
    """Best-effort batch size of a MotionPlanner / BatchMotionPlanner."""
    for attr in ("max_batch_size", "batch_size"):
        val = getattr(planner, attr, None)
        if isinstance(val, int):
            return val
    return 1


# =============================================================================
# Worker entry point
# =============================================================================


def worker_main(
    init_msg: InitMsg,
    request_queue,  # noqa: ANN001
    response_queue,  # noqa: ANN001
) -> None:
    """Worker process main loop.

    Constructs the executor, ACKs initialization, then serves requests until a
    :class:`CloseMsg` (or ``None``) arrives. Each handler is wrapped so an
    exception is returned as an ``("error", ...)`` response rather than killing
    the worker silently; a fatal (init/loop) failure is returned as
    ``("fatal", ...)`` so the parent can raise a clear error instead of hanging.
    """
    _worker_init()
    try:
        executor = _CuroboWorkerExecutor(init_msg)
        response_queue.put(("ok", None))
    except Exception as exc:  # noqa: BLE001
        response_queue.put(
            ("fatal", (type(exc).__name__, str(exc), traceback.format_exc()))
        )
        return

    while True:
        try:
            request = request_queue.get()
        except (EOFError, OSError):
            break
        if request is None or isinstance(request, CloseMsg):
            break
        try:
            if isinstance(request, ConfigureMsg):
                executor.configure(request)
                response = ("ok", None)
            elif isinstance(request, PlanMsg):
                response = ("ok", executor.handle_plan(request))
            elif isinstance(request, UpdateObstacleMsg):
                executor.handle_update_obstacle(request)
                response = ("ok", None)
            else:
                response = (
                    "error",
                    (
                        "ValueError",
                        f"unknown request type {type(request).__name__}",
                        "",
                    ),
                )
            response_queue.put(response)
        except Exception as exc:  # noqa: BLE001
            response_queue.put(
                ("error", (type(exc).__name__, str(exc), traceback.format_exc()))
            )

    try:
        executor.close()
    except Exception:
        pass
