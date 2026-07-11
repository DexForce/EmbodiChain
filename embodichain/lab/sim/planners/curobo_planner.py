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

"""Optional NVIDIA cuRobo V2 collision-aware motion-planning backend.

This module is importable without cuRobo installed. Only constructing a
:class:`CuroboPlanner` triggers the lazy V2 import (and the actionable error
when cuRobo/CUDA are unavailable). cuRobo V2 is an optional runtime dependency;
EmbodiChain never imports it at module load time.

The backend converts EmbodiChain's env-batched ``PlanState`` waypoints into
cuRobo V2 ``JointState`` / ``GoalToolPose`` calls, plans collision-aware
trajectories, and maps the result back into the standard ``PlanResult`` shape.
"""

from __future__ import annotations

import importlib
from dataclasses import MISSING, dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING

import torch

from embodichain.utils import configclass, logger
from embodichain.utils.math import quat_from_matrix

from .base_planner import (
    BasePlanner,
    BasePlannerCfg,
    PlanOptions,
    validate_plan_options,
)
from .utils import MoveType, PlanResult, PlanState

if TYPE_CHECKING:
    from typing import Any

__all__ = [
    "CuroboPlanOptions",
    "CuroboPlanner",
    "CuroboPlannerCfg",
    "CuroboRobotProfileCfg",
    "CuroboWorldCfg",
]


# cuRobo V2 installation extras documented at NVIDIA's installation page.
_CUROBO_INSTALL_URL = (
    "https://nvlabs.github.io/curobo/latest/getting-started/installation.html"
)


@configclass
class CuroboRobotProfileCfg:
    """Per-control-part mapping between an EmbodiChain robot and a cuRobo model.

    Each ``CuroboPlannerCfg.robot_profiles`` key is an EmbodiChain control-part
    name. The profile explicitly maps simulator joint names to cuRobo joint
    names so the backend never assumes simulator and cuRobo joint indices agree.
    Non-controlled joints (e.g. gripper joints) present in the cuRobo model are
    pinned to ``fixed_joint_positions``.
    """

    robot_config_path: str = MISSING
    """Path/identifier of the cuRobo V2 robot profile (e.g. ``"franka.yml"``)."""

    sim_to_curobo_joint_names: dict[str, str] = MISSING
    """Mapping from simulator joint names to cuRobo joint names for this part."""

    active_joint_names: list[str] | None = None
    """Optional explicit cuRobo active-joint ordering, validated against the backend."""

    fixed_joint_positions: dict[str, float] = {}
    """cuRobo joint names pinned to a constant value (e.g. gripper finger joints)."""

    base_link_name: str | None = None
    """cuRobo robot base link name."""

    tool_frame_name: str | None = None
    """cuRobo tool frame name used as the planning target."""


@configclass
class CuroboWorldCfg:
    """Static collision-world configuration for the cuRobo backend."""

    world_config_path: str | None = None
    """Path/identifier of a cuRobo V2 scene profile (cuboid/mesh/voxel obstacles)."""

    collision_cache: dict[str, int] = {"cuboid": 8, "mesh": 2, "voxel": 1}
    """Per-geometry cache capacity created before world updates."""

    dynamic_obstacle_names: list[str] = []
    """Obstacle names whose poses may be updated between plans."""

    multi_env: bool = False
    """Whether the cuRobo world is shared across multiple environments."""


@configclass
class CuroboPlannerCfg(BasePlannerCfg):
    """Configuration for the cuRobo V2 planner backend."""

    planner_type: str = "curobo"

    robot_profiles: dict[str, CuroboRobotProfileCfg] = MISSING
    """Control-part name -> profile mapping. The first release supports one part per request."""

    world: CuroboWorldCfg = CuroboWorldCfg()
    """Static collision-world configuration."""

    warmup: bool = True
    """Whether to warm each cached planner once at construction time."""

    collision_activation_distance: float = 0.01
    """cuRobo collision activation distance (optimizer setting)."""

    max_attempts: int = 5
    """Default per-plan cuRobo attempt count."""

    max_planning_time: float | None = None
    """Post-plan validation budget (seconds). ``None`` skips the timing check."""

    use_cuda_graph: bool = True
    """Whether cuRobo may use CUDA graphs internally."""

    interpolation_dt: float = 0.025
    """Interpolation step (seconds) used by cuRobo and as a dt fallback."""


@configclass
class CuroboPlanOptions(PlanOptions):
    """Per-plan options for :class:`CuroboPlanner`.

    ``start_qpos`` and ``control_part`` are populated from the
    :class:`~embodichain.lab.sim.planners.motion_generator.MotionGenOptions`
    runtime context via :meth:`CuroboPlanner.with_motion_context`.
    """

    start_qpos: torch.Tensor | None = None
    """Planning start joint configuration ``(B, controlled_dof)``."""

    control_part: str | None = None
    """EmbodiChain control-part name to plan for."""

    dynamic_obstacle_poses: dict[str, torch.Tensor] | None = None
    """Per-obstacle world poses ``(B, 4, 4)`` keyed by configured name."""

    max_attempts: int | None = None
    """Per-plan override of ``CuroboPlannerCfg.max_attempts``."""


# =============================================================================
# Pure conversion / validation helpers (no cuRobo import required)
# =============================================================================


def _reorder_by_names(
    values: torch.Tensor,
    from_names: list[str],
    to_names: list[str],
) -> torch.Tensor:
    """Reorder the trailing joint dimension of ``values`` by name.

    Args:
        values: Tensor whose last dimension is ordered by ``from_names``.
        from_names: Joint names describing the current trailing-axis order.
        to_names: Desired joint-name order; must be a permutation of
            ``from_names``.

    Returns:
        Tensor with the trailing axis reordered to ``to_names``.

    Raises:
        ValueError: If the two name sets are not equal as sets.
    """
    if sorted(from_names) != sorted(to_names):
        raise ValueError(
            f"Cannot reorder joints: source names {from_names} and target names "
            f"{to_names} are not the same set."
        )
    if from_names == to_names:
        return values
    perm = [from_names.index(name) for name in to_names]
    return values[..., perm]


def _matrix_to_position_quaternion(
    matrix: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a batched homogeneous pose to cuRobo ``(position, quaternion)``.

    Args:
        matrix: Batched homogeneous transforms of shape ``(B, 4, 4)``.

    Returns:
        Tuple of ``(position (B, 3), quaternion (B, 4))`` where the quaternion
        is in cuRobo's ``(w, x, y, z)`` convention.

    Raises:
        ValueError: If ``matrix`` is not a ``(B, 4, 4)`` tensor.
    """
    if matrix.dim() != 3 or matrix.shape[-2:] != (4, 4):
        raise ValueError(
            f"Expected (B, 4, 4) pose matrices, got shape {tuple(matrix.shape)}."
        )
    matrix = matrix.to(dtype=torch.float32)
    position = matrix[:, :3, 3]
    quaternion = quat_from_matrix(matrix[:, :3, :3])  # wxyz
    return position, quaternion


def _validate_dynamic_obstacles(
    poses: dict[str, torch.Tensor] | None,
    allowed_names: list[str],
) -> None:
    """Validate dynamic-obstacle pose names and shapes.

    Args:
        poses: Mapping of obstacle name -> pose tensor. ``None`` is a no-op.
        allowed_names: Obstacle names declared in :class:`CuroboWorldCfg`.

    Raises:
        ValueError: If a name is not configured, or a pose is not ``(B, 4, 4)``.
    """
    if poses is None:
        return
    for name, pose in poses.items():
        if name not in allowed_names:
            raise ValueError(
                f"unknown obstacle '{name}'; configured dynamic obstacles: "
                f"{allowed_names}."
            )
        if (
            not isinstance(pose, torch.Tensor)
            or pose.dim() != 3
            or pose.shape[-2:] != (4, 4)
        ):
            got = tuple(pose.shape) if isinstance(pose, torch.Tensor) else type(pose)
            raise ValueError(
                f"dynamic obstacle '{name}' pose must be (B, 4, 4), got {got}."
            )


# =============================================================================
# Lazy cuRobo V2 binding acquisition
# =============================================================================


def _require_curobo() -> "Any":
    """Lazily import and bundle the cuRobo V2 public facade types.

    Returns:
        A namespace exposing ``MotionPlanner``, ``MotionPlannerCfg``,
        ``BatchMotionPlanner``, ``JointState``, ``Pose``, and ``GoalToolPose``.

    Raises:
        ImportError: If cuRobo V2 is not installed, with an actionable message
            naming NVIDIA's CUDA-matched extras.
    """
    try:
        planner_mod = importlib.import_module("curobo.motion_planner")
        batch_mod = importlib.import_module("curobo.batch_motion_planner")
        types_mod = importlib.import_module("curobo.types")
    except ModuleNotFoundError as exc:
        raise ImportError(
            "cuRobo V2 is required for the 'curobo' planner but was not found. "
            "Install it using NVIDIA's CUDA-matched extras, e.g. "
            "`pip install .[cu12]` or `pip install .[cu13]` "
            "(also `.[cu12-torch]` / `.[cu13-torch]`). "
            f"See {_CUROBO_INSTALL_URL} for details."
        ) from exc
    return SimpleNamespace(
        MotionPlanner=planner_mod.MotionPlanner,
        MotionPlannerCfg=planner_mod.MotionPlannerCfg,
        BatchMotionPlanner=batch_mod.BatchMotionPlanner,
        JointState=types_mod.JointState,
        Pose=types_mod.Pose,
        GoalToolPose=types_mod.GoalToolPose,
    )


# =============================================================================
# CuroboPlanner
# =============================================================================


class CuroboPlanner(BasePlanner):
    r"""cuRobo V2 collision-aware motion-planning backend.

    The planner lazily imports cuRobo V2 at construction time, builds and caches
    a V2 ``MotionPlanner`` (single-environment) or ``BatchMotionPlanner``
    (multi-environment) per ``(control_part, batch_size, multi_env)`` key, and
    converts standard batched :class:`PlanState` inputs into V2 planning calls.

    Cartesian (``EEF_MOVE``) targets are forwarded to cuRobo unchanged - the
    backend performs its own collision-aware IK and trajectory optimization, so
    EmbodiChain pre-interpolation is disabled (``preinterpolate_targets=False``)
    and returned collision-checked samples are preserved
    (``preserve_plan_samples=True``).

    Args:
        cfg: Configuration for the cuRobo planner.

    Raises:
        ImportError: If cuRobo V2 is not installed.
        RuntimeError: If the robot is not on a CUDA device.
        ValueError: If ``robot_uid`` is missing or the robot is not found.
    """

    preinterpolate_targets = False
    preserve_plan_samples = True

    def __init__(self, cfg: CuroboPlannerCfg) -> None:
        super().__init__(cfg)
        self.cfg: CuroboPlannerCfg = cfg
        if self.device.type != "cuda":
            raise RuntimeError(
                "cuRobo V2 requires a CUDA device, but robot "
                f"'{cfg.robot_uid}' is on {self.device}. Move the simulation "
                "to a CUDA device before constructing the curobo planner."
            )
        self._bindings = _require_curobo()
        # Cached V2 backends keyed by (control_part, batch_size, multi_env).
        self._backend_cache: dict[tuple, "_CuroboBackend"] = {}

    def default_plan_options(self) -> CuroboPlanOptions:
        """Return backend-default planning options."""
        return CuroboPlanOptions()

    def with_motion_context(
        self,
        options: PlanOptions,
        *,
        start_qpos: torch.Tensor | None,
        control_part: str | None,
    ) -> CuroboPlanOptions:
        """Forward MotionGenerator context into :class:`CuroboPlanOptions`."""
        if not isinstance(options, CuroboPlanOptions):
            logger.log_error("CuroboPlanner requires CuroboPlanOptions", TypeError)
        if options.start_qpos is None:
            options.start_qpos = start_qpos
        if options.control_part is None:
            options.control_part = control_part
        return options

    @validate_plan_options(options_cls=CuroboPlanOptions)
    def plan(
        self,
        target_states: list[PlanState],
        options: CuroboPlanOptions = CuroboPlanOptions(),
    ) -> PlanResult:
        r"""Plan a collision-aware trajectory through ``target_states``.

        ``EEF_MOVE`` waypoints are forwarded to cuRobo's ``plan_pose``;
        ``JOINT_MOVE`` waypoints use ``plan_cspace``. Multi-waypoint plans
        chain sequentially: each segment starts from the previous segment's
        final sample, and the returned collision-checked samples are
        concatenated without resampling.

        Args:
            target_states: List of :class:`PlanState` waypoints. ``EEF_MOVE``
                entries carry ``xpos`` ``(B, 4, 4)``; ``JOINT_MOVE`` entries
                carry ``qpos`` ``(B, controlled_dof)``.
            options: :class:`CuroboPlanOptions` carrying the runtime context.

        Returns:
            :class:`PlanResult` with env-batched tensors. ``success`` is
            ``(B,)`` bool; ``positions`` is ``(B, N, controlled_dof)``;
            ``dt`` is ``(B, N)``; ``duration`` is ``(B,)``. Failed environments
            (planning failure or ``total_time`` over budget) hold ``start_qpos``.
        """
        if not target_states:
            return PlanResult(
                success=torch.zeros(0, dtype=torch.bool, device=self.device),
                positions=None,
            )
        control_part, profile = self._resolve_profile(options)
        start = self._resolve_start_qpos(options.start_qpos, control_part)
        backend = self._get_backend(profile, control_part, start.shape[0])
        self.update_dynamic_obstacles(options.dynamic_obstacle_poses, backend)
        return self._plan_segments(target_states, start, backend, options)

    # ------------------------------------------------------------------
    # Profile / start resolution
    # ------------------------------------------------------------------

    def _resolve_profile(
        self, options: CuroboPlanOptions
    ) -> tuple[str, CuroboRobotProfileCfg]:
        """Resolve the requested control part and its cuRobo profile."""
        control_part = options.control_part
        if control_part is None:
            logger.log_error("CuroboPlanOptions.control_part is required.", ValueError)
        if control_part not in self.cfg.robot_profiles:
            logger.log_error(
                f"No cuRobo profile for control part '{control_part}'. "
                f"Configured parts: {sorted(self.cfg.robot_profiles)}.",
                ValueError,
            )
        return control_part, self.cfg.robot_profiles[control_part]

    def _resolve_start_qpos(
        self, start_qpos: torch.Tensor | None, control_part: str
    ) -> torch.Tensor:
        """Resolve the planning start qpos into ``(B, controlled_dof)``."""
        if start_qpos is None:
            start_qpos = self.robot.get_qpos(name=control_part)
        start_qpos = torch.as_tensor(
            start_qpos, dtype=torch.float32, device=self.device
        )
        if start_qpos.dim() == 1:
            start_qpos = start_qpos.unsqueeze(0)
        return start_qpos

    # ------------------------------------------------------------------
    # Backend construction / caching
    # ------------------------------------------------------------------

    def _get_backend(
        self,
        profile: CuroboRobotProfileCfg,
        control_part: str,
        batch_size: int,
    ) -> "_CuroboBackend":
        """Return a cached V2 backend for ``(control_part, batch_size, multi_env)``."""
        multi_env = self.cfg.world.multi_env
        key = (control_part, int(batch_size), bool(multi_env))
        if key in self._backend_cache:
            return self._backend_cache[key]

        world_cfg = self.cfg.world
        collision_cache = (
            dict(world_cfg.collision_cache) if world_cfg.collision_cache else None
        )
        planner_cfg = self._bindings.MotionPlannerCfg.create(
            robot=profile.robot_config_path,
            scene_model=world_cfg.world_config_path,
            collision_cache=collision_cache,
            max_batch_size=int(batch_size),
            multi_env=bool(multi_env),
            optimizer_collision_activation_distance=self.cfg.collision_activation_distance,
            use_cuda_graph=bool(self.cfg.use_cuda_graph),
            interpolation_dt=float(self.cfg.interpolation_dt),
        )
        if batch_size == 1:
            planner = self._bindings.MotionPlanner(planner_cfg)
        else:
            planner = self._bindings.BatchMotionPlanner(planner_cfg)

        if profile.active_joint_names is not None:
            expected = list(profile.active_joint_names)
            actual = list(planner.joint_names)
            if expected != actual:
                logger.log_error(
                    f"active_joint_names {expected} do not match cuRobo model "
                    f"joints {actual} (missing/duplicate/out-of-order).",
                    ValueError,
                )

        backend = _CuroboBackend(
            planner=planner,
            tool_frame=profile.tool_frame_name,
            profile=profile,
            batch_size=int(batch_size),
        )
        if self.cfg.warmup:
            planner.warmup()
        self._backend_cache[key] = backend
        return backend

    # ------------------------------------------------------------------
    # Segment planning
    # ------------------------------------------------------------------

    def _plan_segments(
        self,
        target_states: list[PlanState],
        start: torch.Tensor,
        backend: "_CuroboBackend",
        options: CuroboPlanOptions,
    ) -> PlanResult:
        """Plan each waypoint segment sequentially and assemble a PlanResult."""
        B = start.shape[0]
        D = start.shape[1]
        max_attempts = (
            options.max_attempts
            if options.max_attempts is not None
            else self.cfg.max_attempts
        )
        per_env_samples: list[list[torch.Tensor]] = [[] for _ in range(B)]
        per_env_dt: list[list[torch.Tensor]] = [[] for _ in range(B)]
        alive = torch.ones(B, dtype=torch.bool, device=self.device)
        current = start.clone()

        for seg_idx, target in enumerate(target_states):
            current_state = self._to_curobo_joint_state(current, backend)
            if target.move_type == MoveType.EEF_MOVE:
                if target.xpos is None:
                    logger.log_error(
                        f"Segment {seg_idx} EEF_MOVE target missing xpos.",
                        ValueError,
                    )
                goal = self._to_curobo_pose_goal(target.xpos, backend)
                v2_result = backend.planner.plan_pose(
                    goal, current_state, max_attempts=max_attempts
                )
            elif target.move_type == MoveType.JOINT_MOVE:
                if target.qpos is None:
                    logger.log_error(
                        f"Segment {seg_idx} JOINT_MOVE target missing qpos.",
                        ValueError,
                    )
                goal_state = self._to_curobo_joint_goal(target.qpos, backend)
                v2_result = backend.planner.plan_cspace(
                    goal_state, current_state, max_attempts=max_attempts
                )
            else:
                logger.log_error(
                    f"cuRobo does not support move_type {target.move_type}.",
                    ValueError,
                )

            seg_success, seg_positions, seg_dt = self._extract_segment(
                v2_result, backend
            )
            seg_success = seg_success.to(self.device) & alive
            if self.cfg.max_planning_time is not None:
                total_time = self._extract_total_time(v2_result, B)
                over = total_time > float(self.cfg.max_planning_time)
                seg_success = seg_success & (~over)

            for b in range(B):
                if seg_idx == 0:
                    per_env_samples[b].append(seg_positions[b])
                    per_env_dt[b].append(seg_dt[b])
                elif alive[b]:
                    # Drop the duplicate junction sample (== previous segment's
                    # final) so collision-checked samples are not duplicated.
                    per_env_samples[b].append(seg_positions[b, 1:])
                    per_env_dt[b].append(seg_dt[b, 1:])
                else:
                    per_env_samples[b].append(seg_positions[b, -1:])
                    per_env_dt[b].append(seg_dt[b, -1:])
                if seg_success[b]:
                    current[b] = seg_positions[b, -1]
            alive = seg_success

        return self._assemble_result(per_env_samples, per_env_dt, start, alive, B, D)

    def _extract_segment(
        self, v2_result: "Any", backend: "_CuroboBackend"
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract ``(success, positions, dt)`` for one V2 planning result.

        ``positions`` is ``(B, T, controlled_dof)`` in simulator control-part
        order, trimmed to each env's last valid timestep and padded to a
        rectangular batch by repeating the last valid sample.
        """
        success = torch.as_tensor(v2_result.success)
        if success.dim() == 2:
            success = success.squeeze(-1)
        success = success.to(torch.bool).to(self.device)

        traj = v2_result.interpolated_trajectory
        position = torch.as_tensor(traj.position)
        if position.dim() == 4:
            position = position[:, 0, :, :]  # select seed 0: (B, T, D_full)

        last_tstep = torch.as_tensor(v2_result.interpolated_last_tstep)
        if last_tstep.dim() == 2:
            last_tstep = last_tstep.squeeze(-1)

        B, T, _ = position.shape
        max_len = max(int((last_tstep + 1).max().item()), 1)
        full = torch.zeros(
            B, max_len, position.shape[-1], device=self.device, dtype=torch.float32
        )
        for b in range(B):
            length = min(int(last_tstep[b].item()) + 1, T, max_len)
            full[b, :length] = position[b, :length].float().to(self.device)
            if length < max_len:
                full[b, length:] = position[b, length - 1].float().to(self.device)

        seg_positions = self._map_curobo_to_sim(full, traj.joint_names, backend.profile)
        seg_dt = self._extract_dt(v2_result, traj, max_len, B)
        return success, seg_positions, seg_dt

    def _map_curobo_to_sim(
        self,
        full_positions: torch.Tensor,
        curobo_joint_names: list[str],
        profile: CuroboRobotProfileCfg,
    ) -> torch.Tensor:
        """Map a full cuRobo trajectory to simulator control-part joint order."""
        sim_to_curobo = profile.sim_to_curobo_joint_names
        cols: list[int] = []
        for sim_name in sim_to_curobo:
            cu_name = sim_to_curobo[sim_name]
            if cu_name not in curobo_joint_names:
                logger.log_error(
                    f"cuRobo trajectory is missing active joint '{cu_name}' "
                    f"(mapped from sim joint '{sim_name}'); trajectory joints: "
                    f"{list(curobo_joint_names)}.",
                    ValueError,
                )
            cols.append(curobo_joint_names.index(cu_name))
        return full_positions[..., cols].to(dtype=torch.float32)

    def _extract_dt(
        self, v2_result: "Any", traj: "Any", max_len: int, B: int
    ) -> torch.Tensor:
        """Derive ``(B, max_len)`` per-sample dt from the V2 trajectory."""
        raw_dt = getattr(traj, "dt", None)
        dt = None
        if isinstance(raw_dt, torch.Tensor):
            if raw_dt.dim() == 1:
                dt = raw_dt.unsqueeze(0).expand(B, -1)
            elif raw_dt.dim() == 2:
                dt = raw_dt
        if dt is None:
            return torch.full(
                (B, max_len),
                float(self.cfg.interpolation_dt),
                device=self.device,
                dtype=torch.float32,
            )
        T = dt.shape[-1]
        out = torch.zeros(B, max_len, device=self.device, dtype=torch.float32)
        length = min(T, max_len)
        out[:, :length] = dt[:, :length].to(self.device)
        return out

    def _extract_total_time(self, v2_result: "Any", B: int) -> torch.Tensor:
        """Return a ``(B,)`` total planning time tensor for budget validation."""
        tt = v2_result.total_time
        if isinstance(tt, torch.Tensor):
            if tt.dim() == 0:
                return tt.unsqueeze(0).expand(B).to(self.device)
            if tt.dim() == 2:
                tt = tt.squeeze(-1)
            return tt[:B].to(self.device)
        return torch.full((B,), float(tt), device=self.device)

    def _assemble_result(
        self,
        per_env_samples: list[list[torch.Tensor]],
        per_env_dt: list[list[torch.Tensor]],
        start: torch.Tensor,
        alive: torch.Tensor,
        B: int,
        D: int,
    ) -> PlanResult:
        """Concatenate per-env segment samples into a rectangular PlanResult."""
        env_lengths: list[int] = []
        for b in range(B):
            if alive[b]:
                env_lengths.append(sum(s.shape[0] for s in per_env_samples[b]))
            else:
                env_lengths.append(1)
        max_len = max(env_lengths) if env_lengths else 1

        positions = torch.zeros(B, max_len, D, device=self.device, dtype=torch.float32)
        dt = torch.zeros(B, max_len, device=self.device, dtype=torch.float32)
        for b in range(B):
            if alive[b]:
                cat = torch.cat(per_env_samples[b], dim=0)
                cat_dt = torch.cat(per_env_dt[b], dim=0)
                length = cat.shape[0]
                positions[b, :length] = cat
                positions[b, length:] = cat[-1]
                dt[b, : min(cat_dt.shape[0], max_len)] = cat_dt[:max_len]
            else:
                positions[b, :1] = start[b]
                positions[b, 1:] = start[b]
        duration = dt.sum(dim=1)
        return PlanResult(
            success=alive,
            positions=positions,
            dt=dt,
            duration=duration,
        )

    # ------------------------------------------------------------------
    # cuRobo state / goal construction
    # ------------------------------------------------------------------

    def _to_curobo_joint_state(
        self, current: torch.Tensor, backend: "_CuroboBackend"
    ) -> "Any":
        """Build a full cuRobo ``JointState`` from a sim-order control-part qpos.

        Active joints are filled from ``current`` (reordered to cuRobo order);
        non-active joints present in the cuRobo model are pinned to
        ``fixed_joint_positions``.
        """
        profile = backend.profile
        curobo_names = list(backend.planner.joint_names)
        sim_to_curobo = profile.sim_to_curobo_joint_names
        curobo_to_sim_idx = {
            cu_name: idx
            for idx, sim_name in enumerate(sim_to_curobo)
            for cu_name in [sim_to_curobo[sim_name]]
        }
        B = current.shape[0]
        state = torch.zeros(
            B, len(curobo_names), device=self.device, dtype=torch.float32
        )
        for i, cu_name in enumerate(curobo_names):
            if cu_name in curobo_to_sim_idx:
                state[:, i] = current[:, curobo_to_sim_idx[cu_name]]
            elif cu_name in profile.fixed_joint_positions:
                state[:, i] = float(profile.fixed_joint_positions[cu_name])
        return self._bindings.JointState.from_position(state, joint_names=curobo_names)

    def _to_curobo_pose_goal(
        self, xpos: torch.Tensor, backend: "_CuroboBackend"
    ) -> "Any":
        """Build a cuRobo ``GoalToolPose`` from a batched world-frame pose."""
        xpos = torch.as_tensor(xpos, device=self.device, dtype=torch.float32)
        position, quaternion = _matrix_to_position_quaternion(xpos)
        pose = self._bindings.Pose(position=position, quaternion=quaternion)
        return self._bindings.GoalToolPose.from_poses(
            {backend.tool_frame: pose},
            ordered_tool_frames=[backend.tool_frame],
            num_goalset=1,
        )

    def _to_curobo_joint_goal(
        self, qpos: torch.Tensor, backend: "_CuroboBackend"
    ) -> "Any":
        """Build a cuRobo c-space goal state from a sim-order target qpos."""
        qpos = torch.as_tensor(qpos, dtype=torch.float32, device=self.device)
        if qpos.dim() == 1:
            qpos = qpos.unsqueeze(0)
        full_state = self._to_curobo_joint_state(qpos, backend)
        return self._bindings.JointState.from_position(
            full_state, joint_names=list(backend.planner.joint_names)
        )

    # ------------------------------------------------------------------
    # Collision world + lifecycle
    # ------------------------------------------------------------------

    def update_dynamic_obstacles(
        self,
        poses: dict[str, torch.Tensor] | None,
        backend: "_CuroboBackend | None" = None,
    ) -> None:
        """Update named dynamic obstacle poses on the cuRobo collision world.

        Args:
            poses: Mapping of obstacle name -> ``(B, 4, 4)`` world pose. ``None``
                is a no-op.
            backend: Specific backend to update. If ``None``, updates all cached
                backends.
        """
        if poses is None:
            return
        _validate_dynamic_obstacles(poses, list(self.cfg.world.dynamic_obstacle_names))
        backends = (
            [backend] if backend is not None else list(self._backend_cache.values())
        )
        for name, pose_tensor in poses.items():
            pose_tensor = torch.as_tensor(
                pose_tensor, device=self.device, dtype=torch.float32
            )
            position, quaternion = _matrix_to_position_quaternion(pose_tensor)
            B = pose_tensor.shape[0]
            for b in range(B):
                pose = self._bindings.Pose(
                    position=position[b], quaternion=quaternion[b]
                )
                for be in backends:
                    be.planner.scene_collision_checker.update_obstacle_pose(
                        name, pose, env_idx=b
                    )

    def close(self) -> None:
        """Destroy every cached cuRobo planner and clear the cache."""
        for backend in list(self._backend_cache.values()):
            planner = backend.planner
            close_fn = getattr(planner, "close", None) or getattr(
                planner, "destroy", None
            )
            if close_fn is not None:
                try:
                    close_fn()
                except Exception:
                    pass
        self._backend_cache.clear()

    def __del__(self) -> None:  # pragma: no cover - best-effort GC cleanup
        try:
            self.close()
        except Exception:
            pass


@dataclass
class _CuroboBackend:
    """Internal bundle of a cached V2 planner and its EmbodiChain-side metadata."""

    planner: "Any"
    tool_frame: str | None
    profile: CuroboRobotProfileCfg
    batch_size: int
