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
from copy import deepcopy
from dataclasses import MISSING, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import torch
import yaml

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
    The initial release requires the loaded cuRobo planner's active joints to
    match the selected control part exactly. Lock non-controlled joints (for
    example gripper joints) in the cuRobo V2 robot profile so they are not
    exposed as active planner joints. The simulator values of those joints
    must remain equal to the V2 profile's ``lock_joints`` values while a plan
    is executed; the adapter intentionally preserves non-control simulator
    joints in the full-DoF atomic-action output.
    """

    robot_config_path: str = MISSING
    """Path/identifier of the cuRobo V2 robot profile (e.g. ``"franka.yml"``)."""

    sim_to_curobo_joint_names: dict[str, str] = MISSING
    """Mapping from simulator joint names to cuRobo joint names for this part."""

    active_joint_names: list[str] | None = None
    """Optional explicit cuRobo active-joint ordering, validated against the backend."""

    base_link_name: str | None = None
    """Expected cuRobo robot base link, validated against the loaded V2 model.

    This is a consistency check. Use ``sim_base_to_curobo_base`` when the
    simulator control-part base and this V2 base use different fixed frames.
    Static collision YAML remains authored in the cuRobo base/world frame.
    """

    sim_base_to_curobo_base: list[list[float]] | None = None
    """Fixed transform from the simulator control-part base to the cuRobo base.

    The adapter uses this transform together with the live simulator base pose
    to convert simulator-world Cartesian goals and dynamic obstacle poses into
    cuRobo's base frame. ``None`` means the two base frames coincide.
    """

    sim_base_link_name: str | None = None
    """Simulator link physically equivalent to the control-part base.

    When omitted, the adapter uses the EmbodiChain solver's ``root_link_name``.
    Set this explicitly for a cuRobo-planned control part that has no local
    EmbodiChain IK solver.
    """

    tool_frame_name: str | None = None
    """cuRobo tool frame name used as the planning target."""

    tool_frame_to_tcp: list[list[float]] | None = None
    """Fixed transform from the cuRobo tool frame to the simulator TCP frame.

    EmbodiChain Cartesian targets are expressed in the simulator TCP frame.
    When that frame is not identical to ``tool_frame_name``, provide this
    homogeneous transform so the adapter can convert the target before it is
    sent to cuRobo. ``None`` means the two frames are identical.
    """


@configclass
class CuroboWorldCfg:
    """Static collision-world configuration for the cuRobo backend."""

    world_config_path: str | None = None
    """Path/identifier of a cuRobo V2 scene profile (cuboid/mesh/voxel obstacles)."""

    collision_cache: dict[str, int | dict[str, int | float | list[float]]] = {
        "cuboid": 8,
        "mesh": 2,
    }
    """Per-geometry cache capacity created before world updates.

    cuRobo V2 accepts integer ``cuboid`` and ``mesh`` capacities. A ``voxel``
    cache, when needed for dynamic voxel worlds, must instead be a dictionary
    with V2's ``layers``, ``dims``, and ``voxel_size`` fields.
    """

    dynamic_obstacle_names: list[str] = []
    """Obstacle names whose poses may be updated between plans."""

    multi_env: bool = False
    """Whether cuRobo allocates one collision-world instance per environment.

    ``False`` shares one world and therefore requires equal rebased dynamic
    obstacle poses across batch rows. ``True`` materializes one V2 scene model
    per batch row, supporting independently updated obstacle poses for each
    environment. A mapping YAML is cloned for every row; a top-level YAML list
    may instead provide one explicit mapping per row.
    """


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

    use_cuda_graph: bool = False
    """Whether cuRobo may use CUDA graphs internally.

    Disabled by default because CUDA graph capture can conflict with DexSim's
    GPU physics stream. Enable only after validating the local stream setup.
    """

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
    if (
        len(from_names) != len(set(from_names))
        or len(to_names) != len(set(to_names))
        or sorted(from_names) != sorted(to_names)
    ):
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
    # V2's Pose inverse/update kernels require contiguous float32 tensors.
    # Column/rotation slices of a homogeneous transform are views with strides,
    # so materialize them at the adapter boundary rather than relying on a
    # caller-specific layout.
    position = matrix[:, :3, 3].contiguous()
    quaternion = quat_from_matrix(matrix[:, :3, :3]).contiguous()  # wxyz
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
        DeviceCfg=types_mod.DeviceCfg,
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
    supports_joint_move = True

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

    def _materialize_multi_env_scene_model(
        self, world_config_path: str | None, batch_size: int
    ) -> list[dict]:
        """Return exactly one cuRobo V2 scene mapping for every batch row.

        cuRobo V2 infers the collision-world count from the length of a scene
        model list; ``multi_env=True`` and ``max_batch_size`` alone do not
        allocate per-environment collision data. A single mapping YAML is
        therefore cloned for every row. A top-level YAML list supports
        explicitly different static worlds, but it must contain either one
        mapping (cloned) or exactly ``batch_size`` mappings.

        Args:
            world_config_path: cuRobo scene identifier/path, or ``None`` for
                an initially empty collision world.
            batch_size: Number of simultaneous planning environments.

        Returns:
            A list of independent V2 scene mappings with length ``batch_size``.

        Raises:
            ValueError: If the YAML cannot be loaded or has an incompatible
                top-level structure/count.
        """
        if batch_size < 1:
            logger.log_error(
                f"multi-env cuRobo batch_size must be positive, got {batch_size}.",
                ValueError,
            )

        if world_config_path is None:
            # Even an initially empty collision world needs one scene mapping
            # per row. Otherwise V2's SceneCollisionCfg defaults to num_envs=1
            # despite multi_env=True and later dynamic updates fail for row > 0.
            return [{} for _ in range(batch_size)]

        scene_path = Path(world_config_path)
        if not scene_path.is_absolute():
            content_mod = importlib.import_module("curobo.content")
            scene_path = Path(content_mod.get_scene_configs_path()) / scene_path
        try:
            with scene_path.open(encoding="utf-8") as scene_file:
                scene_model = yaml.safe_load(scene_file)
        except (OSError, yaml.YAMLError) as exc:
            logger.log_error(
                f"Unable to load cuRobo V2 scene configuration "
                f"'{world_config_path}': {exc}",
                ValueError,
            )
            raise AssertionError("unreachable") from exc

        if isinstance(scene_model, dict):
            return [deepcopy(scene_model) for _ in range(batch_size)]
        if isinstance(scene_model, list):
            if not scene_model or not all(
                isinstance(scene, dict) for scene in scene_model
            ):
                logger.log_error(
                    "A multi-env cuRobo scene YAML list must contain one or more "
                    "mapping worlds.",
                    ValueError,
                )
            if len(scene_model) == 1:
                return [deepcopy(scene_model[0]) for _ in range(batch_size)]
            if len(scene_model) == batch_size:
                return [deepcopy(scene) for scene in scene_model]
            logger.log_error(
                "A multi-env cuRobo scene YAML list must have one world to clone "
                f"or exactly batch_size={batch_size} worlds; got {len(scene_model)}.",
                ValueError,
            )
        logger.log_error(
            "A cuRobo V2 scene YAML must contain a mapping world or a list of "
            f"mapping worlds, got {type(scene_model).__name__}.",
            ValueError,
        )
        raise AssertionError("unreachable")

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

        sim_joint_names = self._resolve_sim_joint_names(control_part)
        world_cfg = self.cfg.world
        collision_cache = (
            dict(world_cfg.collision_cache) if world_cfg.collision_cache else None
        )
        curobo_device = self.device
        if curobo_device.index is None:
            # Warp's V2 collision cache indexes CUDA devices by integer and
            # cannot consume the indexless torch.device("cuda") used by
            # DexSim's default CUDA configuration.
            curobo_device = torch.device(f"cuda:{torch.cuda.current_device()}")
        scene_model = world_cfg.world_config_path
        if multi_env:
            scene_model = self._materialize_multi_env_scene_model(
                world_cfg.world_config_path, int(batch_size)
            )
        planner_cfg = self._bindings.MotionPlannerCfg.create(
            robot=profile.robot_config_path,
            scene_model=scene_model,
            collision_cache=collision_cache,
            device_cfg=self._bindings.DeviceCfg(device=curobo_device),
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

        self._validate_profile_joint_names(
            profile, sim_joint_names, list(planner.joint_names)
        )
        self._validate_base_link_name(profile, planner)
        tool_frame = self._resolve_tool_frame(profile, planner)

        backend = _CuroboBackend(
            planner=planner,
            control_part=control_part,
            sim_joint_names=sim_joint_names,
            tool_frame=tool_frame,
            profile=profile,
            batch_size=int(batch_size),
        )
        if self.cfg.warmup:
            planner.warmup(enable_graph=bool(self.cfg.use_cuda_graph))
        self._backend_cache[key] = backend
        return backend

    def _resolve_sim_joint_names(self, control_part: str) -> list[str]:
        """Return simulator control-part joints in the robot's canonical order."""
        control_parts = getattr(self.robot, "control_parts", None)
        if not control_parts or control_part not in control_parts:
            logger.log_error(
                f"Robot '{self.cfg.robot_uid}' has no control part '{control_part}'. "
                "cuRobo requires an explicit ordered control-part joint list.",
                ValueError,
            )
        return list(control_parts[control_part])

    def _validate_profile_joint_names(
        self,
        profile: CuroboRobotProfileCfg,
        sim_joint_names: list[str],
        curobo_joint_names: list[str],
    ) -> None:
        """Validate the profile mapping before a CUDA planning call."""
        sim_to_curobo = profile.sim_to_curobo_joint_names
        if set(sim_to_curobo) != set(sim_joint_names):
            logger.log_error(
                "sim_to_curobo_joint_names keys must exactly match the robot "
                f"control-part joints {sim_joint_names}; got {list(sim_to_curobo)}.",
                ValueError,
            )
        mapped_names = [sim_to_curobo[name] for name in sim_joint_names]
        if len(mapped_names) != len(set(mapped_names)):
            logger.log_error(
                "sim_to_curobo_joint_names maps multiple simulator joints to "
                f"the same cuRobo joint: {mapped_names}.",
                ValueError,
            )
        missing = [name for name in mapped_names if name not in curobo_joint_names]
        if missing:
            logger.log_error(
                "cuRobo profile is missing mapped active joints "
                f"{missing}; planner joints are {curobo_joint_names}.",
                ValueError,
            )
        unmapped = [
            name for name in curobo_joint_names if name not in set(mapped_names)
        ]
        if unmapped:
            logger.log_error(
                "cuRobo planner exposes joints outside the requested control "
                f"part: {unmapped}. Lock non-controlled joints in the V2 robot "
                "profile or select a control part that includes them.",
                ValueError,
            )
        if profile.active_joint_names is not None:
            expected = list(profile.active_joint_names)
            if expected != curobo_joint_names:
                logger.log_error(
                    f"active_joint_names {expected} do not match cuRobo model "
                    f"joints {curobo_joint_names} (missing/duplicate/out-of-order).",
                    ValueError,
                )

    def _resolve_tool_frame(
        self, profile: CuroboRobotProfileCfg, planner: "Any"
    ) -> str:
        """Resolve and validate the single V2 tool frame used for pose goals."""
        tool_frames = list(getattr(planner, "tool_frames", []))
        tool_frame = profile.tool_frame_name
        if tool_frame is None:
            if len(tool_frames) != 1:
                logger.log_error(
                    "tool_frame_name is required when the cuRobo profile exposes "
                    f"multiple tool frames: {tool_frames}.",
                    ValueError,
                )
            return tool_frames[0]
        if tool_frames and tool_frame not in tool_frames:
            logger.log_error(
                f"tool_frame_name '{tool_frame}' is not available in the cuRobo "
                f"profile tool frames {tool_frames}.",
                ValueError,
            )
        return tool_frame

    def _validate_base_link_name(
        self, profile: CuroboRobotProfileCfg, planner: "Any"
    ) -> None:
        """Ensure an explicitly configured base link matches the V2 model."""
        expected = profile.base_link_name
        if expected is None:
            return
        kinematics = getattr(planner, "kinematics", None)
        actual = getattr(kinematics, "base_link", None)
        if actual is None:
            logger.log_error(
                "cuRobo planner did not expose a kinematics.base_link, so "
                f"base_link_name={expected!r} cannot be validated.",
                ValueError,
            )
        if actual != expected:
            logger.log_error(
                f"CuroboRobotProfileCfg.base_link_name={expected!r} does not "
                f"match the loaded cuRobo V2 base link {actual!r}.",
                ValueError,
            )

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
            self._validate_segment_batch(target, B, seg_idx)
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

            if v2_result is None:
                # V2 returns None when no seed reaches a valid solution. Keep
                # the standard EmbodiChain failure contract instead of
                # dereferencing a result that does not exist.
                seg_success = torch.zeros(B, dtype=torch.bool, device=self.device)
                seg_positions = current.unsqueeze(1)
                seg_dt = torch.zeros(B, 1, dtype=torch.float32, device=self.device)
            else:
                seg_success, seg_positions, seg_dt = self._extract_segment(
                    v2_result, backend
                )
            seg_success = seg_success.to(self.device) & alive
            if v2_result is not None and self.cfg.max_planning_time is not None:
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

    def _validate_segment_batch(
        self, target: PlanState, start_batch_size: int, segment_index: int
    ) -> None:
        """Reject target batches that cannot pair with the planning start state."""
        if target.move_type == MoveType.EEF_MOVE:
            values = target.xpos
            expected_dims = (3,)
        elif target.move_type == MoveType.JOINT_MOVE:
            values = target.qpos
            expected_dims = (1, 2)
        else:
            return
        if values is None:
            return
        values = torch.as_tensor(values)
        if values.dim() not in expected_dims:
            # The type-specific conversion path will report the more useful
            # shape error below; only check valid target shapes here.
            return
        target_batch_size = 1 if values.dim() == 1 else values.shape[0]
        if target_batch_size != start_batch_size:
            logger.log_error(
                f"Segment {segment_index} target batch {target_batch_size} does "
                f"not match planning start batch {start_batch_size}.",
                ValueError,
            )

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

        seg_positions = self._map_curobo_to_sim(full, traj.joint_names, backend)
        seg_dt = self._extract_dt(traj, last_tstep, max_len, B)
        return success, seg_positions, seg_dt

    def _map_curobo_to_sim(
        self,
        full_positions: torch.Tensor,
        curobo_joint_names: list[str],
        backend: "_CuroboBackend",
    ) -> torch.Tensor:
        """Map a full cuRobo trajectory to simulator control-part joint order."""
        sim_to_curobo = backend.profile.sim_to_curobo_joint_names
        cols: list[int] = []
        for sim_name in backend.sim_joint_names:
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
        self,
        traj: "Any",
        last_tstep: torch.Tensor,
        max_len: int,
        B: int,
    ) -> torch.Tensor:
        """Derive ``(B, max_len)`` per-point deltas from a V2 trajectory.

        cuRobo V2 uses a scalar ``dt`` per batch/seed for interpolated
        trajectories. EmbodiChain represents deltas at each trajectory point,
        with a zero first point and one interval per following point.
        """
        raw_dt = getattr(traj, "dt", None)
        dt: torch.Tensor | None = None
        if isinstance(raw_dt, torch.Tensor):
            if raw_dt.dim() == 1:
                dt = raw_dt.unsqueeze(0).expand(B, -1)
            elif raw_dt.dim() == 2:
                dt = raw_dt
        if dt is None:
            dt = torch.full(
                (B, 1),
                float(self.cfg.interpolation_dt),
                device=self.device,
                dtype=torch.float32,
            )
        if dt.shape[0] == 1 and B > 1:
            dt = dt.expand(B, -1)
        if dt.shape[0] != B:
            logger.log_error(
                f"cuRobo trajectory dt batch {dt.shape[0]} does not match {B}.",
                ValueError,
            )

        out = torch.zeros(B, max_len, device=self.device, dtype=torch.float32)
        if dt.shape[-1] == 1:
            interval = dt[:, 0].to(self.device, dtype=torch.float32)
            for b in range(B):
                length = min(int(last_tstep[b].item()) + 1, max_len)
                if length > 1:
                    out[b, 1:length] = interval[b]
            return out

        # Preserve an explicitly per-point delta sequence supplied by a V2
        # result or a compatible future API. It already includes the first
        # point's zero delta in EmbodiChain's convention.
        length = min(dt.shape[-1], max_len)
        out[:, :length] = dt[:, :length].to(self.device, dtype=torch.float32)
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

        Every active joint is filled from ``current`` in the simulator control
        part order. Backend construction already rejects profiles whose active
        V2 joints extend beyond that control part, which keeps collision
        checking and the returned trajectory semantically aligned.
        """
        profile = backend.profile
        curobo_names = list(backend.planner.joint_names)
        sim_to_curobo = profile.sim_to_curobo_joint_names
        curobo_to_sim_idx = {
            cu_name: idx
            for idx, sim_name in enumerate(backend.sim_joint_names)
            for cu_name in [sim_to_curobo[sim_name]]
        }
        if current.dim() != 2 or current.shape[1] != len(backend.sim_joint_names):
            logger.log_error(
                "cuRobo start/goal qpos must have shape "
                f"(B, {len(backend.sim_joint_names)}), got {tuple(current.shape)}.",
                ValueError,
            )
        B = current.shape[0]
        state = torch.zeros(
            B, len(curobo_names), device=self.device, dtype=torch.float32
        )
        for i, cu_name in enumerate(curobo_names):
            if cu_name in curobo_to_sim_idx:
                state[:, i] = current[:, curobo_to_sim_idx[cu_name]]
            else:  # Defensive: _validate_profile_joint_names rejects this case.
                logger.log_error(
                    f"cuRobo active joint '{cu_name}' is not mapped to the "
                    "selected EmbodiChain control part.",
                    ValueError,
                )
        return self._bindings.JointState.from_position(state, joint_names=curobo_names)

    def _to_curobo_pose_goal(
        self, xpos: torch.Tensor, backend: "_CuroboBackend"
    ) -> "Any":
        """Build a cuRobo ``GoalToolPose`` from a batched world-frame pose."""
        xpos = torch.as_tensor(xpos, device=self.device, dtype=torch.float32)
        xpos = self._sim_world_to_curobo_base_pose(xpos, backend)
        xpos = self._tcp_to_tool_pose(xpos, backend.profile)
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
        return self._to_curobo_joint_state(qpos, backend)

    def _tcp_to_tool_pose(
        self, tcp_pose: torch.Tensor, profile: CuroboRobotProfileCfg
    ) -> torch.Tensor:
        """Convert a simulator TCP goal into the configured cuRobo tool frame."""
        if tcp_pose.dim() != 3 or tcp_pose.shape[-2:] != (4, 4):
            logger.log_error(
                f"Expected (B, 4, 4) TCP pose matrices, got {tuple(tcp_pose.shape)}.",
                ValueError,
            )
        if profile.tool_frame_to_tcp is None:
            return tcp_pose
        frame_to_tcp = torch.as_tensor(
            profile.tool_frame_to_tcp,
            dtype=torch.float32,
            device=self.device,
        )
        if frame_to_tcp.shape != (4, 4):
            logger.log_error(
                "tool_frame_to_tcp must be a homogeneous (4, 4) transform, "
                f"got {tuple(frame_to_tcp.shape)}.",
                ValueError,
            )
        tool_to_frame = torch.linalg.inv(frame_to_tcp)
        return tcp_pose @ tool_to_frame

    def _sim_world_to_curobo_base_pose(
        self, world_pose: torch.Tensor, backend: "_CuroboBackend"
    ) -> torch.Tensor:
        """Express simulator-world poses in the loaded cuRobo base frame.

        EmbodiChain pose targets and dynamic obstacle poses are world poses,
        while a cuRobo robot profile/world is rooted at the profile's base
        link. The live simulator base pose accounts for arena offsets and
        mobile bases; ``sim_base_to_curobo_base`` accounts for any fixed frame
        convention difference between the two robot descriptions.
        """
        if world_pose.dim() != 3 or world_pose.shape[-2:] != (4, 4):
            logger.log_error(
                f"Expected (B, 4, 4) simulator-world pose matrices, got "
                f"{tuple(world_pose.shape)}.",
                ValueError,
            )
        batch_size = world_pose.shape[0]
        sim_base_pose = self._get_sim_base_pose(backend, batch_size)
        profile_transform = backend.profile.sim_base_to_curobo_base
        if profile_transform is None:
            sim_base_to_curobo = torch.eye(
                4, dtype=torch.float32, device=self.device
            ).expand(batch_size, -1, -1)
        else:
            sim_base_to_curobo = torch.as_tensor(
                profile_transform, dtype=torch.float32, device=self.device
            )
            if sim_base_to_curobo.shape != (4, 4):
                logger.log_error(
                    "sim_base_to_curobo_base must be a homogeneous (4, 4) "
                    f"transform, got {tuple(sim_base_to_curobo.shape)}.",
                    ValueError,
                )
            sim_base_to_curobo = sim_base_to_curobo.expand(batch_size, -1, -1)
        return torch.bmm(
            sim_base_to_curobo,
            torch.bmm(torch.linalg.inv(sim_base_pose), world_pose),
        )

    def _get_sim_base_pose(
        self, backend: "_CuroboBackend", batch_size: int
    ) -> torch.Tensor:
        """Return ``(B, 4, 4)`` world poses of a control part's solver base."""
        control_part = backend.control_part
        root_link_name = backend.profile.sim_base_link_name
        if root_link_name is None:
            solver = self.robot.get_solver(name=control_part)
            root_link_name = getattr(solver, "root_link_name", None)
        if root_link_name is None:
            logger.log_error(
                f"Control part '{control_part}' needs either a solver with "
                "root_link_name or CuroboRobotProfileCfg.sim_base_link_name "
                "for cuRobo world-frame conversion.",
                ValueError,
            )
        base_pose = self.robot.get_link_pose(
            link_name=root_link_name,
            env_ids=list(range(batch_size)),
            to_matrix=True,
        )
        base_pose = torch.as_tensor(base_pose, dtype=torch.float32, device=self.device)
        if base_pose.dim() == 2:
            base_pose = base_pose.unsqueeze(0)
        if base_pose.shape != (batch_size, 4, 4):
            logger.log_error(
                f"Simulator base pose for '{control_part}' must have shape "
                f"({batch_size}, 4, 4), got {tuple(base_pose.shape)}.",
                ValueError,
            )
        return base_pose

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
                backends. In ``multi_env`` mode, cached backends must share one
                batch size; otherwise pass the intended backend explicitly.
        """
        if poses is None:
            return
        _validate_dynamic_obstacles(poses, list(self.cfg.world.dynamic_obstacle_names))
        backends = (
            [backend] if backend is not None else list(self._backend_cache.values())
        )
        if backend is None and self.cfg.world.multi_env:
            batch_sizes = {cached_backend.batch_size for cached_backend in backends}
            if len(batch_sizes) > 1:
                logger.log_error(
                    "Cannot update all cached multi-env cuRobo backends with "
                    "different cached batch sizes. Pass the intended backend "
                    "explicitly.",
                    ValueError,
                )
        for name, pose_tensor in poses.items():
            pose_tensor = torch.as_tensor(
                pose_tensor, device=self.device, dtype=torch.float32
            )
            for be in backends:
                curobo_pose = self._sim_world_to_curobo_base_pose(pose_tensor, be)
                self._update_backend_obstacle(name, curobo_pose, be)

    def _update_backend_obstacle(
        self, name: str, pose_tensor: torch.Tensor, backend: "_CuroboBackend"
    ) -> None:
        """Apply one named obstacle pose tensor under the backend's world policy."""
        if self.cfg.world.multi_env:
            if pose_tensor.shape[0] != backend.batch_size:
                logger.log_error(
                    f"dynamic obstacle '{name}' has batch {pose_tensor.shape[0]}, "
                    f"but this multi-env cuRobo backend expects {backend.batch_size}.",
                    ValueError,
                )
            positions, quaternions = _matrix_to_position_quaternion(pose_tensor)
            for env_idx in range(backend.batch_size):
                pose = self._bindings.Pose(
                    position=positions[env_idx], quaternion=quaternions[env_idx]
                )
                backend.planner.scene_collision_checker.update_obstacle_pose(
                    name, pose, env_idx=env_idx
                )
            return

        # A shared world has one collision environment, so a batched input is
        # only meaningful if every environment supplied the same world pose.
        if pose_tensor.shape[0] > 1 and not torch.allclose(
            pose_tensor, pose_tensor[:1].expand_as(pose_tensor)
        ):
            logger.log_error(
                f"dynamic obstacle '{name}' has different poses across a "
                "shared cuRobo world. Enable world.multi_env for per-env worlds.",
                ValueError,
            )
        position, quaternion = _matrix_to_position_quaternion(pose_tensor[:1])
        pose = self._bindings.Pose(position=position[0], quaternion=quaternion[0])
        backend.planner.scene_collision_checker.update_obstacle_pose(
            name, pose, env_idx=0
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
    control_part: str
    sim_joint_names: list[str]
    tool_frame: str
    profile: CuroboRobotProfileCfg
    batch_size: int
