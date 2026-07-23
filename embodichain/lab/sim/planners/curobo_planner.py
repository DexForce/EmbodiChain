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

import hashlib
import importlib
import os
import queue
import time
from dataclasses import dataclass
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

    from embodichain.lab.sim.objects import RigidObject

__all__ = [
    "CuroboAutoGenCfg",
    "CuroboPlanOptions",
    "CuroboPlanner",
    "CuroboPlannerCfg",
    "CuroboWorldCfg",
]


# cuRobo V2 installation extras documented at NVIDIA's installation page.
_CUROBO_INSTALL_URL = (
    "https://nvlabs.github.io/curobo/latest/getting-started/installation.html"
)


@dataclass
class _CuroboProfile:
    """Auto-derived cuRobo robot profile for one control part (internal).

    Produced by :meth:`CuroboPlanner._materialize_profile` from the robot's URDF
    and IK solver - never user-configured. The cuRobo robot YAML is always
    auto-generated from the URDF (see :class:`CuroboAutoGenCfg`), so the simulator
    and cuRobo share the same joint names and the joint mapping is identity.
    """

    robot_config_path: str
    """Cached path of the auto-generated cuRobo robot YAML."""

    sim_to_curobo_joint_names: dict[str, str]
    """Simulator -> cuRobo joint-name mapping (identity for the auto-gen YAML)."""

    tool_frame_name: str | None = None
    """cuRobo tool frame (a URDF link name) used as the planning target."""

    tool_frame_to_tcp: list[list[float]] | None = None
    """Fixed transform from the cuRobo tool frame to the simulator TCP frame.

    ``None`` means the tool frame is already the TCP (the common auto-derived
    case, where the solver's ``end_link_name`` is the TCP).
    """

    base_link_name: str | None = None
    """cuRobo robot base link, validated against the loaded V2 model."""

    sim_base_link_name: str | None = None
    """Simulator link physically equivalent to the control-part base."""

    sim_base_to_curobo_base: list[list[float]] | None = None
    """Fixed transform from the simulator base to the cuRobo base (``None``=coincide)."""


class _RigidObjectRefList(list):
    """A list of live ``RigidObject`` handles that survives ``@configclass`` deepcopy.

    ``@configclass`` deepcopies every field on construction, but live dexsim
    objects hold non-pickleable C++ handles (e.g. ``dexsim.World``). This
    ``list`` subclass overrides ``__deepcopy__`` to share the object references
    instead of cloning them, so ``CuroboWorldCfg(rigid_objects=[...])`` works.
    """

    def __deepcopy__(self, memo: dict) -> "_RigidObjectRefList":  # noqa: ARG002
        return _RigidObjectRefList(self)


@configclass
class CuroboWorldCfg:
    """Static collision-world configuration for the cuRobo backend.

    The collision world is always auto-generated from live :class:`RigidObject`
    meshes (see :attr:`rigid_objects`); there is no external scene-YAML path.
    """

    rigid_objects: list[RigidObject] | None = None
    """Live :class:`RigidObject` obstacles to bake into the auto-generated world YAML.

    The adapter reads each object's mesh (``get_vertices`` / ``get_triangles``)
    and world pose (``get_local_pose``) and writes a cuRobo V2 scene YAML (cached
    on disk by content hash). Poses are written in the cuRobo world/base frame,
    so this is exact when the robot base sits at the simulator world origin. For
    obstacles that move or live in an offset base frame, also list their names in
    :attr:`dynamic_obstacle_names` to update poses at plan time. ``None`` yields an
    initially empty collision world.
    """

    obstacle_representation: str = "cuboid"
    """Collision representation used when generating the YAML from :attr:`rigid_objects`.

    ``"cuboid"`` (default) emits a local-frame AABB per object, placed as an OBB
    via the object pose - exact for box-shaped obstacles and needs no CUDA.
    ``"mesh"`` emits the full triangle mesh (exact, no CUDA). ``"sphere"`` fits
    spheres with cuRobo's ``fit_spheres_to_mesh`` (faster collision checking, but
    approximate and requires CUDA + cuRobo + trimesh).
    """

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
    environment. The generated world YAML is cloned for every row.
    """

    def __post_init__(self) -> None:
        # Wrap live RigidObjects so the @configclass field-deepcopy (run right
        # after this by custom_post_init) shares references instead of trying to
        # pickle non-pickleable C++ dexsim handles held by each RigidObject.
        if self.rigid_objects is not None and not isinstance(
            self.rigid_objects, _RigidObjectRefList
        ):
            self.rigid_objects = _RigidObjectRefList(self.rigid_objects)


@configclass
class CuroboAutoGenCfg:
    """Auto-generation of the cuRobo robot YAML from the robot's URDF.

    The adapter generates a cuRobo robot configuration YAML from the robot's URDF
    (fitting collision spheres to each link mesh) on the first plan and caches it
    on disk so subsequent inits skip regeneration. The TCP, tool frame, and base
    link are read from the robot's solver, so nothing robot-specific needs to be
    hardcoded.
    """

    cache_dir: str | None = None
    """Directory for cached robot YAMLs.

    ``None`` (default) uses ``$XDG_CACHE_HOME/embodichain_curobo`` or
    ``~/.cache/embodichain_curobo``. The cache key hashes the URDF path, URDF
    content, control part, tool frame, and fit parameters, so editing the URDF
    or changing the fit settings regenerates automatically.
    """

    fit_type: str = "voxel"
    """cuRobo sphere-fit strategy for auto-generation: ``"voxel"`` (default,
    fast), ``"morphit"`` (best, slower), or ``"surface"`` (crude)."""

    num_spheres: int | None = None
    """Per-link sphere count. ``None`` auto-estimates from bounding-box volume
    scaled by :attr:`sphere_density`."""

    sphere_density: float = 0.1
    """Multiplier on the auto-estimated per-link sphere count (ignored when
    :attr:`num_spheres` is set).

    The cuRobo volume-based estimate over-fits at ``1.0`` (~668 spheres for a
    Franka Panda, making planning pathologically slow). ``0.1`` (default) yields
    ~50-100 spheres - enough coverage for collision-aware planning while keeping
    each plan fast. Increase for tighter coverage on complex robots.
    """

    surface_radius: float = 0.005
    """Fixed radius used only by the ``surface`` strategy."""

    iterations: int = 200
    """Adam iterations for the ``morphit`` strategy."""

    collision_sphere_buffer: float = 0.0
    """Padding added to every fitted sphere's radius (m)."""

    force: bool = False
    """Bypass the cache and regenerate the robot YAML on the next plan."""


@configclass
class CuroboPlannerCfg(BasePlannerCfg):
    """Configuration for the cuRobo V2 planner backend.

    cuRobo always runs in a spawned side process with its own CUDA context, so it
    can capture CUDA graphs (~0.02s/plan) without conflicting with DexSim's
    Vulkan/CUDA interop semaphores (graph capture in-process crashes DexSim at
    ``DFGpuSemaphore.cpp:346``). Both the cuRobo robot YAML and the collision-world
    YAML are auto-generated internally (from the robot's URDF and from
    :attr:`world.rigid_objects` respectively); no external YAML is used. The
    per-control-part profile is auto-derived from the robot's solver at plan time.
    """

    planner_type: str = "curobo"

    world: CuroboWorldCfg = CuroboWorldCfg()
    """Collision-world configuration (auto-generated from ``RigidObject`` meshes)."""

    auto_gen: CuroboAutoGenCfg = CuroboAutoGenCfg()
    """Auto-generation settings for the cuRobo robot YAML from the robot's URDF."""

    sim_base_to_curobo_base: list[list[float]] | None = None
    """Fixed transform from the simulator control-part base to the cuRobo base.

    The adapter uses this together with the live simulator base pose to convert
    simulator-world Cartesian goals and dynamic obstacle poses into cuRobo's base
    frame. ``None`` (default) means the two base frames coincide - the common
    case, since the auto-generated robot YAML is rooted at the URDF base link the
    solver reports. Set this only when the simulator base and the URDF base use
    different fixed frame conventions.
    """

    collision_activation_distance: float = 0.01
    """cuRobo collision activation distance (optimizer setting)."""

    max_attempts: int = 5
    """Default per-plan cuRobo attempt count."""

    max_planning_time: float | None = None
    """Post-plan validation budget (seconds). ``None`` skips the timing check."""

    interpolation_dt: float = 0.025
    """Interpolation step (seconds) used by cuRobo and as a dt fallback."""

    warmup_iterations: int = 5
    """cuRobo warmup iterations run once per cached worker planner.

    The worker captures CUDA graphs during warmup so the first real plan is fast.
    Lower to speed up worker startup at the cost of a colder first plan.
    """


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

    The planner lazily imports cuRobo V2 at construction time (as a fail-fast
    check) and runs all cuRobo work in a spawned side process with its own CUDA
    context, where cuRobo can capture CUDA graphs without conflicting with
    DexSim's GPU stream. One worker process per control part is cached; the
    worker itself caches a ``MotionPlanner`` (single-environment) or
    ``BatchMotionPlanner`` (multi-environment) per ``(batch_size, multi_env)``
    key. Cartesian goals are converted to the cuRobo base frame in the parent
    (which holds the live robot) and the solve is RPC'd to the worker.

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
        # Fail fast with an actionable error if cuRobo V2 is not installed; the
        # worker process imports it lazily, but surface the error at construction.
        _require_curobo()
        # Cached subprocess workers keyed by control_part.
        self._isolated_workers: dict[str, "_IsolatedWorker"] = {}
        world_cfg = cfg.world
        if world_cfg.obstacle_representation not in ("cuboid", "mesh", "sphere"):
            logger.log_error(
                "CuroboWorldCfg.obstacle_representation must be 'cuboid', 'mesh', "
                f"or 'sphere', got {world_cfg.obstacle_representation!r}.",
                ValueError,
            )

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
        control_part = self._resolve_control_part(options)
        start = self._resolve_start_qpos(options.start_qpos, control_part)
        backend = self._get_isolated_backend(control_part, start.shape[0])
        self.update_dynamic_obstacles(options.dynamic_obstacle_poses, backend)
        return self._plan_segments(target_states, start, backend, options)

    # ------------------------------------------------------------------
    # Profile / start resolution
    # ------------------------------------------------------------------

    def _resolve_control_part(self, options: CuroboPlanOptions) -> str:
        """Resolve and validate the requested control part against the robot."""
        control_part = options.control_part
        if control_part is None:
            logger.log_error("CuroboPlanOptions.control_part is required.", ValueError)
        control_parts = getattr(self.robot, "control_parts", None) or {}
        if control_part not in control_parts:
            logger.log_error(
                f"Robot '{self.cfg.robot_uid}' has no control part '{control_part}'. "
                f"Available control parts: {sorted(control_parts)}.",
                ValueError,
            )
        return control_part

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

    def _materialize_profile(self, control_part: str) -> _CuroboProfile:
        """Auto-derive the cuRobo profile for ``control_part`` from the robot.

        Reads the tool frame, TCP offset, and base link from the control part's
        IK solver, builds the identity simulator->cuRobo joint mapping (the
        auto-generated robot YAML reuses the URDF joint names), and generates
        the cuRobo robot YAML from the URDF. Nothing robot-specific is hardcoded.
        """
        robot = self.robot
        assert (
            robot is not None
        ), "cuRobo planner has no robot; cannot materialize the profile."
        solver = None
        solvers = getattr(robot, "_solvers", None) or {}
        if solvers and control_part in solvers:
            solver = solvers[control_part]

        # Tool frame: prefer the solver's end link (the TCP), else the control
        # part's last link. Auto-generation needs a concrete tool frame.
        tool_frame = (
            getattr(solver, "end_link_name", None) if solver is not None else None
        )
        if tool_frame is None:
            part_links = robot.get_control_part_link_names(control_part) or []
            if not part_links:
                logger.log_error(
                    f"Control part {control_part!r} has no solver end_link_name and "
                    "no links; cannot derive a cuRobo tool frame.",
                    ValueError,
                )
            tool_frame = part_links[-1]

        # TCP offset: only when the solver's tool frame is not itself the TCP.
        tool_frame_to_tcp = None
        if solver is not None:
            tcp_xpos = getattr(solver, "tcp_xpos", None)
            if tcp_xpos is not None:
                tool_frame_to_tcp = tcp_xpos.tolist()

        base_link = (
            getattr(solver, "root_link_name", None) if solver is not None else None
        )
        sim_base_link = base_link

        sim_joints = self._resolve_sim_joint_names(control_part)
        sim_to_curobo = {j: j for j in sim_joints}

        robot_config_path = self._auto_generate_robot_yaml(control_part, tool_frame)

        return _CuroboProfile(
            robot_config_path=robot_config_path,
            sim_to_curobo_joint_names=sim_to_curobo,
            tool_frame_name=tool_frame,
            tool_frame_to_tcp=tool_frame_to_tcp,
            base_link_name=base_link,
            sim_base_link_name=sim_base_link,
            sim_base_to_curobo_base=self.cfg.sim_base_to_curobo_base,
        )

    def _auto_generate_robot_yaml(
        self, control_part: str, tool_frame: str | None
    ) -> str:
        """Return a cached cuRobo robot YAML path, generating it from the URDF if needed."""
        from .curobo_yaml import generate_curobo_robot_yaml

        robot = self.robot
        assert (
            robot is not None
        ), "cuRobo planner has no robot; cannot auto-generate its YAML."
        auto = self.cfg.auto_gen
        solvers = getattr(robot, "_solvers", None) or {}
        solver = solvers.get(control_part) if solvers else None
        urdf_path = (
            getattr(solver, "urdf_path", None) if solver is not None else None
        ) or robot.cfg.fpath
        cache_dir = auto.cache_dir or os.path.join(
            os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")),
            "embodichain_curobo",
        )
        cache_key = self._robot_yaml_cache_key(
            urdf_path, control_part, tool_frame, auto
        )
        cache_path = os.path.join(cache_dir, f"{cache_key}.yml")
        if not auto.force and os.path.exists(cache_path):
            logger.log_info(f"cuRobo robot YAML cache hit: {cache_path}")
            return cache_path
        logger.log_info(
            f"Auto-generating cuRobo robot YAML from URDF ({urdf_path}) -> {cache_path}"
        )
        return generate_curobo_robot_yaml(
            robot,
            control_part,
            cache_path,
            tool_frame=tool_frame,
            fit_type=auto.fit_type,
            num_spheres=auto.num_spheres,
            sphere_density=auto.sphere_density,
            surface_radius=auto.surface_radius,
            iterations=auto.iterations,
            collision_sphere_buffer=auto.collision_sphere_buffer,
        )

    def _robot_yaml_cache_key(
        self,
        urdf_path: str,
        control_part: str,
        tool_frame: str | None,
        auto: CuroboAutoGenCfg,
    ) -> str:
        """Hash the URDF path/content and fit parameters into a stable cache key."""
        hasher = hashlib.md5()
        hasher.update(urdf_path.encode("utf-8"))
        try:
            with open(urdf_path, "rb") as urdf_file:
                hasher.update(urdf_file.read())
        except OSError:
            pass
        hasher.update(control_part.encode("utf-8"))
        hasher.update((tool_frame or "").encode("utf-8"))
        hasher.update(auto.fit_type.encode("utf-8"))
        hasher.update(str(auto.num_spheres).encode("utf-8"))
        hasher.update(str(auto.sphere_density).encode("utf-8"))
        hasher.update(str(auto.surface_radius).encode("utf-8"))
        hasher.update(str(auto.iterations).encode("utf-8"))
        hasher.update(str(auto.collision_sphere_buffer).encode("utf-8"))
        return hasher.hexdigest()

    def _auto_generate_world_yaml(self, world_cfg: CuroboWorldCfg) -> str:
        """Return a cached cuRobo world YAML path generated from ``rigid_objects``.

        Mirrors :meth:`_auto_generate_robot_yaml`: a content-hashed YAML is written
        to the cuRobo cache directory (reusing :attr:`CuroboAutoGenCfg.cache_dir`)
        on the first plan and reused thereafter. Sphere-fit parameters come from
        :class:`CuroboAutoGenCfg` so robot and world fitting are configured together.
        """
        from .curobo_yaml import generate_curobo_world_yaml

        rigid_objects = world_cfg.rigid_objects
        if not rigid_objects:
            logger.log_error(
                "_auto_generate_world_yaml requires non-empty rigid_objects.",
                ValueError,
            )
        assert rigid_objects is not None  # log_error raises above; narrows type
        auto = self.cfg.auto_gen
        cache_dir = auto.cache_dir or os.path.join(
            os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")),
            "embodichain_curobo",
        )
        cache_key = self._world_yaml_cache_key(world_cfg)
        cache_path = os.path.join(cache_dir, f"world_{cache_key}.yml")
        if not auto.force and os.path.exists(cache_path):
            logger.log_info(f"cuRobo world YAML cache hit: {cache_path}")
            return cache_path
        logger.log_info(
            f"Auto-generating cuRobo world YAML from {len(rigid_objects)} "
            f"RigidObject(s) ({world_cfg.obstacle_representation}) -> {cache_path}"
        )
        return generate_curobo_world_yaml(
            rigid_objects,
            cache_path,
            representation=world_cfg.obstacle_representation,
            fit_type=auto.fit_type,
            num_spheres=auto.num_spheres,
            sphere_density=auto.sphere_density,
            surface_radius=auto.surface_radius,
            iterations=auto.iterations,
            collision_sphere_buffer=auto.collision_sphere_buffer,
        )

    def _world_yaml_cache_key(self, world_cfg: CuroboWorldCfg) -> str:
        """Hash per-object mesh/pose + representation + fit params into a cache key.

        Includes each object's vertex/face/pose bytes so editing the simulator
        geometry or moving a static obstacle regenerates the YAML, matching the
        robot-YAML cache's URDF-content inclusion.
        """
        hasher = hashlib.md5()
        hasher.update(world_cfg.obstacle_representation.encode("utf-8"))
        auto = self.cfg.auto_gen
        hasher.update(auto.fit_type.encode("utf-8"))
        hasher.update(str(auto.num_spheres).encode("utf-8"))
        hasher.update(str(auto.sphere_density).encode("utf-8"))
        hasher.update(str(auto.surface_radius).encode("utf-8"))
        hasher.update(str(auto.iterations).encode("utf-8"))
        hasher.update(str(auto.collision_sphere_buffer).encode("utf-8"))
        for idx, obj in enumerate(world_cfg.rigid_objects or []):
            name = getattr(obj, "uid", None) or f"obstacle_{idx}"
            hasher.update(name.encode("utf-8"))
            vertices = obj.get_vertices(env_ids=[0], scale=True)[0]
            faces = obj.get_triangles(env_ids=[0])[0]
            pose = obj.get_local_pose(to_matrix=False)[0]
            hasher.update(
                vertices.detach().to("cpu").to(torch.float32).numpy().tobytes()
            )
            hasher.update(faces.detach().to("cpu").numpy().tobytes())
            hasher.update(pose.detach().to("cpu").to(torch.float32).numpy().tobytes())
        return hasher.hexdigest()

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
        """Plan each waypoint segment sequentially and assemble a PlanResult.

        Each segment's goal is converted to the cuRobo base frame in-process
        (pure-tensor, using the live robot pose) and the cuRobo solve itself is
        RPC'd to the subprocess worker, which returns a V2-result-like object
        (or ``None``). Everything after the solve - segment extraction, the
        planning-time budget check, junction-sample de-duplication, and
        rectangular assembly - then runs unchanged.
        """
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
            if target.move_type == MoveType.EEF_MOVE:
                if target.xpos is None:
                    logger.log_error(
                        f"Segment {seg_idx} EEF_MOVE target missing xpos.",
                        ValueError,
                    )
                goal_matrix = self._to_curobo_base_tool_matrix(target.xpos, backend)
                position, quaternion = _matrix_to_position_quaternion(goal_matrix)
                start_time = time.time()
                v2_result = self._worker_plan(
                    "eef", current, position, quaternion, None, backend, max_attempts
                )
                logger.log_info(
                    f"cuRobo plan_pose segment {seg_idx} cost time: "
                    f"{time.time() - start_time:.4f}s"
                )
            elif target.move_type == MoveType.JOINT_MOVE:
                if target.qpos is None:
                    logger.log_error(
                        f"Segment {seg_idx} JOINT_MOVE target missing qpos.",
                        ValueError,
                    )
                start_time = time.time()
                v2_result = self._worker_plan(
                    "joint", current, None, None, target.qpos, backend, max_attempts
                )
                logger.log_info(
                    f"cuRobo plan_cspace segment {seg_idx} cost time: "
                    f"{time.time() - start_time:.4f}s"
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

    def _tcp_to_tool_pose(
        self, tcp_pose: torch.Tensor, profile: _CuroboProfile
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
                f"Control part '{control_part}' needs a solver with "
                "root_link_name for cuRobo world-frame conversion.",
                ValueError,
            )
        assert root_link_name is not None  # log_error raises above; narrows type
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
        """Update named dynamic obstacle poses on the cuRobo worker collision worlds.

        Args:
            poses: Mapping of obstacle name -> ``(B, 4, 4)`` world pose. ``None``
                is a no-op.
            backend: Specific control part's worker to update. If ``None``,
                updates all cached workers.
        """
        if poses is None:
            return
        _validate_dynamic_obstacles(poses, list(self.cfg.world.dynamic_obstacle_names))
        from .curobo_process_worker import UpdateObstacleMsg

        if backend is not None:
            targets = [self._isolated_workers[backend.control_part]]
        else:
            targets = list(self._isolated_workers.values())
        for name, pose_tensor in poses.items():
            pose_tensor = torch.as_tensor(
                pose_tensor, device=self.device, dtype=torch.float32
            )
            for iw in targets:
                curobo_pose = self._sim_world_to_curobo_base_pose(
                    pose_tensor, iw.shadow_backend
                )
                position, quaternion = _matrix_to_position_quaternion(curobo_pose)
                self._worker_request(
                    iw,
                    UpdateObstacleMsg(
                        name=name,
                        position=position.detach().to("cpu"),
                        quaternion=quaternion.detach().to("cpu"),
                    ),
                )

    # ------------------------------------------------------------------
    # Subprocess-isolated worker backend
    # ------------------------------------------------------------------

    def _to_curobo_base_tool_matrix(
        self, xpos: torch.Tensor, backend: "_CuroboBackend"
    ) -> torch.Tensor:
        """Convert a batched sim-world TCP pose to a cuRobo-base tool-frame matrix.

        Pure-tensor composition of :meth:`_sim_world_to_curobo_base_pose` and
        :meth:`_tcp_to_tool_pose`, so it runs in the parent (which holds the live
        robot) without constructing any cuRobo type. The worker splits this matrix
        into position/quaternion and builds the ``GoalToolPose``.
        """
        xpos = torch.as_tensor(xpos, device=self.device, dtype=torch.float32)
        xpos = self._sim_world_to_curobo_base_pose(xpos, backend)
        xpos = self._tcp_to_tool_pose(xpos, backend.profile)
        return xpos

    def _get_isolated_backend(
        self, control_part: str, batch_size: int
    ) -> "_CuroboBackend":
        """Return a shadow backend for ``control_part``, spawning its worker once.

        The shadow carries the profile / sim joint names needed by the shared
        post-processing (``_extract_segment``, ``_map_curobo_to_sim``,
        ``_sim_world_to_curobo_base_pose``); the actual cuRobo planner lives in
        the worker process. ``batch_size`` is refreshed on every call so the
        worker builds/caches the right planner.
        """
        iw = self._isolated_workers.get(control_part)
        if iw is None:
            profile = self._materialize_profile(control_part)
            sim_joint_names = self._resolve_sim_joint_names(control_part)
            world_cfg = self.cfg.world
            world_config_path = (
                self._auto_generate_world_yaml(world_cfg)
                if world_cfg.rigid_objects
                else None
            )
            iw = self._spawn_isolated_worker(
                control_part, profile, sim_joint_names, world_config_path
            )
            self._isolated_workers[control_part] = iw
        iw.shadow_backend.batch_size = int(batch_size)
        return iw.shadow_backend

    def _spawn_isolated_worker(
        self,
        control_part: str,
        profile: _CuroboProfile,
        sim_joint_names: list[str],
        world_config_path: str | None,
    ) -> "_IsolatedWorker":
        """Spawn (spawn start method) a cuRobo worker and wait for its init ACK."""
        import multiprocessing as mp

        from .curobo_process_worker import InitMsg, worker_main

        ctx = mp.get_context("spawn")
        req_queue = ctx.Queue()
        resp_queue = ctx.Queue()
        device_index = (
            self.device.index
            if self.device.index is not None
            else torch.cuda.current_device()
        )
        collision_cache = (
            dict(self.cfg.world.collision_cache)
            if self.cfg.world.collision_cache
            else None
        )
        init_msg = InitMsg(
            robot_config_path=profile.robot_config_path,
            world_config_path=world_config_path,
            tool_frame=profile.tool_frame_name,  # type: ignore[arg-type]
            sim_joint_names=list(sim_joint_names),
            sim_to_curobo=dict(profile.sim_to_curobo_joint_names),
            device_index=int(device_index),
            interpolation_dt=float(self.cfg.interpolation_dt),
            collision_activation_distance=float(self.cfg.collision_activation_distance),
            collision_cache=collision_cache,
            multi_env=bool(self.cfg.world.multi_env),
            warmup_iterations=int(self.cfg.warmup_iterations),
        )
        process = ctx.Process(
            target=worker_main,
            args=(init_msg, req_queue, resp_queue),
            daemon=True,
        )
        process.start()
        iw = _IsolatedWorker(
            control_part=control_part,
            process=process,
            req_queue=req_queue,
            resp_queue=resp_queue,
            shadow_backend=_CuroboBackend(
                control_part=control_part,
                sim_joint_names=list(sim_joint_names),
                profile=profile,
                batch_size=1,
            ),
        )
        status, payload = self._worker_request(iw, None)
        if status != "ok":
            self._shutdown_worker(iw)
            details = (
                payload[2]
                if isinstance(payload, tuple) and len(payload) > 2
                else payload
            )
            logger.log_error(
                "cuRobo isolated worker failed to initialize: " f"{details}",
                RuntimeError,
            )
        logger.log_info(
            f"cuRobo isolated worker ready for control part '{control_part}'."
        )
        return iw

    def _worker_request(self, iw: "_IsolatedWorker", msg: "Any") -> tuple[str, "Any"]:
        """Send one request to worker ``iw`` and await its ``(status, payload)`` reply.

        A ``None`` message is the init handshake (the worker ACKs once it has built
        and warmed its first planner is deferred to first plan; here it ACKs after
        executor construction). The loop re-checks liveness on each timeout so a
        worker crash surfaces as a clear error instead of an infinite hang.
        """
        proc = iw.process
        if proc is None or not proc.is_alive():
            logger.log_error(
                "cuRobo isolated worker process is not running.", RuntimeError
            )
        if msg is not None:
            iw.req_queue.put(msg)
        while True:
            try:
                return iw.resp_queue.get(timeout=30.0)
            except queue.Empty:
                if proc is None or not proc.is_alive():
                    logger.log_error(
                        "cuRobo isolated worker process died mid-request.",
                        RuntimeError,
                    )
                # Worker still alive but slow (e.g. first-plan warmup): keep waiting.

    def _worker_plan(
        self,
        move_type: str,
        current: torch.Tensor,
        position: torch.Tensor | None,
        quaternion: torch.Tensor | None,
        goal_qpos: torch.Tensor | None,
        backend: "_CuroboBackend",
        max_attempts: int,
    ) -> "Any":
        """RPC one plan to the worker and wrap the reply as a V2-result-like object."""
        from .curobo_process_worker import PlanMsg

        iw = self._isolated_workers[backend.control_part]
        msg = PlanMsg(
            batch_size=int(backend.batch_size),
            move_type=move_type,
            start_qpos=current.detach().to("cpu", dtype=torch.float32),
            max_attempts=int(max_attempts),
            goal_position=None if position is None else position.detach().to("cpu"),
            goal_quaternion=(
                None if quaternion is None else quaternion.detach().to("cpu")
            ),
            goal_qpos=None if goal_qpos is None else goal_qpos.detach().to("cpu"),
        )
        status: str
        payload: Any
        status, payload = self._worker_request(iw, msg)
        if status != "ok":
            details = (
                payload[2]
                if isinstance(payload, tuple) and len(payload) > 2
                else payload
            )
            logger.log_error(
                f"cuRobo isolated worker plan failed: {details}", RuntimeError
            )
        if payload is None:
            return None
        return SimpleNamespace(
            success=payload.success,
            interpolated_trajectory=SimpleNamespace(
                position=payload.position,
                joint_names=payload.joint_names,
                dt=payload.dt,
            ),
            interpolated_last_tstep=payload.last_tstep,
            total_time=payload.total_time,
        )

    def _shutdown_worker(self, iw: "_IsolatedWorker") -> None:
        """Best-effort close + forcible reap of one isolated worker process.

        Mirrors ``ToppraPlanner._shutdown_pool``: we do not rely on a clean join
        (a worker holding a CUDA context may deadlock in the driver at exit), so
        after signalling close we terminate and, if needed, kill.
        """
        proc = iw.process
        if proc is None:
            return
        try:
            from .curobo_process_worker import CloseMsg

            iw.req_queue.put_nowait(CloseMsg())
        except Exception:
            pass
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=5.0)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=1.0)
        iw.process = None

    def close(self) -> None:
        """Shut down every cached cuRobo worker process."""
        for iw in list(self._isolated_workers.values()):
            self._shutdown_worker(iw)
        self._isolated_workers.clear()

    def __del__(self) -> None:  # pragma: no cover - best-effort GC cleanup
        try:
            self.close()
        except Exception:
            pass


@dataclass
class _CuroboBackend:
    """Parent-side metadata for one control part's subprocess worker.

    Carries the profile / sim joint names the shared post-processing
    (``_extract_segment``, ``_map_curobo_to_sim``,
    ``_sim_world_to_curobo_base_pose``) needs; the real cuRobo planner lives in
    the worker process. ``batch_size`` is refreshed per plan.
    """

    control_part: str
    sim_joint_names: list[str]
    profile: _CuroboProfile
    batch_size: int


@dataclass
class _IsolatedWorker:
    """One subprocess-isolated cuRobo worker and its parent-side handle.

    ``shadow_backend`` carries the profile / sim joint names the shared
    post-processing needs, while the real cuRobo planner lives in ``process``.
    ``batch_size`` on the shadow is refreshed per plan.
    """

    control_part: str
    process: "Any"
    req_queue: "Any"
    resp_queue: "Any"
    shadow_backend: _CuroboBackend
