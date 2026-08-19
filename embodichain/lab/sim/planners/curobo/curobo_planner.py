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
import logging
import os
import threading
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Literal

import torch
import yaml

from embodichain.utils import configclass, logger
from embodichain.utils.math import pose_inv, quat_from_matrix

from embodichain.lab.sim.planners.base_planner import (
    BasePlanner,
    BasePlannerCfg,
    PlanOptions,
    validate_plan_options,
)
from embodichain.lab.sim.planners.curobo.curobo_yaml import _named_rigid_objects
from embodichain.lab.sim.planners.utils import MoveType, PlanResult, PlanState

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

# Bumped whenever the auto-generated robot-YAML schema/logic changes so that
# cached YAMLs from an older generator are regenerated instead of reused. v2:
# exclude URDF mimic joints from cspace/lock_joints (cuRobo folds them into
# their active joint and raises KeyError when locking one).
_CUROBO_ROBOT_YAML_GENERATOR_VERSION = "v2"

# cuRobo 0.8 does not expose PyTorch's CUDA stream-capture error mode. The
# temporary adapter below therefore replaces ``torch.cuda.graph`` only while
# cuRobo can lazily record graphs. Serialize that small process-wide patch.
_TORCH_CUDA_GRAPH_PATCH_LOCK = threading.Lock()


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


class _RigidObjectRefMapping(dict):
    """Registry IDs mapped to live objects without deepcopying their handles."""

    def __deepcopy__(self, memo: dict) -> "_RigidObjectRefMapping":  # noqa: ARG002
        return _RigidObjectRefMapping(self)


@configclass
class CuroboWorldCfg:
    """Static collision-world configuration for the cuRobo backend.

    The collision world is always auto-generated from live :class:`RigidObject`
    meshes (see :attr:`rigid_objects`); there is no external scene-YAML path.
    """

    rigid_objects: list[RigidObject] | Mapping[str, RigidObject] | None = None
    """Live :class:`RigidObject` obstacles to bake into the generated world YAML.

    The adapter reads each object's mesh (``get_vertices`` / ``get_triangles``)
    and world pose (``get_local_pose``) and writes a cuRobo V2 scene YAML (cached
    on disk by content hash). A mapping is the registry-backed path: its keys are
    authoritative obstacle IDs even when they differ from ``RigidObject.uid``.
    The list form remains available for advanced callers and derives names from
    ``uid`` (or ``obstacle_<index>`` when absent). Poses are written in the cuRobo
    world/base frame, so this is exact when the robot base sits at the simulator
    world origin. For obstacles that move or live in an offset base frame, also
    list their canonical names in :attr:`dynamic_obstacle_names` to update poses
    at plan time. ``None`` yields an initially empty collision world.
    """

    obstacle_representation: str = "sphere"
    """Collision representation used when generating the YAML from :attr:`rigid_objects`.

    ``"sphere"`` (default) fits spheres with cuRobo's
    ``fit_spheres_to_mesh`` (fast collision queries, approximate, and requires
    CUDA + cuRobo + trimesh). ``"cuboid"`` emits a local-frame AABB per object,
    placed as an OBB via the object pose. ``"mesh"`` emits the full triangle
    mesh (exact, no CUDA).
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
    """Canonical obstacle IDs whose poses may be updated between plans."""

    multi_env: bool = False
    """Whether cuRobo allocates one collision-world instance per environment.

    This setting concerns collision-world batching only; robot states and goals
    remain env-batched in either mode. The relevant comparison is between
    obstacle poses *after* each simulator-world pose has been rebased into that
    environment's robot-base frame:

    - ``False`` (default) shares one collision world across the batch. Use it
      when every environment has the same robot-relative obstacle layout. The
      raw simulator-world poses may still differ because replicated arenas have
      different world offsets; sharing remains valid when rebasing removes those
      offsets and produces equal poses.
    - ``True`` allocates one collision world per batch row. Use it when obstacle
      poses differ relative to their respective robot bases, such as with
      per-environment object-pose randomization.

    In multi-env mode the scene generated from ``rigid_objects`` (using env 0)
    is cloned for every row; enabling this option alone does not read a distinct
    initial pose from every simulator environment. Per-env pose differences must
    therefore be declared in :attr:`dynamic_obstacle_names` and supplied as
    batched ``(B, 4, 4)`` poses through
    :attr:`CuroboPlanOptions.dynamic_obstacle_poses`. Dynamic updates require
    ``obstacle_representation="cuboid"`` or ``"mesh"``.

    Prefer the shared default when the rebased layouts are identical because
    independent worlds replicate scene data and collision caches across the
    batch.
    """

    def __post_init__(self) -> None:
        if isinstance(self.dynamic_obstacle_names, (str, bytes)):
            raise TypeError(
                "dynamic_obstacle_names must be an iterable of obstacle IDs, "
                "not a string."
            )
        try:
            dynamic_names = list(self.dynamic_obstacle_names)
        except TypeError as exc:
            raise TypeError(
                "dynamic_obstacle_names must be an iterable of obstacle IDs."
            ) from exc
        if not all(
            isinstance(name, str) and name and name == name.strip()
            for name in dynamic_names
        ):
            raise ValueError(
                "dynamic_obstacle_names must contain unique non-empty names "
                "without outer whitespace."
            )
        if len(set(dynamic_names)) != len(dynamic_names):
            raise ValueError(
                "dynamic_obstacle_names must contain unique non-empty names "
                "without outer whitespace."
            )

        if self.rigid_objects is not None and not isinstance(
            self.rigid_objects,
            (list, Mapping),
        ):
            raise TypeError("rigid_objects must be a list, mapping, or None.")
        named_rigid_objects = _named_rigid_objects(self.rigid_objects)
        rigid_names = [name for name, _ in named_rigid_objects]
        if not all(
            isinstance(name, str) and name and name == name.strip()
            for name in rigid_names
        ):
            raise ValueError(
                "CuroboWorldCfg.rigid_objects must have non-empty string obstacle "
                "IDs without outer whitespace."
            )
        if len(set(rigid_names)) != len(rigid_names):
            raise ValueError(
                "CuroboWorldCfg.rigid_objects must have unique obstacle names."
            )
        missing = set(dynamic_names).difference(rigid_names)
        if missing:
            raise ValueError(
                "dynamic_obstacle_names reference objects not present in "
                f"rigid_objects: {sorted(missing)}."
            )
        self.dynamic_obstacle_names = dynamic_names

        # Wrap live RigidObjects so the @configclass field-deepcopy (run right
        # after this by custom_post_init) shares references instead of trying to
        # pickle non-pickleable C++ dexsim handles held by each RigidObject.
        if isinstance(self.rigid_objects, Mapping):
            if not isinstance(self.rigid_objects, _RigidObjectRefMapping):
                self.rigid_objects = _RigidObjectRefMapping(self.rigid_objects)
        elif self.rigid_objects is not None and not isinstance(
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
    ``~/.cache/embodichain_curobo``. The cache key hashes the generator version,
    URDF path, URDF content, control part, tool frame, and fit parameters, so
    editing the URDF, changing the fit settings, or a generator update
    regenerates automatically.
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

    cuRobo runs in the simulator process so it reuses the existing CUDA context
    instead of keeping a spawned Python process and a second CUDA context alive.
    CUDA graphs are enabled by default with renderer-compatible thread-local
    capture. Both the cuRobo robot YAML and the collision-world YAML are
    auto-generated internally (from the robot's URDF and from
    :attr:`world.rigid_objects` respectively); no external YAML is used. The
    per-control-part profile is auto-derived from the robot's solver at plan time.
    """

    planner_type: str = "curobo"

    log_level: str = "error"
    """Log level for cuRobo's Python logger.

    Supported values are ``"debug"``, ``"info"``, ``"warning"`` (or
    ``"warn"``), and ``"error"``. The setting is applied before cuRobo is
    imported, so it also controls messages emitted during backend
    initialization. It does not change EmbodiChain's own log level.
    """

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

    cuda_device: str | int | torch.device | None = None
    """CUDA device used exclusively by cuRobo.

    ``None`` uses the simulator GPU when physics runs on CUDA, otherwise the
    current PyTorch CUDA device. An integer selects ``cuda:<index>``. CPU
    physics is supported, but cuRobo itself always runs on CUDA.
    """

    use_cuda_graph: bool = True
    """Whether cuRobo may capture CUDA graphs in the simulator process.

    ``True`` enables the renderer-compatible fast path by default. Capture is
    serialized with DexSim Newton captures on the same CUDA device and fenced
    with device synchronizations. cuRobo's PyTorch graph captures use
    :attr:`cuda_graph_capture_error_mode`, whose ``"thread_local"`` default
    permits the DexSim render thread to continue submitting CUDA work. Set this
    to ``False`` to reduce one-time initialization and graph-resident memory.
    """

    cuda_graph_fallback: bool = True
    """Use non-graph mode if the capture coordinator times out before capture.

    An error after capture starts is never downgraded in-process because CUDA
    may leave the context in an invalid state. Such errors are raised and the
    simulator process must be restarted.
    """

    cuda_graph_capture_error_mode: str = "thread_local"
    """PyTorch CUDA stream-capture error mode used by cuRobo.

    ``"thread_local"`` isolates capture from CUDA calls made by DexSim's render
    thread and is the recommended mode for the in-process planner. ``"global"``
    retains PyTorch's strict default and is expected to conflict with an active
    renderer. ``"relaxed"`` disables additional capture-safety checks and
    should only be used for diagnosis.
    """

    capture_acquire_timeout: float | None = 2.0
    """Seconds to wait for the per-device capture coordinator.

    ``None`` waits indefinitely. After a finite timeout, graph mode falls back
    to non-graph mode, then waits for the active capture to finish before
    launching planner initialization kernels.
    """

    capture_wait_log_interval: float | None = 10.0
    """Seconds between coordinator wait logs; ``None`` disables them."""

    interpolation_dt: float = 0.025
    """Interpolation step (seconds) used by cuRobo and as a dt fallback."""

    preserve_plan_samples: bool = False
    """Whether callers must retain cuRobo's raw collision-checked samples exactly.

    When ``False`` (default),
    :class:`~embodichain.lab.sim.planners.motion_generator.MotionGenerator`
    resamples the returned trajectory to ``MotionGenOptions.sample_count`` -
    matching the documented contract of
    :attr:`~embodichain.lab.sim.atomic_actions.MotionPolicy.sample_count`
    and the other planners. The resample is arc-length piecewise-linear along
    cuRobo's joint-space path, so the collision-free path is preserved; only the
    sample density changes (cuRobo's own count is derived from
    :attr:`interpolation_dt` and the trajectory duration, e.g. ~82 for a 2 s
    plan at 0.025 s).

    When ``True``, the generator returns cuRobo's own samples unchanged. Use this
    when you need cuRobo's exact time-parameterized, collision-checked samples
    rather than a fixed waypoint count.
    """

    warmup_iterations: int = 1
    """cuRobo warmup iterations run once per cached in-process planner.

    Set to ``0`` to skip warmup when CUDA graphs are disabled. Graph mode always
    runs at least one coordinated warmup because otherwise the first real plan
    would perform an uncoordinated lazy capture.
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
    """World poses ``(B, 4, 4)`` keyed by canonical dynamic-obstacle ID."""

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
        poses: Mapping of canonical obstacle ID -> pose tensor. ``None`` is a no-op.
        allowed_names: Canonical IDs declared in :class:`CuroboWorldCfg`.

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


def _ensure_warp_torch_compat() -> None:
    """Restore ``warp.torch`` for cuRobo 0.8 when using Warp 1.13 or newer."""
    import sys

    import warp as wp

    if hasattr(wp, "torch"):
        return
    try:
        import warp._src.torch as torch_interop
    except ImportError:
        return
    wp.torch = torch_interop  # type: ignore[attr-defined]
    sys.modules["warp.torch"] = torch_interop


class _LocalCaptureCoordinator:
    """Fallback per-device capture lock when DexSim's coordinator is unavailable."""

    _instance: "_LocalCaptureCoordinator | None" = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    @classmethod
    def get(cls) -> "_LocalCaptureCoordinator":
        """Return the process-wide fallback coordinator."""
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def acquire_for_capture(
        self,
        device: str,
        owner: object,  # noqa: ARG002 - API-compatible with DexSim coordinator
        timeout: float | None = None,
        wait_log_interval: float | None = None,  # noqa: ARG002
    ) -> bool:
        """Acquire the lock for ``device`` within ``timeout``."""
        with self._locks_guard:
            lock = self._locks.setdefault(str(device), threading.Lock())
        if timeout is None:
            lock.acquire()
            return True
        return lock.acquire(timeout=max(0.0, float(timeout)))

    def release_for_capture(
        self,
        device: str,
        owner: object,  # noqa: ARG002 - API-compatible with DexSim coordinator
    ) -> None:
        """Release the lock for ``device`` when held."""
        with self._locks_guard:
            lock = self._locks.get(str(device))
        if lock is not None and lock.locked():
            lock.release()


class _CaptureOwner:
    """Weak-referenceable identity for one backend capture attempt."""


@contextmanager
def _torch_cuda_graph_capture_mode(mode: str) -> Iterator[None]:
    """Temporarily force a capture mode for cuRobo's PyTorch graph contexts.

    cuRobo 0.8 creates every graph through ``torch.cuda.graph`` but does not
    forward its ``capture_error_mode`` argument. The patch is deliberately
    scoped to planner warmup/planning and restored even when capture raises.
    """
    valid_modes = ("global", "thread_local", "relaxed")
    if mode not in valid_modes:
        raise ValueError(
            "CuroboPlannerCfg.cuda_graph_capture_error_mode must be one of "
            f"{valid_modes}, got {mode!r}."
        )

    with _TORCH_CUDA_GRAPH_PATCH_LOCK:
        original_graph = torch.cuda.graph

        def graph_with_capture_mode(*args: "Any", **kwargs: "Any") -> "Any":
            kwargs["capture_error_mode"] = mode
            return original_graph(*args, **kwargs)

        torch.cuda.graph = graph_with_capture_mode  # type: ignore[assignment]
        try:
            yield
        finally:
            torch.cuda.graph = original_graph  # type: ignore[assignment]


def _get_capture_coordinator() -> "Any":
    """Return DexSim Newton's shared capture coordinator when available."""
    try:
        coordinator_mod = importlib.import_module(
            "dexsim.engine.newton_physics.capture_coordinator"
        )
    except (ImportError, AttributeError):
        logger.log_warning(
            "DexSim CaptureCoordinator is unavailable; cuRobo CUDA graph "
            "capture will use a local per-device lock."
        )
        return _LocalCaptureCoordinator.get()
    return coordinator_mod.CaptureCoordinator.get()


def _configure_curobo_logging(log_level: str) -> None:
    """Set cuRobo's logger level without reconfiguring the root logger.

    Args:
        log_level: One of ``"debug"``, ``"info"``, ``"warning"``/``"warn"``,
            or ``"error"``. Matching is case-insensitive.

    Raises:
        ValueError: If ``log_level`` is unsupported.
    """
    levels = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warn": logging.WARNING,
        "warning": logging.WARNING,
        "error": logging.ERROR,
    }
    normalized = str(log_level).lower()
    if normalized not in levels:
        raise ValueError(
            "CuroboPlannerCfg.log_level must be one of "
            f"{tuple(levels)}, got {log_level!r}."
        )
    logging.getLogger("curobo").setLevel(levels[normalized])


def _require_curobo(log_level: str = "error") -> "Any":
    """Lazily import and bundle the cuRobo V2 public facade types.

    Args:
        log_level: cuRobo logger level applied before importing the backend.

    Returns:
        A namespace exposing ``MotionPlanner``, ``MotionPlannerCfg``,
        ``BatchMotionPlanner``, ``JointState``, ``Pose``, and ``GoalToolPose``.

    Raises:
        ImportError: If cuRobo V2 is not installed, with an actionable message
            naming NVIDIA's CUDA-matched source variants.
    """
    _configure_curobo_logging(log_level)
    # cuRobo 0.8 references ``wp.torch.*``, which Warp >= 1.13 relocated.
    _ensure_warp_torch_compat()
    try:
        planner_mod = importlib.import_module("curobo.motion_planner")
        batch_mod = importlib.import_module("curobo.batch_motion_planner")
        types_mod = importlib.import_module("curobo.types")
    except ModuleNotFoundError as exc:
        raise ImportError(
            "cuRobo V2 is required for the 'curobo' planner but was not found. "
            "Install NVIDIA's CUDA-matched source package separately, e.g. "
            "`pip install 'nvidia-curobo[cu12] @ "
            "git+https://github.com/NVlabs/curobo.git@v0.8.0'` for CUDA 12.x "
            "or replace `cu12` with `cu13` for CUDA 13.x. "
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


def _resolve_curobo_device(
    configured_device: str | int | torch.device | None,
    simulation_device: torch.device,
) -> torch.device:
    """Resolve cuRobo's concrete CUDA device independently of physics."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "cuRobo V2 requires CUDA even when SimulationManager uses CPU "
            "physics, but torch.cuda.is_available() is False."
        )

    if configured_device is None:
        if simulation_device.type == "cuda" and simulation_device.index is not None:
            device = simulation_device
        else:
            device = torch.device("cuda", torch.cuda.current_device())
    elif isinstance(configured_device, int):
        device = torch.device("cuda", configured_device)
    else:
        device = torch.device(configured_device)
        if device.type != "cuda":
            raise ValueError(
                "CuroboPlannerCfg.cuda_device must select a CUDA device, "
                f"got {configured_device!r}."
            )
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())

    assert device.index is not None
    device_count = torch.cuda.device_count()
    if device.index < 0 or device.index >= device_count:
        raise RuntimeError(
            f"cuRobo CUDA device index {device.index} is unavailable; "
            f"torch reports {device_count} CUDA device(s)."
        )
    return device


# =============================================================================
# CuroboPlanner
# =============================================================================


class CuroboPlanner(BasePlanner):
    r"""cuRobo V2 collision-aware motion-planning backend.

    The planner lazily imports cuRobo V2 at construction time and builds a
    ``MotionPlanner`` (single environment) or ``BatchMotionPlanner`` (batched
    environments) in the simulator process. Backends are cached per control
    part, batch size, collision-world mode, and goal type, so there is no helper
    process, IPC tensor copy, or second CUDA context.

    CUDA graphs are enabled by default. Initialization uses DexSim Newton's
    per-device capture coordinator, synchronizes the CUDA device before and
    after warmup, and records PyTorch graphs in
    ``"thread_local"`` capture mode so the DexSim renderer can continue issuing
    CUDA calls from its own thread. A coordinator timeout may safely downgrade
    before capture begins; a failure during capture is raised because the CUDA
    context may already be invalid.

    Cartesian (``EEF_MOVE``) targets are forwarded to cuRobo unchanged because
    the backend accepts them directly and performs its own collision-aware IK
    and trajectory optimization.
    By default the returned collision-checked samples are arc-length resampled to
    the action's ``sample_interval`` waypoint count
    (``preserve_plan_samples=False``); set
    :attr:`CuroboPlannerCfg.preserve_plan_samples=True` to keep cuRobo's own
    samples unchanged.

    Args:
        cfg: Configuration for the cuRobo planner.

    Raises:
        ImportError: If cuRobo V2 is not installed.
        RuntimeError: If CUDA is unavailable for cuRobo.
        ValueError: If ``robot_uid`` is missing or the robot is not found.
    """

    supported_move_types = frozenset({MoveType.EEF_MOVE, MoveType.JOINT_MOVE})
    supports_collision_world_updates = True

    @property
    def preserve_plan_samples(self) -> bool:
        """Whether callers must retain this planner's raw samples exactly.

        Mirrors :attr:`CuroboPlannerCfg.preserve_plan_samples`; read by the
        atomic-action motion adapter to decide whether to resample the returned
        trajectory to the action's ``sample_interval``.
        """
        return self.cfg.preserve_plan_samples

    @property
    def dynamic_collision_entity_ids(self) -> tuple[str, ...]:
        """Return canonical registry IDs accepted for dynamic pose updates."""
        return tuple(self.cfg.world.dynamic_obstacle_names)

    @property
    def collision_world_entity_ids(self) -> tuple[str, ...]:
        """Return every obstacle ID represented in the generated world."""
        return tuple(
            name for name, _ in _named_rigid_objects(self.cfg.world.rigid_objects)
        )

    @property
    def collision_world_batch_mode(self) -> Literal["shared", "per_env"]:
        """Return the configured collision-world batching policy."""
        return "per_env" if self.cfg.world.multi_env else "shared"

    def __init__(self, cfg: CuroboPlannerCfg) -> None:
        super().__init__(cfg)
        self.cfg: CuroboPlannerCfg = cfg
        self.device = torch.device(self.robot.device)
        self._curobo_device = _resolve_curobo_device(
            cfg.cuda_device,
            self.device,
        )
        # cuRobo and Warp contain a few current-device-sensitive initialization
        # paths, so select the dedicated planning GPU before importing them.
        torch.cuda.set_device(self._curobo_device)
        self._bindings = _require_curobo(cfg.log_level)
        self._backend_cache: dict[tuple[str, int, bool, MoveType], "_CuroboBackend"] = (
            {}
        )
        world_cfg = cfg.world
        if world_cfg.obstacle_representation not in ("cuboid", "mesh", "sphere"):
            logger.log_error(
                "CuroboWorldCfg.obstacle_representation must be 'cuboid', 'mesh', "
                f"or 'sphere', got {world_cfg.obstacle_representation!r}.",
                ValueError,
            )
        if (
            world_cfg.dynamic_obstacle_names
            and world_cfg.obstacle_representation == "sphere"
        ):
            logger.log_error(
                "Dynamic obstacle updates require the 'cuboid' or 'mesh' world "
                "representation. Sphere fitting expands one RigidObject into "
                "multiple independent obstacles that cannot be updated by the "
                "original object name.",
                ValueError,
            )
        if cfg.warmup_iterations < 0:
            logger.log_error(
                "CuroboPlannerCfg.warmup_iterations must be non-negative.",
                ValueError,
            )
        if (
            cfg.capture_acquire_timeout is not None
            and cfg.capture_acquire_timeout < 0.0
        ):
            logger.log_error(
                "CuroboPlannerCfg.capture_acquire_timeout must be non-negative "
                "or None.",
                ValueError,
            )
        if cfg.cuda_graph_capture_error_mode not in (
            "global",
            "thread_local",
            "relaxed",
        ):
            logger.log_error(
                "CuroboPlannerCfg.cuda_graph_capture_error_mode must be "
                "'global', 'thread_local', or 'relaxed'; got "
                f"{cfg.cuda_graph_capture_error_mode!r}.",
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

    def prepare_backend(
        self,
        *,
        control_part: str,
        batch_size: int,
        move_type: MoveType = MoveType.EEF_MOVE,
    ) -> dict[str, object]:
        """Materialize and warm one lazy cuRobo backend without planning a case.

        This explicit lifecycle hook lets deployment tooling and benchmarks
        separate one-time robot/world YAML generation, collision-sphere setup,
        CUDA graph capture, and cuRobo warmup from the first real planning call.
        Repeated calls for the same backend key reuse the cached backend.

        Args:
            control_part: Robot control part to prepare.
            batch_size: Goal batch size used by the future planning calls.
            move_type: Goal type whose cuRobo buffers and graph are prepared.

        Returns:
            Metadata describing the resolved backend and actual CUDA graph mode.

        Raises:
            ValueError: If the batch size or move type is unsupported.
        """
        if batch_size < 1:
            logger.log_error("batch_size must be >= 1.", ValueError)
        if move_type not in self.supported_move_types:
            logger.log_error(
                f"cuRobo cannot prepare unsupported move type {move_type}.",
                ValueError,
            )
        robot_batch_size = int(getattr(self.robot, "num_instances", 1))
        if batch_size not in (1, robot_batch_size):
            logger.log_error(
                f"batch_size={batch_size} must be 1 or robot.num_instances="
                f"{robot_batch_size}.",
                ValueError,
            )
        backend = self._get_backend(control_part, batch_size, move_type)
        return {
            "control_part": backend.control_part,
            "batch_size": backend.batch_size,
            "move_type": backend.planning_mode.name,
            "multi_env": bool(self.cfg.world.multi_env),
            "use_cuda_graph": backend.use_cuda_graph,
        }

    def with_collision_world(
        self,
        options: PlanOptions,
        *,
        obstacle_poses: Mapping[str, torch.Tensor],
    ) -> CuroboPlanOptions:
        """Bind snapshot obstacle poses to one cuRobo planning attempt.

        Args:
            options: Reusable caller options copied by the atomic-action layer.
            obstacle_poses: Batched simulator-world poses keyed by configured
                dynamic obstacle name.

        Returns:
            cuRobo options containing an owned obstacle-pose mapping.
        """
        if not isinstance(options, CuroboPlanOptions):
            logger.log_error("CuroboPlanner requires CuroboPlanOptions", TypeError)
        merged = {
            name: pose.clone()
            for name, pose in (options.dynamic_obstacle_poses or {}).items()
        }
        merged.update({name: pose.clone() for name, pose in obstacle_poses.items()})
        options.dynamic_obstacle_poses = merged or None
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
        move_types = {target.move_type for target in target_states}
        unsupported = move_types.difference((MoveType.EEF_MOVE, MoveType.JOINT_MOVE))
        if unsupported:
            logger.log_error(
                f"cuRobo does not support move types {sorted(str(x) for x in unsupported)}.",
                ValueError,
            )
        backends = {
            move_type: self._get_backend(
                control_part,
                start.shape[0],
                move_type,
            )
            for move_type in move_types
        }
        transform_backend = backends.get(
            MoveType.EEF_MOVE, next(iter(backends.values()))
        )
        # Compute the live sim base pose + its inverse once per plan and reuse it
        # across every EEF segment and every dynamic-obstacle update (the robot
        # does not move during planning), instead of re-querying get_link_pose +
        # re-inverting per segment / obstacle. Skipped for pure
        # joint-move plans with no dynamic obstacles, which never need it.
        needs_base_pose = bool(options.dynamic_obstacle_poses) or any(
            t.move_type == MoveType.EEF_MOVE for t in target_states
        )
        sim_base_pose_inv = (
            pose_inv(self._get_sim_base_pose(transform_backend, start.shape[0]))
            if needs_base_pose
            else None
        )
        for backend in backends.values():
            self.update_dynamic_obstacles(
                options.dynamic_obstacle_poses, backend, sim_base_pose_inv
            )
        return self._plan_segments(
            target_states, start, backends, options, sim_base_pose_inv
        )

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
            start_qpos, dtype=torch.float32, device=self._curobo_device
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
        """Return one independent cuRobo scene mapping for every batch row.

        The auto-generated YAML contains env 0's static scene. Cloning it makes
        the collision worlds independently addressable but does not discover
        per-env simulator poses; dynamic-obstacle updates apply those later.
        """
        if batch_size < 1:
            logger.log_error(
                f"multi-env cuRobo batch_size must be positive, got {batch_size}.",
                ValueError,
            )
        if world_config_path is None:
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
        control_part: str,
        batch_size: int,
        planning_mode: MoveType = MoveType.EEF_MOVE,
    ) -> "_CuroboBackend":
        """Return a cached in-process backend for one goal-buffer shape."""
        multi_env = bool(self.cfg.world.multi_env)
        key = (control_part, int(batch_size), multi_env, planning_mode)
        if key in self._backend_cache:
            return self._backend_cache[key]

        profile = self._materialize_profile(control_part)
        sim_joint_names = self._resolve_sim_joint_names(control_part)
        world_cfg = self.cfg.world
        collision_cache = (
            dict(world_cfg.collision_cache) if world_cfg.collision_cache else None
        )
        world_config_path = (
            self._auto_generate_world_yaml(world_cfg)
            if world_cfg.rigid_objects
            else None
        )
        scene_model: str | list[dict] | None = world_config_path
        if multi_env:
            scene_model = self._materialize_multi_env_scene_model(
                world_config_path, int(batch_size)
            )

        use_cuda_graph = bool(self.cfg.use_cuda_graph)
        coordinator = None
        capture_owner = _CaptureOwner()
        capture_acquired = False
        if use_cuda_graph:
            coordinator = _get_capture_coordinator()
            capture_acquired = coordinator.acquire_for_capture(
                str(self._curobo_device),
                capture_owner,
                timeout=self.cfg.capture_acquire_timeout,
                wait_log_interval=self.cfg.capture_wait_log_interval,
            )
            if not capture_acquired:
                if not self.cfg.cuda_graph_fallback:
                    raise RuntimeError(
                        "Timed out waiting for coordinated cuRobo CUDA graph "
                        f"capture on {self._curobo_device}."
                    )
                logger.log_warning(
                    "Timed out waiting for coordinated cuRobo CUDA graph capture "
                    f"on {self._curobo_device}; waiting for the active capture "
                    "to finish, then using non-graph mode."
                )
                use_cuda_graph = False
                # Non-graph initialization still launches CUDA kernels. Wait
                # for the active capture to finish so those launches cannot
                # invalidate a peer's stream capture.
                capture_acquired = coordinator.acquire_for_capture(
                    str(self._curobo_device),
                    capture_owner,
                    timeout=None,
                    wait_log_interval=self.cfg.capture_wait_log_interval,
                )
                if not capture_acquired:
                    raise RuntimeError(
                        "Unable to coordinate safe non-graph cuRobo "
                        f"initialization on {self._curobo_device}."
                    )

        backend: _CuroboBackend
        try:
            backend = self._build_backend(
                control_part=control_part,
                batch_size=int(batch_size),
                profile=profile,
                sim_joint_names=sim_joint_names,
                scene_model=scene_model,
                collision_cache=collision_cache,
                use_cuda_graph=use_cuda_graph,
                planning_mode=planning_mode,
            )
            try:
                self._warmup_backend(backend)
            except Exception as exc:
                self._close_planner(backend.planner)
                if not use_cuda_graph:
                    raise
                raise RuntimeError(
                    "cuRobo CUDA graph warmup failed after capture may have "
                    "started. The CUDA context may now be invalid, so an "
                    "in-process non-graph fallback is unsafe; restart the "
                    "simulator process. Original error: "
                    f"{exc}"
                ) from exc
        finally:
            if capture_acquired and coordinator is not None:
                try:
                    coordinator.release_for_capture(
                        str(self._curobo_device), capture_owner
                    )
                except Exception as exc:
                    logger.log_warning(
                        f"cuRobo capture coordinator release failed: {exc}"
                    )

        self._backend_cache[key] = backend
        logger.log_info(
            f"cuRobo in-process backend ready for '{control_part}' "
            f"(batch={batch_size}, mode={planning_mode.name}, "
            f"cuda_graph={backend.use_cuda_graph})."
        )
        return backend

    def _build_backend(
        self,
        *,
        control_part: str,
        batch_size: int,
        profile: _CuroboProfile,
        sim_joint_names: list[str],
        scene_model: str | list[dict] | None,
        collision_cache: dict[str, int | dict[str, int | float | list[float]]] | None,
        use_cuda_graph: bool,
        planning_mode: MoveType,
    ) -> "_CuroboBackend":
        """Construct and validate one cuRobo planner on the selected CUDA device."""
        with torch.cuda.device(self._curobo_device):
            planner_cfg = self._bindings.MotionPlannerCfg.create(
                robot=profile.robot_config_path,
                scene_model=scene_model,
                collision_cache=collision_cache,
                device_cfg=self._bindings.DeviceCfg(device=self._curobo_device),
                max_batch_size=batch_size,
                multi_env=bool(self.cfg.world.multi_env),
                optimizer_collision_activation_distance=(
                    self.cfg.collision_activation_distance
                ),
                use_cuda_graph=use_cuda_graph,
            )
            # cuRobo 0.8 reads interpolation_dt from the trajectory optimizer
            # config rather than accepting it in MotionPlannerCfg.create().
            planner_cfg.trajopt_solver_config.interpolation_dt = float(
                self.cfg.interpolation_dt
            )
            planner = (
                self._bindings.MotionPlanner(planner_cfg)
                if batch_size == 1
                else self._bindings.BatchMotionPlanner(planner_cfg)
            )

        try:
            self._validate_profile_joint_names(
                profile, sim_joint_names, list(planner.joint_names)
            )
            self._validate_base_link_name(profile, planner)
            tool_frame = self._resolve_tool_frame(profile, planner)
        except Exception:
            self._close_planner(planner)
            raise
        return _CuroboBackend(
            planner=planner,
            control_part=control_part,
            sim_joint_names=sim_joint_names,
            tool_frame=tool_frame,
            profile=profile,
            batch_size=batch_size,
            use_cuda_graph=use_cuda_graph,
            planning_mode=planning_mode,
        )

    def _warmup_backend(self, backend: "_CuroboBackend") -> None:
        """Warm one goal type without forcing cuRobo to reset captured graphs.

        cuRobo uses structurally different trajectory-optimizer goal buffers for
        pose and c-space solves. Its CUDA backend cannot reset captured graphs,
        so each cached backend is warmed only for its declared planning mode.
        """
        iterations = int(self.cfg.warmup_iterations)
        if not backend.use_cuda_graph and iterations == 0:
            return
        if backend.use_cuda_graph:
            iterations = max(iterations, 1)
        self._synchronize_device()
        capture_mode = (
            _torch_cuda_graph_capture_mode(self.cfg.cuda_graph_capture_error_mode)
            if backend.use_cuda_graph
            else nullcontext()
        )
        with capture_mode, torch.cuda.device(self._curobo_device):
            planner = backend.planner
            default_position = planner.default_joint_state.position
            if default_position.dim() == 1:
                default_position = default_position.unsqueeze(0)
            if default_position.shape[0] == 1 and backend.batch_size > 1:
                default_position = default_position.expand(
                    backend.batch_size, -1
                ).clone()
            current_state = self._bindings.JointState.from_position(
                default_position,
                joint_names=list(planner.joint_names),
            )
            original_exit_early = planner.ik_solver.config.exit_early
            planner.ik_solver.config.exit_early = False
            try:
                for _ in range(iterations):
                    goal_state = current_state.clone()
                    goal_state.position[..., 0] += 0.2
                    if backend.planning_mode == MoveType.EEF_MOVE:
                        goal = planner.compute_kinematics(
                            goal_state
                        ).tool_poses.as_goal()
                        planner.plan_pose(
                            goal,
                            current_state,
                            max_attempts=1,
                            enable_graph_attempt=1,
                        )
                    elif backend.planning_mode == MoveType.JOINT_MOVE:
                        planner.plan_cspace(
                            goal_state,
                            current_state,
                            max_attempts=1,
                            enable_graph_attempt=1,
                        )
                    else:  # pragma: no cover - validated before backend creation
                        raise ValueError(
                            f"Unsupported cuRobo warmup mode {backend.planning_mode}."
                        )
                    planner.reset_seed()

                graph_planner = getattr(planner, "graph_planner", None)
                if backend.use_cuda_graph and graph_planner is not None:
                    graph_planner.warmup(num_warmup_iterations=iterations)
            finally:
                planner.ik_solver.config.exit_early = original_exit_early
        self._synchronize_device()

    def _synchronize_device(self) -> None:
        """Synchronize only the CUDA device used by this planner."""
        torch.cuda.synchronize(self._curobo_device)

    @staticmethod
    def _close_planner(planner: "Any") -> None:
        """Best-effort release of a cuRobo planner's graph resources."""
        close_fn = getattr(planner, "close", None) or getattr(planner, "destroy", None)
        if close_fn is not None:
            try:
                close_fn()
            except Exception:
                pass

    def _validate_profile_joint_names(
        self,
        profile: _CuroboProfile,
        sim_joint_names: list[str],
        curobo_joint_names: list[str],
    ) -> None:
        """Validate the auto-derived joint mapping before a CUDA planning call."""
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
        mapped_set = set(mapped_names)
        unmapped = [name for name in curobo_joint_names if name not in mapped_set]
        if unmapped:
            logger.log_error(
                "cuRobo planner exposes joints outside the requested control "
                f"part: {unmapped}. Lock non-controlled joints in the V2 robot "
                "profile or select a control part that includes them.",
                ValueError,
            )

    def _resolve_tool_frame(self, profile: _CuroboProfile, planner: "Any") -> str:
        """Resolve and validate the V2 tool frame used for pose goals."""
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

    @staticmethod
    def _validate_base_link_name(profile: _CuroboProfile, planner: "Any") -> None:
        """Ensure the auto-derived base link matches the loaded V2 model."""
        expected = profile.base_link_name
        if expected is None:
            return
        actual = getattr(getattr(planner, "kinematics", None), "base_link", None)
        if actual is None:
            logger.log_error(
                "cuRobo planner did not expose kinematics.base_link, so "
                f"base_link_name={expected!r} cannot be validated.",
                ValueError,
            )
        if actual != expected:
            logger.log_error(
                f"Auto-derived base_link_name={expected!r} does not match the "
                f"loaded cuRobo V2 base link {actual!r}.",
                ValueError,
            )

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

        # cuRobo's base is the auto-generated YAML's ``base_link`` (the URDF root
        # link), NOT the solver's control-part root.  For robots whose control
        # part spans the whole arm (franka, ur) the two coincide, but for a
        # control part that is a sub-chain of a larger robot - e.g. w1
        # ``right_arm`` whose root ``right_arm_base`` hangs off a locked torso -
        # they differ.  Cartesian goals and dynamic obstacle poses are converted
        # into this base frame (see :meth:`_sim_world_to_curobo_base_pose`), so it
        # must match the frame cuRobo actually plans in; otherwise cuRobo receives
        # a goal expressed in the control-part base and interprets it in the URDF
        # root, planning to a wrong pose.
        solver_base_link = (
            getattr(solver, "root_link_name", None) if solver is not None else None
        )

        sim_joints = self._resolve_sim_joint_names(control_part)
        sim_to_curobo = {j: j for j in sim_joints}

        robot_config_path = self._auto_generate_robot_yaml(control_part, tool_frame)
        base_link = self._read_curobo_base_link(robot_config_path) or solver_base_link
        sim_base_link = base_link

        return _CuroboProfile(
            robot_config_path=robot_config_path,
            sim_to_curobo_joint_names=sim_to_curobo,
            tool_frame_name=tool_frame,
            tool_frame_to_tcp=tool_frame_to_tcp,
            base_link_name=base_link,
            sim_base_link_name=sim_base_link,
            sim_base_to_curobo_base=self.cfg.sim_base_to_curobo_base,
        )

    @staticmethod
    def _read_curobo_base_link(robot_yaml_path: str) -> str | None:
        """Return cuRobo's ``base_link`` from an auto-generated robot YAML.

        The YAML's ``robot_cfg.kinematics.base_link`` is the URDF root link cuRobo
        roots its kinematics at - the frame Cartesian goals must be expressed in.
        Reading it back (rather than assuming it equals the solver's control-part
        root) keeps the parent's frame conversion in sync with cuRobo's actual
        model for robots whose control part is a sub-chain of a larger URDF.

        Args:
            robot_yaml_path: Path to the cached cuRobo robot YAML.

        Returns:
            The base link name, or ``None`` if the YAML cannot be read.
        """
        try:
            import yaml

            with open(robot_yaml_path, "r") as fh:
                data = yaml.safe_load(fh)
            return data["robot_cfg"]["kinematics"]["base_link"]
        except Exception:  # noqa: BLE001 - fall back to the solver root upstream
            return None

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
        # cuRobo's robot YAML is generated from the *assembled* URDF
        # (robot.cfg.fpath), which includes every mounted component (arm +
        # gripper). A solver's ``urdf_path`` may be a sub-chain URDF (e.g. the
        # UR arm's bare UR10 URDF hardcoded in URSolverCfg) that omits the
        # gripper; keying the cache on it would reuse a stale, gripper-less YAML
        # even after the gripper is attached, and would not invalidate when the
        # gripper changes. Use robot.cfg.fpath for both the cache key and
        # generation so they stay consistent and the gripper links are included.
        urdf_path = robot.cfg.fpath
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
            urdf_path=urdf_path,
            fit_type=auto.fit_type,
            num_spheres=auto.num_spheres,
            sphere_density=auto.sphere_density,
            surface_radius=auto.surface_radius,
            iterations=auto.iterations,
            collision_sphere_buffer=auto.collision_sphere_buffer,
            device=str(self._curobo_device),
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
        hasher.update(_CUROBO_ROBOT_YAML_GENERATOR_VERSION.encode("utf-8"))
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
            device=str(self._curobo_device),
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
        for name, obj in _named_rigid_objects(world_cfg.rigid_objects):
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
        backends: dict[MoveType, "_CuroboBackend"],
        options: CuroboPlanOptions,
        sim_base_pose_inv: torch.Tensor | None = None,
    ) -> PlanResult:
        """Plan each waypoint segment sequentially and assemble a PlanResult.

        Each segment's goal is converted to the cuRobo base frame and solved by
        the cached in-process V2 planner. Segment extraction, planning-time
        budget checks, junction de-duplication, and rectangular assembly all
        stay on cuRobo's CUDA device. The assembled result is copied to the
        simulation device once at the API boundary.
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
        alive = torch.ones(B, dtype=torch.bool, device=self._curobo_device)
        current = start.clone()

        for seg_idx, target in enumerate(target_states):
            self._validate_segment_batch(target, B, seg_idx)
            backend = backends[target.move_type]
            current_state = self._to_curobo_joint_state(current, backend)
            if target.move_type == MoveType.EEF_MOVE:
                if target.xpos is None:
                    logger.log_error(
                        f"Segment {seg_idx} EEF_MOVE target missing xpos.",
                        ValueError,
                    )
                goal = self._to_curobo_pose_goal(
                    target.xpos, backend, sim_base_pose_inv
                )
                start_time = time.time()
                capture_mode = (
                    _torch_cuda_graph_capture_mode(
                        self.cfg.cuda_graph_capture_error_mode
                    )
                    if backend.use_cuda_graph
                    else nullcontext()
                )
                with capture_mode, torch.cuda.device(self._curobo_device):
                    v2_result = backend.planner.plan_pose(
                        goal, current_state, max_attempts=max_attempts
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
                goal_state = self._to_curobo_joint_goal(target.qpos, backend)
                start_time = time.time()
                capture_mode = (
                    _torch_cuda_graph_capture_mode(
                        self.cfg.cuda_graph_capture_error_mode
                    )
                    if backend.use_cuda_graph
                    else nullcontext()
                )
                with capture_mode, torch.cuda.device(self._curobo_device):
                    v2_result = backend.planner.plan_cspace(
                        goal_state, current_state, max_attempts=max_attempts
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
                seg_success = torch.zeros(
                    B, dtype=torch.bool, device=self._curobo_device
                )
                seg_positions = current.unsqueeze(1)
                seg_dt = torch.zeros(
                    B, 1, dtype=torch.float32, device=self._curobo_device
                )
            else:
                seg_success, seg_positions, seg_dt = self._extract_segment(
                    v2_result, backend
                )
            seg_success = seg_success.to(self._curobo_device) & alive
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
        success = success.to(torch.bool).to(self._curobo_device)

        traj = v2_result.interpolated_trajectory
        position = torch.as_tensor(traj.position)
        if position.dim() == 4:
            position = position[:, 0, :, :]  # select seed 0: (B, T, D_full)

        last_tstep = torch.as_tensor(v2_result.interpolated_last_tstep)
        if last_tstep.dim() == 2:
            last_tstep = last_tstep.squeeze(-1)

        B, T, D = position.shape
        # Compute the per-env valid length once. This scalar extraction is the
        # only synchronization needed before rectangular trajectory assembly.
        max_len = max(int((last_tstep + 1).max().item()), 1)
        cap = min(max_len, T)
        lengths = (last_tstep + 1).clamp(min=1, max=cap).long().to(self._curobo_device)
        # A single gather both trims to each env's length and pads by repeating
        # the last valid sample: src[b, t] = t if t < length[b] else length[b] - 1.
        # cap <= T guarantees src < T, so the gather never indexes out of bounds.
        position = position.float().to(self._curobo_device)
        arange = torch.arange(max_len, device=self._curobo_device)
        src = torch.where(
            arange[None, :] < lengths[:, None],
            arange[None, :],
            lengths[:, None] - 1,
        ).long()
        full = position.gather(1, src.unsqueeze(-1).expand(-1, -1, D))

        seg_positions = self._map_curobo_to_sim(full, traj.joint_names, backend)
        seg_dt = self._extract_dt(traj, lengths, max_len, B)
        return success, seg_positions, seg_dt

    def _map_curobo_to_sim(
        self,
        full_positions: torch.Tensor,
        curobo_joint_names: list[str],
        backend: "_CuroboBackend",
    ) -> torch.Tensor:
        """Map a full cuRobo trajectory to simulator control-part joint order.

        The cuRobo joint order is fixed for a planner's life, so the column
        gather index is built once (cached on ``backend``) and reused instead of
        recomputing O(D^2) ``.index()`` lookups on every segment.
        """
        sig = tuple(curobo_joint_names)
        if (
            backend.curobo_joint_names_sig != sig
            or backend.curobo_to_sim_col_idx is None
        ):
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
            backend.curobo_to_sim_col_idx = torch.as_tensor(
                cols, dtype=torch.long, device=self._curobo_device
            )
            backend.curobo_joint_names_sig = sig
        return full_positions[..., backend.curobo_to_sim_col_idx].to(
            dtype=torch.float32
        )

    def _extract_dt(
        self,
        traj: "Any",
        lengths: torch.Tensor,
        max_len: int,
        B: int,
    ) -> torch.Tensor:
        """Derive ``(B, max_len)`` per-point deltas from a V2 trajectory.

        cuRobo V2 uses a scalar ``dt`` per batch/seed for interpolated
        trajectories. EmbodiChain represents deltas at each trajectory point,
        with a zero first point and one interval per following point. ``lengths``
        is the per-env valid-length tensor (computed once in
        :meth:`_extract_segment` and reused here) so this is a vectorized mask
        instead of a per-env Python loop with ``.item()`` syncs.
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
                device=self._curobo_device,
                dtype=torch.float32,
            )
        if dt.shape[0] == 1 and B > 1:
            dt = dt.expand(B, -1)
        if dt.shape[0] != B:
            logger.log_error(
                f"cuRobo trajectory dt batch {dt.shape[0]} does not match {B}.",
                ValueError,
            )

        out = torch.zeros(B, max_len, device=self._curobo_device, dtype=torch.float32)
        if dt.shape[-1] == 1:
            # Scalar dt per env: out[b, t] = interval[b] for 1 <= t < length[b],
            # else 0 - one vectorized mask multiply (was a per-env Python loop).
            interval = dt[:, 0].to(self._curobo_device, dtype=torch.float32)
            arange = torch.arange(max_len, device=self._curobo_device)
            mask = (arange[None, :] >= 1) & (arange[None, :] < lengths[:, None])
            return interval[:, None] * mask

        # Preserve an explicitly per-point delta sequence supplied by a V2
        # result or a compatible future API. It already includes the first
        # point's zero delta in EmbodiChain's convention.
        length = min(dt.shape[-1], max_len)
        out[:, :length] = dt[:, :length].to(self._curobo_device, dtype=torch.float32)
        return out

    def _extract_total_time(self, v2_result: "Any", B: int) -> torch.Tensor:
        """Return a ``(B,)`` total planning time tensor for budget validation."""
        tt = v2_result.total_time
        if isinstance(tt, torch.Tensor):
            if tt.dim() == 0:
                return tt.unsqueeze(0).expand(B).to(self._curobo_device)
            if tt.dim() == 2:
                tt = tt.squeeze(-1)
            return tt[:B].to(self._curobo_device)
        return torch.full((B,), float(tt), device=self._curobo_device)

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
        # One D2H sync for the whole batch (was B per-env `if alive[b]:` syncs,
        # each forcing the GPU pipeline to drain). The rest of the loop reads
        # Python bools and GPU tensors whose .shape / .cat do not sync.
        alive_list = alive.tolist()
        env_lengths: list[int] = []
        for b in range(B):
            if alive_list[b]:
                env_lengths.append(sum(s.shape[0] for s in per_env_samples[b]))
            else:
                env_lengths.append(1)
        max_len = max(env_lengths) if env_lengths else 1

        positions = torch.zeros(
            B, max_len, D, device=self._curobo_device, dtype=torch.float32
        )
        dt = torch.zeros(B, max_len, device=self._curobo_device, dtype=torch.float32)
        for b in range(B):
            if alive_list[b]:
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
            success=alive.to(self.device),
            positions=positions.to(self.device),
            dt=dt.to(self.device),
            duration=duration.to(self.device),
        )

    # ------------------------------------------------------------------
    # cuRobo state / goal construction
    # ------------------------------------------------------------------

    def _to_curobo_joint_state(
        self, current: torch.Tensor, backend: "_CuroboBackend"
    ) -> "Any":
        """Build a cuRobo ``JointState`` from simulator-order joint positions."""
        if current.dim() != 2 or current.shape[1] != len(backend.sim_joint_names):
            logger.log_error(
                "cuRobo start/goal qpos must have shape "
                f"(B, {len(backend.sim_joint_names)}), got {tuple(current.shape)}.",
                ValueError,
            )
        curobo_names = list(backend.planner.joint_names)
        if backend.sim_to_curobo_col_idx is None:
            curobo_to_sim = {
                backend.profile.sim_to_curobo_joint_names[sim_name]: idx
                for idx, sim_name in enumerate(backend.sim_joint_names)
            }
            backend.sim_to_curobo_col_idx = torch.as_tensor(
                [curobo_to_sim[name] for name in curobo_names],
                dtype=torch.long,
                device=self._curobo_device,
            )
        position = current.to(self._curobo_device, dtype=torch.float32).index_select(
            -1, backend.sim_to_curobo_col_idx
        )
        return self._bindings.JointState.from_position(
            position, joint_names=curobo_names
        )

    def _to_curobo_pose_goal(
        self,
        xpos: torch.Tensor,
        backend: "_CuroboBackend",
        sim_base_pose_inv: torch.Tensor | None = None,
    ) -> "Any":
        """Build a cuRobo pose goal from a simulator-world TCP pose."""
        goal_matrix = self._to_curobo_base_tool_matrix(xpos, backend, sim_base_pose_inv)
        position, quaternion = _matrix_to_position_quaternion(goal_matrix)
        pose = self._bindings.Pose(position=position, quaternion=quaternion)
        return self._bindings.GoalToolPose.from_poses(
            {backend.tool_frame: pose},
            ordered_tool_frames=[backend.tool_frame],
            num_goalset=1,
        )

    def _to_curobo_joint_goal(
        self, qpos: torch.Tensor, backend: "_CuroboBackend"
    ) -> "Any":
        """Build a cuRobo c-space goal from simulator-order joint positions."""
        qpos = torch.as_tensor(qpos, dtype=torch.float32, device=self._curobo_device)
        if qpos.dim() == 1:
            qpos = qpos.unsqueeze(0)
        return self._to_curobo_joint_state(qpos, backend)

    def _tcp_to_tool_pose(
        self, tcp_pose: torch.Tensor, backend: "_CuroboBackend"
    ) -> torch.Tensor:
        """Convert a simulator TCP goal into the configured cuRobo tool frame."""
        if tcp_pose.dim() != 3 or tcp_pose.shape[-2:] != (4, 4):
            logger.log_error(
                f"Expected (B, 4, 4) TCP pose matrices, got {tuple(tcp_pose.shape)}.",
                ValueError,
            )
        tool_to_frame = self._tool_to_frame_matrix(backend)
        if tool_to_frame is None:
            return tcp_pose
        return tcp_pose @ tool_to_frame

    def _tool_to_frame_matrix(self, backend: "_CuroboBackend") -> torch.Tensor | None:
        """Cached inverse of the profile's fixed tool_frame->TCP transform.

        ``None`` means the tool frame is already the TCP (the common auto-derived
        case). Built once per backend and reused across plans instead of calling
        ``torch.linalg.inv`` on every EEF segment.
        """
        if backend.tool_to_frame_matrix is not None:
            return backend.tool_to_frame_matrix
        profile = backend.profile
        if profile.tool_frame_to_tcp is None:
            return None
        frame_to_tcp = torch.as_tensor(
            profile.tool_frame_to_tcp,
            dtype=torch.float32,
            device=self._curobo_device,
        )
        if frame_to_tcp.shape != (4, 4):
            logger.log_error(
                "tool_frame_to_tcp must be a homogeneous (4, 4) transform, "
                f"got {tuple(frame_to_tcp.shape)}.",
                ValueError,
            )
        backend.tool_to_frame_matrix = pose_inv(frame_to_tcp)
        return backend.tool_to_frame_matrix

    def _sim_world_to_curobo_base_pose(
        self,
        world_pose: torch.Tensor,
        backend: "_CuroboBackend",
        sim_base_pose_inv: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Express simulator-world poses in the loaded cuRobo base frame.

        EmbodiChain pose targets and dynamic obstacle poses are world poses,
        while a cuRobo robot profile/world is rooted at the profile's base
        link. The live simulator base pose accounts for arena offsets and
        mobile bases; ``sim_base_to_curobo_base`` accounts for any fixed frame
        convention difference between the two robot descriptions.

        ``sim_base_pose_inv`` is the precomputed inverse of the live sim base
        pose; the :meth:`plan` hot path passes it so K segments and N dynamic
        obstacles reuse one inverse (the robot does not move during planning).
        The public path leaves it ``None`` and computes it here via
        :func:`pose_inv` (closed-form, cheaper and more stable than
        ``torch.linalg.inv``).
        """
        if world_pose.dim() != 3 or world_pose.shape[-2:] != (4, 4):
            logger.log_error(
                f"Expected (B, 4, 4) simulator-world pose matrices, got "
                f"{tuple(world_pose.shape)}.",
                ValueError,
            )
        batch_size = world_pose.shape[0]
        if sim_base_pose_inv is None:
            sim_base_pose = self._get_sim_base_pose(backend, batch_size)
            sim_base_pose_inv = pose_inv(sim_base_pose)
        sim_base_to_curobo = self._sim_base_to_curobo_matrix(backend).expand(
            batch_size, -1, -1
        )
        return torch.bmm(
            sim_base_to_curobo,
            torch.bmm(sim_base_pose_inv, world_pose),
        )

    def _sim_base_to_curobo_matrix(self, backend: "_CuroboBackend") -> torch.Tensor:
        """Cached fixed sim-base -> cuRobo-base transform (eye when ``None``).

        Built once per backend and reused across plans instead of
        ``torch.as_tensor``-ing the profile list on every call.
        """
        if backend.sim_base_to_curobo_base_matrix is not None:
            return backend.sim_base_to_curobo_base_matrix
        profile_transform = backend.profile.sim_base_to_curobo_base
        if profile_transform is None:
            matrix = torch.eye(4, dtype=torch.float32, device=self._curobo_device)
        else:
            matrix = torch.as_tensor(
                profile_transform,
                dtype=torch.float32,
                device=self._curobo_device,
            )
            if matrix.shape != (4, 4):
                logger.log_error(
                    "sim_base_to_curobo_base must be a homogeneous (4, 4) "
                    f"transform, got {tuple(matrix.shape)}.",
                    ValueError,
                )
        backend.sim_base_to_curobo_base_matrix = matrix
        return matrix

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
        base_pose = torch.as_tensor(
            base_pose, dtype=torch.float32, device=self._curobo_device
        )
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
        sim_base_pose_inv: torch.Tensor | None = None,
    ) -> None:
        """Update named dynamic obstacle poses on cached cuRobo collision worlds.

        Args:
            poses: Mapping of canonical obstacle ID -> ``(B, 4, 4)`` world pose.
                ``None`` is a no-op.
            backend: Specific cached backend to update. If ``None``, updates all
                cached backends.
            sim_base_pose_inv: Precomputed inverse of the live sim base pose for
                ``backend``'s batch size, reused across all obstacles. Only
                consulted when its batch matches the obstacle pose batch.
        """
        if poses is None:
            return
        _validate_dynamic_obstacles(poses, list(self.cfg.world.dynamic_obstacle_names))
        backends = (
            [backend] if backend is not None else list(self._backend_cache.values())
        )
        if backend is None and self.cfg.world.multi_env:
            batch_sizes = {cached.batch_size for cached in backends}
            if len(batch_sizes) > 1:
                logger.log_error(
                    "Cannot update all cached multi-env cuRobo backends with "
                    "different batch sizes. Pass the intended backend explicitly.",
                    ValueError,
                )

        inv_cache: dict[int, torch.Tensor] = {}
        for name, pose_tensor in poses.items():
            pose_tensor = torch.as_tensor(
                pose_tensor, device=self._curobo_device, dtype=torch.float32
            )
            b = pose_tensor.shape[0]
            for cached_backend in backends:
                key = id(cached_backend)
                inv = inv_cache.get(key)
                if inv is None or inv.shape[0] != b:
                    if (
                        backend is not None
                        and sim_base_pose_inv is not None
                        and sim_base_pose_inv.shape[0] == b
                    ):
                        inv = sim_base_pose_inv
                    else:
                        inv = pose_inv(self._get_sim_base_pose(cached_backend, b))
                    inv_cache[key] = inv
                curobo_pose = self._sim_world_to_curobo_base_pose(
                    pose_tensor, cached_backend, inv
                )
                self._update_backend_obstacle(name, curobo_pose, cached_backend)

    def _update_backend_obstacle(
        self, name: str, pose_tensor: torch.Tensor, backend: "_CuroboBackend"
    ) -> None:
        """Apply one obstacle pose tensor under the backend's world policy."""
        if self.cfg.world.multi_env:
            if pose_tensor.shape[0] != backend.batch_size:
                logger.log_error(
                    f"dynamic obstacle '{name}' has batch {pose_tensor.shape[0]}, "
                    f"but this multi-env backend expects {backend.batch_size}.",
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

        if pose_tensor.shape[0] > 1 and not torch.allclose(
            pose_tensor, pose_tensor[:1].expand_as(pose_tensor)
        ):
            logger.log_error(
                f"dynamic obstacle '{name}' has different poses across a shared "
                "cuRobo world. Enable world.multi_env for per-env worlds.",
                ValueError,
            )
        position, quaternion = _matrix_to_position_quaternion(pose_tensor[:1])
        pose = self._bindings.Pose(position=position[0], quaternion=quaternion[0])
        backend.planner.scene_collision_checker.update_obstacle_pose(
            name, pose, env_idx=0
        )

    # ------------------------------------------------------------------
    # In-process goal conversion and lifecycle
    # ------------------------------------------------------------------

    def _to_curobo_base_tool_matrix(
        self,
        xpos: torch.Tensor,
        backend: "_CuroboBackend",
        sim_base_pose_inv: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Convert a batched sim-world TCP pose to a cuRobo-base tool-frame matrix.

        Pure-tensor composition of :meth:`_sim_world_to_curobo_base_pose` and
        :meth:`_tcp_to_tool_pose`.
        ``sim_base_pose_inv`` is the per-plan cached base-pose inverse (see
        :meth:`plan`).
        """
        xpos = torch.as_tensor(xpos, device=self._curobo_device, dtype=torch.float32)
        xpos = self._sim_world_to_curobo_base_pose(xpos, backend, sim_base_pose_inv)
        xpos = self._tcp_to_tool_pose(xpos, backend)
        return xpos

    def close(self) -> None:
        """Destroy every cached in-process cuRobo planner."""
        for backend in list(self._backend_cache.values()):
            self._close_planner(backend.planner)
        self._backend_cache.clear()

    def __del__(self) -> None:  # pragma: no cover - best-effort GC cleanup
        try:
            self.close()
        except Exception:
            pass


@dataclass
class _CuroboBackend:
    """Cached in-process V2 planner and its EmbodiChain-side metadata."""

    planner: "Any"
    control_part: str
    sim_joint_names: list[str]
    tool_frame: str
    profile: _CuroboProfile
    batch_size: int
    use_cuda_graph: bool
    planning_mode: MoveType
    # Lazily-built device-tensor caches for the shared post-processing. The
    # cuRobo joint order and the profile's fixed transforms are stable for a
    # planner's life, so these are built once on first use and reused across
    # plans instead of recomputing per segment / per plan.
    sim_to_curobo_col_idx: torch.Tensor | None = None
    curobo_to_sim_col_idx: torch.Tensor | None = None
    curobo_joint_names_sig: tuple[str, ...] | None = None
    tool_to_frame_matrix: torch.Tensor | None = None
    sim_base_to_curobo_base_matrix: torch.Tensor | None = None
