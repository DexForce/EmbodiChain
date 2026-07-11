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
from dataclasses import MISSING
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
from .utils import PlanResult, PlanState

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
        planner_mod = importlib.import_module("curobo.planner")
        state_mod = importlib.import_module("curobo.types.state")
        math_mod = importlib.import_module("curobo.types.math")
        goal_mod = importlib.import_module("curobo.types.goal")
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
        BatchMotionPlanner=planner_mod.BatchMotionPlanner,
        JointState=state_mod.JointState,
        Pose=math_mod.Pose,
        GoalToolPose=goal_mod.GoalToolPose,
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
        ValueError: If ``robot_uid`` is missing or the robot is not found.
    """

    preinterpolate_targets = False
    preserve_plan_samples = True

    def __init__(self, cfg: CuroboPlannerCfg) -> None:
        super().__init__(cfg)
        self.cfg: CuroboPlannerCfg = cfg
        self._bindings = _require_curobo()
        # Cached V2 backends keyed by (control_part, batch_size, multi_env).
        self._backend_cache: dict[tuple, "Any"] = {}

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

        .. note::
            Implemented in the backend-integration task. The method currently
            raises ``NotImplementedError`` until the V2 planning path is wired.

        Args:
            target_states: List of :class:`PlanState` waypoints. ``EEF_MOVE``
                entries carry ``xpos`` ``(B, 4, 4)``; ``JOINT_MOVE`` entries
                carry ``qpos`` ``(B, controlled_dof)``.
            options: :class:`CuroboPlanOptions` carrying the runtime context.

        Returns:
            :class:`PlanResult` with env-batched tensors. Failed environments
            hold ``start_qpos``.
        """
        raise NotImplementedError("CuroboPlanner.plan is not implemented yet.")
