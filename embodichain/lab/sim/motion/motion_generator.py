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

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import MISSING
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import torch

from embodichain.lab.sim.motion.planners import (
    BasePlannerCfg,
    CollisionWorldInfo,
    PlanOptions,
    BasePlanner,
    ToppraPlanner,
    ToppraPlannerCfg,
    ToppraPlanOptions,
    TrapezoidalPlanner,
    TrapezoidalPlannerCfg,
    TrapezoidalPlanOptions,
    NeuralPlanner,
    NeuralPlannerCfg,
    CuroboPlanner,
    CuroboPlannerCfg,
    CuroboPlanOptions,
)
from embodichain.compute.trajectory import (
    interpolate_with_distance,
    interpolate_with_nums,
    resample_in_time,
    differentiate_positions,
)
from embodichain.compute.trajectory.interpolation import (
    _minimum_jerk_progress,
    interpolate_minimum_jerk,
)
from embodichain.utils import logger, configclass
from embodichain.utils.math import (
    axis_angle_from_quat,
    axis_angle_to_rotation_matrix,
    quat_error_magnitude,
    quat_from_matrix,
)
from .planners._joint_limits import _joint_velocity_limits, _plan_joint_ids
from .planners.utils import (
    MoveType,
    PlanResult,
    PlanState,
    TrajectorySampleMethod,
    calculate_point_allocations,
    interpolate_xpos_batched,
    normalize_success_mask,
)

__all__ = ["MotionGenerator", "MotionGenCfg", "MotionGenOptions"]

# Limit flattened targets before numerical solvers expand their multi-start
# seeds. One environment with many waypoints must not create an unbounded solve.
# Also bound the path span between seed anchors for single-environment requests.
_MAX_IK_BATCH_TARGETS = 512
_MAX_IK_BATCH_WAYPOINTS = 32


@configclass
class MotionGenCfg:
    planner_cfg: BasePlannerCfg = MISSING
    """Configuration for the underlying planner. Must include 'planner_type' attribute to specify 
    which planner to use, and any additional parameters required by that planner.
    """

    # TODO: More configuration options can be added here in the future.


@configclass
class MotionGenOptions:
    strategy: Literal["motion_gen", "ik_interp"] = "motion_gen"
    """Motion strategy: backend planning or deterministic IK interpolation."""

    sample_count: int | None = None
    """Requested output sample count; ``None`` preserves backend defaults."""

    velocity_limit: float | None = None
    """Optional scalar speed request, capped by effective Robot joint limits."""

    acceleration_limit: float | None = None
    """Optional scalar planner-owned joint acceleration budget."""

    start_qpos: torch.Tensor | None = None
    """Observed controlled-joint start, shape ``(B, DOF)``.

    Required for ``ik_interp``. Backend pre-interpolation reads the current
    robot state when omitted; native backend requests use backend defaults.
    """

    control_part: str | None = None
    """Name of the robot part to control, e.g. 'left_arm'. Must correspond to a valid control part defined in the robot's configuration."""

    plan_opts: PlanOptions | None = None
    """Options to pass to the underlying planner during the planning phase."""

    is_interpolate: bool = False
    """Whether to allow interpolation before planning when the backend needs it.

    Joint-only backends use this to convert Cartesian targets into joint
    waypoints. Backends that accept Cartesian targets directly receive the
    original targets unchanged.
    
    Note:
        - The pre-interpolation only works for PlanState with MoveType.EEF_MOVE or MoveType.JOINT_MOVE.
    """

    interpolation_dt: float | None = None
    """Explicit waypoint interval for deterministic interpolation."""

    interpolate_nums: int | list[int] = 10
    """Number of interpolation points to generate between each pair of waypoints. 
    
    Can be an integer (same for all segments) or a list of integers with len(PlanState) specifying the number of points for each segment."""

    is_linear: bool = False
    """Request Cartesian pre-interpolation for joint-only backends.

    With ``ik_interp``, requires explicit ``preserve_cartesian_samples``;
    sparse endpoint IK followed by joint interpolation is not a Cartesian line.
    """

    preserve_cartesian_samples: bool = False
    """Treat Cartesian targets as an exact sampled path in ``ik_interp`` only.

    This constrained mode requires exactly ``sample_count - 1`` target states;
    the observed start configuration supplies the first sample. Without motion
    limits, targets are emitted unchanged. With limits, their Cartesian path is
    resampled by a minimum-jerk progress law and every emitted pose is solved by
    IK.
    """

    cartesian_path_sampler: Callable[[torch.Tensor], torch.Tensor] | None = None
    """Optional exact-path sampler for normalized progress values.

    The callback receives a one-dimensional tensor spanning zero through one
    and returns poses with shape ``(B, T, 4, 4)``. Use it for paths whose
    geometry cannot be recovered from Cartesian chords, such as rotation about
    an external hinge. It is valid only with ``preserve_cartesian_samples``.
    """

    interpolate_position_step: float = 0.002
    """Position step in meters for Cartesian pre-interpolation."""

    interpolate_angle_step: float = np.pi / 90
    """Orientation step in radians for Cartesian pre-interpolation."""

    def __post_init__(self) -> None:
        """Validate backend-neutral motion generation options."""
        if self.preserve_cartesian_samples and self.strategy != "ik_interp":
            raise ValueError(
                "preserve_cartesian_samples requires strategy='ik_interp'; "
                "planner backends do not guarantee exact Cartesian samples."
            )
        if self.cartesian_path_sampler is not None:
            if not callable(self.cartesian_path_sampler):
                raise TypeError("cartesian_path_sampler must be callable or None.")
            if not self.preserve_cartesian_samples:
                raise ValueError(
                    "cartesian_path_sampler requires preserve_cartesian_samples=True."
                )
        if self.strategy == "ik_interp" and self.plan_opts is not None:
            raise ValueError(
                "strategy='ik_interp' does not consume plan_opts; use "
                "strategy='motion_gen' for backend-specific options."
            )
        if (
            self.strategy == "ik_interp"
            and self.is_linear
            and not self.preserve_cartesian_samples
        ):
            raise ValueError(
                "is_linear with ik_interp requires preserve_cartesian_samples=True "
                "and explicit Cartesian path samples."
            )
        valid_strategies = {"motion_gen", "ik_interp"}
        if self.strategy not in valid_strategies:
            raise ValueError(
                f"strategy must be one of {sorted(valid_strategies)}, "
                f"got {self.strategy!r}."
            )
        if self.sample_count is not None and self.sample_count < 2:
            raise ValueError("sample_count must be at least 2 when set.")
        if self.velocity_limit is not None and (
            not math.isfinite(self.velocity_limit) or self.velocity_limit <= 0.0
        ):
            raise ValueError("velocity_limit must be finite and greater than zero.")
        if self.acceleration_limit is not None and (
            not math.isfinite(self.acceleration_limit) or self.acceleration_limit <= 0.0
        ):
            raise ValueError("acceleration_limit must be finite and greater than zero.")
        if self.interpolation_dt is not None:
            if isinstance(self.interpolation_dt, bool) or not isinstance(
                self.interpolation_dt, (int, float)
            ):
                raise TypeError("interpolation_dt must be a real number or None.")
            if not math.isfinite(self.interpolation_dt) or self.interpolation_dt <= 0.0:
                raise ValueError(
                    "interpolation_dt must be finite and greater than zero when set."
                )


class MotionGenerator:
    r"""Unified motion generator for robot trajectory planning.

    This class provides a unified interface for trajectory planning with and without
    collision checking.

    Args:
        cfg: Configuration object for motion generation, must include 'planner_cfg' attribute
    """

    _support_planner_dict = {
        "toppra": (ToppraPlanner, ToppraPlannerCfg),
        "trapezoidal": (TrapezoidalPlanner, TrapezoidalPlannerCfg),
        "neural": (NeuralPlanner, NeuralPlannerCfg),
        "curobo": (CuroboPlanner, CuroboPlannerCfg),
    }

    def __init__(self, cfg: MotionGenCfg) -> None:

        # Create planner based on planner_type
        self.planner: BasePlanner = self._create_planner(cfg.planner_cfg)

        self.robot = self.planner.robot
        self.device = self.robot.device

    @property
    def collision_world_info(self) -> CollisionWorldInfo | None:
        """Return the selected planner's collision-world contract."""
        info = self.planner.collision_world_info
        if info is not None and not isinstance(info, CollisionWorldInfo):
            raise TypeError(
                "Planner.collision_world_info must be a CollisionWorldInfo or None."
            )
        return info

    @property
    def supports_dynamic_collision_world(self) -> bool:
        """Whether the planner accepts per-plan dynamic obstacle poses.

        Returns:
            ``True`` when the selected planner supports collision-world updates.
        """
        info = self.collision_world_info
        return info is not None and info.supports_updates

    @property
    def supports_joint_trajectory_validation(self) -> bool:
        """Whether the backend checks exact joint samples for collisions."""
        return (
            getattr(
                self.planner,
                "supports_joint_trajectory_validation",
                False,
            )
            is True
        )

    @property
    def supports_heterogeneous_waypoints(self) -> bool:
        """Whether the selected backend accepts mixed waypoint movement types."""
        return getattr(self.planner, "supports_heterogeneous_waypoints", False) is True

    @property
    def dynamic_collision_entity_ids(self) -> tuple[str, ...]:
        """Return canonical dynamic-obstacle IDs declared by the planner."""
        info = self.collision_world_info
        return () if info is None else info.dynamic_entity_ids

    @property
    def collision_world_entity_ids(self) -> tuple[str, ...]:
        """Return every canonical entity ID in the planner collision world."""
        info = self.collision_world_info
        return () if info is None else info.entity_ids

    @staticmethod
    def _validate_collision_pose_keys(
        poses: Mapping[object, object],
        *,
        field_name: str,
    ) -> set[str]:
        """Validate exact canonical IDs on one obstacle-pose mapping."""
        entity_ids = tuple(poses)
        if not all(
            isinstance(entity_id, str) and entity_id and entity_id == entity_id.strip()
            for entity_id in entity_ids
        ):
            raise TypeError(
                f"{field_name} keys must be non-empty strings without outer whitespace."
            )
        return set(entity_ids)

    @property
    def collision_world_batch_mode(self) -> Literal["shared", "per_env"] | None:
        """Return the backend's dynamic collision-world batch-sharing mode."""
        info = self.collision_world_info
        return None if info is None else info.batch_mode

    def bind_collision_world(
        self,
        plan_opts: PlanOptions | None,
        *,
        obstacle_poses: Mapping[str, torch.Tensor],
    ) -> PlanOptions:
        """Bind live obstacle poses to owned planner options.

        Args:
            plan_opts: Optional reusable caller-owned planner options.
            obstacle_poses: Batched world poses keyed by stable obstacle ID.

        Returns:
            Backend-specific options bound to the supplied collision world.

        Raises:
            ValueError: If the selected planner cannot consume dynamic obstacles.
        """
        info = self.collision_world_info
        if info is None or not info.supports_updates:
            logger.log_error(
                f"{type(self.planner).__name__} does not support dynamic "
                "collision-world updates.",
                ValueError,
            )
        assert info is not None
        configured_ids = info.dynamic_entity_ids
        received_ids = tuple(obstacle_poses)
        if not all(
            isinstance(entity_id, str) and entity_id and entity_id == entity_id.strip()
            for entity_id in received_ids
        ):
            raise TypeError(
                "obstacle_poses keys must be non-empty strings without outer "
                "whitespace."
            )
        missing = sorted(set(configured_ids).difference(received_ids))
        extra = sorted(set(received_ids).difference(configured_ids))
        if missing or extra:
            logger.log_error(
                "Dynamic collision obstacle IDs do not match the planner "
                f"configuration; missing={missing}, extra={extra}.",
                ValueError,
            )
        options = (
            deepcopy(plan_opts)
            if plan_opts is not None
            else self.planner.default_plan_options()
        )
        existing_poses = getattr(options, "dynamic_obstacle_poses", None)
        if existing_poses is not None:
            if not isinstance(existing_poses, Mapping):
                raise TypeError(
                    "plan_opts.dynamic_obstacle_poses must be a mapping or None."
                )
            existing_ids = self._validate_collision_pose_keys(
                existing_poses,
                field_name="plan_opts.dynamic_obstacle_poses",
            )
            existing_extra = sorted(existing_ids.difference(configured_ids))
            if existing_extra:
                raise ValueError(
                    "Caller planning options contain dynamic collision IDs that "
                    f"are not configured by the planner: {existing_extra}."
                )
        bound = self.planner.with_collision_world(
            options,
            obstacle_poses=obstacle_poses,
        )
        if hasattr(bound, "dynamic_obstacle_poses"):
            bound_poses = bound.dynamic_obstacle_poses
            if bound_poses is None:
                bound_ids: set[str] = set()
            elif not isinstance(bound_poses, Mapping):
                raise TypeError("Bound dynamic_obstacle_poses must be a mapping.")
            else:
                bound_ids = self._validate_collision_pose_keys(
                    bound_poses,
                    field_name="Bound dynamic_obstacle_poses",
                )
            bound_missing = sorted(set(configured_ids).difference(bound_ids))
            bound_extra = sorted(bound_ids.difference(configured_ids))
            if bound_missing or bound_extra:
                raise ValueError(
                    "Bound dynamic collision obstacle IDs do not match the planner "
                    f"configuration; missing={bound_missing}, extra={bound_extra}."
                )
        return bound

    def validate_joint_trajectory(
        self,
        trajectory: torch.Tensor,
        *,
        control_part: str,
        obstacle_poses: Mapping[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Check exact joint samples through the selected planner backend.

        Args:
            trajectory: Simulator-order joint samples with shape ``(B, T, D)``.
            control_part: Robot control part whose ordered joints form ``D``.
            obstacle_poses: Optional live dynamic-obstacle poses.

        Returns:
            Boolean validity mask with shape ``(B, T)`` on the trajectory device.
        """
        if not self.supports_joint_trajectory_validation:
            raise ValueError(
                f"Planner {type(self.planner).__name__} does not support exact "
                "joint-trajectory collision validation."
            )
        if not isinstance(trajectory, torch.Tensor):
            raise TypeError("trajectory must be a torch.Tensor.")
        if (
            not trajectory.is_floating_point()
            or trajectory.dim() != 3
            or 0 in trajectory.shape
            or not bool(torch.isfinite(trajectory).all().item())
        ):
            raise ValueError(
                "trajectory must be finite floating shape (B, T, D) with "
                "non-zero dimensions."
            )
        if type(control_part) is not str or not control_part:
            raise ValueError("control_part must be a non-empty string.")
        validity = self.planner.validate_joint_trajectory(
            trajectory,
            control_part=control_part,
            obstacle_poses=obstacle_poses,
        )
        if not isinstance(validity, torch.Tensor):
            raise TypeError("Planner.validate_joint_trajectory() must return a tensor.")
        if validity.dtype != torch.bool or validity.shape != trajectory.shape[:2]:
            raise ValueError(
                "Planner.validate_joint_trajectory() must return bool shape "
                f"{tuple(trajectory.shape[:2])}."
            )
        return validity.to(trajectory.device).clone()

    def resolve_plan_options(
        self,
        plan_opts: PlanOptions | None,
        *,
        sample_count: int | None,
        velocity_limit: float | None = None,
        acceleration_limit: float | None = None,
    ) -> PlanOptions:
        """Resolve owned backend options from backend-neutral motion limits.

        Args:
            plan_opts: Optional caller-owned backend-specific options.
            sample_count: Requested output sample count, or ``None`` to use the
                backend default.
            velocity_limit: Optional scalar joint velocity limit.
            acceleration_limit: Optional scalar joint acceleration limit.

        Returns:
            An independently owned options object for the configured backend.

        Raises:
            ValueError: If a supplied ``sample_count`` is smaller than two.
        """
        if sample_count is not None and sample_count < 2:
            raise ValueError("sample_count must be at least 2.")
        if plan_opts is not None:
            resolved = deepcopy(plan_opts)
            if isinstance(resolved, CuroboPlanOptions):
                if velocity_limit is not None:
                    resolved.max_velocity = float(velocity_limit)
                if acceleration_limit is not None:
                    resolved.max_acceleration = float(acceleration_limit)
            return resolved
        planner_type = getattr(getattr(self.planner, "cfg", None), "planner_type", None)
        if planner_type == "curobo":
            return CuroboPlanOptions(
                max_velocity=(
                    None if velocity_limit is None else float(velocity_limit)
                ),
                max_acceleration=(
                    None if acceleration_limit is None else float(acceleration_limit)
                ),
            )
        if planner_type in {"toppra", "trapezoidal"} and (
            sample_count is not None
            or velocity_limit is not None
            or acceleration_limit is not None
        ):
            options_type = (
                ToppraPlanOptions
                if planner_type == "toppra"
                else TrapezoidalPlanOptions
            )
            constraints: dict[str, float | None] = {
                "velocity": velocity_limit,
                "acceleration": acceleration_limit,
            }
            if planner_type == "trapezoidal":
                constraints["jerk"] = 2.0
            options_kwargs: dict[str, object] = {"constraints": constraints}
            if sample_count is not None:
                options_kwargs.update(
                    sample_method=TrajectorySampleMethod.QUANTITY,
                    sample_interval=sample_count,
                )
            return options_type(**options_kwargs)
        return self.planner.default_plan_options()

    def _resolve_velocity_limits(
        self,
        *,
        control_part: str | None,
        batch_size: int,
        dof: int,
        requested: float | None,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        """Intersect one scalar request with effective ordered Robot speed caps.

        Shared planner resolution fills missing runtime caps from the assembled
        asset and projects mimic follower caps before selecting control joints.
        Requests can only tighten known caps; joints still lacking metadata use
        the request or the default interpolation budget.
        """
        joint_names = getattr(self.robot, "joint_names", None)
        use_full_limits = isinstance(joint_names, (list, tuple)) and bool(joint_names)
        if use_full_limits:
            joint_ids = _plan_joint_ids(self.robot, control_part, dof)
            limits = _joint_velocity_limits(
                self.robot, device=reference.device, dtype=reference.dtype
            )
            limits = limits[:, joint_ids]
        else:
            limits = self.robot.get_qvel_limits(name=control_part).detach().clone()
            if limits.ndim != 2 or limits.shape[1] != dof:
                raise ValueError(
                    "Robot velocity limits must have shape (B, interpolation DOF)."
                )
            limits = torch.where(
                torch.isfinite(limits) & (limits >= 0.0) & (limits < 1.0e9),
                limits,
                torch.inf,
            )
        if limits.shape[0] < batch_size:
            raise ValueError(
                "Robot velocity limits do not cover the interpolation batch."
            )
        limits = limits[:batch_size].to(
            device=reference.device,
            dtype=reference.dtype,
        )
        caps = limits.amin(dim=0)
        if requested is None:
            return torch.where(torch.isfinite(caps), caps, 0.2)
        requested_limits = torch.full_like(caps, float(requested))
        return torch.where(
            torch.isfinite(caps),
            torch.minimum(caps, requested_limits),
            requested_limits,
        )

    @classmethod
    def register_planner_type(cls, name: str, planner_class, planner_cfg_class) -> None:
        """
        Register a new planner type.
        """
        cls._support_planner_dict[name] = (planner_class, planner_cfg_class)

    def _create_planner(
        self,
        planner_cfg: BasePlannerCfg,
    ) -> BasePlanner:
        r"""Create planner instance based on planner type.

        Args:
            planner_cfg: Configuration object for the planner, must include 'planner_type' attribute

        Returns:
            Planner instance
        """
        planner_type = planner_cfg.planner_type
        if planner_type not in self._support_planner_dict.keys():
            logger.log_error(
                f"Unsupported planner type: {planner_type}. "
                f"Supported types: {list(self._support_planner_dict.keys())}"
            )
        cls = self._support_planner_dict[planner_type][0](cfg=planner_cfg)
        return cls

    def generate(
        self,
        target_states: list[PlanState],
        options: MotionGenOptions | None = None,
    ) -> PlanResult:
        r"""Generate one normalized, environment-batched motion plan.

        ``options.strategy`` selects either the configured planner backend
        (``"motion_gen"``) or deterministic waypoint IK followed by joint-space
        interpolation (``"ik_interp"``). Unsupported backend target types raise
        rather than silently switching to interpolation.
        Joint-only planners that own sparse waypoints receive IK solutions only
        at the requested Cartesian goals unless ``options.is_linear`` requests
        a dense Cartesian path. Quantity sampling retains every required joint
        waypoint. Automatically resolved trapezoidal options treat the requested
        count as a lower bound; explicit planner options remain authoritative.

        Args:
            target_states: Batched planner waypoints.
            options: Motion-generation strategy and runtime options.

        Returns:
            Normalized result with a per-environment success mask and joint
            positions. Failed rows hold ``start_qpos`` when it is supplied.

        Raises:
            ValueError: If targets or options violate the motion contract.
        """
        if not target_states:
            raise ValueError("target_states must contain at least one waypoint.")
        options = MotionGenOptions() if options is None else deepcopy(options)
        # Options are mutable config objects; revalidate at the call boundary.
        options.__post_init__()

        move_types = {state.move_type for state in target_states}
        heterogeneous = len(move_types) > 1
        if heterogeneous and not self.supports_heterogeneous_waypoints:
            names = sorted(move_type.name for move_type in move_types)
            raise ValueError(
                f"{type(self.planner).__name__} does not support heterogeneous "
                f"waypoints; got {names}."
            )
        if heterogeneous and options.strategy == "ik_interp":
            raise ValueError(
                "strategy='ik_interp' does not support heterogeneous waypoints."
            )
        if options.strategy == "ik_interp":
            raw_result = self._generate_ik_interpolation(target_states, options)
        else:
            raw_result = self._generate_with_planner(target_states, options)
        return self._normalize_plan_result(raw_result, target_states, options)

    def _generate_with_planner(
        self,
        target_states: list[PlanState],
        options: MotionGenOptions,
    ) -> PlanResult:
        """Dispatch batched targets through the configured planner backend."""
        move_types = {state.move_type for state in target_states}
        move_type = target_states[0].move_type
        owns_sparse_waypoints = (
            getattr(self.planner, "uses_sparse_joint_waypoints", False) is True
        )
        uses_sparse_joint_waypoints = (
            len(move_types) == 1
            and move_type is MoveType.JOINT_MOVE
            and owns_sparse_waypoints
        )
        should_preinterpolate = (
            len(move_types) == 1
            and options.is_interpolate
            and not uses_sparse_joint_waypoints
            and not self.planner.supports_move_type(MoveType.EEF_MOVE)
            and self.planner.supports_move_type(MoveType.JOINT_MOVE)
        )

        preparation_success = None
        if should_preinterpolate:
            if move_type == MoveType.EEF_MOVE:
                if any(state.xpos is None for state in target_states):
                    raise ValueError("EEF_MOVE target states require xpos tensors.")
                xpos_list = torch.stack([s.xpos for s in target_states]).transpose(
                    0, 1
                )  # (B, N, 4, 4)
                qpos_list = None
            elif move_type == MoveType.JOINT_MOVE:
                if any(state.qpos is None for state in target_states):
                    raise ValueError("JOINT_MOVE target states require qpos tensors.")
                qpos_list = torch.stack([s.qpos for s in target_states]).transpose(
                    0, 1
                )  # (B, N, DOF)
                xpos_list = None
            else:
                logger.log_error(
                    f"Unsupported move type for pre-interpolation: {move_type}"
                )

            if options.start_qpos is None:
                options.start_qpos = self.robot.get_qpos(name=options.control_part)
            if options.start_qpos is not None:
                start = options.start_qpos
                if start.dim() == 1:
                    start = start.unsqueeze(0)
                if qpos_list is not None:
                    qpos_list = torch.cat([start.unsqueeze(1), qpos_list], dim=1)
                if xpos_list is not None:
                    start_xpos = self.robot.compute_fk(
                        qpos=start, name=options.control_part, to_matrix=True
                    )
                    if start_xpos.dim() == 3:
                        start_xpos = start_xpos.unsqueeze(1)
                    xpos_list = torch.cat([start_xpos, xpos_list], dim=1)

            qpos_interpolated, _, preparation_success = self._interpolate_waypoints(
                control_part=options.control_part,
                xpos_list=xpos_list,
                qpos_list=qpos_list,
                options=options,
                sparse_cartesian=owns_sparse_waypoints and not options.is_linear,
            )
            target_plan_states = [
                PlanState(move_type=MoveType.JOINT_MOVE, qpos=qpos_interpolated[:, j])
                for j in range(qpos_interpolated.shape[1])
            ]
        else:
            target_plan_states = target_states
            if uses_sparse_joint_waypoints and options.start_qpos is not None:
                if any(state.qpos is None for state in target_states):
                    raise ValueError("JOINT_MOVE target states require qpos tensors.")
                reference_qpos = target_states[0].qpos
                assert reference_qpos is not None
                start_qpos = options.start_qpos
                if start_qpos.dim() == 1:
                    start_qpos = start_qpos.unsqueeze(0)
                start_qpos = start_qpos.to(reference_qpos)
                if start_qpos.shape != reference_qpos.shape:
                    raise ValueError(
                        "start_qpos and joint target shapes must match; received "
                        f"{tuple(start_qpos.shape)} and {tuple(reference_qpos.shape)}."
                    )
                # A single target still needs the observed start as an
                # explicit waypoint, including when it is stationary. Repeated
                # waypoints are valid and represent a zero-duration hold.
                if len(target_states) == 1 or not torch.equal(
                    start_qpos, reference_qpos
                ):
                    target_plan_states = [
                        PlanState(move_type=MoveType.JOINT_MOVE, qpos=start_qpos),
                        *target_states,
                    ]

        unsupported_move_types = {
            candidate
            for candidate in move_types
            if not self.planner.supports_move_type(candidate)
        }
        if not should_preinterpolate and unsupported_move_types:
            unsupported_names = sorted(
                move_type.name for move_type in unsupported_move_types
            )
            supported_names = sorted(
                move_type.name for move_type in self.planner.supported_move_types
            )
            logger.log_error(
                f"{type(self.planner).__name__} does not support move types "
                f"{unsupported_names}; supported types are {supported_names}.",
                ValueError,
            )

        explicit_plan_options = options.plan_opts is not None
        plan_opts = self.resolve_plan_options(
            options.plan_opts,
            sample_count=options.sample_count,
            velocity_limit=options.velocity_limit,
            acceleration_limit=options.acceleration_limit,
        )
        if (
            should_preinterpolate
            and not explicit_plan_options
            and isinstance(plan_opts, TrapezoidalPlanOptions)
            and plan_opts.sample_method is TrajectorySampleMethod.QUANTITY
            and int(plan_opts.sample_interval) < len(target_plan_states)
        ):
            # Trapezoidal quantity sampling retains every supplied waypoint.
            # Cartesian targets may have been converted to a denser IK path.
            # For backend-neutral defaults, make the requested count a safe
            # lower bound instead of rejecting an otherwise valid path. An
            # explicit TrapezoidalPlanOptions remains caller-authoritative.
            plan_opts.sample_interval = len(target_plan_states)
        planner_context_method = getattr(
            type(self.planner), "with_motion_context", None
        )
        legacy_backend_limits = planner_context_method in {
            None,
            BasePlanner.with_motion_context,
        }
        if (
            not explicit_plan_options
            and legacy_backend_limits
            and isinstance(plan_opts, (ToppraPlanOptions, TrapezoidalPlanOptions))
        ):
            reference_qpos = target_plan_states[0].qpos
            if reference_qpos is not None:
                if reference_qpos.dim() == 1:
                    reference_qpos = reference_qpos.unsqueeze(0)
                plan_opts.constraints["velocity"] = self._resolve_velocity_limits(
                    control_part=options.control_part,
                    batch_size=reference_qpos.shape[0],
                    dof=reference_qpos.shape[1],
                    requested=options.velocity_limit,
                    reference=reference_qpos,
                )
        plan_opts = self.planner.with_motion_context(
            plan_opts,
            start_qpos=options.start_qpos,
            control_part=options.control_part,
        )
        options.plan_opts = plan_opts
        result = self.planner.plan(
            target_states=target_plan_states,
            options=plan_opts,
        )
        if preparation_success is not None:
            result.success = (
                normalize_success_mask(
                    result.success,
                    num_envs=preparation_success.shape[0],
                    device=self._runtime_device(),
                    name="Planner success",
                )
                & preparation_success
            )
        return result

    def _solve_cartesian_waypoints(
        self,
        poses: torch.Tensor,
        start_qpos: torch.Tensor,
        control_part: str,
        *,
        known_start: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Batch required poses, retaining FK-confirmed starts and IK failures.

        Solvers with continuous candidate selection receive an initial path
        seed. Other solvers first solve each chunk's endpoint from the preceding
        chunk, then solve its interior together using interpolated joint seeds.
        These seeds encourage a consistent branch without claiming continuous
        selection for solvers that do not support it.
        """
        device = self._runtime_device()
        start_qpos = start_qpos.to(device)
        batch_size, controlled_dof = start_qpos.shape
        if poses.ndim != 4 or poses.shape[0] != batch_size or poses.shape[2:] != (4, 4):
            raise ValueError("Cartesian waypoints must have shape (B, N, 4, 4).")
        if poses.shape[1] == 0:
            raise ValueError("Cartesian waypoints must contain at least one pose.")
        poses = poses.to(device)
        solver = self.robot.get_solver(control_part)
        continuous = solver is not None and solver.supports_continuous_batch_ik
        chunk_size = max(
            1, min(_MAX_IK_BATCH_WAYPOINTS, _MAX_IK_BATCH_TARGETS // batch_size)
        )
        success = torch.ones(batch_size, dtype=torch.bool, device=device)
        qpos_seed = start_qpos
        solved_waypoints: list[torch.Tensor] = []
        first_index = 0
        if known_start is not None:
            known_start = known_start.to(device)
            if bool(known_start.all()):
                # FK already grounded this pose, including at singularities
                # where IK may fail. Keep it outside the solver batches.
                solved_waypoints.append(start_qpos[:, None, :])
                first_index = 1
        for offset in range(first_index, poses.shape[1], chunk_size):
            chunk = poses[:, offset : offset + chunk_size]
            count = chunk.shape[1]
            if continuous:
                valid, qpos = self._compute_cartesian_ik_batch(
                    chunk, qpos_seed, control_part, continuous=True
                )
            else:
                end_valid, end_qpos = self._compute_cartesian_ik_batch(
                    chunk[:, -1:], qpos_seed[:, None, :], control_part
                )
                if count == 1:
                    valid, qpos = end_valid, end_qpos
                else:
                    safe_end = torch.where(end_valid, end_qpos[:, 0], qpos_seed)
                    weights = (
                        torch.arange(1, count, dtype=start_qpos.dtype, device=device)
                        / count
                    )
                    seeds = torch.lerp(
                        qpos_seed[:, None, :],
                        safe_end[:, None, :],
                        weights[None, :, None],
                    )
                    inner_valid, inner_qpos = self._compute_cartesian_ik_batch(
                        chunk[:, :-1], seeds, control_part
                    )
                    valid = torch.cat((inner_valid, end_valid), dim=1)
                    qpos = torch.cat((inner_qpos, end_qpos), dim=1)

            if offset == 0 and known_start is not None:
                # Mixed rows still solve the unconfirmed first poses. Only
                # confirmed rows may disregard that first sample's IK failure.
                qpos = qpos.clone()
                valid = valid.clone()
                qpos[:, 0] = torch.where(known_start[:, None], start_qpos, qpos[:, 0])
                valid[:, 0] |= known_start

            # Invalid/nonfinite solutions hold the preceding valid seed, even
            # across chunks. A later reachable pose never rescues a failed row.
            indices = torch.arange(1, count + 1, device=device).expand(batch_size, -1)
            previous_valid = torch.where(valid, indices, 0).cummax(dim=1).values
            candidates = torch.cat((qpos_seed[:, None, :], qpos), dim=1)
            qpos = candidates.gather(
                1, previous_valid[:, :, None].expand(-1, -1, controlled_dof)
            )
            success &= valid.all(dim=1)
            solved_waypoints.append(qpos)
            qpos_seed = qpos[:, -1]
        positions = torch.cat(solved_waypoints, dim=1)
        positions = torch.where(
            success[:, None, None], positions, start_qpos[:, None, :]
        )
        return success, positions

    def _compute_cartesian_ik_batch(
        self,
        poses: torch.Tensor,
        joint_seed: torch.Tensor,
        control_part: str,
        *,
        continuous: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Validate per-waypoint batch results and reject nonfinite solutions."""
        valid, qpos = self.robot.compute_batch_ik(
            pose=poses,
            joint_seed=joint_seed,
            name=control_part,
            **({"continuous": True} if continuous else {}),
        )
        batch_size, count = poses.shape[:2]
        if isinstance(valid, torch.Tensor) and valid.shape != (batch_size, count):
            raise ValueError(
                f"IK success must have shape ({batch_size}, {count}), "
                f"got {tuple(valid.shape)}."
            )
        valid = normalize_success_mask(
            valid.flatten() if isinstance(valid, torch.Tensor) else valid,
            num_envs=batch_size * count,
            device=joint_seed.device,
            name="Batch IK success",
        ).reshape(batch_size, count)
        qpos = torch.as_tensor(qpos, dtype=joint_seed.dtype, device=joint_seed.device)
        expected_shape = (batch_size, count, joint_seed.shape[-1])
        if qpos.shape != expected_shape:
            raise ValueError(
                f"IK qpos must have shape {expected_shape}, got {tuple(qpos.shape)}."
            )
        return valid & torch.isfinite(qpos).all(dim=2), qpos

    def _generate_ik_interpolation(
        self,
        target_states: list[PlanState],
        options: MotionGenOptions,
    ) -> PlanResult:
        """Generate deterministic joint interpolation from batched targets."""
        if options.start_qpos is None:
            raise ValueError("IK interpolation requires start_qpos.")
        if options.sample_count is None:
            raise ValueError("IK interpolation requires sample_count.")
        if options.interpolation_dt is None:
            raise ValueError("IK interpolation requires explicit interpolation_dt.")
        start_qpos = options.start_qpos
        if start_qpos.dim() == 1:
            start_qpos = start_qpos.unsqueeze(0)
        device = self._runtime_device()
        start_qpos = start_qpos.to(device)
        batch_size, controlled_dof = start_qpos.shape
        move_type = target_states[0].move_type

        if move_type is MoveType.JOINT_MOVE:
            qpos_targets: list[torch.Tensor] = []
            for state in target_states:
                if state.qpos is None:
                    raise ValueError("JOINT_MOVE target states require qpos tensors.")
                qpos = state.qpos
                if qpos.dim() == 1:
                    qpos = qpos.unsqueeze(0)
                expected_shape = (batch_size, controlled_dof)
                if qpos.shape != expected_shape:
                    raise ValueError(
                        "JOINT_MOVE target qpos must have shape "
                        f"{expected_shape}, got {tuple(qpos.shape)}."
                    )
                qpos_targets.append(qpos.to(device=device, dtype=start_qpos.dtype))
            keyframes = torch.cat(
                [start_qpos.unsqueeze(1), torch.stack(qpos_targets, dim=1)],
                dim=1,
            )
            positions = self._interpolate_ik_keyframes(keyframes, options)
            dt = self._uniform_dt(
                batch_size=batch_size,
                waypoint_count=positions.shape[1],
                step_dt=options.interpolation_dt,
                device=device,
            )
            return PlanResult(
                success=torch.ones(batch_size, dtype=torch.bool, device=device),
                positions=positions,
                dt=dt,
            )

        if move_type is not MoveType.EEF_MOVE:
            raise ValueError(
                "strategy='ik_interp' supports only EEF_MOVE and JOINT_MOVE, "
                f"got {move_type.name}."
            )
        if options.control_part is None:
            raise ValueError("EEF_MOVE IK interpolation requires control_part.")

        poses = []
        for state in target_states:
            if state.xpos is None:
                raise ValueError("EEF_MOVE target states require xpos tensors.")
            pose = state.xpos
            if pose.dim() == 2:
                pose = pose.unsqueeze(0)
            poses.append(pose)
        success, solved_waypoints = self._solve_cartesian_waypoints(
            torch.stack(poses, dim=1), start_qpos, options.control_part
        )
        keyframes = torch.cat([start_qpos.unsqueeze(1), solved_waypoints], dim=1)
        if options.preserve_cartesian_samples:
            if keyframes.shape[1] != options.sample_count:
                raise ValueError(
                    "Linear Cartesian targets must provide sample_count - 1 "
                    "keyframes so every output sample is IK-grounded; got "
                    f"{len(target_states)} targets for sample_count "
                    f"{options.sample_count}."
                )
            preserve_fixed_grid = False
            if options.velocity_limit is None and options.acceleration_limit is None:
                velocity_limits = self._resolve_velocity_limits(
                    control_part=options.control_part,
                    batch_size=batch_size,
                    dof=controlled_dof,
                    requested=None,
                    reference=keyframes,
                )
                preserve_fixed_grid = bool(
                    (
                        keyframes.diff(dim=1).abs() / options.interpolation_dt
                        <= velocity_limits.view(1, 1, -1) + 1.0e-6
                    )
                    .all()
                    .item()
                )
            if preserve_fixed_grid:
                positions = keyframes
            else:
                limited_success, positions = self._interpolate_cartesian_with_limits(
                    torch.stack(poses, dim=1),
                    keyframes,
                    options,
                )
                success &= limited_success
        else:
            positions = self._interpolate_ik_keyframes(keyframes, options)
        dt = self._uniform_dt(
            batch_size=batch_size,
            waypoint_count=positions.shape[1],
            step_dt=options.interpolation_dt,
            device=device,
        )
        return PlanResult(
            success=success,
            positions=positions,
            dt=dt,
        )

    def _interpolate_cartesian_with_limits(
        self,
        target_poses: torch.Tensor,
        solved_keyframes: torch.Tensor,
        options: MotionGenOptions,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Retain a sampled Cartesian path while enforcing joint motion limits."""
        assert options.control_part is not None
        assert options.sample_count is not None
        assert options.interpolation_dt is not None
        velocity_limits = self._resolve_velocity_limits(
            control_part=options.control_part,
            batch_size=solved_keyframes.shape[0],
            dof=solved_keyframes.shape[2],
            requested=options.velocity_limit,
            reference=solved_keyframes,
        )
        acceleration_limit = (
            0.5 if options.acceleration_limit is None else options.acceleration_limit
        )
        acceleration_limits = torch.full_like(velocity_limits, acceleration_limit)
        variation = solved_keyframes.diff(dim=1).abs().sum(dim=1)
        zero_velocity_motion = (velocity_limits == 0.0) & (variation > 0.0).any(dim=0)
        if zero_velocity_motion.any():
            raise ValueError("A zero-velocity Robot joint cannot follow this path.")
        safe_velocity_limits = torch.where(
            velocity_limits == 0.0,
            torch.ones_like(velocity_limits),
            velocity_limits,
        )
        duration = torch.maximum(
            (1.875 * variation / safe_velocity_limits).amax(dim=1),
            torch.sqrt((10.0 / math.sqrt(3.0)) * variation / acceleration_limits).amax(
                dim=1
            ),
        )
        interval_count = max(
            options.sample_count - 1,
            int(torch.ceil(duration.max() / options.interpolation_dt).item()),
        )
        success = torch.ones(
            solved_keyframes.shape[0],
            dtype=torch.bool,
            device=solved_keyframes.device,
        )
        positions = solved_keyframes
        for _ in range(6):
            if interval_count > 10_000:
                # This is bounded planning failure, not malformed caller input.
                # Retain a safe start hold so Atomic consumers can propagate the
                # failed success mask without executing an unchecked IK path.
                logger.log_warning(
                    "Cartesian timing exhausted its 10000-control-interval budget."
                )
                return torch.zeros_like(success), solved_keyframes[:, :1].repeat(
                    1, 2, 1
                )
            sampled_poses = self._sample_cartesian_path(
                target_poses,
                solved_keyframes,
                interval_count + 1,
                options.control_part,
                path_sampler=options.cartesian_path_sampler,
            )
            iteration_success, solved = self._solve_cartesian_waypoints(
                sampled_poses[:, 1:],
                solved_keyframes[:, 0],
                options.control_part,
            )
            success &= iteration_success
            positions = torch.cat([solved_keyframes[:, :1], solved], dim=1)
            interval_velocity = positions.diff(dim=1) / options.interpolation_dt
            boundary_acceleration = (
                torch.cat(
                    [
                        interval_velocity[:, :1],
                        interval_velocity.diff(dim=1),
                        -interval_velocity[:, -1:],
                    ],
                    dim=1,
                )
                / options.interpolation_dt
            )
            velocity_ratio = float(
                (interval_velocity.abs() / safe_velocity_limits.view(1, 1, -1))
                .amax()
                .item()
            )
            acceleration_ratio = float(
                (boundary_acceleration.abs() / acceleration_limits.view(1, 1, -1))
                .amax()
                .item()
            )
            scale = max(velocity_ratio, math.sqrt(acceleration_ratio))
            if scale <= 1.0 + 1.0e-5:
                return success, positions
            interval_count = math.ceil(interval_count * scale * 1.02)

        interval_velocity = positions.diff(dim=1) / options.interpolation_dt
        boundary_acceleration = (
            torch.cat(
                [
                    interval_velocity[:, :1],
                    interval_velocity.diff(dim=1),
                    -interval_velocity[:, -1:],
                ],
                dim=1,
            )
            / options.interpolation_dt
        )
        success &= (
            interval_velocity.abs() <= velocity_limits.view(1, 1, -1) + 1.0e-6
        ).all(dim=(1, 2))
        success &= (
            boundary_acceleration.abs() <= acceleration_limits.view(1, 1, -1) + 1.0e-5
        ).all(dim=(1, 2))
        return success, positions

    def _sample_cartesian_path(
        self,
        target_poses: torch.Tensor,
        solved_keyframes: torch.Tensor,
        sample_count: int,
        control_part: str,
        *,
        path_sampler: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Sample an exact path or Cartesian polyline with minimum-jerk progress."""
        if self.robot is None:
            raise RuntimeError("MotionGenerator requires a robot before interpolation.")
        start_pose = self.robot.compute_fk(
            qpos=solved_keyframes[:, 0],
            name=control_part,
            to_matrix=True,
        ).to(device=target_poses.device, dtype=target_poses.dtype)
        path = torch.cat([start_pose[:, None], target_poses], dim=1)
        segment_lengths = solved_keyframes.diff(dim=1).abs().amax(dim=-1)
        cumulative = torch.cumsum(segment_lengths, dim=1)
        total = cumulative[:, -1:]
        progress = _minimum_jerk_progress(
            sample_count,
            device=target_poses.device,
            dtype=target_poses.dtype,
        )
        if path_sampler is not None:
            result = path_sampler(progress)
            expected_shape = (target_poses.shape[0], sample_count, 4, 4)
            if not isinstance(result, torch.Tensor) or result.shape != expected_shape:
                raise ValueError(
                    "cartesian_path_sampler must return a tensor with shape "
                    f"{expected_shape}."
                )
            if result.device != target_poses.device:
                raise ValueError(
                    "cartesian_path_sampler output must share the target device."
                )
            if result.dtype != target_poses.dtype:
                raise ValueError(
                    "cartesian_path_sampler output must share the target dtype."
                )
            if not torch.isfinite(result).all():
                raise ValueError("cartesian_path_sampler output must be finite.")
            position_tolerance, rotation_tolerance = self._cartesian_start_tolerances(
                control_part
            )
            start_position_error = torch.linalg.vector_norm(
                result[:, 0, :3, 3] - path[:, 0, :3, 3],
                dim=-1,
            )
            start_rotation_error = quat_error_magnitude(
                quat_from_matrix(result[:, 0, :3, :3]),
                quat_from_matrix(path[:, 0, :3, :3]),
            )
            start_matches = bool(
                (start_position_error <= position_tolerance).all()
                and (start_rotation_error <= rotation_tolerance).all()
                and torch.allclose(
                    result[:, 0, 3],
                    path[:, 0, 3],
                    rtol=0.0,
                    atol=1.0e-6,
                )
            )
            if not start_matches or not torch.allclose(
                result[:, -1], path[:, -1], rtol=1.0e-5, atol=1.0e-6
            ):
                raise ValueError(
                    "cartesian_path_sampler must preserve the requested path endpoints."
                )
            return result
        query = total * progress[None]
        segment_index = torch.searchsorted(
            cumulative.contiguous(), query.contiguous(), right=False
        ).clamp(max=segment_lengths.shape[1] - 1)
        segment_start_distance = torch.cat(
            [torch.zeros_like(cumulative[:, :1]), cumulative[:, :-1]], dim=1
        ).gather(1, segment_index)
        selected_length = segment_lengths.gather(1, segment_index)
        alpha = torch.where(
            selected_length > 0,
            (query - segment_start_distance) / selected_length.clamp_min(1.0e-12),
            torch.zeros_like(query),
        ).clamp(0.0, 1.0)
        gather_index = segment_index[:, :, None, None].expand(-1, -1, 4, 4)
        start = torch.gather(path[:, :-1], 1, gather_index)
        end = torch.gather(path[:, 1:], 1, gather_index)
        result = (
            torch.eye(4, dtype=target_poses.dtype, device=target_poses.device)
            .expand(target_poses.shape[0], sample_count, -1, -1)
            .clone()
        )
        result[:, :, :3, 3] = torch.lerp(
            start[:, :, :3, 3], end[:, :, :3, 3], alpha[:, :, None]
        )
        relative_rotation = torch.matmul(
            start[:, :, :3, :3].transpose(-1, -2), end[:, :, :3, :3]
        )
        axis_angle = axis_angle_from_quat(quat_from_matrix(relative_rotation))
        incremental_rotation = axis_angle_to_rotation_matrix(
            axis_angle * alpha[:, :, None]
        )
        result[:, :, :3, :3] = torch.matmul(start[:, :, :3, :3], incremental_rotation)
        result[:, 0] = path[:, 0]
        result[:, -1] = path[:, -1]
        return result

    def _cartesian_start_tolerances(
        self,
        control_part: str,
    ) -> tuple[float, float]:
        """Return the active IK solver's accepted position and rotation errors."""
        solver = self.robot.get_solver(control_part)
        cfg = None if solver is None else getattr(solver, "cfg", None)

        def configured_tolerance(*names: str) -> float:
            for name in names:
                value = getattr(cfg, name, None)
                if (
                    not isinstance(value, bool)
                    and isinstance(value, (int, float))
                    and math.isfinite(value)
                    and value > 0.0
                ):
                    return max(1.0e-6, float(value))
            return 1.0e-6

        return configured_tolerance(
            "pos_eps", "position_tolerance"
        ), configured_tolerance("rot_eps", "rotation_tolerance")

    def _interpolate_ik_keyframes(
        self,
        keyframes: torch.Tensor,
        options: MotionGenOptions,
    ) -> torch.Tensor:
        """Interpolate IK keyframes under Robot-capped per-joint speeds."""
        assert options.sample_count is not None
        assert options.interpolation_dt is not None
        velocity_limits = self._resolve_velocity_limits(
            control_part=options.control_part,
            batch_size=keyframes.shape[0],
            dof=keyframes.shape[2],
            requested=options.velocity_limit,
            reference=keyframes,
        )
        if options.velocity_limit is None and options.acceleration_limit is None:
            fixed_grid = interpolate_with_distance(
                trajectory=keyframes,
                interp_num=options.sample_count,
                device=keyframes.device,
            )
            if bool(
                (
                    fixed_grid.diff(dim=1).abs() / options.interpolation_dt
                    <= velocity_limits.view(1, 1, -1) + 1.0e-6
                )
                .all()
                .item()
            ):
                return fixed_grid
        return interpolate_minimum_jerk(
            keyframes,
            minimum_sample_count=options.sample_count,
            step_dt=options.interpolation_dt,
            velocity_limit=velocity_limits,
            acceleration_limit=(
                0.5
                if options.acceleration_limit is None
                else options.acceleration_limit
            ),
        )

    @staticmethod
    def _uniform_dt(
        *,
        batch_size: int,
        waypoint_count: int,
        step_dt: float,
        device: torch.device,
    ) -> torch.Tensor:
        """Return explicit uniform arrival intervals for interpolation."""
        dt = torch.zeros(
            (batch_size, waypoint_count),
            dtype=torch.float32,
            device=device,
        )
        if waypoint_count > 1:
            dt[:, 1:] = step_dt
        return dt

    def _normalize_plan_result(
        self,
        result: PlanResult,
        target_states: list[PlanState],
        options: MotionGenOptions,
    ) -> PlanResult:
        """Enforce the public :class:`PlanResult` contract."""
        device = self._runtime_device()
        start_qpos = options.start_qpos
        if start_qpos is not None:
            if start_qpos.dim() == 1:
                start_qpos = start_qpos.unsqueeze(0)
            start_qpos = start_qpos.to(device)
            batch_size = start_qpos.shape[0]
            controlled_dof: int | None = start_qpos.shape[1]
        else:
            sample = target_states[0]
            if sample.qpos is not None:
                batch_size = 1 if sample.qpos.dim() == 1 else sample.qpos.shape[0]
            elif sample.xpos is not None:
                batch_size = 1 if sample.xpos.dim() == 2 else sample.xpos.shape[0]
            else:
                raise ValueError("The first target state has no qpos or xpos tensor.")
            controlled_dof = None

        success = normalize_success_mask(
            result.success,
            num_envs=batch_size,
            device=device,
            name="MotionGenerator PlanResult.success",
        )
        positions = result.positions
        if positions is None or positions.dim() != 3:
            raise ValueError(
                "MotionGenerator must return positions with shape (B, N, DOF)."
            )
        if positions.shape[0] != batch_size:
            raise ValueError(
                "MotionGenerator result batch size does not match its targets: "
                f"{positions.shape[0]} != {batch_size}."
            )
        if controlled_dof is not None and positions.shape[2] != controlled_dof:
            raise ValueError(
                "MotionGenerator result DoF does not match start_qpos: "
                f"{positions.shape[2]} != {controlled_dof}."
            )
        if positions.device != device:
            raise ValueError(
                "MotionGenerator returned positions on "
                f"{positions.device}, expected {device}."
            )
        if not torch.isfinite(positions).all():
            raise ValueError("MotionGenerator returned non-finite positions.")

        dt = result.dt
        if not isinstance(dt, torch.Tensor):
            raise ValueError(
                "MotionGenerator planner results with positions require explicit dt."
            )
        if dt.shape != positions.shape[:2]:
            raise ValueError(
                "MotionGenerator dt must match positions batch and sample "
                f"dimensions, got {tuple(dt.shape)} and "
                f"{tuple(positions.shape[:2])}."
            )
        if dt.device != device or not torch.isfinite(dt).all() or (dt < 0).any():
            raise ValueError("MotionGenerator returned invalid time deltas.")
        resampled = False
        preserve_samples = options.strategy == "motion_gen" and (
            getattr(self.planner, "preserve_plan_samples", False) is True
        )
        if (
            options.sample_count is not None
            and not preserve_samples
            and positions.shape[1] != options.sample_count
            and not (
                options.strategy == "ik_interp"
                and positions.shape[1] >= options.sample_count
            )
        ):
            positions, dt = resample_in_time(positions, dt, options.sample_count)
            resampled = True
        if (
            resampled
            and options.strategy == "motion_gen"
            and self.supports_joint_trajectory_validation
        ):
            control_part = (
                getattr(options.plan_opts, "control_part", None) or options.control_part
            )
            validity = self.validate_joint_trajectory(
                positions,
                control_part=control_part,
                obstacle_poses=getattr(
                    options.plan_opts, "dynamic_obstacle_poses", None
                ),
            )
            success = success & validity.all(dim=1)

        def normalize_derivative(
            value: torch.Tensor | None,
            name: str,
        ) -> torch.Tensor | None:
            if value is None or resampled:
                return None
            if value.shape != positions.shape:
                raise ValueError(
                    f"MotionGenerator {name} must match positions shape, "
                    f"got {tuple(value.shape)} and {tuple(positions.shape)}."
                )
            if value.device != device or not torch.isfinite(value).all():
                raise ValueError(
                    f"MotionGenerator returned invalid or wrong-device {name}."
                )
            return value

        velocities = normalize_derivative(result.velocities, "velocities")
        accelerations = normalize_derivative(result.accelerations, "accelerations")
        constraint_report = None if resampled else result.constraint_report
        preserve_failed_positions = (
            getattr(self.planner, "preserve_failed_plan_positions", False) is True
        )
        if (
            start_qpos is not None
            and not success.all()
            and not preserve_failed_positions
        ):
            # The backend report describes the original failed rows. A generic
            # facade cannot recompute arbitrary planner-specific diagnostics.
            constraint_report = None
            held = (
                start_qpos.to(dtype=positions.dtype).unsqueeze(1).expand_as(positions)
            )
            positions = torch.where(success[:, None, None], positions, held)
            if velocities is not None:
                velocities = torch.where(
                    success[:, None, None],
                    velocities,
                    torch.zeros_like(velocities),
                )
            if accelerations is not None:
                accelerations = torch.where(
                    success[:, None, None],
                    accelerations,
                    torch.zeros_like(accelerations),
                )
        if velocities is None:
            velocities = differentiate_positions(positions, dt)
        if options.strategy == "ik_interp" and positions.shape[1]:
            velocities = velocities.clone()
            velocities[:, 0] = 0.0
            velocities[:, -1] = 0.0
            if positions.shape[1] > 1:
                stationary = positions[:, 1:] == positions[:, :-1]
                velocities[:, :-1] = velocities[:, :-1].masked_fill(stationary, 0.0)
                velocities[:, 1:] = velocities[:, 1:].masked_fill(stationary, 0.0)

        return PlanResult(
            success=success,
            xpos_list=None if resampled else result.xpos_list,
            positions=positions,
            velocities=velocities,
            accelerations=accelerations,
            dt=dt,
            constraint_report=constraint_report,
        )

    def _runtime_device(self) -> torch.device:
        """Return the concrete device used by the generator."""
        device = torch.device(self.device)
        if device.type == "cuda" and device.index is None:
            return torch.device(f"cuda:{torch.cuda.current_device()}")
        return device

    def estimate_trajectory_sample_count(
        self,
        xpos_list: torch.Tensor | list[torch.Tensor] | None = None,
        qpos_list: torch.Tensor | list[torch.Tensor] | None = None,
        step_size: float | torch.Tensor = 0.01,
        angle_step: float | torch.Tensor = np.pi / 90,
        control_part: str | None = None,
    ) -> torch.Tensor:
        """Estimate the number of trajectory sampling points required.

        This function estimates the total number of sampling points needed to generate
        a trajectory based on the given waypoints and sampling parameters. Supports
        parallel computation for batched input trajectories.

        Args:
            xpos_list: Tensor of 4x4 transformation matrices, shape [B, N, 4, 4] or [N, 4, 4]
            qpos_list: Tensor of joint positions, shape [B, N, D] or [N, D] (optional)
            step_size: Maximum allowed distance between points (meters). Float or Tensor [B]
            angle_step: Maximum allowed angular difference between points (radians). Float or Tensor [B]

        Returns:
            torch.Tensor: Estimated number of sampling points per trajectory, shape [B]
                          (or scalar tensor if single trajectory)
        """
        # Input validation
        if xpos_list is None and qpos_list is None:
            return torch.tensor(0)

        # Handle lists gracefully if passed by legacy code
        if isinstance(xpos_list, list):
            xpos_list = torch.stack(
                [
                    x if isinstance(x, torch.Tensor) else torch.tensor(x)
                    for x in xpos_list
                ]
            ).float()
        elif isinstance(xpos_list, np.ndarray):
            xpos_list = torch.as_tensor(xpos_list, dtype=torch.float32)

        if isinstance(qpos_list, list):
            qpos_list = torch.stack(
                [
                    q if isinstance(q, torch.Tensor) else torch.tensor(q)
                    for q in qpos_list
                ]
            ).float()
        elif isinstance(qpos_list, np.ndarray):
            qpos_list = torch.as_tensor(qpos_list, dtype=torch.float32)

        device = qpos_list.device if qpos_list is not None else xpos_list.device

        original_dim = qpos_list.dim() if qpos_list is not None else xpos_list.dim()

        # If joint position list is provided but end effector position list is not,
        # convert through forward kinematics
        if qpos_list is not None and xpos_list is None:
            if original_dim == 2:  # [N, D]
                qpos_list = qpos_list.unsqueeze(0)  # [1, N, D]

            B, N, D = qpos_list.shape

            if N < 2:
                return torch.ones((B,), dtype=torch.int32, device=device)

            xpos_list = self.robot.compute_batch_fk(
                qpos=qpos_list,
                name=control_part,
                to_matrix=True,
            )
        else:
            if original_dim == 3:  # [N, 4, 4]
                xpos_list = xpos_list.unsqueeze(0)
            B, N, _, _ = xpos_list.shape

            if N < 2:
                return torch.ones((B,), dtype=torch.int32, device=device)

        # Convert step metrics to tensors
        if not isinstance(step_size, torch.Tensor):
            step_size = torch.full((B,), step_size, device=device, dtype=torch.float32)
        else:
            step_size = step_size.to(device)

        if not isinstance(angle_step, torch.Tensor):
            angle_step = torch.full(
                (B,), angle_step, device=device, dtype=torch.float32
            )
        else:
            angle_step = angle_step.to(device)

        # Calculate position distances
        start_poses = xpos_list[:, :-1]  # [B, N-1, 4, 4]
        end_poses = xpos_list[:, 1:]  # [B, N-1, 4, 4]

        pos_diffs = end_poses[:, :, :3, 3] - start_poses[:, :, :3, 3]
        pos_dists = torch.norm(pos_diffs, dim=-1)  # [B, N-1]
        total_pos_dist = pos_dists.sum(dim=-1)  # [B]

        # Calculate rotation angles
        start_rot = start_poses[:, :, :3, :3]  # [B, N-1, 3, 3]
        end_rot = end_poses[:, :, :3, :3]  # [B, N-1, 3, 3]

        start_rot_T = start_rot.transpose(-1, -2)
        rel_rot = torch.matmul(start_rot_T, end_rot)

        trace = rel_rot[..., 0, 0] + rel_rot[..., 1, 1] + rel_rot[..., 2, 2]
        cos_angle = (trace - 1.0) / 2.0
        # Add epsilon to prevent NaN in acos at boundaries
        cos_angle = torch.clamp(cos_angle, -1.0 + 1e-6, 1.0 - 1e-6)

        angles = torch.acos(cos_angle)  # [B, N-1]
        total_angle = angles.sum(dim=-1)  # [B]

        # Compute sampling points
        pos_samples = torch.clamp((total_pos_dist / step_size).int(), min=1)
        rot_samples = torch.clamp((total_angle / angle_step).int(), min=1)

        total_samples = torch.max(pos_samples, rot_samples)

        out_samples = torch.clamp(total_samples, min=2)

        if original_dim in (2, 3):  # Reshape back to scalar tensor if not batched
            return out_samples[0]

        return out_samples

    def plot_trajectory(
        self,
        positions: torch.Tensor,
        vels: torch.Tensor | None = None,
        accs: torch.Tensor | None = None,
    ) -> None:
        r"""Plot trajectory data.

        This method visualizes the trajectory by plotting position, velocity, and
        acceleration curves for each joint over time. It also displays the constraint
        limits for reference. Supports plotting batched trajectories.

        Args:
            positions: Position tensor (N, DOF) or (B, N, DOF)
            vels: Velocity tensor (N, DOF) or (B, N, DOF), optional
            accs: Acceleration tensor (N, DOF) or (B, N, DOF), optional

        Note:
            - Creates a multi-subplot figure (position, and optional velocity/acceleration)
            - Shows constraint limits as dashed lines
            - If input is (B, N, DOF), plots elements separately per batch sequence.
            - Requires matplotlib to be installed
        """
        # Ensure we're dealing with CPU tensors for plotting
        positions = positions.detach().cpu()
        if vels is not None:
            vels = vels.detach().cpu()
        if accs is not None:
            accs = accs.detach().cpu()

        time_step = 0.01

        # Helper to unsqueeze unbatched (N, DOF) -> (1, N, DOF)
        def ensure_batch_dim(tensor):
            if tensor is None:
                return None
            return tensor.unsqueeze(0) if tensor.ndim == 2 else tensor

        positions = ensure_batch_dim(positions)
        vels = ensure_batch_dim(vels)
        accs = ensure_batch_dim(accs)

        batch_size, num_steps, _ = positions.shape
        time_steps = np.arange(num_steps) * time_step

        num_plots = 1 + (1 if vels is not None else 0) + (1 if accs is not None else 0)
        fig, axs = plt.subplots(num_plots, 1, figsize=(10, 3 * num_plots))

        # Ensure axs is iterable even if there relies only 1 subplot
        if num_plots == 1:
            axs = [axs]

        for b in range(batch_size):
            line_style = "-" if batch_size == 1 else f"C{b}-"
            alpha = 1.0 if batch_size == 1 else max(0.2, 1.0 / np.sqrt(batch_size))

            for i in range(self.dofs):
                label = f"Joint {i + 1}" if b == 0 else ""
                axs[0].plot(
                    time_steps,
                    positions[b, :, i].numpy(),
                    line_style,
                    alpha=alpha,
                    label=label,
                )

                plot_idx = 1
                if vels is not None:
                    axs[plot_idx].plot(
                        time_steps,
                        vels[b, :, i].numpy(),
                        line_style,
                        alpha=alpha,
                        label=label,
                    )
                    plot_idx += 1
                if accs is not None:
                    axs[plot_idx].plot(
                        time_steps,
                        accs[b, :, i].numpy(),
                        line_style,
                        alpha=alpha,
                        label=label,
                    )

        axs[0].set_title("Position")
        plot_idx = 1
        if vels is not None:
            axs[plot_idx].set_title("Velocity")
            plot_idx += 1
        if accs is not None:
            axs[plot_idx].set_title("Acceleration")

        for ax in axs:
            ax.set_xlabel("Time [s]")
            ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.0))
            ax.grid()

        plt.tight_layout()
        plt.show()

    def interpolate_trajectory(
        self,
        control_part: str | None = None,
        xpos_list: torch.Tensor | None = None,
        qpos_list: torch.Tensor | None = None,
        options: MotionGenOptions | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        r"""Interpolate trajectory based on provided waypoints.

        This method performs interpolation on the provided waypoints to generate a
        smoother trajectory. It supports both Cartesian (end-effector) and joint
        space interpolation based on the control part and options specified.

        Args:
            control_part: Name of the robot part to control, e.g. 'left_arm'. Must
                correspond to a valid control part defined in the robot's configuration.
            xpos_list: End-effector poses, shape ``(B, N, 4, 4)`` or ``(N, 4, 4)``.
                Required if control_part is an end-effector control part.
            qpos_list: Joint positions, shape ``(B, N, DOF)`` or ``(N, DOF)``.
                Required if control_part is a joint control part.
            options: MotionGenOptions containing interpolation settings such as step
                size and whether to use linear interpolation.

        Returns:
            Tuple containing:
                - interpolate_qpos_list: Interpolated joint positions, shape
                  ``(B, M, DOF)``.
                - feasible_pose_targets: Corresponding end-effector poses, shape
                  ``(B, M, 4, 4)``, or ``None`` if not applicable.

        Raises:
            ValueError: If any required Cartesian waypoint fails IK or FK
                consistency. Use :meth:`generate` for per-environment failures.
        """
        qpos, poses, success = self._interpolate_waypoints(
            control_part=control_part,
            xpos_list=xpos_list,
            qpos_list=qpos_list,
            options=options,
        )
        if not success.all():
            raise ValueError(
                "Cartesian interpolation failed for required IK waypoints."
            )
        return qpos, poses

    def _interpolate_waypoints(
        self,
        control_part: str | None = None,
        xpos_list: torch.Tensor | None = None,
        qpos_list: torch.Tensor | None = None,
        options: MotionGenOptions | None = None,
        *,
        sparse_cartesian: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """Prepare required IK waypoints, optionally without Cartesian densification."""
        options = MotionGenOptions() if options is None else options

        # Normalize single-env inputs to batched form.
        if qpos_list is not None and qpos_list.dim() == 2:
            qpos_list = qpos_list.unsqueeze(0)
        if xpos_list is not None and xpos_list.dim() == 3:
            xpos_list = xpos_list.unsqueeze(0)

        if qpos_list is not None and xpos_list is None and options.is_linear:
            # qpos_list is (B, N, DOF); compute_batch_fk handles batched qpos directly.
            xpos_list = self.robot.compute_batch_fk(
                qpos=qpos_list,
                name=control_part,
                to_matrix=True,
            )  # (B, N, 4, 4)

        if xpos_list is None and qpos_list is None:
            logger.log_error("Either xpos_list or qpos_list must be provided")

        # Input validation: the waypoint count is the second-to-last or last batch dim.
        if (xpos_list is not None and xpos_list.shape[-3] < 2) or (
            qpos_list is not None and qpos_list.shape[-2] < 2
        ):
            logger.log_error(
                "xpos_list and qpos_list must contain at least 2 way points"
            )

        qpos_seed = options.start_qpos
        if qpos_seed is not None and qpos_seed.dim() == 1:
            qpos_seed = qpos_seed.unsqueeze(0)
        if qpos_seed is None and qpos_list is not None:
            # First waypoint per env as seed.
            qpos_seed = qpos_list[:, 0]  # (B, DOF)
        if qpos_seed is None:
            # Fallback to current robot state as seed.
            qpos_seed = self.robot.get_qpos(name=control_part)  # (B, DOF)
        qpos_seed = qpos_seed.to(self._runtime_device())

        # Generate trajectory
        if options.is_linear or qpos_list is None:
            if sparse_cartesian:
                # A sparse joint planner owns timing between the requested
                # goals. Densifying a free Cartesian move would turn every IK
                # sample into a mandatory rest point for trapezoidal timing.
                total_interpolated_poses = xpos_list.to(self._runtime_device())
            else:
                # Allocations are per environment; share their segment maxima
                # so the batch has one interpolated pose count.
                per_env_allocations = [
                    calculate_point_allocations(
                        xpos_list[b],
                        step_size=options.interpolate_position_step,
                        angle_step=options.interpolate_angle_step,
                        device=self.device,
                    )
                    for b in range(xpos_list.shape[0])
                ]
                n_segments = xpos_list.shape[1] - 1
                interpolated_point_allocations = [
                    max(alloc[i] for alloc in per_env_allocations)
                    for i in range(n_segments)
                ]

                total_interpolated_poses = []
                for i in range(n_segments):
                    seg = interpolate_xpos_batched(
                        xpos_list[:, i],
                        xpos_list[:, i + 1],
                        interpolated_point_allocations[i],
                    )  # (B, seg, 4, 4)
                    total_interpolated_poses.append(seg)
                total_interpolated_poses = torch.cat(
                    total_interpolated_poses, dim=1
                ).to(
                    self._runtime_device()
                )  # (B, M, 4, 4)

            if control_part is None:
                raise ValueError("Cartesian interpolation requires control_part.")
            # The observed start is already grounded by FK. Re-solving it with
            # analytic IK can fail at a wrist singularity despite later poses
            # being reachable. Preserve it only for rows whose whole pose agrees.
            start_pose = self.robot.compute_fk(
                qpos=qpos_seed, name=control_part, to_matrix=True
            )
            known_start = torch.isclose(
                total_interpolated_poses[:, 0],
                start_pose.to(total_interpolated_poses),
                rtol=0.0,
                atol=1e-6,
            ).all(dim=(-2, -1))
            success, qpos_batch = self._solve_cartesian_waypoints(
                total_interpolated_poses,
                qpos_seed,
                control_part,
                known_start=known_start,
            )
            # Keep the FK consistency guard, but never remove required poses.
            fk_batch = self.robot.compute_batch_fk(
                qpos=qpos_batch, name=control_part, to_matrix=True
            )
            pos_err = torch.linalg.vector_norm(
                fk_batch[..., :3, 3] - total_interpolated_poses[..., :3, 3], dim=-1
            )
            rot_err = torch.linalg.vector_norm(
                fk_batch[..., :3, :3] - total_interpolated_poses[..., :3, :3],
                dim=(-2, -1),
            )
            success &= ((pos_err < 0.02) & (rot_err < 0.2)).all(dim=1)
            interpolate_qpos_list = torch.where(
                success[:, None, None], qpos_batch, qpos_seed[:, None, :]
            )
            feasible_pose_targets = total_interpolated_poses
        else:
            # Joint-space interpolation. qpos_list is (B, N, DOF).
            if isinstance(options.interpolate_nums, int):
                interp_nums = [options.interpolate_nums] * (qpos_list.shape[1] - 1)
            else:
                if len(options.interpolate_nums) != qpos_list.shape[1] - 1:
                    logger.log_error(
                        "Length of interpolate_nums list must equal number of segments",
                        ValueError,
                    )
                interp_nums = options.interpolate_nums

            interpolate_qpos_list = interpolate_with_nums(
                qpos_list, interp_nums=interp_nums, device=self.device
            )  # (B, M, DOF)
            feasible_pose_targets = None
            success = torch.ones(
                qpos_list.shape[0], dtype=torch.bool, device=self._runtime_device()
            )

        return interpolate_qpos_list, feasible_pose_targets, success
