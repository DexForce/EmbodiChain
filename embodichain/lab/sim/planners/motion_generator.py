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

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import MISSING
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import torch

from embodichain.lab.sim.planners import (
    BasePlannerCfg,
    PlanOptions,
    BasePlanner,
    ToppraPlanner,
    ToppraPlannerCfg,
    ToppraPlanOptions,
    NeuralPlanner,
    NeuralPlannerCfg,
    CuroboPlanner,
    CuroboPlannerCfg,
)
from embodichain.lab.sim.utility.action_utils import (
    interpolate_with_distance,
    interpolate_with_nums,
    resample_with_distance,
)
from embodichain.utils import logger, configclass
from .utils import (
    MoveType,
    PlanResult,
    PlanState,
    TrajectorySampleMethod,
    normalize_success_mask,
)
from .utils import (
    calculate_point_allocations,
    interpolate_xpos,
    interpolate_xpos_batched,
)

__all__ = ["MotionGenerator", "MotionGenCfg", "MotionGenOptions"]


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
    """Optional scalar joint velocity limit used by compatible backends."""

    acceleration_limit: float | None = None
    """Optional scalar joint acceleration limit used by compatible backends."""

    start_qpos: torch.Tensor | None = None
    """Optional starting joint configuration for the trajectory, shape (B, DOF). If provided, the planner will ensure that the trajectory starts from this configuration. If not provided, the planner will use the current joint configuration of the robot as the starting point."""

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

    interpolate_nums: int | list[int] = 10
    """Number of interpolation points to generate between each pair of waypoints. 
    
    Can be an integer (same for all segments) or a list of integers with len(PlanState) specifying the number of points for each segment."""

    is_linear: bool = False
    """If True, use cartesian linear interpolation, else joint space"""

    preserve_cartesian_samples: bool = False
    """Treat Cartesian targets as exact output samples and solve each with IK.

    This constrained mode requires exactly ``sample_count - 1`` target states;
    the observed start configuration supplies the first output sample.
    """

    interpolate_position_step: float = 0.002
    """Step size for interpolation. If is_linear is True, this is the step size in Cartesian space (meters). If is_linear is False, this is the step size in joint space (radians)."""

    interpolate_angle_step: float = np.pi / 90
    """Angular step size for interpolation in joint space (radians). Only used if is_linear is False."""

    def __post_init__(self) -> None:
        """Validate backend-neutral motion generation options."""
        valid_strategies = {"motion_gen", "ik_interp"}
        if self.strategy not in valid_strategies:
            raise ValueError(
                f"strategy must be one of {sorted(valid_strategies)}, "
                f"got {self.strategy!r}."
            )
        if self.sample_count is not None and self.sample_count < 2:
            raise ValueError("sample_count must be at least 2 when set.")
        if self.velocity_limit is not None and self.velocity_limit <= 0.0:
            raise ValueError("velocity_limit must be greater than zero when set.")
        if self.acceleration_limit is not None and self.acceleration_limit <= 0.0:
            raise ValueError("acceleration_limit must be greater than zero when set.")


class MotionGenerator:
    r"""Unified motion generator for robot trajectory planning.

    This class provides a unified interface for trajectory planning with and without
    collision checking.

    Args:
        cfg: Configuration object for motion generation, must include 'planner_cfg' attribute
    """

    _support_planner_dict = {
        "toppra": (ToppraPlanner, ToppraPlannerCfg),
        "neural": (NeuralPlanner, NeuralPlannerCfg),
        "curobo": (CuroboPlanner, CuroboPlannerCfg),
    }

    def __init__(self, cfg: MotionGenCfg) -> None:

        # Create planner based on planner_type
        self.planner: BasePlanner = self._create_planner(cfg.planner_cfg)

        self.robot = self.planner.robot
        self.device = self.robot.device

    @property
    def supports_dynamic_collision_world(self) -> bool:
        """Whether the planner accepts per-plan dynamic obstacle poses.

        Returns:
            ``True`` when the selected planner supports collision-world updates.
        """
        return getattr(self.planner, "supports_collision_world_updates", False) is True

    @property
    def dynamic_collision_entity_ids(self) -> tuple[str, ...]:
        """Return canonical dynamic-obstacle IDs declared by the planner."""
        entity_ids = getattr(self.planner, "dynamic_collision_entity_ids", ())
        return self._validate_collision_entity_ids(
            entity_ids,
            field_name="dynamic_collision_entity_ids",
        )

    @property
    def collision_world_entity_ids(self) -> tuple[str, ...]:
        """Return every canonical entity ID in the planner collision world."""
        entity_ids = getattr(self.planner, "collision_world_entity_ids", ())
        return self._validate_collision_entity_ids(
            entity_ids,
            field_name="collision_world_entity_ids",
        )

    @staticmethod
    def _validate_collision_entity_ids(
        entity_ids: object,
        *,
        field_name: str,
    ) -> tuple[str, ...]:
        """Validate one planner-owned canonical collision-ID declaration."""
        if not isinstance(entity_ids, tuple) or not all(
            isinstance(entity_id, str) and entity_id and entity_id == entity_id.strip()
            for entity_id in entity_ids
        ):
            raise TypeError(
                f"Planner.{field_name} must be a tuple of "
                "non-empty strings without outer whitespace."
            )
        if len(set(entity_ids)) != len(entity_ids):
            raise ValueError(f"Planner.{field_name} must contain unique IDs.")
        return entity_ids

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
                f"{field_name} keys must be non-empty strings without outer "
                "whitespace."
            )
        return set(entity_ids)

    @property
    def collision_world_batch_mode(self) -> Literal["shared", "per_env"] | None:
        """Return the backend's dynamic collision-world batch-sharing mode."""
        mode = getattr(self.planner, "collision_world_batch_mode", None)
        if mode not in (None, "shared", "per_env"):
            raise ValueError(
                "Planner.collision_world_batch_mode must be 'shared', 'per_env', "
                "or None."
            )
        return mode

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
        if not self.supports_dynamic_collision_world:
            logger.log_error(
                f"{type(self.planner).__name__} does not support dynamic "
                "collision-world updates.",
                ValueError,
            )
        configured_ids = self.dynamic_collision_entity_ids
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
            return deepcopy(plan_opts)
        if sample_count is not None and self.planner.cfg.planner_type == "toppra":
            return ToppraPlanOptions(
                sample_method=TrajectorySampleMethod.QUANTITY,
                sample_interval=sample_count,
                constraints={
                    "velocity": 0.2 if velocity_limit is None else velocity_limit,
                    "acceleration": (
                        0.5 if acceleration_limit is None else acceleration_limit
                    ),
                },
            )
        return self.planner.default_plan_options()

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
        interpolation (``"ik_interp"``). Joint targets fall back to interpolation
        when the configured backend cannot consume :class:`MoveType.JOINT_MOVE`.

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
        if options.strategy not in {"motion_gen", "ik_interp"}:
            raise ValueError(
                "strategy must be 'motion_gen' or 'ik_interp', "
                f"got {options.strategy!r}."
            )

        move_types = {state.move_type for state in target_states}
        if len(move_types) != 1:
            names = sorted(move_type.name for move_type in move_types)
            raise ValueError(f"All target states must share move_type; got {names}.")
        move_type = target_states[0].move_type
        use_interpolation = (
            options.preserve_cartesian_samples
            or options.strategy == "ik_interp"
            or (
                move_type is MoveType.JOINT_MOVE
                and not self.planner.supports_move_type(MoveType.JOINT_MOVE)
            )
        )
        if use_interpolation:
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
        move_type = target_states[0].move_type
        should_preinterpolate = (
            options.is_interpolate
            and not self.planner.supports_move_type(MoveType.EEF_MOVE)
            and self.planner.supports_move_type(MoveType.JOINT_MOVE)
        )

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

            qpos_interpolated, _ = self.interpolate_trajectory(
                control_part=options.control_part,
                xpos_list=xpos_list,
                qpos_list=qpos_list,
                options=options,
            )
            target_plan_states = [
                PlanState(move_type=MoveType.JOINT_MOVE, qpos=qpos_interpolated[:, j])
                for j in range(qpos_interpolated.shape[1])
            ]
        else:
            target_plan_states = target_states

        unsupported_move_types = (
            set() if self.planner.supports_move_type(move_type) else {move_type}
        )
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

        plan_opts = self.resolve_plan_options(
            options.plan_opts,
            sample_count=options.sample_count,
            velocity_limit=options.velocity_limit,
            acceleration_limit=options.acceleration_limit,
        )
        plan_opts = self.planner.with_motion_context(
            plan_opts,
            start_qpos=options.start_qpos,
            control_part=options.control_part,
        )
        return self.planner.plan(
            target_states=target_plan_states,
            options=plan_opts,
        )

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
            positions = interpolate_with_distance(
                trajectory=keyframes,
                interp_num=options.sample_count,
                device=device,
            )
            return PlanResult(
                success=torch.ones(batch_size, dtype=torch.bool, device=device),
                positions=positions,
            )

        if move_type is not MoveType.EEF_MOVE:
            raise ValueError(
                "strategy='ik_interp' supports only EEF_MOVE and JOINT_MOVE, "
                f"got {move_type.name}."
            )
        if options.control_part is None:
            raise ValueError("EEF_MOVE IK interpolation requires control_part.")

        success = torch.ones(batch_size, dtype=torch.bool, device=device)
        qpos_seed = start_qpos
        solved_waypoints: list[torch.Tensor] = []
        for index, state in enumerate(target_states):
            if state.xpos is None:
                raise ValueError("EEF_MOVE target states require xpos tensors.")
            pose = state.xpos
            if pose.dim() == 2:
                pose = pose.unsqueeze(0)
            expected_shape = (batch_size, 4, 4)
            if pose.shape != expected_shape:
                raise ValueError(
                    f"EEF_MOVE target xpos must have shape {expected_shape}, "
                    f"got {tuple(pose.shape)}."
                )
            step_success, qpos = self.robot.compute_ik(
                pose=pose.to(device),
                name=options.control_part,
                joint_seed=qpos_seed,
            )
            step_success = normalize_success_mask(
                step_success,
                num_envs=batch_size,
                device=device,
                name=f"IK success for target state {index}",
            )
            qpos = torch.as_tensor(
                qpos,
                dtype=start_qpos.dtype,
                device=device,
            )
            if qpos.shape != (batch_size, controlled_dof):
                raise ValueError(
                    "IK qpos must have shape "
                    f"({batch_size}, {controlled_dof}), got {tuple(qpos.shape)}."
                )
            if not step_success.all():
                logger.log_warning(
                    f"Failed to compute IK for target state {index} in some "
                    "environments."
                )
            qpos = torch.where(step_success[:, None], qpos, qpos_seed)
            success &= step_success
            solved_waypoints.append(qpos)
            qpos_seed = qpos

        keyframes = torch.cat(
            [start_qpos.unsqueeze(1), torch.stack(solved_waypoints, dim=1)],
            dim=1,
        )
        if options.preserve_cartesian_samples:
            if keyframes.shape[1] != options.sample_count:
                raise ValueError(
                    "Linear Cartesian targets must provide sample_count - 1 "
                    "keyframes so every output sample is IK-grounded; got "
                    f"{len(target_states)} targets for sample_count "
                    f"{options.sample_count}."
                )
            positions = keyframes
        else:
            positions = interpolate_with_distance(
                trajectory=keyframes,
                interp_num=options.sample_count,
                device=device,
            )
        held = start_qpos.unsqueeze(1).expand_as(positions)
        positions = torch.where(success[:, None, None], positions, held)
        return PlanResult(success=success, positions=positions)

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

        resampled = False
        preserve_samples = getattr(self.planner, "preserve_plan_samples", False) is True
        if (
            options.sample_count is not None
            and not preserve_samples
            and positions.shape[1] != options.sample_count
        ):
            positions = resample_with_distance(
                trajectory=positions,
                interp_num=options.sample_count,
                device=device,
            )
            resampled = True

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
        dt = None if resampled else result.dt
        if dt is not None:
            if not isinstance(dt, torch.Tensor):
                raise TypeError("MotionGenerator dt must be a torch.Tensor.")
            if dt.shape != positions.shape[:2]:
                raise ValueError(
                    "MotionGenerator dt must match positions batch and sample "
                    f"dimensions, got {tuple(dt.shape)} and "
                    f"{tuple(positions.shape[:2])}."
                )
            if dt.device != device or not torch.isfinite(dt).all() or (dt < 0).any():
                raise ValueError("MotionGenerator returned invalid time deltas.")
            duration: float | torch.Tensor = dt.sum(dim=1)
        else:
            duration = result.duration

        if start_qpos is not None and not success.all():
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

        return PlanResult(
            success=success,
            xpos_list=None if resampled else result.xpos_list,
            positions=positions,
            velocities=velocities,
            accelerations=accelerations,
            dt=dt,
            duration=duration,
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
                label = f"Joint {i+1}" if b == 0 else ""
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
        """
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

        # Generate trajectory
        if options.is_linear or qpos_list is None:
            # ``calculate_point_allocations`` only handles single-env (N, 4, 4),
            # so compute allocations per env and use the per-segment maximum so
            # all envs can share the same interpolated pose count.
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

            # Linear cartesian interpolation, batched across B envs.
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
            )  # (B, M, 4, 4)

            qpos_seed_b = qpos_seed
            if qpos_seed_b.dim() == 1:
                qpos_seed_b = qpos_seed_b.unsqueeze(0).repeat(xpos_list.shape[0], 1)
            joint_seed = qpos_seed_b.unsqueeze(1).repeat(
                1, total_interpolated_poses.shape[1], 1
            )  # (B, M, D)
            success_batch, qpos_batch = self.robot.compute_batch_ik(
                pose=total_interpolated_poses,
                joint_seed=joint_seed,
                name=control_part,
            )  # (B, M), (B, M, D)

            has_nan = torch.isnan(qpos_batch).any(dim=-1)
            valid = success_batch.bool() & (~has_nan)  # (B, M)

            # Vectorized FK feasibility check to keep only physically consistent IK outputs.
            if valid.any():
                fk_batch = self.robot.compute_batch_fk(
                    qpos=qpos_batch,
                    name=control_part,
                    to_matrix=True,
                )  # (B, M, 4, 4)
                pos_err = torch.norm(
                    fk_batch[:, :, :3, 3] - total_interpolated_poses[:, :, :3, 3],
                    dim=-1,
                )
                rot_err = torch.norm(
                    fk_batch[:, :, :3, :3] - total_interpolated_poses[:, :, :3, :3],
                    dim=(-2, -1),
                )
                fk_valid = (pos_err < 0.02) & (rot_err < 0.2)
                valid = valid & fk_valid

            # Per-env filter: keep only valid rows; pad short envs by repeating last valid.
            B, M, D = qpos_batch.shape
            max_valid = int(valid.sum(dim=1).max().item())
            max_valid = max(max_valid, 1)
            interp_q = torch.zeros(
                B, max_valid, D, device=self.device, dtype=torch.float32
            )
            feasible = torch.zeros(
                B, max_valid, 4, 4, device=self.device, dtype=torch.float32
            )
            for b in range(B):
                v = qpos_batch[b][valid[b]]
                f = total_interpolated_poses[b][valid[b]]
                if v.shape[0] == 0:
                    v = qpos_batch[b : b + 1, 0]
                    f = total_interpolated_poses[b : b + 1, 0]
                interp_q[b, : v.shape[0]] = v
                interp_q[b, v.shape[0] :] = v[-1]
                feasible[b, : f.shape[0]] = f
                feasible[b, f.shape[0] :] = f[-1]
            interpolate_qpos_list = interp_q
            feasible_pose_targets = feasible
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

        return interpolate_qpos_list, feasible_pose_targets
