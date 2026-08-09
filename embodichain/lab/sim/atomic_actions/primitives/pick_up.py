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

"""PickUp atomic action implementation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

import torch

from embodichain.lab.sim.planners import MoveType, PlanState
from embodichain.utils import logger
from embodichain.utils.math import (
    axis_angle_to_rotation_matrix,
    pose_inv,
    quat_error_magnitude,
    quat_from_matrix,
)

from ._helpers import arm_qpos_from_state
from ..affordance import AntipodalAffordance
from ..bindings import ResolvedControlPart
from ..control import GRASP_COMMAND, OPEN_COMMAND
from ..core import AtomicAction, ObjectSemantics
from ..effects import StateDelta
from ..goals import (
    ObjectActionGoal,
    PoseGoalValue,
    resolve_pose_goal,
    validate_pose_goal,
)
from ..invocation import ActionOptions, ResolvedActionRequest
from ..plans import ActionPlan
from ..policies import MotionPolicy
from ..state import HeldObjectState, PlanningContext


@dataclass(frozen=True, slots=True, eq=False)
class GraspGoal(ObjectActionGoal):
    """Pickup target with an affordance-selected or supplied grasp pose."""

    goal_kind: ClassVar[str] = "grasp"

    grasp_xpos: PoseGoalValue | None = None
    """Optional end-effector grasp pose.

    When omitted, :class:`PickUp` selects a grasp from the target affordance. An
    explicit tensor or late-bound
    :class:`~embodichain.lab.sim.atomic_actions.goals.SceneEntityPose` skips
    grasp sampling. Late-bound poses also declare the scene dependency used by
    closed-loop execution recovery.
    """

    def __post_init__(self) -> None:
        ObjectActionGoal.__post_init__(self)
        if self.grasp_xpos is not None:
            validate_pose_goal(self.grasp_xpos, "grasp_xpos", allow_waypoints=False)


@dataclass(frozen=True, slots=True, eq=False)
class PickUpOptions(ActionOptions):
    """Per-invocation pickup behavior."""

    hand_interp_steps: int = 5
    """Number of waypoints for the gripper close interpolation phase."""

    pick_object_part: str = "center"
    """Name of the object part to pick up (used for grasp pose generation). Currently support [center | top | bottom]."""

    lift_height: float = 0.1
    """Height (m) to lift the end-effector after closing the gripper."""

    pre_grasp_distance: float = 0.15
    """Distance to offset back from the grasp pose along the approach direction."""

    approach_direction: torch.Tensor = torch.tensor([0, 0, -1], dtype=torch.float32)
    """World-frame direction from the pre-grasp pose to the grasp pose."""

    approach_alignment_max_angle: float | None = None
    """Optional maximum TCP z-axis deviation from the approach direction."""

    downstream_object_target_poses: tuple[torch.Tensor, ...] = ()
    """Future object poses that must be reachable with the selected grasp."""

    obj_upright_direction: torch.Tensor | None = None
    """Optional object local direction used to choose the upright grasp rotation."""

    rotate_upright: float | None = None
    """Optional rotation (radians) about the grasp x-axis to apply after grasp selection."""

    def __post_init__(self) -> None:
        if self.hand_interp_steps < 1:
            raise ValueError("hand_interp_steps must be at least 1.")
        if not isinstance(self.pick_object_part, str) or not self.pick_object_part:
            raise ValueError("pick_object_part must be a non-empty string.")
        if self.lift_height < 0.0:
            raise ValueError("lift_height must be non-negative.")
        if self.pre_grasp_distance < 0.0:
            raise ValueError("pre_grasp_distance must be non-negative.")
        if self.approach_direction.shape != (3,):
            raise ValueError("approach_direction must have shape (3,).")
        if not torch.isfinite(self.approach_direction).all():
            raise ValueError("approach_direction must contain finite values.")
        if torch.linalg.vector_norm(self.approach_direction) <= 1.0e-6:
            raise ValueError("approach_direction must be non-zero.")
        if self.approach_alignment_max_angle is not None and not (
            0.0 <= self.approach_alignment_max_angle <= math.pi / 2
        ):
            raise ValueError("approach_alignment_max_angle must be in [0, pi / 2].")
        if self.obj_upright_direction is not None and (
            self.obj_upright_direction.shape != (3,)
            or not torch.isfinite(self.obj_upright_direction).all()
        ):
            raise ValueError("obj_upright_direction must be a finite (3,) tensor.")
        object.__setattr__(self, "approach_direction", self.approach_direction.clone())
        object.__setattr__(
            self,
            "downstream_object_target_poses",
            tuple(value.clone() for value in self.downstream_object_target_poses),
        )
        if self.obj_upright_direction is not None:
            object.__setattr__(
                self, "obj_upright_direction", self.obj_upright_direction.clone()
            )


class PickUp(AtomicAction[GraspGoal, PickUpOptions]):
    """Approach a grasp pose, close the gripper, lift."""

    skill_id: ClassVar[str] = "pick_up"
    GoalType: ClassVar[type] = GraspGoal
    OptionsType: ClassVar[type] = PickUpOptions
    manipulator_roles: ClassVar[tuple[str, ...]] = ("primary",)
    end_effector_roles: ClassVar[tuple[str, ...]] = ("primary",)

    def __init__(
        self,
        default_options: PickUpOptions | None = None,
    ) -> None:
        super().__init__(default_options)

    def _on_bind(self) -> None:
        """Resolve engine-wide resources from the owning engine."""
        self.n_envs = self.robot.get_qpos().shape[0]
        self.robot_dof = self.robot.dof

    def _get_full_pickup_trajectory(
        self,
        grasp_xpos: torch.Tensor,
        start_arm_qpos: torch.Tensor,
        last_qpos: torch.Tensor,
        motion_policy: MotionPolicy,
        options: PickUpOptions,
        approach_direction: torch.Tensor,
        manipulator: ResolvedControlPart,
        end_effector: ResolvedControlPart,
        hand_open_qpos: torch.Tensor,
        hand_grasp_qpos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pre_grasp_xpos = self.builder.apply_local_offset(
            grasp_xpos, -approach_direction * options.pre_grasp_distance
        )

        n_approach, n_close, n_lift = self.builder.split_three_phase(
            motion_policy.sample_count,
            options.hand_interp_steps,
            first_phase_name="approach",
            third_phase_name="lift",
        )

        target_states_list = [
            [
                PlanState(xpos=pre_grasp_xpos[i], move_type=MoveType.EEF_MOVE),
                PlanState(xpos=grasp_xpos[i], move_type=MoveType.EEF_MOVE),
            ]
            for i in range(self.n_envs)
        ]
        approach_success, approach_arm = self.builder.plan_arm_traj(
            target_states_list,
            start_arm_qpos,
            n_approach,
            control_part=manipulator.name,
            arm_dof=manipulator.dof,
            cfg=motion_policy,
        )

        grasp_arm_qpos = approach_arm[:, -1, :]
        lift_xpos = self.builder.apply_local_offset(
            grasp_xpos,
            torch.tensor([0, 0, 1], device=self.device) * options.lift_height,
        )
        target_states_list = [
            [PlanState(xpos=lift_xpos[i], move_type=MoveType.EEF_MOVE)]
            for i in range(self.n_envs)
        ]
        lift_success, lift_arm = self.builder.plan_arm_traj(
            target_states_list,
            grasp_arm_qpos,
            n_lift,
            control_part=manipulator.name,
            arm_dof=manipulator.dof,
            cfg=motion_policy,
        )
        is_success = approach_success & lift_success

        hand_close_path = self.builder.interpolate_hand_qpos(
            hand_open_qpos, hand_grasp_qpos, n_waypoints=n_close
        )
        n_approach_actual = approach_arm.shape[1]
        n_lift_actual = lift_arm.shape[1]
        full = torch.empty(
            (self.n_envs, n_approach_actual + n_close + n_lift_actual, self.robot_dof),
            dtype=torch.float32,
            device=self.device,
        )
        full[:, :, :] = last_qpos.unsqueeze(1)
        arm_joint_ids = list(manipulator.joint_ids)
        hand_joint_ids = list(end_effector.joint_ids)
        full[:, :n_approach_actual, arm_joint_ids] = approach_arm
        full[:, :n_approach_actual, hand_joint_ids] = hand_open_qpos.unsqueeze(1)
        full[:, n_approach_actual : n_approach_actual + n_close, arm_joint_ids] = (
            grasp_arm_qpos.unsqueeze(1)
        )
        full[:, n_approach_actual : n_approach_actual + n_close, hand_joint_ids] = (
            hand_close_path
        )
        full[:, n_approach_actual + n_close :, arm_joint_ids] = lift_arm
        full[:, n_approach_actual + n_close :, hand_joint_ids] = (
            hand_grasp_qpos.unsqueeze(1)
        )
        return is_success, full

    def plan(
        self,
        request: ResolvedActionRequest[GraspGoal, PickUpOptions],
        context: PlanningContext,
    ) -> ActionPlan:
        """Plan approach, close, and lift phases without committing attachment."""
        target = self.require_goal(request)
        options = request.skill_options
        approach_direction = options.approach_direction.to(
            device=self.device, dtype=torch.float32
        )
        approach_direction = approach_direction / torch.linalg.vector_norm(
            approach_direction
        )
        binding = request.binding
        manipulator = binding.manipulator()
        end_effector = binding.end_effector()
        hand_open_qpos = end_effector.joint_positions(
            OPEN_COMMAND,
            n_envs=context.batch_size,
            device=self.device,
            dtype=context.robot.qpos.dtype,
        )
        hand_grasp_qpos = end_effector.joint_positions(
            GRASP_COMMAND,
            n_envs=context.batch_size,
            device=self.device,
            dtype=context.robot.qpos.dtype,
        )
        control_part = manipulator.name
        state = context
        sem = target.semantics
        if target.grasp_xpos is None and not isinstance(
            sem.affordance, AntipodalAffordance
        ):
            logger.log_error(
                "PickUp requires an AntipodalAffordance when grasp_xpos is not set.",
                ValueError,
            )
        if sem.entity is None:
            logger.log_error(
                "PickUp requires an entity on the target semantics.", ValueError
            )
        start_arm_qpos = self.builder.resolve_start_qpos(
            arm_qpos_from_state(state, list(manipulator.joint_ids)),
            n_envs=self.n_envs,
            arm_dof=manipulator.dof,
            control_part=control_part,
        )
        if target.grasp_xpos is None:
            is_success, grasp_xpos = self._resolve_grasp_pose(
                sem, start_arm_qpos, manipulator, options, approach_direction
            )
        else:
            grasp_xpos = self.builder.resolve_pose_target(
                resolve_pose_goal(target.grasp_xpos, context, name="grasp_xpos"),
                n_envs=self.n_envs,
            )
            if options.rotate_upright is not None:
                grasp_xpos = self._upright_adjusted_grasp_poses(
                    sem, grasp_xpos, options
                )
            is_success = torch.ones(self.n_envs, dtype=torch.bool, device=self.device)
        if not self.builder.all_envs_success(is_success):
            logger.log_warning("PickUp failed to resolve a grasp pose.")
            return self.failed_plan(
                request, context, message="Failed to resolve a grasp pose."
            )

        is_success, full = self._get_full_pickup_trajectory(
            grasp_xpos,
            start_arm_qpos,
            state.last_qpos,
            request.motion_policy,
            options,
            approach_direction,
            manipulator,
            end_effector,
            hand_open_qpos,
            hand_grasp_qpos,
        )

        obj_poses = sem.entity.get_local_pose(to_matrix=True)
        object_to_eef = torch.bmm(pose_inv(obj_poses), grasp_xpos)
        held = HeldObjectState(
            semantics=sem, object_to_eef=object_to_eef, grasp_xpos=grasp_xpos
        )
        coordinated_updates = {
            key: None for key in state.coordinated_held_objects if control_part in key
        }
        return self.build_plan(
            request,
            context,
            success=is_success,
            trajectory=full,
            expected_effects=StateDelta(
                held_object_updates={control_part: held},
                coordinated_held_object_updates=coordinated_updates,
            ),
            phase_name="pick",
        )

    def _resolve_grasp_pose(
        self,
        semantics: ObjectSemantics,
        start_qpos: torch.Tensor,
        manipulator: ResolvedControlPart,
        options: PickUpOptions,
        approach_direction: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        obj_poses = semantics.entity.get_local_pose(to_matrix=True)
        grasp_poses_result = semantics.affordance.get_valid_grasp_poses(
            obj_poses=obj_poses,
            approach_direction=approach_direction,
            object_part=options.pick_object_part,
        )
        n_envs = obj_poses.shape[0]
        n_max_pose = max(r[0].shape[0] for r in grasp_poses_result)
        grasp_xpos_padding = torch.zeros(
            (n_envs, n_max_pose, 4, 4), dtype=torch.float32, device=self.device
        )
        grasp_cost_padding = torch.full(
            (n_envs, n_max_pose),
            float("inf"),
            dtype=torch.float32,
            device=self.device,
        )
        for i in range(n_envs):
            n_pose = grasp_poses_result[i][0].shape[0]
            grasp_poses = grasp_poses_result[i][0].to(
                device=self.device, dtype=torch.float32
            )
            grasp_costs = grasp_poses_result[i][1].to(
                device=self.device, dtype=torch.float32
            )
            grasp_xpos_padding[i, :n_pose] = grasp_poses
            grasp_cost_padding[i, :n_pose] = grasp_costs
            grasp_xpos_padding[i, n_pose:] = grasp_poses[0]
            grasp_cost_padding[i, n_pose:] = grasp_costs[0]
        grasp_xpos_padding, ik_success = self._select_feasible_grasp_variants(
            semantics,
            grasp_xpos_padding,
            start_qpos,
            obj_poses,
            manipulator,
            options,
            approach_direction,
        )
        grasp_cost_masked = torch.where(ik_success, grasp_cost_padding, 10000.0)
        best_cost, best_idx = grasp_cost_masked.min(dim=1)
        is_success = best_cost < 9999.0
        best_grasp_xpos = grasp_xpos_padding[
            torch.arange(n_envs, device=self.device), best_idx
        ]
        return is_success, best_grasp_xpos

    def _select_feasible_grasp_variants(
        self,
        semantics: ObjectSemantics,
        grasp_xpos: torch.Tensor,
        start_qpos: torch.Tensor,
        object_poses: torch.Tensor,
        manipulator: ResolvedControlPart,
        options: PickUpOptions,
        approach_direction: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Choose a TCP-roll variant with a feasible pickup and transport path."""
        n_envs, n_pose = grasp_xpos.shape[:2]
        mirrored_grasp_xpos = grasp_xpos.clone()
        mirrored_grasp_xpos[..., :3, 0] = -mirrored_grasp_xpos[..., :3, 0]
        mirrored_grasp_xpos[..., :3, 1] = -mirrored_grasp_xpos[..., :3, 1]
        selection_variants = torch.stack([grasp_xpos, mirrored_grasp_xpos], dim=2)
        grasp_variants = self._upright_adjusted_grasp_poses(
            semantics, selection_variants, options
        )

        pre_grasp_variants = grasp_variants.clone()
        pre_grasp_z = pre_grasp_variants[..., :3, 2]
        pre_grasp_variants[..., :3, 3] -= pre_grasp_z * options.pre_grasp_distance
        lift_variants = grasp_variants.clone()
        lift_variants[..., :3, 3] += torch.tensor(
            [0.0, 0.0, options.lift_height],
            dtype=grasp_variants.dtype,
            device=self.device,
        )

        pre_grasp_success, pre_grasp_qpos = self._compute_batch_candidate_ik(
            pre_grasp_variants, start_qpos, manipulator
        )
        grasp_success, grasp_qpos = self._compute_batch_candidate_ik(
            grasp_variants, pre_grasp_qpos, manipulator
        )
        lift_success, lift_qpos = self._compute_batch_candidate_ik(
            lift_variants, grasp_qpos, manipulator
        )
        alignment_success = self._approach_alignment_mask(
            grasp_variants, options, approach_direction
        )
        pickup_success = (
            alignment_success & pre_grasp_success & grasp_success & lift_success
        )
        downstream_success_counts: list[list[int]] = []
        object_to_eef_variants = torch.matmul(
            pose_inv(object_poses)[:, None, None], grasp_variants
        )
        # MoveHeldObject begins after the lift, so screen its target from the
        # same joint state that the execution stream will use.
        downstream_seed = lift_qpos
        for object_target_pose in options.downstream_object_target_poses:
            object_target_pose = object_target_pose.to(
                device=self.device, dtype=torch.float32
            )
            if object_target_pose.shape == (4, 4):
                object_target_pose = object_target_pose.unsqueeze(0).repeat(
                    n_envs, 1, 1
                )
            if object_target_pose.shape != (n_envs, 4, 4):
                logger.log_error(
                    "downstream_object_target_poses entries must have shape "
                    f"(4, 4) or ({n_envs}, 4, 4), but got "
                    f"{object_target_pose.shape}.",
                    ValueError,
                )
            downstream_eef_variants = torch.matmul(
                object_target_pose[:, None, None], object_to_eef_variants
            )
            downstream_success, downstream_seed = self._compute_batch_candidate_ik(
                downstream_eef_variants, downstream_seed, manipulator
            )
            pickup_success &= downstream_success
            downstream_success_counts.append(pickup_success.sum(dim=(1, 2)).tolist())
        if not pickup_success.any(dim=(1, 2)).all():
            logger.log_warning(
                "PickUp found no candidate with a feasible vertical pickup path: "
                f"aligned={alignment_success.sum(dim=(1, 2)).tolist()}, "
                f"pre_grasp={pre_grasp_success.sum(dim=(1, 2)).tolist()}, "
                f"grasp={(pre_grasp_success & grasp_success).sum(dim=(1, 2)).tolist()}, "
                f"lift={(pre_grasp_success & grasp_success & lift_success).sum(dim=(1, 2)).tolist()}, "
                f"downstream={downstream_success_counts}."
            )

        start_xpos = self.robot.compute_fk(
            qpos=start_qpos,
            name=manipulator.name,
            to_matrix=True,
        )
        start_quat = quat_from_matrix(start_xpos[:, :3, :3])
        # Preserve the established preference between symmetric roll variants;
        # use the upright-adjusted pose only for feasibility and execution.
        variant_quat = quat_from_matrix(selection_variants[..., :3, :3])
        start_quat = start_quat[:, None, None, :].expand_as(variant_quat)
        rotation_error = quat_error_magnitude(
            variant_quat.reshape(-1, 4),
            start_quat.reshape(-1, 4),
        ).reshape(n_envs, n_pose, 2)
        feasible_rotation_error = torch.where(
            pickup_success,
            rotation_error,
            torch.full_like(rotation_error, torch.inf),
        )
        best_variant_idx = feasible_rotation_error.argmin(dim=2)

        env_idx = torch.arange(n_envs, device=self.device)[:, None]
        pose_idx = torch.arange(n_pose, device=self.device)[None, :]
        selected_grasp_xpos = grasp_variants[env_idx, pose_idx, best_variant_idx]
        ik_success = pickup_success[env_idx, pose_idx, best_variant_idx]
        return selected_grasp_xpos, ik_success

    def _approach_alignment_mask(
        self,
        grasp_poses: torch.Tensor,
        options: PickUpOptions,
        approach_direction: torch.Tensor,
    ) -> torch.Tensor:
        """Return candidates whose final TCP z-axis follows the approach direction."""
        max_angle = options.approach_alignment_max_angle
        if options.rotate_upright is not None or max_angle is None:
            return torch.ones(
                grasp_poses.shape[:3], dtype=torch.bool, device=grasp_poses.device
            )
        grasp_z = torch.nn.functional.normalize(grasp_poses[..., :3, 2], dim=-1)
        alignment = torch.sum(grasp_z * approach_direction, dim=-1)
        return alignment >= math.cos(float(max_angle))

    def _compute_batch_candidate_ik(
        self,
        poses: torch.Tensor,
        joint_seed: torch.Tensor,
        manipulator: ResolvedControlPart,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Solve candidate IK poses while preserving the candidate dimensions."""
        n_envs, n_pose, n_variant = poses.shape[:3]
        flat_poses = poses.reshape(n_envs, n_pose * n_variant, 4, 4)
        if joint_seed.dim() == 2:
            joint_seed = joint_seed[:, None, None, :].expand(-1, n_pose, n_variant, -1)
        flat_seed = joint_seed.reshape(n_envs, n_pose * n_variant, manipulator.dof)
        is_success, qpos = self.robot.compute_batch_ik(
            pose=flat_poses,
            name=manipulator.name,
            joint_seed=flat_seed,
        )
        return (
            is_success.reshape(n_envs, n_pose, n_variant),
            qpos.reshape(n_envs, n_pose, n_variant, manipulator.dof),
        )

    def _upright_adjusted_grasp_poses(
        self,
        semantics: ObjectSemantics,
        grasp_xpos: torch.Tensor,
        options: PickUpOptions,
    ) -> torch.Tensor:
        """Return grasp poses after the optional upright-in-place roll adjustment."""
        if options.rotate_upright is None:
            return grasp_xpos

        if options.obj_upright_direction is None:
            upright_direction = torch.tensor(
                [0, 0, 1], dtype=torch.float32, device=self.device
            )
        else:
            upright_direction = options.obj_upright_direction.to(
                device=self.device, dtype=torch.float32
            )
        obj_pose = semantics.entity.get_local_pose(to_matrix=True)
        obj_upright = torch.matmul(obj_pose[:, :3, :3], upright_direction)
        adjusted_grasp_xpos = grasp_xpos.clone()
        grasp_ry = adjusted_grasp_xpos[..., :3, 1]
        object_axes = obj_upright.reshape(
            obj_upright.shape[0], *([1] * (grasp_ry.ndim - 2)), 3
        )
        dot_result = (grasp_ry * object_axes).sum(dim=-1)
        revert_flag = torch.where(dot_result < 0, -1.0, 1.0)
        grasp_rx = adjusted_grasp_xpos[..., :3, 0]
        rota_axis_angle = options.rotate_upright * revert_flag[..., None] * grasp_rx
        rota_offset = axis_angle_to_rotation_matrix(
            rota_axis_angle.reshape(-1, 3)
        ).reshape(*rota_axis_angle.shape[:-1], 3, 3)
        adjusted_grasp_xpos[..., :3, :3] = torch.matmul(
            rota_offset, adjusted_grasp_xpos[..., :3, :3]
        )
        return adjusted_grasp_xpos


__all__ = ["GraspGoal", "PickUp", "PickUpOptions"]
