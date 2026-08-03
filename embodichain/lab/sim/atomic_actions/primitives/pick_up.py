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
from typing import ClassVar

import torch

from embodichain.lab.sim.planners import MoveType, PlanState
from embodichain.utils import configclass, logger
from embodichain.utils.math import (
    axis_angle_to_rotation_matrix,
    pose_inv,
    quat_error_magnitude,
    quat_from_matrix,
)

from ._helpers import arm_qpos_from_state, upright_yaw_pose_variants
from ..affordance import AntipodalAffordance
from ..core import (
    ActionCfg,
    ActionResult,
    AtomicAction,
    GraspTarget,
    HeldObjectState,
    ObjectSemantics,
    WorldState,
)
from ..trajectory import TrajectoryBuilder

_UPRIGHT_SIDE_GRASP_MAX_AXIS_ALIGNMENT = 0.65
_UPRIGHT_SIDE_GRASP_MIN_AXIS_FRACTION = 0.35
_UPRIGHT_SIDE_GRASP_MAX_AXIS_FRACTION = 0.75
_UPRIGHT_SIDE_GRASP_HEIGHT_COST_WEIGHT = 2.0


@configclass
class PickUpCfg(ActionCfg):
    name: str = "pick_up"
    """Name of the action, used for identification and logging."""

    sample_interval: int = 80
    """Number of waypoints for the full trajectory (approach + hand + lift)."""

    hand_interp_steps: int = 5
    """Number of waypoints for the gripper close interpolation phase."""

    hand_control_part: str = "hand"
    """Name of the robot part that controls the hand joints."""

    hand_open_qpos: torch.Tensor | None = None
    """Joint positions for the open hand state, shape ``[hand_dof,]``."""

    hand_close_qpos: torch.Tensor | None = None
    """Joint positions for the closed hand state, shape ``[hand_dof,]``."""

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

    upright_yaw_samples: int = 1
    """Equivalent world-yaw samples for downstream upright targets."""

    obj_upright_direction: torch.Tensor | None = None
    """Optional object local direction used to choose the upright grasp rotation."""

    rotate_upright: float | None = None
    """Optional rotation (radians) about the grasp x-axis to apply after grasp selection."""


class PickUp(AtomicAction):
    """Approach a grasp pose, close the gripper, lift."""

    TargetType: ClassVar[type] = GraspTarget

    def __init__(
        self,
        motion_generator,
        cfg: PickUpCfg | None = None,
    ) -> None:
        super().__init__(motion_generator, cfg or PickUpCfg())
        self.builder = TrajectoryBuilder(motion_generator)
        self.n_envs = self.robot.get_qpos().shape[0]
        self.arm_joint_ids = self.robot.get_joint_ids(name=self.cfg.control_part)
        self.hand_joint_ids = self.robot.get_joint_ids(name=self.cfg.hand_control_part)
        self.arm_dof = len(self.arm_joint_ids)
        self.robot_dof = self.robot.dof

        if self.cfg.hand_open_qpos is None:
            logger.log_error(
                "hand_open_qpos must be specified in PickUpCfg", ValueError
            )
        if self.cfg.hand_close_qpos is None:
            logger.log_error(
                "hand_close_qpos must be specified in PickUpCfg", ValueError
            )
        self.hand_open_qpos = self.cfg.hand_open_qpos.to(self.device)
        self.hand_close_qpos = self.cfg.hand_close_qpos.to(self.device)
        self.approach_direction = self.cfg.approach_direction.to(
            device=self.device, dtype=torch.float32
        )
        approach_norm = torch.linalg.vector_norm(self.approach_direction)
        if approach_norm <= 1.0e-6:
            logger.log_error("approach_direction must be non-zero.", ValueError)
        self.approach_direction = self.approach_direction / approach_norm
        if self.cfg.approach_alignment_max_angle is not None and not (
            0.0 <= self.cfg.approach_alignment_max_angle <= math.pi / 2
        ):
            logger.log_error(
                "approach_alignment_max_angle must be in [0, pi / 2].",
                ValueError,
            )
        if self.cfg.upright_yaw_samples < 1:
            logger.log_error("upright_yaw_samples must be positive.", ValueError)

    def execute(self, target: GraspTarget, state: WorldState) -> ActionResult:
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
            arm_qpos_from_state(state, self.arm_joint_ids),
            n_envs=self.n_envs,
            arm_dof=self.arm_dof,
            control_part=self.cfg.control_part,
        )
        if target.grasp_xpos is None:
            is_success, grasp_xpos = self._resolve_grasp_pose(sem, start_arm_qpos)
        else:
            grasp_xpos = self.builder.resolve_pose_target(
                target.grasp_xpos, n_envs=self.n_envs
            )
            if self.cfg.rotate_upright is not None:
                self._apply_upright_rotation(sem, grasp_xpos)
            is_success = torch.ones(self.n_envs, dtype=torch.bool, device=self.device)
        if not self.builder.all_envs_success(is_success):
            logger.log_warning("PickUp failed to resolve a grasp pose.")
            return self._fail(state)
        pre_grasp_xpos = self.builder.apply_local_offset(
            grasp_xpos, -self.approach_direction * self.cfg.pre_grasp_distance
        )

        n_approach, n_close, n_lift = self.builder.split_three_phase(
            self.cfg.sample_interval,
            self.cfg.hand_interp_steps,
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
            control_part=self.cfg.control_part,
            arm_dof=self.arm_dof,
            cfg=self.cfg,
        )

        grasp_arm_qpos = approach_arm[:, -1, :]
        lift_xpos = self.builder.apply_local_offset(
            grasp_xpos,
            torch.tensor([0, 0, 1], device=self.device) * self.cfg.lift_height,
        )
        target_states_list = [
            [PlanState(xpos=lift_xpos[i], move_type=MoveType.EEF_MOVE)]
            for i in range(self.n_envs)
        ]
        lift_success, lift_arm = self.builder.plan_arm_traj(
            target_states_list,
            grasp_arm_qpos,
            n_lift,
            control_part=self.cfg.control_part,
            arm_dof=self.arm_dof,
            cfg=self.cfg,
        )
        success = approach_success & lift_success

        hand_close_path = self.builder.interpolate_hand_qpos(
            self.hand_open_qpos, self.hand_close_qpos, n_waypoints=n_close
        )

        full = torch.empty(
            (self.n_envs, n_approach + n_close + n_lift, self.robot_dof),
            dtype=torch.float32,
            device=self.device,
        )
        full[:, :, :] = state.last_qpos.unsqueeze(1)
        full[:, :n_approach, self.arm_joint_ids] = approach_arm
        full[:, :n_approach, self.hand_joint_ids] = self.hand_open_qpos
        full[:, n_approach : n_approach + n_close, self.arm_joint_ids] = (
            grasp_arm_qpos.unsqueeze(1)
        )
        full[:, n_approach : n_approach + n_close, self.hand_joint_ids] = (
            hand_close_path
        )
        full[:, n_approach + n_close :, self.arm_joint_ids] = lift_arm
        full[:, n_approach + n_close :, self.hand_joint_ids] = self.hand_close_qpos

        obj_poses = sem.entity.get_local_pose(to_matrix=True)
        object_to_eef = torch.bmm(pose_inv(obj_poses), grasp_xpos)
        held = HeldObjectState(
            semantics=sem, object_to_eef=object_to_eef, grasp_xpos=grasp_xpos
        )
        return ActionResult(
            success=success,
            trajectory=full,
            next_state=WorldState(
                last_qpos=full[:, -1, :].clone(),
                held_object=held,
                coordinated_held_object=state.coordinated_held_object,
            ),
        )

    def _fail(self, state: WorldState) -> ActionResult:
        return ActionResult(
            success=torch.zeros(self.n_envs, dtype=torch.bool, device=self.device),
            trajectory=torch.empty(
                (self.n_envs, 0, self.robot_dof),
                dtype=torch.float32,
                device=self.device,
            ),
            next_state=state,
        )

    def _resolve_grasp_pose(
        self, semantics: ObjectSemantics, start_qpos: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        obj_poses = semantics.entity.get_local_pose(to_matrix=True)
        grasp_cost_fn = None
        if self.cfg.rotate_upright is not None:
            grasp_cost_fn = lambda object_pose, grasp_poses, costs: (
                self._upright_grasp_costs(
                    semantics,
                    object_pose,
                    grasp_poses,
                    costs,
                )
            )
        grasp_poses_result = semantics.affordance.get_valid_grasp_poses(
            obj_poses=obj_poses,
            approach_direction=self.approach_direction,
            grasp_cost_fn=grasp_cost_fn,
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
            semantics, grasp_xpos_padding, start_qpos, obj_poses
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Choose a TCP-roll variant with a feasible pickup and transport path."""
        n_envs, n_pose = grasp_xpos.shape[:2]
        mirrored_grasp_xpos = grasp_xpos.clone()
        mirrored_grasp_xpos[..., :3, 0] = -mirrored_grasp_xpos[..., :3, 0]
        mirrored_grasp_xpos[..., :3, 1] = -mirrored_grasp_xpos[..., :3, 1]
        selection_variants = torch.stack([grasp_xpos, mirrored_grasp_xpos], dim=2)
        grasp_variants = self._upright_adjusted_grasp_poses(
            semantics, selection_variants
        )
        upright_compatible = self._upright_grasp_compatibility_mask(
            semantics,
            grasp_variants,
            object_poses,
        )

        pre_grasp_variants = grasp_variants.clone()
        pre_grasp_variants[..., :3, 3] -= (
            self.approach_direction * self.cfg.pre_grasp_distance
        )
        lift_variants = grasp_variants.clone()
        lift_variants[..., :3, 3] += torch.tensor(
            [0.0, 0.0, self.cfg.lift_height],
            dtype=grasp_variants.dtype,
            device=self.device,
        )

        pre_grasp_success, pre_grasp_qpos = self._compute_batch_candidate_ik(
            pre_grasp_variants, start_qpos
        )
        grasp_success, grasp_qpos = self._compute_batch_candidate_ik(
            grasp_variants, pre_grasp_qpos
        )
        lift_success, lift_qpos = self._compute_batch_candidate_ik(
            lift_variants, grasp_qpos
        )
        alignment_success = self._approach_alignment_mask(grasp_variants)
        pickup_success = (
            upright_compatible
            & alignment_success
            & pre_grasp_success
            & grasp_success
            & lift_success
        )
        downstream_success_counts: list[list[int]] = []
        object_to_eef_variants = torch.matmul(
            pose_inv(object_poses)[:, None, None], grasp_variants
        )
        # MoveHeldObject begins after the lift, so screen its target from the
        # same joint state that the execution stream will use.
        downstream_seed = lift_qpos
        for object_target_pose in self.cfg.downstream_object_target_poses:
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
            object_target_variants = upright_yaw_pose_variants(
                object_target_pose,
                self.cfg.upright_yaw_samples,
            )
            downstream_success = torch.zeros_like(pickup_success)
            selected_qpos = downstream_seed
            for yaw_target in object_target_variants.unbind(dim=1):
                downstream_eef_variants = torch.matmul(
                    yaw_target[:, None, None], object_to_eef_variants
                )
                yaw_success, yaw_qpos = self._compute_batch_candidate_ik(
                    downstream_eef_variants,
                    downstream_seed,
                )
                newly_solved = ~downstream_success & yaw_success
                selected_qpos = torch.where(
                    newly_solved[..., None], yaw_qpos, selected_qpos
                )
                downstream_success |= yaw_success
                if bool((pickup_success & downstream_success).any(dim=(1, 2)).all()):
                    break
            downstream_seed = selected_qpos
            pickup_success &= downstream_success
            downstream_success_counts.append(pickup_success.sum(dim=(1, 2)).tolist())
        if not pickup_success.any(dim=(1, 2)).all():
            logger.log_warning(
                "PickUp found no candidate with a feasible vertical pickup path: "
                f"upright_compatible={upright_compatible.sum(dim=(1, 2)).tolist()}, "
                f"aligned={alignment_success.sum(dim=(1, 2)).tolist()}, "
                f"pre_grasp={pre_grasp_success.sum(dim=(1, 2)).tolist()}, "
                f"grasp={(pre_grasp_success & grasp_success).sum(dim=(1, 2)).tolist()}, "
                f"lift={(pre_grasp_success & grasp_success & lift_success).sum(dim=(1, 2)).tolist()}, "
                f"downstream={downstream_success_counts}."
            )

        start_xpos = self.robot.compute_fk(
            qpos=start_qpos,
            name=self.cfg.control_part,
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

    def _approach_alignment_mask(self, grasp_poses: torch.Tensor) -> torch.Tensor:
        """Return candidates whose final TCP z-axis follows the approach direction."""
        max_angle = self.cfg.approach_alignment_max_angle
        if self.cfg.rotate_upright is not None or max_angle is None:
            return torch.ones(
                grasp_poses.shape[:3], dtype=torch.bool, device=grasp_poses.device
            )
        grasp_z = torch.nn.functional.normalize(grasp_poses[..., :3, 2], dim=-1)
        alignment = torch.sum(grasp_z * self.approach_direction, dim=-1)
        return alignment >= math.cos(float(max_angle))

    def _compute_batch_candidate_ik(
        self, poses: torch.Tensor, joint_seed: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Solve candidate IK poses while preserving the candidate dimensions."""
        n_envs, n_pose, n_variant = poses.shape[:3]
        flat_poses = poses.reshape(n_envs, n_pose * n_variant, 4, 4)
        if joint_seed.dim() == 2:
            joint_seed = joint_seed[:, None, None, :].expand(-1, n_pose, n_variant, -1)
        flat_seed = joint_seed.reshape(n_envs, n_pose * n_variant, self.arm_dof)
        is_success, qpos = self.robot.compute_batch_ik(
            pose=flat_poses,
            name=self.cfg.control_part,
            joint_seed=flat_seed,
        )
        return (
            is_success.reshape(n_envs, n_pose, n_variant),
            qpos.reshape(n_envs, n_pose, n_variant, self.arm_dof),
        )

    def _apply_upright_rotation(
        self, semantics: ObjectSemantics, grasp_xpos: torch.Tensor
    ) -> None:
        """Apply the configured upright-in-place grasp roll adjustment."""
        grasp_xpos.copy_(self._upright_adjusted_grasp_poses(semantics, grasp_xpos))

    def _upright_adjusted_grasp_poses(
        self, semantics: ObjectSemantics, grasp_xpos: torch.Tensor
    ) -> torch.Tensor:
        """Return grasp poses after the optional upright-in-place roll adjustment."""
        if self.cfg.rotate_upright is None:
            return grasp_xpos

        upright_direction = self._normalized_obj_upright_direction()
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
        rota_axis_angle = self.cfg.rotate_upright * revert_flag[..., None] * grasp_rx
        rota_offset = axis_angle_to_rotation_matrix(
            rota_axis_angle.reshape(-1, 3)
        ).reshape(*rota_axis_angle.shape[:-1], 3, 3)
        adjusted_grasp_xpos[..., :3, :3] = torch.matmul(
            rota_offset, adjusted_grasp_xpos[..., :3, :3]
        )
        return adjusted_grasp_xpos

    def _upright_grasp_compatibility_mask(
        self,
        semantics: ObjectSemantics,
        grasp_xpos: torch.Tensor,
        object_poses: torch.Tensor,
    ) -> torch.Tensor:
        """Reject upright grasps that clamp the object's support and top faces."""
        del semantics
        shape = grasp_xpos.shape[:3]
        if self.cfg.rotate_upright is None:
            return torch.ones(shape, dtype=torch.bool, device=grasp_xpos.device)

        local_upright = self._normalized_obj_upright_direction().to(
            device=grasp_xpos.device,
            dtype=grasp_xpos.dtype,
        )
        object_poses = object_poses.to(
            device=grasp_xpos.device,
            dtype=grasp_xpos.dtype,
        )
        world_upright = torch.matmul(object_poses[:, :3, :3], local_upright)
        closing_axes = torch.nn.functional.normalize(
            grasp_xpos[..., :3, 0],
            dim=-1,
        )
        axis_alignment = torch.abs(
            torch.sum(closing_axes * world_upright[:, None, None, :], dim=-1)
        )
        return axis_alignment <= _UPRIGHT_SIDE_GRASP_MAX_AXIS_ALIGNMENT

    def _upright_grasp_costs(
        self,
        semantics: ObjectSemantics,
        object_pose: torch.Tensor,
        grasp_poses: torch.Tensor,
        costs: torch.Tensor,
    ) -> torch.Tensor:
        """Rank side grasps before generator top-k truncation."""
        local_upright = self._normalized_obj_upright_direction().to(
            device=grasp_poses.device,
            dtype=grasp_poses.dtype,
        )
        object_pose = object_pose.to(
            device=grasp_poses.device,
            dtype=grasp_poses.dtype,
        )
        world_upright = torch.matmul(object_pose[:3, :3], local_upright)
        closing_axes = torch.nn.functional.normalize(
            grasp_poses[:, :3, 0],
            dim=-1,
        )
        axis_alignment = torch.abs(
            torch.sum(closing_axes * world_upright[None, :], dim=-1)
        )
        adjusted = torch.where(
            axis_alignment <= _UPRIGHT_SIDE_GRASP_MAX_AXIS_ALIGNMENT,
            costs,
            torch.full_like(costs, torch.inf),
        )

        vertices = semantics.geometry.get("mesh_vertices")
        if vertices is None:
            return adjusted
        vertices = torch.as_tensor(
            vertices,
            dtype=grasp_poses.dtype,
            device=grasp_poses.device,
        )
        if vertices.ndim != 2 or vertices.shape[-1] != 3 or vertices.numel() == 0:
            return adjusted

        vertex_axis_positions = torch.matmul(vertices, local_upright)
        axis_min = vertex_axis_positions.min()
        axis_extent = vertex_axis_positions.max() - axis_min
        if float(axis_extent) <= 1.0e-6:
            return adjusted

        relative_centers = grasp_poses[:, :3, 3] - object_pose[None, :3, 3]
        center_axis_positions = torch.sum(
            relative_centers * world_upright[None, :],
            dim=-1,
        )
        center_fractions = (center_axis_positions - axis_min) / axis_extent
        interval = (
            _UPRIGHT_SIDE_GRASP_MAX_AXIS_FRACTION
            - _UPRIGHT_SIDE_GRASP_MIN_AXIS_FRACTION
        )
        height_penalty = (
            torch.clamp(
                _UPRIGHT_SIDE_GRASP_MIN_AXIS_FRACTION - center_fractions,
                min=0.0,
            )
            + torch.clamp(
                center_fractions - _UPRIGHT_SIDE_GRASP_MAX_AXIS_FRACTION,
                min=0.0,
            )
        ) / interval
        return adjusted + _UPRIGHT_SIDE_GRASP_HEIGHT_COST_WEIGHT * height_penalty

    def _normalized_obj_upright_direction(self) -> torch.Tensor:
        direction = self.cfg.obj_upright_direction
        if direction is None:
            direction = torch.tensor([0, 0, 1], dtype=torch.float32)
        direction = direction.to(device=self.device, dtype=torch.float32)
        norm = torch.linalg.vector_norm(direction)
        if norm <= 1.0e-6:
            logger.log_error("obj_upright_direction must be non-zero.", ValueError)
        return direction / norm


__all__ = ["PickUp", "PickUpCfg"]
