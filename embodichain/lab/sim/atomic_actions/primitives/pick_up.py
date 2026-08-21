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
from dataclasses import dataclass, replace
from typing import ClassVar

import torch

from embodichain.utils import logger
from embodichain.utils.math import (
    axis_angle_to_rotation_matrix,
    pose_inv,
    quat_error_magnitude,
    quat_from_matrix,
)

from ._helpers import arm_qpos_from_state, require_shared_task_state_key
from ..affordance import AntipodalAffordance
from ..bindings import JointPositionTarget
from ..control import GRASP_COMMAND, OPEN_COMMAND, JointPositionCommand
from ..core import AtomicAction, ObjectSemantics
from ..effects import StateDelta
from ..goals import (
    ObjectActionGoal,
    PoseGoalValue,
    _resolve_object_pose,
    collect_scene_dependencies,
    resolve_pose_goal,
    validate_pose_goal,
)
from embodichain.lab.sim.atomic_actions.invocation import (
    ActionOptions,
    ResolvedActionRequest,
)
from embodichain.lab.sim.atomic_actions.plans import (
    ActionPlan,
    TimedTrajectory,
    normalize_success_mask,
)
from embodichain.lab.sim.atomic_actions.policies import MotionPolicy
from embodichain.lab.sim.atomic_actions.requirements import (
    BATCH_INVERSE_KINEMATICS_CAPABILITY,
    CARTESIAN_POSE_CAPABILITY,
    FORWARD_KINEMATICS_CAPABILITY,
    SkillBindingContract,
)
from embodichain.lab.sim.atomic_actions.state import HeldObjectState, PlanningContext
from embodichain.lab.sim.atomic_actions.trajectory_ops import (
    build_pose_plan_states,
    interpolate_hand_qpos,
    resolve_pose_target,
    split_three_segments,
    translate_pose_world,
)
from embodichain.lab.sim.atomic_actions.primitives._binding_contracts import (
    make_manipulation_slot,
)

_UPRIGHT_SIDE_GRASP_MAX_AXIS_ALIGNMENT = 0.65
_UPRIGHT_SIDE_GRASP_MIN_AXIS_FRACTION = 0.35
_UPRIGHT_SIDE_GRASP_MAX_AXIS_FRACTION = 0.75
_UPRIGHT_SIDE_GRASP_HEIGHT_COST_WEIGHT = 2.0


def _upright_yaw_pose_variants(
    target_pose: torch.Tensor,
    sample_count: int,
) -> torch.Tensor:
    """Return object poses with evenly sampled world-Z yaw rotations."""
    signed_steps = [0]
    for step in range(1, (sample_count + 1) // 2):
        signed_steps.extend((step, -step))
    if sample_count % 2 == 0:
        signed_steps.append(sample_count // 2)
    angles = target_pose.new_tensor(signed_steps) * (2.0 * math.pi / sample_count)
    yaw = target_pose.new_zeros((sample_count, 3, 3))
    yaw[:, 0, 0] = torch.cos(angles)
    yaw[:, 0, 1] = -torch.sin(angles)
    yaw[:, 1, 0] = torch.sin(angles)
    yaw[:, 1, 1] = torch.cos(angles)
    yaw[:, 2, 2] = 1.0
    variants = target_pose[:, None].repeat(1, sample_count, 1, 1)
    variants[:, :, :3, :3] = torch.matmul(yaw[None], target_pose[:, None, :3, :3])
    return variants


@dataclass(frozen=True, slots=True, eq=False)
class GraspGoal(ObjectActionGoal):
    """Pickup target with an affordance-selected or supplied grasp pose."""

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
    """Number of waypoints for the gripper-close interpolation segment."""

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

    downstream_object_target_poses: tuple[PoseGoalValue, ...] = ()
    """Future object poses that must be reachable with the selected grasp."""

    upright_yaw_samples: int = 1
    """Equivalent world-yaw samples for semantically upright downstream targets."""

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
        if self.upright_yaw_samples < 1:
            raise ValueError("upright_yaw_samples must be positive.")
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
        downstream_targets: list[PoseGoalValue] = []
        for index, value in enumerate(self.downstream_object_target_poses):
            validate_pose_goal(
                value,
                f"downstream_object_target_poses[{index}]",
                allow_waypoints=False,
            )
            downstream_targets.append(
                value.clone() if isinstance(value, torch.Tensor) else value.snapshot()
            )
        object.__setattr__(
            self,
            "downstream_object_target_poses",
            tuple(downstream_targets),
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
    binding_contract: ClassVar[SkillBindingContract] = SkillBindingContract(
        slots=(
            make_manipulation_slot(
                "primary",
                motion_capabilities=frozenset(
                    {
                        BATCH_INVERSE_KINEMATICS_CAPABILITY,
                        CARTESIAN_POSE_CAPABILITY,
                        FORWARD_KINEMATICS_CAPABILITY,
                    }
                ),
                grasp_commands={
                    OPEN_COMMAND: JointPositionCommand,
                    GRASP_COMMAND: JointPositionCommand,
                },
            ),
        ),
    )

    def _scene_dependencies(
        self,
        request: ResolvedActionRequest[GraspGoal, PickUpOptions],
    ) -> tuple[str, ...]:
        """Include the semantic object when it has a stable scene identity."""
        dependencies = set(super()._scene_dependencies(request))
        entity_id = request.goal.semantics.entity_id
        if entity_id is not None:
            dependencies.add(entity_id)
        dependencies.update(
            collect_scene_dependencies(
                request.skill_options.downstream_object_target_poses
            )
        )
        return tuple(sorted(dependencies))

    def _get_full_pickup_trajectory(
        self,
        grasp_xpos: torch.Tensor,
        start_arm_qpos: torch.Tensor,
        last_qpos: torch.Tensor,
        motion_policy: MotionPolicy,
        options: PickUpOptions,
        approach_direction: torch.Tensor,
        manipulator: JointPositionTarget,
        end_effector: JointPositionTarget,
        hand_open_qpos: torch.Tensor,
        hand_grasp_qpos: torch.Tensor,
        interpolation_dt: float,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
        pre_grasp_xpos = translate_pose_world(
            grasp_xpos, -approach_direction * options.pre_grasp_distance
        )

        n_approach, n_close, n_lift = split_three_segments(
            motion_policy.sample_count,
            options.hand_interp_steps,
            first_segment_name="approach",
            third_segment_name="lift",
        )

        approach_result = self.motion_generator.generate(
            build_pose_plan_states(torch.stack([pre_grasp_xpos, grasp_xpos], dim=1)),
            options=motion_policy.to_motion_gen_options(
                start_qpos=start_arm_qpos,
                control_part=manipulator.control_part,
                sample_count=n_approach,
                interpolation_dt=interpolation_dt,
            ),
        )
        assert isinstance(approach_result.success, torch.Tensor)
        assert approach_result.positions is not None
        approach_success = approach_result.success
        approach_arm = approach_result.positions

        grasp_arm_qpos = approach_arm[:, -1, :]
        lift_xpos = translate_pose_world(
            grasp_xpos,
            torch.tensor([0, 0, 1], device=self.device) * options.lift_height,
        )
        lift_result = self.motion_generator.generate(
            build_pose_plan_states(lift_xpos),
            options=motion_policy.to_motion_gen_options(
                start_qpos=grasp_arm_qpos,
                control_part=manipulator.control_part,
                sample_count=n_lift,
                interpolation_dt=interpolation_dt,
            ),
        )
        assert isinstance(lift_result.success, torch.Tensor)
        assert lift_result.positions is not None
        lift_success = lift_result.success
        lift_arm = lift_result.positions
        is_success = approach_success & lift_success

        hand_close_path = interpolate_hand_qpos(
            hand_open_qpos, hand_grasp_qpos, n_waypoints=n_close
        )
        n_approach_actual = approach_arm.shape[1]
        n_lift_actual = lift_arm.shape[1]
        full = torch.empty(
            (
                self.num_envs,
                n_approach_actual + n_close + n_lift_actual,
                self.robot_dof,
            ),
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
        return (
            is_success,
            full,
            {
                "approach": n_approach_actual,
                "close": n_close,
                "lift": n_lift_actual,
            },
        )

    def _plan(
        self,
        request: ResolvedActionRequest[GraspGoal, PickUpOptions],
        context: PlanningContext,
    ) -> ActionPlan:
        """Plan approach, close, and lift segments without committing attachment."""
        target = self.require_goal(request)
        options = replace(
            request.skill_options,
            downstream_object_target_poses=tuple(
                resolve_pose_goal(
                    downstream_target,
                    context,
                    name=f"downstream_object_target_poses[{index}]",
                )
                for index, downstream_target in enumerate(
                    request.skill_options.downstream_object_target_poses
                )
            ),
        )
        approach_direction = options.approach_direction.to(
            device=self.device, dtype=torch.float32
        )
        approach_direction = approach_direction / torch.linalg.vector_norm(
            approach_direction
        )
        binding = request.binding
        motion = binding.endpoint("primary", "motion")
        grasp = binding.endpoint("primary", "grasp")
        manipulator = motion.require_target(JointPositionTarget)
        end_effector = grasp.require_target(JointPositionTarget)
        task_state_key = require_shared_task_state_key(
            motion,
            grasp,
            participant="PickUp primary participant",
        )
        hand_open_qpos = grasp.joint_positions(
            OPEN_COMMAND,
            num_envs=context.batch_size,
            device=self.device,
            dtype=context.robot.qpos.dtype,
        )
        hand_grasp_qpos = grasp.joint_positions(
            GRASP_COMMAND,
            num_envs=context.batch_size,
            device=self.device,
            dtype=context.robot.qpos.dtype,
        )
        control_part = manipulator.control_part
        state = context
        sem = target.semantics
        object_pose = _resolve_object_pose(
            sem,
            context,
            name="pickup_object_pose",
        )
        if target.grasp_xpos is None and not isinstance(
            sem.affordance, AntipodalAffordance
        ):
            raise ValueError(
                "PickUp requires an AntipodalAffordance when grasp_xpos is not set."
            )
        start_arm_qpos = arm_qpos_from_state(
            state,
            list(manipulator.joint_ids),
        )
        if target.grasp_xpos is None:
            is_success, grasp_xpos = self._resolve_grasp_pose(
                sem,
                object_pose,
                start_arm_qpos,
                manipulator,
                options,
                approach_direction,
            )
        else:
            grasp_xpos = resolve_pose_target(
                resolve_pose_goal(target.grasp_xpos, context, name="grasp_xpos"),
                num_envs=self.num_envs,
                device=self.device,
            )
            if options.rotate_upright is not None:
                grasp_xpos = self._upright_adjusted_grasp_poses(
                    grasp_xpos,
                    object_pose,
                    options,
                )
            is_success = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        grasp_success = normalize_success_mask(
            is_success,
            num_envs=self.num_envs,
            device=self.device,
            name="Grasp-pose success",
        )
        if not grasp_success.any():
            logger.log_warning("PickUp failed to resolve a grasp pose.")
            return self.failed_plan(
                request, context, message="Failed to resolve a grasp pose."
            )

        trajectory_success, full, segment_lengths = self._get_full_pickup_trajectory(
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
            context.require_control_dt(),
        )
        success_mask = grasp_success & normalize_success_mask(
            trajectory_success,
            num_envs=self.num_envs,
            device=self.device,
            name="Pick-up trajectory success",
        )

        object_to_eef = torch.bmm(pose_inv(object_pose), grasp_xpos)
        held = HeldObjectState(
            semantics=sem, object_to_eef=object_to_eef, grasp_xpos=grasp_xpos
        )
        coordinated_updates = {
            key: None
            for key in state.task.coordinated_held_objects
            if task_state_key in key
        }
        return self.build_plan(
            request,
            context,
            success=success_mask,
            trajectory=TimedTrajectory.from_uniform_step(
                full,
                env_ids=context.env_ids,
                step_dt=context.require_control_dt(),
            ),
            expected_effects=StateDelta(
                held_object_updates={task_state_key: held},
                coordinated_held_object_updates=coordinated_updates,
            ),
            segment_lengths=segment_lengths,
            scene_dependency_monitor_until=(
                {}
                if sem.entity_id is None
                else {sem.entity_id: segment_lengths["approach"]}
            ),
        )

    def _resolve_grasp_pose(
        self,
        semantics: ObjectSemantics,
        object_pose: torch.Tensor,
        start_qpos: torch.Tensor,
        manipulator: JointPositionTarget,
        options: PickUpOptions,
        approach_direction: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        grasp_cost_fn = None
        if options.rotate_upright is not None:
            grasp_cost_fn = lambda object_pose, grasp_poses, costs: (
                self._upright_grasp_costs(
                    semantics,
                    object_pose,
                    grasp_poses,
                    costs,
                    options,
                )
            )
        grasp_poses_result = semantics.affordance.get_valid_grasp_poses(
            obj_poses=object_pose,
            approach_direction=approach_direction,
            object_part=options.pick_object_part,
            grasp_cost_fn=grasp_cost_fn,
        )
        num_envs = object_pose.shape[0]
        n_max_pose = max(r[0].shape[0] for r in grasp_poses_result)
        grasp_xpos_padding = torch.zeros(
            (num_envs, n_max_pose, 4, 4), dtype=torch.float32, device=self.device
        )
        grasp_cost_padding = torch.full(
            (num_envs, n_max_pose),
            float("inf"),
            dtype=torch.float32,
            device=self.device,
        )
        for i in range(num_envs):
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
            grasp_xpos_padding,
            start_qpos,
            object_pose,
            manipulator,
            options,
            approach_direction,
        )
        grasp_cost_masked = torch.where(ik_success, grasp_cost_padding, 10000.0)
        best_cost, best_idx = grasp_cost_masked.min(dim=1)
        is_success = best_cost < 9999.0
        best_grasp_xpos = grasp_xpos_padding[
            torch.arange(num_envs, device=self.device), best_idx
        ]
        return is_success, best_grasp_xpos

    def _select_feasible_grasp_variants(
        self,
        grasp_xpos: torch.Tensor,
        start_qpos: torch.Tensor,
        object_poses: torch.Tensor,
        manipulator: JointPositionTarget,
        options: PickUpOptions,
        approach_direction: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Choose a TCP-roll variant with a feasible pickup and transport path."""
        num_envs, n_pose = grasp_xpos.shape[:2]
        mirrored_grasp_xpos = grasp_xpos.clone()
        mirrored_grasp_xpos[..., :3, 0] = -mirrored_grasp_xpos[..., :3, 0]
        mirrored_grasp_xpos[..., :3, 1] = -mirrored_grasp_xpos[..., :3, 1]
        selection_variants = torch.stack([grasp_xpos, mirrored_grasp_xpos], dim=2)
        grasp_variants = self._upright_adjusted_grasp_poses(
            selection_variants,
            object_poses,
            options,
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
        upright_compatible = self._upright_grasp_compatibility_mask(
            grasp_variants,
            object_poses,
            options,
        )
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
        for object_target_pose in options.downstream_object_target_poses:
            object_target_pose = object_target_pose.to(
                device=self.device, dtype=torch.float32
            )
            if object_target_pose.shape == (4, 4):
                object_target_pose = object_target_pose.unsqueeze(0).repeat(
                    num_envs, 1, 1
                )
            if object_target_pose.shape != (num_envs, 4, 4):
                raise ValueError(
                    "downstream_object_target_poses entries must have shape "
                    f"(4, 4) or ({num_envs}, 4, 4), but got "
                    f"{object_target_pose.shape}."
                )
            object_target_variants = _upright_yaw_pose_variants(
                object_target_pose,
                options.upright_yaw_samples,
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
                    manipulator,
                )
                newly_solved = ~downstream_success & yaw_success
                selected_qpos = torch.where(
                    newly_solved[..., None],
                    yaw_qpos,
                    selected_qpos,
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
            name=manipulator.control_part,
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
        ).reshape(num_envs, n_pose, 2)
        feasible_rotation_error = torch.where(
            pickup_success,
            rotation_error,
            torch.full_like(rotation_error, torch.inf),
        )
        best_variant_idx = feasible_rotation_error.argmin(dim=2)

        env_idx = torch.arange(num_envs, device=self.device)[:, None]
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
        manipulator: JointPositionTarget,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Solve candidate IK poses while preserving the candidate dimensions."""
        num_envs, n_pose, n_variant = poses.shape[:3]
        flat_poses = poses.reshape(num_envs, n_pose * n_variant, 4, 4)
        if joint_seed.dim() == 2:
            joint_seed = joint_seed[:, None, None, :].expand(-1, n_pose, n_variant, -1)
        manipulator_dof = len(manipulator.joint_ids)
        flat_seed = joint_seed.reshape(num_envs, n_pose * n_variant, manipulator_dof)
        is_success, qpos = self.robot.compute_batch_ik(
            pose=flat_poses,
            name=manipulator.control_part,
            joint_seed=flat_seed,
        )
        return (
            is_success.reshape(num_envs, n_pose, n_variant),
            qpos.reshape(num_envs, n_pose, n_variant, manipulator_dof),
        )

    def _upright_grasp_compatibility_mask(
        self,
        grasp_xpos: torch.Tensor,
        object_poses: torch.Tensor,
        options: PickUpOptions,
    ) -> torch.Tensor:
        """Reject upright grasps that clamp the object's support and top faces."""
        shape = grasp_xpos.shape[:3]
        if options.rotate_upright is None:
            return torch.ones(shape, dtype=torch.bool, device=grasp_xpos.device)
        local_upright = self._normalized_obj_upright_direction(options).to(
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
        options: PickUpOptions,
    ) -> torch.Tensor:
        """Rank side grasps before generator top-k truncation."""
        local_upright = self._normalized_obj_upright_direction(options).to(
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

    def _normalized_obj_upright_direction(
        self,
        options: PickUpOptions,
    ) -> torch.Tensor:
        direction = options.obj_upright_direction
        if direction is None:
            direction = torch.tensor([0, 0, 1], dtype=torch.float32)
        direction = direction.to(device=self.device, dtype=torch.float32)
        norm = torch.linalg.vector_norm(direction)
        if norm <= 1.0e-6:
            logger.log_error("obj_upright_direction must be non-zero.", ValueError)
        return direction / norm

    def _upright_adjusted_grasp_poses(
        self,
        grasp_xpos: torch.Tensor,
        object_pose: torch.Tensor,
        options: PickUpOptions,
    ) -> torch.Tensor:
        """Return grasp poses after the optional upright-in-place roll adjustment."""
        if options.rotate_upright is None:
            return grasp_xpos

        upright_direction = self._normalized_obj_upright_direction(options).to(
            device=grasp_xpos.device,
            dtype=grasp_xpos.dtype,
        )
        object_pose = object_pose.to(
            device=grasp_xpos.device,
            dtype=grasp_xpos.dtype,
        )
        obj_upright = torch.matmul(object_pose[:, :3, :3], upright_direction)
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
