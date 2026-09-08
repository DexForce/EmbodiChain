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

from embodichain.lab.sim.atomic_actions.primitives._helpers import (
    arm_qpos_from_state,
    require_shared_task_state_key,
    split_joint_trajectory_at_pose,
)
from embodichain.lab.sim.atomic_actions.affordance import AntipodalAffordance
from embodichain.lab.sim.atomic_actions.bindings import JointPositionTarget
from embodichain.lab.sim.atomic_actions.control import (
    GRASP_COMMAND,
    OPEN_COMMAND,
    JointPositionCommand,
)
from embodichain.lab.sim.atomic_actions.core import AtomicAction, ObjectSemantics
from embodichain.lab.sim.atomic_actions.effects import StateDelta
from embodichain.lab.sim.atomic_actions.goals import (
    ObjectActionGoal,
    PoseGoalValue,
    _resolve_object_pose,
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


@dataclass(frozen=True, slots=True, eq=False)
class GraspGoal(ObjectActionGoal):
    """Pickup target with an affordance-selected or supplied grasp pose."""

    grasp_xpos: PoseGoalValue | None = None
    """Optional end-effector grasp pose.

    When omitted, :class:`PickUp` uses the configured fixed object-relative
    grasp when available, otherwise it selects one from the target affordance.
    An explicit tensor or late-bound
    :class:`~embodichain.lab.sim.atomic_actions.goals.SceneEntityPose` skips
    grasp sampling. Late-bound poses also declare the scene dependency used by
    closed-loop execution recovery.
    """

    def __post_init__(self) -> None:
        ObjectActionGoal.__post_init__(self)
        if self.grasp_xpos is not None:
            validate_pose_goal(self.grasp_xpos, "grasp_xpos", allow_waypoints=False)


def _validate_single_se3(value: torch.Tensor, name: str) -> None:
    """Validate one finite, proper SE(3) transform."""
    validate_pose_goal(value, name, allow_waypoints=False)
    if value.shape != (4, 4) or not torch.isfinite(value).all():
        raise ValueError(f"{name} must be one finite 4x4 transform.")
    transform = value.to(dtype=torch.float64)
    if not torch.allclose(
        transform[3],
        transform.new_tensor((0.0, 0.0, 0.0, 1.0)),
        atol=1.0e-6,
        rtol=0.0,
    ):
        raise ValueError(f"{name} must have bottom row [0, 0, 0, 1].")
    rotation = transform[:3, :3]
    if not torch.allclose(
        rotation.T @ rotation,
        torch.eye(3, dtype=transform.dtype, device=transform.device),
        atol=1.0e-6,
        rtol=0.0,
    ) or not torch.isclose(
        torch.linalg.det(rotation),
        transform.new_tensor(1.0),
        atol=1.0e-6,
        rtol=0.0,
    ):
        raise ValueError(f"{name} must contain a proper SE(3) rotation.")


@dataclass(frozen=True, slots=True, eq=False)
class PickUpOptions(ActionOptions):
    """Per-invocation pickup behavior."""

    hand_interp_steps: int = 5
    """Number of waypoints for the gripper-close interpolation segment."""

    grasp_settle_steps: int = 0
    """Fully closed hold frames before lifting the end-effector."""

    pick_object_part: str = "center"
    """Name of the object part to pick up (used for grasp pose generation). Currently support [center | top | bottom]."""

    lift_height: float = 0.1
    """Height (m) to lift the end-effector after closing the gripper."""

    pre_grasp_distance: float = 0.15
    """Distance to offset back from the grasp pose along the approach direction."""

    grasp_commit_fraction: float = 1.0
    """Approach fraction after which contact motion no longer invalidates grasp."""

    approach_direction: torch.Tensor = torch.tensor([0, 0, -1], dtype=torch.float32)
    """World-frame direction from the pre-grasp pose to the grasp pose."""

    approach_alignment_max_angle: float | None = None
    """Optional maximum TCP z-axis deviation from the approach direction."""

    downstream_object_target_poses: tuple[PoseGoalValue, ...] = ()
    """Future object poses that must be reachable with the selected grasp."""

    obj_upright_direction: torch.Tensor | None = None
    """Optional object local direction used to choose the upright grasp rotation."""

    rotate_upright: float | None = None
    """Optional rotation (radians) about the grasp x-axis to apply after grasp selection."""

    grasp_frame_to_eef: torch.Tensor = torch.eye(4, dtype=torch.float32)
    """Canonical grasp-frame to robot end-effector SE(3) calibration."""

    fixed_object_to_eef: torch.Tensor | None = None
    """Optional object-frame to end-effector SE(3) grasp calibration.

    When no explicit goal grasp is supplied, this transform bypasses affordance
    sampling and the sampled-grasp orientation/calibration adjustments.
    """

    def __post_init__(self) -> None:
        if self.hand_interp_steps < 1:
            raise ValueError("hand_interp_steps must be at least 1.")
        if type(self.grasp_settle_steps) is not int or self.grasp_settle_steps < 0:
            raise ValueError("grasp_settle_steps must be a non-negative integer.")
        if not isinstance(self.pick_object_part, str) or not self.pick_object_part:
            raise ValueError("pick_object_part must be a non-empty string.")
        if self.lift_height < 0.0:
            raise ValueError("lift_height must be non-negative.")
        if self.pre_grasp_distance < 0.0:
            raise ValueError("pre_grasp_distance must be non-negative.")
        if isinstance(self.grasp_commit_fraction, bool) or not isinstance(
            self.grasp_commit_fraction, (int, float)
        ):
            raise TypeError("grasp_commit_fraction must be a real number.")
        if not 0.0 < self.grasp_commit_fraction <= 1.0:
            raise ValueError("grasp_commit_fraction must be in (0, 1].")
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
        _validate_single_se3(self.grasp_frame_to_eef, "grasp_frame_to_eef")
        if self.fixed_object_to_eef is not None:
            _validate_single_se3(self.fixed_object_to_eef, "fixed_object_to_eef")
        object.__setattr__(self, "approach_direction", self.approach_direction.clone())
        object.__setattr__(
            self,
            "grasp_frame_to_eef",
            self.grasp_frame_to_eef.clone(),
        )
        if self.fixed_object_to_eef is not None:
            object.__setattr__(
                self,
                "fixed_object_to_eef",
                self.fixed_object_to_eef.clone(),
            )
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
        """Monitor only the object that the current invocation must acquire.

        Downstream object targets guide grasp selection for static look-ahead,
        but they are grounded again at the next Semantic Call boundary.  A
        moving downstream destination must therefore not invalidate an active
        pickup after a feasible grasp has already been selected.
        """
        dependencies = set(super()._scene_dependencies(request))
        entity_id = request.goal.semantics.entity_id
        if entity_id is not None:
            dependencies.add(entity_id)
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

        lift_xpos = translate_pose_world(
            grasp_xpos,
            torch.tensor([0, 0, 1], device=self.device) * options.lift_height,
        )
        motion_options = motion_policy.to_motion_gen_options(
            start_qpos=start_arm_qpos,
            control_part=manipulator.control_part,
            sample_count=n_approach + n_lift,
            interpolation_dt=interpolation_dt,
        )
        if motion_policy.strategy == "motion_gen":
            motion_options.sample_count = None
        motion_result = self.motion_generator.generate(
            build_pose_plan_states(
                torch.stack([pre_grasp_xpos, grasp_xpos, lift_xpos], dim=1)
            ),
            options=motion_options,
        )
        assert isinstance(motion_result.success, torch.Tensor)
        assert motion_result.positions is not None
        approach_arm, lift_arm = split_joint_trajectory_at_pose(
            motion_result.positions,
            grasp_xpos,
            robot=self.robot,
            control_part=manipulator.control_part,
            first_sample_count=n_approach,
            second_sample_count=n_lift,
        )
        grasp_arm_qpos = approach_arm[:, -1, :]
        is_success = motion_result.success

        hand_close_path = interpolate_hand_qpos(
            hand_open_qpos, hand_grasp_qpos, n_waypoints=n_close
        )
        n_settle = options.grasp_settle_steps
        close_start = n_approach
        settle_start = close_start + n_close
        lift_start = settle_start + n_settle
        full = torch.empty(
            (
                self.num_envs,
                lift_start + n_lift,
                self.robot_dof,
            ),
            dtype=torch.float32,
            device=self.device,
        )
        full[:, :, :] = last_qpos.unsqueeze(1)
        arm_joint_ids = list(manipulator.joint_ids)
        hand_joint_ids = list(end_effector.joint_ids)
        full[:, :n_approach, arm_joint_ids] = approach_arm
        full[:, :n_approach, hand_joint_ids] = hand_open_qpos.unsqueeze(1)
        full[:, close_start:settle_start, arm_joint_ids] = grasp_arm_qpos.unsqueeze(1)
        full[:, close_start:settle_start, hand_joint_ids] = hand_close_path
        if n_settle:
            full[:, settle_start:lift_start, arm_joint_ids] = grasp_arm_qpos.unsqueeze(
                1
            )
            full[:, settle_start:lift_start, hand_joint_ids] = (
                hand_grasp_qpos.unsqueeze(1)
            )
        full[:, lift_start:, arm_joint_ids] = lift_arm
        full[:, lift_start:, hand_joint_ids] = hand_grasp_qpos.unsqueeze(1)
        return (
            is_success,
            full,
            {
                "approach": n_approach,
                "close": n_close + n_settle,
                "lift": n_lift,
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
        state = context
        sem = target.semantics
        object_pose = _resolve_object_pose(
            sem,
            context,
            name="pickup_object_pose",
        )
        if (
            target.grasp_xpos is None
            and options.fixed_object_to_eef is None
            and not isinstance(sem.affordance, AntipodalAffordance)
        ):
            raise ValueError(
                "PickUp requires an AntipodalAffordance when neither grasp_xpos "
                "nor fixed_object_to_eef is set."
            )
        start_arm_qpos = arm_qpos_from_state(
            state,
            list(manipulator.joint_ids),
        )
        if target.grasp_xpos is None:
            if options.fixed_object_to_eef is None:
                is_success, grasp_xpos = self._resolve_grasp_pose(
                    sem,
                    object_pose,
                    start_arm_qpos,
                    manipulator,
                    end_effector.target_id,
                    options,
                    approach_direction,
                )
            else:
                object_to_eef = options.fixed_object_to_eef.to(
                    device=self.device,
                    dtype=object_pose.dtype,
                )
                grasp_xpos = torch.matmul(object_pose, object_to_eef)
                is_success = torch.ones(
                    self.num_envs,
                    dtype=torch.bool,
                    device=self.device,
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
            key: None for key in state.coordinated_held_objects if task_state_key in key
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
            # Once the approach is dispatched, contact can move the object and
            # the selected downstream suffix can be grounded again after this
            # semantic boundary.  Neither expected pickup motion nor a later
            # destination revision should invalidate an already acquired
            # grasp during close/lift.
            scene_dependency_monitor_until={
                entity_id: max(
                    1,
                    math.ceil(
                        segment_lengths["approach"] * options.grasp_commit_fraction
                    ),
                )
                for entity_id in self._scene_dependencies(request)
            },
        )

    def _resolve_grasp_pose(
        self,
        semantics: ObjectSemantics,
        object_pose: torch.Tensor,
        start_qpos: torch.Tensor,
        manipulator: JointPositionTarget,
        grasp_target_id: str,
        options: PickUpOptions,
        approach_direction: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        affordance = semantics.affordance
        if not isinstance(affordance, AntipodalAffordance):
            raise ValueError("PickUp grasp sampling requires AntipodalAffordance.")
        generator = self.planning_services.grasp_pose_generator(grasp_target_id)
        obj_longest_axis = None
        is_positive_part = True
        if options.pick_object_part != "center":
            obj_longest_axis = torch.tensor(
                [0.0, 0.0, 1.0], dtype=torch.float32, device=self.device
            )
            is_positive_part = options.pick_object_part == "top"
        grasp_poses_result = generator.get_valid_grasp_poses(
            mesh_vertices=affordance.mesh_vertices,
            mesh_triangles=affordance.mesh_triangles,
            obj_poses=object_pose,
            approach_direction=approach_direction,
            obj_longest_axis=obj_longest_axis,
            is_positive_part=is_positive_part,
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
        grasp_frame_to_eef = options.grasp_frame_to_eef.to(
            device=self.device,
            dtype=grasp_variants.dtype,
        )
        grasp_variants = torch.matmul(grasp_variants, grasp_frame_to_eef)

        pre_grasp_variants = grasp_variants.clone()
        pre_grasp_variants[..., :3, 3] -= (
            approach_direction * options.pre_grasp_distance
        )
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
                    num_envs, 1, 1
                )
            if object_target_pose.shape != (num_envs, 4, 4):
                raise ValueError(
                    "downstream_object_target_poses entries must have shape "
                    f"(4, 4) or ({num_envs}, 4, 4), but got "
                    f"{object_target_pose.shape}."
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
            name=manipulator.control_part,
            to_matrix=True,
        )
        start_quat = quat_from_matrix(start_xpos[:, :3, :3])
        # Preserve the established preference between symmetric roll variants;
        # use the upright-adjusted pose only for feasibility and execution.
        selection_eef_variants = torch.matmul(
            selection_variants,
            grasp_frame_to_eef,
        )
        variant_quat = quat_from_matrix(selection_eef_variants[..., :3, :3])
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
            is_success.to(device=self.device, dtype=torch.bool).reshape(
                num_envs,
                n_pose,
                n_variant,
            ),
            qpos.reshape(num_envs, n_pose, n_variant, manipulator_dof),
        )

    def _upright_adjusted_grasp_poses(
        self,
        grasp_xpos: torch.Tensor,
        object_pose: torch.Tensor,
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
