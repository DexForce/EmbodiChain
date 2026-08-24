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

"""AxisAlign atomic action implementation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

import torch

from embodichain.utils import logger
from embodichain.utils.math import (
    axis_angle_to_rotation_matrix,
    get_relative_rotation,
    pose_inv,
)

from embodichain.lab.sim.atomic_actions.affordance import AxisAlignAffordance
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
from embodichain.lab.sim.atomic_actions.invocation import ResolvedActionRequest
from embodichain.lab.sim.atomic_actions.plans import (
    ActionPlan,
    TimedTrajectory,
    normalize_success_mask,
)
from embodichain.lab.sim.atomic_actions.primitives._binding_contracts import (
    make_manipulation_slot,
)
from embodichain.lab.sim.atomic_actions.primitives._helpers import arm_qpos_from_state
from embodichain.lab.sim.atomic_actions.primitives.pick_up import PickUpOptions
from embodichain.lab.sim.atomic_actions.requirements import (
    CARTESIAN_POSE_CAPABILITY,
    FORWARD_KINEMATICS_CAPABILITY,
    SkillBindingContract,
)
from embodichain.lab.sim.atomic_actions.state import PlanningContext
from embodichain.lab.sim.atomic_actions.trajectory_ops import (
    build_pose_plan_states,
    interpolate_hand_qpos,
    resolve_pose_target,
    translate_pose_world,
)


@dataclass(frozen=True, slots=True, eq=False)
class AxisAlignGoal(ObjectActionGoal):
    """Object whose local axis should be aligned after an antipodal grasp."""

    goal_kind: ClassVar[str] = "axis_align"

    grasp_xpos: PoseGoalValue | None = None
    """Optional explicit end-effector grasp pose; omitted poses are sampled."""

    def __post_init__(self) -> None:
        ObjectActionGoal.__post_init__(self)
        if self.grasp_xpos is not None:
            validate_pose_goal(self.grasp_xpos, "grasp_xpos", allow_waypoints=False)


@dataclass(frozen=True, slots=True, eq=False)
class AxisAlignOptions(PickUpOptions):
    """Per-invocation grasp-and-axis-alignment behavior."""

    target_axis: torch.Tensor = torch.tensor([0.0, 0.0, 1.0])
    """Desired world-frame axis, shape ``(3,)`` or ``(B, 3)``."""

    lower_distance: float = 0.03
    """World-Z distance (m) to lower the aligned object before release."""

    def __post_init__(self) -> None:
        PickUpOptions.__post_init__(self)
        if (
            not isinstance(self.target_axis, torch.Tensor)
            or self.target_axis.dim() not in (1, 2)
            or self.target_axis.shape[-1] != 3
            or not torch.isfinite(self.target_axis).all()
        ):
            raise ValueError("target_axis must be a finite (3,) or (B, 3) tensor.")
        if torch.any(torch.linalg.vector_norm(self.target_axis, dim=-1) <= 1.0e-6):
            raise ValueError("target_axis must be non-zero.")
        if not math.isfinite(self.lower_distance):
            raise ValueError("lower_distance must be finite.")
        if self.lower_distance < 0.0:
            raise ValueError("lower_distance must be non-negative.")
        object.__setattr__(self, "target_axis", self.target_axis.clone())


class AxisAlign(AtomicAction[AxisAlignGoal, AxisAlignOptions]):
    """Grasp an object, align its local axis to a world axis, and release it."""

    skill_id: ClassVar[str] = "axis_align"
    GoalType: ClassVar[type] = AxisAlignGoal
    OptionsType: ClassVar[type] = AxisAlignOptions
    manipulator_roles: ClassVar[tuple[str, ...]] = ("primary",)
    end_effector_roles: ClassVar[tuple[str, ...]] = ("primary",)
    open_loop: ClassVar[bool] = True
    _UPRIGHT_HORIZONTAL_MAX_ABS_Z: ClassVar[float] = 0.5
    _UPRIGHT_TARGET_MIN_Z: ClassVar[float] = math.cos(math.pi / 6.0)
    _UPRIGHT_GRASP_PRE_ROTATION: ClassVar[float] = math.pi / 4.0
    binding_contract: ClassVar[SkillBindingContract] = SkillBindingContract(
        slots=(
            make_manipulation_slot(
                "primary",
                motion_capabilities=frozenset(
                    {
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

    def __init__(self, default_options: AxisAlignOptions | None = None) -> None:
        super().__init__(default_options)

    def _scene_dependencies(
        self,
        request: ResolvedActionRequest[AxisAlignGoal, AxisAlignOptions],
    ) -> tuple[str, ...]:
        """Include the semantic object when it has a stable scene identity."""
        dependencies = set(super()._scene_dependencies(request))
        entity_id = request.goal.semantics.entity_id
        if entity_id is not None:
            dependencies.add(entity_id)
        return tuple(sorted(dependencies))

    def _find_symmetric_nearest_xpos(
        self, target_xpos: torch.Tensor, reference_xpos: torch.Tensor
    ) -> torch.Tensor:
        """Find the nearest symmetric pose to the reference pose."""
        symmetric_xpos = target_xpos.clone()
        symmetric_xpos[:, :3, 0] = -symmetric_xpos[:, :3, 0]
        symmetric_xpos[:, :3, 1] = -symmetric_xpos[:, :3, 1]
        angle_a = get_relative_rotation(
            reference_xpos[:, :3, :3], target_xpos[:, :3, :3]
        )
        angle_b = get_relative_rotation(
            reference_xpos[:, :3, :3], symmetric_xpos[:, :3, :3]
        )
        choose_target = (angle_a < angle_b)[..., None, None]
        target_xpos = torch.where(choose_target, target_xpos, symmetric_xpos)
        return target_xpos

    def _plan(
        self,
        request: ResolvedActionRequest[AxisAlignGoal, AxisAlignOptions],
        context: PlanningContext,
    ) -> ActionPlan:
        """Plan all seven physical actions in two arm-planning phases."""
        target = request.goal
        options = request.skill_options
        affordance = self._require_axis_align_affordance(target.semantics)
        motion_endpoint = request.binding.endpoint("primary", "motion")
        grasp_endpoint = request.binding.endpoint("primary", "grasp")
        manipulator = motion_endpoint.require_target(JointPositionTarget)
        end_effector = grasp_endpoint.require_target(JointPositionTarget)
        arm_joint_ids = list(manipulator.joint_ids)
        hand_joint_ids = list(end_effector.joint_ids)
        start_arm_qpos = arm_qpos_from_state(context, arm_joint_ids)
        hand_open_qpos = grasp_endpoint.joint_positions(
            OPEN_COMMAND,
            num_envs=context.batch_size,
            device=self.device,
            dtype=context.robot.qpos.dtype,
        )
        hand_grasp_qpos = grasp_endpoint.joint_positions(
            GRASP_COMMAND,
            num_envs=context.batch_size,
            device=self.device,
            dtype=context.robot.qpos.dtype,
        )
        approach_direction = options.approach_direction.to(
            device=self.device, dtype=torch.float32
        )
        approach_direction = approach_direction / torch.linalg.vector_norm(
            approach_direction
        )
        object_pose = _resolve_object_pose(
            target.semantics,
            context,
            name="axis_align_object_pose",
        )

        # Resolve the shortest object rotation before selecting a grasp.  The
        # source axis is ``object_rotation @ internal_axis`` in world space;
        # ``rotation_axis`` is the normalized cross product from that source to
        # the requested world-space target.  For opposite axes the helper picks
        # a deterministic perpendicular axis instead of dividing by a near-zero
        # cross product.
        source_axis, target_axis, rotation_axis, rotation_angle = (
            self._axis_alignment_parameters(
                object_pose,
                affordance.internal_axis,
                options.target_axis,
            )
        )

        # If no explicit grasp was supplied, _resolve_grasp_pose first filters
        # invalid affordance samples, then gives priority to the candidates whose
        # TCP y-axis is most perpendicular to ``rotation_axis``.  Grasp-generator
        # cost only breaks ties between equally perpendicular candidates.  This
        # orientation keeps the gripper's roll axis away from the object's
        # rotation axis and generally leaves the arm more room for the alignment
        # motion.  The antipodal pose has a 180-degree symmetric alternative;
        # after choosing the sample, select whichever symmetric orientation is
        # closer to the arm's currently observed FK pose.
        grasp_success, grasp_xpos = self._resolve_grasp_pose(
            target,
            affordance,
            object_pose,
            context,
            approach_direction,
            rotation_axis,
            rotation_angle,
            end_effector.target_id,
            object_part=options.pick_object_part,
        )
        grasp_xpos = self._find_symmetric_nearest_xpos(
            grasp_xpos,
            reference_xpos=self.robot.compute_fk(
                qpos=start_arm_qpos,
                name=manipulator.control_part,
                to_matrix=True,
            ),
        )
        grasp_success = normalize_success_mask(
            grasp_success,
            num_envs=self.num_envs,
            device=self.device,
            name="Axis-align grasp success",
        )
        if not grasp_success.any():
            logger.log_warning("AxisAlign failed to resolve a grasp pose.")
            return self.failed_plan(
                request, context, message="Failed to resolve a grasp pose."
            )

        # Upright handling is enabled independently for each environment when
        # the current object axis is mostly horizontal (|world z| <= 0.5) and
        # the requested target points mostly upward (within 30 degrees of +Z).
        # Before deriving the fixed object-to-EEF grasp transform, rotate only
        # the grasp orientation by 45 degrees *opposite* ``rotation_axis``; its
        # position is unchanged.  The subsequent alignment still rotates the
        # object through the full shortest arc, but the arm starts that arc with
        # a 45-degree bias, reducing the link sweep near the table.
        upright_mask = (
            source_axis[:, 2].abs() <= self._UPRIGHT_HORIZONTAL_MAX_ABS_Z
        ) & (target_axis[:, 2] >= self._UPRIGHT_TARGET_MIN_Z)
        grasp_xpos = self._apply_upright_grasp_pre_rotation(
            grasp_xpos,
            rotation_axis,
            upright_mask,
        )

        pre_grasp_xpos = translate_pose_world(
            grasp_xpos, -approach_direction * options.pre_grasp_distance
        )
        lift_xpos = translate_pose_world(
            grasp_xpos,
            torch.tensor(
                [0.0, 0.0, options.lift_height],
                device=self.device,
                dtype=torch.float32,
            ),
        )
        object_to_eef = torch.bmm(pose_inv(object_pose), grasp_xpos)
        lifted_object_pose = torch.bmm(lift_xpos, pose_inv(object_to_eef))

        n_approach, n_reach, n_lift, n_align, n_lower = self._motion_segment_lengths(
            request.motion_policy.sample_count,
            options.hand_interp_steps,
        )
        interpolation_dt = context.require_control_dt()
        # Only the final aligned pose is a planner target.  Supplying n_align
        # intermediate Cartesian keyframes would make CuRobo call plan_pose once
        # per keyframe; n_align is instead retained as the output sample budget
        # for the continuous post-close phase.
        align_xpos = self._axis_alignment_eef_keyframes(
            lifted_object_pose,
            object_to_eef,
            affordance.internal_axis,
            options.target_axis,
            waypoint_count=1,
        )
        lower_xpos = translate_pose_world(
            align_xpos[:, -1],
            torch.tensor(
                [0.0, 0.0, -options.lower_distance],
                device=self.device,
                dtype=torch.float32,
            ),
        )

        # CuRobo planning is grouped by gripper state.  The open-gripper phase
        # contains both the pre-grasp and grasp waypoints, so one generate call
        # replaces the former independent approach and reach calls.
        pre_close_xpos = torch.stack([pre_grasp_xpos, grasp_xpos], dim=1)
        pre_close_success, pre_close_arm = self._plan_pose_phase(
            pre_close_xpos,
            start_arm_qpos,
            manipulator,
            request,
            n_approach + n_reach,
            interpolation_dt,
        )

        # Once the gripper is closed, lifting, alignment, and lowering form one
        # continuous held-object phase.  Passing only those three semantic
        # endpoints retains the required ordering without expanding the rotation
        # into many CuRobo plan_pose calls.  Together with the open-gripper phase,
        # the action now uses two MotionGenerator.generate calls and five backend
        # target plans instead of n_align + 4 backend target plans.
        post_close_xpos = torch.cat(
            [lift_xpos[:, None], align_xpos, lower_xpos[:, None]], dim=1
        )
        post_close_success, post_close_arm = self._plan_pose_phase(
            post_close_xpos,
            pre_close_arm[:, -1],
            manipulator,
            request,
            n_lift + n_align + n_lower,
            interpolation_dt,
        )
        success = grasp_success & normalize_success_mask(
            pre_close_success & post_close_success,
            num_envs=self.num_envs,
            device=self.device,
            name="Axis-align trajectory success",
        )

        hand_close = interpolate_hand_qpos(
            hand_open_qpos,
            hand_grasp_qpos,
            n_waypoints=options.hand_interp_steps,
        )
        hand_open = interpolate_hand_qpos(
            hand_grasp_qpos,
            hand_open_qpos,
            n_waypoints=options.hand_interp_steps,
        )
        segment_lengths = {
            "approach": pre_close_arm.shape[1],
            "close": hand_close.shape[1],
            "manipulate": post_close_arm.shape[1],
            "open": hand_open.shape[1],
        }
        full = torch.empty(
            (self.num_envs, sum(segment_lengths.values()), self.robot_dof),
            dtype=context.robot.qpos.dtype,
            device=self.device,
        )
        full[:] = context.last_qpos.unsqueeze(1)
        offset = pre_close_arm.shape[1]
        full[:, :offset, arm_joint_ids] = pre_close_arm
        full[:, :offset, hand_joint_ids] = hand_open_qpos.unsqueeze(1)
        stop = offset + hand_close.shape[1]
        full[:, offset:stop, arm_joint_ids] = pre_close_arm[:, -1].unsqueeze(1)
        full[:, offset:stop, hand_joint_ids] = hand_close
        offset = stop
        stop = offset + post_close_arm.shape[1]
        full[:, offset:stop, arm_joint_ids] = post_close_arm
        full[:, offset:stop, hand_joint_ids] = hand_grasp_qpos.unsqueeze(1)
        offset = stop
        full[:, offset:, arm_joint_ids] = post_close_arm[:, -1].unsqueeze(1)
        full[:, offset:, hand_joint_ids] = hand_open

        return self.build_plan(
            request,
            context,
            success=success,
            trajectory=TimedTrajectory.from_uniform_step(
                full,
                env_ids=context.env_ids,
                step_dt=interpolation_dt,
            ),
            expected_effects=StateDelta(),
            segment_lengths=segment_lengths,
        )

    def _resolve_grasp_pose(
        self,
        goal: AxisAlignGoal,
        affordance: AxisAlignAffordance,
        object_pose: torch.Tensor,
        context: PlanningContext,
        approach_direction: torch.Tensor,
        rotation_axis: torch.Tensor,
        rotation_angle: torch.Tensor,
        grasp_target_id: str,
        *,
        object_part: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Resolve an explicit grasp or select the lowest-cost sampled grasp."""
        if goal.grasp_xpos is not None:
            grasp_xpos = resolve_pose_target(
                resolve_pose_goal(goal.grasp_xpos, context, name="grasp_xpos"),
                num_envs=self.num_envs,
                device=self.device,
            )
            return (
                torch.ones(self.num_envs, dtype=torch.bool, device=self.device),
                grasp_xpos,
            )

        obj_longest_axis = None
        is_positive_part = True
        if object_part != "center":
            obj_longest_axis = torch.tensor(
                [0.0, 0.0, 1.0],
                dtype=torch.float32,
                device=self.device,
            )
            is_positive_part = object_part == "top"
        generator = self.planning_services.grasp_pose_generator(grasp_target_id)
        sampled = generator.get_valid_grasp_poses(
            mesh_vertices=affordance.mesh_vertices,
            mesh_triangles=affordance.mesh_triangles,
            obj_poses=object_pose,
            approach_direction=approach_direction,
            obj_longest_axis=obj_longest_axis,
            is_positive_part=is_positive_part,
        )
        poses: list[torch.Tensor] = []
        success: list[bool] = []
        for env_index, (candidates, costs) in enumerate(sampled):
            candidates = candidates.to(device=self.device, dtype=torch.float32)
            costs = costs.to(device=self.device, dtype=torch.float32)
            valid = candidates.shape[0] > 0 and bool(torch.isfinite(costs).any())
            if valid:
                finite_cost = torch.isfinite(costs)
                if rotation_angle[env_index] > 1.0e-6:
                    grasp_y_axis = torch.nn.functional.normalize(
                        candidates[:, :3, 1], dim=1
                    )
                    perpendicularity_error = torch.abs(
                        torch.matmul(grasp_y_axis, rotation_axis[env_index])
                    )
                    best_error = perpendicularity_error[finite_cost].min()
                    preferred = finite_cost & torch.isclose(
                        perpendicularity_error,
                        best_error,
                        atol=1.0e-6,
                        rtol=1.0e-5,
                    )
                    ranked_costs = torch.where(
                        preferred,
                        costs,
                        torch.full_like(costs, torch.inf),
                    )
                    best_index = int(torch.argmin(ranked_costs).item())
                else:
                    best_index = int(torch.argmin(costs).item())
                poses.append(candidates[best_index])
            else:
                poses.append(torch.eye(4, device=self.device, dtype=torch.float32))
            success.append(valid)
        return (
            torch.tensor(success, dtype=torch.bool, device=self.device),
            torch.stack(poses),
        )

    def _plan_pose_phase(
        self,
        target_pose: torch.Tensor,
        start_qpos: torch.Tensor,
        manipulator: JointPositionTarget,
        request: ResolvedActionRequest[AxisAlignGoal, AxisAlignOptions],
        sample_count: int,
        interpolation_dt: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Plan one continuous arm phase with a fixed gripper command."""
        result = self.motion_generator.generate(
            build_pose_plan_states(target_pose),
            options=request.motion_policy.to_motion_gen_options(
                start_qpos=start_qpos,
                control_part=manipulator.control_part,
                sample_count=sample_count,
                interpolation_dt=interpolation_dt,
            ),
        )
        assert isinstance(result.success, torch.Tensor)
        assert result.positions is not None
        return result.success, result.positions

    def _axis_alignment_eef_keyframes(
        self,
        object_pose: torch.Tensor,
        object_to_eef: torch.Tensor,
        internal_axis: torch.Tensor,
        target_axis: torch.Tensor,
        *,
        waypoint_count: int,
    ) -> torch.Tensor:
        """Rotate the object in place along the shortest axis-alignment arc."""
        _, _, axis, angle = self._axis_alignment_parameters(
            object_pose,
            internal_axis,
            target_axis,
        )

        fractions = torch.linspace(
            1.0 / waypoint_count,
            1.0,
            waypoint_count,
            dtype=torch.float32,
            device=self.device,
        )
        rotation_vectors = (
            axis[:, None, :] * angle[:, None, None] * fractions[None, :, None]
        )
        delta_rotation = axis_angle_to_rotation_matrix(rotation_vectors)
        object_keyframes = object_pose[:, None].repeat(1, waypoint_count, 1, 1)
        object_keyframes[:, :, :3, :3] = torch.matmul(
            delta_rotation, object_pose[:, None, :3, :3]
        )
        return torch.matmul(object_keyframes, object_to_eef[:, None])

    def _axis_alignment_parameters(
        self,
        object_pose: torch.Tensor,
        internal_axis: torch.Tensor,
        target_axis: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return normalized source/target axes and the shortest rotation."""
        internal = internal_axis.to(device=self.device, dtype=torch.float32)
        internal = internal / torch.linalg.vector_norm(internal)
        source = torch.matmul(object_pose[:, :3, :3], internal)
        source = torch.nn.functional.normalize(source, dim=1)
        target = target_axis.to(device=self.device, dtype=torch.float32)
        if target.shape == (3,):
            target = target.unsqueeze(0).expand(self.num_envs, -1)
        elif target.shape != (self.num_envs, 3):
            raise ValueError(
                f"target_axis must have shape (3,) or ({self.num_envs}, 3)."
            )
        target = torch.nn.functional.normalize(target, dim=1)

        cross = torch.linalg.cross(source, target, dim=1)
        sin_angle = torch.linalg.vector_norm(cross, dim=1)
        cos_angle = torch.sum(source * target, dim=1).clamp(-1.0, 1.0)
        axis = cross / sin_angle.clamp_min(1.0e-8).unsqueeze(1)
        basis = torch.eye(3, dtype=torch.float32, device=self.device)
        reference = basis[torch.argmin(torch.abs(source), dim=1)]
        fallback_axis = torch.nn.functional.normalize(
            torch.linalg.cross(source, reference, dim=1), dim=1
        )
        degenerate = sin_angle <= 1.0e-6
        axis = torch.where(degenerate.unsqueeze(1), fallback_axis, axis)
        angle = torch.atan2(sin_angle, cos_angle)
        opposite = degenerate & (cos_angle < 0.0)
        angle = torch.where(opposite, torch.full_like(angle, torch.pi), angle)
        return source, target, axis, angle

    def _apply_upright_grasp_pre_rotation(
        self,
        grasp_xpos: torch.Tensor,
        rotation_axis: torch.Tensor,
        upright_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Pre-rotate upright grasps to reduce the arm's table-side sweep."""
        if not upright_mask.any():
            return grasp_xpos
        delta = axis_angle_to_rotation_matrix(
            -rotation_axis * self._UPRIGHT_GRASP_PRE_ROTATION
        )
        rotated = grasp_xpos.clone()
        rotated[:, :3, :3] = torch.matmul(delta, grasp_xpos[:, :3, :3])
        return torch.where(upright_mask[:, None, None], rotated, grasp_xpos)

    @staticmethod
    def _motion_segment_lengths(
        sample_count: int,
        hand_interp_steps: int,
    ) -> tuple[int, int, int, int, int]:
        motion_count = sample_count - 2 * hand_interp_steps
        if motion_count < 5:
            raise ValueError(
                "Not enough waypoints for AxisAlign. Increase sample_count or "
                "decrease hand_interp_steps."
            )
        base, remainder = divmod(motion_count, 5)
        values = [base + (index < remainder) for index in range(5)]
        return values[0], values[1], values[2], values[3], values[4]

    @staticmethod
    def _require_axis_align_affordance(
        semantics: ObjectSemantics,
    ) -> AxisAlignAffordance:
        affordance = semantics.affordance
        if not isinstance(affordance, AxisAlignAffordance):
            raise ValueError("AxisAlign requires an AxisAlignAffordance.")
        return affordance


__all__ = ["AxisAlign", "AxisAlignGoal", "AxisAlignOptions"]
