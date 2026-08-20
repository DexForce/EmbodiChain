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
from embodichain.utils.math import axis_angle_to_rotation_matrix, pose_inv

from embodichain.lab.sim.atomic_actions.affordance import AxisAlignAffordance
from embodichain.lab.sim.atomic_actions.bindings import ResolvedControlPart
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

    def _plan(
        self,
        request: ResolvedActionRequest[AxisAlignGoal, AxisAlignOptions],
        context: PlanningContext,
    ) -> ActionPlan:
        """Plan all seven segments without stepping the simulator."""
        target = request.goal
        options = request.skill_options
        affordance = self._require_axis_align_affordance(target.semantics)
        manipulator = request.binding.manipulator()
        end_effector = request.binding.end_effector()
        arm_joint_ids = list(manipulator.joint_ids)
        hand_joint_ids = list(end_effector.joint_ids)
        start_arm_qpos = arm_qpos_from_state(context, arm_joint_ids)
        hand_open_qpos = end_effector.joint_positions(
            OPEN_COMMAND,
            num_envs=context.batch_size,
            device=self.device,
            dtype=context.robot.qpos.dtype,
        )
        hand_grasp_qpos = end_effector.joint_positions(
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
        source_axis, target_axis, rotation_axis, rotation_angle = (
            self._axis_alignment_parameters(
                object_pose,
                affordance.internal_axis,
                options.target_axis,
            )
        )
        grasp_success, grasp_xpos = self._resolve_grasp_pose(
            target,
            affordance,
            object_pose,
            context,
            approach_direction,
            rotation_axis,
            rotation_angle,
            object_part=options.pick_object_part,
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
        align_xpos = self._axis_alignment_eef_keyframes(
            lifted_object_pose,
            object_to_eef,
            affordance.internal_axis,
            options.target_axis,
            waypoint_count=n_align,
        )
        lower_xpos = translate_pose_world(
            align_xpos[:, -1],
            torch.tensor(
                [0.0, 0.0, -options.lower_distance],
                device=self.device,
                dtype=torch.float32,
            ),
        )

        approach_success, approach_arm = self._plan_pose_segment(
            pre_grasp_xpos,
            start_arm_qpos,
            manipulator,
            request,
            n_approach,
            interpolation_dt,
        )
        reach_success, reach_arm = self._plan_pose_segment(
            grasp_xpos,
            approach_arm[:, -1],
            manipulator,
            request,
            n_reach,
            interpolation_dt,
        )
        lift_success, lift_arm = self._plan_pose_segment(
            lift_xpos,
            reach_arm[:, -1],
            manipulator,
            request,
            n_lift,
            interpolation_dt,
        )
        align_success, align_arm = self._plan_pose_segment(
            align_xpos,
            lift_arm[:, -1],
            manipulator,
            request,
            n_align,
            interpolation_dt,
        )
        lower_success, lower_arm = self._plan_pose_segment(
            lower_xpos,
            align_arm[:, -1],
            manipulator,
            request,
            n_lower,
            interpolation_dt,
        )
        success = grasp_success & normalize_success_mask(
            approach_success
            & reach_success
            & lift_success
            & align_success
            & lower_success,
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
        arm_parts = (approach_arm, reach_arm, lift_arm, align_arm, lower_arm)
        segment_lengths = {
            "approach": approach_arm.shape[1],
            "reach": reach_arm.shape[1],
            "close": hand_close.shape[1],
            "lift": lift_arm.shape[1],
            "align": align_arm.shape[1],
            "lower": lower_arm.shape[1],
            "open": hand_open.shape[1],
        }
        full = torch.empty(
            (self.num_envs, sum(segment_lengths.values()), self.robot_dof),
            dtype=context.robot.qpos.dtype,
            device=self.device,
        )
        full[:] = context.last_qpos.unsqueeze(1)
        offset = 0
        for arm, hand in (
            (arm_parts[0], hand_open_qpos),
            (arm_parts[1], hand_open_qpos),
        ):
            stop = offset + arm.shape[1]
            full[:, offset:stop, arm_joint_ids] = arm
            full[:, offset:stop, hand_joint_ids] = hand.unsqueeze(1)
            offset = stop
        stop = offset + hand_close.shape[1]
        full[:, offset:stop, arm_joint_ids] = reach_arm[:, -1].unsqueeze(1)
        full[:, offset:stop, hand_joint_ids] = hand_close
        offset = stop
        for arm in arm_parts[2:]:
            stop = offset + arm.shape[1]
            full[:, offset:stop, arm_joint_ids] = arm
            full[:, offset:stop, hand_joint_ids] = hand_grasp_qpos.unsqueeze(1)
            offset = stop
        full[:, offset:, arm_joint_ids] = lower_arm[:, -1].unsqueeze(1)
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

        sampled = affordance.get_valid_grasp_poses(
            obj_poses=object_pose,
            approach_direction=approach_direction,
            object_part=object_part,
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
                    grasp_x_axis = torch.nn.functional.normalize(
                        candidates[:, :3, 0], dim=1
                    )
                    perpendicularity_error = torch.abs(
                        torch.matmul(grasp_x_axis, rotation_axis[env_index])
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

    def _plan_pose_segment(
        self,
        target_pose: torch.Tensor,
        start_qpos: torch.Tensor,
        manipulator: ResolvedControlPart,
        request: ResolvedActionRequest[AxisAlignGoal, AxisAlignOptions],
        sample_count: int,
        interpolation_dt: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        result = self.motion_generator.generate(
            build_pose_plan_states(target_pose),
            options=request.motion_policy.to_motion_gen_options(
                start_qpos=start_qpos,
                control_part=manipulator.name,
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
