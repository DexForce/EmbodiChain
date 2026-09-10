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

"""Slide atomic action implementation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Real
from typing import ClassVar, Literal

import torch

from embodichain.lab.sim.atomic_actions.affordance import SlideAffordance
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
    resolve_pose_goal,
    validate_pose_goal,
)
from embodichain.lab.sim.atomic_actions.invocation import (
    ActionOptions,
    ResolvedActionRequest,
)
from embodichain.lab.sim.atomic_actions.plans import (
    ActionPlan,
    PlannerDiagnostics,
    TimedTrajectory,
    normalize_success_mask,
)
from embodichain.lab.sim.atomic_actions.primitives._binding_contracts import (
    make_manipulation_slot,
)
from embodichain.lab.sim.atomic_actions.primitives._helpers import arm_qpos_from_state
from embodichain.lab.sim.atomic_actions.requirements import (
    CARTESIAN_POSE_CAPABILITY,
    SkillBindingContract,
)
from embodichain.lab.sim.atomic_actions.state import PlanningContext
from embodichain.lab.sim.atomic_actions.trajectory_ops import (
    axis_translation_keyframes,
    build_pose_plan_states,
    interpolate_hand_qpos,
    resolve_pose_target,
    translate_pose_world,
)


@dataclass(frozen=True, slots=True)
class SlideJointTarget:
    """Absolute prismatic coordinate, in metres, with a calibrated axis sign.

    Args:
        articulation_id: Canonical articulation identity in the scene snapshot.
        joint_name: Exact observed prismatic joint name.
        position: Requested absolute joint position in metres.
        axis_sign: Maps increasing joint position to the affordance axis (+1/-1).
        tolerance: Position tolerance for an already-satisfied planning row.
    """

    articulation_id: str
    joint_name: str
    position: float
    axis_sign: int = field(kw_only=True)
    tolerance: float = 1.0e-4

    def __post_init__(self) -> None:
        for name in ("articulation_id", "joint_name"):
            value = getattr(self, name)
            if type(value) is not str or not value or value != value.strip():
                raise ValueError(f"{name} must be an exact non-empty identifier.")
        for name in ("position", "tolerance"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be a finite real number.")
            object.__setattr__(self, name, float(value))
        if (
            self.tolerance <= 0
            or type(self.axis_sign) is not int
            or self.axis_sign not in (-1, 1)
        ):
            raise ValueError(
                "Slide joint targets require positive tolerance and axis_sign +/-1."
            )


@dataclass(frozen=True, slots=True, eq=False)
class SlideGoal(ObjectActionGoal):
    """Translating articulation link described by a slide affordance."""

    target_pose: PoseGoalValue
    """Link pose snapshot or late-bound stable scene-entity reference."""

    joint_target: SlideJointTarget | None = field(default=None, kw_only=True)
    """Optional absolute joint intent; otherwise preserve fixed-distance Options.

    Joint-target mode resolves fresh row-local distances and the pull/push
    direction during planning. Active rows must share one direction; already
    satisfied rows hold their observed pose. This does not verify physical effect.
    """

    def __post_init__(self) -> None:
        ObjectActionGoal.__post_init__(self)
        validate_pose_goal(self.target_pose, "target_pose", allow_waypoints=False)
        if (
            self.joint_target is not None
            and type(self.joint_target) is not SlideJointTarget
        ):
            raise TypeError("joint_target must be exactly SlideJointTarget or None.")


@dataclass(frozen=True, slots=True, eq=False)
class SlideOptions(ActionOptions):
    """Per-invocation sliding behavior for a translating articulation link."""

    direction: Literal["pull", "push"] = "pull"
    """Whether to pull the part open or push it closed."""

    hand_interp_steps: int = 5
    """Number of waypoints used for each close/open hand segment."""

    approach_distance: float = 0.1
    """Pre-grasp distance opposite the approach/push axis."""

    translation_distance: float = 0.15
    """Distance traveled along the pull or push direction."""

    def __post_init__(self) -> None:
        if self.direction not in ("pull", "push"):
            raise ValueError("direction must be either 'pull' or 'push'.")
        if self.hand_interp_steps < 1:
            raise ValueError("hand_interp_steps must be at least 1.")
        if not math.isfinite(self.approach_distance):
            raise ValueError("approach_distance must be finite.")
        if self.approach_distance < 0.0:
            raise ValueError("approach_distance must be non-negative.")
        if not math.isfinite(self.translation_distance):
            raise ValueError("translation_distance must be finite.")
        if self.translation_distance <= 0.0:
            raise ValueError("translation_distance must be positive.")


class Slide(AtomicAction[SlideGoal, SlideOptions]):
    """Open-loop approach, grasp, and axis-constrained sliding motion."""

    skill_id: ClassVar[str] = "slide"
    GoalType: ClassVar[type] = SlideGoal
    OptionsType: ClassVar[type] = SlideOptions
    open_loop: ClassVar[bool] = True
    binding_contract: ClassVar[SkillBindingContract] = SkillBindingContract(
        slots=(
            make_manipulation_slot(
                "primary",
                motion_capabilities=frozenset({CARTESIAN_POSE_CAPABILITY}),
                grasp_commands={
                    OPEN_COMMAND: JointPositionCommand,
                    GRASP_COMMAND: JointPositionCommand,
                },
            ),
        ),
    )

    def _on_bind(self) -> None:
        """Resolve dimensions owned by the engine's robot."""
        self.num_envs = self.robot.get_qpos().shape[0]
        self.robot_dof = self.robot.dof

    def _plan(
        self,
        request: ResolvedActionRequest[
            SlideGoal,
            SlideOptions,
        ],
        context: PlanningContext,
    ) -> ActionPlan:
        """Plan the complete pull/push sequence without stepping simulation."""
        target = self.require_goal(request)
        affordance = self._require_slide_affordance(target.semantics)
        options = request.skill_options
        interpolation_dt = context.require_control_dt()
        direction = options.direction
        displacement = None
        joint_valid = torch.ones(
            context.batch_size, dtype=torch.bool, device=self.device
        )
        reached = torch.zeros_like(joint_valid)
        diagnostics = {}
        if target.joint_target is not None:
            displacement, joint_valid, reached = self._joint_displacement(
                target.joint_target, affordance, context
            )
            active = joint_valid & ~reached
            diagnostics["diagnostics"] = PlannerDiagnostics(
                backend=self.planning_services.planner_name,
                metadata={
                    "joint_target": {
                        "articulation_id": target.joint_target.articulation_id,
                        "joint_name": target.joint_target.joint_name,
                        "position": target.joint_target.position,
                        "already_satisfied": reached.detach().cpu().tolist(),
                    }
                },
            )
            if not active.any():
                return self.build_plan(
                    request,
                    context,
                    success=reached,
                    trajectory=TimedTrajectory.from_uniform_step(
                        context.robot.qpos[:, None],
                        env_ids=context.env_ids,
                        step_dt=interpolation_dt,
                    ),
                    segment_lengths={"already_satisfied": 1},
                    **diagnostics,
                )
            positive = displacement[active] > 0
            if positive.any() and not positive.all():
                raise ValueError(
                    "One Slide invocation requires the same direction for active joint-target rows."
                )
            direction = "push" if positive.all() else "pull"
        binding = request.binding
        motion_target = binding.endpoint("primary", "motion").require_target(
            JointPositionTarget
        )
        grasp = binding.endpoint("primary", "grasp")
        grasp_target = grasp.require_target(JointPositionTarget)
        control_part = motion_target.control_part
        arm_joint_ids = list(motion_target.joint_ids)
        hand_joint_ids = list(grasp_target.joint_ids)
        start_arm_qpos = arm_qpos_from_state(context, arm_joint_ids)
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

        link_pose = resolve_pose_target(
            resolve_pose_goal(target.target_pose, context, name="target_pose"),
            num_envs=self.num_envs,
            device=self.device,
        )
        translation_axis = affordance.translation_axis.to(
            device=self.device, dtype=torch.float32
        )
        translation_axis = translation_axis / torch.linalg.vector_norm(translation_axis)
        translation_axis_world = torch.matmul(link_pose[:, :3, :3], translation_axis)
        grasp_generator = self.planning_services.grasp_pose_generator(
            grasp_target.target_id
        )
        grasp_success, grasp_xpos, _ = grasp_generator.get_best_grasp_poses(
            mesh_vertices=affordance.mesh_vertices,
            mesh_triangles=affordance.mesh_triangles,
            obj_poses=link_pose,
            approach_direction=translation_axis_world,
        )
        grasp_xpos = grasp_xpos.to(device=self.device, dtype=torch.float32)
        grasp_success = normalize_success_mask(
            grasp_success,
            num_envs=self.num_envs,
            device=self.device,
            name="Slide grasp-pose success",
        )
        if not grasp_success.any():
            if target.joint_target is not None and reached.any():
                return self.build_plan(
                    request,
                    context,
                    success=reached,
                    trajectory=TimedTrajectory.from_uniform_step(
                        context.robot.qpos[:, None],
                        env_ids=context.env_ids,
                        step_dt=interpolation_dt,
                    ),
                    segment_lengths={"already_satisfied": 1},
                    **diagnostics,
                )
            return self.failed_plan(
                request,
                context,
                message="Failed to resolve an articulated-part grasp pose.",
            )
        approach_xpos = translate_pose_world(
            grasp_xpos,
            -translation_axis_world * options.approach_distance,
        )
        translation_sign = -1.0 if direction == "pull" else 1.0
        translated_xpos = translate_pose_world(
            grasp_xpos,
            translation_axis_world
            * (
                translation_sign * options.translation_distance
                if displacement is None
                else displacement[:, None]
            ),
        )

        motion_lengths = self._motion_segment_lengths(
            request.motion_policy.sample_count,
            options.hand_interp_steps,
            direction=direction,
        )
        approach_success, approach_arm = self._plan_pose_segment(
            approach_xpos,
            start_arm_qpos,
            control_part,
            request,
            motion_lengths[0],
            interpolation_dt=interpolation_dt,
        )
        reach_keyframes = axis_translation_keyframes(
            approach_xpos,
            grasp_xpos,
            translation_axis_world,
            n_waypoints=motion_lengths[1] - 1,
        )
        reach_success, reach_arm = self._plan_pose_segment(
            reach_keyframes,
            approach_arm[:, -1],
            control_part,
            request,
            motion_lengths[1],
            interpolation_dt=interpolation_dt,
            cartesian_linear=True,
        )
        translate_keyframes = axis_translation_keyframes(
            grasp_xpos,
            translated_xpos,
            translation_axis_world,
            n_waypoints=motion_lengths[2] - 1,
        )
        translate_success, translate_arm = self._plan_pose_segment(
            translate_keyframes,
            reach_arm[:, -1],
            control_part,
            request,
            motion_lengths[2],
            interpolation_dt=interpolation_dt,
            cartesian_linear=True,
        )
        success = grasp_success & approach_success & reach_success & translate_success

        return_arm: torch.Tensor | None = None
        if direction == "push":
            return_keyframes = axis_translation_keyframes(
                translated_xpos,
                approach_xpos,
                translation_axis_world,
                n_waypoints=motion_lengths[3] - 1,
            )
            return_success, return_arm = self._plan_pose_segment(
                return_keyframes,
                translate_arm[:, -1],
                control_part,
                request,
                motion_lengths[3],
                interpolation_dt=interpolation_dt,
                cartesian_linear=True,
            )
            success = success & return_success

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
        named_parts: list[tuple[str, torch.Tensor]] = [
            ("approach", approach_arm),
            ("reach", reach_arm),
            ("close", hand_close),
            (direction, translate_arm),
            ("open", hand_open),
        ]
        if return_arm is not None:
            named_parts.append(("return", return_arm))

        segment_lengths = {name: part.shape[1] for name, part in named_parts}
        full = torch.empty(
            (self.num_envs, sum(segment_lengths.values()), self.robot_dof),
            dtype=context.robot.qpos.dtype,
            device=self.device,
        )
        full[:] = context.last_qpos.unsqueeze(1)
        offset = 0

        for arm in (approach_arm, reach_arm):
            stop = offset + arm.shape[1]
            full[:, offset:stop, arm_joint_ids] = arm
            full[:, offset:stop, hand_joint_ids] = hand_open_qpos.unsqueeze(1)
            offset = stop

        stop = offset + hand_close.shape[1]
        full[:, offset:stop, arm_joint_ids] = reach_arm[:, -1].unsqueeze(1)
        full[:, offset:stop, hand_joint_ids] = hand_close
        offset = stop

        stop = offset + translate_arm.shape[1]
        full[:, offset:stop, arm_joint_ids] = translate_arm
        full[:, offset:stop, hand_joint_ids] = hand_grasp_qpos.unsqueeze(1)
        offset = stop

        stop = offset + hand_open.shape[1]
        full[:, offset:stop, arm_joint_ids] = translate_arm[:, -1].unsqueeze(1)
        full[:, offset:stop, hand_joint_ids] = hand_open
        offset = stop

        if return_arm is not None:
            full[:, offset:, arm_joint_ids] = return_arm
            full[:, offset:, hand_joint_ids] = hand_open_qpos.unsqueeze(1)

        if target.joint_target is not None:
            full[reached] = context.robot.qpos[reached, None]
            success = (success & joint_valid) | reached

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
            # Once reach completes, contact or the commanded slide may move the
            # articulated target. That self-induced motion must not recover.
            scene_dependency_end_segment=(
                "reach" if self._scene_dependencies(request) else None
            ),
            **diagnostics,
        )

    @staticmethod
    def _joint_displacement(
        target: SlideJointTarget, affordance: SlideAffordance, context: PlanningContext
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not math.isclose(
            context.scene.timestamp, context.robot.timestamp, abs_tol=1e-9, rel_tol=0
        ):
            raise ValueError(
                "Slide joint and robot observations must share a timestamp."
            )
        limits = affordance.joint_limits
        if affordance.joint_name != target.joint_name or limits is None:
            raise ValueError(
                "Slide joint target requires matching named affordance limits."
            )
        if not limits[0] <= target.position <= limits[1]:
            raise ValueError("Slide joint target is outside its declared limits.")
        state = context.scene.articulation_joints.get(
            (target.articulation_id, target.joint_name)
        )
        if state is None:
            raise ValueError(
                "The exact Slide articulation joint observation is absent."
            )
        position = state.position
        if position.shape == (1,):
            position = position.reshape(1, 1).expand(context.batch_size, 1)
        if (
            position.shape != (context.batch_size, 1)
            or position.device != context.robot.qpos.device
        ):
            raise ValueError(
                "Slide joint observations must have shape (B, 1) on the planning device."
            )
        position = position[:, 0]
        valid = (
            torch.isfinite(position)
            & (position >= limits[0] - target.tolerance)
            & (position <= limits[1] + target.tolerance)
        )
        if state.valid_mask is not None:
            valid &= state.valid_mask
        delta = (target.position - position) * target.axis_sign
        reached = valid & (delta.abs() <= target.tolerance)
        return (
            torch.where(valid & ~reached, delta, torch.zeros_like(delta)),
            valid,
            reached,
        )

    @staticmethod
    def _require_slide_affordance(
        semantics: ObjectSemantics,
    ) -> SlideAffordance:
        affordance = semantics.affordance
        if not isinstance(affordance, SlideAffordance):
            raise ValueError("Slide requires a SlideAffordance.")
        return affordance

    @staticmethod
    def _motion_segment_lengths(
        sample_count: int,
        hand_interp_steps: int,
        *,
        direction: Literal["pull", "push"],
    ) -> tuple[int, ...]:
        motion_segment_count = 3 if direction == "pull" else 4
        motion_count = sample_count - 2 * hand_interp_steps
        if motion_count < 2 * motion_segment_count:
            raise ValueError(
                "Not enough waypoints for Slide. Increase "
                "sample_count or decrease hand_interp_steps."
            )
        base, remainder = divmod(motion_count, motion_segment_count)
        return tuple(
            base + (index < remainder) for index in range(motion_segment_count)
        )

    def _plan_pose_segment(
        self,
        target_pose: torch.Tensor,
        start_qpos: torch.Tensor,
        control_part: str,
        request: ResolvedActionRequest[
            SlideGoal,
            SlideOptions,
        ],
        sample_count: int,
        *,
        interpolation_dt: float,
        cartesian_linear: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        result = self.motion_generator.generate(
            build_pose_plan_states(target_pose),
            options=request.motion_policy.to_motion_gen_options(
                start_qpos=start_qpos,
                control_part=control_part,
                sample_count=sample_count,
                interpolation_dt=interpolation_dt,
                cartesian_linear=cartesian_linear,
            ),
        )
        assert isinstance(result.success, torch.Tensor)
        assert result.positions is not None
        return result.success, result.positions


__all__ = [
    "Slide",
    "SlideGoal",
    "SlideJointTarget",
    "SlideOptions",
]
