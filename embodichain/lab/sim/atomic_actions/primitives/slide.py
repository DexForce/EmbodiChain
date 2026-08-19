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
from dataclasses import dataclass
from typing import ClassVar, Literal

import torch

from embodichain.lab.sim.atomic_actions.affordance import SlideAffordance
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
from embodichain.lab.sim.atomic_actions.plans import ActionPlan, normalize_success_mask
from embodichain.lab.sim.atomic_actions.primitives._helpers import arm_qpos_from_state
from embodichain.lab.sim.atomic_actions.requirements import (
    ActionBindingRoute,
    CARTESIAN_POSE_CAPABILITY,
    DisjointSlotEndpoints,
    GRASP_CAPABILITY,
    SkillBindingContract,
    SkillEndpointRequirement,
    SkillResourceSlot,
)
from embodichain.lab.sim.atomic_actions.state import PlanningContext
from embodichain.lab.sim.atomic_actions.trajectory_ops import (
    axis_translation_keyframes,
    build_pose_plan_states,
    interpolate_hand_qpos,
    resolve_pose_target,
    translate_pose_world,
)


@dataclass(frozen=True, slots=True, eq=False)
class SlideGoal(ObjectActionGoal):
    """Translating articulation link described by a slide affordance."""

    goal_kind: ClassVar[str] = "slide"

    target_pose: PoseGoalValue
    """Link pose snapshot or late-bound stable scene-entity reference."""

    def __post_init__(self) -> None:
        ObjectActionGoal.__post_init__(self)
        validate_pose_goal(self.target_pose, "target_pose", allow_waypoints=False)


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
    manipulator_roles: ClassVar[tuple[str, ...]] = ("primary",)
    end_effector_roles: ClassVar[tuple[str, ...]] = ("primary",)
    open_loop: ClassVar[bool] = True
    binding_contract: ClassVar[SkillBindingContract] = SkillBindingContract(
        slots=(
            SkillResourceSlot(
                slot_id="primary",
                endpoints=(
                    SkillEndpointRequirement(
                        endpoint_id="motion",
                        capabilities=frozenset({CARTESIAN_POSE_CAPABILITY}),
                        route=ActionBindingRoute("manipulator", "primary"),
                    ),
                    SkillEndpointRequirement(
                        endpoint_id="grasp",
                        capabilities=frozenset({GRASP_CAPABILITY}),
                        required_commands={
                            OPEN_COMMAND: JointPositionCommand,
                            GRASP_COMMAND: JointPositionCommand,
                        },
                        route=ActionBindingRoute("end_effector", "primary"),
                    ),
                ),
                constraints=(DisjointSlotEndpoints(("motion", "grasp")),),
            ),
        ),
    )

    def __init__(
        self,
        default_options: SlideOptions | None = None,
    ) -> None:
        super().__init__(default_options)

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
        grasp_success, grasp_xpos, _ = affordance.get_best_grasp_poses(
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
            return self.failed_plan(
                request,
                context,
                message="Failed to resolve an articulated-part grasp pose.",
            )
        approach_xpos = translate_pose_world(
            grasp_xpos,
            -translation_axis_world * options.approach_distance,
        )
        translation_sign = -1.0 if options.direction == "pull" else 1.0
        translated_xpos = translate_pose_world(
            grasp_xpos,
            translation_axis_world * (translation_sign * options.translation_distance),
        )

        motion_lengths = self._motion_segment_lengths(
            request.motion_policy.sample_count,
            options.hand_interp_steps,
            direction=options.direction,
        )
        approach_success, approach_arm = self._plan_pose_segment(
            approach_xpos,
            start_arm_qpos,
            manipulator.name,
            request,
            motion_lengths[0],
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
            manipulator.name,
            request,
            motion_lengths[1],
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
            manipulator.name,
            request,
            motion_lengths[2],
            cartesian_linear=True,
        )
        success = grasp_success & approach_success & reach_success & translate_success

        return_arm: torch.Tensor | None = None
        if options.direction == "push":
            return_keyframes = axis_translation_keyframes(
                translated_xpos,
                approach_xpos,
                translation_axis_world,
                n_waypoints=motion_lengths[3] - 1,
            )
            return_success, return_arm = self._plan_pose_segment(
                return_keyframes,
                translate_arm[:, -1],
                manipulator.name,
                request,
                motion_lengths[3],
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
            (options.direction, translate_arm),
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

        return self.build_plan(
            request,
            context,
            success=success,
            trajectory=full,
            expected_effects=StateDelta(),
            segment_lengths=segment_lengths,
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
        cartesian_linear: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        result = self.motion_generator.generate(
            build_pose_plan_states(target_pose),
            options=request.motion_policy.to_motion_gen_options(
                start_qpos=start_qpos,
                control_part=control_part,
                sample_count=sample_count,
                cartesian_linear=cartesian_linear,
            ),
        )
        assert isinstance(result.success, torch.Tensor)
        assert result.positions is not None
        return result.success, result.positions


__all__ = [
    "Slide",
    "SlideGoal",
    "SlideOptions",
]
