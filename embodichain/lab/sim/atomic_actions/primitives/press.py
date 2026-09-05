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

"""Press atomic action implementation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

import torch

from embodichain.lab.sim.atomic_actions.primitives._helpers import arm_qpos_from_state
from embodichain.lab.sim.atomic_actions.affordance import PressAffordance
from embodichain.lab.sim.atomic_actions.bindings import JointPositionTarget
from embodichain.lab.sim.atomic_actions.control import (
    GRASP_COMMAND,
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
from embodichain.lab.sim.atomic_actions.plans import ActionPlan, TimedTrajectory
from embodichain.lab.sim.atomic_actions.requirements import (
    CARTESIAN_POSE_CAPABILITY,
    FORWARD_KINEMATICS_CAPABILITY,
    SkillBindingContract,
)
from embodichain.utils.math import get_relative_rotation
from embodichain.lab.sim.atomic_actions.state import PlanningContext
from embodichain.lab.sim.atomic_actions.trajectory_ops import (
    axis_translation_keyframes,
    build_pose_plan_states,
    interpolate_hand_qpos,
    resolve_pose_target,
    translate_pose_world,
)
from embodichain.lab.sim.atomic_actions.primitives._binding_contracts import (
    make_manipulation_slot,
)


@dataclass(frozen=True, slots=True, eq=False)
class PressGoal(ObjectActionGoal):
    """Target object described by a press affordance."""

    target_pose: PoseGoalValue
    """Target pose snapshot or late-bound stable scene-entity reference."""

    def __post_init__(self) -> None:
        ObjectActionGoal.__post_init__(self)
        validate_pose_goal(self.target_pose, "target_pose", allow_waypoints=False)


@dataclass(frozen=True, slots=True, eq=False)
class PressOptions(ActionOptions):
    """Per-invocation pressing behavior."""

    hand_interp_steps: int = 5
    """Number of waypoints used to close the hand."""

    approach_distance: float = 0.1
    """Distance from the press position opposite the press direction."""

    press_distance: float = 0.05
    """Distance traveled into the target along its press axis."""

    press_position: tuple[float, float, float] | None = None
    """Optional local-frame position overriding the affordance press position."""

    def __post_init__(self) -> None:
        if self.hand_interp_steps < 1:
            raise ValueError("hand_interp_steps must be at least 1.")
        if not math.isfinite(self.approach_distance):
            raise ValueError("approach_distance must be finite.")
        if self.approach_distance < 0.0:
            raise ValueError("approach_distance must be non-negative.")
        if not math.isfinite(self.press_distance):
            raise ValueError("press_distance must be finite.")
        if self.press_distance <= 0.0:
            raise ValueError("press_distance must be positive.")
        if self.press_position is not None:
            position = torch.as_tensor(self.press_position, dtype=torch.float32)
            if position.shape != (3,) or not torch.isfinite(position).all():
                raise ValueError("press_position must be a finite (x, y, z) tuple.")
            object.__setattr__(
                self,
                "press_position",
                tuple(float(component) for component in position),
            )


class Press(AtomicAction[PressGoal, PressOptions]):
    """Open-loop motion primitive that approaches, presses, and retracts."""

    skill_id: ClassVar[str] = "press"
    GoalType: ClassVar[type] = PressGoal
    OptionsType: ClassVar[type] = PressOptions
    open_loop: ClassVar[bool] = True
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
                grasp_commands={GRASP_COMMAND: JointPositionCommand},
            ),
        ),
    )

    def _on_bind(self) -> None:
        """Resolve dimensions owned by the engine's robot."""
        self.num_envs = self.robot.get_qpos().shape[0]
        self.robot_dof = self.robot.dof

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
        request: ResolvedActionRequest[PressGoal, PressOptions],
        context: PlanningContext,
    ) -> ActionPlan:
        """Plan close, approach, press, and retract without stepping simulation."""
        target = self.require_goal(request)
        affordance = self._require_press_affordance(target.semantics)
        options = request.skill_options
        interpolation_dt = context.require_control_dt()
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
        start_hand_qpos = context.last_qpos[:, hand_joint_ids]
        hand_grasp_qpos = grasp.joint_positions(
            GRASP_COMMAND,
            num_envs=context.batch_size,
            device=self.device,
            dtype=context.robot.qpos.dtype,
        )

        target_pose = resolve_pose_target(
            resolve_pose_goal(target.target_pose, context, name="target_pose"),
            num_envs=self.num_envs,
            device=self.device,
        )
        contact_xpos = affordance.get_press_pose(
            target_pose,
            press_position=options.press_position,
        ).to(device=self.device, dtype=torch.float32)
        contact_xpos = self._find_symmetric_nearest_xpos(
            contact_xpos,
            reference_xpos=self.robot.compute_fk(
                qpos=start_arm_qpos, name=control_part, to_matrix=True
            ),
        )
        approach_xpos = translate_pose_world(
            contact_xpos,
            -contact_xpos[:, :3, 2] * options.approach_distance,
        )
        pressed_xpos = translate_pose_world(
            contact_xpos,
            contact_xpos[:, :3, 2] * options.press_distance,
        )
        n_approach, n_contact, n_press, n_retract = self._motion_segment_lengths(
            request.motion_policy.sample_count,
            options.hand_interp_steps,
        )
        hand_close = interpolate_hand_qpos(
            start_hand_qpos,
            hand_grasp_qpos,
            n_waypoints=options.hand_interp_steps,
        )
        approach_success, approach_arm = self._plan_pose_segment(
            approach_xpos,
            start_arm_qpos,
            control_part,
            request,
            n_approach,
            interpolation_dt=interpolation_dt,
        )
        contact_keyframes = axis_translation_keyframes(
            approach_xpos,
            contact_xpos,
            contact_xpos[:, :3, 2],
            n_waypoints=n_contact - 1,
        )
        contact_success, contact_arm = self._plan_pose_segment(
            contact_keyframes,
            approach_arm[:, -1],
            control_part,
            request,
            n_contact,
            interpolation_dt=interpolation_dt,
            cartesian_linear=True,
        )
        press_keyframes = axis_translation_keyframes(
            contact_xpos,
            pressed_xpos,
            contact_xpos[:, :3, 2],
            n_waypoints=n_press - 1,
        )
        press_success, press_arm = self._plan_pose_segment(
            press_keyframes,
            contact_arm[:, -1],
            control_part,
            request,
            n_press,
            interpolation_dt=interpolation_dt,
            cartesian_linear=True,
        )
        retract_keyframes = axis_translation_keyframes(
            pressed_xpos,
            approach_xpos,
            contact_xpos[:, :3, 2],
            n_waypoints=n_retract - 1,
        )
        retract_success, retract_arm = self._plan_pose_segment(
            retract_keyframes,
            press_arm[:, -1],
            control_part,
            request,
            n_retract,
            interpolation_dt=interpolation_dt,
            cartesian_linear=True,
        )
        success = approach_success & contact_success & press_success & retract_success

        parts = (hand_close, approach_arm, contact_arm, press_arm, retract_arm)
        lengths = tuple(part.shape[1] for part in parts)
        full = torch.empty(
            (self.num_envs, sum(lengths), self.robot_dof),
            dtype=context.robot.qpos.dtype,
            device=self.device,
        )
        full[:] = context.last_qpos.unsqueeze(1)
        offset = 0

        stop = offset + hand_close.shape[1]
        full[:, offset:stop, arm_joint_ids] = start_arm_qpos.unsqueeze(1)
        full[:, offset:stop, hand_joint_ids] = hand_close
        offset = stop

        for arm in (approach_arm, contact_arm, press_arm, retract_arm):
            stop = offset + arm.shape[1]
            full[:, offset:stop, arm_joint_ids] = arm
            full[:, offset:stop, hand_joint_ids] = hand_grasp_qpos.unsqueeze(1)
            offset = stop

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
            segment_lengths={
                "close": lengths[0],
                "approach": lengths[1],
                "contact": lengths[2],
                "press": lengths[3],
                "retract": lengths[4],
            },
        )

    @staticmethod
    def _require_press_affordance(
        semantics: ObjectSemantics,
    ) -> PressAffordance:
        affordance = semantics.affordance
        if not isinstance(affordance, PressAffordance):
            raise ValueError("Press requires a PressAffordance.")
        return affordance

    @staticmethod
    def _motion_segment_lengths(
        sample_count: int,
        hand_interp_steps: int,
    ) -> tuple[int, int, int, int]:
        motion_count = sample_count - hand_interp_steps
        if motion_count < 8:
            raise ValueError(
                "Not enough waypoints for Press. Increase sample_count or "
                "decrease hand_interp_steps."
            )
        base, remainder = divmod(motion_count, 4)
        values = [base + (index < remainder) for index in range(4)]
        return values[0], values[1], values[2], values[3]

    def _plan_pose_segment(
        self,
        target_pose: torch.Tensor,
        start_qpos: torch.Tensor,
        control_part: str,
        request: ResolvedActionRequest[PressGoal, PressOptions],
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


__all__ = ["Press", "PressGoal", "PressOptions"]
