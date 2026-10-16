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

from embodichain.compute.trajectory import retime_to_control_grid
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
from embodichain.lab.sim.atomic_actions.plans import (
    ActionPlan,
    PlannerDiagnostics,
    TimedTrajectory,
)
from embodichain.lab.sim.atomic_actions.policies import DynamicCollisionMode
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
        contact_sample = affordance.sample_press_pose(
            target_pose,
            press_position=options.press_position,
            sampling=context.affordance_sampling,
            env_ids=context.env_ids,
            key=(request.invocation_id or self.skill_id) + ":contact",
        )
        contact_xpos = contact_sample.poses.to(device=self.device, dtype=torch.float32)
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
        if request.motion_policy.strategy == "motion_gen":
            targets = torch.stack(
                (approach_xpos, contact_xpos, pressed_xpos, approach_xpos), dim=1
            )
            motion_options = request.motion_policy.to_motion_gen_options(
                start_qpos=start_arm_qpos,
                control_part=control_part,
                interpolation_dt=interpolation_dt,
            )
            motion_options.sample_count = None
            result = self.motion_generator.generate(
                build_pose_plan_states(targets), options=motion_options
            )
            assert isinstance(result.success, torch.Tensor)
            assert result.positions is not None
            motion_phases = self._split_motion_path(
                result.positions,
                result.dt,
                targets[:, :3],
                control_part,
                (n_approach, n_contact, n_press, n_retract),
                interpolation_dt,
            )
            success = contact_sample.success & result.success
        else:
            approach_success, approach_phase = self._plan_pose_segment(
                approach_xpos,
                start_arm_qpos,
                control_part,
                request,
                n_approach,
                interpolation_dt=interpolation_dt,
            )
            approach_arm = approach_phase[0]
            contact_keyframes = axis_translation_keyframes(
                approach_xpos,
                contact_xpos,
                contact_xpos[:, :3, 2],
                n_waypoints=n_contact - 1,
            )
            contact_success, contact_phase = self._plan_pose_segment(
                contact_keyframes,
                approach_arm[:, -1],
                control_part,
                request,
                n_contact,
                interpolation_dt=interpolation_dt,
                cartesian_linear=True,
            )
            contact_arm = contact_phase[0]
            press_keyframes = axis_translation_keyframes(
                contact_xpos,
                pressed_xpos,
                contact_xpos[:, :3, 2],
                n_waypoints=n_press - 1,
            )
            press_success, press_phase = self._plan_pose_segment(
                press_keyframes,
                contact_arm[:, -1],
                control_part,
                request,
                n_press,
                interpolation_dt=interpolation_dt,
                cartesian_linear=True,
            )
            press_arm = press_phase[0]
            retract_keyframes = axis_translation_keyframes(
                pressed_xpos,
                approach_xpos,
                contact_xpos[:, :3, 2],
                n_waypoints=n_retract - 1,
            )
            retract_success, retract_phase = self._plan_pose_segment(
                retract_keyframes,
                press_arm[:, -1],
                control_part,
                request,
                n_retract,
                interpolation_dt=interpolation_dt,
                cartesian_linear=True,
            )
            success = (
                contact_sample.success
                & approach_success
                & contact_success
                & press_success
                & retract_success
            )
            motion_phases = (approach_phase, contact_phase, press_phase, retract_phase)

        parts = (hand_close, *(phase[0] for phase in motion_phases))
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

        hand_dt = torch.full(hand_close.shape[:2], interpolation_dt, device=self.device)
        hand_dt[:, 0] = 0.0
        phase_intervals = [hand_dt]
        for arm, dt in motion_phases:
            stop = offset + arm.shape[1]
            full[:, offset:stop, arm_joint_ids] = arm
            full[:, offset:stop, hand_joint_ids] = hand_grasp_qpos.unsqueeze(1)
            # Keep one control-period hold at each phase junction, as with the
            # original uniform composition, without replacing backend timing.
            dt = dt.clone()
            dt[:, 0] = interpolation_dt
            phase_intervals.append(dt)
            offset = stop

        validation_messages: tuple[str, ...] = ()
        if self.motion_generator.supports_joint_trajectory_validation:
            obstacle_poses = getattr(
                request.motion_policy.plan_opts, "dynamic_obstacle_poses", None
            )
            dynamic_obstacle_ids = self.motion_generator.dynamic_collision_entity_ids
            obstacle_poses = {} if obstacle_poses is None else dict(obstacle_poses)
            missing_ids = set(dynamic_obstacle_ids).difference(obstacle_poses)
            if (
                missing_ids
                and request.motion_policy.dynamic_collision_mode
                is not DynamicCollisionMode.OFF
            ):
                # ik_interp bypasses backend planning options, but validating its
                # final samples still needs the live collision-world snapshot.
                scene_obstacle_poses = context.scene.collision_obstacle_poses(
                    batch_size=context.batch_size,
                    device=context.robot.qpos.device,
                    dtype=context.robot.qpos.dtype,
                )
                obstacle_poses.update(
                    {
                        entity_id: scene_obstacle_poses[entity_id]
                        for entity_id in missing_ids
                        if entity_id in scene_obstacle_poses
                    }
                )
                missing_ids.difference_update(obstacle_poses)
            if missing_ids:
                # Exact validation cannot certify any row with an incomplete
                # world. OFF deliberately forbids filling gaps from the scene.
                success = torch.zeros_like(success)
                validation_messages = (
                    "Press cannot validate the final trajectory: missing dynamic "
                    f"obstacle poses {sorted(missing_ids)} "
                    f"(dynamic_collision_mode={request.motion_policy.dynamic_collision_mode.value}).",
                )
            else:
                validity = self.motion_generator.validate_joint_trajectory(
                    full[:, :, arm_joint_ids],
                    control_part=control_part,
                    obstacle_poses=(
                        {
                            entity_id: obstacle_poses[entity_id]
                            for entity_id in dynamic_obstacle_ids
                        }
                        if dynamic_obstacle_ids
                        else None
                    ),
                )
                success = success & validity.all(dim=1)
        elif (
            request.motion_policy.strategy == "motion_gen"
            and self.motion_generator.collision_world_info is not None
        ):
            raise ValueError(
                "Press requires joint-trajectory validation when retiming a "
                "collision-aware backend path."
            )

        return self.build_plan(
            request,
            context,
            success=success,
            trajectory=TimedTrajectory.from_positions(
                full,
                env_ids=context.env_ids,
                dt=torch.cat(phase_intervals, dim=1),
            ),
            expected_effects=StateDelta(),
            diagnostics=PlannerDiagnostics(
                backend=self.planning_services.planner_name,
                messages=validation_messages,
                metadata={"affordance_sample": contact_sample.metadata},
            ),
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

    def _split_motion_path(
        self,
        trajectory: torch.Tensor,
        dt: torch.Tensor | None,
        split_poses: torch.Tensor,
        control_part: str,
        sample_counts: tuple[int, int, int, int],
        control_dt: float,
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor],
    ]:
        """Split at ordered native boundaries and safely retime each phase."""
        if dt is None:
            raise ValueError("Press requires explicit backend trajectory timing.")
        poses = self.robot.compute_batch_fk(
            qpos=trajectory, name=control_part, to_matrix=True
        )
        position_error = torch.linalg.vector_norm(
            poses[:, None, :, :3, 3] - split_poses[:, :, None, :3, 3], dim=-1
        )
        relative_rotation = torch.matmul(
            poses[:, None, :, :3, :3].transpose(-1, -2),
            split_poses[:, :, None, :3, :3],
        )
        trace = relative_rotation.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        error = position_error + torch.acos(((trace - 1.0) * 0.5).clamp(-1.0, 1.0))
        cost = error[:, 0]
        predecessors = []
        for goal_index in (1, 2):
            prefix_cost, prefix_index = torch.cummin(cost, dim=1)
            predecessors.append(prefix_index)
            cost = prefix_cost + error[:, goal_index]
        index = cost.argmin(dim=1)
        boundaries = [index]
        for predecessor in reversed(predecessors):
            index = predecessor.gather(1, index[:, None]).squeeze(1)
            boundaries.append(index)
        boundaries.reverse()
        boundaries = (
            torch.stack(
                (
                    torch.zeros_like(index),
                    *boundaries,
                    torch.full_like(index, trajectory.shape[1] - 1),
                ),
                dim=1,
            )
            .cpu()
            .tolist()
        )
        segments = []
        for segment_index, count in enumerate(sample_counts):
            native_count = max(
                indices[segment_index + 1] - indices[segment_index] + 1
                for indices in boundaries
            )
            positions = trajectory.new_empty(
                trajectory.shape[0], native_count, trajectory.shape[2]
            )
            intervals = dt.new_zeros(trajectory.shape[0], native_count)
            for row, indices in enumerate(boundaries):
                start, stop = indices[segment_index : segment_index + 2]
                path = trajectory[row, start : stop + 1]
                positions[row] = path[-1]
                positions[row, : path.shape[0]] = path
                intervals[row, : path.shape[0]] = dt[row, start : stop + 1]
            if segment_index:
                # The shared boundary has already arrived in the prior phase.
                intervals[:, 0] = 0.0
            segments.append(
                self._retime_motion_phase(positions, intervals, count, control_dt)
            )
        return segments[0], segments[1], segments[2], segments[3]

    @staticmethod
    def _retime_motion_phase(
        positions: torch.Tensor,
        dt: torch.Tensor | None,
        sample_count: int,
        control_dt: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Use the requested count as a minimum, preserving source time profile."""
        if dt is None:
            raise ValueError("Press requires explicit backend trajectory timing.")
        # An initial arrival offset is a hold, not time that may be discarded.
        positions = torch.cat((positions[:, :1], positions), dim=1)
        dt = torch.cat((dt.new_zeros(dt.shape[0], 1), dt), dim=1)
        duration = dt.sum(dim=1)
        stationary = (positions == positions[:, :1]).all(dim=2).all(dim=1)
        if ((duration == 0) & ~stationary).any():
            raise ValueError("A zero-duration Press phase cannot change position.")
        minimum_duration = (sample_count - 1) * control_dt
        scale = torch.maximum(
            torch.ones_like(duration),
            minimum_duration / torch.where(duration > 0, duration, 1.0),
        )
        scaled_dt = dt * torch.where(duration > 0, scale, 1.0)[:, None]
        scaled_dt[:, -1] = torch.where(duration > 0, scaled_dt[:, -1], minimum_duration)
        retimed, _, intervals, _ = retime_to_control_grid(
            positions, scaled_dt, control_dt
        )
        return retimed, intervals

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
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
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
        return result.success, self._retime_motion_phase(
            result.positions, result.dt, sample_count, interpolation_dt
        )


__all__ = ["Press", "PressGoal", "PressOptions"]
