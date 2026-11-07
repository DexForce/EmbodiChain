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
from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import ClassVar

import torch

from embodichain.compute.trajectory import retime_to_control_grid
from embodichain.compute.trajectory.limits import joint_velocity_validity
from embodichain.lab.sim.motion.ik import (
    resolve_continuous_ik_path,
    sample_cartesian_targets,
)
from embodichain.lab.sim.motion.limits import authored_joint_velocity_limits
from embodichain.lab.sim.atomic_actions.primitives._helpers import (
    arm_qpos_from_state,
    find_symmetric_nearest_xpos,
)
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
    PlanQualification,
    PlanningFailure,
    TimedTrajectory,
)
from embodichain.lab.sim.atomic_actions.policies import (
    DynamicCollisionMode,
    PlanningRecovery,
)
from embodichain.lab.sim.atomic_actions.requirements import (
    CARTESIAN_POSE_CAPABILITY,
    FORWARD_KINEMATICS_CAPABILITY,
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
from embodichain.lab.sim.atomic_actions.primitives._binding_contracts import (
    make_manipulation_slot,
)


def validate_press_extra_distance(value: float | None) -> None:
    """Validate an optional declaration-owned addition to an authored press stroke.

    Args:
        value: Additional travel in metres, or None to retain the declaration's
            existing selection. Exact Python integers/floats in ``[0, 0.010]``
            are accepted; booleans and non-finite values are rejected.

    Raises:
        ValueError: The value is outside the supported declaration contract.

    This does not change :class:`PressOptions` defaults or certify physical
    travel. The caller still owns the authored stroke and measured acceptance.
    """
    if value is not None and (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or not 0.0 <= value <= 0.010
    ):
        raise ValueError("press_extra_distance must be between 0 and 0.010 m or None.")


@dataclass(frozen=True, slots=True, eq=False)
class PressGoal(ObjectActionGoal):
    """Target object described by a press affordance."""

    target_pose: PoseGoalValue
    """Target pose snapshot or late-bound stable scene-entity reference."""

    def __post_init__(self) -> None:
        ObjectActionGoal.__post_init__(self)
        validate_pose_goal(self.target_pose, "target_pose", allow_waypoints=False)


@dataclass(frozen=True, slots=True)
class PressPlanningRecovery(PlanningRecovery):
    """Failure-only B1 Press planning rules with the fixed source attempt budget.

    The two complete plans include the nominal plan and at most one declared
    contact-roll alternative. Failed Cartesian phases may use the shared bounded
    IK continuation on unchanged targets and time. Inner IK search limits remain
    owned by the motion service. Separate dense-grid engine attempts each get a
    fresh local scope; this declaration does not change their RNG boundary.
    """

    kind: str = field(default="press_bounded", init=False)
    skill_id: str = field(default="press", init=False)
    revision: int = field(default=1, init=False)
    max_complete_plans: int = field(default=2, init=False)
    retry_failed_cartesian_phases: bool = field(default=True, init=False)
    single_environment_only: bool = field(default=True, init=False)
    qualify_emitted_velocity: bool = field(default=True, init=False)


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

    contact_roll_alternatives: tuple[float, ...] = ()
    """Allowed extra contact-frame rolls in radians; point and press axis stay fixed.

    Planning these alternatives requires an explicit PressPlanningRecovery.
    The source-compatible declaration supplies only ``(pi / 2,)``.
    """

    def __post_init__(self) -> None:
        if not isinstance(self.contact_roll_alternatives, (tuple, list)):
            raise TypeError("contact_roll_alternatives must be a sequence of radians.")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or abs(value) > math.pi
            for value in self.contact_roll_alternatives
        ):
            raise ValueError(
                "contact_roll_alternatives must be finite radians in [-pi, pi]."
            )
        object.__setattr__(
            self,
            "contact_roll_alternatives",
            tuple(float(value) for value in self.contact_roll_alternatives),
        )
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


class _RolledPressAffordance(PressAffordance):
    def __init__(self, source: PressAffordance, roll: float) -> None:
        super().__init__(
            object_label=source.object_label,
            custom_config=deepcopy(source.custom_config),
            press_axis=source.press_axis,
            press_position=source.press_position,
        )
        self._roll = roll

    def get_press_pose(
        self,
        target_pose: torch.Tensor,
        press_position: tuple[float, float, float] | None = None,
    ) -> torch.Tensor:
        pose = super().get_press_pose(target_pose, press_position=press_position)
        x, y = pose[:, :3, 0].clone(), pose[:, :3, 1].clone()
        if self._roll == math.pi / 2:
            pose[:, :3, 0], pose[:, :3, 1] = y, -x
        else:
            cosine, sine = math.cos(self._roll), math.sin(self._roll)
            pose[:, :3, 0] = cosine * x + sine * y
            pose[:, :3, 1] = -sine * x + cosine * y
        return pose


class Press(AtomicAction[PressGoal, PressOptions]):
    """Open-loop motion primitive that approaches, presses, and retracts."""

    skill_id: ClassVar[str] = "press"
    GoalType: ClassVar[type] = PressGoal
    OptionsType: ClassVar[type] = PressOptions
    PlanningRecoveryType: ClassVar[type[PlanningRecovery]] = PressPlanningRecovery
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

    def _plan(
        self,
        request: ResolvedActionRequest[PressGoal, PressOptions],
        context: PlanningContext,
    ) -> ActionPlan:
        self.descriptor().validate_planning_recovery(request.recovery_policy)
        recovery = request.recovery_policy.planning_recovery
        alternatives = request.skill_options.contact_roll_alternatives
        if recovery is None:
            if alternatives:
                raise ValueError(
                    "Press contact-roll alternatives require PressPlanningRecovery."
                )
            return self._plan_nominal(request, context)
        assert isinstance(recovery, PressPlanningRecovery)
        if context.batch_size != 1:
            raise ValueError("PressPlanningRecovery supports exactly one environment.")
        if len(alternatives) + 1 > recovery.max_complete_plans:
            raise ValueError(
                "Press recovery permits at most 2 complete plans, including nominal."
            )
        original = self._plan_checked(request, context)
        affordance = request.goal.semantics.affordance
        if (
            original.plan_success.all()
            or type(affordance) is not PressAffordance
            or not alternatives
        ):
            return original
        roll = alternatives[0]
        selected = replace(
            request,
            goal=replace(
                request.goal,
                semantics=replace(
                    request.goal.semantics,
                    affordance=_RolledPressAffordance(affordance, roll),
                ),
            ),
        )
        # The source does not restore RNG between its two complete plans.
        recovered = self._plan_checked(selected, context)
        result = recovered if recovered.plan_success.all() else original
        if result.diagnostics is not None:
            result = replace(
                result,
                diagnostics=replace(
                    result.diagnostics,
                    metadata={
                        **result.diagnostics.metadata,
                        "press_roll_recovery": {
                            "maximum_complete_plans": recovery.max_complete_plans,
                            "complete_plans": 2,
                            "roll_rad": roll,
                            "accepted": bool(recovered.plan_success.all()),
                            "alternate_phases": (
                                recovered.diagnostics.metadata.get(
                                    "press_continuous_ik", []
                                )
                                if recovered.diagnostics is not None
                                else []
                            ),
                            "scope": "same_contact_point_axis_and_resource_not_physical_evidence",
                        },
                    },
                ),
            )
        return result

    def _plan_checked(
        self,
        request: ResolvedActionRequest[PressGoal, PressOptions],
        context: PlanningContext,
    ) -> ActionPlan:
        records: list[dict[str, object]] = []
        plan = self._plan_nominal(request, context, recovery_records=records)
        if records and plan.diagnostics is not None:
            plan = replace(
                plan,
                diagnostics=replace(
                    plan.diagnostics,
                    metadata={
                        **plan.diagnostics.metadata,
                        "press_continuous_ik": records,
                    },
                ),
            )
        qualified = self._qualify_final_plan(request, context, plan)
        if qualified is not None and not qualified.valid_mask.all():
            return self.failed_plan(
                request,
                context,
                message="Press exceeds final resampled joint velocity limits.",
            )
        return plan

    def _qualify_final_plan(
        self,
        request: ResolvedActionRequest[PressGoal, PressOptions],
        context: PlanningContext,
        plan: ActionPlan,
    ) -> PlanQualification | None:
        if (
            request.recovery_policy.planning_recovery is None
            or not plan.plan_success.any()
        ):
            return None
        trajectory = plan.joint_trajectory
        if trajectory is None:
            return PlanQualification(
                torch.zeros_like(plan.plan_success),
                PlannerDiagnostics(
                    backend=self.planning_services.planner_name,
                    messages=(
                        "Press planning recovery requires an emitted joint trajectory.",
                    ),
                    failure=PlanningFailure(
                        "press_final_joint_motion_missing", retryable=False
                    ),
                ),
            )
        limits = authored_joint_velocity_limits(self.robot, None).to(
            trajectory.positions
        )
        valid = joint_velocity_validity(trajectory.positions, trajectory.dt, limits)
        mask = torch.full_like(plan.plan_success, bool(valid.all()))
        diagnostics = None
        if not mask.all():
            diagnostics = PlannerDiagnostics(
                backend=self.planning_services.planner_name,
                failure=PlanningFailure(
                    "press_final_velocity_invalid", retryable=False
                ),
                metadata={
                    "press_final_velocity": {
                        "valid_rows": valid.tolist(),
                        "scope": "final_emitted_command_grid",
                    }
                },
            )
        return PlanQualification(mask, diagnostics)

    def _plan_nominal(
        self,
        request: ResolvedActionRequest[PressGoal, PressOptions],
        context: PlanningContext,
        *,
        recovery_records: list[dict[str, object]] | None = None,
    ) -> ActionPlan:
        """Plan close, approach, press, and retract without stepping simulation."""
        if request.motion_policy.strategy != "ik_interp":
            raise ValueError(
                "Press requires strategy='ik_interp' because its contact and "
                "press phases must preserve every Cartesian path sample."
            )
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
        contact_xpos = find_symmetric_nearest_xpos(
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
        approach_success, approach_phase = self._plan_press_phase(
            approach_xpos,
            start_arm_qpos,
            control_part,
            request,
            n_approach,
            interpolation_dt=interpolation_dt,
            phase_name="approach",
            recovery_records=recovery_records,
        )
        approach_arm = approach_phase[0]
        contact_keyframes = axis_translation_keyframes(
            approach_xpos,
            contact_xpos,
            contact_xpos[:, :3, 2],
            n_waypoints=n_contact - 1,
        )
        contact_success, contact_phase = self._plan_press_phase(
            contact_keyframes,
            approach_arm[:, -1],
            control_part,
            request,
            n_contact,
            interpolation_dt=interpolation_dt,
            cartesian_linear=True,
            phase_name="contact",
            recovery_records=recovery_records,
        )
        contact_arm = contact_phase[0]
        press_keyframes = axis_translation_keyframes(
            contact_xpos,
            pressed_xpos,
            contact_xpos[:, :3, 2],
            n_waypoints=n_press - 1,
        )
        press_success, press_phase = self._plan_press_phase(
            press_keyframes,
            contact_arm[:, -1],
            control_part,
            request,
            n_press,
            interpolation_dt=interpolation_dt,
            cartesian_linear=True,
            phase_name="press",
            recovery_records=recovery_records,
        )
        press_arm = press_phase[0]
        retract_keyframes = axis_translation_keyframes(
            pressed_xpos,
            approach_xpos,
            contact_xpos[:, :3, 2],
            n_waypoints=n_retract - 1,
        )
        retract_success, retract_phase = self._plan_press_phase(
            retract_keyframes,
            press_arm[:, -1],
            control_part,
            request,
            n_retract,
            interpolation_dt=interpolation_dt,
            cartesian_linear=True,
            phase_name="retract",
            recovery_records=recovery_records,
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

    def _plan_press_phase(
        self,
        target_pose: torch.Tensor,
        start_qpos: torch.Tensor,
        control_part: str,
        request: ResolvedActionRequest[PressGoal, PressOptions],
        sample_count: int,
        *,
        interpolation_dt: float,
        phase_name: str,
        recovery_records: list[dict[str, object]] | None,
        cartesian_linear: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        original = self._plan_pose_segment(
            target_pose,
            start_qpos,
            control_part,
            request,
            sample_count,
            interpolation_dt=interpolation_dt,
            cartesian_linear=cartesian_linear,
        )
        if (
            recovery_records is None
            or original[0].all()
            or start_qpos.shape[0] != 1
            or request.motion_policy.strategy != "ik_interp"
        ):
            return original
        start = self.robot.compute_fk(
            qpos=start_qpos, name=control_part, to_matrix=True
        )
        targets = sample_cartesian_targets(
            start, build_pose_plan_states(target_pose), sample_count
        )
        options = request.motion_policy.to_motion_gen_options(
            start_qpos=start_qpos,
            control_part=control_part,
            sample_count=sample_count,
            interpolation_dt=interpolation_dt,
            cartesian_linear=True,
        )
        recovered = resolve_continuous_ik_path(
            self.motion_generator, targets, options, operation="Press"
        )
        recovery_records.append(
            {
                "phase": phase_name,
                "sample_count": sample_count,
                "control_dt": interpolation_dt,
                "accepted": recovered is not None,
                "scope": "failure_only_seeded_ik_same_targets_and_time_not_physical_evidence",
            }
        )
        if recovered is None:
            return original
        assert recovered.positions is not None
        return recovered.success, self._retime_motion_phase(
            recovered.positions, recovered.dt, sample_count, interpolation_dt
        )

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
        positions, dt = result.positions, result.dt
        if (
            request.recovery_policy.planning_recovery is not None
            and not result.success.all()
        ):
            candidate = result.rejected_candidate
            if candidate is not None:
                # Inspect rejected samples only inside bounded replanning. The
                # original false success mask is retained and never executed.
                positions, dt = candidate.positions, candidate.dt
            elif positions is None:
                positions = start_qpos[:, None].expand(-1, sample_count, -1).clone()
                dt = positions.new_full(positions.shape[:2], interpolation_dt)
                dt[:, 0] = 0.0
        assert positions is not None
        return result.success, self._retime_motion_phase(
            positions, dt, sample_count, interpolation_dt
        )


__all__ = [
    "Press",
    "PressGoal",
    "PressOptions",
    "PressPlanningRecovery",
    "validate_press_extra_distance",
]
