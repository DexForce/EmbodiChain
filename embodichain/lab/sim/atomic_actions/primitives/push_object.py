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

"""Planar rigid-object pushing atomic action."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

import torch

from embodichain.lab.sim.atomic_actions.bindings import JointPositionTarget
from embodichain.lab.sim.atomic_actions.control import (
    GRASP_COMMAND,
    JointPositionCommand,
)
from embodichain.lab.sim.atomic_actions.core import AtomicAction
from embodichain.lab.sim.atomic_actions.effects import StateDelta
from embodichain.lab.sim.atomic_actions.goals import (
    ObjectActionGoal,
    PoseGoalValue,
    SceneEntityPose,
    collect_scene_dependencies,
    resolve_pose_goal,
    validate_pose_goal,
)
from embodichain.lab.sim.atomic_actions.invocation import (
    ActionOptions,
    ResolvedActionRequest,
)
from embodichain.lab.sim.atomic_actions.plans import ActionPlan, TimedTrajectory
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


def _validate_transform(value: torch.Tensor, name: str) -> torch.Tensor:
    """Validate, copy, and return one proper SE(3) transform."""
    validate_pose_goal(value, name, allow_waypoints=False)
    if value.shape != (4, 4) or not torch.isfinite(value).all():
        raise ValueError(f"{name} must be one finite 4x4 transform.")
    transform = value.to(dtype=torch.float64)
    if not torch.allclose(
        transform[3],
        torch.tensor(
            (0.0, 0.0, 0.0, 1.0),
            dtype=torch.float64,
            device=transform.device,
        ),
        atol=1.0e-6,
        rtol=0.0,
    ):
        raise ValueError(f"{name} must have bottom row [0, 0, 0, 1].")
    rotation = transform[:3, :3]
    if not torch.allclose(
        rotation.T @ rotation,
        torch.eye(3, dtype=torch.float64, device=transform.device),
        atol=1.0e-6,
        rtol=0.0,
    ) or not torch.isclose(
        torch.linalg.det(rotation),
        torch.tensor(1.0, dtype=torch.float64, device=transform.device),
        atol=1.0e-6,
        rtol=0.0,
    ):
        raise ValueError(f"{name} must contain a proper SE(3) rotation.")
    return value.clone()


@dataclass(frozen=True, slots=True, eq=False)
class PushObjectGoal(ObjectActionGoal):
    """Push one rigid object toward a target pose on the target support plane."""

    target_pose: PoseGoalValue
    """Desired object pose or a late-bound scene-entity target reference."""

    def __post_init__(self) -> None:
        ObjectActionGoal.__post_init__(self)
        validate_pose_goal(self.target_pose, "target_pose", allow_waypoints=False)


@dataclass(frozen=True, slots=True, eq=False)
class PushObjectToolCalibration:
    """End-effector calibration selected by a bound motion control part.

    Args:
        control_part: Exact motion control-part identifier that selects this
            calibration.
        contact_frame_to_eef: Contact-frame to robot end-effector SE(3)
            calibration.
        contact_distance: Optional tool-specific planar clearance behind the
            object. ``None`` uses :class:`PushObjectOptions`' default.
    """

    control_part: str
    """Exact motion control-part identifier that selects this calibration."""

    contact_frame_to_eef: torch.Tensor
    """Contact-frame to robot end-effector SE(3) calibration."""

    contact_distance: float | None = None
    """Optional tool-specific planar clearance behind the object."""

    def __post_init__(self) -> None:
        if (
            type(self.control_part) is not str
            or not self.control_part
            or self.control_part != self.control_part.strip()
        ):
            raise ValueError(
                "control_part must be a non-empty string without outer whitespace."
            )
        object.__setattr__(
            self,
            "contact_frame_to_eef",
            _validate_transform(
                self.contact_frame_to_eef,
                "contact_frame_to_eef",
            ),
        )
        if self.contact_distance is not None and (
            not math.isfinite(self.contact_distance) or self.contact_distance < 0.0
        ):
            raise ValueError(
                "contact_distance must be None or finite and non-negative."
            )


@dataclass(frozen=True, slots=True, eq=False)
class PushObjectOptions(ActionOptions):
    """Per-invocation planar pushing behavior."""

    hand_interp_steps: int = 5
    """Number of waypoints used to close the end effector before approach."""

    approach_height: float = 0.1
    """Distance above the contact pose used for the free-space approach."""

    retract_height: float = 0.1
    """Distance above the pushed pose used for the final retraction."""

    contact_distance: float = 0.03
    """Initial planar clearance behind the object along the push direction."""

    push_overshoot: float = 0.0
    """Additional end-effector travel beyond the object's target displacement."""

    completion_tolerance: float = 0.0
    """Planar target distance at which the action succeeds without moving."""

    object_contact_offset: torch.Tensor = torch.zeros(3, dtype=torch.float32)
    """Object-local point used as the center of the planar contact frame."""

    support_frame_planar_contact_offset: torch.Tensor | None = None
    """Optional target-support-frame override for the contact's planar offset."""

    contact_frame_to_eef: torch.Tensor = torch.eye(4, dtype=torch.float32)
    """Contact-frame to robot end-effector SE(3) calibration."""

    tool_calibrations: tuple[PushObjectToolCalibration, ...] = ()
    """Per-control-part tool-frame overrides for asymmetric robot arms."""

    def __post_init__(self) -> None:
        if type(self.hand_interp_steps) is not int or self.hand_interp_steps < 1:
            raise ValueError("hand_interp_steps must be a positive integer.")
        for name in (
            "approach_height",
            "retract_height",
            "contact_distance",
            "push_overshoot",
            "completion_tolerance",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
        offset = self.object_contact_offset
        if (
            not isinstance(offset, torch.Tensor)
            or offset.shape != (3,)
            or not torch.isfinite(offset).all()
        ):
            raise ValueError("object_contact_offset must be a finite (3,) tensor.")
        object.__setattr__(self, "object_contact_offset", offset.clone())
        support_offset = self.support_frame_planar_contact_offset
        if support_offset is not None:
            if (
                not isinstance(support_offset, torch.Tensor)
                or support_offset.shape != (3,)
                or not torch.isfinite(support_offset).all()
            ):
                raise ValueError(
                    "support_frame_planar_contact_offset must be None or a finite "
                    "(3,) tensor."
                )
            object.__setattr__(
                self,
                "support_frame_planar_contact_offset",
                support_offset.clone(),
            )
        object.__setattr__(
            self,
            "contact_frame_to_eef",
            _validate_transform(
                self.contact_frame_to_eef,
                "contact_frame_to_eef",
            ),
        )
        if type(self.tool_calibrations) is not tuple:
            raise TypeError("tool_calibrations must be an exact tuple.")
        control_parts: list[str] = []
        owned_calibrations: list[PushObjectToolCalibration] = []
        for index, calibration in enumerate(self.tool_calibrations):
            if type(calibration) is not PushObjectToolCalibration:
                raise TypeError(
                    f"tool_calibrations[{index}] must be exactly "
                    "PushObjectToolCalibration."
                )
            control_parts.append(calibration.control_part)
            owned_calibrations.append(
                PushObjectToolCalibration(
                    control_part=calibration.control_part,
                    contact_frame_to_eef=calibration.contact_frame_to_eef,
                    contact_distance=calibration.contact_distance,
                )
            )
        if len(set(control_parts)) != len(control_parts):
            raise ValueError("tool_calibrations must select unique control parts.")
        object.__setattr__(self, "tool_calibrations", tuple(owned_calibrations))

    def _calibration_for(self, control_part: str) -> torch.Tensor:
        """Return an owned tool calibration for one bound control part."""
        for calibration in self.tool_calibrations:
            if calibration.control_part == control_part:
                return calibration.contact_frame_to_eef.clone()
        return self.contact_frame_to_eef.clone()

    def _contact_distance_for(self, control_part: str) -> float:
        """Return the tool-specific or default planar contact clearance."""
        for calibration in self.tool_calibrations:
            if (
                calibration.control_part == control_part
                and calibration.contact_distance is not None
            ):
                return calibration.contact_distance
        return self.contact_distance


class PushObject(AtomicAction[PushObjectGoal, PushObjectOptions]):
    """Close the end effector, contact a rigid object, and push it in-plane."""

    skill_id: ClassVar[str] = "push_object"
    GoalType: ClassVar[type] = PushObjectGoal
    OptionsType: ClassVar[type] = PushObjectOptions
    open_loop: ClassVar[bool] = True
    binding_contract: ClassVar[SkillBindingContract] = SkillBindingContract(
        slots=(
            make_manipulation_slot(
                "primary",
                motion_capabilities=frozenset({CARTESIAN_POSE_CAPABILITY}),
                grasp_commands={GRASP_COMMAND: JointPositionCommand},
            ),
        ),
    )

    def _on_bind(self) -> None:
        """Resolve dimensions owned by the engine's robot."""
        self.num_envs = self.robot.get_qpos().shape[0]
        self.robot_dof = self.robot.dof

    def _scene_dependencies(
        self,
        request: ResolvedActionRequest[PushObjectGoal, PushObjectOptions],
    ) -> tuple[str, ...]:
        """Monitor the pushed object and any late-bound target before contact."""
        dependencies = set(super()._scene_dependencies(request))
        dependencies.add(request.goal.semantics.entity_id)
        dependencies.update(collect_scene_dependencies(request.goal.target_pose))
        return tuple(sorted(dependencies))

    def _plan(
        self,
        request: ResolvedActionRequest[PushObjectGoal, PushObjectOptions],
        context: PlanningContext,
    ) -> ActionPlan:
        """Plan close, approach, contact, planar push, and retract segments."""
        goal = self.require_goal(request)
        options = request.skill_options
        interpolation_dt = context.require_control_dt()
        motion_target = request.binding.endpoint("primary", "motion").require_target(
            JointPositionTarget
        )
        grasp = request.binding.endpoint("primary", "grasp")
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

        object_pose = resolve_pose_target(
            resolve_pose_goal(
                SceneEntityPose(goal.semantics.entity_id),
                context,
                name="object_pose",
            ),
            num_envs=self.num_envs,
            device=self.device,
        )
        target_pose = resolve_pose_target(
            resolve_pose_goal(goal.target_pose, context, name="target_pose"),
            num_envs=self.num_envs,
            device=self.device,
        )
        support_normal = target_pose[:, :3, 2]
        support_normal = support_normal / torch.linalg.vector_norm(
            support_normal,
            dim=1,
            keepdim=True,
        ).clamp_min(1.0e-6)
        displacement = target_pose[:, :3, 3] - object_pose[:, :3, 3]
        planar_displacement = (
            displacement
            - (displacement * support_normal).sum(dim=1, keepdim=True) * support_normal
        )
        planar_distance = torch.linalg.vector_norm(
            planar_displacement,
            dim=1,
            keepdim=True,
        )
        completed = (
            planar_distance[:, 0] <= options.completion_tolerance
            if options.completion_tolerance > 0.0
            else torch.zeros(
                self.num_envs,
                dtype=torch.bool,
                device=self.device,
            )
        )
        n_approach, n_contact, n_push, n_retract = self._motion_segment_lengths(
            request.motion_policy.sample_count,
            options.hand_interp_steps,
        )
        lengths = (
            options.hand_interp_steps,
            n_approach,
            n_contact,
            n_push,
            n_retract,
        )
        if bool(completed.all()):
            hold = context.last_qpos.unsqueeze(1).expand(-1, sum(lengths), -1).clone()
            return self.build_plan(
                request,
                context,
                success=completed,
                trajectory=TimedTrajectory.from_uniform_step(
                    hold,
                    env_ids=context.env_ids,
                    step_dt=interpolation_dt,
                ),
                expected_effects=StateDelta(),
                segment_lengths={
                    "close": lengths[0],
                    "approach": lengths[1],
                    "contact": lengths[2],
                    "push": lengths[3],
                    "retract": lengths[4],
                },
                scene_dependency_end_segment="approach",
            )
        valid_direction = planar_distance[:, 0] > 1.0e-6
        resolved_direction = planar_displacement / planar_distance.clamp_min(1.0e-6)
        push_direction = torch.where(
            valid_direction[:, None],
            resolved_direction,
            target_pose[:, :3, 0],
        )

        contact_frame = torch.eye(
            4,
            dtype=torch.float32,
            device=self.device,
        ).repeat(self.num_envs, 1, 1)
        contact_frame[:, :3, 2] = -support_normal
        target_x_axis = target_pose[:, :3, 0]
        target_x_axis = (
            target_x_axis
            - (target_x_axis * support_normal).sum(dim=1, keepdim=True) * support_normal
        )
        contact_frame[:, :3, 0] = target_x_axis / torch.linalg.vector_norm(
            target_x_axis,
            dim=1,
            keepdim=True,
        ).clamp_min(1.0e-6)
        contact_frame[:, :3, 1] = torch.linalg.cross(
            contact_frame[:, :3, 2],
            contact_frame[:, :3, 0],
            dim=1,
        )
        local_contact = options.object_contact_offset.to(
            device=self.device,
            dtype=torch.float32,
        )
        contact_offset = torch.matmul(object_pose[:, :3, :3], local_contact)
        support_contact_offset = options.support_frame_planar_contact_offset
        if support_contact_offset is not None:
            support_offset = torch.matmul(
                target_pose[:, :3, :3],
                support_contact_offset.to(
                    device=self.device,
                    dtype=torch.float32,
                ),
            )
            support_planar_offset = (
                support_offset
                - (support_offset * support_normal).sum(dim=1, keepdim=True)
                * support_normal
            )
            object_normal_offset = (contact_offset * support_normal).sum(
                dim=1, keepdim=True
            ) * support_normal
            contact_offset = support_planar_offset + object_normal_offset
        contact_center = contact_offset + object_pose[:, :3, 3]
        contact_distance = options._contact_distance_for(control_part)
        contact_frame[:, :3, 3] = contact_center - push_direction * contact_distance
        calibration = options._calibration_for(control_part).to(
            device=self.device,
            dtype=torch.float32,
        )
        contact_xpos = torch.matmul(contact_frame, calibration)
        approach_xpos = translate_pose_world(
            contact_xpos,
            support_normal * options.approach_height,
        )
        pushed_xpos = translate_pose_world(
            contact_xpos,
            push_direction * (planar_distance + options.push_overshoot),
        )
        retract_xpos = translate_pose_world(
            pushed_xpos,
            support_normal * options.retract_height,
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
            support_normal,
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
        push_keyframes = axis_translation_keyframes(
            contact_xpos,
            pushed_xpos,
            push_direction,
            n_waypoints=n_push - 1,
        )
        push_success, push_arm = self._plan_pose_segment(
            push_keyframes,
            contact_arm[:, -1],
            control_part,
            request,
            n_push,
            interpolation_dt=interpolation_dt,
            cartesian_linear=True,
        )
        if options.retract_height <= 1.0e-6:
            retract_success = push_success.clone()
            retract_arm = push_arm[:, -1:].expand(-1, n_retract, -1).clone()
        else:
            retract_keyframes = axis_translation_keyframes(
                pushed_xpos,
                retract_xpos,
                support_normal,
                n_waypoints=n_retract - 1,
            )
            retract_success, retract_arm = self._plan_pose_segment(
                retract_keyframes,
                push_arm[:, -1],
                control_part,
                request,
                n_retract,
                interpolation_dt=interpolation_dt,
                cartesian_linear=True,
            )
        success = completed | (
            valid_direction
            & approach_success
            & contact_success
            & push_success
            & retract_success
        )
        parts = (hand_close, approach_arm, contact_arm, push_arm, retract_arm)
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

        for arm in (approach_arm, contact_arm, push_arm, retract_arm):
            stop = offset + arm.shape[1]
            full[:, offset:stop, arm_joint_ids] = arm
            full[:, offset:stop, hand_joint_ids] = hand_grasp_qpos.unsqueeze(1)
            offset = stop

        full[completed] = context.last_qpos[completed].unsqueeze(1)

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
                "push": lengths[3],
                "retract": lengths[4],
            },
            # Contact and push intentionally move the dynamic object.
            scene_dependency_end_segment="approach",
        )

    @staticmethod
    def _motion_segment_lengths(
        sample_count: int,
        hand_interp_steps: int,
    ) -> tuple[int, int, int, int]:
        motion_count = sample_count - hand_interp_steps
        if motion_count < 8:
            raise ValueError(
                "Not enough waypoints for PushObject. Increase sample_count or "
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
        request: ResolvedActionRequest[PushObjectGoal, PushObjectOptions],
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
    "PushObject",
    "PushObjectGoal",
    "PushObjectOptions",
    "PushObjectToolCalibration",
]
