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

"""OpenDoor atomic action implementation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import ClassVar

import torch

from embodichain.lab.sim.atomic_actions.affordance import OpenDoorAffordance
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
from embodichain.lab.sim.atomic_actions.state import (
    ObservedArticulationJointState,
    PlanningContext,
)
from embodichain.lab.sim.atomic_actions.trajectory_ops import (
    axis_translation_keyframes,
    build_pose_plan_states,
    interpolate_hand_qpos,
    resolve_pose_target,
    translate_pose_world,
)
from embodichain.utils.math import axis_angle_to_rotation_matrix, pose_inv


@dataclass(frozen=True, slots=True, eq=False)
class OpenDoorGoal(ObjectActionGoal):
    """Door handle and desired absolute opening state."""

    target_pose: PoseGoalValue
    """Handle-link pose snapshot or late-bound scene-entity reference."""

    open_fraction: float | torch.Tensor
    """Desired hinge position normalized from its closed to open legal endpoint."""

    def __post_init__(self) -> None:
        ObjectActionGoal.__post_init__(self)
        validate_pose_goal(self.target_pose, "target_pose", allow_waypoints=False)
        if isinstance(self.open_fraction, bool) or not isinstance(
            self.open_fraction,
            (Real, torch.Tensor),
        ):
            raise TypeError("open_fraction must be a real number or torch.Tensor.")
        if isinstance(self.open_fraction, torch.Tensor):
            if self.open_fraction.dim() > 1 or self.open_fraction.numel() == 0:
                raise ValueError(
                    "open_fraction tensor must be scalar or have shape (B,)."
                )
            if not self.open_fraction.is_floating_point():
                raise TypeError("open_fraction tensor must be floating point.")
            object.__setattr__(self, "open_fraction", self.open_fraction.clone())
        else:
            object.__setattr__(self, "open_fraction", float(self.open_fraction))


@dataclass(frozen=True, slots=True, eq=False)
class OpenDoorOptions(ActionOptions):
    """Per-invocation approach, interpolation, release, and retract behavior."""

    hand_interp_steps: int = 5
    """Number of waypoints used for each close/open hand segment."""

    door_waypoint_count: int = 20
    """Number of Cartesian keyframes along the handle's circular arc."""

    approach_distance: float = 0.1
    """Pre-grasp distance opposite the automatically inferred approach axis."""

    retract_distance: float = 0.1
    """Post-release retreat distance opposite the rotated approach axis."""

    joint_position_tolerance: float = 1.0e-4
    """Tolerance for legal-limit and already-open comparisons in radians."""

    def __post_init__(self) -> None:
        if self.hand_interp_steps < 1:
            raise ValueError("hand_interp_steps must be at least 1.")
        if self.door_waypoint_count < 1:
            raise ValueError("door_waypoint_count must be at least 1.")
        if not math.isfinite(self.approach_distance):
            raise ValueError("approach_distance must be finite.")
        if self.approach_distance < 0.0:
            raise ValueError("approach_distance must be non-negative.")
        if not math.isfinite(self.retract_distance):
            raise ValueError("retract_distance must be finite.")
        if self.retract_distance < 0.0:
            raise ValueError("retract_distance must be non-negative.")
        if not math.isfinite(self.joint_position_tolerance):
            raise ValueError("joint_position_tolerance must be finite.")
        if self.joint_position_tolerance < 0.0:
            raise ValueError("joint_position_tolerance must be non-negative.")


class OpenDoor(AtomicAction[OpenDoorGoal, OpenDoorOptions]):
    """Approach, grasp, rotate a door about its hinge, release, and retract."""

    skill_id: ClassVar[str] = "open_door"
    GoalType: ClassVar[type] = OpenDoorGoal
    OptionsType: ClassVar[type] = OpenDoorOptions
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
        request: ResolvedActionRequest[OpenDoorGoal, OpenDoorOptions],
        context: PlanningContext,
    ) -> ActionPlan:
        """Plan the complete six-stage door-opening motion without side effects."""
        target = self.require_goal(request)
        affordance = self._require_open_door_affordance(target.semantics)
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

        n_approach, n_reach, n_open, n_retract = self._motion_segment_lengths(
            request.motion_policy.sample_count,
            options.hand_interp_steps,
            options.door_waypoint_count,
        )
        segment_lengths = {
            "approach": n_approach,
            "reach": n_reach,
            "close": options.hand_interp_steps,
            "open": n_open,
            "release": options.hand_interp_steps,
            "retract": n_retract,
        }
        hinge_state, hinge_error = self._resolve_hinge_state(
            context,
            affordance.joint_name,
        )
        if hinge_state is None:
            return self.failed_plan(request, context, message=hinge_error)
        hinge_rotation, active, already_open, semantic_valid = (
            self._resolve_hinge_rotation(
                target.open_fraction,
                hinge_state.position,
                hinge_state.valid_mask,
                affordance.joint_limits,
                affordance.opening_direction,
                context,
                tolerance=options.joint_position_tolerance,
            )
        )
        if not active.any():
            message = None
            if not semantic_valid.all():
                message = (
                    "OpenDoor target or observed hinge state is invalid for one or "
                    "more environments."
                )
            return self._hold_plan(
                request,
                context,
                success=already_open,
                segment_lengths=segment_lengths,
                diagnostics_message=message,
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

        link_pose = resolve_pose_target(
            resolve_pose_goal(target.target_pose, context, name="target_pose"),
            num_envs=self.num_envs,
            device=self.device,
        )
        approach_direction_local = self._approach_direction_local(affordance)
        approach_direction_world = torch.matmul(
            link_pose[:, :3, :3], approach_direction_local
        )
        grasp_generator = self.planning_services.grasp_pose_generator(
            grasp_target.target_id
        )
        grasp_success, grasp_xpos, _ = grasp_generator.get_best_grasp_poses(
            mesh_vertices=affordance.mesh_vertices,
            mesh_triangles=affordance.mesh_triangles,
            obj_poses=link_pose,
            approach_direction=approach_direction_world,
        )
        grasp_xpos = grasp_xpos.to(device=self.device, dtype=torch.float32)
        grasp_success = normalize_success_mask(
            grasp_success,
            num_envs=self.num_envs,
            device=self.device,
            name="OpenDoor grasp-pose success",
        )
        actionable_grasp = active & grasp_success
        if not actionable_grasp.any():
            return self._hold_plan(
                request,
                context,
                success=already_open,
                segment_lengths=segment_lengths,
                diagnostics_message="Failed to resolve a door-handle grasp pose.",
            )

        approach_xpos = translate_pose_world(
            grasp_xpos,
            -approach_direction_world * options.approach_distance,
        )
        opened_link_poses, opened_grasp_xpos = self._opened_link_and_eef_poses(
            link_pose,
            grasp_xpos,
            affordance.rotation_axis,
            affordance.axis_origin,
            hinge_rotation,
            options.door_waypoint_count,
        )
        rotated_approach_direction = torch.matmul(
            opened_link_poses[:, -1, :3, :3],
            approach_direction_local,
        )
        retract_xpos = translate_pose_world(
            opened_grasp_xpos[:, -1],
            -rotated_approach_direction * options.retract_distance,
        )

        approach_success, approach_arm = self._plan_pose_segment(
            approach_xpos,
            start_arm_qpos,
            control_part,
            request,
            n_approach,
            interpolation_dt=interpolation_dt,
        )
        reach_keyframes = axis_translation_keyframes(
            approach_xpos,
            grasp_xpos,
            approach_direction_world,
            n_waypoints=n_reach - 1,
        )
        reach_success, reach_arm = self._plan_pose_segment(
            reach_keyframes,
            approach_arm[:, -1],
            control_part,
            request,
            n_reach,
            interpolation_dt=interpolation_dt,
            cartesian_linear=True,
        )
        open_success, open_arm = self._plan_pose_segment(
            opened_grasp_xpos,
            reach_arm[:, -1],
            control_part,
            request,
            n_open,
            interpolation_dt=interpolation_dt,
        )
        retract_keyframes = axis_translation_keyframes(
            opened_grasp_xpos[:, -1],
            retract_xpos,
            rotated_approach_direction,
            n_waypoints=n_retract - 1,
        )
        retract_success, retract_arm = self._plan_pose_segment(
            retract_keyframes,
            open_arm[:, -1],
            control_part,
            request,
            n_retract,
            interpolation_dt=interpolation_dt,
            cartesian_linear=True,
        )
        planned_success = (
            actionable_grasp
            & approach_success
            & reach_success
            & open_success
            & retract_success
        )
        success = already_open | planned_success

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
        stop = offset + open_arm.shape[1]
        full[:, offset:stop, arm_joint_ids] = open_arm
        full[:, offset:stop, hand_joint_ids] = hand_grasp_qpos.unsqueeze(1)
        offset = stop
        stop = offset + hand_open.shape[1]
        full[:, offset:stop, arm_joint_ids] = open_arm[:, -1].unsqueeze(1)
        full[:, offset:stop, hand_joint_ids] = hand_open
        offset = stop
        full[:, offset:, arm_joint_ids] = retract_arm
        full[:, offset:, hand_joint_ids] = hand_open_qpos.unsqueeze(1)
        full[already_open] = context.last_qpos[already_open].unsqueeze(1)

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
            diagnostics=self._semantic_diagnostics(semantic_valid),
            segment_lengths=segment_lengths,
            scene_dependency_end_segment=(
                "reach" if self._scene_dependencies(request) else None
            ),
        )

    @staticmethod
    def _require_open_door_affordance(
        semantics: ObjectSemantics,
    ) -> OpenDoorAffordance:
        affordance = semantics.affordance
        if not isinstance(affordance, OpenDoorAffordance):
            raise ValueError("OpenDoor requires an OpenDoorAffordance.")
        return affordance

    @staticmethod
    def _approach_direction_local(
        affordance: OpenDoorAffordance,
    ) -> torch.Tensor:
        """Infer approach opposite the positive hinge-opening tangent."""
        axis = affordance.rotation_axis.to(dtype=torch.float32)
        axis = axis / torch.linalg.vector_norm(axis)
        origin = torch.tensor(
            affordance.axis_origin,
            device=axis.device,
            dtype=torch.float32,
        )
        assert affordance.mesh_vertices is not None
        handle_center = affordance.mesh_vertices.to(
            device=axis.device,
            dtype=torch.float32,
        ).mean(dim=0)
        radial = handle_center - origin
        radial = radial - torch.dot(radial, axis) * axis
        if torch.linalg.vector_norm(radial) <= 1.0e-6:
            raise ValueError(
                "Door-handle center must not lie on the resolved hinge axis."
            )
        opening_tangent = torch.linalg.cross(axis, radial)
        opening_tangent = opening_tangent / torch.linalg.vector_norm(opening_tangent)
        return -affordance.opening_direction * opening_tangent

    @staticmethod
    def _resolve_hinge_state(
        context: PlanningContext,
        joint_name: str,
    ) -> tuple[ObservedArticulationJointState | None, str | None]:
        """Resolve one live hinge observation by its affordance-owned joint name."""
        matches = [
            state
            for (
                _,
                observed_joint_name,
            ), state in context.scene.articulation_joints.items()
            if observed_joint_name == joint_name
        ]
        if not matches:
            return None, f"No observed articulation joint named {joint_name!r}."
        if len(matches) > 1:
            return (
                None,
                f"Observed articulation joint name {joint_name!r} is ambiguous.",
            )
        return matches[0], None

    def _resolve_hinge_rotation(
        self,
        open_fraction: float | torch.Tensor,
        observed_position: torch.Tensor,
        observed_valid_mask: torch.Tensor | None,
        joint_limits: tuple[float, float] | None,
        opening_direction: int,
        context: PlanningContext,
        *,
        tolerance: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Resolve an opening-directed row-local delta from an absolute fraction."""
        if joint_limits is None:
            invalid = torch.zeros(
                context.batch_size,
                dtype=torch.bool,
                device=self.device,
            )
            return (
                torch.zeros_like(invalid, dtype=torch.float32),
                invalid,
                invalid,
                invalid,
            )
        lower, upper = joint_limits
        if not math.isfinite(lower) or not math.isfinite(upper) or upper <= lower:
            invalid = torch.zeros(
                context.batch_size,
                dtype=torch.bool,
                device=self.device,
            )
            return (
                torch.zeros_like(invalid, dtype=torch.float32),
                invalid,
                invalid,
                invalid,
            )

        fractions = torch.as_tensor(
            open_fraction,
            dtype=torch.float32,
            device=self.device,
        )
        if fractions.dim() == 0 or fractions.shape == (1,):
            fractions = fractions.reshape(1).expand(context.batch_size)
        elif fractions.shape != (context.batch_size,):
            raise ValueError(
                "OpenDoorGoal.open_fraction must be scalar or match the planning "
                f"batch size ({context.batch_size},)."
            )

        position = observed_position.to(device=self.device, dtype=torch.float32)
        if position.shape == (1,):
            position = position.expand(context.batch_size)
        elif position.shape == (context.batch_size, 1):
            position = position[:, 0]
        else:
            raise ValueError(
                "OpenDoor hinge observation must have shape (1,) or (B, 1)."
            )
        if observed_valid_mask is None:
            observation_valid = torch.ones(
                context.batch_size,
                dtype=torch.bool,
                device=self.device,
            )
        else:
            observation_valid = observed_valid_mask.to(device=self.device)

        closed_position = lower if opening_direction > 0 else upper
        open_position = upper if opening_direction > 0 else lower
        target_position = closed_position + fractions * (
            open_position - closed_position
        )
        fraction_valid = (
            torch.isfinite(fractions) & (fractions >= 0.0) & (fractions <= 1.0)
        )
        position_valid = (
            observation_valid
            & torch.isfinite(position)
            & (position >= lower - tolerance)
            & (position <= upper + tolerance)
        )
        rotation = target_position - position
        directed_rotation = rotation * opening_direction
        forward_or_reached = directed_rotation >= -tolerance
        semantic_valid = fraction_valid & position_valid & forward_or_reached
        already_open = semantic_valid & (directed_rotation.abs() <= tolerance)
        active = semantic_valid & (directed_rotation > tolerance)
        safe_rotation = torch.where(active, rotation, torch.zeros_like(rotation))
        return safe_rotation, active, already_open, semantic_valid

    def _hold_plan(
        self,
        request: ResolvedActionRequest[OpenDoorGoal, OpenDoorOptions],
        context: PlanningContext,
        *,
        success: torch.Tensor,
        segment_lengths: dict[str, int],
        diagnostics_message: str | None,
    ) -> ActionPlan:
        """Return a segmented full-robot hold for reached and failed rows."""
        frame_count = sum(segment_lengths.values())
        positions = context.last_qpos.unsqueeze(1).expand(-1, frame_count, -1).clone()
        diagnostics = PlannerDiagnostics(
            backend=self.planning_services.planner_name,
            messages=(() if diagnostics_message is None else (diagnostics_message,)),
        )
        return self.build_plan(
            request,
            context,
            success=success,
            trajectory=TimedTrajectory.from_uniform_step(
                positions,
                env_ids=context.env_ids,
                step_dt=context.require_control_dt(),
            ),
            diagnostics=diagnostics,
            segment_lengths=segment_lengths,
            scene_dependency_end_segment=(
                "reach" if self._scene_dependencies(request) else None
            ),
        )

    def _semantic_diagnostics(
        self,
        semantic_valid: torch.Tensor,
    ) -> PlannerDiagnostics:
        """Describe row-local semantic rejection without changing success masking."""
        messages = ()
        if not semantic_valid.all():
            messages = (
                "OpenDoor target or observed hinge state is invalid for one or "
                "more environments.",
            )
        return PlannerDiagnostics(
            backend=self.planning_services.planner_name,
            messages=messages,
        )

    @staticmethod
    def _motion_segment_lengths(
        sample_count: int,
        hand_interp_steps: int,
        door_waypoint_count: int,
    ) -> tuple[int, int, int, int]:
        minimum_sample_count = 2 * hand_interp_steps + door_waypoint_count + 7
        if sample_count < minimum_sample_count:
            raise ValueError(
                "Not enough waypoints for OpenDoor: sample_count must be at "
                f"least {minimum_sample_count} for hand_interp_steps="
                f"{hand_interp_steps} and door_waypoint_count="
                f"{door_waypoint_count}."
            )
        motion_count = sample_count - 2 * hand_interp_steps
        base, remainder = divmod(motion_count, 4)
        values = [base + (index < remainder) for index in range(4)]
        minimum_open_count = door_waypoint_count + 1
        if values[2] >= minimum_open_count:
            return values[0], values[1], values[2], values[3]

        remaining_count = motion_count - minimum_open_count
        other_base, other_remainder = divmod(remaining_count, 3)
        other_values = [other_base + (index < other_remainder) for index in range(3)]
        return (
            other_values[0],
            other_values[1],
            minimum_open_count,
            other_values[2],
        )

    def _plan_pose_segment(
        self,
        target_pose: torch.Tensor,
        start_qpos: torch.Tensor,
        control_part: str,
        request: ResolvedActionRequest[OpenDoorGoal, OpenDoorOptions],
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

    def _opened_link_and_eef_poses(
        self,
        link_pose: torch.Tensor,
        grasp_xpos: torch.Tensor,
        rotation_axis: torch.Tensor,
        axis_origin: tuple[float, float, float],
        hinge_rotation: torch.Tensor,
        waypoint_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rotate link poses about the hinge and recover corresponding EEF poses."""
        axis = rotation_axis.to(device=self.device, dtype=torch.float32)
        axis = axis / torch.linalg.vector_norm(axis)
        fractions = torch.linspace(
            1.0 / waypoint_count,
            1.0,
            waypoint_count,
            dtype=torch.float32,
            device=self.device,
        )
        angles = hinge_rotation.to(device=self.device, dtype=torch.float32)[:, None]
        angles = angles * fractions[None]
        rotations = (
            torch.eye(4, dtype=torch.float32, device=self.device)
            .reshape(1, 1, 4, 4)
            .repeat(link_pose.shape[0], waypoint_count, 1, 1)
        )
        rotations[:, :, :3, :3] = axis_angle_to_rotation_matrix(
            angles[:, :, None] * axis
        )
        origin = torch.tensor(axis_origin, dtype=torch.float32, device=self.device)
        to_origin = torch.eye(4, dtype=torch.float32, device=self.device)
        from_origin = torch.eye(4, dtype=torch.float32, device=self.device)
        to_origin[:3, 3] = origin
        from_origin[:3, 3] = -origin
        local_rotations = torch.matmul(
            torch.matmul(to_origin.reshape(1, 1, 4, 4), rotations),
            from_origin.reshape(1, 1, 4, 4),
        )
        opened_link_poses = torch.matmul(link_pose[:, None], local_rotations)
        link_to_eef = torch.bmm(pose_inv(link_pose), grasp_xpos)
        opened_eef_poses = torch.matmul(opened_link_poses, link_to_eef[:, None])
        return opened_link_poses, opened_eef_poses


__all__ = ["OpenDoor", "OpenDoorGoal", "OpenDoorOptions"]
