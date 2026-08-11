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

"""Reusable contact-and-drag operation for articulated mechanisms."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import ClassVar

import torch

from ..bindings import JointPositionTarget
from ..control import GRASP_COMMAND, OPEN_COMMAND, JointPositionCommand
from ..core import AtomicAction
from ..effects import StateDelta
from ..goals import SceneArticulationOperationGeometry
from ..invocation import ActionOptions, ResolvedActionRequest
from ..plans import (
    ActionPlan,
    EffectVerificationRequirement,
    PlannerDiagnostics,
    TimedTrajectory,
)
from ..requirements import (
    CARTESIAN_POSE_CAPABILITY,
    DisjointSlotEndpoints,
    GRASP_CAPABILITY,
    JOINT_POSITION_CAPABILITY,
    SkillBindingContract,
    SkillEndpointRequirement,
    SkillResourceSlot,
)
from ..state import ArticulationJointState, PlanningContext
from ..trajectory_ops import (
    build_pose_plan_states,
    interpolate_hand_qpos,
    resolve_pose_target,
)


def _validate_identifier(value: str, *, field_name: str) -> None:
    """Validate one canonical articulation identifier."""
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a non-empty canonical identifier.")


@dataclass(frozen=True, slots=True, eq=False)
class OperateArticulationGoal:
    """Grounded interaction path and desired state for one articulation joint.

    The semantic compiler copies immutable affordance geometry and records the
    live source joint position. The atomic planner combines those values with
    the latest handle and joint observation, so the same resolved request can
    safely replan drawers, doors, sliders, and similar interactions.
    """

    goal_kind: ClassVar[str] = "operate_articulation"

    articulation_id: str
    """Canonical scene-registry articulation identifier."""

    joint_id: str
    """Canonical joint identifier within the articulation."""

    geometry: SceneArticulationOperationGeometry
    """Handle-relative geometry resolved again for every plan and replan."""

    source_position: torch.Tensor
    """Live joint position at semantic grounding, shape ``(1,)`` or ``(B, 1)``."""

    target_position: torch.Tensor
    """Absolute desired joint position, shape ``(1,)`` or ``(B, 1)``."""

    target_displacement: float
    """Signed handle displacement from source position to target position."""

    def __post_init__(self) -> None:
        _validate_identifier(self.articulation_id, field_name="articulation_id")
        _validate_identifier(self.joint_id, field_name="joint_id")
        if not isinstance(self.geometry, SceneArticulationOperationGeometry):
            raise TypeError("geometry must be a SceneArticulationOperationGeometry.")
        for field_name in ("source_position", "target_position"):
            position = getattr(self, field_name)
            if not isinstance(position, torch.Tensor):
                raise TypeError(f"{field_name} must be a torch.Tensor.")
            if position.dim() not in (1, 2) or position.shape[-1:] != (1,):
                raise ValueError(f"{field_name} must have shape (1,) or (B, 1).")
            if not position.is_floating_point():
                raise TypeError(f"{field_name} must use a floating-point dtype.")
            if not torch.isfinite(position).all():
                raise ValueError(f"{field_name} must contain only finite values.")
            object.__setattr__(self, field_name, position.clone())
        displacement = self.target_displacement
        if isinstance(displacement, bool) or not isinstance(displacement, (int, float)):
            raise TypeError("target_displacement must be a finite scalar.")
        displacement = float(displacement)
        if not math.isfinite(displacement):
            raise ValueError("target_displacement must be finite.")
        object.__setattr__(self, "target_displacement", displacement)


@dataclass(frozen=True, slots=True, eq=False)
class OperateArticulationOptions(ActionOptions):
    """Per-invocation contact sequencing for articulation operations."""

    engage_steps: int = 5
    """Number of gripper-closing waypoints at the contact pose."""

    release_steps: int = 5
    """Number of gripper-opening waypoints before retracting."""

    def __post_init__(self) -> None:
        for field_name in ("engage_steps", "release_steps"):
            value = getattr(self, field_name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer.")


class OperateArticulation(
    AtomicAction[OperateArticulationGoal, OperateArticulationOptions]
):
    """Approach, engage, move, release, and retract an articulated affordance."""

    skill_id: ClassVar[str] = "operate_articulation"
    GoalType: ClassVar[type] = OperateArticulationGoal
    OptionsType: ClassVar[type] = OperateArticulationOptions
    binding_contract: ClassVar[SkillBindingContract] = SkillBindingContract(
        slots=(
            SkillResourceSlot(
                slot_id="primary",
                endpoints=(
                    SkillEndpointRequirement(
                        endpoint_id="motion",
                        capabilities=frozenset(
                            {
                                CARTESIAN_POSE_CAPABILITY,
                                JOINT_POSITION_CAPABILITY,
                            }
                        ),
                    ),
                    SkillEndpointRequirement(
                        endpoint_id="interaction",
                        capabilities=frozenset({GRASP_CAPABILITY}),
                        required_commands={
                            GRASP_COMMAND: JointPositionCommand,
                            OPEN_COMMAND: JointPositionCommand,
                        },
                    ),
                ),
                constraints=(DisjointSlotEndpoints(("motion", "interaction")),),
            ),
        ),
    )

    def __init__(
        self,
        default_options: OperateArticulationOptions | None = None,
    ) -> None:
        super().__init__(default_options)

    def _on_bind(self) -> None:
        """Capture immutable robot dimensions from engine-owned services."""
        self.n_envs = int(self.robot.get_qpos().shape[0])
        self.robot_dof = int(self.robot.dof)

    def _plan(
        self,
        request: ResolvedActionRequest[
            OperateArticulationGoal,
            OperateArticulationOptions,
        ],
        context: PlanningContext,
    ) -> ActionPlan:
        """Plan the complete contact interaction from one observed context."""
        goal = self.require_goal(request)
        options = request.skill_options
        motion = request.binding.endpoint("primary", "motion").require_target(
            JointPositionTarget
        )
        interaction = request.binding.endpoint("primary", "interaction")
        interaction_target = interaction.require_target(JointPositionTarget)
        arm_joint_ids = list(motion.joint_ids)
        interaction_joint_ids = list(interaction_target.joint_ids)

        remaining_displacement = self._remaining_displacement(goal, context)
        poses = tuple(
            resolve_pose_target(
                pose,
                num_envs=context.batch_size,
                device=self.device,
            )
            for pose in goal.geometry.resolve(
                context,
                displacement=remaining_displacement,
            )
        )
        motion_counts = self._motion_sample_counts(
            request.motion_policy.sample_count,
            options,
        )
        arm_segments: list[torch.Tensor] = []
        phase_diagnostics: dict[str, dict[str, object]] = {}
        arm_start = context.robot.qpos[:, arm_joint_ids]
        success = torch.ones(
            context.batch_size,
            dtype=torch.bool,
            device=context.robot.qpos.device,
        )
        phase_names = ("approach", "contact", "operate", "retract")
        for phase_name, pose, sample_count in zip(
            phase_names,
            poses,
            motion_counts,
            strict=True,
        ):
            result = self.motion_generator.generate(
                build_pose_plan_states(pose),
                options=request.motion_policy.to_motion_gen_options(
                    start_qpos=arm_start,
                    control_part=motion.control_part,
                    sample_count=sample_count,
                    interpolation_dt=context.require_control_dt(),
                ),
            )
            if result.positions is None or not isinstance(result.success, torch.Tensor):
                return self.failed_plan(
                    request,
                    context,
                    message=(
                        "The articulation motion planner returned no trajectory for "
                        f"phase {phase_name!r}."
                    ),
                )
            phase_success = result.success.to(
                device=success.device,
                dtype=torch.bool,
            )
            failed_rows = (
                (~phase_success).nonzero(as_tuple=False).flatten().detach().cpu()
            )
            phase_diagnostics[phase_name] = {
                "success": phase_success.detach().cpu().tolist(),
                "failed_rows": failed_rows.tolist(),
                "waypoint_count": int(result.positions.shape[1]),
            }
            arm_segments.append(result.positions)
            arm_start = result.positions[:, -1]
            success &= phase_success

        grasp_qpos = interaction.joint_positions(
            GRASP_COMMAND,
            num_envs=self.num_envs,
            device=self.device,
            dtype=context.robot.qpos.dtype,
        )
        open_qpos = interaction.joint_positions(
            OPEN_COMMAND,
            num_envs=self.num_envs,
            device=self.device,
            dtype=context.robot.qpos.dtype,
        )
        initial_interaction = context.robot.qpos[:, interaction_joint_ids]
        engage_path = interpolate_hand_qpos(
            initial_interaction,
            grasp_qpos,
            n_waypoints=options.engage_steps,
        )
        release_path = interpolate_hand_qpos(
            grasp_qpos,
            open_qpos,
            n_waypoints=options.release_steps,
        )

        approach_arm, contact_arm, operation_arm, retract_arm = arm_segments
        lengths = {
            "approach": int(approach_arm.shape[1]),
            "engage": int(contact_arm.shape[1] + engage_path.shape[1]),
            "operate": int(operation_arm.shape[1]),
            "release": int(release_path.shape[1]),
            "retract": int(retract_arm.shape[1]),
        }
        full = torch.empty(
            (
                context.batch_size,
                sum(lengths.values()),
                self.robot_dof,
            ),
            dtype=context.robot.qpos.dtype,
            device=context.robot.qpos.device,
        )
        full[:] = context.robot.qpos.unsqueeze(1)
        cursor = 0

        def append_motion(
            segment: torch.Tensor,
            interaction_qpos: torch.Tensor,
        ) -> None:
            nonlocal cursor
            count = int(segment.shape[1])
            full[:, cursor : cursor + count, arm_joint_ids] = segment
            full[:, cursor : cursor + count, interaction_joint_ids] = (
                interaction_qpos.unsqueeze(1)
            )
            cursor += count

        append_motion(approach_arm, initial_interaction)
        append_motion(contact_arm, initial_interaction)
        full[:, cursor : cursor + options.engage_steps, arm_joint_ids] = contact_arm[
            :, -1
        ].unsqueeze(1)
        full[:, cursor : cursor + options.engage_steps, interaction_joint_ids] = (
            engage_path
        )
        cursor += options.engage_steps
        append_motion(operation_arm, grasp_qpos)
        full[:, cursor : cursor + options.release_steps, arm_joint_ids] = operation_arm[
            :, -1
        ].unsqueeze(1)
        full[:, cursor : cursor + options.release_steps, interaction_joint_ids] = (
            release_path
        )
        cursor += options.release_steps
        append_motion(retract_arm, open_qpos)
        assert cursor == full.shape[1]

        target_position = goal.target_position.to(
            device=context.robot.qpos.device,
            dtype=context.robot.qpos.dtype,
        )
        expected = StateDelta(
            articulation_joint_updates={
                (goal.articulation_id, goal.joint_id): ArticulationJointState(
                    target_position
                )
            }
        )
        return self.build_plan(
            request,
            context,
            success=success,
            trajectory=TimedTrajectory.from_uniform_step(
                full,
                env_ids=context.env_ids,
                step_dt=context.require_control_dt(),
            ),
            expected_effects=expected,
            effect_verification=EffectVerificationRequirement(
                kind="articulation.joint_progress"
            ),
            segment_lengths=lengths,
            scene_dependency_monitor_until={
                goal.geometry.handle_pose.entity_id: lengths["approach"]
                + lengths["engage"]
            },
            diagnostics=PlannerDiagnostics(
                backend=self.planning_services.planner_name,
                messages=tuple(
                    f"Articulation motion phase {phase_name!r} failed for rows "
                    f"{details['failed_rows']}."
                    for phase_name, details in phase_diagnostics.items()
                    if details["failed_rows"]
                ),
                metadata={"motion_phases": phase_diagnostics},
            ),
        )

    @staticmethod
    def _position_batch(
        value: torch.Tensor,
        context: PlanningContext,
        *,
        field_name: str,
    ) -> torch.Tensor:
        """Broadcast one scalar joint position to the planning batch."""
        position = value.to(
            device=context.robot.qpos.device,
            dtype=context.robot.qpos.dtype,
        )
        if position.shape == (1,):
            return position.unsqueeze(0).expand(context.batch_size, -1).clone()
        if position.shape != (context.batch_size, 1):
            raise ValueError(
                f"{field_name} must have shape (1,) or " f"({context.batch_size}, 1)."
            )
        return position.clone()

    @classmethod
    def _remaining_displacement(
        cls,
        goal: OperateArticulationGoal,
        context: PlanningContext,
    ) -> torch.Tensor:
        """Map remaining joint stroke to a bounded signed handle displacement.

        For each row, ``remaining = target_displacement * clamp(
        (target - current) / (target - source), 0, 1)``. A zero-length source
        stroke, a reached target, and an overshot target all resolve to zero.
        """
        observed = context.scene.get_articulation_joint_state(
            goal.articulation_id,
            goal.joint_id,
        )
        address = (goal.articulation_id, goal.joint_id)
        if observed is None:
            raise ValueError(
                "OperateArticulation recovery-safe planning requires a live "
                f"ObservedArticulationJointState for {address!r}."
            )
        current = cls._position_batch(
            observed.position,
            context,
            field_name=f"observed articulation joint {address!r}",
        )
        if observed.valid_mask is not None:
            valid = observed.valid_mask.to(device=context.robot.qpos.device)
            if not bool(valid.all()):
                invalid_rows = (~valid).nonzero(as_tuple=False).flatten().tolist()
                raise ValueError(
                    f"Live articulation joint {address!r} is invalid for planning "
                    f"rows {invalid_rows}."
                )
        source = cls._position_batch(
            goal.source_position,
            context,
            field_name="source_position",
        )
        target = cls._position_batch(
            goal.target_position,
            context,
            field_name="target_position",
        )
        total = target - source
        tolerance = torch.finfo(total.dtype).eps * 16.0
        nonzero_stroke = total.abs() > tolerance
        fraction = torch.zeros_like(total)
        fraction[nonzero_stroke] = (
            (target - current)[nonzero_stroke] / total[nonzero_stroke]
        ).clamp(0.0, 1.0)
        return fraction[:, 0] * goal.target_displacement

    @staticmethod
    def _motion_sample_counts(
        sample_count: int,
        options: OperateArticulationOptions,
    ) -> tuple[int, int, int, int]:
        """Allocate the preset sample budget across four motion phases."""
        remaining = sample_count - options.engage_steps - options.release_steps
        if remaining < 8:
            raise ValueError(
                "MotionPolicy.sample_count must leave at least two waypoints for "
                "each articulation motion phase."
            )
        base, remainder = divmod(remaining, 4)
        counts = tuple(base + (1 if index < remainder else 0) for index in range(4))
        assert len(counts) == 4 and all(value >= 2 for value in counts)
        return counts


__all__ = [
    "OperateArticulation",
    "OperateArticulationGoal",
    "OperateArticulationOptions",
]
