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
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, ClassVar, Literal

import torch

from embodichain.lab.sim.atomic_actions.affordance import SlideAffordance
from embodichain.lab.sim.atomic_actions.bindings import JointPositionTarget
from embodichain.lab.sim.atomic_actions.candidates import (
    AtomicCandidateBatch,
    AtomicCandidateSelection,
    _input_fingerprint,
    _value_fingerprint,
)
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
from embodichain.lab.sim.atomic_actions.primitives._candidate_helpers import (
    _CANDIDATE_MAX_JOINT_STEP,
    _candidate_cartesian_segment,
    _checked_candidate_ik,
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

if TYPE_CHECKING:
    from embodichain.toolkits.graspkit.candidates import GraspCandidateBatch


@dataclass(frozen=True, slots=True, eq=False, kw_only=True)
class SlideCandidateBatch(AtomicCandidateBatch):
    """One certified slide per original grasp, without extra roll variants.

    Pose fields are ``(E,G,4,4)`` and ``qpos_anchors`` is ``(E,G,3,arm_dof)``
    in pre-grasp, grasp, translated order. Selection retains these exact
    anchors and is valid only at the enumerated invocation/context.
    """

    grasp_ids: tuple[tuple[str, ...], ...]
    pre_grasp_poses: torch.Tensor
    grasp_poses: torch.Tensor
    translated_poses: torch.Tensor
    qpos_anchors: torch.Tensor
    evaluation_fingerprint: str = ""

    def __post_init__(self) -> None:
        AtomicCandidateBatch.__post_init__(self)
        shape = self.valid_mask.shape
        if self.skill_id != "slide":
            raise ValueError("Slide candidates must have skill_id='slide'.")
        if len(self.grasp_ids) != shape[0] or any(
            len(row) != shape[1] for row in self.grasp_ids
        ):
            raise ValueError("grasp_ids must have E rows and G columns.")
        object.__setattr__(
            self, "grasp_ids", tuple(tuple(row) for row in self.grasp_ids)
        )
        if self.qpos_anchors.ndim != 4 or self.qpos_anchors.shape[:3] != (*shape, 3):
            raise ValueError("qpos_anchors must have shape (E,G,3,A).")
        for name in (
            "pre_grasp_poses",
            "grasp_poses",
            "translated_poses",
            "qpos_anchors",
        ):
            value = getattr(self, name)
            if name != "qpos_anchors" and value.shape != (*shape, 4, 4):
                raise ValueError(f"{name} must have shape (E,G,4,4).")
            if (
                not value.is_floating_point()
                or value.dtype != self.qpos_anchors.dtype
                or value.device != self.env_ids.device
                or not torch.isfinite(value).all()
            ):
                raise ValueError(
                    "Slide candidate payloads must be finite and share device/dtype."
                )
            object.__setattr__(self, name, value.detach().clone())
        fingerprint = _value_fingerprint(
            self.env_ids,
            self.valid_mask,
            self.costs,
            self.candidate_ids,
            self.failure_stages,
            self.failure_reasons,
            self.grasp_ids,
            self.pre_grasp_poses,
            self.grasp_poses,
            self.translated_poses,
            self.qpos_anchors,
            self.input_fingerprint,
        )
        if self.evaluation_fingerprint and self.evaluation_fingerprint != fingerprint:
            raise ValueError(
                "Evaluated candidate payload was modified; enumerate again."
            )
        object.__setattr__(self, "evaluation_fingerprint", fingerprint)


@dataclass(frozen=True, slots=True, eq=False)
class SlideGoal(ObjectActionGoal):
    """Translating articulation link described by a slide affordance."""

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

    def _enumerate_candidates(
        self,
        request: ResolvedActionRequest[SlideGoal, SlideOptions],
        context: PlanningContext,
        *,
        grasp_candidates: GraspCandidateBatch | None,
        active_mask: torch.Tensor,
    ) -> SlideCandidateBatch:
        """Evaluate all original grasps without a winner reduction or roll changes."""
        from embodichain.toolkits.graspkit.candidates import GraspCandidateBatch

        goal = self.require_goal(request)
        affordance = self._require_slide_affordance(goal.semantics)
        manipulator = request.binding.endpoint("primary", "motion").require_target(
            JointPositionTarget
        )
        hand = request.binding.endpoint("primary", "grasp").require_target(
            JointPositionTarget
        )
        link_pose = resolve_pose_target(
            resolve_pose_goal(goal.target_pose, context, name="target_pose"),
            num_envs=context.batch_size,
            device=self.device,
        ).to(dtype=context.robot.qpos.dtype)
        axis = affordance.translation_axis.to(link_pose)
        axis = link_pose[:, :3, :3] @ (axis / torch.linalg.vector_norm(axis))
        if grasp_candidates is None:
            empty = (link_pose.new_empty((0, 4, 4)), link_pose.new_empty((0,)))
            ragged = [empty for _ in range(context.batch_size)]
            active_rows = torch.nonzero(active_mask).flatten()
            if active_rows.numel():
                sampled = self.planning_services.grasp_pose_generator(
                    hand.target_id
                ).get_valid_grasp_poses(
                    mesh_vertices=affordance.mesh_vertices,
                    mesh_triangles=affordance.mesh_triangles,
                    obj_poses=link_pose[active_rows],
                    approach_direction=axis[active_rows],
                )
                if len(sampled) != active_rows.numel():
                    raise ValueError(
                        "Grasp sampler must return one result per input row."
                    )
                for index, row in enumerate(active_rows.tolist()):
                    ragged[row] = sampled[index]
            grasp_candidates = GraspCandidateBatch.from_ragged(
                ragged, object_poses=link_pose
            )
        if not isinstance(grasp_candidates, GraspCandidateBatch):
            raise TypeError("grasp_candidates must be GraspCandidateBatch.")
        raw = replace(grasp_candidates)
        if raw.poses.shape[0] != context.batch_size or raw.poses.device != self.device:
            raise ValueError(
                "Grasp candidates must be bound to the real E rows/device."
            )
        if raw.frame != "local_arena":
            raise ValueError("Atomic grasp candidates must use the local_arena frame.")
        poses = raw.poses.to(dtype=context.robot.qpos.dtype)
        pre_grasp, translated = poses.clone(), poses.clone()
        pre_grasp[..., :3, 3] -= axis[:, None] * request.skill_options.approach_distance
        sign = -1.0 if request.skill_options.direction == "pull" else 1.0
        translated[..., :3, 3] += (
            axis[:, None] * sign * request.skill_options.translation_distance
        )
        valid = raw.valid_mask & active_mask[:, None]
        rows, columns = valid.shape
        reasons: list[list[str | None]] = [[None] * columns for _ in range(rows)]
        for row in range(rows):
            for column in range(columns):
                if not active_mask[row]:
                    reasons[row][column] = "INACTIVE"
                elif not valid[row, column]:
                    reasons[row][
                        column
                    ] = f"grasp:{raw.rejection_reasons[row][column] or 'INVALID_GEOMETRY'}"
        start = context.robot.qpos[:, list(manipulator.joint_ids)]
        seed = start[:, None].expand(-1, columns, -1).clone()
        anchors = []
        for stage, target in (
            ("pre_grasp", pre_grasp),
            ("grasp", poses),
            ("translated", translated),
        ):
            stage_valid, seed, failures = _checked_candidate_ik(
                self.robot, target, seed, manipulator, context, valid
            )
            for row, column in torch.nonzero(valid & ~stage_valid).tolist():
                reasons[row][column] = f"{stage}:{failures[row][column]}"
            valid &= stage_valid
            anchors.append(seed.clone())
        return SlideCandidateBatch(
            skill_id=self.skill_id,
            env_ids=context.env_ids,
            valid_mask=valid,
            costs=torch.where(valid, raw.costs, torch.inf),
            candidate_ids=tuple(
                tuple(f"{value}/slide" for value in row) for row in raw.grasp_ids
            ),
            failure_stages=tuple(tuple(row) for row in reasons),
            input_fingerprint=_input_fingerprint(request, context),
            grasp_ids=raw.grasp_ids,
            pre_grasp_poses=pre_grasp,
            grasp_poses=poses,
            translated_poses=translated,
            qpos_anchors=torch.stack(anchors, dim=2),
        )

    def _plan_candidate(
        self,
        request: ResolvedActionRequest[SlideGoal, SlideOptions],
        selection: AtomicCandidateSelection,
        context: PlanningContext,
        *,
        active_mask: torch.Tensor,
    ) -> ActionPlan:
        """Materialize a selected original grasp with strict Cartesian phases."""
        candidates = selection.candidates
        if not isinstance(candidates, SlideCandidateBatch):
            raise TypeError("Slide requires SlideCandidateBatch selections.")
        if candidates.input_fingerprint != _input_fingerprint(request, context):
            raise ValueError(
                "Candidate evaluation is stale; enumerate at the current invocation/context."
            )
        if request.motion_policy.strategy != "ik_interp":
            raise NotImplementedError(
                "Selected Slide candidates currently require ik_interp."
            )
        options = request.skill_options
        manipulator = request.binding.endpoint("primary", "motion").require_target(
            JointPositionTarget
        )
        grasp = request.binding.endpoint("primary", "grasp")
        hand = grasp.require_target(JointPositionTarget)
        rows = torch.arange(context.batch_size, device=self.device)
        indices = selection.indices.clamp_min(0)
        capacity = candidates.valid_mask.shape[1]
        success = active_mask.clone()
        if capacity:
            success &= candidates.valid_mask[rows, indices]
        else:
            success[:] = False
        ids: list[str | None] = [None] * context.batch_size
        reasons: list[str | None] = [None] * context.batch_size
        for row in range(context.batch_size):
            if not active_mask[row]:
                reasons[row] = "INACTIVE"
            elif not capacity:
                reasons[row] = "grasp:NO_CANDIDATES"
            else:
                column = int(indices[row])
                ids[row] = candidates.candidate_ids[row][column]
                if not success[row]:
                    stage = candidates.failure_stages[row][column] or "grasp"
                    reason = candidates.failure_reasons[row][column] or "IK_NOT_FOUND"
                    reasons[row] = f"{stage}:{reason}"

        def diagnostics() -> PlannerDiagnostics:
            return PlannerDiagnostics(
                backend="selected_slide_ik",
                metadata={
                    "candidate_ids": tuple(ids),
                    "candidate_failure_reasons": tuple(reasons),
                    "selected_joint_branch_preserved": True,
                    "max_joint_step": _CANDIDATE_MAX_JOINT_STEP,
                },
            )

        if not success.any():
            return self.build_plan(
                request,
                context,
                success=success,
                trajectory=TimedTrajectory.from_uniform_step(
                    context.robot.qpos[:, None],
                    env_ids=context.env_ids,
                    step_dt=context.require_control_dt(),
                ),
                diagnostics=diagnostics(),
            )
        arm_ids, hand_ids = list(manipulator.joint_ids), list(hand.joint_ids)
        anchors = candidates.qpos_anchors[rows, indices]
        pre = candidates.pre_grasp_poses[rows, indices]
        contact = candidates.grasp_poses[rows, indices]
        translated = candidates.translated_poses[rows, indices]
        lengths = self._motion_segment_lengths(
            request.motion_policy.sample_count,
            options.hand_interp_steps,
            direction=options.direction,
        )
        weights = torch.linspace(
            0.0, 1.0, lengths[0], device=self.device, dtype=anchors.dtype
        )
        transit = torch.lerp(
            context.robot.qpos[:, None, arm_ids],
            anchors[:, 0, None],
            weights[None, :, None],
        )

        def cartesian(
            name: str,
            begin_pose: torch.Tensor,
            end_pose: torch.Tensor,
            begin_qpos: torch.Tensor,
            end_qpos: torch.Tensor,
            count: int,
        ) -> torch.Tensor:
            nonlocal success
            valid, qpos, failures = _candidate_cartesian_segment(
                self.robot,
                begin_pose,
                end_pose,
                begin_qpos,
                end_qpos,
                manipulator,
                context,
                success,
                count,
            )
            for row in torch.nonzero(success & ~valid).flatten().tolist():
                reasons[row] = f"{name}:{failures[row]}"
            success &= valid
            return qpos

        reach = cartesian(
            "reach", pre, contact, anchors[:, 0], anchors[:, 1], lengths[1]
        )
        travel = cartesian(
            options.direction,
            contact,
            translated,
            anchors[:, 1],
            anchors[:, 2],
            lengths[2],
        )
        returning = (
            cartesian(
                "return", translated, pre, anchors[:, 2], anchors[:, 0], lengths[3]
            )
            if options.direction == "push"
            else None
        )
        opened = grasp.joint_positions(
            OPEN_COMMAND,
            num_envs=context.batch_size,
            device=self.device,
            dtype=context.robot.qpos.dtype,
        )
        closed = grasp.joint_positions(
            GRASP_COMMAND,
            num_envs=context.batch_size,
            device=self.device,
            dtype=context.robot.qpos.dtype,
        )
        close_hand = interpolate_hand_qpos(
            opened, closed, n_waypoints=options.hand_interp_steps
        )
        open_hand = interpolate_hand_qpos(
            closed, opened, n_waypoints=options.hand_interp_steps
        )
        if options.hand_interp_steps == 1:
            # A one-command phase still issues its terminal semantic command.
            close_hand, open_hand = closed[:, None], opened[:, None]
        parts = [
            (
                "approach",
                transit,
                torch.lerp(
                    context.robot.qpos[:, None, hand_ids],
                    opened[:, None],
                    weights[None, :, None],
                ),
            ),
            ("reach", reach, opened[:, None]),
            (
                "close",
                anchors[:, 1, None].expand(-1, options.hand_interp_steps, -1),
                close_hand,
            ),
            (options.direction, travel, closed[:, None]),
            (
                "open",
                anchors[:, 2, None].expand(-1, options.hand_interp_steps, -1),
                open_hand,
            ),
        ]
        if returning is not None:
            parts.append(("return", returning, opened[:, None]))
        full = (
            context.robot.qpos[:, None]
            .expand(-1, sum(part[1].shape[1] for part in parts), -1)
            .clone()
        )
        offset = 0
        for _, arm_qpos, hand_qpos in parts:
            stop = offset + arm_qpos.shape[1]
            full[:, offset:stop, arm_ids] = arm_qpos
            full[:, offset:stop, hand_ids] = hand_qpos
            offset = stop
        limits = self.robot.get_qpos_limits(env_ids=context.env_ids.tolist())
        if (
            not isinstance(limits, torch.Tensor)
            or limits.shape != (context.batch_size, self.robot_dof, 2)
            or limits.device != full.device
            or limits.dtype != full.dtype
            or torch.isnan(limits).any()
            or (limits[..., 0] > limits[..., 1]).any()
        ):
            raise ValueError(
                "Selected trajectory requires valid full-robot joint limits."
            )
        within_limits = (
            torch.isfinite(full)
            & (full >= limits[:, None, :, 0])
            & (full <= limits[:, None, :, 1])
        ).all(dim=(1, 2))
        continuous = (
            torch.abs(full[:, 1:, arm_ids] - full[:, :-1, arm_ids])
            <= _CANDIDATE_MAX_JOINT_STEP
        ).all(dim=(1, 2))
        for row in (
            torch.nonzero(success & ~(within_limits & continuous)).flatten().tolist()
        ):
            code = "JOINT_LIMIT" if not within_limits[row] else "BRANCH_DISCONTINUITY"
            reasons[row] = f"trajectory:{code}"
        success &= within_limits & continuous
        return self.build_plan(
            request,
            context,
            success=success,
            trajectory=TimedTrajectory.from_uniform_step(
                full,
                env_ids=context.env_ids,
                step_dt=context.require_control_dt(),
            ),
            expected_effects=StateDelta(),
            segment_lengths={name: arm.shape[1] for name, arm, _ in parts},
            scene_dependency_end_segment=(
                "reach" if self._scene_dependencies(request) else None
            ),
            diagnostics=diagnostics(),
        )

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
    "SlideCandidateBatch",
    "SlideGoal",
    "SlideOptions",
]
