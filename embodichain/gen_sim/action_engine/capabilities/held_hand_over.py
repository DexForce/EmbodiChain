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

"""GenSim-local handover for transferring an already-held object."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch

from embodichain.lab.sim.atomic_actions import (
    ActionOptions,
    ActionPlan,
    AntipodalAffordance,
    AtomicAction,
    CARTESIAN_POSE_CAPABILITY,
    DisjointResourceSlots,
    FORWARD_KINEMATICS_CAPABILITY,
    GRASP_COMMAND,
    GraspGoal,
    HeldObjectState,
    JointPositionCommand,
    JointPositionTarget,
    ObjectSemantics,
    OPEN_COMMAND,
    PlanningContext,
    ResolvedActionRequest,
    SkillBindingContract,
    StateDelta,
    TimedTrajectory,
)
from embodichain.lab.sim.atomic_actions.goals import (
    PoseGoalValue,
    collect_scene_dependencies,
    resolve_pose_goal,
    validate_pose_goal,
)
from embodichain.lab.sim.atomic_actions.primitives._binding_contracts import (
    make_manipulation_slot,
)
from embodichain.lab.sim.atomic_actions.primitives._helpers import (
    assemble_full_robot_trajectory,
    plan_named_arm_trajectory,
    repeat_qpos,
    require_shared_task_state_key,
    resolve_batched_pose,
)
from embodichain.lab.sim.atomic_actions.trajectory_ops import (
    interpolate_hand_qpos,
    translate_pose_world,
)
from embodichain.lab.sim.planners.utils import normalize_success_mask
from embodichain.utils.math import pose_inv


@dataclass(frozen=True, slots=True, eq=False)
class HeldObjectHandOverOptions(ActionOptions):
    """Per-invocation behavior for transferring an already-held object."""

    receive_pick_object_part: str = "bottom"
    middle_object_pose: PoseGoalValue | None = None
    final_object_pose: PoseGoalValue | None = None
    receive_approach_direction: torch.Tensor = torch.tensor([0.0, 0.0, -1.0])
    pre_grasp_distance: float = 0.10
    lift_height: float = 0.08
    hand_interp_steps: int = 10
    hold_steps: int = 4
    retreat_steps: int = 24

    def __post_init__(self) -> None:
        if self.receive_pick_object_part not in frozenset({"center", "top", "bottom"}):
            raise ValueError(
                "receive_pick_object_part must be 'center', 'top', or 'bottom'."
            )
        direction = self.receive_approach_direction
        if (
            not isinstance(direction, torch.Tensor)
            or direction.shape != (3,)
            or not torch.isfinite(direction).all()
            or torch.linalg.vector_norm(direction) <= 1.0e-6
        ):
            raise ValueError(
                "receive_approach_direction must be a finite non-zero (3,) tensor."
            )
        if self.pre_grasp_distance < 0.0 or self.lift_height < 0.0:
            raise ValueError("Handover distances must be non-negative.")
        for name in ("hand_interp_steps", "hold_steps", "retreat_steps"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")

        object.__setattr__(self, "receive_approach_direction", direction.clone())
        for name in ("middle_object_pose", "final_object_pose"):
            value = getattr(self, name)
            if value is None:
                continue
            validate_pose_goal(value, name, allow_waypoints=False)
            object.__setattr__(
                self,
                name,
                value.clone() if isinstance(value, torch.Tensor) else value.snapshot(),
            )


@dataclass(frozen=True, slots=True)
class _HandoverResources:
    source_state_key: str
    destination_state_key: str
    source_arm: JointPositionTarget
    destination_arm: JointPositionTarget
    source_hand: JointPositionTarget
    destination_hand: JointPositionTarget
    source_hand_open_qpos: torch.Tensor
    source_hand_grasp_qpos: torch.Tensor
    destination_hand_open_qpos: torch.Tensor
    destination_hand_grasp_qpos: torch.Tensor


class HeldObjectHandOver(AtomicAction[GraspGoal, HeldObjectHandOverOptions]):
    """Transfer an existing attachment while leaving the receiver holding it."""

    skill_id: ClassVar[str] = "hand_over"
    GoalType: ClassVar[type] = GraspGoal
    OptionsType: ClassVar[type] = HeldObjectHandOverOptions
    manipulator_roles: ClassVar[tuple[str, ...]] = ("source", "destination")
    end_effector_roles: ClassVar[tuple[str, ...]] = ("source", "destination")
    open_loop: ClassVar[bool] = True
    binding_contract: ClassVar[SkillBindingContract] = SkillBindingContract(
        slots=(
            make_manipulation_slot(
                "source",
                motion_capabilities=frozenset(
                    {CARTESIAN_POSE_CAPABILITY, FORWARD_KINEMATICS_CAPABILITY}
                ),
                grasp_commands={
                    OPEN_COMMAND: JointPositionCommand,
                    GRASP_COMMAND: JointPositionCommand,
                },
            ),
            make_manipulation_slot(
                "destination",
                motion_capabilities=frozenset({CARTESIAN_POSE_CAPABILITY}),
                grasp_commands={
                    OPEN_COMMAND: JointPositionCommand,
                    GRASP_COMMAND: JointPositionCommand,
                },
            ),
        ),
        constraints=(DisjointResourceSlots(("source", "destination")),),
    )

    def __init__(
        self, default_options: HeldObjectHandOverOptions | None = None
    ) -> None:
        super().__init__(default_options)

    def _scene_dependencies(
        self,
        request: ResolvedActionRequest[GraspGoal, HeldObjectHandOverOptions],
    ) -> tuple[str, ...]:
        """Return scene entities whose poses materially affect this plan."""
        return collect_scene_dependencies(
            tuple(
                value
                for value in (
                    request.skill_options.middle_object_pose,
                    request.skill_options.final_object_pose,
                )
                if value is not None
            )
        )

    def _plan(
        self,
        request: ResolvedActionRequest[GraspGoal, HeldObjectHandOverOptions],
        context: PlanningContext,
    ) -> ActionPlan:
        goal = self.require_goal(request)
        options = request.skill_options
        self._require_exchange_pose(options)
        resources = self._resolve_resources(request)

        if (
            request.motion_policy.strategy == "motion_gen"
            and self.motion_generator.planner.cfg.planner_type == "curobo"
        ):
            raise ValueError(
                "Coordinated dual-arm planning is not supported by cuRobo."
            )

        held = context.get_held_object(resources.source_state_key)
        if held is None:
            raise ValueError(
                "HeldObjectHandOver requires the source participant to hold an object."
            )
        self._require_same_object(goal.semantics, held.semantics)
        eligible = context.task.exclusive_held_object_mask(resources.source_state_key)
        if not eligible.any():
            return self.failed_plan(
                request,
                context,
                message="Source object must be held exclusively.",
            )

        source_start, destination_start = self._start_qpos(context, resources)
        object_to_source = self._pose(held.object_to_eef, "held.object_to_eef")
        source_eef = self.robot.compute_fk(
            qpos=source_start,
            name=resources.source_arm.control_part,
            to_matrix=True,
        )
        current_object_pose = torch.bmm(source_eef, pose_inv(object_to_source))
        assert options.middle_object_pose is not None
        assert options.final_object_pose is not None
        middle_object_pose = self._pose(
            resolve_pose_goal(
                options.middle_object_pose,
                context,
                name="middle_object_pose",
            ),
            "middle_object_pose",
        )
        final_object_pose = self._pose(
            resolve_pose_goal(
                options.final_object_pose,
                context,
                name="final_object_pose",
            ),
            "final_object_pose",
        )
        middle_object_pose[:, :3, :3] = current_object_pose[:, :3, :3]
        final_object_pose[:, :3, :3] = current_object_pose[:, :3, :3]
        if not torch.allclose(
            middle_object_pose,
            final_object_pose,
            atol=1.0e-5,
            rtol=1.0e-5,
        ):
            raise ValueError(
                "HeldObjectHandOver requires final_object_pose to match the exchange "
                "pose so the receiver remains stationary."
            )

        source_middle_eef = torch.bmm(middle_object_pose, object_to_source)
        destination_grasp, grasp_success = self._destination_grasp(
            held.semantics,
            middle_object_pose,
            resources.destination_hand.control_part,
            options,
        )
        success = normalize_success_mask(
            grasp_success,
            num_envs=self.num_envs,
            device=self.device,
            name="Receiving-grasp success",
        )
        success &= eligible
        if not success.any():
            return self.failed_plan(
                request,
                context,
                message="No receiving grasp was available.",
            )

        object_to_destination = torch.bmm(
            pose_inv(middle_object_pose), destination_grasp
        )
        destination_pre_grasp = translate_pose_world(
            destination_grasp,
            -destination_grasp[:, :3, 2] * options.pre_grasp_distance,
        )
        source_retreat_eef = translate_pose_world(
            source_middle_eef,
            source_middle_eef.new_tensor([0.0, 0.0, options.lift_height]),
        )
        lengths = self._segment_lengths(request.motion_policy.sample_count, options)

        segment_success, source_transfer = plan_named_arm_trajectory(
            self.motion_generator,
            resources.source_arm.control_part,
            source_start,
            source_middle_eef.unsqueeze(1),
            lengths["transfer"],
            request.motion_policy,
            context.control_dt,
        )
        success &= self._success(segment_success, "Source transfer")
        segment_success, destination_approach = plan_named_arm_trajectory(
            self.motion_generator,
            resources.destination_arm.control_part,
            destination_start,
            torch.stack((destination_pre_grasp, destination_grasp), dim=1),
            lengths["approach"],
            request.motion_policy,
            context.control_dt,
        )
        success &= self._success(segment_success, "Destination approach")
        source_hold = source_transfer[:, -1]
        destination_hold = destination_approach[:, -1]
        segment_success, source_retreat = plan_named_arm_trajectory(
            self.motion_generator,
            resources.source_arm.control_part,
            source_hold,
            source_retreat_eef.unsqueeze(1),
            lengths["retreat"],
            request.motion_policy,
            context.control_dt,
        )
        success &= self._success(segment_success, "Source retreat")
        if not success.any():
            return self.failed_plan(
                request,
                context,
                message="Handover arm planning failed.",
            )

        segments = [
            (
                "transfer",
                self._segment(
                    context,
                    resources,
                    source_transfer,
                    repeat_qpos(destination_start, lengths["transfer"]),
                    repeat_qpos(resources.source_hand_grasp_qpos, lengths["transfer"]),
                    repeat_qpos(
                        resources.destination_hand_open_qpos, lengths["transfer"]
                    ),
                ),
            ),
            (
                "approach",
                self._segment(
                    context,
                    resources,
                    repeat_qpos(source_hold, lengths["approach"]),
                    destination_approach,
                    repeat_qpos(resources.source_hand_grasp_qpos, lengths["approach"]),
                    repeat_qpos(
                        resources.destination_hand_open_qpos, lengths["approach"]
                    ),
                ),
            ),
            (
                "close",
                self._segment(
                    context,
                    resources,
                    repeat_qpos(source_hold, lengths["close"]),
                    repeat_qpos(destination_hold, lengths["close"]),
                    repeat_qpos(resources.source_hand_grasp_qpos, lengths["close"]),
                    interpolate_hand_qpos(
                        resources.destination_hand_open_qpos,
                        resources.destination_hand_grasp_qpos,
                        n_waypoints=lengths["close"],
                    ),
                ),
            ),
        ]
        if lengths["hold"]:
            segments.append(
                (
                    "hold",
                    self._segment(
                        context,
                        resources,
                        repeat_qpos(source_hold, lengths["hold"]),
                        repeat_qpos(destination_hold, lengths["hold"]),
                        repeat_qpos(resources.source_hand_grasp_qpos, lengths["hold"]),
                        repeat_qpos(
                            resources.destination_hand_grasp_qpos, lengths["hold"]
                        ),
                    ),
                )
            )
        segments.extend(
            (
                (
                    "release",
                    self._segment(
                        context,
                        resources,
                        repeat_qpos(source_hold, lengths["release"]),
                        repeat_qpos(destination_hold, lengths["release"]),
                        interpolate_hand_qpos(
                            resources.source_hand_grasp_qpos,
                            resources.source_hand_open_qpos,
                            n_waypoints=lengths["release"],
                        ),
                        repeat_qpos(
                            resources.destination_hand_grasp_qpos,
                            lengths["release"],
                        ),
                    ),
                ),
                (
                    "retreat",
                    self._segment(
                        context,
                        resources,
                        source_retreat,
                        repeat_qpos(destination_hold, lengths["retreat"]),
                        repeat_qpos(
                            resources.source_hand_open_qpos, lengths["retreat"]
                        ),
                        repeat_qpos(
                            resources.destination_hand_grasp_qpos,
                            lengths["retreat"],
                        ),
                    ),
                ),
            )
        )

        trajectory = torch.cat([value for _, value in segments], dim=1)
        received = HeldObjectState(
            semantics=held.semantics,
            object_to_eef=object_to_destination,
            grasp_xpos=destination_grasp,
        )
        return self.build_plan(
            request,
            context,
            success=success,
            trajectory=TimedTrajectory.from_uniform_step(
                trajectory,
                env_ids=context.env_ids,
                step_dt=context.require_control_dt(),
            ),
            expected_effects=StateDelta(
                held_object_updates={
                    resources.source_state_key: None,
                    resources.destination_state_key: received,
                }
            ),
            segment_lengths={name: value.shape[1] for name, value in segments},
        )

    def _resolve_resources(
        self,
        request: ResolvedActionRequest[GraspGoal, HeldObjectHandOverOptions],
    ) -> _HandoverResources:
        binding = request.binding
        source_motion = binding.endpoint("source", "motion")
        source_grasp = binding.endpoint("source", "grasp")
        destination_motion = binding.endpoint("destination", "motion")
        destination_grasp = binding.endpoint("destination", "grasp")
        source_arm = source_motion.require_target(JointPositionTarget)
        source_hand = source_grasp.require_target(JointPositionTarget)
        destination_arm = destination_motion.require_target(JointPositionTarget)
        destination_hand = destination_grasp.require_target(JointPositionTarget)
        source_key = require_shared_task_state_key(
            source_motion,
            source_grasp,
            participant="HeldObjectHandOver source",
        )
        destination_key = require_shared_task_state_key(
            destination_motion,
            destination_grasp,
            participant="HeldObjectHandOver destination",
        )
        if source_key == destination_key:
            raise ValueError("Handover participants require different state keys.")
        return _HandoverResources(
            source_state_key=source_key,
            destination_state_key=destination_key,
            source_arm=source_arm,
            destination_arm=destination_arm,
            source_hand=source_hand,
            destination_hand=destination_hand,
            source_hand_open_qpos=source_grasp.joint_positions(
                OPEN_COMMAND,
                num_envs=self.num_envs,
                device=self.device,
                dtype=torch.float32,
            ),
            source_hand_grasp_qpos=source_grasp.joint_positions(
                GRASP_COMMAND,
                num_envs=self.num_envs,
                device=self.device,
                dtype=torch.float32,
            ),
            destination_hand_open_qpos=destination_grasp.joint_positions(
                OPEN_COMMAND,
                num_envs=self.num_envs,
                device=self.device,
                dtype=torch.float32,
            ),
            destination_hand_grasp_qpos=destination_grasp.joint_positions(
                GRASP_COMMAND,
                num_envs=self.num_envs,
                device=self.device,
                dtype=torch.float32,
            ),
        )

    def _destination_grasp(
        self,
        semantics: ObjectSemantics,
        object_pose: torch.Tensor,
        grasp_target_id: str,
        options: HeldObjectHandOverOptions,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        affordance = semantics.affordance
        if not isinstance(affordance, AntipodalAffordance):
            raise ValueError("HeldObjectHandOver requires AntipodalAffordance.")
        direction = options.receive_approach_direction.to(
            device=self.device, dtype=torch.float32
        )
        direction = direction / torch.linalg.vector_norm(direction)
        direction = direction.expand(self.num_envs, -1)
        axis = None
        positive: bool | torch.Tensor = True
        if options.receive_pick_object_part != "center":
            local_axis = object_pose.new_tensor([0.0, 0.0, 1.0])
            axis = torch.matmul(object_pose[:, :3, :3], local_axis)
            positive = torch.full(
                (self.num_envs,),
                options.receive_pick_object_part == "top",
                dtype=torch.bool,
                device=self.device,
            )

        generator = self.planning_services.grasp_pose_generator(grasp_target_id)
        sampled = generator.get_valid_grasp_poses(
            mesh_vertices=affordance.mesh_vertices,
            mesh_triangles=affordance.mesh_triangles,
            obj_poses=object_pose,
            approach_direction=direction,
            obj_longest_axis=axis,
            is_positive_part=positive,
        )
        poses = torch.eye(4, dtype=torch.float32, device=self.device).repeat(
            self.num_envs, 1, 1
        )
        success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for env_index, (candidates, costs) in enumerate(sampled):
            candidates = candidates.to(device=self.device, dtype=torch.float32)
            costs = costs.to(device=self.device, dtype=torch.float32)
            finite = torch.isfinite(costs)
            if candidates.shape[0] == 0 or not finite.any():
                continue
            ranked = torch.where(finite, costs, torch.inf)
            poses[env_index] = candidates[torch.argmin(ranked)]
            success[env_index] = True
        return poses, success

    @staticmethod
    def _segment(
        context: PlanningContext,
        resources: _HandoverResources,
        source_arm: torch.Tensor,
        destination_arm: torch.Tensor,
        source_hand: torch.Tensor,
        destination_hand: torch.Tensor,
    ) -> torch.Tensor:
        return assemble_full_robot_trajectory(
            context.robot.qpos,
            (
                (resources.source_arm.joint_ids, source_arm),
                (resources.destination_arm.joint_ids, destination_arm),
                (resources.source_hand.joint_ids, source_hand),
                (resources.destination_hand.joint_ids, destination_hand),
            ),
        )

    def _success(self, value: torch.Tensor, name: str) -> torch.Tensor:
        return normalize_success_mask(
            value,
            num_envs=self.num_envs,
            device=self.device,
            name=name,
        )

    def _pose(self, value: torch.Tensor, name: str) -> torch.Tensor:
        return resolve_batched_pose(
            value,
            num_envs=self.num_envs,
            device=self.device,
            name=name,
        )

    @staticmethod
    def _require_exchange_pose(options: HeldObjectHandOverOptions) -> None:
        if options.middle_object_pose is None or options.final_object_pose is None:
            raise ValueError(
                "middle_object_pose and final_object_pose are required for "
                "HeldObjectHandOver."
            )

    @staticmethod
    def _require_same_object(
        requested: ObjectSemantics,
        held: ObjectSemantics,
    ) -> None:
        if requested.entity_id is not None and held.entity_id is not None:
            matches = requested.entity_id == held.entity_id
        elif requested.entity is not None and held.entity is not None:
            matches = requested.entity is held.entity
        else:
            matches = bool(requested.label) and requested.label == held.label
        if not matches:
            raise ValueError(
                "Handover goal must identify the object held by the source."
            )

    @staticmethod
    def _start_qpos(
        context: PlanningContext,
        resources: _HandoverResources,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        qpos = context.robot.qpos.to(dtype=torch.float32)
        return (
            qpos[:, list(resources.source_arm.joint_ids)],
            qpos[:, list(resources.destination_arm.joint_ids)],
        )

    @staticmethod
    def _segment_lengths(
        sample_count: int,
        options: HeldObjectHandOverOptions,
    ) -> dict[str, int]:
        close = max(2, options.hand_interp_steps)
        release = max(2, options.hand_interp_steps)
        retreat = max(2, options.retreat_steps)
        hold = options.hold_steps
        reserved = close + release + retreat + hold
        transfer = max(2, (sample_count - reserved) // 2)
        approach = sample_count - reserved - transfer
        if approach < 2:
            raise ValueError(
                "Not enough handover waypoints; increase sample_count or reduce "
                "handover segment lengths."
            )
        return {
            "transfer": transfer,
            "approach": approach,
            "close": close,
            "hold": hold,
            "release": release,
            "retreat": retreat,
        }


__all__ = ["HeldObjectHandOver", "HeldObjectHandOverOptions"]
