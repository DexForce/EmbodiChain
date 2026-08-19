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

"""HandOver atomic action implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch

from embodichain.utils import logger
from embodichain.utils.math import pose_inv

from ..bindings import JointPositionTarget
from ..control import GRASP_COMMAND, OPEN_COMMAND, JointPositionCommand
from ..core import AtomicAction, ObjectSemantics, _same_object_identity
from ..effects import StateDelta
from ..invocation import ActionOptions, ResolvedActionRequest
from ..plans import ActionPlan, TimedTrajectory, normalize_success_mask
from ..requirements import (
    CARTESIAN_POSE_CAPABILITY,
    DisjointResourceSlots,
    FORWARD_KINEMATICS_CAPABILITY,
    SkillBindingContract,
)
from ..state import HeldObjectState, PlanningContext
from ..trajectory_ops import (
    interpolate_hand_qpos,
    translate_pose_world,
)
from ._helpers import (
    assemble_full_robot_trajectory,
    plan_named_arm_trajectory,
    repeat_qpos,
    resolve_batched_pose,
)
from ._binding_contracts import make_manipulation_slot
from .pick_up import GraspGoal


@dataclass(frozen=True, slots=True, eq=False)
class HandOverOptions(ActionOptions):
    """Per-invocation handover behavior and object-pose targets."""

    receive_pick_object_part: str = "bottom"
    """Object part the receiving arm grasps during the handover
    (see :meth:`AntipodalAffordance.get_valid_grasp_poses`)."""

    middle_object_pose: torch.Tensor | None = None
    """Object pose at the handover point where the receiving arm grasps it,
    shape ``(4, 4)`` or ``(num_envs, 4, 4)``. Must be set by the caller."""

    final_object_pose: torch.Tensor | None = None
    """Object pose the receiving arm delivers the object to, shape ``(4, 4)``
    or ``(num_envs, 4, 4)``. Must be set by the caller."""

    receive_approach_direction: torch.Tensor = torch.tensor(
        [0.0, 0.0, -1.0], dtype=torch.float32
    )
    """World-frame approach direction used to sample and approach the receiving
    grasp. Tune this (e.g. to ``[0, 0, 1]`` for a from-below receive grasp) so
    the receiving arm does not collide with the transferring arm."""

    pre_grasp_distance: float = 0.10
    """World distance to offset back from the receiving grasp pose along the
    negative approach direction."""

    lift_height: float = 0.08
    """World-Z lift distance for the transferring arm after it releases."""

    hand_interp_steps: int = 10
    """Number of waypoints used for the receiving-hand close and the
    transferring-hand release interpolations."""

    hold_steps: int = 4
    """Number of waypoints to hold the handoff pose before releasing."""

    retreat_steps: int = 24
    """Number of waypoints used for the final deliver/retreat segment."""

    def __post_init__(self) -> None:
        if not isinstance(self.receive_pick_object_part, str) or not (
            self.receive_pick_object_part
        ):
            raise ValueError("receive_pick_object_part must be non-empty.")
        if self.receive_approach_direction.shape != (3,):
            raise ValueError("receive_approach_direction must have shape (3,).")
        if not torch.isfinite(self.receive_approach_direction).all() or (
            torch.linalg.vector_norm(self.receive_approach_direction) <= 1.0e-6
        ):
            raise ValueError("receive_approach_direction must be finite and non-zero.")
        if self.pre_grasp_distance < 0.0 or self.lift_height < 0.0:
            raise ValueError("pre_grasp_distance and lift_height must be non-negative.")
        for name in ("hand_interp_steps", "hold_steps", "retreat_steps"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative.")
        object.__setattr__(
            self,
            "receive_approach_direction",
            self.receive_approach_direction.clone(),
        )
        for name in ("middle_object_pose", "final_object_pose"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, value.clone())


@dataclass(frozen=True, slots=True, eq=False)
class _HandOverResources:
    """Invocation-bound control parts and compatible hand commands."""

    transfer_arm: JointPositionTarget
    receive_arm: JointPositionTarget
    transfer_hand: JointPositionTarget
    receive_hand: JointPositionTarget
    transfer_hand_open_qpos: torch.Tensor
    transfer_hand_close_qpos: torch.Tensor
    receive_hand_open_qpos: torch.Tensor
    receive_hand_close_qpos: torch.Tensor


class HandOver(AtomicAction[GraspGoal, HandOverOptions]):
    """Hand an object from one arm to the other.

    The transferring arm (already holding the object) moves it to a middle
    handover pose, the receiving arm approaches and grasps a different part of
    the object, the transferring arm releases and retreats, and the receiving
    arm carries the object to a final pose.
    """

    skill_id: ClassVar[str] = "hand_over"
    GoalType: ClassVar[type] = GraspGoal
    OptionsType: ClassVar[type] = HandOverOptions
    binding_contract: ClassVar[SkillBindingContract] = SkillBindingContract(
        slots=(
            make_manipulation_slot(
                "source",
                motion_capabilities=frozenset(
                    {
                        CARTESIAN_POSE_CAPABILITY,
                        FORWARD_KINEMATICS_CAPABILITY,
                    }
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
    _repeat_qpos = staticmethod(repeat_qpos)

    def _scene_dependencies(
        self,
        request: ResolvedActionRequest[GraspGoal, HandOverOptions],
    ) -> tuple[str, ...]:
        """Return no goal-pose dependency because handover ignores grasp_xpos."""
        del request
        return ()

    def _resolve_resources(
        self,
        request: ResolvedActionRequest[GraspGoal, HandOverOptions],
    ) -> _HandOverResources:
        """Resolve source/destination roles from robot control parts."""
        binding = request.binding
        transfer_motion = binding.endpoint("source", "motion")
        receive_motion = binding.endpoint("destination", "motion")
        transfer_grasp = binding.endpoint("source", "grasp")
        receive_grasp = binding.endpoint("destination", "grasp")
        transfer_arm = transfer_motion.require_target(JointPositionTarget)
        receive_arm = receive_motion.require_target(JointPositionTarget)
        transfer_hand = transfer_grasp.require_target(JointPositionTarget)
        receive_hand = receive_grasp.require_target(JointPositionTarget)
        if transfer_arm.control_part == receive_arm.control_part:
            raise ValueError(
                "HandOver source and destination must use different manipulator "
                "control parts."
            )
        if transfer_hand.control_part == receive_hand.control_part:
            raise ValueError(
                "HandOver source and destination must use different end-effector "
                "control parts."
            )
        return _HandOverResources(
            transfer_arm=transfer_arm,
            receive_arm=receive_arm,
            transfer_hand=transfer_hand,
            receive_hand=receive_hand,
            transfer_hand_open_qpos=transfer_grasp.joint_positions(
                OPEN_COMMAND,
                num_envs=self.num_envs,
                device=self.device,
                dtype=torch.float32,
            ),
            transfer_hand_close_qpos=transfer_grasp.joint_positions(
                GRASP_COMMAND,
                num_envs=self.num_envs,
                device=self.device,
                dtype=torch.float32,
            ),
            receive_hand_open_qpos=receive_grasp.joint_positions(
                OPEN_COMMAND,
                num_envs=self.num_envs,
                device=self.device,
                dtype=torch.float32,
            ),
            receive_hand_close_qpos=receive_grasp.joint_positions(
                GRASP_COMMAND,
                num_envs=self.num_envs,
                device=self.device,
                dtype=torch.float32,
            ),
        )

    # ------------------------------------------------------------------
    # Public contract
    # ------------------------------------------------------------------

    def _plan(
        self,
        request: ResolvedActionRequest[GraspGoal, HandOverOptions],
        context: PlanningContext,
    ) -> ActionPlan:
        """Plan a handover without committing the attachment transfer."""
        target = request.goal
        options = request.skill_options
        self._validate_pose_options(options)
        resources = self._resolve_resources(request)
        if (
            request.motion_policy.strategy == "motion_gen"
            and self.motion_generator.planner.cfg.planner_type == "curobo"
        ):
            raise ValueError(
                "Coordinated dual-arm planning is not supported by the cuRobo backend."
            )
        state = context
        transfer_control_part = resources.transfer_arm.control_part
        transfer_held_object = self._resolve_transfer_held_object(
            state, transfer_control_part
        )
        self._validate_requested_object(
            target.semantics, transfer_held_object.semantics
        )
        semantics = transfer_held_object.semantics
        eligible = context.task.exclusive_held_object_mask(transfer_control_part)
        if not eligible.any():
            logger.log_warning("HandOver requires an exclusively held source object.")
            return self.failed_plan(
                request,
                context,
                message="Source object must be held exclusively.",
            )
        transfer_object_to_eef = self._resolve_matrix(
            transfer_held_object.object_to_eef,
            "held_object.object_to_eef",
        )
        transfer_start_qpos, receive_start_qpos = self._resolve_start_qpos(
            state,
            resources,
        )
        assert options.middle_object_pose is not None
        assert options.final_object_pose is not None
        middle_object_pose = self._resolve_matrix(
            options.middle_object_pose, "middle_object_pose"
        )
        final_object_pose = self._resolve_matrix(
            options.final_object_pose, "final_object_pose"
        )
        receive_approach_direction = options.receive_approach_direction.to(
            device=self.device, dtype=torch.float32
        )
        receive_approach_direction = (
            receive_approach_direction
            / torch.linalg.vector_norm(receive_approach_direction)
        )
        # Keep the requested object orientation consistent with the verified
        # attachment and the transferring arm's current measured pose.
        transfer_current_eef = self.robot.compute_fk(
            qpos=transfer_start_qpos,
            name=resources.transfer_arm.control_part,
            to_matrix=True,
        )
        current_object_pose = torch.bmm(
            transfer_current_eef,
            pose_inv(transfer_object_to_eef),
        )
        middle_object_pose[:, :3, :3] = current_object_pose[:, :3, :3]
        final_object_pose[:, :3, :3] = current_object_pose[:, :3, :3]

        # 2.1 - EEF target that keeps the object at the handover pose.
        transfer_middle_eef = torch.bmm(middle_object_pose, transfer_object_to_eef)

        # 2.2 - receiving grasp on the requested object part at the handover pose.
        receive_grasp_xpos, grasp_success = self._resolve_receive_grasp(
            semantics,
            middle_object_pose,
            options.receive_pick_object_part,
            receive_approach_direction,
        )
        success_mask = normalize_success_mask(
            grasp_success,
            num_envs=self.num_envs,
            device=self.device,
            name="Receiving-grasp success",
        )
        success_mask &= eligible
        if not success_mask.any():
            logger.log_warning("HandOver failed to resolve a receiving grasp pose.")
            return self.failed_plan(request, context, message="No receiving grasp.")
        receive_object_to_eef = torch.bmm(
            pose_inv(middle_object_pose), receive_grasp_xpos
        )
        receive_grasp_z = receive_grasp_xpos[..., :3, 2]
        receive_pre_grasp_eef = translate_pose_world(
            receive_grasp_xpos,
            -receive_grasp_z * options.pre_grasp_distance,
        )
        # 2.4 - receiving arm delivers the object to the final pose.
        receive_final_eef = torch.bmm(final_object_pose, receive_object_to_eef)
        # 2.3 - transferring arm retreats upward after releasing.
        transfer_retreat_eef = translate_pose_world(
            transfer_middle_eef,
            torch.tensor(
                [0.0, 0.0, options.lift_height],
                dtype=torch.float32,
                device=self.device,
            ),
        )

        segments = self._compute_segment_lengths(
            request.motion_policy.sample_count, options
        )

        segment_success, transfer_move_traj = plan_named_arm_trajectory(
            self.motion_generator,
            resources.transfer_arm.control_part,
            transfer_start_qpos,
            transfer_middle_eef.unsqueeze(1),
            segments["transfer"],
            request.motion_policy,
            context.control_dt,
        )
        success_mask &= normalize_success_mask(
            segment_success,
            num_envs=self.num_envs,
            device=self.device,
            name="Transfer-move success",
        )
        if not success_mask.any():
            logger.log_warning("HandOver failed to plan the transfer move.")
            return self.failed_plan(request, context, message="Transfer move failed.")

        segment_success, receive_approach_traj = plan_named_arm_trajectory(
            self.motion_generator,
            resources.receive_arm.control_part,
            receive_start_qpos,
            torch.stack([receive_pre_grasp_eef, receive_grasp_xpos], dim=1),
            segments["approach"],
            request.motion_policy,
            context.control_dt,
        )
        success_mask &= normalize_success_mask(
            segment_success,
            num_envs=self.num_envs,
            device=self.device,
            name="Receiving-approach success",
        )
        if not success_mask.any():
            logger.log_warning("HandOver failed to plan the receiving approach.")
            return self.failed_plan(
                request, context, message="Receiving approach failed."
            )

        transfer_hold_qpos = transfer_move_traj[:, -1]
        receive_grasp_qpos = receive_approach_traj[:, -1]

        segment_success, transfer_retreat_traj = plan_named_arm_trajectory(
            self.motion_generator,
            resources.transfer_arm.control_part,
            transfer_hold_qpos,
            transfer_retreat_eef.unsqueeze(1),
            segments["deliver"],
            request.motion_policy,
            context.control_dt,
        )
        success_mask &= normalize_success_mask(
            segment_success,
            num_envs=self.num_envs,
            device=self.device,
            name="Transfer-retreat success",
        )
        if not success_mask.any():
            logger.log_warning("HandOver failed to plan the transfer retreat.")
            return self.failed_plan(
                request, context, message="Transfer retreat failed."
            )

        segment_success, receive_deliver_traj = plan_named_arm_trajectory(
            self.motion_generator,
            resources.receive_arm.control_part,
            receive_grasp_qpos,
            receive_final_eef.unsqueeze(1),
            segments["deliver"],
            request.motion_policy,
            context.control_dt,
        )
        success_mask &= normalize_success_mask(
            segment_success,
            num_envs=self.num_envs,
            device=self.device,
            name="Receiving-delivery success",
        )
        if not success_mask.any():
            logger.log_warning("HandOver failed to plan the receiving delivery.")
            return self.failed_plan(
                request, context, message="Receiving delivery failed."
            )

        segment_trajectories: list[torch.Tensor] = []
        # 2.1 transfer: transferring arm carries the object to the middle pose.
        segment_trajectories.append(
            self._assemble_segment(
                state,
                transfer_move_traj,
                self._repeat_qpos(receive_start_qpos, segments["transfer"]),
                self._repeat_qpos(
                    resources.transfer_hand_close_qpos, segments["transfer"]
                ),
                self._repeat_qpos(
                    resources.receive_hand_open_qpos, segments["transfer"]
                ),
                resources=resources,
            )
        )
        # 2.2 approach: receiving arm moves to the grasp pose; transferring arm holds.
        segment_trajectories.append(
            self._assemble_segment(
                state,
                self._repeat_qpos(transfer_hold_qpos, segments["approach"]),
                receive_approach_traj,
                self._repeat_qpos(
                    resources.transfer_hand_close_qpos, segments["approach"]
                ),
                self._repeat_qpos(
                    resources.receive_hand_open_qpos, segments["approach"]
                ),
                resources=resources,
            )
        )
        # 2.2 close: receiving hand closes; transferring arm keeps holding.
        segment_trajectories.append(
            self._assemble_segment(
                state,
                self._repeat_qpos(transfer_hold_qpos, segments["close"]),
                self._repeat_qpos(receive_grasp_qpos, segments["close"]),
                self._repeat_qpos(
                    resources.transfer_hand_close_qpos, segments["close"]
                ),
                interpolate_hand_qpos(
                    resources.receive_hand_open_qpos,
                    resources.receive_hand_close_qpos,
                    n_waypoints=segments["close"],
                ),
                resources=resources,
            )
        )
        if segments["hold"] > 0:
            segment_trajectories.append(
                self._assemble_segment(
                    state,
                    self._repeat_qpos(transfer_hold_qpos, segments["hold"]),
                    self._repeat_qpos(receive_grasp_qpos, segments["hold"]),
                    self._repeat_qpos(
                        resources.transfer_hand_close_qpos, segments["hold"]
                    ),
                    self._repeat_qpos(
                        resources.receive_hand_close_qpos, segments["hold"]
                    ),
                    resources=resources,
                )
            )
        # 2.3 release: transferring hand opens; receiving arm keeps holding.
        segment_trajectories.append(
            self._assemble_segment(
                state,
                self._repeat_qpos(transfer_hold_qpos, segments["release"]),
                self._repeat_qpos(receive_grasp_qpos, segments["release"]),
                interpolate_hand_qpos(
                    resources.transfer_hand_close_qpos,
                    resources.transfer_hand_open_qpos,
                    n_waypoints=segments["release"],
                ),
                self._repeat_qpos(
                    resources.receive_hand_close_qpos, segments["release"]
                ),
                resources=resources,
            )
        )
        # 2.4 deliver: receiving arm carries the object away; transferring arm retreats.
        segment_trajectories.append(
            self._assemble_segment(
                state,
                transfer_retreat_traj,
                receive_deliver_traj,
                self._repeat_qpos(
                    resources.transfer_hand_open_qpos, segments["deliver"]
                ),
                self._repeat_qpos(
                    resources.receive_hand_close_qpos, segments["deliver"]
                ),
                resources=resources,
            )
        )
        full = torch.cat(segment_trajectories, dim=1)
        segment_names = ["transfer", "approach", "close"]
        if segments["hold"] > 0:
            segment_names.append("hold")
        segment_names.extend(("release", "deliver"))
        segment_lengths = {
            name: trajectory.shape[1]
            for name, trajectory in zip(
                segment_names, segment_trajectories, strict=True
            )
        }
        held_object = HeldObjectState(
            semantics=semantics,
            object_to_eef=receive_object_to_eef,
            grasp_xpos=receive_grasp_xpos,
        )
        return self.build_plan(
            request,
            context,
            success=success_mask,
            trajectory=TimedTrajectory.from_uniform_step(
                full,
                env_ids=context.env_ids,
                step_dt=context.require_control_dt(),
            ),
            expected_effects=StateDelta(
                held_object_updates={
                    resources.transfer_arm.control_part: None,
                    resources.receive_arm.control_part: held_object,
                }
            ),
            segment_lengths=segment_lengths,
        )

    # ------------------------------------------------------------------
    # Resolution helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_pose_options(options: HandOverOptions) -> None:
        for name in ("middle_object_pose", "final_object_pose"):
            if getattr(options, name) is None:
                raise ValueError(f"{name} must be specified in HandOverOptions")

    def _resolve_matrix(self, matrix: torch.Tensor, name: str) -> torch.Tensor:
        return resolve_batched_pose(
            matrix,
            num_envs=self.num_envs,
            device=self.device,
            name=name,
        )

    def _resolve_transfer_held_object(
        self,
        state: PlanningContext,
        transfer_control_part: str,
    ) -> HeldObjectState:
        held = state.get_held_object(transfer_control_part)
        if held is None:
            raise ValueError(
                "HandOver requires an object held by transfer control part "
                f"{transfer_control_part!r} (run PickUp first)."
            )
        return held

    @staticmethod
    def _validate_requested_object(
        requested: ObjectSemantics,
        held: ObjectSemantics,
    ) -> None:
        """Reject a request that names a different grounded object."""
        if not _same_object_identity(requested, held):
            raise ValueError(
                "HandOver goal semantics must identify the object held by the "
                "source control part."
            )

    def _resolve_receive_grasp(
        self,
        semantics: ObjectSemantics,
        object_pose: torch.Tensor,
        object_part: str,
        approach_direction: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select the lowest-cost receiving grasp on ``object_part`` at ``object_pose``."""
        grasp_poses_result = semantics.affordance.get_valid_grasp_poses(
            obj_poses=object_pose,
            approach_direction=approach_direction,
            object_part=object_part,
        )
        num_envs = object_pose.shape[0]
        grasp_xpos = (
            torch.eye(4, device=self.device, dtype=torch.float32)
            .unsqueeze(0)
            .repeat(num_envs, 1, 1)
        )
        is_success = torch.ones(num_envs, dtype=torch.bool, device=self.device)
        for i in range(num_envs):
            poses, costs = grasp_poses_result[i]
            poses = poses.to(device=self.device, dtype=torch.float32)
            costs = costs.to(device=self.device, dtype=torch.float32)
            if poses.shape[0] == 0:
                is_success[i] = False
                continue
            best_idx = torch.argmin(costs)
            if not torch.isfinite(costs[best_idx]):
                is_success[i] = False
            grasp_xpos[i] = poses[best_idx]
        return grasp_xpos, is_success

    def _resolve_start_qpos(
        self,
        state: PlanningContext,
        resources: _HandOverResources,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if state.last_qpos.shape != (self.num_envs, self.robot_dof):
            raise ValueError(
                f"PlanningContext.last_qpos must have shape "
                f"({self.num_envs}, {self.robot_dof}), but got "
                f"{state.last_qpos.shape}"
            )
        start_qpos = state.last_qpos.to(device=self.device, dtype=torch.float32)
        return (
            start_qpos[:, list(resources.transfer_arm.joint_ids)],
            start_qpos[:, list(resources.receive_arm.joint_ids)],
        )

    def _compute_segment_lengths(
        self, sample_count: int, options: HandOverOptions
    ) -> dict[str, int]:
        """Split the invocation sample budget across handover segments."""
        n_close = max(2, options.hand_interp_steps)
        n_release = max(2, options.hand_interp_steps)
        n_deliver = max(2, options.retreat_steps)
        n_hold = max(0, options.hold_steps)
        reserved = n_close + n_release + n_deliver + n_hold
        n_transfer = max(2, (sample_count - reserved) // 2)
        n_approach = sample_count - reserved - n_transfer
        if n_approach < 2:
            raise ValueError(
                "Not enough waypoints for handover. Increase sample_count or "
                "decrease hand_interp_steps/hold_steps/retreat_steps."
            )
        return {
            "transfer": n_transfer,
            "approach": n_approach,
            "close": n_close,
            "hold": n_hold,
            "release": n_release,
            "deliver": n_deliver,
        }

    # ------------------------------------------------------------------
    # Planning / assembly helpers
    # ------------------------------------------------------------------

    def _assemble_segment(
        self,
        state: PlanningContext,
        transfer_arm_traj: torch.Tensor,
        receive_arm_traj: torch.Tensor,
        transfer_hand_traj: torch.Tensor,
        receive_hand_traj: torch.Tensor,
        *,
        resources: _HandOverResources,
    ) -> torch.Tensor:
        return assemble_full_robot_trajectory(
            state.last_qpos,
            (
                (resources.transfer_arm.joint_ids, transfer_arm_traj),
                (resources.receive_arm.joint_ids, receive_arm_traj),
                (resources.transfer_hand.joint_ids, transfer_hand_traj),
                (resources.receive_hand.joint_ids, receive_hand_traj),
            ),
        )


__all__ = ["HandOver", "HandOverOptions"]
