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

"""Place atomic action implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal

import torch

from embodichain.lab.sim.planners import MoveType, PlanState
from embodichain.utils import configclass, logger
from embodichain.utils.math import quat_error_magnitude, quat_from_matrix

from ._helpers import arm_qpos_from_state, resolve_object_target
from ..affordance import AssembleAffordance
from ..core import (
    ActionTarget,
    ActionCfg,
    ActionResult,
    AtomicAction,
    WorldState,
    _validate_pose_tensor,
)
from ..trajectory import TrajectoryBuilder

TcpSymmetry = Literal["none", "z_roll_180"]


@dataclass(frozen=True, slots=True, eq=False)
class PlaceTarget(ActionTarget):
    """End-effector release-pose target used by :class:`Place`."""

    xpos: torch.Tensor
    """Target end-effector release pose.

    Accepts ``(4, 4)``, ``(n_envs, 4, 4)``, or
    ``(n_envs, n_waypoint, 4, 4)``.
    """

    tcp_symmetry: TcpSymmetry = "none"
    """Optional TCP-frame symmetry allowed by the placement semantics.

    ``"none"`` preserves the pose exactly. ``"z_roll_180"`` lets placement
    choose between the pose and its TCP z-roll 180 equivalent, which flips TCP
    x/y while preserving TCP z and translation.
    """

    def __post_init__(self) -> None:
        _validate_pose_tensor(self.xpos, "xpos", allow_waypoints=True)
        if self.tcp_symmetry not in ("none", "z_roll_180"):
            raise ValueError(
                "tcp_symmetry must be one of 'none' or 'z_roll_180', "
                f"but got {self.tcp_symmetry!r}"
            )


@dataclass(frozen=True, slots=True, eq=False)
class AssembleTarget(ActionTarget):
    """Place a held assemble object onto a base object at a relative pose.

    The base object pose is read at planning time from
    :attr:`AssembleAffordance.base_object_entity`, and the assemble object's
    target pose is ``base_pose @ assemble_to_base_pose``. The held-object
    transform (``object_to_eef``) is read from :attr:`WorldState.held_objects`
    for the place control part, which a prior :class:`PickUp` populates.
    """

    affordance: AssembleAffordance
    """Assembly affordance anchoring the assemble object to the base object."""


@configclass
class PlaceCfg(ActionCfg):
    name: str = "place"
    """Name of the action, used for identification and logging."""

    sample_interval: int = 80
    """Number of waypoints for the full trajectory (down + hand + back)."""

    hand_interp_steps: int = 5
    """Number of waypoints for the gripper open interpolation phase."""

    post_hold_steps: int = 0
    """Number of stationary open-gripper waypoints before retracting."""

    hand_control_part: str = "hand"
    """Name of the robot part that controls the hand joints."""

    hand_open_qpos: torch.Tensor | None = None
    """Joint positions for the open hand state, shape ``[hand_dof,]``."""

    hand_close_qpos: torch.Tensor | None = None
    """Joint positions for the closed hand state, shape ``[hand_dof,]``."""

    lift_height: float = 0.1
    """Height (m) to retract the end-effector after opening the gripper."""

    max_approach_retract_z: float | None = None
    """Optional maximum world-frame TCP z for approach and retract poses (m)."""

    cartesian_waypoint_count: int = 1
    """Number of fixed-orientation Cartesian keyframes per translation segment."""


class Place(AtomicAction[PlaceTarget | AssembleTarget]):
    """Lower the held object to a place pose, open the gripper, retract.

    The :class:`PlaceTarget` may carry either a single waypoint
    ``(n_envs, 4, 4)`` (or a broadcastable ``(4, 4)``) or a multi-waypoint
    trajectory ``(n_envs, n_waypoint, 4, 4)``. In the multi-waypoint case the
    down phase visits every waypoint in order; approaching from above the
    first waypoint, descending through each waypoint, then opening the gripper
    at the final waypoint and retracting to above the last waypoint. Starting
    joint positions are inherited from ``WorldState.last_qpos``.

    An :class:`AssembleTarget` replaces the explicit EEF pose with an assembly
    affordance: the place pose is derived from the base object's current pose
    and ``assemble_to_base_pose``, converted to an EEF pose through the held
    object's ``object_to_eef`` (read from ``WorldState.held_objects``).
    """

    TargetType: ClassVar[type | tuple[type, ...]] = (
        PlaceTarget,
        AssembleTarget,
    )

    def __init__(
        self,
        motion_generator,
        cfg: PlaceCfg | None = None,
    ) -> None:
        super().__init__(motion_generator, cfg or PlaceCfg())
        self.builder = TrajectoryBuilder(motion_generator)
        self.n_envs = self.robot.get_qpos().shape[0]
        self.arm_joint_ids = self.robot.get_joint_ids(name=self.cfg.control_part)
        self.hand_joint_ids = self.robot.get_joint_ids(name=self.cfg.hand_control_part)
        self.arm_dof = len(self.arm_joint_ids)
        self.robot_dof = self.robot.dof

        if self.cfg.hand_open_qpos is None:
            logger.log_error("hand_open_qpos must be specified in PlaceCfg", ValueError)
        if self.cfg.hand_close_qpos is None:
            logger.log_error(
                "hand_close_qpos must be specified in PlaceCfg", ValueError
            )
        self.hand_open_qpos = self.cfg.hand_open_qpos.to(self.device)
        self.hand_close_qpos = self.cfg.hand_close_qpos.to(self.device)
        if self.cfg.cartesian_waypoint_count < 1:
            logger.log_error("cartesian_waypoint_count must be at least 1.", ValueError)
        if self.cfg.post_hold_steps < 0:
            logger.log_error("post_hold_steps must be non-negative.", ValueError)

    def execute(
        self, target: PlaceTarget | AssembleTarget, state: WorldState
    ) -> ActionResult:
        place_xpos = self._resolve_place_xpos(target, state)
        if place_xpos.dim() == 3:
            place_xpos = place_xpos.unsqueeze(1)

        start_arm_qpos = self.builder.resolve_start_qpos(
            arm_qpos_from_state(state, self.arm_joint_ids),
            n_envs=self.n_envs,
            arm_dof=self.arm_dof,
            control_part=self.cfg.control_part,
        )
        if isinstance(target, PlaceTarget) and target.tcp_symmetry == "z_roll_180":
            place_xpos = self._select_tcp_symmetric_place_variant(
                place_xpos, start_arm_qpos
            )
        n_down, n_open, n_back = self.builder.split_three_phase(
            self.cfg.sample_interval,
            self.cfg.hand_interp_steps,
            first_phase_name="approach",
            third_phase_name="back",
        )

        approach_xpos = self._lifted_pose(place_xpos[:, 0])
        retract_xpos = self._lifted_pose(place_xpos[:, -1])

        start_xpos = self.robot.compute_fk(
            qpos=start_arm_qpos,
            name=self.cfg.control_part,
            to_matrix=True,
        )
        down_xpos = torch.cat([approach_xpos.unsqueeze(1), place_xpos], dim=1)
        down_xpos = self._translation_keyframes(start_xpos, down_xpos)

        target_states_list = [
            [
                PlanState(xpos=down_xpos[i, j], move_type=MoveType.EEF_MOVE)
                for j in range(down_xpos.shape[1])
            ]
            for i in range(self.n_envs)
        ]
        down_success, down_arm = self.builder.plan_arm_traj(
            target_states_list,
            start_arm_qpos,
            n_down,
            control_part=self.cfg.control_part,
            arm_dof=self.arm_dof,
            cfg=self.cfg,
        )
        reach_arm_qpos = down_arm[:, -1, :]

        back_xpos = self._translation_keyframes(
            place_xpos[:, -1], retract_xpos.unsqueeze(1)
        )
        target_states_list = [
            [
                PlanState(xpos=back_xpos[i, j], move_type=MoveType.EEF_MOVE)
                for j in range(back_xpos.shape[1])
            ]
            for i in range(self.n_envs)
        ]
        back_success, back_arm = self.builder.plan_arm_traj(
            target_states_list,
            reach_arm_qpos,
            n_back,
            control_part=self.cfg.control_part,
            arm_dof=self.arm_dof,
            cfg=self.cfg,
        )
        success = down_success & back_success

        hand_open_path = self.builder.interpolate_hand_qpos(
            self.hand_close_qpos, self.hand_open_qpos, n_waypoints=n_open
        )

        # Allocate from the actually-returned phase lengths so collision-aware
        # planners can preserve their own sample count.
        n_down_actual = down_arm.shape[1]
        n_back_actual = back_arm.shape[1]
        n_hold = int(self.cfg.post_hold_steps)
        full = torch.empty(
            (
                self.n_envs,
                n_down_actual + n_open + n_hold + n_back_actual,
                self.robot_dof,
            ),
            dtype=torch.float32,
            device=self.device,
        )
        full[:, :, :] = state.last_qpos.unsqueeze(1)
        full[:, :n_down_actual, self.arm_joint_ids] = down_arm
        full[:, :n_down_actual, self.hand_joint_ids] = self.hand_close_qpos
        full[:, n_down_actual : n_down_actual + n_open, self.arm_joint_ids] = (
            reach_arm_qpos.unsqueeze(1)
        )
        full[:, n_down_actual : n_down_actual + n_open, self.hand_joint_ids] = (
            hand_open_path
        )
        hold_end = n_down_actual + n_open + n_hold
        if n_hold:
            full[:, n_down_actual + n_open : hold_end, self.arm_joint_ids] = (
                reach_arm_qpos.unsqueeze(1)
            )
            full[:, n_down_actual + n_open : hold_end, self.hand_joint_ids] = (
                self.hand_open_qpos
            )
        full[:, hold_end:, self.arm_joint_ids] = back_arm
        full[:, hold_end:, self.hand_joint_ids] = self.hand_open_qpos

        held_objects = dict(state.held_objects)
        held_objects.pop(self.cfg.control_part, None)
        coordinated_held_objects = {
            key: value
            for key, value in state.coordinated_held_objects.items()
            if self.cfg.control_part not in key
        }
        return ActionResult(
            success=success,
            trajectory=full,
            next_state=state.with_updates(
                last_qpos=full[:, -1, :].clone(),
                held_objects=held_objects,
                coordinated_held_objects=coordinated_held_objects,
            ),
        )

    def _resolve_place_xpos(
        self, target: PlaceTarget | AssembleTarget, state: WorldState
    ) -> torch.Tensor:
        """Resolve the place EEF poses from a typed target.

        Args:
            target: Either an explicit EEF pose target or an assembly target.
            state: World state carrying the held-object transform.

        Returns:
            Place EEF poses with shape ``(n_envs, 4, 4)`` or
            ``(n_envs, n_waypoint, 4, 4)``.
        """
        if isinstance(target, PlaceTarget):
            return self.builder.resolve_pose_target(target.xpos, n_envs=self.n_envs)
        return self._resolve_assemble_place_xpos(target, state)

    def _resolve_assemble_place_xpos(
        self, target: AssembleTarget, state: WorldState
    ) -> torch.Tensor:
        """Derive the place EEF pose from an assembly affordance.

        The assemble object target pose is ``base_pose @ assemble_to_base_pose``;
        the EEF pose is that target posed through the held object's
        ``object_to_eef``.

        Args:
            target: Assembly target carrying the base/assemble affordance.
            state: World state carrying the held-object transform.

        Returns:
            Place EEF poses with shape ``(n_envs, 4, 4)``.

        Raises:
            ValueError: If no held object or no base object entity is available.
        """
        held = state.get_held_object(self.cfg.control_part)
        if held is None:
            logger.log_error(
                "Place with AssembleTarget requires an object held by control "
                f"part {self.cfg.control_part!r} (run PickUp first).",
                ValueError,
            )
        affordance = target.affordance
        if affordance.base_object_entity is None:
            logger.log_error(
                "AssembleAffordance.base_object_entity must be set to assemble "
                "onto a base object.",
                ValueError,
            )
        base_pose = affordance.base_object_entity.get_local_pose(to_matrix=True).to(
            device=self.device, dtype=torch.float32
        )
        assemble_object_pose = affordance.get_assemble_object_pose(base_pose)
        object_to_eef = resolve_object_target(
            held.object_to_eef,
            n_envs=self.n_envs,
            device=self.device,
            name="object_to_eef",
        )
        return torch.bmm(assemble_object_pose, object_to_eef)

    def _lifted_pose(self, release_xpos: torch.Tensor) -> torch.Tensor:
        """Build an above-release pose while respecting the optional TCP z cap."""
        lifted_xpos = release_xpos.clone()
        lifted_z = release_xpos[:, 2, 3] + self.cfg.lift_height
        if self.cfg.max_approach_retract_z is not None:
            max_z = torch.as_tensor(
                self.cfg.max_approach_retract_z,
                dtype=release_xpos.dtype,
                device=release_xpos.device,
            )
            lifted_z = torch.maximum(
                release_xpos[:, 2, 3],
                torch.clamp_max(lifted_z, max_z),
            )
        lifted_xpos[:, 2, 3] = lifted_z
        return lifted_xpos

    def _translation_keyframes(
        self, start_xpos: torch.Tensor, target_xpos: torch.Tensor
    ) -> torch.Tensor:
        """Interpolate translations while holding each segment's target rotation."""
        count = self.cfg.cartesian_waypoint_count
        if count == 1:
            return target_xpos

        segment_starts = torch.cat(
            [start_xpos.unsqueeze(1), target_xpos[:, :-1]], dim=1
        )
        alpha = torch.linspace(
            1.0 / count,
            1.0,
            count,
            dtype=target_xpos.dtype,
            device=self.device,
        )
        keyframes = target_xpos.unsqueeze(2).repeat(1, 1, count, 1, 1)
        start_position = segment_starts[..., :3, 3].unsqueeze(2)
        target_position = target_xpos[..., :3, 3].unsqueeze(2)
        keyframes[..., :3, 3] = start_position + alpha[None, None, :, None] * (
            target_position - start_position
        )
        return keyframes.flatten(1, 2)

    def _fail(self, state: WorldState) -> ActionResult:
        return ActionResult(
            success=torch.zeros(self.n_envs, dtype=torch.bool, device=self.device),
            trajectory=torch.empty(
                (self.n_envs, 0, self.robot_dof),
                dtype=torch.float32,
                device=self.device,
            ),
            next_state=state,
        )

    def _select_tcp_symmetric_place_variant(
        self, place_xpos: torch.Tensor, start_qpos: torch.Tensor
    ) -> torch.Tensor:
        """Choose the closest TCP z-roll variant for an opt-in place target."""
        mirrored_place_xpos = place_xpos.clone()
        mirrored_place_xpos[..., :3, 0] = -mirrored_place_xpos[..., :3, 0]
        mirrored_place_xpos[..., :3, 1] = -mirrored_place_xpos[..., :3, 1]
        place_variants = torch.stack([place_xpos, mirrored_place_xpos], dim=2)

        start_xpos = self.robot.compute_fk(
            qpos=start_qpos,
            name=self.cfg.control_part,
            to_matrix=True,
        )
        start_quat = quat_from_matrix(start_xpos[:, :3, :3])
        first_waypoint_quat = quat_from_matrix(place_variants[:, 0, :, :3, :3])
        start_quat = start_quat[:, None, :].expand_as(first_waypoint_quat)
        rotation_error = quat_error_magnitude(
            first_waypoint_quat.reshape(-1, 4),
            start_quat.reshape(-1, 4),
        ).reshape(self.n_envs, 2)
        best_variant_idx = rotation_error.argmin(dim=1)

        env_idx = torch.arange(self.n_envs, device=self.device)[:, None]
        waypoint_idx = torch.arange(place_xpos.shape[1], device=self.device)[None, :]
        return place_variants[
            env_idx,
            waypoint_idx,
            best_variant_idx[:, None],
        ]


__all__ = ["Place", "PlaceCfg", "PlaceTarget", "AssembleTarget"]
