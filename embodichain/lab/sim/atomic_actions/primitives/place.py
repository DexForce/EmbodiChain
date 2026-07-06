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

from typing import ClassVar

import torch

from embodichain.lab.sim.planners import MoveType, PlanState
from embodichain.utils import configclass, logger
from embodichain.utils.math import quat_error_magnitude, quat_from_matrix

from ._helpers import arm_qpos_from_state
from ..core import (
    ActionCfg,
    ActionResult,
    AtomicAction,
    EndEffectorPoseTarget,
    WorldState,
)
from ..trajectory import TrajectoryBuilder


@configclass
class PlaceCfg(ActionCfg):
    name: str = "place"
    """Name of the action, used for identification and logging."""

    sample_interval: int = 80
    """Number of waypoints for the full trajectory (down + hand + back)."""

    hand_interp_steps: int = 5
    """Number of waypoints for the gripper open interpolation phase."""

    hand_control_part: str = "hand"
    """Name of the robot part that controls the hand joints."""

    hand_open_qpos: torch.Tensor | None = None
    """Joint positions for the open hand state, shape ``[hand_dof,]``."""

    hand_close_qpos: torch.Tensor | None = None
    """Joint positions for the closed hand state, shape ``[hand_dof,]``."""

    lift_height: float = 0.1
    """Height (m) to retract the end-effector after opening the gripper."""


class Place(AtomicAction):
    """Lower the held object to a place pose, open the gripper, retract.

    The :class:`EndEffectorPoseTarget` may carry either a single waypoint
    ``(n_envs, 4, 4)`` (or a broadcastable ``(4, 4)``) or a multi-waypoint
    trajectory ``(n_envs, n_waypoint, 4, 4)``. In the multi-waypoint case the
    down phase visits every waypoint in order; approaching from above the
    first waypoint, descending through each waypoint, then opening the gripper
    at the final waypoint and retracting to above the last waypoint. Starting
    joint positions are inherited from ``WorldState.last_qpos``.
    """

    TargetType: ClassVar[type] = EndEffectorPoseTarget

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

    def execute(self, target: EndEffectorPoseTarget, state: WorldState) -> ActionResult:
        place_xpos = self.builder.resolve_pose_target(target.xpos, n_envs=self.n_envs)
        if place_xpos.dim() == 3:
            place_xpos = place_xpos.unsqueeze(1)

        start_arm_qpos = self.builder.resolve_start_qpos(
            arm_qpos_from_state(state, self.arm_joint_ids),
            n_envs=self.n_envs,
            arm_dof=self.arm_dof,
            control_part=self.cfg.control_part,
        )
        if target.tcp_symmetry == "z_roll_180":
            place_xpos = self._select_tcp_symmetric_place_variant(
                place_xpos, start_arm_qpos
            )
        n_waypoint = place_xpos.shape[1]
        n_down, n_open, n_back = self.builder.split_three_phase(
            self.cfg.sample_interval,
            self.cfg.hand_interp_steps,
            first_phase_name="approach",
            third_phase_name="back",
        )

        lift_offset = torch.tensor([0, 0, 1], device=self.device) * self.cfg.lift_height
        approach_xpos = self.builder.apply_local_offset(place_xpos[:, 0], lift_offset)
        retract_xpos = self.builder.apply_local_offset(place_xpos[:, -1], lift_offset)

        target_states_list = [
            [PlanState(xpos=approach_xpos[i], move_type=MoveType.EEF_MOVE)]
            + [
                PlanState(xpos=place_xpos[i, j], move_type=MoveType.EEF_MOVE)
                for j in range(n_waypoint)
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

        target_states_list = [
            [PlanState(xpos=retract_xpos[i], move_type=MoveType.EEF_MOVE)]
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

        full = torch.empty(
            (self.n_envs, n_down + n_open + n_back, self.robot_dof),
            dtype=torch.float32,
            device=self.device,
        )
        full[:, :, :] = state.last_qpos.unsqueeze(1)
        full[:, :n_down, self.arm_joint_ids] = down_arm
        full[:, :n_down, self.hand_joint_ids] = self.hand_close_qpos
        full[:, n_down : n_down + n_open, self.arm_joint_ids] = (
            reach_arm_qpos.unsqueeze(1)
        )
        full[:, n_down : n_down + n_open, self.hand_joint_ids] = hand_open_path
        full[:, n_down + n_open :, self.arm_joint_ids] = back_arm
        full[:, n_down + n_open :, self.hand_joint_ids] = self.hand_open_qpos

        return ActionResult(
            success=success,
            trajectory=full,
            next_state=WorldState(
                last_qpos=full[:, -1, :].clone(),
                held_object=None,
                coordinated_held_object=state.coordinated_held_object,
            ),
        )

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


__all__ = ["Place", "PlaceCfg"]
