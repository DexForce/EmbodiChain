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

"""CoordinatedPlacement atomic action implementation."""

from __future__ import annotations

from typing import ClassVar

import torch

from embodichain.lab.sim.planners import MoveType, PlanState
from embodichain.utils import configclass, logger

from ._helpers import resolve_object_target
from ..core import (
    ActionCfg,
    ActionResult,
    AtomicAction,
    CoordinatedPlacementTarget,
    HeldObjectState,
    WorldState,
)
from ..trajectory import TrajectoryBuilder


@configclass
class CoordinatedPlacementCfg(ActionCfg):
    name: str = "coordinated_placement"
    """Name of the action, used for identification and logging."""

    control_part: str = "dual_arm"
    """Robot control part containing both placing and support arms."""

    placing_arm_control_part: str = "left_arm"
    """Arm that places and releases its held object."""

    support_arm_control_part: str = "right_arm"
    """Arm that moves the support object and keeps holding it."""

    placing_hand_control_part: str = "left_hand"
    """Hand attached to the placing arm."""

    support_hand_control_part: str = "right_hand"
    """Hand attached to the support arm."""

    placing_hand_open_qpos: torch.Tensor | None = None
    """Placing-hand qpos for the open state, shape ``[hand_dof,]``."""

    placing_hand_close_qpos: torch.Tensor | None = None
    """Placing-hand qpos for the closed state, shape ``[hand_dof,]``."""

    support_hand_close_qpos: torch.Tensor | None = None
    """Support-hand qpos for the closed state, shape ``[hand_dof,]``."""

    release: bool = True
    """Whether to open the placing hand at the aligned placement pose."""

    placing_height_offset: float = 0.0
    """Default World-Z offset above the placing object target pose."""

    support_height_offset: float = 0.0
    """Default World-Z offset above the support object target pose."""

    lift_height: float = 0.08
    """World-Z lift distance for the placing arm after release."""

    sample_interval: int = 100
    """Number of waypoints for the full coordinated placement trajectory."""

    hand_interp_steps: int = 10
    """Number of waypoints for the placing-hand release interpolation."""

    hold_steps: int = 4
    """Number of waypoints to hold alignment before releasing."""

    retreat_steps: int = 16
    """Number of waypoints used for the placing-arm lift retreat."""


class CoordinatedPlacement(AtomicAction):
    """Coordinate two held objects: support object below, placing object above."""

    TargetType: ClassVar[type] = CoordinatedPlacementTarget

    def __init__(
        self,
        motion_generator,
        cfg: CoordinatedPlacementCfg | None = None,
    ) -> None:
        super().__init__(motion_generator, cfg or CoordinatedPlacementCfg())
        self.builder = TrajectoryBuilder(motion_generator)
        self.n_envs = self.robot.get_qpos().shape[0]
        self.robot_dof = self.robot.dof

        self.dual_arm_joint_ids = self.robot.get_joint_ids(name=self.cfg.control_part)
        self.placing_arm_joint_ids = self.robot.get_joint_ids(
            name=self.cfg.placing_arm_control_part
        )
        self.support_arm_joint_ids = self.robot.get_joint_ids(
            name=self.cfg.support_arm_control_part
        )
        self.placing_hand_joint_ids = self.robot.get_joint_ids(
            name=self.cfg.placing_hand_control_part
        )
        self.support_hand_joint_ids = self.robot.get_joint_ids(
            name=self.cfg.support_hand_control_part
        )
        self.joint_ids = (
            self.dual_arm_joint_ids
            + self.placing_hand_joint_ids
            + self.support_hand_joint_ids
        )
        self.placing_arm_dof = len(self.placing_arm_joint_ids)
        self.support_arm_dof = len(self.support_arm_joint_ids)
        self.placing_hand_dof = len(self.placing_hand_joint_ids)
        self.support_hand_dof = len(self.support_hand_joint_ids)

        self._validate_hand_qpos_cfg()
        self.placing_hand_open_qpos = self.builder.expand_hand_qpos(
            self.cfg.placing_hand_open_qpos,
            n_envs=self.n_envs,
            hand_dof=self.placing_hand_dof,
        )
        self.placing_hand_close_qpos = self.builder.expand_hand_qpos(
            self.cfg.placing_hand_close_qpos,
            n_envs=self.n_envs,
            hand_dof=self.placing_hand_dof,
        )
        self.support_hand_close_qpos = self.builder.expand_hand_qpos(
            self.cfg.support_hand_close_qpos,
            n_envs=self.n_envs,
            hand_dof=self.support_hand_dof,
        )

    def execute(
        self, target: CoordinatedPlacementTarget, state: WorldState
    ) -> ActionResult:
        placing_xpos, support_xpos, release, support_held_object = self._resolve_target(
            target
        )
        placing_start_qpos, support_start_qpos = self._resolve_start_qpos(state)
        segments = self._compute_segment_lengths(release)

        placing_lift_xpos = self.builder.apply_local_offset(
            placing_xpos,
            torch.tensor(
                [0.0, 0.0, self.cfg.lift_height],
                dtype=torch.float32,
                device=self.device,
            ),
        )

        ok, placing_approach_traj = self._plan_named_arm_trajectory(
            self.cfg.placing_arm_control_part,
            placing_start_qpos,
            torch.stack([placing_lift_xpos, placing_xpos], dim=1),
            segments["approach"],
        )
        if not ok:
            logger.log_warning("CoordinatedPlacement failed to plan placing approach.")
            return self._fail(state)

        ok, support_approach_traj = self._plan_named_arm_trajectory(
            self.cfg.support_arm_control_part,
            support_start_qpos,
            support_xpos.unsqueeze(1),
            segments["approach"],
        )
        if not ok:
            logger.log_warning("CoordinatedPlacement failed to plan support approach.")
            return self._fail(state)

        placing_place_qpos = placing_approach_traj[:, -1]
        support_place_qpos = support_approach_traj[:, -1]
        approach_trajectory = self._assemble_phase(
            state.last_qpos,
            placing_approach_traj,
            support_approach_traj,
            self._repeat_qpos(self.placing_hand_close_qpos, segments["approach"]),
            self._repeat_qpos(self.support_hand_close_qpos, segments["approach"]),
        )

        hold_trajectory = self._empty_phase()
        if segments["hold"] > 0:
            hold_trajectory = self._assemble_phase(
                state.last_qpos,
                self._repeat_qpos(placing_place_qpos, segments["hold"]),
                self._repeat_qpos(support_place_qpos, segments["hold"]),
                self._repeat_qpos(self.placing_hand_close_qpos, segments["hold"]),
                self._repeat_qpos(self.support_hand_close_qpos, segments["hold"]),
            )

        release_trajectory = self._empty_phase()
        if release:
            release_trajectory = self._assemble_phase(
                state.last_qpos,
                self._repeat_qpos(placing_place_qpos, segments["release"]),
                self._repeat_qpos(support_place_qpos, segments["release"]),
                self.builder.interpolate_hand_qpos(
                    self.placing_hand_close_qpos,
                    self.placing_hand_open_qpos,
                    n_waypoints=segments["release"],
                ),
                self._repeat_qpos(self.support_hand_close_qpos, segments["release"]),
            )

        ok, placing_retreat_traj = self._plan_named_arm_trajectory(
            self.cfg.placing_arm_control_part,
            placing_place_qpos,
            placing_lift_xpos.unsqueeze(1),
            segments["retreat"],
        )
        if not ok:
            logger.log_warning("CoordinatedPlacement failed to plan placing retreat.")
            return self._fail(state)

        placing_hand_retreat_qpos = (
            self.placing_hand_open_qpos if release else self.placing_hand_close_qpos
        )
        retreat_trajectory = self._assemble_phase(
            state.last_qpos,
            placing_retreat_traj,
            self._repeat_qpos(support_place_qpos, segments["retreat"]),
            self._repeat_qpos(placing_hand_retreat_qpos, segments["retreat"]),
            self._repeat_qpos(self.support_hand_close_qpos, segments["retreat"]),
        )

        full = torch.cat(
            [
                approach_trajectory,
                hold_trajectory,
                release_trajectory,
                retreat_trajectory,
            ],
            dim=1,
        )
        return ActionResult(
            success=True,
            trajectory=full,
            next_state=WorldState(
                last_qpos=full[:, -1, :].clone(),
                held_object=support_held_object,
            ),
        )

    def _validate_hand_qpos_cfg(self) -> None:
        required_names = (
            "placing_hand_open_qpos",
            "placing_hand_close_qpos",
            "support_hand_close_qpos",
        )
        for name in required_names:
            if getattr(self.cfg, name) is None:
                logger.log_error(
                    f"{name} must be specified in CoordinatedPlacementCfg",
                    ValueError,
                )

    def _resolve_object_pose(
        self,
        pose: torch.Tensor,
        height_offset: float,
        name: str,
    ) -> torch.Tensor:
        object_pose = resolve_object_target(
            pose,
            n_envs=self.n_envs,
            device=self.device,
            name=name,
        )
        return self.builder.apply_local_offset(
            object_pose,
            torch.tensor(
                [0.0, 0.0, height_offset],
                dtype=torch.float32,
                device=self.device,
            ),
        )

    def _resolve_object_to_eef(
        self,
        held_state: HeldObjectState,
        name: str,
    ) -> torch.Tensor:
        return self._resolve_held_matrix(
            held_state.object_to_eef,
            f"{name}.object_to_eef",
        )

    def _resolve_held_matrix(self, matrix: torch.Tensor, name: str) -> torch.Tensor:
        matrix = matrix.to(device=self.device, dtype=torch.float32)
        if matrix.shape == (4, 4):
            matrix = matrix.unsqueeze(0).repeat(self.n_envs, 1, 1)
        if matrix.shape != (self.n_envs, 4, 4):
            logger.log_error(
                f"{name} must have shape (4, 4) or ({self.n_envs}, 4, 4), "
                f"but got {matrix.shape}",
                ValueError,
            )
        return matrix

    def _resolve_held_state(
        self,
        held_state: HeldObjectState,
        name: str,
        object_to_eef: torch.Tensor,
    ) -> HeldObjectState:
        return HeldObjectState(
            semantics=held_state.semantics,
            object_to_eef=object_to_eef,
            grasp_xpos=self._resolve_held_matrix(
                held_state.grasp_xpos,
                f"{name}.grasp_xpos",
            ),
        )

    def _resolve_target(
        self,
        target: CoordinatedPlacementTarget,
    ) -> tuple[torch.Tensor, torch.Tensor, bool, HeldObjectState]:
        placing_height_offset = (
            self.cfg.placing_height_offset
            if target.placing_height_offset is None
            else target.placing_height_offset
        )
        support_height_offset = (
            self.cfg.support_height_offset
            if target.support_height_offset is None
            else target.support_height_offset
        )
        placing_object_pose = self._resolve_object_pose(
            target.placing_object_target_pose,
            placing_height_offset,
            "placing_object_target_pose",
        )
        support_object_pose = self._resolve_object_pose(
            target.support_object_target_pose,
            support_height_offset,
            "support_object_target_pose",
        )
        placing_object_to_eef = self._resolve_object_to_eef(
            target.placing_held_object,
            "placing_held_object",
        )
        support_object_to_eef = self._resolve_object_to_eef(
            target.support_held_object,
            "support_held_object",
        )
        placing_xpos = torch.bmm(placing_object_pose, placing_object_to_eef)
        support_xpos = torch.bmm(support_object_pose, support_object_to_eef)
        release = self.cfg.release if target.release is None else target.release
        return (
            placing_xpos,
            support_xpos,
            release,
            self._resolve_held_state(
                target.support_held_object,
                "support_held_object",
                support_object_to_eef,
            ),
        )

    def _resolve_start_qpos(
        self, state: WorldState
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if state.last_qpos.shape != (self.n_envs, self.robot_dof):
            logger.log_error(
                f"WorldState.last_qpos must have shape ({self.n_envs}, {self.robot_dof}), "
                f"but got {state.last_qpos.shape}",
                ValueError,
            )
        start_qpos = state.last_qpos.to(device=self.device, dtype=torch.float32)
        return (
            start_qpos[:, self.placing_arm_joint_ids],
            start_qpos[:, self.support_arm_joint_ids],
        )

    def _compute_segment_lengths(self, release: bool) -> dict[str, int]:
        n_release = max(2, self.cfg.hand_interp_steps) if release else 0
        n_hold = max(0, self.cfg.hold_steps)
        n_retreat = max(2, self.cfg.retreat_steps)
        n_approach = self.cfg.sample_interval - n_hold - n_release - n_retreat
        if n_approach < 2:
            logger.log_error(
                "Not enough waypoints for coordinated placement. Increase "
                "sample_interval or decrease hold/release/retreat steps.",
                ValueError,
            )
        return {
            "approach": n_approach,
            "hold": n_hold,
            "release": n_release,
            "retreat": n_retreat,
        }

    def _plan_named_arm_trajectory(
        self,
        control_part: str,
        start_qpos: torch.Tensor,
        target_poses: torch.Tensor,
        n_waypoints: int,
    ) -> tuple[bool, torch.Tensor]:
        target_states_list = [
            [
                PlanState(xpos=target_poses[i, j], move_type=MoveType.EEF_MOVE)
                for j in range(target_poses.shape[1])
            ]
            for i in range(self.n_envs)
        ]
        return self.builder.plan_arm_traj(
            target_states_list,
            start_qpos,
            n_waypoints,
            control_part=control_part,
            arm_dof=start_qpos.shape[-1],
        )

    @staticmethod
    def _repeat_qpos(qpos: torch.Tensor, n_waypoints: int) -> torch.Tensor:
        return qpos.unsqueeze(1).repeat(1, n_waypoints, 1)

    def _empty_phase(self) -> torch.Tensor:
        return torch.empty(
            (self.n_envs, 0, self.robot_dof),
            dtype=torch.float32,
            device=self.device,
        )

    def _assemble_phase(
        self,
        base_full_qpos: torch.Tensor,
        placing_arm_traj: torch.Tensor,
        support_arm_traj: torch.Tensor,
        placing_hand_traj: torch.Tensor,
        support_hand_traj: torch.Tensor,
    ) -> torch.Tensor:
        n_waypoints = placing_arm_traj.shape[1]
        full = base_full_qpos.to(device=self.device, dtype=torch.float32)
        full = full.unsqueeze(1).repeat(1, n_waypoints, 1).clone()
        full[:, :, self.placing_arm_joint_ids] = placing_arm_traj
        full[:, :, self.support_arm_joint_ids] = support_arm_traj
        full[:, :, self.placing_hand_joint_ids] = placing_hand_traj
        full[:, :, self.support_hand_joint_ids] = support_hand_traj
        return full

    def _fail(self, state: WorldState) -> ActionResult:
        return ActionResult(
            success=False,
            trajectory=torch.empty(
                (self.n_envs, 0, self.robot_dof),
                dtype=torch.float32,
                device=self.device,
            ),
            next_state=state,
        )


__all__ = ["CoordinatedPlacement", "CoordinatedPlacementCfg"]
