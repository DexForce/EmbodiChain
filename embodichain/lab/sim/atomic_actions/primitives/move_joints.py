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

"""MoveJoints atomic action implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch

from embodichain.utils import configclass, logger

from ..core import (
    ActionTarget,
    ActionCfg,
    ActionResult,
    AtomicAction,
    WorldState,
)
from ..trajectory import TrajectoryBuilder


@dataclass(frozen=True, slots=True, eq=False)
class JointPositionTarget(ActionTarget):
    """Joint-space target for a configured robot control part."""

    qpos: torch.Tensor
    """Target joint positions.

    Accepts:

    - ``(control_dof,)`` or ``(n_envs, control_dof)`` — a single waypoint.
    - ``(n_envs, n_waypoint, control_dof)`` — a multi-waypoint trajectory;
      waypoints are visited in order.
    """

    def __post_init__(self) -> None:
        if not isinstance(self.qpos, torch.Tensor):
            raise TypeError(
                f"qpos must be a torch.Tensor, got {type(self.qpos).__name__}."
            )
        if self.qpos.dim() not in (1, 2, 3) or self.qpos.shape[-1] == 0:
            raise ValueError(
                "qpos must have shape (control_dof,), (n_envs, control_dof), "
                "or (n_envs, n_waypoint, control_dof), "
                f"got {tuple(self.qpos.shape)}."
            )


@dataclass(frozen=True, slots=True, eq=False)
class NamedJointPositionTarget(ActionTarget):
    """Named joint-space target resolved from :class:`MoveJointsCfg`."""

    name: str
    """Name of a joint-position target in ``MoveJointsCfg.named_joint_positions``."""

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise TypeError(f"name must be a str, got {type(self.name).__name__}.")
        if not self.name.strip():
            raise ValueError("name must not be empty.")


@configclass
class MoveJointsCfg(ActionCfg):
    name: str = "move_joints"
    """Name of the action, used for identification and logging."""

    sample_interval: int = 50
    """Number of waypoints in the interpolated joint-space trajectory."""

    named_joint_positions: dict[str, torch.Tensor] | None = None
    """Optional named joint targets resolved by ``NamedJointPositionTarget``."""


class MoveJoints(AtomicAction[JointPositionTarget | NamedJointPositionTarget]):
    """Plan a joint-space move for the configured control part.

    The :class:`JointPositionTarget` may carry either a single waypoint
    ``(n_envs, control_dof)`` or a multi-waypoint trajectory
    ``(n_envs, n_waypoint, control_dof)``. In the multi-waypoint case the
    action plans a single trajectory that visits every waypoint in order,
    starting from the inherited ``WorldState.last_qpos``.
    """

    TargetType: ClassVar[tuple[type, ...]] = (
        JointPositionTarget,
        NamedJointPositionTarget,
    )

    def __init__(
        self,
        motion_generator,
        cfg: MoveJointsCfg | None = None,
    ) -> None:
        super().__init__(motion_generator, cfg or MoveJointsCfg())
        self.builder = TrajectoryBuilder(motion_generator)
        self.n_envs = self.robot.get_qpos().shape[0]
        self.joint_ids = self.robot.get_joint_ids(name=self.cfg.control_part)
        self.joint_dof = len(self.joint_ids)
        self.robot_dof = self.robot.dof
        self.named_joint_positions = self.cfg.named_joint_positions or {}

    def execute(
        self,
        target: JointPositionTarget | NamedJointPositionTarget,
        state: WorldState,
    ) -> ActionResult:
        target_qpos = self.builder.resolve_joint_target(
            self._resolve_target_qpos(target),
            n_envs=self.n_envs,
            joint_dof=self.joint_dof,
            control_part=self.cfg.control_part,
        )
        start_qpos = self.builder.resolve_start_qpos(
            state.last_qpos[:, self.joint_ids],
            n_envs=self.n_envs,
            arm_dof=self.joint_dof,
            control_part=self.cfg.control_part,
        )
        success, joint_traj = self.builder.plan_joint_motion(
            start_qpos,
            target_qpos,
            self.cfg.sample_interval,
            control_part=self.cfg.control_part,
            arm_dof=self.joint_dof,
            cfg=self.cfg,
        )
        full = self._embed(joint_traj, state.last_qpos)
        return ActionResult(
            success=success,
            trajectory=full,
            next_state=state.with_updates(
                last_qpos=full[:, -1, :].clone(),
            ),
        )

    def _resolve_target_qpos(
        self, target: JointPositionTarget | NamedJointPositionTarget
    ) -> torch.Tensor:
        if isinstance(target, JointPositionTarget):
            return target.qpos
        if target.name not in self.named_joint_positions:
            logger.log_error(
                f"Unknown named joint-position target '{target.name}' for "
                f"MoveJoints. Available targets: {sorted(self.named_joint_positions)}",
                KeyError,
            )
        return self.named_joint_positions[target.name]

    def _embed(
        self, joint_traj: torch.Tensor, last_full_qpos: torch.Tensor
    ) -> torch.Tensor:
        n_wp = joint_traj.shape[1]
        full = torch.empty(
            (self.n_envs, n_wp, self.robot_dof),
            dtype=torch.float32,
            device=self.device,
        )
        full[:, :, :] = last_full_qpos.unsqueeze(1)
        full[:, :, self.joint_ids] = joint_traj
        return full


__all__ = [
    "JointPositionTarget",
    "MoveJoints",
    "MoveJointsCfg",
    "NamedJointPositionTarget",
]
