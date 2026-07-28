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

from typing import ClassVar

import torch

from embodichain.utils import configclass, logger
from embodichain.utils.math import matrix_from_quat, quat_from_matrix

from embodichain.lab.sim.planners import MoveType, PlanState
from ..core import (
    ActionCfg,
    ActionResult,
    AtomicAction,
    CoordinatedHeldObjectState,
    CoordinatedPickmentTarget,
    GraspTarget,
    ObjectSemantics,
    WorldState,
)
from ..trajectory import TrajectoryBuilder


"""HandOver atomic action implementation."""
"""First arm pick object in the bottom, then the second arm pick object in the top"""

@configclass
class HandOverCfg(ActionCfg):
    name: str = "coordinated_placement"
    """Name of the action, used for identification and logging."""

    control_part: str = "dual_arm"
    """Robot control part containing both placing and support arms."""

    left_arm_control_part: str = "left_arm"
    """Arm that places and releases its held object."""

    right_arm_control_part: str = "right_arm"
    """Arm that moves the support object and keeps holding it."""

    left_hand_control_part: str = "left_hand"
    """Hand attached to the placing arm."""

    right_hand_control_part: str = "right_hand"
    """Hand attached to the support arm."""

    left_hand_open_qpos: torch.Tensor | None = None
    """Left-hand qpos for the open state, shape ``[hand_dof,]``."""

    left_hand_close_qpos: torch.Tensor | None = None
    """Left-hand qpos for the closed state, shape ``[hand_dof,]``."""

    right_hand_open_qpos: torch.Tensor | None = None
    """Right-hand qpos for the open state, shape ``[hand_dof,]``."""

    right_hand_close_qpos: torch.Tensor | None = None
    """Right-hand qpos for the closed state, shape ``[hand_dof,]``."""

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


class HandOver(AtomicAction):
    TargetType: ClassVar[type] = GraspTarget
    
    def __init__(
        self,
        motion_generator,
        cfg: HandOverCfg | None = None,
    ):
        super().__init__(motion_generator, cfg or HandOverCfg())
        if (
            self.cfg.motion_source == "motion_gen"
            and self.motion_generator.planner.cfg.planner_type == "curobo"
        ):
            logger.log_error(
                "Coordinated dual-arm planning is not supported by the cuRobo "
                "backend. Use a single-arm action or a dedicated multi-arm "
                "planner.",
                ValueError,
            )

    def _init_dual_arm_parts(self):
        self.left_arm_part = self.cfg.left_arm_control_part
        self.right_arm_part = self.cfg.right_arm_control_part
        self.left_joint_ids = self.robot.get_joint_ids(self.left_arm_part)
        self.right_joint_ids = self.robot.get_joint_ids(self.right_arm_part)
        