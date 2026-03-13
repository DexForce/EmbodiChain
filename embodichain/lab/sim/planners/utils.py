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

from enum import Enum
from typing import Union
from embodichain.utils import logger
import torch
from enum import Enum
from dataclasses import dataclass


class TrajectorySampleMethod(Enum):
    r"""Enumeration for different trajectory sampling methods.

    This enum defines various methods for sampling trajectories,
    providing meaningful names for different sampling strategies.
    """

    TIME = "time"
    """Sample based on time intervals."""

    QUANTITY = "quantity"
    """Sample based on a specified number of points."""

    DISTANCE = "distance"
    """Sample based on distance intervals."""

    @classmethod
    def from_str(
        cls, value: Union[str, "TrajectorySampleMethod"]
    ) -> "TrajectorySampleMethod":
        if isinstance(value, cls):
            return value
        try:
            return cls[value.upper()]
        except KeyError:
            valid_values = [e.name for e in cls]
            logger.log_error(
                f"Invalid version '{value}'. Valid values are: {valid_values}",
                ValueError,
            )

    def __str__(self):
        """Override string representation for better readability."""
        return self.value.capitalize()


class MovePart(Enum):
    """Enumeration for different robot parts to move."""

    LEFT = 0  # left arm|eef
    RIGHT = 1  # right arm|eef
    BOTH = 2  # left arm|eef and right arm|eef
    TORSO = 3  # torso for humanoid robot
    ALL = 4  # all joints of the robot. Only for joint control.


class MoveType(Enum):
    """Enumeration for different types of movements."""

    TOOL = 0  # Tool open or close
    TCP_MOVE = 1  # Move the end-effector to a target pose (xpos) using IK and trajectory planning
    JOINT_MOVE = (
        2  # Directly move joints to target angles (qpos) using trajectory planning
    )
    SYNC = 3  # Synchronized left and right arm movement (for dual-arm robots)
    PAUSE = 4  # Pause for a specified duration (use pause_seconds in PlanState)


@dataclass
class PlanState:
    """Data class representing the state for a motion plan."""

    move_type: MoveType = MoveType.PAUSE  # Type of movement
    move_part: MovePart = MovePart.LEFT  # Part of the robot to move
    xpos: torch.Tensor = None  # target tcp pose (4x4 matrix) for TCP_MOVE
    qpos: torch.Tensor = None  # target joint angles for JOINT_MOVE (shape: (DOF,))
    qvel: torch.Tensor = None  # target joint velocities for JOINT_MOVE (shape: (DOF,))
    qacc: torch.Tensor = (
        None  # target joint accelerations for JOINT_MOVE (shape: (DOF,))
    )
    is_open: bool = True  # for TOOL move type, whether to open or close the tool
    is_world_coordinate: bool = (
        True  # whether the target pose is in world coordinates (True) or relative to current pose (False)
    )
    pause_seconds: float = 0.0  # duration to pause for PAUSE move type
