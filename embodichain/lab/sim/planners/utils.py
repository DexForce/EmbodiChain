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

import torch

from dataclasses import dataclass
from enum import Enum
from typing import Union, __all__

from embodichain.utils import logger


__all__ = ["TrajectorySampleMethod", "MovePart", "MoveType", "PlanState", "PlanResult"]


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
class PlanResult:
    r"""Data class representing the result of a motion plan."""

    success: bool
    """Whether planning succeeded."""

    positions: torch.Tensor | None = None
    """Joint positions along trajectory with shape `(N, DOF)`."""

    velocities: torch.Tensor | None = None
    """Joint velocities along trajectory with shape `(N, DOF)`."""

    accelerations: torch.Tensor | None = None
    """Joint accelerations along trajectory with shape `(N, DOF)`."""

    times: torch.Tensor | None = None
    """Time stamps for each point with shape `(N,)`."""

    duration: float = 0.0
    """Total trajectory duration in seconds."""

    error_msg: str | None = None
    """Optional error message if planning failed."""


@dataclass
class PlanState:
    r"""Data class representing the state for a motion plan."""

    move_type: MoveType = MoveType.PAUSE
    """Type of movement used by the plan."""

    move_part: MovePart = MovePart.LEFT
    """Robot part that should move."""

    xpos: torch.Tensor = None
    """Target TCP pose (4x4 matrix) for `MoveType.TCP_MOVE`."""

    qpos: torch.Tensor = None
    """Target joint angles for `MoveType.JOINT_MOVE` with shape `(DOF,)`."""

    qvel: torch.Tensor = None
    """Target joint velocities for `MoveType.JOINT_MOVE` with shape `(DOF,)`."""

    qacc: torch.Tensor = None
    """Target joint accelerations for `MoveType.JOINT_MOVE` with shape `(DOF,)`."""

    is_open: bool = True
    """For `MoveType.TOOL`, indicates whether to open (`True`) or close (`False`) the tool."""

    is_world_coordinate: bool = True
    """`True` if the target pose is in world coordinates, `False` if relative to the current pose."""

    pause_seconds: float = 0.0
    """Duration of a pause when `move_type` is `MoveType.PAUSE`."""
