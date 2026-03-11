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
    LEFT = 0
    RIGHT = 1
    BOTH = 2
    TORSO = 3
    ALL = 4


class MoveType(Enum):
    TOOL = 0
    TCP_MOVE = 1
    JOINT_MOVE = 2
    SYNC = 3
    PAUSE = 4


@dataclass
class PlanState:
    move_type: MoveType = MoveType.PAUSE
    move_part: MovePart = MovePart.LEFT
    xpos: torch.Tensor = None
    qpos: torch.Tensor = None
    qacc: torch.Tensor = None
    qvel: torch.Tensor = None
    is_open: bool = True
    is_world_coordinate: bool = True
    pause_seconds: float = 0.0
