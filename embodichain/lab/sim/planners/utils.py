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
import numpy as np
from dataclasses import dataclass
from scipy.spatial.transform import Rotation, Slerp
from enum import Enum
from typing import Union, List

from embodichain.utils import logger


__all__ = [
    "TrajectorySampleMethod",
    "MovePart",
    "MoveType",
    "PlanState",
    "PlanResult",
    "calculate_point_allocations",
    "interpolate_xpos",
]


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
    r"""Enumeration for different robot parts to move.

    Defines robot part selection for motion planning.

    Attributes:
        LEFT (int): left arm or end-effector.
        RIGHT (int): right arm or end-effector.
        BOTH (int): both arms or end-effectors.
        TORSO (int): torso for humanoid robot.
        ALL (int): all joints of the robot (joint control only).
    """

    LEFT = 0  # left arm|eef
    RIGHT = 1  # right arm|eef
    BOTH = 2  # left arm|eef and right arm|eef
    TORSO = 3  # torso for humanoid robot
    ALL = 4  # all joints of the robot. Only for joint control.


class MoveType(Enum):
    r"""Enumeration for different types of movements.

    Defines movement types for robot planning.

    Attributes:
        TOOL (int): Tool open or close.
        EEF_MOVE (int): Move end-effector to target pose (IK + trajectory).
        JOINT_MOVE (int): Move joints to target angles (trajectory planning).
        SYNC (int): Synchronized left/right arm movement (dual-arm robots).
        PAUSE (int): Pause for specified duration (see PlanState.pause_seconds).
    """

    TOOL = 0  # Tool open or close
    EEF_MOVE = 1  # Move the end-effector to a target pose (xpos) using IK and trajectory planning
    JOINT_MOVE = (
        2  # Directly move joints to target angles (qpos) using trajectory planning
    )
    SYNC = 3  # Synchronized left and right arm movement (for dual-arm robots)
    PAUSE = 4  # Pause for a specified duration (use pause_seconds in PlanState)


@dataclass
class PlanResult:
    r"""Data class representing the result of a motion plan."""

    success: bool | torch.Tensor = False
    """Whether planning succeeded."""

    xpos_list: torch.Tensor | None = None
    """End-effector poses along trajectory with shape `(N, 4, 4)`."""

    positions: torch.Tensor | None = None
    """Joint positions along trajectory with shape `(N, DOF)`."""

    velocities: torch.Tensor | None = None
    """Joint velocities along trajectory with shape `(N, DOF)`."""

    accelerations: torch.Tensor | None = None
    """Joint accelerations along trajectory with shape `(N, DOF)`."""

    dt: torch.Tensor | None = None
    """Time duration between each point with shape `(N,)`."""

    duration: float | torch.Tensor = 0.0
    """Total trajectory duration in seconds."""


@dataclass
class PlanState:
    r"""Data class representing the state for a motion plan."""

    move_type: MoveType = MoveType.JOINT_MOVE
    """Type of movement used by the plan."""

    move_part: MovePart = MovePart.LEFT
    """Robot part that should move."""

    xpos: torch.Tensor | None = None
    """Target TCP pose (4x4 matrix) for `MoveType.EEF_MOVE`."""

    qpos: torch.Tensor | None = None
    """Target joint angles for `MoveType.JOINT_MOVE` with shape `(DOF,)`."""

    qvel: torch.Tensor | None = None
    """Target joint velocities for `MoveType.JOINT_MOVE` with shape `(DOF,)`."""

    qacc: torch.Tensor | None = None
    """Target joint accelerations for `MoveType.JOINT_MOVE` with shape `(DOF,)`."""

    is_open: bool = True
    """For `MoveType.TOOL`, indicates whether to open (`True`) or close (`False`) the tool."""

    is_world_coordinate: bool = True
    """`True` if the target pose is in world coordinates, `False` if relative to the current pose."""

    pause_seconds: float = 0.0
    """Duration of a pause when `move_type` is `MoveType.PAUSE`."""


def interpolate_xpos(
    current_xpos: np.ndarray, target_xpos: np.ndarray, num_samples: int
) -> np.ndarray:
    """Interpolate between two poses using vectorized Slerp + linear translation."""
    num_samples = max(2, int(num_samples))

    interp_ratios = np.linspace(0.0, 1.0, num_samples)
    slerp = Slerp(
        [0.0, 1.0],
        Rotation.from_matrix([current_xpos[:3, :3], target_xpos[:3, :3]]),
    )
    interp_rots = slerp(interp_ratios).as_matrix()
    interp_trans = (1.0 - interp_ratios[:, None]) * current_xpos[:3, 3] + interp_ratios[
        :, None
    ] * target_xpos[:3, 3]

    interp_poses = np.repeat(np.eye(4)[None, :, :], num_samples, axis=0)
    interp_poses[:, :3, :3] = interp_rots
    interp_poses[:, :3, 3] = interp_trans
    return interp_poses


def calculate_point_allocations(
    xpos_list: torch.Tensor | np.ndarray,
    step_size: float = 0.002,
    angle_step: float = np.pi / 90,
    device: torch.device = torch.device("cpu"),
) -> List[int]:
    """Calculate interpolation points for each segment with vectorized tensor ops."""
    if not isinstance(xpos_list, torch.Tensor):
        xpos_tensor = torch.as_tensor(
            np.asarray(xpos_list), dtype=torch.float32, device=device
        )
    else:
        xpos_tensor = xpos_list.to(dtype=torch.float32, device=device)

    if xpos_tensor.dim() != 3 or xpos_tensor.shape[0] < 2:
        return []

    start_poses = xpos_tensor[:-1]  # [N-1, 4, 4]
    end_poses = xpos_tensor[1:]  # [N-1, 4, 4]

    pos_dists = torch.norm(end_poses[:, :3, 3] - start_poses[:, :3, 3], dim=-1)
    pos_points = torch.clamp((pos_dists / step_size).int(), min=1)

    rel_rot = torch.matmul(
        start_poses[:, :3, :3].transpose(-1, -2), end_poses[:, :3, :3]
    )
    trace = rel_rot[:, 0, 0] + rel_rot[:, 1, 1] + rel_rot[:, 2, 2]
    cos_angle = torch.clamp((trace - 1.0) / 2.0, -1.0 + 1e-6, 1.0 - 1e-6)
    angles = torch.acos(cos_angle)
    rot_points = torch.clamp((angles / angle_step).int(), min=1)

    return torch.maximum(pos_points, rot_points).tolist()
