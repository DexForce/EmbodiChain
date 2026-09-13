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

"""Interactive viewer, marker, and recording configuration."""

from __future__ import annotations

from typing import List, Literal

import torch
from dexsim.types import AxisArrowType, AxisCornerType

from embodichain.utils import configclass


@configclass
class MarkerCfg:
    """Configuration for visual markers in the simulation.

    This class defines properties for creating visual markers such as coordinate frames,
    lines, and points that can be used for debugging, visualization, or reference purposes
    in the simulation environment.
    """

    name: str = "empty-mesh"
    """Name of the marker for identification purposes."""

    marker_type: Literal["axis", "line", "point"] = "axis"
    """Type of marker to display. Can be 'axis' (3D coordinate frame), 'line', or 'point'. (only axis supported now)"""

    axis_xpos: torch.Tensor | None = None
    """List of 4x4 transformation matrices defining the position and orientation of each axis marker."""

    axis_size: float = 0.002
    """Thickness/size of the axis lines in meters."""

    axis_len: float = 0.005
    """Length of each axis arm in meters."""

    line_color: List[float] = [1, 1, 0, 1.0]
    """RGBA color values for the marker lines. Values should be between 0.0 and 1.0."""

    arrow_type: AxisArrowType = AxisArrowType.CONE
    """Type of arrow head for axis markers (e.g., CONE, ARROW, etc.)."""

    corner_type: AxisCornerType = AxisCornerType.SPHERE
    """Type of corner/joint visualization for axis markers (e.g., SPHERE, CUBE, etc.)."""

    arena_index: int = -1
    """Index of the arena where the marker should be placed. -1 means all arenas."""


@configclass
class WindowRecordCfg:
    """Configuration for interactive viewer window recording."""

    enable_hotkey: bool = True
    """Whether to register the ``r`` hotkey for viewer recording when the window opens."""

    save_path: str | None = None
    """Optional output path for viewer recordings. If None, use the default outputs directory."""

    fps: int = 20
    """Frames per second for viewer recording."""

    max_memory: int = 1024
    """Maximum buffered recording memory in MB before auto-stopping capture."""

    video_prefix: str = "viewer_record"
    """Video file prefix used when no explicit save path is provided."""


@configclass
class WindowCameraPoseCfg:
    """Configuration for printing the interactive viewer camera pose."""

    enable_hotkey: bool = True
    """Whether to register the ``p`` hotkey when the window opens."""

    convert_to_look_at: bool = True
    """Whether the hotkey prints a ``set_look_at`` call instead of a matrix."""
