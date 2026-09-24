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

"""Task-owned camera presets for native policy evaluation."""

from __future__ import annotations

from embodichain.utils import configclass

__all__ = ["PolicyViewerCameraCfg"]


@configclass
class PolicyViewerCameraCfg:
    """Configure planar tracking in the native policy Viewer.

    A task opts in by exposing ``policy_viewer_camera_cfg`` and a callable
    ``get_policy_viewer_target_pose()`` returning a world pose as seven values
    ``(x, y, z, qx, qy, qz, qw)``. Tasks without a preset retain their own view.

    On reset, the eye offset is rotated by the target's initial yaw. Subsequent
    updates translate the camera only in world X-Y, preserving orbit and zoom
    adjustments. Target height is fixed in world coordinates for flat scenes.
    """

    #: Eye position relative to the look-at point, in meters in the target's
    #: initial heading frame. Negative X looks from behind; the horizontal
    #: offset must be nonzero.
    eye_offset: tuple[float, float, float] = (-1.8, -1.4, 0.8)
    #: Fixed world Z of the look-at point, in meters.
    target_height: float = 0.6
