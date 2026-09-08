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

"""Trajectory computations on arrays and tensors.

Interpolation retains keyframe boundaries; resampling treats interior points
as optional path samples. Warping applies keyframe offsets to a trajectory."""

from __future__ import annotations

from .interpolation import interpolate_with_distance, interpolate_with_nums
from .resampling import resample_with_distance
from .warping import sort_and_padding_key_frame, warp_trajectory_qpos

__all__ = [
    "interpolate_with_distance",
    "interpolate_with_nums",
    "resample_with_distance",
    "sort_and_padding_key_frame",
    "warp_trajectory_qpos",
]
