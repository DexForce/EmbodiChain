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
"""Sample pose transitions and axis-relative helices on a control grid."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

__all__: list[str] = []


def _quintic_progress(u: np.ndarray) -> np.ndarray:
    u = np.clip(u, 0.0, 1.0)
    return u**3 * (10 - 15 * u + 6 * u * u)


def _interpolate_pose_path(
    start: np.ndarray, end: np.ndarray, progress: np.ndarray
) -> np.ndarray:
    weight = progress[:, None]
    result = np.tile(np.eye(4), (len(weight), 1, 1))
    result[:, :3, 3] = (1 - weight) * start[:3, 3] + weight * end[:3, 3]
    result[:, :3, :3] = Slerp(
        [0.0, 1.0], Rotation.from_matrix([start[:3, :3], end[:3, :3]])
    )(progress).as_matrix()
    return result


def _sample_axis_arc(
    axis: np.ndarray,
    radius: float,
    angles: np.ndarray,
    heights: np.ndarray,
    offset: np.ndarray,
) -> np.ndarray:
    local = np.stack(
        (radius * np.cos(angles), radius * np.sin(angles), heights), axis=-1
    )
    return local @ axis[:3, :3].T + axis[:3, 3] + offset
