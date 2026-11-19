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


"""Convert packed rigid-body positions and xyzw quaternions to matrices."""

from __future__ import annotations

import warp as wp


@wp.kernel(enable_backward=False)
def _rigid_pose_to_matrix(
    poses: wp.array2d(dtype=wp.float32),
    matrices: wp.array(dtype=wp.mat44),
):
    row = wp.tid()
    x = poses[row, 3]
    y = poses[row, 4]
    z = poses[row, 5]
    w = poses[row, 6]
    scale = 2.0 / (x * x + y * y + z * z + w * w)
    matrices[row] = wp.mat44(
        1.0 - scale * (y * y + z * z),
        scale * (x * y - z * w),
        scale * (x * z + y * w),
        poses[row, 0],
        scale * (x * y + z * w),
        1.0 - scale * (x * x + z * z),
        scale * (y * z - x * w),
        poses[row, 1],
        scale * (x * z - y * w),
        scale * (y * z + x * w),
        1.0 - scale * (x * x + y * y),
        poses[row, 2],
        0.0,
        0.0,
        0.0,
        1.0,
    )
