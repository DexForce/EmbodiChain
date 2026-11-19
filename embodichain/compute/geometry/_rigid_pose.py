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


"""Device-resident matrix conversion for batched rigid poses."""

from __future__ import annotations

import torch
import warp as wp

from embodichain.utils.device_utils import current_warp_stream

from ._warp.rigid_pose import _rigid_pose_to_matrix


def _rigid_pose_matrices(poses: torch.Tensor) -> torch.Tensor:
    """Convert float32 (x, y, z, qx, qy, qz, qw) poses on the caller's stream."""
    matrices = torch.empty(
        (poses.shape[0], 4, 4), dtype=poses.dtype, device=poses.device
    )
    wp.launch(
        _rigid_pose_to_matrix,
        dim=poses.shape[0],
        inputs=[wp.from_torch(poses)],
        outputs=[wp.from_torch(matrices, dtype=wp.mat44)],
        device=str(poses.device),
        stream=current_warp_stream(poses.device),
    )
    return matrices
