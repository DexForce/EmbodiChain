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

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from embodichain.compute.geometry._warp.convex_query import (
    convex_signed_distance_kernel,
)
from embodichain.compute.image._warp.tiling import reshape_tiled_image
from embodichain.compute.trajectory._warp.warping import compute_offset_key_poses_kernel


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.gpu)]
)
def test_convex_query_returns_maximum_halfspace_value(device: str) -> None:
    wp.init()
    # Unit cube: inside, face, and outside a corner. The last value is the
    # maximum plane distance (1), not the Euclidean distance to the cube.
    points = wp.array(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 2.0, 0.0]]],
        dtype=wp.float32,
        device=device,
    )
    planes = wp.array(
        [
            [
                [1.0, 0.0, 0.0, -1.0],
                [-1.0, 0.0, 0.0, -1.0],
                [0.0, 1.0, 0.0, -1.0],
                [0.0, -1.0, 0.0, -1.0],
                [0.0, 0.0, 1.0, -1.0],
                [0.0, 0.0, -1.0, -1.0],
            ]
        ],
        dtype=wp.float32,
        device=device,
    )
    counts = wp.array([6], dtype=wp.int32, device=device)
    output = wp.full((1, 3, 1), -float("inf"), dtype=wp.float32, device=device)
    wp.launch(
        convex_signed_distance_kernel,
        dim=(1, 3, 1),
        inputs=[points, planes, counts],
        outputs=[output],
        device=device,
    )
    np.testing.assert_allclose(output.numpy().reshape(-1), [-1.0, 0.0, 1.0])


@pytest.mark.parametrize("dtype", [wp.uint8, wp.uint32, wp.float32])
def test_tiled_image_preserves_registered_overloads(dtype: type) -> None:
    wp.init()
    # Four one-pixel tiles retain the existing bottom-to-top ordering.
    source = wp.array([0, 1, 2, 3], dtype=dtype, device="cpu")
    output = wp.empty((4, 1, 1, 1), dtype=dtype, device="cpu")
    wp.launch(
        reshape_tiled_image,
        dim=(4, 1, 1),
        inputs=[source, output, 1, 1, 1, 2],
        device="cpu",
    )
    np.testing.assert_array_equal(output.numpy().reshape(-1), [2, 3, 0, 1])


def test_offset_kernel_outputs_transformed_poses() -> None:
    wp.init()
    key_pose = np.eye(4, dtype=np.float32)
    key_pose[0, 3] = 2.0
    offset = np.eye(4, dtype=np.float32)
    offset[1, 3] = 3.0
    base_inv = np.eye(4, dtype=np.float32)
    base_inv[0, 3] = -1.0
    output = wp.zeros(16, dtype=wp.float32, device="cpu")
    wp.launch(
        compute_offset_key_poses_kernel,
        dim=(1, 1),
        inputs=[
            wp.array([0], dtype=wp.int32, device="cpu"),
            wp.array(offset.ravel(), device="cpu"),
            wp.array(key_pose.ravel(), device="cpu"),
            wp.mat44f(base_inv),
            1,
            1,
        ],
        outputs=[output],
        device="cpu",
    )
    np.testing.assert_allclose(
        output.numpy().reshape(4, 4), base_inv @ key_pose @ offset
    )
