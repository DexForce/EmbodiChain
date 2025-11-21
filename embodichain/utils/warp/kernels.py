# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
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

import warp as wp
from typing import Any


@wp.kernel(enable_backward=False)
def reshape_tiled_image(
    tiled_image_buffer: Any,
    batched_image: Any,
    image_height: int,
    image_width: int,
    num_channels: int,
    num_tiles_x: int,
):
    """Reshapes a tiled image into a batch of images.

    This function reshapes the input tiled image buffer into a batch of images. The input image buffer
    is assumed to be tiled in the x and y directions. The output image is a batch of images with the
    specified height, width, and number of channels.

    Args:
        tiled_image_buffer: The input image buffer. Shape is (height * width * num_channels * num_cameras,).
        batched_image: The output image. Shape is (num_cameras, height, width, num_channels).
        image_width: The width of the image.
        image_height: The height of the image.
        num_channels: The number of channels in the image.
        num_tiles_x: The number of tiles in x-direction.
    """
    # get the thread id
    camera_id, height_id, width_id = wp.tid()

    # resolve the tile indices
    tile_x_id = camera_id % num_tiles_x
    # TODO: Currently, the tiles arranged in the bottom-to-top order, which should be changed.
    tile_y_id = (
        num_tiles_x - 1 - (camera_id // num_tiles_x)
    )  # Adjust for bottom-to-top tiling
    # compute the start index of the pixel in the tiled image buffer
    pixel_start = (
        num_channels
        * num_tiles_x
        * image_width
        * (image_height * tile_y_id + height_id)
        + num_channels * tile_x_id * image_width
        + num_channels * width_id
    )

    # copy the pixel values into the batched image
    for i in range(num_channels):
        batched_image[camera_id, height_id, width_id, i] = batched_image.dtype(
            tiled_image_buffer[pixel_start + i]
        )


# uint32 -> int32 conversion is required for non-colored segmentation annotators
wp.overload(
    reshape_tiled_image,
    {
        "tiled_image_buffer": wp.array(dtype=wp.uint32),
        "batched_image": wp.array(dtype=wp.uint32, ndim=4),
    },
)
# uint8 is used for 4 channel annotators
wp.overload(
    reshape_tiled_image,
    {
        "tiled_image_buffer": wp.array(dtype=wp.uint8),
        "batched_image": wp.array(dtype=wp.uint8, ndim=4),
    },
)
# float32 is used for single channel annotators
wp.overload(
    reshape_tiled_image,
    {
        "tiled_image_buffer": wp.array(dtype=wp.float32),
        "batched_image": wp.array(dtype=wp.float32, ndim=4),
    },
)
