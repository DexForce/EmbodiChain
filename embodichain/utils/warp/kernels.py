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


@wp.kernel(enable_backward=False)
def scatter_contact_data(
    contact_data: wp.array(dtype=wp.float32, ndim=2),
    user_ids: wp.array(dtype=wp.int32, ndim=2),
    env_ids: wp.array(dtype=wp.int32, ndim=1),
    num_contacts_per_env: wp.array(dtype=wp.int32, ndim=1),
    max_contacts_per_env: int,
    # Output buffers
    position: wp.array(dtype=wp.float32, ndim=3),
    normal: wp.array(dtype=wp.float32, ndim=3),
    friction: wp.array(dtype=wp.float32, ndim=3),
    impulse: wp.array(dtype=wp.float32, ndim=2),
    distance: wp.array(dtype=wp.float32, ndim=2),
    user_ids_out: wp.array(dtype=wp.int32, ndim=3),
    is_valid: wp.array(dtype=wp.bool, ndim=2),
):
    """Scatters contact data into per-environment buffers.

    This kernel takes filtered contact data and scatters it into per-environment
    buffers. For each contact, it determines which environment it belongs to and
    the contact index within that environment (using atomic add for thread-safe counting).

    Args:
        contact_data: Input contact data. Shape is (n_contact, 11).
            Columns: [x, y, z, nx, ny, nz, fx, fy, fz, impulse, distance]
        user_ids: Input user IDs for each contact. Shape is (n_contact, 2).
        env_ids: Environment ID for each contact. Shape is (n_contact,).
        num_contacts_per_env: Output counter for contacts per environment. Shape is (num_envs,).
            Updated atomically during kernel execution.
        max_contacts_per_env: Maximum contacts per environment (buffer capacity).
        position: Output position buffer. Shape is (num_envs, max_contacts_per_env, 3).
        normal: Output normal buffer. Shape is (num_envs, max_contacts_per_env, 3).
        friction: Output friction buffer. Shape is (num_envs, max_contacts_per_env, 3).
        impulse: Output impulse buffer. Shape is (num_envs, max_contacts_per_env).
        distance: Output distance buffer. Shape is (num_envs, max_contacts_per_env).
        user_ids_out: Output user IDs buffer. Shape is (num_envs, max_contacts_per_env, 2).
        is_valid: Output validity mask. Shape is (num_envs, max_contacts_per_env).

    Note:
        If an environment has more contacts than max_contacts_per_env, excess contacts
        are silently dropped. The num_contacts_per_env output will reflect the actual
        number of contacts written (capped at max_contacts_per_env).
    """
    i = wp.tid()
    n_contact = contact_data.shape[0]

    if i >= n_contact:
        return

    env_id = env_ids[i]

    # Atomically increment contact counter for this environment
    contact_idx = wp.atomic_add(num_contacts_per_env, env_id, 1)

    # Drop excess contacts if buffer is full
    if contact_idx >= max_contacts_per_env:
        # Decrement counter since we didn't write this contact
        wp.atomic_sub(num_contacts_per_env, env_id, 1)
        return

    # Extract contact data columns
    x = contact_data[i, 0]
    y = contact_data[i, 1]
    z = contact_data[i, 2]
    nx = contact_data[i, 3]
    ny = contact_data[i, 4]
    nz = contact_data[i, 5]
    fx = contact_data[i, 6]
    fy = contact_data[i, 7]
    fz = contact_data[i, 8]
    impulse_val = contact_data[i, 9]
    distance_val = contact_data[i, 10]

    # Write to output buffers
    position[env_id, contact_idx, 0] = x
    position[env_id, contact_idx, 1] = y
    position[env_id, contact_idx, 2] = z

    normal[env_id, contact_idx, 0] = nx
    normal[env_id, contact_idx, 1] = ny
    normal[env_id, contact_idx, 2] = nz

    friction[env_id, contact_idx, 0] = fx
    friction[env_id, contact_idx, 1] = fy
    friction[env_id, contact_idx, 2] = fz

    impulse[env_id, contact_idx] = impulse_val
    distance[env_id, contact_idx] = distance_val

    user_ids_out[env_id, contact_idx, 0] = user_ids[i, 0]
    user_ids_out[env_id, contact_idx, 1] = user_ids[i, 1]

    is_valid[env_id, contact_idx] = True
