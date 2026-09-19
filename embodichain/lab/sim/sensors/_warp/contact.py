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

"""Scatter simulation contacts into per-environment sensor buffers."""

from __future__ import annotations

import warp as wp


@wp.func
def _scatter_contact_row(
    i: int,
    contact_data: wp.array(dtype=wp.float32, ndim=2),
    user_ids: wp.array(dtype=wp.int32, ndim=2),
    env_ids: wp.array(dtype=wp.int32),
    num_contacts_per_env: wp.array(dtype=wp.int32),
    max_contacts_per_env: int,
    position: wp.array(dtype=wp.float32, ndim=3),
    normal: wp.array(dtype=wp.float32, ndim=3),
    friction: wp.array(dtype=wp.float32, ndim=3),
    impulse: wp.array(dtype=wp.float32, ndim=2),
    distance: wp.array(dtype=wp.float32, ndim=2),
    user_ids_out: wp.array(dtype=wp.int32, ndim=3),
    is_valid: wp.array(dtype=wp.bool, ndim=2),
    record_overflow: bool,
    dropped: wp.array(dtype=wp.int32),
):
    env_id = env_ids[i]
    if env_id < 0 or env_id >= num_contacts_per_env.shape[0]:
        return
    # Atomically increment contact counter for this environment
    contact_idx = wp.atomic_add(num_contacts_per_env, env_id, 1)

    # Drop excess contacts if buffer is full
    if contact_idx >= max_contacts_per_env:
        if record_overflow:
            wp.atomic_add(dropped, env_id, 1)
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


@wp.kernel(enable_backward=False)
def scatter_contact_data(
    contact_data: wp.array(dtype=wp.float32, ndim=2),
    user_ids: wp.array(dtype=wp.int32, ndim=2),
    env_ids: wp.array(dtype=wp.int32),
    num_contacts_per_env: wp.array(dtype=wp.int32),
    max_contacts_per_env: int,
    position: wp.array(dtype=wp.float32, ndim=3),
    normal: wp.array(dtype=wp.float32, ndim=3),
    friction: wp.array(dtype=wp.float32, ndim=3),
    impulse: wp.array(dtype=wp.float32, ndim=2),
    distance: wp.array(dtype=wp.float32, ndim=2),
    user_ids_out: wp.array(dtype=wp.int32, ndim=3),
    is_valid: wp.array(dtype=wp.bool, ndim=2),
):
    """Scatter a compact array, retaining the legacy kernel argument order."""
    if wp.tid() >= contact_data.shape[0]:
        return
    _scatter_contact_row(
        wp.tid(),
        contact_data,
        user_ids,
        env_ids,
        num_contacts_per_env,
        max_contacts_per_env,
        position,
        normal,
        friction,
        impulse,
        distance,
        user_ids_out,
        is_valid,
        False,
        num_contacts_per_env,
    )


@wp.kernel(enable_backward=False)
def scatter_contact_rows(
    contact_data: wp.array(dtype=wp.float32, ndim=2),
    user_ids: wp.array(dtype=wp.int32, ndim=2),
    env_ids: wp.array(dtype=wp.int32),
    num_contacts_per_env: wp.array(dtype=wp.int32),
    max_contacts_per_env: int,
    position: wp.array(dtype=wp.float32, ndim=3),
    normal: wp.array(dtype=wp.float32, ndim=3),
    friction: wp.array(dtype=wp.float32, ndim=3),
    impulse: wp.array(dtype=wp.float32, ndim=2),
    distance: wp.array(dtype=wp.float32, ndim=2),
    user_ids_out: wp.array(dtype=wp.int32, ndim=3),
    is_valid: wp.array(dtype=wp.bool, ndim=2),
    counts: wp.array(dtype=wp.int32),
    dropped: wp.array(dtype=wp.int32),
):
    """Scatter valid device-counted rows and record per-environment overflow."""
    i = wp.tid()
    if i < counts[0]:
        _scatter_contact_row(
            i,
            contact_data,
            user_ids,
            env_ids,
            num_contacts_per_env,
            max_contacts_per_env,
            position,
            normal,
            friction,
            impulse,
            distance,
            user_ids_out,
            is_valid,
            True,
            dropped,
        )
