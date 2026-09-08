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
