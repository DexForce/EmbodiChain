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

"""Sparse sensor contact reduction and interval timing kernels."""

from __future__ import annotations

import warp as wp


@wp.func
def find_actor(keys: wp.array(dtype=wp.int64), env: int, actor: int) -> int:
    if actor < 0:
        return -1
    key = wp.int64(env) * wp.int64(4294967296) + wp.int64(actor)
    lo = int(0)
    hi = keys.shape[0]
    while lo < hi:
        middle = (lo + hi) // 2
        if keys[middle] < key:
            lo = middle + 1
        else:
            hi = middle
    if lo < keys.shape[0] and keys[lo] == key:
        return lo
    return -1


@wp.func
def reduce_pair(
    env: int,
    actor0: int,
    actor1: int,
    force: wp.vec3,
    keys: wp.array(dtype=wp.int64),
    rows: wp.array(dtype=wp.int32),
    counterparts: wp.array(dtype=wp.int64),
    accept_all: bool,
    accept_unknown: bool,
    threshold: float,
    body_count: int,
    forces: wp.array(dtype=wp.float32, ndim=3),
    hits: wp.array(dtype=wp.int32, ndim=2),
    env_hits: wp.array(dtype=wp.int32),
):
    for side in range(2):
        actor = actor0
        other = actor1
        sign = float(-1.0)
        if side == 1:
            actor = actor1
            other = actor0
            sign = 1.0
        index = find_actor(keys, env, actor)
        if index >= 0:
            allowed = accept_all or (accept_unknown and other == -1)
            if not allowed:
                allowed = find_actor(counterparts, env, other) >= 0
            if allowed:
                body = rows[index] % body_count
                for axis in range(3):
                    wp.atomic_add(forces, env, body, axis, sign * force[axis])
                if threshold <= 0.0 or wp.length(force) > threshold:
                    wp.atomic_max(hits, env, body, 1)
                    wp.atomic_max(env_hits, env, 1)


@wp.kernel(enable_backward=False)
def reduce_contact_rows(
    data: wp.array(dtype=wp.float32, ndim=2),
    actors: wp.array(dtype=wp.int32, ndim=2),
    env_ids: wp.array(dtype=wp.int32),
    counts: wp.array(dtype=wp.int32),
    keys: wp.array(dtype=wp.int64),
    rows: wp.array(dtype=wp.int32),
    counterparts: wp.array(dtype=wp.int64),
    accept_all: bool,
    accept_unknown: bool,
    threshold: float,
    dt: float,
    forces: wp.array(dtype=wp.float32, ndim=3),
    hits: wp.array(dtype=wp.int32, ndim=2),
    env_hits: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    if i >= counts[0]:
        return
    env = env_ids[i]
    if env < 0 or env >= hits.shape[0] or hits.shape[1] == 0:
        return
    force = (
        wp.vec3(
            data[i, 3] * data[i, 9] + data[i, 6],
            data[i, 4] * data[i, 9] + data[i, 7],
            data[i, 5] * data[i, 9] + data[i, 8],
        )
        / dt
    )
    reduce_pair(
        env,
        actors[i, 0],
        actors[i, 1],
        force,
        keys,
        rows,
        counterparts,
        accept_all,
        accept_unknown,
        threshold,
        hits.shape[1],
        forces,
        hits,
        env_hits,
    )


@wp.kernel(enable_backward=False)
def reduce_contact_batch(
    actors: wp.array(dtype=wp.int32, ndim=3),
    valid: wp.array(dtype=wp.bool, ndim=2),
    normal: wp.array(dtype=wp.float32, ndim=3),
    friction: wp.array(dtype=wp.float32, ndim=3),
    impulse: wp.array(dtype=wp.float32, ndim=2),
    keys: wp.array(dtype=wp.int64),
    rows: wp.array(dtype=wp.int32),
    counterparts: wp.array(dtype=wp.int64),
    accept_all: bool,
    accept_unknown: bool,
    threshold: float,
    dt: float,
    forces: wp.array(dtype=wp.float32, ndim=3),
    hits: wp.array(dtype=wp.int32, ndim=2),
    env_hits: wp.array(dtype=wp.int32),
):
    env, i = wp.tid()
    if not valid[env, i] or hits.shape[1] == 0:
        return
    force = (
        wp.vec3(
            normal[env, i, 0] * impulse[env, i] + friction[env, i, 0],
            normal[env, i, 1] * impulse[env, i] + friction[env, i, 1],
            normal[env, i, 2] * impulse[env, i] + friction[env, i, 2],
        )
        / dt
    )
    reduce_pair(
        env,
        actors[env, i, 0],
        actors[env, i, 1],
        force,
        keys,
        rows,
        counterparts,
        accept_all,
        accept_unknown,
        threshold,
        hits.shape[1],
        forces,
        hits,
        env_hits,
    )


@wp.kernel(enable_backward=False)
def finish_contact_sample(
    dt: float,
    hits: wp.array(dtype=wp.int32, ndim=2),
    env_hits: wp.array(dtype=wp.int32),
    force: wp.array(dtype=wp.float32, ndim=3),
    peak: wp.array(dtype=wp.float32, ndim=3),
    contact: wp.array(dtype=wp.bool, ndim=2),
    found: wp.array(dtype=wp.bool, ndim=2),
    first: wp.array(dtype=wp.bool, ndim=2),
    air: wp.array(dtype=wp.float32, ndim=2),
    last_air: wp.array(dtype=wp.float32, ndim=2),
    count: wp.array(dtype=wp.float32),
):
    env, body = wp.tid()
    touching = hits[env, body] != 0
    landed = touching and not contact[env, body]
    elapsed = air[env, body] + dt
    if landed:
        last_air[env, body] = elapsed
    air[env, body] = elapsed
    if touching:
        air[env, body] = 0.0
    contact[env, body] = touching
    found[env, body] = found[env, body] or touching
    first[env, body] = first[env, body] or landed
    current = wp.vec3(force[env, body, 0], force[env, body, 1], force[env, body, 2])
    previous = wp.vec3(peak[env, body, 0], peak[env, body, 1], peak[env, body, 2])
    if wp.length_sq(current) > wp.length_sq(previous):
        for axis in range(3):
            peak[env, body, axis] = current[axis]
    if body == 0:
        count[env] = count[env] + float(env_hits[env])
