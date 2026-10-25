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

"""SSIK branch generation, original-model acceptance and bounded selection."""

from __future__ import annotations

import warp as wp

wp.set_module_options({"enable_backward": False})

from .ssik_6r import ssik_three_parallel_limited, mat73d, mat83d
from .ssik_7r import solve_srs, seed_swivel, vec7d, mat87d, vec8i, _rotation, _identity

__all__: list[str] = []


_mat812d = wp.types.matrix(shape=(8, 12), dtype=wp.float64)


@wp.struct
class _Model:
    h: mat73d
    p: mat83d
    home: wp.mat33d
    shoulder: wp.vec3d
    wrist: wp.vec3d
    dof: wp.int32
    frames: _mat812d
    local_axes: mat73d


@wp.func
def _fk(model: _Model, q: vec7d) -> wp.mat44d:
    rotation = _identity()
    position = wp.vec3d()
    # Acceptance retains original fixed frames, independently of the
    # normalized POE representation used for analytical reconstruction.
    for j in range(model.dof + 1):
        fixed = wp.mat33d()
        translation = wp.vec3d()
        for row in range(3):
            translation[row] = model.frames[j, row * 4 + 3]
            for column in range(3):
                fixed[row, column] = model.frames[j, row * 4 + column]
        position += rotation @ translation
        rotation = rotation @ fixed
        if j < model.dof:
            axis = wp.vec3d(
                model.local_axes[j, 0], model.local_axes[j, 1], model.local_axes[j, 2]
            )
            magnitude = wp.length(axis)
            rotation = rotation @ _rotation(axis / magnitude, q[j] * magnitude)
    result = wp.mat44d()
    for j in range(3):
        for k in range(3):
            result[j, k] = rotation[j, k]
        result[j, 3] = position[j]
    result[3, 3] = wp.float64(1.0)
    return result


@wp.func
def _pose_ok(
    actual: wp.mat44d, goal: wp.mat44d, ptol: wp.float64, rtol: wp.float64
) -> bool:
    displacement = wp.vec3d(
        actual[0, 3] - goal[0, 3], actual[1, 3] - goal[1, 3], actual[2, 3] - goal[2, 3]
    )
    r = wp.mat33d()
    for j in range(3):
        for k in range(3):
            for l in range(3):
                r[j, k] += actual[l, j] * goal[l, k]
    skew = wp.vec3d(r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1])
    angle = wp.atan2(wp.length(skew), r[0, 0] + r[1, 1] + r[2, 2] - wp.float64(1.0))
    return wp.length(displacement) <= ptol and angle <= rtol


@wp.kernel
def generate(
    model: _Model,
    axes: wp.array(dtype=wp.vec3d),
    offsets: wp.array(dtype=wp.vec3d),
    target: wp.array(dtype=wp.mat44d),
    tcp_inverse: wp.mat44d,
    tcp: wp.mat44d,
    seed: wp.array2d(dtype=wp.float64),
    lower: wp.array(dtype=wp.float64),
    upper: wp.array(dtype=wp.float64),
    samples: int,
    position_tolerance: wp.float64,
    rotation_tolerance: wp.float64,
    max_joint_step: wp.float64,
    validity: wp.array2d(dtype=wp.int32),
    candidates: wp.array3d(dtype=wp.float32),
):
    i, sample = wp.tid()
    qseed = vec7d()
    finite = bool(True)
    for j in range(model.dof):
        value = seed[i, j]
        finite = finite and wp.isfinite(value)
        if not wp.isfinite(value):
            value = (lower[j] + upper[j]) * wp.float64(0.5)
        qseed[j] = wp.clamp(value, lower[j], upper[j])
    for j in range(4):
        for k in range(4):
            finite = finite and wp.isfinite(target[i][j, k])
    goal = target[i] @ tcp_inverse
    rotation = wp.mat33d()
    for j in range(3):
        for k in range(3):
            rotation[j, k] = goal[j, k]
    gram = wp.transpose(rotation) @ rotation
    for j in range(3):
        for k in range(3):
            expected = wp.float64(0.0)
            if j == k:
                expected = wp.float64(1.0)
            finite = finite and wp.abs(gram[j, k] - expected) <= wp.float64(1e-5)
        finite = finite and wp.abs(target[i][3, j]) <= wp.float64(1e-6)
    finite = finite and wp.abs(target[i][3, 3] - wp.float64(1.0)) <= wp.float64(1e-6)
    finite = finite and wp.determinant(rotation) > wp.float64(0.0)
    position = wp.vec3d(goal[0, 3], goal[1, 3], goal[2, 3])
    # Both branch functions have identical fixed-sized result types.
    branches = mat87d()
    flags = vec8i()
    if model.dof == 6:
        low, high = vec7d(), vec7d()
        for j in range(model.dof):
            low[j] = lower[j]
            high[j] = upper[j]
        branches, flags = ssik_three_parallel_limited(
            model.h,
            model.p,
            position,
            rotation @ wp.transpose(model.home),
            qseed,
            low,
            high,
        )
    else:
        phi = -wp.float64(wp.pi) + wp.float64(2.0 * wp.pi) * wp.float64(
            sample - 1
        ) / wp.float64(samples)
        if sample == 0:
            phi, defined = seed_swivel(
                axes, offsets, model.home, model.shoulder, model.wrist, qseed
            )
            finite = finite and defined
        branches, flags = solve_srs(
            axes, offsets, model.home, model.shoulder, model.wrist, goal, phi, qseed
        )
    for branch in range(8):
        slot = sample * 8 + branch
        q = vec7d()
        accepted = finite and flags[branch] != 0
        for j in range(model.dof):
            value = branches[branch, j]
            turns = wp.floor(
                (qseed[j] - value) / wp.float64(2.0 * wp.pi) + wp.float64(0.5)
            )
            # A tangent branch can be a few float32 ulps past an authored
            # bound. Project only that roundoff-sized band, then validate the
            # returned coordinates against the original pose and hard limits.
            first = wp.ceil(
                (lower[j] - wp.float64(1e-6) - value) / wp.float64(2.0 * wp.pi)
            )
            last = wp.floor(
                (upper[j] + wp.float64(1e-6) - value) / wp.float64(2.0 * wp.pi)
            )
            turns = wp.clamp(turns, first, last)
            # Validate the same float32 coordinates that are returned to callers.
            q[j] = wp.float64(
                wp.float32(
                    wp.clamp(
                        value + wp.float64(2.0 * wp.pi) * turns, lower[j], upper[j]
                    )
                )
            )
            accepted = accepted and first <= last and wp.isfinite(q[j])
            accepted = accepted and q[j] >= lower[j] and q[j] <= upper[j]
            accepted = accepted and wp.abs(q[j] - qseed[j]) <= max_joint_step
        accepted = accepted and _pose_ok(
            _fk(model, q) @ tcp, target[i], position_tolerance, rotation_tolerance
        )
        validity[i, slot] = wp.int32(accepted)
        for j in range(model.dof):
            value = qseed[j]
            if accepted:
                value = q[j]
            candidates[i, slot, j] = wp.float32(value)


@wp.kernel
def select_nearest(
    candidates: wp.array3d(dtype=wp.float32),
    validity: wp.array2d(dtype=wp.int32),
    seed: wp.array2d(dtype=wp.float64),
    lower: wp.array(dtype=wp.float64),
    upper: wp.array(dtype=wp.float64),
    weights: wp.array(dtype=wp.float32),
    count: int,
    dof: int,
    result_valid: wp.array(dtype=wp.bool),
    result: wp.array2d(dtype=wp.float32),
):
    i = wp.tid()
    best = int(-1)
    cost = wp.float64(wp.inf)
    for k in range(count):
        if validity[i, k] != 0:
            distance = wp.float64(0.0)
            for j in range(dof):
                delta = (
                    wp.float64(candidates[i, k, j])
                    - wp.clamp(seed[i, j], lower[j], upper[j])
                ) * wp.float64(weights[j])
                distance += delta * delta
            if distance < cost:
                cost, best = distance, k
    result_valid[i] = best >= 0
    for j in range(dof):
        fallback = seed[i, j]
        if not wp.isfinite(fallback):
            fallback = (lower[j] + upper[j]) * wp.float64(0.5)
        value = wp.float32(wp.clamp(fallback, lower[j], upper[j]))
        if best >= 0:
            value = candidates[i, best, j]
        result[i, j] = value


@wp.kernel
def select_path(
    candidates: wp.array4d(dtype=wp.float32),
    validity: wp.array3d(dtype=wp.bool),
    initial: wp.array2d(dtype=wp.float32),
    lower: wp.array(dtype=wp.float64),
    upper: wp.array(dtype=wp.float64),
    weights: wp.array(dtype=wp.float32),
    steps: int,
    count: int,
    dof: int,
    max_joint_step: wp.float64,
    result_valid: wp.array2d(dtype=wp.bool),
    result: wp.array3d(dtype=wp.float32),
):
    i = wp.tid()
    previous = vec7d()
    for j in range(dof):
        value = wp.float64(initial[i, j])
        if not wp.isfinite(value):
            value = (lower[j] + upper[j]) * wp.float64(0.5)
        previous[j] = wp.clamp(value, lower[j], upper[j])
    for step in range(steps):
        best = int(-1)
        cost = wp.float64(wp.inf)
        for k in range(count):
            if validity[i, step, k]:
                distance = wp.float64(0.0)
                allowed = bool(True)
                for j in range(dof):
                    delta = wp.float64(candidates[i, step, k, j]) - previous[j]
                    allowed = (
                        allowed
                        and wp.isfinite(delta)
                        and wp.abs(delta) <= max_joint_step
                    )
                    distance += (
                        delta * delta * wp.float64(weights[j]) * wp.float64(weights[j])
                    )
                if allowed and distance < cost:
                    cost, best = distance, k
        result_valid[i, step] = best >= 0
        for j in range(dof):
            if best >= 0:
                previous[j] = wp.float64(candidates[i, step, best, j])
            result[i, step, j] = wp.float32(previous[j])
