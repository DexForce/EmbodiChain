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

from .ssik_6r import (
    mat73d,
    mat83d,
)
from .ssik_7r import vec7d, _rotation, _identity

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
    frames: wp.array2d(dtype=wp.float64)
    local_axes: wp.array2d(dtype=wp.float64)


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


@wp.func
def _validate_candidate(
    model: _Model,
    raw: vec7d,
    target: wp.mat44d,
    tcp: wp.mat44d,
    seed: vec7d,
    lower: vec7d,
    upper: vec7d,
    ptol: wp.float64,
    rtol: wp.float64,
    max_step: wp.float64,
):
    """Validate one returned-precision branch against the original model."""
    q, fallback = vec7d(), vec7d()
    valid = bool(True)
    rotation = wp.mat33d()
    for row in range(4):
        for column in range(4):
            valid = valid and wp.isfinite(target[row, column])
            if row < 3 and column < 3:
                rotation[row, column] = target[row, column]
    gram = wp.transpose(rotation) @ rotation
    for row in range(3):
        for column in range(3):
            expected = wp.float64(0.0)
            if row == column:
                expected = wp.float64(1.0)
            valid = valid and wp.abs(gram[row, column] - expected) <= wp.float64(1e-5)
        valid = valid and wp.abs(target[3, row]) <= wp.float64(1e-6)
    valid = valid and wp.abs(target[3, 3] - wp.float64(1.0)) <= wp.float64(1e-6)
    valid = valid and wp.determinant(rotation) > wp.float64(0.0)
    for j in range(model.dof):
        origin = seed[j]
        valid = valid and wp.isfinite(origin)
        if not wp.isfinite(origin):
            origin = (lower[j] + upper[j]) * wp.float64(0.5)
        origin = wp.clamp(origin, lower[j], upper[j])
        fallback[j] = origin
        value = raw[j]
        valid = valid and wp.isfinite(value)
        first = wp.ceil((lower[j] - wp.float64(1e-6) - value) / wp.float64(2.0 * wp.pi))
        last = wp.floor((upper[j] + wp.float64(1e-6) - value) / wp.float64(2.0 * wp.pi))
        turns = wp.clamp(
            wp.floor((origin - value) / wp.float64(2.0 * wp.pi) + wp.float64(0.5)),
            first,
            last,
        )
        q[j] = wp.float64(
            wp.float32(
                wp.clamp(value + wp.float64(2.0 * wp.pi) * turns, lower[j], upper[j])
            )
        )
        valid = valid and first <= last and q[j] >= lower[j] and q[j] <= upper[j]
        valid = valid and wp.isfinite(q[j]) and wp.abs(q[j] - origin) <= max_step
    valid = valid and _pose_ok(_fk(model, q) @ tcp, target, ptol, rtol)
    if not valid:
        q = fallback
    return q, valid


@wp.kernel
def accept_raw(
    model: _Model,
    raw: wp.array3d(dtype=wp.float64),
    raw_valid: wp.array2d(dtype=wp.int32),
    target: wp.array(dtype=wp.mat44d),
    tcp: wp.mat44d,
    seed: wp.array2d(dtype=wp.float64),
    lower: wp.array(dtype=wp.float64),
    upper: wp.array(dtype=wp.float64),
    ptol: wp.float64,
    rtol: wp.float64,
    max_step: wp.float64,
    validity: wp.array2d(dtype=wp.int32),
    candidates: wp.array3d(dtype=wp.float32),
):
    i, branch = wp.tid()
    initial, origin, lo, hi = vec7d(), vec7d(), vec7d(), vec7d()
    for j in range(model.dof):
        initial[j] = raw[i, branch, j]
        origin[j] = seed[i, j]
        lo[j] = lower[j]
        hi[j] = upper[j]
    q, accepted = _validate_candidate(
        model, initial, target[i], tcp, origin, lo, hi, ptol, rtol, max_step
    )
    accepted = accepted and raw_valid[i, branch] != 0
    validity[i, branch] = wp.int32(accepted)
    for j in range(model.dof):
        value = q[j]
        if not accepted:
            value = origin[j]
            if not wp.isfinite(value):
                value = (lo[j] + hi[j]) * wp.float64(0.5)
            value = wp.clamp(value, lo[j], hi[j])
        candidates[i, branch, j] = wp.float32(value)


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
    model: _Model,
    targets: wp.array2d(dtype=wp.mat44d),
    tcp: wp.mat44d,
    ptol: wp.float64,
    rtol: wp.float64,
    check_pose: bool,
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
        best_q = vec7d()
        cost = wp.float64(wp.inf)
        for k in range(count):
            if validity[i, step, k]:
                distance = wp.float64(0.0)
                allowed = bool(True)
                changed = bool(False)
                q = vec7d()
                for j in range(dof):
                    original = wp.float64(candidates[i, step, k, j])
                    value = original
                    # An already-near representative cannot be improved by
                    # another winding. Keep the ordinary path inexpensive.
                    if wp.isfinite(value) and wp.abs(value - previous[j]) > wp.float64(
                        wp.pi
                    ):
                        period = wp.float64(2.0 * wp.pi)
                        first = wp.ceil((lower[j] - wp.float64(1e-6) - value) / period)
                        last = wp.floor((upper[j] + wp.float64(1e-6) - value) / period)
                        turns = wp.clamp(
                            wp.floor((previous[j] - value) / period + wp.float64(0.5)),
                            first,
                            last,
                        )
                        value = wp.float64(
                            wp.float32(
                                wp.clamp(value + period * turns, lower[j], upper[j])
                            )
                        )
                        allowed = allowed and first <= last
                    q[j] = value
                    changed = changed or value != original
                    delta = value - previous[j]
                    allowed = (
                        allowed
                        and value >= lower[j]
                        and value <= upper[j]
                        and wp.isfinite(delta)
                        and wp.abs(delta) <= max_joint_step
                    )
                    distance += (
                        delta * delta * wp.float64(weights[j]) * wp.float64(weights[j])
                    )
                if allowed and distance < cost:
                    # Changing a float32 winding can change its FK rounding.
                    # Reject this branch, rather than the whole waypoint, if
                    # its actual returned coordinates no longer close.
                    if check_pose and changed:
                        allowed = _pose_ok(
                            _fk(model, q) @ tcp, targets[i, step], ptol, rtol
                        )
                    if allowed:
                        cost, best, best_q = distance, k, q
        result_valid[i, step] = best >= 0
        for j in range(dof):
            if best >= 0:
                previous[j] = best_q[j]
            result[i, step, j] = wp.float32(previous[j])


# The legacy CPU generator keeps its import path without sharing CUDA code.
def __getattr__(name: str) -> object:
    if name == "generate":
        from .ssik_legacy import generate

        return generate
    raise AttributeError(name)
