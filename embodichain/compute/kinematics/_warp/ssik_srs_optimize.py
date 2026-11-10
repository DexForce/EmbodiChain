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

"""Bounded SRS redundancy optimization using cached target geometry."""

from __future__ import annotations

import warp as wp

from .ssik_7r import _rotation, _mat23d, vec7d
from .ssik_srs_geometry import (
    _m,
    _v,
    _srs_shoulder,
    srs_context_branch,
    _decompose_cached,
)
from .ssik_srs_search import _srs_cost

wp.set_module_options({"enable_backward": False})

__all__: list[str] = []


@wp.func
def _closest_angle(value: wp.float64, seed: wp.float64) -> wp.float64:
    return value + wp.float64(6.283185307179586) * wp.floor(
        (seed - value) / wp.float64(6.283185307179586) + wp.float64(0.5)
    )


@wp.func
def _triple_best(
    angles: _mat23d,
    seed: vec7d,
    weights: vec7d,
    start: int,
) -> wp.vec3d:
    best = wp.vec3d()
    best_cost = wp.float64(wp.inf)
    for branch in range(2):
        q = wp.vec3d()
        cost = wp.float64(0)
        for j in range(3):
            q[j] = _closest_angle(angles[branch, j], seed[start + j])
            d = (q[j] - seed[start + j]) * weights[start + j]
            cost += d * d
        if cost < best_cost:
            best, best_cost = q, cost
    return best


@wp.func
def context_nearest(
    context: wp.array2d(dtype=wp.float64),
    i: int,
    axes: wp.array(dtype=wp.vec3d),
    phi: wp.float64,
    roll: wp.float64,
    weights: vec7d,
) -> tuple[vec7d, bool]:
    """Minimize separable shoulder/wrist costs without an8x7 branch array.

    For each elbow root the Cartesian product of two shoulder and two wrist
    sheets has an additive joint-distance objective. Selecting its two minima
    preserves the complete eight-branch objective and first-branch tie order.
    """
    seed, best = vec7d(), vec7d()
    for j in range(7):
        seed[j] = context[i, 93 + j]
    found = bool(False)
    best_cost = wp.float64(wp.inf)
    if context[i, 0] != wp.float64(0):
        for elbow in range(2):
            rsh = _srs_shoulder(context, i, elbow, phi, roll)
            residual = wp.transpose(rsh * _m(context, i, 57 + elbow * 9)) * _m(
                context, i, 22
            )
            sha, shv = _decompose_cached(
                rsh,
                axes[0],
                axes[1],
                axes[2],
                seed[0],
                context[i, 102],
                context[i, 103],
            )
            wra, wrv = _decompose_cached(
                residual,
                axes[4],
                axes[5],
                axes[6],
                seed[4],
                context[i, 104],
                context[i, 105],
            )
            if shv and wrv:
                shoulder = _triple_best(sha, seed, weights, 0)
                wrist = _triple_best(wra, seed, weights, 4)
                q = vec7d()
                for j in range(3):
                    q[j] = shoulder[j]
                    q[j + 4] = wrist[j]
                q[3] = _closest_angle(context[i, 40 + elbow], seed[3])
                cost = _srs_cost(q, seed, weights)
                if cost < best_cost:
                    best, best_cost, found = q, cost, True
    return best, found


@wp.func
def context_cost(
    context: wp.array2d(dtype=wp.float64),
    i: int,
    axes: wp.array(dtype=wp.vec3d),
    parameter: wp.float64,
    free_roll: bool,
    weights: vec7d,
) -> wp.float64:
    phi, roll = parameter, wp.float64(0)
    if free_roll:
        phi, roll = wp.float64(0), parameter
    q, found = context_nearest(context, i, axes, phi, roll, weights)
    cost = wp.float64(wp.inf)
    if found:
        seed = vec7d()
        for j in range(7):
            seed[j] = context[i, 93 + j]
        cost = _srs_cost(q, seed, weights)
    return cost


@wp.kernel
def optimize_srs_context(
    context: wp.array2d(dtype=wp.float64),
    axes: wp.array(dtype=wp.vec3d),
    weights_array: wp.array(dtype=wp.float32),
    count: int,
):
    """Preserve all pole anchors and40 golden iterations of the scalar search."""
    target = wp.tid()
    i = target + count
    phi = wp.float64(0)
    defined = context[i, 0] != wp.float64(0)
    free = context[i, 1] != wp.float64(0)
    seed, weights = vec7d(), vec7d()
    for j in range(7):
        seed[j] = context[i, 93 + j]
        weights[j] = wp.float64(weights_array[j])
    if defined:
        if not free:
            upper_seed = _m(context, i, 31) * (_v(context, i, 7) * context[i, 2])
            direction = _v(context, i, 13)
            radial = upper_seed - direction * wp.dot(upper_seed, direction)
            if wp.length(radial) > wp.float64(1e-12):
                phi = wp.atan2(
                    wp.dot(radial, _v(context, i, 19)),
                    wp.dot(radial, _v(context, i, 16)),
                )
        initial, found = context_nearest(context, i, axes, phi, wp.float64(0), weights)
        defined = defined and found
        if defined:
            best_cost = _srs_cost(initial, seed, weights)
            if not free:
                anchor = phi
                direction = _v(context, i, 13)
                for elbow in range(2):
                    q, valid = srs_context_branch(
                        context, i, axes, anchor, wp.float64(0), elbow * 4
                    )
                    if valid:
                        rsh = (
                            _rotation(axes[0], q[0])
                            * _rotation(axes[1], q[1])
                            * _rotation(axes[2], q[2])
                        )
                        for triple in range(2):
                            first = axes[0]
                            vector = rsh * axes[2]
                            sign = wp.float64(1)
                            if triple == 1:
                                first = rsh * _rotation(axes[3], q[3]) * axes[4]
                                vector = _m(context, i, 22) * axes[6]
                                sign = wp.float64(-1)
                            a = wp.dot(first, vector) - wp.dot(
                                first, direction
                            ) * wp.dot(direction, vector)
                            b = sign * wp.dot(first, wp.cross(direction, vector))
                            phase = wp.atan2(b, a)
                            for extremum in range(2):
                                trial = (
                                    anchor
                                    + phase
                                    + wp.float64(extremum)
                                    * wp.float64(3.141592653589793)
                                )
                                cost = context_cost(
                                    context, i, axes, trial, False, weights
                                )
                                if cost < best_cost:
                                    phi, best_cost = trial, cost
            lower, upper = phi - wp.float64(0.2), phi + wp.float64(0.2)
            ratio = wp.float64(0.6180339887498949)
            left = upper - ratio * (upper - lower)
            right = lower + ratio * (upper - lower)
            first_cost = context_cost(context, i, axes, left, free, weights)
            second_cost = context_cost(context, i, axes, right, free, weights)
            for iteration in range(40):
                if first_cost < best_cost:
                    phi, best_cost = left, first_cost
                if second_cost < best_cost:
                    phi, best_cost = right, second_cost
                if first_cost <= second_cost:
                    upper, right, second_cost = right, left, first_cost
                    left = upper - ratio * (upper - lower)
                    first_cost = context_cost(context, i, axes, left, free, weights)
                else:
                    lower, left, first_cost = left, right, second_cost
                    right = lower + ratio * (upper - lower)
                    second_cost = context_cost(context, i, axes, right, free, weights)
    context[i, 100] = phi
    context[i, 101] = wp.float64(defined)


@wp.kernel
def srs_raw(
    context: wp.array2d(dtype=wp.float64),
    axes: wp.array(dtype=wp.vec3d),
    samples: int,
    local_search: int,
    count: int,
    raw_valid: wp.array2d(dtype=wp.int32),
    raw: wp.array3d(dtype=wp.float64),
):
    """Reconstruct one branch per thread over the complete unchanged sample set."""
    i, slot = wp.tid()
    sample, branch = slot // 8, slot % 8
    row = i
    phi, roll = wp.float64(0), wp.float64(0)
    defined = bool(True)
    if local_search != 0 and sample == samples + 1:
        row = i + count
        defined = context[row, 101] != wp.float64(0)
        if context[row, 1] != wp.float64(0):
            roll = context[row, 100]
        else:
            phi = context[row, 100]
    elif sample == 0:
        phi = context[i, 106]
        defined = context[i, 107] != wp.float64(0)
    else:
        phi = -wp.float64(wp.pi) + wp.float64(2.0 * wp.pi) * wp.float64(
            sample - 1
        ) / wp.float64(samples)
    q, valid = srs_context_branch(context, row, axes, phi, roll, branch)
    raw_valid[i, slot] = wp.int32(defined and valid)
    for j in range(7):
        raw[i, slot, j] = q[j]
