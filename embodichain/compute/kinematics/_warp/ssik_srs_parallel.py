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

"""One warp cooperates on the complete8-sheet SRS scalar objective.

All lanes keep identical bounded search state. Eight lanes evaluate one sheet
in parallel; a warp minimum broadcasts the unchanged full objective. This is
not branch pruning and keeps every anchor and all40 golden-section iterations.
"""

from __future__ import annotations

import warp as wp

from .ssik_7r import _rotation, vec7d
from .ssik_srs_geometry import _m, _v, srs_context_branch
from .ssik_srs_optimize import _closest_angle
from .ssik_srs_search import _srs_cost

wp.set_module_options({"enable_backward": False})

__all__: list[str] = []


@wp.func_native(snippet=r"""
#if defined(__CUDA_ARCH__)
    const unsigned mask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset /= 2) {
        double other = __shfl_down_sync(mask, value, offset);
        if (other < value) value = other;
    }
    return __shfl_sync(mask, value, 0);
#else
    return value;
#endif
""")
def _warp_min(value: wp.float64) -> wp.float64: ...


@wp.func
def _branch_context_cost(
    context: wp.array2d(dtype=wp.float64),
    i: int,
    axes: wp.array(dtype=wp.vec3d),
    parameter: wp.float64,
    free_roll: bool,
    weights: vec7d,
    branch: int,
) -> wp.float64:
    phi, roll = parameter, wp.float64(0)
    if free_roll:
        phi, roll = wp.float64(0), parameter
    cost = wp.float64(wp.inf)
    q, valid = srs_context_branch(context, i, axes, phi, roll, branch)
    if valid:
        seed = vec7d()
        for j in range(7):
            seed[j] = context[i, 93 + j]
            q[j] = _closest_angle(q[j], seed[j])
        cost = _srs_cost(q, seed, weights)
        if not wp.isfinite(cost):
            cost = wp.float64(wp.inf)
    return cost


@wp.func_native(snippet=r"""
#if defined(__CUDA_ARCH__)
    const unsigned mask = 0xffffffffu;
    for (int offset = 4; offset > 0; offset /= 2) {
        double other = __shfl_down_sync(mask, value, offset, 8);
        if (other < value) value = other;
    }
    return __shfl_sync(mask, value, 0, 8);
#else
    return value;
#endif
""")
def _group_min(value: wp.float64) -> wp.float64: ...


@wp.func_native(snippet=r"""
#if defined(__CUDA_ARCH__)
    return __shfl_sync(0xffffffffu, value, source);
#else
    return value;
#endif
""")
def _warp_read(value: wp.float64, source: int) -> wp.float64: ...


@wp.func
def parallel_context_cost(
    context: wp.array2d(dtype=wp.float64),
    i: int,
    axes: wp.array(dtype=wp.vec3d),
    parameter: wp.float64,
    free_roll: bool,
    weights: vec7d,
    lane: int,
) -> wp.float64:
    cost = wp.float64(wp.inf)
    if lane < 8:
        cost = _branch_context_cost(
            context, i, axes, parameter, free_roll, weights, lane
        )
    return _warp_min(cost)


@wp.func
def _anchor_context_cost(
    context: wp.array2d(dtype=wp.float64),
    i: int,
    axes: wp.array(dtype=wp.vec3d),
    parameter: wp.float64,
    weights: vec7d,
    lane: int,
) -> wp.float64:
    cost = _branch_context_cost(context, i, axes, parameter, False, weights, lane % 8)
    return _group_min(cost)


@wp.kernel
def optimize_srs_parallel(
    context: wp.array2d(dtype=wp.float64),
    axes: wp.array(dtype=wp.vec3d),
    weights_array: wp.array(dtype=wp.float32),
    count: int,
):
    """Preserve all pole anchors and40 golden iterations of the scalar search."""
    target, lane = wp.tid()
    i = target + count
    phi = wp.float64(0)
    defined = context[i, 0] != wp.float64(0)
    free = context[i, 1] != wp.float64(0)
    seed, weights = (vec7d(), vec7d())
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
        initial_cost = parallel_context_cost(context, i, axes, phi, free, weights, lane)
        defined = defined and wp.isfinite(initial_cost)
        if defined:
            best_cost = initial_cost
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
                        group = lane // 8
                        triple = group // 2
                        extremum = group % 2
                        first = axes[0]
                        vector = rsh * axes[2]
                        sign = wp.float64(1)
                        if triple == 1:
                            first = rsh * _rotation(axes[3], q[3]) * axes[4]
                            vector = _m(context, i, 22) * axes[6]
                            sign = wp.float64(-1)
                        a = wp.dot(first, vector) - wp.dot(first, direction) * wp.dot(
                            direction, vector
                        )
                        b = sign * wp.dot(first, wp.cross(direction, vector))
                        phase = wp.atan2(b, a)
                        trial = (
                            anchor
                            + phase
                            + wp.float64(extremum) * wp.float64(3.141592653589793)
                        )
                        cost = _anchor_context_cost(
                            context, i, axes, trial, weights, lane
                        )
                        # The shared elbow test is warp-uniform: every lane
                        # participates in these shuffles or every lane skips it.
                        # Preserve the original elbow/triple/extremum tie order.
                        for candidate in range(4):
                            candidate_cost = _warp_read(cost, candidate * 8)
                            candidate_phi = _warp_read(trial, candidate * 8)
                            if candidate_cost < best_cost:
                                phi, best_cost = candidate_phi, candidate_cost
            lower, upper = (phi - wp.float64(0.2), phi + wp.float64(0.2))
            ratio = wp.float64(0.6180339887498949)
            left = upper - ratio * (upper - lower)
            right = lower + ratio * (upper - lower)
            first_cost = parallel_context_cost(
                context, i, axes, left, free, weights, lane
            )
            second_cost = parallel_context_cost(
                context, i, axes, right, free, weights, lane
            )
            for iteration in range(40):
                if first_cost < best_cost:
                    phi, best_cost = (left, first_cost)
                if second_cost < best_cost:
                    phi, best_cost = (right, second_cost)
                if first_cost <= second_cost:
                    upper, right, second_cost = (right, left, first_cost)
                    left = upper - ratio * (upper - lower)
                    first_cost = parallel_context_cost(
                        context, i, axes, left, free, weights, lane
                    )
                else:
                    lower, left, first_cost = (left, right, second_cost)
                    right = lower + ratio * (upper - lower)
                    second_cost = parallel_context_cost(
                        context, i, axes, right, free, weights, lane
                    )
    if lane == 0:
        context[i, 100] = phi
        context[i, 101] = wp.float64(defined)
