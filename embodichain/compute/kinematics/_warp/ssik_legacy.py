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

"""Unchanged CPU/reference scheduling of all strict-geometry branches.

Kept in its own compilation module so CUDA acceptance/selection kernels do not
inherit the stack or code footprint of the legacy runtime-family generator.
"""

from __future__ import annotations

import warp as wp

from .ssik import _Model, _fk, _pose_ok
from .ssik_6r import ssik_three_parallel_limited
from .ssik_7r import mat87d, vec8i, vec7d, solve_srs, seed_swivel
from .ssik_srs_search import _srs_search_candidates

wp.set_module_options({"enable_backward": False})

__all__: list[str] = []


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
    weights: wp.array(dtype=wp.float32),
    samples: int,
    seed_local_search: int,
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
        if seed_local_search != 0 and sample == samples + 1:
            weight = vec7d()
            for j in range(model.dof):
                weight[j] = wp.float64(weights[j])
            branches, flags = _srs_search_candidates(
                axes,
                offsets,
                model.home,
                model.shoulder,
                model.wrist,
                goal,
                position_tolerance,
                qseed,
                weight,
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
