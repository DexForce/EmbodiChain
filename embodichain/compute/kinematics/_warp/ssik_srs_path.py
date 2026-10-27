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

"""Target-aware continuous SRS IK with explicit per-path seed ownership."""

from __future__ import annotations

import warp as wp

from .ssik import _Model, _validate_candidate
from .ssik_7r import mat87d, seed_swivel, solve_srs, vec7d, vec8i
from .ssik_srs_search import _srs_search_candidates

wp.set_module_options({"enable_backward": False})

__all__: list[str] = []


@wp.func
def _srs_valid_target(target: wp.mat44d, inverse_tcp: wp.mat44d) -> bool:
    valid = bool(True)
    for j in range(4):
        for k in range(4):
            valid = valid and wp.isfinite(target[j, k])
    goal = target @ inverse_tcp
    rotation = wp.mat33d(
        goal[0, 0],
        goal[0, 1],
        goal[0, 2],
        goal[1, 0],
        goal[1, 1],
        goal[1, 2],
        goal[2, 0],
        goal[2, 1],
        goal[2, 2],
    )
    gram = wp.transpose(rotation) @ rotation
    for j in range(3):
        for k in range(3):
            expected = wp.float64(0)
            if j == k:
                expected = wp.float64(1)
            valid = valid and wp.abs(gram[j, k] - expected) <= wp.float64(1e-5)
        valid = valid and wp.abs(target[3, j]) <= wp.float64(1e-6)
    return (
        valid
        and wp.determinant(rotation) > wp.float64(0)
        and wp.abs(target[3, 3] - wp.float64(1)) <= wp.float64(1e-6)
    )


@wp.kernel
def srs_continuous_path(
    model: _Model,
    axes: wp.array(dtype=wp.vec3d),
    offsets: wp.array(dtype=wp.vec3d),
    targets: wp.array2d(dtype=wp.mat44d),
    tcp: wp.mat44d,
    inverse_tcp: wp.mat44d,
    initial: wp.array2d(dtype=wp.float32),
    lower: wp.array(dtype=wp.float64),
    upper: wp.array(dtype=wp.float64),
    weights: wp.array(dtype=wp.float32),
    samples: int,
    position_tolerance: wp.float64,
    rotation_tolerance: wp.float64,
    max_joint_step: wp.float64,
    result_valid: wp.array2d(dtype=wp.bool),
    result: wp.array3d(dtype=wp.float32),
):
    """Evaluate the ordinary seed-local candidate pool using each prior result.

    Launch one thread per independent path. A block size of one spreads small
    path batches over GPU multiprocessors. Scratch is fixed local storage;
    outputs belong to the caller and no cross-call state is retained.
    """
    path = wp.tid()
    previous, lo, hi, weight = vec7d(), vec7d(), vec7d(), vec7d()
    seed_finite = bool(True)
    for j in range(7):
        lo[j] = lower[j]
        hi[j] = upper[j]
        weight[j] = wp.float64(weights[j])
        value = wp.float64(initial[path, j])
        seed_finite = seed_finite and wp.isfinite(value)
        if not wp.isfinite(value):
            value = (lo[j] + hi[j]) * wp.float64(0.5)
        previous[j] = wp.clamp(value, lo[j], hi[j])
    for frame in range(targets.shape[1]):
        target = targets[path, frame]
        best = previous
        best_cost = wp.float64(wp.inf)
        found = bool(False)
        if seed_finite and _srs_valid_target(target, inverse_tcp):
            goal = target @ inverse_tcp
            local_candidates, local_flags = _srs_search_candidates(
                axes,
                offsets,
                model.home,
                model.shoulder,
                model.wrist,
                goal,
                position_tolerance,
                previous,
                weight,
            )
            for sample in range(samples + 2):
                candidates, valid = mat87d(), vec8i()
                phi = wp.float64(0)
                usable = bool(True)
                if sample == samples + 1:
                    candidates, valid = local_candidates, local_flags
                    usable = False
                elif sample == 0:
                    phi, usable = seed_swivel(
                        axes,
                        offsets,
                        model.home,
                        model.shoulder,
                        model.wrist,
                        previous,
                    )
                else:
                    phi = -wp.float64(wp.pi) + wp.float64(2.0 * wp.pi) * wp.float64(
                        sample - 1
                    ) / wp.float64(samples)
                if usable:
                    candidates, valid = solve_srs(
                        axes,
                        offsets,
                        model.home,
                        model.shoulder,
                        model.wrist,
                        goal,
                        phi,
                        previous,
                    )
                for branch in range(8):
                    if valid[branch] != 0:
                        raw = vec7d()
                        for j in range(7):
                            raw[j] = candidates[branch, j]
                        qpos, accepted = _validate_candidate(
                            model,
                            raw,
                            target,
                            tcp,
                            previous,
                            lo,
                            hi,
                            position_tolerance,
                            rotation_tolerance,
                            max_joint_step,
                        )
                        if accepted:
                            cost = wp.float64(0)
                            for j in range(7):
                                difference = (qpos[j] - previous[j]) * weight[j]
                                cost += difference * difference
                            if cost < best_cost:
                                best_cost, best, found = cost, qpos, True
        result_valid[path, frame] = found
        previous = best
        for j in range(7):
            result[path, frame, j] = wp.float32(previous[j])
