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
"""Fused bounded DLS correction for general offset seven-revolute chains.

Each thread owns one numerical search, including FK, Jacobian and all four
backtracking trials. Double precision protects the tiny normal equations;
public joints remain float32 and are independently verified by the host.
"""

from __future__ import annotations

import warp as wp

__all__ = []

_vec6 = wp.types.vector(length=6, dtype=wp.float64)
_vec7 = wp.types.vector(length=7, dtype=wp.float64)
_mat73 = wp.types.matrix(shape=(7, 3), dtype=wp.float64)
_mat67 = wp.types.matrix(shape=(6, 7), dtype=wp.float64)
_mat66 = wp.types.matrix(shape=(6, 6), dtype=wp.float64)

wp.set_module_options({"enable_backward": False})


@wp.func
def _rotation(axis: wp.vec3d, angle: wp.float64) -> wp.mat44d:
    x, y, z = axis[0], axis[1], axis[2]
    c = wp.cos(angle)
    s = wp.sin(angle)
    t = wp.float64(1.0) - c
    return wp.mat44d(
        c + x * x * t,
        x * y * t - z * s,
        x * z * t + y * s,
        wp.float64(0.0),
        y * x * t + z * s,
        c + y * y * t,
        y * z * t - x * s,
        wp.float64(0.0),
        z * x * t - y * s,
        z * y * t + x * s,
        c + z * z * t,
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(1.0),
    )


@wp.func
def _fk(
    qpos: _vec7,
    origins: wp.array(dtype=wp.mat44d),
    axes: wp.array(dtype=wp.vec3d),
    tcp: wp.mat44d,
) -> wp.mat44d:
    pose = wp.identity(n=4, dtype=wp.float64)
    for joint in range(7):
        pose = pose @ origins[joint] @ _rotation(axes[joint], qpos[joint])
    return pose @ origins[7] @ tcp


@wp.func
def _jacobian(
    qpos: _vec7,
    origins: wp.array(dtype=wp.mat44d),
    axes: wp.array(dtype=wp.vec3d),
    tcp: wp.mat44d,
) -> _mat67:
    pose = wp.identity(n=4, dtype=wp.float64)
    positions = _mat73()
    directions = _mat73()
    for joint in range(7):
        pose = pose @ origins[joint]
        axis = axes[joint]
        for row in range(3):
            positions[joint, row] = pose[row, 3]
            directions[joint, row] = (
                pose[row, 0] * axis[0] + pose[row, 1] * axis[1] + pose[row, 2] * axis[2]
            )
        pose = pose @ _rotation(axis, qpos[joint])
    pose = pose @ origins[7] @ tcp
    jac = _mat67()
    for joint in range(7):
        axis = wp.vec3d(
            directions[joint, 0], directions[joint, 1], directions[joint, 2]
        )
        offset = wp.vec3d(
            pose[0, 3] - positions[joint, 0],
            pose[1, 3] - positions[joint, 1],
            pose[2, 3] - positions[joint, 2],
        )
        linear = wp.cross(axis, offset)
        for row in range(3):
            jac[row, joint] = linear[row]
            jac[row + 3, joint] = axis[row]
    return jac


@wp.func
def _error(target: wp.mat44d, actual: wp.mat44d) -> _vec6:
    rotation = wp.mat33d()
    for i in range(3):
        for j in range(3):
            rotation[i, j] = (
                target[i, 0] * actual[j, 0]
                + target[i, 1] * actual[j, 1]
                + target[i, 2] * actual[j, 2]
            )
    a, b, c = rotation[0, 0], rotation[1, 1], rotation[2, 2]
    parts = wp.vec4d(
        wp.float64(1.0) + a + b + c,
        wp.float64(1.0) + a - b - c,
        wp.float64(1.0) - a + b - c,
        wp.float64(1.0) - a - b + c,
    )
    index = int(0)
    for i in range(1, 4):
        if parts[i] > parts[index]:
            index = i
    # Match Torch's largest-component quaternion convention, including the
    # sign of w; configuration searches keep their original local policy.
    quat = wp.vec4d()
    if index == 0:
        quat = wp.vec4d(
            parts[0],
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        )
    elif index == 1:
        quat = wp.vec4d(
            rotation[2, 1] - rotation[1, 2],
            parts[1],
            rotation[1, 0] + rotation[0, 1],
            rotation[0, 2] + rotation[2, 0],
        )
    elif index == 2:
        quat = wp.vec4d(
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] + rotation[0, 1],
            parts[2],
            rotation[2, 1] + rotation[1, 2],
        )
    else:
        quat = wp.vec4d(
            rotation[1, 0] - rotation[0, 1],
            rotation[2, 0] + rotation[0, 2],
            rotation[2, 1] + rotation[1, 2],
            parts[3],
        )
    quat = quat / (wp.float64(2.0) * wp.sqrt(wp.max(parts[index], wp.float64(1e-30))))
    vector = wp.vec3d(quat[1], quat[2], quat[3])
    norm = wp.length(vector)
    factor = wp.float64(2.0)
    if norm > wp.float64(1e-12):
        factor = wp.float64(2.0) * wp.atan2(norm, quat[0]) / norm
    return _vec6(
        target[0, 3] - actual[0, 3],
        target[1, 3] - actual[1, 3],
        target[2, 3] - actual[2, 3],
        factor * quat[1],
        factor * quat[2],
        factor * quat[3],
    )


@wp.func
def _within(
    error: _vec6, position_tolerance: wp.float64, rotation_tolerance: wp.float64
) -> bool:
    position = error[0] * error[0] + error[1] * error[1] + error[2] * error[2]
    angle = error[3] * error[3] + error[4] * error[4] + error[5] * error[5]
    return (
        wp.isfinite(position)
        and wp.isfinite(angle)
        and position <= position_tolerance * position_tolerance
        and angle <= rotation_tolerance * rotation_tolerance
    )


@wp.func
def _step(
    jac: _mat67, error: _vec6, damping: wp.float64, max_step: wp.float64
) -> _vec7:
    lower = _mat66()
    good = bool(True)
    # Cholesky of J J^T + lambda^2 I; no global solver workspace or host check.
    for i in range(6):
        for j in range(i + 1):
            value = wp.float64(0.0)
            for k in range(7):
                value += jac[i, k] * jac[j, k]
            if i == j:
                value += damping * damping
            for k in range(j):
                value -= lower[i, k] * lower[j, k]
            if i == j:
                if value <= wp.float64(0.0) or not wp.isfinite(value):
                    good = False
                lower[i, j] = wp.sqrt(wp.max(value, wp.float64(1e-30)))
            else:
                lower[i, j] = value / lower[j, j]
    forward = _vec6()
    dual = _vec6()
    for i in range(6):
        value = error[i]
        for j in range(i):
            value -= lower[i, j] * forward[j]
        forward[i] = value / lower[i, i]
    for reverse in range(6):
        i = 5 - reverse
        value = forward[i]
        for j in range(i + 1, 6):
            value -= lower[j, i] * dual[j]
        dual[i] = value / lower[i, i]
    result = _vec7()
    maximum = max_step
    if good:
        for joint in range(7):
            for i in range(6):
                result[joint] += jac[i, joint] * dual[i]
            maximum = wp.max(maximum, wp.abs(result[joint]))
    return result * (max_step / maximum)


@wp.kernel
def solve(
    targets: wp.array(dtype=wp.mat44f),
    seeds: wp.array2d(dtype=wp.float32),
    lower: wp.array(dtype=wp.float32),
    upper: wp.array(dtype=wp.float32),
    origins: wp.array(dtype=wp.mat44d),
    axes: wp.array(dtype=wp.vec3d),
    tcp: wp.array(dtype=wp.mat44d),
    max_iterations: int,
    damping: wp.float64,
    max_step: wp.float64,
    position_tolerance: wp.float64,
    rotation_tolerance: wp.float64,
    valid: wp.array(dtype=wp.int32),
    joints: wp.array2d(dtype=wp.float32),
):
    tid = wp.tid()
    qpos = _vec7()
    target = wp.mat44d(targets[tid])
    for joint in range(7):
        qpos[joint] = wp.float64(
            wp.clamp(seeds[tid, joint], lower[joint], upper[joint])
        )
    error = _error(target, _fk(qpos, origins, axes, tcp[0]))
    for iteration in range(max_iterations):
        if _within(error, position_tolerance, rotation_tolerance):
            break
        jac = _jacobian(qpos, origins, axes, tcp[0])
        step = _step(jac, error, damping, max_step)
        best = qpos
        best_error = error
        best_cost = wp.dot(error, error)
        scale = wp.float64(1.0)
        improved = bool(False)
        for trial_index in range(4):
            trial = qpos + scale * step
            for joint in range(7):
                trial[joint] = wp.clamp(
                    trial[joint], wp.float64(lower[joint]), wp.float64(upper[joint])
                )
            trial_error = _error(target, _fk(trial, origins, axes, tcp[0]))
            cost = wp.dot(trial_error, trial_error)
            if wp.isfinite(cost) and cost < best_cost:
                best, best_error, best_cost = trial, trial_error, cost
                improved = True
            scale *= wp.float64(0.5)
        if not improved:
            break
        qpos, error = best, best_error
    valid[tid] = int(_within(error, position_tolerance, rotation_tolerance))
    for joint in range(7):
        joints[tid, joint] = wp.float32(qpos[joint])
