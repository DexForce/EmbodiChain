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

"""Independent parameterized POE RR kernels; no simulation or CPU solve calls."""

from __future__ import annotations

import warp as wp

from .ssik_rr import _v12, _v14, _v6, _v8, _v9, _projective_angle, _rigid_inverse

wp.set_module_options({"enable_backward": False, "max_unroll": 0})

__all__: list[str] = []


@wp.struct
class _ParametricData:
    p_affine: wp.array4d(dtype=wp.float64)
    factor_affine: wp.array4d(dtype=wp.float64)
    mono_factors: wp.array2d(dtype=wp.int32)
    cell_offsets: wp.array(dtype=wp.int32)
    coefficient_monomials: wp.array(dtype=wp.int32)
    coefficient_values: wp.array(dtype=wp.float64)
    global_angles: wp.array(dtype=wp.float64)
    local_offsets: wp.array(dtype=wp.float64)
    model_frames: wp.array2d(dtype=wp.float64)
    model_axes: wp.array2d(dtype=wp.float64)
    w: wp.array2d(dtype=wp.float64)
    pre_inverse: wp.mat44d
    source_to_artifact: wp.mat44d


@wp.func
def _rotation_z(angle: wp.float64) -> wp.mat44d:
    c, s = wp.cos(angle), wp.sin(angle)
    return wp.mat44d(
        c,
        -s,
        wp.float64(0),
        wp.float64(0),
        s,
        c,
        wp.float64(0),
        wp.float64(0),
        wp.float64(0),
        wp.float64(0),
        wp.float64(1),
        wp.float64(0),
        wp.float64(0),
        wp.float64(0),
        wp.float64(0),
        wp.float64(1),
    )


@wp.func
def _factor_at(data: _ParametricData, phi: wp.float64, joint: int) -> wp.mat44d:
    result = wp.mat44d()
    sine, cosine = wp.sin(phi), wp.cos(phi)
    for i in range(4):
        for j in range(4):
            result[i, j] = (
                data.factor_affine[0, joint, i, j]
                + sine * data.factor_affine[1, joint, i, j]
                + cosine * data.factor_affine[2, joint, i, j]
            )
    return result


@wp.kernel
def parametric_refine_roots(
    pencil: wp.array4d(dtype=wp.float64),
    roots: wp.array3d(dtype=wp.float64),
    root_valid: wp.array3d(dtype=wp.int32),
    matrix: wp.array4d(dtype=wp.float64),
    derivative: wp.array4d(dtype=wp.float64),
):
    """Refine existing real algebraic roots against their original pencil.

    Shift-invert linearization can amplify roundoff in a non-normal pencil.
    Three bounded determinant Newton steps use the original quadratic matrix,
    with simultaneous LU differentiation computing ``d(log(det(M)))/dx``.
    Both 12-by-12 workspaces are global and reused by back substitution; no
    joint-space or Cartesian IK iterate is introduced and masks stay unchanged.
    """
    target, sample, branch = wp.tid()
    slot = sample * 24 + branch
    if root_valid[target, sample, branch] == 0:
        return
    original = roots[target, sample, branch]
    if not wp.isfinite(original):
        return
    reciprocal = wp.abs(original) > wp.float64(1.0)
    initial = original
    if reciprocal:
        initial = wp.float64(1.0) / original
    value = initial
    for iteration in range(3):
        for i in range(12):
            scale = wp.float64(0.0)
            for j in range(12):
                a = pencil[target, sample, i, j]
                b = pencil[target, sample, i + 12, j]
                c = pencil[target, sample, i + 24, j]
                if reciprocal:
                    temp = a
                    a = c
                    c = temp
                entry = (a * value + b) * value + c
                matrix[target, slot, i, j] = entry
                derivative[target, slot, i, j] = wp.float64(2.0) * a * value + b
                scale = wp.max(scale, wp.abs(entry))
            if scale > wp.float64(0.0):
                for j in range(12):
                    matrix[target, slot, i, j] /= scale
                    derivative[target, slot, i, j] /= scale
        safe = bool(True)
        for k in range(12):
            pivot = int(k)
            for i in range(k + 1, 12):
                if wp.abs(matrix[target, slot, i, k]) > wp.abs(
                    matrix[target, slot, pivot, k]
                ):
                    pivot = i
            diagonal = matrix[target, slot, pivot, k]
            if not wp.isfinite(diagonal) or wp.abs(diagonal) <= wp.float64(1e-28):
                safe = False
                break
            if pivot != k:
                for j in range(k, 12):
                    temp = matrix[target, slot, k, j]
                    matrix[target, slot, k, j] = matrix[target, slot, pivot, j]
                    matrix[target, slot, pivot, j] = temp
                    temp = derivative[target, slot, k, j]
                    derivative[target, slot, k, j] = derivative[target, slot, pivot, j]
                    derivative[target, slot, pivot, j] = temp
            diagonal_derivative = derivative[target, slot, k, k]
            for i in range(k + 1, 12):
                factor = matrix[target, slot, i, k] / diagonal
                factor_derivative = (
                    derivative[target, slot, i, k] - factor * diagonal_derivative
                ) / diagonal
                for j in range(k + 1, 12):
                    derivative[target, slot, i, j] -= (
                        factor_derivative * matrix[target, slot, k, j]
                        + factor * derivative[target, slot, k, j]
                    )
                    matrix[target, slot, i, j] -= factor * matrix[target, slot, k, j]
                matrix[target, slot, i, k] = wp.float64(0.0)
                derivative[target, slot, i, k] = wp.float64(0.0)
        if not safe:
            break
        log_derivative = wp.float64(0.0)
        for k in range(12):
            log_derivative += (
                derivative[target, slot, k, k] / matrix[target, slot, k, k]
            )
        if not wp.isfinite(log_derivative) or wp.abs(log_derivative) <= wp.float64(
            1e-28
        ):
            break
        delta = wp.float64(1.0) / log_derivative
        proposed = value - delta
        # A chart correction is deliberately local to the existing root. An
        # unsafe correction preserves the last safe value, never adds a root.
        if (
            not wp.isfinite(delta)
            or not wp.isfinite(proposed)
            or wp.abs(delta) > wp.float64(0.01)
            or wp.abs(proposed - initial) > wp.float64(0.01)
        ):
            break
        value = proposed
    refined = value
    if reciprocal:
        refined = original
        if wp.abs(value) > wp.float64(1e-30):
            refined = wp.float64(1.0) / value
    if wp.isfinite(refined):
        roots[target, sample, branch] = refined


@wp.kernel
def parametric_back_substitute(
    data: _ParametricData,
    goals: wp.array(dtype=wp.mat44d),
    p_target: wp.array3d(dtype=wp.float64),
    q_inverse: wp.array3d(dtype=wp.float64),
    pencil: wp.array4d(dtype=wp.float64),
    roots: wp.array3d(dtype=wp.float64),
    root_valid: wp.array3d(dtype=wp.int32),
    angles: wp.array2d(dtype=wp.float64),
    matrix: wp.array4d(dtype=wp.float64),
    vectors: wp.array4d(dtype=wp.float64),
    candidates: wp.array3d(dtype=wp.float64),
    validity: wp.array2d(dtype=wp.int32),
):
    target, sample, branch = wp.tid()
    slot = sample * 24 + branch
    validity[target, slot] = 0
    for j in range(7):
        candidates[target, slot, j] = wp.float64(0.0)
    if root_valid[target, sample, branch] != 0:
        x = roots[target, sample, branch]
        # Scale projectively; x may encode a pole as 1e30.
        norm = wp.float64(1.0) + x * x
        a_scale, b_scale, c_scale = x * x / norm, x / norm, wp.float64(1.0) / norm
        v = _v12()
        for j in range(12):
            v[j] = wp.sin(wp.float64(j + 1) * wp.float64(1.23456789))
        for iteration in range(3):
            for i in range(12):
                vectors[target, slot, 0, i] = v[i]
                for j in range(12):
                    matrix[target, slot, i, j] = (
                        a_scale * pencil[target, sample, i, j]
                        + b_scale * pencil[target, sample, i + 12, j]
                        + c_scale * pencil[target, sample, i + 24, j]
                    )
            for k in range(12):
                pivot = int(k)
                for i in range(k + 1, 12):
                    if wp.abs(matrix[target, slot, i, k]) > wp.abs(
                        matrix[target, slot, pivot, k]
                    ):
                        pivot = i
                if pivot != k:
                    for j in range(k, 12):
                        temp = matrix[target, slot, k, j]
                        matrix[target, slot, k, j] = matrix[target, slot, pivot, j]
                        matrix[target, slot, pivot, j] = temp
                    temp = vectors[target, slot, 0, k]
                    vectors[target, slot, 0, k] = vectors[target, slot, 0, pivot]
                    vectors[target, slot, 0, pivot] = temp
                diagonal = matrix[target, slot, k, k]
                if wp.abs(diagonal) < wp.float64(1e-28):
                    diagonal = wp.float64(1e-28)
                for i in range(k + 1, 12):
                    factor = matrix[target, slot, i, k] / diagonal
                    for j in range(k + 1, 12):
                        matrix[target, slot, i, j] -= (
                            factor * matrix[target, slot, k, j]
                        )
                    vectors[target, slot, 0, i] -= factor * vectors[target, slot, 0, k]
            for offset in range(12):
                i = 11 - offset
                value = vectors[target, slot, 0, i]
                for j in range(i + 1, 12):
                    value -= matrix[target, slot, i, j] * vectors[target, slot, 1, j]
                diagonal = matrix[target, slot, i, i]
                if wp.abs(diagonal) < wp.float64(1e-28):
                    diagonal = wp.float64(1e-28)
                vectors[target, slot, 1, i] = value / diagonal
            scale = wp.float64(0.0)
            for j in range(12):
                scale = wp.max(scale, wp.abs(vectors[target, slot, 1, j]))
            if scale > wp.float64(0.0):
                for j in range(12):
                    v[j] = vectors[target, slot, 1, j] / scale
        phi = angles[target, sample]
        phase_s, phase_c = wp.sin(phi), wp.cos(phi)
        q_lin = wp.float64(2.0) * wp.atan(x)
        q_left0, q_left1 = _projective_angle(v, 0), _projective_angle(v, 1)
        sl0, cl0, sl1, cl1 = (
            wp.sin(q_left0),
            wp.cos(q_left0),
            wp.sin(q_left1),
            wp.cos(q_left1),
        )
        left = _v9(
            sl0 * sl1,
            sl0 * cl1,
            cl0 * sl1,
            cl0 * cl1,
            sl0,
            cl0,
            sl1,
            cl1,
            wp.float64(1.0),
        )
        rhs = _v14()
        for i in range(14):
            for j in range(9):
                ps = (
                    data.p_affine[0, 0, i, j]
                    + phase_s * data.p_affine[0, 1, i, j]
                    + phase_c * data.p_affine[0, 2, i, j]
                )
                pc = (
                    data.p_affine[1, 0, i, j]
                    + phase_s * data.p_affine[1, 1, i, j]
                    + phase_c * data.p_affine[1, 2, i, j]
                )
                po = (
                    p_target[target, i, j]
                    + phase_s * data.p_affine[2, 1, i, j]
                    + phase_c * data.p_affine[2, 2, i, j]
                )
                rhs[i] += (ps * wp.sin(q_lin) + pc * wp.cos(q_lin) + po) * left[j]
        right = _v8()
        for i in range(8):
            for j in range(14):
                right[i] += q_inverse[target, i, j] * rhs[j]
        q = _v6()
        q[1] = q_lin
        q[2] = q_left0
        q[3] = q_left1
        q[0] = wp.atan2(right[4], right[5])
        q[5] = wp.atan2(right[6], right[7])
        before = wp.identity(n=4, dtype=wp.float64)
        for j in range(4):
            before = before @ _rotation_z(q[j]) @ _factor_at(data, phi, j)
        after = _rotation_z(q[5]) @ _factor_at(data, phi, 5)
        residual = (
            _rigid_inverse(before)
            @ goals[target]
            @ _rigid_inverse(after)
            @ _rigid_inverse(_factor_at(data, phi, 4))
        )
        q[4] = wp.atan2(residual[1, 0], residual[0, 0])
        finite = bool(True)
        for j in range(6):
            index = int(j)
            if j >= 2:
                index += 1
            value = q[j]
            candidates[target, slot, index] = value
            finite = finite and wp.isfinite(value)
        candidates[target, slot, 2] = phi
        validity[target, slot] = wp.int32(finite)
