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

"""Scalar Warp scheduling of the parameterized POE RR algebraic pipeline.

The coefficients, conditional samples, shifted pencil and projective recovery
are shared with the cooperative CUDA specialization. CPU threads own separate
targets or lock samples; scratch resides in bounded arrays. No LAPACK, SciPy,
NumPy target processing or numerical pose solver participates in these kernels.
"""

from __future__ import annotations

import warp as wp

from .ssik_7r import _rotation
from .ssik_rr import _eigenvalues_global
from .ssik_rr_parametric import _ParametricData

wp.set_module_options({"enable_backward": False, "max_unroll": 0})

__all__: list[str] = []

_mat148 = wp.types.matrix(shape=(14, 8), dtype=wp.float64)
_mat1414 = wp.types.matrix(shape=(14, 14), dtype=wp.float64)
_vec14 = wp.types.vector(length=14, dtype=wp.float64)
_mat73 = wp.types.matrix(shape=(7, 3), dtype=wp.float64)
_mat67 = wp.types.matrix(shape=(6, 7), dtype=wp.float64)
_mat66 = wp.types.matrix(shape=(6, 6), dtype=wp.float64)
_mat62 = wp.types.matrix(shape=(6, 2), dtype=wp.float64)
_vec6 = wp.types.vector(length=6, dtype=wp.float64)
_mat69 = wp.types.matrix(shape=(6, 9), dtype=wp.float64)


@wp.kernel
def parametric_target_prepare_cpu(
    data: _ParametricData,
    targets: wp.array(dtype=wp.mat44d),
    goals: wp.array(dtype=wp.mat44d),
    p_target: wp.array3d(dtype=wp.float64),
    q_basis: wp.array3d(dtype=wp.float64),
    q_inverse: wp.array3d(dtype=wp.float64),
    target_valid: wp.array(dtype=wp.int32),
):
    target = wp.tid()
    original = targets[target]
    rotation = wp.mat33d()
    ok = bool(True)
    for i in range(4):
        for j in range(4):
            ok = ok and wp.isfinite(original[i, j])
            if i < 3 and j < 3:
                rotation[i, j] = original[i, j]
    gram = wp.transpose(rotation) @ rotation
    for i in range(3):
        for j in range(3):
            ok = ok and wp.abs(gram[i, j] - wp.float64(i == j)) <= wp.float64(1e-5)
    ok = ok and wp.determinant(rotation) > wp.float64(0.0)
    for j in range(4):
        ok = ok and wp.abs(original[3, j] - wp.float64(j == 3)) <= wp.float64(1e-6)
    if not ok:
        target_valid[target] = 0
        return
    identity = wp.identity(n=3, dtype=wp.float64)
    for step in range(2):
        gram = wp.transpose(rotation) @ rotation
        rotation = rotation @ (wp.float64(1.5) * identity - wp.float64(0.5) * gram)
    conditioned = original
    for i in range(3):
        for j in range(3):
            conditioned[i, j] = rotation[i, j]
    goal = data.pre_inverse @ conditioned @ data.source_to_artifact
    goals[target] = goal
    q = _mat148()
    qt = _mat1414()
    for i in range(14):
        qt[i, i] = wp.float64(1.0)
    for cell in range(238):
        total = wp.float64(0.0)
        for k in range(data.cell_offsets[cell], data.cell_offsets[cell + 1]):
            value = data.coefficient_values[k]
            mono = data.coefficient_monomials[k]
            for f in range(4):
                variable = data.mono_factors[mono, f]
                if variable >= 0:
                    value *= goal[variable // 4, variable % 4]
            total += value
        if cell < 126:
            p_target[target, cell // 9, cell % 9] = total
        else:
            q[(cell - 126) // 8, (cell - 126) % 8] = total
    for k in range(8):
        v = _vec14()
        norm = wp.float64(0.0)
        for i in range(k, 14):
            v[i] = q[i, k]
            norm += v[i] * v[i]
        norm = wp.sqrt(norm)
        beta = wp.float64(0.0)
        if norm < wp.float64(1e-14):
            ok = False
        else:
            if v[k] >= wp.float64(0.0):
                norm = -norm
            v[k] -= norm
            vv = wp.float64(0.0)
            for i in range(k, 14):
                vv += v[i] * v[i]
            if vv > wp.float64(1e-28):
                beta = wp.float64(2.0) / vv
        for j in range(k, 8):
            dot = wp.float64(0.0)
            for i in range(k, 14):
                dot += v[i] * q[i, j]
            for i in range(k, 14):
                q[i, j] -= beta * v[i] * dot
        for j in range(14):
            dot = wp.float64(0.0)
            for i in range(k, 14):
                dot += v[i] * qt[i, j]
            for i in range(k, 14):
                qt[i, j] -= beta * v[i] * dot
    for j in range(14):
        for offset in range(8):
            i = 7 - offset
            value = qt[i, j]
            for k in range(i + 1, 8):
                value -= q[i, k] * q_inverse[target, k, j]
            diagonal = q[i, i]
            if wp.abs(diagonal) <= wp.float64(1e-20):
                diagonal = wp.float64(1e-20)
            q_inverse[target, i, j] = value / diagonal
        for i in range(6):
            q_basis[target, i, j] = qt[i + 8, j]
    target_valid[target] = wp.int32(ok)


@wp.kernel
def parametric_lock_predictor_cpu(
    data: _ParametricData,
    targets: wp.array(dtype=wp.mat44d),
    seed: wp.array2d(dtype=wp.float64),
    lower: wp.array(dtype=wp.float64),
    upper: wp.array(dtype=wp.float64),
    max_step: wp.float64,
    predictions: wp.array2d(dtype=wp.float64),
):
    target = wp.tid()
    rotation = wp.identity(n=3, dtype=wp.float64)
    position = wp.vec3d()
    origins = _mat73()
    axes = _mat73()
    for joint in range(8):
        fixed = wp.mat33d()
        translation = wp.vec3d()
        for i in range(3):
            translation[i] = data.model_frames[joint, i * 4 + 3]
            for j in range(3):
                fixed[i, j] = data.model_frames[joint, i * 4 + j]
        position += rotation @ translation
        rotation = rotation @ fixed
        if joint < 7:
            axis = wp.vec3d(
                data.model_axes[joint, 0],
                data.model_axes[joint, 1],
                data.model_axes[joint, 2],
            )
            world = rotation @ axis
            for i in range(3):
                origins[joint, i] = position[i]
                axes[joint, i] = world[i]
            magnitude = wp.length(axis)
            rotation = rotation @ _rotation(
                axis / magnitude, seed[target, joint] * magnitude
            )
    error = _vec6()
    goal = targets[target]
    relative = wp.mat33d()
    for i in range(3):
        error[i] = goal[i, 3] - position[i]
        for j in range(3):
            for k in range(3):
                relative[i, j] += goal[i, k] * rotation[j, k]
    for i in range(3):
        a, b = (i + 1) % 3, (i + 2) % 3
        error[i + 3] = wp.float64(0.5) * (relative[b, a] - relative[a, b])
    jacobian = _mat67()
    for joint in range(7):
        axis = wp.vec3d(axes[joint, 0], axes[joint, 1], axes[joint, 2])
        delta = position - wp.vec3d(
            origins[joint, 0], origins[joint, 1], origins[joint, 2]
        )
        linear = wp.cross(axis, delta)
        for i in range(3):
            jacobian[i, joint] = linear[i]
            jacobian[i + 3, joint] = axis[i]
    prediction0, prediction1 = wp.float64(0.0), wp.float64(0.0)
    solution = _mat62()
    usable = bool(False)
    fallback = int(0)
    for attempt in range(2):
        matrix = _mat66()
        rhs = _mat62()
        usable = bool(True)
        for i in range(6):
            rhs[i, 0] = error[i]
            rhs[i, 1] = jacobian[i, 2]
            for j in range(6):
                if attempt == 0:
                    index = int(j)
                    if j >= 2:
                        index += 1
                    matrix[i, j] = jacobian[i, index]
                else:
                    matrix[i, j] = wp.float64(1e-10) * wp.float64(i == j)
                    for k in range(7):
                        matrix[i, j] += jacobian[i, k] * jacobian[j, k]
        for k in range(6):
            pivot = int(k)
            for i in range(k + 1, 6):
                if wp.abs(matrix[i, k]) > wp.abs(matrix[pivot, k]):
                    pivot = i
            threshold = wp.float64(1e-12)
            if attempt != 0:
                threshold = wp.float64(1e-20)
            usable = usable and wp.abs(matrix[pivot, k]) >= threshold
            if pivot != k:
                for j in range(6):
                    temp = matrix[k, j]
                    matrix[k, j] = matrix[pivot, j]
                    matrix[pivot, j] = temp
                for j in range(2):
                    temp = rhs[k, j]
                    rhs[k, j] = rhs[pivot, j]
                    rhs[pivot, j] = temp
            diagonal = matrix[k, k]
            if wp.abs(diagonal) <= wp.float64(1e-25):
                diagonal = wp.float64(1e-25)
            for i in range(k + 1, 6):
                factor = matrix[i, k] / diagonal
                for j in range(k + 1, 6):
                    matrix[i, j] -= factor * matrix[k, j]
                for j in range(2):
                    rhs[i, j] -= factor * rhs[k, j]
        for j in range(2):
            for offset in range(6):
                i = 5 - offset
                value = rhs[i, j]
                for k in range(i + 1, 6):
                    value -= matrix[i, k] * solution[k, j]
                diagonal = matrix[i, i]
                if wp.abs(diagonal) <= wp.float64(1e-25):
                    diagonal = wp.float64(1e-25)
                solution[i, j] = value / diagonal
        if attempt == 1 or usable:
            fallback = attempt
            break
    trust = wp.min(max_step, wp.float64(0.2))
    first = wp.max(-trust, lower[2] - seed[target, 2])
    last = wp.min(trust, upper[2] - seed[target, 2])
    if first <= last and usable:
        numerator, denominator = wp.float64(0.0), wp.float64(1.0)
        for i in range(6):
            if fallback != 0:
                numerator += jacobian[i, 2] * solution[i, 0]
            else:
                numerator += solution[i, 0] * solution[i, 1]
                denominator += solution[i, 1] * solution[i, 1]
        unconstrained = numerator / denominator
        prediction0 = wp.clamp(unconstrained, first, last)
        prediction1 = wp.float64(0.5) * prediction0
        if fallback == 0:
            lo, hi = first, last
            for i in range(6):
                joint = int(i)
                if i >= 2:
                    joint += 1
                low = wp.max(-trust, lower[joint] - seed[target, joint])
                high = wp.min(trust, upper[joint] - seed[target, joint])
                u, v = solution[i, 0], solution[i, 1]
                if wp.abs(v) > wp.float64(1e-14):
                    bound_first, bound_last = (u - low) / v, (u - high) / v
                    lo = wp.max(lo, wp.min(bound_first, bound_last))
                    hi = wp.min(hi, wp.max(bound_first, bound_last))
                elif u < low or u > high:
                    hi = lo - wp.float64(1.0)
            if lo <= hi:
                prediction0 = wp.clamp(unconstrained, lo, hi)
                prediction1 = wp.float64(0.5) * (lo + hi)
    predictions[target, 0] = prediction0
    predictions[target, 1] = prediction1


@wp.kernel
def parametric_pencil_roots_cpu(
    data: _ParametricData,
    p_target: wp.array3d(dtype=wp.float64),
    q_basis: wp.array3d(dtype=wp.float64),
    target_valid: wp.array(dtype=wp.int32),
    seed: wp.array2d(dtype=wp.float64),
    lower: wp.array(dtype=wp.float64),
    upper: wp.array(dtype=wp.float64),
    max_step: wp.float64,
    predictions: wp.array2d(dtype=wp.float64),
    angles: wp.array2d(dtype=wp.float64),
    pencil: wp.array4d(dtype=wp.float64),
    lhs: wp.array4d(dtype=wp.float64),
    rhs: wp.array4d(dtype=wp.float64),
    real: wp.array4d(dtype=wp.float64),
    imag: wp.array4d(dtype=wp.float64),
    vectors: wp.array4d(dtype=wp.float64),
    roots: wp.array3d(dtype=wp.float64),
    root_imag: wp.array3d(dtype=wp.float64),
    valid: wp.array3d(dtype=wp.int32),
):
    target, sample = wp.tid()
    initial, lo, hi = seed[target, 2], lower[2], upper[2]
    raw = wp.float64(0.0)
    if sample < 16:
        raw = data.global_angles[sample]
    else:
        phase_offset = wp.float64(0.0)
        if sample < 23:
            phase_offset = data.local_offsets[sample - 16]
        else:
            phase_offset = predictions[target, sample - 23]
        raw = wp.clamp(initial + phase_offset, lo, hi)
    period = wp.float64(2.0 * wp.pi)
    first = wp.ceil((lo - wp.float64(1e-6) - raw) / period)
    last = wp.floor((hi + wp.float64(1e-6) - raw) / period)
    turns = wp.clamp(wp.floor((initial - raw) / period + wp.float64(0.5)), first, last)
    phi = wp.clamp(raw + period * turns, lo, hi)
    angles[target, sample] = phi
    ok = (
        target_valid[target] != 0
        and first <= last
        and wp.isfinite(phi)
        and wp.abs(wp.float64(wp.float32(phi)) - initial) <= max_step
    )
    if not ok:
        for branch in range(24):
            valid[target, sample, branch] = 0
        return
    sine, cosine = wp.sin(phi), wp.cos(phi)
    e_a, e_b, e_c = _mat69(), _mat69(), _mat69()
    for i in range(6):
        for j in range(9):
            es, ec, eo = wp.float64(0.0), wp.float64(0.0), wp.float64(0.0)
            for k in range(14):
                n = q_basis[target, i, k]
                es += n * (
                    data.p_affine[0, 0, k, j]
                    + sine * data.p_affine[0, 1, k, j]
                    + cosine * data.p_affine[0, 2, k, j]
                )
                ec += n * (
                    data.p_affine[1, 0, k, j]
                    + sine * data.p_affine[1, 1, k, j]
                    + cosine * data.p_affine[1, 2, k, j]
                )
                eo += n * (
                    p_target[target, k, j]
                    + sine * data.p_affine[2, 1, k, j]
                    + cosine * data.p_affine[2, 2, k, j]
                )
            e_a[i, j] = eo - ec
            e_b[i, j] = wp.float64(2.0) * es
            e_c[i, j] = eo + ec
    for i in range(36):
        for j in range(12):
            pencil[target, sample, i, j] = wp.float64(0.0)
    for i in range(6):
        for j in range(9):
            a, b, c = wp.float64(0.0), wp.float64(0.0), wp.float64(0.0)
            for k in range(9):
                w = data.w[k, j]
                a += e_a[i, k] * w
                b += e_b[i, k] * w
                c += e_c[i, k] * w
            shifted = int(j - 3)
            if j < 3:
                shifted = j + 9
            pencil[target, sample, i, j] = a
            pencil[target, sample, i + 12, j] = b
            pencil[target, sample, i + 24, j] = c
            pencil[target, sample, i + 6, shifted] = a
            pencil[target, sample, i + 18, shifted] = b
            pencil[target, sample, i + 30, shifted] = c

    shift = wp.float64(0.2718281828459045)
    for i in range(24):
        for j in range(24):
            lhs[target, sample, i, j] = wp.float64(0.0)
            rhs[target, sample, i, j] = wp.float64(0.0)
            imag[target, sample, i, j] = wp.float64(0.0)
    for i in range(12):
        lhs[target, sample, i, i] = -shift
        lhs[target, sample, i, i + 12] = wp.float64(1.0)
        rhs[target, sample, i, i] = wp.float64(1.0)
        for j in range(12):
            lhs[target, sample, i + 12, j] = -pencil[target, sample, i + 24, j]
            lhs[target, sample, i + 12, j + 12] = (
                -pencil[target, sample, i + 12, j]
                - shift * pencil[target, sample, i, j]
            )
            rhs[target, sample, i + 12, j + 12] = pencil[target, sample, i, j]
    for i in range(24):
        scale = wp.float64(0.0)
        for j in range(24):
            scale = wp.max(scale, wp.abs(lhs[target, sample, i, j]))
        if scale > wp.float64(0.0):
            for j in range(24):
                lhs[target, sample, i, j] /= scale
                rhs[target, sample, i, j] /= scale
    for k in range(24):
        pivot = int(k)
        for i in range(k + 1, 24):
            if wp.abs(lhs[target, sample, i, k]) > wp.abs(
                lhs[target, sample, pivot, k]
            ):
                pivot = i
        if wp.abs(lhs[target, sample, pivot, k]) < wp.float64(1e-28):
            ok = False
        if pivot != k:
            for j in range(24):
                temp = lhs[target, sample, k, j]
                lhs[target, sample, k, j] = lhs[target, sample, pivot, j]
                lhs[target, sample, pivot, j] = temp
                temp = rhs[target, sample, k, j]
                rhs[target, sample, k, j] = rhs[target, sample, pivot, j]
                rhs[target, sample, pivot, j] = temp
        diagonal = lhs[target, sample, k, k]
        if wp.abs(diagonal) < wp.float64(1e-28):
            diagonal = wp.float64(1e-28)
        for i in range(k + 1, 24):
            factor = lhs[target, sample, i, k] / diagonal
            for j in range(k + 1, 24):
                lhs[target, sample, i, j] -= factor * lhs[target, sample, k, j]
            for j in range(24):
                rhs[target, sample, i, j] -= factor * rhs[target, sample, k, j]
    for offset in range(24):
        i = 23 - offset
        diagonal = lhs[target, sample, i, i]
        if wp.abs(diagonal) < wp.float64(1e-28):
            diagonal = wp.float64(1e-28)
        for j in range(24):
            value = rhs[target, sample, i, j]
            for k in range(i + 1, 24):
                value -= lhs[target, sample, i, k] * real[target, sample, k, j]
            real[target, sample, i, j] = value / diagonal
    # Retain the fixed-chart solver's Householder reduction and QR layout.
    # Each task owns one bounded (target, sample) scratch slab.
    for k in range(22):
        norm = wp.float64(0.0)
        for i in range(k + 1, 24):
            value = real[target, sample, i, k]
            vectors[target, sample, 0, i] = value
            norm += value * value
        norm = wp.sqrt(norm)
        if norm > wp.float64(1e-28):
            if vectors[target, sample, 0, k + 1] >= wp.float64(0.0):
                norm = -norm
            vectors[target, sample, 0, k + 1] -= norm
            vv = wp.float64(0.0)
            for i in range(k + 1, 24):
                vv += vectors[target, sample, 0, i] * vectors[target, sample, 0, i]
            beta = wp.float64(2.0) / vv
            for j in range(k, 24):
                dot = wp.float64(0.0)
                for i in range(k + 1, 24):
                    dot += vectors[target, sample, 0, i] * real[target, sample, i, j]
                for i in range(k + 1, 24):
                    real[target, sample, i, j] -= (
                        beta * vectors[target, sample, 0, i] * dot
                    )
            for i in range(24):
                dot = wp.float64(0.0)
                for j in range(k + 1, 24):
                    dot += real[target, sample, i, j] * vectors[target, sample, 0, j]
                for j in range(k + 1, 24):
                    real[target, sample, i, j] -= (
                        beta * dot * vectors[target, sample, 0, j]
                    )
            for i in range(k + 2, 24):
                real[target, sample, i, k] = wp.float64(0.0)
    converged = _eigenvalues_global(
        real, imag, vectors, roots, root_imag, target, sample
    )
    for j in range(24):
        real_part, imag_part = roots[target, sample, j], root_imag[target, sample, j]
        denominator = real_part * real_part + imag_part * imag_part
        x, y = wp.float64(1e30), wp.float64(0.0)
        if denominator > wp.float64(1e-26):
            x = shift + real_part / denominator
            y = -imag_part / denominator
        spurious = wp.abs(wp.abs(y) - wp.float64(1.0)) < wp.float64(0.1) and wp.abs(
            x
        ) < wp.float64(0.1)
        is_real = wp.abs(y) <= wp.float64(1e-3) * wp.max(wp.float64(1.0), wp.abs(x))
        roots[target, sample, j] = x
        valid[target, sample, j] = wp.int32(
            ok and converged and not spurious and is_real and wp.isfinite(x)
        )
