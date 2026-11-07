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

"""Device-resident Raghavan--Roth elimination and matrix-pencil roots.

The baked coefficient convention and projective back-substitution follow SSIK
4109147f37a7ca3e2516c83853aca583612783f3 (BSD-3-Clause, see ssik_LICENSE.txt).
The shifted pencil and complex QR implementation here run entirely on the
selected Warp device. Neither CPU LAPACK nor a numerical IK fallback is used.
"""

from __future__ import annotations

import warp as wp

wp.set_module_options({"enable_backward": False, "max_unroll": 0})

__all__: list[str] = []

_v14 = wp.types.vector(length=14, dtype=wp.float64)
_v12 = wp.types.vector(length=12, dtype=wp.float64)
_v6 = wp.types.vector(length=6, dtype=wp.float64)
_v9 = wp.types.vector(length=9, dtype=wp.float64)
_v8 = wp.types.vector(length=8, dtype=wp.float64)


@wp.struct
class _RRData:
    p_sin: wp.array3d(dtype=wp.float64)
    p_cos: wp.array3d(dtype=wp.float64)
    factors: wp.array2d(dtype=wp.int32)
    po_indices: wp.array2d(dtype=wp.int32)
    po_coeff: wp.array(dtype=wp.float64)
    po_offsets: wp.array(dtype=wp.int32)
    q_indices: wp.array2d(dtype=wp.int32)
    q_coeff: wp.array(dtype=wp.float64)
    q_offsets: wp.array(dtype=wp.int32)
    w: wp.array2d(dtype=wp.float64)
    dh: wp.array2d(dtype=wp.float64)
    theta_offset: wp.array2d(dtype=wp.float64)
    t_pre_inv: wp.array(dtype=wp.mat44d)
    t_post_inv: wp.array(dtype=wp.mat44d)
    metadata: wp.array2d(dtype=wp.int32)
    lock_angles: wp.array(dtype=wp.float64)
    lock_joint: wp.int32
    source_to_artifact: wp.mat44d


@wp.func
def _factor(goal: wp.mat44d, data: _RRData, mono: int) -> wp.float64:
    value = wp.float64(1.0)
    for k in range(3):
        index = data.factors[mono, k]
        if index >= 0:
            value *= goal[index // 4, index % 4]
    return value


@wp.func
def _cmul(a: wp.vec2d, b: wp.vec2d) -> wp.vec2d:
    return wp.vec2d(a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0])


@wp.func
def _conj(a: wp.vec2d) -> wp.vec2d:
    return wp.vec2d(a[0], -a[1])


@wp.func
def _csqrt(a: wp.vec2d) -> wp.vec2d:
    mag = wp.length(a)
    real = wp.sqrt(wp.max(wp.float64(0.0), (mag + a[0]) * wp.float64(0.5)))
    imag = wp.sqrt(wp.max(wp.float64(0.0), (mag - a[0]) * wp.float64(0.5)))
    if a[1] < wp.float64(0.0):
        imag = -imag
    return wp.vec2d(real, imag)


@wp.func
def _eigenvalues_global(
    real: wp.array4d(dtype=wp.float64),
    imag: wp.array4d(dtype=wp.float64),
    vectors: wp.array4d(dtype=wp.float64),
    roots_real: wp.array3d(dtype=wp.float64),
    roots_imag: wp.array3d(dtype=wp.float64),
    target: int,
    sample: int,
) -> bool:
    n = int(24)
    iteration = int(0)
    while n > 0 and iteration < 1600:
        iteration += 1
        small = bool(n == 1)
        if n > 1:
            value = wp.vec2d(
                real[target, sample, n - 1, n - 2], imag[target, sample, n - 1, n - 2]
            )
            scale = wp.length(
                wp.vec2d(
                    real[target, sample, n - 1, n - 1],
                    imag[target, sample, n - 1, n - 1],
                )
            )
            scale += wp.length(
                wp.vec2d(
                    real[target, sample, n - 2, n - 2],
                    imag[target, sample, n - 2, n - 2],
                )
            )
            small = wp.length(value) <= wp.float64(1e-14) * (scale + wp.float64(1.0))
        if small:
            roots_real[target, sample, n - 1] = real[target, sample, n - 1, n - 1]
            roots_imag[target, sample, n - 1] = imag[target, sample, n - 1, n - 1]
            n -= 1
        else:
            a = wp.vec2d(
                real[target, sample, n - 2, n - 2], imag[target, sample, n - 2, n - 2]
            )
            b = wp.vec2d(
                real[target, sample, n - 2, n - 1], imag[target, sample, n - 2, n - 1]
            )
            c = wp.vec2d(
                real[target, sample, n - 1, n - 2], imag[target, sample, n - 1, n - 2]
            )
            d = wp.vec2d(
                real[target, sample, n - 1, n - 1], imag[target, sample, n - 1, n - 1]
            )
            discriminant = _csqrt(_cmul(a - d, a - d) + wp.float64(4.0) * _cmul(b, c))
            mu = (a + d + discriminant) * wp.float64(0.5)
            alternative = (a + d - discriminant) * wp.float64(0.5)
            if wp.length(alternative - d) < wp.length(mu - d):
                mu = alternative
            if iteration % 50 == 49:
                mu = d + wp.length(c) * wp.vec2d(wp.float64(0.75), wp.float64(0.25))
            for j in range(n):
                real[target, sample, j, j] -= mu[0]
                imag[target, sample, j, j] -= mu[1]
            for j in range(n - 1):
                x = wp.vec2d(real[target, sample, j, j], imag[target, sample, j, j])
                y = wp.vec2d(
                    real[target, sample, j + 1, j], imag[target, sample, j + 1, j]
                )
                norm = wp.sqrt(wp.dot(x, x) + wp.dot(y, y))
                cosine = wp.vec2d(wp.float64(1.0), wp.float64(0.0))
                sine = wp.vec2d()
                if norm > wp.float64(1e-300):
                    cosine = x / norm
                    sine = y / norm
                vectors[target, sample, 0, j] = cosine[0]
                vectors[target, sample, 1, j] = cosine[1]
                vectors[target, sample, 2, j] = sine[0]
                vectors[target, sample, 3, j] = sine[1]
                for k in range(j, n):
                    u = wp.vec2d(real[target, sample, j, k], imag[target, sample, j, k])
                    v = wp.vec2d(
                        real[target, sample, j + 1, k], imag[target, sample, j + 1, k]
                    )
                    top = _cmul(_conj(cosine), u) + _cmul(_conj(sine), v)
                    bottom = -_cmul(sine, u) + _cmul(cosine, v)
                    real[target, sample, j, k] = top[0]
                    imag[target, sample, j, k] = top[1]
                    real[target, sample, j + 1, k] = bottom[0]
                    imag[target, sample, j + 1, k] = bottom[1]
                real[target, sample, j + 1, j] = wp.float64(0.0)
                imag[target, sample, j + 1, j] = wp.float64(0.0)
            for j in range(n - 1):
                cosine = wp.vec2d(
                    vectors[target, sample, 0, j], vectors[target, sample, 1, j]
                )
                sine = wp.vec2d(
                    vectors[target, sample, 2, j], vectors[target, sample, 3, j]
                )
                for k in range(n):
                    u = wp.vec2d(real[target, sample, k, j], imag[target, sample, k, j])
                    v = wp.vec2d(
                        real[target, sample, k, j + 1], imag[target, sample, k, j + 1]
                    )
                    left = _cmul(cosine, u) + _cmul(sine, v)
                    right = -_cmul(_conj(sine), u) + _cmul(_conj(cosine), v)
                    real[target, sample, k, j] = left[0]
                    imag[target, sample, k, j] = left[1]
                    real[target, sample, k, j + 1] = right[0]
                    imag[target, sample, k, j + 1] = right[1]
            for j in range(n):
                real[target, sample, j, j] += mu[0]
                imag[target, sample, j, j] += mu[1]
    return n == 0


@wp.func
def _projective_angle(v: _v12, variable: int) -> wp.float64:
    numerator, denominator = wp.float64(0.0), wp.float64(0.0)
    best = wp.float64(-1.0)
    # Adjacent degrees are homogeneous coordinates [tan(q/2) : 1].
    # Select by pair signal, so a joint at pi remains readable.
    count = int(8)
    if variable == 0:
        count = 7
    for i in range(count):
        a, b = int(0), int(0)
        if variable == 0:
            if i == 0:
                a, b = 5, 8
            elif i == 1:
                a, b = 2, 5
            elif i == 2:
                a, b = 11, 2
            elif i == 3:
                a, b = 4, 7
            elif i == 4:
                a, b = 10, 1
            elif i == 5:
                a, b = 3, 6
            else:
                a, b = 9, 0
        else:
            if i == 0:
                a, b = 7, 8
            elif i == 1:
                a, b = 6, 7
            elif i == 2:
                a, b = 1, 2
            elif i == 3:
                a, b = 4, 5
            elif i == 4:
                a, b = 10, 11
            elif i == 5:
                a, b = 0, 1
            elif i == 6:
                a, b = 3, 4
            else:
                a, b = 9, 10
        signal = wp.max(wp.abs(v[a]), wp.abs(v[b]))
        if signal > best:
            best, numerator, denominator = signal, v[a], v[b]
    if denominator < wp.float64(0.0) or (
        denominator == wp.float64(0.0) and numerator < wp.float64(0.0)
    ):
        numerator, denominator = -numerator, -denominator
    return wp.float64(2.0) * wp.atan2(numerator, denominator)


@wp.func
def _dh_transform(
    q: wp.float64, alpha: wp.float64, a: wp.float64, d: wp.float64
) -> wp.mat44d:
    cq, sq, ca, sa = wp.cos(q), wp.sin(q), wp.cos(alpha), wp.sin(alpha)
    result = wp.mat44d()
    result[0, 0] = cq
    result[0, 1] = -sq * ca
    result[0, 2] = sq * sa
    result[0, 3] = a * cq
    result[1, 0] = sq
    result[1, 1] = cq * ca
    result[1, 2] = -cq * sa
    result[1, 3] = a * sq
    result[2, 1] = sa
    result[2, 2] = ca
    result[2, 3] = d
    result[3, 3] = wp.float64(1.0)
    return result


@wp.func
def _rigid_inverse(t: wp.mat44d) -> wp.mat44d:
    result = wp.mat44d()
    for i in range(3):
        for j in range(3):
            result[i, j] = t[j, i]
            result[i, 3] -= t[j, i] * t[j, 3]
    result[3, 3] = wp.float64(1.0)
    return result


@wp.kernel
def rr_roots(
    data: _RRData,
    targets: wp.array(dtype=wp.mat44d),
    p_one: wp.array4d(dtype=wp.float64),
    q: wp.array4d(dtype=wp.float64),
    qt: wp.array4d(dtype=wp.float64),
    q_inverse: wp.array4d(dtype=wp.float64),
    pencil: wp.array4d(dtype=wp.float64),
    lhs: wp.array4d(dtype=wp.float64),
    rhs: wp.array4d(dtype=wp.float64),
    real: wp.array4d(dtype=wp.float64),
    imag: wp.array4d(dtype=wp.float64),
    vectors: wp.array4d(dtype=wp.float64),
    roots: wp.array3d(dtype=wp.float64),
    root_imag: wp.array3d(dtype=wp.float64),
    valid: wp.array3d(dtype=wp.int32),
    cooperative_eigen: wp.int32,
):
    target, sample = wp.tid()
    goal = (
        data.t_pre_inv[sample]
        @ targets[target]
        @ data.source_to_artifact
        @ data.t_post_inv[sample]
    )
    for i in range(14):
        for j in range(14):
            qt[target, sample, i, j] = wp.float64(i == j)
        for j in range(9):
            p_one[target, sample, i, j] = wp.float64(0.0)
        for j in range(8):
            q[target, sample, i, j] = wp.float64(0.0)
    for i in range(data.po_offsets[sample], data.po_offsets[sample + 1]):
        row, column = int(data.po_indices[i, 0]), int(data.po_indices[i, 1])
        p_one[target, sample, row, column] += data.po_coeff[i] * _factor(
            goal, data, data.po_indices[i, 2]
        )
    for i in range(data.q_offsets[sample], data.q_offsets[sample + 1]):
        row, column = int(data.q_indices[i, 0]), int(data.q_indices[i, 1])
        q[target, sample, row, column] += data.q_coeff[i] * _factor(
            goal, data, data.q_indices[i, 2]
        )
    ok = bool(True)
    for k in range(8):
        norm = wp.float64(0.0)
        for i in range(k, 14):
            value = q[target, sample, i, k]
            vectors[target, sample, 0, i] = value
            norm += value * value
        norm = wp.sqrt(norm)
        if norm > wp.float64(1e-14):
            if vectors[target, sample, 0, k] >= wp.float64(0.0):
                norm = -norm
            vectors[target, sample, 0, k] -= norm
            vv = wp.float64(0.0)
            for i in range(k, 14):
                vv += vectors[target, sample, 0, i] * vectors[target, sample, 0, i]
            beta = wp.float64(2.0) / vv
            for j in range(k, 8):
                dot = wp.float64(0.0)
                for i in range(k, 14):
                    dot += vectors[target, sample, 0, i] * q[target, sample, i, j]
                for i in range(k, 14):
                    q[target, sample, i, j] -= (
                        beta * vectors[target, sample, 0, i] * dot
                    )
            for j in range(14):
                dot = wp.float64(0.0)
                for i in range(k, 14):
                    dot += vectors[target, sample, 0, i] * qt[target, sample, i, j]
                for i in range(k, 14):
                    qt[target, sample, i, j] -= (
                        beta * vectors[target, sample, 0, i] * dot
                    )
        else:
            ok = False
    for offset in range(8):
        i = 7 - offset
        for j in range(14):
            value = qt[target, sample, i, j]
            for k in range(i + 1, 8):
                value -= q[target, sample, i, k] * q_inverse[target, sample, k, j]
            diagonal = q[target, sample, i, i]
            if wp.abs(diagonal) < wp.float64(1e-20):
                diagonal = wp.float64(1e-20)
            q_inverse[target, sample, i, j] = value / diagonal
    for i in range(36):
        for j in range(12):
            pencil[target, sample, i, j] = wp.float64(0.0)
    for i in range(6):
        for j in range(9):
            es, ec, eo = wp.float64(0.0), wp.float64(0.0), wp.float64(0.0)
            for k in range(14):
                for l in range(9):
                    w = qt[target, sample, i + 8, k] * data.w[l, j]
                    es += w * data.p_sin[sample, k, l]
                    ec += w * data.p_cos[sample, k, l]
                    eo += w * p_one[target, sample, k, l]
            destination = int(j - 3)
            if j < 3:
                destination = j + 9
            pencil[target, sample, i, j] = eo - ec
            pencil[target, sample, i + 12, j] = wp.float64(2.0) * es
            pencil[target, sample, i + 24, j] = eo + ec
            pencil[target, sample, i + 6, destination] = eo - ec
            pencil[target, sample, i + 18, destination] = wp.float64(2.0) * es
            pencil[target, sample, i + 30, destination] = eo + ec
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
    # Hessenberg similarity and QR use bounded global arrays, not CUDA local
    # matrix temporaries. Large per-thread local matrices allocate gigabytes of
    # context backing storage on high-SM-count GPUs even for a singleton batch.
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
    if cooperative_eigen != 0:
        for j in range(24):
            valid[target, sample, j] = wp.int32(ok)
        return
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


@wp.kernel
def rr_back_substitute(
    data: _RRData,
    targets: wp.array(dtype=wp.mat44d),
    p_one: wp.array4d(dtype=wp.float64),
    q_inverse: wp.array4d(dtype=wp.float64),
    pencil: wp.array4d(dtype=wp.float64),
    roots: wp.array3d(dtype=wp.float64),
    root_valid: wp.array3d(dtype=wp.int32),
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
                rhs[i] += (
                    data.p_sin[sample, i, j] * wp.sin(q_lin)
                    + data.p_cos[sample, i, j] * wp.cos(q_lin)
                    + p_one[target, sample, i, j]
                ) * left[j]
        right = _v8()
        for i in range(8):
            for j in range(14):
                right[i] += q_inverse[target, sample, i, j] * rhs[j]
        q = _v6()
        q[data.metadata[sample, 0]] = q_lin
        q[data.metadata[sample, 1]] = q_left0
        q[data.metadata[sample, 2]] = q_left1
        q[data.metadata[sample, 3]] = wp.atan2(right[4], right[5])
        q[data.metadata[sample, 4]] = wp.atan2(right[6], right[7])
        drop = data.metadata[sample, 5]
        before, after = wp.identity(n=4, dtype=wp.float64), wp.identity(
            n=4, dtype=wp.float64
        )
        for j in range(drop):
            before = before @ _dh_transform(
                q[j],
                data.dh[sample, j],
                data.dh[sample, j + 6],
                data.dh[sample, j + 12],
            )
        for j in range(drop + 1, 6):
            after = after @ _dh_transform(
                q[j],
                data.dh[sample, j],
                data.dh[sample, j + 6],
                data.dh[sample, j + 12],
            )
        goal = (
            data.t_pre_inv[sample]
            @ targets[target]
            @ data.source_to_artifact
            @ data.t_post_inv[sample]
        )
        residual = _rigid_inverse(before) @ goal @ _rigid_inverse(after)
        q[drop] = wp.atan2(residual[1, 0], residual[0, 0])
        finite = bool(True)
        for j in range(6):
            index = int(j)
            if j >= data.lock_joint:
                index += 1
            value = q[j] - data.theta_offset[sample, j]
            candidates[target, slot, index] = value
            finite = finite and wp.isfinite(value)
        candidates[target, slot, data.lock_joint] = data.lock_angles[sample]
        validity[target, slot] = wp.int32(finite)
