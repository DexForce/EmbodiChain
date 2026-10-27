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

"""Cooperative parameterized POE coefficient and matrix-pencil preparation."""

from __future__ import annotations

import warp as wp

from .ssik_rr_parametric import _ParametricData

wp.set_module_options({"enable_backward": False, "max_unroll": 0})

__all__: list[str] = []


@wp.func_native(snippet=r"""
#if defined(__CUDA_ARCH__)
    // A single tangent chooses two scalar lock coordinates. No tangent joint
    // vector is an IK result: both coordinates feed fresh algebraic RR solves.
    __shared__ double rotation[9], temporary[9], position[3], origins[21], axes[21];
    __shared__ double jacobian[42], error[6], matrix[36], rhs[12], solution[12];
    __shared__ double prediction0, prediction1;
    __shared__ int pivot, usable, fallback;
    const unsigned mask = 0xffffffffu;
    if (lane == 0) {
        for (int i = 0; i < 9; ++i) rotation[i] = i / 3 == i % 3 ? 1.0 : 0.0;
        for (int i = 0; i < 3; ++i) position[i] = 0.0;
        for (int joint = 0; joint < 8; ++joint) {
            for (int i = 0; i < 3; ++i) {
                double shift = 0.0;
                for (int k = 0; k < 3; ++k) shift += rotation[i * 3 + k] * wp::index(data.model_frames, joint, k * 4 + 3);
                position[i] += shift;
                for (int j = 0; j < 3; ++j) {
                    double value = 0.0;
                    for (int k = 0; k < 3; ++k) value += rotation[i * 3 + k] * wp::index(data.model_frames, joint, k * 4 + j);
                    temporary[i * 3 + j] = value;
                }
            }
            for (int i = 0; i < 9; ++i) rotation[i] = temporary[i];
            if (joint < 7) {
                const double ax = wp::index(data.model_axes, joint, 0), ay = wp::index(data.model_axes, joint, 1), az = wp::index(data.model_axes, joint, 2);
                const double magnitude = sqrt(ax * ax + ay * ay + az * az);
                const double ux = ax / magnitude, uy = ay / magnitude, uz = az / magnitude;
                const double angle = wp::index(seed, target, joint) * magnitude;
                const double sine = sin(angle), cosine = cos(angle);
                for (int i = 0; i < 3; ++i) { origins[joint * 3 + i] = position[i]; axes[joint * 3 + i] = rotation[i * 3] * ax + rotation[i * 3 + 1] * ay + rotation[i * 3 + 2] * az; }
                for (int i = 0; i < 3; ++i) {
                    const double radial = axes[joint * 3 + i] / magnitude;
                    temporary[i * 3] = cosine * rotation[i * 3] + (1.0 - cosine) * radial * ux + sine * (rotation[i * 3 + 1] * uz - rotation[i * 3 + 2] * uy);
                    temporary[i * 3 + 1] = cosine * rotation[i * 3 + 1] + (1.0 - cosine) * radial * uy + sine * (-rotation[i * 3] * uz + rotation[i * 3 + 2] * ux);
                    temporary[i * 3 + 2] = cosine * rotation[i * 3 + 2] + (1.0 - cosine) * radial * uz + sine * (rotation[i * 3] * uy - rotation[i * 3 + 1] * ux);
                }
                for (int i = 0; i < 9; ++i) rotation[i] = temporary[i];
            }
        }
        for (int i = 0; i < 3; ++i) error[i] = goal.data[i][3] - position[i];
        for (int i = 0; i < 3; ++i) {
            const int a = (i + 1) % 3, b = (i + 2) % 3;
            double value = 0.0;
            for (int k = 0; k < 3; ++k) value += goal.data[b][k] * rotation[a * 3 + k] - goal.data[a][k] * rotation[b * 3 + k];
            error[i + 3] = 0.5 * value;
        }
        prediction0 = prediction1 = 0.0;
    }
    __syncwarp(mask);
    if (lane < 7) {
        const double dx = position[0] - origins[lane * 3], dy = position[1] - origins[lane * 3 + 1], dz = position[2] - origins[lane * 3 + 2];
        const double ax = axes[lane * 3], ay = axes[lane * 3 + 1], az = axes[lane * 3 + 2];
        jacobian[lane] = ay * dz - az * dy;
        jacobian[7 + lane] = az * dx - ax * dz;
        jacobian[14 + lane] = ax * dy - ay * dx;
        jacobian[21 + lane] = ax; jacobian[28 + lane] = ay; jacobian[35 + lane] = az;
    }
    __syncwarp(mask);
    for (int attempt = 0; attempt < 2; ++attempt) {
        if (lane == 0) usable = 1;
        if (lane < 6) {
            rhs[lane * 2] = error[lane]; rhs[lane * 2 + 1] = jacobian[lane * 7 + 2];
            for (int j = 0; j < 6; ++j) {
                if (attempt == 0) matrix[lane * 6 + j] = jacobian[lane * 7 + (j < 2 ? j : j + 1)];
                else {
                    double value = lane == j ? 1e-10 : 0.0;
                    for (int k = 0; k < 7; ++k) value += jacobian[lane * 7 + k] * jacobian[j * 7 + k];
                    matrix[lane * 6 + j] = value;
                }
            }
        }
        __syncwarp(mask);
        for (int k = 0; k < 6; ++k) {
            if (lane == 0) {
                pivot = k;
                for (int i = k + 1; i < 6; ++i) if (fabs(matrix[i * 6 + k]) > fabs(matrix[pivot * 6 + k])) pivot = i;
                if (fabs(matrix[pivot * 6 + k]) < (attempt == 0 ? 1e-12 : 1e-20)) usable = 0;
            }
            __syncwarp(mask);
            if (lane < 6 && pivot != k) { const double swap = matrix[k * 6 + lane]; matrix[k * 6 + lane] = matrix[pivot * 6 + lane]; matrix[pivot * 6 + lane] = swap; }
            if (lane < 2 && pivot != k) { const double swap = rhs[k * 2 + lane]; rhs[k * 2 + lane] = rhs[pivot * 2 + lane]; rhs[pivot * 2 + lane] = swap; }
            __syncwarp(mask);
            if (lane > k && lane < 6) {
                const double diagonal = fabs(matrix[k * 6 + k]) > 1e-25 ? matrix[k * 6 + k] : 1e-25;
                const double factor = matrix[lane * 6 + k] / diagonal;
                for (int j = k + 1; j < 6; ++j) matrix[lane * 6 + j] -= factor * matrix[k * 6 + j];
                for (int j = 0; j < 2; ++j) rhs[lane * 2 + j] -= factor * rhs[k * 2 + j];
            }
            __syncwarp(mask);
        }
        if (lane < 2) {
            for (int i = 5; i >= 0; --i) {
                double value = rhs[i * 2 + lane];
                for (int j = i + 1; j < 6; ++j) value -= matrix[i * 6 + j] * solution[j * 2 + lane];
                const double diagonal = fabs(matrix[i * 6 + i]) > 1e-25 ? matrix[i * 6 + i] : 1e-25;
                solution[i * 2 + lane] = value / diagonal;
            }
        }
        __syncwarp(mask);
        if (attempt == 1 || usable) { if (lane == 0) fallback = attempt; __syncwarp(mask); break; }
    }
    if (lane == 0) {
        const double trust = fmin(max_step, .2);
        double first = fmax(-trust, wp::index(lower, 2) - wp::index(seed, target, 2));
        double last = fmin(trust, wp::index(upper, 2) - wp::index(seed, target, 2));
        if (first <= last && usable) {
            double unconstrained = 0.0;
            if (fallback) for (int i = 0; i < 6; ++i) unconstrained += jacobian[i * 7 + 2] * solution[i * 2];
            else {
                double numerator = 0.0, denominator = 1.0;
                for (int i = 0; i < 6; ++i) { numerator += solution[i * 2] * solution[i * 2 + 1]; denominator += solution[i * 2 + 1] * solution[i * 2 + 1]; }
                unconstrained = numerator / denominator;
            }
            prediction0 = fmin(last, fmax(first, unconstrained)); prediction1 = .5 * prediction0;
            if (!fallback) {
                double lo = first, hi = last;
                for (int i = 0; i < 6; ++i) {
                    const int joint = i < 2 ? i : i + 1;
                    const double low = fmax(-trust, wp::index(lower, joint) - wp::index(seed, target, joint));
                    const double high = fmin(trust, wp::index(upper, joint) - wp::index(seed, target, joint));
                    const double u = solution[i * 2], v0 = solution[i * 2 + 1];
                    if (fabs(v0) > 1e-14) { const double a = (u - low) / v0, b = (u - high) / v0; lo = fmax(lo, fmin(a, b)); hi = fmin(hi, fmax(a, b)); }
                    else if (u < low || u > high) hi = lo - 1.0;
                }
                if (lo <= hi) { prediction0 = fmin(hi, fmax(lo, unconstrained)); prediction1 = .5 * (lo + hi); }
            }
        }
        wp::index(predictions, target, 0) = prediction0;
        wp::index(predictions, target, 1) = prediction1;
    }
#endif
""")
def _lock_predict(
    data: _ParametricData,
    goal: wp.mat44d,
    seed: wp.array2d(dtype=wp.float64),
    lower: wp.array(dtype=wp.float64),
    upper: wp.array(dtype=wp.float64),
    max_step: wp.float64,
    predictions: wp.array2d(dtype=wp.float64),
    target: int,
    lane: int,
): ...


@wp.kernel
def parametric_lock_predictor(
    data: _ParametricData,
    targets: wp.array(dtype=wp.mat44d),
    seed: wp.array2d(dtype=wp.float64),
    lower: wp.array(dtype=wp.float64),
    upper: wp.array(dtype=wp.float64),
    max_step: wp.float64,
    predictions: wp.array2d(dtype=wp.float64),
):
    target, lane = wp.tid()
    _lock_predict(
        data, targets[target], seed, lower, upper, max_step, predictions, target, lane
    )


@wp.func_native(snippet=r"""
#if defined(__CUDA_ARCH__)
    __shared__ double po[126], q[112], qt[196], v[14], beta;
    __shared__ double rotation[9], gram[9], updated[9];
    __shared__ double conditioned[16], intermediate[16], goal[16];
    __shared__ int ok;
    const unsigned mask = 0xffffffffu;
    if (lane == 0) {
        ok = 1;
        for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) {
            const double value = original.data[i][j];
            conditioned[i * 4 + j] = value;
            if (!isfinite(value)) ok = 0;
            if (i < 3 && j < 3) rotation[i * 3 + j] = value;
        }
        for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) {
            double value = 0.0;
            for (int k = 0; k < 3; ++k) value += rotation[k * 3 + i] * rotation[k * 3 + j];
            if (fabs(value - (i == j ? 1.0 : 0.0)) > 1e-5) ok = 0;
        }
        const double determinant = rotation[0] * (rotation[4] * rotation[8] - rotation[5] * rotation[7])
                                 - rotation[1] * (rotation[3] * rotation[8] - rotation[5] * rotation[6])
                                 + rotation[2] * (rotation[3] * rotation[7] - rotation[4] * rotation[6]);
        if (!(determinant > 0.0)) ok = 0;
        for (int j = 0; j < 4; ++j) if (fabs(original.data[3][j] - (j == 3 ? 1.0 : 0.0)) > 1e-6) ok = 0;
        if (ok) {
            // The RR reduction assumes exact SO(3) identities. Condition only
            // after checking the original input; final FK still uses that input.
            for (int step = 0; step < 2; ++step) {
                for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) {
                    double value = 0.0;
                    for (int k = 0; k < 3; ++k) value += rotation[k * 3 + i] * rotation[k * 3 + j];
                    gram[i * 3 + j] = value;
                }
                for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) {
                    double value = 0.0;
                    for (int k = 0; k < 3; ++k) value += rotation[i * 3 + k] * ((k == j ? 1.5 : 0.0) - 0.5 * gram[k * 3 + j]);
                    updated[i * 3 + j] = value;
                }
                for (int i = 0; i < 9; ++i) rotation[i] = updated[i];
            }
            for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) conditioned[i * 4 + j] = rotation[i * 3 + j];
            for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) {
                double value = 0.0;
                for (int k = 0; k < 4; ++k) value += data.pre_inverse.data[i][k] * conditioned[k * 4 + j];
                intermediate[i * 4 + j] = value;
            }
            for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) {
                double value = 0.0;
                for (int k = 0; k < 4; ++k) value += intermediate[i * 4 + k] * data.source_to_artifact.data[k][j];
                goal[i * 4 + j] = value;
                wp::index(goals, target).data[i][j] = value;
            }
        }
    }
    for (int index = lane; index < 196; index += 32) qt[index] = index / 14 == index % 14 ? 1.0 : 0.0;
    __syncwarp(mask);
    if (!ok) { if (lane == 0) wp::index(target_valid, target) = 0; return; }
    // CSR by output cell gives deterministic accumulation: no shared atomics.
    for (int cell = lane; cell < 238; cell += 32) {
        double sum = 0.0;
        for (int k = wp::index(data.cell_offsets, cell); k < wp::index(data.cell_offsets, cell + 1); ++k) {
            double value = wp::index(data.coefficient_values, k);
            const int mono = wp::index(data.coefficient_monomials, k);
            for (int f = 0; f < 4; ++f) { const int variable = wp::index(data.mono_factors, mono, f); if (variable >= 0) value *= goal[variable]; }
            sum += value;
        }
        if (cell < 126) po[cell] = sum; else q[cell - 126] = sum;
    }
    __syncwarp(mask);
    for (int k = 0; k < 8; ++k) {
        double value = lane >= k && lane < 14 ? q[lane * 8 + k] : 0.0;
        if (lane < 14) v[lane] = value;
        __syncwarp(mask); // Shuffle reductions do not order shared-memory writes.
        value *= value;
        for (int offset = 16; offset > 0; offset /= 2) value += __shfl_down_sync(mask, value, offset);
        double norm = sqrt(__shfl_sync(mask, value, 0));
        if (lane == 0) { if (norm < 1e-14) { ok = 0; beta = 0.0; } else { if (v[k] >= 0.0) norm = -norm; v[k] -= norm; } }
        __syncwarp(mask);
        value = lane >= k && lane < 14 ? v[lane] * v[lane] : 0.0;
        for (int offset = 16; offset > 0; offset /= 2) value += __shfl_down_sync(mask, value, offset);
        if (lane == 0) beta = value > 1e-28 ? 2.0 / value : 0.0;
        __syncwarp(mask);
        if (lane >= k && lane < 8) { double dot = 0.0; for (int i = k; i < 14; ++i) dot += v[i] * q[i * 8 + lane]; for (int i = k; i < 14; ++i) q[i * 8 + lane] -= beta * v[i] * dot; }
        if (lane < 14) { double dot = 0.0; for (int i = k; i < 14; ++i) dot += v[i] * qt[i * 14 + lane]; for (int i = k; i < 14; ++i) qt[i * 14 + lane] -= beta * v[i] * dot; }
        __syncwarp(mask);
    }
    if (lane < 14) {
        for (int i = 7; i >= 0; --i) { double value = qt[i * 14 + lane]; for (int k = i + 1; k < 8; ++k) value -= q[i * 8 + k] * wp::index(q_inverse, target, k, lane); const double div = fabs(q[i * 8 + i]) > 1e-20 ? q[i * 8 + i] : 1e-20; wp::index(q_inverse, target, i, lane) = value / div; }
        for (int i = 0; i < 6; ++i) wp::index(q_basis, target, i, lane) = qt[(i + 8) * 14 + lane];
    }
    for (int index = lane; index < 126; index += 32) wp::index(p_target, target, index / 9, index % 9) = po[index];
    if (lane == 0) wp::index(target_valid, target) = ok;
#endif
""")
def _target_prepare(
    data: _ParametricData,
    original: wp.mat44d,
    goals: wp.array(dtype=wp.mat44d),
    p_target: wp.array3d(dtype=wp.float64),
    q_basis: wp.array3d(dtype=wp.float64),
    q_inverse: wp.array3d(dtype=wp.float64),
    target_valid: wp.array(dtype=wp.int32),
    target: int,
    lane: int,
): ...


@wp.kernel
def parametric_target_prepare(
    data: _ParametricData,
    targets: wp.array(dtype=wp.mat44d),
    goals: wp.array(dtype=wp.mat44d),
    p_target: wp.array3d(dtype=wp.float64),
    q_basis: wp.array3d(dtype=wp.float64),
    q_inverse: wp.array3d(dtype=wp.float64),
    target_valid: wp.array(dtype=wp.int32),
):
    target, lane = wp.tid()
    _target_prepare(
        data,
        targets[target],
        goals,
        p_target,
        q_basis,
        q_inverse,
        target_valid,
        target,
        lane,
    )


@wp.func_native(snippet=r"""
#if defined(__CUDA_ARCH__)
    __shared__ double ps[126], pc[126], po[126], v[24], e[162];
    __shared__ double aa[144], bb[144], cc[144], lhs[576], rhs[576], h[576];
    __shared__ double beta, phi;
    __shared__ int pivot, ok;
    const unsigned mask = 0xffffffffu;
    if (lane == 0) {
        const double initial = wp::index(seed, target, 2), lo = wp::index(lower, 2), hi = wp::index(upper, 2);
        double offset = sample < 16 ? 0.0 : (sample < 23 ? wp::index(data.local_offsets, sample - 16) : wp::index(predictions, target, sample - 23));
        double raw = sample < 16 ? wp::index(data.global_angles, sample) : fmin(hi, fmax(lo, initial + offset));
        const double first = ceil((lo - 1e-6 - raw) / (2 * M_PI)), last = floor((hi + 1e-6 - raw) / (2 * M_PI));
        const double turns = fmin(last, fmax(first, floor((initial - raw) / (2 * M_PI) + .5)));
        phi = fmin(hi, fmax(lo, raw + 2 * M_PI * turns));
        ok = wp::index(target_valid, target) && first <= last && fabs((double)(float)phi - initial) <= max_step && isfinite(phi);
        wp::index(angles, target, sample) = phi;
    }
    __syncwarp(mask);
    if (!ok) { for (int k = lane; k < 576; k += 32) wp::index(real, target, sample, k / 24, k % 24) = 0.0; if (lane < 24) wp::index(valid, target, sample, lane) = 0; return; }
    const double ss = sin(phi), cc0 = cos(phi);
    for (int k = lane; k < 126; k += 32) {
        const int i = k / 9, j = k % 9;
        ps[k] = wp::index(data.p_affine, 0, 0, i, j) + ss * wp::index(data.p_affine, 0, 1, i, j) + cc0 * wp::index(data.p_affine, 0, 2, i, j);
        pc[k] = wp::index(data.p_affine, 1, 0, i, j) + ss * wp::index(data.p_affine, 1, 1, i, j) + cc0 * wp::index(data.p_affine, 1, 2, i, j);
        po[k] = wp::index(p_target, target, i, j) + ss * wp::index(data.p_affine, 2, 1, i, j) + cc0 * wp::index(data.p_affine, 2, 2, i, j);
    }
    __syncwarp(mask);
    for (int index = lane; index < 54; index += 32) {
        const int row = index / 9, column = index % 9;
        double es = 0.0, ec = 0.0, eo = 0.0;
        for (int k = 0; k < 14; ++k) {
            const double n = wp::index(q_basis, target, row, k);
            es += n * ps[k * 9 + column];
            ec += n * pc[k * 9 + column];
            eo += n * po[k * 9 + column];
        }
        e[index] = eo - ec; e[index + 54] = 2.0 * es; e[index + 108] = eo + ec;
    }
    for (int k = lane; k < 144; k += 32) { aa[k] = 0.0; bb[k] = 0.0; cc[k] = 0.0; }
    __syncwarp(mask);
    for (int index = lane; index < 54; index += 32) {
        const int row = index / 9, column = index % 9;
        double a = 0.0, b = 0.0, c = 0.0;
        for (int k = 0; k < 9; ++k) {
            const double w = wp::index(data.w, k, column);
            a += e[row * 9 + k] * w; b += e[row * 9 + k + 54] * w; c += e[row * 9 + k + 108] * w;
        }
        const int shifted = column < 3 ? column + 9 : column - 3;
        aa[row * 12 + column] = a; bb[row * 12 + column] = b; cc[row * 12 + column] = c;
        aa[(row + 6) * 12 + shifted] = a; bb[(row + 6) * 12 + shifted] = b; cc[(row + 6) * 12 + shifted] = c;
    }
    __syncwarp(mask);
    for (int k = lane; k < 144; k += 32) {
        wp::index(pencil, target, sample, k / 12, k % 12) = aa[k];
        wp::index(pencil, target, sample, k / 12 + 12, k % 12) = bb[k];
        wp::index(pencil, target, sample, k / 12 + 24, k % 12) = cc[k];
    }
    const double shift = 0.2718281828459045;
    for (int k = lane; k < 576; k += 32) {
        const int i = k / 24, j = k % 24;
        lhs[k] = 0.0; rhs[k] = 0.0;
        if (i < 12) {
            if (i == j) { lhs[k] = -shift; rhs[k] = 1.0; }
            if (j == i + 12) lhs[k] = 1.0;
        } else {
            if (j < 12) lhs[k] = -cc[(i - 12) * 12 + j];
            else { lhs[k] = -bb[(i - 12) * 12 + j - 12] - shift * aa[(i - 12) * 12 + j - 12]; rhs[k] = aa[(i - 12) * 12 + j - 12]; }
        }
    }
    __syncwarp(mask);
    if (lane < 24) {
        double scale = 0.0;
        for (int j = 0; j < 24; ++j) scale = fmax(scale, fabs(lhs[lane * 24 + j]));
        if (scale > 0.0) for (int j = 0; j < 24; ++j) { lhs[lane * 24 + j] /= scale; rhs[lane * 24 + j] /= scale; }
    }
    __syncwarp(mask);
    for (int k = 0; k < 24; ++k) {
        if (lane == 0) {
            pivot = k;
            for (int i = k + 1; i < 24; ++i) if (fabs(lhs[i * 24 + k]) > fabs(lhs[pivot * 24 + k])) pivot = i;
            if (fabs(lhs[pivot * 24 + k]) < 1e-28) ok = 0;
        }
        __syncwarp(mask);
        if (lane < 24 && pivot != k) {
            double t = lhs[k * 24 + lane]; lhs[k * 24 + lane] = lhs[pivot * 24 + lane]; lhs[pivot * 24 + lane] = t;
            t = rhs[k * 24 + lane]; rhs[k * 24 + lane] = rhs[pivot * 24 + lane]; rhs[pivot * 24 + lane] = t;
        }
        __syncwarp(mask);
        if (lane > k && lane < 24) {
            const double div = fabs(lhs[k * 24 + k]) > 1e-28 ? lhs[k * 24 + k] : 1e-28;
            const double f = lhs[lane * 24 + k] / div;
            for (int j = k + 1; j < 24; ++j) lhs[lane * 24 + j] -= f * lhs[k * 24 + j];
            for (int j = 0; j < 24; ++j) rhs[lane * 24 + j] -= f * rhs[k * 24 + j];
        }
        __syncwarp(mask);
    }
    if (lane < 24) {
        for (int i = 23; i >= 0; --i) {
            double value = rhs[i * 24 + lane];
            for (int k = i + 1; k < 24; ++k) value -= lhs[i * 24 + k] * h[k * 24 + lane];
            const double div = fabs(lhs[i * 24 + i]) > 1e-28 ? lhs[i * 24 + i] : 1e-28;
            h[i * 24 + lane] = value / div;
        }
    }
    __syncwarp(mask);
    for (int k = 0; k < 22; ++k) {
        double value = lane > k && lane < 24 ? h[lane * 24 + k] : 0.0;
        if (lane < 24) v[lane] = value;
        __syncwarp(mask); // Publish Householder coordinates before lane0 reads.
        value *= value;
        for (int offset = 16; offset > 0; offset /= 2) value += __shfl_down_sync(mask, value, offset);
        double norm = sqrt(__shfl_sync(mask, value, 0));
        if (lane == 0) {
            if (v[k + 1] >= 0.0) norm = -norm;
            v[k + 1] -= norm;
        }
        __syncwarp(mask);
        value = lane > k && lane < 24 ? v[lane] * v[lane] : 0.0;
        for (int offset = 16; offset > 0; offset /= 2) value += __shfl_down_sync(mask, value, offset);
        if (lane == 0) beta = value > 1e-56 ? 2.0 / value : 0.0;
        __syncwarp(mask);
        if (lane >= k && lane < 24) {
            double dot = 0.0;
            for (int i = k + 1; i < 24; ++i) dot += v[i] * h[i * 24 + lane];
            for (int i = k + 1; i < 24; ++i) h[i * 24 + lane] -= beta * v[i] * dot;
        }
        __syncwarp(mask);
        if (lane < 24) {
            double dot = 0.0;
            for (int j = k + 1; j < 24; ++j) dot += h[lane * 24 + j] * v[j];
            for (int j = k + 1; j < 24; ++j) h[lane * 24 + j] -= beta * dot * v[j];
            if (lane > k + 1) h[lane * 24 + k] = 0.0;
        }
        __syncwarp(mask);
    }
    for (int k = lane; k < 576; k += 32) wp::index(real, target, sample, k / 24, k % 24) = h[k];
    if (lane < 24) wp::index(valid, target, sample, lane) = ok;
#endif
""")
def _pencil_prepare(
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
    real: wp.array4d(dtype=wp.float64),
    valid: wp.array3d(dtype=wp.int32),
    target: int,
    sample: int,
    lane: int,
): ...


@wp.kernel
def parametric_pencil_prepare(
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
    real: wp.array4d(dtype=wp.float64),
    valid: wp.array3d(dtype=wp.int32),
):
    target, sample, lane = wp.tid()
    _pencil_prepare(
        data,
        p_target,
        q_basis,
        target_valid,
        seed,
        lower,
        upper,
        max_step,
        predictions,
        angles,
        pencil,
        real,
        valid,
        target,
        sample,
        lane,
    )
