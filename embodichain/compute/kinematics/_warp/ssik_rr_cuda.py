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

"""Cooperative CUDA QR for the small non-Hermitian SSIK matrix pencil.

One CUDA warp owns one 24-by-24 Hessenberg matrix. Its real and imaginary
parts reside in shared memory; this avoids the large per-thread local-memory
allocation of an automatic matrix and parallelizes each Givens update.
"""

from __future__ import annotations

import warp as wp

from .ssik_rr import _RRData

wp.set_module_options({"enable_backward": False, "max_unroll": 0})

__all__: list[str] = []


@wp.func_native(snippet=r"""
#if defined(__CUDA_ARCH__)
    __shared__ double po[126], q[112], qt[196], v[24], e[162];
    __shared__ double aa[144], bb[144], cc[144], lhs[576], rhs[576], h[576];
    __shared__ double beta, diagonal;
    __shared__ int pivot, ok;
    const unsigned mask = 0xffffffffu;
    for (int k = lane; k < 126; k += 32) po[k] = 0.0;
    for (int k = lane; k < 112; k += 32) q[k] = 0.0;
    for (int k = lane; k < 196; k += 32) qt[k] = (k / 14 == k % 14) ? 1.0 : 0.0;
    if (lane == 0) {
        ok = 1;
        for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) if (!isfinite(goal.data[i][j])) ok = 0;
    }
    __syncwarp(mask);
    if (!ok) {
        for (int k = lane; k < 576; k += 32) wp::index(real, target, sample, k / 24, k % 24) = 0.0;
        if (lane < 24) wp::index(valid, target, sample, lane) = 0;
        return;
    }
    for (int k = wp::index(data.po_offsets, sample) + lane; k < wp::index(data.po_offsets, sample + 1); k += 32) {
        const int m = wp::index(data.po_indices, k, 2);
        double value = wp::index(data.po_coeff, k);
        for (int f = 0; f < 3; ++f) {
            const int index = wp::index(data.factors, m, f);
            if (index >= 0) value *= goal.data[index / 4][index % 4];
        }
        atomicAdd(&po[wp::index(data.po_indices, k, 0) * 9 + wp::index(data.po_indices, k, 1)], value);
    }
    for (int k = wp::index(data.q_offsets, sample) + lane; k < wp::index(data.q_offsets, sample + 1); k += 32) {
        const int m = wp::index(data.q_indices, k, 2);
        double value = wp::index(data.q_coeff, k);
        for (int f = 0; f < 3; ++f) {
            const int index = wp::index(data.factors, m, f);
            if (index >= 0) value *= goal.data[index / 4][index % 4];
        }
        atomicAdd(&q[wp::index(data.q_indices, k, 0) * 8 + wp::index(data.q_indices, k, 1)], value);
    }
    __syncwarp(mask);
    for (int k = 0; k < 8; ++k) {
        double value = (lane >= k && lane < 14) ? q[lane * 8 + k] : 0.0;
        if (lane < 14) v[lane] = value;
        __syncwarp(mask); // Shuffle reductions do not order shared-memory writes.
        value *= value;
        for (int offset = 16; offset > 0; offset /= 2) value += __shfl_down_sync(mask, value, offset);
        double norm = sqrt(__shfl_sync(mask, value, 0));
        if (lane == 0) {
            if (norm < 1e-14) { ok = 0; beta = 0.0; }
            else { if (v[k] >= 0.0) norm = -norm; v[k] -= norm; }
        }
        __syncwarp(mask);
        value = (lane >= k && lane < 14) ? v[lane] * v[lane] : 0.0;
        for (int offset = 16; offset > 0; offset /= 2) value += __shfl_down_sync(mask, value, offset);
        if (lane == 0) beta = value > 1e-28 ? 2.0 / value : 0.0;
        __syncwarp(mask);
        if (lane >= k && lane < 8) {
            double dot = 0.0;
            for (int i = k; i < 14; ++i) dot += v[i] * q[i * 8 + lane];
            for (int i = k; i < 14; ++i) q[i * 8 + lane] -= beta * v[i] * dot;
        }
        if (lane < 14) {
            double dot = 0.0;
            for (int i = k; i < 14; ++i) dot += v[i] * qt[i * 14 + lane];
            for (int i = k; i < 14; ++i) qt[i * 14 + lane] -= beta * v[i] * dot;
        }
        __syncwarp(mask);
    }
    if (lane < 14) {
        for (int i = 7; i >= 0; --i) {
            double value = qt[i * 14 + lane];
            for (int k = i + 1; k < 8; ++k) value -= q[i * 8 + k] * wp::index(q_inverse, target, sample, k, lane);
            const double div = fabs(q[i * 8 + i]) > 1e-20 ? q[i * 8 + i] : 1e-20;
            wp::index(q_inverse, target, sample, i, lane) = value / div;
        }
    }
    for (int index = lane; index < 54; index += 32) {
        const int row = index / 9, column = index % 9;
        double es = 0.0, ec = 0.0, eo = 0.0;
        for (int k = 0; k < 14; ++k) {
            const double n = qt[(row + 8) * 14 + k];
            es += n * wp::index(data.p_sin, sample, k, column);
            ec += n * wp::index(data.p_cos, sample, k, column);
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
    for (int k = lane; k < 126; k += 32) wp::index(p_one, target, sample, k / 9, k % 9) = po[k];
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
def _cooperative_prepare(
    data: _RRData,
    goal: wp.mat44d,
    p_one: wp.array4d(dtype=wp.float64),
    q_inverse: wp.array4d(dtype=wp.float64),
    pencil: wp.array4d(dtype=wp.float64),
    real: wp.array4d(dtype=wp.float64),
    valid: wp.array3d(dtype=wp.int32),
    target: int,
    sample: int,
    lane: int,
): ...


@wp.kernel
def rr_prepare_cooperative(
    data: _RRData,
    targets: wp.array(dtype=wp.mat44d),
    p_one: wp.array4d(dtype=wp.float64),
    q_inverse: wp.array4d(dtype=wp.float64),
    pencil: wp.array4d(dtype=wp.float64),
    real: wp.array4d(dtype=wp.float64),
    valid: wp.array3d(dtype=wp.int32),
):
    target, sample, lane = wp.tid()
    goal = (
        data.t_pre_inv[sample]
        @ targets[target]
        @ data.source_to_artifact
        @ data.t_post_inv[sample]
    )
    _cooperative_prepare(
        data, goal, p_one, q_inverse, pencil, real, valid, target, sample, lane
    )


@wp.func_native(snippet=r"""
#if defined(__CUDA_ARCH__)
    // The caller launches exactly one full warp per matrix/problem.
    __shared__ double h[576], g[576], cr[24], ci[24], sr[24], si[24];
    __shared__ double wr[24], wi[24], mur, mui;
    __shared__ int n, iteration;
    const unsigned mask = 0xffffffffu;
    for (int k = lane; k < 576; k += 32) {
        h[k] = wp::index(real, target, sample, k / 24, k % 24);
        g[k] = 0.0;
    }
    if (lane < 24) { wr[lane] = 0.0; wi[lane] = 0.0; }
    if (lane == 0) { n = wp::index(valid, target, sample, 0) ? 24 : 0; iteration = 0; }
    __syncwarp(mask);
    while (n > 0 && iteration < 1600) {
        // All lanes must finish reading the loop guard before lane0 changes it.
        // Uniform control flow alone does not order shared reads and writes.
        __syncwarp(mask);
        if (lane == 0) {
            ++iteration;
            const int d = (n - 1) * 24 + n - 1;
            bool small = n == 1;
            if (n > 1) {
                const int a = (n - 2) * 24 + n - 2;
                small = hypot(h[d - 1], g[d - 1]) <= 1e-14 * (hypot(h[d], g[d]) + hypot(h[a], g[a]) + 1.0);
            }
            if (small) { wr[n - 1] = h[d]; wi[n - 1] = g[d]; --n; mur = 0.0; mui = 0.0; }
            else {
                const int a = (n - 2) * 24 + n - 2, b = a + 1, c = d - 1;
                const double ar = h[a] - h[d], ai = g[a] - g[d];
                const double dr = ar * ar - ai * ai + 4.0 * (h[b] * h[c] - g[b] * g[c]);
                const double di = 2.0 * ar * ai + 4.0 * (h[b] * g[c] + g[b] * h[c]);
                const double mag = hypot(dr, di);
                const double pr = sqrt(fmax(0.0, 0.5 * (mag + dr)));
                const double pi = copysign(sqrt(fmax(0.0, 0.5 * (mag - dr))), di);
                const double m0r = 0.5 * (h[a] + h[d] + pr), m0i = 0.5 * (g[a] + g[d] + pi);
                const double m1r = 0.5 * (h[a] + h[d] - pr), m1i = 0.5 * (g[a] + g[d] - pi);
                mur = m0r; mui = m0i;
                if (hypot(m1r - h[d], m1i - g[d]) < hypot(m0r - h[d], m0i - g[d])) { mur = m1r; mui = m1i; }
                if (iteration % 50 == 49) { const double r = hypot(h[c], g[c]); mur = h[d] + 0.75 * r; mui = g[d] + 0.25 * r; }
            }
            // Signal whether this pass only deflated an eigenvalue.
            cr[23] = small ? 1.0 : 0.0;
        }
        __syncwarp(mask);
        if (cr[23] != 0.0) continue;
        if (lane < n) { h[lane * 24 + lane] -= mur; g[lane * 24 + lane] -= mui; }
        __syncwarp(mask);
        for (int j = 0; j < n - 1; ++j) {
            if (lane == 0) {
                const int a = j * 24 + j, b = a + 24;
                const double norm = sqrt(h[a] * h[a] + g[a] * g[a] + h[b] * h[b] + g[b] * g[b]);
                cr[j] = norm > 1e-300 ? h[a] / norm : 1.0;
                ci[j] = norm > 1e-300 ? g[a] / norm : 0.0;
                sr[j] = norm > 1e-300 ? h[b] / norm : 0.0;
                si[j] = norm > 1e-300 ? g[b] / norm : 0.0;
            }
            __syncwarp(mask);
            if (lane >= j && lane < n) {
                const int a = j * 24 + lane, b = a + 24;
                const double ur = h[a], ui = g[a], vr = h[b], vi = g[b];
                h[a] = cr[j] * ur + ci[j] * ui + sr[j] * vr + si[j] * vi;
                g[a] = cr[j] * ui - ci[j] * ur + sr[j] * vi - si[j] * vr;
                h[b] = -sr[j] * ur + si[j] * ui + cr[j] * vr - ci[j] * vi;
                g[b] = -sr[j] * ui - si[j] * ur + cr[j] * vi + ci[j] * vr;
            }
            __syncwarp(mask);
            if (lane == 0) { h[(j + 1) * 24 + j] = 0.0; g[(j + 1) * 24 + j] = 0.0; }
            __syncwarp(mask);
        }
        for (int j = 0; j < n - 1; ++j) {
            if (lane < n) {
                const int a = lane * 24 + j, b = a + 1;
                const double ur = h[a], ui = g[a], vr = h[b], vi = g[b];
                h[a] = cr[j] * ur - ci[j] * ui + sr[j] * vr - si[j] * vi;
                g[a] = cr[j] * ui + ci[j] * ur + sr[j] * vi + si[j] * vr;
                h[b] = -sr[j] * ur - si[j] * ui + cr[j] * vr + ci[j] * vi;
                g[b] = -sr[j] * ui + si[j] * ur + cr[j] * vi - ci[j] * vr;
            }
            __syncwarp(mask);
        }
        if (lane < n) { h[lane * 24 + lane] += mur; g[lane * 24 + lane] += mui; }
        __syncwarp(mask);
    }
    if (lane < 24) {
        const double den = wr[lane] * wr[lane] + wi[lane] * wi[lane];
        const double x = den > 1e-26 ? 0.2718281828459045 + wr[lane] / den : 1e30;
        const double y = den > 1e-26 ? -wi[lane] / den : 0.0;
        const bool spurious = fabs(fabs(y) - 1.0) < 0.1 && fabs(x) < 0.1;
        const bool is_real = fabs(y) <= 1e-3 * fmax(1.0, fabs(x));
        wp::index(roots, target, sample, lane) = x;
        wp::index(valid, target, sample, lane) = wp::index(valid, target, sample, lane) && n == 0 && !spurious && is_real && isfinite(x);
    }
#endif
""")
def _cooperative_qr(
    real: wp.array4d(dtype=wp.float64),
    roots: wp.array3d(dtype=wp.float64),
    valid: wp.array3d(dtype=wp.int32),
    target: int,
    sample: int,
    lane: int,
): ...


@wp.kernel
def rr_eigen_cooperative(
    real: wp.array4d(dtype=wp.float64),
    roots: wp.array3d(dtype=wp.float64),
    valid: wp.array3d(dtype=wp.int32),
):
    target, sample, lane = wp.tid()
    _cooperative_qr(real, roots, valid, target, sample, lane)
