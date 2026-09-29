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

"""Warp kernels for time-optimal path parameterization under joint box limits.

Three launches per call. The first two are parallel over every environment and
grid point: one converts path derivatives into per-stage constraint rows, the
other intersects those rows into the maximum velocity curve. The third runs one
thread per environment through the two sequential sweeps.

Reverse mode needs every loop that updates a local to be unrolled, so the joint
count and constraint-set count are compile-time constants captured by closure
and each loop bound stays within Warp's unroll limit. The grid sweeps are
dynamic loops, but each iteration stores to its own array slot and never
revisits it, which is the pattern Warp differentiates correctly.
"""

from __future__ import annotations

import functools

import warp as wp

__all__ = ["MAX_JOINTS", "build_topp_kernels"]

MAX_JOINTS = 16
"""Largest joint count whose per-joint loops Warp still unrolls."""

_BIG = 1.0e12


@functools.lru_cache(maxsize=None)
def build_topp_kernels(num_joints: int, num_sets: int, double: bool) -> tuple:
    """Compile the three parameterization kernels for one problem shape.

    Args:
        num_joints: Joints per path sample, at most :data:`MAX_JOINTS`.
        num_sets: Constraint rows per joint and stage; 1 for collocation, 2 for
            interpolation.
        double: Whether to compile for ``float64`` instead of ``float32``.

    Returns:
        The ``(rows, cap, sweep)`` kernels.
    """
    J = int(num_joints)
    G = int(num_sets)
    F = wp.float64 if double else wp.float32
    EPS = 1.0e-12 if double else 1.0e-7
    SQRT_FLOOR = 1.0e-12 if double else 1.0e-10
    TINY_SQUARE = 1.0e-30
    DENOMINATOR_FLOOR = 1.0e-12
    BIG = _BIG

    @wp.kernel(module="unique")
    def rows_kernel(
        qs: wp.array3d(dtype=F),
        qss: wp.array3d(dtype=F),
        ds: wp.array(dtype=F),
        vmax: wp.array(dtype=F),
        amax: wp.array(dtype=F),
        n: int,
        k: wp.array4d(dtype=F),
        c: wp.array4d(dtype=F),
        xcap: wp.array2d(dtype=F),
    ):
        b, i = wp.tid()
        cap = F(BIG)
        for j in range(J):
            a = qs[b, i, j]
            cap = wp.min(cap, vmax[j] * vmax[j] / wp.max(a * a, F(TINY_SQUARE)))
        for g in range(G):
            for j in range(J):
                a = qs[b, i, j]
                bb = qss[b, i, j]
                if g == 1 and i < n:
                    a = qs[b, i + 1, j] + F(2.0) * ds[i] * qss[b, i + 1, j]
                    bb = qss[b, i + 1, j]
                kv = F(0.0)
                cv = F(BIG)
                if wp.abs(a) > F(EPS):
                    kv = bb / a
                    cv = amax[j] / wp.abs(a)
                elif wp.abs(bb) > F(EPS):
                    cap = wp.min(cap, amax[j] / wp.abs(bb))
                k[b, i, g, j] = kv
                c[b, i, g, j] = cv
        xcap[b, i] = cap

    @wp.kernel(module="unique")
    def cap_kernel(
        k: wp.array4d(dtype=F),
        c: wp.array4d(dtype=F),
        xcap: wp.array2d(dtype=F),
        mvc: wp.array2d(dtype=F),
    ):
        b, i = wp.tid()
        cap = xcap[b, i]
        # Every lower acceleration bound must sit below every upper one.
        for g1 in range(G):
            for j1 in range(J):
                for g2 in range(G):
                    for j2 in range(J):
                        dk = k[b, i, g2, j2] - k[b, i, g1, j1]
                        if dk > F(EPS):
                            cap = wp.min(cap, (c[b, i, g2, j2] + c[b, i, g1, j1]) / dk)
        mvc[b, i] = cap

    @wp.kernel(module="unique")
    def sweep_kernel(
        k: wp.array4d(dtype=F),
        c: wp.array4d(dtype=F),
        mvc: wp.array2d(dtype=F),
        ds: wp.array(dtype=F),
        n: int,
        xmax: wp.array2d(dtype=F),
        x: wp.array2d(dtype=F),
        u: wp.array2d(dtype=F),
        seg: wp.array2d(dtype=F),
    ):
        b = wp.tid()
        # Backward: the largest squared speed at each stage from which some
        # admissible acceleration still lands inside the next stage's set.
        xmax[b, n] = F(0.0)
        for ii in range(n):
            i = n - 1 - ii
            two = F(2.0) * ds[i]
            nxt = xmax[b, i + 1]
            cap = mvc[b, i]
            for g in range(G):
                for j in range(J):
                    coef = F(1.0) - two * k[b, i, g, j]
                    if coef > F(EPS):
                        cap = wp.min(cap, (nxt + two * c[b, i, g, j]) / coef)
                    elif coef < -F(EPS):
                        cap = wp.min(cap, two * c[b, i, g, j] / (-coef))
            xmax[b, i] = wp.max(cap, F(0.0))
        # Forward: take the largest admissible acceleration at every stage.
        x[b, 0] = F(0.0)
        for i in range(n):
            two = F(2.0) * ds[i]
            xi = x[b, i]
            uhi = F(BIG)
            for g in range(G):
                for j in range(J):
                    uhi = wp.min(uhi, c[b, i, g, j] - k[b, i, g, j] * xi)
            ui = wp.min(uhi, (xmax[b, i + 1] - xi) / two)
            u[b, i] = ui
            x[b, i + 1] = wp.min(wp.max(xi + two * ui, F(0.0)), xmax[b, i + 1])
        for i in range(n):
            s0 = F(0.0)
            s1 = F(0.0)
            if x[b, i] > F(SQRT_FLOOR):
                s0 = wp.sqrt(x[b, i])
            if x[b, i + 1] > F(SQRT_FLOOR):
                s1 = wp.sqrt(x[b, i + 1])
            seg[b, i] = F(2.0) * ds[i] / wp.max(s0 + s1, F(DENOMINATOR_FLOOR))

    return rows_kernel, cap_kernel, sweep_kernel
