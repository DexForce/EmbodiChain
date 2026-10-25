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
# Three-parallel decomposition and singular wrist construction adapted from
# personalrobotics/ssik, commit 4109147f37a7ca3e2516c83853aca583612783f3.
# Its BSD-3-Clause notice is retained below. Polynomial elimination/root
# isolation and the Warp implementation are original to this integration.
#
# Copyright (c) 2026, Siddhartha Srinivasa. All rights reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Double precision geometric IK for a consecutive parallel-axis 6R chain.

The chain is normalized into home-frame axes H and segment vectors P:
``T = Trans(P0) Rot(H0,q0) Trans(P1) ... Rot(H5,q5) Trans(P6)``.
The caller removes the home tool rotation before invoking this module and
owns joint limits, periodic representatives, and final original-model FK
verification. Parallel axes may have either sign.

SP6 is reduced to a scalar trigonometric equation when its constant block
has rank one (including Aloha Mini and UR), or to a quartic otherwise.
Quartic roots are isolated between the derivative's algebraic roots; its
bounded scalar bisection is polynomial evaluation, never iterative IK.
"""

from __future__ import annotations

import warp as wp

__all__ = [
    "mat73d",
    "mat83d",
    "vec7d",
    "mat87d",
    "vec8i",
    "ssik_three_parallel",
    "ssik_three_parallel_limited",
]

mat73d = wp.types.matrix(shape=(7, 3), dtype=wp.float64)
mat83d = wp.types.matrix(shape=(8, 3), dtype=wp.float64)
vec7d = wp.types.vector(length=7, dtype=wp.float64)
mat87d = wp.types.matrix(shape=(8, 7), dtype=wp.float64)
vec8i = wp.types.vector(length=8, dtype=wp.int32)
_vec4d = wp.types.vector(length=4, dtype=wp.float64)
_vec5d = wp.types.vector(length=5, dtype=wp.float64)
_vec6d = wp.types.vector(length=6, dtype=wp.float64)
_mat42d = wp.types.matrix(shape=(4, 2), dtype=wp.float64)


@wp.func
def _row_h(h: mat73d, i: int) -> wp.vec3d:
    return wp.vec3d(h[i, 0], h[i, 1], h[i, 2])


@wp.func
def _row_p(p: mat83d, i: int) -> wp.vec3d:
    return wp.vec3d(p[i, 0], p[i, 1], p[i, 2])


@wp.func
def _rot(k: wp.vec3d, a: wp.float64) -> wp.mat33d:
    c = wp.cos(a)
    s = wp.sin(a)
    return wp.mat33d(
        c + k[0] * k[0] * (wp.float64(1.0) - c),
        k[0] * k[1] * (wp.float64(1.0) - c) - k[2] * s,
        k[0] * k[2] * (wp.float64(1.0) - c) + k[1] * s,
        k[1] * k[0] * (wp.float64(1.0) - c) + k[2] * s,
        c + k[1] * k[1] * (wp.float64(1.0) - c),
        k[1] * k[2] * (wp.float64(1.0) - c) - k[0] * s,
        k[2] * k[0] * (wp.float64(1.0) - c) - k[1] * s,
        k[2] * k[1] * (wp.float64(1.0) - c) + k[0] * s,
        c + k[2] * k[2] * (wp.float64(1.0) - c),
    )


@wp.func
def _wrap(a: wp.float64) -> wp.float64:
    return wp.atan2(wp.sin(a), wp.cos(a))


@wp.func
def _sp1(k: wp.vec3d, p: wp.vec3d, q: wp.vec3d) -> wp.float64:
    a = wp.cross(k, p)
    b = wp.cross(k, q)
    return wp.atan2(wp.dot(k, wp.cross(a, b)), wp.dot(a, b))


@wp.func
def _trig_roots_with_band(
    a: wp.float64, b: wp.float64, d: wp.float64, tangent_band: wp.float64
):
    """Return the two roots of a*cos(t)+b*sin(t)=d and a validity flag."""
    out = wp.vec2d(wp.float64(0.0))
    r = wp.sqrt(a * a + b * b)
    valid = int(0)
    rhs = d
    if r > wp.float64(1e-14):
        if wp.abs(wp.abs(d) - r) <= tangent_band:
            rhs = r
            if d < wp.float64(0.0):
                rhs = -r
        if wp.abs(rhs) <= r + wp.float64(1e-12) * wp.max(r, wp.float64(1.0)):
            delta = wp.acos(wp.clamp(rhs / r, wp.float64(-1.0), wp.float64(1.0)))
            phi = wp.atan2(b, a)
            out[0] = _wrap(phi + delta)
            out[1] = _wrap(phi - delta)
            valid = 1
    elif wp.abs(d) <= wp.float64(1e-12):
        valid = 1
    return out, valid


@wp.func
def _trig_roots(a: wp.float64, b: wp.float64, d: wp.float64):
    return _trig_roots_with_band(a, b, d, wp.float64(0.0))


@wp.func
def _elbows(k: wp.vec3d, p2: wp.vec3d, p3: wp.vec3d, inner: wp.vec3d):
    # Coordinate rounding perturbs a squared distance by |inner|*delta_inner.
    # Collapse either side of a numerically unresolved tangent; a fixed
    # absolute squared-distance allowance is unsafe for nearly folded arms.
    axial = -wp.dot(p2, k) * wp.dot(k, p3)
    a = -wp.dot(p2, p3) - axial
    b = -wp.dot(p2, wp.cross(k, p3))
    rhs = (wp.dot(p2, p2) + wp.dot(p3, p3) - wp.dot(inner, inner)) * wp.float64(
        0.5
    ) - axial
    band = wp.float64(1e-7) * wp.length(inner) + wp.float64(1e-14)
    return _trig_roots_with_band(a, b, rhs, band)


@wp.func
def _sp4(h: wp.vec3d, k: wp.vec3d, p: wp.vec3d, d: wp.float64):
    c = wp.dot(h, k) * wp.dot(k, p)
    return _trig_roots(wp.dot(h, p) - c, wp.dot(h, wp.cross(k, p)), d - c)


@wp.func
def _cbrt(x: wp.float64) -> wp.float64:
    y = wp.pow(wp.abs(x), wp.float64(1.0 / 3.0))
    if x < wp.float64(0.0):
        y = -y
    return y


@wp.func
def _cubic_real(a: wp.float64, b: wp.float64, c: wp.float64, d: wp.float64):
    """Sorted real roots of a cubic (including reduced quadratic cases)."""
    roots = wp.vec3d(wp.float64(0.0))
    n = int(0)
    if wp.abs(a) < wp.float64(1e-15):
        if wp.abs(b) < wp.float64(1e-15):
            if wp.abs(c) > wp.float64(1e-15):
                roots[0] = -d / c
                n = 1
        else:
            disc = c * c - wp.float64(4.0) * b * d
            if disc >= wp.float64(0.0):
                rt = wp.sqrt(disc)
                roots[0] = (-c - rt) / (wp.float64(2.0) * b)
                roots[1] = (-c + rt) / (wp.float64(2.0) * b)
                n = 2
    else:
        aa = b / a
        bb = c / a
        cc = d / a
        pp = bb - aa * aa / wp.float64(3.0)
        qq = wp.float64(2.0 / 27.0) * aa * aa * aa - aa * bb / wp.float64(3.0) + cc
        disc = qq * qq / wp.float64(4.0) + pp * pp * pp / wp.float64(27.0)
        if disc > wp.float64(0.0):
            rr = wp.sqrt(disc)
            roots[0] = (
                _cbrt(-qq / wp.float64(2.0) + rr)
                + _cbrt(-qq / wp.float64(2.0) - rr)
                - aa / wp.float64(3.0)
            )
            n = 1
        elif pp < wp.float64(-1e-30):
            radius = wp.float64(2.0) * wp.sqrt(-pp / wp.float64(3.0))
            arg = -qq / (wp.float64(2.0) * wp.sqrt(-pp * pp * pp / wp.float64(27.0)))
            phase = wp.acos(
                wp.clamp(arg, wp.float64(-1.0), wp.float64(1.0))
            ) / wp.float64(3.0)
            roots[0] = radius * wp.cos(phase) - aa / wp.float64(3.0)
            roots[1] = radius * wp.cos(
                phase + wp.float64(2.09439510239319549)
            ) - aa / wp.float64(3.0)
            roots[2] = radius * wp.cos(
                phase + wp.float64(4.18879020478639098)
            ) - aa / wp.float64(3.0)
            n = 3
        else:
            roots[0] = -aa / wp.float64(3.0)
            n = 1
    for i in range(3):
        for j in range(i + 1, 3):
            if j < n and roots[j] < roots[i]:
                tmp = roots[i]
                roots[i] = roots[j]
                roots[j] = tmp
    return roots, n


@wp.func
def _poly4(c: _vec5d, t: wp.float64) -> wp.float64:
    return (((c[4] * t + c[3]) * t + c[2]) * t + c[1]) * t + c[0]


@wp.func
def _quartic_real(c: _vec5d):
    roots = _vec4d(wp.float64(0.0))
    n = int(0)
    if wp.abs(c[4]) < wp.float64(1e-15):
        reduced, count = _cubic_real(c[3], c[2], c[1], c[0])
        for i in range(3):
            if i < count:
                roots[i] = reduced[i]
        n = count
    else:
        critical, count = _cubic_real(
            wp.float64(4.0) * c[4], wp.float64(3.0) * c[3], wp.float64(2.0) * c[2], c[1]
        )
        bound = wp.float64(1.0)
        for i in range(4):
            bound = wp.max(bound, wp.float64(1.0) + wp.abs(c[i] / c[4]))
        points = _vec5d(wp.float64(0.0))
        points[0] = -bound
        for i in range(3):
            if i < count:
                points[i + 1] = critical[i]
        points[count + 1] = bound
        for i in range(4):
            if i <= count:
                lo = points[i]
                hi = points[i + 1]
                flo = _poly4(c, lo)
                fhi = _poly4(c, hi)
                if (flo < wp.float64(0.0) and fhi > wp.float64(0.0)) or (
                    flo > wp.float64(0.0) and fhi < wp.float64(0.0)
                ):
                    for it in range(64):
                        mid = (lo + hi) * wp.float64(0.5)
                        fm = _poly4(c, mid)
                        if (fm < wp.float64(0.0)) == (flo < wp.float64(0.0)):
                            lo = mid
                            flo = fm
                        else:
                            hi = mid
                    x = (lo + hi) * wp.float64(0.5)
                    unique = bool(True)
                    for j in range(4):
                        if j < n and wp.abs(x - roots[j]) <= wp.float64(1e-9) * (
                            wp.float64(1.0) + wp.abs(x)
                        ):
                            unique = False
                    if unique and n < 4:
                        roots[n] = x
                        n = n + 1
        # A multiple root is a derivative root; inspect these explicitly.
        for i in range(3):
            if i < count:
                x = critical[i]
                scale = wp.float64(0.0)
                mag = wp.float64(1.0)
                for j in range(5):
                    scale = scale + wp.abs(c[j]) * mag
                    mag = mag * wp.abs(x)
                if (
                    wp.abs(_poly4(c, x))
                    <= wp.float64(2e-13) * wp.max(scale, wp.float64(1.0))
                    and n < 4
                ):
                    roots[n] = x
                    n = n + 1
    return roots, n


@wp.func
def _sp6(
    h: wp.vec3d,
    k0: wp.vec3d,
    k4: wp.vec3d,
    p16: wp.vec3d,
    p5: wp.vec3d,
    rh5: wp.vec3d,
    h5: wp.vec3d,
    d1: wp.float64,
    seed0: wp.float64,
):
    """Coupled shoulder/wrist roots, with an algebraic rank-one fast path."""
    a0 = wp.dot(h, p16) - wp.dot(h, k0) * wp.dot(k0, p16)
    a1 = wp.dot(h, wp.cross(-k0, p16))
    a2 = wp.dot(h, rh5) - wp.dot(h, k0) * wp.dot(k0, rh5)
    a3 = wp.dot(h, wp.cross(-k0, rh5))
    b0 = -wp.dot(h, p5) + wp.dot(h, k4) * wp.dot(k4, p5)
    b1 = -wp.dot(h, wp.cross(k4, p5))
    b2 = -wp.dot(h, h5) + wp.dot(h, k4) * wp.dot(k4, h5)
    b3 = -wp.dot(h, wp.cross(k4, h5))
    d0 = d1 - wp.dot(h, k0) * wp.dot(k0, p16) + wp.dot(h, k4) * wp.dot(k4, p5)
    d2 = -wp.dot(h, k0) * wp.dot(k0, rh5) + wp.dot(h, k4) * wp.dot(k4, h5)
    pairs = _mat42d(wp.float64(0.0))
    count = int(0)
    det = b0 * b3 - b1 * b2
    scale = wp.sqrt((b0 * b0 + b1 * b1) * (b2 * b2 + b3 * b3))
    if wp.abs(det) <= wp.float64(1e-12) * wp.max(scale, wp.float64(1.0)):
        # Eliminate the common cos/sin projection of theta4.
        u = b0
        v = b2
        if b1 * b1 + b3 * b3 > b0 * b0 + b2 * b2:
            u = b1
            v = b3
        # Target matrices may have been rounded to float32 before promotion.
        # At a tangent, treating opposite signs of that coordinate roundoff
        # differently creates both false infeasibility and split fake roots.
        shoulder_band = wp.float64(5e-8) * (wp.float64(1.0) + wp.length(p16))
        theta0, valid0 = _trig_roots_with_band(
            v * a0 - u * a2, v * a1 - u * a3, v * d0 - u * d2, shoulder_band
        )
        if wp.abs(v * a0 - u * a2) + wp.abs(v * a1 - u * a3) < wp.float64(1e-14):
            theta0[0] = seed0
            theta0[1] = seed0 + wp.float64(3.141592653589793)
        if valid0 == 1:
            for i in range(2):
                t0 = theta0[i]
                e0 = d0 - a0 * wp.cos(t0) - a1 * wp.sin(t0)
                e2 = d2 - a2 * wp.cos(t0) - a3 * wp.sin(t0)
                t4, valid4 = _trig_roots(b0, b1, e0)
                if b2 * b2 + b3 * b3 > b0 * b0 + b1 * b1:
                    t4, valid4 = _trig_roots(b2, b3, e2)
                if valid4 == 1:
                    for j in range(2):
                        pairs[count, 0] = _wrap(t0)
                        pairs[count, 1] = t4[j]
                        count = count + 1
    else:
        # [cos(theta4),sin(theta4)] = U*cos(root0)+V*sin(root0)+W.
        uvec = wp.vec2d((-b3 * a0 + b1 * a2) / det, (b2 * a0 - b0 * a2) / det)
        vvec = wp.vec2d((-b3 * a1 + b1 * a3) / det, (b2 * a1 - b0 * a3) / det)
        wvec = wp.vec2d((b3 * d0 - b1 * d2) / det, (-b2 * d0 + b0 * d2) / det)
        # Choose a tan-half-angle chart with a large leading coefficient.
        shift = wp.float64(0.0)
        best = wp.float64(-1.0)
        for i in range(4):
            angle = wp.float64(i) * wp.float64(1.5707963267948966)
            test = -wp.cos(angle) * uvec - wp.sin(angle) * vvec + wvec
            score = wp.abs(wp.dot(test, test) - wp.float64(1.0))
            if score > best:
                shift = angle
                best = score
        uu = wp.cos(shift) * uvec + wp.sin(shift) * vvec
        vv = -wp.sin(shift) * uvec + wp.cos(shift) * vvec
        g0 = uu + wvec
        g1 = wp.float64(2.0) * vv
        g2 = wvec - uu
        coeff = _vec5d(
            wp.dot(g0, g0) - wp.float64(1.0),
            wp.float64(2.0) * wp.dot(g0, g1),
            wp.dot(g1, g1) + wp.float64(2.0) * wp.dot(g0, g2) - wp.float64(2.0),
            wp.float64(2.0) * wp.dot(g1, g2),
            wp.dot(g2, g2) - wp.float64(1.0),
        )
        coeff_scale = wp.float64(0.0)
        for i in range(5):
            coeff_scale = wp.max(coeff_scale, wp.abs(coeff[i]))
        if coeff_scale > wp.float64(1e-15):
            coeff = coeff / coeff_scale
            roots, nroots = _quartic_real(coeff)
            for i in range(4):
                if i < nroots:
                    root0 = _wrap(wp.float64(2.0) * wp.atan(roots[i]) + shift)
                    cs = uvec * wp.cos(root0) + vvec * wp.sin(root0) + wvec
                    if wp.abs(wp.dot(cs, cs) - wp.float64(1.0)) < wp.float64(1e-7):
                        pairs[count, 0] = root0
                        pairs[count, 1] = wp.atan2(cs[1], cs[0])
                        count = count + 1
    return pairs, count


@wp.func
def ssik_three_parallel(
    h: mat73d,
    p: mat83d,
    target_position: wp.vec3d,
    target_rotation: wp.mat33d,
    seed: vec7d,
):
    """Return eight padded 7-joint candidates and geometric validity flags.

    ``target_rotation`` has the home tool rotation removed. Candidate rows
    are physical joint angles, including anti-parallel trio sign correction.
    Final feasibility requires the caller's limits and independent FK gate.
    """
    out = mat87d(wp.float64(0.0))
    valid = vec8i(int(0))
    h0 = _row_h(h, 0)
    hh = _row_h(h, 1)
    h4 = _row_h(h, 4)
    h5 = _row_h(h, 5)
    p1 = _row_p(p, 1)
    p2 = _row_p(p, 2)
    p3 = _row_p(p, 3)
    p4 = _row_p(p, 4)
    p5 = _row_p(p, 5)
    p16 = target_position - _row_p(p, 0) - target_rotation * _row_p(p, 6)
    d1 = wp.dot(hh, p1 + p2 + p3 + p4)
    pairs, count = _sp6(hh, h0, h4, p16, p5, target_rotation * h5, h5, d1, seed[0])
    sign2 = wp.float64(1.0)
    sign3 = wp.float64(1.0)
    if wp.dot(_row_h(h, 2), hh) < wp.float64(0.0):
        sign2 = wp.float64(-1.0)
    if wp.dot(_row_h(h, 3), hh) < wp.float64(0.0):
        sign3 = wp.float64(-1.0)
    for i in range(4):
        if i < count:
            q0 = pairs[i, 0]
            q4 = pairs[i, 1]
            r01 = _rot(h0, q0)
            r45 = _rot(h4, q4)
            wrist_axis = r45 * h5
            theta = _sp1(hh, wrist_axis, wp.transpose(r01) * target_rotation * h5)
            q5 = _sp1(
                -h5, wp.transpose(r45) * hh, wp.transpose(target_rotation) * r01 * hh
            )
            # At a wrist lock the two SP1 angles are individually undefined.
            # Construct seed-near finite representatives of that continuum.
            if wp.length(wp.cross(hh, wrist_axis)) < wp.float64(1e-7):
                axis_sign = wp.float64(1.0)
                if wp.dot(hh, wrist_axis) < wp.float64(0.0):
                    axis_sign = wp.float64(-1.0)
                q4 = _wrap(q4 + _sp1(h4, wrist_axis, axis_sign * hh))
                r45 = _rot(h4, q4)
                pan_roots, pan_valid = _sp4(hh, -h0, p16, d1 + wp.dot(hh, r45 * p5))
                if pan_valid == 1:
                    q0_new = pan_roots[0]
                    if wp.abs(_wrap(pan_roots[1] - q0)) < wp.abs(_wrap(q0_new - q0)):
                        q0_new = pan_roots[1]
                    q0 = q0_new
                    r01 = _rot(h0, q0)
                # A duplicate tangent root supplies the other wrist side.
                q5 = seed[5] + wp.float64(i % 2) * wp.float64(3.141592653589793)
                m = (
                    wp.transpose(r01)
                    * target_rotation
                    * wp.transpose(_rot(h5, q5))
                    * wp.transpose(r45)
                )
                theta = _sp1(hh, h4, m * h4)
                arm_target = wp.transpose(r01) * p16 - p1
                wrist_offset = r45 * p5 + p4
                inner = arm_target - _rot(hh, theta) * wrist_offset
                distance_sq = wp.dot(inner, inner)
                # The planar two-link arm reaches a closed distance interval.
                # Project an infeasible seed wrist onto its analytical boundary,
                # rather than sampling and missing arbitrarily narrow intervals.
                arm_center = (
                    wp.dot(p2, p2)
                    + wp.dot(p3, p3)
                    + wp.float64(2.0) * wp.dot(hh, p2) * wp.dot(hh, p3)
                )
                arm_radius = (
                    wp.float64(2.0)
                    * wp.length(wp.cross(hh, p2))
                    * wp.length(wp.cross(hh, p3))
                )
                near_distance = wp.clamp(
                    distance_sq,
                    wp.max(wp.float64(0.0), arm_center - arm_radius),
                    arm_center + arm_radius,
                )
                if wp.abs(near_distance - distance_sq) > wp.float64(1e-14):
                    bound_angles, bound_valid = _sp4(
                        arm_target,
                        hh,
                        wrist_offset,
                        (
                            wp.dot(arm_target, arm_target)
                            + wp.dot(wrist_offset, wrist_offset)
                            - near_distance
                        )
                        * wp.float64(0.5),
                    )
                    if bound_valid == 1:
                        closest = bound_angles[0]
                        if wp.abs(_wrap(bound_angles[1] - theta)) < wp.abs(
                            _wrap(closest - theta)
                        ):
                            closest = bound_angles[1]
                        theta = closest
                        residual_rotation = (
                            wp.transpose(r45)
                            * wp.transpose(_rot(hh, theta))
                            * wp.transpose(r01)
                            * target_rotation
                        )
                        q5 = _sp1(h5, h4, residual_rotation * h4)
            r14 = _rot(hh, theta)
            inner = wp.transpose(r01) * p16 - p1 - r14 * r45 * p5 - r14 * p4
            elbows, feasible = _elbows(hh, p2, p3, inner)
            if feasible == 1:
                for j in range(2):
                    q2 = elbows[j]
                    q1 = _sp1(hh, p2 + _rot(hh, q2) * p3, inner)
                    row = i * 2 + j
                    out[row, 0] = _wrap(q0)
                    out[row, 1] = _wrap(q1)
                    out[row, 2] = _wrap(sign2 * q2)
                    out[row, 3] = _wrap(sign3 * (theta - q1 - q2))
                    out[row, 4] = _wrap(q4)
                    out[row, 5] = _wrap(q5)
                    valid[row] = 1
    return out, valid


@wp.func
def ssik_three_parallel_limited(
    h: mat73d,
    p: mat83d,
    target_position: wp.vec3d,
    target_rotation: wp.mat33d,
    seed: vec7d,
    lower: vec7d,
    upper: vec7d,
):
    """Recover a near-singular branch lying just outside its wrist limit.

    The ordinary geometric candidates are preserved whenever their wrist
    coordinate is in range. Only a wrist excess of at most 1e-3 radians is
    eligible: fixing that coordinate at the limit gives an SP4 for shoulder
    pan, followed by the same analytical wrist and elbow decomposition.
    A shoulder coordinate outside its limit by at most 1e-5 radians may
    similarly be fixed and the remaining planar two-link geometry recovered.
    These constructions replace otherwise invalid representatives of an
    ill-conditioned branch. The caller must still enforce every limit and the original FK
    residual at returned precision; this construction cannot accept a pose.
    """
    out, valid = ssik_three_parallel(h, p, target_position, target_rotation, seed)
    h0 = _row_h(h, 0)
    hh = _row_h(h, 1)
    h4 = _row_h(h, 4)
    h5 = _row_h(h, 5)
    p1 = _row_p(p, 1)
    p2 = _row_p(p, 2)
    p3 = _row_p(p, 3)
    p4 = _row_p(p, 4)
    p5 = _row_p(p, 5)
    p16 = target_position - _row_p(p, 0) - target_rotation * _row_p(p, 6)
    sign2 = wp.float64(1.0)
    sign3 = wp.float64(1.0)
    if wp.dot(_row_h(h, 2), hh) < wp.float64(0.0):
        sign2 = wp.float64(-1.0)
    if wp.dot(_row_h(h, 3), hh) < wp.float64(0.0):
        sign3 = wp.float64(-1.0)
    for i in range(4):
        row = i * 2
        if valid[row] == 1:
            wrist = out[row, 4]
            first_turn = wp.ceil((lower[4] - wrist) / wp.float64(6.283185307179586))
            last_turn = wp.floor((upper[4] - wrist) / wp.float64(6.283185307179586))
            # Compare a periodic representative, just as the caller does.
            turn = wp.round((seed[4] - wrist) / wp.float64(6.283185307179586))
            wrist = wrist + turn * wp.float64(6.283185307179586)
            projected = wp.clamp(wrist, lower[4], upper[4])
            excess = wp.abs(wrist - projected)
            if (
                first_turn > last_turn
                and excess > wp.float64(0.0)
                and excess <= wp.float64(1e-3)
            ):
                r45 = _rot(h4, projected)
                pans, feasible = _sp4(
                    hh, -h0, target_rotation * h5, wp.dot(hh, r45 * h5)
                )
                if feasible == 1:
                    pan = pans[0]
                    if wp.abs(_wrap(pans[1] - out[row, 0])) < wp.abs(
                        _wrap(pan - out[row, 0])
                    ):
                        pan = pans[1]
                    r01 = _rot(h0, pan)
                    theta = _sp1(hh, r45 * h5, wp.transpose(r01) * target_rotation * h5)
                    last = _sp1(
                        -h5,
                        wp.transpose(r45) * hh,
                        wp.transpose(target_rotation) * r01 * hh,
                    )
                    r14 = _rot(hh, theta)
                    inner = wp.transpose(r01) * p16 - p1 - r14 * r45 * p5 - r14 * p4
                    elbows, elbow_valid = _elbows(hh, p2, p3, inner)
                    if elbow_valid == 1:
                        for j in range(2):
                            elbow = elbows[j]
                            shoulder = _sp1(hh, p2 + _rot(hh, elbow) * p3, inner)
                            out[row + j, 0] = _wrap(pan)
                            out[row + j, 1] = _wrap(shoulder)
                            out[row + j, 2] = _wrap(sign2 * elbow)
                            out[row + j, 3] = _wrap(sign3 * (theta - shoulder - elbow))
                            out[row + j, 4] = _wrap(projected)
                            out[row + j, 5] = _wrap(last)
    for row in range(8):
        if valid[row] == 1:
            shoulder = out[row, 1]
            first_turn = wp.ceil((lower[1] - shoulder) / wp.float64(6.283185307179586))
            last_turn = wp.floor((upper[1] - shoulder) / wp.float64(6.283185307179586))
            turn = wp.round((seed[1] - shoulder) / wp.float64(6.283185307179586))
            shoulder = shoulder + turn * wp.float64(6.283185307179586)
            bounded = wp.clamp(shoulder, lower[1], upper[1])
            excess = wp.abs(shoulder - bounded)
            if (
                first_turn > last_turn
                and excess > wp.float64(0.0)
                and excess <= wp.float64(1e-5)
            ):
                theta = out[row, 1] + sign2 * out[row, 2] + sign3 * out[row, 3]
                r01 = _rot(h0, out[row, 0])
                r45 = _rot(h4, out[row, 4])
                r14 = _rot(hh, theta)
                inner = wp.transpose(r01) * p16 - p1 - r14 * r45 * p5 - r14 * p4
                elbow_target = _rot(hh, -bounded) * inner - p2
                elbow = _sp1(hh, p3, elbow_target)
                out[row, 1] = _wrap(bounded)
                out[row, 2] = _wrap(sign2 * elbow)
                out[row, 3] = _wrap(sign3 * (theta - bounded - elbow))
    return out, valid
