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
# Portions adapted from SSIK, revision 4109147f37a7ca3e2516c83853aca583612783f3.
# Copyright (c) 2026, Siddhartha Srinivasa. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
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

"""Cached per-target SRS geometry and one-branch reconstruction.

The circle, elbow roots and home-frame bases are invariant during redundancy
search. They are stored in a bounded caller-owned array; individual branches
never materialize the eight-by-seven candidate matrix on a CUDA thread stack.
"""

from __future__ import annotations

import warp as wp

from .ssik_7r import (
    _angle_about,
    _mat23d,
    _min_rotation,
    _rotation,
    _swivel_basis,
    _unit_perp,
    seed_swivel,
    vec7d,
)
from .ssik_srs_search import _srs_search_target

wp.set_module_options({"enable_backward": False})

__all__: list[str] = []

# One packed row per target avoids passing a large value struct through every
# scalar-search call. Layout (float64 columns): valid/collapsed[0:2], segment
# lengths/circle/distance[2:7], upper-unit/SW/direction/basis[7:22], target-home
# rotation[22:31], seed rotation[31:40], elbow roots[40:42], forearm[42:45],
# the two home-frame bases[45:57], elbow rotations[57:75], collapsed shoulder
# bases[75:93], projected seed[93:100], optimum/defined[100:102], cached seed
# sine/cosine[102:106], and the original seed swivel/defined[106:108].
CONTEXT_SIZE = 112


@wp.func
def _rotate_cs(
    axis: wp.vec3d, vector: wp.vec3d, cosine: wp.float64, sine: wp.float64
) -> wp.vec3d:
    return (
        cosine * vector
        + sine * wp.cross(axis, vector)
        + (wp.float64(1) - cosine) * wp.dot(axis, vector) * axis
    )


@wp.func
def _decompose_cached(
    rotation: wp.mat33d,
    first: wp.vec3d,
    middle: wp.vec3d,
    last: wp.vec3d,
    seed_first: wp.float64,
    seed_sine: wp.float64,
    seed_cosine: wp.float64,
) -> tuple[_mat23d, bool]:
    """The same Davenport roots using algebraic sine/cosine and vector actions.

    The center/offset angles and gimbal threshold match the reference function.
    Their known sine/cosine avoids reevaluating trigonometry for both middle
    roots; Rodrigues vector products replace three-by-three middle rotations.
    """
    result = _mat23d()
    fm, ml = wp.dot(first, middle), wp.dot(middle, last)
    a = wp.dot(first, last) - fm * ml
    b = wp.dot(first, wp.cross(middle, last))
    target_last = rotation * last
    c = wp.dot(first, target_last) - fm * ml
    amplitude = wp.sqrt(a * a + b * b)
    valid = amplitude > wp.float64(1e-12)
    if valid:
        valid = wp.abs(c) <= amplitude + wp.float64(5e-7)
        center = wp.atan2(b, a)
        direction = target_last
        norm = wp.length(direction)
        if norm > wp.float64(1e-12):
            direction /= norm
        projection = wp.dot(first, direction)
        cross = wp.cross(first, direction)
        discriminant = (
            wp.dot(cross, cross)
            - fm * fm
            - ml * ml
            + wp.float64(2) * fm * ml * projection
        )
        sine_offset = wp.sqrt(wp.max(discriminant, wp.float64(0)))
        cosine_offset = projection - fm * ml
        offset = wp.atan2(sine_offset, cosine_offset)
        denominator = amplitude * wp.sqrt(
            sine_offset * sine_offset + cosine_offset * cosine_offset
        )
        inverse_first = wp.transpose(rotation) * first
        for branch in range(2):
            sign = wp.float64(1)
            if branch == 1:
                sign = wp.float64(-1)
            angle_middle = center + sign * offset
            cosine = (a * cosine_offset - sign * b * sine_offset) / denominator
            sine = (b * cosine_offset + sign * a * sine_offset) / denominator
            mapped_last = _rotate_cs(middle, last, cosine, sine)
            inverse_mapped_first = _rotate_cs(middle, first, cosine, -sine)
            angle_first = _angle_about(first, mapped_last, target_last)
            angle_last = _angle_about(last, inverse_first, inverse_mapped_first)
            mapped_perp = wp.length(mapped_last - first * wp.dot(first, mapped_last))
            target_perp = wp.length(target_last - first * wp.dot(first, target_last))
            if mapped_perp < wp.float64(1e-6) or target_perp < wp.float64(1e-6):
                angle_first = seed_first
                perpendicular = _unit_perp(last)
                vector = _rotate_cs(
                    first, rotation * perpendicular, seed_cosine, -seed_sine
                )
                vector = _rotate_cs(middle, vector, cosine, -sine)
                angle_last = _angle_about(last, perpendicular, vector)
            result[branch, 0] = angle_first
            result[branch, 1] = angle_middle
            result[branch, 2] = angle_last
    return result, valid


@wp.func
def _decompose_one(
    rotation: wp.mat33d,
    first: wp.vec3d,
    middle: wp.vec3d,
    last: wp.vec3d,
    seed_first: wp.float64,
    seed_sine: wp.float64,
    seed_cosine: wp.float64,
    branch: int,
) -> tuple[wp.vec3d, bool]:
    """One requested Davenport sheet, without computing its unused sibling.

    The center/offset angles and gimbal threshold match the reference function.
    Their known sine/cosine avoids reevaluating trigonometry for both middle
    roots; Rodrigues vector products replace three-by-three middle rotations.
    """
    result = wp.vec3d()
    fm, ml = (wp.dot(first, middle), wp.dot(middle, last))
    a = wp.dot(first, last) - fm * ml
    b = wp.dot(first, wp.cross(middle, last))
    target_last = rotation * last
    c = wp.dot(first, target_last) - fm * ml
    amplitude = wp.sqrt(a * a + b * b)
    valid = amplitude > wp.float64(1e-12)
    if valid:
        valid = wp.abs(c) <= amplitude + wp.float64(5e-07)
        center = wp.atan2(b, a)
        direction = target_last
        norm = wp.length(direction)
        if norm > wp.float64(1e-12):
            direction /= norm
        projection = wp.dot(first, direction)
        cross = wp.cross(first, direction)
        discriminant = (
            wp.dot(cross, cross)
            - fm * fm
            - ml * ml
            + wp.float64(2) * fm * ml * projection
        )
        sine_offset = wp.sqrt(wp.max(discriminant, wp.float64(0)))
        cosine_offset = projection - fm * ml
        offset = wp.atan2(sine_offset, cosine_offset)
        denominator = amplitude * wp.sqrt(
            sine_offset * sine_offset + cosine_offset * cosine_offset
        )
        sign = wp.float64(1)
        if branch == 1:
            sign = wp.float64(-1)
        angle_middle = center + sign * offset
        cosine = (a * cosine_offset - sign * b * sine_offset) / denominator
        sine = (b * cosine_offset + sign * a * sine_offset) / denominator
        mapped_last = _rotate_cs(middle, last, cosine, sine)
        mapped_perp = wp.length(mapped_last - first * wp.dot(first, mapped_last))
        target_perp = wp.length(target_last - first * wp.dot(first, target_last))
        if mapped_perp < wp.float64(1e-06) or target_perp < wp.float64(1e-06):
            angle_first = seed_first
            perpendicular = _unit_perp(last)
            vector = _rotate_cs(
                first, rotation * perpendicular, seed_cosine, -seed_sine
            )
            vector = _rotate_cs(middle, vector, cosine, -sine)
            angle_last = _angle_about(last, perpendicular, vector)
        else:
            inverse_first = wp.transpose(rotation) * first
            inverse_mapped_first = _rotate_cs(middle, first, cosine, -sine)
            angle_first = _angle_about(first, mapped_last, target_last)
            angle_last = _angle_about(last, inverse_first, inverse_mapped_first)
        result[0] = angle_first
        result[1] = angle_middle
        result[2] = angle_last
    return (result, valid)


@wp.func
def _v(context: wp.array2d(dtype=wp.float64), i: int, offset: int) -> wp.vec3d:
    return wp.vec3d(context[i, offset], context[i, offset + 1], context[i, offset + 2])


@wp.func
def _m(context: wp.array2d(dtype=wp.float64), i: int, offset: int) -> wp.mat33d:
    result = wp.mat33d()
    for j in range(3):
        for k in range(3):
            result[j, k] = context[i, offset + 3 * j + k]
    return result


@wp.func
def _put_v(context: wp.array2d(dtype=wp.float64), i: int, offset: int, value: wp.vec3d):
    for j in range(3):
        context[i, offset + j] = value[j]


@wp.func
def _put_m(
    context: wp.array2d(dtype=wp.float64), i: int, offset: int, value: wp.mat33d
):
    for j in range(3):
        for k in range(3):
            context[i, offset + 3 * j + k] = value[j, k]


@wp.func
def _prepare_problem(
    axes: wp.array(dtype=wp.vec3d),
    offsets: wp.array(dtype=wp.vec3d),
    home: wp.mat33d,
    shoulder: wp.vec3d,
    wrist: wp.vec3d,
    goal: wp.mat44d,
    seed: vec7d,
    finite: bool,
    context: wp.array2d(dtype=wp.float64),
    i: int,
):
    for j in range(CONTEXT_SIZE):
        context[i, j] = wp.float64(0)
    rotation = wp.mat33d()
    home_position = wp.vec3d()
    for j in range(8):
        home_position += offsets[j]
    for j in range(3):
        for k in range(3):
            rotation[j, k] = goal[j, k]
    elbow = offsets[0] + offsets[1] + offsets[2] + offsets[3]
    upper, forearm = elbow - shoulder, wrist - elbow
    ul, fl = wp.length(upper), wp.length(forearm)
    sw = (
        wp.vec3d(goal[0, 3], goal[1, 3], goal[2, 3])
        - rotation * (wp.transpose(home) * (home_position - wrist))
        - shoulder
    )
    distance = wp.length(sw)
    valid = (
        finite
        and ul > wp.float64(1e-12)
        and fl > wp.float64(1e-12)
        and distance > wp.float64(1e-12)
    )
    valid = (
        valid
        and distance <= ul + fl + wp.float64(1e-7)
        and distance >= wp.abs(ul - fl) - wp.float64(1e-7)
    )
    for j in range(7):
        context[i, 93 + j] = seed[j]
    context[i, 102] = wp.sin(seed[0])
    context[i, 103] = wp.cos(seed[0])
    context[i, 104] = wp.sin(seed[4])
    context[i, 105] = wp.cos(seed[4])
    phi_seed, phi_defined = seed_swivel(axes, offsets, home, shoulder, wrist, seed)
    context[i, 106] = phi_seed
    context[i, 107] = wp.float64(phi_defined)
    if valid:
        direction = sw / distance
        x = (ul * ul - fl * fl + distance * distance) / (wp.float64(2) * distance)
        radius = wp.sqrt(wp.max(ul * ul - x * x, wp.float64(0)))
        if wp.abs(distance - ul - fl) <= wp.float64(1e-13):
            x, radius = ul, wp.float64(0)
        elif wp.abs(distance - wp.abs(ul - fl)) <= wp.float64(1e-13):
            x = ul
            if ul < fl:
                x = -ul
            radius = wp.float64(0)
        first, second = _swivel_basis(direction)
        uh = upper / ul
        hk, kp = wp.dot(uh, axes[3]), wp.dot(axes[3], forearm)
        a = wp.dot(uh, forearm) - hk * kp
        b = wp.dot(uh, wp.cross(axes[3], forearm))
        c = (x * distance - ul * ul) / ul - hk * kp
        amplitude = wp.sqrt(a * a + b * b)
        valid = (
            valid
            and amplitude > wp.float64(1e-12)
            and wp.abs(c) <= amplitude + wp.float64(1e-7)
        )
        context[i, 1] = wp.float64(radius == wp.float64(0))
        context[i, 2] = ul
        context[i, 3] = fl
        context[i, 4] = x
        context[i, 5] = radius
        context[i, 6] = distance
        _put_v(context, i, 7, uh)
        _put_v(context, i, 10, sw)
        _put_v(context, i, 13, direction)
        _put_v(context, i, 16, first)
        _put_v(context, i, 19, second)
        _put_v(context, i, 42, forearm)
        _put_m(context, i, 22, rotation * wp.transpose(home))
        seed_rotation = (
            _rotation(axes[0], seed[0])
            * _rotation(axes[1], seed[1])
            * _rotation(axes[2], seed[2])
        )
        _put_m(context, i, 31, seed_rotation)
        if valid:
            center = wp.atan2(b, a)
            offset = wp.acos(wp.clamp(c / amplitude, wp.float64(-1), wp.float64(1)))
            if radius == wp.float64(0):
                if wp.abs(c / amplitude - wp.float64(1)) <= wp.float64(1e-13):
                    offset = wp.float64(0)
                elif wp.abs(c / amplitude + wp.float64(1)) <= wp.float64(1e-13):
                    offset = wp.float64(3.141592653589793)
            for branch in range(2):
                angle = center + offset
                if branch == 1:
                    angle = center - offset
                context[i, 40 + branch] = angle
                re = _rotation(axes[3], angle)
                _put_m(context, i, 57 + branch * 9, re)
                f = re * forearm
                p = f - uh * wp.dot(uh, f)
                norm = wp.length(p)
                if norm > wp.float64(1e-9):
                    p /= norm
                _put_v(context, i, 45 + branch * 6, p)
                _put_v(context, i, 48 + branch * 6, wp.cross(uh, p))
                if radius == wp.float64(0):
                    u_target = x * direction
                    d = u_target / wp.length(u_target)
                    r0 = _min_rotation(upper, u_target)
                    perpendicular = _unit_perp(uh)
                    roll = _angle_about(
                        d, r0 * perpendicular, seed_rotation * perpendicular
                    )
                    _put_m(context, i, 75 + branch * 9, _rotation(d, roll) * r0)
    context[i, 0] = wp.float64(valid)


@wp.kernel
def prepare_srs_context(
    axes: wp.array(dtype=wp.vec3d),
    offsets: wp.array(dtype=wp.vec3d),
    home: wp.mat33d,
    shoulder: wp.vec3d,
    wrist: wp.vec3d,
    targets: wp.array(dtype=wp.mat44d),
    inverse_tcp: wp.mat44d,
    seeds: wp.array2d(dtype=wp.float64),
    lower: wp.array(dtype=wp.float64),
    upper: wp.array(dtype=wp.float64),
    position_tolerance: wp.float64,
    prepare_local: int,
    context: wp.array2d(dtype=wp.float64),
):
    """Prepare one row for global samples and an optional projected local row."""
    i = wp.tid()
    seed = vec7d()
    finite = bool(True)
    for j in range(7):
        value = seeds[i, j]
        finite = finite and wp.isfinite(value)
        if not wp.isfinite(value):
            value = (lower[j] + upper[j]) * wp.float64(0.5)
        seed[j] = wp.clamp(value, lower[j], upper[j])
    target = targets[i]
    goal = target @ inverse_tcp
    rotation = wp.mat33d()
    for j in range(4):
        for k in range(4):
            finite = finite and wp.isfinite(target[j, k])
            if j < 3 and k < 3:
                rotation[j, k] = goal[j, k]
    gram = wp.transpose(rotation) * rotation
    for j in range(3):
        for k in range(3):
            expected = wp.float64(0)
            if j == k:
                expected = wp.float64(1)
            finite = finite and wp.abs(gram[j, k] - expected) <= wp.float64(1e-5)
        finite = finite and wp.abs(target[3, j]) <= wp.float64(1e-6)
    finite = (
        finite
        and wp.abs(target[3, 3] - wp.float64(1)) <= wp.float64(1e-6)
        and wp.determinant(rotation) > wp.float64(0)
    )
    if not finite:
        goal = wp.mat44d(wp.float64(1))
    _prepare_problem(
        axes, offsets, home, shoulder, wrist, goal, seed, finite, context, i
    )
    if prepare_local != 0:
        projected, within = _srs_search_target(
            axes, offsets, home, shoulder, wrist, goal, position_tolerance
        )
        _prepare_problem(
            axes,
            offsets,
            home,
            shoulder,
            wrist,
            projected,
            seed,
            finite and within,
            context,
            i + targets.shape[0],
        )


@wp.func
def _srs_shoulder(
    context: wp.array2d(dtype=wp.float64),
    i: int,
    elbow: int,
    phi: wp.float64,
    roll: wp.float64,
) -> wp.mat33d:
    direction = _v(context, i, 13)
    if context[i, 1] != wp.float64(0):
        d = direction
        if context[i, 4] < wp.float64(0):
            d = -d
        return _rotation(d, roll) * _m(context, i, 75 + elbow * 9)
    upper_target = context[i, 4] * direction + context[i, 5] * (
        wp.cos(phi) * _v(context, i, 16) + wp.sin(phi) * _v(context, i, 19)
    )
    d = upper_target / wp.length(upper_target)
    vector = _v(context, i, 10) - upper_target
    vp = vector - d * wp.dot(d, vector)
    vn = wp.length(vp)
    hp = _v(context, i, 45 + elbow * 6)
    if vn > wp.float64(1e-9) and wp.length(hp) > wp.float64(1e-9):
        vp /= vn
        up = _v(context, i, 7)
        vc = wp.cross(d, vp)
        hc = _v(context, i, 48 + elbow * 6)
        result = wp.mat33d()
        for row in range(3):
            for column in range(3):
                result[row, column] = (
                    d[row] * up[column] + vp[row] * hp[column] + vc[row] * hc[column]
                )
        return result
    r0 = _min_rotation(_v(context, i, 7), upper_target)
    perpendicular = _unit_perp(_v(context, i, 7))
    angle = _angle_about(d, r0 * perpendicular, _m(context, i, 31) * perpendicular)
    return _rotation(d, angle) * r0


@wp.func
def srs_context_branch(
    context: wp.array2d(dtype=wp.float64),
    i: int,
    axes: wp.array(dtype=wp.vec3d),
    phi: wp.float64,
    roll: wp.float64,
    branch: int,
) -> tuple[vec7d, bool]:
    """Reconstruct one conditional branch without an eight-branch local array."""
    result = vec7d()
    valid = context[i, 0] != wp.float64(0)
    if valid:
        elbow, shoulder, wrist = branch // 4, (branch // 2) % 2, branch % 2
        rsh = _srs_shoulder(context, i, elbow, phi, roll)
        residual = wp.transpose(rsh * _m(context, i, 57 + elbow * 9)) * _m(
            context, i, 22
        )
        sha, shv = _decompose_one(
            rsh,
            axes[0],
            axes[1],
            axes[2],
            context[i, 93],
            context[i, 102],
            context[i, 103],
            shoulder,
        )
        wra, wrv = _decompose_one(
            residual,
            axes[4],
            axes[5],
            axes[6],
            context[i, 97],
            context[i, 104],
            context[i, 105],
            wrist,
        )
        for j in range(3):
            result[j] = sha[j]
            result[j + 4] = wra[j]
        result[3] = context[i, 40 + elbow]
        valid = valid and shv and wrv
    return result, valid


@wp.kernel
def reconstruct_srs_context(
    context: wp.array2d(dtype=wp.float64),
    axes: wp.array(dtype=wp.vec3d),
    phi: wp.array(dtype=wp.float64),
    roll: wp.array(dtype=wp.float64),
    raw_valid: wp.array2d(dtype=wp.int32),
    raw: wp.array3d(dtype=wp.float64),
):
    """Independent branch fixture and cross-backend reconstruction entry."""
    i, branch = wp.tid()
    q, valid = srs_context_branch(context, i, axes, phi[i], roll[i], branch)
    raw_valid[i, branch] = wp.int32(valid)
    for j in range(7):
        raw[i, branch, j] = q[j]
