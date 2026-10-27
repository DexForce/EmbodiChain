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

"""Double precision, fixed-swivel geometry for concurrent-axis 7R arms.

One invocation returns the eight elbow/shoulder/wrist branches for a single
swivel. The caller owns parallelism over targets and swivels, joint-limit
handling and final FK acceptance. This module performs no iterative IK.
"""

from __future__ import annotations

import warp as wp

__all__ = ["mat87d", "seed_swivel", "solve_srs", "vec7d", "vec8i"]

vec7d = wp.types.vector(length=7, dtype=wp.float64)
vec8i = wp.types.vector(length=8, dtype=wp.int32)
mat87d = wp.types.matrix(shape=(8, 7), dtype=wp.float64)
_mat23d = wp.types.matrix(shape=(2, 3), dtype=wp.float64)


@wp.func
def _identity() -> wp.mat33d:
    return wp.mat33d(
        wp.float64(1),
        wp.float64(0),
        wp.float64(0),
        wp.float64(0),
        wp.float64(1),
        wp.float64(0),
        wp.float64(0),
        wp.float64(0),
        wp.float64(1),
    )


@wp.func
def _rotation(axis: wp.vec3d, angle: wp.float64) -> wp.mat33d:
    """Rodrigues rotation about a normalized axis."""
    x = axis[0]
    y = axis[1]
    z = axis[2]
    c = wp.cos(angle)
    s = wp.sin(angle)
    v = wp.float64(1) - c
    return wp.mat33d(
        c + x * x * v,
        x * y * v - z * s,
        x * z * v + y * s,
        y * x * v + z * s,
        c + y * y * v,
        y * z * v - x * s,
        z * x * v - y * s,
        z * y * v + x * s,
        c + z * z * v,
    )


@wp.func
def _unit_perp(axis: wp.vec3d) -> wp.vec3d:
    ref = wp.vec3d(wp.float64(1), wp.float64(0), wp.float64(0))
    if wp.abs(axis[0]) >= wp.float64(0.9):
        ref = wp.vec3d(wp.float64(0), wp.float64(1), wp.float64(0))
    p = ref - axis * wp.dot(axis, ref)
    return p / wp.length(p)


@wp.func
def _swivel_basis(direction: wp.vec3d) -> tuple[wp.vec3d, wp.vec3d]:
    ref = wp.vec3d(wp.float64(0), wp.float64(0), wp.float64(1))
    if wp.abs(direction[2]) >= wp.float64(0.99):
        ref = wp.vec3d(wp.float64(1), wp.float64(0), wp.float64(0))
    first = ref - direction * wp.dot(ref, direction)
    first = first / wp.length(first)
    return first, wp.cross(direction, first)


@wp.func
def _angle_about(axis: wp.vec3d, before: wp.vec3d, after: wp.vec3d) -> wp.float64:
    vf = before - axis * wp.dot(axis, before)
    vt = after - axis * wp.dot(axis, after)
    result = wp.float64(0)
    if wp.length(vf) > wp.float64(1e-12) and wp.length(vt) > wp.float64(1e-12):
        result = wp.atan2(wp.dot(axis, wp.cross(vf, vt)), wp.dot(vf, vt))
    return result


@wp.func
def _min_rotation(before: wp.vec3d, after: wp.vec3d) -> wp.mat33d:
    u = before / wp.length(before)
    v = after / wp.length(after)
    cross = wp.cross(u, v)
    sine = wp.length(cross)
    cosine = wp.clamp(wp.dot(u, v), wp.float64(-1), wp.float64(1))
    result = _identity()
    if sine > wp.float64(1e-12):
        result = _rotation(cross / sine, wp.atan2(sine, cosine))
    elif cosine < wp.float64(0):
        result = _rotation(_unit_perp(u), wp.float64(3.141592653589793))
    return result


@wp.func
def _decompose_3axis(
    rotation: wp.mat33d,
    first: wp.vec3d,
    middle: wp.vec3d,
    last: wp.vec3d,
    seed_first: wp.float64,
) -> tuple[_mat23d, bool]:
    """General Davenport decomposition, preserving seed at Euler gimbals."""
    branches = _mat23d()
    first_middle = wp.dot(first, middle)
    middle_last = wp.dot(middle, last)
    a = wp.dot(first, last) - first_middle * middle_last
    b = wp.dot(first, wp.cross(middle, last))
    c = wp.dot(first, rotation * last) - first_middle * middle_last
    amplitude = wp.sqrt(a * a + b * b)
    valid = amplitude > wp.float64(1e-12)
    if valid:
        # A float32 FK target may put the exact Euler double root slightly
        # beyond the unit-circle envelope. Keep its tangent representative;
        # original-model FK validation remains authoritative for acceptance.
        valid = wp.abs(c) <= amplitude + wp.float64(5e-7)
        center = wp.atan2(b, a)
        # Evaluate rho²-C² from cross products rather than 1-cos² near an
        # Euler pole. This identity applies to arbitrary unit axis triples.
        target_last = rotation * last
        target_norm = wp.length(target_last)
        direction = target_last
        if target_norm > wp.float64(1e-12):
            direction = target_last / target_norm
        projection = wp.dot(first, direction)
        cross = wp.cross(first, direction)
        discriminant = (
            wp.dot(cross, cross)
            - first_middle * first_middle
            - middle_last * middle_last
            + wp.float64(2) * first_middle * middle_last * projection
        )
        offset = wp.atan2(
            wp.sqrt(wp.max(discriminant, wp.float64(0))),
            projection - first_middle * middle_last,
        )
        for branch in range(2):
            angle_middle = center + offset
            if branch == 1:
                angle_middle = center - offset
            rm = _rotation(middle, angle_middle)
            mapped_last = rm * last
            angle_first = _angle_about(first, mapped_last, rotation * last)
            angle_last = _angle_about(
                last, wp.transpose(rotation) * first, wp.transpose(rm) * first
            )
            # acos loses relative precision at a double root. Inspect the
            # target projection as well, so exact gimbals do not become a
            # spurious 1e-8-radian middle angle with inconsistent outer angles.
            mapped_perp = wp.length(mapped_last - first * wp.dot(first, mapped_last))
            target_perp = wp.length(target_last - first * wp.dot(first, target_last))
            if mapped_perp < wp.float64(1e-6) or target_perp < wp.float64(1e-6):
                angle_first = seed_first
                residual = wp.transpose(rm) * _rotation(first, -angle_first) * rotation
                perpendicular = _unit_perp(last)
                angle_last = _angle_about(last, perpendicular, residual * perpendicular)
            branches[branch, 0] = angle_first
            branches[branch, 1] = angle_middle
            branches[branch, 2] = angle_last
    return branches, valid


@wp.func
def seed_swivel(
    axes: wp.array(dtype=wp.vec3d),
    offsets: wp.array(dtype=wp.vec3d),
    home_rotation: wp.mat33d,
    shoulder: wp.vec3d,
    wrist: wp.vec3d,
    seed: vec7d,
) -> tuple[wp.float64, bool]:
    """Recover the seed elbow swivel; return false when its circle collapses.

    ``offsets`` and ``axes`` use the normalized POE chain convention. The home
    rotation is accepted for symmetry with :func:`solve_srs`; the swivel is
    independent of the end-effector orientation.
    """
    elbow = offsets[0] + offsets[1] + offsets[2] + offsets[3]
    r_shoulder = (
        _rotation(axes[0], seed[0])
        * _rotation(axes[1], seed[1])
        * _rotation(axes[2], seed[2])
    )
    upper = r_shoulder * (elbow - shoulder)
    sw = upper + r_shoulder * _rotation(axes[3], seed[3]) * (wrist - elbow)
    distance = wp.length(sw)
    angle = wp.float64(0)
    valid = distance > wp.float64(1e-12)
    if valid:
        direction = sw / distance
        radial = upper - direction * wp.dot(upper, direction)
        valid = wp.length(radial) > wp.float64(1e-10)
        if valid:
            first, second = _swivel_basis(direction)
            angle = wp.atan2(wp.dot(radial, second), wp.dot(radial, first))
    return angle, valid


@wp.func
def solve_srs(
    axes: wp.array(dtype=wp.vec3d),
    offsets: wp.array(dtype=wp.vec3d),
    home_rotation: wp.mat33d,
    shoulder: wp.vec3d,
    wrist: wp.vec3d,
    target: wp.mat44d,
    swivel: wp.float64,
    seed: vec7d,
    free_roll: wp.float64 = wp.float64(0),
) -> tuple[mat87d, vec8i]:
    """Return eight conditional branches for a strict SRS chain and swivel.

    The home origins of axes 0--2 and 4--6 must be concurrent at ``shoulder``
    and ``wrist`` respectively. Translations may place intermediate origins
    away from these pivots; no ZYZ or orthogonality assumption is made.

    A 1e-7 m reach-envelope allowance accommodates float32 target rounding.
    It does not accept a solution: the caller must verify every rounded joint
    vector against the original chain and its configured FK tolerances.
    """
    candidates = mat87d()
    validity = vec8i()
    rotation_target = wp.mat33d(
        target[0, 0],
        target[0, 1],
        target[0, 2],
        target[1, 0],
        target[1, 1],
        target[1, 2],
        target[2, 0],
        target[2, 1],
        target[2, 2],
    )
    position_target = wp.vec3d(target[0, 3], target[1, 3], target[2, 3])
    home_position = wp.vec3d()
    for i in range(8):
        home_position = home_position + offsets[i]
    elbow = offsets[0] + offsets[1] + offsets[2] + offsets[3]
    upper_home = elbow - shoulder
    forearm_home = wrist - elbow
    upper_length = wp.length(upper_home)
    forearm_length = wp.length(forearm_home)
    ee_offset = wp.transpose(home_rotation) * (home_position - wrist)
    wrist_target = position_target - rotation_target * ee_offset
    shoulder_wrist = wrist_target - shoulder
    distance = wp.length(shoulder_wrist)
    reachable = upper_length > wp.float64(1e-12) and forearm_length > wp.float64(1e-12)
    reachable = reachable and distance > wp.float64(1e-12)
    reachable = reachable and distance <= upper_length + forearm_length + wp.float64(
        1e-7
    )
    reachable = reachable and distance >= wp.abs(
        upper_length - forearm_length
    ) - wp.float64(1e-7)
    if reachable:
        direction = shoulder_wrist / distance
        x_circle = (
            upper_length * upper_length
            - forearm_length * forearm_length
            + distance * distance
        ) / (wp.float64(2) * distance)
        radius = wp.sqrt(
            wp.max(upper_length * upper_length - x_circle * x_circle, wp.float64(0))
        )
        # Do not create a fictitious swivel circle from positive float64
        # roundoff at an exact radial boundary (at most 1e-13 m affected).
        if wp.abs(distance - upper_length - forearm_length) <= wp.float64(1e-13):
            x_circle = upper_length
            radius = wp.float64(0)
        elif wp.abs(distance - wp.abs(upper_length - forearm_length)) <= wp.float64(
            1e-13
        ):
            x_circle = upper_length
            if upper_length < forearm_length:
                x_circle = -upper_length
            radius = wp.float64(0)
        basis_first, basis_second = _swivel_basis(direction)
        upper_target = x_circle * direction + radius * (
            wp.cos(swivel) * basis_first + wp.sin(swivel) * basis_second
        )
        d = upper_target / wp.length(upper_target)
        wrist_vector = shoulder_wrist - upper_target
        r0 = _min_rotation(upper_home, upper_target)
        k_elbow = r0 * axes[3]
        forearm0 = r0 * forearm_home
        hk = wp.dot(d, k_elbow)
        kp = wp.dot(k_elbow, forearm0)
        a = wp.dot(d, forearm0) - hk * kp
        b = wp.dot(d, wp.cross(k_elbow, forearm0))
        c = wp.dot(d, wrist_vector) - hk * kp
        amplitude = wp.sqrt(a * a + b * b)
        elbow_valid = amplitude > wp.float64(1e-12)
        elbow_valid = elbow_valid and wp.abs(c) <= amplitude + wp.float64(1e-7)
        if elbow_valid:
            center = wp.atan2(b, a)
            offset = wp.acos(wp.clamp(c / amplitude, wp.float64(-1), wp.float64(1)))
            if radius == wp.float64(0):
                if wp.abs(c / amplitude - wp.float64(1)) <= wp.float64(1e-13):
                    offset = wp.float64(0)
                elif wp.abs(c / amplitude + wp.float64(1)) <= wp.float64(1e-13):
                    offset = wp.float64(3.141592653589793)
            seed_rotation = (
                _rotation(axes[0], seed[0])
                * _rotation(axes[1], seed[1])
                * _rotation(axes[2], seed[2])
            )
            for elbow_branch in range(2):
                angle_elbow = center + offset
                if elbow_branch == 1:
                    angle_elbow = center - offset
                r_elbow = _rotation(axes[3], angle_elbow)
                forearm = r0 * r_elbow * forearm_home
                phi = _angle_about(d, forearm, wrist_vector)
                if radius == wp.float64(0) or wp.length(
                    forearm - d * wp.dot(d, forearm)
                ) < wp.float64(1e-9):
                    perpendicular = _unit_perp(upper_home / upper_length)
                    phi = _angle_about(
                        d, r0 * perpendicular, seed_rotation * perpendicular
                    )
                    if radius == wp.float64(0):
                        phi += free_roll
                r_shoulder = _rotation(d, phi) * r0
                residual = (
                    wp.transpose(r_shoulder * r_elbow)
                    * rotation_target
                    * wp.transpose(home_rotation)
                )
                shoulder_angles, shoulder_valid = _decompose_3axis(
                    r_shoulder, axes[0], axes[1], axes[2], seed[0]
                )
                wrist_angles, wrist_valid = _decompose_3axis(
                    residual, axes[4], axes[5], axes[6], seed[4]
                )
                for shoulder_branch in range(2):
                    for wrist_branch in range(2):
                        branch = elbow_branch * 4 + shoulder_branch * 2 + wrist_branch
                        for j in range(3):
                            candidates[branch, j] = shoulder_angles[shoulder_branch, j]
                            candidates[branch, j + 4] = wrist_angles[wrist_branch, j]
                        candidates[branch, 3] = angle_elbow
                        if shoulder_valid and wrist_valid:
                            validity[branch] = 1
    return candidates, validity
