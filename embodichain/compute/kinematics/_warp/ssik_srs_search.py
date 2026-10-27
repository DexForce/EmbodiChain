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

"""Bounded geometric redundancy search along the analytic SRS solution curve.

The scalar search changes the redundant swivel while remaining on an analytic
IK branch. It never solves a Cartesian residual or calls numerical IK. Callers
must validate the returned joint coordinates against their original TCP target.
"""

from __future__ import annotations

import warp as wp

from .ssik_7r import (
    _rotation,
    _swivel_basis,
    mat87d,
    solve_srs,
    vec7d,
    vec8i,
)

wp.set_module_options({"enable_backward": False})

__all__: list[str] = []


@wp.func
def _srs_search_target(
    axes: wp.array(dtype=wp.vec3d),
    offsets: wp.array(dtype=wp.vec3d),
    home: wp.mat33d,
    shoulder: wp.vec3d,
    wrist: wp.vec3d,
    target: wp.mat44d,
    position_tolerance: wp.float64,
) -> tuple[wp.mat44d, bool]:
    """Project only the tolerance-sized band outside the exact radial shell.

    The shell is derived from the elbow rotation, including tilted/offset
    elbows. The original, unmodified TCP target remains the acceptance goal.
    """
    elbow = offsets[0] + offsets[1] + offsets[2] + offsets[3]
    upper, forearm = elbow - shoulder, wrist - elbow
    axis = axes[3]
    constant = wp.dot(upper, axis) * wp.dot(axis, forearm)
    a = wp.dot(upper, forearm) - constant
    b = wp.dot(upper, wp.cross(axis, forearm))
    amplitude = wp.sqrt(a * a + b * b)
    norm_squared = wp.dot(upper, upper) + wp.dot(forearm, forearm)
    minimum = wp.sqrt(
        wp.max(norm_squared + wp.float64(2) * (constant - amplitude), wp.float64(0))
    )
    maximum = wp.sqrt(
        wp.max(norm_squared + wp.float64(2) * (constant + amplitude), wp.float64(0))
    )
    position = wp.vec3d(target[0, 3], target[1, 3], target[2, 3])
    rotation = wp.mat33d(
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
    home_position = wp.vec3d()
    for j in range(8):
        home_position += offsets[j]
    ee_offset = wp.transpose(home) * (home_position - wrist)
    sw = position - rotation * ee_offset - shoulder
    distance = wp.length(sw)
    result = target
    valid = wp.isfinite(distance) and distance > wp.float64(1e-12)
    if valid:
        projected = wp.clamp(distance, minimum, maximum)
        valid = wp.abs(projected - distance) <= position_tolerance
        if valid:
            displacement = sw * ((projected - distance) / distance)
            for j in range(3):
                result[j, 3] += displacement[j]
    return result, valid


@wp.func
def _target_seed_swivel(
    axes: wp.array(dtype=wp.vec3d),
    offsets: wp.array(dtype=wp.vec3d),
    home: wp.mat33d,
    shoulder: wp.vec3d,
    wrist: wp.vec3d,
    target: wp.mat44d,
    seed: vec7d,
) -> tuple[wp.float64, bool]:
    """Project the seed elbow onto the current target's swivel plane."""
    elbow = offsets[0] + offsets[1] + offsets[2] + offsets[3]
    seed_rotation = (
        _rotation(axes[0], seed[0])
        * _rotation(axes[1], seed[1])
        * _rotation(axes[2], seed[2])
    )
    upper_seed = seed_rotation * (elbow - shoulder)
    home_position = wp.vec3d()
    for j in range(8):
        home_position += offsets[j]
    rotation = wp.mat33d(
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
    sw = (
        wp.vec3d(target[0, 3], target[1, 3], target[2, 3])
        - rotation * (wp.transpose(home) * (home_position - wrist))
        - shoulder
    )
    distance = wp.length(sw)
    angle = wp.float64(0)
    valid = wp.isfinite(distance) and distance > wp.float64(1e-12)
    if valid:
        direction = sw / distance
        radial = upper_seed - direction * wp.dot(upper_seed, direction)
        if wp.length(radial) > wp.float64(1e-12):
            first, second = _swivel_basis(direction)
            angle = wp.atan2(wp.dot(radial, second), wp.dot(radial, first))
    return angle, valid


@wp.func
def _srs_lift(qpos: vec7d, reference: vec7d) -> vec7d:
    result = vec7d()
    for j in range(7):
        result[j] = qpos[j] + wp.float64(6.283185307179586) * wp.floor(
            (reference[j] - qpos[j]) / wp.float64(6.283185307179586) + wp.float64(0.5)
        )
    return result


@wp.func
def _srs_row(candidates: mat87d, branch: int) -> vec7d:
    result = vec7d()
    for j in range(7):
        result[j] = candidates[branch, j]
    return result


@wp.func
def _srs_cost(qpos: vec7d, seed: vec7d, weights: vec7d) -> wp.float64:
    result = wp.float64(0)
    for j in range(7):
        difference = (qpos[j] - seed[j]) * weights[j]
        result += difference * difference
    return result


@wp.func
def _srs_nearest_branch(
    candidates: mat87d, validity: vec8i, reference: vec7d, weights: vec7d
) -> tuple[vec7d, bool]:
    """Follow the nearest sheet through Euler/elbow branch intersections."""
    best = reference
    best_cost = wp.float64(1e30)
    found = bool(False)
    for branch in range(8):
        if validity[branch] != 0:
            qpos = _srs_lift(_srs_row(candidates, branch), reference)
            cost = _srs_cost(qpos, reference, weights)
            if cost < best_cost:
                best, best_cost, found = qpos, cost, True
    return best, found


@wp.func
def _srs_phi_cost(
    axes: wp.array(dtype=wp.vec3d),
    offsets: wp.array(dtype=wp.vec3d),
    home: wp.mat33d,
    shoulder: wp.vec3d,
    wrist: wp.vec3d,
    target: wp.mat44d,
    seed: vec7d,
    weights: vec7d,
    phi: wp.float64,
    free_roll: bool = False,
) -> wp.float64:
    swivel, roll = phi, wp.float64(0)
    if free_roll:
        swivel, roll = wp.float64(0), phi
    candidates, validity = solve_srs(
        axes, offsets, home, shoulder, wrist, target, swivel, seed, roll
    )
    nearest, found = _srs_nearest_branch(candidates, validity, seed, weights)
    cost = wp.float64(wp.inf)
    if found:
        cost = _srs_cost(nearest, seed, weights)
    return cost


@wp.func
def _refine_srs_swivel(
    axes: wp.array(dtype=wp.vec3d),
    offsets: wp.array(dtype=wp.vec3d),
    home: wp.mat33d,
    shoulder: wp.vec3d,
    wrist: wp.vec3d,
    target: wp.mat44d,
    seed: vec7d,
    weights: vec7d,
) -> tuple[wp.float64, bool]:
    """Add one locally optimized swivel, with fixed iteration/work bounds.

    Analytic Euler-chart extrema supply pole-safe anchors. Forty bounded
    golden-section steps minimize weighted joint displacement over a local
    0.4-radian interval, following the nearest branch at every evaluation.
    This searches redundancy and never iterates a Cartesian IK residual.
    """
    phi, defined = _target_seed_swivel(
        axes, offsets, home, shoulder, wrist, target, seed
    )
    candidates, validity = solve_srs(
        axes, offsets, home, shoulder, wrist, target, phi, seed
    )
    initial, found = _srs_nearest_branch(candidates, validity, seed, weights)
    defined = defined and found
    if defined:
        # Euler charts meet where their middle-angle sinusoid is extremal.
        # Those swivel angles are available in closed form. Considering them
        # explicitly avoids a finite-difference step jumping over a very narrow
        # pole neighborhood and needlessly changing both outer joint angles.
        home_position = wp.vec3d()
        for j in range(8):
            home_position += offsets[j]
        rotation = wp.mat33d(
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
        sw = (
            wp.vec3d(target[0, 3], target[1, 3], target[2, 3])
            - rotation * (wp.transpose(home) * (home_position - wrist))
            - shoulder
        )
        direction = sw / wp.length(sw)
        anchor = phi
        best_cost = _srs_cost(initial, seed, weights)
        for elbow_branch in range(2):
            index = elbow_branch * 4
            if validity[index] != 0:
                q = _srs_row(candidates, index)
                r_shoulder = (
                    _rotation(axes[0], q[0])
                    * _rotation(axes[1], q[1])
                    * _rotation(axes[2], q[2])
                )
                for triple in range(2):
                    first = axes[0]
                    vector = r_shoulder * axes[2]
                    sign = wp.float64(1)
                    if triple == 1:
                        first = r_shoulder * _rotation(axes[3], q[3]) * axes[4]
                        vector = rotation * wp.transpose(home) * axes[6]
                        sign = wp.float64(-1)
                    a = wp.dot(first, vector) - wp.dot(first, direction) * wp.dot(
                        direction, vector
                    )
                    b = sign * wp.dot(first, wp.cross(direction, vector))
                    phase = wp.atan2(b, a)
                    for extremum in range(2):
                        trial_phi = (
                            anchor
                            + phase
                            + wp.float64(extremum) * wp.float64(3.141592653589793)
                        )
                        critical, critical_valid = solve_srs(
                            axes,
                            offsets,
                            home,
                            shoulder,
                            wrist,
                            target,
                            trial_phi,
                            seed,
                        )
                        qcritical, found_critical = _srs_nearest_branch(
                            critical, critical_valid, seed, weights
                        )
                        critical_cost = _srs_cost(qcritical, seed, weights)
                        if found_critical and critical_cost < best_cost:
                            phi, best_cost = trial_phi, critical_cost
        lower, upper = phi - wp.float64(0.2), phi + wp.float64(0.2)
        ratio = wp.float64(0.6180339887498949)
        left_point = upper - ratio * (upper - lower)
        right_point = lower + ratio * (upper - lower)
        first_cost = _srs_phi_cost(
            axes, offsets, home, shoulder, wrist, target, seed, weights, left_point
        )
        second_cost = _srs_phi_cost(
            axes, offsets, home, shoulder, wrist, target, seed, weights, right_point
        )
        for iteration in range(40):
            if first_cost < best_cost:
                phi, best_cost = left_point, first_cost
            if second_cost < best_cost:
                phi, best_cost = right_point, second_cost
            if first_cost <= second_cost:
                upper, right_point, second_cost = right_point, left_point, first_cost
                left_point = upper - ratio * (upper - lower)
                first_cost = _srs_phi_cost(
                    axes,
                    offsets,
                    home,
                    shoulder,
                    wrist,
                    target,
                    seed,
                    weights,
                    left_point,
                )
            else:
                lower, left_point, first_cost = left_point, right_point, second_cost
                right_point = lower + ratio * (upper - lower)
                second_cost = _srs_phi_cost(
                    axes,
                    offsets,
                    home,
                    shoulder,
                    wrist,
                    target,
                    seed,
                    weights,
                    right_point,
                )
    return phi, defined


@wp.func
def _refine_srs_free_roll(
    axes: wp.array(dtype=wp.vec3d),
    offsets: wp.array(dtype=wp.vec3d),
    home: wp.mat33d,
    shoulder: wp.vec3d,
    wrist: wp.vec3d,
    target: wp.mat44d,
    seed: vec7d,
    weights: vec7d,
) -> tuple[wp.float64, bool]:
    """Optimize the remaining shoulder roll when the elbow circle collapses."""
    best = wp.float64(0)
    best_cost = _srs_phi_cost(
        axes, offsets, home, shoulder, wrist, target, seed, weights, best, True
    )
    lower, upper = wp.float64(-0.2), wp.float64(0.2)
    ratio = wp.float64(0.6180339887498949)
    first = upper - ratio * (upper - lower)
    second = lower + ratio * (upper - lower)
    first_cost = _srs_phi_cost(
        axes, offsets, home, shoulder, wrist, target, seed, weights, first, True
    )
    second_cost = _srs_phi_cost(
        axes, offsets, home, shoulder, wrist, target, seed, weights, second, True
    )
    for iteration in range(40):
        if first_cost < best_cost:
            best, best_cost = first, first_cost
        if second_cost < best_cost:
            best, best_cost = second, second_cost
        if first_cost <= second_cost:
            upper, second, second_cost = second, first, first_cost
            first = upper - ratio * (upper - lower)
            first_cost = _srs_phi_cost(
                axes, offsets, home, shoulder, wrist, target, seed, weights, first, True
            )
        else:
            lower, first, first_cost = first, second, second_cost
            second = lower + ratio * (upper - lower)
            second_cost = _srs_phi_cost(
                axes,
                offsets,
                home,
                shoulder,
                wrist,
                target,
                seed,
                weights,
                second,
                True,
            )
    return best, wp.isfinite(best_cost)


@wp.func
def _srs_search_candidates(
    axes: wp.array(dtype=wp.vec3d),
    offsets: wp.array(dtype=wp.vec3d),
    home: wp.mat33d,
    shoulder: wp.vec3d,
    wrist: wp.vec3d,
    target: wp.mat44d,
    position_tolerance: wp.float64,
    seed: vec7d,
    weights: vec7d,
) -> tuple[mat87d, vec8i]:
    """Return the eight explicit seed-local branches used by every adapter.

    The search parameter is elbow swivel inside the radial shell and free
    shoulder roll at a collapsed circle. Both use the same weighted joint
    objective; all output acceptance remains the caller's responsibility.
    """
    result, validity = mat87d(), vec8i()
    projected, within = _srs_search_target(
        axes, offsets, home, shoulder, wrist, target, position_tolerance
    )
    if within:
        elbow = offsets[0] + offsets[1] + offsets[2] + offsets[3]
        upper_length = wp.length(elbow - shoulder)
        forearm_length = wp.length(wrist - elbow)
        home_position = wp.vec3d()
        for j in range(8):
            home_position += offsets[j]
        rotation = wp.mat33d(
            projected[0, 0],
            projected[0, 1],
            projected[0, 2],
            projected[1, 0],
            projected[1, 1],
            projected[1, 2],
            projected[2, 0],
            projected[2, 1],
            projected[2, 2],
        )
        sw = (
            wp.vec3d(projected[0, 3], projected[1, 3], projected[2, 3])
            - rotation * (wp.transpose(home) * (home_position - wrist))
            - shoulder
        )
        distance = wp.length(sw)
        collapsed = wp.abs(distance - upper_length - forearm_length) <= wp.float64(
            1e-13
        ) or wp.abs(distance - wp.abs(upper_length - forearm_length)) <= wp.float64(
            1e-13
        )
        if collapsed:
            roll, defined = _refine_srs_free_roll(
                axes, offsets, home, shoulder, wrist, projected, seed, weights
            )
            if defined:
                result, validity = solve_srs(
                    axes,
                    offsets,
                    home,
                    shoulder,
                    wrist,
                    projected,
                    wp.float64(0),
                    seed,
                    roll,
                )
        else:
            phi, defined = _refine_srs_swivel(
                axes, offsets, home, shoulder, wrist, projected, seed, weights
            )
            if defined:
                result, validity = solve_srs(
                    axes, offsets, home, shoulder, wrist, projected, phi, seed
                )
    return result, validity
