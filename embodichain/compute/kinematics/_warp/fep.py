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
"""Fixed-q7 geometric Franka IK, shared by Warp CPU and CUDA.

Algorithm: HolisticMotion ``src/kinematics/fep/FEPGeometry.cpp`` (64b97a5),
following Lopez-Custodio, Gong and Figueredo's GeoFIK, arXiv:2503.03992.
Reconstruct screw-axis directions by triangle and plane/circle intersection,
then recover joint angles by signed rotations. No Jacobian iteration is used.
"""

from __future__ import annotations

import warp as wp

__all__ = []

_axes = wp.types.matrix(shape=(7, 3), dtype=wp.float64)
_vec7 = wp.types.vector(length=7, dtype=wp.float64)
_vec8 = wp.types.vector(length=8, dtype=wp.float64)
_vec9 = wp.types.vector(length=9, dtype=wp.float64)
_solutions = wp.types.matrix(shape=(8, 7), dtype=wp.float32)
wp.set_module_options({"enable_backward": False})


@wp.func
def _home_axes() -> _axes:
    axes = _axes()
    axes[0, 2] = wp.float64(1.0)
    axes[1, 1] = wp.float64(1.0)
    axes[2, 2] = wp.float64(1.0)
    axes[3, 1] = wp.float64(-1.0)
    axes[4, 2] = wp.float64(1.0)
    axes[5, 1] = wp.float64(-1.0)
    axes[6, 2] = wp.float64(-1.0)
    return axes


@wp.func
def _rotate(v: wp.vec3d, axis: wp.vec3d, angle: wp.float64) -> wp.vec3d:
    c = wp.cos(angle)
    return (
        c * v
        + wp.sin(angle) * wp.cross(axis, v)
        + (wp.float64(1.0) - c) * wp.dot(axis, v) * axis
    )


@wp.func
def _orthogonal(axis: wp.vec3d) -> wp.vec3d:
    # Pick the least aligned coordinate to avoid cancellation.
    basis = wp.vec3d(wp.float64(1.0), wp.float64(0.0), wp.float64(0.0))
    if wp.abs(axis[1]) < wp.abs(axis[0]) and wp.abs(axis[1]) <= wp.abs(axis[2]):
        basis = wp.vec3d(wp.float64(0.0), wp.float64(1.0), wp.float64(0.0))
    elif wp.abs(axis[2]) < wp.abs(axis[0]):
        basis = wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(1.0))
    return wp.normalize(wp.cross(axis, basis))


@wp.func
def _rotation(axis: wp.vec3d, angle: wp.float64) -> wp.mat33d:
    x, y, z = axis[0], axis[1], axis[2]
    c, s = wp.cos(angle), wp.sin(angle)
    t = wp.float64(1.0) - c
    return wp.mat33d(
        c + x * x * t,
        x * y * t - z * s,
        x * z * t + y * s,
        y * x * t + z * s,
        c + y * y * t,
        y * z * t - x * s,
        z * x * t - y * s,
        z * y * t + x * s,
        c + z * z * t,
    )


@wp.func
def _arm_angle(
    qpos: _vec7,
    dimensions: wp.array(dtype=wp.float64),
    scales: wp.array(dtype=wp.float64),
) -> wp.float64:
    # GeoFIK's oriented shoulder/elbow/joint-7 plane, independent of TCP.
    axes = _home_axes()
    x = wp.vec3d(wp.float64(1.0), wp.float64(0.0), wp.float64(0.0))
    y = wp.vec3d(wp.float64(0.0), wp.float64(1.0), wp.float64(0.0))
    z = wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(1.0))
    for i in range(6):
        # Accumulate the canonical rotation once, instead of rotating all
        # remaining axes (21 Rodrigues operations per candidate).
        angle = scales[i] * qpos[i]
        if i % 2 == 0:
            axes[i] = z
            c, s = wp.cos(angle), wp.sin(angle)
            next_x = c * x + s * y
            y = -s * x + c * y
            x = next_x
        else:
            axes[i] = y
            if i >= 3:
                axes[i] = -y
                angle = -angle
            c, s = wp.cos(angle), wp.sin(angle)
            next_x = c * x - s * z
            z = s * x + c * z
            x = next_x
    axes[6] = -z
    elbow = dimensions[0] * axes[2] + dimensions[2] * wp.cross(axes[2], axes[3])
    endpoint = (
        elbow
        + dimensions[1] * axes[4]
        - dimensions[3] * wp.cross(axes[4], axes[3])
        + dimensions[4] * wp.cross(axes[5], axes[6])
    )
    reference = wp.vec3d(endpoint[1], -endpoint[0], wp.float64(0.0))
    normal = wp.cross(endpoint, elbow)
    if wp.length(reference) <= wp.float64(1e-8) or wp.length(normal) <= wp.float64(
        1e-8
    ):
        return wp.float64(wp.nan)
    if wp.dot(normal, axes[3]) < wp.float64(0.0):
        normal = -normal
    return wp.atan2(
        wp.dot(wp.normalize(endpoint), wp.cross(reference, normal)),
        wp.dot(reference, normal),
    )


@wp.kernel
def arm_angles(
    joints: wp.array2d(dtype=wp.float32),
    dimensions: wp.array(dtype=wp.float64),
    scales: wp.array(dtype=wp.float64),
    angles: wp.array(dtype=wp.float32),
):
    row = wp.tid()
    qpos = _vec7()
    for i in range(7):
        qpos[i] = wp.float64(joints[row, i])
    angles[row] = wp.float32(_arm_angle(qpos, dimensions, scales))


@wp.kernel
def score_redundancy(
    seeds: wp.array2d(dtype=wp.float32),
    candidates: wp.array3d(dtype=wp.float32),
    flags: wp.array2d(dtype=wp.int32),
    references: wp.array(dtype=wp.float32),
    lower: wp.array(dtype=wp.float32),
    upper: wp.array(dtype=wp.float32),
    weights: wp.array(dtype=wp.float32),
    dimensions: wp.array(dtype=wp.float64),
    scales: wp.array(dtype=wp.float64),
    angle_weight: wp.float64,
    limit_weight: wp.float64,
    max_step: wp.float64,
    guidance: bool,
    costs: wp.array2d(dtype=wp.float64),
    guide_costs: wp.array2d(dtype=wp.float64),
):
    row, candidate = wp.tid()
    costs[row, candidate] = wp.float64(wp.inf)
    if guidance:
        guide_costs[row, candidate] = wp.float64(wp.inf)
    if flags[row, candidate] == 0:
        return
    qpos = _vec7()
    cost = wp.float64(0.0)
    violation = wp.float64(0.0)
    for i in range(7):
        qpos[i] = wp.float64(candidates[row, candidate, i])
        delta = qpos[i] - wp.float64(seeds[row, i])
        if not guidance and wp.abs(delta) > max_step:
            return
        if guidance and wp.isfinite(max_step):
            excess = wp.max(wp.abs(delta) - max_step, wp.float64(0.0)) / max_step
            violation += excess * excess
        cost += wp.float64(weights[i]) * delta * delta
        width = wp.float64(upper[i]) - wp.float64(lower[i])
        if guidance:
            outside = wp.max(
                wp.max(wp.float64(lower[i]) - qpos[i], qpos[i] - wp.float64(upper[i])),
                wp.float64(0.0),
            ) / wp.max(width, wp.float64(1e-6))
            violation += outside * outside
        if width > wp.float64(1e-8):
            margin = (
                wp.min(qpos[i] - wp.float64(lower[i]), wp.float64(upper[i]) - qpos[i])
                / width
            )
            cost += limit_weight / (wp.max(margin, wp.float64(0.0)) + wp.float64(1e-3))
    if guidance:
        guide_costs[row, candidate] = violation
    # Flag 2 is geometric guidance outside a joint limit, never an IK result.
    if flags[row, candidate] != 1 or violation > wp.float64(0.0):
        return
    reference = wp.float64(references[row])
    if angle_weight > wp.float64(0.0) and wp.isfinite(reference):
        angle = _arm_angle(qpos, dimensions, scales)
        if not wp.isfinite(angle):
            return
        delta = wp.atan2(wp.sin(angle - reference), wp.cos(angle - reference))
        cost += angle_weight * delta * delta
    costs[row, candidate] = cost


@wp.kernel
def select_redundancy(
    rows: wp.array(dtype=wp.int64),
    seeds: wp.array2d(dtype=wp.float32),
    lower: wp.array(dtype=wp.float32),
    upper: wp.array(dtype=wp.float32),
    candidates: wp.array3d(dtype=wp.float32),
    costs: wp.array2d(dtype=wp.float64),
    guide_costs: wp.array2d(dtype=wp.float64),
    guidance: bool,
    valid: wp.array2d(dtype=wp.bool),
    joints: wp.array3d(dtype=wp.float32),
    selected_costs: wp.array2d(dtype=wp.float64),
    guide_centers: wp.array(dtype=wp.float32),
    guide_scores: wp.array(dtype=wp.float64),
):
    row = wp.tid()
    target_row = int(rows[row])
    count = valid.shape[1]
    # Merge in place without concatenating candidate batches or scattering
    # temporary winners. Snapshot the at most eight previous solutions first.
    previous_costs = _vec8(wp.float64(wp.inf))
    previous_joints = _solutions()
    for slot in range(count):
        previous_costs[slot] = selected_costs[target_row, slot]
        for i in range(7):
            previous_joints[slot, i] = joints[target_row, slot, i]
    if guidance:
        best_guide = guide_scores[target_row]
        center = guide_centers[target_row]
        for candidate in range(costs.shape[1]):
            if guide_costs[row, candidate] < best_guide:
                best_guide = guide_costs[row, candidate]
                center = candidates[row, candidate, 6]
        guide_scores[target_row] = best_guide
        guide_centers[target_row] = center
    for slot in range(valid.shape[1]):
        best, best_cost = int(-1), wp.float64(wp.inf)
        for previous in range(count):
            if previous_costs[previous] < best_cost:
                best = previous
                best_cost = previous_costs[previous]
        for candidate in range(costs.shape[1]):
            if costs[row, candidate] < best_cost:
                best = count + candidate
                best_cost = wp.float64(costs[row, candidate])
        success = best >= 0
        valid[target_row, slot] = success
        selected_costs[target_row, slot] = best_cost
        for i in range(7):
            value = wp.clamp(seeds[target_row, i], lower[i], upper[i])
            if success:
                if best < count:
                    value = previous_joints[best, i]
                else:
                    value = candidates[row, best - count, i]
            joints[target_row, slot, i] = value
        if success and count > 1:
            # Remove this solution and duplicates from clamped/repeated q7 probes.
            for previous in range(count):
                distance = wp.float32(0.0)
                for i in range(7):
                    distance = wp.max(
                        distance,
                        wp.abs(
                            previous_joints[previous, i] - joints[target_row, slot, i]
                        ),
                    )
                if distance <= wp.float32(1e-6):
                    previous_costs[previous] = wp.float64(wp.inf)
            for candidate in range(costs.shape[1]):
                distance = wp.float32(0.0)
                for i in range(7):
                    distance = wp.max(
                        distance,
                        wp.abs(
                            candidates[row, candidate, i] - joints[target_row, slot, i]
                        ),
                    )
                if distance <= wp.float32(1e-6):
                    costs[row, candidate] = wp.float64(wp.inf)


@wp.func
def _matches_urdf_fk(
    qpos: _vec7,
    target: wp.mat44d,
    frames: wp.array(dtype=wp.mat44d),
    local_axes: wp.array(dtype=wp.vec3d),
    position_tolerance: wp.float64,
    rotation_tolerance: wp.float64,
) -> bool:
    # frames[0:2] hold canonical conversion; [2:9] are actual URDF
    # joint origins, and [9] is the fixed tip including the current TCP.
    # Compose rigid transforms directly; the homogeneous last row is constant.
    rotation = wp.identity(n=3, dtype=wp.float64)
    position = wp.vec3d()
    for i in range(8):
        origin = frames[i + 2]
        position += rotation @ wp.vec3d(origin[0, 3], origin[1, 3], origin[2, 3])
        rotation = rotation @ wp.mat33d(
            origin[0, 0],
            origin[0, 1],
            origin[0, 2],
            origin[1, 0],
            origin[1, 1],
            origin[1, 2],
            origin[2, 0],
            origin[2, 1],
            origin[2, 2],
        )
        if i < 7:
            rotation = rotation @ _rotation(local_axes[i], qpos[i])
    offset = wp.vec3d(target[0, 3], target[1, 3], target[2, 3]) - position
    position2 = wp.dot(offset, offset)
    if (
        not wp.isfinite(position2)
        or position2 > position_tolerance * position_tolerance
    ):
        return False
    relative = wp.mat33d()
    for i in range(3):
        for j in range(3):
            relative[i, j] = (
                target[i, 0] * rotation[j, 0]
                + target[i, 1] * rotation[j, 1]
                + target[i, 2] * rotation[j, 2]
            )
    sine = wp.length(
        wp.vec3d(
            relative[2, 1] - relative[1, 2],
            relative[0, 2] - relative[2, 0],
            relative[1, 0] - relative[0, 1],
        )
    )
    cosine = relative[0, 0] + relative[1, 1] + relative[2, 2] - wp.float64(1.0)
    # atan2 remains accurate near zero and rejects a half-turn (unlike a
    # skew-only residual). No quaternion conversion or intermediate FK batch.
    angle = wp.atan2(sine, cosine)
    return wp.isfinite(angle) and angle <= rotation_tolerance


@wp.kernel
def validate_inputs(
    targets: wp.array(dtype=wp.mat44f),
    seeds: wp.array2d(dtype=wp.float32),
    lower: wp.array(dtype=wp.float32),
    upper: wp.array(dtype=wp.float32),
    weights: wp.array(dtype=wp.float32),
    errors: wp.array(dtype=wp.int32),
):
    row = wp.tid()
    code = int(0)
    if row == 0:
        for i in range(7):
            if not wp.isfinite(weights[i]) or weights[i] < wp.float32(0.0):
                code = code | 16
            if (
                not wp.isfinite(lower[i])
                or not wp.isfinite(upper[i])
                or lower[i] > upper[i]
            ):
                code = code | 4
    if row < targets.shape[0]:
        target = wp.mat44d(targets[row])
        for i in range(4):
            for j in range(4):
                if not wp.isfinite(target[i, j]):
                    code = code | 1
        for i in range(7):
            if not wp.isfinite(seeds[row, i]):
                code = code | 2
        for i in range(3):
            if wp.abs(target[3, i]) > wp.float64(1e-6):
                code = code | 8
            for j in range(3):
                dot = (
                    target[0, i] * target[0, j]
                    + target[1, i] * target[1, j]
                    + target[2, i] * target[2, j]
                )
                expected = wp.float64(0.0)
                if i == j:
                    expected = wp.float64(1.0)
                if wp.abs(dot - expected) > wp.float64(1e-5):
                    code = code | 8
        x = wp.vec3d(target[0, 0], target[1, 0], target[2, 0])
        y = wp.vec3d(target[0, 1], target[1, 1], target[2, 1])
        z = wp.vec3d(target[0, 2], target[1, 2], target[2, 2])
        if wp.abs(wp.dot(x, wp.cross(y, z)) - wp.float64(1.0)) > wp.float64(
            1e-5
        ) or wp.abs(target[3, 3] - wp.float64(1.0)) > wp.float64(1e-6):
            code = code | 8
    if code != 0:
        wp.atomic_or(errors, 0, code)


@wp.kernel
def solve(
    targets: wp.array(dtype=wp.mat44f),
    seeds: wp.array2d(dtype=wp.float32),
    q7_samples: wp.array2d(dtype=wp.float32),
    lower: wp.array(dtype=wp.float32),
    upper: wp.array(dtype=wp.float32),
    dimensions: wp.array(dtype=wp.float64),
    scales: wp.array(dtype=wp.float64),
    frames: wp.array(dtype=wp.mat44d),
    local_axes: wp.array(dtype=wp.vec3d),
    position_tolerance: wp.float64,
    rotation_tolerance: wp.float64,
    guide_limits: bool,
    weights: wp.array(dtype=wp.float32),
    score_bounds: wp.array(dtype=wp.float64),
    max_step: wp.float64,
    prune: bool,
    valid: wp.array2d(dtype=wp.int32),
    joints: wp.array3d(dtype=wp.float32),
):
    probe, branch = wp.tid()
    row = probe // q7_samples.shape[1]
    sample = probe % q7_samples.shape[1]
    candidate = sample * 8 + branch
    valid[row, candidate] = 0
    for i in range(7):
        joints[row, candidate, i] = seeds[row, i]
    # Preserve a known feasible seed even when inverse geometry is ill
    # conditioned near extension. Keep all eight reconstructed branches.
    # One extra slot per target, not per probe; scoring still enforces posture
    # and step constraints just as for every geometric candidate.
    target = wp.mat44d(targets[row])
    if sample == 0 and branch == 0:
        seed_slot = valid.shape[1] - 1
        valid[row, seed_slot] = 0
        seed = _vec7()
        inside = bool(True)
        for i in range(7):
            joints[row, seed_slot, i] = seeds[row, i]
            seed[i] = wp.float64(seeds[row, i])
            if seeds[row, i] < lower[i] or seeds[row, i] > upper[i]:
                inside = False
        if inside and _matches_urdf_fk(
            seed, target, frames, local_axes, position_tolerance, rotation_tolerance
        ):
            valid[row, seed_slot] = 1
    q7 = wp.float64(q7_samples[row, sample])
    # Clipped grids can repeat an endpoint. Keep its first occurrence; use
    # exact equality so thin but distinct feasible intervals are preserved.
    if sample > 0 and q7 == wp.float64(q7_samples[row, sample - 1]):
        return
    if q7 < wp.float64(lower[6]) or q7 > wp.float64(upper[6]):
        return
    distance = wp.float64(0.0)
    score_bound = wp.float64(wp.inf)
    if prune:
        # Nonnegative posture/limit terms make partial weighted distance a
        # lower bound (HolisticMotion FEPAuto). Keep a rounding allowance.
        score_bound = score_bounds[row]
        score_bound += wp.float64(1e-12) * (wp.float64(1.0) + score_bound)
        delta = q7 - wp.float64(seeds[row, 6])
        distance = wp.float64(weights[6]) * delta * delta
        if wp.abs(delta) > max_step or distance > score_bound:
            return
    d3, d5, a4, a5, a7 = (
        dimensions[0],
        dimensions[1],
        dimensions[2],
        dimensions[3],
        dimensions[4],
    )
    endpoint = frames[0] @ target @ frames[1]
    s7 = wp.normalize(wp.vec3d(endpoint[0, 2], endpoint[1, 2], endpoint[2, 2]))
    radial = _rotate(
        wp.vec3d(endpoint[0, 0], endpoint[1, 0], endpoint[2, 0]), s7, -scales[6] * q7
    )
    # Remove float32 FK rotation drift before intersecting geometric objects.
    s6 = wp.normalize(wp.cross(s7, radial))
    radial = wp.cross(s6, s7)
    wrist = wp.vec3d(endpoint[0, 3], endpoint[1, 3], endpoint[2, 3]) - a7 * radial
    upper_length, lower_length = dimensions[5], dimensions[6]
    cosine = (
        wp.dot(wrist, wrist) - upper_length * upper_length - lower_length * lower_length
    ) / (wp.float64(2.0) * upper_length * lower_length)
    # Test distances in metres instead of a fixed dimensionless cosine
    # allowance. TCP/flange offsets turn angular error into wrist error.
    tool_offset = wp.vec3d(frames[1][0, 3], frames[1][1, 3], frames[1][2, 3])
    geometry_tolerance = (
        position_tolerance + (a7 + wp.length(tool_offset)) * rotation_tolerance
    )
    reach = wp.length(wrist)
    if (
        reach > upper_length + lower_length + geometry_tolerance
        or reach < wp.abs(upper_length - lower_length) - geometry_tolerance
    ):
        return
    bend = wp.acos(wp.clamp(cosine, wp.float64(-1.0), wp.float64(1.0)))
    elbow_sign = wp.float64(-1.0)
    if branch >= 4:
        elbow_sign = wp.float64(1.0)
    beta1 = dimensions[7]
    q4 = elbow_sign * bend - beta1 - dimensions[8]
    # The triangle already determines q4: reject its out-of-limit elbow
    # branch before reconstructing axes and recovering the other angles.
    joint4 = q4 / scales[3]
    period4 = wp.float64(6.283185307179586) / wp.abs(scales[3])
    if wp.ceil((wp.float64(lower[3]) - joint4 - wp.float64(1e-6)) / period4) > wp.floor(
        (wp.float64(upper[3]) - joint4 + wp.float64(1e-6)) / period4
    ):
        nearest4 = joint4 + period4 * wp.round(
            ((wp.float64(lower[3]) + wp.float64(upper[3])) * wp.float64(0.5) - joint4)
            / period4
        )
        slack4 = wp.float64(0.05) * (wp.float64(upper[3]) - wp.float64(lower[3]))
        if (
            not guide_limits
            or nearest4 < wp.float64(lower[3]) - slack4
            or nearest4 > wp.float64(upper[3]) + slack4
        ):
            return
    projection = wrist - s6 * wp.dot(wrist, s6)
    radius = wp.length(projection)
    height = d5 + d3 * wp.cos(q4) - a4 * wp.sin(q4)
    cross_scale = a5 - a4 * wp.cos(q4) - d3 * wp.sin(q4)
    seed_axes = _home_axes()
    if radius <= wp.float64(1e-10) or wp.abs(cross_scale) <= wp.float64(1e-10):
        for i in range(5):
            axis = seed_axes[i]
            angle = scales[i] * wp.float64(seeds[row, i])
            for j in range(i + 1, 5):
                seed_axes[j] = _rotate(seed_axes[j], axis, angle)
    wrist_sign = wp.float64(1.0)
    if (branch // 2) % 2 == 1:
        wrist_sign = wp.float64(-1.0)
    s5 = wp.vec3d()
    if radius <= wp.float64(1e-10):
        if wp.abs(height) > wp.float64(1e-8):
            return
        direction = seed_axes[4] - s6 * wp.dot(s6, seed_axes[4])
        if wp.length(direction) <= wp.float64(1e-10):
            direction = _orthogonal(s6)
        s5 = wrist_sign * wp.normalize(direction)
    else:
        ratio = height / radius
        if wp.abs(height) > radius + geometry_tolerance:
            return
        ratio = wp.clamp(ratio, wp.float64(-1.0), wp.float64(1.0))
        s5 = (
            ratio * projection / radius
            + wrist_sign
            * wp.sqrt(wp.max(wp.float64(0.0), wp.float64(1.0) - ratio * ratio))
            * wp.cross(s6, projection)
            / radius
        )
    s4 = wp.vec3d()
    if wp.abs(cross_scale) <= wp.float64(1e-10):
        s4 = seed_axes[3] - s5 * wp.dot(s5, seed_axes[3])
        if wp.length(s4) <= wp.float64(1e-10):
            s4 = _orthogonal(s5)
    else:
        s4 = wp.cross(s5, wrist) / cross_scale
    if wp.length(s4) <= wp.float64(1e-10):
        return
    s4 = wp.normalize(s4)
    elbow = wrist - d5 * s5 + a5 * wp.cross(s5, s4)
    s3 = _rotate(elbow, s4, beta1)
    if wp.length(s3) <= wp.float64(1e-10):
        return
    s3 = wp.normalize(s3)
    s2 = wp.vec3d(-s3[1], s3[0], wp.float64(0.0))
    if wp.length(s2) <= wp.float64(1e-6):
        q1 = scales[0] * wp.float64(seeds[row, 0])
        s2 = wp.vec3d(-wp.sin(q1), wp.cos(q1), wp.float64(0.0))
    else:
        s2 = wp.normalize(s2)
    if branch % 2 == 1:
        s2 = -s2
    axes = _home_axes()
    axes[1] = s2
    axes[2] = s3
    axes[3] = s4
    axes[4] = s5
    axes[5] = s6
    axes[6] = s7
    qpos = _vec7()
    qpos[6] = q7
    feasible = bool(True)
    for i in range(6):
        # In the canonical layout h[i+1] is parallel to h[i-1]. Rotation
        # about axis i-1 leaves that axis unchanged, so its reconstructed
        # direction gives the next axis before rotation i. Recovering angles
        # therefore needs no propagation of all downstream home axes.
        before = wp.vec3d(wp.float64(0.0), wp.float64(1.0), wp.float64(0.0))
        if i > 0:
            before = axes[i - 1]
            if i == 2 or i == 5:
                before = -before
        after = axes[i + 1]
        angle = wp.atan2(
            wp.dot(axes[i], wp.cross(before, after)), wp.dot(before, after)
        )
        joint = angle / scales[i]
        period = wp.float64(6.283185307179586) / wp.abs(scales[i])
        # Small boundary allowance for float32 URDF/target input; the kernel
        # verifies every rounded/clamped candidate against the actual FK.
        first = wp.ceil((wp.float64(lower[i]) - joint - wp.float64(1e-6)) / period)
        last = wp.floor((wp.float64(upper[i]) - joint + wp.float64(1e-6)) / period)
        if first > last:
            if not guide_limits:
                return
            # Preserve the geometric angle outside a nearby limit for search
            # guidance. It is neither clamped nor returned as a valid solution.
            joint += period * wp.round(
                (
                    (wp.float64(lower[i]) + wp.float64(upper[i])) * wp.float64(0.5)
                    - joint
                )
                / period
            )
            slack = wp.float64(0.05) * (wp.float64(upper[i]) - wp.float64(lower[i]))
            if (
                joint < wp.float64(lower[i]) - slack
                or joint > wp.float64(upper[i]) + slack
            ):
                return
            feasible = False
        else:
            turns = wp.clamp(
                wp.round((wp.float64(seeds[row, i]) - joint) / period), first, last
            )
            joint = wp.clamp(
                joint + period * turns, wp.float64(lower[i]), wp.float64(upper[i])
            )
        if not wp.isfinite(joint):
            return
        # Verify the same rounded joint values that will be returned.
        qpos[i] = wp.float64(wp.float32(joint))
        if prune:
            delta = qpos[i] - wp.float64(seeds[row, i])
            distance += wp.float64(weights[i]) * delta * delta
            if wp.abs(delta) > max_step or distance > score_bound:
                return
    if _matches_urdf_fk(
        qpos, target, frames, local_axes, position_tolerance, rotation_tolerance
    ):
        valid[row, candidate] = 1
        if not feasible:
            valid[row, candidate] = 2
        for i in range(7):
            joints[row, candidate, i] = wp.float32(qpos[i])


@wp.kernel
def select(
    seeds: wp.array2d(dtype=wp.float32),
    lower: wp.array(dtype=wp.float32),
    upper: wp.array(dtype=wp.float32),
    weights: wp.array(dtype=wp.float32),
    flags: wp.array2d(dtype=wp.int32),
    candidates: wp.array3d(dtype=wp.float32),
    valid: wp.array2d(dtype=wp.bool),
    joints: wp.array3d(dtype=wp.float32),
):
    row = wp.tid()
    costs = _vec9(wp.float64(wp.inf))
    for branch in range(flags.shape[1]):
        if flags[row, branch] == 1:
            cost = wp.float64(0.0)
            for i in range(7):
                delta = wp.float64(candidates[row, branch, i]) - wp.float64(
                    seeds[row, i]
                )
                cost += delta * delta * wp.float64(weights[i])
            costs[branch] = cost
    count = valid.shape[1]
    for slot in range(count):
        best = int(0)
        for branch in range(1, flags.shape[1]):
            if costs[branch] < costs[best]:
                best = branch
        success = wp.isfinite(costs[best])
        costs[best] = wp.float64(wp.inf)
        if count > 1 and success:
            # Only the all-solutions path needs sorting and deduplication.
            for previous in range(slot):
                if valid[row, previous]:
                    maximum = wp.float64(0.0)
                    for i in range(7):
                        maximum = wp.max(
                            maximum,
                            wp.abs(
                                wp.float64(joints[row, previous, i])
                                - wp.float64(candidates[row, best, i])
                            ),
                        )
                    if maximum < wp.float64(1e-6):
                        success = False
        valid[row, slot] = success
        for i in range(7):
            value = wp.clamp(seeds[row, i], lower[i], upper[i])
            if success:
                value = candidates[row, best, i]
            joints[row, slot, i] = value
