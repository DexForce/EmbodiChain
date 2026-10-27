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

"""One-branch near-parallel IK with explicitly bounded global workspace.

The six-by-six systems live in caller-owned arrays. Keeping them out of
per-thread CUDA local memory avoids a device-wide multi-gigabyte stack reserve.
"""

from __future__ import annotations

import warp as wp

from .ssik_6r import _rot, mat73d, mat83d, ssik_three_parallel_near_branch, vec7d

wp.set_module_options({"enable_backward": False})

__all__: list[str] = []

WORKSPACE_COLUMNS = 120
CANDIDATE_COUNT = 9


@wp.func
def _source_pose(
    frames: wp.array2d(dtype=wp.float64),
    axes: wp.array2d(dtype=wp.float64),
    q: vec7d,
    tcp_translation: wp.vec3d,
    work: wp.array2d(dtype=wp.float64),
    index: int,
    jacobian: bool,
):
    rotation = wp.identity(n=3, dtype=wp.float64)
    position = wp.vec3d()
    joint = int(0)
    while joint < 7:
        fixed, translation = wp.mat33d(), wp.vec3d()
        for row in range(3):
            translation[row] = frames[joint, row * 4 + 3]
            for column in range(3):
                fixed[row, column] = frames[joint, row * 4 + column]
        position += rotation @ translation
        rotation = rotation @ fixed
        if joint < 6:
            local = wp.vec3d(axes[joint, 0], axes[joint, 1], axes[joint, 2])
            direction = rotation @ local
            if jacobian:
                for row in range(3):
                    work[index, 36 + joint * 3 + row] = position[row]
                    work[index, 54 + joint * 3 + row] = direction[row]
            magnitude = wp.length(local)
            rotation = rotation @ _rot(local / magnitude, q[joint] * magnitude)
        joint += 1
    position += rotation @ tcp_translation
    if jacobian:
        for j in range(6):
            origin = wp.vec3d(
                work[index, 36 + j * 3],
                work[index, 37 + j * 3],
                work[index, 38 + j * 3],
            )
            direction = wp.vec3d(
                work[index, 54 + j * 3],
                work[index, 55 + j * 3],
                work[index, 56 + j * 3],
            )
            velocity = wp.cross(direction, position - origin)
            for row in range(3):
                work[index, row * 6 + j] = velocity[row]
                work[index, (row + 3) * 6 + j] = direction[row]
    return position, rotation


@wp.func
def _orientation_residual(rotation: wp.mat33d, goal: wp.mat44d) -> wp.vec3d:
    residual = wp.vec3d()
    for column in range(3):
        current = wp.vec3d(
            rotation[0, column], rotation[1, column], rotation[2, column]
        )
        target = wp.vec3d(goal[0, column], goal[1, column], goal[2, column])
        residual += wp.cross(current, target) * wp.float64(0.5)
    return residual


@wp.func
def _solve_system(work: wp.array2d(dtype=wp.float64), index: int) -> bool:
    valid = bool(True)
    column = int(0)
    while column < 6:
        pivot = column
        magnitude = wp.abs(work[index, 72 + column * 6 + column])
        for row in range(column + 1, 6):
            value = wp.abs(work[index, 72 + row * 6 + column])
            if value > magnitude:
                magnitude, pivot = value, row
        valid = valid and magnitude > wp.float64(1e-15)
        if magnitude > wp.float64(1e-15):
            for k in range(6):
                temporary = work[index, 72 + column * 6 + k]
                work[index, 72 + column * 6 + k] = work[index, 72 + pivot * 6 + k]
                work[index, 72 + pivot * 6 + k] = temporary
            temporary = work[index, 108 + column]
            work[index, 108 + column] = work[index, 108 + pivot]
            work[index, 108 + pivot] = temporary
            diagonal = work[index, 72 + column * 6 + column]
            for k in range(column, 6):
                work[index, 72 + column * 6 + k] /= diagonal
            work[index, 108 + column] /= diagonal
            for row in range(6):
                if row != column:
                    factor = work[index, 72 + row * 6 + column]
                    for k in range(column, 6):
                        work[index, 72 + row * 6 + k] -= (
                            factor * work[index, 72 + column * 6 + k]
                        )
                    work[index, 108 + row] -= factor * work[index, 108 + column]
        column += 1
    return valid


@wp.func
def _polish_branch(
    frames: wp.array2d(dtype=wp.float64),
    axes: wp.array2d(dtype=wp.float64),
    initial: vec7d,
    goal: wp.mat44d,
    tcp_translation: wp.vec3d,
    position_tolerance: wp.float64,
    rotation_tolerance: wp.float64,
    work: wp.array2d(dtype=wp.float64),
    index: int,
) -> vec7d:
    q = initial
    goal_rotation = wp.mat33d()
    for row in range(3):
        for column in range(3):
            goal_rotation[row, column] = goal[row, column]
    goal_position = (
        wp.vec3d(goal[0, 3], goal[1, 3], goal[2, 3]) + goal_rotation @ tcp_translation
    )
    normalization = wp.max(position_tolerance, rotation_tolerance)
    iteration = int(0)
    while iteration < 12:
        position, rotation = _source_pose(
            frames, axes, q, tcp_translation, work, index, True
        )
        displacement = goal_position - position
        orientation = _orientation_residual(rotation, goal)
        position_error = wp.length(displacement) / position_tolerance
        rotation_error = wp.length(orientation) / rotation_tolerance
        previous_error = wp.max(position_error, rotation_error)
        if previous_error <= wp.float64(1e-4):
            break
        # Independent position/rotation acceptance uses a product of balls,
        # not a single L2 ball. Balance the active error blocks, retaining a
        # strictly positive weight for both. Backtracking below decreases the
        # actual worst normalized error, rather than only a quadratic proxy.
        position_ratio = wp.max(wp.float64(0.25), position_error / previous_error)
        rotation_ratio = wp.max(wp.float64(0.25), rotation_error / previous_error)
        position_weight = (
            wp.pow(position_ratio, wp.float64(8.0))
            * normalization
            * normalization
            / (position_tolerance * position_tolerance)
        )
        rotation_weight = (
            wp.pow(rotation_ratio, wp.float64(8.0))
            * normalization
            * normalization
            / (rotation_tolerance * rotation_tolerance)
        )
        for j in range(3):
            work[index, 114 + j] = displacement[j]
            work[index, 117 + j] = orientation[j]
        row = int(0)
        while row < 6:
            for column in range(6):
                value = wp.float64(0.0)
                for k in range(6):
                    weight = position_weight
                    if k >= 3:
                        weight = rotation_weight
                    value += (
                        work[index, k * 6 + row] * work[index, k * 6 + column] * weight
                    )
                if row == column:
                    value += wp.float64(1e-8) * wp.max(position_weight, rotation_weight)
                work[index, 72 + row * 6 + column] = value
            value = wp.float64(0.0)
            for k in range(6):
                weight = position_weight
                if k >= 3:
                    weight = rotation_weight
                value += work[index, k * 6 + row] * work[index, 114 + k] * weight
            work[index, 108 + row] = value
            row += 1
        regular = _solve_system(work, index)
        largest = wp.float64(0.0)
        for j in range(6):
            largest = wp.max(largest, wp.abs(work[index, 108 + j]))
        scale = wp.min(
            wp.float64(1.0), wp.float64(0.025) / wp.max(largest, wp.float64(1e-30))
        )
        improved = bool(False)
        attempt = int(0)
        while regular and attempt < 6:
            next_q = q
            for j in range(6):
                next_q[j] += work[index, 108 + j] * scale
            next_position, next_rotation = _source_pose(
                frames, axes, next_q, tcp_translation, work, index, False
            )
            next_displacement = goal_position - next_position
            next_orientation = _orientation_residual(next_rotation, goal)
            next_error = wp.max(
                wp.length(next_displacement) / position_tolerance,
                wp.length(next_orientation) / rotation_tolerance,
            )
            if next_error < previous_error:
                q = next_q
                improved = True
                break
            scale *= wp.float64(0.5)
            attempt += 1
        if not improved:
            break
        iteration += 1
    return q


@wp.kernel
def generate_near(
    h: mat73d,
    p: mat83d,
    home: wp.mat33d,
    frames: wp.array2d(dtype=wp.float64),
    local_axes: wp.array2d(dtype=wp.float64),
    targets: wp.array(dtype=wp.mat44d),
    seeds: wp.array2d(dtype=wp.float64),
    axis_tolerance: wp.float64,
    tcp_translation: wp.vec3d,
    position_tolerance: wp.float64,
    rotation_tolerance: wp.float64,
    work: wp.array2d(dtype=wp.float64),
    validity: wp.array2d(dtype=wp.int32),
    candidates: wp.array3d(dtype=wp.float64),
):
    target, branch = wp.tid()
    seed = vec7d()
    for j in range(6):
        seed[j] = seeds[target, j]
    goal = targets[target]
    rotation = wp.mat33d()
    for row in range(3):
        for column in range(3):
            rotation[row, column] = goal[row, column]
    position = wp.vec3d(goal[0, 3], goal[1, 3], goal[2, 3])
    q, valid = seed, bool(True)
    if branch < 8:
        q, valid = ssik_three_parallel_near_branch(
            h, p, position, rotation @ wp.transpose(home), seed, branch, axis_tolerance
        )
    else:
        # This initialization is always evaluated alongside all geometric
        # slots, never dispatched conditionally after another solver fails.
        for j in range(6):
            valid = valid and wp.isfinite(seed[j])
    if valid:
        q = _polish_branch(
            frames,
            local_axes,
            q,
            goal,
            tcp_translation,
            position_tolerance,
            rotation_tolerance,
            work,
            target * CANDIDATE_COUNT + branch,
        )
    validity[target, branch] = wp.int32(valid)
    for j in range(6):
        candidates[target, branch, j] = q[j]
