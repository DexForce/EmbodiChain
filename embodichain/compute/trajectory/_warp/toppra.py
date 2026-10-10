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

"""Double-precision TOPPRA kernels; parallelism is across paths and intervals.

Integer searches live in separate functions so Warp replays their selected
indices before differentiating the active expression. Loop-carried min/max
accumulators do not retain the intermediate values needed by the adjoint.
"""

from __future__ import annotations

import warp as wp

__all__ = []


@wp.kernel
def _prepare(
    points: wp.array3d(dtype=wp.float64),
    clean: wp.array3d(dtype=wp.float64),
    kept: wp.array2d(dtype=int),
    counts: wp.array(dtype=int),
    intervals: wp.array(dtype=int),
    success: wp.array(dtype=int),
    grid_size: int,
    single_precision: int,
    duplicate_tolerance: wp.float64,
):
    row = wp.tid()
    count = int(0)
    last_kept = int(-1)
    valid = int(1)
    for i in range(points.shape[1]):
        different = int(0)
        for j in range(points.shape[2]):
            value = points[row, i, j]
            if not wp.isfinite(value):
                valid = 0
            if count == 0:
                different = 1
            elif single_precision != 0:
                # Match the legacy subtraction and threshold rounding before
                # the numerical solve is promoted to double precision.
                if (
                    wp.float64(
                        wp.abs(wp.float32(value) - wp.float32(clean[row, count - 1, j]))
                    )
                    >= duplicate_tolerance
                ):
                    different = 1
            elif wp.abs(value - clean[row, count - 1, j]) >= duplicate_tolerance:
                different = 1
        if different != 0:
            kept[row, count] = i
            for j in range(points.shape[2]):
                clean[row, count, j] = points[row, i, j]
            count += 1
            last_kept = i
    if last_kept != points.shape[1] - 1:
        kept[row, count] = points.shape[1] - 1
        for j in range(points.shape[2]):
            clean[row, count, j] = points[row, points.shape[1] - 1, j]
        count += 1
    if count == 2:
        stationary = int(1)
        for j in range(points.shape[2]):
            if clean[row, 0, j] != clean[row, 1, j]:
                stationary = 0
        if stationary != 0:
            count = 1
    counts[row] = count
    success[row] = valid
    if count > 1 and valid != 0:
        base = wp.max(grid_size - 1, 10 * count - 1)
        subdivisions = (base + count - 2) // (count - 1)
        intervals[row] = (count - 1) * subdivisions


@wp.kernel
def _gather(
    points: wp.array3d(dtype=wp.float64),
    kept: wp.array2d(dtype=int),
    counts: wp.array(dtype=int),
    success: wp.array(dtype=int),
    clean: wp.array3d(dtype=wp.float64),
):
    row, i, joint = wp.tid()
    if success[row] != 0 and i < counts[row]:
        clean[row, i, joint] = points[row, kept[row, i], joint]


@wp.func
def _speed(x: wp.float64):
    # Zero speed at the fixed rest boundaries is constant. Avoid 0 * inf in
    # the adjoint of sqrt; degenerate interior stops use the same convention.
    value = wp.float64(0.0)
    if x > wp.float64(0.0):
        value = wp.sqrt(x)
    return value


@wp.kernel
def _spline(
    points: wp.array3d(dtype=wp.float64),
    counts: wp.array(dtype=int),
    success: wp.array(dtype=int),
    diagonal: wp.array3d(dtype=wp.float64),
    eliminated: wp.array3d(dtype=wp.float64),
    second: wp.array3d(dtype=wp.float64),
    coefficients: wp.array4d(dtype=wp.float64),
):
    row, joint = wp.tid()
    n = counts[row]
    if success[row] == 0 or n < 2:
        return
    h = wp.float64(1.0) / wp.float64(n - 1)
    if n == 3:
        value = (
            points[row, 2, joint]
            - wp.float64(2.0) * points[row, 1, joint]
            + points[row, 0, joint]
        ) / (h * h)
        for i in range(3):
            second[row, i, joint] = value
    elif n > 3:
        # Uniform not-a-knot boundaries eliminate the endpoint unknowns.
        # Keep elimination RHS values separate from back substitution so the
        # adjoint can read every forward intermediate without overwritten data.
        for i in range(1, n - 1):
            diag = wp.float64(4.0)
            rhs = (
                wp.float64(6.0)
                * (
                    points[row, i + 1, joint]
                    - wp.float64(2.0) * points[row, i, joint]
                    + points[row, i - 1, joint]
                )
                / (h * h)
            )
            if i == 1 or i == n - 2:
                diag = wp.float64(6.0)
            if i > 1 and i < n - 2:
                factor = wp.float64(1.0) / diagonal[row, i - 1, joint]
                if i > 2:
                    diag -= factor
                rhs -= factor * eliminated[row, i - 1, joint]
            diagonal[row, i, joint] = diag
            eliminated[row, i, joint] = rhs
        for k in range(n - 2):
            i = n - 2 - k
            rhs = eliminated[row, i, joint] + wp.float64(0.0)
            if i > 1 and i < n - 2:
                rhs -= second[row, i + 1, joint]
            second[row, i, joint] = rhs / diagonal[row, i, joint]
        second[row, 0, joint] = (
            wp.float64(2.0) * second[row, 1, joint] - second[row, 2, joint]
        )
        second[row, n - 1, joint] = (
            wp.float64(2.0) * second[row, n - 2, joint] - second[row, n - 3, joint]
        )
    for i in range(n - 1):
        left, right = second[row, i, joint], second[row, i + 1, joint]
        coefficients[row, i, joint, 0] = points[row, i, joint]
        coefficients[row, i, joint, 1] = (
            points[row, i + 1, joint] - points[row, i, joint]
        ) / h - h * (wp.float64(2.0) * left + right) / wp.float64(6.0)
        coefficients[row, i, joint, 2] = left / wp.float64(2.0)
        coefficients[row, i, joint, 3] = (right - left) / (wp.float64(6.0) * h)


@wp.func
def _joint_constraints(
    coefficients: wp.array4d(dtype=wp.float64),
    velocity: wp.array2d(dtype=wp.float64),
    acceleration: wp.array2d(dtype=wp.float64),
    a: wp.array3d(dtype=wp.float64),
    b: wp.array3d(dtype=wp.float64),
    c: wp.array3d(dtype=wp.float64),
    row: int,
    i: int,
    joint: int,
    segment: int,
    delta: wp.float64,
    ds: wp.float64,
):
    dofs = coefficients.shape[2]
    cap = wp.float64(wp.inf)
    cb = coefficients[row, segment, joint, 1]
    cc = coefficients[row, segment, joint, 2]
    cd = coefficients[row, segment, joint, 3]
    end = delta + ds
    first = cb + delta * (wp.float64(2.0) * cc + wp.float64(3.0) * delta * cd)
    last = cb + end * (wp.float64(2.0) * cc + wp.float64(3.0) * end * cd)
    second_first = wp.float64(2.0) * cc + wp.float64(6.0) * delta * cd
    second_last = wp.float64(2.0) * cc + wp.float64(6.0) * end * cd
    middle = first + wp.float64(0.5) * ds * second_first
    quadratic = first - wp.float64(2.0) * middle + last
    ratio = wp.float64(0.0)
    if quadratic != wp.float64(0.0):
        ratio = wp.clamp((first - middle) / quadratic, wp.float64(0.0), wp.float64(1.0))
    one = wp.float64(1.0) - ratio
    extremum = (
        one * one * first
        + wp.float64(2.0) * ratio * one * middle
        + ratio * ratio * last
    )
    high, low = wp.max(wp.max(first, last), extremum), wp.min(
        wp.min(first, last), extremum
    )
    if high > wp.float64(0.0):
        limit = velocity[joint, 1] / high
        cap = wp.min(cap, limit * limit)
    if low < wp.float64(0.0):
        limit = velocity[joint, 0] / low
        cap = wp.min(cap, limit * limit)
    upper, lower = acceleration[joint, 1], -acceleration[joint, 0]
    us = wp.where(upper > wp.float64(0.0), upper, wp.float64(1.0))
    ls = wp.where(lower > wp.float64(0.0), lower, wp.float64(1.0))
    for k in range(3):
        deriv, sx, sy = first, second_first, wp.float64(0.0)
        if k == 1:
            deriv, sx, sy = (
                middle,
                wp.float64(0.5) * second_last,
                wp.float64(0.5) * second_first,
            )
        elif k == 2:
            deriv, sx, sy = last, wp.float64(0.0), second_last
        ax = sx - deriv / (wp.float64(2.0) * ds)
        ay = sy + deriv / (wp.float64(2.0) * ds)
        index = k * dofs + joint
        a[row, i, index] = ax / us
        b[row, i, index] = ay / us
        c[row, i, index] = upper / us
        index += 3 * dofs
        a[row, i, index] = -ax / ls
        b[row, i, index] = -ay / ls
        c[row, i, index] = lower / ls
    return cap


@wp.kernel
def _constraints(
    coefficients: wp.array4d(dtype=wp.float64),
    counts: wp.array(dtype=int),
    intervals: wp.array(dtype=int),
    success: wp.array(dtype=int),
    velocity: wp.array2d(dtype=wp.float64),
    acceleration: wp.array2d(dtype=wp.float64),
    velocity_caps: wp.array3d(dtype=wp.float64),
    a: wp.array3d(dtype=wp.float64),
    b: wp.array3d(dtype=wp.float64),
    c: wp.array3d(dtype=wp.float64),
):
    row, i = wp.tid()
    total = intervals[row]
    if success[row] == 0 or i >= total:
        return
    dofs = coefficients.shape[2]
    subdivisions = total // (counts[row] - 1)
    segment = i // subdivisions
    ds = wp.float64(1.0) / wp.float64(total)
    delta = wp.float64(i % subdivisions) * ds
    for joint in range(dofs):
        velocity_caps[row, i, joint] = _joint_constraints(
            coefficients,
            velocity,
            acceleration,
            a,
            b,
            c,
            row,
            i,
            joint,
            segment,
            delta,
            ds,
        )


@wp.func
def _velocity_active(velocity_caps: wp.array3d(dtype=wp.float64), row: int, i: int):
    active = int(0)
    for joint in range(1, velocity_caps.shape[2]):
        if velocity_caps[row, i, joint] < velocity_caps[row, i, active]:
            active = joint
    return active


@wp.kernel
def _velocity_caps(
    velocity_caps: wp.array3d(dtype=wp.float64),
    intervals: wp.array(dtype=int),
    success: wp.array(dtype=int),
    a: wp.array3d(dtype=wp.float64),
    b: wp.array3d(dtype=wp.float64),
    c: wp.array3d(dtype=wp.float64),
):
    row, i = wp.tid()
    if success[row] == 0 or i >= intervals[row]:
        return
    active = _velocity_active(velocity_caps, row, i)
    cap = velocity_caps[row, i, active]
    index = 6 * velocity_caps.shape[2]
    a[row, i, index] = wp.float64(1.0)
    b[row, i, index] = wp.float64(0.0)
    c[row, i, index] = cap
    a[row, i, index + 1] = wp.float64(0.0)
    b[row, i, index + 1] = wp.float64(1.0)
    c[row, i, index + 1] = cap


@wp.func
def _projection_active(
    a: wp.array3d(dtype=wp.float64),
    b: wp.array3d(dtype=wp.float64),
    c: wp.array3d(dtype=wp.float64),
    row: int,
    i: int,
):
    cap = wp.float64(wp.inf)
    active_j, active_k = int(-1), int(-1)
    for j in range(a.shape[2]):
        aj, bj, cj = a[row, i, j], b[row, i, j], c[row, i, j]
        if bj >= wp.float64(0.0) and aj > wp.float64(0.0):
            if cj / aj < cap:
                cap = cj / aj
                active_j, active_k = j, -1
        if bj < wp.float64(0.0):
            for k in range(a.shape[2]):
                bk = b[row, i, k]
                if bk > wp.float64(0.0):
                    coefficient = aj * bk - bj * a[row, i, k]
                    if coefficient > wp.float64(0.0):
                        rhs = cj * bk - bj * c[row, i, k]
                        if rhs / coefficient < cap:
                            cap = rhs / coefficient
                            active_j, active_k = j, k
    return active_j, active_k


@wp.kernel
def _project_caps(
    a: wp.array3d(dtype=wp.float64),
    b: wp.array3d(dtype=wp.float64),
    c: wp.array3d(dtype=wp.float64),
    intervals: wp.array(dtype=int),
    caps: wp.array2d(dtype=wp.float64),
):
    row, i = wp.tid()
    if i >= intervals[row]:
        return
    active_j, active_k = _projection_active(a, b, c, row, i)
    value = wp.float64(wp.inf)
    if active_j >= 0:
        selected_aj, selected_bj, selected_cj = (
            a[row, i, active_j],
            b[row, i, active_j],
            c[row, i, active_j],
        )
        if active_k < 0:
            value = selected_cj / selected_aj
        else:
            selected_ak, selected_bk, selected_ck = (
                a[row, i, active_k],
                b[row, i, active_k],
                c[row, i, active_k],
            )
            value = (selected_cj * selected_bk - selected_bj * selected_ck) / (
                selected_aj * selected_bk - selected_bj * selected_ak
            )
    caps[row, i] = value


@wp.func
def _controllable_active(
    a: wp.array3d(dtype=wp.float64),
    b: wp.array3d(dtype=wp.float64),
    c: wp.array3d(dtype=wp.float64),
    caps: wp.array2d(dtype=wp.float64),
    controllable: wp.array2d(dtype=wp.float64),
    row: int,
    i: int,
):
    cap = caps[row, i] + wp.float64(0.0)
    active = int(-1)
    for j in range(a.shape[2]):
        aj, bj = a[row, i, j], b[row, i, j]
        if aj > wp.float64(0.0) and bj < wp.float64(0.0):
            bound = (c[row, i, j] - bj * controllable[row, i + 1]) / aj
            if bound < cap:
                cap = bound
                active = j
    return active


@wp.func
def _controllable_step(
    a: wp.array3d(dtype=wp.float64),
    b: wp.array3d(dtype=wp.float64),
    c: wp.array3d(dtype=wp.float64),
    caps: wp.array2d(dtype=wp.float64),
    controllable: wp.array2d(dtype=wp.float64),
    row: int,
    i: int,
):
    active = _controllable_active(a, b, c, caps, controllable, row, i)
    value = caps[row, i] + wp.float64(0.0)
    if active >= 0:
        value = (c[row, i, active] - b[row, i, active] * controllable[row, i + 1]) / a[
            row, i, active
        ]
    return value


@wp.func
def _forward_bound(a: wp.float64, b: wp.float64, c: wp.float64, x: wp.float64):
    product = a * x
    rounding = wp.float64(3.552713678800501e-15) * wp.max(wp.abs(product), wp.abs(c))
    return (c - product + rounding) / b


@wp.func
def _reachable_active(
    a: wp.array3d(dtype=wp.float64),
    b: wp.array3d(dtype=wp.float64),
    c: wp.array3d(dtype=wp.float64),
    controllable: wp.array2d(dtype=wp.float64),
    x: wp.array2d(dtype=wp.float64),
    row: int,
    i: int,
):
    high = controllable[row, i + 1] + wp.float64(0.0)
    active = int(-1)
    for j in range(a.shape[2]):
        if b[row, i, j] > wp.float64(0.0):
            bound = _forward_bound(a[row, i, j], b[row, i, j], c[row, i, j], x[row, i])
            if bound < high:
                high = bound
                active = j
    return active


@wp.func
def _reachable_step(
    a: wp.array3d(dtype=wp.float64),
    b: wp.array3d(dtype=wp.float64),
    c: wp.array3d(dtype=wp.float64),
    controllable: wp.array2d(dtype=wp.float64),
    x: wp.array2d(dtype=wp.float64),
    row: int,
    i: int,
):
    active = _reachable_active(a, b, c, controllable, x, row, i)
    value = controllable[row, i + 1] + wp.float64(0.0)
    if active >= 0:
        value = _forward_bound(
            a[row, i, active], b[row, i, active], c[row, i, active], x[row, i]
        )
    return wp.max(wp.float64(0.0), value)


@wp.kernel
def _solve(
    a: wp.array3d(dtype=wp.float64),
    b: wp.array3d(dtype=wp.float64),
    c: wp.array3d(dtype=wp.float64),
    intervals: wp.array(dtype=int),
    caps: wp.array2d(dtype=wp.float64),
    controllable: wp.array2d(dtype=wp.float64),
    x: wp.array2d(dtype=wp.float64),
    times: wp.array2d(dtype=wp.float64),
    durations: wp.array(dtype=wp.float64),
    success: wp.array(dtype=int),
):
    row = wp.tid()
    total = intervals[row]
    if success[row] == 0 or total == 0:
        return
    for reverse in range(total):
        i = total - 1 - reverse
        cap = _controllable_step(a, b, c, caps, controllable, row, i)
        if not wp.isfinite(cap) or cap < wp.float64(0.0):
            success[row] = 0
            return
        controllable[row, i] = cap
    for i in range(total):
        current = x[row, i]
        high = _reachable_step(a, b, c, controllable, x, row, i)
        for j in range(a.shape[2]):
            left, right, check_bound = (
                a[row, i, j] * current,
                b[row, i, j] * high,
                c[row, i, j],
            )
            tolerance = wp.float64(1e-10) * wp.max(
                wp.max(wp.abs(left), wp.abs(right)), wp.abs(check_bound)
            )
            if left + right - check_bound > tolerance:
                success[row] = 0
                return
        x[row, i + 1] = high
        denominator = _speed(current) + _speed(high)
        if denominator <= wp.float64(0.0):
            success[row] = 0
            return
        arrival = times[row, i] + wp.float64(2.0) / (wp.float64(total) * denominator)
        if not wp.isfinite(arrival) or arrival <= times[row, i]:
            success[row] = 0
            return
        times[row, i + 1] = arrival
    durations[row] = times[row, total]


@wp.func
def _sample_interval(
    times: wp.array2d(dtype=wp.float64), row: int, total: int, query: wp.float64
):
    lo = int(0)
    hi = total + 0
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if times[row, mid] <= query:
            lo = mid
        else:
            hi = mid
    return wp.min(lo, total - 1)


@wp.kernel
def _sample(
    points: wp.array3d(dtype=wp.float64),
    coefficients: wp.array4d(dtype=wp.float64),
    counts: wp.array(dtype=int),
    intervals: wp.array(dtype=int),
    success: wp.array(dtype=int),
    x: wp.array2d(dtype=wp.float64),
    times: wp.array2d(dtype=wp.float64),
    durations: wp.array(dtype=wp.float64),
    sample_counts: wp.array(dtype=int),
    positions: wp.array3d(dtype=wp.float64),
    velocities: wp.array3d(dtype=wp.float64),
    accelerations: wp.array3d(dtype=wp.float64),
    dt: wp.array2d(dtype=wp.float64),
):
    row, sample, joint = wp.tid()
    if success[row] == 0:
        return
    n = counts[row]
    valid = sample_counts[row]
    duration = durations[row]
    if sample >= valid or n == 1:
        positions[row, sample, joint] = points[row, n - 1, joint]
        return
    step = duration / wp.float64(valid - 1)
    query = step * wp.float64(sample)
    if joint == 0 and sample > 0:
        dt[row, sample] = step
    total = intervals[row]
    i = _sample_interval(times, row, total, query)
    elapsed = query - times[row, i]
    accel = (x[row, i + 1] - x[row, i]) * wp.float64(total) / wp.float64(2.0)
    initial = _speed(x[row, i])
    speed = wp.max(wp.float64(0.0), initial + accel * elapsed)
    s = wp.clamp(
        wp.float64(i) / wp.float64(total)
        + initial * elapsed
        + wp.float64(0.5) * accel * elapsed * elapsed,
        wp.float64(i) / wp.float64(total),
        wp.float64(i + 1) / wp.float64(total),
    )
    segment = wp.min(int(s * wp.float64(n - 1)), n - 2)
    delta = s - wp.float64(segment) / wp.float64(n - 1)
    ca, cb = coefficients[row, segment, joint, 0], coefficients[row, segment, joint, 1]
    cc, cd = coefficients[row, segment, joint, 2], coefficients[row, segment, joint, 3]
    first = cb + delta * (wp.float64(2.0) * cc + wp.float64(3.0) * delta * cd)
    second = wp.float64(2.0) * cc + wp.float64(6.0) * delta * cd
    position = ca + delta * (cb + delta * (cc + delta * cd))
    velocity_value = first * speed
    accelerations[row, sample, joint] = second * speed * speed + first * accel
    if sample == 0:
        position = points[row, 0, joint]
        velocity_value = wp.float64(0.0)
    elif sample == valid - 1:
        position = points[row, n - 1, joint]
        velocity_value = wp.float64(0.0)
    positions[row, sample, joint] = position
    velocities[row, sample, joint] = velocity_value
