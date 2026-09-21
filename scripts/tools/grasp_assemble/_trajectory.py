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

"""CPU trajectory timing for continuous joint paths on a fixed command grid."""

from __future__ import annotations

from numbers import Real
from typing import Any

import numpy as np
from scipy.interpolate import CubicHermiteSpline, CubicSpline, PchipInterpolator, PPoly
from scipy.optimize import brentq

__all__ = ["time_parameterize"]


def _positive(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite positive number")
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return value


def _evaluate(
    path: CubicSpline, clock: PPoly, times: np.ndarray
) -> tuple[np.ndarray, ...]:
    s, sd, sdd, sddd = (clock(times, order) for order in range(4))
    s = np.clip(s, 0.0, 1.0)
    q1, q2, q3 = (path(s, order) for order in (1, 2, 3))
    return (
        path(s),
        q1 * sd[:, None],
        q2 * sd[:, None] ** 2 + q1 * sdd[:, None],
        q3 * sd[:, None] ** 3
        + 3 * q2 * sd[:, None] * sdd[:, None]
        + q1 * sddd[:, None],
    )


def _toppra_parameterization(
    path: CubicSpline, knots: np.ndarray, v: float, a: float
) -> tuple[np.ndarray, np.ndarray]:
    import toppra as ta
    from toppra.constraint import JointAccelerationConstraint, JointVelocityConstraint

    toppra_path = ta.SplineInterpolator(knots, path(knots))
    grid = np.unique(np.concatenate((np.linspace(0.0, 1.0, 501), knots)))
    # Chord-length knots can differ from uniform samples by one ULP. Such
    # intervals can round to zero duration when accumulated into the clock.
    # Coalesce only the solver's constraint grid, preserving all spline knots
    # and the independent exact waypoint-passage calculations below.
    separated = [0.0]
    for value in grid[1:-1]:
        if value - separated[-1] > 1e-12 and 1.0 - value > 1e-12:
            separated.append(float(value))
    grid = np.asarray([*separated, 1.0])
    constraints = [
        JointVelocityConstraint(np.full(path.c.shape[-1], v)),
        JointAccelerationConstraint(np.full(path.c.shape[-1], a)),
    ]
    solver = ta.algorithm.TOPPRA(constraints, toppra_path, gridpoints=grid)
    _, speeds, _ = solver.compute_parameterization(0.0, 0.0)
    if (
        speeds is None
        or np.shape(speeds) != grid.shape
        or not np.isfinite(speeds).all()
        or np.any(speeds < 0)
    ):
        raise ValueError("TOPPRA could not time-parameterize the joint path")
    speed_sum = speeds[:-1] + speeds[1:]
    if np.any(speed_sum <= 0):
        raise ValueError("TOPPRA produced a stationary path interval")
    times = np.r_[0.0, np.cumsum(2 * np.diff(grid) / speed_sum)]
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("TOPPRA produced invalid path timing")
    return times, speeds


def _smooth_clock(times: np.ndarray, speeds: np.ndarray, spacing: float) -> PPoly:
    # Closely spaced TOPPRA grid points can amplify numerical speed noise into
    # large jerk. Coarsen only this clock fit; every geometric knot is retained.
    keep = [0]
    for index in range(1, len(times) - 1):
        if (
            times[index] - times[keep[-1]] >= spacing
            and times[-1] - times[index] >= spacing
        ):
            keep.append(index)
    keep.append(len(times) - 1)
    t, v = times[keep], speeds[keep]
    slopes = PchipInterpolator(t, v).derivative()(t)
    slopes[[0, -1]] = 0.0
    speed = CubicHermiteSpline(t, v, slopes)
    critical = speed.derivative().roots(extrapolate=False)
    critical = critical[np.isfinite(critical)]
    if np.min(speed(np.r_[t, critical])) < -1e-12 * max(1.0, np.max(v)):
        raise ValueError("smoothed TOPPRA clock would move backwards")
    clock = speed.antiderivative()
    distance = float(clock(times[-1]))
    if distance <= 0 or not np.isfinite(distance):
        raise ValueError("smoothed TOPPRA clock has no forward motion")
    clock.c /= distance
    return clock


def _passage_times(clock: PPoly, knots: np.ndarray) -> np.ndarray:
    return np.r_[
        0.0,
        [brentq(lambda t: float(clock(t)) - s, 0.0, clock.x[-1]) for s in knots[1:-1]],
        clock.x[-1],
    ]


def _verification_times(clock: PPoly, passages: np.ndarray) -> np.ndarray:
    boundaries = np.unique(np.r_[clock.x, passages])
    duration = clock.x[-1]
    return np.unique(
        np.r_[
            np.linspace(0.0, duration, max(2001, 40 * len(boundaries))),
            np.concatenate(
                [np.linspace(a, b, 41) for a, b in zip(boundaries[:-1], boundaries[1:])]
            ),
            np.clip(boundaries - 1e-10 * duration, 0.0, duration),
            np.clip(boundaries + 1e-10 * duration, 0.0, duration),
        ]
    )


def time_parameterize(
    joint_path: np.ndarray,
    *,
    control_dt: float,
    velocity_limit: float,
    acceleration_limit: float,
    jerk_limit: float,
    duration_scale: float = 1.0,
) -> dict[str, Any]:
    """Produce a smooth rest-to-rest joint trajectory on a fixed command grid.

    A cubic spline passes through all supplied joint samples in chord-length
    order. TOPPRA is called once for the whole continuous path. Its scalar speed
    profile is smoothed with a nonnegative cubic Hermite interpolant and integrated
    to produce a C2 path clock with zero endpoint velocity and acceleration.
    A small, deterministic set of clock fitting spacings removes numerical
    speed noise; the fastest densely validated candidate is used. Every input
    geometric waypoint is retained, without imposing intermediate stops.

    Analytical derivatives determine any extra duration dilation required by
    the velocity, acceleration and jerk limits, with a two-percent margin.
    Duration is rounded up to whole control intervals. This preserves the path
    but is not TOPPRA's original time-optimal trajectory.

    Position, velocity and acceleration are continuous. Jerk is piecewise
    continuous because a cubic path's third derivative can jump at its knots;
    both sides of each knot participate in limit verification. Limits are
    densely checked, not a formal continuous-time certificate. Collision and
    joint-position-limit checks remain the caller's responsibility.

    Args:
        joint_path: Finite joint positions with shape ``(N, DOF)``, ``N >= 1``.
        control_dt: Positive command interval in seconds.
        velocity_limit: Positive per-joint velocity magnitude limit.
        acceleration_limit: Positive per-joint acceleration magnitude limit.
        jerk_limit: Positive per-joint jerk magnitude limit.
        duration_scale: Additional slowdown, at least one.

    Returns:
        Positions and analytical derivatives as ``(M, DOF)`` arrays, arrival
        intervals ``dt`` with ``dt[0] == 0``, total duration, TOPPRA reference
        duration, and a JSON-serializable ``validation`` dictionary.
        ``waypoint_times`` records each original input row's passage time;
        ``waypoint_indices`` maps those times to their nearest output samples.
        Consecutive duplicates share a passage time. For a wholly stationary
        path, the input rows are spread across its one-interval hold instead.
        A single input row maps to the start of that hold.

    Raises:
        ValueError: For malformed inputs, failed timing, or excessive output.
    """
    q = np.asarray(joint_path, dtype=np.float64)
    if q.ndim != 2 or min(q.shape) < 1 or not np.isfinite(q).all():
        raise ValueError("joint_path must be a finite nonempty (N, DOF) array")
    control_dt = _positive(control_dt, "control_dt")
    velocity_limit = _positive(velocity_limit, "velocity_limit")
    acceleration_limit = _positive(acceleration_limit, "acceleration_limit")
    jerk_limit = _positive(jerk_limit, "jerk_limit")
    duration_scale = _positive(duration_scale, "duration_scale")
    if duration_scale < 1:
        raise ValueError("duration_scale must be at least one")
    original_count = len(q)
    unique_mask = np.r_[True, np.any(np.diff(q, axis=0) != 0.0, axis=1)]
    original_to_unique = np.cumsum(unique_mask) - 1
    q = q[unique_mask]
    limits = np.array([velocity_limit, acceleration_limit, jerk_limit])
    reference_duration = 0.0
    verification_samples = 2
    clock_spacing = 0.0
    clock_knots = 2
    dilation = 1.0
    if len(q) == 1:
        positions = np.repeat(q, 2, axis=0)
        derivatives = [np.zeros_like(positions) for _ in range(3)]
        peaks = np.zeros(3)
        duration = control_dt
        waypoint_times = np.linspace(0.0, duration, original_count)
    else:
        distances = np.linalg.norm(np.diff(q, axis=0), axis=1)
        knots = np.r_[0.0, np.cumsum(distances)]
        knots /= knots[-1]
        if np.any(np.diff(knots) <= 0):
            raise ValueError("joint_path contains numerically indistinguishable knots")
        path = CubicSpline(knots, q)
        reference_times, reference_speeds = _toppra_parameterization(
            path, knots, velocity_limit, acceleration_limit
        )
        reference_duration = float(reference_times[-1])
        # Acceleration / jerk gives a meaningful smoothing time scale. Cap it
        # for short paths so the fit always retains interior moving samples.
        spacings = np.unique(
            np.minimum(
                acceleration_limit / jerk_limit * np.array([0.25, 0.5, 1.0, 1.5, 2.0]),
                reference_duration / 10,
            )
        )
        choices = []
        for spacing in spacings:
            try:
                clock = _smooth_clock(reference_times, reference_speeds, float(spacing))
            except ValueError:
                # A coarser fit can discard the only moving sample in a
                # degenerate timing profile; the other spacings remain valid.
                continue
            passages = _passage_times(clock, knots)
            verification_t = _verification_times(clock, passages)
            reference_peaks = np.array(
                [np.max(np.abs(d)) for d in _evaluate(path, clock, verification_t)[1:]]
            )
            scale = max(1.0, *(reference_peaks / limits) ** (1.0 / np.arange(1, 4)))
            required_duration = reference_duration * scale * 1.02 * duration_scale
            choices.append(
                (
                    required_duration,
                    clock,
                    passages,
                    reference_peaks,
                    len(verification_t),
                    spacing,
                )
            )
        if not choices:
            raise ValueError("no monotone smoothed TOPPRA clock could be constructed")
        (
            required_duration,
            clock,
            passages,
            reference_peaks,
            verification_samples,
            spacing,
        ) = min(choices, key=lambda item: item[0])
        intervals = max(1, int(np.ceil(required_duration / control_dt)))
        if intervals > 1_000_000:
            raise ValueError("trajectory exceeds one million control intervals")
        duration = intervals * control_dt
        scale = duration / reference_duration
        dilation = scale
        clock_spacing = float(spacing)
        clock_knots = len(clock.x)
        waypoint_times = passages[original_to_unique] * scale
        reference_t = np.linspace(0.0, reference_duration, intervals + 1)
        positions, *derivatives = _evaluate(path, clock, reference_t)
        derivatives = [
            values / scale**order for order, values in enumerate(derivatives, 1)
        ]
        peaks = np.maximum(
            reference_peaks / scale ** np.arange(1, 4),
            [np.max(np.abs(values)) for values in derivatives],
        )
        if np.any(peaks > limits * (1 + 1e-10)):
            raise ValueError("timed trajectory failed derivative-limit verification")
        # Preserve exact input endpoints; remove polynomial roundoff at rest.
        positions[[0, -1]] = q[[0, -1]]
        derivatives[0][[0, -1]] = 0.0
        derivatives[1][[0, -1]] = 0.0
        verification_samples += len(reference_t)
    dt = np.full(len(positions), control_dt)
    dt[0] = 0.0
    return {
        "positions": positions,
        "velocities": derivatives[0],
        "accelerations": derivatives[1],
        "jerks": derivatives[2],
        "dt": dt,
        "duration": float(duration),
        "toppra_duration": reference_duration,
        "waypoint_times": waypoint_times,
        "waypoint_indices": np.clip(
            np.rint(waypoint_times / control_dt).astype(np.int64), 0, len(positions) - 1
        ),
        "validation": {
            "method": "toppra_smoothed_speed_clock",
            "path_waypoints": len(q),
            "peak_velocity": float(peaks[0]),
            "peak_acceleration": float(peaks[1]),
            "peak_jerk": float(peaks[2]),
            "velocity_limit": velocity_limit,
            "acceleration_limit": acceleration_limit,
            "jerk_limit": jerk_limit,
            "limits_satisfied": bool(np.all(peaks <= limits * (1 + 1e-10))),
            "verification_samples": verification_samples,
            "clock_knot_spacing_seconds": clock_spacing,
            "clock_knots": clock_knots,
            "duration_dilation": dilation,
            "toppra_calls": int(len(q) > 1),
            "smoothness": "C2; piecewise continuous jerk; sampled limit checks",
        },
    }
