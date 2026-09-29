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

"""Internal NumPy reference for rest-to-rest TOPPRA planning.

The path remains the planner's uniform, not-a-knot cubic spline. Squared
path speeds at adjacent gridpoints obey half-planes a*x + b*y <= c.
Bernstein acceleration bounds and analytic derivative extrema constrain the
whole interval. Backward controllability and a greedy forward pass optimize
this conservative discretization; they do not claim the continuous optimum.
"""

from __future__ import annotations

from numbers import Integral

import numpy as np
from scipy.interpolate import CubicSpline

__all__ = []


def _limits(value: object, dofs: int) -> np.ndarray:
    """Canonicalize symmetric magnitudes or signed lower/upper bounds."""
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        array = np.full(dofs, float(array))
    if array.shape == (dofs,):
        array = np.stack((-array, array), axis=-1)
    if array.shape != (dofs, 2) or not np.isfinite(array).all():
        raise ValueError("limits must be finite scalar, (DOF,), or (DOF, 2)")
    if np.any(array[:, 0] > 0) or np.any(array[:, 1] < 0):
        raise ValueError("rest-to-rest limits must contain zero")
    return array


def _validate_sampling(sample_count: int | None, sample_dt: float | None) -> None:
    if (sample_count is None) == (sample_dt is None):
        raise ValueError("specify exactly one of sample_count and sample_dt")
    if sample_count is not None and (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, Integral)
        or sample_count < 2
    ):
        raise ValueError("sample_count must be an integer >= 2")
    if sample_dt is not None and (not np.isfinite(sample_dt) or sample_dt <= 0):
        raise ValueError("sample_dt must be finite and positive")


def _validate_grid_size(grid_size: int) -> None:
    if (
        isinstance(grid_size, bool)
        or not isinstance(grid_size, Integral)
        or grid_size < 3
    ):
        raise ValueError("grid_size must be an integer >= 3")


class _NumpyToppra:
    """Numerical reference; construction raises on infeasible input rows."""

    def __init__(
        self,
        points: np.ndarray,
        velocity: object,
        acceleration: object,
        grid_size: int = 100,
    ) -> None:
        _validate_grid_size(grid_size)
        points = np.asarray(points)
        if points.ndim != 2 or min(points.shape) < 1 or points.shape[0] < 2:
            raise ValueError("waypoints must have shape (N >= 2, DOF >= 1)")
        if not np.isfinite(points).all():
            raise ValueError("waypoints must be finite")
        self.vlim = _limits(velocity, points.shape[1])
        self.alim = _limits(acceleration, points.shape[1])
        # A run of samples within the tolerance is one point. When the run ends
        # the path, move the last kept knot onto the final sample instead of
        # appending it: a repeated end knot is a zero-length segment the cubic
        # fit must bend back through, and it changes every knot's spacing. A
        # converged environment holding its pose in a batched rollout produces
        # exactly that tail. A lone first point still gains the final sample
        # when they differ at all, so a tiny move keeps a real duration.
        keep = [0]
        for i in range(1, len(points)):
            if np.max(np.abs(points[i] - points[keep[-1]])) >= 1e-6:
                keep.append(i)
        last = len(points) - 1
        if keep[-1] != last:
            if len(keep) > 1:
                keep[-1] = last
            elif np.any(points[last] != points[0]):
                keep.append(last)
        self.points = points[keep].astype(np.float64)
        self.duration = 0.0
        if np.all(self.points == self.points[0]):
            return
        knot_count = len(self.points)
        base_intervals = max(grid_size - 1, 10 * knot_count - 1)
        self.path = CubicSpline(np.linspace(0, 1, knot_count), self.points)
        subdivisions = (base_intervals + knot_count - 2) // (knot_count - 1)
        self.grid = np.linspace(0, 1, (knot_count - 1) * subdivisions + 1)
        # Segment-aligned subdivision never creates almost-duplicate knots.
        self.a, self.b, self.c = self._constraints()
        self._solve()

    def _constraints(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ds = np.diff(self.grid)[:, None]
        first, last = self.path(self.grid[:-1], 1), self.path(self.grid[1:], 1)
        second_first, second_last = self.path(self.grid[:-1], 2), self.path(
            self.grid[1:], 2
        )
        derivative = np.stack((first, first + 0.5 * ds * second_first, last), axis=1)
        zero = np.zeros_like(first)
        ax = np.stack((second_first, 0.5 * second_last, zero), axis=1) - derivative / (
            2 * ds[:, None]
        )
        ay = np.stack((zero, 0.5 * second_first, second_last), axis=1) + derivative / (
            2 * ds[:, None]
        )
        # Normalize each sign independently, including zero one-sided bounds.
        upper, lower = self.alim[:, 1], -self.alim[:, 0]
        us, ls = np.where(upper > 0, upper, 1.0), np.where(lower > 0, lower, 1.0)
        shape = (len(ds), -1)
        a = np.concatenate(
            ((ax / us).reshape(shape), (-ax / ls).reshape(shape)), axis=1
        )
        b = np.concatenate(
            ((ay / us).reshape(shape), (-ay / ls).reshape(shape)), axis=1
        )
        c = np.broadcast_to(
            np.r_[np.tile(upper / us, 3), np.tile(lower / ls, 3)], a.shape
        ).copy()
        p0, p1, p2 = np.moveaxis(derivative, 1, 0)
        quadratic = p0 - 2 * p1 + p2
        ratio = np.clip(
            np.divide(p0 - p1, quadratic, out=np.zeros_like(p0), where=quadratic != 0),
            0,
            1,
        )
        extremum = (1 - ratio) ** 2 * p0 + 2 * ratio * (1 - ratio) * p1 + ratio**2 * p2
        high = np.maximum(np.maximum(p0, p2), extremum)
        low = np.minimum(np.minimum(p0, p2), extremum)
        cap = np.full_like(first, np.inf)
        np.divide(self.vlim[:, 1], high, out=cap, where=high > 0)
        negative_cap = np.full_like(first, np.inf)
        np.divide(self.vlim[:, 0], low, out=negative_cap, where=low < 0)
        cap = np.min(np.minimum(cap, negative_cap), axis=1) ** 2
        a = np.column_stack((a, np.ones(len(ds)), np.zeros(len(ds))))
        b = np.column_stack((b, np.zeros(len(ds)), np.ones(len(ds))))
        c = np.column_stack((c, cap, cap))
        return a, b, c

    def _solve(self) -> None:
        count = len(self.grid)
        controllable = np.zeros(count)
        # All limits contain rest, so the lower controllable bound is zero.
        for i in range(count - 2, -1, -1):
            a, b, c = self.a[i], self.b[i], self.c[i]
            positive, negative = b > 0, b < 0
            coefficients = (
                a[negative, None] * b[positive] - b[negative, None] * a[positive]
            )
            rhs = c[negative, None] * b[positive] - b[negative, None] * c[positive]
            mask = coefficients > 0
            cap = np.min(rhs[mask] / coefficients[mask], initial=np.inf)
            bound = c.copy()
            bound[negative] -= b[negative] * controllable[i + 1]
            mask = a > 0
            cap = min(cap, np.min(bound[mask] / a[mask], initial=np.inf))
            if not np.isfinite(cap) or cap < 0:
                raise ValueError("path has no finite controllable speed")
            controllable[i] = cap
        x = np.zeros(count)
        for i in range(count - 1):
            a, b, c = self.a[i], self.b[i], self.c[i]
            residual = c - a * x[i]
            rounding = (
                16 * np.finfo(float).eps * np.maximum(np.abs(c), np.abs(a * x[i]))
            )
            positive = b > 0
            high = min(
                controllable[i + 1],
                np.min(
                    (residual[positive] + rounding[positive]) / b[positive],
                    initial=np.inf,
                ),
            )
            x[i + 1] = max(0.0, high)
            violation = a * x[i] + b * x[i + 1] - c
            tolerance = 1e-10 * np.maximum(
                np.maximum(np.abs(a * x[i]), np.abs(b * x[i + 1])), np.abs(c)
            )
            if np.any(violation > tolerance):
                raise ValueError("path forward pass is infeasible")
        self.speeds = np.sqrt(x)
        ds = np.diff(self.grid)
        denominator = self.speeds[:-1] + self.speeds[1:]
        if np.any(denominator <= 0):
            raise ValueError("path contains a zero-speed interval")
        self.accelerations = np.diff(x) / (2 * ds)
        self.times = np.r_[0.0, np.cumsum(2 * ds / denominator)]
        if not np.isfinite(self.times).all() or np.any(np.diff(self.times) <= 0):
            raise ValueError("path timing is not representable")
        self.duration = float(self.times[-1])

    def sample(self, times: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        times = np.clip(np.asarray(times), 0.0, self.duration)
        if self.duration == 0:
            positions = np.broadcast_to(
                self.points[0], (len(times), self.points.shape[1])
            ).copy()
            return positions, np.zeros_like(positions), np.zeros_like(positions)
        index = np.clip(
            np.searchsorted(self.times, times, side="right") - 1, 0, len(self.times) - 2
        )
        elapsed = times - self.times[index]
        accel = self.accelerations[index]
        speed = np.maximum(0, self.speeds[index] + accel * elapsed)
        s = self.grid[index] + self.speeds[index] * elapsed + 0.5 * accel * elapsed**2
        s = np.clip(s, self.grid[index], self.grid[index + 1])
        positions = self.path(s)
        velocities = self.path(s, 1) * speed[:, None]
        accelerations = (
            self.path(s, 2) * speed[:, None] ** 2 + self.path(s, 1) * accel[:, None]
        )
        positions[times == 0] = self.points[0]
        positions[times == self.duration] = self.points[-1]
        velocities[(times == 0) | (times == self.duration)] = 0
        return positions, velocities, accelerations
