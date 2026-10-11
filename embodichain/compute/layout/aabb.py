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

"""Array-only constrained XY layout with iterative AABB separation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.optimize import minimize

__all__ = ["AABBLayoutResult", "solve_aabb_layout"]


@dataclass(frozen=True)
class AABBLayoutResult:
    """Numerical result; residuals are nonnegative violations in metres.

    Failure statuses describe this search, not mathematical infeasibility.
    """

    status: Literal["solved", "no_solution", "budget_exhausted"]
    xy: np.ndarray
    residuals: dict[str, float]
    message: str


def solve_aabb_layout(
    seed_xy: np.ndarray,
    half_extents_xy: np.ndarray,
    bounds_xy: np.ndarray,
    fixed_mask: np.ndarray,
    preferred_weights: np.ndarray,
    relations: tuple[tuple[int, int, int, float], ...] = (),
    *,
    relation_clearance_m: float = 0.03,
    collision_margin_m: float = 0.02,
    max_slsqp_iterations: int = 500,
    slsqp_ftol: float = 1e-6,
    max_collision_rounds: int = 8,
    max_added_collision_pairs: int = 64,
    feasibility_tolerance: float = 1e-7,
    skip_fixed_collisions: bool = False,
    enforce_all_clearances: bool = True,
) -> AABBLayoutResult:
    """Solve horizontal, fixed-orientation AABB placement without side effects.

    Args:
        seed_xy: Initial centers, shaped ``(N, 2)``; fixed rows stay here.
        half_extents_xy: Nonnegative oriented-AABB half sizes, ``(N, 2)``.
        bounds_xy: Allowed full-box bounds, ``(N, 2, 2)``; infinity means unbounded.
        fixed_mask: Boolean fixed-position mask, ``(N,)``.
        preferred_weights: Nonnegative squared displacement weights, ``(N,)``.
        relations: Tuples ``(source, target, axis, sign)`` meaning the signed
            source-minus-target coordinate must be smaller than negative spacing.
        relation_clearance_m: Additional relation spacing.
        collision_margin_m: Additional repaired-pair spacing.
        max_slsqp_iterations: Iteration budget for each constrained solve.
        slsqp_ftol: SLSQP objective convergence tolerance.
        max_collision_rounds: Number of overlap discovery passes.
        max_added_collision_pairs: Maximum pairs processed per pass.
        feasibility_tolerance: Maximum accepted linear or overlap violation.
        skip_fixed_collisions: Exempt fixed-fixed pairs for legacy layouts.
        enforce_all_clearances: Require margin on every pair. False preserves
            legacy behavior that only repairs intersecting boxes.

    Returns:
        Candidate centers and measured violations, including on failure.
    """
    initial = np.array(seed_xy, dtype=float, copy=True)
    half = np.asarray(half_extents_xy, dtype=float)
    bounds = np.asarray(bounds_xy, dtype=float)
    fixed = np.asarray(fixed_mask)
    if fixed.dtype != np.bool_:
        raise ValueError("fixed_mask must contain boolean values.")
    if (
        type(skip_fixed_collisions) is not bool
        or type(enforce_all_clearances) is not bool
    ):
        raise ValueError("Collision policy flags must be booleans.")
    weights = np.asarray(preferred_weights, dtype=float)
    n = len(initial)
    if initial.shape != (n, 2) or half.shape != (n, 2):
        raise ValueError("Centers and half extents must have shape (N, 2).")
    if bounds.shape != (n, 2, 2) or fixed.shape != (n,) or weights.shape != (n,):
        raise ValueError("Bounds, fixed mask or weights have inconsistent shapes.")
    if not all(np.all(np.isfinite(a)) for a in (initial, half, weights)):
        raise ValueError("Centers, half extents and weights must be finite.")
    if np.any(half < 0) or np.any(weights < 0) or np.any(np.isnan(bounds)):
        raise ValueError("Invalid extents, weights or bounds.")
    if np.any(np.isposinf(bounds[:, 0])) or np.any(np.isneginf(bounds[:, 1])):
        raise ValueError("Infinite bounds must extend outward.")
    if np.any(bounds[:, 0] > bounds[:, 1]):
        raise ValueError("Bounds must be ordered minimum to maximum.")
    for value in (relation_clearance_m, collision_margin_m):
        if not np.isfinite(value) or value < 0:
            raise ValueError("Clearances must be finite and nonnegative.")
    for value in (slsqp_ftol, feasibility_tolerance):
        if not np.isfinite(value) or value <= 0:
            raise ValueError("Tolerances must be finite and positive.")
    for value in (
        max_slsqp_iterations,
        max_collision_rounds,
        max_added_collision_pairs,
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(
                "Iteration and collision budgets must be positive integers."
            )

    rows: list[np.ndarray] = []
    limits: list[float] = []
    lower = bounds[:, 0] + half
    upper = bounds[:, 1] - half
    for i in range(n):
        for axis in range(2):
            for sign, limit in ((1.0, upper[i, axis]), (-1.0, -lower[i, axis])):
                if np.isfinite(limit):
                    row = np.zeros((n, 2))
                    row[i, axis] = sign
                    rows.append(row.reshape(-1))
                    limits.append(float(limit))
    for source, target, axis, sign in relations:
        if (
            isinstance(source, (bool, np.bool_))
            or isinstance(target, (bool, np.bool_))
            or not isinstance(source, (int, np.integer))
            or not isinstance(target, (int, np.integer))
            or not (0 <= source < n and 0 <= target < n)
            or source == target
        ):
            raise ValueError("Relations must reference two distinct entity indices.")
        if (
            isinstance(axis, (bool, np.bool_))
            or not isinstance(axis, (int, np.integer))
            or axis not in (0, 1)
            or sign not in (-1.0, 1.0)
        ):
            raise ValueError("Relations require axis 0/1 and sign -1/+1.")
        row = np.zeros((n, 2))
        row[source, axis], row[target, axis] = sign, -sign
        rows.append(row.reshape(-1))
        limits.append(
            -float(half[source, axis] + half[target, axis] + relation_clearance_m)
        )

    def overlaps(xy: np.ndarray) -> list[tuple[float, int, int]]:
        pairs = []
        margin = collision_margin_m if enforce_all_clearances else 0.0
        for i in range(n):
            for j in range(i + 1, n):
                if skip_fixed_collisions and fixed[i] and fixed[j]:
                    continue
                penetration = half[i] + half[j] + margin - np.abs(xy[i] - xy[j])
                if np.all(penetration > feasibility_tolerance):
                    pairs.append((float(np.min(penetration)), i, j))
        return sorted(pairs, reverse=True)

    def report(xy: np.ndarray, status: str, message: str) -> AABBLayoutResult:
        linear = max(
            (float(row @ xy.reshape(-1) - bound) for row, bound in zip(rows, limits)),
            default=0.0,
        )
        fixed_error = (
            float(np.max(np.abs(xy[fixed] - initial[fixed]))) if np.any(fixed) else 0.0
        )
        residuals = {
            "linear": max(0.0, linear),
            "fixed": fixed_error,
            "collision": max((p[0] for p in overlaps(xy)), default=0.0),
        }
        return AABBLayoutResult(status, xy.copy(), residuals, message)

    if np.any(lower > upper):
        return report(initial, "no_solution", "An AABB exceeds its assigned region.")
    # Eliminate fixed variables instead of feeding redundant equalities to SLSQP.
    moving = np.repeat(~fixed, 2)
    fixed_values = initial.reshape(-1).copy()

    def solve(
        start: np.ndarray,
        constraint_rows: list[np.ndarray],
        constraint_limits: list[float],
    ) -> tuple[np.ndarray, str, str]:
        if not np.any(moving):
            residual = max(
                (
                    float(row @ fixed_values - bound)
                    for row, bound in zip(constraint_rows, constraint_limits)
                ),
                default=0.0,
            )
            return (
                initial.copy(),
                "solved" if residual <= feasibility_tolerance else "no_solution",
                "All positions are fixed.",
            )
        matrix = np.asarray(constraint_rows, dtype=float).reshape(-1, 2 * n)
        bound = np.asarray(constraint_limits)
        effective = bound - matrix[:, ~moving] @ fixed_values[~moving]
        matrix = matrix[:, moving]
        # Constant constraints need checking but must not produce zero rows in SLSQP.
        constant = np.all(matrix == 0, axis=1)
        if np.any(effective[constant] < -feasibility_tolerance):
            return start.copy(), "no_solution", "Fixed positions violate a constraint."
        matrix, effective = matrix[~constant], effective[~constant]
        reference = initial.reshape(-1)[moving]
        row_weights = np.repeat(weights, 2)[moving]
        result = minimize(
            lambda x: float(np.sum(row_weights * (x - reference) ** 2)),
            start.reshape(-1)[moving],
            jac=lambda x: 2 * row_weights * (x - reference),
            method="SLSQP",
            constraints=(
                [
                    {
                        "type": "ineq",
                        "fun": lambda x: effective - matrix @ x,
                        "jac": lambda x: -matrix,
                    }
                ]
                if len(matrix)
                else []
            ),
            options={
                "maxiter": max_slsqp_iterations,
                "ftol": slsqp_ftol,
                "disp": False,
            },
        )
        full = fixed_values.copy()
        full[moving] = result.x
        if not np.all(np.isfinite(full)):
            return (
                start.copy(),
                "no_solution",
                "Optimizer returned nonfinite coordinates.",
            )
        residual = float(np.max(matrix @ result.x - effective)) if len(matrix) else 0.0
        status = (
            "solved"
            if result.success and residual <= feasibility_tolerance
            else ("budget_exhausted" if result.status == 9 else "no_solution")
        )
        return full.reshape(n, 2), status, str(result.message)

    current, status, message = solve(initial, rows, limits)
    if status != "solved":
        return report(current, status, message)
    for _ in range(max_collision_rounds):
        pairs = overlaps(current)
        if not pairs:
            return report(current, "solved", "Linear and AABB checks passed.")
        for _, i, j in pairs[:max_added_collision_pairs]:
            if not any(a == i and b == j for _, a, b in overlaps(current)):
                continue
            penetration = half[i] + half[j] - np.abs(current[i] - current[j])
            exhausted = False
            accepted = False
            for axis in np.argsort(penetration):
                sign = 1.0 if current[i, axis] <= current[j, axis] else -1.0
                for direction in (sign, -sign):
                    row = np.zeros((n, 2))
                    row[i, axis], row[j, axis] = direction, -direction
                    row = row.reshape(-1)
                    bound = -float(half[i, axis] + half[j, axis] + collision_margin_m)
                    candidate, candidate_status, _ = solve(
                        current, [*rows, row], [*limits, bound]
                    )
                    exhausted |= candidate_status == "budget_exhausted"
                    if candidate_status == "solved":
                        rows.append(row)
                        limits.append(bound)
                        current = candidate
                        accepted = True
                        break
                if accepted:
                    break
            if not accepted:
                return report(
                    current,
                    "budget_exhausted" if exhausted else "no_solution",
                    f"No accepted AABB separation for indices {i}, {j}.",
                )
    if not overlaps(current):
        return report(current, "solved", "Linear and AABB checks passed.")
    return report(current, "budget_exhausted", "Collision refinement budget exhausted.")
