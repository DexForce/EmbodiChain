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

"""Select continuous analytical IK branches over an entire Cartesian path."""

from __future__ import annotations

from itertools import product
from numbers import Real

import numpy as np
from scipy.spatial import cKDTree

__all__ = ["select_ik_paths"]


def _joint_cost_weights(value: object, joint_count: int) -> np.ndarray:
    """Validate independent joint penalties without changing motion limits."""
    if value is None:
        return np.ones(joint_count, dtype=float)
    raw = np.asarray(value, dtype=object)
    if raw.shape != (joint_count,) or any(
        isinstance(item, (bool, np.bool_)) or not isinstance(item, Real)
        for item in raw.flat
    ):
        raise ValueError("joint_cost_weights must contain one numeric weight per joint")
    weights = raw.astype(float)
    if not np.isfinite(weights).all() or np.any(weights < 0) or not np.any(weights > 0):
        raise ValueError(
            "joint_cost_weights must be finite and nonnegative, with a positive weight"
        )
    return weights


def _bounded_layer(values: np.ndarray, limits: np.ndarray) -> np.ndarray:
    """Expand physical solutions into every winding allowed by the joint limits."""
    canonical = np.arctan2(np.sin(values), np.cos(values))
    # UR analytical IK emits float32 full-turn aliases whose wrapped angles
    # differ by around 1e-6 rad. Merge those before Cartesian-product expansion.
    _, unique_ids = np.unique(np.round(canonical, 4), axis=0, return_index=True)
    expanded = []
    period = 2 * np.pi
    for row in canonical[np.sort(unique_ids)]:
        lower = np.ceil((limits[:, 0] - row - 1e-9) / period).astype(int)
        upper = np.floor((limits[:, 1] - row + 1e-9) / period).astype(int)
        if np.any(lower > upper):
            continue
        shifts = np.asarray(
            list(product(*(range(low, high + 1) for low, high in zip(lower, upper)))),
            dtype=float,
        )
        # Clip only numerical noise at an exactly inclusive physical limit.
        expanded.append(np.clip(row + period * shifts, limits[:, 0], limits[:, 1]))
    if not expanded:
        return np.empty((0, values.shape[1]), dtype=float)
    result = np.concatenate(expanded)
    _, unique_ids = np.unique(np.round(result, 4), axis=0, return_index=True)
    return result[np.sort(unique_ids)]


def select_ik_paths(
    solution_layers: list[np.ndarray],
    seed: np.ndarray,
    limits: np.ndarray,
    max_step_degrees: float,
    *,
    joint_cost_weights: list[float] | np.ndarray | None = None,
) -> list[np.ndarray]:
    """Find bounded low-travel branch paths, considering future reachability.

    Args:
        solution_layers: Valid analytical solutions (K, J) for each pose, in radians.
        seed: Reference starting posture (J,); home-to-start travel is not executed.
        limits: Physical joint limits (J, 2).
        max_step_degrees: Maximum continuous joint increment between IK samples.
        joint_cost_weights: Nonnegative joint penalties, or uniform weights when
            omitted. They affect path ranking and the starting-posture preference;
            physical joint limits and the maximum increment remain unweighted.

    Returns:
        Up to eight physical paths (N, J), sorted by weighted squared joint-step
        cost with a small starting-posture preference. Every allowed full-turn alias is a
        separate search state, so joint-limit feasibility is preserved through
        the whole route. Paths equivalent modulo full turns are returned once.
        This searches the supplied discrete IK layers; it does not prove that an
        arbitrary Cartesian route is unreachable when no candidate is returned.
    """
    seed, limits = np.asarray(seed, dtype=float), np.asarray(limits, dtype=float)
    if (
        seed.ndim != 1
        or seed.size == 0
        or limits.shape != (len(seed), 2)
        or not np.isfinite(limits).all()
        or not np.isfinite(seed).all()
        or np.any(limits[:, 0] >= limits[:, 1])
    ):
        raise ValueError("Invalid IK seed or joint limits")
    if not np.isfinite(max_step_degrees) or max_step_degrees <= 0:
        raise ValueError("max_step_degrees must be finite and positive")
    weights = _joint_cost_weights(joint_cost_weights, len(seed))
    layers = []
    for values in solution_layers:
        values = np.asarray(values, dtype=float)
        if (
            values.ndim != 2
            or values.shape[1] != len(seed)
            or not np.isfinite(values).all()
        ):
            raise ValueError("Invalid IK solution layer")
        bounded = _bounded_layer(values, limits)
        if len(bounded) == 0:
            return []
        layers.append(bounded)
    if not layers:
        return []
    # A tiny home preference breaks ties; all executed edges use actual angles.
    cost = 1e-5 * np.sum(weights * (layers[0] - seed) ** 2, axis=1)
    parents = []
    max_step = np.deg2rad(max_step_degrees)
    for before, after in zip(layers, layers[1:]):
        reachable = np.flatnonzero(np.isfinite(cost))
        neighbors = cKDTree(before[reachable]).query_ball_point(
            after, max_step + 1e-12, p=np.inf
        )
        counts = np.fromiter((len(ids) for ids in neighbors), dtype=int)
        if counts.sum() == 0:
            return []
        destination = np.repeat(np.arange(len(after)), counts)
        source = reachable[np.concatenate(neighbors).astype(int)]
        delta = after[destination] - before[source]
        edge_cost = cost[source] + np.einsum("ij,j,ij->i", delta, weights, delta)
        next_cost = np.full(len(after), np.inf)
        np.minimum.at(next_cost, destination, edge_cost)
        best = edge_cost == next_cost[destination]
        parent = np.full(len(after), len(before), dtype=int)
        np.minimum.at(parent, destination[best], source[best])
        parent[~np.isfinite(next_cost)] = -1
        parents.append(parent)
        cost = next_cost
    paths, seen = [], set()
    for index in np.argsort(cost, kind="stable"):
        if not np.isfinite(cost[index]):
            break
        selected = [int(index)]
        for parent in reversed(parents):
            selected.append(int(parent[selected[-1]]))
        path = np.array([layer[i] for layer, i in zip(layers, reversed(selected))])
        # Trigonometric signatures identify aliases even across the +/-pi cut.
        signature = np.round(np.stack((np.sin(path), np.cos(path)), axis=-1), 4)
        signature[signature == 0] = 0.0
        key = signature.tobytes()
        if key in seen:
            continue
        seen.add(key)
        paths.append(path)
        if len(paths) == 8:
            break
    return paths
