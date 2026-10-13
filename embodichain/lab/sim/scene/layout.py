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

"""UID-based horizontal layout proposals independent of simulation execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from embodichain.lab.sim.scene._layout_cpu import solve_aabb_layout
from embodichain.utils import configclass

__all__ = [
    "LayoutCfg",
    "PlanarRelation",
    "LayoutProblem",
    "LayoutSolution",
    "solve_layout",
]


@configclass
class LayoutCfg:
    """Budgets and metric tolerances for fixed-orientation planar layout.

    ``enforce_all_clearances=False`` is a legacy compatibility option: only
    intersecting boxes are repaired, with the requested collision margin.
    """

    relation_clearance_m: float = 0.03
    collision_margin_m: float = 0.02
    max_slsqp_iterations: int = 500
    slsqp_ftol: float = 1e-6
    max_collision_rounds: int = 8
    max_added_collision_pairs: int = 64
    imported_seed_weight: float = 5.0
    feasibility_tolerance: float = 1e-7
    enforce_all_clearances: bool = True


@dataclass(frozen=True)
class PlanarRelation:
    """Axis-aligned spatial relation in an explicitly named reference frame.

    Arena XY means left=-X, right=+X, behind=-Y and front=+Y. Unsupported
    frames are reported instead of silently interpreting local axes as arena.
    """

    source_uid: str
    relation: str
    target_uid: str
    frame: str = "arena"


@dataclass(frozen=True)
class LayoutProblem:
    """A horizontal support problem with fixed orientations and AABB geometry.

    Bounds contain minimum and maximum corners for each complete AABB, not
    center bounds. Missing bounds mean unbounded placement (e.g. obstacles).
    Fixed positions override seeds. The support UID identifies the support
    only; it is not a reserved entity name. Callers own support height, mesh
    measurement, grouping and hierarchical solve order.
    """

    support_uid: str
    seed_xy_by_uid: dict[str, tuple[float, float]]
    half_extents_xy_by_uid: dict[str, tuple[float, float]]
    bounds_xy_by_uid: dict[str, tuple[tuple[float, float], tuple[float, float]]]
    fixed_xy_by_uid: dict[str, tuple[float, float]] = field(default_factory=dict)
    relations: tuple[PlanarRelation, ...] = ()
    preferred_uids: tuple[str, ...] | None = None
    frame: str = "arena"
    skip_fixed_collisions: bool = False


@dataclass(frozen=True)
class LayoutSolution:
    """Proposed centers and measured violations; never a physical-validity claim.

    ``no_solution`` describes the numerical search outcome, not a proof that
    all possible layouts are infeasible. Failure candidates must not be applied.
    """

    status: Literal["solved", "no_solution", "budget_exhausted", "unsupported"]
    xy_by_uid: dict[str, tuple[float, float]]
    residuals: dict[str, float]
    message: str
    geometry: str = "fixed_orientation_aabb"


def solve_layout(
    problem: LayoutProblem, cfg: LayoutCfg | None = None
) -> LayoutSolution:
    """Generate an arena-local XY proposal using the CPU SLSQP scene strategy.

    This policy does not read or change physics. Shared tensor geometry kernels
    support other devices independently; this search remains CPU-only.

    Args:
        problem: UID, geometry, boundary and relation inputs in metres.
        cfg: Numerical controls; defaults to :class:`LayoutCfg`.

    Returns:
        Candidate centers and residuals. Unsupported coordinate frames return
        ``unsupported``; malformed inputs raise ``ValueError``.
    """
    cfg = cfg or LayoutCfg()
    cfg.validate()
    if not np.isfinite(cfg.imported_seed_weight) or cfg.imported_seed_weight < 0:
        raise ValueError("imported_seed_weight must be finite and nonnegative.")
    if not isinstance(problem.support_uid, str) or not problem.support_uid:
        raise ValueError("A nonempty support UID is required.")
    uids = list(problem.seed_xy_by_uid)
    if any(not isinstance(uid, str) or not uid for uid in uids):
        raise ValueError("Entity UIDs must be nonempty strings.")
    uid_set = set(uids)
    if set(problem.half_extents_xy_by_uid) != uid_set:
        raise ValueError("Half-extents must match the entity UID set.")
    for mapping in (problem.bounds_xy_by_uid, problem.fixed_xy_by_uid):
        if not set(mapping).issubset(uid_set):
            raise ValueError("Bounds or fixed positions reference an unknown UID.")
    preferred = (
        uid_set if problem.preferred_uids is None else set(problem.preferred_uids)
    )
    if not preferred.issubset(uid_set):
        raise ValueError("Preferred positions reference an unknown UID.")
    for relation in problem.relations:
        if relation.source_uid not in uid_set or relation.target_uid not in uid_set:
            raise ValueError("A relation references an unknown UID.")
    relation_axes = {
        "left_of": (0, 1.0),
        "right_of": (0, -1.0),
        "behind": (1, 1.0),
        "in_front_of": (1, -1.0),
    }
    if problem.frame != "arena" or any(
        r.frame != "arena" or r.relation not in relation_axes for r in problem.relations
    ):
        return LayoutSolution(
            "unsupported",
            {},
            {},
            "Only arena-frame axis-aligned planar relations are supported.",
        )
    index = {uid: i for i, uid in enumerate(uids)}
    count = len(uids)
    seeds = np.asarray(
        [problem.fixed_xy_by_uid.get(uid, problem.seed_xy_by_uid[uid]) for uid in uids],
        dtype=float,
    ).reshape(count, 2)
    half = np.asarray(
        [problem.half_extents_xy_by_uid[uid] for uid in uids], dtype=float
    ).reshape(count, 2)
    bounds = np.asarray(
        [
            problem.bounds_xy_by_uid.get(uid, ((-np.inf, -np.inf), (np.inf, np.inf)))
            for uid in uids
        ],
        dtype=float,
    ).reshape(count, 2, 2)
    fixed = np.asarray([uid in problem.fixed_xy_by_uid for uid in uids], dtype=bool)
    weights = np.asarray(
        [cfg.imported_seed_weight if uid in preferred else 0.0 for uid in uids]
    )
    result = solve_aabb_layout(
        seeds,
        half,
        bounds,
        fixed,
        weights,
        tuple(
            (index[r.source_uid], index[r.target_uid], *relation_axes[r.relation])
            for r in problem.relations
        ),
        relation_clearance_m=cfg.relation_clearance_m,
        collision_margin_m=cfg.collision_margin_m,
        max_slsqp_iterations=cfg.max_slsqp_iterations,
        slsqp_ftol=cfg.slsqp_ftol,
        max_collision_rounds=cfg.max_collision_rounds,
        max_added_collision_pairs=cfg.max_added_collision_pairs,
        feasibility_tolerance=cfg.feasibility_tolerance,
        skip_fixed_collisions=problem.skip_fixed_collisions,
        enforce_all_clearances=cfg.enforce_all_clearances,
    )
    return LayoutSolution(
        result.status,
        {uid: tuple(map(float, result.xy[i])) for i, uid in enumerate(uids)},
        result.residuals,
        result.message,
    )
