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

"""Seeded tabletop recipes using the same layout and proposal contracts as hosts."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, replace

import numpy as np

from ._validation import _identifier, _nonnegative_integer
from .composition import SceneFragment, _config_pose
from .expansion import ScenePoseChange, SceneVariant
from .layout import LayoutCfg, LayoutProblem, LayoutSolution, solve_layout
from .validation import SceneCheck, SceneCheckReport, validate_scene

__all__ = ["SceneRecipe", "GeneratedScene", "generate", "check_layout"]


class SceneRecipe:
    """Own resolved asset configurations and an explicit planar layout recipe.

    Args:
        recipe_id: Versioned scene/asset identity supplied by the recipe author.
        fragment: Existing entity configurations. Asset generation is upstream.
        layout: Geometry-center seeds and AABBs at the fragment's initial poses.
            Fixed virtual entities without configurations represent keep-out
            zones. All movable entities must have matching configurations.

    Geometry proxies already include asset scale and initial orientation. The
    first implementation changes XY translation only; objects retain their
    authored height and rotation. A recipe supplies the physical support height
    rather than inferring it from a filename or semantic category.

    A layout UID may represent a whole-group AABB whose center differs from
    the configured root origin. Generation translates that root by the solved
    center displacement; it does not traverse support graphs or move members
    absent from the layout. The caller must expand group transforms and movable
    permissions to every member before applying a complete group proposal.
    """

    def __init__(
        self, recipe_id: str, fragment: SceneFragment, layout: LayoutProblem
    ) -> None:
        _identifier(recipe_id, "recipe_id")
        if not isinstance(fragment, SceneFragment) or not isinstance(
            layout, LayoutProblem
        ):
            raise TypeError("recipe requires a SceneFragment and LayoutProblem")
        from embodichain.lab.sim.cfg import RigidObjectCfg

        entities = {cfg.uid: cfg for cfg in fragment.entities}
        if layout.support_uid not in entities:
            raise ValueError("Recipe support must name a fragment entity")
        if type(layout.skip_fixed_collisions) is not bool:
            raise ValueError("skip_fixed_collisions must be a boolean")
        uids = set(layout.seed_xy_by_uid)
        for uid in uids:
            _identifier(uid, "layout UID")
        if set(layout.half_extents_xy_by_uid) != uids:
            raise ValueError("Recipe geometry must match the layout UID set")
        if not set(layout.fixed_xy_by_uid).issubset(uids) or not set(
            layout.bounds_xy_by_uid
        ).issubset(uids):
            raise ValueError(
                "Recipe bounds or fixed positions reference an unknown UID"
            )
        if layout.preferred_uids is not None and not set(
            layout.preferred_uids
        ).issubset(uids):
            raise ValueError("Recipe preferences reference an unknown UID")
        if any(
            r.source_uid not in uids or r.target_uid not in uids
            for r in layout.relations
        ):
            raise ValueError("Recipe relations reference an unknown UID")
        movable = uids - set(layout.fixed_xy_by_uid)
        if not movable.issubset(entities):
            raise ValueError("Every movable layout entity must have a configuration")
        if layout.support_uid in movable:
            raise ValueError("The recipe support cannot be a movable layout entity")
        if any(not isinstance(entities[uid], RigidObjectCfg) for uid in movable):
            raise ValueError(
                "Movable recipe entities must be rigid-object configurations"
            )
        if not movable.issubset(layout.bounds_xy_by_uid):
            raise ValueError("Recipe sampling requires bounds for every movable entity")
        for uid in uids:
            center = np.asarray(layout.seed_xy_by_uid[uid], dtype=float)
            half = np.asarray(layout.half_extents_xy_by_uid[uid], dtype=float)
            if (
                center.shape != (2,)
                or half.shape != (2,)
                or not np.isfinite(center).all()
                or not np.isfinite(half).all()
                or np.any(half < 0)
            ):
                raise ValueError(
                    "Recipe centers and half extents must be finite XY pairs with nonnegative extents"
                )
            if uid in layout.fixed_xy_by_uid:
                fixed = np.asarray(layout.fixed_xy_by_uid[uid], dtype=float)
                if fixed.shape != (2,) or not np.isfinite(fixed).all():
                    raise ValueError("Recipe fixed positions must be finite XY pairs")
                if uid in entities and not np.array_equal(center, fixed):
                    raise ValueError(
                        "Configured fixed positions must match their authored geometry-center seeds"
                    )
            if uid in layout.bounds_xy_by_uid:
                bounds = np.asarray(layout.bounds_xy_by_uid[uid], dtype=float)
                if (
                    bounds.shape != (2, 2)
                    or np.isnan(bounds).any()
                    or np.any(bounds[0] > bounds[1])
                ):
                    raise ValueError(
                        "Recipe bounds must be ordered XY minimum/maximum pairs"
                    )
                if uid in movable and not np.isfinite(bounds).all():
                    raise ValueError("Recipe sampling requires finite XY bounds")
        self._recipe_id = recipe_id
        self._fragment = SceneFragment(fragment.entities, anchors=fragment.anchors)
        self._layout = deepcopy(layout)

    @property
    def recipe_id(self) -> str:
        """Return the author-supplied versioned asset/scene identity."""
        return self._recipe_id

    @property
    def fragment(self) -> SceneFragment:
        """Return a detached copy of the original configuration fragment."""
        return SceneFragment(self._fragment.entities, anchors=self._fragment.anchors)

    @property
    def layout(self) -> LayoutProblem:
        """Return an owned copy of the original proxy geometry and constraints."""
        return deepcopy(self._layout)


@dataclass(frozen=True)
class GeneratedScene:
    """One recipe attempt with separate proposal and geometry evidence.

    Failed layout attempts have no fragment or variant. Even a passed geometry
    report is not physical stability, reachability, or task acceptance.
    """

    fragment: SceneFragment | None
    variant: SceneVariant | None
    layout: LayoutSolution
    checks: SceneCheckReport


def generate(
    recipe: SceneRecipe,
    *,
    seed: int = 0,
    ordinal: int = 0,
    cfg: LayoutCfg | None = None,
) -> GeneratedScene:
    """Sample a bounded recipe and solve it without creating a simulator.

    Args:
        recipe: Resolved configurations and planar constraints.
        seed: Nonnegative proposal seed.
        ordinal: Logical candidate index, independent of slots and scheduling.
        cfg: Shared solver controls.

    Returns:
        A fresh fragment and an existing-scene-compatible pose variant when
        solved, or a failed attempt with the original solver status and checks.
    """
    _nonnegative_integer(seed, "seed")
    _nonnegative_integer(ordinal, "ordinal")
    problem = recipe.layout
    rng = np.random.default_rng(np.random.SeedSequence([seed, ordinal]))
    sampled = dict(problem.seed_xy_by_uid)
    for uid in sorted(set(sampled) - set(problem.fixed_xy_by_uid)):
        bounds = np.asarray(problem.bounds_xy_by_uid[uid], dtype=float)
        half = np.asarray(problem.half_extents_xy_by_uid[uid], dtype=float)
        if (
            bounds.shape != (2, 2)
            or half.shape != (2,)
            or not np.isfinite(bounds).all()
            or not np.isfinite(half).all()
        ):
            raise ValueError(
                "Recipe sampling requires finite XY bounds and half extents"
            )
        minimum, maximum = bounds[0] + half, bounds[1] - half
        if np.all(minimum <= maximum):
            sampled[uid] = tuple(rng.uniform(minimum, maximum))
    solution = solve_layout(replace(problem, seed_xy_by_uid=sampled), cfg)
    checks = validate_scene(
        (
            SceneCheck(
                "scene.layout",
                (
                    "passed"
                    if solution.status == "solved"
                    else "unverified" if solution.status == "unsupported" else "failed"
                ),
                solution.message,
                solution.residuals,
            ),
        )
    )
    if solution.status != "solved":
        return GeneratedScene(None, None, solution, checks)
    entities = recipe.fragment.entities
    changes = []
    for entity in entities:
        if entity.uid not in sampled or entity.uid in problem.fixed_xy_by_uid:
            continue
        pose = _config_pose(entity)
        delta = np.asarray(solution.xy_by_uid[entity.uid]) - np.asarray(
            problem.seed_xy_by_uid[entity.uid]
        )
        pose[:2, 3] += delta
        entity.init_local_pose = pose
        entity.init_pos = tuple(pose[:3, 3])
        changes.append(ScenePoseChange(entity.uid, pose.tolist()))
    variant = SceneVariant(
        recipe.recipe_id,
        seed,
        ordinal,
        tuple(changes),
        (("generator", "scene.procedural/v1"),),
    )
    return GeneratedScene(
        SceneFragment(entities, anchors=recipe.fragment.anchors),
        variant,
        solution,
        checks,
    )


def check_layout(
    recipe: SceneRecipe,
    poses_by_uid: dict[str, object],
    *,
    cfg: LayoutCfg | None = None,
    actual_aabbs_xy_by_uid: Mapping[str, object] | None = None,
) -> SceneCheckReport:
    """Measure actual poses against a recipe's arena-local XY constraints.

    Args:
        recipe: Original asset poses and geometry-center proxy layout.
        poses_by_uid: Measured 4x4 arena-local transforms for every configured
            layout entity and its support. Virtual obstacles retain their fixed
            recipe positions, even when their seed positions differ.
        cfg: The same constraint clearances used during generation.
        actual_aabbs_xy_by_uid: Optional measured arena-frame AABB minimum and
            maximum XY corners, shaped ``(2, 2)``, for configured entities.
            These update centers and half extents and permit orientation changes
            of movable objects. For group proxies, supply the union of all
            members' measured AABBs under the root UID. The caller owns geometry
            measurement and separately checks member poses, fixed members and
            preservation of the internal group structure. Without measurements,
            a root translation only shifts the authored proxy; it cannot prove
            that every group member followed the root.

    Returns:
        Measured layout evidence. Missing poses, changed support or orientation
        changes without measured AABBs are unverified. Moving a fixed entity
        fails even when updated AABBs are supplied. This check does not establish
        vertical support, physical stability or task success.
    """
    from ._validation import _rigid_pose

    cfg = cfg or LayoutCfg()
    problem = recipe.layout
    positions = {
        uid: problem.fixed_xy_by_uid.get(uid, center)
        for uid, center in problem.seed_xy_by_uid.items()
    }
    half_extents = dict(problem.half_extents_xy_by_uid)
    entities = {entity.uid: entity for entity in recipe.fragment.entities}
    measured_aabbs = {}
    for uid, value in (actual_aabbs_xy_by_uid or {}).items():
        if uid not in entities:
            raise ValueError(f"Measured AABB requires a configured entity UID: {uid!r}")
        bounds = np.asarray(value, dtype=float)
        if (
            bounds.shape != (2, 2)
            or not np.isfinite(bounds).all()
            or np.any(bounds[0] > bounds[1])
        ):
            raise ValueError(
                "Measured AABBs must contain finite ordered XY minimum/maximum corners"
            )
        measured_aabbs[uid] = bounds
    checks = []
    for uid in sorted({problem.support_uid, *set(positions).intersection(entities)}):
        expected = _config_pose(entities[uid])
        if uid not in poses_by_uid:
            checks.append(
                SceneCheck(
                    f"scene.pose.{uid}", "unverified", "Required actual pose is missing"
                )
            )
            continue
        actual = np.asarray(_rigid_pose(poses_by_uid[uid]))
        unchanged = np.allclose(actual, expected, atol=1e-5, rtol=0)
        if uid == problem.support_uid and not unchanged:
            checks.append(
                SceneCheck(
                    f"scene.pose.{uid}",
                    "unverified",
                    "Changed support invalidates the arena-frame recipe",
                )
            )
            continue
        if uid in problem.fixed_xy_by_uid and not unchanged:
            checks.append(
                SceneCheck(f"scene.fixed.{uid}", "failed", "Fixed entity pose changed")
            )
            continue
        if uid not in measured_aabbs and not np.allclose(
            actual[:3, :3], expected[:3, :3], atol=1e-5, rtol=0
        ):
            checks.append(
                SceneCheck(
                    f"scene.pose.{uid}",
                    "unverified",
                    "Changed orientation requires measured geometry",
                )
            )
            continue
        if uid in positions:
            if uid in measured_aabbs:
                bounds = measured_aabbs[uid]
                positions[uid] = tuple((bounds[0] + bounds[1]) / 2)
                half_extents[uid] = tuple((bounds[1] - bounds[0]) / 2)
            else:
                positions[uid] = tuple(
                    np.asarray(problem.seed_xy_by_uid[uid])
                    + actual[:2, 3]
                    - expected[:2, 3]
                )
            if uid in problem.fixed_xy_by_uid and not np.allclose(
                positions[uid], problem.fixed_xy_by_uid[uid], atol=1e-5, rtol=0
            ):
                checks.append(
                    SceneCheck(
                        f"scene.fixed.{uid}", "failed", "Fixed geometry center moved"
                    )
                )
    if checks:
        return validate_scene(checks)
    # Fix all variables to their measurements to evaluate boundaries and relations
    # without repairing the scene. Handle collisions separately so only the
    # recipe's ORIGINAL fixed-fixed exemptions apply.
    fixed_problem = replace(
        problem,
        seed_xy_by_uid=positions,
        half_extents_xy_by_uid=half_extents,
        fixed_xy_by_uid=positions,
        skip_fixed_collisions=True,
    )
    result = solve_layout(fixed_problem, cfg)
    if result.status == "unsupported":
        return validate_scene(
            (SceneCheck("scene.layout.actual", "unverified", result.message),)
        )
    collision = 0.0
    uids = list(positions)
    margin = cfg.collision_margin_m if cfg.enforce_all_clearances else 0.0
    for index, first in enumerate(uids):
        for second in uids[index + 1 :]:
            if (
                problem.skip_fixed_collisions
                and first in problem.fixed_xy_by_uid
                and second in problem.fixed_xy_by_uid
            ):
                continue
            penetration = (
                np.asarray(half_extents[first])
                + half_extents[second]
                + margin
                - np.abs(np.asarray(positions[first]) - positions[second])
            )
            collision = max(collision, max(0.0, float(np.min(penetration))))
    residuals = {**result.residuals, "collision": collision}
    passed = result.status == "solved" and collision <= cfg.feasibility_tolerance
    return validate_scene(
        (
            SceneCheck(
                "scene.layout.actual",
                "passed" if passed else "failed",
                "Measured XY proxy constraints" + (" passed" if passed else " failed"),
                residuals,
            ),
        )
    )
