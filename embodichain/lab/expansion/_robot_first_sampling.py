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

"""Owned workspace priors for conditional layout completion."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from embodichain.lab.sim.scene.expansion import SceneLayoutCondition
from embodichain.lab.sim.scene.feasibility.proposals import _workspace_fk_samples

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import Robot
    from embodichain.lab.sim.scene.expansion import PreparedSceneRandomization
    from embodichain.lab.sim.scene.feasibility import (
        FeasibilityBudget,
        PhaseTarget,
        TaskOperationMode,
    )

__all__: list[str] = []


@dataclass(frozen=True)
class _AnchorTarget:
    phase_id: str
    target: PhaseTarget
    root_uid: str


def _anchor_targets(
    mode: TaskOperationMode, layout: Mapping, policy: Mapping, selections: object
) -> tuple[_AnchorTarget, ...]:
    """Resolve each named target to a rigid frame in the initial support tree."""
    if not isinstance(selections, list) or not selections:
        raise ValueError(
            "robot_anchors requires a nonempty list of phase/part selectors"
        )
    targets = {}
    moved = set()
    support, parents = layout["support_uid"], layout["parent_by_uid"]
    variable_members = policy["intra_group"]["targets"]
    for phase in mode.phases:
        for target in phase.targets:
            targets[phase.phase_id, target.control_part] = (target, moved.copy())
        moved.update(held.entity_uid for held in phase.held_objects)
        moved.update(change.entity_uid for change in phase.resulting_poses)
    result, seen = [], set()
    for sel in selections:
        if (
            not isinstance(sel, Mapping)
            or set(sel) != {"phase_id", "control_part"}
            or any(type(value) is not str for value in sel.values())
        ):
            raise ValueError("robot_anchors selects explicit phase_id and control_part")
        key = (sel["phase_id"], sel["control_part"])
        if key not in targets or key in seen:
            raise ValueError("robot_anchors must select unique declared phase targets")
        seen.add(key)
        target, earlier_moves = targets[key]
        root = target.frame_uid
        if root not in parents:
            raise ValueError("A robot anchor target needs a frame in the support tree")
        while True:
            if root in variable_members or root in earlier_moves:
                raise ValueError(
                    "A robot anchor needs an initial frame rigidly attached to its layout root"
                )
            if parents[root] == support:
                break
            root = parents[root]
        groups = policy["group_targets"]
        if not policy["randomize_groups"] or (
            groups is not None and root not in groups
        ):
            raise ValueError("A robot anchor requires a movable layout root")
        if root in layout.get("conditions", {}):
            raise ValueError("A robot anchor cannot replace an authored root condition")
        result.append(_AnchorTarget(key[0], target, root))
    return tuple(result)


@dataclass(frozen=True)
class _AnchorPool:
    selection: _AnchorTarget
    base: np.ndarray
    buckets: tuple[np.ndarray, ...]

    def condition(
        self, visit: int, fixed: SceneLayoutCondition
    ) -> SceneLayoutCondition | None:
        """Cycle over occupied cells and shuffle within each cell once per job."""
        if not self.buckets:
            return None
        cycle, cell = divmod(visit, len(self.buckets))
        points = self.buckets[cell]
        pose = self.base.copy()
        pose[:2, 3] = points[cycle % len(points)]
        return SceneLayoutCondition({**fixed.poses, self.selection.root_uid: pose})

    def metrics(self) -> dict[str, object]:
        return {
            "phase_id": self.selection.phase_id,
            "control_part": self.selection.target.control_part,
            "root_uid": self.selection.root_uid,
            "samples_in_domain": sum(map(len, self.buckets)),
            "occupied_xy_cells": len(self.buckets),
        }


def _prepare_anchor_pools(
    robot: Robot,
    budget: FeasibilityBudget,
    prepared: PreparedSceneRandomization,
    config: Mapping[str, Any],
    selections: Sequence[_AnchorTarget],
    scene: Mapping[str, Any],
    nominal: Mapping[str, np.ndarray],
    *,
    seed: int,
) -> tuple[_AnchorPool, ...]:
    """Precompute at most sample_count FK rows per arm, retaining only legal XY."""
    policy = prepared.policy
    movable_roots = {
        root
        for root, parent in prepared.source.parent_by_uid.items()
        if parent == prepared.source.support_uid and root in prepared.movable_uids
    }
    nodes_by_part: dict[str, list[tuple[int, _AnchorTarget]]] = {}
    for index, selection in enumerate(selections):
        root = selection.root_uid
        if root not in movable_roots or root not in scene or "follow" in scene[root]:
            raise ValueError(
                f"Robot anchor root {root!r} must have independent XY bounds"
            )
        nodes_by_part.setdefault(selection.target.control_part, []).append(
            (index, selection)
        )

    pools = {}
    for part, entries in nodes_by_part.items():
        part_index = entries[0][0]
        fk = _workspace_fk_samples(
            robot,
            budget,
            control_part=part,
            source=config["workspace_source"],
            sample_count=config["sample_count"],
            seed=seed + part_index,
        )
        for index, selection in entries:
            root, target = selection.root_uid, selection.target
            base = np.array(prepared.source.poses[root], dtype=float)
            bounds = np.asarray(scene[root]["translation_m"], dtype=float)[:2].T
            bounds = bounds + np.asarray(nominal[root])[:2, 3]
            offsets = policy["group_xy_offset_bounds_m"].get(root)
            if offsets is not None:
                layout_bounds = base[:2, 3] + np.asarray(offsets)
                bounds[0] = np.maximum(bounds[0], layout_bounds[0])
                bounds[1] = np.minimum(bounds[1], layout_bounds[1])
            lower, upper = bounds
            if np.any(upper < lower):
                raise ValueError(f"Robot anchor root {root!r} has disjoint XY bounds")

            # Preserve sampled TCP XY under the root's allowed rotation/height.
            # Sampled TCP height/orientation are not retained by this projection.
            frame = np.asarray(prepared.source.poses[target.frame_uid])
            root_from_tcp = np.linalg.solve(base, frame @ np.asarray(target.pose))
            xy = fk[:, :2, 3] - (base[:3, :3] @ root_from_tcp[:3, 3])[:2]
            xy[:, lower == upper] = lower[lower == upper]
            xy = xy[np.all((lower <= xy) & (xy <= upper), axis=1)]
            # Balance across occupied cells, so a dense set of joint samples in
            # one region cannot dominate every proposal. No extrapolation.
            size = budget.cfg.archive_bins
            scaled = np.divide(
                xy - lower, upper - lower, out=np.zeros_like(xy), where=upper > lower
            )
            cells = np.minimum((scaled * size).astype(int), size - 1)
            occupied = cells[:, 0] * size + cells[:, 1]
            rng = np.random.default_rng(np.random.SeedSequence([seed, index]))
            buckets = []
            for cell in rng.permutation(np.unique(occupied)):
                points = rng.permutation(xy[occupied == cell])
                points.setflags(write=False)
                buckets.append(points)
            base.setflags(write=False)
            pools[index] = _AnchorPool(selection, base, tuple(buckets))
        budget.check()
    return tuple(pools[index] for index in range(len(selections)))
