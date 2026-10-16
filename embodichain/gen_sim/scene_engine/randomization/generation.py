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

"""Generate grouped XY scene proposals through the shared scene services."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, fields
from typing import Literal

import numpy as np

from embodichain.lab.sim.cfg import RigidObjectCfg
from embodichain.lab.sim.scene.composition import SceneFragment
from embodichain.lab.sim.scene.expansion import ScenePoseChange, SceneVariant
from embodichain.lab.sim.scene.layout import LayoutCfg, LayoutProblem
from embodichain.lab.sim.scene.procedural import GeneratedScene, SceneRecipe, generate

from .baseline import (
    PreparedSceneBaseline,
    _entry_pose,
    _read_json,
    _read_pose,
    _rigid_entries,
    _source_digest,
    prepare_scene_baseline,
)
from .cfg import SceneRandomizationCfg, _validate_generation_parameters

__all__ = ["SceneRandomizationResult", "randomize_scene"]

_GENERATOR = "scene_engine.group_randomization/v1"


@dataclass(frozen=True)
class SceneRandomizationResult:
    """In-memory grouped scene candidates and their generation accounting.

    Attributes:
        baseline: Validated shared geometry and relative member poses.
        candidates: Successful proposals with every group member expanded.
        attempts: Number of calls to the shared procedural generator.
        status: Whether the requested count was reached, the attempt budget
            was exhausted, or the shared generator reported unsupported input.
        solver_status_counts: Number of attempts for each observed solver status.

    Candidates establish fixed-orientation group AABB bounds and XY clearances
    only. They do not establish physical stability or task acceptance.
    """

    baseline: PreparedSceneBaseline
    candidates: tuple[GeneratedScene, ...]
    attempts: int
    status: Literal["completed", "budget_exhausted", "unsupported"]
    solver_status_counts: dict[str, int]


def randomize_scene(
    cfg: SceneRandomizationCfg,
    *,
    seed: int | None = None,
    candidate_count: int | None = None,
    max_attempts: int | None = None,
    collision_margin_m: float | None = None,
    rebuild: bool = False,
) -> SceneRandomizationResult:
    """Randomize every support group using shared seeded XY layout generation.

    Every attempt starts from the same baseline. A group consists of a direct
    table child and all descendants; its full AABB must remain inside the
    table's persisted safe rectangle. Only XY translations change. This
    function prepares the existing baseline cache but does not save candidates
    or modify the source scene.

    Args:
        cfg: Validated scene paths and generation parameters from the loader.
        seed: Optional nonnegative seed override; ``None`` uses ``cfg.seed``.
        candidate_count: Optional successful layout count override; ``None``
            uses ``cfg.candidate_count``.
        max_attempts: Optional total attempt budget override; ``None`` uses
            ``cfg.max_attempts``.
        collision_margin_m: Optional group clearance override; ``None`` uses
            ``cfg.collision_margin_m``.
        rebuild: Recompute the baseline instead of reusing a valid cache.

    Returns:
        Successful full-scene proposals and explicit completion accounting.
        A partial result retains successful candidates if the budget runs out.

    Raises:
        OSError: Scene files or the baseline cannot be read or written.
        ValueError: Parameters or source data are invalid, a group cannot fit
            on the table, or the source changed after baseline preparation.
    """
    seed = cfg.seed if seed is None else seed
    candidate_count = (
        cfg.candidate_count if candidate_count is None else candidate_count
    )
    max_attempts = cfg.max_attempts if max_attempts is None else max_attempts
    collision_margin_m = (
        cfg.collision_margin_m if collision_margin_m is None else collision_margin_m
    )
    _validate_generation_parameters(
        seed=seed,
        candidate_count=candidate_count,
        max_attempts=max_attempts,
        collision_margin_m=collision_margin_m,
    )
    if type(rebuild) is not bool:
        raise ValueError("rebuild must be a boolean.")

    baseline = prepare_scene_baseline(cfg, rebuild=rebuild)
    recipe = _build_recipe(baseline)
    layout_cfg = LayoutCfg(collision_margin_m=float(collision_margin_m))
    candidates: list[GeneratedScene] = []
    counts: dict[str, int] = {}
    status: Literal["completed", "budget_exhausted", "unsupported"] = "budget_exhausted"
    attempts = 0
    for proposal_ordinal in range(max_attempts):
        proposal = generate(recipe, seed=seed, ordinal=proposal_ordinal, cfg=layout_cfg)
        attempts += 1
        outcome = proposal.layout.status
        counts[outcome] = counts.get(outcome, 0) + 1
        if outcome == "unsupported":
            status = "unsupported"
            break
        if outcome != "solved":
            continue
        candidates.append(
            _expand_groups(
                baseline,
                recipe,
                proposal,
                seed=seed,
                proposal_ordinal=proposal_ordinal,
                candidate_ordinal=len(candidates),
            )
        )
        if len(candidates) == candidate_count:
            status = "completed"
            break
    return SceneRandomizationResult(
        baseline, tuple(candidates), attempts, status, counts
    )


def _build_recipe(baseline: PreparedSceneBaseline) -> SceneRecipe:
    data = baseline.data
    source = data["source_scene"]
    export_root = (baseline.path.parent / source["export_dir"]).resolve()
    config_bytes, config = _read_json(export_root / "scene_config.json")
    graph_bytes, _ = _read_json(export_root / "scene_graph.json")
    entries = _rigid_entries(config)
    files = [export_root / entry["shape"]["fpath"] for entry in entries]
    if (
        _source_digest(export_root, config_bytes, graph_bytes, files)
        != source["content_sha256"]
    ):
        raise ValueError("Source scene changed after baseline preparation.")
    groups = data["groups"]
    if not groups:
        raise ValueError("Scene randomization requires at least one table group.")

    table_center = np.asarray(data["table"]["center_xy"], dtype=float)
    table_half = np.asarray(data["table"]["size_xy"], dtype=float) / 2
    bounds = tuple(map(tuple, (table_center - table_half, table_center + table_half)))
    centers: dict[str, tuple[float, float]] = {}
    half_extents: dict[str, tuple[float, float]] = {}
    for group in groups:
        uid = group["root_id"]
        half = np.asarray(group["aabb_2d"]["size_xy"], dtype=float) / 2
        if np.any(half > table_half):
            raise ValueError(f"Group {uid!r} cannot fit within the table bounds.")
        centers[uid] = tuple(group["aabb_2d"]["center_xy"])
        half_extents[uid] = tuple(half)

    allowed_fields = {field.name for field in fields(RigidObjectCfg)}
    entities = []
    for entry in entries:
        # The export also includes semantic/support metadata, which is not a
        # runtime object field. Preserve every authored runtime configuration.
        runtime = deepcopy(
            {key: value for key, value in entry.items() if key in allowed_fields}
        )
        runtime["shape"]["fpath"] = str(
            (export_root / entry["shape"]["fpath"]).resolve()
        )
        entity = RigidObjectCfg.from_dict(runtime)
        # from_dict infers extrinsic xyz. Portable exports and SceneFragment
        # use intrinsic XYZ, so supply the authoritative rigid matrix explicitly.
        _set_pose(entity, _entry_pose(entry))
        entity.init_rot = tuple(entry.get("init_rot", (0.0, 0.0, 0.0)))
        entities.append(entity)
    return SceneRecipe(
        source["content_sha256"],
        SceneFragment(entities),
        LayoutProblem(
            support_uid=data["table"]["entity_id"],
            seed_xy_by_uid=centers,
            half_extents_xy_by_uid=half_extents,
            bounds_xy_by_uid={uid: bounds for uid in centers},
        ),
    )


def _expand_groups(
    baseline: PreparedSceneBaseline,
    recipe: SceneRecipe,
    proposal: GeneratedScene,
    *,
    seed: int,
    proposal_ordinal: int,
    candidate_ordinal: int,
) -> GeneratedScene:
    # Do not start from a prior candidate or the root-only generated fragment.
    entities = {entity.uid: entity for entity in recipe.fragment.entities}
    changes = []
    for group in baseline.data["groups"]:
        uid = group["root_id"]
        root_pose = _read_pose(group["root_pose"])
        delta = np.asarray(proposal.layout.xy_by_uid[uid]) - np.asarray(
            group["aabb_2d"]["center_xy"]
        )
        root_pose[:2, 3] += delta
        poses = [(uid, root_pose)]
        poses.extend(
            (
                member["entity_id"],
                root_pose @ _read_pose(member["pose_in_root"]),
            )
            for member in group["members"]
        )
        for member_uid, pose in poses:
            _set_pose(entities[member_uid], pose)
            changes.append(ScenePoseChange(member_uid, pose.tolist()))
    variant = SceneVariant(
        recipe.recipe_id,
        seed,
        candidate_ordinal,
        tuple(changes),
        (
            ("generator", _GENERATOR),
            ("baseline_sha256", baseline.data["data_sha256"]),
            ("proposal_ordinal", str(proposal_ordinal)),
            ("geometry_scope", "fixed_orientation_group_aabb_xy"),
        ),
    )
    return GeneratedScene(
        SceneFragment(tuple(entities.values())),
        variant,
        proposal.layout,
        proposal.checks,
    )


def _set_pose(entity: RigidObjectCfg, pose: np.ndarray) -> None:
    entity.init_local_pose = pose.copy()
    entity.init_pos = tuple(pose[:3, 3])
