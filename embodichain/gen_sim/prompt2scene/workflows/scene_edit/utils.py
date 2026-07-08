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

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

from embodichain.gen_sim.prompt2scene.agent_tools.managers.layout_manager import (
    LayoutManager,
)
from embodichain.gen_sim.prompt2scene.agent_tools.tools.text_asset_generation import (
    generate_text_object_assets,
)
from embodichain.gen_sim.prompt2scene.utils.io import relative_path, write_json
from embodichain.gen_sim.prompt2scene.workflows.gym_export import (
    _bake_glb_bottom_center_to_origin,
    _render_scene_state_topdown,
)
from embodichain.gen_sim.prompt2scene.workflows.paths import PipelinePaths
from embodichain.gen_sim.prompt2scene.agent_tools.tools.spatial_relations import (
    transitive_relation_closure,
)
from embodichain.gen_sim.prompt2scene.utils.log import log_info

SCENE_EDIT_OBJECT_CLEARANCE_M = 0.05

__all__ = [
    "build_scene_edit_layout",
    "extract_current_grids",
    "extract_current_relations",
    "extract_scene_edit_support_region",
    "extract_scene_object_footprints",
    "extract_scene_objects",
    "generate_scene_edit_object_assets",
    "export_scene_edit_gym_state",
    "load_json_object",
    "match_prompt_scene_objects",
    "resolve_scene_edit_intent",
    "resolve_scene_state_snapshot_path",
    "scene_state_path",
    "tokenize_text",
    "validate_scene_edit_intent",
]


def scene_state_path(output_root: Path) -> Path:
    return output_root / "gym_export" / "scene_state" / "result.json"


def load_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return data


def extract_scene_objects(scene_state: dict[str, Any]) -> list[dict[str, str]]:
    """Return the minimal object view used by the edit-intent LLM."""
    objects: list[dict[str, str]] = []
    for obj in scene_state.get("objects", []) or []:
        if not isinstance(obj, dict):
            continue
        object_id = str(obj.get("id", "")).strip()
        if not object_id:
            continue
        objects.append(
            {
                "id": object_id,
                "name": str(obj.get("name", "")).strip(),
                "description": str(obj.get("description", "")).strip(),
            }
        )
    return objects


def extract_scene_object_footprints(
    scene_state: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Return normalized object 2D footprints keyed by object id."""
    footprints: dict[str, dict[str, Any]] = {}
    for obj in scene_state.get("objects", []) or []:
        if not isinstance(obj, dict):
            continue
        object_id = str(obj.get("id", "")).strip()
        footprint = obj.get("footprint_2d")
        if not object_id or not isinstance(footprint, dict):
            continue
        center_xy = footprint.get("center_xy")
        aabb_xy = footprint.get("aabb_xy")
        size_xy = footprint.get("size_xy")
        if not (
            isinstance(center_xy, list)
            and len(center_xy) == 2
            and isinstance(aabb_xy, list)
            and len(aabb_xy) == 2
            and all(isinstance(item, list) and len(item) == 2 for item in aabb_xy)
            and isinstance(size_xy, list)
            and len(size_xy) == 2
        ):
            continue
        footprints[object_id] = {
            "unit": str(footprint.get("unit", "m")).strip() or "m",
            "center_xy": [float(value) for value in center_xy],
            "aabb_xy": [
                [float(value) for value in aabb_xy[0]],
                [float(value) for value in aabb_xy[1]],
            ],
            "size_xy": [float(value) for value in size_xy],
        }
    return footprints


def extract_scene_edit_support_region(scene_state: dict[str, Any]) -> dict[str, Any]:
    """Return the table support-region 2D manifest from the previous scene."""
    table = scene_state.get("table")
    if not isinstance(table, dict):
        return {
            "unit": "m",
            "center_xy": [],
            "aabb_xy": [],
            "size_xy": [],
            "corners_xy": [],
        }
    support_region = table.get("support_region_2d")
    if not isinstance(support_region, dict):
        return {
            "unit": "m",
            "center_xy": [],
            "aabb_xy": [],
            "size_xy": [],
            "corners_xy": [],
        }
    return support_region


def resolve_scene_state_snapshot_path(
    *,
    output_root: Path,
    scene_state: dict[str, Any],
    snapshot_name: str,
) -> Path | None:
    """Resolve a snapshot path recorded in gym_export/scene_state/result.json."""
    source_snapshots = scene_state.get("source_snapshots")
    if not isinstance(source_snapshots, dict):
        return None
    snapshot_value = source_snapshots.get(snapshot_name)
    if not isinstance(snapshot_value, str) or not snapshot_value:
        return None
    snapshot_path = Path(snapshot_value)
    if snapshot_path.is_absolute():
        return snapshot_path
    return output_root / "gym_export" / snapshot_path


def extract_current_relations(
    *,
    output_root: Path,
    scene_state: dict[str, Any],
) -> list[dict[str, str]]:
    """Load canonical relations from the unified_scene snapshot if available."""
    snapshot_path = resolve_scene_state_snapshot_path(
        output_root=output_root,
        scene_state=scene_state,
        snapshot_name="unified_scene",
    )
    if snapshot_path is None or not snapshot_path.is_file():
        return []
    unified_scene = load_json_object(snapshot_path)
    spatial = unified_scene.get("spatial")
    if not isinstance(spatial, dict):
        return []
    relations = spatial.get("relations")
    if not isinstance(relations, list):
        return []

    normalized: list[dict[str, str]] = []
    for relation in relations:
        if not isinstance(relation, dict):
            continue
        subject = str(relation.get("subject", "")).strip()
        relation_name = str(relation.get("relation", "")).strip()
        object_id = str(relation.get("object", "")).strip()
        if not subject or not relation_name or not object_id:
            continue
        normalized.append(
            {
                "subject": subject,
                "relation": relation_name,
                "object": object_id,
                "source": str(relation.get("source", "")).strip(),
            }
        )
    return normalized


def extract_current_grids(
    *,
    output_root: Path,
    scene_state: dict[str, Any],
) -> dict[str, str]:
    """Load object 9-grid assignments from the unified_scene snapshot."""
    snapshot_path = resolve_scene_state_snapshot_path(
        output_root=output_root,
        scene_state=scene_state,
        snapshot_name="unified_scene",
    )
    if snapshot_path is None or not snapshot_path.is_file():
        return {}
    unified_scene = load_json_object(snapshot_path)
    objects = unified_scene.get("objects")
    if not isinstance(objects, list):
        return {}

    grids: dict[str, str] = {}
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        object_id = str(obj.get("id", "")).strip()
        grid = str(obj.get("grid", "") or "").strip()
        if object_id and grid:
            grids[object_id] = grid
    return grids


def resolve_scene_edit_intent(
    *,
    intent: dict[str, Any],
    scene_objects: list[dict[str, str]],
    current_relations: list[dict[str, str]],
    current_grids: dict[str, str],
) -> dict[str, Any]:
    """Resolve LLM edit operations into program-computed relations and grids."""
    intent = _normalize_scene_edit_intent_ids(
        intent=intent,
        scene_objects=scene_objects,
    )
    validate_scene_edit_intent(intent=intent, scene_objects=scene_objects)

    operations = [op for op in intent.get("operations", []) if isinstance(op, dict)]
    generated_objects = _normalize_generated_objects(
        operations=operations,
        generated_objects=[
            obj for obj in intent.get("generated_objects", []) if isinstance(obj, dict)
        ],
    )
    generated_ids = {
        str(obj.get("temp_id", "")).strip()
        for obj in generated_objects
        if str(obj.get("temp_id", "")).strip()
    }
    deleted_ids = _string_set(intent.get("deleted_object_ids"), "deleted_object_ids")

    replacement_map: dict[str, str] = {}
    replacement_inherits: set[str] = set()
    for operation in operations:
        if operation.get("type") != "replace":
            continue
        target_id = str(operation.get("target_object_id", "")).strip()
        new_id = str(operation.get("new_object_temp_id", "")).strip()
        if not target_id or not new_id:
            continue
        replacement_map[target_id] = new_id
        placement = operation.get("placement")
        placement_type = (
            str(placement.get("type", "")).strip()
            if isinstance(placement, dict)
            else ""
        )
        if placement_type in {"", "preserve_target"}:
            replacement_inherits.add(target_id)

    direct_relations: list[dict[str, str]] = []
    for relation in current_relations:
        subject = str(relation.get("subject", "")).strip()
        object_id = str(relation.get("object", "")).strip()
        relation_name = str(relation.get("relation", "")).strip()
        mapped_subject = _map_relation_endpoint(
            object_id=subject,
            deleted_ids=deleted_ids,
            replacement_map=replacement_map,
            replacement_inherits=replacement_inherits,
        )
        mapped_object = _map_relation_endpoint(
            object_id=object_id,
            deleted_ids=deleted_ids,
            replacement_map=replacement_map,
            replacement_inherits=replacement_inherits,
        )
        if mapped_subject is None or mapped_object is None:
            continue
        if mapped_subject == mapped_object:
            continue
        direct_relations.append(
            {
                "subject": mapped_subject,
                "relation": relation_name,
                "object": mapped_object,
                "source": (
                    "replacement_inherited"
                    if mapped_subject != subject or mapped_object != object_id
                    else "preserved"
                ),
            }
        )

    updated_grids: dict[str, str] = {}
    for object_id, grid in current_grids.items():
        if object_id in deleted_ids:
            replacement_id = replacement_map.get(object_id)
            if replacement_id and object_id in replacement_inherits:
                updated_grids[replacement_id] = grid
            continue
        updated_grids[object_id] = grid

    for operation in operations:
        op_type = str(operation.get("type", "")).strip()
        if op_type not in {"add", "replace"}:
            continue
        new_id = str(operation.get("new_object_temp_id", "")).strip()
        if new_id not in generated_ids:
            continue
        placement = operation.get("placement")
        if not isinstance(placement, dict):
            continue
        placement_type = str(placement.get("type", "")).strip()
        if placement_type == "grid":
            grid = str(placement.get("grid", "")).strip()
            if grid:
                updated_grids[new_id] = grid
        elif placement_type == "relative_to_object":
            reference_id = _map_reference_endpoint(
                object_id=str(placement.get("reference_object_id", "")).strip(),
                deleted_ids=deleted_ids,
                replacement_map=replacement_map,
            )
            relation = _placement_relation_to_canonical(
                new_object_id=new_id,
                relation=str(placement.get("relation", "")).strip(),
                reference_object_id=reference_id or "",
            )
            if relation is not None:
                direct_relations.append({**relation, "source": "new_prompt"})

    return {
        "deleted_object_ids": sorted(deleted_ids),
        "generated_objects": generated_objects,
        "operations": operations,
        "updated_relations": _close_relations_with_sources(direct_relations),
        "updated_grid_assignments": dict(sorted(updated_grids.items())),
        "unresolved": intent.get("unresolved", []),
        "reason": intent.get("reason", ""),
    }


def tokenize_text(value: str) -> set[str]:
    return {
        token for token in re.split(r"[^a-zA-Z0-9]+", value.lower()) if len(token) >= 2
    }


def match_prompt_scene_objects(
    *,
    prompt: str,
    scene_state: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return rough object candidates mentioned by the edit prompt."""
    prompt_tokens = tokenize_text(prompt)
    matches: list[dict[str, Any]] = []
    for obj in scene_state.get("objects", []) or []:
        if not isinstance(obj, dict):
            continue
        text = " ".join(str(obj.get(key, "")) for key in ("id", "name", "description"))
        object_tokens = tokenize_text(text.replace("_", " "))
        overlap = sorted(prompt_tokens & object_tokens)
        if not overlap:
            continue
        score = len(overlap) / max(len(object_tokens), 1)
        matches.append(
            {
                "id": obj.get("id", ""),
                "name": obj.get("name", ""),
                "description": obj.get("description", ""),
                "matched_tokens": overlap,
                "score": score,
                "footprint_2d": obj.get("footprint_2d"),
            }
        )
    return sorted(matches, key=lambda item: float(item["score"]), reverse=True)


def generate_scene_edit_object_assets(
    *,
    generated_objects: list[dict[str, Any]],
    output_root: Path,
    output_dir: Path,
    gravity_settle_mode: str = "geometry",
) -> dict[str, Any]:
    """Generate simready assets for scene-edit add/replace objects."""
    image_gen_dir = output_dir / "image_gen"
    glb_gen_dir = output_dir / "glb_gen"
    debug_dir = output_dir / "debug"
    image_gen_dir.mkdir(parents=True, exist_ok=True)
    glb_gen_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    object_specs = [
        _scene_edit_object_spec(generated_object)
        for generated_object in generated_objects
    ]
    log_info(
        "scene_edit object asset generation started "
        f"count={len(object_specs)} output_dir={output_dir}"
    )
    object_results = generate_text_object_assets(
        object_specs=object_specs,
        image_gen_dir=image_gen_dir,
        glb_gen_dir=glb_gen_dir,
        debug_dir=debug_dir,
        gravity_settle_mode=gravity_settle_mode,
    )
    normalized_results = [
        _normalize_generated_asset_result(result, output_root=output_root)
        for result in object_results
    ]
    succeeded = sum(
        str(result.get("status", "")).strip() == "ok" for result in normalized_results
    )
    status = "ok" if succeeded == len(normalized_results) else "partial"
    if not normalized_results:
        status = "ok"
    log_info(
        "scene_edit object asset generation completed "
        f"succeeded={succeeded} failed={len(normalized_results) - succeeded}"
    )
    return {
        "status": status,
        "object_count": len(normalized_results),
        "generated_assets": normalized_results,
    }


def build_scene_edit_layout(
    *,
    scene_state: dict[str, Any],
    resolved_intent: dict[str, Any],
    generated_assets: list[dict[str, Any]],
    output_root: Path,
    optimize_new_objects_only: bool = False,
) -> dict[str, Any]:
    """Build an edited 2D layout on top of the previous scene state."""
    support_region = extract_scene_edit_support_region(scene_state)
    old_footprints = extract_scene_object_footprints(scene_state)
    old_objects_by_id = {
        str(obj.get("id", "")).strip(): obj
        for obj in scene_state.get("objects", []) or []
        if isinstance(obj, dict) and str(obj.get("id", "")).strip()
    }
    deleted_ids = {
        str(object_id).strip()
        for object_id in resolved_intent.get("deleted_object_ids", []) or []
        if str(object_id).strip()
    }
    operations = [
        op for op in resolved_intent.get("operations", []) or [] if isinstance(op, dict)
    ]
    updated_relations = [
        relation
        for relation in resolved_intent.get("updated_relations", []) or []
        if isinstance(relation, dict)
    ]
    updated_grids = {
        str(object_id).strip(): str(grid).strip()
        for object_id, grid in (
            resolved_intent.get("updated_grid_assignments") or {}
        ).items()
        if str(object_id).strip() and str(grid).strip()
    }
    generated_asset_by_id = {
        str(asset.get("id", "")).strip(): asset
        for asset in generated_assets
        if isinstance(asset, dict)
        and str(asset.get("id", "")).strip()
        and str(asset.get("status", "")).strip() == "ok"
    }

    replacement_target_by_new_id: dict[str, str] = {}
    placement_by_new_id: dict[str, dict[str, Any]] = {}
    added_ids: list[str] = []
    replaced_ids: list[str] = []
    explicit_reposition_replace_ids: set[str] = set()
    for operation in operations:
        op_type = str(operation.get("type", "")).strip()
        new_id = str(operation.get("new_object_temp_id", "")).strip()
        if not new_id:
            continue
        placement = operation.get("placement")
        if isinstance(placement, dict):
            placement_by_new_id[new_id] = placement
        if op_type == "replace":
            target_id = str(operation.get("target_object_id", "")).strip()
            if target_id:
                replacement_target_by_new_id[new_id] = target_id
                replaced_ids.append(new_id)
                placement_type = (
                    str(placement.get("type", "")).strip()
                    if isinstance(placement, dict)
                    else ""
                )
                if placement_type not in {"", "preserve_target"}:
                    explicit_reposition_replace_ids.add(new_id)
        elif op_type == "add":
            added_ids.append(new_id)

    final_items: dict[str, dict[str, Any]] = {}
    for object_id, obj in old_objects_by_id.items():
        if object_id in deleted_ids:
            continue
        footprint = old_footprints.get(object_id)
        if footprint is None:
            continue
        final_items[object_id] = {
            "id": object_id,
            "name": str(obj.get("name", "")).strip(),
            "description": str(obj.get("description", "")).strip(),
            "action": "keep",
            "center_xy": list(footprint["center_xy"]),
            "size_xy": list(footprint["size_xy"]),
            "footprint_2d": footprint,
            "source": "previous_scene",
        }

    generated_ids = sorted(generated_asset_by_id)
    if not generated_ids:
        return {
            "status": "ok",
            "support_region": support_region,
            "deleted_object_ids": sorted(deleted_ids),
            "layout_updates": sorted(final_items.values(), key=lambda item: item["id"]),
            "optimization": {
                "method": "reuse_previous_scene",
                "generated_object_count": 0,
            },
        }

    xy_sizes = {
        object_id: np.asarray(
            LayoutManager.compute_simready_glb_xy_size(
                glb_path=_resolve_generated_asset_path(
                    generated_asset_by_id[object_id],
                    output_root=output_root,
                ),
                metric_scale=generated_asset_by_id[object_id].get("metric_scale"),
            ),
            dtype=np.float64,
        )
        for object_id in generated_ids
    }
    fixed_ids = set(replaced_ids)

    for object_id in replaced_ids:
        if object_id not in generated_asset_by_id:
            continue
        target_id = replacement_target_by_new_id.get(object_id, "")
        target_footprint = old_footprints.get(target_id)
        if target_footprint is None:
            continue
        asset = generated_asset_by_id[object_id]
        center_xy = LayoutManager.clamp_center_to_support_region(
            center_xy=list(target_footprint["center_xy"]),
            size_xy=xy_sizes[object_id].tolist(),
            support_region=support_region,
        )
        final_items[object_id] = {
            "id": object_id,
            "name": str(asset.get("name", "")).strip(),
            "description": str(asset.get("description", "")).strip(),
            "action": "replace",
            "replaces": target_id,
            "center_xy": center_xy,
            "size_xy": xy_sizes[object_id].tolist(),
            "footprint_2d": LayoutManager.build_xy_footprint(
                center_xy=center_xy,
                size_xy=xy_sizes[object_id].tolist(),
            ),
            "source": "generated_asset",
            "simready_geometry_path": asset.get("simready_geometry_path")
            or asset.get("mesh_path"),
        }

    initialized_added_centers = _initialize_added_object_centers(
        added_ids=[
            object_id for object_id in added_ids if object_id in generated_asset_by_id
        ],
        placement_by_new_id=placement_by_new_id,
        updated_grids=updated_grids,
        updated_relations=updated_relations,
        stable_items=final_items,
        support_region=support_region,
        xy_sizes=xy_sizes,
    )
    for object_id in added_ids:
        if (
            object_id not in generated_asset_by_id
            or object_id not in initialized_added_centers
        ):
            continue
        asset = generated_asset_by_id[object_id]
        center_xy = initialized_added_centers[object_id].tolist()
        size_xy = xy_sizes[object_id].tolist()
        final_items[object_id] = {
            "id": object_id,
            "name": str(asset.get("name", "")).strip(),
            "description": str(asset.get("description", "")).strip(),
            "action": "add",
            "replaces": "",
            "center_xy": center_xy,
            "size_xy": size_xy,
            "footprint_2d": LayoutManager.build_xy_footprint(
                center_xy=center_xy, size_xy=size_xy
            ),
            "source": "generated_asset",
            "simready_geometry_path": asset.get("simready_geometry_path")
            or asset.get("mesh_path"),
        }

    initial_centers_all = {
        object_id: np.asarray(item["center_xy"], dtype=np.float64)
        for object_id, item in final_items.items()
    }
    optimized_centers = {
        object_id: center.copy() for object_id, center in initial_centers_all.items()
    }
    optimization_metadata: dict[str, Any] | None = None
    all_object_ids = sorted(final_items)
    if all_object_ids:
        fixed_object_ids: list[str] = []
        if optimize_new_objects_only:
            movable_ids = set(added_ids) | explicit_reposition_replace_ids
            fixed_object_ids = [
                object_id
                for object_id in all_object_ids
                if object_id not in movable_ids
            ]
        gym_config = load_json_object(PipelinePaths(output_root).gym_config)
        rigid_objects = gym_config.get("rigid_object")
        if not isinstance(rigid_objects, list):
            raise ValueError("gym_config rigid_object must be a list.")
        rigid_by_id = {
            str(item.get("uid", "")).strip(): item
            for item in rigid_objects
            if isinstance(item, dict) and str(item.get("uid", "")).strip()
        }
        sa_runtime_root = output_root / "scene_edit" / "sa_node3_5_runtime"
        optimized_layout = LayoutManager.optimize_scene_edit_layout_with_sa_node3_5(
            output_root=output_root,
            support_region=support_region,
            layout_items=final_items,
            updated_relations=updated_relations,
            updated_grids=updated_grids,
            fixed_object_ids=fixed_object_ids,
            rigid_by_id=rigid_by_id,
            generated_asset_by_id=generated_asset_by_id,
            runtime_root=sa_runtime_root,
        )
        all_optimized = {
            object_id: np.asarray(center, dtype=np.float64)
            for object_id, center in optimized_layout.get("centers", {}).items()
        }
        for object_id, center in all_optimized.items():
            optimized_centers[object_id] = np.asarray(
                LayoutManager.clamp_center_to_support_region(
                    center_xy=center.tolist(),
                    size_xy=final_items[object_id]["size_xy"],
                    support_region=support_region,
                ),
                dtype=np.float64,
            )
        optimization_metadata = optimized_layout.get("metadata")

    for object_id, item in final_items.items():
        center_xy = optimized_centers[object_id].tolist()
        size_xy = item["size_xy"]
        item["center_xy"] = center_xy
        item["footprint_2d"] = LayoutManager.build_xy_footprint(
            center_xy=center_xy, size_xy=size_xy
        )

    return {
        "status": "ok",
        "support_region": support_region,
        "deleted_object_ids": sorted(deleted_ids),
        "layout_updates": sorted(final_items.values(), key=lambda item: item["id"]),
        "optimization": {
            "method": "delete_then_replace_then_add_initialize_then_optimize",
            "generated_object_count": len(generated_ids),
            "fixed_replacement_count": len(fixed_ids),
            "replaced_object_count": len(replaced_ids),
            "added_object_count": len(initialized_added_centers),
            "initialized_added_object_count": len(initialized_added_centers),
            "optimized_object_count": len(all_object_ids),
            "optimize_new_objects_only": optimize_new_objects_only,
            "added_layout_optimization": optimization_metadata,
        },
    }


def export_scene_edit_gym_state(
    *,
    output_root: Path,
    scene_state: dict[str, Any],
    generated_assets: list[dict[str, Any]],
    layout_updates: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    """Update gym_export files from scene-edit layout results."""
    paths = PipelinePaths(output_root)
    gym_config_path = paths.gym_config
    if not gym_config_path.is_file():
        raise FileNotFoundError(f"gym_config.json not found: {gym_config_path}")
    gym_config = load_json_object(gym_config_path)
    rigid_objects = gym_config.get("rigid_object")
    if not isinstance(rigid_objects, list):
        raise ValueError("gym_config rigid_object must be a list.")

    scene_objects = scene_state.get("objects")
    if not isinstance(scene_objects, list):
        raise ValueError("scene_state objects must be a list.")

    rigid_by_id = {
        str(item.get("uid", "")).strip(): item
        for item in rigid_objects
        if isinstance(item, dict) and str(item.get("uid", "")).strip()
    }
    scene_by_id = {
        str(item.get("id", "")).strip(): item
        for item in scene_objects
        if isinstance(item, dict) and str(item.get("id", "")).strip()
    }
    generated_asset_by_id = {
        str(item.get("id", "")).strip(): item
        for item in generated_assets
        if isinstance(item, dict) and str(item.get("id", "")).strip()
    }
    layout_by_id = {
        str(item.get("id", "")).strip(): item
        for item in layout_updates
        if isinstance(item, dict) and str(item.get("id", "")).strip()
    }

    scene_state_dir = output_root / "gym_export" / "scene_state"
    mesh_assets_dir = output_root / "gym_export" / "mesh_assets"
    scene_state_dir.mkdir(parents=True, exist_ok=True)
    mesh_assets_dir.mkdir(parents=True, exist_ok=True)

    gym_config = load_json_object(PipelinePaths(output_root).gym_config)
    table_surface_height = _infer_scene_edit_table_surface_height(
        output_root=output_root,
        gym_config=gym_config,
    )

    updated_rigid_objects: list[dict[str, Any]] = []
    updated_scene_objects: list[dict[str, Any]] = []
    updated_files: list[str] = []

    for object_id, layout_item in layout_by_id.items():
        action = str(layout_item.get("action", "")).strip()
        center_xy = layout_item.get("center_xy")
        size_xy = layout_item.get("size_xy")
        if not (
            isinstance(center_xy, list)
            and len(center_xy) == 2
            and isinstance(size_xy, list)
            and len(size_xy) == 2
        ):
            continue
        old_rigid = rigid_by_id.get(object_id)
        old_scene_obj = scene_by_id.get(object_id)
        if action == "keep" and old_rigid is None:
            continue

        if action == "keep":
            updated_rigid = _update_existing_rigid_object(
                object_id=object_id,
                rigid_object=old_rigid,
                old_scene_object=old_scene_obj,
                layout_item=layout_item,
            )
        else:
            generated_asset = generated_asset_by_id.get(object_id)
            if generated_asset is None:
                raise ValueError(
                    f"Missing generated asset for edited object: {object_id}"
                )
            updated_rigid = _build_generated_rigid_object(
                object_id=object_id,
                layout_item=layout_item,
                generated_asset=generated_asset,
                output_root=output_root,
                mesh_assets_dir=mesh_assets_dir,
                table_height=table_surface_height + 0.01,
            )
            shape = updated_rigid.get("shape")
            if isinstance(shape, dict):
                updated_files.append(str(shape.get("fpath", "")))

        updated_rigid_objects.append(updated_rigid)
        updated_scene_objects.append(
            _build_scene_state_object(
                object_id=object_id,
                layout_item=layout_item,
                rigid_object=updated_rigid,
                output_root=output_root,
            )
        )

    gym_config["rigid_object"] = updated_rigid_objects
    write_json(gym_config_path, gym_config)
    updated_files.append(relative_path(gym_config_path, output_root))

    topdown_path = scene_state_dir / "topdown_2d.png"
    _render_scene_state_topdown(
        support_region=extract_scene_edit_support_region(scene_state),
        objects=updated_scene_objects,
        output_path=topdown_path,
    )
    updated_files.append(relative_path(topdown_path, output_root))

    state_payload = dict(scene_state)
    state_payload["gym_config_path"] = str(
        gym_config_path.relative_to(output_root / "gym_export")
    )
    state_payload["topdown_2d_plot_path"] = str(
        topdown_path.relative_to(output_root / "gym_export")
    )
    state_payload["objects"] = updated_scene_objects
    source_snapshots = dict(scene_state.get("source_snapshots") or {})
    layout_snapshot_path = scene_state_dir / "scene_edit_layout.json"
    write_json(
        layout_snapshot_path,
        {"layout_updates": layout_updates},
    )
    source_snapshots["scene_edit_layout"] = str(
        layout_snapshot_path.relative_to(output_root / "gym_export")
    )
    state_payload["source_snapshots"] = source_snapshots
    scene_state_result_path = scene_state_dir / "result.json"
    write_json(scene_state_result_path, state_payload)
    updated_files.append(relative_path(scene_state_result_path, output_root))
    updated_files.append(relative_path(layout_snapshot_path, output_root))

    return {
        "status": "ok",
        "updated_files": sorted(set(updated_files)),
        "object_count": len(updated_scene_objects),
        "gym_config_path": str(gym_config_path),
        "scene_state_path": str(scene_state_result_path),
    }


def validate_scene_edit_intent(
    *,
    intent: dict[str, Any],
    scene_objects: list[dict[str, str]],
) -> None:
    """Validate that an edit intent only references legal object ids."""
    existing_ids = {obj["id"] for obj in scene_objects if obj.get("id")}
    deleted_ids = _string_set(intent.get("deleted_object_ids"), "deleted_object_ids")
    unknown_deleted = sorted(deleted_ids - existing_ids)
    if unknown_deleted:
        raise ValueError(
            "Scene edit intent deleted unknown object ids: " f"{unknown_deleted}"
        )

    generated_objects = intent.get("generated_objects")
    if not isinstance(generated_objects, list):
        raise ValueError("Scene edit intent generated_objects must be a list.")
    generated_ids: set[str] = set()
    for generated in generated_objects:
        if not isinstance(generated, dict):
            raise ValueError(
                "Scene edit intent generated_objects entries must be objects."
            )
        temp_id = str(generated.get("temp_id", "")).strip()
        if not temp_id:
            raise ValueError("Scene edit intent generated object has empty temp_id.")
        if temp_id in existing_ids:
            raise ValueError(
                f"Scene edit generated temp_id collides with existing id: {temp_id}"
            )
        if temp_id in generated_ids:
            raise ValueError(f"Scene edit generated temp_id is duplicated: {temp_id}")
        generated_ids.add(temp_id)

    operations = intent.get("operations")
    if not isinstance(operations, list):
        raise ValueError("Scene edit intent operations must be a list.")
    for operation in operations:
        if not isinstance(operation, dict):
            raise ValueError("Scene edit intent operation entries must be objects.")
        op_type = str(operation.get("type", "")).strip()
        target_id = str(operation.get("target_object_id", "")).strip()
        new_temp_id = str(operation.get("new_object_temp_id", "")).strip()
        if op_type in {"delete", "replace"} and target_id not in existing_ids:
            raise ValueError(
                f"Scene edit {op_type} operation targets unknown object id: "
                f"{target_id}"
            )
        if op_type == "delete" and target_id not in deleted_ids:
            raise ValueError(
                f"Scene edit delete target is missing from deleted_object_ids: "
                f"{target_id}"
            )
        if op_type == "replace":
            if target_id not in deleted_ids:
                raise ValueError(
                    "Scene edit replace target is missing from deleted_object_ids: "
                    f"{target_id}"
                )
            if new_temp_id not in generated_ids:
                raise ValueError(
                    "Scene edit replace operation references unknown generated "
                    f"temp_id: {new_temp_id}"
                )
        if op_type == "add" and new_temp_id not in generated_ids:
            raise ValueError(
                f"Scene edit add operation references unknown generated temp_id: {new_temp_id}"
            )
        placement = operation.get("placement")
        if isinstance(placement, dict):
            reference_id = str(placement.get("reference_object_id", "")).strip()
            if reference_id and reference_id not in existing_ids:
                raise ValueError(
                    "Scene edit placement references unknown object id: "
                    f"{reference_id}"
                )


def _scene_edit_object_spec(generated_object: dict[str, Any]) -> dict[str, Any]:
    temp_id = str(generated_object.get("temp_id", "")).strip()
    name = str(generated_object.get("name", "")).strip()
    class_candidates = [name] if name else []
    return {
        "id": temp_id,
        "name": name,
        "description": str(generated_object.get("description", "")).strip(),
        "class_candidate": class_candidates,
    }


def _normalize_generated_asset_result(
    result: dict[str, Any],
    *,
    output_root: Path,
) -> dict[str, Any]:
    normalized = dict(result)
    for key in (
        "image_path",
        "raw_geometry_path",
        "mesh_path",
        "simready_geometry_path",
    ):
        value = normalized.get(key)
        if value:
            normalized[key] = relative_path(value, output_root)
    return normalized


def _map_relation_endpoint(
    *,
    object_id: str,
    deleted_ids: set[str],
    replacement_map: dict[str, str],
    replacement_inherits: set[str],
) -> str | None:
    if object_id in deleted_ids:
        replacement_id = replacement_map.get(object_id)
        if replacement_id and object_id in replacement_inherits:
            return replacement_id
        return None
    return object_id


def _normalize_generated_objects(
    *,
    operations: list[dict[str, Any]],
    generated_objects: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    operation_type_by_temp_id: dict[str, str] = {}
    for operation in operations:
        new_temp_id = str(operation.get("new_object_temp_id", "")).strip()
        op_type = str(operation.get("type", "")).strip()
        if new_temp_id and op_type in {"add", "replace"}:
            operation_type_by_temp_id[new_temp_id] = op_type

    normalized: list[dict[str, Any]] = []
    for generated in generated_objects:
        temp_id = str(generated.get("temp_id", "")).strip()
        if not temp_id:
            continue
        source_operation = str(generated.get("source_operation", "")).strip()
        normalized.append(
            {
                **generated,
                "source_operation": (
                    source_operation or operation_type_by_temp_id.get(temp_id, "add")
                ),
            }
        )
    return normalized


def _normalize_scene_edit_intent_ids(
    *,
    intent: dict[str, Any],
    scene_objects: list[dict[str, str]],
) -> dict[str, Any]:
    """Make generated temp ids internally consistent and unique for this scene."""
    normalized = json.loads(json.dumps(intent))
    existing_ids = {
        str(obj.get("id", "")).strip()
        for obj in scene_objects
        if str(obj.get("id", "")).strip()
    }
    generated_objects = [
        obj for obj in normalized.get("generated_objects", []) if isinstance(obj, dict)
    ]
    operations = [op for op in normalized.get("operations", []) if isinstance(op, dict)]

    generated_ids = {
        str(obj.get("temp_id", "")).strip()
        for obj in generated_objects
        if str(obj.get("temp_id", "")).strip()
    }
    referenced_ids = {
        str(op.get("new_object_temp_id", "")).strip()
        for op in operations
        if str(op.get("type", "")).strip() in {"add", "replace"}
        and str(op.get("new_object_temp_id", "")).strip()
    }

    unused_generated_ids = [
        str(obj.get("temp_id", "")).strip()
        for obj in generated_objects
        if str(obj.get("temp_id", "")).strip()
        and str(obj.get("temp_id", "")).strip() not in referenced_ids
    ]
    for operation in operations:
        op_type = str(operation.get("type", "")).strip()
        if op_type not in {"add", "replace"}:
            continue
        new_temp_id = str(operation.get("new_object_temp_id", "")).strip()
        if not new_temp_id or new_temp_id in generated_ids:
            continue
        if len(unused_generated_ids) == 1:
            operation["new_object_temp_id"] = unused_generated_ids[0]
        elif len(generated_objects) == 1:
            operation["new_object_temp_id"] = str(
                generated_objects[0].get("temp_id", "")
            ).strip()

    reserved = set(existing_ids)
    seen_generated: set[str] = set()
    rename_by_old_id: dict[str, str] = {}
    for generated in generated_objects:
        old_id = str(generated.get("temp_id", "")).strip()
        if not old_id:
            continue
        new_id = old_id
        if new_id in reserved or new_id in seen_generated:
            new_id = _unique_scene_edit_generated_id(
                base_id=old_id,
                reserved_ids=reserved | seen_generated,
            )
            generated["temp_id"] = new_id
        seen_generated.add(new_id)
        reserved.add(new_id)
        if new_id != old_id:
            rename_by_old_id[old_id] = new_id

    if rename_by_old_id:
        for operation in operations:
            new_temp_id = str(operation.get("new_object_temp_id", "")).strip()
            if new_temp_id in rename_by_old_id:
                operation["new_object_temp_id"] = rename_by_old_id[new_temp_id]

    return normalized


def _unique_scene_edit_generated_id(
    *,
    base_id: str,
    reserved_ids: set[str],
) -> str:
    base = re.sub(r"_\d+$", "", base_id.strip()) or "new_object"
    index = 0
    while True:
        candidate = f"{base}_{index}"
        if candidate not in reserved_ids:
            return candidate
        index += 1


def _map_reference_endpoint(
    *,
    object_id: str,
    deleted_ids: set[str],
    replacement_map: dict[str, str],
) -> str | None:
    if object_id in replacement_map:
        return replacement_map[object_id]
    if object_id in deleted_ids:
        return None
    return object_id


def _placement_relation_to_canonical(
    *,
    new_object_id: str,
    relation: str,
    reference_object_id: str,
) -> dict[str, str] | None:
    if not new_object_id or not reference_object_id:
        return None
    if relation == "left_of":
        return {
            "subject": new_object_id,
            "relation": "left_of",
            "object": reference_object_id,
        }
    if relation == "right_of":
        return {
            "subject": reference_object_id,
            "relation": "left_of",
            "object": new_object_id,
        }
    if relation == "front_of":
        return {
            "subject": new_object_id,
            "relation": "front_of",
            "object": reference_object_id,
        }
    if relation == "back_of":
        return {
            "subject": reference_object_id,
            "relation": "front_of",
            "object": new_object_id,
        }
    return None


def _close_relations_with_sources(
    direct_relations: list[dict[str, str]],
) -> list[dict[str, str]]:
    if not direct_relations:
        return []
    source_by_edge = {
        (
            str(relation.get("subject", "")).strip(),
            str(relation.get("relation", "")).strip(),
            str(relation.get("object", "")).strip(),
        ): str(relation.get("source", "")).strip()
        for relation in direct_relations
    }
    closed = transitive_relation_closure(direct_relations)
    result: list[dict[str, str]] = []
    for relation in closed:
        key = (
            relation["subject"],
            relation["relation"],
            relation["object"],
        )
        source = source_by_edge.get(key)
        result.append(
            {
                "subject": relation["subject"],
                "relation": relation["relation"],
                "object": relation["object"],
                "source": source or "transitive_closure",
            }
        )
    return result


def _string_set(value: Any, context: str) -> set[str]:
    if not isinstance(value, list):
        raise ValueError(f"Scene edit intent {context} must be a list.")
    result: set[str] = set()
    for item in value:
        text = str(item).strip()
        if not text:
            raise ValueError(f"Scene edit intent {context} contains an empty id.")
        result.add(text)
    return result


def _resolve_generated_asset_path(asset: dict[str, Any], *, output_root: Path) -> Path:
    value = asset.get("simready_geometry_path") or asset.get("mesh_path")
    path = Path(str(value or "")).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (output_root / path).resolve()


def _infer_scene_edit_table_surface_height(
    *,
    output_root: Path,
    gym_config: dict[str, Any],
) -> float:
    try:
        import trimesh
        import trimesh.transformations as tt
    except ImportError:
        return 0.0

    background = gym_config.get("background")
    if not isinstance(background, list) or not background:
        return 0.0
    table = background[0]
    if not isinstance(table, dict):
        return 0.0

    shape = table.get("shape")
    if not isinstance(shape, dict):
        return 0.0
    fpath = str(shape.get("fpath", "") or "").strip()
    if not fpath:
        return 0.0
    table_mesh_path = (output_root / "gym_export" / fpath).resolve()
    if not table_mesh_path.is_file():
        return 0.0

    scene = trimesh.load(table_mesh_path, force="scene")
    if isinstance(scene, trimesh.Trimesh):
        mesh = scene
    else:
        dumped = scene.dump(concatenate=True)
        if isinstance(dumped, trimesh.Trimesh):
            mesh = dumped
        else:
            meshes = [item for item in dumped if isinstance(item, trimesh.Trimesh)]
            if not meshes:
                return 0.0
            mesh = trimesh.util.concatenate(meshes)

    verts = np.asarray(mesh.vertices, dtype=np.float64)
    if verts.size == 0:
        return 0.0

    body_scale = np.asarray(
        table.get("body_scale") or [1.0, 1.0, 1.0], dtype=np.float64
    )
    if body_scale.shape != (3,) or not np.all(np.isfinite(body_scale)):
        body_scale = np.ones(3, dtype=np.float64)
    glb_scale = np.asarray(
        [body_scale[0], body_scale[2], body_scale[1]], dtype=np.float64
    )
    verts = verts * glb_scale.reshape(1, 3)

    init_rot = np.asarray(table.get("init_rot") or [0.0, 0.0, 0.0], dtype=np.float64)
    if init_rot.shape == (3,) and np.any(np.abs(init_rot) > 1.0e-8):
        rot = tt.euler_matrix(
            float(np.deg2rad(init_rot[0])),
            float(np.deg2rad(init_rot[1])),
            float(np.deg2rad(init_rot[2])),
            axes="sxyz",
        )
        verts = (rot[:3, :3] @ verts.T).T

    init_pos = np.asarray(table.get("init_pos") or [0.0, 0.0, 0.0], dtype=np.float64)
    if init_pos.shape != (3,) or not np.all(np.isfinite(init_pos)):
        init_pos = np.zeros(3, dtype=np.float64)

    return float(init_pos[2] + np.max(verts[:, 1]))


def _update_existing_rigid_object(
    *,
    object_id: str,
    rigid_object: dict[str, Any] | None,
    old_scene_object: dict[str, Any] | None,
    layout_item: dict[str, Any],
) -> dict[str, Any]:
    if rigid_object is None:
        raise ValueError(f"Missing rigid_object for existing scene object: {object_id}")
    updated = json.loads(json.dumps(rigid_object))
    old_center = _scene_edit_center_xy(old_scene_object)
    new_center = np.asarray(layout_item.get("center_xy", []), dtype=np.float64)
    init_pos = list(updated.get("init_pos") or [0.0, 0.0, 0.0])
    if old_center is not None and new_center.shape == (2,):
        delta = new_center - old_center
        init_pos[0] = float(init_pos[0]) + float(delta[0])
        init_pos[1] = float(init_pos[1]) + float(delta[1])
    updated["init_pos"] = [float(value) for value in init_pos]
    updated["description"] = (
        str(layout_item.get("description", "")).strip()
        or str(updated.get("description", "")).strip()
    )
    return updated


def _build_generated_rigid_object(
    *,
    object_id: str,
    layout_item: dict[str, Any],
    generated_asset: dict[str, Any],
    output_root: Path,
    mesh_assets_dir: Path,
    table_height: float,
) -> dict[str, Any]:
    simready_path = _resolve_generated_asset_path(
        generated_asset, output_root=output_root
    )
    if not simready_path.is_file():
        raise FileNotFoundError(f"Generated simready GLB not found: {simready_path}")
    safe_name = object_id.replace("interact_", "").strip("_") or "object"
    object_dir = mesh_assets_dir / safe_name / object_id
    object_dir.mkdir(parents=True, exist_ok=True)
    object_dst = object_dir / f"{object_id}.glb"

    metric_scale = generated_asset.get("metric_scale")
    scale_factor = 1.0
    if isinstance(metric_scale, dict):
        try:
            scale_factor = float(metric_scale.get("scale_factor", 1.0))
        except (TypeError, ValueError):
            scale_factor = 1.0
    if not np.isfinite(scale_factor) or scale_factor <= 0.0:
        scale_factor = 1.0
    _bake_glb_bottom_center_to_origin(
        simready_path,
        object_dst,
        scale_factor=scale_factor,
    )
    body_scale = [1.0, 1.0, 1.0]
    init_rot = [0.0, 0.0, 0.0]
    target_center = np.asarray(layout_item.get("center_xy", []), dtype=np.float64)
    if target_center.shape != (2,):
        raise ValueError(f"Missing center_xy for generated object: {object_id}")
    init_pos = [
        float(target_center[0]),
        float(target_center[1]),
        float(table_height),
    ]
    return {
        "uid": object_id,
        "description": str(layout_item.get("description", "")).strip(),
        "shape": {
            "shape_type": "Mesh",
            "fpath": str(object_dst.relative_to(output_root / "gym_export")),
            "compute_uv": False,
        },
        "attrs": {
            "mass": 0.01,
            "contact_offset": 0.003,
            "rest_offset": 0.001,
            "restitution": 0.01,
            "max_depenetration_velocity": 10.0,
            "min_position_iters": 32,
            "min_velocity_iters": 8,
        },
        "body_type": "dynamic",
        "init_pos": init_pos,
        "init_rot": init_rot,
        "body_scale": body_scale,
        "max_convex_hull_num": 16,
    }


def _build_scene_state_object(
    *,
    object_id: str,
    layout_item: dict[str, Any],
    rigid_object: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    init_rot = [
        float(value) for value in rigid_object.get("init_rot") or [0.0, 0.0, 0.0]
    ]
    body_scale = [
        float(value) for value in rigid_object.get("body_scale") or [1.0, 1.0, 1.0]
    ]
    init_pos = [
        float(value) for value in rigid_object.get("init_pos") or [0.0, 0.0, 0.0]
    ]
    footprint_2d = layout_item.get("footprint_2d") or LayoutManager.build_xy_footprint(
        center_xy=list(layout_item.get("center_xy", [0.0, 0.0])),
        size_xy=list(layout_item.get("size_xy", [0.0, 0.0])),
    )
    return {
        "id": object_id,
        "name": str(layout_item.get("name", "")).strip() or object_id,
        "role": "interact",
        "description": str(layout_item.get("description", "")).strip(),
        "init_pos": init_pos,
        "init_rot": init_rot,
        "body_scale": body_scale,
        "footprint_2d": footprint_2d,
    }


def _scene_edit_center_xy(scene_object: dict[str, Any] | None) -> np.ndarray | None:
    if not isinstance(scene_object, dict):
        return None
    footprint = scene_object.get("footprint_2d")
    if not isinstance(footprint, dict):
        return None
    center_xy = footprint.get("center_xy")
    if not isinstance(center_xy, list) or len(center_xy) != 2:
        return None
    return np.asarray(center_xy, dtype=np.float64)


def _compute_anchor_targets(
    *,
    generated_ids: list[str],
    replacement_target_by_new_id: dict[str, str],
    placement_by_new_id: dict[str, dict[str, Any]],
    updated_grids: dict[str, str],
    old_footprints: dict[str, dict[str, Any]],
    support_region: dict[str, Any],
    xy_sizes: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    targets: dict[str, np.ndarray] = {}
    unresolved = set(generated_ids)
    for _ in range(max(len(generated_ids), 1) * 2):
        progressed = False
        for object_id in list(unresolved):
            replacement_target = replacement_target_by_new_id.get(object_id)
            if replacement_target:
                target_footprint = old_footprints.get(replacement_target)
                if target_footprint is None:
                    continue
                targets[object_id] = np.asarray(
                    target_footprint["center_xy"],
                    dtype=np.float64,
                )
                unresolved.remove(object_id)
                progressed = True
                continue

            placement = placement_by_new_id.get(object_id, {})
            placement_type = str(placement.get("type", "")).strip()
            if placement_type == "relative_to_object":
                reference_id = str(placement.get("reference_object_id", "")).strip()
                relation = str(placement.get("relation", "")).strip()
                reference_center = targets.get(reference_id)
                reference_size = xy_sizes.get(reference_id)
                if reference_center is None:
                    reference = old_footprints.get(reference_id)
                    if reference is not None:
                        reference_center = np.asarray(
                            reference["center_xy"],
                            dtype=np.float64,
                        )
                        reference_size = np.asarray(
                            reference["size_xy"],
                            dtype=np.float64,
                        )
                if reference_center is not None and reference_size is not None:
                    targets[object_id] = _offset_center_by_relation(
                        reference_center=reference_center,
                        reference_size=reference_size,
                        object_size=xy_sizes[object_id],
                        relation=relation,
                    )
                    unresolved.remove(object_id)
                    progressed = True
                    continue

            grid_name = updated_grids.get(object_id)
            if grid_name:
                targets[object_id] = LayoutManager.support_region_grid_center(
                    support_region=support_region,
                    grid_name=grid_name,
                )
                unresolved.remove(object_id)
                progressed = True
                continue
        if not progressed:
            break
    return targets


def _initialize_added_object_centers(
    *,
    added_ids: list[str],
    placement_by_new_id: dict[str, dict[str, Any]],
    updated_grids: dict[str, str],
    updated_relations: list[dict[str, Any]],
    stable_items: dict[str, dict[str, Any]],
    support_region: dict[str, Any],
    xy_sizes: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    if not added_ids:
        return {}
    support_center = LayoutManager.support_region_default_center(
        support_region=support_region
    )
    results: dict[str, np.ndarray] = {}
    for object_id in added_ids:
        seed_center = support_center.copy()
        grid_name = updated_grids.get(object_id)
        if grid_name:
            seed_center = LayoutManager.support_region_grid_center(
                support_region=support_region,
                grid_name=grid_name,
            )
        else:
            placement = placement_by_new_id.get(object_id, {})
            placement_type = str(placement.get("type", "")).strip()
            if placement_type == "relative_to_object":
                reference_id = str(placement.get("reference_object_id", "")).strip()
                stable_item = stable_items.get(reference_id)
                if stable_item is not None:
                    reference_center = np.asarray(
                        stable_item.get("center_xy", support_center.tolist()),
                        dtype=np.float64,
                    )
                    if reference_center.shape == (2,):
                        seed_center = reference_center
        results[object_id] = np.asarray(
            LayoutManager.clamp_center_to_support_region(
                center_xy=seed_center.tolist(),
                size_xy=xy_sizes[object_id].tolist(),
                support_region=support_region,
            ),
            dtype=np.float64,
        )
    return results


def _offset_center_by_relation(
    *,
    reference_center: np.ndarray,
    reference_size: np.ndarray,
    object_size: np.ndarray,
    relation: str,
    padding: float = SCENE_EDIT_OBJECT_CLEARANCE_M,
) -> np.ndarray:
    gap_x = 0.5 * (reference_size[0] + object_size[0]) + padding
    gap_y = 0.5 * (reference_size[1] + object_size[1]) + padding
    offset = np.zeros(2, dtype=np.float64)
    if relation == "left_of":
        offset[0] = -gap_x
    elif relation == "right_of":
        offset[0] = gap_x
    elif relation == "front_of":
        offset[1] = -gap_y
    elif relation in {"back_of", "behind"}:
        offset[1] = gap_y
    else:
        offset = np.asarray([gap_x, 0.0], dtype=np.float64)
    return reference_center + offset
