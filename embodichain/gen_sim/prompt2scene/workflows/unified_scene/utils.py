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

from collections import defaultdict
from typing import Any

from embodichain.gen_sim.prompt2scene.workflows.image_relations.schema import (
    ImageAnchor,
    ImageRelationSpec,
)
from embodichain.gen_sim.prompt2scene.agent_tools.tools.spatial_relations import (
    assign_grids_from_anchor_and_orders,
    derive_relations_from_orders,
    transitive_relation_closure,
)
from embodichain.gen_sim.prompt2scene.workflows.unified_scene.schema import (
    UnifiedObject,
    UnifiedSceneSpec,
    UnifiedSpatialAnchor,
    UnifiedSpatialRelation,
    UnifiedSpatial,
    UnifiedTable,
)
from embodichain.gen_sim.prompt2scene.workflows.scene_intake.schema import (
    SceneIntakeAsset,
    SceneIntakeSpec,
)
from embodichain.gen_sim.prompt2scene.workflows.text_relations.schema import (
    TextObjectLayout,
    TextRelationSpec,
)

__all__ = [
    "build_unified_object",
    "build_unified_object_specs",
    "build_unified_scene_from_image_relations",
    "build_unified_scene_from_text_relations",
    "build_unified_spatial_anchor",
    "build_unified_table",
    "grid_cells_from_objects",
    "object_ids_by_name",
    "relations_by_object_id",
    "resolve_image_layout",
    "resolve_text_layout",
    "text_grids_by_object_id",
]


def build_unified_object_specs(
    assets: list[SceneIntakeAsset],
) -> list[dict[str, Any]]:
    """Expand scene-intake assets into unified object instance specs."""
    specs: list[dict[str, Any]] = []
    for asset in assets:
        for index in range(asset.count):
            specs.append(
                {
                    "id": f"{asset.id}_{index}",
                    "name": asset.name,
                    "description": asset.description,
                    "class_candidate": list(asset.class_candidate),
                }
            )
    return specs


def object_ids_by_name(object_specs: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Group expanded object ids by object name."""
    grouped: dict[str, list[str]] = defaultdict(list)
    for spec in object_specs:
        grouped[str(spec["name"])].append(str(spec["id"]))
    return dict(grouped)


def build_unified_table(
    scene_intake: SceneIntakeSpec,
    *,
    grid_cells: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Build a unified table record from scene intake."""
    table: dict[str, Any] = {
        "id": scene_intake.table.id,
        "name": scene_intake.table.name,
        "description": scene_intake.table.description,
        "complete_table_description": (scene_intake.table.complete_table_description),
        "is_complete_visible_table": scene_intake.table.is_complete_visible_table,
        "class_candidate": list(scene_intake.table.class_candidate),
        "image_path": None,
        "mesh_path": None,
        "grid_cells": grid_cells,
    }
    if scene_intake.table.object_coverage_percent is not None:
        table["object_coverage_percent"] = scene_intake.table.object_coverage_percent
    return table


def build_unified_spatial_anchor(anchor: ImageAnchor | None) -> dict[str, Any] | None:
    """Convert the image anchor to a unified spatial anchor record."""
    if anchor is None:
        return None
    return {
        "object_id": anchor.asset_id,
        "grid": anchor.grid,
        "reason": anchor.reason,
    }


def build_unified_object(
    *,
    spec: dict[str, Any],
    grid: str | None,
    is_arbitrary_layout: bool,
    layout_reason: str,
) -> dict[str, Any]:
    """Build one unified object record."""
    return {
        "id": spec["id"],
        "name": spec["name"],
        "description": spec["description"],
        "class_candidate": list(spec["class_candidate"]),
        "grid": grid,
        "is_arbitrary_layout": is_arbitrary_layout,
        "layout_reason": layout_reason,
        "image_path": None,
        "mesh_path": None,
    }


def resolve_image_layout(
    asset_id: str,
    layout_by_id: dict[str, Any],
) -> tuple[bool, str]:
    """Resolve an image asset's layout state."""
    layout = layout_by_id.get(asset_id)
    if layout is None:
        return False, ""
    return bool(layout.is_arbitrary_layout), str(layout.reason)


def resolve_text_layout(
    name: str,
    layout_by_name: dict[str, TextObjectLayout],
) -> tuple[bool, str]:
    """Resolve a text asset's layout state."""
    layout = layout_by_name.get(name)
    if layout is None:
        return False, ""
    return bool(layout.is_arbitrary_layout), str(layout.reason)


def text_grids_by_object_id(
    *,
    text_relations: TextRelationSpec,
    ids_by_name: dict[str, list[str]],
) -> dict[str, str | None]:
    """Assign explicit text table constraints to object ids."""
    grids: dict[str, str | None] = {
        object_id: None for ids in ids_by_name.values() for object_id in ids
    }
    for constraint in text_relations.table_constraints:
        for object_id in ids_by_name.get(constraint.asset, []):
            grids[object_id] = constraint.grid
    return grids


def grid_cells_from_objects(
    objects: list[dict[str, Any]],
) -> dict[str, list[str]] | None:
    """Build table grid cell membership from unified objects."""
    grid_cells: dict[str, list[str]] = {
        "center": [],
        "front": [],
        "back": [],
        "left_center": [],
        "right_center": [],
        "left_front": [],
        "right_front": [],
        "left_back": [],
        "right_back": [],
    }
    any_grid = False
    for obj in objects:
        grid = obj.get("grid")
        if not grid:
            continue
        any_grid = True
        grid_cells.setdefault(str(grid), []).append(str(obj["id"]))
    return grid_cells if any_grid else None


def relations_by_object_id(
    *,
    text_relations: TextRelationSpec,
    ids_by_name: dict[str, list[str]],
) -> list[dict[str, str]]:
    """Expand text relations to object-id relations."""
    relations: list[dict[str, str]] = []
    for relation in text_relations.object_relations:
        subjects = ids_by_name.get(relation.subject, [])
        objects = ids_by_name.get(relation.object, [])
        for subject in subjects:
            for object_id in objects:
                if subject == object_id:
                    continue
                relations.append(
                    {
                        "subject": subject,
                        "relation": relation.relation,
                        "object": object_id,
                        "source": "input",
                    }
                )
    return relations


def build_unified_scene_from_image_relations(
    *,
    scene_intake: SceneIntakeSpec,
    image_relations: ImageRelationSpec,
) -> UnifiedSceneSpec:
    """Build a unified scene from image relation outputs."""
    object_specs = build_unified_object_specs(scene_intake.assets)
    anchor = build_unified_spatial_anchor(image_relations.anchor)
    if anchor is None:
        raise ValueError("Image unified scene requires an anchor.")
    layout_by_id = {layout.asset_id: layout for layout in image_relations.asset_layouts}
    objects = []
    for spec in object_specs:
        is_arbitrary_layout, layout_reason = resolve_image_layout(
            spec["id"],
            layout_by_id,
        )
        objects.append(
            UnifiedObject(
                **build_unified_object(
                    spec=spec,
                    grid=anchor["grid"] if spec["id"] == anchor["object_id"] else None,
                    is_arbitrary_layout=is_arbitrary_layout,
                    layout_reason=layout_reason,
                )
            )
        )
    relations = [
        UnifiedSpatialRelation(**relation)
        for relation in derive_relations_from_orders(
            x_order=image_relations.x_order,
            y_order=image_relations.y_order,
        )
    ]
    return UnifiedSceneSpec(
        input=scene_intake.input.to_manifest(),
        table=UnifiedTable(
            **build_unified_table(
                scene_intake,
                grid_cells=grid_cells_from_objects(
                    [object_.to_manifest() for object_ in objects]
                ),
            )
        ),
        objects=objects,
        spatial=UnifiedSpatial(
            anchor=UnifiedSpatialAnchor(**anchor),
            relations=relations,
        ),
    )


def build_unified_scene_from_text_relations(
    *,
    scene_intake: SceneIntakeSpec,
    text_relations: TextRelationSpec,
) -> UnifiedSceneSpec:
    """Build a unified scene from text relation outputs."""
    object_specs = build_unified_object_specs(scene_intake.assets)
    ids_by_name = object_ids_by_name(object_specs)
    grid_by_id = text_grids_by_object_id(
        text_relations=text_relations,
        ids_by_name=ids_by_name,
    )
    layout_by_name = {layout.asset: layout for layout in text_relations.object_layouts}
    objects = []
    for spec in object_specs:
        is_arbitrary_layout, layout_reason = resolve_text_layout(
            spec["name"],
            layout_by_name,
        )
        objects.append(
            UnifiedObject(
                **build_unified_object(
                    spec=spec,
                    grid=grid_by_id.get(spec["id"]),
                    is_arbitrary_layout=is_arbitrary_layout,
                    layout_reason=layout_reason,
                )
            )
        )
    relations = [
        UnifiedSpatialRelation(**relation)
        for relation in transitive_relation_closure(
            relations_by_object_id(
                text_relations=text_relations,
                ids_by_name=ids_by_name,
            )
        )
    ]
    return UnifiedSceneSpec(
        input=scene_intake.input.to_manifest(),
        table=UnifiedTable(
            **build_unified_table(
                scene_intake,
                grid_cells=grid_cells_from_objects(
                    [object_.to_manifest() for object_ in objects]
                ),
            )
        ),
        objects=objects,
        spatial=UnifiedSpatial(anchor=None, relations=relations),
    )
