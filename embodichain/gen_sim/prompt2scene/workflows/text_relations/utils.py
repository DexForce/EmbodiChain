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

from typing import Any

from embodichain.gen_sim.prompt2scene.workflows.scene_intake.schema import (
    SceneIntakeSpec,
)
from embodichain.gen_sim.prompt2scene.workflows.spatial import (
    GRID_VALUES,
    RELATION_VALUES,
)
from embodichain.gen_sim.prompt2scene.workflows.scene_intake.utils import (
    normalize_asset_name,
)
from embodichain.gen_sim.prompt2scene.workflows.text_relations.schema import (
    TextObjectLayout,
    TextObjectRelation,
    TextRelationSpec,
    TextTableConstraint,
)

__all__ = [
    "build_text_relation_spec",
]


def build_text_relation_spec(
    *,
    scene_intake: SceneIntakeSpec,
    model_output: dict[str, Any],
) -> TextRelationSpec:
    """Normalize raw LLM JSON into text relation constraints."""
    asset_names = {asset.name for asset in scene_intake.assets}
    object_relations = _parse_object_relations(
        model_output.get("object_relations"),
        asset_names=asset_names,
    )
    table_constraints = _parse_table_constraints(
        model_output.get("table_constraints"),
        asset_names=asset_names,
    )
    object_layouts = _parse_object_layouts(
        model_output.get("object_layouts"),
        asset_names=asset_names,
    )
    return TextRelationSpec(
        source_text=scene_intake.input.text or "",
        object_relations=object_relations,
        table_constraints=table_constraints,
        object_layouts=object_layouts,
    )


def _parse_object_relations(
    raw_relations: Any,
    *,
    asset_names: set[str],
) -> list[TextObjectRelation]:
    if not isinstance(raw_relations, list):
        raise ValueError("text_relations.object_relations must be a list.")
    relations: list[TextObjectRelation] = []
    seen: set[tuple[str, str, str]] = set()
    for index, raw_relation in enumerate(raw_relations):
        if not isinstance(raw_relation, dict):
            raise ValueError(
                f"text_relations.object_relations[{index}] must be an object."
            )
        subject = _parse_asset_name(raw_relation.get("subject"), asset_names, index)
        relation = str(raw_relation.get("relation") or "").strip()
        object_name = _parse_asset_name(raw_relation.get("object"), asset_names, index)
        evidence = str(raw_relation.get("evidence") or "").strip()
        if relation not in RELATION_VALUES:
            raise ValueError(
                f"text_relations.object_relations[{index}].relation is invalid."
            )
        if not evidence:
            raise ValueError(
                f"text_relations.object_relations[{index}].evidence is required."
            )
        key = (subject, relation, object_name)
        if key in seen:
            continue
        seen.add(key)
        relations.append(
            TextObjectRelation(
                subject=subject,
                relation=relation,
                object=object_name,
                evidence=evidence,
            )
        )
    return relations


def _parse_table_constraints(
    raw_constraints: Any,
    *,
    asset_names: set[str],
) -> list[TextTableConstraint]:
    if not isinstance(raw_constraints, list):
        raise ValueError("text_relations.table_constraints must be a list.")
    constraints: list[TextTableConstraint] = []
    seen: set[tuple[str, str]] = set()
    for index, raw_constraint in enumerate(raw_constraints):
        if not isinstance(raw_constraint, dict):
            raise ValueError(
                f"text_relations.table_constraints[{index}] must be an object."
            )
        asset = _parse_asset_name(raw_constraint.get("asset"), asset_names, index)
        grid = str(raw_constraint.get("grid") or "").strip()
        evidence = str(raw_constraint.get("evidence") or "").strip()
        if grid not in GRID_VALUES:
            raise ValueError(
                f"text_relations.table_constraints[{index}].grid is invalid."
            )
        if not evidence:
            raise ValueError(
                f"text_relations.table_constraints[{index}].evidence is required."
            )
        key = (asset, grid)
        if key in seen:
            continue
        seen.add(key)
        constraints.append(
            TextTableConstraint(asset=asset, grid=grid, evidence=evidence)
        )
    return constraints


def _parse_object_layouts(
    raw_layouts: Any,
    *,
    asset_names: set[str],
) -> list[TextObjectLayout]:
    if not isinstance(raw_layouts, list):
        raise ValueError("text_relations.object_layouts must be a list.")
    layouts: list[TextObjectLayout] = []
    seen: set[str] = set()
    for index, raw_layout in enumerate(raw_layouts):
        if not isinstance(raw_layout, dict):
            raise ValueError(
                f"text_relations.object_layouts[{index}] must be an object."
            )
        asset = _parse_asset_name(raw_layout.get("asset"), asset_names, index)
        is_arbitrary_layout = raw_layout.get("is_arbitrary_layout")
        reason = str(raw_layout.get("reason") or "").strip()
        if not isinstance(is_arbitrary_layout, bool):
            raise ValueError(
                "text_relations.object_layouts"
                f"[{index}].is_arbitrary_layout must be boolean."
            )
        if not reason:
            raise ValueError(
                f"text_relations.object_layouts[{index}].reason is required."
            )
        if asset in seen:
            continue
        seen.add(asset)
        layouts.append(
            TextObjectLayout(
                asset=asset,
                is_arbitrary_layout=is_arbitrary_layout,
                reason=reason,
            )
        )
    return layouts


def _parse_asset_name(raw_name: Any, asset_names: set[str], index: int) -> str:
    name = normalize_asset_name(str(raw_name or ""))
    if name not in asset_names:
        raise ValueError(
            f"text_relations item {index} references unknown scene asset: {name!r}."
        )
    return name
