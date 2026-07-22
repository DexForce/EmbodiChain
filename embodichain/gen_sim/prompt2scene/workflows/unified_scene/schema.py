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

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "UnifiedObject",
    "UnifiedSceneSpec",
    "UnifiedSpatial",
    "UnifiedSpatialAnchor",
    "UnifiedSpatialRelation",
    "UnifiedTable",
]


@dataclass(frozen=True)
class UnifiedTable:
    """Unified table/support object."""

    id: str
    name: str
    description: str
    complete_table_description: str
    is_complete_visible_table: bool
    class_candidate: list[str]
    image_path: str | None = None
    mesh_path: str | None = None
    grid_cells: dict[str, list[str]] | None = None
    object_coverage_percent: int | None = None

    def to_manifest(self) -> dict[str, Any]:
        """Convert the table to JSON-safe data."""
        manifest: dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "complete_table_description": self.complete_table_description,
            "is_complete_visible_table": self.is_complete_visible_table,
            "class_candidate": list(self.class_candidate),
            "image_path": self.image_path,
            "mesh_path": self.mesh_path,
            "grid_cells": self.grid_cells,
        }
        if self.object_coverage_percent is not None:
            manifest["object_coverage_percent"] = self.object_coverage_percent
        return manifest


@dataclass(frozen=True)
class UnifiedObject:
    """Unified object instance used by downstream scene generation."""

    id: str
    name: str
    description: str
    class_candidate: list[str]
    grid: str | None = None
    is_arbitrary_layout: bool = False
    layout_reason: str = ""
    image_path: str | None = None
    mesh_path: str | None = None

    def to_manifest(self) -> dict[str, Any]:
        """Convert the object to JSON-safe data."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "class_candidate": list(self.class_candidate),
            "grid": self.grid,
            "is_arbitrary_layout": self.is_arbitrary_layout,
            "layout_reason": self.layout_reason,
            "image_path": self.image_path,
            "mesh_path": self.mesh_path,
        }


@dataclass(frozen=True)
class UnifiedSpatialAnchor:
    """Spatial anchor used to infer a full table grid."""

    object_id: str
    grid: str
    reason: str = ""

    def to_manifest(self) -> dict[str, str]:
        """Convert the anchor to JSON-safe data."""
        return {
            "object_id": self.object_id,
            "grid": self.grid,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class UnifiedSpatialRelation:
    """Unified pairwise spatial relation between two objects."""

    subject: str
    relation: str
    object: str
    source: str

    def to_manifest(self) -> dict[str, str]:
        """Convert the relation to JSON-safe data."""
        return {
            "subject": self.subject,
            "relation": self.relation,
            "object": self.object,
            "source": self.source,
        }


@dataclass(frozen=True)
class UnifiedSpatial:
    """Unified spatial relations for a scene."""

    anchor: UnifiedSpatialAnchor | None = None
    relations: list[UnifiedSpatialRelation] = field(default_factory=list)

    def to_manifest(self) -> dict[str, Any]:
        """Convert the spatial record to JSON-safe data."""
        return {
            "anchor": self.anchor.to_manifest() if self.anchor else None,
            "relations": [relation.to_manifest() for relation in self.relations],
        }


@dataclass(frozen=True)
class UnifiedSceneSpec:
    """Unified scene representation consumed by downstream generation steps."""

    input: dict[str, Any]
    table: UnifiedTable
    objects: list[UnifiedObject]
    spatial: UnifiedSpatial

    def to_manifest(self) -> dict[str, Any]:
        """Convert the unified scene to JSON-safe data."""
        return {
            "input": dict(self.input),
            "table": self.table.to_manifest(),
            "objects": [obj.to_manifest() for obj in self.objects],
            "spatial": self.spatial.to_manifest(),
        }
