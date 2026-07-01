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

from embodichain.gen_sim.prompt2scene.workflows.spatial import (
    GRID_VALUE_LIST,
    RELATION_VALUE_LIST,
)

__all__ = [
    "TEXT_RELATIONS_JSON_SCHEMA",
    "TextObjectLayout",
    "TextObjectRelation",
    "TextRelationSpec",
    "TextTableConstraint",
]

TEXT_RELATIONS_JSON_SCHEMA: dict[str, Any] = {
    "title": "TextRelationsOutput",
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "object_relations": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "subject": {"type": "string", "minLength": 1},
                    "relation": {
                        "type": "string",
                        "enum": RELATION_VALUE_LIST,
                    },
                    "object": {"type": "string", "minLength": 1},
                    "evidence": {"type": "string", "minLength": 1},
                },
                "required": ["subject", "relation", "object", "evidence"],
            },
        },
        "table_constraints": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "asset": {"type": "string", "minLength": 1},
                    "grid": {
                        "type": "string",
                        "enum": GRID_VALUE_LIST,
                    },
                    "evidence": {"type": "string", "minLength": 1},
                },
                "required": ["asset", "grid", "evidence"],
            },
        },
        "object_layouts": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "asset": {"type": "string", "minLength": 1},
                    "is_arbitrary_layout": {"type": "boolean"},
                    "reason": {"type": "string", "minLength": 1},
                },
                "required": ["asset", "is_arbitrary_layout", "reason"],
            },
        },
    },
    "required": ["object_relations", "table_constraints", "object_layouts"],
}


@dataclass(frozen=True)
class TextObjectRelation:
    """Text-stated relation between two scene-intake asset groups."""

    subject: str
    relation: str
    object: str
    evidence: str

    def to_manifest(self) -> dict[str, str]:
        """Convert the relation to JSON-safe data."""
        return {
            "subject": self.subject,
            "relation": self.relation,
            "object": self.object,
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class TextTableConstraint:
    """Text-stated table grid constraint for one asset group."""

    asset: str
    grid: str
    evidence: str

    def to_manifest(self) -> dict[str, str]:
        """Convert the table constraint to JSON-safe data."""
        return {
            "asset": self.asset,
            "grid": self.grid,
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class TextObjectLayout:
    """Text-stated object support-pose constraint."""

    asset: str
    is_arbitrary_layout: bool
    reason: str

    def to_manifest(self) -> dict[str, object]:
        """Convert the layout constraint to JSON-safe data."""
        return {
            "asset": self.asset,
            "is_arbitrary_layout": self.is_arbitrary_layout,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class TextRelationSpec:
    """Spatial constraints explicitly extracted from a text prompt."""

    source_text: str
    object_relations: list[TextObjectRelation] = field(default_factory=list)
    table_constraints: list[TextTableConstraint] = field(default_factory=list)
    object_layouts: list[TextObjectLayout] = field(default_factory=list)

    def to_manifest(self) -> dict[str, object]:
        """Convert the text relations to JSON-safe data."""
        return {
            "source_text": self.source_text,
            "object_relations": [
                relation.to_manifest() for relation in self.object_relations
            ],
            "table_constraints": [
                constraint.to_manifest() for constraint in self.table_constraints
            ],
            "object_layouts": [layout.to_manifest() for layout in self.object_layouts],
        }
