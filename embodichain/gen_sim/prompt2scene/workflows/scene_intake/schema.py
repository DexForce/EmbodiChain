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

from embodichain.gen_sim.prompt2scene.workflows.request import (
    InputKind,
    Prompt2SceneInput,
)

__all__ = [
    "SCENE_INTAKE_JSON_SCHEMA",
    "SceneIntakeAsset",
    "SceneIntakeInputRecord",
    "SceneIntakeSpec",
    "SceneIntakeTable",
]

SCENE_INTAKE_JSON_SCHEMA: dict[str, Any] = {
    "title": "SceneIntakeModelOutput",
    "description": (
        "Objects and table information extracted from a text or image input."
    ),
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "table": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "name": {
                    "type": "string",
                    "description": (
                        "Canonical English class name for the visible table "
                        "or tabletop target, such as table, desk, dining_table, "
                        "coffee_table, workbench, or tabletop."
                    ),
                },
                "description": {
                    "type": "string",
                    "minLength": 20,
                    "maxLength": 180,
                    "description": (
                        "One concise standalone appearance description of the "
                        "visible table or tabletop region."
                    ),
                },
                "complete_table_description": {
                    "type": "string",
                    "minLength": 20,
                    "maxLength": 220,
                    "description": (
                        "One concise standalone description of a complete table "
                        "asset for text-to-3D generation, matching the visible "
                        "tabletop color, material, and texture."
                    ),
                },
                "is_complete_visible_table": {
                    "type": "boolean",
                    "description": (
                        "For image input, whether a mostly complete table is "
                        "visible and suitable as the final table geometry source. "
                        "For text input, this should be false."
                    ),
                },
                "class_candidate": {
                    "type": "array",
                    "minItems": 5,
                    "maxItems": 5,
                    "description": (
                        "Exactly five likely class names for segmenting the "
                        "visible table or tabletop target."
                    ),
                    "items": {
                        "type": "string",
                        "minLength": 1,
                    },
                },
                "object_coverage_percent": {
                    "type": "integer",
                    "enum": [10, 30, 50, 70],
                    "description": (
                        "For image input with a complete visible table ONLY: "
                        "choose the closest coverage bucket for objects on the "
                        "tabletop: 10 (mostly empty, a few small objects), "
                        "30 (lightly cluttered), 50 (moderately cluttered), "
                        "70 (densely packed). Omit this field entirely for "
                        "text input or when is_complete_visible_table is false."
                    ),
                },
            },
            "required": [
                "name",
                "description",
                "complete_table_description",
                "is_complete_visible_table",
                "class_candidate",
            ],
        },
        "assets": {
            "type": "array",
            "description": (
                "Object category groups on or intended for the tabletop scene."
            ),
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "name": {
                        "type": "string",
                        "description": (
                            "Canonical English object name, singular, "
                            "snake_case preferred."
                        ),
                    },
                    "description": {
                        "type": "string",
                        "minLength": 20,
                        "maxLength": 180,
                        "description": (
                            "One concise appearance description of the object for "
                            "image and 3D geometry generation."
                        ),
                    },
                    "class_candidate": {
                        "type": "array",
                        "minItems": 5,
                        "maxItems": 5,
                        "description": (
                            "Exactly five likely object class names for later "
                            "image detection or segmentation."
                        ),
                        "items": {
                            "type": "string",
                            "minLength": 1,
                        },
                    },
                    "count": {
                        "type": "integer",
                        "description": (
                            "Number of repeated instances in this object category "
                            "group. Only group objects that can share the same name, "
                            "description, and class_candidate list."
                        ),
                        "minimum": 1,
                    },
                },
                "required": ["name", "description", "class_candidate", "count"],
            },
        },
    },
    "required": ["table", "assets"],
}


@dataclass(frozen=True)
class SceneIntakeInputRecord:
    """Normalized input source recorded by scene intake."""

    input_kind: InputKind
    text: str | None = None
    image_path: str | None = None

    @classmethod
    def from_request(cls, request: Prompt2SceneInput) -> "SceneIntakeInputRecord":
        """Create an input record from a prompt2scene request."""
        return cls(
            input_kind=request.input_kind,
            text=request.text,
            image_path=str(request.image_path) if request.image_path else None,
        )

    def to_manifest(self) -> dict[str, str | None]:
        """Convert the input record to JSON-safe data."""
        return {
            "input_kind": self.input_kind.value,
            "text": self.text,
            "image_path": self.image_path,
        }


@dataclass(frozen=True)
class SceneIntakeTable:
    """Table/support information extracted during scene intake."""

    id: str = "table"
    name: str = "table"
    description: str = ""
    complete_table_description: str = ""
    is_complete_visible_table: bool = False
    class_candidate: list[str] = field(default_factory=list)
    object_coverage_percent: int | None = None

    def to_manifest(self) -> dict[str, object]:
        """Convert the table record to JSON-safe data."""
        manifest: dict[str, object] = {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "complete_table_description": self.complete_table_description,
            "is_complete_visible_table": self.is_complete_visible_table,
            "class_candidate": list(self.class_candidate),
        }
        if self.object_coverage_percent is not None:
            manifest["object_coverage_percent"] = self.object_coverage_percent
        return manifest


@dataclass(frozen=True)
class SceneIntakeAsset:
    """Object category group extracted during scene intake."""

    id: str
    name: str
    count: int = 1
    description: str = ""
    class_candidate: list[str] = field(default_factory=list)

    def to_manifest(self) -> dict[str, object]:
        """Convert the asset record to JSON-safe data."""
        return {
            "id": self.id,
            "name": self.name,
            "count": self.count,
            "description": self.description,
            "class_candidate": list(self.class_candidate),
        }


@dataclass(frozen=True)
class SceneIntakeSpec:
    """Unified first-step scene intake output for text and image inputs."""

    input: SceneIntakeInputRecord
    table: SceneIntakeTable
    assets: list[SceneIntakeAsset]

    def to_manifest(self) -> dict[str, object]:
        """Convert the intake spec to JSON-safe data."""
        return {
            "input": self.input.to_manifest(),
            "table": self.table.to_manifest(),
            "assets": [asset.to_manifest() for asset in self.assets],
        }
