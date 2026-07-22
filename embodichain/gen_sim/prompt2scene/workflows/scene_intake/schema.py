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
    "SceneIntakeAsset",
    "SceneIntakeInputRecord",
    "SceneIntakeSpec",
    "SceneIntakeTable",
]


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
            text=None,
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
