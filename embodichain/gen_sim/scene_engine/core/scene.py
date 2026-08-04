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

from embodichain.gen_sim.scene_engine.core.scene_object import SceneObject


@dataclass
class Scene:
    """A scene containing one table object and zero or more asset objects."""

    objects: list[SceneObject] = field(default_factory=list)

    @property
    def table(self) -> SceneObject | None:
        """Return the sole table object, or ``None`` before understanding."""
        tables = [
            scene_object
            for scene_object in self.objects
            if scene_object.kind == "table"
        ]
        if len(tables) > 1:
            raise ValueError("A scene may contain only one table object.")
        return tables[0] if tables else None

    @property
    def assets(self) -> list[SceneObject]:
        """Return movable asset objects in their scene order."""
        return [
            scene_object
            for scene_object in self.objects
            if scene_object.kind == "asset"
        ]

    def to_dict(self) -> dict[str, object]:
        """Serialize the canonical object collection for debugging artifacts."""
        return {"objects": [scene_object.to_dict() for scene_object in self.objects]}
