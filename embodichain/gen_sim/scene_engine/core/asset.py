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

from dataclasses import dataclass


@dataclass
class Asset:
    """A scene asset identified during scene understanding."""

    id: str
    category: str
    name: str
    description: str
    # Path to a binary mask image aligned with the input image. White pixels
    # identify this asset; black pixels identify the background.
    mask_path: str | None = None
    # Absolute path to the canonicalized GLB used by the final simulation.
    simready_glb_path: str | None = None
    # Final y-up layout after scene refinement and gravity settling.
    rot: list[float] | None = None
    pos: list[float] | None = None
    scale: list[float] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "category": self.category,
            "name": self.name,
            "description": self.description,
            "mask_path": self.mask_path,
            "simready_glb_path": self.simready_glb_path,
            "rot": self.rot,
            "pos": self.pos,
            "scale": self.scale,
        }
