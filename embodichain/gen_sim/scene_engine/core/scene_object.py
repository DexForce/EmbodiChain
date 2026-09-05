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
from typing import Literal


@dataclass
class ObjectPhysics:
    """Physics and collision settings shared by settling and scene export."""

    body_type: Literal["dynamic", "kinematic"]  # Runtime behaviour in simulation.
    attrs: dict[str, object]  # Grouped rigid-body physics configuration.
    max_convex_hull_num: int  # Collision-decomposition hull budget.

    def __post_init__(self) -> None:
        """Validate physics settings before a later stage consumes them."""
        if self.body_type not in {"dynamic", "kinematic"}:
            raise ValueError("body_type must be 'dynamic' or 'kinematic'.")
        if self.max_convex_hull_num <= 0:
            raise ValueError("max_convex_hull_num must be positive.")
        if not self.attrs:
            raise ValueError("attrs must contain at least one physics attribute.")
        if not all(isinstance(name, str) for name in self.attrs):
            raise ValueError("attrs must use string configuration keys.")

    def to_dict(self) -> dict[str, object]:
        """Serialize the physics settings for scene debugging artifacts."""
        return {
            "body_type": self.body_type,
            "attrs": self.attrs,
            "max_convex_hull_num": self.max_convex_hull_num,
        }


@dataclass
class SceneObject:
    """One semantic object progressing through the Scene Engine pipeline."""

    id: str  # Stable scene-unique identifier.
    kind: Literal["table", "asset"]  # Table support body or movable scene asset.
    category: str  # Semantic category identified by scene understanding.
    name: str  # Human-readable visual name.
    description: str  # Detailed semantic and spatial description.
    is_articulated: bool = False  # Whether this object has movable links or joints.
    mask_path: str | None = None  # Absolute path to the validated binary image mask.
    visible_rgba_path: str | None = None  # None for future unsegmented objects.
    simready_glb_path: str | None = None  # Absolute path to the canonical SimReady GLB.
    articulated_usdc_path: str | None = (
        None  # Generated articulation asset, unused by the GLB pipeline for now.
    )
    articulated_usdc_scale: list[float] | None = (
        None  # Y-up runtime scale retained because the GLB pipeline bakes coarse scale.
    )
    rot: list[float] | None = None  # Final y-up Euler XYZ rotation in degrees.
    pos: list[float] | None = None  # Final y-up world position in metres.
    scale: list[float] | None = None  # Final y-up object scale.
    center_xy: list[float] | None = None  # Z-up table-frame XY AABB center.
    support_surface_z: float | None = None  # Detected tabletop height in z-up.
    support_contour_xy: list[list[float]] | None = None  # Outer support contour.
    support_optimization_rect_xy: list[list[float]] | None = None  # Safe XY rectangle.
    physics: ObjectPhysics | None = None  # Assigned when SimReady processing succeeds.

    def to_dict(self) -> dict[str, object]:
        """Serialize this object and its currently available pipeline artifacts."""
        return {
            "id": self.id,
            "kind": self.kind,
            "category": self.category,
            "name": self.name,
            "description": self.description,
            "is_articulated": self.is_articulated,
            "mask_path": self.mask_path,
            "visible_rgba_path": self.visible_rgba_path,
            "simready_glb_path": self.simready_glb_path,
            "articulated_usdc_path": self.articulated_usdc_path,
            "articulated_usdc_scale": self.articulated_usdc_scale,
            "rot": self.rot,
            "pos": self.pos,
            "scale": self.scale,
            "center_xy": self.center_xy,
            "support_surface_z": self.support_surface_z,
            "support_contour_xy": self.support_contour_xy,
            "support_optimization_rect_xy": self.support_optimization_rect_xy,
            "physics": self.physics.to_dict() if self.physics is not None else None,
        }
