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

from embodichain.gen_sim.prompt2scene.workflows.spatial import GRID_VALUE_LIST

__all__ = [
    "FILTER_EXTRA_INSTANCES_JSON_SCHEMA",
    "ImageAnchor",
    "ImageAssetLayout",
    "ImageAssetSegment",
    "ImageRelationGroup",
    "ImageRelationSpec",
    "SPATIAL_LAYOUT_JSON_SCHEMA",
]

FILTER_EXTRA_INSTANCES_JSON_SCHEMA: dict[str, Any] = {
    "title": "FilterExtraImageInstancesOutput",
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "extra_instance_numbers": {
            "type": "array",
            "description": "1-based mask numbers that should be removed.",
            "items": {"type": "integer", "minimum": 1},
        },
        "reason": {
            "type": "string",
            "description": "Brief reason for the removal decision.",
        },
    },
    "required": ["extra_instance_numbers", "reason"],
}

SPATIAL_LAYOUT_JSON_SCHEMA: dict[str, Any] = {
    "title": "ImageSpatialLayoutOutput",
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "anchor": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "asset_id": {"type": "string", "minLength": 1},
                "grid": {
                    "type": "string",
                    "enum": GRID_VALUE_LIST,
                },
                "reason": {"type": "string"},
            },
            "required": ["asset_id", "grid", "reason"],
        },
        "x_order": {
            "type": "array",
            "description": "Asset-id groups ordered from left to right.",
            "items": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "minItems": 1,
            },
            "minItems": 1,
        },
        "y_order": {
            "type": "array",
            "description": "Asset-id groups ordered from front to back.",
            "items": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "minItems": 1,
            },
            "minItems": 1,
        },
        "asset_states": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": True,
                "properties": {
                    "asset_id": {"type": "string", "minLength": 1},
                    "is_arbitrary_layout": {"type": "boolean"},
                    "reason": {"type": "string", "minLength": 1},
                },
                "required": [
                    "asset_id",
                    "is_arbitrary_layout",
                    "reason",
                ],
            },
        },
    },
    "required": ["anchor", "x_order", "y_order", "asset_states"],
}


@dataclass(frozen=True)
class ImageAssetSegment:
    """Image segmentation result aligned to one scene-intake asset."""

    asset_id: str
    name: str
    segment_id: str
    bbox_xyxy: list[float]
    score: float
    source_prompt: str
    mask_rle: dict[str, Any] | None = None

    def to_manifest(self) -> dict[str, Any]:
        """Convert the segment to JSON-safe data."""
        return {
            "asset_id": self.asset_id,
            "name": self.name,
            "segment_id": self.segment_id,
            "bbox_xyxy": list(self.bbox_xyxy),
            "score": self.score,
            "source_prompt": self.source_prompt,
            "mask_rle": self.mask_rle,
        }


@dataclass(frozen=True)
class ImageRelationGroup:
    """Segmentation alignment status for assets sharing one object name."""

    name: str
    expected_count: int
    detected_count: int
    status: str
    tried_prompts: list[str] = field(default_factory=list)
    asset_ids: list[str] = field(default_factory=list)
    debug_images: list[str] = field(default_factory=list)
    error: str | None = None

    def to_manifest(self) -> dict[str, Any]:
        """Convert the group to JSON-safe data."""
        return {
            "name": self.name,
            "expected_count": self.expected_count,
            "detected_count": self.detected_count,
            "status": self.status,
            "tried_prompts": list(self.tried_prompts),
            "asset_ids": list(self.asset_ids),
            "debug_images": list(self.debug_images),
            "error": self.error,
        }


@dataclass(frozen=True)
class ImageAnchor:
    """Anchor object used to place relative ordering onto the table grid."""

    asset_id: str
    grid: str
    reason: str = ""

    def to_manifest(self) -> dict[str, Any]:
        """Convert the anchor to JSON-safe data."""
        return {
            "asset_id": self.asset_id,
            "grid": self.grid,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ImageAssetLayout:
    """Support state for one image asset instance."""

    asset_id: str
    is_arbitrary_layout: bool
    reason: str = ""

    def to_manifest(self) -> dict[str, Any]:
        """Convert the layout to JSON-safe data."""
        return {
            "asset_id": self.asset_id,
            "is_arbitrary_layout": self.is_arbitrary_layout,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ImageRelationSpec:
    """Image asset segmentation alignment and spatial relations."""

    status: str
    image_path: str
    asset_segments: list[ImageAssetSegment]
    groups: list[ImageRelationGroup]
    table_segment: ImageAssetSegment | None = None
    table_group: ImageRelationGroup | None = None
    bbox_name_image_path: str | None = None
    anchor: ImageAnchor | None = None
    x_order: list[list[str]] = field(default_factory=list)
    y_order: list[list[str]] = field(default_factory=list)
    asset_layouts: list[ImageAssetLayout] = field(default_factory=list)

    def to_manifest(self) -> dict[str, Any]:
        """Convert the image relation spec to JSON-safe data."""
        manifest = self.to_segmentation_manifest()
        manifest.update(self.to_spatial_manifest())
        return manifest

    def to_segmentation_manifest(self) -> dict[str, Any]:
        """Convert only the segmentation alignment result to JSON-safe data."""
        return {
            "image_path": self.image_path,
            "bbox_name_image_path": self.bbox_name_image_path,
            "asset_segments": [
                segment.to_manifest() for segment in self.asset_segments
            ],
            "groups": [group.to_manifest() for group in self.groups],
            "table_segment": (
                self.table_segment.to_manifest() if self.table_segment else None
            ),
            "table_group": (
                self.table_group.to_manifest() if self.table_group else None
            ),
        }

    def to_spatial_manifest(self) -> dict[str, Any]:
        """Convert only spatial relations and layout states to JSON-safe data."""
        return {
            "image_path": self.image_path,
            "bbox_name_image_path": self.bbox_name_image_path,
            "anchor": self.anchor.to_manifest() if self.anchor else None,
            "spatial_order": {
                "left_to_right": [list(group) for group in self.x_order],
                "front_to_back": [list(group) for group in self.y_order],
            },
            "objects": [
                layout.to_manifest() for layout in self.asset_layouts
            ],
        }
