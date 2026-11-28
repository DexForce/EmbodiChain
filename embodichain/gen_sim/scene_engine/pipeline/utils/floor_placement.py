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

"""Identify visible floor assets and place them beside the finalized table."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import ConvexHull, QhullError

from embodichain.gen_sim.scene_engine.pipeline.utils.simready_processor_utils import (
    _strip_json_code_fence,
)
from embodichain.gen_sim.services.compat import vision_for_role
from embodichain.gen_sim.services.errors import ServiceError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from embodichain.gen_sim.scene_engine.core.scene_object import SceneObject
    from embodichain.gen_sim.services.compat import VisionClient

__all__: list[str] = []

FloorSide = Literal["left", "right", "front", "back"]
_SIDES = ("left", "right", "front", "back")
_MIN_OUTSIDE_FRACTION = 0.9
_TABLE_CLEARANCE_M = 0.02
_PROMPT = """Review the original scene and its labeled segmentation overview.
For each supplied candidate, decide whether it is clearly a separate object
resting directly on the floor beside the main table. Omit candidates on the
table, hung or held objects, objects on another low support, robot hardware,
and any ambiguous candidate. Use visible contact and depth cues; a mask
outside the tabletop alone is insufficient.
Return only a JSON object mapping a subset of the supplied candidate IDs to
"left", "right", "front", or "back", relative to the main table as seen in
the original image. Front is the side nearer the viewer; back is farther
away. For a corner, choose a side only if one clearly dominates. Return {}
when no candidate is certain."""


def _read_mask(path: str | None) -> np.ndarray | None:
    if path is None or not Path(path).is_file():
        return None
    with Image.open(path) as image:
        return np.asarray(image.convert("L")) != 0


def _filled_table_outline(table_mask: np.ndarray) -> np.ndarray | None:
    """Fill the outer footprint so tabletop holes do not look like floor."""
    rows = np.flatnonzero(table_mask.any(axis=1))
    if rows.size < 2:
        return None
    active = table_mask[rows]
    left = active.argmax(axis=1)
    right = active.shape[1] - 1 - active[:, ::-1].argmax(axis=1)
    points = np.concatenate([np.column_stack((edge, rows)) for edge in (left, right)])
    try:
        hull = ConvexHull(points)
    except QhullError:
        return None
    filled = Image.new("1", (table_mask.shape[1], table_mask.shape[0]))
    ImageDraw.Draw(filled).polygon(
        [tuple(point) for point in points[hull.vertices].tolist()], fill=1
    )
    return np.asarray(filled)


def identify_floor_placements(
    *,
    image_path: Path,
    table: SceneObject,
    assets: Sequence[SceneObject],
    vlm_client: VisionClient,
    debug_root: Path,
) -> dict[str, FloorSide]:
    """Use masks to gate one floor-side review of rigid table-root assets."""
    if not assets or table.mask_path is None:
        return {}
    overview = Path(table.mask_path).with_name("asset_masks_with_ids.png")
    if not overview.is_file():
        return {}
    table_mask = _read_mask(table.mask_path)
    if table_mask is None or (outline := _filled_table_outline(table_mask)) is None:
        return {}

    candidates = {}
    for asset in assets:
        if asset.is_articulated:
            continue
        mask = _read_mask(asset.mask_path)
        if mask is None or mask.shape != outline.shape or not mask.any():
            continue
        outside = np.count_nonzero(mask & ~outline) / np.count_nonzero(mask)
        if outside >= _MIN_OUTSIDE_FRACTION:
            candidates[asset.id] = {
                "description": asset.description,
                "outside_table_fraction": round(outside, 3),
            }
    if not candidates:
        return {}

    user_prompt = json.dumps({"candidates": candidates}, ensure_ascii=False)
    client = vision_for_role(vlm_client, "scene_understanding")
    complete_images = getattr(client, "complete_images", None)
    audit: dict[str, object] = {"candidates": candidates}
    placements: dict[str, FloorSide] = {}
    try:
        if callable(complete_images):
            raw = complete_images(
                system_prompt=_PROMPT,
                user_prompt=user_prompt,
                image_paths=[image_path, overview],
            )
        else:
            raw = client.complete(
                system_prompt=_PROMPT, user_prompt=user_prompt, image_path=overview
            )
        audit["raw_response"] = raw
        result = json.loads(_strip_json_code_fence(raw))
        if not isinstance(result, dict) or any(
            uid not in candidates or side not in _SIDES for uid, side in result.items()
        ):
            raise ValueError("Expected candidate IDs mapped to valid floor sides.")
        placements = result
    except (ServiceError, ValueError) as exc:
        audit["error"] = exc.code if isinstance(exc, ServiceError) else str(exc)
    audit["placements"] = placements
    debug_root.mkdir(parents=True, exist_ok=True)
    (debug_root / "floor_placement_review.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return placements


def floor_placement_delta(
    table_bounds_z_up: np.ndarray, asset_bounds_z_up: np.ndarray, side: FloorSide
) -> np.ndarray:
    """Translate an asset to the requested edge in the final front-camera frame."""
    delta = np.zeros(3)
    axis = 0 if side in {"left", "right"} else 1
    if side in {"left", "front"}:
        delta[axis] = (
            table_bounds_z_up[0, axis] - _TABLE_CLEARANCE_M - asset_bounds_z_up[1, axis]
        )
    else:
        delta[axis] = (
            table_bounds_z_up[1, axis] + _TABLE_CLEARANCE_M - asset_bounds_z_up[0, axis]
        )
    delta[2] = table_bounds_z_up[0, 2] - asset_bounds_z_up[0, 2]
    return delta
