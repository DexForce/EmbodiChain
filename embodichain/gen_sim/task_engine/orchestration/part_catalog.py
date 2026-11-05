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
"""Semantic spatial labels from shared qualified articulation descriptors."""

from __future__ import annotations
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any
import numpy as np
from embodichain.lab.sim.scene.articulation_inputs import (
    discover_prismatic_parts,
    prismatic_reset_state,
    prismatic_handle_center_world,
)
from .source_scene import PreparedScene

__all__ = ["discover_part_catalogs"]


def discover_part_catalogs(
    prepared: PreparedScene,
) -> dict[str, tuple[dict[str, Any], ...]]:
    """Describe qualified source parts with conservative semantic spatial ranks.

    Args:
        prepared: Detached source/runtime declaration views sharing canonical UIDs.

    Returns:
        Ordered part descriptions per articulation UID. Positions and authored
        reset labels come from public Scene readers; absent or ambiguous inputs
        remain unlabeled. Spatial ranks are planning metadata, not observations
        or physical acceptance results.
    """
    catalogs: dict[str, tuple[dict[str, Any], ...]] = {}
    for config in prepared.articulations:
        try:
            parts = discover_prismatic_parts(dict(config))
        except (KeyError, TypeError, ValueError):
            continue
        payloads = []
        for part in parts:
            payload = part.payload()
            payload["initial_state"] = prismatic_reset_state(config, part.part_id)
            try:
                payload["handle_center_world"] = prismatic_handle_center_world(
                    config, part.part_id
                )
            except (KeyError, TypeError, ValueError):
                pass
            payloads.append(payload)
        catalogs[str(config["uid"])] = _rank_parts_by_lateral_position(
            _rank_parts_by_vertical_position(payloads)
        )
    return catalogs


def _rank_parts_by_vertical_position(
    parts: Sequence[Mapping[str, Any]], *, tolerance: float = 0.01
) -> tuple[dict[str, Any], ...]:
    ranked = [deepcopy(dict(part)) for part in parts]
    if len(ranked) < 2:
        return tuple(ranked)
    centers = [
        np.asarray(part.get("handle_center_world", ()), dtype=float) for part in ranked
    ]
    if any(center.shape != (3,) or not np.isfinite(center).all() for center in centers):
        return tuple(ranked)
    order = sorted(range(len(ranked)), key=lambda index: float(centers[index][2]))
    heights = [float(centers[index][2]) for index in order]
    if any(right - left <= tolerance for left, right in zip(heights, heights[1:])):
        return tuple(ranked)
    for vertical_index, part_index in enumerate(order):
        ranked[part_index]["vertical_index"] = vertical_index
        ranked[part_index]["part_count"] = len(ranked)
    ranked[order[0]]["vertical_rank"] = "bottom"
    ranked[order[-1]]["vertical_rank"] = "top"
    if len(ranked) % 2 == 1:
        ranked[order[len(ranked) // 2]]["vertical_rank"] = "middle"
    return tuple(ranked)


def _rank_parts_by_lateral_position(
    parts: Sequence[Mapping[str, Any]], *, tolerance: float = 0.01
) -> tuple[dict[str, Any], ...]:
    ranked = [deepcopy(dict(part)) for part in parts]
    if not 2 <= len(ranked) <= 3:
        return tuple(ranked)
    centers = [
        np.asarray(part.get("handle_center_world", ()), dtype=float) for part in ranked
    ]
    if any(center.shape != (3,) or not np.isfinite(center).all() for center in centers):
        return tuple(ranked)
    order = sorted(range(len(ranked)), key=lambda index: float(centers[index][1]))
    offsets = [float(centers[index][1]) for index in order]
    if any(right - left <= tolerance for left, right in zip(offsets, offsets[1:])):
        return tuple(ranked)
    ranked[order[0]]["lateral_rank"] = "left"
    ranked[order[-1]]["lateral_rank"] = "right"
    if len(ranked) % 2 == 1:
        ranked[order[len(ranked) // 2]]["lateral_rank"] = "center"
    return tuple(ranked)
