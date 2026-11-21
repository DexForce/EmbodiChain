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

"""Pure source-envelope orientation and support proposals without simulation."""

from __future__ import annotations
from collections.abc import Mapping, Sequence
from typing import Any
import math
import numpy as np

__all__ = [
    "source_orientation_label",
    "source_support_proposal",
    "source_rigid_policy_facts",
]


def source_orientation_label(geometry: Mapping[str, Any] | None) -> str:
    """Classify a source dominant axis using the unchanged geometric thresholds.

    Args:
        geometry: Local extents, source axis rotation and primitive kind, or None.

    Returns:
        Standing, lying or unknown source orientation; no task success judgment.
    """
    if geometry is None or geometry["shape_type"] == "Sphere":
        return "unknown"
    extents = np.asarray(geometry["local_extents"], dtype=float)
    ordered = np.sort(extents)
    if ordered[-1] <= 0.0 or ordered[-1] / max(ordered[-2], 1.0e-9) < 1.2:
        return "unknown"
    dominant = int(np.argmax(extents))
    axis = np.asarray(geometry["axis_transform"], dtype=float)[:, dominant]
    vertical = abs(float(axis[2])) / max(float(np.linalg.norm(axis)), 1.0e-9)
    if vertical >= 0.75:
        return "standing"
    if vertical <= 0.35:
        return "lying"
    return "unknown"


def _geometry_from_record(item: Mapping[str, Any]) -> np.ndarray | None:
    aabb = item.get("world_aabb")
    if not isinstance(aabb, Mapping):
        return None
    return np.asarray([aabb["min"], aabb["max"]], dtype=float)


def source_support_proposal(
    uid: str,
    child: np.ndarray,
    objects: Mapping[str, Mapping[str, Any]],
    *,
    tolerance: float,
) -> dict[str, Any] | None:
    """Propose one source AABB support with original overlap/tie ordering.

    Args:
        uid: Child identity excluded from parent candidates.
        child: World bounds (2, 3).
        objects: Ordered source records with optional world_aabb min/max metadata.
        tolerance: Original vertical gap allowance in metres.

    Returns:
        Geometric proposal with original confidence/gap/overlap values, or None.
        This proposal establishes no physical load path or retention certificate.
    """
    child_bottom = float(child[0, 2])
    child_area = max(
        float((child[1, 0] - child[0, 0]) * (child[1, 1] - child[0, 1])),
        1.0e-9,
    )
    candidates = []
    for parent_uid, parent_item in objects.items():
        if parent_uid == uid:
            continue
        parent = _geometry_from_record(parent_item)
        if parent is None:
            continue
        overlap_x = max(
            0.0, min(child[1, 0], parent[1, 0]) - max(child[0, 0], parent[0, 0])
        )
        overlap_y = max(
            0.0, min(child[1, 1], parent[1, 1]) - max(child[0, 1], parent[0, 1])
        )
        overlap_ratio = float(overlap_x * overlap_y / child_area)
        gap = child_bottom - float(parent[1, 2])
        if overlap_ratio >= 0.1 and -tolerance <= gap <= tolerance:
            candidates.append((overlap_ratio, -abs(gap), parent_uid, gap))
    if not candidates:
        return None
    overlap_ratio, _, parent_uid, gap = max(candidates)
    confidence = min(1.0, overlap_ratio * max(0.0, 1.0 - abs(gap) / tolerance))
    return {
        "parent_uid": parent_uid,
        "relation": "on",
        "confidence": float(confidence),
        "gap_m": float(gap),
        "xy_overlap_ratio": float(overlap_ratio),
    }


def source_rigid_policy_facts(
    subject: Mapping[str, Any], objects: Sequence[Mapping[str, Any]]
) -> tuple[bool, bool]:
    """Read source AABB confinement and height proposals without runtime queries.

    Args:
        subject: Source object with optional final_world_aabb attributes.
        objects: Original source inventory order, including articulation bounds.

    Returns:
        Whether an articulation AABB encloses the source, and whether its
        vertical span exceeds both horizontal spans. These are planning
        permissions, not physical containment or contact evidence.
    """

    def bounds(item: Mapping[str, Any]) -> tuple | None:
        value = item.get("attributes", {}).get("final_world_aabb", {})
        low, high = value.get("min", ()), value.get("max", ())
        if (
            len(low) != 3
            or len(high) != 3
            or not all(
                math.isfinite(a) and math.isfinite(b) and a < b
                for a, b in zip(low, high, strict=True)
            )
        ):
            return None
        return low, high

    source = bounds(subject)
    if source is None:
        return False, False
    confined = False
    for item in objects:
        if item.get("role") != "articulation":
            continue
        enclosing = bounds(item)
        if enclosing is not None and all(
            enclosing[0][i] <= source[0][i] and source[1][i] <= enclosing[1][i]
            for i in range(3)
        ):
            confined = True
            break
    spans = [high - low for low, high in zip(*source, strict=True)]
    return confined, spans[2] > max(spans[:2])
