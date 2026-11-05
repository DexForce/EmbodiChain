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

"""Numerical upward contact-patch measurements without simulation dependencies."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING
from collections.abc import Mapping
import math

if TYPE_CHECKING:
    import torch

import numpy as np
from shapely.geometry import MultiPoint, Point

__all__ = ["support_from_contacts", "contact_load_path", "contact_support_heights"]


def support_from_contacts(
    pairs: np.ndarray,
    positions: np.ndarray,
    normals: np.ndarray,
    impulses: np.ndarray,
    center_of_mass: np.ndarray,
    *,
    object_actor: int,
    reference_actor: int,
    tolerance: float,
) -> tuple[bool, dict[str, Any]]:
    """Measure the upward load patch between two explicit actors.

    Args:
        pairs: Native contact actor pairs, shape ``(C, 2)``.
        positions: Matching contact positions in one common frame, shape ``(C, 3)``.
        normals: Matching native normals directed from the first to second actor.
        impulses: Matching scalar normal impulses, shape ``(C,)``.
        center_of_mass: Observed payload COM in the same frame, shape ``(3,)``.
        object_actor: Exact native payload actor ID.
        reference_actor: Exact native support actor ID.
        tolerance: Declared COM-to-contact-patch tolerance in metres.

    Returns:
        Whether the upward patch geometrically supports the COM and its measured
        patch, distance and upward impulse. This instantaneous observation does
        not prove a complete stability window or task acceptance. Contact order,
        normal arithmetic and the source's upward-normal threshold are retained.

    Raises:
        ValueError: Contact observations are nonfinite or selected normals are zero.
    """
    if not all(
        np.isfinite(v).all() for v in (positions, normals, impulses, center_of_mass)
    ):
        raise ValueError("Native support observations must be finite.")
    selected = (
        ((pairs[:, 0] == reference_actor) & (pairs[:, 1] == object_actor))
        | ((pairs[:, 1] == reference_actor) & (pairs[:, 0] == object_actor))
    ) & (impulses > 0)
    directions = np.where(
        (pairs[selected, 0] == reference_actor)[:, None],
        normals[selected],
        -normals[selected],
    )
    lengths = np.linalg.norm(directions, axis=-1)
    if (lengths <= 1e-8).any():
        raise ValueError("Native support normals must be nonzero.")
    upward = directions[:, 2] / lengths >= 0.9
    patch = positions[selected][upward]
    distance = None
    if len(patch):
        distance = float(
            MultiPoint(patch[:, :2]).convex_hull.distance(Point(center_of_mass[:2]))
        )
    accepted = distance is not None and distance <= tolerance
    return accepted, {
        "contacts_world": patch.tolist(),
        "center_of_mass_world": center_of_mass.tolist(),
        "com_patch_distance_m": distance,
        "com_tolerance_m": tolerance,
        "upward_impulse": float(
            (impulses[selected][upward] * directions[upward, 2] / lengths[upward]).sum()
        ),
    }


def contact_load_path(
    pairs: list[list[int]],
    normals: list[list[float]],
    impulses: list[float],
    *,
    object_actor: int,
    reference_actor: int,
    dynamic_actors: set[int],
) -> tuple[bool, bool]:
    """Measure one substep's upward connectivity and external contact coupling.

    Args:
        pairs: Native actor pairs, shape ``(C, 2)``.
        normals: Matching normals from first to second actor, shape ``(C, 3)``.
        impulses: Matching nonnegative normal impulses, shape ``(C,)``.
        object_actor: Payload actor whose load path is queried.
        reference_actor: Exact root support actor.
        dynamic_actors: Explicit dynamic cargo actors allowed on intermediate paths.

    Returns:
        Whether a positive upward load path reaches the payload, and whether
        its coupled dynamic cargo has an external contact outside that rooted
        path. Substeps are never combined. Window coverage and qualification
        remain the caller's responsibility.

    Raises:
        ValueError: Actor IDs or contact arrays violate the numerical contract.
    """
    if any(type(value) is not int for pair in pairs for value in pair):
        raise ValueError("Stable support native actor IDs must be integers.")
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    normals = np.asarray(normals, dtype=float).reshape(-1, 3)
    impulse = np.asarray(impulses, dtype=float)
    if (
        normals.shape != (len(pairs), 3)
        or impulse.shape != (len(pairs),)
        or not np.isfinite(normals).all()
        or not np.isfinite(impulse).all()
        or (impulse < 0).any()
    ):
        raise ValueError("Stable support native contact arrays are invalid.")
    if (np.linalg.norm(normals, axis=-1)[impulse > 0] <= 1e-8).any():
        raise ValueError("Stable support has a zero native contact normal.")
    edges, contacts = [], []
    for (a, b), normal, value in zip(pairs, normals, impulse, strict=True):
        if value <= 0:
            continue
        contacts.append((a, b))
        if normal[2] > 0:
            edges.append((a, b))
        elif normal[2] < 0:
            edges.append((b, a))
    rooted = {reference_actor}
    coupled = {object_actor}
    # Do not combine paths across substeps, or traverse the reference into its
    # table. Only known dynamic rigid cargo may extend a native load path.
    while True:
        next_rooted = rooted | {
            b for a, b in edges if a in rooted and b in dynamic_actors
        }
        next_coupled = coupled | {
            b
            for a, b in contacts + [(b, a) for a, b in contacts]
            if a in coupled and b in dynamic_actors and b != reference_actor
        }
        if next_rooted == rooted and next_coupled == coupled:
            break
        rooted, coupled = next_rooted, next_coupled
    external = any(
        (a in coupled and b not in rooted) or (b in coupled and a not in rooted)
        for a, b in contacts
    )
    return object_actor in rooted, external


def contact_support_heights(
    supported: torch.Tensor,
    observation: Mapping[str, Any],
    like: torch.Tensor,
) -> torch.Tensor:
    """Refine heights only from NativeSupportObservation's qualified patch.

    Its contact points already require positive upward load and COM support.
    Keep the highest qualified point so penetration is not hidden by averaging.
    Unavailable rows cannot replace the existing conservative mesh envelope.

    Args:
        supported: Measured support flags, one boolean per environment.
        observation: Original contact-patch measurements and COM distances.
        like: Per-environment numeric reference for output shape, dtype and device.

    Returns:
        Highest qualified contact height for each row, or NaN when the row lacks
        finite positive upward load and a qualified COM patch. No pose is sampled.
    """
    import torch

    heights = torch.full_like(like, torch.nan)
    rows = observation.get("rows")
    if (
        supported.dtype != torch.bool
        or supported.shape != like.shape
        or observation.get("scope") != "native_contact_support"
        or not isinstance(rows, list)
        or len(rows) != like.shape[0]
    ):
        return heights
    for index, row in enumerate(rows):
        if not supported[index] or not isinstance(row, Mapping):
            continue
        values = tuple(
            row.get(name)
            for name in ("com_patch_distance_m", "com_tolerance_m", "upward_impulse")
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in values
        ):
            continue
        distance, tolerance, impulse = values
        if tolerance <= 0 or not 0 <= distance <= tolerance or impulse <= 0:
            continue
        if row.get("contacts_world") is None or row.get("center_of_mass_world") is None:
            continue
        try:
            points = torch.as_tensor(
                row["contacts_world"], device=like.device, dtype=like.dtype
            )
            center = torch.as_tensor(
                row["center_of_mass_world"], device=like.device, dtype=like.dtype
            )
        except (TypeError, ValueError):
            continue
        if (
            points.ndim != 2
            or points.shape[1] != 3
            or not points.shape[0]
            or center.shape != (3,)
            or not torch.isfinite(points).all()
            or not torch.isfinite(center).all()
        ):
            continue
        heights[index] = points[:, 2].amax()
    return heights
