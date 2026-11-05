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

from typing import Any

import numpy as np
from shapely.geometry import MultiPoint, Point

__all__ = ["support_from_contacts"]


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
