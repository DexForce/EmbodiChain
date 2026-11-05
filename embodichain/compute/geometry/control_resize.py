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

"""Numeric bounds and original ordered XY proposals for source-control preparation."""

from __future__ import annotations
import numpy as np

__all__ = ["control_bounds_overlap", "control_clearance_translations"]


def control_bounds_overlap(first: np.ndarray, second: np.ndarray) -> bool:
    """Return the source positive-volume AABB intersection.

    Args:
        first: First lower/upper bounds with shape ``(2, 3)``.
        second: Second lower/upper bounds with the same shape and frame.

    Returns:
        Whether all three overlap lengths exceed the original 1e-6 threshold.
        No native geometry, clearance query or semantic state is evaluated.
    """
    overlap = np.minimum(first[1], second[1]) - np.maximum(first[0], second[0])
    return bool(np.all(overlap > 1e-6))


def control_clearance_translations(
    target: np.ndarray, other: np.ndarray, margin: float
) -> tuple[tuple[np.ndarray, str], ...]:
    """Return the four original axis-only proposals in stable shortest-first order.

    Args:
        target: Protected target lower/upper world bounds (2, 3).
        other: Movable source object's lower/upper world bounds (2, 3).
        margin: Explicit unchanged collision reserve from the host layout policy.

    Returns:
        Owned XY translations and source direction labels. Tied distances retain
        the original -X,+X,-Y,+Y order; no source placement is mutated here.
    """
    proposals = [
        (np.array([target[0, 0] - other[1, 0] - margin, 0.0]), "x_negative"),
        (np.array([target[1, 0] - other[0, 0] + margin, 0.0]), "x_positive"),
        (np.array([0.0, target[0, 1] - other[1, 1] - margin]), "y_negative"),
        (np.array([0.0, target[1, 1] - other[0, 1] + margin]), "y_positive"),
    ]
    return tuple(sorted(proposals, key=lambda item: float(np.linalg.norm(item[0]))))
