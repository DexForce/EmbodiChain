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

"""Lightweight parallel-jaw physical descriptions; concrete values are model-owned."""

from __future__ import annotations

import math
from embodichain.utils import configclass

__all__ = ["ParallelJawGripperModelCfg"]


def _validate_gripper_transform(value: object) -> tuple[tuple[float, ...], ...]:
    """Normalize a finite SE(3) matrix without importing numerical backends."""
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("Gripper transform must have shape (4, 4).")
    try:
        rows = tuple(tuple(float(x) for x in row) for row in value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Gripper transform must contain finite numbers.") from exc
    if any(len(row) != 4 or any(not math.isfinite(x) for x in row) for row in rows):
        raise ValueError("Gripper transform must have finite shape (4, 4).")
    if any(abs(a - b) > 1e-7 for a, b in zip(rows[3], (0, 0, 0, 1))):
        raise ValueError("Gripper transform must have homogeneous last row.")
    if any(
        abs(sum(rows[k][i] * rows[k][j] for k in range(3)) - (i == j)) > 1e-6
        for i in range(3)
        for j in range(3)
    ):
        raise ValueError("Gripper transform rotation must be orthonormal.")
    r = rows
    determinant = (
        r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
        - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
        + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0])
    )
    if abs(determinant - 1) > 1e-6:
        raise ValueError("Gripper transform must be a proper rotation.")
    return rows


def _positive_finite(value: float, *, field_name: str) -> float:
    """Return one validated positive finite value."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a real number.")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{field_name} must be finite and positive.")
    return normalized


@configclass
class ParallelJawGripperModelCfg:
    """Named antipodal contact-width limits, independent of collision geometry.

    Opening width is the full distance between opposing object contacts. Actual
    gripper surfaces, opening commands and contact/TCP transforms are mandatory
    inputs to the concrete generator; scalar box dimensions are unsupported.
    """

    model_id: str = "parallel_jaw"
    """Identity of the physical hand whose contact-width limits are supplied."""

    min_opening_width: float = 0.001
    """Minimum usable full distance between antipodal contacts in metres."""

    max_opening_width: float = 0.1
    """Maximum usable full distance between antipodal contacts in metres."""

    catalog_id: str | None = None
    """Optional originating configuration ID; algorithms never resolve it."""

    root_link: str | None = None
    """Native hand subtree root; required when building physical geometry."""

    collision_links: tuple[str, ...] = ()
    """Native links whose complete collision surfaces are sampled."""

    pad_links: tuple[str, str] | None = None
    """Optional ordered contact-pad links for sampled closing services."""

    root_from_contact_frame: tuple[tuple[float, ...], ...] = (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )
    """Canonical contact frame expressed in the native hand-root frame."""

    def __post_init__(self) -> None:
        if (
            type(self.model_id) is not str
            or not self.model_id
            or self.model_id != self.model_id.strip()
        ):
            raise ValueError(
                "model_id must be a non-empty string without outer whitespace."
            )
        self.min_opening_width = _positive_finite(
            self.min_opening_width, field_name="min_opening_width"
        )
        self.max_opening_width = _positive_finite(
            self.max_opening_width, field_name="max_opening_width"
        )
        if self.min_opening_width >= self.max_opening_width:
            raise ValueError("min_opening_width must be less than max_opening_width.")

        if self.root_link is not None and (
            type(self.root_link) is not str
            or not self.root_link
            or self.root_link != self.root_link.strip()
        ):
            raise ValueError("root_link must be an explicit native link name.")
        self.collision_links = tuple(self.collision_links)
        if any(type(name) is not str or not name for name in self.collision_links):
            raise ValueError("collision_links must contain native link names.")
        if len(set(self.collision_links)) != len(self.collision_links):
            raise ValueError("collision_links must be unique.")
        if self.pad_links is not None:
            self.pad_links = tuple(self.pad_links)
            if len(self.pad_links) != 2 or len(set(self.pad_links)) != 2:
                raise ValueError("pad_links must contain two distinct links.")
        self.root_from_contact_frame = _validate_gripper_transform(
            self.root_from_contact_frame
        )
