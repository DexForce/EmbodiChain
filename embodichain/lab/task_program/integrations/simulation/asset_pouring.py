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

"""Measured rigid-body acceptance for bottle-to-cup pouring motions.

The monitor verifies sampled motion geometry, target stability and bottle
return. It does not measure liquid transfer or claim fluid simulation success.
Hosts supply live poses after each control step and identify the pour segment.
"""

from __future__ import annotations

from collections.abc import Sequence
import math
from numbers import Real
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from embodichain.lab.sim.motion.expansion import ValidationCheck, ValidationResult
from embodichain.lab.sim.scene._validation import _rigid_pose

__all__ = ["PouringAcceptanceMonitor"]


def _finite(
    value: float, name: str, *, lower: float = 0.0, strict: bool = False
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or value < lower
        or (strict and value == lower)
    ):
        qualifier = "greater than" if strict else "at least"
        raise ValueError(f"{name} must be finite and {qualifier} {lower}.")
    return float(value)


def _point(value: Sequence[float], name: str) -> tuple[float, float, float]:
    point = tuple(value)
    if len(point) != 3 or any(
        isinstance(item, bool) or not isinstance(item, Real) or not math.isfinite(item)
        for item in point
    ):
        raise ValueError(f"{name} must contain three finite real coordinates.")
    return tuple(float(item) for item in point)


def _pose(value: ArrayLike) -> NDArray[np.float64]:
    return np.asarray(_rigid_pose(value), dtype=np.float64)


def _tilt(pose: NDArray[np.float64]) -> float:
    return math.acos(float(np.clip(pose[2, 2], -1.0, 1.0)))


class PouringAcceptanceMonitor:
    """Accumulate bounded pose evidence for one rigid bottle/cup episode.

    Both assets use meter-scale local coordinates with +Z as their upright axis.
    Mouth alignment is computed in the cup frame: the transformed bottle mouth
    must project inside the cup opening and lie within its positive height band
    while the bottle is tilted, on the same sampled control steps.

    Args:
        bottle_mouth_xyz: Bottle mouth center in the bottle's local frame.
        cup_mouth_xyz: Cup opening center in the cup's local frame.
        bottle_mouth_radius_m: Conservative radius enclosing the bottle opening.
        cup_opening_radius_m: Radius of the usable circular cup opening.
        min_tilt_rad: Minimum bottle tilt from world +Z during pouring.
        max_cup_displacement_m: Maximum cup-origin excursion from its initial pose.
        return_position_tolerance_m: Final bottle-origin distance from its initial pose.
        upright_tolerance_rad: Maximum final bottle tilt and any sampled cup tilt.
        min_mouth_height_m: Minimum positive bottle-mouth height above cup opening.
        max_mouth_height_m: Maximum bottle-mouth height above cup opening.
        min_pour_samples: Minimum simultaneous qualifying control-step samples.

    Only constant-size aggregates are retained. Sampling frequency is owned by
    the host; sample counts do not establish a continuous-time hold duration.
    """

    def __init__(
        self,
        *,
        bottle_mouth_xyz: Sequence[float],
        cup_mouth_xyz: Sequence[float],
        bottle_mouth_radius_m: float,
        cup_opening_radius_m: float,
        min_tilt_rad: float = math.radians(50),
        max_cup_displacement_m: float = 0.005,
        return_position_tolerance_m: float = 0.02,
        upright_tolerance_rad: float = math.radians(10),
        min_mouth_height_m: float = 0.015,
        max_mouth_height_m: float = 0.25,
        min_pour_samples: int = 2,
    ) -> None:
        self._bottle_mouth = _point(bottle_mouth_xyz, "bottle_mouth_xyz")
        self._cup_mouth = _point(cup_mouth_xyz, "cup_mouth_xyz")
        self._bottle_radius = _finite(
            bottle_mouth_radius_m, "bottle_mouth_radius_m", strict=True
        )
        self._radius = _finite(
            cup_opening_radius_m, "cup_opening_radius_m", strict=True
        )
        if self._bottle_radius > self._radius:
            raise ValueError(
                "Bottle mouth radius cannot exceed the cup opening radius."
            )
        self._min_tilt = _finite(min_tilt_rad, "min_tilt_rad", strict=True)
        self._max_cup_displacement = _finite(
            max_cup_displacement_m, "max_cup_displacement_m"
        )
        self._return_tolerance = _finite(
            return_position_tolerance_m, "return_position_tolerance_m"
        )
        self._upright_tolerance = _finite(
            upright_tolerance_rad, "upright_tolerance_rad"
        )
        self._min_height = _finite(
            min_mouth_height_m, "min_mouth_height_m", strict=True
        )
        self._max_height = _finite(
            max_mouth_height_m,
            "max_mouth_height_m",
            lower=self._min_height,
            strict=True,
        )
        if self._min_tilt > math.pi or self._upright_tolerance >= math.pi / 2:
            raise ValueError(
                "Tilt must not exceed pi; upright tolerance must be less than pi/2."
            )
        if type(min_pour_samples) is not int or min_pour_samples < 2:
            raise ValueError("min_pour_samples must be an integer of at least two.")
        self._min_pour_samples = min_pour_samples
        self._clear()

    def _clear(self) -> None:
        self._initial_bottle: NDArray[np.float64] | None = None
        self._initial_cup: NDArray[np.float64] | None = None
        self._samples = 0
        self._pour_samples = 0
        self._qualifying_samples = 0
        self._invalid_samples = 0
        self._max_pour_tilt = 0.0
        self._min_radial: float | None = None
        self._min_pour_height: float | None = None
        self._max_pour_height: float | None = None
        self._max_cup_displacement_seen = 0.0
        self._max_cup_tilt_seen = 0.0
        self._final_return_distance: float | None = None
        self._final_bottle_tilt: float | None = None
        self._accepted: bool | None = None

    def reset(
        self, initial_bottle_pose: ArrayLike, initial_cup_pose: ArrayLike
    ) -> None:
        """Start a fresh episode from owned, finite proper SE(3) poses.

        Any invalid reset clears prior evidence before raising ``ValueError``.
        """
        self._clear()
        bottle = _pose(initial_bottle_pose)
        cup = _pose(initial_cup_pose)
        self._initial_bottle = bottle
        self._initial_cup = cup
        self._track_cup(cup)

    def _track_cup(self, cup: NDArray[np.float64]) -> None:
        assert self._initial_cup is not None
        displacement = float(np.linalg.norm(cup[:3, 3] - self._initial_cup[:3, 3]))
        self._max_cup_displacement_seen = max(
            self._max_cup_displacement_seen, displacement
        )
        self._max_cup_tilt_seen = max(self._max_cup_tilt_seen, _tilt(cup))

    def observe(
        self,
        bottle_pose: ArrayLike,
        cup_pose: ArrayLike,
        *,
        in_pour_segment: bool,
    ) -> None:
        """Record one control-step sample, retaining excursions outside pouring.

        Invalid poses or a non-boolean segment marker poison this episode's
        evidence and raise. Catching that error cannot later accept the episode.
        """
        if self._initial_bottle is None or self._initial_cup is None:
            raise RuntimeError("reset must establish initial poses before observe.")
        self._accepted = None
        try:
            if type(in_pour_segment) is not bool:
                raise TypeError("in_pour_segment must be a bool.")
            bottle = _pose(bottle_pose)
            cup = _pose(cup_pose)
        except (TypeError, ValueError):
            self._invalid_samples += 1
            raise
        self._samples += 1
        self._track_cup(cup)
        if not in_pour_segment:
            return
        self._pour_samples += 1
        tilt = _tilt(bottle)
        bottle_mouth_world = bottle[:3, :3] @ self._bottle_mouth + bottle[:3, 3]
        cup_mouth_world = cup[:3, :3] @ self._cup_mouth + cup[:3, 3]
        offset_in_cup = cup[:3, :3].T @ (bottle_mouth_world - cup_mouth_world)
        radial = math.hypot(float(offset_in_cup[0]), float(offset_in_cup[1]))
        height = float(offset_in_cup[2])
        self._max_pour_tilt = max(self._max_pour_tilt, tilt)
        self._min_radial = (
            radial if self._min_radial is None else min(self._min_radial, radial)
        )
        self._min_pour_height = (
            height
            if self._min_pour_height is None
            else min(self._min_pour_height, height)
        )
        self._max_pour_height = (
            height
            if self._max_pour_height is None
            else max(self._max_pour_height, height)
        )
        if (
            tilt >= self._min_tilt
            and radial + self._bottle_radius <= self._radius
            and self._min_height <= height <= self._max_height
        ):
            self._qualifying_samples += 1

    def validate_final(
        self, bottle_pose: ArrayLike, cup_pose: ArrayLike
    ) -> ValidationResult:
        """Check sampled pouring geometry, all cup excursions and final return.

        Missing reset or samples produce unavailable checks. Final poses cannot
        add qualifying pour samples. Invalid final poses raise and poison the
        episode, preventing a later valid pose from hiding the missing evidence.
        """
        try:
            bottle = _pose(bottle_pose)
            cup = _pose(cup_pose)
        except (TypeError, ValueError):
            self._invalid_samples += 1
            self._accepted = False
            raise
        if self._initial_bottle is None or self._initial_cup is None:
            self._accepted = False
            return ValidationResult(
                (
                    ValidationCheck(
                        "pour.initial_state",
                        "unavailable",
                        "No initial pose snapshot was recorded.",
                    ),
                )
            )
        self._track_cup(cup)
        distance = float(np.linalg.norm(bottle[:3, 3] - self._initial_bottle[:3, 3]))
        tilt = _tilt(bottle)
        self._final_return_distance = distance
        self._final_bottle_tilt = tilt
        result = ValidationResult(
            (
                ValidationCheck(
                    "pour.samples_valid",
                    (
                        "passed"
                        if self._samples and not self._invalid_samples
                        else "failed" if self._invalid_samples else "unavailable"
                    ),
                    "Every observed sample must contain finite proper rigid transforms.",
                    {
                        "samples": self._samples,
                        "invalid_samples": self._invalid_samples,
                    },
                ),
                ValidationCheck(
                    "pour.simultaneous_alignment",
                    (
                        "passed"
                        if self._qualifying_samples >= self._min_pour_samples
                        else "failed" if self._pour_samples else "unavailable"
                    ),
                    "Tilt, aperture alignment and positive mouth height must hold simultaneously during pouring.",
                    {
                        "pour_samples": self._pour_samples,
                        "qualifying_samples": self._qualifying_samples,
                        "required_samples": self._min_pour_samples,
                    },
                ),
                ValidationCheck(
                    "pour.cup_stable",
                    (
                        "passed"
                        if self._max_cup_displacement_seen <= self._max_cup_displacement
                        and self._max_cup_tilt_seen <= self._upright_tolerance
                        else "failed"
                    ),
                    "Cup displacement and tilt limits apply throughout the observed episode.",
                    {
                        "max_displacement_m": self._max_cup_displacement_seen,
                        "max_tilt_rad": self._max_cup_tilt_seen,
                    },
                ),
                ValidationCheck(
                    "pour.bottle_returned",
                    "passed" if distance <= self._return_tolerance else "failed",
                    "Bottle origin must return near its initial position.",
                    {"distance_m": distance},
                ),
                ValidationCheck(
                    "pour.bottle_upright",
                    "passed" if tilt <= self._upright_tolerance else "failed",
                    "Bottle +Z must be upright at the end of the episode.",
                    {"tilt_rad": tilt},
                ),
            )
        )
        self._accepted = result.accepted
        return result

    def to_metadata(self) -> dict[str, Any]:
        """Return an independent JSON-compatible evidence and threshold snapshot."""
        return {
            "schema_version": 1,
            "evidence_kind": "sampled_rigid_pouring_motion",
            "fluid_transfer_measured": False,
            "initial_bottle_pose": (
                None if self._initial_bottle is None else self._initial_bottle.tolist()
            ),
            "initial_cup_pose": (
                None if self._initial_cup is None else self._initial_cup.tolist()
            ),
            "sample_count": self._samples,
            "pour_sample_count": self._pour_samples,
            "qualifying_sample_count": self._qualifying_samples,
            "invalid_sample_count": self._invalid_samples,
            "max_pour_tilt_rad": self._max_pour_tilt,
            "min_pour_radial_distance_m": self._min_radial,
            "min_pour_mouth_height_m": self._min_pour_height,
            "max_pour_mouth_height_m": self._max_pour_height,
            "max_cup_displacement_m": self._max_cup_displacement_seen,
            "max_cup_tilt_rad": self._max_cup_tilt_seen,
            "final_bottle_return_distance_m": self._final_return_distance,
            "final_bottle_tilt_rad": self._final_bottle_tilt,
            "accepted": self._accepted,
            "thresholds": {
                "bottle_mouth_xyz": list(self._bottle_mouth),
                "cup_mouth_xyz": list(self._cup_mouth),
                "bottle_mouth_radius_m": self._bottle_radius,
                "cup_opening_radius_m": self._radius,
                "min_tilt_rad": self._min_tilt,
                "max_cup_displacement_m": self._max_cup_displacement,
                "return_position_tolerance_m": self._return_tolerance,
                "upright_tolerance_rad": self._upright_tolerance,
                "min_mouth_height_m": self._min_height,
                "max_mouth_height_m": self._max_height,
                "min_pour_samples": self._min_pour_samples,
            },
        }
