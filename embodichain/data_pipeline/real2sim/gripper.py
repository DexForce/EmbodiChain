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

"""Explicit scalar-to-driver calibration, independent of a gripper model."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

__all__ = ["GripperPositionCalibration", "gripper_release_duration"]


def gripper_release_duration(
    timestamps: np.ndarray,
    gripper_open: np.ndarray,
    bilateral_contact: np.ndarray,
) -> float:
    """Measure continuous trailing release evidence in a recorded trace.

    The host computes gripper_open from measured positions and its explicit
    calibration/tolerance. Release requires an open gripper without bilateral
    grasp contact. This measures elapsed time, not a count of samples, and does
    not itself establish object stability, task success or source eligibility.

    Args:
        timestamps: Finite, strictly increasing (N,) elapsed seconds.
        gripper_open: Boolean (N,) measured-open evidence, not controller targets.
        bilateral_contact: Boolean (N,) simultaneous left/right grasp contacts.

    Returns:
        Seconds since the start of the final uninterrupted release interval.
        Empty, singleton or currently unreleased traces return zero.

    Raises:
        ValueError: Invalid shapes, masks, timestamps or nonfinite intervals.
    """
    times = np.asarray(timestamps, dtype=np.float64)
    opened = np.asarray(gripper_open)
    contact = np.asarray(bilateral_contact)
    if (
        times.ndim != 1
        or opened.shape != times.shape
        or contact.shape != times.shape
        or opened.dtype != np.bool_
        or contact.dtype != np.bool_
        or not np.isfinite(times).all()
    ):
        raise ValueError(
            "Release evidence requires finite (N,) times and boolean masks"
        )
    intervals = np.diff(times)
    if not np.isfinite(intervals).all() or np.any(intervals <= 0):
        raise ValueError(
            "Release timestamps must strictly increase with finite intervals"
        )
    released = opened & ~contact
    if len(times) < 2 or not released[-1]:
        return 0.0
    interruptions = np.flatnonzero(~released)
    start = int(interruptions[-1] + 1) if len(interruptions) else 0
    return float(times[-1] - times[start])


@dataclass(frozen=True)
class GripperPositionCalibration:
    """Decode a declared width convention through a reviewed lookup table.

    Args:
        encoding: One of normalized_closure, normalized_open or width_m.
        max_width_m: Width corresponding to a fully open source gripper.
        widths_m: Increasing measured or model-derived jaw widths in metres.
        driver_positions_rad: Corresponding independent driver angles.
        calibration_id: Versioned evidence identity.
        evidence: Source/calibration reference; never inferred from sample range.
        status: Review state; model-only estimates remain unverified.
    """

    encoding: str
    max_width_m: float
    widths_m: tuple[float, ...]
    driver_positions_rad: tuple[float, ...]
    calibration_id: str
    evidence: str
    status: str = "unverified"

    def __post_init__(self) -> None:
        if self.encoding not in {"normalized_closure", "normalized_open", "width_m"}:
            raise ValueError("Gripper encoding must be explicitly supported")
        widths = tuple(float(x) for x in self.widths_m)
        angles = tuple(float(x) for x in self.driver_positions_rad)
        if (
            not np.isfinite(self.max_width_m)
            or self.max_width_m <= 0
            or len(widths) < 2
            or len(widths) != len(angles)
            or not np.isfinite([*widths, *angles]).all()
            or np.any(np.diff(widths) <= 0)
            or widths[0] > 0
            or widths[-1] < self.max_width_m
        ):
            raise ValueError("Calibration table must cover [0, max_width_m]")
        if not (np.all(np.diff(angles) >= 0) or np.all(np.diff(angles) <= 0)):
            raise ValueError("Gripper driver calibration must be monotone")
        if any(
            not isinstance(x, str) or not x.strip()
            for x in (self.calibration_id, self.evidence)
        ):
            raise ValueError("Calibration ID and evidence are required")
        if self.status not in {"verified", "unverified", "rejected"}:
            raise ValueError("Unknown gripper calibration status")
        object.__setattr__(self, "widths_m", widths)
        object.__setattr__(self, "driver_positions_rad", angles)

    def decode(self, values: np.ndarray) -> np.ndarray:
        """Return driver radians without clipping malformed source values.

        Args:
            values: Finite scalar source samples under the declared encoding.

        Returns:
            Driver angles with the same shape as the source samples.
        """
        if self.status == "rejected":
            raise ValueError("Rejected gripper calibration cannot decode samples")
        values = np.asarray(values, dtype=np.float64)
        upper = self.max_width_m if self.encoding == "width_m" else 1.0
        if not np.isfinite(values).all() or np.any((values < 0) | (values > upper)):
            raise ValueError("Gripper samples exceed their declared encoding domain")
        width = (
            values
            if self.encoding == "width_m"
            else self.max_width_m
            * (1 - values if self.encoding == "normalized_closure" else values)
        )
        return np.interp(width, self.widths_m, self.driver_positions_rad)
