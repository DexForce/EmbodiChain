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

"""Numeric interpretation of explicit articulation contact-event parameters."""

from __future__ import annotations
from collections.abc import Iterable
import math
from typing import Any, Literal, Protocol

__all__: list[str] = []


class PressEventParameters(Protocol):
    """Explicit runtime-coordinate endpoints; configuration defaults stay with callers."""

    released_position: float
    pressed_position: float

    @property
    def stroke(self) -> float:
        """Return the absolute endpoint distance in runtime metres."""
        ...


class TwistEventParameters(Protocol):
    """Explicit angular coordinate and event criterion operands, all in radians."""

    target_qpos: float
    limits: tuple[float, float]
    success_criterion: Literal["contact_backed_coarse_turn", "calibrated_setting"]
    angle_tolerance: float
    coarse_turn_threshold: float
    path_deadband: float


class JointContactValues(Protocol):
    """One ordered measured joint/contact sample, without a simulation owner."""

    timestamp: float
    qpos: float
    phase: str
    target_contact: bool | None
    valid: bool


def evaluate_press_event(
    binding: PressEventParameters, samples: Iterable[JointContactValues]
) -> dict[str, Any]:
    """Require a released onset and continuous contact-backed press-phase travel.

    Callers must mark missing, stale, ambiguous or saturated contact observations
    invalid. This records a transient press event, not persistent activation or
    a terminal Task Program success certificate.

    Args:
        binding: Explicit physical released and pressed endpoints.
        samples: Ordered measured joint/contact samples, including released onset.

    Returns:
        Source-compatible event measurements and reason. ``invalid_observation``
        is unavailable evidence, not a measured negative or a successful event.
    """
    trace = tuple(samples)
    tolerance = min(0.0005, binding.stroke * 0.1)
    direction = math.copysign(1.0, binding.pressed_position - binding.released_position)
    result: dict[str, Any] = {
        "accepted": False,
        "criterion": "contact_press_0.05mm",
        "minimum_contact_stroke_m": 0.00005,
        "activation_verified": False,
        "reason": "missing_samples",
        "max_stroke": 0.0,
        "contact_stroke": 0.0,
        "contact_peak": 0.0,
    }
    if not trace:
        return result
    times = [s.timestamp for s in trace]
    if any(
        not s.valid
        or not math.isfinite(s.qpos)
        or not math.isfinite(s.timestamp)
        or s.timestamp < 0
        or type(s.target_contact) is not bool
        for s in trace
    ) or any(b <= a for a, b in zip(times, times[1:])):
        return {**result, "reason": "invalid_observation"}
    travel = [(s.qpos - binding.released_position) * direction for s in trace]
    result["max_stroke"] = max(travel)
    if any(p < -tolerance or p > binding.stroke + tolerance for p in travel):
        return {**result, "reason": "joint_out_of_bounds"}
    if (
        abs(travel[0]) > tolerance
        or trace[0].phase == "press"
        or trace[0].target_contact
    ):
        return {**result, "reason": "not_initially_released"}
    started = False
    anchor: float | None = None
    previous = 0.0
    for sample, progress in zip(trace, travel):
        if sample.phase != "press":
            if not started and progress > tolerance:
                return {**result, "reason": "passive_before_press"}
            anchor = None
            continue
        started = True
        if not sample.target_contact:
            anchor = None
            continue
        if anchor is None:
            if progress > tolerance:
                continue
            anchor = progress
        if progress < previous - tolerance:
            anchor = None
        elif anchor is not None:
            result["contact_stroke"] = max(result["contact_stroke"], progress - anchor)
            result["contact_peak"] = max(result["contact_peak"], progress)
        previous = progress
    if result["contact_stroke"] >= result["minimum_contact_stroke_m"]:
        return {**result, "accepted": True, "reason": "contact_press_event"}
    return {**result, "reason": "insufficient_contact_stroke"}


def _debounced_contact_path(
    samples: tuple[JointContactValues, ...], deadband: float
) -> float:
    path, anchor = 0.0, None
    for sample in samples:
        if sample.target_contact is not True:
            anchor = None
        elif anchor is None:
            anchor = sample.qpos
        elif abs(sample.qpos - anchor) >= deadband:
            path += abs(sample.qpos - anchor)
            anchor = sample.qpos
    return path


def evaluate_twist_event(
    target: TwistEventParameters,
    samples: Iterable[JointContactValues],
    *,
    final_qpos: float,
) -> dict[str, Any]:
    """Interpret measured contact-backed rotation using the declared criterion.

    This preserves coarse path credit, release/regrasp reverse-travel subtraction
    and calibrated-setting qualification. It is physical effect evidence, not a
    planning outcome or TaskSpec certificate. Invalid measurements retain the
    ``invalid_observation`` reason and must remain unavailable to consumers.
    """
    samples = tuple(samples)
    b = target
    result = {
        "accepted": False,
        "criterion": "contact_backed_coarse_turn",
        "target_qpos": b.target_qpos,
        "observed_qpos": final_qpos,
        "angle_tolerance": target.angle_tolerance,
        "coarse_turn_threshold": target.coarse_turn_threshold,
        "coarse_turn_reached": False,
        "target_contact_seen": False,
        "reason": "invalid_observation",
    }
    if (
        not samples
        or any(
            not s.valid
            or not math.isfinite(s.qpos)
            or not math.isfinite(s.timestamp)
            or s.timestamp < 0
            or type(s.target_contact) is not bool
            for s in samples
        )
        or any(b.timestamp <= a.timestamp for a, b in zip(samples, samples[1:]))
    ):
        return result
    if samples[0].target_contact:
        return {**result, "reason": "unstable_initial_state"}
    if any(not b.limits[0] - 0.02 <= s.qpos <= b.limits[1] + 0.02 for s in samples):
        return {**result, "reason": "joint_out_of_bounds"}
    sign = math.copysign(1.0, b.target_qpos - samples[0].qpos)
    signed_contact_travel, uncontacted_reverse_travel, raw_contact_path = 0.0, 0.0, 0.0
    for previous, current in zip(samples, samples[1:]):
        delta = sign * (current.qpos - previous.qpos)
        if previous.target_contact and current.target_contact:
            signed_contact_travel += delta
            raw_contact_path += abs(delta)
        elif delta < 0.0:
            # Release/regrasp reversals must not let repeated excursions earn credit.
            uncontacted_reverse_travel -= delta
    directional_travel = max(0.0, signed_contact_travel - uncontacted_reverse_travel)
    contact_path = _debounced_contact_path(samples, target.path_deadband)
    error = abs(final_qpos - b.target_qpos)
    target_contact_seen = any(s.target_contact for s in samples)
    required_directional_travel = min(
        target.coarse_turn_threshold, abs(b.target_qpos - samples[0].qpos)
    )
    accepted = (
        math.isfinite(final_qpos)
        and target_contact_seen
        and contact_path >= target.coarse_turn_threshold
    )
    if target.success_criterion == "calibrated_setting":
        accepted = (
            accepted
            and directional_travel >= required_directional_travel
            and error <= target.angle_tolerance
        )
    return {
        **result,
        "accepted": accepted,
        "criterion": target.success_criterion,
        "contact_travel": directional_travel,
        "signed_contact_travel": signed_contact_travel,
        "uncontacted_reverse_travel": uncontacted_reverse_travel,
        "contact_path": contact_path,
        "raw_contact_path": raw_contact_path,
        "target_contact_seen": target_contact_seen,
        "directional_travel_required": required_directional_travel,
        "target_error": error,
        "target_qpos_reached": error <= target.angle_tolerance,
        "coarse_turn_reached": contact_path >= target.coarse_turn_threshold,
        "reason": (
            "coarse_turn_reached"
            if accepted
            else (
                "target_contact_missing"
                if not target_contact_seen
                else "coarse_contact_travel_insufficient"
            )
        ),
    }


def initial_twist_state_stable(
    positions: list[float],
    velocities: list[float],
    *,
    position_tolerance: float,
    velocity_tolerance: float,
) -> bool:
    """Reject a stationary-looking joint with unresolved angular velocity."""
    return bool(
        positions
        and len(positions) == len(velocities)
        and all(math.isfinite(q) for q in positions)
        and max(positions) - min(positions) <= position_tolerance
        and all(math.isfinite(v) and abs(v) <= velocity_tolerance for v in velocities)
    )
