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

"""Physical readiness and release validation for explicit articulation controls."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import math

import torch

from .articulation_evidence import (
    JointContactSample,
    PressEventTarget,
    TwistEventTarget,
    evaluate_press_event,
    evaluate_twist_event,
    initial_twist_state_stable,
)

__all__ = [
    "TwistStartupSample",
    "press_start_is_released",
    "evaluate_twist_startup",
    "validate_press_ready",
    "validate_press_release",
    "validate_twist_ready",
    "validate_twist_terminal",
]


@dataclass(frozen=True, slots=True)
class TwistStartupSample:
    """One observed joint/contact startup fact, before control is armed."""

    timestamp: float
    qpos: float
    qvel: float
    table_target_contact: bool
    robot_target_contact: bool
    valid: bool
    dropped_contacts: int


def press_start_is_released(position: float, target: PressEventTarget) -> bool:
    """Check the original finite, ten-percent-stroke Press start condition.

    This is an instantaneous physical precondition, not released stability or
    task completion. Native observation and semantic binding remain host-owned.
    """
    return (
        math.isfinite(position)
        and abs(position - target.released_position) <= target.stroke * 0.1
    )


def evaluate_twist_startup(
    samples: Sequence[TwistStartupSample],
    *,
    complete: bool,
    overflow: bool,
) -> dict[str, object]:
    """Qualify source startup stability and absence of table/robot target contact."""
    trace = tuple(samples)
    valid = bool(
        complete
        and not overflow
        and len(trace) >= 2
        and all(frame.valid and frame.dropped_contacts == 0 for frame in trace)
        and all(
            math.isfinite(value)
            for frame in trace
            for value in (frame.timestamp, frame.qpos, frame.qvel)
        )
        and all(
            current.timestamp > previous.timestamp
            for previous, current in zip(trace, trace[1:])
        )
    )
    table_contacts = sum(frame.table_target_contact for frame in trace)
    robot_contacts = sum(frame.robot_target_contact for frame in trace)
    stable = initial_twist_state_stable(
        [frame.qpos for frame in trace], [frame.qvel for frame in trace]
    )
    reason = (
        "invalid_startup_evidence"
        if not valid
        else (
            "startup_table_contact"
            if table_contacts
            else (
                "startup_robot_contact"
                if robot_contacts
                else "startup_stable" if stable else "startup_motion"
            )
        )
    )
    return {
        "accepted": valid and not table_contacts and not robot_contacts and stable,
        "reason": reason,
        "sample_count": len(trace),
        "table_contact_samples": table_contacts,
        "robot_contact_samples": robot_contacts,
        "max_abs_qpos": max((abs(frame.qpos) for frame in trace), default=None),
        "max_abs_qvel": max((abs(frame.qvel) for frame in trace), default=None),
        "initial_angle_tolerance": math.radians(2.0),
        "initial_velocity_tolerance": math.radians(2.0),
    }


def validate_press_ready(
    target: PressEventTarget,
    positions: Sequence[float],
    initial: JointContactSample,
    *,
    read_limits: Callable[[], torch.Tensor],
    expected_limits: tuple[float, float],
    window_available: bool,
) -> dict[str, object]:
    """Retain released-state, no-contact and native limit checks at readiness."""
    values = list(positions)
    if not values:
        raise ValueError("Press readiness requires measured hold samples.")
    good = all(
        math.isfinite(value)
        and abs(value - target.released_position) <= target.stroke * 0.1
        for value in values
    )
    good = bool(
        good and initial.valid and initial.target_contact is False and window_available
    )
    limits_available = True
    if good:
        measured_limits = read_limits()
        limits_available = bool(torch.isfinite(measured_limits).all())
        good = torch.allclose(
            measured_limits.cpu(), torch.tensor(expected_limits), atol=1e-7, rtol=1e-5
        )
    return {
        "accepted": good,
        "phase": "released_stable",
        "joint_positions": values,
        "observation_available": bool(
            window_available
            and limits_available
            and all(math.isfinite(value) for value in values)
        ),
    }


def _released(samples: Sequence[JointContactSample], physics_dt: float) -> bool:
    if isinstance(physics_dt, bool) or not math.isfinite(physics_dt) or physics_dt <= 0:
        raise ValueError("Control release requires positive physics cadence.")
    count = max(1, math.ceil(0.15 / physics_dt))
    return bool(samples) and all(
        sample.target_contact is False and sample.valid for sample in samples[-count:]
    )


def validate_press_release(
    target: PressEventTarget,
    samples: Sequence[JointContactSample],
    *,
    parent_contacts: Sequence[bool | None],
    physics_dt: float,
    terminal: bool,
    retreat_verified: bool,
    retreat_clearance: float,
    tcp_clearance: float,
    terminal_clearance: float | None,
    observed_qpos: float,
    window_available: bool,
) -> dict[str, object]:
    """Qualify a measured press event plus source retreat/park release conditions."""
    if len(samples) != len(parent_contacts):
        raise ValueError("Press contacts must align with the complete sample window.")
    event = evaluate_press_event(target, samples)
    no_contact = _released(samples, physics_dt)
    count = max(1, math.ceil(0.15 / physics_dt))
    no_housing = bool(samples) and all(
        contact is False and sample.valid
        for sample, contact in zip(samples[-count:], parent_contacts[-count:])
    )
    required = terminal_clearance if terminal else retreat_clearance
    reason = event["reason"]
    good = False
    if not window_available:
        reason = "invalid_observation"
    elif event["accepted"]:
        if not no_contact:
            reason = "contact_not_released"
        elif terminal and not retreat_verified:
            reason = "retreat_not_verified"
        elif terminal and not no_housing:
            reason = "housing_contact_not_released"
        elif required is None or not math.isfinite(required) or required < 0.04:
            reason = (
                "terminal_clearance_insufficient"
                if terminal
                else "retreat_clearance_insufficient"
            )
        else:
            good = True
    if not terminal:
        retreat_verified = good
    return {
        **event,
        "accepted": good,
        "reason": reason,
        "event_accepted": event["accepted"],
        "retreat_clearance": retreat_clearance,
        "retreat_verified": retreat_verified,
        "terminal_clearance": terminal_clearance,
        "terminal_clearance_metric": "world_mesh_aabb_separation",
        "tcp_clearance": tcp_clearance,
        "contact_released": no_contact,
        "housing_contact_released": no_housing,
        "terminal": terminal,
        "observed_joint_position": observed_qpos,
    }


def validate_twist_ready(
    startup: Mapping[str, object],
    positions: Sequence[float],
    velocities: Sequence[float],
    initial: JointContactSample,
    *,
    window_available: bool,
) -> dict[str, object]:
    """Consume measured startup, stable-hold and initial released-contact facts."""
    values, speeds = list(positions), list(velocities)
    good = bool(
        startup["accepted"]
        and initial_twist_state_stable(values, speeds)
        and initial.valid
        and initial.target_contact is False
        and window_available
    )
    return {
        "accepted": good,
        "phase": "initial_stable",
        "reason": (
            startup["reason"]
            if not startup["accepted"]
            else "initial_stable" if good else "unstable_initial_state"
        ),
        "joint_positions": values,
        "joint_velocities": speeds,
        "initial_velocity_tolerance": math.radians(2.0),
        "startup_check": dict(startup),
    }


def validate_twist_terminal(
    target: TwistEventTarget,
    samples: Sequence[JointContactSample],
    *,
    parent_contacts: Sequence[bool | None],
    positions: Sequence[float],
    velocities: Sequence[float],
    physics_dt: float,
    clearance: float,
    others_unchanged: bool,
    terminal: bool,
    early_stop: bool,
    window_available: bool,
) -> dict[str, object]:
    """Qualify retained rotation, release, park clearance and non-target joints."""
    values, speeds = list(positions), list(velocities)
    if not values or len(values) != len(speeds) or len(samples) != len(parent_contacts):
        raise ValueError("Twist terminal validation requires aligned measured windows.")
    result = evaluate_twist_event(target, samples, final_qpos=values[-1])
    stable = bool(
        max(values) - min(values) <= math.radians(1.0)
        and all(
            math.isfinite(value) and abs(value) <= math.radians(2.0) for value in speeds
        )
    )
    released = _released(samples, physics_dt)
    good = bool(
        result["accepted"]
        and stable
        and released
        and (not terminal or clearance >= 0.04)
        and others_unchanged
        and window_available
    )
    parent_fraction = (
        sum(bool(value) for value in parent_contacts) / len(parent_contacts)
        if parent_contacts
        else 1.0
    )
    path = result.get("contact_path", 0.0)
    result.update(
        accepted=good,
        contact_released=released,
        stable=stable,
        joint_velocities=speeds,
        clearance=clearance,
        clearance_required=0.04 if terminal else None,
        others_unchanged=others_unchanged,
        parent_contact_fraction=parent_fraction,
        parent_contact_fraction_limit=0.25,
        parent_contact_warning=parent_fraction > 0.25,
        terminal=terminal,
        coarse_turn_threshold=target.coarse_turn_threshold,
        cumulative_contact_path=path,
        coarse_turn_reached=path >= target.coarse_turn_threshold,
        early_stop=early_stop,
    )
    return result
