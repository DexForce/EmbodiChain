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

"""Physical articulation event interpretation independent of task acceptance."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import math
from typing import Any, Literal
from embodichain.compute.articulation import (
    evaluate_press_event as _evaluate_press_event,
    evaluate_twist_event as _evaluate_twist_event,
    initial_twist_state_stable as _initial_twist_state_stable,
)

__all__ = [
    "PressEventTarget",
    "TwistEventTarget",
    "JointContactSample",
    "evaluate_press_event",
    "evaluate_twist_event",
    "initial_twist_state_stable",
]


@dataclass(frozen=True, slots=True)
class PressEventTarget:
    """Explicit released/pressed joint endpoints in runtime metres.

    Source geometry inspection and robot/sensor binding remain with their
    respective owners. This value supplies only the physical event coordinate.
    """

    released_position: float
    pressed_position: float

    def __post_init__(self) -> None:
        for name in ("released_position", "pressed_position"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError("Press endpoints must be finite runtime coordinates.")
            object.__setattr__(self, name, float(value))
        if self.released_position == self.pressed_position:
            raise ValueError("Press endpoints must be distinct.")

    @property
    def stroke(self) -> float:
        """Return the authored endpoint distance in metres."""
        return abs(self.pressed_position - self.released_position)


@dataclass(frozen=True, slots=True)
class JointContactSample:
    """Measured state after the preceding command; contact is exact target/hand."""

    timestamp: float
    qpos: float
    phase: str
    target_contact: bool | None
    valid: bool = True


def evaluate_press_event(
    binding: PressEventTarget, samples: Iterable[JointContactSample]
) -> dict[str, Any]:
    """Interpret the declared physical Press event through shared numeric kernels.

    Args:
        binding: Explicit released and pressed runtime endpoints.
        samples: Original ordered measured joint/contact samples.

    Returns:
        Source-compatible event measurements, not a task certificate.
    """
    return _evaluate_press_event(binding, samples)


@dataclass(frozen=True, slots=True)
class TwistEventTarget:
    """Explicit physical angular target, limits and event qualification in radians.

    The 15-degree coarse event threshold is distinct from the planning arc cap
    and the occurrence coverage budget. Selecting this value changes no shared
    skill default or program success criterion.
    """

    target_qpos: float
    limits: tuple[float, float]
    success_criterion: Literal["contact_backed_coarse_turn", "calibrated_setting"] = (
        "contact_backed_coarse_turn"
    )
    angle_tolerance: float = math.radians(5.0)
    coarse_turn_threshold: float = math.radians(15.0)
    path_deadband: float = math.radians(0.5)

    def __post_init__(self) -> None:
        for name in (
            "target_qpos",
            "angle_tolerance",
            "coarse_turn_threshold",
            "path_deadband",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError("Twist event coordinates must be finite radians.")
            if name != "target_qpos" and value <= 0:
                raise ValueError("Twist event tolerances must be positive.")
            object.__setattr__(self, name, float(value))
        limits = tuple(self.limits)
        if (
            len(limits) != 2
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                for value in limits
            )
            or limits[0] >= limits[1]
        ):
            raise ValueError("Twist event limits must be ordered finite radians.")
        object.__setattr__(self, "limits", tuple(float(value) for value in limits))
        if not self.limits[0] <= self.target_qpos <= self.limits[1]:
            raise ValueError("Twist event target must lie within the declared limits.")
        if self.success_criterion not in (
            "contact_backed_coarse_turn",
            "calibrated_setting",
        ):
            raise ValueError("Unknown Twist event criterion.")


def evaluate_twist_event(
    target: TwistEventTarget,
    samples: Iterable[JointContactSample],
    *,
    final_qpos: float,
) -> dict[str, Any]:
    """Interpret the explicitly selected physical rotation criterion.

    Args:
        target: Original coordinate, limits and configured event criterion.
        samples: Ordered measured joint/contact samples.
        final_qpos: Actual measured terminal joint position.

    Returns:
        Source-compatible event measurements retaining coarse/calibrated semantics.
    """
    return _evaluate_twist_event(target, samples, final_qpos=final_qpos)


def initial_twist_state_stable(
    positions: list[float],
    velocities: list[float],
    *,
    position_tolerance: float = math.radians(2.0),
    velocity_tolerance: float = math.radians(2.0),
) -> bool:
    """Apply the source physical stability tolerances to explicit measured values."""
    return _initial_twist_state_stable(
        positions,
        velocities,
        position_tolerance=position_tolerance,
        velocity_tolerance=velocity_tolerance,
    )
