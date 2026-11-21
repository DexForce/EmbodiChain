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

"""Static and runtime control timing share one environment-owned contract."""

from __future__ import annotations
from dataclasses import dataclass
import math
from numbers import Integral

__all__ = ["ControlTiming", "resolve_control_timing"]


@dataclass(frozen=True, slots=True)
class ControlTiming:
    """Validated cadence from file-owned physics and control declarations."""

    physics_dt: float
    sim_steps_per_control: int

    @property
    def step_dt(self) -> float:
        """Return the exact unresampled control interval."""
        return self.physics_dt * self.sim_steps_per_control


def resolve_control_timing(
    physics_dt: float,
    sim_steps_per_control: int,
    *,
    target_control_frequency: float | None = None,
) -> ControlTiming:
    """Resolve exactly the same integer physics grid before or during environment setup.

    Args:
        physics_dt: File-owned physics interval, or its existing public default.
        sim_steps_per_control: Explicit positive integral decimation.
        target_control_frequency: Optional exactly representable frequency;
            retains its existing priority over explicit decimation.

    Returns:
        Exact validated timing without creating any simulator or controller.

    Raises:
        ValueError: Physics/control inputs are invalid or the requested rate is
            not exactly representable on the physics time grid.
    """
    sim_steps = sim_steps_per_control
    target_frequency = target_control_frequency
    try:
        physics_dt_value = float(physics_dt)
    except (TypeError, ValueError):
        physics_dt_value = math.nan
    if (
        isinstance(physics_dt, bool)
        or not math.isfinite(physics_dt_value)
        or physics_dt_value <= 0.0
    ):
        raise ValueError(
            f"physics_dt must be a finite positive number, got {physics_dt!r}."
        )

    if target_frequency is not None:
        try:
            target_frequency_value = float(target_frequency)
        except (TypeError, ValueError):
            target_frequency_value = math.nan
        if (
            isinstance(target_frequency, bool)
            or not math.isfinite(target_frequency_value)
            or target_frequency_value <= 0.0
        ):
            raise ValueError(
                "target_control_frequency must be a finite positive number, "
                f"got {target_frequency!r}."
            )

        ideal_steps = 1.0 / (physics_dt_value * target_frequency_value)
        resolved_steps = max(1, round(ideal_steps))
        if not math.isclose(
            ideal_steps, float(resolved_steps), rel_tol=0.0, abs_tol=1e-9
        ):
            achievable_frequency = 1.0 / (physics_dt_value * resolved_steps)
            raise ValueError(
                f"target_control_frequency={target_frequency!r} cannot be "
                f"represented exactly with physics_dt={physics_dt!r}. The nearest "
                f"integer sim_steps_per_control is {resolved_steps}, which gives "
                f"{achievable_frequency:g} Hz. Set sim_steps_per_control explicitly "
                "or choose an exactly representable target frequency."
            )
        sim_steps = resolved_steps

    if isinstance(sim_steps, bool) or not isinstance(sim_steps, Integral):
        raise ValueError(
            "sim_steps_per_control must be a positive integer, " f"got {sim_steps!r}."
        )
    if sim_steps <= 0:
        raise ValueError(
            "sim_steps_per_control must be a positive integer, " f"got {sim_steps!r}."
        )

    return ControlTiming(physics_dt_value, sim_steps)
