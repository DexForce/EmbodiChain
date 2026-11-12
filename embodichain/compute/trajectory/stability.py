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

"""Pure sampled-coordinate range windows, independent of simulation and tasks."""

from __future__ import annotations

import math
import torch

__all__ = ["PositionRangeMonitor"]


class PositionRangeMonitor:
    """Track continuous qualifying intervals with reset-on-range-drift semantics.

    Args:
        initial: Initial floating coordinates with shape ``(B,)``. Units are
            supplied by the caller; periodic coordinates are not unwrapped.
        required_steps: Positive number of completed sample intervals required.
        position_drift: Positive maximum range in one candidate window.

    The caller supplies a separate pointwise target predicate. Tick zero does
    not count as an elapsed interval. Re-entering the target starts a new window
    at zero completed intervals. Success is current, not latched: a later
    violation resets the row even if an earlier window succeeded. Duplicate
    step observations are idempotent; missing intervals cannot prove stability.
    This numerical monitor reads no Robot, clock, simulator or normative spec.
    """

    def __init__(
        self, initial: torch.Tensor, *, required_steps: int, position_drift: float
    ) -> None:
        if (
            not isinstance(initial, torch.Tensor)
            or initial.ndim != 1
            or not initial.numel()
            or not initial.is_floating_point()
        ):
            raise ValueError("initial must be a nonempty floating coordinate vector.")
        if type(required_steps) is not int or required_steps < 1:
            raise ValueError("required_steps must be a positive integer.")
        if (
            isinstance(position_drift, bool)
            or not isinstance(position_drift, (int, float))
            or not math.isfinite(position_drift)
            or position_drift <= 0
        ):
            raise ValueError("position_drift must be finite and positive.")
        self._low = initial.clone()
        self._high = initial.clone()
        self._counts = torch.zeros_like(initial, dtype=torch.long)
        self._satisfied = torch.zeros_like(initial, dtype=torch.bool)
        self._previous_good = torch.zeros_like(initial, dtype=torch.bool)
        self._required = required_steps
        self._drift = float(position_drift)
        self._last_step = -1

    def observe(
        self, values: torch.Tensor, within_target: torch.Tensor, *, elapsed_steps: int
    ) -> torch.Tensor:
        """Consume one actual sample without inventing unobserved intervals.

        Args:
            values: Coordinates with the original batch shape, dtype and device.
            within_target: Per-row pointwise target predicate, independently chosen
                by the caller. Non-finite values always invalidate this predicate.
            elapsed_steps: Monotonic completed sample intervals since entry.

        Returns:
            Independently owned current window-satisfaction mask.

        Raises:
            ValueError: Shapes/devices differ, counters regress, or a sample
                interval is missing. The first observation must be tick zero.
        """
        if (
            not isinstance(values, torch.Tensor)
            or values.shape != self._low.shape
            or values.device != self._low.device
            or values.dtype != self._low.dtype
        ):
            raise ValueError("Coordinate samples must match the initial vector.")
        if (
            not isinstance(within_target, torch.Tensor)
            or within_target.shape != values.shape
            or within_target.dtype != torch.bool
            or within_target.device != values.device
        ):
            raise ValueError("within_target must be a matching boolean vector.")
        if type(elapsed_steps) is not int or elapsed_steps < 0:
            raise ValueError("elapsed_steps must be a nonnegative integer.")
        if elapsed_steps == self._last_step:
            return self._satisfied.clone()
        if elapsed_steps != self._last_step + 1:
            raise ValueError(
                "Continuous stability requires every consecutive interval."
            )
        good = within_target & torch.isfinite(values)
        low = torch.minimum(self._low, values)
        high = torch.maximum(self._high, values)
        reset = (
            ~good
            | (~self._previous_good & (elapsed_steps > 0))
            | ~torch.isfinite(self._low)
            | ((high - low) > self._drift)
        )
        self._low = torch.where(reset, values, low)
        self._high = torch.where(reset, values, high)
        self._counts = torch.where(reset, 0, self._counts + int(elapsed_steps > 0))
        self._satisfied = good & (self._counts >= self._required)
        self._previous_good = good
        self._last_step = elapsed_steps
        return self._satisfied.clone()

    @property
    def stable_counts(self) -> torch.Tensor:
        """Return independently owned per-row completed qualifying intervals."""
        return self._counts.clone()

    @property
    def minimum(self) -> torch.Tensor:
        """Return the current candidate-window coordinate minima."""
        return self._low.clone()

    @property
    def maximum(self) -> torch.Tensor:
        """Return the current candidate-window coordinate maxima."""
        return self._high.clone()
