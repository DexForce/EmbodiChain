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

"""Shared near-target evaluation and bounded control-step holds."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING

from embodichain.compute.geometry._distance import (
    _measure_near_target,
    _positions,
    _validate_near_target,
)

if TYPE_CHECKING:
    import torch

__all__ = [
    "NearTargetHoldSpec",
    "EvaluationStatus",
    "EvaluationResult",
    "EvaluationSession",
]


class EvaluationStatus(IntEnum):
    """Codes used in the per-row predicate and window status tensors."""

    NOT_RUN = 0
    UNKNOWN = 1
    UNAVAILABLE = 2
    FAILED = 3
    SATISFIED = 4


@dataclass(frozen=True, slots=True)
class NearTargetHoldSpec:
    """One bound near-target predicate and its bounded consecutive hold.

    Args:
        position_field: Raw observation field supplying the object's pose.
        target_field: Raw observation field supplying the target's pose.
        tolerance: Positive distance threshold in metres.
        hold_steps: Consecutive qualified control samples required.
        window_steps: Number of control slots in the inclusive search window.
        axes: Distinct translation axes to compare.
        inclusive: Whether a distance equal to the tolerance passes.
    """

    position_field: str
    target_field: str
    tolerance: float
    hold_steps: int
    window_steps: int
    axes: tuple[int, ...] = (0, 1, 2)
    inclusive: bool = True

    def __post_init__(self) -> None:
        fields = (self.position_field, self.target_field)
        if any(
            type(name) is not str or not name or name != name.strip() for name in fields
        ):
            raise ValueError("Evaluation fields must be nonempty observation IDs.")
        if self.position_field == self.target_field:
            raise ValueError(
                "Position and target must select distinct observation fields."
            )
        if (
            type(self.hold_steps) is not int
            or type(self.window_steps) is not int
            or not 0 < self.hold_steps <= self.window_steps
        ):
            raise ValueError("Hold steps must be positive and fit within the window.")
        axes = _validate_near_target(self.tolerance, self.axes, self.inclusive)
        object.__setattr__(self, "axes", axes)

    @property
    def required_fields(self) -> tuple[str, str]:
        """Return the two observation IDs in position, target order."""
        return self.position_field, self.target_field


@dataclass(frozen=True, slots=True, eq=False)
class EvaluationResult:
    """Owned results with physical truth separate from historical achievement.

    Status and distance tensors follow ``env_ids``. Unavailable distances are
    NaN. Witness tensors contain the start and end step or acquisition revision
    of the first qualified hold. A missing witness or revision is encoded as -1.
    """

    env_ids: tuple[int, ...]
    episode_ids: tuple[int, ...]
    predicate_status: torch.Tensor
    window_status: torch.Tensor
    distance: torch.Tensor
    tolerance: float
    step: int | None
    revisions: torch.Tensor
    witness_steps: torch.Tensor
    witness_revisions: torch.Tensor


class EvaluationSession:
    """Evaluate one window for a fixed cohort and its source episodes.

    Args:
        spec: Bound evaluation and window specification.
        env_ids: Original environment IDs in input row order.
        episode_ids: Expected episode identity for each selected row.
        start_step: First expected control slot.
        device: Device for state and qualified observations.

    One instance represents a single occurrence. To evaluate a new occurrence
    or episode, construct a new session. Providers qualify source ownership
    before submitting values. The evaluator retains a fixed number of tensors
    per row regardless of window length.
    """

    def __init__(
        self,
        spec: NearTargetHoldSpec,
        *,
        env_ids: Sequence[int],
        episode_ids: Sequence[int],
        start_step: int = 0,
        device: str | torch.device = "cpu",
    ) -> None:
        import torch

        self.spec = spec
        self.env_ids = tuple(env_ids)
        self.episode_ids = tuple(episode_ids)
        if (
            not self.env_ids
            or len(self.env_ids) != len(self.episode_ids)
            or any(
                type(value) is not int or value < 0
                for value in (*self.env_ids, *self.episode_ids)
            )
            or len(set(self.env_ids)) != len(self.env_ids)
        ):
            raise ValueError(
                "Expected unique environment IDs and one initialized episode per row."
            )
        if type(start_step) is not int or start_step < 0:
            raise ValueError("The first control slot must be a nonnegative integer.")
        self.start_step = start_step
        self.end_step = start_step + spec.window_steps - 1
        self._last_step: int | None = None
        self._observation_owner: object | None = None
        shape = (len(self.env_ids),)
        self._consecutive = torch.zeros(shape, device=device, dtype=torch.long)
        self._missing = torch.zeros_like(self._consecutive, dtype=torch.bool)
        self._run_start_revision = torch.full_like(self._consecutive, -1)
        self._max_revision = torch.full_like(self._consecutive, -1)
        self._revisions = torch.full_like(self._consecutive, -1)
        self._predicate = torch.full_like(self._consecutive, EvaluationStatus.NOT_RUN)
        self._distance = torch.full(shape, float("nan"), device=device)
        self._witness_steps = torch.full(
            (*shape, 2), -1, device=device, dtype=torch.long
        )
        self._witness_revisions = torch.full_like(self._witness_steps, -1)

    @property
    def device(self) -> torch.device:
        """Return the device that owns the evaluator state."""
        return self._consecutive.device

    def advance(
        self,
        step: int,
        position: torch.Tensor,
        target: torch.Tensor,
        *,
        valid: torch.Tensor,
        revisions: torch.Tensor,
    ) -> None:
        """Consume a qualified batch for one strictly increasing control slot.

        Args:
            step: Control slot shared by the batch. Skipped slots break holds.
            position: Observed object positions with shape (batch, 3).
            target: Observed target positions with shape (batch, 3).
            valid: Combined raw-field availability in cohort order.
            revisions: Acquisition revision per row, in the source clock domain.
        """
        import torch

        previous = self.start_step - 1 if self._last_step is None else self._last_step
        if type(step) is not int or not previous < step <= self.end_step:
            raise ValueError(
                "Control slots must increase within the evaluation window."
            )
        batch_shape = self._consecutive.shape
        _positions(position)
        _positions(target)
        if (
            position.shape != (*batch_shape, 3)
            or target.shape != position.shape
            or valid.shape != batch_shape
            or revisions.shape != batch_shape
            or valid.dtype != torch.bool
            or revisions.dtype != torch.long
            or any(
                tensor.device != self.device
                for tensor in (position, target, valid, revisions)
            )
        ):
            raise ValueError(
                "Observations and identities must match the evaluation batch and device."
            )

        satisfied, finite, distance = _measure_near_target(
            position, target, self.spec.tolerance, self.spec.axes, self.spec.inclusive
        )
        fresh = revisions > self._max_revision
        qualified = valid & finite & fresh
        if step != previous + 1:
            self._missing.fill_(True)
            self._consecutive.zero_()
        self._missing |= ~qualified
        passed = qualified & satisfied
        self._run_start_revision = torch.where(
            passed & (self._consecutive == 0), revisions, self._run_start_revision
        )
        self._consecutive = torch.where(
            passed, (self._consecutive + 1).clamp(max=self.spec.hold_steps), 0
        )
        reached = (self._consecutive == self.spec.hold_steps) & (
            self._witness_steps[:, 1] < 0
        )
        self._witness_steps[:, 0].masked_fill_(reached, step - self.spec.hold_steps + 1)
        self._witness_steps[:, 1].masked_fill_(reached, step)
        self._witness_revisions[:, 0] = torch.where(
            reached, self._run_start_revision, self._witness_revisions[:, 0]
        )
        self._witness_revisions[:, 1] = torch.where(
            reached, revisions, self._witness_revisions[:, 1]
        )
        self._max_revision = torch.maximum(self._max_revision, revisions)
        self._revisions = torch.where(fresh, revisions, -1)
        self._distance = torch.where(qualified, distance.detach(), float("nan"))
        self._predicate.fill_(EvaluationStatus.UNAVAILABLE)
        self._predicate.masked_fill_(qualified & ~satisfied, EvaluationStatus.FAILED)
        self._predicate.masked_fill_(passed, EvaluationStatus.SATISFIED)
        self._last_step = step

    def snapshot(self, *, closed: bool = False) -> EvaluationResult:
        """Read owned results without sampling or advancing time.

        Args:
            closed: Interpret the stream as ended even if some slots are missing.

        Returns:
            The latest predicate, historical hold status and first witness.
        """
        import torch

        if self._last_step == self.end_step:
            status = torch.full_like(self._predicate, EvaluationStatus.FAILED)
            status.masked_fill_(self._missing, EvaluationStatus.UNAVAILABLE)
        elif closed:
            status = torch.full_like(self._predicate, EvaluationStatus.UNAVAILABLE)
        else:
            pending = (
                EvaluationStatus.NOT_RUN
                if self._last_step is None
                else EvaluationStatus.UNKNOWN
            )
            status = torch.full_like(self._predicate, pending)
        status.masked_fill_(self._witness_steps[:, 1] >= 0, EvaluationStatus.SATISFIED)
        return EvaluationResult(
            self.env_ids,
            self.episode_ids,
            self._predicate.clone(),
            status,
            self._distance.clone(),
            self.spec.tolerance,
            self._last_step,
            self._revisions.clone(),
            self._witness_steps.clone(),
            self._witness_revisions.clone(),
        )
