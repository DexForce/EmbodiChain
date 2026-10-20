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

"""Versioned physical-evidence ports for closed-loop Atomic Action execution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import torch

if TYPE_CHECKING:
    from .state import PlanningContext

__all__ = [
    "PhysicalEvidenceBatch",
    "PhysicalEvidenceFrame",
    "PhysicalEvidenceProvider",
    "PhysicalEvidenceRequest",
]


def _validate_env_ids(
    env_ids: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
    name: str,
) -> torch.Tensor:
    """Validate and own an ordered environment-id vector."""
    if not isinstance(env_ids, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if env_ids.dtype != torch.long or env_ids.dim() != 1:
        raise ValueError(f"{name} must be a one-dimensional int64 tensor.")
    if env_ids.shape != (batch_size,):
        raise ValueError(
            f"{name} must have shape ({batch_size},), got {tuple(env_ids.shape)}."
        )
    if env_ids.device != device:
        raise ValueError(f"{name} must use device {device}, got {env_ids.device}.")
    if torch.unique(env_ids).numel() != batch_size:
        raise ValueError(f"{name} must contain unique environment IDs.")
    return env_ids.clone()


def _validate_timestamp(value: float, *, name: str) -> float:
    """Normalize a non-negative finite timestamp."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric.")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return normalized


@dataclass(frozen=True, slots=True, eq=False)
class PhysicalEvidenceRequest:
    """One synchronized request for runtime physical evidence."""

    timestamp: float
    observation_revision: int
    env_ids: torch.Tensor
    channels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        timestamp = _validate_timestamp(self.timestamp, name="timestamp")
        if type(self.observation_revision) is not int or self.observation_revision < 0:
            raise ValueError("observation_revision must be a non-negative integer.")
        if not isinstance(self.env_ids, torch.Tensor):
            raise TypeError("env_ids must be a torch.Tensor.")
        if self.env_ids.numel() == 0:
            raise ValueError("env_ids must contain at least one environment row.")
        env_ids = _validate_env_ids(
            self.env_ids,
            device=self.env_ids.device,
            batch_size=int(self.env_ids.numel()),
            name="env_ids",
        )
        channels = tuple(self.channels)
        if len(channels) != len(set(channels)) or not all(
            isinstance(channel, str) and channel.strip() for channel in channels
        ):
            raise ValueError("channels must contain unique non-empty strings.")
        object.__setattr__(self, "timestamp", timestamp)
        object.__setattr__(self, "env_ids", env_ids)
        object.__setattr__(self, "channels", channels)

    def snapshot(self) -> "PhysicalEvidenceRequest":
        """Return an independently owned request snapshot."""
        return PhysicalEvidenceRequest(
            timestamp=self.timestamp,
            observation_revision=self.observation_revision,
            env_ids=self.env_ids,
            channels=self.channels,
        )


@dataclass(frozen=True, slots=True, eq=False)
class PhysicalEvidenceBatch:
    """One typed-by-convention physical channel for an environment batch.

    ``values`` keeps the channel-specific trailing shape. The first dimension
    always follows ``env_ids``. ``valid`` and ``acquisition_errors`` distinguish
    unavailable observations from measured negative evidence.
    """

    evidence_id: str
    values: torch.Tensor
    valid: torch.Tensor
    acquisition_errors: tuple[str | None, ...]
    timestamp: float
    env_ids: torch.Tensor
    observation_revision: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.evidence_id, str)
            or not self.evidence_id
            or self.evidence_id != self.evidence_id.strip()
        ):
            raise ValueError("evidence_id must be a non-empty string.")
        if not isinstance(self.values, torch.Tensor) or self.values.dim() == 0:
            raise ValueError("values must be a tensor with a batch dimension.")
        if self.values.shape[0] == 0:
            raise ValueError("values must contain at least one environment row.")
        if self.values.device != self.valid.device:
            raise ValueError("values and valid must share a device.")
        if self.valid.dtype != torch.bool or self.valid.shape != self.values.shape[:1]:
            raise ValueError("valid must be a bool vector matching values batch size.")
        errors = tuple(self.acquisition_errors)
        if len(errors) != self.values.shape[0] or any(
            error is not None and (not isinstance(error, str) or not error.strip())
            for error in errors
        ):
            raise ValueError(
                "acquisition_errors must contain one optional message per row."
            )
        if (
            self.values.is_floating_point()
            and not torch.isfinite(self.values[self.valid]).all()
        ):
            raise ValueError("Valid floating evidence values must be finite.")
        timestamp = _validate_timestamp(self.timestamp, name="timestamp")
        if type(self.observation_revision) is not int or self.observation_revision < 0:
            raise ValueError("observation_revision must be a non-negative integer.")
        if not isinstance(self.env_ids, torch.Tensor):
            raise TypeError("env_ids must be a torch.Tensor.")
        if self.env_ids.numel() == 0:
            raise ValueError("env_ids must contain at least one environment row.")
        env_ids = _validate_env_ids(
            self.env_ids,
            device=self.values.device,
            batch_size=self.values.shape[0],
            name="env_ids",
        )
        object.__setattr__(self, "values", self.values.clone())
        object.__setattr__(self, "valid", self.valid.clone())
        object.__setattr__(self, "acquisition_errors", errors)
        object.__setattr__(self, "timestamp", timestamp)
        object.__setattr__(self, "env_ids", env_ids)

    def snapshot(self) -> "PhysicalEvidenceBatch":
        """Return an independently owned evidence batch."""
        return PhysicalEvidenceBatch(
            evidence_id=self.evidence_id,
            values=self.values,
            valid=self.valid,
            acquisition_errors=self.acquisition_errors,
            timestamp=self.timestamp,
            env_ids=self.env_ids,
            observation_revision=self.observation_revision,
        )


@dataclass(frozen=True, slots=True, eq=False)
class PhysicalEvidenceFrame:
    """Synchronized evidence channels from one execution observation tick."""

    timestamp: float
    observation_revision: int
    env_ids: torch.Tensor
    batches: Mapping[str, PhysicalEvidenceBatch] = field(default_factory=dict)

    def __post_init__(self) -> None:
        timestamp = _validate_timestamp(self.timestamp, name="timestamp")
        if type(self.observation_revision) is not int or self.observation_revision < 0:
            raise ValueError("observation_revision must be a non-negative integer.")
        if not isinstance(self.env_ids, torch.Tensor):
            raise TypeError("env_ids must be a torch.Tensor.")
        if not isinstance(self.batches, Mapping):
            raise TypeError("batches must be a mapping of evidence IDs to batches.")
        if any(
            not isinstance(batch, PhysicalEvidenceBatch)
            for batch in self.batches.values()
        ):
            raise TypeError("batches must contain PhysicalEvidenceBatch values.")
        if self.env_ids.numel() == 0:
            raise ValueError("env_ids must contain at least one environment row.")
        env_ids = _validate_env_ids(
            self.env_ids,
            device=self.env_ids.device,
            batch_size=int(self.env_ids.numel()),
            name="env_ids",
        )
        normalized: dict[str, PhysicalEvidenceBatch] = {}
        for evidence_id, batch in self.batches.items():
            if evidence_id != batch.evidence_id:
                raise ValueError("Evidence mapping keys must match batch IDs.")
            if batch.timestamp != timestamp:
                raise ValueError("All evidence batches must share frame timestamp.")
            if batch.observation_revision != self.observation_revision:
                raise ValueError(
                    "All evidence batches must share frame observation revision."
                )
            if batch.env_ids.device != env_ids.device or not torch.equal(
                batch.env_ids, env_ids
            ):
                raise ValueError("All evidence batches must share frame env_ids.")
            normalized[evidence_id] = batch.snapshot()
        object.__setattr__(self, "timestamp", timestamp)
        object.__setattr__(self, "env_ids", env_ids)
        object.__setattr__(self, "batches", MappingProxyType(normalized))

    @property
    def batch_size(self) -> int:
        """Return the number of environment rows in the frame."""
        return int(self.env_ids.numel())

    def get(self, evidence_id: str) -> PhysicalEvidenceBatch:
        """Return one channel or raise a descriptive ``KeyError``."""
        try:
            return self.batches[evidence_id].snapshot()
        except KeyError as exc:
            raise KeyError(
                f"Unknown physical evidence {evidence_id!r}; available: "
                f"{sorted(self.batches)}."
            ) from exc

    def snapshot(self) -> "PhysicalEvidenceFrame":
        """Return an independently owned frame snapshot."""
        return PhysicalEvidenceFrame(
            timestamp=self.timestamp,
            observation_revision=self.observation_revision,
            env_ids=self.env_ids,
            batches=self.batches,
        )


@runtime_checkable
class PhysicalEvidenceProvider(Protocol):
    """Acquire synchronized physical evidence for an execution observation."""

    def collect(
        self,
        context: "PlanningContext",
        request: PhysicalEvidenceRequest,
    ) -> PhysicalEvidenceFrame:
        """Return evidence aligned with the supplied planning context."""
        ...
