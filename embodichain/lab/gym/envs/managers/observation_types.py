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

"""Raw observation channels and explicit acquisition boundaries.

Policy tensors remain ordinary observations. Only described original channels
are eligible for physical recording; their values precede policy transforms.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, TYPE_CHECKING
import math

if TYPE_CHECKING:
    from ..episode_context import EpisodeContext

import torch

__all__ = [
    "ObservationDescriptor",
    "ObservationSample",
    "ObservationFrame",
    "ObservationHistorySnapshot",
]


@dataclass(frozen=True, slots=True)
class ObservationDescriptor:
    """Semantic identity of an untransformed physical channel.

    Args:
        entity_uid: Resolved native scene entity identifier.
        quantity: Measured quantity, such as pose, qpos or qvel.
        coordinate_frame: Explicit reference frame, e.g. arena or joint.
        representation: Value convention, e.g. xyz_xyzw or joint_order.
        units: Units in representation order.
        shape: Per-environment value shape.
        dtype: Original tensor dtype.
        joint_names: Ordered joint identities for joint-space channels.
    """

    entity_uid: str
    quantity: str
    coordinate_frame: str
    representation: str
    units: tuple[str, ...]
    shape: tuple[int, ...]
    dtype: torch.dtype
    joint_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("entity_uid", "quantity", "coordinate_frame", "representation"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or value != value.strip():
                raise ValueError(f"{name} must be an exact nonempty identifier.")
        for name in ("units", "shape", "joint_names"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if not isinstance(self.dtype, torch.dtype) or any(
            type(size) is not int or size < 0 for size in self.shape
        ):
            raise ValueError("Observation shape and dtype must be explicit.")


@dataclass(frozen=True, slots=True, eq=False)
class ObservationSample:
    """A described raw value with row validity and missing-acquisition reasons.

    Args:
        descriptor: Physical meaning and original representation.
        values: Tensor with a leading environment axis.
        valid: Boolean acquisition mask, one entry per environment.
        acquisition_errors: Missing-data reasons; None for valid rows.

    Producers own their tensors. Acquisition boundaries copy a sample once;
    consumers must not modify it. Policy views receive independent tensors.
    Expensive physical validation belongs at the evidence conversion boundary.
    """

    descriptor: ObservationDescriptor
    values: torch.Tensor
    valid: torch.Tensor
    acquisition_errors: tuple[str | None, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.descriptor, ObservationDescriptor):
            raise TypeError("Raw observations require an ObservationDescriptor.")
        if (
            not isinstance(self.values, torch.Tensor)
            or self.values.ndim < 1
            or tuple(self.values.shape[1:]) != self.descriptor.shape
            or self.values.dtype != self.descriptor.dtype
        ):
            raise ValueError("Observation values differ from their descriptor.")
        size = self.values.shape[0]
        if (
            not isinstance(self.valid, torch.Tensor)
            or self.valid.shape != (size,)
            or self.valid.dtype != torch.bool
            or self.valid.device != self.values.device
        ):
            raise ValueError("Observation validity must align with the value rows.")
        errors = tuple(self.acquisition_errors)
        if len(errors) != size or any(
            error is not None and (not isinstance(error, str) or not error)
            for error in errors
        ):
            raise ValueError("Acquisition errors must align with the value rows.")
        object.__setattr__(self, "acquisition_errors", errors)


@dataclass(frozen=True, slots=True, eq=False)
class ObservationFrame:
    """One explicit observation boundary, independent of history advancement.

    Args:
        policy: Actual policy observation, including the sampled noise.
        fields: Selected raw recording/evidence fields before policy transforms.
        episode_ids: Row-local host episode identities at acquisition.
        revision: Unique acquisition identity; repeated queries in one control
            tick are separate revisions.
        timestamp: Source simulation time, or None when unavailable.
        clock_domain: Clock identity; never inferred from a trajectory slot.
        episode_owner: Exact lifecycle owner at acquisition, preventing foreign
            captures with coincidentally equal numeric episode IDs.
    """

    policy: Any
    fields: Mapping[str, ObservationSample]
    episode_ids: tuple[int, ...]
    revision: int
    timestamp: float | None = None
    clock_domain: str = "unavailable"
    episode_owner: EpisodeContext | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "fields", MappingProxyType(dict(self.fields)))
        object.__setattr__(self, "episode_ids", tuple(self.episode_ids))
        if type(self.revision) is not int:
            raise ValueError("An observation revision must be an integer.")
        if self.timestamp is not None and (
            isinstance(self.timestamp, bool)
            or not isinstance(self.timestamp, (int, float))
            or not math.isfinite(self.timestamp)
        ):
            raise ValueError("An available observation timestamp must be finite.")


@dataclass(frozen=True, slots=True, eq=False)
class ObservationHistorySnapshot:
    """Owned values selected by field ID and exact temporal row identity.

    Args:
        values: Selected field tensors, without a second semantic schema.
        valid: Per-field row availability.
        acquisition_errors: Per-field missing-acquisition reasons.
        env_ids: Original environment rows in query order.
        episode_ids: Source episode identities.
        frame_indices: Exact row-local state slots.
        revisions: Original acquisition revisions.
        timestamps: Original simulation times, or None when unavailable.
        clock_domain: Explicit source clock domain.
    """

    values: Mapping[str, torch.Tensor]
    valid: Mapping[str, torch.Tensor]
    acquisition_errors: Mapping[str, tuple[str | None, ...]]
    env_ids: tuple[int, ...]
    episode_ids: tuple[int, ...]
    frame_indices: tuple[int, ...]
    revisions: tuple[int, ...]
    timestamps: tuple[float | None, ...]
    clock_domain: str = "simulation_time"
