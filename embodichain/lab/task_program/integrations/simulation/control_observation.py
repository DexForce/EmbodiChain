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

"""Guarded physical contact-window adaptation for articulation control."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, asdict, replace
import math
from typing import Any

import torch

from embodichain.lab.sim.evidence import PhysicalEvidenceFrame
from embodichain.lab.sim.atomic_actions.articulation_evidence import JointContactSample
from embodichain.lab.sim.sensors.contact.sampling import (
    CONTACT_ACTOR_IDS,
    CONTACT_VALID_MASK,
    CONTACT_NORMALS,
    CONTACT_IMPULSES,
    CONTACT_POSITIONS,
    CONTACT_DISTANCES,
)
from embodichain.lab.sim.sensors.contact.observation import (
    JointContactObservation,
    JointContactObserver,
)

__all__ = [
    "ControlContactActors",
    "ControlObservationSnapshot",
    "ControlObservationProjection",
    "ControlObservationWindow",
    "ControlPolicyObservation",
]


@dataclass(frozen=True, slots=True)
class ControlContactActors:
    """Exact native target, parent and full hand actor roster for one environment."""

    target: int
    parent: int
    hand: tuple[int, ...]

    def __post_init__(self) -> None:
        hand = tuple(self.hand)
        if not hand or any(
            type(value) is not int or value < 0
            for value in (self.target, self.parent, *hand)
        ):
            raise ValueError("Control contacts require explicit nonnegative actor IDs.")
        if (
            self.target == self.parent
            or self.target in hand
            or self.parent in hand
            or len(set(hand)) != len(hand)
        ):
            raise ValueError(
                "Control actor roles must be distinct and the hand roster unique."
            )
        object.__setattr__(self, "hand", hand)


@dataclass(frozen=True, slots=True)
class ControlObservationProjection:
    """Owned scalar projection for repeated effect checks, without raw-history copies."""

    generation: int
    legacy_time_origin: float
    physics_dt: float
    samples: tuple[JointContactSample, ...]
    parent_contacts: tuple[bool | None, ...]
    errors: tuple[str, ...]

    @property
    def available(self) -> bool:
        """Whether the complete current window has valid scalar measurements."""
        return (
            bool(self.samples)
            and not self.errors
            and all(sample.valid for sample in self.samples)
        )


@dataclass(frozen=True, slots=True)
class ControlObservationSnapshot:
    """Owned complete raw records and the source-compatible legacy projection.

    Raw frame timestamps remain simulation time. The explicit legacy origin only
    maps projected sample timestamps; it never retags or fabricates acquisitions.
    Window errors retain incomplete cadence/capacity/scope qualification separately
    from per-channel validity. This value asserts no control or task success.
    """

    generation: int
    legacy_time_origin: float
    physics_dt: float
    records: tuple[JointContactObservation, ...]
    samples: tuple[JointContactSample, ...]
    trace: tuple[dict[str, Any], ...]
    errors: tuple[str, ...]

    @property
    def available(self) -> bool:
        """Whether a current complete window has valid projected measurements."""
        return (
            bool(self.records)
            and not self.errors
            and all(sample.valid for sample in self.samples)
        )


@dataclass(frozen=True, slots=True)
class ControlPolicyObservation:
    """Owned same-policy physical reads and exact source window identity.

    Frame timestamps are simulation time; their revisions identify policy reads,
    not native contact refreshes. No acceptance or non-target-unchanged verdict
    is stored. The host retains one snapshot per actual compiled occurrence.
    """

    phase: str
    terminal: bool
    control_dt: float
    joint_index: int
    generation: int
    legacy_time_origin: float
    physics_dt: float
    window: tuple[float, float] | None
    sample_count: int
    hold_start_step: int
    hold_end_step: int
    hold_frames: tuple[PhysicalEvidenceFrame, ...]
    joint_limits: PhysicalEvidenceFrame | None = None
    reference_qpos: PhysicalEvidenceFrame | None = None
    current_qpos: PhysicalEvidenceFrame | None = None
    clearance: PhysicalEvidenceFrame | None = None

    def __post_init__(self) -> None:
        if (
            type(self.phase) is not str
            or not self.phase
            or type(self.terminal) is not bool
        ):
            raise ValueError(
                "Control observations require exact policy phase metadata."
            )
        for name, minimum in (
            ("joint_index", 0),
            ("generation", 1),
            ("sample_count", 0),
            ("hold_start_step", 0),
            ("hold_end_step", 0),
        ):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError("Control observation indices are invalid.")
        if (
            self.hold_end_step < self.hold_start_step
            or any(
                type(value) not in (int, float) or not math.isfinite(value) or value < 0
                for value in (
                    self.control_dt,
                    self.physics_dt,
                    self.legacy_time_origin,
                    *(self.window or ()),
                )
            )
            or self.control_dt == 0
            or self.physics_dt == 0
            or (self.window is None) != (self.sample_count == 0)
            or self.window is not None
            and (len(self.window) != 2 or self.window[0] > self.window[1])
        ):
            raise ValueError("Control observation clocks/window are invalid.")
        object.__setattr__(
            self, "hold_frames", tuple(frame.snapshot() for frame in self.hold_frames)
        )
        for name in ("joint_limits", "reference_qpos", "current_qpos", "clearance"):
            frame = getattr(self, name)
            if frame is not None:
                object.__setattr__(self, name, frame.snapshot())

    def snapshot(self) -> ControlPolicyObservation:
        """Copy all physical channels without reading current native state."""
        return replace(self)


def _project(
    record: JointContactObservation,
    actors: ControlContactActors,
    origin: float,
) -> tuple[JointContactSample, dict[str, Any], tuple[str, ...]]:
    # This private adapter receives a fresh owned delta from snapshot(start).
    # No external references can mutate it between acquisition and projection.
    frame = record.frame
    if frame.env_ids.tolist() != [0]:
        raise ValueError("Control contact windows support native B=1/env0 only.")
    errors = []
    channels = (
        CONTACT_VALID_MASK,
        CONTACT_ACTOR_IDS,
        CONTACT_IMPULSES,
        CONTACT_POSITIONS,
        CONTACT_NORMALS,
        CONTACT_DISTANCES,
        "articulation.joint_position",
    )
    for channel in channels:
        batch = frame.batches.get(channel)
        if batch is None:
            errors.append(f"Missing control channel: {channel}.")
        elif not bool(batch.valid[0]):
            errors.append(
                batch.acquisition_errors[0]
                or f"Unavailable control channel: {channel}."
            )
    pairs, impulses, qpos = [], [], float("nan")
    try:
        values = {channel: frame.batches[channel].values for channel in channels}
        mask = values[CONTACT_VALID_MASK]
        if mask.dtype != torch.bool or mask.ndim != 2 or mask.shape[0] != 1:
            raise ValueError("Control contact mask must have shape (1, K).")
        count = mask.shape[1]
        shapes = {
            CONTACT_ACTOR_IDS: (1, count, 2),
            CONTACT_IMPULSES: (1, count),
            CONTACT_POSITIONS: (1, count, 3),
            CONTACT_NORMALS: (1, count, 3),
            CONTACT_DISTANCES: (1, count),
            "articulation.joint_position": (1,),
        }
        if any(values[channel].shape != shape for channel, shape in shapes.items()):
            raise ValueError("Control channels differ from their declared B=1 shape.")
        if values[CONTACT_ACTOR_IDS].dtype not in (torch.int32, torch.int64):
            raise ValueError("Control actor pairs must contain integer identities.")
        if any(
            not values[channel].is_floating_point()
            for channel in shapes
            if channel != CONTACT_ACTOR_IDS
        ):
            raise ValueError("Control numeric channels must be floating tensors.")
        pairs = values[CONTACT_ACTOR_IDS][0, mask[0]].tolist()
        impulses = values[CONTACT_IMPULSES][0, mask[0]].tolist()
        qpos = float(values["articulation.joint_position"][0])
        if any(
            not bool(torch.isfinite(values[channel][0, mask[0]]).all())
            for channel in (
                CONTACT_IMPULSES,
                CONTACT_POSITIONS,
                CONTACT_NORMALS,
                CONTACT_DISTANCES,
            )
        ) or not math.isfinite(qpos):
            errors.append("Nonfinite active control observation.")
    except (KeyError, TypeError, ValueError, IndexError) as error:
        errors.append(str(error) or type(error).__name__)
    if record.dropped_contacts:
        errors.append("Control contact capacity was exceeded.")
    timestamp = frame.timestamp - origin
    if timestamp < 0:
        errors.append("Legacy time origin follows the actual acquisition.")
    valid = not errors

    def contact(target: int) -> bool:
        return any(
            target in pair
            and any(actor in actors.hand for actor in pair)
            and impulse > 0
            for pair, impulse in zip(pairs, impulses)
        )

    sample = JointContactSample(
        timestamp, qpos, record.phase, contact(actors.target) if valid else None, valid
    )
    trace = {
        **asdict(sample),
        "contact_pairs": pairs,
        "impulses": impulses,
        "dropped_contacts": record.dropped_contacts,
        "parent_contact": contact(actors.parent) if valid else None,
    }
    return sample, trace, tuple(dict.fromkeys(errors))


class ControlObservationWindow:
    """Bind a sensor-owned generation and explicit actor roles to a runtime owner.

    Args:
        observer: Standard joint/contact observer registered in the sensor suite.
        generation: Exact generation returned by its begin_window call.
        actors: Explicit native actor binding from the registered physical entities.
        legacy_time_origin: Actual simulation-clock origin used by the source
            projected trace. Raw frames remain in their actual clock domain.
        physics_dt: Declared fixed physics-substep cadence from the environment.
        current_time: Same simulation clock used by the observer.
        validate_scope: Guard rejecting replaced invocation/episode ownership.
    """

    def __init__(
        self,
        observer: JointContactObserver,
        generation: int,
        actors: ControlContactActors,
        *,
        legacy_time_origin: float,
        physics_dt: float,
        current_time: Callable[[], float],
        validate_scope: Callable[[], None],
    ) -> None:
        if (
            type(generation) is not int
            or generation < 1
            or type(actors) is not ControlContactActors
        ):
            raise ValueError(
                "Control window requires exact generation and actor roles."
            )
        if (
            isinstance(legacy_time_origin, bool)
            or not math.isfinite(legacy_time_origin)
            or legacy_time_origin < 0
        ):
            raise ValueError("Legacy time origin must be finite and nonnegative.")
        if (
            isinstance(physics_dt, bool)
            or not math.isfinite(physics_dt)
            or physics_dt <= 0
        ):
            raise ValueError("Control observation requires a positive physics cadence.")
        self._physics_dt = float(physics_dt)
        if not callable(current_time) or not callable(validate_scope):
            raise TypeError(
                "Control observation requires explicit clock and owner guards."
            )
        self._observer, self._generation, self._actors = observer, generation, actors
        self._origin, self._time, self._guard = (
            float(legacy_time_origin),
            current_time,
            validate_scope,
        )
        self._seen = 0
        self._last_timestamp: float | None = None
        self._samples: list[JointContactSample] = []
        self._trace: list[dict[str, Any]] = []
        self._parents: list[bool | None] = []
        self._errors: list[str] = []

    @property
    def generation(self) -> int:
        """Return the sensor generation bound at the ready/startup boundary."""
        return self._generation

    @property
    def physics_dt(self) -> float:
        """Return the declared fixed substep cadence used to qualify this window."""
        return self._physics_dt

    def _require_generation(self) -> None:
        self._guard()
        if self._observer.generation != self._generation:
            raise ValueError("Control observation window was reset or replaced.")

    def _refresh(self) -> None:
        """Project only newly appended owned records in this exact generation."""
        self._require_generation()
        count = self._observer.record_count
        if count < self._seen:
            raise ValueError("Control observation lost an earlier window prefix.")
        records = self._observer.snapshot(self._seen)
        if len(records) != count - self._seen:
            raise ValueError("Control observation changed during its delta read.")
        for record in records:
            if self._last_timestamp is not None and not math.isclose(
                record.frame.timestamp - self._last_timestamp,
                self._physics_dt,
                rel_tol=0,
                abs_tol=1e-8,
            ):
                self._errors.append(
                    "Control observation window has missing or duplicated physics substeps."
                )
            sample, trace, missing = _project(record, self._actors, self._origin)
            self._samples.append(sample)
            self._trace.append(trace)
            self._parents.append(trace["parent_contact"])
            self._errors.extend(missing)
            self._last_timestamp = record.frame.timestamp
        self._seen = count
        self._errors = list(dict.fromkeys(self._errors))
        self._require_generation()
        if self._observer.record_count != self._seen:
            raise ValueError("Control observation changed during scalar projection.")

    def projection(self) -> ControlObservationProjection:
        """Return current scalar facts, processing only new append-only acquisitions.

        Ownership is the observer generation, not tensor/object identity. Reset
        invalidates the reader; earlier records cannot be rewritten within a
        generation. Returned samples and parent flags are immutable scalar data.
        Raw frames are copied only when :meth:`snapshot` is explicitly requested.
        Calls must be serialized with the owning sensor/runtime lifecycle.
        """
        self._refresh()
        timestamp = self._time()
        errors = list(self._errors)
        if isinstance(timestamp, bool) or not math.isfinite(timestamp) or timestamp < 0:
            errors.append("Current simulation acquisition clock is unavailable.")
        if self._observer.overflow:
            errors.append("Control observation window exceeded its declared capacity.")
        if self._last_timestamp is None:
            errors.append("Control observation window has no acquisitions.")
        elif not math.isclose(self._last_timestamp, timestamp, rel_tol=0, abs_tol=1e-8):
            errors.append(
                "Control observation does not reach the current physics boundary."
            )
        self._require_generation()
        return ControlObservationProjection(
            self._generation,
            self._origin,
            self._physics_dt,
            tuple(self._samples),
            tuple(self._parents),
            tuple(dict.fromkeys(errors)),
        )

    def snapshot(self) -> ControlObservationSnapshot:
        """Copy the complete raw window for an explicit artifact/receipt boundary."""
        projection = self.projection()
        records = self._observer.snapshot()
        self._require_generation()
        if len(records) != self._seen:
            raise ValueError("Control observation changed during raw snapshot.")
        return ControlObservationSnapshot(
            self._generation,
            self._origin,
            self._physics_dt,
            records,
            projection.samples,
            tuple(deepcopy(self._trace)),
            projection.errors,
        )
