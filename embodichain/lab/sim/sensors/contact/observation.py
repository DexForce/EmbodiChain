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

"""Joint and contact facts observed through the standard sensor lifecycle."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
import math

import torch

from embodichain.lab.sim.evidence import (
    PhysicalEvidenceBatch,
    PhysicalEvidenceFrame,
    PhysicalEvidenceRequest,
)
from .sampling import (
    CONTACT_DISTANCES,
    CONTACT_VALID_MASK,
    DEFAULT_CHANNELS,
    ContactEvidenceSampler,
    ContactSourceStamp,
)

__all__ = ["JointContactObservation", "JointContactObserver"]


@dataclass(frozen=True, slots=True)
class JointContactObservation:
    """Owned joint/contact acquisition, without an action or task verdict.

    Frame timestamps use simulation time; observation revisions count this
    observer's acquisitions. Native contact revision remains independent.
    """

    frame: PhysicalEvidenceFrame
    phase: str
    native_revision: int | None
    topology_revision: int | None
    dropped_contacts: int

    def snapshot(self) -> JointContactObservation:
        """Return detached evidence tensors with the same acquisition identity."""
        return replace(self, frame=self.frame.snapshot())


class JointContactObserver:
    """Collect explicit B=1 joint channels alongside sampled native contacts.

    ContactSensor drives substep callbacks from its completed sample. Windows
    own phase labels and storage limits; integrations own episode/occurrence
    binding and verdicts. No callback advances simulation or fetches again.

    Args:
        sampler: Existing standard contact source, with exactly one environment.
        joint_position: Read the explicitly selected joint as a floating (1,) tensor.
        clock: Actual nonnegative simulation time, not a synthetic sample clock.
        joint_velocity: Optional independent (1,) joint-velocity reader.
    """

    def __init__(
        self,
        sampler: ContactEvidenceSampler,
        joint_position: Callable[[], torch.Tensor],
        clock: Callable[[], float],
        *,
        joint_velocity: Callable[[], torch.Tensor] | None = None,
    ) -> None:
        if sampler.num_envs != 1:
            raise ValueError("Joint contact observation supports B=1 only.")
        self._sampler, self._clock = sampler, clock
        self._joints = {"articulation.joint_position": joint_position}
        if joint_velocity is not None:
            self._joints["articulation.joint_velocity"] = joint_velocity
        self._generation = 0
        self._revision = 0
        self._closed = False
        self._active = False
        self._records: list[JointContactObservation] = []
        self._phase = ""
        self._started_at = 0.0
        self._max_samples: int | None = None
        self._max_duration: float | None = None
        self._overflow = False

    @property
    def generation(self) -> int:
        """Observer-local window generation, advanced on begin/reset/close."""
        return self._generation

    @property
    def record_count(self) -> int:
        """Return the current window's retained count without copying or sampling."""
        return len(self._records)

    @property
    def overflow(self) -> bool:
        """Whether an explicit window limit prevented a complete acquisition."""
        return self._overflow

    def begin_window(
        self,
        phase: str,
        *,
        max_samples: int | None = None,
        max_duration: float | None = None,
    ) -> int:
        """Start a fresh owned window without sampling or changing simulation.

        Returns:
            Generation for the integration's existing occurrence binding. A new
            window discards earlier live records; snapshots remain detached.
        """
        if self._closed:
            raise RuntimeError("Joint contact observer is closed.")
        if max_samples is not None and (
            type(max_samples) is not int or max_samples < 1
        ):
            raise ValueError("max_samples must be a positive integer.")
        if max_duration is not None and (
            isinstance(max_duration, bool)
            or not math.isfinite(max_duration)
            or max_duration <= 0
        ):
            raise ValueError("max_duration must be finite and positive.")
        self.set_phase(phase)
        started = self._time()
        self.reset()
        self._max_samples, self._max_duration = max_samples, max_duration
        self._started_at, self._active = started, True
        return self._generation

    def set_phase(self, phase: str) -> None:
        """Set an explicit descriptive phase; no behavior is selected by its name."""
        if type(phase) is not str or not phase or phase != phase.strip():
            raise ValueError("Observation phase must be an exact nonempty string.")
        self._phase = phase

    def _time(self) -> float:
        value = self._clock()
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError(
                "Joint observation requires actual nonnegative simulation time."
            )
        return float(value)

    def _joint_batch(
        self,
        reader: Callable[[], torch.Tensor],
        name: str,
        request: PhysicalEvidenceRequest,
    ) -> PhysicalEvidenceBatch:
        error = None
        try:
            raw = reader()
            if (
                not isinstance(raw, torch.Tensor)
                or raw.shape != (1,)
                or not raw.is_floating_point()
            ):
                raise ValueError("Joint readers must return a floating B=1 vector.")
            value = raw.detach().to(request.env_ids.device).clone()
            valid = torch.isfinite(value)
            if not bool(valid.all()):
                error = "Nonfinite joint observation."
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError) as exc:
            value = torch.full((1,), float("nan"), device=request.env_ids.device)
            valid = torch.zeros(1, dtype=torch.bool, device=request.env_ids.device)
            error = str(exc) or type(exc).__name__
        return PhysicalEvidenceBatch(
            name,
            value,
            valid,
            (error,),
            request.timestamp,
            request.env_ids,
            request.observation_revision,
        )

    def capture(
        self,
        *,
        refresh: bool,
        source_stamp: ContactSourceStamp | None = None,
    ) -> None:
        """Append one joint/contact sample to the active window.

        Args:
            refresh: Explicitly refresh the sensor for a prepare sample. Sensor
                callbacks reuse the completed substep with ``False``.
            source_stamp: Before-sample stamp supplied by the sensor callback.

        Read stored data with :meth:`snapshot`. A reached limit marks overflow
        without acquiring another sample. Missing channels remain unavailable.
        """
        if self._closed or not self._active:
            raise RuntimeError("Joint contact observation has no active window.")
        generation, phase = self._generation, self._phase
        timestamp = self._time()
        if (
            timestamp < self._started_at
            or self._records
            and timestamp < self._records[-1].frame.timestamp
        ):
            raise ValueError(
                "Simulation time moved backwards in an observation window."
            )
        if (
            self._max_samples is not None
            and len(self._records) >= self._max_samples
            or self._max_duration is not None
            and timestamp - self._started_at > self._max_duration + 1e-8
        ):
            self._overflow = True
            return
        self._revision += 1
        request = PhysicalEvidenceRequest(
            timestamp,
            self._revision,
            torch.tensor([0]),
            (*DEFAULT_CHANNELS, CONTACT_DISTANCES),
        )
        sample = self._sampler.sample(
            request, refresh=refresh, source_stamp=source_stamp
        )
        # Retain valid contacts rather than fixed-capacity padding per substep.
        mask = sample.frame.batches[CONTACT_VALID_MASK].values[0]
        batches = {
            name: replace(batch, values=batch.values[:, mask])
            for name, batch in sample.frame.batches.items()
        }
        batches.update(
            (name, self._joint_batch(reader, name, request))
            for name, reader in self._joints.items()
        )
        if self._closed or not self._active or generation != self._generation:
            raise ValueError("Observation window changed during acquisition.")
        if self._time() != timestamp:
            raise ValueError(
                "Simulation time changed during joint/contact acquisition."
            )
        record = JointContactObservation(
            PhysicalEvidenceFrame._from_owned_batches(
                timestamp, self._revision, request.env_ids, batches
            ),
            phase,
            sample.native_revision,
            sample.topology_revision,
            sample.dropped_contacts,
        )
        self._records.append(record)

    def snapshot(self, start: int = 0) -> tuple[JointContactObservation, ...]:
        """Copy all records or only the suffix beginning at an earlier offset.

        Args:
            start: Record count already consumed in the current generation.
                Callers check :attr:`generation` before reusing an offset.

        Returns:
            Owned observations; later samples and resets cannot modify them.

        Raises:
            ValueError: The offset does not address the current window prefix.
        """
        if type(start) is not int or not 0 <= start <= len(self._records):
            raise ValueError(
                "Observation offset must address the current window prefix."
            )
        return tuple(record.snapshot() for record in self._records[start:])

    def begin_control_step(self) -> None:
        """Keep the window across control steps; the sensor owns interval counters."""

    def before_sample(self) -> object:
        """Capture source identity and current window before sensor fetch."""
        if self._closed or not self._active or self._overflow:
            return None
        return (self, self._generation, self._sampler.source_stamp())

    def after_sample(self, dt: float, token: object) -> None:
        """Reuse one completed sensor sample; stale-window callbacks are ignored."""
        if self._closed or not self._active or token is None:
            return
        if (
            not isinstance(token, tuple)
            or len(token) != 3
            or token[0] is not self
            or token[1] != self._generation
        ):
            return
        if isinstance(dt, bool) or not math.isfinite(dt) or dt <= 0:
            raise ValueError("Joint contact substep duration must be positive.")
        if self._records and not math.isclose(
            self._time() - self._records[-1].frame.timestamp,
            dt,
            rel_tol=0,
            abs_tol=1e-8,
        ):
            raise ValueError("Joint contact samples do not cover consecutive substeps.")
        self.capture(refresh=False, source_stamp=token[2])

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """End the owned window when its single native row resets."""
        if env_ids is not None and 0 not in env_ids:
            return
        self._generation += 1
        self._active = False
        self._records.clear()
        self._overflow = False

    def close(self) -> None:
        """Invalidate late callbacks without sampling, stepping or issuing commands."""
        self.reset()
        self._closed = True
