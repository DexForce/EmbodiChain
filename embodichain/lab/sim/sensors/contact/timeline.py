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

"""Bounded per-environment contact history on the host's physics/control clock."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import MISSING
import math
from typing import Any

import torch

from embodichain.lab.sim.evidence import PhysicalEvidenceRequest
from embodichain.utils import configclass
from .sampling import (
    CONTACT_ACTOR_IDS,
    CONTACT_VALID_MASK,
    CONTACT_NORMALS,
    CONTACT_IMPULSES,
    CONTACT_POSITIONS,
    ContactEvidenceSampler,
    ContactSourceStamp,
)

__all__ = ["ContactTimeline", "ContactTimelineCfg"]


@configclass
class ContactTimelineCfg:
    """Explicit timing and memory bounds for physical contact history."""

    step_dt: float = MISSING
    max_control_steps: int = 30000
    max_contact_rows: int = 1_000_000
    """Soft stop threshold checked before each complete native sample.

    A retained sample is never truncated. With N environments and sensor capacity
    C contacts per environment, charged rows stay below this threshold plus N*C.
    Once the threshold is reached, the next sample is unavailable and records a
    capacity error. This is not a strict allocation/RSS ceiling.
    """
    max_substeps_per_control: int = 256
    max_errors_per_env: int = 16


class ContactTimeline:
    """Collect packed valid contacts through a ContactSensor sampling observer.

    The clock callback reports ``(completed_control_steps, elapsed_control_time)``
    or None while recording is inactive. The host owns task/episode activation;
    this collector reads no bridge internals or task labels. Samples are packed
    to CPU lists rather than retaining fixed-capacity sensor padding. Contact
    capacity is checked before each sample, preserving the existing rule: at
    most one sampled batch can cross the configured contact-row budget.
    """

    def __init__(
        self,
        sampler: ContactEvidenceSampler,
        cfg: ContactTimelineCfg,
        *,
        clock: Callable[[], tuple[int, float] | None],
        control_parts: tuple[str, ...] = (),
        opening_reader: Callable[[str, int], Mapping[str, object]] | None = None,
    ) -> None:
        """Bind explicit acquisition, clock and optional hand-opening inputs.

        Args:
            sampler: Contact sampler with required native/topology source readers.
            cfg: Control period and bounded retention budgets.
            clock: Host-owned control clock; no scene or task-name inference.
            control_parts: Explicit hands whose measured opening should be recorded.
            opening_reader: Read-only calibrated opening reader for those hands.
        """
        cfg.validate()
        if (
            isinstance(cfg.step_dt, bool)
            or not math.isfinite(cfg.step_dt)
            or cfg.step_dt <= 0
        ):
            raise ValueError("step_dt must be finite and positive.")
        for name in (
            "max_control_steps",
            "max_contact_rows",
            "max_substeps_per_control",
            "max_errors_per_env",
        ):
            if type(getattr(cfg, name)) is not int or getattr(cfg, name) < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if not callable(clock):
            raise TypeError("clock must be callable.")
        if (
            type(control_parts) is not tuple
            or len(set(control_parts)) != len(control_parts)
            or any(
                type(part) is not str or not part or part != part.strip()
                for part in control_parts
            )
        ):
            raise ValueError("control_parts must contain unique explicit identifiers.")
        if control_parts and not callable(opening_reader):
            raise ValueError("Declared control parts require an opening reader.")
        self.sampler = sampler
        self._cfg = deepcopy(cfg)
        self._clock = clock
        self._control_parts, self._opening_reader = control_parts, opening_reader
        count = sampler.num_envs
        self._rows: list[dict[int, dict[str, Any]]] = [{} for _ in range(count)]
        self._errors: list[list[str]] = [[] for _ in range(count)]
        self._generation = [0] * count
        self._charged_contacts = [0] * count
        self._pending: list[list[dict[str, Any]]] = [[] for _ in range(count)]
        self._blocked_rows: set[int] = set()
        self._index: int | None = None
        self._elapsed = 0.0
        self._sample_revision = 0

    def _invalidate(self, reason: str, rows: Sequence[int] | None = None) -> None:
        selected = range(self.sampler.num_envs) if rows is None else rows
        for row in selected:
            if (
                reason not in self._errors[row]
                and len(self._errors[row]) < self._cfg.max_errors_per_env
            ):
                self._errors[row].append(reason)

    def begin_control_step(self) -> None:
        """Begin an explicitly active host interval after sensor counters reset."""
        if self._index is not None and 0 < self._elapsed + 1e-9 < self._cfg.step_dt:
            self._invalidate(
                "Previous control interval has incomplete contact samples."
            )
        self._index, self._elapsed = None, 0.0
        self._pending = [[] for _ in range(self.sampler.num_envs)]
        self._blocked_rows.clear()
        try:
            value = self._clock()
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError) as error:
            self._invalidate(str(error) or type(error).__name__)
            return
        if value is None:
            return
        completed, timestamp = value
        if (
            type(completed) is not int
            or completed < 0
            or not math.isfinite(timestamp)
            or abs(timestamp - completed * self._cfg.step_dt) > 1e-8
        ):
            self._invalidate("Contact control clock differs from the episode.")
            return
        self._index = completed + 1
        for row, records in enumerate(self._rows):
            if self._index in records:
                self._invalidate(
                    "Contact control interval would overwrite an existing sample.",
                    (row,),
                )
                self._blocked_rows.add(row)

    def before_sample(self) -> ContactSourceStamp | None:
        """Capture the source revision before ContactSensor fetches the substep."""
        return None if self._index is None else self.sampler.source_stamp()

    def after_sample(self, dt: float, token: object) -> None:
        """Observe the completed sensor sample without refetching or stepping.

        Args:
            dt: Physics substep duration in seconds.
            token: Source stamp captured before this same sample was fetched.
        """
        if self._index is None:
            return
        if isinstance(dt, bool) or not math.isfinite(dt) or dt <= 0:
            self._invalidate("Contact substep duration must be finite and positive.")
            return
        self._elapsed += dt
        if self._elapsed > self._cfg.step_dt + 1e-8:
            self._invalidate("Control interval differs from the environment clock.")
            return
        active = []
        for row in range(self.sampler.num_envs):
            if row in self._blocked_rows:
                continue
            if (
                len(self._rows[row]) >= self._cfg.max_control_steps
                or len(self._pending[row]) >= self._cfg.max_substeps_per_control
            ):
                self._invalidate("Contact timeline sample capacity exceeded.", (row,))
                self._blocked_rows.add(row)
            else:
                active.append(row)
        if not active:
            return
        if sum(self._charged_contacts) >= self._cfg.max_contact_rows:
            self._invalidate("Contact timeline capacity exceeded.", active)
            return
        if (
            type(token) is not ContactSourceStamp
            or token.native_revision is None
            or token.topology_revision is None
        ):
            if type(token) is ContactSourceStamp:
                for error in token.errors:
                    self._invalidate(error, active)
            self._invalidate(
                "Native contact source revision or topology stamp is unavailable.",
                active,
            )
            return
        self._sample_revision += 1
        request = PhysicalEvidenceRequest(
            (self._index - 1) * self._cfg.step_dt + self._elapsed,
            self._sample_revision,
            torch.tensor(active, dtype=torch.long),
        )
        sample = self.sampler.sample(request, refresh=False, source_stamp=token)
        batches = sample.frame.batches
        for offset, env_id in enumerate(active):
            mask = batches[CONTACT_VALID_MASK].values[offset]
            contacts = {
                label: batches[channel].values[offset, mask].detach().cpu().tolist()
                for label, channel in (
                    ("pairs", CONTACT_ACTOR_IDS),
                    ("positions", CONTACT_POSITIONS),
                    ("normals", CONTACT_NORMALS),
                    ("impulses", CONTACT_IMPULSES),
                )
            }
            available = all(bool(batch.valid[offset]) for batch in batches.values())
            if not available:
                reasons = [
                    batch.acquisition_errors[offset]
                    for batch in batches.values()
                    if batch.acquisition_errors[offset]
                ]
                self._invalidate(
                    "; ".join(dict.fromkeys(reasons))
                    or "Contact sample is unavailable.",
                    (env_id,),
                )
            self._charged_contacts[env_id] += len(contacts["pairs"])
            record = {
                "offset": self._elapsed,
                "dt": dt,
                "timestamp": sample.frame.timestamp,
                "observation_revision": sample.frame.observation_revision,
                "native_revision": sample.native_revision,
                "topology_revision": sample.topology_revision,
                "dropped_contacts": sample.dropped_contacts,
                "available": available,
                "contacts": contacts,
            }
            self._pending[env_id].append(record)
            if self._elapsed + 1e-9 < self._cfg.step_dt:
                continue
            opening = {}
            for part in self._control_parts:
                try:
                    opening[part] = deepcopy(dict(self._opening_reader(part, env_id)))
                except (
                    AttributeError,
                    KeyError,
                    RuntimeError,
                    TypeError,
                    ValueError,
                ) as error:
                    self._invalidate(str(error) or type(error).__name__, (env_id,))
                    opening[part] = {
                        "finite": False,
                        "open": False,
                        "error": str(error) or type(error).__name__,
                    }
            self._rows[env_id][self._index] = {
                "step": self._index,
                "time": self._index * self._cfg.step_dt,
                "episode_generation": self._generation[env_id],
                "native_revision": sample.native_revision,
                "topology_revision": sample.topology_revision,
                "dropped_contacts": sample.dropped_contacts,
                "available": all(item["available"] for item in self._pending[env_id])
                and not self._errors[env_id],
                "contacts": contacts,
                "substeps": self._pending[env_id],
                "hand_opening": opening,
            }

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Clear only selected rows and advance their episode generations.

        Args:
            env_ids: Sensor-compatible indices, or None for every row. The same
                torch.long conversion and negative-index aliases as ContactSensor
                apply; duplicates increment each selected row's generation once.
                A reset during an interval suppresses those rows until the next
                complete interval.
        """
        if env_ids is None:
            rows = list(range(self.sampler.num_envs))
        else:
            # Match the sensor's actual index_fill_ contract, including scalar
            # tensors, negative aliases and conversion errors. Build the mask
            # before changing any recorder state, then reset each native row once.
            indices = torch.as_tensor(env_ids, dtype=torch.long).detach().cpu()
            selected = torch.zeros(self.sampler.num_envs, dtype=torch.bool)
            selected.index_fill_(0, indices, True)
            rows = selected.nonzero(as_tuple=False).flatten().tolist()
        for row in rows:
            self._rows[row].clear()
            self._errors[row].clear()
            self._pending[row] = []
            self._charged_contacts[row] = 0
            self._generation[row] += 1
            self._blocked_rows.add(row)
        if len(rows) == self.sampler.num_envs:
            self._index = None

    def snapshot(self) -> dict[str, Any]:
        """Own packed native data and per-row loss state without evaluating a task.

        Returns:
            Application-neutral raw metadata. Frontends own serialized schema IDs,
            actor/semantic label maps and task-specific evidence evaluation.
        """
        return deepcopy(
            {
                "scope": "native_substep_samples_not_continuous_contact_proof",
                "step_dt": self._cfg.step_dt,
                "retained_contact_rows": sum(self._charged_contacts),
                "environments": [
                    {
                        "env_id": row,
                        "episode_generation": self._generation[row],
                        "errors": self._errors[row],
                        "retained_contact_rows": self._charged_contacts[row],
                        "samples": list(self._rows[row].values()),
                        "incomplete_control_step": (
                            self._index
                            if self._pending[row] and self._index not in self._rows[row]
                            else None
                        ),
                    }
                    for row in range(self.sampler.num_envs)
                ],
            }
        )
