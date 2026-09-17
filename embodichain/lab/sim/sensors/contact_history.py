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

"""Sensor-owned contact reduction and timing for a selected set of actors."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

__all__ = ["ContactHistory"]


class ContactHistory:
    """Accumulate selected contacts across one control interval.

    ``contact`` and ``force`` describe the latest physics sample; ``found``,
    ``peak_force`` and ``first_contact`` retain events from every substep.
    ``current_air_time`` measures an unfinished flight; ``last_air_time``
    captures its duration on landing, including the landing sampling interval.
    Reading these tensors never advances time or clears an event.

    Args:
        actor_ids: Selected body IDs, shaped (environments, bodies).
        counterpart_ids: Allowed counterpart IDs per environment; None accepts all.
        include_unknown_counterpart: Accept the query's unknown counterpart (-1)
            in addition to explicitly selected IDs. This does not identify ground.
        force_threshold: Minimum individual contact-force magnitude for a timing event.
    """

    def __init__(
        self,
        actor_ids: torch.Tensor,
        *,
        counterpart_ids: torch.Tensor | None = None,
        include_unknown_counterpart: bool = False,
        force_threshold: float = 0.0,
    ) -> None:
        if actor_ids.ndim != 2:
            raise ValueError("actor_ids must have shape (environments, bodies).")
        self.actor_ids = actor_ids.clone()
        self.counterpart_ids = (
            None if counterpart_ids is None else counterpart_ids.to(actor_ids).clone()
        )
        self.include_unknown_counterpart = include_unknown_counterpart
        self.force_threshold = float(force_threshold)
        shape = actor_ids.shape
        self.contact = torch.zeros(shape, device=actor_ids.device, dtype=torch.bool)
        self.found = torch.zeros_like(self.contact)
        self.first_contact = torch.zeros_like(self.contact)
        self.force = torch.zeros((*shape, 3), device=actor_ids.device)
        self.peak_force = torch.zeros_like(self.force)
        self.current_air_time = torch.zeros(shape, device=actor_ids.device)
        self.last_air_time = torch.zeros_like(self.current_air_time)
        self.contact_count = torch.zeros(shape[0], device=actor_ids.device)

    def begin_control_step(self) -> None:
        """Clear interval aggregates while retaining contact timing."""
        self.found.zero_()
        self.first_contact.zero_()
        self.peak_force.zero_()
        self.contact_count.zero_()

    def update(self, data: Mapping[str, torch.Tensor], dt: float) -> None:
        """Reduce a new physics sample and advance timing exactly once.

        Args:
            data: ContactSensor data in its documented actor/impulse convention.
            dt: Duration of the physics sample in seconds.
        """
        if dt <= 0:
            raise ValueError("Contact sampling dt must be positive.")
        pair = data["user_ids"]
        valid = data["is_valid"]
        force = (data["normal"] * data["impulse"].unsqueeze(-1) + data["friction"]) / dt
        # Invalid slots can retain arbitrary data from previous query updates.
        force = torch.where(valid.unsqueeze(-1), force, 0.0)
        matches = []
        for side in (0, 1):
            selected = pair[:, :, side, None] == self.actor_ids[:, None, :]
            allowed = valid
            if self.counterpart_ids is not None:
                other = pair[:, :, 1 - side]
                allowed = allowed & (
                    (other[:, :, None] == self.counterpart_ids[:, None, :]).any(-1)
                    | (self.include_unknown_counterpart & (other == -1))
                )
            matches.append(selected & allowed.unsqueeze(-1))
        signed = matches[1].to(force.dtype) - matches[0].to(force.dtype)
        self.force.copy_((force.unsqueeze(2) * signed.unsqueeze(-1)).sum(1))
        events = matches[0] | matches[1]
        if self.force_threshold > 0:
            events &= (force.norm(dim=-1) > self.force_threshold).unsqueeze(-1)
        contact = events.any(1)
        landed = contact & ~self.contact
        elapsed = self.current_air_time + dt
        self.last_air_time.copy_(torch.where(landed, elapsed, self.last_air_time))
        self.current_air_time.copy_(torch.where(contact, 0.0, elapsed))
        self.contact.copy_(contact)
        self.found.logical_or_(contact)
        self.first_contact.logical_or_(landed)
        stronger = self.force.norm(dim=-1) > self.peak_force.norm(dim=-1)
        self.peak_force.copy_(
            torch.where(stronger.unsqueeze(-1), self.force, self.peak_force)
        )
        self.contact_count.add_(contact.any(dim=-1))

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Clear only selected environment rows, including unfinished timing.

        Args:
            env_ids: Rows to clear. None selects every environment.
        """
        ids = slice(None) if env_ids is None else env_ids
        for value in (
            self.contact,
            self.found,
            self.first_contact,
            self.force,
            self.peak_force,
            self.current_air_time,
            self.last_air_time,
            self.contact_count,
        ):
            value[ids] = 0
