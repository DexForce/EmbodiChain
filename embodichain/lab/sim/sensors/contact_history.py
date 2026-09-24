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
from typing import TYPE_CHECKING

import torch
import warp as wp

from embodichain.lab.sim.sensors._warp.contact_history import (
    reduce_contact_rows,
    reduce_contact_batch,
    finish_contact_sample,
)

if TYPE_CHECKING:
    from dexsim.scene import ContactBuffer

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
        wp.init()
        if actor_ids.ndim != 2:
            raise ValueError("actor_ids must have shape (environments, bodies).")
        if actor_ids.dtype not in (torch.int32, torch.int64) or (actor_ids < 0).any():
            raise ValueError("actor_ids must contain nonnegative integer identities.")
        if counterpart_ids is not None and (
            counterpart_ids.ndim != 2 or counterpart_ids.shape[0] != actor_ids.shape[0]
        ):
            raise ValueError("counterpart_ids must have one row per environment.")
        self.actor_ids = actor_ids.clone()
        self.counterpart_ids = (
            None if counterpart_ids is None else counterpart_ids.to(actor_ids).clone()
        )
        self.include_unknown_counterpart = include_unknown_counterpart
        self.force_threshold = float(force_threshold)
        shape = actor_ids.shape
        # Pooled allocations: one storage per dtype/shape family so a reset
        # fuses into a handful of launches instead of one CUDA write per
        # field. The public attributes remain per-field contiguous views.
        self._bool_pool = torch.zeros(
            (3, *shape), device=actor_ids.device, dtype=torch.bool
        )
        self.contact = self._bool_pool[0]
        self.found = self._bool_pool[1]
        self.first_contact = self._bool_pool[2]
        self._force_pool = torch.zeros((2, *shape, 3), device=actor_ids.device)
        self.force = self._force_pool[0]
        self.peak_force = self._force_pool[1]
        self._air_pool = torch.zeros((2, *shape), device=actor_ids.device)
        self.current_air_time = self._air_pool[0]
        self.last_air_time = self._air_pool[1]
        self.contact_count = torch.zeros(shape[0], device=actor_ids.device)
        self._hits = torch.zeros(shape, device=actor_ids.device, dtype=torch.int32)
        self._env_hits = torch.zeros(
            shape[0], device=actor_ids.device, dtype=torch.int32
        )
        self._actor_keys, self._actor_rows = self._index_actors(self.actor_ids)
        if (self._actor_keys[1:] == self._actor_keys[:-1]).any():
            raise ValueError(
                "Tracked actor IDs must be unique within each environment."
            )
        self._counterpart_keys = (
            torch.empty(0, device=actor_ids.device, dtype=torch.int64)
            if self.counterpart_ids is None
            else self._index_actors(self.counterpart_ids)[0]
        )

    @staticmethod
    def _index_actors(ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        env = torch.arange(ids.shape[0], device=ids.device, dtype=torch.int64)
        keys = env[:, None] * (2**32) + ids.to(torch.int64)
        keys, rows = keys.flatten().sort()
        return keys, rows.to(torch.int32)

    def _start_sample(self, dt: float) -> list:
        if dt <= 0:
            raise ValueError("Contact sampling dt must be positive.")
        self.force.zero_()
        self._hits.zero_()
        self._env_hits.zero_()
        return [
            wp.from_torch(self._actor_keys),
            wp.from_torch(self._actor_rows),
            wp.from_torch(self._counterpart_keys),
            self.counterpart_ids is None,
            self.include_unknown_counterpart,
            self.force_threshold,
            dt,
            wp.from_torch(self.force),
            wp.from_torch(self._hits),
            wp.from_torch(self._env_hits),
        ]

    def _launch(self, kernel, dim: int | tuple[int, ...], inputs: list) -> None:
        wp.launch(
            kernel,
            dim=dim,
            inputs=inputs,
            device=str(self.actor_ids.device),
            stream=(
                wp.stream_from_torch(self.actor_ids.device)
                if self.actor_ids.is_cuda
                else None
            ),
        )

    def _finish_sample(self, dt: float) -> None:
        self._launch(
            finish_contact_sample,
            self.actor_ids.shape,
            [
                dt,
                *[
                    wp.from_torch(value)
                    for value in (
                        self._hits,
                        self._env_hits,
                        self.force,
                        self.peak_force,
                        self.contact,
                        self.found,
                        self.first_contact,
                        self.current_air_time,
                        self.last_air_time,
                        self.contact_count,
                    )
                ],
            ],
        )

    def _update_from_query(self, buffer: ContactBuffer, dt: float) -> None:
        """Reduce compact query rows using its device-resident valid-row count."""
        args = self._start_sample(dt)
        self._launch(
            reduce_contact_rows,
            buffer.capacity,
            [
                wp.from_torch(buffer.data),
                wp.from_torch(buffer.actor_ids),
                wp.from_torch(buffer.env_ids),
                wp.from_torch(buffer.count_device),
                *args,
            ],
        )
        self._finish_sample(dt)

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
        args = self._start_sample(dt)
        self._launch(
            reduce_contact_batch,
            data["is_valid"].shape,
            [
                wp.from_torch(data["user_ids"].to(torch.int32)),
                *[
                    wp.from_torch(data[name])
                    for name in ("is_valid", "normal", "friction", "impulse")
                ],
                *args,
            ],
        )
        self._finish_sample(dt)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Clear only selected environment rows, including unfinished timing.

        Args:
            env_ids: Rows to clear. None selects every environment.
        """
        if env_ids is None:
            # One fused multi-tensor zero for the full-reset case.
            torch._foreach_zero_([
                self._bool_pool,
                self._force_pool,
                self._air_pool,
                self.contact_count,
                self._hits,
                self._env_hits,
            ])
            return
        ids = env_ids
        # Pooled rows: one advanced-indexing write per dtype/shape family.
        self._bool_pool[:, ids] = False
        self._force_pool[:, ids] = 0
        self._air_pool[:, ids] = 0
        self.contact_count[ids] = 0
        self._hits[ids] = 0
        self._env_hits[ids] = 0

