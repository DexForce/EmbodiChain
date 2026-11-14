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

"""Device-side permission checks over complete native contact samples."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING
import torch

if TYPE_CHECKING:
    from .sensor import ContactSensor

__all__ = ["ContactPermissionChecker"]


class ContactPermissionChecker:
    """Classify every valid measured contact without constructing Python pairs.

    Args:
        sensor: Initialized native sensor with applied-impulse validity flags.
        actor_frames: Explicit actor ID to frame name mapping for every row.
            A global static actor may appear in several rows with the same name.
        allowed_contacts: Structural and task-owned unordered frame exemptions.
        moving_frames: Robot and dynamic frames requiring permission checks.

    This checker does not infer adjacency, change filters, step physics or select
    a sampling frequency. A host requiring substep validation must invoke it at
    every original physical sample. Unknown actors and unavailable impulses fail
    closed; cumulative native/scatter loss invalidates every row conservatively.
    """

    def __init__(
        self,
        sensor: ContactSensor,
        actor_frames: Sequence[Mapping[int, str]],
        *,
        allowed_contacts: Sequence[tuple[str, str]],
        moving_frames: Sequence[str],
    ) -> None:
        self.sensor = sensor
        self.device = torch.device(sensor.device)
        self.rows = sensor.num_instances
        if len(actor_frames) != self.rows or any(not row for row in actor_frames):
            raise ValueError("actor_frames must declare every nonempty sensor row")
        if any(
            type(actor) is not int
            or actor < 0
            or type(name) is not str
            or not name.strip()
            for row in actor_frames
            for actor, name in row.items()
        ):
            raise ValueError(
                "contact actors require nonnegative IDs and explicit frame names"
            )
        names = sorted({name for row in actor_frames for name in row.values()})
        if any(
            len(pair) != 2 or any(name not in names for name in pair)
            for pair in allowed_contacts
        ):
            raise ValueError("contact permissions reference an absent frame")
        if set(moving_frames) - set(names):
            raise ValueError("moving_frames references an absent contact frame")
        codes = {name: index for index, name in enumerate(names)}
        actors = {}
        for row, frames in enumerate(actor_frames):
            for actor, name in frames.items():
                code, rows = actors.setdefault(actor, (codes[name], set()))
                if code != codes[name]:
                    raise ValueError("contact actor has ambiguous frame identities")
                rows.add(row)
        actors = sorted(actors.items())
        allowed = {frozenset(pair) for pair in allowed_contacts}
        self._actors = torch.tensor(
            [actor for actor, _ in actors], dtype=torch.int32, device=self.device
        )
        self._memberships = torch.tensor(
            [
                [row in members for row in range(self.rows)]
                for _, (_, members) in actors
            ],
            dtype=torch.bool,
            device=self.device,
        )
        self._codes = torch.tensor(
            [code for _, (code, _) in actors], dtype=torch.long, device=self.device
        )
        self._moving = torch.tensor(
            [name in moving_frames for name in names],
            dtype=torch.bool,
            device=self.device,
        )
        self._allowed = torch.tensor(
            [[frozenset((a, b)) in allowed for b in names] for a in names],
            dtype=torch.bool,
            device=self.device,
        )
        self._first = torch.empty(self.rows, dtype=torch.int32, device=self.device)
        self._drop = torch.zeros(1, dtype=torch.int32, device=self.device)
        self._output = torch.empty(
            (self.rows, 3), dtype=torch.int32, device=self.device
        )
        self._ready = torch.cuda.Event() if self.device.type == "cuda" else None
        if self._ready is not None:
            self._ready.record(torch.cuda.current_stream(self.device))
        self._actors_token = id(sensor._query.actors)

    def check(self) -> torch.Tensor:
        """Return owned per-row ``(reason, actor0, actor1)`` integer summaries.

        Eager CUDA checks wait for the latest sensor sample on the caller's
        stream without refreshing contacts or advancing histories.

        Returns:
            Tensor ``(B, 3)`` on the sensor device. Reason zero passes; one means
            unavailable applied impulse, two unknown/cross-row actor, three
            forbidden moving contact, four dropped evidence. The earliest bad
            buffer slot supplies actor IDs. No valid contact is subsampled.

        Raises:
            RuntimeError: Sensor topology changed after permission binding.
        """
        import warp as wp
        from embodichain.compute.geometry._warp.contact_permissions import (
            classify_contacts,
            contact_failures,
        )

        if self._ready is not None:
            torch.cuda.current_stream(self.device).wait_event(self._ready)
        # The checker event orders its own buffers, not the sensor's producer.
        self.sensor._wait_for_sample()
        data = self.sensor.get_data()
        if id(self.sensor._query.actors) != self._actors_token:
            raise RuntimeError(
                "contact topology changed; bind a new permission checker"
            )
        pairs = data["user_ids"]
        self._first.fill_(pairs.shape[1] * 8)
        self._drop.zero_()
        # Use the current Torch stream on every call, rather than assuming that
        # the stream used at construction is still the caller's producer.
        stream = (
            wp.stream_from_torch(torch.cuda.current_stream(self.device))
            if self.device.type == "cuda"
            else None
        )
        wp.launch(
            classify_contacts,
            pairs.shape[:2],
            inputs=[
                wp.from_torch(pairs),
                wp.from_torch(data["is_valid"]),
                wp.from_torch(data["impulse_valid"]),
                wp.from_torch(self._actors),
                wp.from_torch(self._memberships),
                wp.from_torch(self._codes),
                wp.from_torch(self._moving),
                wp.from_torch(self._allowed),
                wp.from_torch(self.sensor.dropped_contacts_mask),
                wp.from_torch(self._first),
                wp.from_torch(self._drop),
            ],
            device=str(self.device),
            stream=stream,
        )
        wp.launch(
            contact_failures,
            self.rows,
            inputs=[
                wp.from_torch(pairs),
                wp.from_torch(self._first),
                wp.from_torch(self._drop),
                wp.from_torch(self._output),
            ],
            device=str(self.device),
            stream=stream,
        )
        result = self._output.clone()
        if self._ready is not None:
            current = torch.cuda.current_stream(self.device)
            self._ready.record(current)
            for value in (
                self._actors,
                self._memberships,
                self._codes,
                self._moving,
                self._allowed,
                self._first,
                self._drop,
                self._output,
            ):
                value.record_stream(current)
        return result
