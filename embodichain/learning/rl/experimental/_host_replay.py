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

"""Private synchronous host-replay transfer boundary for large native experiments.

Host storage retains its sampling and episode/history semantics. Only sampled
batches cross to the learner device; returned tensors own storage independently
of later host writes. Sampling uses the host buffer's RNG backend.
"""

from __future__ import annotations

from typing import Protocol, TypedDict

import torch

__all__: list[str] = []


class _HostReplayStorage(Protocol):
    """Single-threaded host storage that copies added transitions before returning."""

    device: torch.device

    def add(self, transition: dict[str, torch.Tensor]) -> None: ...
    def sample_windows(self, batch: int, window: int) -> dict[str, torch.Tensor]: ...
    def sample_starts(
        self, batch: int, history: int
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]: ...


class _HostReplayRing(Protocol):
    """CPU time/environment ring whose writer is serialized by its owner."""

    device: torch.device
    steps: int
    num_envs: int
    ptr: int
    size: int
    data: dict[str, torch.Tensor]


class _ReplayState(TypedDict):
    version: int
    steps: int
    num_envs: int
    ptr: int
    size: int
    data: dict[str, torch.Tensor]


def _host_replay_state_dict(storage: _HostReplayRing) -> _ReplayState:
    """Borrow CPU ring tensors for synchronous serialization without a full copy.

    The caller must finish serialization before any writer resumes. This is
    intentionally a borrowed view, like a module state dictionary, rather than
    an independently mutable checkpoint. Restore copies into owned ring memory.
    """
    if torch.device(storage.device).type != "cpu" or any(
        value.device.type != "cpu" for value in storage.data.values()
    ):
        raise ValueError("Host replay checkpoint requires CPU storage")
    return {
        "version": 1,
        "steps": storage.steps,
        "num_envs": storage.num_envs,
        "ptr": storage.ptr,
        "size": storage.size,
        "data": dict(storage.data),
    }


@torch.no_grad()
def _load_host_replay_state_dict(storage: _HostReplayRing, state: _ReplayState) -> None:
    """Validate a complete ring receipt before copying or changing its cursor.

    Episode discontinuities caused by restarting the simulator are the caller's
    responsibility; transport restores the recorded masks and history exactly.
    """
    if (
        state.get("version") != 1
        or state.get("steps") != storage.steps
        or state.get("num_envs") != storage.num_envs
        or torch.device(storage.device).type != "cpu"
    ):
        raise ValueError("Host replay checkpoint version/capacity/device mismatch")
    ptr, size = state["ptr"], state["size"]
    if (
        type(ptr) is not int
        or type(size) is not int
        or not 0 <= ptr < storage.steps
        or not 0 <= size <= storage.steps
        or (size < storage.steps and ptr != size)
    ):
        raise ValueError("Invalid host replay checkpoint cursor or size")
    data = state["data"]
    if data.keys() != storage.data.keys():
        raise ValueError("Host replay checkpoint fields mismatch")
    for key, target in storage.data.items():
        value = data[key]
        if (
            value.shape != target.shape
            or value.dtype != target.dtype
            or value.device.type != "cpu"
            or target.device.type != "cpu"
        ):
            raise ValueError(f"Host replay checkpoint tensor mismatch: {key}")
    for key, target in storage.data.items():
        target.copy_(data[key])
    storage.ptr, storage.size = ptr, size


class _HostReplayAdapter:
    """Transfer complete replay batches while preserving source sampling rules.

    Copies are synchronous and deliberately unpinned. This bounds transient
    memory to one transition or sampled batch and avoids borrowed asynchronous
    host buffers. The caller owns thread serialization and storage lifetime.
    """

    def __init__(self, storage: _HostReplayStorage, device: torch.device | str) -> None:
        if torch.device(storage.device).type != "cpu":
            raise ValueError("Host replay storage must reside on CPU")
        self.storage = storage
        self.device = torch.device(device)

    @torch.no_grad()
    def add(self, transition: dict[str, torch.Tensor]) -> None:
        """Complete host copies before the storage consumes a transition."""
        self.storage.add(
            {
                key: value.detach().to("cpu", non_blocking=False)
                for key, value in transition.items()
            }
        )

    def _move(self, values: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {
            key: value.detach().to(self.device, copy=True, non_blocking=False)
            for key, value in values.items()
        }

    @torch.no_grad()
    def sample_windows(self, batch: int, window: int) -> dict[str, torch.Tensor]:
        """Return owned learner-device windows and masks from the host sampler."""
        return self._move(self.storage.sample_windows(batch, window))

    @torch.no_grad()
    def sample_starts(
        self, batch: int, history: int
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Return owned learner-device starts and clamped source histories."""
        starts, histories = self.storage.sample_starts(batch, history)
        return self._move(starts), self._move(histories)
