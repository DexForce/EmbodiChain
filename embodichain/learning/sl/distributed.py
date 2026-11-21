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

"""Single-node process-group lifecycle and small SL control messages."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import timedelta
import io
import math
import os
import pickle
from typing import Any, TypeVar

import torch
import torch.distributed as dist

__all__ = ["distributed_run"]

_T = TypeVar("_T")


@contextmanager
def distributed_run(
    device: str | torch.device, *, timeout_seconds: float = 1800.0
) -> Iterator[torch.device]:
    """Bind one device and own a process group when launched by ``torchrun``.

    CUDA jobs use NCCL and ``cuda:LOCAL_RANK``; CPU jobs use Gloo. Ordinary
    single-process calls need no process group. An existing group is borrowed
    and is never destroyed here. All ranks must enter this context together.

    Args:
        device: Recipe device. Under torchrun a CUDA index is replaced by the
            local rank within ``CUDA_VISIBLE_DEVICES``.
        timeout_seconds: Positive collective timeout, including rank-zero
            validation time. It does not enable elastic world-size recovery.

    Yields:
        The device to use before constructing a model and optimizer.
    """
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("Distributed timeout must be finite and positive")
    resolved = torch.device(device)
    requested = int(os.environ.get("WORLD_SIZE", "1"))
    if requested < 1:
        raise ValueError("WORLD_SIZE must be positive")
    if requested > 1:
        if int(os.environ.get("LOCAL_WORLD_SIZE", str(requested))) != requested:
            raise ValueError("SL torchrun currently supports a single node")
        if resolved.type not in ("cpu", "cuda"):
            raise ValueError("SL DDP supports CPU/Gloo or CUDA/NCCL")
        if resolved.type == "cuda":
            resolved = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    if resolved.type == "cuda":
        resolved = torch.device(
            "cuda",
            (
                resolved.index
                if resolved.index is not None
                else torch.cuda.current_device()
            ),
        )
        torch.cuda.set_device(resolved)
    owned = requested > 1 and not dist.is_initialized()
    if owned:
        dist.init_process_group(
            backend="nccl" if resolved.type == "cuda" else "gloo",
            timeout=timedelta(seconds=timeout_seconds),
        )
    try:
        yield resolved
    finally:
        if owned and dist.is_initialized():
            dist.destroy_process_group()


class _DistributedContext:
    """Coordinate control boundaries; model gradients use native Torch DDP."""

    def __init__(self, device: str | torch.device) -> None:
        self.device = torch.device(device)
        self.enabled = dist.is_initialized() and dist.get_world_size() > 1
        self.rank = dist.get_rank() if self.enabled else 0
        self.world_size = dist.get_world_size() if self.enabled else 1
        self.control_device = (
            self.device
            if self.enabled and dist.get_backend() == "nccl"
            else torch.device("cpu")
        )

    def check(self, error: Exception | None, stage: str) -> None:
        """Propagate reachable local failures before the next collective phase."""
        if not self.enabled:
            if error is not None:
                raise error
            return
        failed = torch.tensor(
            int(error is not None), dtype=torch.int32, device=self.control_device
        )
        dist.all_reduce(failed, op=dist.ReduceOp.MAX)
        if not failed.item():
            return
        message = None if error is None else f"{type(error).__name__}: {error}"
        messages: list[Any] = [None] * self.world_size
        dist.all_gather_object(messages, message)
        for rank, failure in enumerate(messages):
            if failure is not None:
                raise RuntimeError(
                    f"Distributed SL {stage} failed on rank {rank}: {failure}"
                ) from error

    def run(self, function: Callable[[], _T], stage: str) -> _T:
        error = None
        result: Any = None
        try:
            result = function()
        except Exception as caught:
            error = caught
        self.check(error, stage)
        return result

    def root(self, function: Callable[[], _T], stage: str) -> _T:
        if not self.enabled:
            return function()
        payload: list[Any] = [None]
        if self.rank == 0:
            try:
                payload[0] = (function(), None)
            except Exception as error:
                payload[0] = (None, f"{type(error).__name__}: {error}")
        dist.broadcast_object_list(payload, src=0)
        value, failure = payload[0]
        if failure is not None:
            raise RuntimeError(f"Distributed SL {stage} failed on rank 0: {failure}")
        return value

    def agree(self, value: Any, stage: str) -> None:
        if not self.enabled:
            return
        values: list[Any] = [None] * self.world_size
        dist.all_gather_object(values, value)
        if any(item != values[0] for item in values[1:]):
            raise ValueError(f"Distributed SL {stage} differs between ranks")

    def gather(self, value: Any) -> list[Any] | None:
        if not self.enabled:
            return [value]

        # Participant snapshots can contain a full model EMA. Broadcasting one
        # bounded chunk at a time keeps NCCL's device allocation independent of
        # both the snapshot size and world size; only rank zero retains copies.
        def serialize() -> io.BytesIO | None:
            if self.rank == 0:
                return None
            stream = io.BytesIO()
            pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
            return stream

        payload = self.run(serialize, "checkpoint state serialization")
        sizes: list[int] = [0] * self.world_size
        dist.all_gather_object(sizes, payload.tell() if payload is not None else 0)
        chunk_size = min(max(sizes), 16 * 1024 * 1024)
        buffer = self.run(
            lambda: torch.empty(
                chunk_size, dtype=torch.uint8, device=self.control_device
            ),
            "checkpoint transfer buffer",
        )
        values: list[Any] | None = [value] if self.rank == 0 else None
        sender = (
            torch.frombuffer(payload.getbuffer(), dtype=torch.uint8)
            if payload is not None
            else None
        )
        for source in range(1, self.world_size):
            received = io.BytesIO() if self.rank == 0 else None
            failure = None
            for offset in range(0, sizes[source], chunk_size):
                count = min(chunk_size, sizes[source] - offset)
                if self.rank == source:
                    assert sender is not None
                    buffer[:count].copy_(sender[offset : offset + count])
                dist.broadcast(buffer[:count], src=source)
                if received is not None and failure is None:
                    try:
                        received.write(buffer[:count].cpu().numpy().tobytes())
                    except Exception as error:
                        failure = error
            if received is not None and failure is None:
                try:
                    received.seek(0)
                    assert values is not None
                    values.append(pickle.load(received))
                except Exception as error:
                    failure = error
            self.check(failure, "checkpoint state collection")
        return values
