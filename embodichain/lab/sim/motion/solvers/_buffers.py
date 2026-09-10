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

"""Bounded analytic IK scratch storage with caller-owned results."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from functools import wraps
from threading import RLock

import torch
import warp as wp

__all__: list[str] = []


class _IKBuffers:
    """Reuse candidate storage, serializing callers across threads and streams."""

    def __init__(self, device: torch.device) -> None:
        self.device = torch.device(device)
        self.capacity = 0
        self.arrays: dict[str, torch.Tensor] = {}
        self.lock = RLock()
        self.event: torch.cuda.Event | None = None

    def reserve(self, max_batch: int) -> None:
        if max_batch < 1:
            raise ValueError("max_batch must be positive")
        with self.lock:
            self.capacity = max(self.capacity, max_batch)

    @contextmanager
    def borrow(self):
        with self.lock:
            stream = (
                torch.cuda.current_stream(self.device)
                if self.device.type == "cuda"
                else None
            )
            if stream is not None and self.event is not None:
                stream.wait_event(self.event)
            scope = (
                wp.ScopedStream(wp.stream_from_torch(stream))
                if stream is not None
                else nullcontext()
            )
            with scope:
                try:
                    yield
                finally:
                    if stream is not None:
                        self.event = torch.cuda.Event()
                        self.event.record(stream)

    def zeros(self, name: str, batch: int, width: int, dtype: torch.dtype) -> wp.array:
        self.capacity = max(self.capacity, batch)
        size = self.capacity * width
        tensor = self.arrays.get(name)
        if tensor is None or tensor.numel() < size:
            tensor = torch.empty(size, dtype=dtype, device=self.device)
            self.arrays[name] = tensor
        if self.device.type == "cuda":
            tensor.record_stream(torch.cuda.current_stream(self.device))
        view = tensor[: batch * width]
        view.zero_()
        return wp.from_torch(view)


def _with_ik_buffers(method):
    """Keep candidate buffers borrowed until result copies have been queued."""

    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with self._ik_buffers.borrow():
            return method(self, *args, **kwargs)

    return wrapped
