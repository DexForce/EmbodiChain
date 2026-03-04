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

from __future__ import annotations

import time
from typing import Any, Callable, Iterator, Optional, Tuple

try:  # Python >=3.8 ships sharedctypes everywhere
    from multiprocessing.sharedctypes import SynchronizedArray
except ImportError:  # pragma: no cover - fallback for static type checking only
    SynchronizedArray = Any

from torch.utils.data import IterableDataset
from tensordict import TensorDict

from embodichain.utils.logger import log_warning


class OnlineRolloutDataset(IterableDataset):
    """Dataset that streams rollouts emitted by :class:`OnlineDataEngine`.

    The dataset expects access to the same shared :class:`TensorDict` buffer and
    the multiprocessing ``index_list`` used by the producer. Every time the
    engine finishes a rollout it advances the indices; this dataset blocks until
    that happens, clones the finished slice to detach it from shared memory, and
    yields individual environment rollouts from the slice. As long as the
    producer keeps running, the iterator produces an infinite stream of samples.

    Args:
        shared_buffer: Shared rollout buffer managed by the engine.
        index_list: Two-element multiprocessing array storing the current
            ``[start, end)`` slice inside ``shared_buffer`` where the producer
            is writing next. The dataset watches changes to detect when new data
            is ready.
        poll_interval_s: Sleep interval (in seconds) when waiting for fresh
            data. Choose a smaller value for lower latency at the cost of more
            CPU usage.
        timeout_s: Optional timeout (in seconds). If provided, the iterator
            raises :class:`TimeoutError` when no new data arrives before the
            deadline. ``None`` waits indefinitely.
        transform: Optional callable applied to every rollout before yielding
            (e.g. to flatten the time dimension or convert to numpy).
        copy_tensors: When ``True`` (default) the data slice is cloned before
            yielding so that the producer can safely overwrite the shared memory
            afterwards. Disable only if the consumer finishes using the data
            before the producer can wrap around.
    """

    def __init__(
        self,
        shared_buffer: TensorDict,
        index_list: SynchronizedArray,
        *,
        poll_interval_s: float = 0.01,
        timeout_s: Optional[float] = None,
        transform: Optional[Callable[[TensorDict], TensorDict]] = None,
        copy_tensors: bool = True,
    ) -> None:
        super().__init__()
        if shared_buffer.batch_size is None or not shared_buffer.batch_size:
            raise ValueError("shared_buffer must have a leading batch dimension")
        self.shared_buffer = shared_buffer
        self.index_list = index_list
        self.poll_interval_s = max(poll_interval_s, 1e-4)
        self.timeout_s = timeout_s
        self.transform = transform
        self.copy_tensors = copy_tensors
        self._buffer_size = int(shared_buffer.batch_size[0])
        self._lock = getattr(index_list, "get_lock", lambda: None)()

    def __iter__(self) -> Iterator[TensorDict]:
        start, end = self._read_indices()

        while True:
            next_start, next_end = self._wait_for_new_range((start, end))
            chunk = self._materialize_chunk(start, end)
            start, end = next_start, next_end

            if chunk is None:
                continue

            for rollout_idx in range(chunk.batch_size[0]):
                rollout_td = chunk[rollout_idx]
                if self.transform is not None:
                    rollout_td = self.transform(rollout_td)
                yield rollout_td

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _read_indices(self) -> Tuple[int, int]:
        if self._lock is None:
            return int(self.index_list[0]), int(self.index_list[1])
        with self._lock:  # type: ignore[attr-defined]
            return int(self.index_list[0]), int(self.index_list[1])

    def _wait_for_new_range(self, current_range: Tuple[int, int]) -> Tuple[int, int]:
        start_time = time.monotonic()
        while True:
            candidate = self._read_indices()
            if candidate != current_range:
                return candidate

            if (
                self.timeout_s is not None
                and (time.monotonic() - start_time) > self.timeout_s
            ):
                raise TimeoutError(
                    "Timed out while waiting for OnlineDataEngine to publish new rollouts."
                )

            time.sleep(self.poll_interval_s)

    def _materialize_chunk(self, start: int, end: int) -> Optional[TensorDict]:
        if end <= start:
            log_warning(
                "Received an empty index range from OnlineDataEngine; waiting for the next chunk."
            )
            return None

        if end > self._buffer_size or start < 0:
            raise ValueError(
                f"Invalid buffer slice [{start}, {end}) for buffer size {self._buffer_size}."
            )

        chunk_view = self.shared_buffer[start:end]
        return chunk_view.clone() if self.copy_tensors else chunk_view

    # IterableDataset does not define __len__ for infinite streams.
    def __len__(self) -> int:  # pragma: no cover - make intent explicit
        raise TypeError(
            "OnlineRolloutDataset is an infinite stream; length is undefined."
        )
