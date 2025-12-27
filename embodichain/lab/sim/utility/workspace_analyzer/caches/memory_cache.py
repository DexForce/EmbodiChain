# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
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

import gc
from typing import List
import numpy as np

from embodichain.lab.sim.utility.workspace_analyzer.caches.base_cache import BaseCache

all = [
    "MemoryCache",
]


class MemoryCache(BaseCache):
    """In-memory cache for workspace sampling.

    Stores all pose samples in RAM for fast access. Suitable for
    smaller datasets or when memory is not a constraint.
    """

    def __init__(self, batch_size: int = 5000, save_threshold: int = 10000000):
        """Initialize memory cache.

        Args:
            batch_size: Number of samples per processing batch
            save_threshold: Threshold for triggering garbage collection
        """
        super().__init__(batch_size, save_threshold)
        self._poses: List[np.ndarray] = []

    def add(self, poses: List[np.ndarray]) -> None:
        """Add poses to in-memory storage.

        Args:
            poses: List of 4x4 transformation matrices
        """
        self._poses.extend(poses)
        self._total_processed += len(poses)

        # Trigger garbage collection periodically
        if len(self._poses) % 1000 == 0:
            gc.collect()

    def flush(self) -> None:
        """Flush operation (no-op for memory cache, but triggers GC)."""
        gc.collect()

    def get_all(self) -> List[np.ndarray] | None:
        """Retrieve all cached poses.

        Returns:
            List of all cached poses, or None if empty
        """
        return self._poses if self._poses else None

    def clear(self) -> None:
        """Clear all cached data and free memory."""
        self._poses.clear()
        self._total_processed = 0
        gc.collect()

    def __len__(self) -> int:
        """Return number of cached poses."""
        return len(self._poses)
