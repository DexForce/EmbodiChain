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

from abc import ABC, abstractmethod
from typing import List
import numpy as np


all = [
    "BaseCache",
]


class BaseCache(ABC):
    """Abstract base class for workspace sampling cache strategies.

    Defines the interface for different caching mechanisms (memory, disk)
    used during workspace analysis sampling operations.
    """

    def __init__(self, batch_size: int = 5000, save_threshold: int = 10000000):
        """Initialize base cache parameters.

        Args:
            batch_size: Number of samples to process in each batch
            save_threshold: Number of samples to accumulate before triggering save/flush
        """
        self.batch_size = batch_size
        self.save_threshold = save_threshold
        self._total_processed = 0

    @abstractmethod
    def add(self, poses: List[np.ndarray]) -> None:
        """Add pose samples to the cache.

        Args:
            poses: List of 4x4 transformation matrices
        """
        pass

    @abstractmethod
    def flush(self) -> None:
        """Flush any pending data in the cache."""
        pass

    @abstractmethod
    def get_all(self) -> List[np.ndarray] | None:
        """Retrieve all cached poses.

        Returns:
            List of all cached 4x4 transformation matrices, or None if unavailable
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all cached data."""
        pass

    @property
    def total_processed(self) -> int:
        """Total number of poses processed by this cache."""
        return self._total_processed
