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

from typing import Literal

from embodichain.lab.sim.utility.workspace_analyzer.caches.base_cache import BaseCache
from embodichain.lab.sim.utility.workspace_analyzer.caches.memory_cache import (
    MemoryCache,
)
from embodichain.lab.sim.utility.workspace_analyzer.caches.disk_cache import DiskCache
from embodichain.lab.sim.utility.workspace_analyzer.configs.cache_config import (
    CacheConfig,
)


all = [
    "CacheManager",
]


class CacheManager:
    """Factory and manager for workspace sampling caches.

    Provides a unified interface for creating and managing different
    cache strategies (memory vs disk).
    """

    @staticmethod
    def create_cache(
        cache_mode: Literal["memory", "disk"],
        save_dir: str | None = None,
        batch_size: int = 5000,
        save_threshold: int = 10000000,
        use_cached: bool = True,
    ) -> BaseCache:
        """Create a cache instance based on the specified mode.

        Args:
            cache_mode: Caching strategy - "memory" or "disk"
            save_dir: Directory for disk cache. If None in disk mode, uses
                     ~/.cache/embodichain/workspace_analyzer/session_TIMESTAMP
            batch_size: Number of samples per batch
            save_threshold: Threshold for saving/flushing data
            use_cached: Whether to use existing cached data (disk mode only)

        Returns:
            Configured cache instance

        Raises:
            ValueError: If cache_mode is invalid
        """
        if cache_mode not in ["memory", "disk"]:
            raise ValueError(
                f"Invalid cache_mode '{cache_mode}'. Must be 'memory' or 'disk'"
            )

        if cache_mode == "disk":

            return DiskCache(
                save_dir=save_dir,
                batch_size=batch_size,
                save_threshold=save_threshold,
                use_cached=use_cached,
            )
        else:  # memory
            return MemoryCache(batch_size=batch_size, save_threshold=save_threshold)

    @staticmethod
    def create_cache_from_config(config: CacheConfig) -> BaseCache | None:
        """Create a cache instance from a CacheConfig object.

        Args:
            config: CacheConfig instance with cache settings

        Returns:
            Configured cache instance if enabled, None otherwise
        """
        if not config.enabled:
            return None

        cache_mode = "disk" if config.cache_dir is not None else "memory"
        save_dir = str(config.cache_dir) if config.cache_dir else None

        return CacheManager.create_cache(
            cache_mode=cache_mode,
            save_dir=save_dir,
            batch_size=5000,  # Default batch size
            save_threshold=config.max_cache_size_mb
            * 1024
            * 1024,  # Convert MB to bytes
            use_cached=True,
        )
