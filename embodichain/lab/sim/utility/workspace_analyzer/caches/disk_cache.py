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
import os
from typing import List
import numpy as np
from tqdm import tqdm


from embodichain.lab.sim.utility.workspace_analyzer.caches.base_cache import BaseCache

from embodichain.utils import logger

all = [
    "DiskCache",
]


class DiskCache(BaseCache):
    """Disk-based cache for workspace sampling.

    Saves pose samples to disk in batches to minimize memory usage.
    Suitable for large-scale sampling operations.

    Default cache location: ~/.cache/embodichain/workspace_analyzer/
    """

    @staticmethod
    def get_default_cache_dir(subdir: str = "default") -> str:
        """Get default cache directory in user's home.

        Args:
            subdir: Subdirectory name under workspace_analyzer cache

        Returns:
            Path to cache directory: ~/.cache/embodichain/workspace_analyzer/{subdir}
        """
        cache_home = os.path.expanduser("~/.cache")
        cache_dir = os.path.join(
            cache_home, "embodichain", "workspace_analyzer", subdir
        )
        return cache_dir

    def __init__(
        self,
        save_dir: str | None = None,
        batch_size: int = 5000,
        save_threshold: int = 10000000,
        use_cached: bool = True,
    ):
        """Initialize disk cache.

        Args:
            save_dir: Directory path for saving batch files.
                     If None, uses ~/.cache/embodichain/workspace_analyzer/default
            batch_size: Number of samples per batch
            save_threshold: Number of samples to accumulate before writing to disk
            use_cached: Whether to use existing cached files if available
        """
        super().__init__(batch_size, save_threshold)

        # Use default cache dir if not specified
        if save_dir is None:
            import time

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            save_dir = self.get_default_cache_dir(subdir=f"session_{timestamp}")
            logger.log_info(f"Using default cache directory: {save_dir}")

        self.save_dir = save_dir
        self.use_cached = use_cached
        self._buffer: List[np.ndarray] = []
        self._batch_count = 0

        # Create batches directory
        self._batches_dir = os.path.join(save_dir, "batches")
        os.makedirs(self._batches_dir, exist_ok=True)

        # Check for existing cached data
        if use_cached and self._has_cached_data():
            logger.log_info(f"Found existing cached data in {self._batches_dir}")

    def _has_cached_data(self) -> bool:
        """Check if cached batch files exist.

        Returns:
            True if cached .npy files exist
        """
        if not os.path.exists(self._batches_dir):
            return False
        npy_files = [f for f in os.listdir(self._batches_dir) if f.endswith(".npy")]
        return len(npy_files) > 0

    def add(self, poses: List[np.ndarray]) -> None:
        """Add poses to buffer and save to disk when threshold is reached.

        Args:
            poses: List of 4x4 transformation matrices
        """
        self._buffer.extend(poses)
        self._total_processed += len(poses)

        # Write to disk when buffer reaches threshold
        if len(self._buffer) >= self.save_threshold:
            self._save_batch()

    def _save_batch(self) -> None:
        """Save current buffer to disk as a batch file."""
        if not self._buffer:
            return

        batch_path = os.path.join(
            self._batches_dir, f"batch_{self._batch_count:04d}.npy"
        )
        np.save(batch_path, np.array(self._buffer))
        logger.log_info(
            f"Saved batch {self._batch_count}: "
            f"{len(self._buffer)} poses -> {batch_path}"
        )

        self._batch_count += 1
        self._buffer.clear()
        gc.collect()

    def flush(self) -> None:
        """Flush any remaining data in buffer to disk."""
        if self._buffer:
            self._save_batch()

    def get_all(self) -> List[np.ndarray] | None:
        """Load and merge all batch files from disk.

        Returns:
            List of all cached poses merged from batch files, or None if no data
        """
        # First flush any pending data
        self.flush()

        # Get all batch files
        npy_files = sorted(
            [f for f in os.listdir(self._batches_dir) if f.endswith(".npy")]
        )

        if not npy_files:
            return None

        logger.log_info(f"Loading {len(npy_files)} batch files...")
        all_poses = []

        for npy_file in tqdm(npy_files, desc="Merging batches"):
            batch_path = os.path.join(self._batches_dir, npy_file)
            try:
                batch_data = np.load(batch_path)
                all_poses.extend(batch_data)
            except Exception as e:
                logger.log_warning(f"Error loading {npy_file}: {str(e)}")

        logger.log_info(f"Loaded {len(all_poses)} total poses")
        return all_poses if all_poses else None

    def clear(self) -> None:
        """Clear all cached data and remove batch files."""
        self._buffer.clear()
        self._batch_count = 0
        self._total_processed = 0

        # Remove all batch files
        if os.path.exists(self._batches_dir):
            for file in os.listdir(self._batches_dir):
                if file.endswith(".npy"):
                    os.remove(os.path.join(self._batches_dir, file))

        gc.collect()

    def get_batch_count(self) -> int:
        """Get number of batches written to disk.

        Returns:
            Number of batch files on disk
        """
        if not os.path.exists(self._batches_dir):
            return 0
        return len([f for f in os.listdir(self._batches_dir) if f.endswith(".npy")])
