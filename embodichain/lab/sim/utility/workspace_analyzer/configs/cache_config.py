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

from dataclasses import dataclass
from pathlib import Path


@dataclass
class CacheConfig:
    """Configuration for caching workspace analysis results."""

    enabled: bool = True
    """Whether to enable caching of analysis results."""

    cache_dir: Path | None = None
    """Directory to store cache files. If None, uses default system cache directory."""

    use_hash: bool = True
    """Whether to use hash-based cache keys (based on robot config and parameters)."""

    compression: bool = True
    """Whether to compress cache files to save disk space."""

    max_cache_size_mb: int = 1000
    """Maximum total size of cache directory in megabytes. Old files will be removed if exceeded."""

    cache_format: str = "npz"
    """Format for cache files. Options: 'npz' (numpy), 'pkl' (pickle), 'h5' (hdf5)."""
