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


import os
from pathlib import Path

EMBODICHAIN_DOWNLOAD_PREFIX = os.environ.get(
    "EMBODICHAIN_DOWNLOAD_PREFIX",
    "http://192.168.3.120/CoreEngine/Data/embodychain_data/",
)

# When True, all assets are served flat (no subdirectory) under the prefix.
# Detected automatically: local/http servers typically use a flat layout.
_is_flat = not EMBODICHAIN_DOWNLOAD_PREFIX.startswith("https://")


def get_download_url(*path_parts: str) -> str:
    """Build a download URL from the configured prefix.

    On the local server assets are stored flat (no subdirectories),
    so the intermediate directory components are dropped.
    """
    if _is_flat:
        return EMBODICHAIN_DOWNLOAD_PREFIX + path_parts[-1]
    return os.path.join(EMBODICHAIN_DOWNLOAD_PREFIX, *path_parts)
EMBODICHAIN_DEFAULT_DATA_ROOT = os.environ.get(
    "EMBODICHAIN_DATA_ROOT", str(Path.home() / ".cache" / "embodichain_data")
)
EMBODICHAIN_DEFAULT_DATASET_ROOT = os.environ.get(
    "EMBODICHAIN_DATASET_ROOT", str(Path.home() / ".cache" / "embodichain_datasets")
)
EMBODICHAIN_DEFAULT_DATABASE_ROOT = str(
    Path.home() / ".cache" / "embodichain" / "database"
)
