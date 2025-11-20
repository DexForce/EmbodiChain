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

from pathlib import Path
import logging


def ensure_directory_exists(path: str, logger: logging.Logger = None):
    """Ensure the directory exists, create if not."""
    try:
        path_obj = Path(path)
        path_obj.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        if logger:
            logger.error(f"Failed to create directory {path}: {e}")
        else:
            print(f"Failed to create directory {path}: {e}")
