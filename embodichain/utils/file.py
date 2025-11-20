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

import os
import re

from typing import Optional, List


def get_all_files_in_directory(
    directory: str,
    exts: Optional[List[str]] = None,
    patterns: Optional[List[str]] = None,
) -> List[str]:
    """Get all files in a directory with optional filtering by extensions or regex patterns.

    Args:
        directory (str): The directory to search for files.
        exts (Optional[List[str]]): List of file extensions to filter by. If None, all files are returned.
        patterns (Optional[List[str]]): List of regex patterns to match file names. If None, no pattern matching is applied.

    Returns:
        List[str]: List of file paths in the directory matching the specified extensions or patterns.
    """
    all_files = []
    compiled_patterns = (
        [re.compile(pattern) for pattern in patterns] if patterns else []
    )

    for root, _, files in os.walk(directory):
        for file in files:
            match_ext = exts is None or any(
                file.lower().endswith(ext.lower()) for ext in exts
            )
            match_pattern = not compiled_patterns or any(
                pattern.search(file) for pattern in compiled_patterns
            )

            if match_ext and match_pattern:
                all_files.append(os.path.join(root, file))
    return all_files
