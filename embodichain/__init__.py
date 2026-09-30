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

from pathlib import Path

embodichain_dir = Path(__file__).resolve().parent


# Read version from VERSION file
def _get_version():
    version_files = (embodichain_dir / "VERSION", embodichain_dir.parent / "VERSION")
    for version_file in version_files:
        try:
            return version_file.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            continue
    return "unknown"


__version__ = _get_version()
