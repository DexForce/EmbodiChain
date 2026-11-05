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
"""Record the loaded SDK source and native library identities outside timers."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys

__all__ = ["loaded_sdk_identity"]


def loaded_sdk_identity() -> dict[str, dict[str, str]]:
    """Hash loaded DexSim/DexUni modules and native SDK libraries."""
    files = {}
    for name, module in tuple(sys.modules.items()):
        if not (
            name.startswith("dexsim.")
            or name.startswith("newton._src.solvers.mjvbd_v2")
        ):
            continue
        path = getattr(module, "__file__", None)
        if path and Path(path).is_file():
            files[name] = {
                "path": path,
                "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            }
    for line in Path("/proc/self/maps").read_text().splitlines():
        path = line.split()[-1]
        if "libDexSim" in path and Path(path).is_file():
            files[Path(path).name] = {
                "path": path,
                "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            }
    return files
