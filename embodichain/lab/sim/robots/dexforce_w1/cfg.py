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

"""Compatibility alias for :mod:`embodichain_robots.dexforce_w1.cfg`."""

from __future__ import annotations

from importlib import import_module
import sys

__all__ = ["DexforceW1Cfg"]

if __name__ == "__main__":
    from pathlib import Path
    import runpy

    sys.path[0] = str(Path(__file__).resolve().parents[5])
    runpy.run_module("embodichain_robots.dexforce_w1.cfg", run_name="__main__")
else:
    sys.modules[__name__] = import_module("embodichain_robots.dexforce_w1.cfg")
