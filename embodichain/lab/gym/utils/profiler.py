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

"""Backward-compatible environment aliases for the simulation profiler.

The implementation lives in :mod:`embodichain.lab.sim.profiler` so standalone
simulation code and Gym environments can share the same profiler instance.
"""

from __future__ import annotations

from embodichain.lab.sim.profiler import Profiler, ProfilerCfg
from embodichain.utils import logger

# Preserve the old module-level logger hook used by downstream report capture.
EnvProfiler = Profiler
EnvProfilerCfg = ProfilerCfg

__all__ = ["EnvProfilerCfg", "EnvProfiler"]
