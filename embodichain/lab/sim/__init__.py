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

"""EmbodiChain's simulation core.

Organized around the ``SimulationManager`` (the DexSim scene handle), the scene-object hierarchy, sensors, IK solvers, motion planners, the atomic-action layer, and shared configuration types.
"""

from __future__ import annotations

from .material import (
    VisualMaterialCfg,
    VisualMaterial,
    VisualMaterialInst,
    ReuseSegmentState,
)
from .common import BatchEntity
from .profiler import Profiler, ProfilerCfg

from .sim_manager import *

__all__ = [
    "VisualMaterialCfg",
    "VisualMaterial",
    "VisualMaterialInst",
    "ReuseSegmentState",
    "BatchEntity",
    "Profiler",
    "ProfilerCfg",
    "SimulationManager",
    "SimulationManagerCfg",
    "SIM_CACHE_DIR",
    "MATERIAL_CACHE_DIR",
    "CONVEX_DECOMP_DIR",
    "REACHABLE_XPOS_DIR",
]


from .utility.dynamic_pybind import init_dynamic_pybind

init_dynamic_pybind()
