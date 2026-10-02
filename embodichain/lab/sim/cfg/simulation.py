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

"""Compatibility facade for rendering and physics configuration.

New code should import rendering types from the rendering module and physics
types from the physics module. The historical module remains available for
callers that import both configuration domains from ``simulation``.
"""

from __future__ import annotations

from .physics import (
    DefaultPhysicsCfg,
    GPUMemoryCfg,
    NewtonCollisionPipelineCfg,
    NewtonPhysicsCfg,
    PhysicsBackendCfg,
    _gravity_vector,
    _newton_solver_cfg_to_dexsim,
    _normalize_newton_solver_type,
    physics_backend_from_cfg,
    physics_cfg_for_backend,
    validate_physics_cfg,
)
from .rendering import (
    DenoisingCfg,
    DenoisingMode,
    DLSSCfg,
    NRDCfg,
    RenderCfg,
)

__all__ = [
    "DenoisingCfg",
    "DenoisingMode",
    "DLSSCfg",
    "NRDCfg",
    "RenderCfg",
    "GPUMemoryCfg",
    "PhysicsBackendCfg",
    "DefaultPhysicsCfg",
    "NewtonCollisionPipelineCfg",
    "NewtonPhysicsCfg",
    "physics_cfg_for_backend",
    "physics_backend_from_cfg",
    "validate_physics_cfg",
]
