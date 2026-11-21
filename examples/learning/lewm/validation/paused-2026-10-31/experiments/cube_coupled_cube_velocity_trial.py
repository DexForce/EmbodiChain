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
from unittest.mock import patch
from cube_coupled_source_trial import CoupledSourceCube
from embodichain.lab.sim import SimulationManager

SourceBase = CoupledSourceCube.__bases__[0]


class CubeVelocityOne(CoupledSourceCube):
    def _prepare_scene(self, **kwargs):
        original = SimulationManager.add_rigid_object

        def declare(sim, cfg):
            if cfg.attrs.rigid_props is not None:
                cfg.attrs.rigid_props.min_velocity_iters = 1
            return original(sim, cfg)

        with patch.object(SimulationManager, "add_rigid_object", declare):
            return SourceBase._prepare_scene(self, **kwargs)
