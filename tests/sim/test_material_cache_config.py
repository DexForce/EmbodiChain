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

import pytest

from embodichain.lab.gym.utils.gym_utils import config_to_cfg
from embodichain.lab.sim.cfg import DefaultPhysicsCfg


@pytest.mark.parametrize("enabled", [False, True])
def test_material_cache_reaches_backend_arguments(enabled: bool) -> None:
    cfg = DefaultPhysicsCfg(cache_material=enabled)
    assert cfg.to_dexsim_args()["cache_material"] is enabled


@pytest.mark.parametrize("enabled", [False, True])
def test_training_config_accepts_material_cache(enabled: bool) -> None:
    cfg = config_to_cfg(
        {
            "id": "Test-Material-Cache-v0",
            "robot": {},
            "env": {},
            "physics": "default",
            "device": "cpu",
            "physics_config": {"cache_material": enabled},
        }
    )
    assert cfg.sim_cfg.physics_cfg.cache_material is enabled
