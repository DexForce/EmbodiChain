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

import torch

from embodichain.lab.sim import SimulationManagerCfg
from embodichain.lab.sim.cfg import NewtonPhysicsCfg


def test_physics_runtime_fields_are_stored_on_physics_cfg() -> None:
    cfg = SimulationManagerCfg(
        headless=True,
        physics_dt=0.02,
        device=torch.device("cpu"),
    )

    assert cfg.physics_dt == 0.02
    assert cfg.device == torch.device("cpu")
    assert cfg.physics_cfg.physics_dt == 0.02
    assert cfg.physics_cfg.device == torch.device("cpu")

    serialized = cfg.to_dict()
    assert "physics_dt" not in serialized
    assert "device" not in serialized
    assert serialized["physics_cfg"]["physics_dt"] == 0.02
    assert serialized["physics_cfg"]["device"] == torch.device("cpu")


def test_simulation_manager_cfg_keeps_legacy_physics_accessors() -> None:
    cfg = SimulationManagerCfg(physics_cfg=NewtonPhysicsCfg())

    cfg.physics_dt = 0.005
    cfg.device = "cuda:0"

    assert cfg.physics_cfg.physics_dt == 0.005
    assert cfg.physics_cfg.device == "cuda:0"


def test_newton_physics_cfg_uses_device() -> None:
    cfg = NewtonPhysicsCfg(device="cuda:1")

    serialized = cfg.to_dict()
    assert serialized["device"] == "cuda:1"
    assert serialized["physics_dt"] == 1.0 / 100.0
