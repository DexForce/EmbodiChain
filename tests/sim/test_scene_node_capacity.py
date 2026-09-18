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
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from embodichain.lab.gym.utils.gym_utils import config_to_cfg
from embodichain.lab.sim import SimulationManager

pytestmark = pytest.mark.no_sim


@pytest.mark.parametrize("capacity", [None, 262144])
def test_training_node_capacity_reaches_world_config(capacity: int | None) -> None:
    data = {
        "id": "Test-Node-Capacity-v0",
        "robot": {},
        "env": {},
        "physics": "default",
        "device": "cpu",
        "render_cfg": {"renderer": "hybrid"},
        "headless": True,
    }
    if capacity is not None:
        data["scene_node_capacity"] = capacity
    cfg = config_to_cfg(data).sim_cfg
    assert cfg.scene_node_capacity == capacity
    manager = object.__new__(SimulationManager)
    manager._material_cache_dir = Path("unused-material-cache")
    manager.physics = SimpleNamespace(configure_world=Mock())
    world_cfg = manager._convert_sim_config(cfg)
    assert world_cfg.scene_node_capacity == (65536 if capacity is None else capacity)
