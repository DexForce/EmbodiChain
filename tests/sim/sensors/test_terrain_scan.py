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

"""Native headless terrain acquisition through registration and manager stepping."""

from __future__ import annotations

import pytest
import torch

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import DefaultPhysicsCfg, RigidObjectCfg
from embodichain.lab.sim.sensors import SensorCfg
from embodichain.lab.sim.shapes import CubeCfg

pytestmark = pytest.mark.requires_sim


def test_registered_terrain_scanner_samples_headless_ground_and_static_cube() -> None:
    manager = SimulationManager(
        SimulationManagerCfg(
            headless=True, num_envs=2, physics_cfg=DefaultPhysicsCfg(device="cpu")
        )
    )
    try:
        manager.add_rigid_object(
            RigidObjectCfg(
                uid="target",
                shape=CubeCfg(size=(0.5, 0.5, 0.4)),
                body_type="static",
                init_pos=(0.0, 0.0, 0.2),
            )
        )
        sensor = manager.add_sensor(
            SensorCfg.from_dict(
                {
                    "sensor_type": "TerrainScan",
                    "uid": "scan",
                    "size": [0.0, 0.0],
                    "mount": {"pos": [0.0, 0.0, 1.0]},
                    "target_uids": ["default_plane", "target"],
                }
            )
        )
        manager.update(step=1, render_final_step=False)
        torch.testing.assert_close(
            sensor.get_data()["distance"], torch.full((2, 1), 0.6)
        )
        assert sensor.get_data()["shape_id"].tolist() == [[1], [1]]
        assert sensor.get_sample_metadata()["valid"].tolist() == [True, True]
    finally:
        manager.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()
