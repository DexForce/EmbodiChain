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

"""Native acceleration acquisition through a completed Newton step."""

from __future__ import annotations

import pytest
import torch

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import NewtonPhysicsCfg, RigidObjectCfg
from embodichain.lab.sim.sensors import IMUCfg, PVACfg, SensorCfg
from embodichain.lab.sim.shapes import CubeCfg
from embodichain.utils.math import matrix_from_quat

pytestmark = [pytest.mark.requires_sim, pytest.mark.renderer]


def test_newton_native_inertial_samples_offset_and_reset() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA MuJoCo Warp is required")
    sim = SimulationManager(
        SimulationManagerCfg(
            headless=True,
            num_envs=2,
            physics_dt=0.01,
            physics_cfg=NewtonPhysicsCfg(
                num_substeps=3, solver_cfg={"solver_type": "mujoco_warp"}
            ),
        )
    )
    try:
        body = sim.add_rigid_object(
            RigidObjectCfg(
                uid="payload",
                shape=CubeCfg(size=(0.1, 0.1, 0.1)),
                body_type="dynamic",
                init_pos=(0.0, 0.0, 1.0),
            )
        )
        pva = sim.add_sensor(
            PVACfg(
                uid="pva",
                asset_uid=body.uid,
                acceleration_source="native",
                sample_period=0.02,
                offset=SensorCfg.OffsetCfg(pos=(0.2, 0.0, 0.0)),
            )
        )
        imu = sim.add_sensor(
            IMUCfg(uid="imu", asset_uid=body.uid, acceleration_source="auto")
        )
        device = sim.device
        pva.update()
        assert not pva.get_data()["acceleration_valid"].any()
        body.set_com_pose(torch.tensor([[0.1, 0, 0, 0, 0, 0, 1]] * 2, device=device))
        body.set_velocity(ang_vel=torch.tensor([[0.0, 0, 2], [0, 0, 0]], device=device))
        sim.update(step=1, render_final_step=False)
        reading = pva.get_data()
        assert reading["acceleration_valid"].tolist() == [True, True]
        assert imu.get_data_descriptors()["acceleration_valid"]["source"] == "native"
        rotation = matrix_from_quat(reading["pose"][0, 3:])
        gravity = torch.tensor([0.0, 0.0, -9.81], device=device)
        expected = torch.stack((gravity - 0.4 * rotation[:, 0], gravity))
        torch.testing.assert_close(
            reading["linear_acceleration"], expected, atol=0.02, rtol=0
        )
        torch.testing.assert_close(
            imu.get_data()["specific_force"][1], torch.zeros_like(gravity)
        )
        # The first step after acquisition is held by sample_period. Reset
        # must suppress direct acquisition of that step's pre-reset report.
        sampled_step = sim.physics_step_index
        sim.update(step=1, render_final_step=False)
        assert pva.get_sample_metadata()["physics_step"].tolist() == [sampled_step] * 2
        sim.reset_objects_state(env_ids=[0])
        pva.update()
        assert pva.get_data()["acceleration_valid"].tolist() == [False, True]
        assert pva.get_sample_metadata()["valid"].tolist() == [False, True]
        sim.update(step=1, render_final_step=False)
        assert pva.get_data()["acceleration_valid"].tolist() == [True, True]
    finally:
        sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()
