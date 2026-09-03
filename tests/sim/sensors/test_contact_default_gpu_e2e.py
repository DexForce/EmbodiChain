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
"""Direct-GPU regression contract for the backend-neutral contact sensor."""

from __future__ import annotations

import pytest
import warp as wp

from embodichain.lab.sim.cfg import (
    DefaultPhysicsCfg,
    MassPropertiesCfg,
    RigidBodyPhysicsCfg,
    RigidObjectCfg,
)
from embodichain.lab.sim.sensors import ContactSensorCfg
from embodichain.lab.sim.shapes import CubeCfg
from embodichain.lab.sim.sim_manager import SimulationManager, SimulationManagerCfg

pytestmark = [pytest.mark.requires_sim, pytest.mark.gpu]


def test_default_direct_gpu_contact_sensor_reports_each_arena() -> None:
    wp.init()
    if not wp.is_cuda_available():
        pytest.skip("CUDA is required for the Direct-GPU contact E2E contract.")

    sim = SimulationManager(
        SimulationManagerCfg(
            headless=True,
            num_envs=2,
            arena_space=2.0,
            physics_cfg=DefaultPhysicsCfg(device="cuda:0"),
        )
    )
    try:
        sim.add_rigid_object(
            RigidObjectCfg(
                uid="ground",
                shape=CubeCfg(size=[1.0, 1.0, 0.1]),
                attrs=RigidBodyPhysicsCfg(),
                body_type="static",
            )
        )
        sim.add_rigid_object(
            RigidObjectCfg(
                uid="cube",
                shape=CubeCfg(size=[0.2, 0.2, 0.2]),
                attrs=RigidBodyPhysicsCfg(mass_props=MassPropertiesCfg(mass=1.0)),
                body_type="dynamic",
                init_pos=(0.0, 0.0, 0.4),
            )
        )
        sensor = sim.add_sensor(
            ContactSensorCfg(
                uid="contacts",
                rigid_uid_list=["cube"],
                filter_need_both_actor=False,
                max_contacts_per_env=16,
            )
        )

        counts = [0, 0]
        for _ in range(80):
            sim.update(step=1)
            sensor.update()
            counts = sensor._num_contacts_per_env.cpu().tolist()
            if all(count > 0 for count in counts):
                break

        assert not sensor.contact_capabilities.friction
        assert all(count > 0 for count in counts)
        data = sensor.get_data()
        actor_ids = data["user_ids"][data["is_valid"]]
        assert (actor_ids >= 0).any(dim=1).all()
        assert all(
            sensor.get_actor_info(actor_id).path.endswith("/cube")
            for actor_id in set(actor_ids[actor_ids >= 0].cpu().tolist())
        )
    finally:
        sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()
