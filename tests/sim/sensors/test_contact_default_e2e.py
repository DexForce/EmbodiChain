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
"""Default-PhysX contact-point multiplicity regression contract."""

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

pytestmark = pytest.mark.requires_sim


@pytest.mark.parametrize(
    ("device", "expected_contacts_per_env"),
    [
        pytest.param("cpu", 4, id="cpu"),
        pytest.param("cuda:0", 4, marks=pytest.mark.gpu, id="direct-gpu"),
    ],
)
def test_default_contact_sensor_preserves_physx_contact_multiplicity(
    device: str,
    expected_contacts_per_env: int,
) -> None:
    wp.init()
    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("CUDA is required for the Direct-GPU contact E2E contract.")

    sim = SimulationManager(
        SimulationManagerCfg(
            headless=True,
            num_envs=2,
            arena_space=2.0,
            physics_cfg=DefaultPhysicsCfg(device=device),
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
            if counts == [expected_contacts_per_env] * 2:
                break

        assert not sensor.contact_capabilities.friction
        assert counts == [expected_contacts_per_env] * 2
        data = sensor.get_data()
        valid = data["is_valid"]
        assert (data["impulse"][valid] > 1.0e-7).all()
        positions = data["position"][valid]
        assert positions[:, 0].abs().max().item() < 0.2
        assert (positions[:, 2] - 0.05).abs().max().item() < 0.02
        actor_ids = data["user_ids"][valid]
        assert (actor_ids >= 0).any(dim=1).all()
        cube_actor_ids = set(sensor.item_user_ids.cpu().tolist())
        assert all(
            any(actor_id in cube_actor_ids for actor_id in pair)
            for pair in actor_ids.cpu().tolist()
        )
        assert all(
            sensor.get_actor_info(actor_id).path.endswith("/cube")
            for actor_id in cube_actor_ids
        )
    finally:
        sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()
