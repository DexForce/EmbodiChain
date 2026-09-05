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
"""Newton end-to-end contract for the backend-neutral contact sensor."""

from __future__ import annotations

import pytest
import warp as wp

from embodichain.lab.sim.cfg import (
    MassPropertiesCfg,
    NewtonPhysicsCfg,
    RigidBodyPhysicsCfg,
    RigidObjectCfg,
)
from embodichain.lab.sim.sensors import ContactSensorCfg
from embodichain.lab.sim.shapes import CubeCfg
from embodichain.lab.sim.sim_manager import SimulationManager, SimulationManagerCfg

pytestmark = [pytest.mark.requires_sim, pytest.mark.gpu]


def test_newton_contact_sensor_reports_each_arena() -> None:
    wp.init()
    if not wp.is_cuda_available():
        pytest.skip("CUDA is required for the Newton contact E2E contract.")

    sim = SimulationManager(
        SimulationManagerCfg(
            headless=True,
            num_envs=2,
            arena_space=2.0,
            physics_cfg=NewtonPhysicsCfg(
                device="cuda:0",
                num_substeps=1,
                use_cuda_graph=False,
                # Exercise a multi-point manifold so query-side reductions
                # cannot satisfy this regression contract accidentally.
                solver_cfg={
                    "solver_type": "mujoco_warp",
                    "enable_multiccd": True,
                },
            ),
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
                rigid_uid_list=["ground", "cube"],
                max_contacts_per_env=16,
            )
        )

        sim.update(step=120)
        sensor.update()

        assert sensor.contact_capabilities.geometry
        assert sensor.contact_capabilities.impulse
        counts = sensor._num_contacts_per_env.cpu().tolist()
        assert counts == [4, 4]
        data = sensor.get_data()
        assert data["is_valid"].sum(dim=1).cpu().tolist() == counts
        assert (data["impulse"][data["is_valid"]] > 1.0e-7).all()
        actor_ids = data["user_ids"][data["is_valid"]]
        assert all(
            sensor.get_actor_info(actor_id).path.endswith(("/ground", "/cube"))
            for actor_id in set(actor_ids.flatten().cpu().tolist())
        )
    finally:
        sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()
        from dexsim.engine.newton_physics import teardown_newton_physics

        teardown_newton_physics()
