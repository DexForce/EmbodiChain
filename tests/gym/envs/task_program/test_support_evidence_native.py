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

"""Small native Default support contract, independent of task qualification."""

from __future__ import annotations

import pytest
from embodichain.lab.gym.envs.task_program.bridge import EnvironmentStepClock
from embodichain.lab.sim.cfg import (
    DefaultPhysicsCfg,
    RigidObjectCfg,
    RigidBodyPhysicsCfg,
    MassPropertiesCfg,
)
from embodichain.lab.sim.shapes import CubeCfg
from embodichain.lab.sim.sim_manager import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.sensors.contact_sensor import ContactSensorCfg
from embodichain.lab.task_program.integrations.simulation.support_evidence import (
    SupportObservationRegistry,
)

pytestmark = [pytest.mark.requires_sim, pytest.mark.renderer]


def test_default_boxes_preserve_native_support_witness_and_clock_domains():
    sim = SimulationManager(
        SimulationManagerCfg(
            headless=True,
            num_envs=1,
            physics_dt=0.01,
            physics_cfg=DefaultPhysicsCfg(device="cpu"),
        )
    )
    registry = None
    try:
        sim.add_rigid_object(
            RigidObjectCfg(
                uid="support",
                shape=CubeCfg(size=[0.5, 0.5, 0.1]),
                body_type="static",
                attrs=RigidBodyPhysicsCfg(),
            )
        )
        sim.add_rigid_object(
            RigidObjectCfg(
                uid="payload",
                shape=CubeCfg(size=[0.06, 0.06, 0.06]),
                body_type="dynamic",
                attrs=RigidBodyPhysicsCfg(mass_props=MassPropertiesCfg(mass=0.1)),
                init_pos=(0.0, 0.0, 0.14),
            )
        )
        sim.update(physics_dt=0.01, step=3)
        offset = sim.simulation_time
        obj, reference = sim.get_rigid_object("payload"), sim.get_rigid_object(
            "support"
        )
        registry = SupportObservationRegistry(
            sim, acquisition_clock=lambda: sim.simulation_time
        )
        observation = registry.acquire(
            obj,
            reference,
            entity_id="item",
            reference_id="surface",
            observer_name="support",
            sensor_cfg=ContactSensorCfg(
                uid="native_support_test",
                rigid_uid_list=["payload", "support"],
                max_contacts_per_env=512,
            ),
        )
        clock = EnvironmentStepClock(0.04)
        observation.bind_clock(clock)
        calls = []
        fetch = observation.sensor._query.fetch

        def counted_fetch():
            calls.append(True)
            return fetch()

        observation.sensor._query.fetch = counted_fetch

        def window():
            before = len(calls)
            observation.sensor.begin_control_step()
            sim.update(
                physics_dt=0.01,
                step=4,
                after_substep=observation.sensor.update_physics_step,
            )
            clock.advance_after_env_step()
            result = observation.observe(
                obj.get_local_pose(to_matrix=True),
                reference.get_local_pose(to_matrix=True),
                support_contract_revision=1,
            )
            assert len(calls) - before == 4
            assert result.support.valid.all(), result.support.acquisition_errors
            assert result.support.timestamp == clock.now()
            assert sim.simulation_time - clock.now() == pytest.approx(offset, abs=1e-8)
            return result

        positive = retained = False
        for _ in range(100):
            result = window()
            positive |= bool(result.support.values[0])
            retained |= bool(result.metadata["rows"][0]["retained"])
            if positive and retained:
                break
        assert positive and retained
        moved = obj.get_local_pose(to_matrix=True)
        moved[:, 0, 3] += 0.25
        moved[:, 2, 3] += 0.3
        obj.set_local_pose(moved)
        assert not window().support.values.any()
    finally:
        if registry is not None:
            registry.close()
        sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()
