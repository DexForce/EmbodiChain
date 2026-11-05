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
from embodichain.lab.sim.sensors.contact.sensor import ContactSensorCfg
from embodichain.lab.sim.sensors.support_observation import (
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

        # Exercise the typed standard policy against the same actual native
        # bodies, including a cold acquisition window and normal substep holds.
        from types import SimpleNamespace
        import torch
        from embodichain.lab.task_program import (
            TaskProgramCompiler,
            decode_task_program,
        )
        from embodichain.lab.task_program.integrations.simulation.bindings import (
            SimulationSceneBinding,
            SimulationRigidObjectBinding,
        )
        from embodichain.lab.task_program.integrations.simulation.policies import (
            SimulationSegmentPolicyPort,
        )
        from embodichain.lab.task_program.integrations.simulation.stability import (
            SimulationStabilityConditionCfg,
            SupportSensorDeclarationCfg,
        )

        registry.release("payload", "support")
        restored = obj.get_local_pose(to_matrix=True)
        restored[:, 0, 3] = 0.0
        restored[:, 2, 3] = 0.14
        obj.set_local_pose(restored)
        binding = SimulationSceneBinding(
            "native_support_policy",
            rigid_objects=(
                SimulationRigidObjectBinding("item", "payload"),
                SimulationRigidObjectBinding("surface", "support"),
            ),
        )
        qpos = torch.zeros(1, 1)
        sensors = {}
        port = SimulationSegmentPolicyPort(
            sim,
            SimpleNamespace(get_qpos=lambda **kw: qpos.clone()),
            binding,
            settle_presets={
                "stable": SimulationStabilityConditionCfg(
                    entity="item",
                    kind="stack",
                    reference="surface",
                    local_axis=(0, 0, 1),
                    reference_axis=(0, 0, 1),
                    reference_half_extents=(0.25, 0.25),
                    object_bottom=-0.03,
                    reference_top=0.05,
                    duration=0.08,
                    timeout=2.0,
                    support_contract_revision=1,
                    support_sensor=SupportSensorDeclarationCfg(
                        uid="native_support_test"
                    ),
                )
            },
            step_dt=0.04,
            support_registry=registry,
            sensor_suite=sensors,
        )
        compiled = TaskProgramCompiler(binding.declare()).compile(
            decode_task_program(
                dict(
                    program_id="native_support_policy",
                    targets={},
                    integration=dict(
                        scene_registry="native_support_policy",
                        robot_profile="robot",
                        runtime_preset="default",
                    ),
                    program=dict(
                        kind="segment",
                        name="support",
                        steps=dict(
                            kind="invoke", call=dict(kind="pick", object="item")
                        ),
                        post=[dict(kind="wait_stable", entity="item", preset="stable")],
                    ),
                )
            )
        )
        segment = next(compiled.iter_segments())
        policy = segment.post_policies[0]
        clock = EnvironmentStepClock(0.04)
        port.bind_clock(clock)
        sensor = sensors["native_support_test"]
        commands = 0
        for command in port.actions(
            policy, segment=segment, active_mask=torch.ones(1, dtype=torch.bool)
        ):
            assert torch.equal(command, qpos)
            before = len(calls)
            sensor.begin_control_step()
            sim.update(
                physics_dt=0.01, step=4, after_substep=sensor.update_physics_step
            )
            clock.advance_after_env_step()
            assert len(calls) - before == 4
            commands += 1
        assert commands >= 2
        assert port.post_policy_result(policy, segment=segment).tolist() == [True]
        metadata = port.post_policy_metadata(policy, segment=segment)
        assert metadata["support_unavailable_observations"][0] >= 1
        assert metadata["support_result_available"] == [True]
        assert metadata["clock_ended_at"] == clock.now()

    finally:
        if registry is not None:
            registry.close()
        sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()
