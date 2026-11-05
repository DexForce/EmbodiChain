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

from dataclasses import replace
from types import SimpleNamespace
import pytest
import torch

from embodichain.lab.sim.atomic_actions.bindings import JointPositionTarget
from embodichain.lab.sim.evidence import PhysicalObservationStamp
from embodichain.lab.task_program.integrations.simulation.bindings import (
    SimulationSceneBinding,
    SimulationRigidObjectBinding,
)
from embodichain.lab.task_program.integrations.simulation.handover_scene import (
    SimulationHandOverSceneProvider,
)


@pytest.mark.parametrize("projected", [False, True])
def test_native_handover_provider_binds_exact_receiver_and_complete_world_stamp(
    monkeypatch,
    projected,
):
    from embodichain.lab.task_program.integrations.simulation import handover_scene

    stamp = PhysicalObservationStamp("source", "default.contacts", 7, 1.0)
    current = [stamp]
    robot, simulation = object(), object()
    scene = SimulationSceneBinding(
        "scene", rigid_objects=(SimulationRigidObjectBinding("item", "native-item"),)
    )
    provider = SimulationHandOverSceneProvider(
        robot,
        simulation,
        scene_binding=scene,
        required_owner_uids=("native-item", "table"),
        read_source_stamp=lambda: current[0],
    )
    context = SimpleNamespace(
        robot=SimpleNamespace(
            qpos=torch.zeros(1, 4), root_pose=torch.eye(4)[None], timestamp=0.5
        ),
        scene=SimpleNamespace(timestamp=0.5, version=3),
        env_ids=torch.tensor([0]),
        world_source_stamp=stamp,
        observed_robot=None,
    )
    actual = context.robot
    if projected:
        context.observed_robot = actual
        context.robot = SimpleNamespace(
            qpos=torch.ones(1, 4), root_pose=torch.eye(4)[None], timestamp=0.5
        )
    targets = {
        "motion": JointPositionTarget("custom_arm", (0, 1, 2)),
        "grasp": JointPositionTarget("custom_hand", (3,)),
    }

    def endpoint(slot, name):
        assert slot == "destination"
        return SimpleNamespace(require_target=lambda cls: targets[name])

    request = SimpleNamespace(
        skill_id="hand_over",
        goal=SimpleNamespace(semantics=SimpleNamespace(entity_id="item")),
        binding=SimpleNamespace(endpoint=endpoint),
    )
    captured = []

    def capture(actual_robot, actual_sim, **kwargs):
        assert actual_robot is robot and actual_sim is simulation
        assert kwargs["read_native_revision"]() == 7
        captured.append(kwargs)
        return object()

    monkeypatch.setattr(handover_scene, "capture_native_hand_scene", capture)
    if projected:
        with pytest.raises(ValueError, match="projected joint starts"):
            provider.prepare(request, context)
        assert captured == []
        return
    provider.prepare(request, context)
    provider.prepare(request, context)
    first = captured[0]
    assert first["scope"].query_id != captured[1]["scope"].query_id
    assert first["receiver_control_part"] == "custom_arm"
    assert first["receiver_joint_ids"] == (3,)
    assert first["excluded_owner_uid"] == "native-item"
    assert first["required_owner_uids"] == ("native-item", "table")
    assert first["expected_qpos"] is actual.qpos
    assert first["root_pose"] is actual.root_pose
    # Equal revision alone cannot authorize a foreign source or another time.
    for changed in (
        replace(stamp, source_id="foreign"),
        replace(stamp, simulation_time=2.0),
        replace(stamp, revision=8),
    ):
        current[0] = changed
        with pytest.raises(ValueError, match="source changed"):
            first["read_native_revision"]()
        with pytest.raises(ValueError, match="source changed"):
            provider.prepare(request, context)
    assert len(captured) == 2
