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
"""Live drawer geometry refresh and explicit per-call placement behavior."""

from __future__ import annotations

from types import SimpleNamespace
import torch
import pytest
from embodichain.lab.sim.scene.articulation_inputs import PrismaticBinding
from embodichain.lab.sim.scene.drawer_observation import DrawerPayloadObservation


def _observation():
    pose = torch.eye(4)[None]
    pose[:, 2, 3] = 0.02
    link = torch.eye(4)[None]
    qpos = torch.tensor([[0.2]])
    vertices = torch.tensor([[[-0.01, -0.01, -0.01], [0.01, 0.01, 0.01]]])
    obj = SimpleNamespace(
        get_vertices=lambda **kw: vertices, get_local_pose=lambda **kw: pose
    )
    art = SimpleNamespace(
        joint_names=["rail"], get_link_pose=lambda *a, **kw: link, get_qpos=lambda: qpos
    )
    simulation = SimpleNamespace(
        get_rigid_object=lambda _: obj, get_articulation=lambda _: art
    )
    binding = PrismaticBinding(
        "cabinet",
        "rail",
        "drawer",
        "base",
        "/drawer/grip",
        "0" * 64,
        1.0,
        (1.0, 0.0, 0.0),
        (0.0, 0.4),
    )
    return (
        DrawerPayloadObservation(
            simulation, binding, "payload", (-0.2, -0.2, 0.0), (0.2, 0.2, 0.2)
        ),
        pose,
        link,
        qpos,
    )


def test_live_opening_updates_exposed_floor_without_mutating_geometry():
    observation, pose, link, qpos = _observation()
    first = observation.target_poses()[0].clone()
    qpos[:] = 0.4
    second = observation.target_poses()[0]
    assert first[0, 0, 3] == pytest.approx(0.1)
    assert second[0, 0, 3] == pytest.approx(0.0)
    assert torch.equal(pose[0, :3, 3], torch.tensor([0.0, 0.0, 0.02]))
    assert torch.equal(link, torch.eye(4)[None])


def test_explicit_upright_view_does_not_change_later_ordinary_proposals():
    observation, _, _, _ = _observation()
    ordinary = observation.entry_pose_candidates()
    upright = observation.entry_pose_candidates(
        preserve_vertical=True, release_lift=0.0
    )
    repeated = observation.entry_pose_candidates()
    for a, b in zip(ordinary, repeated):
        for x, y in zip(a, b):
            assert torch.equal(x, y)
    assert upright[0][2][0, 2, 3] < ordinary[0][2][0, 2, 3]
    for row in upright:
        assert torch.allclose(row[2][0, :3, 2], torch.tensor([0.0, 0.0, 1.0]))


def test_close_containment_reads_current_payload_pose():
    observation, pose, _, _ = _observation()
    assert observation.containment_report(lateral_margin=0.0)["accepted"]
    pose[:, 0, 3] = 0.191
    assert not observation.containment_report(lateral_margin=0.0)["accepted"]


def test_drawer_observation_rejects_batch_rows_before_geometry_selection():
    observation, _, _, _ = _observation()
    observation._obj.get_local_pose = lambda **kw: torch.eye(4).repeat(2, 1, 1)
    with pytest.raises(ValueError, match="batch_size=1"):
        observation.target_poses()


def test_actual_drawer_provider_guards_original_observation_and_native_views():
    from embodichain.lab.sim.atomic_actions import (
        PlanningContext,
        RobotObservation,
        SceneSnapshot,
        TaskState,
    )
    from embodichain.lab.sim.evidence import PhysicalObservationStamp
    from embodichain.lab.task_program.integrations.simulation.drawer_planning import (
        DrawerInvocationBinding,
        SimulationDrawerPlanningProvider,
    )

    observation, pose, _, _ = _observation()
    qpos, root = torch.zeros(1, 3), torch.eye(4)[None]
    robot = SimpleNamespace(
        uid="robot", get_qpos=lambda: qpos, get_local_pose=lambda **kwargs: root
    )
    owners = {"cabinet": observation._art, "payload": observation._obj}
    simulation = SimpleNamespace(
        num_envs=1,
        physics_backend="default",
        get_robot=lambda uid: robot,
        get_articulation=owners.get,
        get_rigid_object=owners.get,
    )
    stamp = [PhysicalObservationStamp("source", "default.contacts", 1, 0.0)]
    provider = SimulationDrawerPlanningProvider(
        robot,
        simulation,
        bindings=(
            DrawerInvocationBinding(
                "place",
                "place",
                "target",
                "drawer",
                "payload",
                observation.binding,
                observation.lower,
                observation.upper,
            ),
        ),
        read_source_stamp=lambda: stamp[0],
    )
    context = PlanningContext(
        RobotObservation(
            0.0, qpos.clone(), torch.zeros_like(qpos), root_pose=root.clone()
        ),
        TaskState(1, "cpu"),
        SceneSnapshot(0.0, 0, {}),
        torch.tensor([0]),
        world_source_stamp=stamp[0],
    )
    request = SimpleNamespace(invocation_id="place", skill_id="place")
    view = provider.inputs(request, context)[0].observation
    assert view.containment_report(lateral_margin=0.0)["accepted"]
    pose[:, 0, 3] = 0.191
    assert not view.containment_report(lateral_margin=0.0)["accepted"]
    owners["payload"] = object()
    with pytest.raises(ValueError, match="ownership changed"):
        view.poses()
    owners["payload"] = observation._obj
    stamp[0] = PhysicalObservationStamp("source", "default.contacts", 2, 0.0)
    with pytest.raises(ValueError, match="complete native observation"):
        provider.inputs(request, context)
