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

"""Configured Pour staging preserves measured-entry geometry and retained effects."""

from __future__ import annotations
from dataclasses import replace
from types import SimpleNamespace as NS
from pathlib import Path
import json
import numpy as np
import pytest
import torch
import trimesh
from embodichain.lab.task_program.integrations.configured import (
    _decode_registered_lowerer,
)
from embodichain.lab.task_program.integrations.simulation.pour_transport import (
    PourTransportFactory,
    PourRegionTransportFactory,
)
from embodichain.lab.task_program.semantics import (
    SceneRegistry,
    SceneObjectRef,
    SceneEntityRegistration,
    RegisteredSemanticCall,
    SemanticEffectKind,
)
from embodichain.lab.sim.atomic_actions import (
    MoveHeldObjectOptions,
    PlanningContext,
    RobotObservation,
    TaskState,
    SceneSnapshot,
    EntityState,
    JointPositionTarget,
)
from embodichain.lab.sim.atomic_actions.goals import resolve_pose_goal


def fixture(region=False, cache_mode=None):
    reads = []
    data = {}
    registrations = []
    for name, size in (("source", (0.06, 0.08, 0.15)), ("receiver", (0.4, 0.3, 0.04))):
        mesh = trimesh.creation.box(extents=size)
        vertices = torch.tensor(mesh.vertices.copy()).float()[None]
        faces = torch.tensor(mesh.faces.copy()).long()[None]

        def get_vertices(*, env_ids, scale, name=name, vertices=vertices):
            assert env_ids == [0] and scale is True
            reads.append((name, "vertices"))
            return vertices.clone()

        def get_triangles(*, env_ids, name=name, faces=faces):
            assert env_ids == [0]
            reads.append((name, "faces"))
            return faces.clone()

        entity = NS(
            num_instances=1, get_vertices=get_vertices, get_triangles=get_triangles
        )
        registrations.append(
            SceneEntityRegistration(
                SceneObjectRef(name),
                state_provider=NS(
                    entity=entity,
                    observe=lambda **kw: pytest.fail(
                        "geometry capture must not read poses"
                    ),
                ),
            )
        )
        data[name] = (vertices, faces)
    registry = SceneRegistry(registrations)
    simulation = NS(
        num_envs=1,
        physics_backend="default",
        sim_config=NS(physics_cfg=NS(gravity=(0.0, 0.0, -9.81))),
    )
    root = torch.eye(4)[None]
    root[0, :3, 3] = torch.tensor([-0.5, 0.4, 0.1])
    robot = NS(
        get_solver=lambda part: NS(root_link_name="custom_root"),
        get_link_pose=lambda **kwargs: root.clone(),
    )
    cfg = {
        "kind": "pour_region_transport" if region else "pour_transport",
        "routes": [
            {
                "object_id": "source",
                "target_id": "stage",
                "pose": {"kind": "scene_entity", "entity_id": "receiver"},
            },
            {
                "object_id": "source",
                "target_id": "home",
                "pose": {
                    "kind": "pose",
                    "position": [0.1, -0.2, 0.3],
                    "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
            },
        ],
        "staging_routes": [
            {
                "object_id": "source",
                "target_id": "stage",
                "receiver_id": "receiver",
                "clearance_m": 0.05,
            }
        ],
        **({"proxy_position_tolerance_m": 0.03} if region else {}),
    }
    factory = _decode_registered_lowerer(cfg, path="lowerer")
    engine = NS(robot=robot)
    if cache_mode is not None:
        from embodichain.lab.sim.scene.pour_region import PourRegionGeometry

        cached_vertices = {
            key: value[0][0].numpy().copy() for key, value in data.items()
        }
        cached_region = PourRegionGeometry(
            cached_vertices,
            {"receiver": data["receiver"][1][0].numpy()},
            (0.0, 0.0, -9.81),
            0.03,
        )

        def region_lookup(object_id, receiver_id, budget):
            assert (object_id, receiver_id, budget) == ("source", "receiver", 0.03)
            if cache_mode == "conflict":
                raise ValueError("declared cached budget conflicts")
            return cached_region

        provider = NS(
            scene_vertices=cached_vertices,
            gravity_world=(
                (0.0, 0.0, -1.0) if cache_mode == "gravity" else (0.0, 0.0, -9.81)
            ),
            region_geometry=region_lookup,
        )
        engine.planning_services = NS(pour_grasp_projection_provider=provider)
    lowerer = factory.create(
        simulation=simulation, robot=robot, scene_registry=registry, engine=engine
    )
    source = torch.eye(4)[None]
    source[0, :3, 3] = torch.tensor([0.1, -0.2, 0.3])
    receiver = torch.eye(4)[None]
    receiver[0, :3, 3] = torch.tensor([0.4, 0.2, 0.3])
    context = PlanningContext(
        robot=RobotObservation(
            timestamp=0.0, qpos=torch.zeros(1, 2), qvel=torch.zeros(1, 2)
        ),
        task=TaskState.empty(1, "cpu"),
        scene=SceneSnapshot(
            timestamp=0.0,
            version=1,
            entities={"source": EntityState(source), "receiver": EntityState(receiver)},
        ),
        env_ids=torch.tensor([0]),
    )
    bound = NS(
        binding=NS(
            action_binding=NS(
                endpoint=lambda *args: NS(
                    require_target=lambda cls: JointPositionTarget("custom_arm", (0, 1))
                )
            )
        )
    )
    return factory, lowerer, context, bound, reads, data


@pytest.mark.parametrize("region", [False, True])
def test_pour_transport_captures_only_declared_geometry_and_uses_fresh_poses(region):
    factory, lowerer, context, bound, reads, data = fixture(region)
    assert type(factory) is (
        PourRegionTransportFactory if region else PourTransportFactory
    )
    assert reads == [("source", "vertices"), ("receiver", "vertices")] + (
        [("receiver", "faces")] if region else []
    )
    call = RegisteredSemanticCall(
        call_id=factory.call_id,
        arguments={"object": "source", "target": "stage", "reference": "receiver"},
    )
    initial = context.scene.entities["source"].pose.clone()
    result = lowerer.lower(
        call, context=context, bound=bound, option_template=MoveHeldObjectOptions()
    )
    assert result.registered_effect.effect_kind is SemanticEffectKind.ATTACH
    assert result.goal.object_target_pose.entity_id == "receiver"
    pose = resolve_pose_goal(result.goal.object_target_pose, context, name="stage")
    torch.testing.assert_close(pose[:, :3, :3], initial[:, :3, :3])
    assert (
        lowerer.pick_lookahead_targets(
            call,
            picked_object=SceneObjectRef("source"),
            bound=bound,
            previous_target=None,
        )
        == ()
    )
    home = replace(call, arguments={"object": "source", "target": "home"})
    returned = lowerer.lower(
        home, context=context, bound=bound, option_template=MoveHeldObjectOptions()
    )
    torch.testing.assert_close(
        returned.goal.object_target_pose[:3, 3], torch.tensor([0.1, -0.2, 0.3])
    )
    assert (
        len(
            lowerer.pick_lookahead_targets(
                home,
                picked_object=SceneObjectRef("source"),
                bound=bound,
                previous_target=None,
            )
        )
        == 1
    )
    old = lowerer.scene_vertices["source"].copy()
    data["source"][0].fill_(99)
    np.testing.assert_array_equal(lowerer.scene_vertices["source"], old)
    assert not lowerer.scene_vertices["source"].flags.writeable
    assert reads == [("source", "vertices"), ("receiver", "vertices")] + (
        [("receiver", "faces")] if region else []
    )


def test_pour_transport_rejects_crosswired_stages_and_unavailable_region():
    factory, lowerer, context, bound, _, _ = fixture(True)
    with pytest.raises(ValueError, match="exact configured"):
        replace(
            factory,
            staging_routes=(
                replace(factory.staging_routes[0], receiver_id="other_receiver"),
            ),
        )
    geometry = lowerer.region_geometry
    too_large = type(geometry)(
        geometry.vertices, geometry.faces, geometry.gravity_world, 10.0
    )
    with pytest.raises(ValueError, match="inset"):
        too_large.staging_pose(
            "source", "receiver", np.eye(4), np.eye(4), np.zeros(3), clearance_m=0.05
        )


@pytest.mark.parametrize("region", [False, True])
def test_pour_transport_reuses_shared_provider_geometry_without_native_recapture(
    region,
):
    _, cached, context, bound, reads, _ = fixture(region, cache_mode="shared")
    _, original, _, _, _, _ = fixture(region)
    assert reads == []
    call = RegisteredSemanticCall(
        call_id="simulation.move_held_object",
        arguments={"object": "source", "target": "stage"},
    )
    a = cached.lower(
        call, context=context, bound=bound, option_template=MoveHeldObjectOptions()
    )
    b = original.lower(
        call, context=context, bound=bound, option_template=MoveHeldObjectOptions()
    )
    torch.testing.assert_close(
        a.goal.object_target_pose.world_displacement,
        b.goal.object_target_pose.world_displacement,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        a.goal.object_target_pose.world_orientation,
        b.goal.object_target_pose.world_orientation,
        rtol=0,
        atol=0,
    )


def test_conflicting_pour_cache_does_not_refresh_or_substitute_geometry():
    with pytest.raises(ValueError, match="different configured gravity"):
        fixture(False, cache_mode="gravity")
    with pytest.raises(ValueError, match="cached budget conflicts"):
        fixture(True, cache_mode="conflict")
