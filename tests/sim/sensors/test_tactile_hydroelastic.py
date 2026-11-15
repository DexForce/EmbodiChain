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


"""Completed Newton face projection, ownership and model rebind."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import warp as wp
from tensordict import TensorDict

from embodichain.lab.sim.sensors._hydroelastic_grid import _HydroelasticPadProjection


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.gpu)]
)
def test_scatter_same_arena_faces_and_report_truncation(device: str) -> None:
    wp.init()
    points = torch.tensor(
        [[-0.01, -0.01, 0], [0.01, -0.01, 0], [-0.01, 0.01, 0], [0.01, 0.01, 0]],
        device=device,
    )
    centers = torch.tensor(
        [[1.01, -0.01, 0], [1.01, -0.01, 0], [2.99, 0.01, 0], [1.01, -0.01, 0]],
        device=device,
    )
    offset = torch.tensor(
        [[-0.001, 0, 0], [0.001, -0.001, 0], [0, 0.001, 0]], device=device
    )
    snapshot = SimpleNamespace(
        capacity=4,
        face_count=torch.tensor([5], device=device, dtype=torch.int32),
        vertices=centers[:, None] + offset,
        shape_pairs=torch.tensor(
            [[0, 1], [1, 0], [3, 2], [0, 3]], device=device, dtype=torch.int32
        ),
        separation=torch.tensor([-0.001, -0.003, 0.002, -0.010], device=device),
        body_poses=torch.tensor([[1, 0, 0, 0, 0, 0, 1.0]], device=device),
        collision_generation=17,
        substep=2,
    )
    fixed = torch.eye(4, device=device).repeat(2, 1, 1)
    fixed[1, 0, 3] = -3
    source = SimpleNamespace(
        read=lambda: snapshot,
        pad_env=torch.tensor([0, -1, 1, -1], device=device, dtype=torch.int32),
        target_env=torch.tensor([-1, 0, -1, 1], device=device, dtype=torch.int32),
        body_ids=torch.tensor([0, -1], device=device, dtype=torch.int32),
        static_inverse=fixed,
        mount_inverse=wp.mat44(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1),
    )
    grid = TensorDict(
        {
            "surface_face_count": torch.empty(
                (2, 2, 2), device=device, dtype=torch.int32
            ),
            "surface_separation": torch.empty((2, 2, 2), device=device),
            "surface_complete": torch.empty(2, device=device, dtype=torch.bool),
            "surface_generation": torch.empty(2, device=device, dtype=torch.int64),
            "surface_substep": torch.empty(2, device=device, dtype=torch.int64),
        },
        batch_size=[2],
    )
    projection = _HydroelasticPadProjection(
        source,
        points,
        torch.tensor([0.0, 0.0, 1.0], device=device).expand_as(points),
        grid,
        (0.04, 0.04),
        (2, 2),
        True,
        0.02,
    )
    projection.sample()
    assert grid["surface_face_count"].tolist() == [[[0, 2], [0, 0]], [[0, 0], [1, 0]]]
    torch.testing.assert_close(
        grid["surface_separation"],
        torch.tensor([[[0, -0.003], [0, 0]], [[0, 0], [0.002, 0]]], device=device),
    )
    assert grid["surface_complete"].tolist() == [False, False]
    assert grid["surface_generation"].tolist() == [17, 17]
    assert grid["surface_substep"].tolist() == [2, 2]

    snapshot.face_count.zero_()
    projection.sample()
    assert grid["surface_complete"].tolist() == [True, True]
    assert not grid["surface_face_count"].any()
    assert not grid["surface_separation"].any()
    source.read = lambda: None
    projection.sample()
    assert not grid["surface_complete"].any()
    assert grid["surface_generation"].tolist() == [-1, -1]


@pytest.mark.requires_sim
@pytest.mark.gpu
@pytest.mark.parametrize("pad_kind", ["static", "kinematic"])
def test_native_pad_uses_completed_surface_and_rebinds_after_rebuild(
    pad_kind: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dexsim.scene import Scene

    if not hasattr(Scene, "get_hydroelastic_contact_capabilities"):
        pytest.skip(
            "The installed DexSim SDK does not export completed surface bindings."
        )
    trimesh = pytest.importorskip("trimesh")
    from newton.geometry import HydroelasticSDF

    from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
    from embodichain.lab.sim.cfg import (
        NewtonCollisionPipelineCfg,
        NewtonPhysicsCfg,
        RigidObjectCfg,
    )
    from embodichain.lab.sim.sensors import SensorCfg, TactileIndentationCfg
    from embodichain.lab.sim.shapes import CubeCfg, MeshCfg, MeshCollisionCfg

    device = "cuda:0"
    sim = SimulationManager(
        SimulationManagerCfg(
            headless=True,
            num_envs=2,
            startup_summary="off",
            arena_space=0.5,
            physics_cfg=NewtonPhysicsCfg(
                device=device,
                gravity=(0.0, 0.0, 0.0),
                num_substeps=3,
                use_cuda_graph=True,
                solver_cfg={
                    "solver_type": "mujoco_warp",
                    "use_mujoco_contacts": False,
                    "nconmax": 256,
                },
                collision_cfg=NewtonCollisionPipelineCfg(
                    update_interval=1,
                    max_triangle_pairs=16384,
                    sdf_hydroelastic_config=HydroelasticSDF.Config(
                        output_contact_surface=True, pre_prune_contacts=True
                    ),
                ),
            ),
        )
    )
    try:
        mesh = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
        for uid, kind, z in (("pad", pad_kind, 0.5), ("tool", "dynamic", 0.58)):
            sim.add_rigid_object(
                RigidObjectCfg(
                    uid=uid,
                    body_type=kind,
                    init_pos=(0.0, 0.0, z),
                    shape=MeshCfg(
                        vertices=mesh.vertices,
                        triangles=mesh.faces,
                        collision=MeshCollisionCfg(
                            approximation="sdf", is_hydroelastic=True, sdf_resolution=32
                        ),
                    ),
                )
            )
        tool = sim.get_rigid_object("tool")
        config = TactileIndentationCfg(
            uid="grid",
            target_uids=("tool",),
            mount=SensorCfg.OffsetCfg(parent="pad", pos=(0.0, 0.0, 0.05)),
            size=(0.1, 0.1),
            resolution=(8, 8),
            surface=TactileIndentationCfg.SurfaceCfg(max_distance=0.03),
        )
        sensor = sim.add_sensor(config)
        second = sim.add_sensor(config.replace(uid="second-grid"))
        sim.prepare()
        scene = sim.spawn_result
        assert scene.get_hydroelastic_contact_surface() is None
        get_surface = Scene.get_hydroelastic_contact_surface
        reads = 0

        def count_read(scene):
            nonlocal reads
            reads += 1
            return get_surface(scene)

        monkeypatch.setattr(Scene, "get_hydroelastic_contact_surface", count_read)
        sim.update(step=1, render_final_step=False)
        assert reads == 1
        data = sensor.get_data()
        torch.testing.assert_close(
            data["surface_face_count"], second.get_data()["surface_face_count"]
        )
        first = get_surface(scene)
        binding = scene.get_hydroelastic_contact_binding(
            sim._spawn_scene.handles("pad")[0]
        )
        assert (binding.body_index is None) == (pad_kind == "static")
        assert (first.step, first.substep, first.collision_generation) == (
            scene.get_simulation_step(),
            2,
            3,
        )
        assert sensor.get_sample_metadata()["physics_step"].tolist() == [1, 1]
        assert data["surface_generation"].tolist() == [3, 3]
        assert data["surface_substep"].tolist() == [2, 2]
        assert data["surface_complete"].all()
        assert (data["surface_face_count"].flatten(1).sum(1) > 0).all()
        before_reset = first.vertices.clone()
        held_poses = first.body_poses.clone()
        tool.set_local_pose(
            torch.tensor(
                [[0.0, 0, 0.58, 0, 0, 0, 1.0], [0.3, 0, 0.58, 0, 0, 0, 1.0]],
                device=device,
            )
        )
        zero_velocity = torch.zeros((2, 3), device=device)
        tool.set_velocity(zero_velocity, zero_velocity)
        sim.update(step=1, render_final_step=False)
        assert reads == 2
        assert data["surface_face_count"][0].any()
        assert not data["surface_face_count"][1].any()
        assert data["surface_generation"].tolist() == [6, 6]
        sim.add_rigid_object(
            RigidObjectCfg(
                uid="rebuild",
                shape=CubeCfg(size=(0.01, 0.01, 0.01)),
                init_pos=(0.0, 0.0, 3.0),
            )
        )
        sim.prepare()
        assert get_surface(sim.spawn_result) is None
        assert not sensor.get_data()["surface_complete"].any()
        assert sim.spawn_result.topology_revision != first.topology_revision
        sim.update(step=1, render_final_step=False)
        assert reads == 3
        assert sensor.get_data()["surface_generation"].tolist() == [3, 3]
        torch.testing.assert_close(first.vertices, before_reset)
        torch.testing.assert_close(first.body_poses, held_poses)
    finally:
        sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()
