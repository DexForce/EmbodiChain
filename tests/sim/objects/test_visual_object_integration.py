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

import pytest
import numpy as np
import torch

from embodichain.lab.sim import (
    SimulationManager,
    SimulationManagerCfg,
    VisualMaterialCfg,
)
from embodichain.lab.sim.cfg import (
    RenderCfg,
    RigidObjectCfg,
    ArticulationCfg,
    UsdSourceCfg,
    VisualObjectCfg,
    physics_cfg_for_backend,
)
from embodichain.lab.sim.shapes import CubeCfg, MeshCfg
from embodichain.utils.math import matrix_from_quat

pytestmark = [pytest.mark.requires_sim, pytest.mark.renderer]


@pytest.mark.parametrize("physics", ["default", "newton"])
def test_render_only_object_native_lifecycle(physics):
    sim = SimulationManager(
        SimulationManagerCfg(
            headless=True,
            device="cpu",
            num_envs=2,
            scene_node_capacity=128,
            physics_cfg=physics_cfg_for_backend(physics),
            render_cfg=RenderCfg(renderer="hybrid"),
        )
    )
    visual = None
    try:
        visual = sim.add_visual_object(
            VisualObjectCfg(
                uid="prop",
                shape=CubeCfg(size=(0.1, 0.2, 0.3)),
                init_pos=(0.0, 0.0, 1.0),
            )
        )
        sim.prepare()
        assert all(
            h.object_desc.physics is None and not h.object_desc.collisions
            for h in visual._entities
        )
        before = visual.get_local_pose().clone()
        sim.update()
        torch.testing.assert_close(visual.get_local_pose(), before)
        target = before[1:].clone()
        target[:, 0] += 0.3
        visual.set_local_pose(target, [1])
        visual.set_visible(False, [1])
        source_material = visual._entities[1].get_render_body().get_material(0)
        source_name = None if source_material is None else source_material.get_name()
        material = sim.create_visual_material(
            VisualMaterialCfg(uid="alternate", base_color=[0.8, 0.1, 0.2, 1.0])
        )
        visual.set_visual_material(material, [1])
        assigned_name = visual._entities[1].get_render_body().get_material(0).get_name()
        sim.sync_render_state()
        torch.testing.assert_close(visual.get_local_pose()[1:], target)
        torch.testing.assert_close(visual.get_local_pose()[:1], before[:1])
        assert visual.get_visibility().tolist() == [True, False]
        other = sim.add_visual_object(
            VisualObjectCfg(uid="other", shape=CubeCfg(size=(0.1, 0.1, 0.1)))
        )
        torch.testing.assert_close(visual.get_local_pose()[1:], target)
        assert visual.get_visibility().tolist() == [True, False]
        assert (
            visual._entities[1].get_render_body().get_material(0).get_name()
            == assigned_name
        )
        assert sim.remove_asset("other")
        del other
        torch.testing.assert_close(visual.get_local_pose()[1:], target)
        sim.reset_objects_state(torch.tensor([1], dtype=torch.int64))
        torch.testing.assert_close(visual.get_local_pose(), before)
        assert visual.get_visibility().all()
        restored = visual._entities[1].get_render_body().get_material(0)
        assert (None if restored is None else restored.get_name()) == source_name
        assert sim.get_rigid_object("prop") is None
        assert sim.remove_asset("prop")
        assert sim.get_asset("prop") is None
        assert not visual.is_spawn_bound
    finally:
        sim.destroy(exit_process=False)
        del visual
        SimulationManager.flush_cleanup_queue()


@pytest.mark.parametrize("physics", ["default", "newton"])
def test_visual_mesh_sources_use_native_batch_pose(physics, tmp_path):
    import trimesh
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics

    mesh = trimesh.creation.box(extents=(0.1, 0.2, 0.3))
    paths = []
    for suffix in ("obj", "stl", "glb"):
        path = tmp_path / f"box.{suffix}"
        mesh.export(str(path))
        paths.append(path)
    gltf_root = tmp_path / "gltf"
    gltf_root.mkdir()
    for name, payload in trimesh.Scene(mesh).export(file_type="gltf").items():
        (gltf_root / name).write_bytes(payload)
    paths.append(gltf_root / "model.gltf")
    usd_path = tmp_path / "mixed.usda"
    stage = Usd.Stage.CreateNew(str(usd_path))
    for name in ("Visual", "Physical"):
        prim = UsdGeom.Xform.Define(stage, f"/World/{name}").GetPrim()
        prim.CreateAttribute("MeshObject", Sdf.ValueTypeNames.Token)
        cube = UsdGeom.Cube.Define(stage, f"/World/{name}/mesh")
        cube.CreateSizeAttr().Set(0.1)
        if name == "Physical":
            UsdPhysics.RigidBodyAPI.Apply(prim)
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    stage.GetRootLayer().Save()
    paths.append(usd_path)
    sim = SimulationManager(
        SimulationManagerCfg(
            headless=True,
            device="cpu",
            num_envs=3,
            scene_node_capacity=128,
            physics_cfg=physics_cfg_for_backend(physics),
            render_cfg=RenderCfg(renderer="hybrid"),
        )
    )
    visuals = []
    visual = other = handle = rigid = None
    try:
        for index, path in enumerate(paths):
            data = {
                "uid": f"mesh_{index}",
                "shape": {"shape_type": "Mesh", "fpath": str(path)},
                "body_scale": (1.2, 0.8, 1.5),
                "init_pos": (0.2 * index, 0.1, 1.0),
                "init_rot": (10.0, 20.0, 30.0),
            }
            if path == usd_path:
                del data["shape"]
                data["source"] = {
                    "file_path": str(path),
                    "source_prim_path": "/World/Physical",
                }
            cfg = VisualObjectCfg.from_dict(data)
            visuals.append(sim.add_visual_object(cfg))
        rigid = sim.add_rigid_object(
            RigidObjectCfg(
                uid="selected_rigid",
                source=UsdSourceCfg(
                    file_path=str(usd_path), source_prim_path="/World/Physical"
                ),
            )
        )
        sim.prepare()
        assert all(h.object_desc.physics is not None for h in rigid._entities)
        rows = torch.tensor([2, 0], dtype=torch.int64)
        for visual in visuals:
            assert visual.get_vertices().shape[1] > 0
            assert visual.get_triangles().shape[1] > 0
            assert all(
                h.object_desc.physics is None and not h.object_desc.collisions
                for h in visual._entities
            )
            initial = visual.get_local_pose(to_matrix=True)
            torch.testing.assert_close(
                initial,
                torch.tensor(visual.cfg.init_local_pose, dtype=torch.float32).expand(
                    3, -1, -1
                ),
            )
            storage = torch.zeros(2, 14)
            target = storage[:, ::2]
            target[:] = torch.tensor(
                [
                    [0.3, 0.4, 1.2, 0.1, 0.2, 0.3, 0.9],
                    [-0.2, 0.5, 1.4, -0.3, 0.2, 0.1, 0.9],
                ]
            )
            target[:, 3:] = torch.nn.functional.normalize(target[:, 3:], dim=-1)
            expected = torch.eye(4).repeat(2, 1, 1)
            expected[:, :3, :3] = matrix_from_quat(target[:, 3:])
            expected[:, :3, 3] = target[:, :3]
            visual.set_local_pose(target, rows)
            torch.testing.assert_close(
                visual.get_local_pose(to_matrix=True, env_ids=rows),
                expected,
                atol=1e-5,
                rtol=1e-5,
            )
            torch.testing.assert_close(
                visual.get_local_pose(to_matrix=True)[1], initial[1]
            )
            for row, index in enumerate(rows.tolist()):
                handle = visual._entities[index]
                np.testing.assert_allclose(
                    handle.get_local_pose(), expected[row].numpy(), atol=1e-5
                )
            torch.testing.assert_close(
                visual.get_body_scale(), torch.tensor([[1.2, 0.8, 1.5]]).expand(3, -1)
            )
            saved = visual.get_local_pose(to_matrix=True)
            other = sim.add_visual_object(
                VisualObjectCfg(uid="rebuild", shape=CubeCfg())
            )
            torch.testing.assert_close(
                visual.get_local_pose(to_matrix=True), saved, atol=1e-5, rtol=1e-5
            )
            assert sim.remove_asset("rebuild")
            other = None
            visual.reset(rows)
            torch.testing.assert_close(
                visual.get_local_pose(to_matrix=True), initial, atol=1e-5, rtol=1e-5
            )
    finally:
        sim.destroy(exit_process=False)
        visuals.clear()
        visual = other = handle = rigid = None
        SimulationManager.flush_cleanup_queue()


def test_visual_native_cuda_pose_buffers():
    sim = SimulationManager(
        SimulationManagerCfg(
            headless=True,
            device="cuda",
            num_envs=2,
            scene_node_capacity=128,
            render_cfg=RenderCfg(renderer="hybrid"),
        )
    )
    visual = None
    try:
        visual = sim.add_visual_object(
            VisualObjectCfg(uid="cuda_visual", shape=CubeCfg())
        )
        sim.prepare()
        initial = visual.get_local_pose(to_matrix=True)
        assert initial.device.type == "cuda"
        rows = torch.tensor([1], dtype=torch.int32, device="cuda")
        target = initial[1:].clone()
        target[:, :3, 3] = torch.tensor([0.3, -0.4, 1.2], device="cuda")
        visual.set_local_pose(target, rows)
        torch.testing.assert_close(
            visual.get_local_pose(to_matrix=True, env_ids=rows), target
        )
        torch.testing.assert_close(
            visual.get_local_pose(to_matrix=True)[:1], initial[:1]
        )
        visual.reset(rows)
        torch.testing.assert_close(visual.get_local_pose(to_matrix=True), initial)
    finally:
        sim.destroy(exit_process=False)
        visual = None
        SimulationManager.flush_cleanup_queue()


@pytest.mark.parametrize("physics", ["default", "newton"])
def test_usd_source_selects_articulation_root(physics, tmp_path):
    from pxr import Usd, UsdGeom, UsdPhysics

    path = tmp_path / "arms.usda"
    stage = Usd.Stage.CreateNew(str(path))
    for name in ("ArmA", "ArmB"):
        root = UsdGeom.Xform.Define(stage, f"/World/{name}").GetPrim()
        UsdPhysics.ArticulationRootAPI.Apply(root)
        for link in ("Base", "Tip"):
            body = UsdGeom.Xform.Define(stage, f"/World/{name}/{link}").GetPrim()
            UsdPhysics.RigidBodyAPI.Apply(body)
            UsdPhysics.MassAPI.Apply(body).CreateMassAttr().Set(1.0)
            mesh = UsdGeom.Cube.Define(stage, f"/World/{name}/{link}/mesh")
            mesh.CreateSizeAttr().Set(0.1)
            UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        fixed = UsdPhysics.FixedJoint.Define(stage, f"/World/{name}/root_joint")
        fixed.CreateBody1Rel().SetTargets([f"/World/{name}/Base"])
        hinge = UsdPhysics.RevoluteJoint.Define(stage, f"/World/{name}/hinge")
        hinge.CreateBody0Rel().SetTargets([f"/World/{name}/Base"])
        hinge.CreateBody1Rel().SetTargets([f"/World/{name}/Tip"])
        hinge.CreateAxisAttr().Set("Z")
        hinge.CreateLowerLimitAttr().Set(-45.0)
        hinge.CreateUpperLimitAttr().Set(45.0)
    stage.GetRootLayer().Save()
    sim = SimulationManager(
        SimulationManagerCfg(
            headless=True,
            device="cpu",
            num_envs=2,
            scene_node_capacity=128,
            physics_cfg=physics_cfg_for_backend(physics),
            render_cfg=RenderCfg(renderer="hybrid"),
        )
    )
    articulation = None
    try:
        articulation = sim.add_articulation(
            ArticulationCfg(
                source=UsdSourceCfg(file_path=str(path), source_prim_path="/World/ArmB")
            )
        )
        sim.prepare()
        assert sim.get_articulation_uid_list() == ["arms"]
        assert articulation.dof == 1
        assert articulation.num_instances == 2
        assert articulation.pk_chain is not None
        assert articulation.cfg.fpath is None
        assert articulation.cfg.source.source_prim_path == "/World/ArmB"
    finally:
        sim.destroy(exit_process=False)
        articulation = None
        SimulationManager.flush_cleanup_queue()
