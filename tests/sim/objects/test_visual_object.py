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

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from dexsim.spawn import CollisionDesc, ObjectDesc, RigidBodyPhysicsDesc

from embodichain.lab.sim.cfg import VisualObjectCfg, UsdSourceCfg
from embodichain.lab.sim.cfg import ObjectBaseCfg, RigidObjectCfg
from embodichain.lab.sim.objects import RigidObject, VisualObject
from embodichain.lab.sim.shapes import CubeCfg, MeshCfg
from embodichain.lab.sim.spawn.descriptors import visual_desc_from_cfg
from embodichain.lab.sim.spawn._usd_assets import _UsdTextureFiles
from embodichain.lab.sim.spawn.usd import rigid_desc_from_usd, visual_desc_from_usd

pytestmark = pytest.mark.no_sim


class _Render:
    def __init__(self):
        self.materials = [None]

    def get_mesh_count(self):
        return len(self.materials)

    def get_material(self, index):
        return self.materials[index]

    def set_material(self, index, material):
        self.materials[index] = material

    def clean_material(self):
        self.materials = [None] * len(self.materials)


class _Handle:
    def __init__(self, offset):
        self.object_desc = ObjectDesc(name="visual")
        self.pose = np.eye(4, dtype=np.float32)
        self.pose[0, 3] = offset
        self.parent = np.eye(4, dtype=np.float32)
        self.parent[:3, 3] = (5.0 * offset, 2.0 * offset, 0.0)
        self.visible = True
        self.user_id = 10 + int(offset)
        self.render = _Render()

    def get_user_id(self):
        return self.user_id

    def get_local_pose(self):
        return self.pose.copy()

    def set_local_pose(self, pose):
        self.pose = pose.copy()

    def get_world_pose(self):
        return self.parent @ self.pose

    def set_visible(self, visible):
        self.visible = visible

    def is_visible(self):
        return self.visible

    def get_render_body(self):
        return self.render

    def set_material(self, material):
        self.render.materials = [material]

    def get_body_scale(self):
        return np.array([2.0, 3.0, 4.0])

    def get_vertices(self):
        return np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    def get_triangles(self):
        return np.array([[0, 1, 2]])


class _World:
    def __init__(self, handles):
        self.handles = {handle.get_user_id(): handle for handle in handles}
        self.fetch_calls = 0
        self.apply_calls = 0
        self.fetch_status = 0
        self.apply_status = 0

    def fetch_object_transform(self, transforms, indices):
        self.fetch_calls += 1
        if self.fetch_status < 0:
            return self.fetch_status
        for row, uid in enumerate(indices.tolist()):
            handle = self.handles[uid]
            transforms[row].copy_(torch.from_numpy(handle.parent @ handle.pose))
        return 0

    def apply_object_transform(self, transforms, indices):
        self.apply_calls += 1
        if self.apply_status < 0:
            return self.apply_status
        for matrix, uid in zip(transforms.numpy(), indices.tolist()):
            handle = self.handles[uid]
            handle.pose = np.linalg.inv(handle.parent) @ matrix
        return len(indices)


def _visual():
    visual = VisualObject(VisualObjectCfg(uid="visual"))
    visual._initialize_spawn_declaration(2)
    handles = [_Handle(0.0), _Handle(1.0)]
    visual.attach_spawn_handles(handles)
    visual.bind_spawn(SimpleNamespace(topology_revision=1, world=_World(handles)))
    return visual, handles


def test_visual_declaration_requires_prepare_and_has_no_physics_interface():
    visual = VisualObject(VisualObjectCfg(uid="visual"))
    visual._initialize_spawn_declaration(2)
    assert visual.num_instances == 2
    assert "visual" in str(visual)
    assert not isinstance(visual, RigidObject)
    assert not hasattr(visual, "get_mass")
    with pytest.raises(RuntimeError, match="prepare"):
        visual.get_local_pose()


@pytest.mark.parametrize("physical", ["body", "collider"])
def test_visual_binding_rejects_physical_handles(physical):
    visual = VisualObject(VisualObjectCfg(uid="visual"))
    visual._initialize_spawn_declaration(1)
    handle = _Handle(0.0)
    if physical == "body":
        handle.object_desc.physics = RigidBodyPhysicsDesc.static()
    else:
        handle.object_desc.collisions = [CollisionDesc()]
    with pytest.raises(ValueError, match="physical"):
        visual.attach_spawn_handles([handle])


def test_visual_selected_pose_visibility_and_reset_preserve_other_environment():
    visual, handles = _visual()
    before = visual.get_local_pose().clone()
    assert visual.user_ids.tolist() == [10, 11]
    # A non-symmetric xyzw rotation catches quaternion-order regressions.
    pose = torch.tensor([[3.0, 4.0, 5.0, 0.1, 0.2, 0.3, 0.9]])
    pose[:, 3:] = torch.nn.functional.normalize(pose[:, 3:], dim=-1)
    visual.set_local_pose(pose, [1])
    visual.set_visible(False, [1])
    torch.testing.assert_close(visual.get_local_pose()[1:], pose)
    torch.testing.assert_close(visual.get_local_pose()[:1], before[:1])
    assert visual.get_visibility().tolist() == [True, False]
    visual.reset([1])
    torch.testing.assert_close(visual.get_local_pose(), before)
    assert all(h.visible for h in handles)


@pytest.mark.parametrize("indices", [[-1], [2], [True], [0, 0]])
def test_invalid_visual_selection_fails_before_writes(indices):
    visual, _ = _visual()
    before = visual.get_local_pose().clone()
    with pytest.raises((ValueError, IndexError)):
        visual.set_visible(False, indices)
    torch.testing.assert_close(visual.get_local_pose(), before)
    assert visual.get_visibility().all()


def test_invalid_visual_pose_fails_before_writes():
    visual, _ = _visual()
    before = visual.get_local_pose().clone()
    matrices = torch.eye(4).repeat(2, 1, 1)
    matrices[1, 0, 0] = 2.0
    with pytest.raises(ValueError, match="rigid transforms"):
        visual.set_local_pose(matrices)
    torch.testing.assert_close(visual.get_local_pose(), before)


def test_visual_geometry_returns_scaled_combined_mesh():
    visual, _ = _visual()
    vertices = visual.get_vertices([1], scale=True)
    torch.testing.assert_close(vertices[0, 1], torch.tensor([2.0, 0.0, 0.0]))
    assert visual.get_triangles([1]).tolist() == [[[0, 1, 2]]]


def test_visual_reset_restores_source_material_for_only_selected_environment():
    visual, handles = _visual()
    replacement = object()
    template = SimpleNamespace(
        uid="new", create_instance=lambda uid: SimpleNamespace(mat=replacement)
    )
    visual.set_visual_material(template)
    visual.reset([1])
    assert handles[0].render.materials == [replacement]
    assert handles[1].render.materials == [None]


def test_visual_config_and_descriptor_never_create_physics():
    cfg = VisualObjectCfg.from_dict(
        {"uid": "prop", "shape": {"shape_type": "Cube", "size": [0.2, 0.3, 0.4]}}
    )
    assert isinstance(cfg.shape, CubeCfg)
    descriptor, _ = visual_desc_from_cfg(cfg)
    assert descriptor.physics is None
    assert descriptor.collisions == []
    assert len(descriptor.renders) == 1


def _usd_scene(tmp_path):
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics

    path = tmp_path / "mixed.usda"
    stage = Usd.Stage.CreateNew(str(path))
    for name in ("Visual", "Physical"):
        prim = UsdGeom.Xform.Define(stage, f"/World/{name}").GetPrim()
        prim.CreateAttribute("MeshObject", Sdf.ValueTypeNames.Token)
        mesh = UsdGeom.Cube.Define(stage, f"/World/{name}/mesh").GetPrim()
        if name == "Physical":
            UsdPhysics.RigidBodyAPI.Apply(prim)
            UsdPhysics.CollisionAPI.Apply(mesh)
    stage.GetRootLayer().Save()
    return path


def test_explicit_visual_usd_selection_imports_only_appearance(tmp_path):
    path = _usd_scene(tmp_path)
    descriptor, _ = visual_desc_from_usd(
        VisualObjectCfg(
            uid="selected",
            source=UsdSourceCfg(
                file_path=str(path), source_prim_path="/World/Physical"
            ),
        )
    )
    assert descriptor.name == "selected"
    assert descriptor.physics is None
    assert descriptor.collisions == []
    assert descriptor.renders
    with pytest.raises(ValueError, match="exactly one"):
        visual_desc_from_usd(
            VisualObjectCfg(uid="selected", shape=MeshCfg(fpath=str(path)))
        )


def test_add_usd_classifies_visuals_without_promoting_collision_only_objects(tmp_path):
    from embodichain.lab.sim import SimulationManager
    from embodichain.lab.sim.spawn.scene import SpawnScene
    from dexsim.kit.usd import parse_usd
    from unittest.mock import patch

    path = _usd_scene(tmp_path)
    source = parse_usd(str(path))
    source_visual = next(d for d in source.mesh_objects if d.name == "Visual")
    source_visual.name = "default_plane"
    collision_only = next(d for d in source.mesh_objects if d.name == "Physical")
    collision_only.physics = None
    assert collision_only.collisions
    manager = object.__new__(SimulationManager)
    manager.device = torch.device("cpu")
    for key in (
        "_visual_objects",
        "_rigid_objects",
        "_rigid_object_groups",
        "_deformable_objects",
        "_articulations",
        "_robots",
        "_lights",
        "_sensors",
    ):
        setattr(manager, key, {})
    scene = object.__new__(SpawnScene)
    scene._num_envs = 2
    scene._assets = {}
    scene._usd_texture_files = _UsdTextureFiles()
    scene.builder = SimpleNamespace(
        is_finalized=False,
        result=None,
        add_scene=lambda source, **kw: source.mesh_objects,
    )
    manager._spawn_scene = scene
    manager.notify_visualization_topology_changed = MagicMock()
    with patch("dexsim.kit.usd.parse_usd", return_value=source):
        assets = manager.add_usd("scene", str(path))
    assert isinstance(assets["/World/Visual"], VisualObject)
    assert isinstance(assets["/World/Physical"], RigidObject)
    assert manager.get_asset("scene_default_plane") is assets["/World/Visual"]
    assert manager.get_visual_object_uid_list() == ["scene_default_plane"]


def test_visual_tensor_and_empty_environment_selections():
    visual, _ = _visual()
    visual.set_visible(False, torch.tensor([1], dtype=torch.int64))
    assert visual.get_visibility().tolist() == [True, False]
    visual.reset(torch.tensor([1], dtype=torch.int32))
    assert visual.get_visibility().all()
    assert visual.get_vertices([]).shape == (0, 3, 3)
    assert visual.get_triangles([]).shape == (0, 1, 3)
    visual.set_local_pose(torch.empty((0, 7)), [])


def test_visual_release_invalidates_runtime_operations():
    visual, _ = _visual()
    visual._release_spawn_references()
    assert not visual.is_spawn_bound
    with pytest.raises(RuntimeError, match="prepare"):
        visual.get_local_pose()
    assert not visual._default_materials


def test_invalid_visual_scale_fails_without_building_physics():
    for scale in ((0.0, 1.0, 1.0), (float("nan"), 1.0, 1.0)):
        with pytest.raises(ValueError, match="body_scale"):
            visual_desc_from_cfg(
                VisualObjectCfg(
                    uid="bad", shape=CubeCfg(size=(1.0, 1.0, 1.0)), body_scale=scale
                )
            )


def test_visual_state_survives_native_rebuild_without_changing_reset_baseline():
    visual, handles = _visual()
    initial = visual.get_local_pose().clone()
    target = initial[1:].clone()
    target[:, 2] = 3.0
    visual.set_local_pose(target, [1])
    visual.set_visible(False, [1])
    replacement = object()
    visual.set_visual_material(
        SimpleNamespace(
            uid="test", create_instance=lambda uid: SimpleNamespace(mat=replacement)
        ),
        [1],
    )
    # Simulate a backend recreating render actors from initial descriptors.
    handles[1].pose = np.eye(4, dtype=np.float32)
    handles[1].user_id = 51
    angle = 0.3
    handles[1].parent[:2, :2] = [
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)],
    ]
    visual._spawn_result.world.handles = {h.get_user_id(): h for h in handles}
    handles[1].visible = True
    handles[1].render.materials = [None]
    visual._spawn_result.topology_revision += 1
    visual._sync_spawn_state()
    assert visual.user_ids.tolist() == [10, 51]
    torch.testing.assert_close(visual.get_local_pose()[1:], target)
    assert not handles[1].visible
    assert handles[1].render.materials == [replacement]
    visual.reset([1])
    torch.testing.assert_close(visual.get_local_pose(), initial)
    assert handles[1].visible
    assert handles[1].render.materials == [None]


def test_visual_config_inherited_annotations_resolve():
    from typing import get_type_hints

    import sys

    hints = get_type_hints(
        VisualObjectCfg, globalns=vars(sys.modules[VisualObjectCfg.__module__])
    )
    assert hints["init_local_pose"] == np.ndarray | None


@pytest.mark.parametrize("suffix", ["obj", "stl", "glb", "gltf"])
def test_visual_mesh_file_config_and_spawn_match_rigid_convention(tmp_path, suffix):
    path = tmp_path / f"mesh.{suffix}"
    cfg = VisualObjectCfg.from_dict(
        {"uid": "mesh", "shape": {"shape_type": "Mesh", "fpath": str(path)}}
    )
    assert isinstance(cfg.shape, MeshCfg)
    descriptor, _ = visual_desc_from_cfg(cfg)
    assert descriptor.renders[0].file_path == str(path)
    assert descriptor.physics is None
    assert not descriptor.collisions


def test_usd_source_selection_is_shared_by_visual_and_rigid(tmp_path):
    assert "prim_path" not in ObjectBaseCfg.__annotations__
    path = _usd_scene(tmp_path)
    data = {
        "uid": "selected",
        "source": {"file_path": str(path), "source_prim_path": "/World/Physical"},
    }
    visual, _ = visual_desc_from_usd(VisualObjectCfg.from_dict(data))
    rigid, _ = rigid_desc_from_usd(RigidObjectCfg.from_dict(data))
    assert (
        visual.usd.prim_path
        == rigid.usd.prim_path
        == data["source"]["source_prim_path"]
    )
    assert visual.physics is None
    assert rigid.physics is not None


def test_source_preserves_existing_positional_config_arguments():
    cfg = RigidObjectCfg("rigid", (1.0, 2.0, 3.0))
    assert cfg.uid == "rigid"
    assert cfg.init_pos == (1.0, 2.0, 3.0)
    assert cfg.source is None


def test_visual_spawn_rejects_non_usd_source_file(tmp_path):
    cfg = VisualObjectCfg(
        uid="bad",
        source=UsdSourceCfg(
            file_path=str(tmp_path / "mesh.obj"), source_prim_path="/World/Mesh"
        ),
    )
    with pytest.raises(ValueError, match="requires a USD source"):
        visual_desc_from_cfg(cfg)


def test_visual_pose_uses_one_native_batch_and_preserves_selection_order(monkeypatch):
    visual, handles = _visual()
    world = visual._spawn_result.world
    before = visual.get_local_pose(to_matrix=True)
    for handle in handles:
        monkeypatch.setattr(
            handle, "get_local_pose", lambda: pytest.fail("scalar read")
        )
        monkeypatch.setattr(
            handle, "set_local_pose", lambda _: pytest.fail("scalar write")
        )
    world.fetch_calls = world.apply_calls = 0
    rows = torch.tensor([1, 0], dtype=torch.int32)
    # A strided input also exercises native tensor preparation.
    storage = torch.eye(4).repeat(2, 1, 2)
    target = storage[:, :, ::2]
    target.copy_(before.flip(0))
    target[:, :3, 3] += torch.tensor([0.3, 0.5, 0.7])
    assert not target.is_contiguous()
    visual.set_local_pose(target, rows)
    result = visual.get_local_pose(to_matrix=True, env_ids=rows)
    torch.testing.assert_close(result, target)
    assert world.apply_calls == 1
    assert world.fetch_calls == 1
    saved = result.clone()
    visual.set_local_pose(before)
    torch.testing.assert_close(result, saved)


def test_visual_pose_empty_selection_does_not_dispatch():
    visual, _ = _visual()
    world = visual._spawn_result.world
    world.fetch_calls = world.apply_calls = 0
    assert visual.get_local_pose(env_ids=[]).shape == (0, 7)
    assert visual.get_local_pose(to_matrix=True, env_ids=[]).shape == (0, 4, 4)
    visual.set_local_pose(torch.empty(0, 7), [])
    assert world.fetch_calls == world.apply_calls == 0


@pytest.mark.parametrize("operation", ["fetch", "apply"])
def test_visual_pose_native_failure_propagates_without_corrupting_cache(operation):
    visual, handles = _visual()
    cached = visual._poses.clone()
    native = [h.pose.copy() for h in handles]
    world = visual._spawn_result.world
    setattr(world, f"{operation}_status", -6)
    with pytest.raises(RuntimeError, match="status -6"):
        if operation == "fetch":
            visual.get_local_pose()
        else:
            visual.set_local_pose(torch.eye(4).repeat(2, 1, 1))
    torch.testing.assert_close(visual._poses, cached)
    for handle, pose in zip(handles, native):
        np.testing.assert_array_equal(handle.pose, pose)


def test_visual_material_read_respects_selected_environments():
    visual, _ = _visual()
    assert visual.get_visual_material_inst([1]) == [None]
    assert visual.get_visual_material_inst([]) == []
