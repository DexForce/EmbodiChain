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

from enum import Enum
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

import numpy as np
import pytest
import torch
import trimesh
from pxr import Usd, UsdGeom, UsdPhysics, Gf

from embodichain.compute.geometry.mesh_bounds import transform_vertex_bounds
from embodichain.lab.sim.scene.mesh_snapshot import (
    SceneMeshComponent,
    SceneMeshSnapshot,
    capture_scene_meshes,
    iter_scene_components,
)
from embodichain.lab.sim.scene.collision_sources import (
    CollisionObjectSnapshot,
    read_articulation_link_boxes,
    snapshot_collision_sources,
)


@pytest.mark.parametrize(
    "module",
    [
        "embodichain.lab.sim.scene.mesh_snapshot",
        "embodichain.lab.sim.scene.collision_sources",
        "embodichain.lab.sim.motion.planners.curobo.payload",
    ],
)
def test_shared_inputs_do_not_import_task_program_gensim_or_native_curobo(module):
    script = f"""
import importlib, importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        forbidden=('embodichain.gen_sim','embodichain.lab.task_program','curobo')
        if any(fullname==name or fullname.startswith(name+'.') for name in forbidden):
            raise AssertionError('Unexpected import: '+fullname)
sys.meta_path.insert(0,Block())
importlib.import_module({module!r})
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_public_snapshot_copies_supplied_component_arrays():
    vertices = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    faces = torch.tensor([[0, 1, 2]])
    component = SceneMeshComponent("owner", "pose", "triangle", vertices, faces)
    snapshot = SceneMeshSnapshot({}, components=(component,))
    component.vertices.fill_(9.0)
    component.faces.fill_(0)
    torch.testing.assert_close(snapshot.components[0].vertices, vertices)
    torch.testing.assert_close(snapshot.components[0].faces, faces)


@pytest.mark.parametrize("invalid", ["scaled", "sheared", "reflected", "last_row"])
def test_collision_snapshot_rejects_nonrigid_owner_and_projection_poses(invalid):
    pose = torch.eye(4)[None]
    if invalid == "scaled":
        pose[0, 0, 0] = 2
    elif invalid == "sheared":
        pose[0, 0, 1] = 0.2
    elif invalid == "reflected":
        pose[0, 0, 0] = -1
    else:
        pose[0, 3, 0] = 0.1
    identity = torch.eye(4)[None]
    with pytest.raises(ValueError, match="rigid transform"):
        CollisionObjectSnapshot((), pose)
    with pytest.raises(ValueError, match="rigid transform"):
        snapshot_collision_sources({"body": ()}, {"body": pose}, identity)
    with pytest.raises(ValueError, match="rigid transform"):
        snapshot_collision_sources({"body": ()}, {"body": identity}, pose)


def _usd(path: Path, *, nonconvex=False):
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    body = UsdGeom.Xform.Define(stage, "/World/Panel")
    body.AddTranslateOp().Set(Gf.Vec3d(5, 1, 2))
    body.AddRotateZOp().Set(37.0)
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    if nonconvex:
        geometry = trimesh.creation.annulus(
            r_min=0.2, r_max=0.4, height=0.1, sections=12
        )
    else:
        first = trimesh.creation.box(extents=[0.4, 0.2, 0.1])
        second = first.copy()
        second.apply_translation([0, 1, 0])
        geometry = trimesh.util.concatenate((first, second))
    mesh = UsdGeom.Mesh.Define(stage, "/World/Panel/Collider")
    mesh.CreatePointsAttr(geometry.vertices.tolist())
    mesh.CreateFaceVertexCountsAttr([3] * len(geometry.faces))
    mesh.CreateFaceVertexIndicesAttr(geometry.faces.reshape(-1).tolist())
    UsdGeom.Xformable(mesh).AddTranslateOp().Set(Gf.Vec3d(1, 0, 0))
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    stage.GetRootLayer().Save()


def _simulation(path: Path, *, backend="newton", scale=(2.0, 2.0, 2.0)):
    vertices = torch.tensor([[[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]]])
    rigid = SimpleNamespace(get_vertices=lambda *, scale: vertices)
    art = SimpleNamespace(cfg=SimpleNamespace(fpath=str(path), body_scale=scale))
    sim = SimpleNamespace(
        physics_backend=backend,
        num_envs=1,
        spawn_result=None,
        get_rigid_object=lambda uid: rigid if uid == "native_rigid" else None,
        get_articulation=lambda uid: art if uid == "native_art" else None,
    )
    return sim, vertices


def test_capture_keeps_visual_and_input_components_owned_local_and_ordered(tmp_path):
    path = tmp_path / "panels.usda"
    _usd(path)
    sim, original = _simulation(path)
    snapshot = capture_scene_meshes(
        sim,
        rigid_uids={"item": "native_rigid"},
        articulation_uids={"fixture": "native_art"},
        link_bindings=[("fixture", "Panel", "observed_panel")],
        collision_roles={"item": "none", "fixture": "none"},
    )
    assert snapshot.unavailable is None
    assert snapshot.rigid_geometry_kind == "rigid_visual_envelope"
    assert snapshot.collision_roles == {"item": "none", "fixture": "none"}
    assert [c.name for c in snapshot.components] == [
        "/World/Panel/Collider:0",
        "/World/Panel/Collider:1",
    ]
    assert all(
        c.geometry_kind == "articulation_collision_input"
        and c.provenance == "usd_collision_inputs"
        for c in snapshot.components
    )
    original.fill_(99)
    torch.testing.assert_close(
        snapshot["item"], torch.tensor([[[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]]])
    )
    centers = [
        (c.vertices.amin(1) + c.vertices.amax(1))[0] / 2 for c in snapshot.components
    ]
    torch.testing.assert_close(
        torch.stack(centers),
        torch.tensor([[2.0, 0.0, 0.0], [2.0, 2.0, 0.0]]),
        atol=1e-6,
        rtol=0,
    )
    pose = torch.tensor(
        [[[0.0, -1, 0, 10], [1.0, 0, 0, 20], [0, 0, 1.0, 30], [0, 0, 0, 1.0]]]
    )
    observed = list(
        iter_scene_components(snapshot, {"observed_panel": pose}, excluded_owner="item")
    )
    low, high = transform_vertex_bounds(observed[0][2], observed[0][3])
    torch.testing.assert_close(low, torch.tensor([[9.8, 21.6, 29.9]]))
    torch.testing.assert_close(high, torch.tensor([[10.2, 22.4, 30.1]]))
    boxes = read_articulation_link_boxes(path, [2.0, 2.0, 2.0])["Panel"]
    assert [box.name for box in boxes] == ["component_0", "component_1"]
    torch.testing.assert_close(boxes[0].half_extents, torch.tensor([0.4, 0.2, 0.1]))


@pytest.mark.parametrize(
    "defect",
    [
        "missing_binding",
        "duplicate_binding",
        "nonuniform_scale",
        "missing_scene",
        "nonconvex",
    ],
)
def test_articulation_capture_remains_unavailable_with_captured_rigids(
    tmp_path, defect
):
    path = tmp_path / "panels.usda"
    _usd(path, nonconvex=defect == "nonconvex")
    sim, _ = _simulation(
        path,
        backend="default" if defect == "missing_scene" else "newton",
        scale=(2.0, 3.0, 2.0) if defect == "nonuniform_scale" else (2.0, 2.0, 2.0),
    )
    links = (
        [] if defect == "missing_binding" else [("fixture", "Panel", "observed_panel")]
    )
    if defect == "duplicate_binding":
        links.append(("fixture", "Panel", "other_pose"))
    snapshot = capture_scene_meshes(
        sim,
        rigid_uids={"item": "native_rigid"},
        articulation_uids={"fixture": "native_art"},
        link_bindings=links,
    )
    assert list(snapshot) == ["item"] and snapshot.unavailable
    assert snapshot.components == ()
    with pytest.raises(ValueError):
        list(iter_scene_components(snapshot, {}, excluded_owner="item"))
    if defect == "nonconvex":
        # The independent cuRobo input-box policy retains its conservative AABB.
        assert len(read_articulation_link_boxes(path, [2.0, 2.0, 2.0])["Panel"]) == 1


def test_rigid_only_missing_geometry_and_neutral_roles(tmp_path):
    sim, _ = _simulation(tmp_path / "unused.usda")
    assert not capture_scene_meshes(sim, rigid_uids={"missing": "absent"})

    class Role(str, Enum):
        NONE = "none"

    with pytest.raises(TypeError, match="neutral"):
        capture_scene_meshes(
            sim,
            rigid_uids={"item": "native_rigid"},
            collision_roles={"item": Role.NONE},
        )


@pytest.mark.parametrize("decompose", [False, True])
def test_default_descriptor_route_preserves_local_transform_scale_and_part_order(
    tmp_path, monkeypatch, decompose
):
    from dexsim import scene as sdk_scene
    from dexsim.spawn.descs import CollisionDesc, CollisionApproximation, GeometryType
    from dexsim.kit.meshproc import convex_cache

    mesh = trimesh.creation.box(extents=[1, 1, 1])
    transform = np.eye(4)
    transform[:3, 3] = [1, 2, 3]
    geom = CollisionDesc(
        geometry_type=GeometryType.MESH,
        segment_name="panel",
        vertices=mesh.vertices,
        triangles=mesh.faces,
        local_transform=transform,
        approximation=(
            CollisionApproximation.CONVEX_DECOMPOSITION
            if decompose
            else CollisionApproximation.CONVEX_HULL
        ),
    )
    path = tmp_path / "descriptor.usda"

    class Handle:
        arena_name = "env0"
        desc = SimpleNamespace(
            urdf_path=None,
            usd=SimpleNamespace(file_path=str(path)),
            links=[SimpleNamespace(name="Panel")],
            body_scale=[2.0, 2.0, 2.0],
        )

        def get_name(self):
            return "native_art"

        def get_link_names(self):
            return ["Panel"]

    monkeypatch.setattr(sdk_scene, "SpawnedArticulation", Handle)
    handle = Handle()
    scene = SimpleNamespace(
        backend="dexsim",
        arenas=SimpleNamespace(names=("global", "env0")),
        get_paths=lambda: ["/fixture"],
        get=lambda path: handle,
        get_link_desc=lambda *args: SimpleNamespace(collisions=[geom]),
    )
    if decompose:

        def decomp(vertices, faces, **kwargs):
            assert kwargs["max_convex_hulls"] == geom.decomp_max_hulls
            return (vertices, faces)

        monkeypatch.setattr(convex_cache, "convex_decomposition_to_obj", decomp)
        monkeypatch.setattr(
            convex_cache, "load_obj_as_convex_parts", lambda pair: [pair, pair]
        )
    sim, _ = _simulation(path, backend="default")
    sim.spawn_result = scene
    snapshot = capture_scene_meshes(
        sim,
        rigid_uids={},
        articulation_uids={"fixture": "native_art"},
        link_bindings=[("fixture", "Panel", "pose")],
    )
    assert snapshot.unavailable is None
    assert len(snapshot.components) == (2 if decompose else 1)
    for index, component in enumerate(snapshot.components):
        assert component.name == f"panel:0:{index}:0"
        assert component.provenance == "retained_default_usd_collision_inputs"
        torch.testing.assert_close(
            component.vertices.amin(1), torch.tensor([[1.0, 3.0, 5.0]])
        )
        torch.testing.assert_close(
            component.vertices.amax(1), torch.tensor([[3.0, 5.0, 7.0]])
        )
