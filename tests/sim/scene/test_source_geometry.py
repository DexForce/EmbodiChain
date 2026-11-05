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
"""Source geometry preserves authored topology and owns its numeric inputs."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from embodichain.lab.sim.scene.source_geometry import LinkMesh


def test_input_mesh_preserves_order_and_detaches_mutable_buffers():
    vertices = np.array(
        [[2.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 3.0, 0.0]], dtype=np.float32
    )
    faces = np.array([[2, 0, 1]], dtype=np.int32)
    mesh = LinkMesh("body/collider", "body", vertices, faces, "convexHull")
    copied = deepcopy(mesh)
    vertices[:] = 99
    faces[:] = 0
    for owned in (mesh, copied):
        np.testing.assert_array_equal(
            owned.vertices, [[2.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 3.0, 0.0]]
        )
        np.testing.assert_array_equal(owned.faces, [[2, 0, 1]])
        assert owned.vertices.dtype == np.float32
        assert owned.faces.dtype == np.int32
        assert not owned.vertices.flags.writeable
        assert not owned.faces.flags.writeable
        with pytest.raises(ValueError):
            owned.vertices.setflags(write=True)
        with pytest.raises(ValueError):
            owned.faces.setflags(write=True)
    assert not np.shares_memory(mesh.vertices, copied.vertices)
    cloud = LinkMesh(
        "cloud", "body", mesh.vertices, np.empty((0, 3), dtype=np.int64), "none"
    )
    assert cloud.faces.shape == (0, 3)


@pytest.mark.parametrize(
    "vertices,faces",
    [
        ([[float("nan"), 0, 0]], [[0, 0, 0]]),
        ([[0.0, 0.0, 0.0]], [[0, 1, 0]]),
        ([[0.0, 0.0, 0.0]], [[-1, 0, 0]]),
        ([[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]]),
    ],
)
def test_source_mesh_rejects_invalid_geometry(vertices, faces):
    with pytest.raises(ValueError):
        LinkMesh("collider", "link", np.asarray(vertices), np.asarray(faces), "none")


def test_source_support_frame_preserves_raw_bounds_and_explicit_top():
    from embodichain.lab.sim.scene.declaration_placement import (
        source_support_target_pose,
    )

    support = {
        "shape": {"shape_type": "Cube", "size": [0.1, 0.2, 0.4]},
        "init_pos": [0.0, 0.0, 0.6],
    }
    child = {"shape": {"shape_type": "Cube", "size": [0.04, 0.06, 0.08]}}
    kwargs = dict(
        clearance=0.01,
        support_fallback_extents=(0.1, 0.1, 0.1),
        child_fallback_extents=(0.05, 0.05, 0.05),
    )
    pose = np.asarray(
        source_support_target_pose(support, child, axis_aligned=True, **kwargs)
    ).reshape(4, 4)
    assert pose[2, 3] == pytest.approx(0.25)
    other = np.asarray(
        source_support_target_pose(
            support, child, axis_aligned=False, world_support_top=0.9, **kwargs
        )
    ).reshape(4, 4)
    assert other[2, 3] == pytest.approx(0.35)


def test_collision_source_reader_retains_owner_frame_polygon_order_and_boundary(
    tmp_path,
):
    from pxr import Usd, UsdGeom, UsdPhysics
    from embodichain.lab.sim.scene.source_collisions import read_source_collision_meshes

    path = tmp_path / "colliders.usda"
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    owner = UsdGeom.Xform.Define(stage, "/owner")
    owner.AddTranslateOp().Set((10.0, 0.0, 0.0))
    UsdPhysics.RigidBodyAPI.Apply(owner.GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, "/owner/input")
    mesh.CreatePointsAttr(
        [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)]
    )
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr(True)
    disabled = UsdGeom.Mesh.Define(stage, "/owner/disabled")
    UsdPhysics.CollisionAPI.Apply(disabled.GetPrim()).CreateCollisionEnabledAttr(False)
    stage.GetRootLayer().Save()
    original = path.read_bytes()
    result = read_source_collision_meshes(
        {"fpath": str(path), "body_scale": [2.0, 2.0, 2.0]}
    )
    assert list(result) == [("owner", "/owner/input")]
    np.testing.assert_array_equal(
        result[("owner", "/owner/input")].vertices,
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 2.0, 0.0], [0.0, 2.0, 0.0]],
    )
    np.testing.assert_array_equal(
        result[("owner", "/owner/input")].faces, [[0, 1, 2], [0, 2, 3]]
    )
    assert path.read_bytes() == original
    with pytest.raises(ValueError, match="uniform scale"):
        read_source_collision_meshes(
            {"fpath": str(path), "body_scale": [1.0, 2.0, 1.0]}
        )
    mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2, 5])
    stage.GetRootLayer().Save()
    with pytest.raises(ValueError, match="invalid topology"):
        read_source_collision_meshes({"fpath": str(path)})


@pytest.fixture(autouse=True)
def forbid_native_runtime(monkeypatch):
    from embodichain.lab.sim.objects import Robot
    from embodichain.lab.sim.sim_manager import SimulationManager
    from embodichain.lab.sim.motion.solvers import BaseSolver

    def forbidden(*args, **kwargs):
        pytest.fail("authored placement inputs constructed a native runtime")

    monkeypatch.setattr(Robot, "__init__", forbidden)
    monkeypatch.setattr(SimulationManager, "__init__", forbidden)
    monkeypatch.setattr(BaseSolver, "__init__", forbidden)


@pytest.fixture
def declaration_placement_reference(tmp_path):
    import json
    from pathlib import Path
    import trimesh

    reference = json.loads(
        (Path(__file__).parent / "data/declaration_placement_b0.json").read_text()
    )
    sources = deepcopy(reference["sources"])
    for name, geometry in reference["meshes"].items():
        path = tmp_path / (name + ".glb")
        trimesh.Trimesh(
            vertices=geometry["vertices"], faces=geometry["faces"], process=False
        ).export(path)
        sources[name]["shape"]["fpath"] = str(path)
    return reference, sources


def test_source_container_adapter_preserves_b0_reservations_and_tilt_boundary(
    declaration_placement_reference,
):
    from embodichain.lab.sim.scene.declaration_placement import (
        source_container_target_pose,
    )

    reference, sources = declaration_placement_reference
    before = deepcopy(sources)
    arm = np.asarray(reference["container"]["arm_root_world"])
    first = source_container_target_pose(sources["tray"], sources["child"], arm)
    second = source_container_target_pose(
        sources["tray"],
        sources["child"],
        arm,
        occupied=[(first, sources["child"])],
        occupied_rotations=[np.eye(3)],
    )
    assert first == reference["container"]["first"]
    assert second == reference["container"]["second"]
    tilted = deepcopy(sources["tray"])
    tilted["init_rot"] = [0.0, 6.0, 0.0]
    assert source_container_target_pose(tilted, sources["child"], arm) is None
    with pytest.raises(ValueError, match="rotations must match"):
        source_container_target_pose(
            sources["tray"],
            sources["child"],
            arm,
            occupied=[(first, sources["child"])],
            occupied_rotations=[np.eye(3), np.eye(3)],
        )
    assert sources == before


@pytest.mark.parametrize("case_index", [0, 1])
def test_source_pour_adapter_matches_original_b0_staging_and_pivot(
    declaration_placement_reference,
    case_index,
):
    from embodichain.lab.sim.scene.declaration_placement import source_pour_target_pose

    reference, sources = declaration_placement_reference
    case = reference["pour"]["cases"][case_index]
    sources["vessel"]["init_rot"] = case["angles"]
    before = deepcopy(sources)
    result = source_pour_target_pose(
        sources["vessel"],
        sources["receiver"],
        finger_radius=reference["pour"]["finger_radius"],
    )
    assert result == tuple(case["expected"])
    assert sources == before


def test_source_horizontal_selector_retains_role_and_authored_pose_boundary(
    declaration_placement_reference,
):
    from embodichain.lab.sim.scene.declaration_placement import (
        source_horizontal_slender_object,
    )

    _, sources = declaration_placement_reference
    source = deepcopy(sources["vessel"])
    # A 5:1 full-mesh extent has an unambiguous authored long axis.
    source["body_scale"] = [1.0, 1.0, 2.0]
    source["init_rot"] = [90.0, 0.0, 0.0]
    assert source_horizontal_slender_object(source) is True
    source["init_rot"] = [0.0, 0.0, 0.0]
    assert source_horizontal_slender_object(source) is False
    source["role"] = "articulation"
    assert source_horizontal_slender_object(source) is False
