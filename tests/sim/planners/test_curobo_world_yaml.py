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

"""Dependency-free unit tests for cuRobo world-YAML generation from RigidObjects.

The core conversion (``cuboid`` / ``mesh``) never requires CUDA or the real
``curobo`` package - only the optional cuRobo round-trip test uses
``pytest.importorskip``. A ``_FakeRigidObject`` stands in for the simulator
object so these tests exercise the full :func:`generate_curobo_world_yaml`
assembly path without dexsim.
"""

from __future__ import annotations

import math

import pytest
import torch
import yaml

from embodichain.lab.sim.planners.curobo.curobo_planner import CuroboWorldCfg
from embodichain.lab.sim.planners.curobo.curobo_yaml import (
    _mesh_to_obstacle_entry,
    generate_curobo_world_yaml,
)


def _unit_cube_vertices() -> torch.Tensor:
    """8 vertices of a unit cube centered at the local-frame origin."""
    s = 0.5
    return torch.tensor(
        [
            [-s, -s, -s],
            [s, -s, -s],
            [s, s, -s],
            [-s, s, -s],
            [-s, -s, s],
            [s, -s, s],
            [s, s, s],
            [-s, s, s],
        ],
        dtype=torch.float32,
    )


def _cube_faces() -> torch.Tensor:
    """12 triangle indices into :func:`_unit_cube_vertices`."""
    return torch.tensor(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [2, 3, 7],
            [2, 7, 6],
            [1, 2, 6],
            [1, 6, 5],
            [0, 3, 7],
            [0, 7, 4],
        ],
        dtype=torch.int32,
    )


def _identity_pose(trans=(0.45, 0.0, 0.18)) -> torch.Tensor:
    return torch.tensor([*trans, 1.0, 0.0, 0.0, 0.0], dtype=torch.float32)


class _FakeRigidObject:
    """Minimal stand-in exposing the mesh/pose API used by the generator."""

    def __init__(
        self, uid: str, vertices: torch.Tensor, faces: torch.Tensor, pose: torch.Tensor
    ) -> None:
        self.uid = uid
        self._vertices = vertices
        self._faces = faces
        self._pose = pose

    def get_vertices(self, env_ids=None, scale=False):  # noqa: ARG002
        return self._vertices.unsqueeze(0)

    def get_triangles(self, env_ids=None):  # noqa: ARG002
        return self._faces.unsqueeze(0)

    def get_local_pose(self, to_matrix=False):  # noqa: ARG002
        return self._pose.unsqueeze(0)


# ---------------------------------------------------------------------------
# _mesh_to_obstacle_entry: cuboid
# ---------------------------------------------------------------------------


def test_cuboid_entry_centered_mesh_matches_aabb_and_pose():
    """A centered mesh + identity pose yields dims == AABB and center == translation."""
    entries = _mesh_to_obstacle_entry(
        "demo_block",
        _unit_cube_vertices(),
        _cube_faces(),
        _identity_pose(),
        representation="cuboid",
    )
    assert len(entries) == 1
    top_key, name, fields = entries[0]
    assert (top_key, name) == ("cuboid", "demo_block")
    assert fields["dims"] == pytest.approx([1.0, 1.0, 1.0])
    # Centered mesh -> cuboid center coincides with the pose translation, which
    # is the convention the cuRobo planner's auto-generated world YAML uses.
    assert fields["pose"][:3] == pytest.approx([0.45, 0.0, 0.18])
    assert fields["pose"][3:] == pytest.approx([1.0, 0.0, 0.0, 0.0])


def test_cuboid_entry_off_origin_mesh_offsets_center():
    """A mesh whose origin is a corner offsets the cuboid center by the AABB center."""
    verts = _unit_cube_vertices() + 0.5  # now spans [0, 1]^3; center_local = 0.5
    _, _, fields = _mesh_to_obstacle_entry(
        "b", verts, _cube_faces(), _identity_pose(), representation="cuboid"
    )[0]
    assert fields["dims"] == pytest.approx([1.0, 1.0, 1.0])
    # center_world = pose translation + center_local (identity rotation).
    assert fields["pose"][:3] == pytest.approx([0.95, 0.5, 0.68])


def test_cuboid_entry_rotated_pose_keeps_centered_center_at_translation():
    """A centered mesh under a rotated pose keeps the cuboid center at the translation."""
    qz = torch.tensor(
        [math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)], dtype=torch.float32
    )
    pose = torch.cat([torch.tensor([0.45, 0.0, 0.18]), qz])
    _, _, fields = _mesh_to_obstacle_entry(
        "b", _unit_cube_vertices(), _cube_faces(), pose, representation="cuboid"
    )[0]
    assert fields["dims"] == pytest.approx([1.0, 1.0, 1.0])
    assert fields["pose"][:3] == pytest.approx([0.45, 0.0, 0.18])
    assert fields["pose"][3:] == pytest.approx(qz.tolist())


def test_cuboid_entry_accepts_homogeneous_pose():
    """A (4, 4) pose matrix is accepted and converted to xyz-quaternion."""
    mat = torch.eye(4, dtype=torch.float32)
    mat[:3, 3] = torch.tensor([0.45, 0.0, 0.18])
    _, _, fields = _mesh_to_obstacle_entry(
        "b", _unit_cube_vertices(), _cube_faces(), mat, representation="cuboid"
    )[0]
    assert fields["pose"][:3] == pytest.approx([0.45, 0.0, 0.18])
    assert fields["pose"][3:] == pytest.approx([1.0, 0.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# _mesh_to_obstacle_entry: mesh
# ---------------------------------------------------------------------------


def test_mesh_entry_serializes_flat_face_buffer():
    """The mesh representation flattens faces to 3 ints per triangle."""
    top_key, name, fields = _mesh_to_obstacle_entry(
        "demo_block",
        _unit_cube_vertices(),
        _cube_faces(),
        _identity_pose(),
        representation="mesh",
    )[0]
    assert (top_key, name) == ("mesh", "demo_block")
    assert len(fields["vertices"]) == 8
    assert len(fields["faces"]) == 12 * 3  # 12 triangles, 3 indices each
    assert fields["pose"] == pytest.approx(_identity_pose().tolist())


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def test_invalid_representation_raises():
    with pytest.raises(ValueError, match="representation"):
        _mesh_to_obstacle_entry(
            "x",
            _unit_cube_vertices(),
            _cube_faces(),
            _identity_pose(),
            representation="banana",
        )


def test_empty_mesh_raises_for_cuboid():
    with pytest.raises(ValueError, match="no vertices"):
        _mesh_to_obstacle_entry(
            "x",
            torch.zeros((0, 3), dtype=torch.float32),
            torch.zeros((0, 3), dtype=torch.int32),
            _identity_pose(),
            representation="cuboid",
        )


# ---------------------------------------------------------------------------
# generate_curobo_world_yaml: assembly
# ---------------------------------------------------------------------------


def test_generate_cuboid_world_yaml_assembles_schema(tmp_path):
    obj = _FakeRigidObject(
        "demo_block", _unit_cube_vertices(), _cube_faces(), _identity_pose()
    )
    out = tmp_path / "world.yml"
    result = generate_curobo_world_yaml([obj], str(out), representation="cuboid")
    assert result == str(out)
    data = yaml.safe_load(out.read_text())
    assert list(data.keys()) == ["cuboid"]
    block = data["cuboid"]["demo_block"]
    assert block["dims"] == pytest.approx([1.0, 1.0, 1.0])
    assert block["pose"][:3] == pytest.approx([0.45, 0.0, 0.18])


def test_generate_mesh_world_yaml_assembles_schema(tmp_path):
    obj = _FakeRigidObject(
        "demo_block", _unit_cube_vertices(), _cube_faces(), _identity_pose()
    )
    out = tmp_path / "world_mesh.yml"
    generate_curobo_world_yaml([obj], str(out), representation="mesh")
    data = yaml.safe_load(out.read_text())
    assert list(data.keys()) == ["mesh"]
    assert len(data["mesh"]["demo_block"]["vertices"]) == 8


def test_generate_world_yaml_supports_multiple_objects(tmp_path):
    pose_a = _identity_pose((0.45, 0.0, 0.18))
    pose_b = _identity_pose((0.0, 0.3, 0.1))
    objs = [
        _FakeRigidObject("block_a", _unit_cube_vertices(), _cube_faces(), pose_a),
        _FakeRigidObject("block_b", _unit_cube_vertices(), _cube_faces(), pose_b),
    ]
    out = tmp_path / "multi.yml"
    generate_curobo_world_yaml(objs, str(out), representation="cuboid")
    data = yaml.safe_load(out.read_text())
    assert set(data["cuboid"].keys()) == {"block_a", "block_b"}
    assert data["cuboid"]["block_b"]["pose"][:3] == pytest.approx([0.0, 0.3, 0.1])


def test_generate_world_yaml_rejects_empty_input(tmp_path):
    with pytest.raises(ValueError, match="at least one"):
        generate_curobo_world_yaml([], str(tmp_path / "x.yml"))


def test_generate_world_yaml_rejects_duplicate_names(tmp_path):
    pose = _identity_pose()
    a = _FakeRigidObject("block", _unit_cube_vertices(), _cube_faces(), pose)
    b = _FakeRigidObject("block", _unit_cube_vertices(), _cube_faces(), pose)
    with pytest.raises(ValueError, match="Duplicate"):
        generate_curobo_world_yaml([a, b], str(tmp_path / "y.yml"))


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


def test_curobo_world_cfg_defaults():
    cfg = CuroboWorldCfg()
    assert cfg.obstacle_representation == "cuboid"
    assert cfg.rigid_objects is None
    assert not hasattr(cfg, "world_config_path")


# ---------------------------------------------------------------------------
# cuRobo round-trip (optional; skipped without cuRobo)
# ---------------------------------------------------------------------------


def test_generated_yaml_loads_in_curobo_scene_cfg(tmp_path):
    """The generated YAML must be accepted by cuRobo's SceneCfg.create."""
    pytest.importorskip("curobo")
    from curobo._src.geom.types import SceneCfg

    obj = _FakeRigidObject(
        "demo_block", _unit_cube_vertices(), _cube_faces(), _identity_pose()
    )
    out = tmp_path / "world.yml"
    generate_curobo_world_yaml([obj], str(out), representation="cuboid")
    data = yaml.safe_load(out.read_text())
    scene = SceneCfg.create(data)
    assert len(scene.cuboid) == 1
    assert scene.cuboid[0].name == "demo_block"
    assert scene.cuboid[0].dims == pytest.approx([1.0, 1.0, 1.0])


def test_generated_mesh_yaml_loads_in_curobo_scene_cfg(tmp_path):
    pytest.importorskip("curobo")
    from curobo._src.geom.types import SceneCfg

    obj = _FakeRigidObject(
        "demo_block", _unit_cube_vertices(), _cube_faces(), _identity_pose()
    )
    out = tmp_path / "world_mesh.yml"
    generate_curobo_world_yaml([obj], str(out), representation="mesh")
    data = yaml.safe_load(out.read_text())
    scene = SceneCfg.create(data)
    assert len(scene.mesh) == 1
    assert scene.mesh[0].name == "demo_block"
    assert len(scene.mesh[0].vertices) == 8
