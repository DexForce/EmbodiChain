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

"""Authored collision frames, source order and explicit surface evidence."""

from __future__ import annotations

import math

import numpy as np
import pytest
import trimesh

from embodichain.lab.sim.utility.urdf_collision import load_urdf_collision_inputs


def _urdf(tmp_path, *, degenerate=False):
    mesh = trimesh.Trimesh(
        vertices=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0] if degenerate else [0.0, 1.0, 0.0],
        ],
        faces=[[0, 1, 2]],
        process=False,
    )
    mesh.export(tmp_path / "mesh.ply")
    path = tmp_path / "model.urdf"
    path.write_text(f"""<robot name="fixture">
      <link name="first">
        <collision><origin xyz="1 2 3" rpy="0 0 {math.pi / 2}"/>
          <geometry><mesh filename="mesh.ply" scale="2 3 1"/></geometry></collision>
        <collision><geometry><box size="0.2 0.4 0.6"/></geometry></collision>
      </link>
      <link name="second"><collision><geometry><box size="1 1 1"/></geometry></collision></link>
      <link name="visual_only"><visual><geometry><box size="2 2 2"/></geometry></visual></link>
    </robot>""")
    return path


@pytest.mark.parametrize("vertices_only", [False, True])
def test_explicit_link_order_and_scale_origin_order_are_preserved(
    tmp_path, vertices_only
):
    path = _urdf(tmp_path)
    result = load_urdf_collision_inputs(
        path,
        ("second", "first"),
        body_scale=(2.0, 1.0, 0.5),
        vertices_only=vertices_only,
    )
    assert tuple(result) == ("second", "first")
    mesh, box = result["first"]
    np.testing.assert_allclose(
        mesh.vertices,
        [[2.0, 2.0, 1.5], [2.0, 4.0, 1.5], [-4.0, 2.0, 1.5]],
        rtol=0,
        atol=1e-14,
    )
    np.testing.assert_allclose(box.vertices.min(0), [-0.2, -0.2, -0.15])
    np.testing.assert_allclose(box.vertices.max(0), [0.2, 0.2, 0.15])
    assert mesh.owner_path == "first" and "#first/collision[0]:" in mesh.path
    assert "#first/collision[1]:box:" in box.path
    assert mesh.vertices.dtype == np.float64 and mesh.faces.dtype == np.int64
    assert not mesh.vertices.flags.writeable and not mesh.faces.flags.writeable
    if vertices_only:
        assert mesh.faces.shape == box.faces.shape == (0, 3)
        assert mesh.approximation == "urdfMeshVertexEnvelope"
        assert box.approximation == "urdfBoxVertexEnvelope"
    else:
        np.testing.assert_array_equal(mesh.faces, [[0, 1, 2]])
        assert mesh.approximation == "urdfMeshInput"


def test_degenerate_triangles_are_only_usable_as_vertex_envelopes(tmp_path):
    path = _urdf(tmp_path, degenerate=True)
    with pytest.raises(ValueError, match="degenerate"):
        load_urdf_collision_inputs(path, ("first",))
    result = load_urdf_collision_inputs(path, ("first",), vertices_only=True)
    assert result["first"][0].vertices.shape == (3, 3)
    assert result["first"][0].faces.shape == (0, 3)


@pytest.mark.parametrize("names", [("missing",), ("first", "first"), ("visual_only",)])
def test_missing_duplicate_and_visual_only_selection_is_rejected(tmp_path, names):
    with pytest.raises(ValueError):
        load_urdf_collision_inputs(_urdf(tmp_path), names)


@pytest.mark.parametrize(
    "geometry", ['<sphere radius="1"/>', '<mesh filename="asset://shape.obj"/>']
)
def test_unsupported_shapes_or_unresolved_uri_never_substitute_geometry(
    tmp_path, geometry
):
    path = tmp_path / "model.urdf"
    path.write_text(
        f'<robot name="fixture"><link name="part"><collision><geometry>{geometry}</geometry></collision></link></robot>'
    )
    with pytest.raises(ValueError):
        load_urdf_collision_inputs(path, ("part",))
