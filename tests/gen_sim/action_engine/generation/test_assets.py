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

"""Tests for generated Action Engine runtime meshes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import trimesh

from embodichain.gen_sim.action_engine.generation import assets


def test_bake_glb_splits_face_corners_for_renderer_safe_normals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.glb"
    destination = tmp_path / "baked.glb"
    trimesh.creation.box().export(source, file_type="glb")
    monkeypatch.setattr(
        assets,
        "_has_inconsistent_shading_normals",
        lambda _mesh: True,
    )

    assets._bake_glb(source, destination, [1.0, 1.0, 1.0])

    baked = trimesh.load(destination, force="scene", process=False)
    geometry = tuple(baked.geometry.values())
    assert len(geometry) == 1
    assert len(geometry[0].vertices) == 3 * len(geometry[0].faces)


def test_bake_glb_preserves_safe_mesh_topology(tmp_path: Path) -> None:
    source = tmp_path / "source.glb"
    destination = tmp_path / "baked.glb"
    trimesh.creation.box().export(source, file_type="glb")
    source_scene = trimesh.load(source, force="scene", process=False)
    source_mesh = tuple(source_scene.geometry.values())[0]

    assets._bake_glb(source, destination, [1.0, 1.0, 1.0])

    baked = trimesh.load(destination, force="scene", process=False)
    geometry = tuple(baked.geometry.values())
    assert len(geometry) == 1
    assert len(geometry[0].vertices) == len(source_mesh.vertices)


def test_inconsistent_shading_normals_detect_opposed_face_corner() -> None:
    mesh = trimesh.Trimesh(
        vertices=np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ),
        faces=np.asarray(
            [
                [0, 1, 2],
                [0, 1, 2],
                [0, 1, 2],
                [0, 2, 1],
            ]
        ),
        process=False,
    )

    assert assets._has_inconsistent_shading_normals(mesh) is True
