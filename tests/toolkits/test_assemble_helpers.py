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

"""Standalone assembly helper contracts, without model calls or a renderer."""

from __future__ import annotations

from pathlib import Path
import sys
from types import ModuleType

import numpy as np
import pytest
import trimesh

from scripts.tools.assemble import _json_io
from scripts.tools.assemble._geometry import load_mesh
from scripts.tools.assemble._json_io import pose_matrix, read_json, write_json


@pytest.mark.parametrize(
    "source",
    [
        '{"value": NaN}',
        '{"value": Infinity}',
        '{"value": -Infinity}',
        '{"nested": [1e999]}',
        "[]",
        "null",
    ],
)
def test_read_json_rejects_nonfinite_numbers_and_nonobject_roots(
    tmp_path: Path, source: str
) -> None:
    path = tmp_path / "invalid.json"
    path.write_text(source, encoding="utf-8")
    with pytest.raises(ValueError):
        read_json(path)


def test_json_round_trip_preserves_unicode_and_replaces_checkpoint(
    tmp_path: Path,
) -> None:
    path = tmp_path / "nested" / "result.json"
    write_json(path, {"status": "running"})
    result = {"description": "树状支架", "T_base_assemble": np.eye(4).tolist()}
    write_json(path, result)
    assert read_json(path) == result
    assert "树状支架" in path.read_text(encoding="utf-8")
    assert list(path.parent.iterdir()) == [path]


@pytest.mark.parametrize("invalid", [{"v": float("nan")}, {"v": float("inf")}, []])
def test_invalid_json_write_preserves_previous_checkpoint(
    tmp_path: Path, invalid: object
) -> None:
    path = tmp_path / "result.json"
    write_json(path, {"status": "running"})
    before = path.read_bytes()
    with pytest.raises(ValueError):
        write_json(path, invalid)
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def test_atomic_write_failure_cleans_up_and_preserves_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "result.json"
    write_json(path, {"status": "running"})

    def fail_replace(source: Path, target: Path) -> None:
        # Until replacement, readers must still observe a complete old record.
        assert read_json(target) == {"status": "running"}
        assert read_json(source) == {"status": "success"}
        raise OSError("replacement failed")

    monkeypatch.setattr(_json_io.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replacement failed"):
        write_json(path, {"status": "success"})
    assert read_json(path) == {"status": "running"}
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize(
    "invalid",
    [
        np.eye(3),
        np.full((4, 4), float("nan")),
        np.diag([1, 1, 1, 2]),
        np.diag([-1, 1, 1, 1]),
        np.diag([2, 1, 1, 1]),
        [[1, 0.1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        np.eye(4).astype(str).tolist(),
        [[1], [0, 1]],
    ],
)
def test_pose_matrix_rejects_nonrigid_or_malformed_values(invalid: object) -> None:
    with pytest.raises(ValueError):
        pose_matrix(invalid)


def test_pose_matrix_preserves_relative_frame_and_returns_independent_array() -> None:
    pose = trimesh.transformations.rotation_matrix(0.4, [1, 2, 3])
    pose[:3, 3] = [0.1, -0.2, 0.3]
    result = pose_matrix(pose.tolist())
    np.testing.assert_allclose(result, pose)
    assert result.dtype == np.float64
    pose_matrix(pose)[0, 3] = 99
    assert pose[0, 3] == 0.1


@pytest.fixture
def cup() -> trimesh.Trimesh:
    # A 90 mm hollow cup with a 6 mm floor and a +Z mouth.
    return trimesh.creation.revolve(
        [
            [0, 0],
            [0.04, 0],
            [0.04, 0.09],
            [0.035, 0.09],
            [0.035, 0.006],
            [0, 0.006],
            [0, 0],
        ],
        sections=64,
    )


def test_load_welds_seams_preserves_cavity_and_rejects_open_surface(
    tmp_path: Path, cup: trimesh.Trimesh
) -> None:
    path = tmp_path / "cup.obj"
    unmerged = cup.copy()
    unmerged.unmerge_vertices()
    unmerged.export(path)
    loaded = load_mesh(path)
    assert loaded.is_volume
    assert loaded.volume == pytest.approx(cup.volume, rel=1e-5)
    assert loaded.volume < loaded.convex_hull.volume / 2
    cup.update_faces(np.arange(len(cup.faces) - 1))
    cup.export(path)
    with pytest.raises(ValueError, match="not watertight"):
        load_mesh(path)


def test_load_mesh_converts_units(tmp_path: Path) -> None:
    path = tmp_path / "millimeters.obj"
    trimesh.creation.box([20, 40, 60]).export(path)
    np.testing.assert_allclose(load_mesh(path, scale=0.001).extents, [0.02, 0.04, 0.06])


@pytest.mark.parametrize("scale", [0, -1, float("nan"), float("inf")])
def test_load_mesh_rejects_invalid_scale(tmp_path: Path, scale: float) -> None:
    with pytest.raises(ValueError, match="scale"):
        load_mesh(tmp_path / "unused.obj", scale)


@pytest.fixture
def collision_module() -> ModuleType:
    pytest.importorskip("fcl")
    pytest.importorskip("manifold3d")
    from scripts.tools.assemble import _collision

    return _collision


def test_containment_detected_without_triangle_intersection(
    collision_module: ModuleType,
) -> None:
    outer = trimesh.creation.box([1, 1, 1])
    inner = trimesh.creation.box([0.1, 0.1, 0.1])
    exact = collision_module.ExactCollision(outer, inner)
    assert not exact.surface_collision(np.eye(4))
    assert exact.intersection_volume(np.eye(4)) == pytest.approx(0.001)
    assert collision_module.HullCollision([outer], [inner]).collides(np.eye(4))


def test_original_collision_keeps_cavity_even_when_hulls_overlap(
    collision_module: ModuleType, cup: trimesh.Trimesh
) -> None:
    inner = trimesh.creation.box([0.01, 0.01, 0.01])
    pose = np.eye(4)
    pose[2, 3] = 0.05
    exact = collision_module.ExactCollision(cup, inner)
    assert collision_module.HullCollision([cup.convex_hull], [inner]).collides(pose)
    assert not exact.surface_collision(pose)
    assert exact.intersection_volume(pose) == 0
    assert exact.distance(pose) > 0
    pose[2, 3] = 0.006
    assert exact.surface_collision(pose)
    assert exact.intersection_volume(pose) > 0
    assert exact.distance(pose) == 0


def test_hull_collision_applies_rotation_and_translation(
    collision_module: ModuleType,
) -> None:
    base = trimesh.creation.box([0.1, 0.1, 0.1])
    moving = trimesh.creation.box([0.3, 0.02, 0.02])
    hulls = collision_module.HullCollision([base], [moving])
    pose = np.eye(4)
    pose[0, 3] = 0.15
    assert hulls.collides(pose)
    pose[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    assert not hulls.collides(pose)


def test_visacd_cache_reuses_hulls_and_invalidates_changed_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, collision_module: ModuleType
) -> None:
    pytest.importorskip("open3d")
    backend = ModuleType("dexsim.kit.meshproc")
    calls = []

    def fake_visacd(mesh: object, **settings: object) -> tuple[bool, list]:
        calls.append(settings)
        return True, [mesh]

    backend.convex_decomposition_visacd = fake_visacd
    monkeypatch.setitem(sys.modules, "dexsim.kit.meshproc", backend)
    monkeypatch.setattr(collision_module, "version", lambda name: "test-v1")
    mesh = trimesh.creation.box([0.1, 0.2, 0.3])
    first = collision_module.decompose(mesh, tmp_path, 0.015, 32)
    cached = collision_module.decompose(mesh, tmp_path, 0.015, 32)
    assert len(calls) == 1
    assert len(first) == len(cached) == 1
    np.testing.assert_array_equal(first[0].vertices, cached[0].vertices)
    np.testing.assert_array_equal(first[0].faces, cached[0].faces)
    assert calls[0]["max_convex_hull_num"] == 32
    collision_module.decompose(mesh, tmp_path, 0.02, 32)
    collision_module.decompose(mesh, tmp_path, 0.02, 16)
    mesh.apply_scale(2)
    collision_module.decompose(mesh, tmp_path, 0.02, 16)
    monkeypatch.setattr(collision_module, "version", lambda name: "test-v2")
    collision_module.decompose(mesh, tmp_path, 0.02, 16)
    assert len(calls) == 5


def test_visacd_failure_does_not_cache_empty_hulls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, collision_module: ModuleType
) -> None:
    pytest.importorskip("open3d")
    backend = ModuleType("dexsim.kit.meshproc")
    backend.convex_decomposition_visacd = lambda *args, **kwargs: (False, [])
    monkeypatch.setitem(sys.modules, "dexsim.kit.meshproc", backend)
    monkeypatch.setattr(collision_module, "version", lambda name: "test-v1")
    with pytest.raises(RuntimeError, match="VISACD returned no hulls"):
        collision_module.decompose(trimesh.creation.box(), tmp_path, 0.015, 32)
    assert not list(tmp_path.iterdir())
