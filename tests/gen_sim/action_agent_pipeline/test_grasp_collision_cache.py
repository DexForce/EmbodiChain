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

import hashlib
import pickle

import numpy as np
import torch

import embodichain.lab.sim
from embodichain.gen_sim.action_agent_pipeline.runtime import atom_actions
from embodichain.gen_sim.action_agent_pipeline.runtime import grasp_collision_cache
from embodichain.gen_sim.action_agent_pipeline.runtime.grasp_collision_cache import (
    ensure_vhacd_grasp_collision_cache,
    main_grasp_collision_cache_path,
)
from embodichain.toolkits.graspkit.pg_grasp import collision_checker

_MAX_HULLS = 4


def _tetra_mesh() -> tuple[torch.Tensor, torch.Tensor]:
    vertices = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    triangles = torch.tensor(
        [
            [0, 2, 1],
            [0, 1, 3],
            [1, 2, 3],
            [2, 0, 3],
        ],
        dtype=torch.int64,
    )
    return vertices, triangles


def _tetra_plane_equations() -> list[tuple[np.ndarray, np.ndarray]]:
    normals = np.array(
        [
            [0.0, 0.0, -1.0],
            [0.0, -1.0, 0.0],
            [1.0, 1.0, 1.0],
            [-1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    offsets = np.array([0.0, 0.0, -1.0, 0.0], dtype=np.float32)
    return [(normals, offsets)]


def test_main_cache_path_matches_pg_grasp_contract(tmp_path) -> None:
    vertices, triangles = _tetra_mesh()
    expected_hash = hashlib.md5(
        vertices.numpy().tobytes() + triangles.numpy().tobytes()
    ).hexdigest()

    cache_path = main_grasp_collision_cache_path(
        vertices,
        triangles,
        _MAX_HULLS,
        cache_dir=tmp_path,
    )

    assert cache_path == tmp_path / f"{expected_hash}_{_MAX_HULLS}.pkl"


def test_vhacd_cache_is_loaded_by_main_without_coacd(monkeypatch, tmp_path) -> None:
    vertices, triangles = _tetra_mesh()
    monkeypatch.setattr(
        grasp_collision_cache,
        "_compute_vhacd_plane_equations",
        lambda *args: _tetra_plane_equations(),
    )
    result = ensure_vhacd_grasp_collision_cache(
        mesh_vertices=vertices,
        mesh_triangles=triangles,
        max_decomposition_hulls=_MAX_HULLS,
        cache_dir=tmp_path,
    )

    def fail_coacd(*args, **kwargs):
        raise AssertionError("Main must load the prepared V-HACD cache")

    monkeypatch.setattr(embodichain.lab.sim, "CONVEX_DECOMP_DIR", tmp_path)
    monkeypatch.setattr(collision_checker, "convex_decomposition_coacd", fail_coacd)
    checker = collision_checker.ConvexCollisionChecker(
        vertices,
        triangles,
        max_decomposition_hulls=_MAX_HULLS,
    )

    assert result["status"] == "generated"
    assert checker.cache_path == result["grasp_cache_path"]
    assert checker.plane_equations["plane_equation_counts"].tolist() == [4]


def test_vhacd_cache_replaces_unlabelled_or_modified_cache(
    monkeypatch, tmp_path
) -> None:
    vertices, triangles = _tetra_mesh()
    compute_calls = []

    def fake_compute(*args):
        compute_calls.append(args)
        return _tetra_plane_equations()

    monkeypatch.setattr(
        grasp_collision_cache,
        "_compute_vhacd_plane_equations",
        fake_compute,
    )
    kwargs = {
        "mesh_vertices": vertices,
        "mesh_triangles": triangles,
        "max_decomposition_hulls": _MAX_HULLS,
        "cache_dir": tmp_path,
    }

    first = ensure_vhacd_grasp_collision_cache(**kwargs)
    second = ensure_vhacd_grasp_collision_cache(**kwargs)
    with open(first["grasp_cache_path"], "wb") as cache_file:
        cache_file.write(b"unlabelled-coacd-cache")
    third = ensure_vhacd_grasp_collision_cache(**kwargs)

    assert first["status"] == "generated"
    assert second["status"] == "hit"
    assert third["status"] == "replaced"
    assert len(compute_calls) == 2
    with open(third["grasp_cache_path"], "rb") as cache_file:
        cache = pickle.load(cache_file)
    assert set(cache) == {"plane_equations", "plane_equation_counts"}


def test_vhacd_plane_generation_uses_vhacd_backend(monkeypatch) -> None:
    vertices, triangles = _tetra_mesh()
    calls = []

    def fake_vhacd(mesh, *, max_convex_hull_num):
        calls.append(max_convex_hull_num)
        return True, [mesh]

    import dexsim.kit.meshproc

    monkeypatch.setattr(
        dexsim.kit.meshproc,
        "convex_decomposition_vhacd",
        fake_vhacd,
    )

    equations = grasp_collision_cache._compute_vhacd_plane_equations(
        vertices,
        triangles,
        _MAX_HULLS,
    )

    assert calls == [_MAX_HULLS]
    assert len(equations) == 1


def test_atomic_runtime_prepares_vhacd_cache(monkeypatch) -> None:
    vertices, triangles = _tetra_mesh()
    received = {}

    def fake_prepare(**kwargs):
        received.update(kwargs)
        return {
            "status": "hit",
            "grasp_cache_path": "/tmp/cache.pkl",
            "metadata_path": "/tmp/cache.pkl.action_agent.json",
        }

    monkeypatch.setattr(
        atom_actions, "ensure_vhacd_grasp_collision_cache", fake_prepare
    )

    atom_actions._prepare_grasp_collision_cache(
        obj_name="apple",
        mesh_vertices=vertices,
        mesh_triangles=triangles,
        source_mesh_path=None,
        max_decomposition_hulls=_MAX_HULLS,
        convex_decomposition_method="vhacd",
        body_scale=None,
        runtime_kwargs={},
    )

    assert received["mesh_vertices"] is vertices
    assert received["mesh_triangles"] is triangles
    assert received["max_decomposition_hulls"] == _MAX_HULLS
