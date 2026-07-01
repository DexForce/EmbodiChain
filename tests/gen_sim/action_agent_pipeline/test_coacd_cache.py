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

import pytest
import torch

from embodichain.gen_sim.action_agent_pipeline.generation.coacd_cache import (
    coacd_cache_path_for_mesh,
    dexsim_coacd_cache_key_for_mesh,
)
from embodichain.gen_sim.action_agent_pipeline.runtime.coacd_cache_bridge import (
    ensure_grasp_collision_cache_from_env_coacd,
)


def test_coacd_cache_path_matches_dexsim_load_actor_key(tmp_path) -> None:
    mesh_path = tmp_path / "object.obj"
    mesh_path.write_text("# placeholder mesh\n", encoding="utf-8")
    cache_dir = tmp_path / "cache"

    cache_path = coacd_cache_path_for_mesh(
        mesh_path,
        16,
        cache_dir,
    )

    expected_key = hashlib.sha256(
        f"{mesh_path.resolve()}|mesh_count=1".encode("utf-8")
    ).hexdigest()
    assert dexsim_coacd_cache_key_for_mesh(mesh_path) == expected_key
    assert cache_path == cache_dir.resolve() / f"{expected_key}_16.obj"


def test_grasp_cache_bridge_uses_existing_env_coacd_obj(tmp_path) -> None:
    pytest.importorskip("dexsim.kit.meshproc.convex_cache")
    source_mesh_path = tmp_path / "source.obj"
    _write_tetra_obj(source_mesh_path)

    cache_dir = tmp_path / "cache"
    env_cache_path = coacd_cache_path_for_mesh(
        source_mesh_path,
        4,
        cache_dir,
    )
    env_cache_path.parent.mkdir(parents=True, exist_ok=True)
    _write_tetra_obj(env_cache_path)

    mesh_vertices = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 2.0],
        ],
        dtype=torch.float32,
    )
    mesh_triangles = torch.tensor(
        [
            [0, 2, 1],
            [0, 1, 3],
            [1, 2, 3],
            [2, 0, 3],
        ],
        dtype=torch.int64,
    )

    result = ensure_grasp_collision_cache_from_env_coacd(
        mesh_vertices=mesh_vertices,
        mesh_triangles=mesh_triangles,
        source_mesh_path=source_mesh_path,
        max_decomposition_hulls=4,
        body_scale=[2.0, 2.0, 2.0],
        cache_dir=cache_dir,
    )

    assert result["status"] == "generated"
    assert result["env_cache_path"] == env_cache_path.as_posix()
    with open(result["grasp_cache_path"], "rb") as cache_file:
        cache = pickle.load(cache_file)
    assert set(cache) == {"plane_equations", "plane_equation_counts"}
    assert cache["plane_equations"].shape[-1] == 4
    assert cache["plane_equation_counts"].numel() == 1
    assert not list(env_cache_path.parent.glob("*.tmp.*"))

    second_result = ensure_grasp_collision_cache_from_env_coacd(
        mesh_vertices=mesh_vertices,
        mesh_triangles=mesh_triangles,
        source_mesh_path=source_mesh_path,
        max_decomposition_hulls=4,
        body_scale=[2.0, 2.0, 2.0],
        cache_dir=cache_dir,
    )
    assert second_result["status"] == "hit"


def _write_tetra_obj(path) -> None:
    path.write_text(
        "\n".join(
            [
                "o convex_0",
                "v 0.0 0.0 0.0",
                "v 1.0 0.0 0.0",
                "v 0.0 1.0 0.0",
                "v 0.0 0.0 1.0",
                "f 1 3 2",
                "f 1 2 4",
                "f 2 3 4",
                "f 3 1 4",
                "",
            ]
        ),
        encoding="utf-8",
    )
