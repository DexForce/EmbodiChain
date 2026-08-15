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
import json
import os
from pathlib import Path
import pickle
from typing import Callable

import numpy as np
import pytest
import torch

from embodichain.gen_sim.action_engine.runtime import grasp_collision_cache
from embodichain.gen_sim.action_engine.runtime.grasp_collision_cache import (
    GraspCollisionCacheError,
    ensure_vhacd_grasp_collision_cache,
    grasp_collision_cache_path,
)


def _tetrahedron() -> tuple[torch.Tensor, torch.Tensor]:
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
            [0, 3, 2],
            [1, 2, 3],
        ],
        dtype=torch.int64,
    )
    return vertices, triangles


def _plane_equations() -> list[tuple[np.ndarray, np.ndarray]]:
    return [
        (
            np.asarray(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            ),
            np.asarray([-1.0, -1.0, -1.0], dtype=np.float32),
        ),
        (
            np.asarray([[1.0, 1.0, 1.0]], dtype=np.float32),
            np.asarray([-1.0], dtype=np.float32),
        ),
    ]


def _install_fake_decomposer(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[tuple[int, ...], tuple[int, ...], int]]:
    calls: list[tuple[tuple[int, ...], tuple[int, ...], int]] = []

    def fake_decompose(
        vertices: np.ndarray,
        triangles: np.ndarray,
        max_decomposition_hulls: int,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        calls.append(
            (
                tuple(vertices.shape),
                tuple(triangles.shape),
                max_decomposition_hulls,
            )
        )
        return _plane_equations()

    monkeypatch.setattr(
        grasp_collision_cache,
        "_compute_vhacd_plane_equations",
        fake_decompose,
    )
    return calls


def test_cache_key_and_payload_match_main_checker_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    vertices, triangles = _tetrahedron()
    _install_fake_decomposer(monkeypatch)

    result = ensure_vhacd_grasp_collision_cache(
        mesh_vertices=vertices,
        mesh_triangles=triangles,
        max_decomposition_hulls=16,
        cache_dir=tmp_path,
    )

    expected_hash = hashlib.md5(
        vertices.numpy().tobytes() + triangles.numpy().tobytes()
    ).hexdigest()
    assert result.cache_path == tmp_path / f"{expected_hash}_16.pkl"
    with result.cache_path.open("rb") as cache_file:
        payload = pickle.load(cache_file)
    assert set(payload) == {"plane_equations", "plane_equation_counts"}
    assert payload["plane_equations"].shape == (2, 3, 4)
    assert payload["plane_equations"].dtype == torch.float32
    assert payload["plane_equation_counts"].tolist() == [3, 1]
    assert payload["plane_equation_counts"].dtype == torch.int32


def test_main_checker_loads_prepared_cache_without_running_coacd(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import embodichain.lab.sim
    from embodichain.toolkits.graspkit.pg_grasp import collision_checker

    vertices, triangles = _tetrahedron()
    _install_fake_decomposer(monkeypatch)
    result = ensure_vhacd_grasp_collision_cache(
        mesh_vertices=vertices,
        mesh_triangles=triangles,
        max_decomposition_hulls=16,
        cache_dir=tmp_path,
    )

    def fail_coacd(*args: object, **kwargs: object) -> None:
        raise AssertionError("The prepared V-HACD cache must bypass CoACD.")

    monkeypatch.setattr(embodichain.lab.sim, "CONVEX_DECOMP_DIR", tmp_path)
    monkeypatch.setattr(collision_checker, "convex_decomposition_coacd", fail_coacd)
    checker = collision_checker.ConvexCollisionChecker(
        vertices,
        triangles,
        max_decomposition_hulls=16,
    )

    assert checker.cache_path == result.cache_path.as_posix()
    assert checker.plane_equations["plane_equation_counts"].tolist() == [3, 1]


def test_matching_vhacd_metadata_returns_cache_hit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    vertices, triangles = _tetrahedron()
    calls = _install_fake_decomposer(monkeypatch)
    kwargs = {
        "mesh_vertices": vertices,
        "mesh_triangles": triangles,
        "max_decomposition_hulls": 16,
        "cache_dir": tmp_path,
    }

    first = ensure_vhacd_grasp_collision_cache(**kwargs)
    second = ensure_vhacd_grasp_collision_cache(**kwargs)

    assert first.status == "generated"
    assert second.status == "hit"
    assert len(calls) == 1


def test_non_vhacd_metadata_forces_cache_replacement(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    vertices, triangles = _tetrahedron()
    calls = _install_fake_decomposer(monkeypatch)
    kwargs = {
        "mesh_vertices": vertices,
        "mesh_triangles": triangles,
        "max_decomposition_hulls": 16,
        "cache_dir": tmp_path,
    }
    first = ensure_vhacd_grasp_collision_cache(**kwargs)
    metadata = json.loads(first.metadata_path.read_text(encoding="utf-8"))
    metadata["backend"] = "coacd"
    first.metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    replaced = ensure_vhacd_grasp_collision_cache(**kwargs)

    assert replaced.status == "replaced"
    assert len(calls) == 2
    repaired = json.loads(replaced.metadata_path.read_text(encoding="utf-8"))
    assert repaired["backend"] == "vhacd"


def test_modified_cache_fails_checksum_and_is_rebuilt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    vertices, triangles = _tetrahedron()
    calls = _install_fake_decomposer(monkeypatch)
    kwargs = {
        "mesh_vertices": vertices,
        "mesh_triangles": triangles,
        "max_decomposition_hulls": 16,
        "cache_dir": tmp_path,
    }
    first = ensure_vhacd_grasp_collision_cache(**kwargs)
    first.cache_path.write_bytes(b"not a valid collision cache")

    replaced = ensure_vhacd_grasp_collision_cache(**kwargs)

    assert replaced.status == "replaced"
    assert len(calls) == 2
    with replaced.cache_path.open("rb") as cache_file:
        assert "plane_equations" in pickle.load(cache_file)


def test_cache_and_metadata_are_published_by_atomic_replace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    vertices, triangles = _tetrahedron()
    _install_fake_decomposer(monkeypatch)
    replacements: list[tuple[Path, Path]] = []
    real_replace: Callable[[os.PathLike[str], os.PathLike[str]], None] = os.replace

    def recording_replace(
        source: os.PathLike[str],
        destination: os.PathLike[str],
    ) -> None:
        replacements.append((Path(source), Path(destination)))
        real_replace(source, destination)

    monkeypatch.setattr(grasp_collision_cache.os, "replace", recording_replace)

    result = ensure_vhacd_grasp_collision_cache(
        mesh_vertices=vertices,
        mesh_triangles=triangles,
        max_decomposition_hulls=16,
        cache_dir=tmp_path,
    )

    assert [destination for _, destination in replacements] == [
        result.cache_path,
        result.metadata_path,
    ]
    assert all(
        source.parent == destination.parent for source, destination in replacements
    )
    assert all(not source.exists() for source, _ in replacements)


@pytest.mark.parametrize(
    ("vertices", "triangles", "message"),
    [
        (
            torch.empty((0, 3), dtype=torch.float32),
            torch.tensor([[0, 1, 2]], dtype=torch.int64),
            "mesh_vertices",
        ),
        (
            torch.zeros((3, 3), dtype=torch.float32),
            torch.tensor([[0, 1]], dtype=torch.int64),
            "mesh_triangles",
        ),
        (
            torch.tensor([[0.0, 0.0, 0.0], [1.0, float("nan"), 0.0], [0.0, 1.0, 0.0]]),
            torch.tensor([[0, 1, 2]], dtype=torch.int64),
            "finite",
        ),
        (
            torch.zeros((3, 3), dtype=torch.float32),
            torch.tensor([[0, 1, 3]], dtype=torch.int64),
            "indices",
        ),
    ],
)
def test_invalid_mesh_is_rejected_before_decomposition(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    vertices: torch.Tensor,
    triangles: torch.Tensor,
    message: str,
) -> None:
    calls = _install_fake_decomposer(monkeypatch)

    with pytest.raises(ValueError, match=message):
        ensure_vhacd_grasp_collision_cache(
            mesh_vertices=vertices,
            mesh_triangles=triangles,
            max_decomposition_hulls=16,
            cache_dir=tmp_path,
        )

    assert calls == []


@pytest.mark.parametrize("max_decomposition_hulls", [True, 0, -1, 1.5])
def test_invalid_hull_limit_is_rejected(
    tmp_path: Path,
    max_decomposition_hulls: object,
) -> None:
    vertices, triangles = _tetrahedron()

    with pytest.raises((TypeError, ValueError), match="max_decomposition_hulls"):
        ensure_vhacd_grasp_collision_cache(
            mesh_vertices=vertices,
            mesh_triangles=triangles,
            max_decomposition_hulls=max_decomposition_hulls,  # type: ignore[arg-type]
            cache_dir=tmp_path,
        )


def test_symlinked_cache_path_is_refused(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    vertices, triangles = _tetrahedron()
    _install_fake_decomposer(monkeypatch)
    cache_path = grasp_collision_cache_path(
        vertices,
        triangles,
        16,
        cache_dir=tmp_path,
    )
    victim = tmp_path / "victim.pkl"
    victim.write_bytes(b"do not overwrite")
    cache_path.symlink_to(victim)

    with pytest.raises(GraspCollisionCacheError, match="symlink"):
        ensure_vhacd_grasp_collision_cache(
            mesh_vertices=vertices,
            mesh_triangles=triangles,
            max_decomposition_hulls=16,
            cache_dir=tmp_path,
        )

    assert victim.read_bytes() == b"do not overwrite"
