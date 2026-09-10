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
from pathlib import Path

import numpy as np
import pytest
import torch

from embodichain.toolkits.graspkit.pg_grasp import collision_checker as module
from embodichain.toolkits.graspkit.pg_grasp.collision_checker import (
    ConvexCollisionChecker,
)


def _tetrahedron() -> tuple[np.ndarray, np.ndarray]:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    faces = np.array(
        [
            [0, 2, 1],
            [0, 1, 3],
            [0, 3, 2],
            [1, 2, 3],
        ],
        dtype=np.int32,
    )
    return vertices, faces


def test_plane_equations_use_vhacd(monkeypatch: pytest.MonkeyPatch) -> None:
    vertices, faces = _tetrahedron()
    calls: list[int] = []

    def fake_vhacd(mesh, *, max_convex_hull_num: int):
        calls.append(max_convex_hull_num)
        return True, (mesh,)

    monkeypatch.setattr(module, "convex_decomposition_vhacd", fake_vhacd)

    plane_equations = ConvexCollisionChecker._compute_plane_equations(
        vertices,
        faces,
        max_decomposition_hulls=16,
    )

    assert calls == [16]
    assert len(plane_equations) == 1


def test_vhacd_failure_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    vertices, faces = _tetrahedron()

    monkeypatch.setattr(
        module,
        "convex_decomposition_vhacd",
        lambda *_args, **_kwargs: (False, ()),
    )

    with pytest.raises(RuntimeError, match="V-HACD convex decomposition failed"):
        ConvexCollisionChecker._compute_plane_equations(
            vertices,
            faces,
            max_decomposition_hulls=16,
        )


def test_vhacd_cache_does_not_reuse_legacy_backend(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    vertices, faces = _tetrahedron()
    mesh_hash = hashlib.md5(vertices.tobytes() + faces.tobytes()).hexdigest()
    legacy_path = tmp_path / f"{mesh_hash}_16.pkl"
    legacy_path.write_bytes(b"legacy CoACD cache")
    calls: list[int] = []

    def fake_plane_equations(
        _vertices: np.ndarray,
        _faces: np.ndarray,
        max_decomposition_hulls: int,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        calls.append(max_decomposition_hulls)
        return [
            (
                np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
                np.array([0.0], dtype=np.float32),
            )
        ]

    monkeypatch.setattr(module, "CONVEX_DECOMPOSITION_CACHE_DIR", tmp_path)
    monkeypatch.setattr(
        ConvexCollisionChecker,
        "_compute_plane_equations",
        staticmethod(fake_plane_equations),
    )

    checker = ConvexCollisionChecker(
        torch.from_numpy(vertices),
        torch.from_numpy(faces),
        max_decomposition_hulls=16,
    )

    assert calls == [16]
    assert checker.cache_path == str(tmp_path / f"{mesh_hash}_16_vhacd_v1.pkl")
