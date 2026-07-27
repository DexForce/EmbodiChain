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
import pickle
import stat
from pathlib import Path

import numpy as np
import torch

__all__ = [
    "GraspCollisionCachePreparationError",
    "ensure_vhacd_grasp_collision_cache",
    "main_grasp_collision_cache_path",
]

_CACHE_SCHEMA_VERSION = 1
_CACHE_METADATA_SUFFIX = ".action_agent.json"
_DEFAULT_CONVEX_DECOMP_DIR = (
    Path.home() / ".cache" / "embodichain_cache" / "convex_decomposition"
)


class GraspCollisionCachePreparationError(RuntimeError):
    """Raised when a Main-compatible grasp collision cache cannot be prepared."""


def main_grasp_collision_cache_path(
    mesh_vertices: torch.Tensor | np.ndarray,
    mesh_triangles: torch.Tensor | np.ndarray,
    max_decomposition_hulls: int,
    *,
    cache_dir: str | Path | None = None,
) -> Path:
    """Return the cache path used by Main's ``ConvexCollisionChecker``.

    Args:
        mesh_vertices: Object mesh vertices passed to the grasp collision checker.
        mesh_triangles: Object mesh triangle indices passed to the checker.
        max_decomposition_hulls: Maximum number of convex hulls.
        cache_dir: Optional cache directory override.

    Returns:
        The exact unsuffixed pickle path expected by Main.

    Raises:
        ValueError: If ``max_decomposition_hulls`` is not positive.
    """
    if int(max_decomposition_hulls) <= 0:
        raise ValueError("max_decomposition_hulls must be positive.")
    mesh_hash = _main_mesh_hash(mesh_vertices, mesh_triangles)
    return _resolve_cache_dir(cache_dir) / (
        f"{mesh_hash}_{int(max_decomposition_hulls)}.pkl"
    )


def ensure_vhacd_grasp_collision_cache(
    *,
    mesh_vertices: torch.Tensor | np.ndarray,
    mesh_triangles: torch.Tensor | np.ndarray,
    max_decomposition_hulls: int,
    cache_dir: str | Path | None = None,
) -> dict[str, str]:
    """Prepare a V-HACD cache that Main's grasp checker loads directly.

    Main's cache key does not encode the decomposition backend. A sidecar records
    the backend and cache checksum so an existing unlabelled legacy cache is
    never mistaken for a V-HACD result.

    Args:
        mesh_vertices: Object mesh vertices used by the grasp checker.
        mesh_triangles: Object mesh triangle indices used by the checker.
        max_decomposition_hulls: Maximum number of V-HACD convex hulls.
        cache_dir: Optional cache directory override.

    Returns:
        A report containing ``status``, ``grasp_cache_path``, and
        ``metadata_path``.

    Raises:
        GraspCollisionCachePreparationError: If V-HACD or cache writing fails.
        ValueError: If ``max_decomposition_hulls`` is not positive.
    """
    cache_path = main_grasp_collision_cache_path(
        mesh_vertices,
        mesh_triangles,
        max_decomposition_hulls,
        cache_dir=cache_dir,
    )
    metadata_path = _metadata_path(cache_path)
    mesh_hash = _main_mesh_hash(mesh_vertices, mesh_triangles)
    expected_metadata = {
        "schema_version": _CACHE_SCHEMA_VERSION,
        "algorithm": "vhacd",
        "mesh_hash": mesh_hash,
        "max_decomposition_hulls": int(max_decomposition_hulls),
    }

    _prepare_private_cache_directory(cache_path.parent)
    _refuse_symlink(cache_path)
    _refuse_symlink(metadata_path)
    if _cache_matches_metadata(cache_path, metadata_path, expected_metadata):
        return {
            "status": "hit",
            "grasp_cache_path": cache_path.as_posix(),
            "metadata_path": metadata_path.as_posix(),
        }

    status = "replaced" if cache_path.exists() else "generated"
    try:
        plane_equations = _compute_vhacd_plane_equations(
            mesh_vertices,
            mesh_triangles,
            int(max_decomposition_hulls),
        )
        _write_grasp_collision_cache(cache_path, plane_equations)
        metadata = {
            **expected_metadata,
            "cache_sha256": _file_sha256(cache_path),
        }
        _write_json_atomic(metadata_path, metadata)
    except GraspCollisionCachePreparationError:
        raise
    except (
        ImportError,
        ModuleNotFoundError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        raise GraspCollisionCachePreparationError(
            f"Failed to prepare V-HACD grasp collision cache {cache_path}: {exc}"
        ) from exc

    return {
        "status": status,
        "grasp_cache_path": cache_path.as_posix(),
        "metadata_path": metadata_path.as_posix(),
    }


def _compute_vhacd_plane_equations(
    mesh_vertices: torch.Tensor | np.ndarray,
    mesh_triangles: torch.Tensor | np.ndarray,
    max_decomposition_hulls: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    import open3d as o3d
    from dexsim.kit.meshproc import convex_decomposition_vhacd

    from embodichain.toolkits.graspkit.pg_grasp.collision_checker import (
        extract_plane_equations,
    )

    vertices = _as_numpy(mesh_vertices).astype(np.float32, copy=False)
    triangles = _as_numpy(mesh_triangles).astype(np.int32, copy=False)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.shape[0] == 0:
        raise ValueError("mesh_vertices must have non-empty shape [N, 3].")
    if triangles.ndim != 2 or triangles.shape[1] != 3 or triangles.shape[0] == 0:
        raise ValueError("mesh_triangles must have non-empty shape [M, 3].")

    mesh = o3d.t.geometry.TriangleMesh()
    mesh.vertex.positions = o3d.core.Tensor(vertices, dtype=o3d.core.float32)
    mesh.triangle.indices = o3d.core.Tensor(triangles, dtype=o3d.core.int32)
    is_success, hull_meshes = convex_decomposition_vhacd(
        mesh,
        max_convex_hull_num=max_decomposition_hulls,
    )
    if not is_success or not hull_meshes:
        raise GraspCollisionCachePreparationError(
            "V-HACD returned no convex hulls for the grasp collision mesh."
        )

    convex_parts = [
        (
            np.asarray(hull.vertex.positions.numpy()),
            np.asarray(hull.triangle.indices.numpy()),
        )
        for hull in hull_meshes
    ]
    plane_equations = extract_plane_equations(convex_parts)
    if not plane_equations:
        raise GraspCollisionCachePreparationError(
            "V-HACD hulls produced no grasp collision plane equations."
        )
    return plane_equations


def _write_grasp_collision_cache(
    cache_path: Path,
    plane_equations_np: list[tuple[np.ndarray, np.ndarray]],
) -> None:
    n_convex = len(plane_equations_np)
    n_max_equation = max(normals.shape[0] for normals, _ in plane_equations_np)
    plane_equations = torch.zeros(
        (n_convex, n_max_equation, 4), dtype=torch.float32, device="cpu"
    )
    plane_equation_counts = torch.zeros(n_convex, dtype=torch.int32, device="cpu")
    for index, (normals, offsets) in enumerate(plane_equations_np):
        n_equation = normals.shape[0]
        plane_equations[index, :n_equation, :3] = torch.as_tensor(
            normals, dtype=torch.float32
        )
        plane_equations[index, :n_equation, 3] = torch.as_tensor(
            offsets, dtype=torch.float32
        )
        plane_equation_counts[index] = n_equation

    _write_pickle_atomic(
        cache_path,
        {
            "plane_equations": plane_equations,
            "plane_equation_counts": plane_equation_counts,
        },
    )


def _cache_matches_metadata(
    cache_path: Path,
    metadata_path: Path,
    expected_metadata: dict[str, object],
) -> bool:
    if not cache_path.is_file() or not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    if any(metadata.get(key) != value for key, value in expected_metadata.items()):
        return False
    checksum = metadata.get("cache_sha256")
    try:
        return isinstance(checksum, str) and checksum == _file_sha256(cache_path)
    except OSError:
        return False


def _write_pickle_atomic(path: Path, value: object) -> None:
    temp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        with temp_path.open("wb") as cache_file:
            _chmod_if_possible(temp_path, 0o600)
            pickle.dump(value, cache_file)
        os.replace(temp_path, path)
        _chmod_if_possible(path, 0o600)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _write_json_atomic(path: Path, value: dict[str, object]) -> None:
    temp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        with temp_path.open("w", encoding="utf-8") as metadata_file:
            _chmod_if_possible(temp_path, 0o600)
            json.dump(value, metadata_file, indent=2, sort_keys=True)
            metadata_file.write("\n")
        os.replace(temp_path, path)
        _chmod_if_possible(path, 0o600)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _prepare_private_cache_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    _chmod_if_possible(path, 0o700)
    if path.stat().st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        raise GraspCollisionCachePreparationError(
            f"Refusing group/world-writable grasp cache directory: {path}"
        )


def _refuse_symlink(path: Path) -> None:
    if path.is_symlink():
        raise GraspCollisionCachePreparationError(
            f"Refusing symlinked grasp collision cache path: {path}"
        )


def _metadata_path(cache_path: Path) -> Path:
    return cache_path.with_name(f"{cache_path.name}{_CACHE_METADATA_SUFFIX}")


def _main_mesh_hash(
    mesh_vertices: torch.Tensor | np.ndarray,
    mesh_triangles: torch.Tensor | np.ndarray,
) -> str:
    vertices = _as_numpy(mesh_vertices)
    triangles = _as_numpy(mesh_triangles)
    return hashlib.md5(vertices.tobytes() + triangles.tobytes()).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as cache_file:
        for chunk in iter(lambda: cache_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_cache_dir(cache_dir: str | Path | None) -> Path:
    if cache_dir is not None:
        return Path(cache_dir).expanduser().resolve()
    try:
        from embodichain.lab.sim import CONVEX_DECOMP_DIR
    except Exception:
        return _DEFAULT_CONVEX_DECOMP_DIR
    return Path(CONVEX_DECOMP_DIR).expanduser().resolve()


def _as_numpy(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.ascontiguousarray(value)


def _chmod_if_possible(path: Path, mode: int) -> None:
    try:
        path.chmod(mode)
    except OSError:
        pass
