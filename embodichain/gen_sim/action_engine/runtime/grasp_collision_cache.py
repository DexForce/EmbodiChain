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

"""Prepare checksummed V-HACD caches for the shared grasp collision checker.

The sidecar identifies the backend without changing Main's cache key or pickle
payload, so an unlabelled CoACD cache is never silently reused as V-HACD.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import operator
import os
from pathlib import Path
import pickle
import stat
import tempfile
from typing import Literal

import numpy as np
import torch

__all__ = [
    "GraspCollisionCacheError",
    "GraspCollisionCacheResult",
    "ensure_vhacd_grasp_collision_cache",
    "grasp_collision_cache_path",
]

_CACHE_SCHEMA_VERSION = 1
_METADATA_SUFFIX = ".action_engine.json"
_DEFAULT_CACHE_DIR = (
    Path.home() / ".cache" / "embodichain_cache" / "convex_decomposition"
)

CacheStatus = Literal["hit", "generated", "replaced"]


class GraspCollisionCacheError(RuntimeError):
    """Raised when a safe, Main-compatible V-HACD cache cannot be prepared."""


@dataclass(frozen=True)
class GraspCollisionCacheResult:
    """Describe the prepared cache files and whether decomposition ran."""

    status: CacheStatus
    cache_path: Path
    metadata_path: Path


def grasp_collision_cache_path(
    mesh_vertices: torch.Tensor | np.ndarray,
    mesh_triangles: torch.Tensor | np.ndarray,
    max_decomposition_hulls: int,
    *,
    cache_dir: str | Path | None = None,
) -> Path:
    """Return Main's exact ``<mesh-md5>_<hulls>.pkl`` cache path."""
    vertices, triangles = _validate_mesh(mesh_vertices, mesh_triangles)
    hull_limit = _validate_hull_limit(max_decomposition_hulls)
    mesh_hash = hashlib.md5(vertices.tobytes() + triangles.tobytes()).hexdigest()
    return _resolve_cache_dir(cache_dir) / f"{mesh_hash}_{hull_limit}.pkl"


def ensure_vhacd_grasp_collision_cache(
    *,
    mesh_vertices: torch.Tensor | np.ndarray,
    mesh_triangles: torch.Tensor | np.ndarray,
    max_decomposition_hulls: int,
    cache_dir: str | Path | None = None,
) -> GraspCollisionCacheResult:
    """Create or validate a V-HACD cache and its checksummed backend sidecar."""
    vertices, triangles = _validate_mesh(mesh_vertices, mesh_triangles)
    hull_limit = _validate_hull_limit(max_decomposition_hulls)
    mesh_hash = hashlib.md5(vertices.tobytes() + triangles.tobytes()).hexdigest()
    cache_path = _resolve_cache_dir(cache_dir) / f"{mesh_hash}_{hull_limit}.pkl"
    metadata_path = cache_path.with_name(f"{cache_path.name}{_METADATA_SUFFIX}")
    expected_metadata: dict[str, object] = {
        "schema_version": _CACHE_SCHEMA_VERSION,
        "backend": "vhacd",
        "mesh_hash": mesh_hash,
        "max_decomposition_hulls": hull_limit,
    }

    _prepare_private_directory(cache_path.parent)
    _refuse_symlink(cache_path)
    _refuse_symlink(metadata_path)
    if _cache_matches_metadata(cache_path, metadata_path, expected_metadata):
        return GraspCollisionCacheResult("hit", cache_path, metadata_path)

    exists = cache_path.exists() or metadata_path.exists()
    status: CacheStatus = "replaced" if exists else "generated"
    try:
        plane_equations = _compute_vhacd_plane_equations(
            vertices,
            triangles,
            hull_limit,
        )
        cache_bytes = _serialize_checker_payload(plane_equations)
        metadata = {
            **expected_metadata,
            "cache_sha256": hashlib.sha256(cache_bytes).hexdigest(),
        }

        # Publish the complete pickle before its sidecar. A crash between the
        # two replaces leaves a cache miss on retry, never a partial pickle.
        _write_bytes_atomic(cache_path, cache_bytes)
        metadata_bytes = (json.dumps(metadata, sort_keys=True) + "\n").encode()
        _write_bytes_atomic(metadata_path, metadata_bytes)
    except GraspCollisionCacheError:
        raise
    except Exception as exc:
        raise GraspCollisionCacheError(
            f"Failed to prepare V-HACD grasp collision cache {cache_path}: {exc}"
        ) from exc

    return GraspCollisionCacheResult(status, cache_path, metadata_path)


def _compute_vhacd_plane_equations(
    vertices: np.ndarray,
    triangles: np.ndarray,
    max_decomposition_hulls: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Run DexSim V-HACD and convert its hulls to checker plane equations."""
    import open3d as o3d
    from dexsim.kit.meshproc import convex_decomposition_vhacd

    from embodichain.toolkits.graspkit.pg_grasp.collision_checker import (
        extract_plane_equations,
    )

    mesh = o3d.t.geometry.TriangleMesh()
    mesh.vertex.positions = o3d.core.Tensor(vertices.astype(np.float32, copy=False))
    mesh.triangle.indices = o3d.core.Tensor(triangles.astype(np.int32, copy=False))
    is_success, hull_meshes = convex_decomposition_vhacd(
        mesh,
        max_convex_hull_num=max_decomposition_hulls,
    )
    if not is_success or not hull_meshes:
        raise GraspCollisionCacheError(
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
        raise GraspCollisionCacheError(
            "V-HACD hulls produced no grasp collision plane equations."
        )
    return plane_equations


def _serialize_checker_payload(
    plane_equations: list[tuple[np.ndarray, np.ndarray]],
) -> bytes:
    """Pack plane equations in the exact tensor dictionary Main unpickles."""
    if not plane_equations:
        raise ValueError("V-HACD must produce at least one convex hull.")

    normalized: list[tuple[np.ndarray, np.ndarray]] = []
    for normals_value, offsets_value in plane_equations:
        normals = np.asarray(normals_value, dtype=np.float32)
        offsets = np.asarray(offsets_value, dtype=np.float32)
        if normals.ndim != 2 or normals.shape[1:] != (3,) or not len(normals):
            raise ValueError("Each V-HACD hull must have normals shaped [K, 3].")
        if offsets.shape != (len(normals),):
            raise ValueError("Each hull needs one offset per plane normal.")
        if not np.isfinite(normals).all() or not np.isfinite(offsets).all():
            raise ValueError("V-HACD plane equations must contain finite values.")
        normalized.append((normals, offsets))

    max_plane_count = max(normals.shape[0] for normals, _ in normalized)
    equations = torch.zeros((len(normalized), max_plane_count, 4))
    counts = torch.zeros(len(normalized), dtype=torch.int32)
    for index, (normals, offsets) in enumerate(normalized):
        plane_count = normals.shape[0]
        equations[index, :plane_count, :3] = torch.from_numpy(normals)
        equations[index, :plane_count, 3] = torch.from_numpy(offsets)
        counts[index] = plane_count

    stream = io.BytesIO()
    payload = {"plane_equations": equations, "plane_equation_counts": counts}
    pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    return stream.getvalue()


def _cache_matches_metadata(
    cache_path: Path,
    metadata_path: Path,
    expected_metadata: dict[str, object],
) -> bool:
    if not cache_path.is_file() or not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        checksum = metadata.get("cache_sha256")
        expected_checksum = hashlib.sha256(cache_path.read_bytes()).hexdigest()
        return (
            isinstance(metadata, dict)
            and all(
                metadata.get(key) == value for key, value in expected_metadata.items()
            )
            and isinstance(checksum, str)
            and checksum == expected_checksum
        )
    except (AttributeError, OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False


def _validate_mesh(
    mesh_vertices: torch.Tensor | np.ndarray,
    mesh_triangles: torch.Tensor | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(mesh_vertices, torch.Tensor):
        mesh_vertices = mesh_vertices.detach().cpu().numpy()
    if isinstance(mesh_triangles, torch.Tensor):
        mesh_triangles = mesh_triangles.detach().cpu().numpy()
    if not isinstance(mesh_vertices, np.ndarray):
        raise TypeError("mesh_vertices must be a torch.Tensor or numpy.ndarray.")
    if not isinstance(mesh_triangles, np.ndarray):
        raise TypeError("mesh_triangles must be a torch.Tensor or numpy.ndarray.")
    vertices = np.ascontiguousarray(mesh_vertices)
    triangles = np.ascontiguousarray(mesh_triangles)
    if vertices.ndim != 2 or vertices.shape[1:] != (3,) or len(vertices) == 0:
        raise ValueError("mesh_vertices must have non-empty shape [N, 3].")
    if triangles.ndim != 2 or triangles.shape[1:] != (3,) or len(triangles) == 0:
        raise ValueError("mesh_triangles must have non-empty shape [M, 3].")
    if not np.issubdtype(vertices.dtype, np.number):
        raise TypeError("mesh_vertices must contain numeric values.")
    if not np.isfinite(vertices).all():
        raise ValueError("mesh_vertices must contain only finite values.")
    if not np.issubdtype(triangles.dtype, np.integer):
        raise TypeError("mesh_triangles must contain integer indices.")
    if triangles.min() < 0 or triangles.max() >= len(vertices):
        raise ValueError("mesh_triangles contains out-of-range vertex indices.")
    return vertices, triangles


def _validate_hull_limit(value: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("max_decomposition_hulls must be an integer.")
    try:
        hull_limit = operator.index(value)
    except TypeError as exc:
        raise TypeError("max_decomposition_hulls must be an integer.") from exc
    if hull_limit <= 0:
        raise ValueError("max_decomposition_hulls must be positive.")
    return hull_limit


def _resolve_cache_dir(cache_dir: str | Path | None) -> Path:
    if cache_dir is not None:
        return Path(cache_dir).expanduser().resolve()
    try:
        from embodichain.lab.sim import CONVEX_DECOMP_DIR
    except Exception:
        return _DEFAULT_CACHE_DIR
    return Path(CONVEX_DECOMP_DIR).expanduser().resolve()


def _prepare_private_directory(path: Path) -> None:
    try:
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
        path.chmod(0o700)
    except OSError as exc:
        raise GraspCollisionCacheError(
            f"Cannot secure grasp collision cache directory: {path}"
        ) from exc
    if path.stat().st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        raise GraspCollisionCacheError(f"Refusing writable cache directory: {path}")


def _refuse_symlink(path: Path) -> None:
    if path.is_symlink():
        raise GraspCollisionCacheError(
            f"Refusing symlinked grasp collision cache path: {path}"
        )


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    """Publish one complete file with a same-directory atomic replacement."""
    _refuse_symlink(path)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        os.fchmod(file_descriptor, 0o600)
        with os.fdopen(file_descriptor, "wb") as output:
            file_descriptor = -1
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        _refuse_symlink(path)
        os.replace(temporary_path, path)
        path.chmod(0o600)
    finally:
        if file_descriptor >= 0:
            os.close(file_descriptor)
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
