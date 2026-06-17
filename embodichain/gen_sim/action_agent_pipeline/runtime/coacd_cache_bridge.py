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
from pathlib import Path
from typing import Any

import numpy as np
import torch

from embodichain.gen_sim.action_agent_pipeline.generation.coacd_cache import (
    coacd_cache_path_for_mesh,
)

__all__ = [
    "ensure_grasp_collision_cache_from_env_coacd",
    "grasp_collision_cache_path",
]


_DEFAULT_CONVEX_DECOMP_DIR = (
    Path.home() / ".cache" / "embodichain_cache" / "convex_decomposition"
)


def grasp_collision_cache_path(
    mesh_vertices: torch.Tensor | np.ndarray,
    mesh_triangles: torch.Tensor | np.ndarray,
    max_decomposition_hulls: int,
    *,
    cache_dir: str | Path | None = None,
) -> Path:
    """Return the grasp collision checker cache path for a scaled mesh."""

    vertices = _as_numpy(mesh_vertices)
    triangles = _as_numpy(mesh_triangles)
    mesh_hash = hashlib.md5(vertices.tobytes() + triangles.tobytes()).hexdigest()
    return _resolve_cache_dir(cache_dir) / (
        f"{mesh_hash}_{int(max_decomposition_hulls)}.pkl"
    )


def ensure_grasp_collision_cache_from_env_coacd(
    *,
    mesh_vertices: torch.Tensor | np.ndarray,
    mesh_triangles: torch.Tensor | np.ndarray,
    source_mesh_path: str | Path | None,
    max_decomposition_hulls: int,
    body_scale: Any = None,
    cache_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Prepare grasp collision cache from the environment CoACD OBJ cache.

    The environment and grasp collision paths use different cache formats. This
    bridge avoids running CoACD again during grasp annotation when the
    environment-side convex OBJ cache is already available.
    """

    grasp_cache_path = grasp_collision_cache_path(
        mesh_vertices,
        mesh_triangles,
        max_decomposition_hulls,
        cache_dir=cache_dir,
    )
    if grasp_cache_path.is_file():
        return {
            "status": "hit",
            "grasp_cache_path": grasp_cache_path.as_posix(),
        }

    if source_mesh_path is None:
        return {
            "status": "missing_source_mesh",
            "grasp_cache_path": grasp_cache_path.as_posix(),
        }

    env_cache_path = coacd_cache_path_for_mesh(
        source_mesh_path,
        max_decomposition_hulls,
        _resolve_cache_dir(cache_dir),
    )
    if not env_cache_path.is_file():
        return {
            "status": "missing_env_cache",
            "env_cache_path": env_cache_path.as_posix(),
            "grasp_cache_path": grasp_cache_path.as_posix(),
        }

    try:
        plane_equations = _plane_equations_from_env_cache(env_cache_path, body_scale)
        _write_grasp_collision_cache(grasp_cache_path, plane_equations)
    except Exception as exc:
        return {
            "status": "skipped",
            "reason": str(exc),
            "env_cache_path": env_cache_path.as_posix(),
            "grasp_cache_path": grasp_cache_path.as_posix(),
        }

    return {
        "status": "generated",
        "env_cache_path": env_cache_path.as_posix(),
        "grasp_cache_path": grasp_cache_path.as_posix(),
    }


def _plane_equations_from_env_cache(
    env_cache_path: Path,
    body_scale: Any,
) -> list[tuple[np.ndarray, np.ndarray]]:
    from dexsim.kit.meshproc.convex_cache import load_obj_as_convex_parts

    from embodichain.toolkits.graspkit.pg_grasp.collision_checker import (
        extract_plane_equations,
    )

    convex_parts = load_obj_as_convex_parts(env_cache_path.as_posix())
    if not convex_parts:
        raise ValueError(f"No convex parts found in {env_cache_path}.")

    scale = _body_scale(body_scale)
    if not np.allclose(scale, np.ones(3, dtype=np.float32)):
        convex_parts = [
            (vertices.astype(np.float32, copy=False) * scale, faces)
            for vertices, faces in convex_parts
        ]

    plane_equations = extract_plane_equations(convex_parts)
    if not plane_equations:
        raise ValueError(f"No plane equations extracted from {env_cache_path}.")
    return plane_equations


def _write_grasp_collision_cache(
    cache_path: Path,
    plane_equations_np: list[tuple[np.ndarray, np.ndarray]],
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    n_convex = len(plane_equations_np)
    n_max_equation = max(normals.shape[0] for normals, _ in plane_equations_np)
    plane_equations = torch.zeros(
        size=(n_convex, n_max_equation, 4),
        dtype=torch.float32,
        device="cpu",
    )
    plane_equation_counts = torch.zeros(n_convex, dtype=torch.int32, device="cpu")
    for index, (normals, offsets) in enumerate(plane_equations_np):
        n_equation = normals.shape[0]
        plane_equations[index, :n_equation, :3] = torch.as_tensor(
            normals,
            dtype=torch.float32,
        )
        plane_equations[index, :n_equation, 3] = torch.as_tensor(
            offsets,
            dtype=torch.float32,
        )
        plane_equation_counts[index] = n_equation

    with cache_path.open("wb") as cache_file:
        pickle.dump(
            {
                "plane_equations": plane_equations,
                "plane_equation_counts": plane_equation_counts,
            },
            cache_file,
        )


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


def _body_scale(body_scale: Any) -> np.ndarray:
    if body_scale is None:
        return np.ones(3, dtype=np.float32)
    if isinstance(body_scale, torch.Tensor):
        body_scale = body_scale.detach().cpu().numpy()
    scale = np.asarray(body_scale, dtype=np.float32).reshape(-1)
    if scale.size == 1:
        scale = np.repeat(scale, 3)
    if scale.size != 3 or not np.all(np.isfinite(scale)):
        raise ValueError(f"Invalid body scale: {body_scale!r}.")
    return scale.reshape(1, 3)
