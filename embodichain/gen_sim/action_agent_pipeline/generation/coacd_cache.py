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

from collections.abc import Mapping
from pathlib import Path
from typing import Any
import hashlib

from embodichain.utils.logger import log_info

__all__ = [
    "coacd_cache_path_for_mesh",
    "dexsim_coacd_cache_key_for_mesh",
    "prewarm_coacd_cache_for_gym_config",
]

_DEFAULT_CONVEX_DECOMP_DIR = (
    Path.home() / ".cache" / "embodichain_cache" / "convex_decomposition"
)


def coacd_cache_path_for_mesh(
    mesh_path: str | Path,
    max_convex_hull_num: int,
    cache_dir: str | Path | None = None,
    *,
    mesh_count: int = 1,
) -> Path:
    """Return the DexSim environment-side CoACD cache path for a mesh."""

    if cache_dir is None:
        cache_dir = _DEFAULT_CONVEX_DECOMP_DIR

    mesh_md5_key = dexsim_coacd_cache_key_for_mesh(mesh_path, mesh_count=mesh_count)
    return Path(cache_dir).expanduser().resolve() / (
        f"{mesh_md5_key}_{int(max_convex_hull_num)}.obj"
    )


def dexsim_coacd_cache_key_for_mesh(
    mesh_path: str | Path,
    *,
    mesh_count: int = 1,
) -> str:
    """Return the cache key used by DexSim ``load_actor_with_coacd``."""

    resolved_mesh_path = Path(mesh_path).expanduser().resolve(strict=False)
    mesh_key_data = f"{resolved_mesh_path}|mesh_count={int(mesh_count)}"
    return hashlib.md5(mesh_key_data.encode("utf-8")).hexdigest()


def prewarm_coacd_cache_for_gym_config(
    gym_config: Mapping[str, Any],
    *,
    cache_dir: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Precompute DexSim environment-side CoACD cache files for mesh objects."""

    entries = []
    for obj in _iter_mesh_object_configs(gym_config):
        max_convex_hull_num = int(obj.get("max_convex_hull_num", 1))
        if max_convex_hull_num <= 1:
            continue
        entries.append((obj, max_convex_hull_num))
    if not entries:
        return []

    if cache_dir is None:
        cache_dir = _DEFAULT_CONVEX_DECOMP_DIR

    cache_dir = Path(cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    repo_root = Path(repo_root).expanduser().resolve() if repo_root else _repo_root()

    reports: list[dict[str, Any]] = []
    seen_cache_paths: set[Path] = set()
    for obj, max_convex_hull_num in entries:
        uid = str(obj.get("uid", ""))
        raw_fpath = str(obj.get("shape", {}).get("fpath", ""))
        mesh_path = _resolve_mesh_path(raw_fpath, repo_root)
        cache_path = coacd_cache_path_for_mesh(
            mesh_path,
            max_convex_hull_num,
            cache_dir,
        )
        report = {
            "uid": uid,
            "mesh_path": mesh_path.as_posix(),
            "mesh_count": 1,
            "max_convex_hull_num": max_convex_hull_num,
            "cache_path": cache_path.as_posix(),
        }
        if cache_path in seen_cache_paths:
            report["status"] = "duplicate"
        elif cache_path.is_file():
            report["status"] = "hit"
        else:
            try:
                _generate_coacd_cache(mesh_path, cache_path, max_convex_hull_num)
            except Exception as exc:
                report["status"] = "skipped"
                report["reason"] = str(exc)
            else:
                report["status"] = "generated"
        seen_cache_paths.add(cache_path)
        reports.append(report)
    return reports


def _iter_mesh_object_configs(
    gym_config: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    objects = []
    for section in ("background", "rigid_object"):
        value = gym_config.get(section, [])
        if isinstance(value, Mapping):
            value = [value]
        if not isinstance(value, list):
            continue
        for obj in value:
            if not isinstance(obj, Mapping):
                continue
            shape = obj.get("shape", {})
            if isinstance(shape, Mapping) and shape.get("shape_type") == "Mesh":
                objects.append(obj)
    return objects


def _resolve_mesh_path(raw_fpath: str, repo_root: Path) -> Path:
    path = Path(raw_fpath).expanduser()
    if path.is_absolute():
        candidate = path.resolve()
    else:
        candidate = (repo_root / path).resolve()
        if not candidate.is_file():
            cwd_candidate = (Path.cwd() / path).resolve()
            if cwd_candidate.is_file():
                candidate = cwd_candidate
    if not candidate.is_file():
        raise FileNotFoundError(f"Mesh path for CoACD prewarm not found: {raw_fpath}")
    return candidate


def _generate_coacd_cache(
    mesh_path: Path,
    cache_path: Path,
    max_convex_hull_num: int,
) -> None:
    import open3d as o3d
    from dexsim.kit.meshproc import convex_decomposition_coacd
    from dexsim.kit.meshproc.utility import mesh_list_to_file

    log_info(
        "Prewarming environment CoACD cache: "
        f"mesh={mesh_path.as_posix()}, hulls={max_convex_hull_num}"
    )
    in_mesh = o3d.t.io.read_triangle_mesh(mesh_path.as_posix())
    _, out_mesh_list = convex_decomposition_coacd(
        in_mesh,
        max_convex_hull_num=int(max_convex_hull_num),
    )
    mesh_list_to_file(cache_path.as_posix(), out_mesh_list)


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "setup.py").is_file() and (parent / "embodichain").is_dir():
            return parent
    return Path.cwd().resolve()
