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

from embodichain.gen_sim.action_agent_pipeline.generation.scene_objects import (
    iter_mesh_object_configs,
)

__all__ = [
    "coacd_cache_path_for_mesh",
    "dexsim_coacd_cache_key_for_mesh",
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

    mesh_cache_key = dexsim_coacd_cache_key_for_mesh(mesh_path, mesh_count=mesh_count)
    return Path(cache_dir).expanduser().resolve() / (
        f"{mesh_cache_key}_{int(max_convex_hull_num)}.obj"
    )


def dexsim_coacd_cache_key_for_mesh(
    mesh_path: str | Path,
    *,
    mesh_count: int = 1,
) -> str:
    """Return the cache key used by DexSim ``load_actor_with_coacd``."""

    resolved_mesh_path = Path(mesh_path).expanduser().resolve(strict=False)
    mesh_key_data = f"{resolved_mesh_path}|mesh_count={int(mesh_count)}"
    return hashlib.sha256(mesh_key_data.encode("utf-8")).hexdigest()


def _iter_mesh_object_configs(
    gym_config: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    return iter_mesh_object_configs(gym_config)


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "setup.py").is_file() and (parent / "embodichain").is_dir():
            return parent
    return Path.cwd().resolve()
