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

"""Stable facade for Action Agent atomic-action runtime APIs.

Execution, target resolution, geometry, IK, and trajectory adaptation live in
focused modules. This facade keeps the historical import path stable for graph
execution and external scripts.
"""

from __future__ import annotations

from typing import Any

from embodichain.gen_sim.action_agent_pipeline.runtime.action_execution import (
    execute_atomic_action,
)
from embodichain.gen_sim.action_agent_pipeline.runtime.atomic_action_spec import (
    AtomicActionSpec,
    normalize_atomic_action_spec,
)
from embodichain.gen_sim.action_agent_pipeline.runtime.grasp_collision_cache import (
    GraspCollisionCachePreparationError as VhacdCachePreparationError,
    ensure_vhacd_grasp_collision_cache,
)
from embodichain.gen_sim.action_agent_pipeline.runtime.parallel_execution import (
    build_parallel_action_stream,
    execute_parallel_atomic_actions,
    init_parallel_world_states,
    step_env_with_actions,
)
from embodichain.utils.logger import log_info

__all__ = [
    "AtomicActionSpec",
    "build_parallel_action_stream",
    "execute_atomic_action",
    "execute_parallel_atomic_actions",
    "init_parallel_world_states",
    "normalize_atomic_action_spec",
    "step_env_with_actions",
]


def _prepare_grasp_collision_cache(
    *,
    obj_name: str,
    mesh_vertices: Any,
    mesh_triangles: Any,
    max_decomposition_hulls: int,
    convex_decomposition_method: str,
    **_compat_kwargs: Any,
) -> None:
    """Preserve the historical monkeypatch boundary for cache preparation.

    The local wrapper is deliberate: existing tests and downstream tools patch
    ``atom_actions.ensure_vhacd_grasp_collision_cache`` before invoking this
    private compatibility hook.
    """
    if convex_decomposition_method != "vhacd":
        raise ValueError("convex_decomposition_method must be 'vhacd'")
    try:
        result = ensure_vhacd_grasp_collision_cache(
            mesh_vertices=mesh_vertices,
            mesh_triangles=mesh_triangles,
            max_decomposition_hulls=max_decomposition_hulls,
        )
    except VhacdCachePreparationError as exc:
        raise VhacdCachePreparationError(
            f"Failed to prepare V-HACD grasp collision cache for target={obj_name}: {exc}"
        ) from exc
    if result.get("status") != "hit":
        log_info(
            "Prepared Main-compatible V-HACD grasp collision cache: "
            f"target={obj_name}, cache={result.get('grasp_cache_path')}.",
            color="green",
        )
