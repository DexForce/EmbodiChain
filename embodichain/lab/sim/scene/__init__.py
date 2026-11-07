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

"""Shared scene construction, layout and runtime services.

Submodules are loaded on first access. Scene services own no simulator, task
lifecycle, semantic registry, or model-generation clients. Expansion contracts
are available separately in :mod:`embodichain.lab.sim.scene.expansion`.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "SupportSurfaceConfig",
    "GroupSupportClampConfig",
    "AABBLayoutConfig",
    "SceneFragment",
    "compose",
    "declare_fragment",
    "SceneRecipe",
    "GeneratedScene",
    "generate",
    "check_layout",
    "SupportRegion",
    "detect_support_region",
    "clamp_group",
    "separate_aabbs",
    "align_group_height",
    "LayoutCfg",
    "PlanarRelation",
    "LayoutProblem",
    "LayoutSolution",
    "solve_layout",
    "SceneView",
    "EntityState",
    "SceneState",
    "capture_state",
    "restore_state",
    "restore_render_state",
    "SceneChangeSet",
    "SceneChangeResult",
    "apply_changes",
    "apply_changes_batch",
    "SceneCheck",
    "SceneCheckReport",
    "SettlingCfg",
    "validate_scene",
    "compare_states",
    "settle_scene",
    "RobotSceneWorkspace",
    "SceneWorkspaceCandidate",
    "convex_input_clearance",
    "SceneMeshComponent",
    "SceneMeshSnapshot",
    "capture_scene_meshes",
    "iter_scene_components",
    "CollisionObjectSnapshot",
    "read_articulation_link_boxes",
    "snapshot_collision_sources",
]

_EXPORT_MODULES = {
    "SupportSurfaceConfig": "geometry",
    "GroupSupportClampConfig": "placement",
    "AABBLayoutConfig": "placement",
    "SceneFragment": "composition",
    "compose": "composition",
    "declare_fragment": "composition",
    "SceneRecipe": "procedural",
    "GeneratedScene": "procedural",
    "generate": "procedural",
    "check_layout": "procedural",
    "SupportRegion": "geometry",
    "detect_support_region": "geometry",
    "clamp_group": "placement",
    "separate_aabbs": "placement",
    "align_group_height": "placement",
    "LayoutCfg": "layout",
    "PlanarRelation": "layout",
    "LayoutProblem": "layout",
    "LayoutSolution": "layout",
    "solve_layout": "layout",
    "SceneView": "query",
    "EntityState": "state",
    "SceneState": "state",
    "capture_state": "state",
    "restore_state": "state",
    "restore_render_state": "state",
    "SceneChangeSet": "operations",
    "SceneChangeResult": "operations",
    "apply_changes": "operations",
    "apply_changes_batch": "operations",
    "SceneCheck": "validation",
    "SceneCheckReport": "validation",
    "SettlingCfg": "validation",
    "validate_scene": "validation",
    "compare_states": "validation",
    "settle_scene": "validation",
    "RobotSceneWorkspace": "workspace",
    "SceneWorkspaceCandidate": "workspace",
    "convex_input_clearance": "convex_clearance",
    "SceneMeshComponent": "mesh_snapshot",
    "SceneMeshSnapshot": "mesh_snapshot",
    "capture_scene_meshes": "mesh_snapshot",
    "iter_scene_components": "mesh_snapshot",
    "CollisionObjectSnapshot": "collision_sources",
    "read_articulation_link_boxes": "collision_sources",
    "snapshot_collision_sources": "collision_sources",
}


def __getattr__(name: str) -> Any:
    module = _EXPORT_MODULES.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{module}"), name)
    globals()[name] = value
    return value
