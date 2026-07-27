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
import copy
from pathlib import Path
from typing import Any
import re

from embodichain.gen_sim.action_agent_pipeline.generation.config_types import (
    _SceneObject,
)
from embodichain.gen_sim.action_agent_pipeline.generation.naming import (
    _object_text,
)

__all__ = [
    "_arm_side_for_position",
    "_collect_scene_objects",
    "_infer_project_name",
    "is_prompt2scene_gym_export",
    "iter_mesh_object_configs",
    "iter_scene_object_configs",
    "_pick_table",
    "_position_side_axis_value",
    "_resolve_gym_config_path",
]

_PROJECT_NAME_RE = re.compile(r"^[0-9]+_gym_project$")
_GYM_CONFIG_FILENAMES = frozenset({"gym_config.json", "gym_config_merged.json"})
_GYM_CONFIG_PREFERENCE = ("gym_config_merged.json", "gym_config.json")
_ROBOT_VIEW_SIDE_AXIS_INDEX = 1


def is_prompt2scene_gym_export(path: Path) -> bool:
    """Return whether a path belongs to a prompt2scene gym export.

    Detection intentionally lives below the CLI orchestration layer so direct
    config generation does not import scene-generation stages, pipeline
    history, or optional service clients.
    """
    candidates = [path.parent] if path.is_file() else [path, path / "gym_export"]
    return any(
        (candidate / "scene_state" / "result.json").is_file()
        for candidate in candidates
    )


def _resolve_gym_config_path(input_path: Path) -> Path:
    if input_path.is_file():
        if input_path.name not in _GYM_CONFIG_FILENAMES:
            expected = ", ".join(sorted(_GYM_CONFIG_FILENAMES))
            raise ValueError(f"Expected one of {expected}, got: {input_path}")
        return input_path

    direct = _preferred_gym_config_in_dir(input_path)
    if direct is not None:
        return direct

    formatted_scene_dirs = sorted(
        {
            path.parent
            for filename in _GYM_CONFIG_FILENAMES
            for path in input_path.glob(f"formatted_tabletop_scene/*/{filename}")
        }
    )
    formatted_matches = [
        path
        for scene_dir in formatted_scene_dirs
        if (path := _preferred_gym_config_in_dir(scene_dir)) is not None
    ]
    if len(formatted_matches) == 1:
        return formatted_matches[0]
    if len(formatted_matches) > 1:
        matches = ", ".join(path.as_posix() for path in formatted_matches)
        raise ValueError(f"Multiple formatted gym config files found: {matches}")

    recursive_scene_dirs = sorted(
        {
            path.parent
            for filename in _GYM_CONFIG_FILENAMES
            for path in input_path.rglob(filename)
        }
    )
    recursive_matches = [
        path
        for scene_dir in recursive_scene_dirs
        if (path := _preferred_gym_config_in_dir(scene_dir)) is not None
    ]
    if len(recursive_matches) == 1:
        return recursive_matches[0]
    if not recursive_matches:
        expected = " or ".join(_GYM_CONFIG_PREFERENCE)
        raise FileNotFoundError(f"{expected} not found under: {input_path}")
    matches = ", ".join(path.as_posix() for path in recursive_matches)
    raise ValueError(f"Multiple gym config files found: {matches}")


def _preferred_gym_config_in_dir(scene_dir: Path) -> Path | None:
    for filename in _GYM_CONFIG_PREFERENCE:
        path = scene_dir / filename
        if path.is_file():
            return path
    return None


def _infer_project_name(input_path: Path, scene_dir: Path) -> str:
    for part in input_path.parts:
        if _PROJECT_NAME_RE.match(part):
            return part
    for part in scene_dir.parts:
        if _PROJECT_NAME_RE.match(part):
            return part
    return scene_dir.name


def iter_scene_object_configs(
    gym_config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Return every ``background``/``rigid_object`` dict in a gym config.

    Each section may be a single dict or a list; both are normalized to a flat
    list of dict configs. Non-dict entries are skipped. Returned dicts are the
    live references in ``gym_config`` (no copy), matching the prior inline
    behavior callers relied on for in-place mutation.
    """
    objects: list[dict[str, Any]] = []
    for section in ("background", "rigid_object"):
        value = gym_config.get(section, [])
        if isinstance(value, Mapping):
            value = [value]
        if not isinstance(value, list):
            continue
        objects.extend(obj for obj in value if isinstance(obj, dict))
    return objects


def iter_mesh_object_configs(
    gym_config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Return scene-object dicts whose ``shape.shape_type`` is ``"Mesh"``.

    Mesh-bearing objects are the ones that carry a GLB path and therefore need
    geometry baking, convex-decomposition caching, or mesh-based pose
    derivation. Non-mesh objects (primitives) are filtered out.
    """
    objects: list[dict[str, Any]] = []
    for obj in iter_scene_object_configs(gym_config):
        shape = obj.get("shape", {})
        if isinstance(shape, Mapping) and shape.get("shape_type") == "Mesh":
            objects.append(obj)
    return objects


def _collect_scene_objects(scene_config: Mapping[str, Any]) -> list[_SceneObject]:
    scene_objects = []
    for source_role in ("background", "rigid_object"):
        for obj_config in scene_config.get(source_role, []) or []:
            source_uid = str(obj_config.get("uid", "")).strip()
            if not source_uid:
                raise ValueError(f"Scene object without uid in {source_role}.")
            scene_objects.append(
                _SceneObject(
                    source_uid=source_uid,
                    source_role=source_role,
                    config=copy.deepcopy(dict(obj_config)),
                )
            )

    if not scene_objects:
        raise ValueError("No background or rigid_object entries found in gym config.")
    return scene_objects


def _pick_table(background_objects: list[_SceneObject]) -> _SceneObject:
    for obj in background_objects:
        text = _object_text(obj)
        if "table" in text:
            return obj
    return background_objects[0]


def _position_side_axis_value(position: list[float]) -> float:
    return -float(position[_ROBOT_VIEW_SIDE_AXIS_INDEX])


def _arm_side_for_position(position: list[float]) -> str:
    return "left" if _position_side_axis_value(position) < 0.0 else "right"


def _vector3(value: Any) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"Expected a 3-vector, got {value!r}.")
    return [float(item) for item in value]
