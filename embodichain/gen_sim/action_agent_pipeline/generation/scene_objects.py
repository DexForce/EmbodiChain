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
    _BasketTaskRoles,
    _SceneObject,
)
from embodichain.gen_sim.action_agent_pipeline.generation.naming import (
    _base_name,
    _container_runtime_uid,
    _display_noun,
    _is_container_like,
    _object_text,
    _target_noun,
)

__all__ = [
    "_arm_side_for_position",
    "_collect_scene_objects",
    "_infer_basket_task_roles",
    "_infer_project_name",
    "_pick_container",
    "_pick_left_right_targets",
    "_pick_table",
    "_position_side_axis_value",
    "_resolve_gym_config_path",
    "_side_axis_value",
]

_PROJECT_NAME_RE = re.compile(r"^[0-9]+_gym_project$")
_GYM_CONFIG_FILENAMES = frozenset({"gym_config.json", "gym_config_merged.json"})
_GYM_CONFIG_PREFERENCE = ("gym_config_merged.json", "gym_config.json")
_ROBOT_VIEW_SIDE_AXIS_INDEX = 1


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


def _infer_basket_task_roles(scene_objects: list[_SceneObject]) -> _BasketTaskRoles:
    background_objects = [
        obj for obj in scene_objects if obj.source_role == "background"
    ]
    rigid_objects = [obj for obj in scene_objects if obj.source_role == "rigid_object"]
    if not background_objects:
        raise ValueError("Basket generation requires a table/background object.")
    if len(rigid_objects) < 3:
        raise ValueError(
            "Basket generation requires at least two target objects and one "
            "basket-like container."
        )

    table = _pick_table(background_objects)
    container = _pick_container(rigid_objects)
    target_candidates = [
        obj for obj in rigid_objects if obj.source_uid != container.source_uid
    ]
    if len(target_candidates) < 2:
        raise ValueError("Expected at least two non-container target objects.")

    left_target, right_target = _pick_left_right_targets(target_candidates)
    target_noun = _target_noun(left_target, right_target)
    container_noun = _display_noun(_base_name(container))
    return _BasketTaskRoles(
        table_source_uid=table.source_uid,
        container_source_uid=container.source_uid,
        left_target_source_uid=left_target.source_uid,
        right_target_source_uid=right_target.source_uid,
        container_runtime_uid=_container_runtime_uid(container),
        left_target_runtime_uid=f"left_{target_noun}",
        right_target_runtime_uid=f"right_{target_noun}",
        target_noun=target_noun,
        left_target_noun=target_noun,
        right_target_noun=target_noun,
        container_noun=container_noun,
    )


def _pick_table(background_objects: list[_SceneObject]) -> _SceneObject:
    for obj in background_objects:
        text = _object_text(obj)
        if "table" in text:
            return obj
    return background_objects[0]


def _pick_container(rigid_objects: list[_SceneObject]) -> _SceneObject:
    candidates = [obj for obj in rigid_objects if _is_container_like(obj)]
    if not candidates:
        names = ", ".join(obj.source_uid for obj in rigid_objects)
        raise ValueError(f"No basket-like container object found among: {names}")

    def score(obj: _SceneObject) -> tuple[int, float]:
        text = _object_text(obj)
        keyword_score = 0 if "basket" in text else 1
        pos = _vector3(obj.config.get("init_pos", [0.0, 0.0, 0.0]))
        center_distance = abs(pos[0]) + abs(pos[1])
        return keyword_score, center_distance

    return sorted(candidates, key=score)[0]


def _pick_left_right_targets(
    target_candidates: list[_SceneObject],
) -> tuple[_SceneObject, _SceneObject]:
    if len(target_candidates) == 2:
        picked = target_candidates
    else:
        grouped: dict[str, list[_SceneObject]] = {}
        for obj in target_candidates:
            grouped.setdefault(_base_name(obj), []).append(obj)
        repeated_groups = [group for group in grouped.values() if len(group) >= 2]
        if repeated_groups:
            picked = sorted(
                repeated_groups,
                key=_target_group_sort_key,
            )[0]
            if len(picked) > 2:
                picked = sorted(
                    picked,
                    key=lambda obj: abs(_side_axis_value(obj)),
                    reverse=True,
                )[:2]
        else:
            picked = sorted(
                target_candidates,
                key=lambda obj: abs(_side_axis_value(obj)),
                reverse=True,
            )[:2]
    left, right = sorted(picked, key=_side_axis_value)
    return left, right


def _target_group_sort_key(group: list[_SceneObject]) -> tuple[float, int]:
    side_values = [_side_axis_value(obj) for obj in group]
    side_spread = max(side_values) - min(side_values)
    return -side_spread, -len(group)


def _side_axis_value(obj: _SceneObject) -> float:
    return _position_side_axis_value(
        _vector3(obj.config.get("init_pos", [0.0, 0.0, 0.0]))
    )


def _position_side_axis_value(position: list[float]) -> float:
    return -float(position[_ROBOT_VIEW_SIDE_AXIS_INDEX])


def _arm_side_for_position(position: list[float]) -> str:
    return "left" if _position_side_axis_value(position) < 0.0 else "right"


def _vector3(value: Any) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"Expected a 3-vector, got {value!r}.")
    return [float(item) for item in value]
