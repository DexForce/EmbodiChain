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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import copy
import json
import re

__all__ = [
    "GeneratedUR5BasketConfigPaths",
    "TargetReplacementSpec",
    "generate_ur5_basket_config_from_project",
]

_DIGIT_SUFFIX_RE = re.compile(r"_[0-9]+$")
_INVALID_UID_CHARS_RE = re.compile(r"[^0-9a-zA-Z_]+")
_PROJECT_NAME_RE = re.compile(r"^[0-9]+_gym_project$")
_GYM_CONFIG_FILENAMES = frozenset({"gym_config.json", "gym_config_merged.json"})
_GYM_CONFIG_PREFERENCE = ("gym_config_merged.json", "gym_config.json")

_CONTAINER_KEYWORDS = (
    "basket",
    "container",
    "bowl",
    "box",
    "bin",
    "tray",
    "crate",
)

_RELATIVE_RELATIONS = {
    "inside",
    "on",
    "left_of",
    "right_of",
    "front_of",
    "behind",
}

_RELATION_ALIASES = {
    "in": "inside",
    "into": "inside",
    "inside": "inside",
    "放入": "inside",
    "放进": "inside",
    "里面": "inside",
    "on": "on",
    "onto": "on",
    "on_top": "on",
    "on_top_of": "on",
    "above": "on",
    "top": "on",
    "上": "on",
    "上方": "on",
    "上面": "on",
    "叠放": "on",
    "left": "left_of",
    "left_of": "left_of",
    "to_the_left_of": "left_of",
    "左": "left_of",
    "左边": "left_of",
    "右": "right_of",
    "右边": "right_of",
    "right": "right_of",
    "right_of": "right_of",
    "to_the_right_of": "right_of",
    "front": "front_of",
    "front_of": "front_of",
    "in_front_of": "front_of",
    "前": "front_of",
    "前方": "front_of",
    "前面": "front_of",
    "back": "behind",
    "behind": "behind",
    "back_of": "behind",
    "后": "behind",
    "后方": "behind",
    "后面": "behind",
}

_SIDE_RELATION_DISTANCE = 0.16
_SIDE_RELEASE_Z_OFFSET = 0.12
_STAGING_Z_DELTA = 0.10
_ON_RELEASE_Z_OFFSET = 0.05
_DUAL_UR5_LEGACY_INIT_Z = 0.5
_DUAL_UR5_HIGH_TABLETOP_THRESHOLD = 1.0
_DUAL_UR5_TABLETOP_Z_OFFSET = 0.05
_DUAL_UR5_SIDE_AXIS_INDEX = 1
_BACKGROUND_MAX_CONVEX_HULL_NUM = 1
_TARGET_MAX_CONVEX_HULL_NUM = 4
_CONTAINER_MAX_CONVEX_HULL_NUM = 8
_EXTRA_RIGID_MAX_CONVEX_HULL_NUM = 1

_BACKGROUND_ATTRS = {
    "mass": 10.0,
    "static_friction": 0.95,
    "dynamic_friction": 0.9,
    "restitution": 0.01,
}

_RIGID_OBJECT_ATTRS = {
    "mass": 0.01,
    "contact_offset": 0.003,
    "rest_offset": 0.001,
    "restitution": 0.01,
    "max_depenetration_velocity": 10.0,
    "min_position_iters": 32,
    "min_velocity_iters": 8,
}


@dataclass(frozen=True)
class GeneratedUR5BasketConfigPaths:
    """Paths written by the UR5 basket config generator."""

    output_dir: Path
    gym_config: Path
    agent_config: Path
    task_prompt: Path
    basic_background: Path
    atom_actions: Path
    summary: dict[str, Any]


@dataclass(frozen=True)
class TargetReplacementSpec:
    """Prompt-to-geometry replacement for one source target object."""

    source_uid: str
    prompt: str
    output_dir_name: str


@dataclass(frozen=True)
class _SceneObject:
    source_uid: str
    source_role: str
    config: dict[str, Any]


@dataclass(frozen=True)
class _BasketTaskRoles:
    table_source_uid: str
    container_source_uid: str
    left_target_source_uid: str
    right_target_source_uid: str
    container_runtime_uid: str
    left_target_runtime_uid: str
    right_target_runtime_uid: str
    target_noun: str
    left_target_noun: str
    right_target_noun: str
    container_noun: str


@dataclass(frozen=True)
class _ResolvedTargetReplacement:
    source_uid: str
    prompt: str
    output_dir_name: str
    mesh_path: Path
    runtime_noun: str


@dataclass(frozen=True)
class _RelativePlacementSpec:
    table_source_uid: str
    moved_source_uid: str
    reference_source_uid: str
    moved_runtime_uid: str
    reference_runtime_uid: str
    relation: str
    active_side: str
    task_description: str
    task_prompt_summary: str
    basic_background_notes: str
    action_sketch: list[str]
    release_offset: list[float]
    high_offset: list[float]


def generate_ur5_basket_config_from_project(
    gym_project: str | Path,
    output_dir: str | Path,
    *,
    task_name: str = "UR5BreadBasket",
    task_description: str | None = None,
    use_llm_roles: bool = False,
    llm_model: str | None = None,
    target_body_scale: float | list[float] | tuple[float, float, float] = 0.7,
    target_replacements: Sequence[TargetReplacementSpec] | None = None,
    sync_replacement_names: bool = False,
    overwrite: bool = False,
    max_episodes: int = 1,
    max_episode_steps: int = 1000,
) -> GeneratedUR5BasketConfigPaths:
    """Generate Dual-UR5 basket placement configs from an exported gym project.

    This first-stage generator intentionally keeps the UR5BreadBasket task
    structure fixed: the left arm grasps the left target object, the right arm
    grasps the right target object, and both objects are placed into one
    basket-like container.

    Args:
        gym_project: Project root, formatted scene folder, ``gym_config.json``,
            or ``gym_config_merged.json``.
        output_dir: Destination config directory.
        task_name: Name passed to ``run_agent``.
        task_description: Optional natural-language relative-placement task.
            When provided, the generator asks the shared LLM for a constrained
            config-level task spec and generates prompts from that spec.
        use_llm_roles: If true, use an LLM only to refine object role mapping.
        llm_model: Optional model override for role refinement.
        target_body_scale: Uniform or xyz scale applied to generated target
            objects. Basket-like containers keep their source ``body_scale``.
        target_replacements: Optional prompt-generated GLB replacements for
            selected default basket target objects. Each replacement writes to
            ``<gym_project>/mesh_assets/<output_dir_name>`` and only affects the
            generated config, not the original source mesh file.
        sync_replacement_names: If true, update runtime target UIDs and prompts
            from the replacement prompts. If false, only mesh paths are replaced.
        overwrite: If false, fail when generated files already exist.
        max_episodes: Value written to ``fast_gym_config.json``.
        max_episode_steps: Value written to ``fast_gym_config.json``.

    Returns:
        Paths of generated config files.
    """

    input_path = Path(gym_project).expanduser().resolve()
    gym_config_path = _resolve_gym_config_path(input_path)
    scene_dir = gym_config_path.parent
    source_config = _read_json(gym_config_path)
    project_name = _infer_project_name(input_path, scene_dir)
    replacement_specs = _normalize_target_replacements(target_replacements)

    scene_objects = _collect_scene_objects(source_config)
    if task_description:
        if replacement_specs:
            raise ValueError(
                "target_replacements are only supported by the default basket "
                "template. Do not combine them with task_description."
            )
        spec = _build_relative_placement_spec_with_llm(
            scene_objects=scene_objects,
            project_name=project_name,
            task_description=task_description,
            model=llm_model,
        )
        bundle = _build_relative_placement_bundle(
            scene_dir=scene_dir,
            source_config=source_config,
            spec=spec,
            project_name=project_name,
            task_name=task_name,
            target_body_scale=target_body_scale,
            max_episodes=max_episodes,
            max_episode_steps=max_episode_steps,
        )
        _validate_relative_bundle(bundle, spec)
        return _write_config_bundle(
            output_dir=Path(output_dir).expanduser().resolve(),
            bundle=bundle,
            overwrite=overwrite,
        )

    roles = _infer_basket_task_roles(scene_objects)
    if use_llm_roles:
        roles = _refine_roles_with_llm(
            roles=roles,
            scene_objects=scene_objects,
            project_name=project_name,
            model=llm_model,
        )

    _validate_target_replacement_sources(roles, replacement_specs)
    resolved_replacements = _run_target_replacements(
        scene_dir=scene_dir,
        replacement_specs=replacement_specs,
    )
    if sync_replacement_names:
        roles = _apply_replacement_names(
            roles,
            resolved_replacements,
        )

    bundle = _build_ur5_basket_bundle(
        scene_dir=scene_dir,
        source_config=source_config,
        roles=roles,
        project_name=project_name,
        task_name=task_name,
        target_body_scale=target_body_scale,
        target_replacements=resolved_replacements,
        max_episodes=max_episodes,
        max_episode_steps=max_episode_steps,
    )
    _validate_bundle(bundle, roles)
    return _write_config_bundle(
        output_dir=Path(output_dir).expanduser().resolve(),
        bundle=bundle,
        overwrite=overwrite,
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
        raise ValueError("UR5 basket generation requires a table/background object.")
    if len(rigid_objects) < 3:
        raise ValueError(
            "UR5 basket generation requires at least two target objects and one "
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
    candidates = [
        obj
        for obj in rigid_objects
        if any(keyword in _object_text(obj) for keyword in _CONTAINER_KEYWORDS)
    ]
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
    return float(position[_DUAL_UR5_SIDE_AXIS_INDEX])


def _arm_side_for_position(position: list[float]) -> str:
    return "left" if _position_side_axis_value(position) < 0.0 else "right"


def _target_noun(left_target: _SceneObject, right_target: _SceneObject) -> str:
    left_base = _base_name(left_target)
    right_base = _base_name(right_target)
    if left_base == right_base:
        return _target_runtime_suffix(left_base)
    return "target_object"


def _object_text(obj: _SceneObject) -> str:
    shape = obj.config.get("shape", {}) or {}
    return f"{obj.source_uid} {shape.get('fpath', '')}".lower()


def _base_name(obj: _SceneObject) -> str:
    base = _DIGIT_SUFFIX_RE.sub("", obj.source_uid)
    if base == obj.source_uid:
        fpath = str(obj.config.get("shape", {}).get("fpath", ""))
        path = Path(fpath)
        if len(path.parts) >= 2:
            base = path.parts[-2]
    return _normalize_runtime_uid(base)


def _target_runtime_suffix(base: str) -> str:
    if base == "bread":
        return "bread_roll"
    return base


def _container_runtime_uid(container: _SceneObject) -> str:
    base = _base_name(container)
    if "basket" in base:
        return "wicker_basket"
    return f"target_{base}"


def _display_noun(uid: str) -> str:
    return uid.replace("_", " ")


def _plural(noun: str) -> str:
    if noun.endswith("s"):
        return noun
    if noun.endswith(("ch", "sh", "x")):
        return f"{noun}es"
    return f"{noun}s"


def _left_target_text(roles: _BasketTaskRoles) -> str:
    return _display_noun(roles.left_target_noun)


def _right_target_text(roles: _BasketTaskRoles) -> str:
    return _display_noun(roles.right_target_noun)


def _target_pair_text(roles: _BasketTaskRoles) -> str:
    left_text = _left_target_text(roles)
    right_text = _right_target_text(roles)
    if left_text == right_text:
        return f"two {left_text} objects"
    return f"the left {left_text} and right {right_text}"


def _target_plural_text(roles: _BasketTaskRoles) -> str:
    left_text = _left_target_text(roles)
    right_text = _right_target_text(roles)
    if left_text == right_text:
        return _plural(left_text)
    return "target objects"


def _generic_target_text(roles: _BasketTaskRoles) -> str:
    left_text = _left_target_text(roles)
    right_text = _right_target_text(roles)
    if left_text == right_text:
        return left_text
    return "target object"


def _target_task_description_text(roles: _BasketTaskRoles) -> str:
    left_text = _left_target_text(roles)
    right_text = _right_target_text(roles)
    if left_text == right_text:
        return _plural(left_text)
    return f"{left_text}-and-{right_text}"


def _normalize_runtime_uid(value: str) -> str:
    uid = _INVALID_UID_CHARS_RE.sub("_", value.strip()).strip("_").lower()
    if not uid:
        raise ValueError(f"Invalid runtime uid: {value!r}")
    return uid


def _normalize_target_replacements(
    target_replacements: Sequence[TargetReplacementSpec] | None,
) -> tuple[TargetReplacementSpec, ...]:
    if not target_replacements:
        return ()

    normalized = []
    seen_source_uids = set()
    seen_output_dirs = set()
    for replacement in target_replacements:
        if not isinstance(replacement, TargetReplacementSpec):
            raise TypeError(
                "target_replacements must contain TargetReplacementSpec values."
            )
        source_uid = str(replacement.source_uid).strip()
        prompt = str(replacement.prompt).strip()
        output_dir_name = str(replacement.output_dir_name).strip()
        if not source_uid:
            raise ValueError("target replacement source_uid must be non-empty.")
        if not prompt:
            raise ValueError("target replacement prompt must be non-empty.")
        if not output_dir_name:
            raise ValueError("target replacement output_dir_name must be non-empty.")
        output_dir_path = Path(output_dir_name)
        if (
            output_dir_path.is_absolute()
            or len(output_dir_path.parts) != 1
            or output_dir_name in {".", ".."}
        ):
            raise ValueError(
                "target replacement output_dir_name must be a single relative "
                f"directory name, got: {output_dir_name!r}"
            )
        if source_uid in seen_source_uids:
            raise ValueError(f"Duplicate target replacement source uid: {source_uid}")
        if output_dir_name in seen_output_dirs:
            raise ValueError(
                f"Duplicate target replacement output dir: {output_dir_name}"
            )
        seen_source_uids.add(source_uid)
        seen_output_dirs.add(output_dir_name)
        normalized.append(
            TargetReplacementSpec(
                source_uid=source_uid,
                prompt=prompt,
                output_dir_name=output_dir_name,
            )
        )
    return tuple(normalized)


def _validate_target_replacement_sources(
    roles: _BasketTaskRoles,
    replacement_specs: Sequence[TargetReplacementSpec],
) -> None:
    if not replacement_specs:
        return

    target_source_uids = {
        roles.left_target_source_uid,
        roles.right_target_source_uid,
    }
    unknown = [
        replacement.source_uid
        for replacement in replacement_specs
        if replacement.source_uid not in target_source_uids
    ]
    if unknown:
        raise ValueError(
            "target_replacements must reference the selected basket target "
            f"source uid(s) {sorted(target_source_uids)}, got: {unknown}"
        )


def _run_target_replacements(
    *,
    scene_dir: Path,
    replacement_specs: Sequence[TargetReplacementSpec],
) -> tuple[_ResolvedTargetReplacement, ...]:
    resolved = []
    for replacement in replacement_specs:
        runtime_noun = _replacement_runtime_noun(replacement.prompt)
        output_root = scene_dir / "mesh_assets" / replacement.output_dir_name
        result = _run_prompt2geometry_replacement(
            prompt=replacement.prompt,
            output_root=output_root,
            output_name=f"{runtime_noun}.glb",
        )
        mesh_path = _resolve_prompt2geometry_mesh_path(result, output_root)
        resolved.append(
            _ResolvedTargetReplacement(
                source_uid=replacement.source_uid,
                prompt=replacement.prompt,
                output_dir_name=replacement.output_dir_name,
                mesh_path=mesh_path,
                runtime_noun=runtime_noun,
            )
        )
    return tuple(resolved)


def _run_prompt2geometry_replacement(
    *,
    prompt: str,
    output_root: Path,
    output_name: str,
) -> dict[str, Any]:
    from embodichain.gen_sim.action_agent_pipeline.gym_project_api.prompt2geometry import (
        Prompt2GeometryRequest,
        load_prompt2geometry_config,
        run_prompt2geometry,
    )

    cfg = load_prompt2geometry_config()
    return run_prompt2geometry(
        Prompt2GeometryRequest(
            prompt=prompt,
            output_root=output_root,
            output_name=output_name,
            zimage_base_url=cfg.zimage_base_url,
            sam3_base_url=cfg.sam3_base_url,
            sam3d_base_url=cfg.sam3d_base_url,
            llm_api_key=cfg.llm_api_key,
            llm_model=cfg.llm_model,
            llm_base_url=cfg.llm_base_url,
            llm_timeout_s=cfg.llm_timeout_s,
        )
    )


def _resolve_prompt2geometry_mesh_path(
    result: Mapping[str, Any],
    output_root: Path,
) -> Path:
    raw_path = result.get("scaled_mesh_path") or result.get("mesh_path")
    if not raw_path:
        raise ValueError("prompt2geometry result did not include a GLB mesh path.")

    mesh_path = Path(str(raw_path)).expanduser()
    if not mesh_path.is_absolute():
        mesh_path = (output_root / mesh_path).resolve()
    else:
        mesh_path = mesh_path.resolve()

    if not mesh_path.is_file():
        raise FileNotFoundError(f"Generated replacement GLB not found: {mesh_path}")
    return mesh_path


def _replacement_runtime_noun(prompt: str) -> str:
    tokens = re.findall(r"[a-z0-9]+", prompt.lower())
    while tokens and tokens[0] in {"a", "an", "the"}:
        tokens.pop(0)
    stem = "_".join(tokens)
    if not stem:
        stem = "replacement_object"
    return _normalize_runtime_uid(stem)


def _apply_replacement_names(
    roles: _BasketTaskRoles,
    resolved_replacements: Sequence[_ResolvedTargetReplacement],
) -> _BasketTaskRoles:
    replacement_by_uid = {
        replacement.source_uid: replacement for replacement in resolved_replacements
    }
    left_replacement = replacement_by_uid.get(roles.left_target_source_uid)
    right_replacement = replacement_by_uid.get(roles.right_target_source_uid)
    left_target_noun = (
        left_replacement.runtime_noun
        if left_replacement is not None
        else roles.left_target_noun
    )
    right_target_noun = (
        right_replacement.runtime_noun
        if right_replacement is not None
        else roles.right_target_noun
    )
    target_noun = (
        left_target_noun if left_target_noun == right_target_noun else "target_object"
    )
    return _BasketTaskRoles(
        table_source_uid=roles.table_source_uid,
        container_source_uid=roles.container_source_uid,
        left_target_source_uid=roles.left_target_source_uid,
        right_target_source_uid=roles.right_target_source_uid,
        container_runtime_uid=roles.container_runtime_uid,
        left_target_runtime_uid=f"left_{left_target_noun}",
        right_target_runtime_uid=f"right_{right_target_noun}",
        target_noun=target_noun,
        left_target_noun=left_target_noun,
        right_target_noun=right_target_noun,
        container_noun=roles.container_noun,
    )


def _refine_roles_with_llm(
    *,
    roles: _BasketTaskRoles,
    scene_objects: list[_SceneObject],
    project_name: str,
    model: str | None,
) -> _BasketTaskRoles:
    response = _call_role_llm(
        project_name=project_name,
        scene_summary=[
            {
                "source_uid": obj.source_uid,
                "role": obj.source_role,
                "mesh": obj.config.get("shape", {}).get("fpath"),
                "init_pos": obj.config.get("init_pos"),
            }
            for obj in scene_objects
        ],
        default_roles={
            "container_object": roles.container_source_uid,
            "left_target_object": roles.left_target_source_uid,
            "right_target_object": roles.right_target_source_uid,
            "target_noun": roles.target_noun,
            "container_runtime_uid": roles.container_runtime_uid,
        },
        model=model,
    )
    source_uids = {obj.source_uid for obj in scene_objects}
    left_target = str(response.get("left_target_object", roles.left_target_source_uid))
    right_target = str(
        response.get("right_target_object", roles.right_target_source_uid)
    )
    container = str(response.get("container_object", roles.container_source_uid))
    for uid in (left_target, right_target, container):
        if uid not in source_uids:
            raise ValueError(f"LLM returned unknown source uid: {uid!r}")
    if len({left_target, right_target, container}) != 3:
        raise ValueError("LLM role mapping must use three distinct source objects.")

    target_noun = _normalize_runtime_uid(
        str(response.get("target_noun", roles.target_noun))
    )
    container_runtime_uid = _normalize_runtime_uid(
        str(response.get("container_runtime_uid", roles.container_runtime_uid))
    )
    return _BasketTaskRoles(
        table_source_uid=roles.table_source_uid,
        container_source_uid=container,
        left_target_source_uid=left_target,
        right_target_source_uid=right_target,
        container_runtime_uid=container_runtime_uid,
        left_target_runtime_uid=f"left_{target_noun}",
        right_target_runtime_uid=f"right_{target_noun}",
        target_noun=target_noun,
        left_target_noun=target_noun,
        right_target_noun=target_noun,
        container_noun=_display_noun(container_runtime_uid),
    )


def _call_role_llm(
    *,
    project_name: str,
    scene_summary: list[dict[str, Any]],
    default_roles: dict[str, Any],
    model: str | None,
) -> dict[str, Any]:
    from langchain_core.messages import HumanMessage, SystemMessage

    from embodichain.gen_sim.mllm import create_chat_openai
    from embodichain.utils.llm_json import extract_json_object

    prompt = (
        "Identify roles for a fixed Dual-UR5 basket-placement simulation task. "
        "Return only one JSON object with keys: container_object, "
        "left_target_object, right_target_object, target_noun, "
        "container_runtime_uid. Use only source_uid values from the scene. The "
        "left target starts on the negative-y side, and the right target starts "
        "on the positive-y side.\n\n"
        f"Project: {project_name}\n"
        f"Scene objects:\n{json.dumps(scene_summary, ensure_ascii=False, indent=2)}\n"
        f"Default roles:\n{json.dumps(default_roles, ensure_ascii=False, indent=2)}"
    )
    llm = create_chat_openai(temperature=0.0, model=model)
    response = llm.invoke(
        [
            SystemMessage(
                content=(
                    "You produce strict JSON role mappings for simulation config "
                    "generation. Do not include markdown."
                )
            ),
            HumanMessage(content=prompt),
        ]
    )
    content = getattr(response, "content", response)
    return extract_json_object(content)


def _build_relative_placement_spec_with_llm(
    *,
    scene_objects: list[_SceneObject],
    project_name: str,
    task_description: str,
    model: str | None,
) -> _RelativePlacementSpec:
    background_objects = [
        obj for obj in scene_objects if obj.source_role == "background"
    ]
    rigid_objects = [obj for obj in scene_objects if obj.source_role == "rigid_object"]
    if not background_objects:
        raise ValueError("Relative placement generation requires a background table.")
    if len(rigid_objects) < 2:
        raise ValueError(
            "Relative placement generation requires at least two rigid objects."
        )

    table = _pick_table(background_objects)
    response = _call_relative_task_llm(
        project_name=project_name,
        task_description=task_description,
        scene_summary=[
            {
                "source_uid": obj.source_uid,
                "role": obj.source_role,
                "object_type": _base_name(obj),
                "is_container_like": _is_container_like(obj),
                "mesh": obj.config.get("shape", {}).get("fpath"),
                "init_pos": obj.config.get("init_pos"),
            }
            for obj in scene_objects
        ],
        model=model,
    )
    return _apply_relative_task_response(
        response=response,
        table_source_uid=table.source_uid,
        rigid_objects=rigid_objects,
        task_description=task_description,
    )


def _call_relative_task_llm(
    *,
    project_name: str,
    task_description: str,
    scene_summary: list[dict[str, Any]],
    model: str | None,
) -> dict[str, Any]:
    from langchain_core.messages import HumanMessage, SystemMessage

    from embodichain.gen_sim.mllm import create_chat_openai
    from embodichain.utils.llm_json import extract_json_object

    prompt = (
        "Parse a simple Dual-UR5 tabletop relative-placement task and produce "
        "a constrained config-level JSON spec. This JSON is used to generate "
        "task_prompt.txt, basic_background.txt, atom_actions.txt, and "
        "agent_success; a second LLM will later read those prompts to generate "
        "the executable graph JSON.\n\n"
        "Return exactly one JSON object with this schema:\n"
        "{\n"
        '  "moved_object": "<source_uid from rigid_object>",\n'
        '  "reference_object": "<different source_uid from rigid_object>",\n'
        '  "goal_relation": '
        '"inside|on|left_of|right_of|front_of|behind",\n'
        '  "task_prompt_summary": "<one or two sentences for task_prompt>",\n'
        '  "basic_background_notes": "<short scene/task notes>",\n'
        '  "action_sketch": [\n'
        '    "grasp moved_object",\n'
        '    "move above the relation target pose",\n'
        '    "lower to the release pose",\n'
        '    "open gripper",\n'
        '    "retreat upward"\n'
        "  ]\n"
        "}\n\n"
        "Rules:\n"
        "- Use only source_uid values from rigid_object entries.\n"
        "- moved_object is the object to grasp and move.\n"
        "- reference_object is the object used as the spatial reference, "
        "container, or support.\n"
        "- The two objects must be different.\n"
        "- For Chinese/English left/right/front/back, use the relation enums. "
        "front_of means negative world-y, closer to the Dual-UR5 bases; "
        "behind means positive world-y.\n"
        "- If the task says to release an object above a basket/container so it "
        "falls into it, use goal_relation='inside'.\n"
        "- If the task says to stack/place one object on another non-container "
        "support, use goal_relation='on'.\n"
        "- Do not return numeric offsets, object poses, scales, success JSON, "
        "robot config, or full prompt files. The generator computes those "
        "deterministically.\n\n"
        f"Project: {project_name}\n"
        f"Task description:\n{task_description}\n"
        f"Scene objects:\n{json.dumps(scene_summary, ensure_ascii=False, indent=2)}"
    )
    llm = create_chat_openai(temperature=0.0, model=model)
    response = llm.invoke(
        [
            SystemMessage(
                content=(
                    "You produce strict JSON specs for simulation config "
                    "generation. Do not include markdown."
                )
            ),
            HumanMessage(content=prompt),
        ]
    )
    content = getattr(response, "content", response)
    return extract_json_object(content)


def _apply_relative_task_response(
    *,
    response: Mapping[str, Any],
    table_source_uid: str,
    rigid_objects: list[_SceneObject],
    task_description: str,
) -> _RelativePlacementSpec:
    moved_source_uid = _resolve_rigid_source_uid(
        response.get("moved_object"),
        rigid_objects,
        field_name="moved_object",
    )
    reference_source_uid = _resolve_rigid_source_uid(
        response.get("reference_object"),
        rigid_objects,
        field_name="reference_object",
    )
    if moved_source_uid == reference_source_uid:
        raise ValueError(
            "Relative placement requires distinct moved/reference objects."
        )

    by_uid = {obj.source_uid: obj for obj in rigid_objects}
    reference_obj = by_uid[reference_source_uid]
    relation = _normalize_relative_relation(response.get("goal_relation"))
    if relation == "on" and _is_container_like(reference_obj):
        relation = "inside"

    runtime_uids = _relative_runtime_uid_mapping(rigid_objects)
    moved_runtime_uid = runtime_uids[moved_source_uid]
    reference_runtime_uid = runtime_uids[reference_source_uid]
    if moved_runtime_uid == reference_runtime_uid:
        raise ValueError(
            f"Relative placement produced duplicate runtime uid {moved_runtime_uid!r}."
        )

    release_offset = _relative_release_offset(relation)
    high_offset = list(release_offset)
    high_offset[2] += _STAGING_Z_DELTA
    moved_position = _vector3(
        by_uid[moved_source_uid].config.get("init_pos", [0, 0, 0])
    )
    active_side = _arm_side_for_position(moved_position)
    summary = str(response.get("task_prompt_summary", "")).strip()
    if not summary:
        summary = _default_relative_task_summary(
            moved_runtime_uid,
            reference_runtime_uid,
            relation,
        )
    background_notes = str(response.get("basic_background_notes", "")).strip()
    action_sketch = _string_list(response.get("action_sketch"))
    if not action_sketch:
        action_sketch = [
            f"grasp {moved_runtime_uid}",
            f"move above the {relation} release pose relative to {reference_runtime_uid}",
            "lower to the release pose",
            "open the gripper",
            "retreat upward",
        ]

    return _RelativePlacementSpec(
        table_source_uid=table_source_uid,
        moved_source_uid=moved_source_uid,
        reference_source_uid=reference_source_uid,
        moved_runtime_uid=moved_runtime_uid,
        reference_runtime_uid=reference_runtime_uid,
        relation=relation,
        active_side=active_side,
        task_description=task_description,
        task_prompt_summary=summary,
        basic_background_notes=background_notes,
        action_sketch=action_sketch,
        release_offset=release_offset,
        high_offset=high_offset,
    )


def _resolve_rigid_source_uid(
    value: Any,
    rigid_objects: list[_SceneObject],
    *,
    field_name: str,
) -> str:
    if value is None:
        raise ValueError(f"LLM response missing required {field_name}.")
    text = str(value).strip()
    by_uid = {obj.source_uid: obj for obj in rigid_objects}
    if text in by_uid:
        return text

    normalized = _normalize_runtime_uid(text)
    matches = [
        obj.source_uid
        for obj in rigid_objects
        if _normalize_runtime_uid(obj.source_uid) == normalized
        or _base_name(obj) == normalized
        or _candidate_relative_runtime_uid(obj) == normalized
    ]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(f"LLM returned unknown {field_name}: {text!r}.")
    raise ValueError(
        f"LLM returned ambiguous {field_name}: {text!r}; candidates: {matches}."
    )


def _normalize_relative_relation(value: Any) -> str:
    relation = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    relation = _RELATION_ALIASES.get(relation, relation)
    if relation not in _RELATIVE_RELATIONS:
        raise ValueError(
            f"Unsupported relative placement relation {value!r}; expected one "
            f"of {sorted(_RELATIVE_RELATIONS)}."
        )
    return relation


def _relative_release_offset(relation: str) -> list[float]:
    relation = _normalize_relative_relation(relation)
    if relation == "inside":
        return [0.0, 0.0, _SIDE_RELEASE_Z_OFFSET]
    if relation == "on":
        return [0.0, 0.0, _ON_RELEASE_Z_OFFSET]
    if relation == "left_of":
        return [-_SIDE_RELATION_DISTANCE, 0.0, _SIDE_RELEASE_Z_OFFSET]
    if relation == "right_of":
        return [_SIDE_RELATION_DISTANCE, 0.0, _SIDE_RELEASE_Z_OFFSET]
    if relation == "front_of":
        return [0.0, -_SIDE_RELATION_DISTANCE, _SIDE_RELEASE_Z_OFFSET]
    if relation == "behind":
        return [0.0, _SIDE_RELATION_DISTANCE, _SIDE_RELEASE_Z_OFFSET]
    raise ValueError(f"Unsupported relative placement relation: {relation!r}.")


def _relative_runtime_uid_mapping(
    rigid_objects: list[_SceneObject],
) -> dict[str, str]:
    candidates: dict[str, str] = {}
    for obj in rigid_objects:
        if _is_container_like(obj):
            candidates[obj.source_uid] = _container_runtime_uid(obj)
            continue

        base = _target_runtime_suffix(_base_name(obj))
        base_count = sum(
            1 for other in rigid_objects if _base_name(other) == _base_name(obj)
        )
        candidates[obj.source_uid] = (
            base if base_count == 1 else _normalize_runtime_uid(obj.source_uid)
        )

    counts: dict[str, int] = {}
    for runtime_uid in candidates.values():
        counts[runtime_uid] = counts.get(runtime_uid, 0) + 1
    return {
        source_uid: (
            runtime_uid
            if counts[runtime_uid] == 1
            else _normalize_runtime_uid(source_uid)
        )
        for source_uid, runtime_uid in candidates.items()
    }


def _candidate_relative_runtime_uid(obj: _SceneObject) -> str:
    if _is_container_like(obj):
        return _container_runtime_uid(obj)
    return _target_runtime_suffix(_base_name(obj))


def _is_container_like(obj: _SceneObject) -> bool:
    return any(keyword in _object_text(obj) for keyword in _CONTAINER_KEYWORDS)


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _default_relative_task_summary(
    moved_uid: str,
    reference_uid: str,
    relation: str,
) -> str:
    return (
        f"Move `{moved_uid}` so its final state is "
        f"{_relative_relation_phrase(relation)} `{reference_uid}`."
    )


def _relative_relation_phrase(relation: str) -> str:
    relation = _normalize_relative_relation(relation)
    if relation == "inside":
        return "inside"
    if relation == "on":
        return "on top of"
    if relation == "left_of":
        return "to the left of"
    if relation == "right_of":
        return "to the right of"
    if relation == "front_of":
        return "in front of"
    if relation == "behind":
        return "behind"
    raise ValueError(f"Unsupported relative placement relation: {relation!r}.")


def _build_ur5_basket_bundle(
    *,
    scene_dir: Path,
    source_config: Mapping[str, Any],
    roles: _BasketTaskRoles,
    project_name: str,
    task_name: str,
    target_body_scale: float | list[float] | tuple[float, float, float],
    target_replacements: Sequence[_ResolvedTargetReplacement],
    max_episodes: int,
    max_episode_steps: int,
) -> dict[str, Any]:
    scene_objects = _collect_scene_objects(source_config)
    by_uid = {obj.source_uid: obj for obj in scene_objects}
    replacement_by_source_uid = {
        replacement.source_uid: replacement for replacement in target_replacements
    }
    object_scale = _target_body_scale_vector(target_body_scale)
    container_scale = _source_body_scale(by_uid[roles.container_source_uid])
    task_source_uids = {
        roles.container_source_uid,
        roles.left_target_source_uid,
        roles.right_target_source_uid,
    }
    extra_rigid_objects = [
        obj
        for obj in scene_objects
        if obj.source_role == "rigid_object" and obj.source_uid not in task_source_uids
    ]
    extra_background_objects = [
        obj
        for obj in scene_objects
        if obj.source_role == "background" and obj.source_uid != roles.table_source_uid
    ]
    robot_init_z = _estimate_dual_ur5_init_z(scene_objects)

    gym_config = {
        "id": "AtomicActionsAgent-v3",
        "max_episodes": int(max_episodes),
        "max_episode_steps": int(max_episode_steps),
        "env": {
            "extensions": _make_extensions_config(roles),
            "events": _make_events_config(roles),
            "observations": _make_observations_config(),
            "dataset": _make_dataset_config(project_name, roles),
        },
        "robot": _make_dual_ur5_robot_config(robot_init_z=robot_init_z),
        "sensor": _make_sensor_config(),
        "light": _make_light_config(),
        "background": [
            _make_background_config(scene_dir, by_uid[roles.table_source_uid]),
            *[
                _make_extra_background_config(scene_dir, obj)
                for obj in extra_background_objects
            ],
        ],
        "rigid_object": [
            _make_target_object_config(
                scene_dir,
                by_uid[roles.right_target_source_uid],
                roles.right_target_runtime_uid,
                object_scale,
                replacement_by_source_uid.get(roles.right_target_source_uid),
            ),
            _make_target_object_config(
                scene_dir,
                by_uid[roles.left_target_source_uid],
                roles.left_target_runtime_uid,
                object_scale,
                replacement_by_source_uid.get(roles.left_target_source_uid),
            ),
            _make_container_object_config(
                scene_dir,
                by_uid[roles.container_source_uid],
                roles.container_runtime_uid,
                container_scale,
            ),
            *[
                _make_extra_rigid_object_config(scene_dir, obj, _source_body_scale(obj))
                for obj in extra_rigid_objects
            ],
        ],
    }
    return {
        "gym_config": gym_config,
        "agent_config": _make_agent_config(),
        "task_prompt": _make_task_prompt(task_name, project_name, roles),
        "basic_background": _make_basic_background(project_name, roles),
        "atom_actions": _make_atom_actions_prompt(roles),
        "summary": {
            "mode": "basket_template",
            "left_target": roles.left_target_runtime_uid,
            "right_target": roles.right_target_runtime_uid,
            "container": roles.container_runtime_uid,
            "target_replacements": [
                {
                    "source_uid": replacement.source_uid,
                    "prompt": replacement.prompt,
                    "output_dir_name": replacement.output_dir_name,
                    "mesh_path": replacement.mesh_path.as_posix(),
                    "runtime_noun": replacement.runtime_noun,
                }
                for replacement in target_replacements
            ],
        },
    }


def _build_relative_placement_bundle(
    *,
    scene_dir: Path,
    source_config: Mapping[str, Any],
    spec: _RelativePlacementSpec,
    project_name: str,
    task_name: str,
    target_body_scale: float | list[float] | tuple[float, float, float],
    max_episodes: int,
    max_episode_steps: int,
) -> dict[str, Any]:
    scene_objects = _collect_scene_objects(source_config)
    background_objects = [
        obj for obj in scene_objects if obj.source_role == "background"
    ]
    rigid_objects = [obj for obj in scene_objects if obj.source_role == "rigid_object"]
    by_uid = {obj.source_uid: obj for obj in scene_objects}
    runtime_uids = _relative_runtime_uid_mapping(rigid_objects)
    object_scale = _target_body_scale_vector(target_body_scale)
    robot_init_z = _estimate_dual_ur5_init_z(scene_objects)

    gym_config = {
        "id": "AtomicActionsAgent-v3",
        "max_episodes": int(max_episodes),
        "max_episode_steps": int(max_episode_steps),
        "env": {
            "extensions": _make_relative_extensions_config(spec),
            "events": _make_relative_events_config(spec, list(runtime_uids.values())),
            "observations": _make_observations_config(),
            "dataset": _make_relative_dataset_config(project_name, spec),
        },
        "robot": _make_dual_ur5_robot_config(robot_init_z=robot_init_z),
        "sensor": _make_sensor_config(),
        "light": _make_light_config(),
        "background": [
            _make_background_config(scene_dir, by_uid[spec.table_source_uid]),
            *[
                _make_extra_background_config(scene_dir, obj, object_scale)
                for obj in background_objects
                if obj.source_uid != spec.table_source_uid
            ],
        ],
        "rigid_object": [
            _make_relative_rigid_object_config(
                scene_dir=scene_dir,
                obj=obj,
                runtime_uid=runtime_uids[obj.source_uid],
                body_scale=_relative_object_body_scale(
                    obj,
                    target_scale=object_scale,
                ),
                max_convex_hull_num=_relative_rigid_object_max_convex_hull_num(
                    runtime_uids[obj.source_uid],
                    spec,
                ),
            )
            for obj in rigid_objects
        ],
    }
    return {
        "gym_config": gym_config,
        "agent_config": _make_agent_config(),
        "task_prompt": _make_relative_task_prompt(task_name, project_name, spec),
        "basic_background": _make_relative_basic_background(project_name, spec),
        "atom_actions": _make_relative_atom_actions_prompt(spec),
        "summary": {
            "mode": "relative_placement",
            "moved_object": spec.moved_runtime_uid,
            "reference_object": spec.reference_runtime_uid,
            "relation": spec.relation,
            "active_arm": f"{spec.active_side}_arm",
            "release_offset": spec.release_offset,
        },
    }


def _target_body_scale_vector(
    target_body_scale: float | list[float] | tuple[float, float, float],
) -> list[float]:
    if isinstance(target_body_scale, (int, float)):
        value = float(target_body_scale)
        return [value, value, value]
    return _clean_vector3(target_body_scale)


def _source_body_scale(obj: _SceneObject) -> list[float]:
    return _clean_vector3(obj.config.get("body_scale", [1.0, 1.0, 1.0]))


def _relative_object_body_scale(
    obj: _SceneObject,
    *,
    target_scale: list[float],
) -> list[float]:
    if _is_container_object(obj):
        return _source_body_scale(obj)
    return target_scale


def _is_container_object(obj: _SceneObject) -> bool:
    text = _object_text(obj)
    return any(keyword in text for keyword in _CONTAINER_KEYWORDS)


def _estimate_dual_ur5_init_z(scene_objects: list[_SceneObject]) -> float:
    """Estimate robot base height from source tabletop object heights."""

    rigid_z_values = []
    for obj in scene_objects:
        if obj.source_role != "rigid_object":
            continue
        init_pos = obj.config.get("init_pos")
        if not isinstance(init_pos, (list, tuple)) or len(init_pos) < 3:
            continue
        try:
            rigid_z_values.append(float(init_pos[2]))
        except (TypeError, ValueError):
            continue

    if not rigid_z_values:
        return _DUAL_UR5_LEGACY_INIT_Z

    sorted_z = sorted(rigid_z_values)
    mid = len(sorted_z) // 2
    if len(sorted_z) % 2:
        tabletop_z = sorted_z[mid]
    else:
        tabletop_z = (sorted_z[mid - 1] + sorted_z[mid]) / 2.0

    if tabletop_z <= _DUAL_UR5_HIGH_TABLETOP_THRESHOLD:
        return _DUAL_UR5_LEGACY_INIT_Z
    return max(
        _DUAL_UR5_LEGACY_INIT_Z,
        tabletop_z - _DUAL_UR5_TABLETOP_Z_OFFSET,
    )


def _make_extensions_config(roles: _BasketTaskRoles) -> dict[str, Any]:
    return {
        "agent_arm_slots": {
            "left": {
                "arm": "left_arm",
                "eef": "left_eef",
            },
            "right": {
                "arm": "right_arm",
                "eef": "right_eef",
            },
        },
        "arm_aim_yaw_offset": {
            "left": 0.0,
            "right": 3.141592653589793,
        },
        "gripper_open_state": [0.0],
        "gripper_close_state": [0.04],
        "ignore_terminations_during_agent": True,
        "agent_success": {
            "op": "all",
            "terms": [
                _object_in_container_success(
                    roles.left_target_runtime_uid,
                    roles.container_runtime_uid,
                ),
                _object_in_container_success(
                    roles.right_target_runtime_uid,
                    roles.container_runtime_uid,
                ),
            ],
        },
        "agent_grasp_pose_overrides": [
            {
                "type": "top_down",
                "object": roles.left_target_runtime_uid,
                "side": "left",
                "height_offset": 0.036,
            },
            {
                "type": "top_down",
                "object": roles.right_target_runtime_uid,
                "side": "right",
                "height_offset": 0.036,
            },
        ],
    }


def _object_in_container_success(object_uid: str, container_uid: str) -> dict[str, Any]:
    return {
        "type": "object_in_container",
        "object": object_uid,
        "container": container_uid,
        "radius": 0.2,
        "min_z_offset": -0.05,
        "max_z_offset": 0.35,
    }


def _make_relative_extensions_config(spec: _RelativePlacementSpec) -> dict[str, Any]:
    return {
        "agent_arm_slots": {
            "left": {
                "arm": "left_arm",
                "eef": "left_eef",
            },
            "right": {
                "arm": "right_arm",
                "eef": "right_eef",
            },
        },
        "arm_aim_yaw_offset": {
            "left": 0.0,
            "right": 3.141592653589793,
        },
        "gripper_open_state": [0.0],
        "gripper_close_state": [0.04],
        "ignore_terminations_during_agent": True,
        "agent_success": _make_relative_success_spec(spec),
        "agent_grasp_pose_overrides": [
            {
                "type": "top_down",
                "object": spec.moved_runtime_uid,
                "side": spec.active_side,
                "height_offset": 0.036,
            },
        ],
    }


def _make_relative_success_spec(spec: _RelativePlacementSpec) -> dict[str, Any]:
    if spec.relation == "inside":
        return _object_in_container_success(
            spec.moved_runtime_uid,
            spec.reference_runtime_uid,
        )
    if spec.relation == "on":
        return {
            "type": "object_on_object",
            "object": spec.moved_runtime_uid,
            "support": spec.reference_runtime_uid,
            "xy_radius": 0.08,
            "min_z_offset": 0.02,
            "max_z_offset": 0.35,
        }

    primary_axis, primary_offset, secondary_axis = _side_relation_axes(spec.relation)
    return {
        "op": "all",
        "terms": [
            {
                "type": "object_axis_offset_near",
                "object": spec.moved_runtime_uid,
                "reference": spec.reference_runtime_uid,
                "axis": primary_axis,
                "offset": primary_offset,
                "tolerance": 0.05,
            },
            {
                "type": "object_axis_offset_near",
                "object": spec.moved_runtime_uid,
                "reference": spec.reference_runtime_uid,
                "axis": secondary_axis,
                "offset": 0.0,
                "tolerance": 0.06,
            },
            {
                "type": "object_not_fallen",
                "object": spec.moved_runtime_uid,
                "max_tilt": 0.9,
            },
        ],
    }


def _side_relation_axes(relation: str) -> tuple[str, float, str]:
    if relation == "left_of":
        return "x", -_SIDE_RELATION_DISTANCE, "y"
    if relation == "right_of":
        return "x", _SIDE_RELATION_DISTANCE, "y"
    if relation == "front_of":
        return "y", -_SIDE_RELATION_DISTANCE, "x"
    if relation == "behind":
        return "y", _SIDE_RELATION_DISTANCE, "x"
    raise ValueError(f"Unsupported side relation: {relation!r}.")


def _make_relative_events_config(
    spec: _RelativePlacementSpec,
    rigid_runtime_uids: list[str],
) -> dict[str, Any]:
    return {
        "record_camera": {
            "func": "record_camera_data",
            "mode": "interval",
            "interval_step": 1,
            "params": {
                "name": "cam1",
                "resolution": [320, 240],
                "eye": [0.0, -1.8, 1.45],
                "target": [0.0, 0.0, 0.72],
            },
        },
        "validation_cameras": {
            "func": "validation_cameras",
            "mode": "trigger",
            "params": {
                "cameras": [
                    _validation_camera(
                        "valid_cam_front",
                        eye=[0.0, -1.8, 1.45],
                    ),
                    _validation_camera(
                        "valid_cam_left",
                        eye=[-1.4, -0.7, 1.35],
                    ),
                    _validation_camera(
                        "valid_cam_right",
                        eye=[1.4, -0.7, 1.35],
                    ),
                ]
            },
        },
        "prepare_extra_attr": {
            "func": "prepare_extra_attr",
            "mode": "reset",
            "params": {
                "attrs": [
                    {
                        "name": "object_lengths",
                        "mode": "callable",
                        "entity_uids": "all_objects",
                        "func_name": "compute_object_length",
                        "func_kwargs": {
                            "is_svd_frame": True,
                            "sample_points": 5000,
                        },
                    },
                    {
                        "name": "grasp_pose_object",
                        "mode": "static",
                        "entity_uids": [spec.moved_runtime_uid],
                        "value": [
                            [
                                [0.0, 0.0, 1.0, 0.0],
                                [1.0, 0.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0, 0.036],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        ],
                    },
                ]
            },
        },
        "register_info_to_env": {
            "func": "register_info_to_env",
            "mode": "reset",
            "params": {
                "registry": [
                    _object_registry_entry(uid) for uid in sorted(rigid_runtime_uids)
                ],
                "registration": "affordance_datas",
                "sim_update": True,
            },
        },
    }


def _make_events_config(roles: _BasketTaskRoles) -> dict[str, Any]:
    return {
        "record_camera": {
            "func": "record_camera_data",
            "mode": "interval",
            "interval_step": 1,
            "params": {
                "name": "cam1",
                "resolution": [320, 240],
                "eye": [0.0, -1.8, 1.45],
                "target": [0.0, 0.0, 0.72],
            },
        },
        "validation_cameras": {
            "func": "validation_cameras",
            "mode": "trigger",
            "params": {
                "cameras": [
                    _validation_camera(
                        "valid_cam_front",
                        eye=[0.0, -1.8, 1.45],
                    ),
                    _validation_camera(
                        "valid_cam_left",
                        eye=[-1.4, -0.7, 1.35],
                    ),
                    _validation_camera(
                        "valid_cam_right",
                        eye=[1.4, -0.7, 1.35],
                    ),
                ]
            },
        },
        "prepare_extra_attr": {
            "func": "prepare_extra_attr",
            "mode": "reset",
            "params": {
                "attrs": [
                    {
                        "name": "object_lengths",
                        "mode": "callable",
                        "entity_uids": "all_objects",
                        "func_name": "compute_object_length",
                        "func_kwargs": {
                            "is_svd_frame": True,
                            "sample_points": 5000,
                        },
                    },
                    {
                        "name": "grasp_pose_object",
                        "mode": "static",
                        "entity_uids": [
                            roles.left_target_runtime_uid,
                            roles.right_target_runtime_uid,
                        ],
                        "value": [
                            [
                                [0.0, 0.0, 1.0, 0.0],
                                [1.0, 0.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0, 0.036],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        ],
                    },
                ]
            },
        },
        "register_info_to_env": {
            "func": "register_info_to_env",
            "mode": "reset",
            "params": {
                "registry": [
                    _object_registry_entry(roles.left_target_runtime_uid),
                    _object_registry_entry(roles.right_target_runtime_uid),
                    _object_registry_entry(roles.container_runtime_uid),
                ],
                "registration": "affordance_datas",
                "sim_update": True,
            },
        },
    }


def _validation_camera(uid: str, *, eye: list[float]) -> dict[str, Any]:
    return {
        "uid": uid,
        "width": 1280,
        "height": 960,
        "enable_mask": False,
        "intrinsics": [1400, 1400, 640, 480],
        "extrinsics": {
            "eye": eye,
            "target": [0.0, 0.0, 0.72],
        },
    }


def _object_registry_entry(uid: str) -> dict[str, Any]:
    return {
        "entity_cfg": {
            "uid": uid,
        },
        "pose_register_params": {
            "compute_relative": False,
            "compute_pose_object_to_arena": True,
            "to_matrix": True,
        },
    }


def _make_observations_config() -> dict[str, Any]:
    return {
        "norm_robot_eef_joint": {
            "func": "normalize_robot_joint_data",
            "mode": "modify",
            "name": "robot/qpos",
            "params": {
                "joint_ids": [12, 13, 14, 15],
            },
        }
    }


def _make_dataset_config(
    project_name: str,
    roles: _BasketTaskRoles,
) -> dict[str, Any]:
    left_target_text = _left_target_text(roles)
    right_target_text = _right_target_text(roles)
    target_description = _target_task_description_text(roles)
    return {
        "lerobot": {
            "func": "LeRobotRecorder",
            "mode": "save",
            "params": {
                "robot_meta": {
                    "robot_type": "DualUR5",
                    "control_freq": 25,
                },
                "instruction": {
                    "lang": (
                        f"Use the left UR5 to place the left {left_target_text} into "
                        f"the {roles.container_runtime_uid}, then use the right "
                        f"UR5 to place the right {right_target_text} into the "
                        f"{roles.container_runtime_uid}."
                    ),
                },
                "extra": {
                    "scene_type": project_name,
                    "task_description": (
                        f"Dual UR5 {target_description}-to-container placement"
                    ),
                    "data_type": "sim",
                },
                "use_videos": True,
            },
        }
    }


def _make_relative_dataset_config(
    project_name: str,
    spec: _RelativePlacementSpec,
) -> dict[str, Any]:
    return {
        "lerobot": {
            "func": "LeRobotRecorder",
            "mode": "save",
            "params": {
                "robot_meta": {
                    "robot_type": "DualUR5",
                    "control_freq": 25,
                },
                "instruction": {
                    "lang": (
                        f"Use the {spec.active_side} UR5 to move "
                        f"{spec.moved_runtime_uid} "
                        f"{_relative_relation_phrase(spec.relation)} "
                        f"{spec.reference_runtime_uid}."
                    ),
                },
                "extra": {
                    "scene_type": project_name,
                    "task_description": spec.task_description,
                    "data_type": "sim",
                },
                "use_videos": True,
            },
        }
    }


def _make_dual_ur5_robot_config(*, robot_init_z: float) -> dict[str, Any]:
    return {
        "uid": "DualUR5",
        "urdf_cfg": {
            "fname": "dual_ur5_dh_pgi_basket",
            "components": [
                {
                    "component_type": "left_arm",
                    "urdf_path": "UniversalRobots/UR5/UR5.urdf",
                    "transform": [
                        [0.0, -1.0, 0.0, -0.3],
                        [1.0, 0.0, 0.0, -1.45],
                        [0.0, 0.0, 1.0, 0.4],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                },
                {
                    "component_type": "left_hand",
                    "urdf_path": "DH_PGI_140_80/DH_PGI_140_80.urdf",
                },
                {
                    "component_type": "right_arm",
                    "urdf_path": "UniversalRobots/UR5/UR5.urdf",
                    "transform": [
                        [0.0, -1.0, 0.0, 0.3],
                        [1.0, 0.0, 0.0, -1.45],
                        [0.0, 0.0, 1.0, 0.4],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                },
                {
                    "component_type": "right_hand",
                    "urdf_path": "DH_PGI_140_80/DH_PGI_140_80.urdf",
                },
            ],
        },
        "init_pos": [-2.0, 0.0, float(robot_init_z)],
        "init_rot": [0.0, 0.0, 90.0],
        "init_qpos": [
            0.0,
            0.0,
            -1.57,
            -1.57,
            1.57,
            1.57,
            1.57,
            1.57,
            1.57,
            1.57,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        "drive_pros": {
            "stiffness": {
                "LEFT_JOINT[1-6]": 10000.0,
                "RIGHT_JOINT[1-6]": 10000.0,
                "LEFT_GRIPPER_FINGER[1-2]_JOINT_1": 100.0,
                "RIGHT_GRIPPER_FINGER[1-2]_JOINT_1": 100.0,
            },
            "damping": {
                "LEFT_JOINT[1-6]": 1000.0,
                "RIGHT_JOINT[1-6]": 1000.0,
                "LEFT_GRIPPER_FINGER[1-2]_JOINT_1": 10.0,
                "RIGHT_GRIPPER_FINGER[1-2]_JOINT_1": 10.0,
            },
            "max_effort": {
                "LEFT_JOINT[1-6]": 100000.0,
                "RIGHT_JOINT[1-6]": 100000.0,
                "LEFT_GRIPPER_FINGER[1-2]_JOINT_1": 1000.0,
                "RIGHT_GRIPPER_FINGER[1-2]_JOINT_1": 1000.0,
            },
        },
        "control_parts": {
            "left_arm": ["LEFT_JOINT[1-6]"],
            "left_eef": ["LEFT_GRIPPER_FINGER[1-2]_JOINT_1"],
            "right_arm": ["RIGHT_JOINT[1-6]"],
            "right_eef": ["RIGHT_GRIPPER_FINGER[1-2]_JOINT_1"],
        },
        "solver_cfg": {
            "left_arm": _ur5_solver_config("left"),
            "right_arm": _ur5_solver_config("right"),
        },
    }


def _ur5_solver_config(side: str) -> dict[str, Any]:
    return {
        "class_type": "PytorchSolver",
        "end_link_name": f"{side}_ee_link",
        "root_link_name": f"{side}_base_link",
        "tcp": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.16],
            [0.0, 0.0, 0.0, 1.0],
        ],
    }


def _make_sensor_config() -> list[dict[str, Any]]:
    return [
        {
            "sensor_type": "Camera",
            "uid": "cam_high",
            "width": 960,
            "height": 540,
            "intrinsics": [488.1665344238281, 488.1665344238281, 480, 270],
            "extrinsics": {
                "pos": [0.02, 0.13, 1.71],
                "quat": [0, 0, 0.9996550532166887, -0.095845],
            },
        },
        {
            "sensor_type": "Camera",
            "uid": "cam_wrist_left",
            "width": 640,
            "height": 480,
            "intrinsics": [600, 600, 320, 240],
            "extrinsics": {
                "parent": "left_ee_link",
                "pos": [0.0, 0.12, 0.08],
                "quat": [
                    -0.0012598701,
                    -0.029051816664441618998,
                    0.9094039177564813,
                    0.41489627504330695,
                ],
            },
        },
        {
            "sensor_type": "Camera",
            "uid": "cam_wrist_right",
            "width": 640,
            "height": 480,
            "intrinsics": [600, 600, 320, 240],
            "extrinsics": {
                "parent": "right_ee_link",
                "pos": [0.0, 0.12, 0.08],
                "quat": [
                    -0.0012598701,
                    -0.029051816664441618998,
                    0.9094039177564813,
                    0.41489627504330695,
                ],
            },
        },
    ]


def _make_light_config() -> dict[str, Any]:
    return {
        "direct": [
            {
                "uid": "main_light",
                "light_type": "point",
                "color": [1.0, 1.0, 1.0],
                "intensity": 40.0,
                "init_pos": [0.0, -0.4, 2.2],
                "radius": 10.0,
            }
        ]
    }


def _make_background_config(scene_dir: Path, obj: _SceneObject) -> dict[str, Any]:
    return {
        "uid": "table",
        "shape": _make_shape_config(scene_dir, obj.config),
        "attrs": dict(_BACKGROUND_ATTRS),
        "body_scale": _clean_vector3(obj.config.get("body_scale", [1.0, 1.0, 1.0])),
        "body_type": "kinematic",
        "init_pos": _clean_vector3(obj.config.get("init_pos", [0.0, 0.0, 0.0])),
        "init_rot": _clean_vector3(obj.config.get("init_rot", [0.0, 0.0, 0.0])),
        "max_convex_hull_num": _role_limited_max_convex_hull_num(
            obj,
            _BACKGROUND_MAX_CONVEX_HULL_NUM,
        ),
    }


def _make_extra_background_config(
    scene_dir: Path,
    obj: _SceneObject,
    body_scale: Any | None = None,
) -> dict[str, Any]:
    config = {
        "uid": _normalize_runtime_uid(obj.source_uid),
        "shape": _make_shape_config(scene_dir, obj.config),
        "attrs": copy.deepcopy(dict(obj.config.get("attrs", _BACKGROUND_ATTRS))),
        "body_scale": _clean_vector3(
            obj.config.get("body_scale", [1.0, 1.0, 1.0])
            if body_scale is None
            else body_scale
        ),
        "body_type": str(obj.config.get("body_type", "static")),
        "init_pos": _clean_vector3(obj.config.get("init_pos", [0.0, 0.0, 0.0])),
        "init_rot": _clean_vector3(obj.config.get("init_rot", [0.0, 0.0, 0.0])),
        "max_convex_hull_num": _role_limited_max_convex_hull_num(
            obj,
            _BACKGROUND_MAX_CONVEX_HULL_NUM,
        ),
    }
    return config


def _make_target_object_config(
    scene_dir: Path,
    obj: _SceneObject,
    runtime_uid: str,
    target_scale: list[float],
    replacement: _ResolvedTargetReplacement | None = None,
) -> dict[str, Any]:
    return _make_rigid_object_config(
        scene_dir,
        obj,
        runtime_uid,
        target_scale,
        max_convex_hull_num=_role_limited_max_convex_hull_num(
            obj,
            _TARGET_MAX_CONVEX_HULL_NUM,
        ),
        mesh_fpath=replacement.mesh_path if replacement else None,
    )


def _make_container_object_config(
    scene_dir: Path,
    obj: _SceneObject,
    runtime_uid: str,
    body_scale: Any,
) -> dict[str, Any]:
    return _make_rigid_object_config(
        scene_dir,
        obj,
        runtime_uid,
        body_scale,
        max_convex_hull_num=_role_limited_max_convex_hull_num(
            obj,
            _CONTAINER_MAX_CONVEX_HULL_NUM,
        ),
    )


def _make_extra_rigid_object_config(
    scene_dir: Path,
    obj: _SceneObject,
    body_scale: Any,
) -> dict[str, Any]:
    return _make_rigid_object_config(
        scene_dir,
        obj,
        _normalize_runtime_uid(obj.source_uid),
        body_scale,
        max_convex_hull_num=_role_limited_max_convex_hull_num(
            obj,
            _EXTRA_RIGID_MAX_CONVEX_HULL_NUM,
        ),
    )


def _make_relative_rigid_object_config(
    *,
    scene_dir: Path,
    obj: _SceneObject,
    runtime_uid: str,
    body_scale: Any,
    max_convex_hull_num: int,
) -> dict[str, Any]:
    return _make_rigid_object_config(
        scene_dir,
        obj,
        runtime_uid,
        body_scale,
        max_convex_hull_num=_role_limited_max_convex_hull_num(
            obj,
            max_convex_hull_num,
        ),
    )


def _make_rigid_object_config(
    scene_dir: Path,
    obj: _SceneObject,
    runtime_uid: str,
    body_scale: Any,
    max_convex_hull_num: int,
    mesh_fpath: str | Path | None = None,
) -> dict[str, Any]:
    config = {
        "uid": runtime_uid,
        "shape": _make_shape_config(scene_dir, obj.config, mesh_fpath=mesh_fpath),
        "attrs": dict(_RIGID_OBJECT_ATTRS),
        "init_pos": _clean_vector3(obj.config.get("init_pos", [0.0, 0.0, 0.0])),
        "init_rot": _clean_vector3(obj.config.get("init_rot", [0.0, 0.0, 0.0])),
        "body_scale": _clean_vector3(body_scale),
        "max_convex_hull_num": int(max_convex_hull_num),
    }
    if "body_type" in obj.config:
        config["body_type"] = str(obj.config["body_type"])
    return config


def _role_limited_max_convex_hull_num(
    obj: _SceneObject,
    role_max_convex_hull_num: int,
) -> int:
    source_max_convex_hull_num = obj.config.get("max_convex_hull_num")
    if source_max_convex_hull_num is None:
        return role_max_convex_hull_num
    return max(1, min(int(source_max_convex_hull_num), role_max_convex_hull_num))


def _relative_rigid_object_max_convex_hull_num(
    runtime_uid: str,
    spec: _RelativePlacementSpec,
) -> int:
    if spec.relation == "inside" and runtime_uid == spec.reference_runtime_uid:
        return _CONTAINER_MAX_CONVEX_HULL_NUM
    if runtime_uid in {spec.moved_runtime_uid, spec.reference_runtime_uid}:
        return _TARGET_MAX_CONVEX_HULL_NUM
    return _EXTRA_RIGID_MAX_CONVEX_HULL_NUM


def _make_shape_config(
    scene_dir: Path,
    source_config: Mapping[str, Any],
    *,
    mesh_fpath: str | Path | None = None,
) -> dict[str, Any]:
    shape = copy.deepcopy(dict(source_config.get("shape", {})))
    if mesh_fpath is not None:
        shape["shape_type"] = "Mesh"
        shape["fpath"] = str(mesh_fpath)
    if shape.get("shape_type") == "Mesh" and "fpath" in shape:
        shape["fpath"] = _asset_path_for_config(scene_dir, str(shape["fpath"]))
    shape.setdefault("compute_uv", False)
    return shape


def _asset_path_for_config(scene_dir: Path, fpath: str) -> str:
    raw_path = Path(fpath)
    if raw_path.is_absolute():
        candidate = raw_path.resolve()
    else:
        candidate = (scene_dir / raw_path).resolve()

    repo_root = _repo_root()
    if _is_relative_to(candidate, repo_root):
        return candidate.relative_to(repo_root).as_posix()
    return candidate.as_posix()


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "setup.py").exists() and (parent / "embodichain").exists():
            return parent
    return current.parents[3]


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _make_agent_config() -> dict[str, Any]:
    return {
        "TaskAgent": {
            "prompt_name": "generate_task_graph",
        },
        "RecoveryAgent": {
            "prompt_name": "augment_task_graph",
        },
        "CompileAgent": {
            "prompt_name": "compile_agent_graph",
        },
        "Agent": {
            "prompt_kwargs": {
                "task_prompt": {
                    "type": "text",
                    "name": "task_prompt.txt",
                },
                "basic_background": {
                    "type": "text",
                    "name": "basic_background.txt",
                },
                "atom_actions": {
                    "type": "text",
                    "name": "atom_actions.txt",
                },
                "error_functions": {
                    "type": "text",
                    "name": "error_functions.txt",
                },
                "monitor_functions": {
                    "type": "text",
                    "name": "monitor_functions.txt",
                },
                "recovery_rules": {
                    "type": "text",
                    "name": "recovery_rules.txt",
                },
            }
        },
    }


def _make_relative_task_prompt(
    task_name: str,
    project_name: str,
    spec: _RelativePlacementSpec,
) -> str:
    active_arm = f"{spec.active_side}_arm"
    inactive_arm = "right_arm" if spec.active_side == "left" else "left_arm"
    active_slot = f"{spec.active_side}_arm_action"
    inactive_slot = (
        "right_arm_action" if spec.active_side == "left" else "left_arm_action"
    )
    high_x, high_y, high_z = spec.high_offset
    rel_x, rel_y, rel_z = spec.release_offset
    action_sketch = _format_action_sketch(spec.action_sketch)
    return f"""Task:
{task_name}: {spec.task_prompt_summary}

This config was generated from a simple task description by the config-stage
LLM. The execution-stage LLM must now generate the graph JSON from this prompt.

Original simple task description:
{spec.task_description}

Config-stage LLM interpretation:
{action_sketch}

Object and arm mapping:
- Move `{spec.moved_runtime_uid}`. Source object: `{spec.moved_source_uid}`.
- Use `{spec.reference_runtime_uid}` as the spatial reference. Source object:
  `{spec.reference_source_uid}`.
- Goal relation: `{spec.relation}` ({_relative_relation_phrase(spec.relation)}).
- Active arm: `{active_arm}`.
- Keep every `{inactive_slot}` as null.

Coordinate convention for relative placement:
- `left_of` means negative world x relative to the reference object.
- `right_of` means positive world x relative to the reference object.
- `front_of` means negative world y, closer to the Dual-UR5 bases.
- `behind` means positive world y, farther from the Dual-UR5 bases.
- `inside` and `on` use the reference object's xy center.

Generate one deterministic nominal graph with exactly 6 nominal edges. Use only
the atomic actions shown below. Do not add recovery, monitor, search, alignment,
or extra lift edges. The inactive arm must remain null in every edge.

1. Grasp the moved object:
   - {active_slot}: grasp(robot_name="{active_arm}",
     obj_name="{spec.moved_runtime_uid}", pre_grasp_dis=0.08, sample_num=90)
   - {inactive_slot}: null

2. Move the held object to the high staging pose relative to the reference:
   - {active_slot}: move_relative_to_object(robot_name="{active_arm}",
     obj_name="{spec.reference_runtime_uid}", x_offset={_format_prompt_float(high_x)},
     y_offset={_format_prompt_float(high_y)},
     z_offset={_format_prompt_float(high_z)}, sample_num=90)
   - {inactive_slot}: null

3. Lower the held object to the release pose:
   - {active_slot}: move_relative_to_object(robot_name="{active_arm}",
     obj_name="{spec.reference_runtime_uid}", x_offset={_format_prompt_float(rel_x)},
     y_offset={_format_prompt_float(rel_y)},
     z_offset={_format_prompt_float(rel_z)}, sample_num=60)
   - {inactive_slot}: null

4. Release the moved object:
   - {active_slot}: open_gripper(robot_name="{active_arm}",
     sample_num=30, open_threshold=-1.0, settle_steps=50)
   - {inactive_slot}: null

5. Move the empty gripper upward to clear the object:
   - {active_slot}: move_by_relative_offset(robot_name="{active_arm}",
     dx=0.0, dy=0.0, dz=0.14, mode="extrinsic", sample_num=40)
   - {inactive_slot}: null

6. Return the active arm to its initial pose:
   - {active_slot}: back_to_initial_pose(robot_name="{active_arm}", sample_num=60)
   - {inactive_slot}: null

Final state: `{spec.moved_runtime_uid}` must be
{_relative_relation_phrase(spec.relation)} `{spec.reference_runtime_uid}`. Always
plan to the current object poses from the exported {project_name} environment
config. Do not hard-code absolute object coordinates in the generated graph.
"""


def _make_relative_basic_background(
    project_name: str,
    spec: _RelativePlacementSpec,
) -> str:
    active_arm = f"{spec.active_side}_arm"
    inactive_arm = "right_arm" if spec.active_side == "left" else "left_arm"
    notes = spec.basic_background_notes or (
        "No extra scene notes were provided by the config-stage LLM."
    )
    return f"""The scene comes from the exported {project_name} mesh environment.

This configuration directory is for a Dual-UR5 relative-placement task generated
from a simple natural-language task description.

The robot is a dual-UR5 composite robot with DH_PGI_140_80 parallel grippers:
- left_arm is the UR5 outside the left side of the table's near long edge.
- right_arm is the UR5 outside the right side of the table's near long edge.

The active arm for this task is `{active_arm}`. The inactive arm
`{inactive_arm}` must stay null in the nominal graph.

Interactive task objects:
- {spec.moved_runtime_uid}: moved object from source `{spec.moved_source_uid}`.
- {spec.reference_runtime_uid}: reference object from source
  `{spec.reference_source_uid}`.

Config-stage LLM notes:
{notes}

The execution-stage LLM should generate graph JSON that grasps the moved object,
moves it to a high staging pose relative to the current reference object pose,
lowers to the release pose, opens the gripper, retreats upward, and returns the
active arm to its initial pose.
"""


def _make_relative_atom_actions_prompt(spec: _RelativePlacementSpec) -> str:
    active_arm = f"{spec.active_side}_arm"
    inactive_arm = "right_arm" if spec.active_side == "left" else "left_arm"
    high_x, high_y, high_z = spec.high_offset
    rel_x, rel_y, rel_z = spec.release_offset
    return f"""### Atom Functions for Dual-UR5 Relative Placement

Use the existing atomic action API and gripper semantics in this configuration.
The active arm is `{active_arm}`. Do not call atomic actions for `{inactive_arm}`
in the nominal graph.

Each atomic function returns a list of joint-space trajectories (`list[np.ndarray]`).
All atom functions are public-only wrappers backed by
`embodichain.lab.sim.atomic_actions.AtomicActionEngine`. Public action failures
raise immediately. Use only the parameters listed below.

"grasp":
    def grasp(robot_name: str, obj_name: str, pre_grasp_dis: float, **kwargs) -> list[np.ndarray]

    Use exactly:
    `grasp(robot_name="{active_arm}", obj_name="{spec.moved_runtime_uid}",
    pre_grasp_dis=0.08, sample_num=90)`.

"move_relative_to_object":
    def move_relative_to_object(robot_name: str,
                                obj_name: str,
                                x_offset=0.0,
                                y_offset=0.0,
                                z_offset=0.0,
                                **kwargs) -> list[np.ndarray]

    Moves the selected end-effector to the current reference object pose plus a
    world-frame xyz offset while preserving the current end-effector orientation.
    For this task, use `obj_name="{spec.reference_runtime_uid}"`.

    High staging pose:
    `move_relative_to_object(robot_name="{active_arm}",
    obj_name="{spec.reference_runtime_uid}",
    x_offset={_format_prompt_float(high_x)}, y_offset={_format_prompt_float(high_y)},
    z_offset={_format_prompt_float(high_z)}, sample_num=90)`.

    Release pose:
    `move_relative_to_object(robot_name="{active_arm}",
    obj_name="{spec.reference_runtime_uid}",
    x_offset={_format_prompt_float(rel_x)}, y_offset={_format_prompt_float(rel_y)},
    z_offset={_format_prompt_float(rel_z)}, sample_num=60)`.

"move_by_relative_offset":
    def move_by_relative_offset(robot_name: str,
                                dx=0.0,
                                dy=0.0,
                                dz=0.0,
                                mode="extrinsic",
                                **kwargs) -> list[np.ndarray]

    Use this only for the short upward retreat after releasing the object:
    `move_by_relative_offset(robot_name="{active_arm}", dx=0.0, dy=0.0,
    dz=0.14, mode="extrinsic", sample_num=40)`.

"open_gripper":
    def open_gripper(robot_name: str, **kwargs) -> list[np.ndarray]

    Opens the selected gripper to release the held object. Use:
    `open_gripper(robot_name="{active_arm}", sample_num=30,
    open_threshold=-1.0, settle_steps=50)`.

"close_gripper":
    def close_gripper(robot_name: str, **kwargs) -> list[np.ndarray]

    Do not use this separately in the nominal graph, because `grasp` already
    closes the gripper.

"back_to_initial_pose":
    def back_to_initial_pose(robot_name: str, **kwargs) -> list[np.ndarray]

    Returns the selected UR5 arm to its initial joint configuration. Use it only
    after `open_gripper` and the upward retreat:
    `back_to_initial_pose(robot_name="{active_arm}", sample_num=60)`.
"""


def _format_action_sketch(action_sketch: list[str]) -> str:
    return "\n".join(f"- {item}" for item in action_sketch)


def _format_prompt_float(value: float) -> str:
    return f"{float(value):.12g}"


def _make_task_prompt(
    task_name: str,
    project_name: str,
    roles: _BasketTaskRoles,
) -> str:
    left_target_text = _left_target_text(roles)
    right_target_text = _right_target_text(roles)
    target_pair_text = _target_pair_text(roles)
    target_plural = _target_plural_text(roles)
    return f"""Task:
{task_name}: use the current two-UR5 configuration to place
{target_pair_text} into the {roles.container_runtime_uid}.

The task starts with both arms acting simultaneously:
the left UR5 grasps the left {left_target_text} while the right UR5 grasps the
right {right_target_text} in the same nominal graph edge. After both
{target_plural} are grasped, the left UR5 places its {left_target_text} into the
{roles.container_runtime_uid} and retreats upward. While the left UR5 returns
to its initial pose, the right UR5 must simultaneously begin placing its
already-grasped {right_target_text} by moving it to the high staging pose above
the {roles.container_runtime_uid}. The right UR5 then completes its placement
and returns to its initial pose.

Object and arm mapping:
- left_arm must only manipulate `{roles.left_target_runtime_uid}`.
- right_arm must only manipulate `{roles.right_target_runtime_uid}`.
- Both target objects must be released into `{roles.container_runtime_uid}`.

Generate one deterministic nominal graph with the following semantic sequence.
Do not add extra alignment, search, recovery, or monitor steps. Do include the
specified post-release retreat and return-to-initial steps. The left arm must
finish its upward retreat before the right arm enters the shared container
workspace, but the left return-to-initial action and the right high-staging
action must execute simultaneously in one graph edge. Generate exactly 10
nominal edges, one edge for each numbered step below. Do not split the
simultaneous grasp or the simultaneous left-return/right-staging action into
separate edges. Do not merge, reorder, or omit the lower-to-release,
open-gripper, upward-retreat, or final right return-to-initial edges.

A target object is not considered placed when it is only above the
{roles.container_runtime_uid}. For each arm, the placement order must be: move
to a high staging pose above the container, lower to the release pose inside the
container, open the gripper, move the empty gripper upward, then return the arm
to its initial pose. Never call `back_to_initial_pose` for an arm that has not
already executed `open_gripper` for its held target object.

1. Grasp both target objects simultaneously:
   - left_arm_action: grasp(robot_name="left_arm",
     obj_name="{roles.left_target_runtime_uid}", pre_grasp_dis=0.08,
     sample_num=90)
   - right_arm_action: grasp(robot_name="right_arm",
     obj_name="{roles.right_target_runtime_uid}", pre_grasp_dis=0.08,
     sample_num=90)

2. Move the held left target object directly above the left half of the
   {roles.container_runtime_uid} while the right arm keeps holding its target:
   - left_arm_action: move_relative_to_object(robot_name="left_arm",
     obj_name="{roles.container_runtime_uid}", x_offset=0.0, y_offset=-0.04,
     z_offset=0.22, sample_num=90)
   - right_arm_action: close_gripper(robot_name="right_arm", sample_num=20)

3. Lower the held left target object to the left release pose inside the
   {roles.container_runtime_uid}:
   - left_arm_action: move_relative_to_object(robot_name="left_arm",
     obj_name="{roles.container_runtime_uid}", x_offset=0.0, y_offset=-0.04,
     z_offset=0.12, sample_num=60)
   - right_arm_action: close_gripper(robot_name="right_arm", sample_num=20)

4. Release the left target object into the {roles.container_runtime_uid}:
   - left_arm_action: open_gripper(robot_name="left_arm",
     sample_num=30, open_threshold=-1.0, settle_steps=50)
   - right_arm_action: close_gripper(robot_name="right_arm", sample_num=20)

5. Move the empty left gripper upward to clear the container:
   - left_arm_action: move_by_relative_offset(robot_name="left_arm",
     dx=0.0, dy=0.0, dz=0.14, mode="extrinsic", sample_num=40)
   - right_arm_action: close_gripper(robot_name="right_arm", sample_num=20)

6. After the left gripper has retreated upward, return the left UR5 to its
   initial pose while simultaneously moving the held right target object
   directly above the right half of the {roles.container_runtime_uid}. This
   parallel handoff must remain one graph edge:
   - left_arm_action: back_to_initial_pose(robot_name="left_arm", sample_num=60)
   - right_arm_action: move_relative_to_object(robot_name="right_arm",
     obj_name="{roles.container_runtime_uid}", x_offset=0.0, y_offset=0.04,
     z_offset=0.22, sample_num=90)

7. Lower the held right target object to the right release pose inside the
   {roles.container_runtime_uid}:
   - left_arm_action: null
   - right_arm_action: move_relative_to_object(robot_name="right_arm",
     obj_name="{roles.container_runtime_uid}", x_offset=0.0, y_offset=0.04,
     z_offset=0.12, sample_num=60)

8. Release the right target object into the {roles.container_runtime_uid}:
   - left_arm_action: null
   - right_arm_action: open_gripper(robot_name="right_arm",
     sample_num=30, open_threshold=-1.0, settle_steps=50)

9. Move the empty right gripper upward to clear the container:
   - left_arm_action: null
   - right_arm_action: move_by_relative_offset(robot_name="right_arm",
     dx=0.0, dy=0.0, dz=0.14, mode="extrinsic", sample_num=40)

10. Return the right UR5 to its initial pose after releasing the target object:
   - left_arm_action: null
   - right_arm_action: back_to_initial_pose(robot_name="right_arm", sample_num=60)

The final state is both `{roles.left_target_runtime_uid}` and
`{roles.right_target_runtime_uid}` resting inside `{roles.container_runtime_uid}`,
with both arms moved away from the container workspace. Always plan to the
current `{roles.container_runtime_uid}` object pose from the exported
{project_name} environment config.
"""


def _make_basic_background(project_name: str, roles: _BasketTaskRoles) -> str:
    left_target_text = _left_target_text(roles)
    right_target_text = _right_target_text(roles)
    target_plural = _target_plural_text(roles)
    return f"""The scene comes from the exported {project_name} mesh environment.

This configuration directory is for the UR5BreadBasket task template. The
current robot is a dual-UR5 composite robot with DH_PGI_140_80 parallel
grippers.

The robot is a dual-UR5 composite robot with two parallel grippers:
- left_arm is the UR5 outside the left side of the table's near long edge.
- right_arm is the UR5 outside the right side of the table's near long edge.

Both UR5 bases are on the same long side of the table and face inward toward
the central {roles.container_runtime_uid}. The bases are intentionally kept
outside the table edge to avoid initial robot-table contact.

The interactive objects are:
- {roles.left_target_runtime_uid}: the {left_target_text} mesh initially on the
  negative-y side (source object {roles.left_target_source_uid}).
- {roles.right_target_runtime_uid}: the {right_target_text} mesh initially on the
  positive-y side (source object {roles.right_target_source_uid}).
- {roles.container_runtime_uid}: the target container near the center of the
  table (source object {roles.container_source_uid}).

The nominal task starts with simultaneous dual-arm grasping. The left UR5 must
grasp {roles.left_target_runtime_uid} while the right UR5 grasps
{roles.right_target_runtime_uid} in the same graph edge. After both
{target_plural} are held, the left UR5 places
{roles.left_target_runtime_uid} into {roles.container_runtime_uid}, releases
it, and retreats upward. The next graph edge is a parallel handoff: the left
UR5 returns to its initial pose while the right UR5 simultaneously moves its
already-grasped {roles.right_target_runtime_uid} to the high staging pose above
{roles.container_runtime_uid}. The right UR5 then lowers and releases
{roles.right_target_runtime_uid}, retreats upward, and returns to its initial
pose. To change the insertion order later, edit the task prompt sequence and
keep the same atomic action API.

The {roles.container_runtime_uid} area is a shared workspace. After a UR5
releases a target object, it should retreat upward before the other UR5 moves
to the container, otherwise the two arms may collide near the container. The
right UR5 should keep holding {roles.right_target_runtime_uid} while the left
UR5 performs its placement and upward retreat. Once that retreat is complete,
the right UR5 may move toward the container while the left UR5 simultaneously
returns to its initial pose; it must not wait for the left return-to-initial
motion to finish.

A target object at a high pose above `{roles.container_runtime_uid}` is only
staged, not placed. Each arm must lower the held object into the container
release pose and open the gripper before any return-to-initial motion.

Always plan to the current `{roles.container_runtime_uid}` object pose from the
environment config. Do not hard-code container coordinates in generated graph
actions.
"""


def _make_atom_actions_prompt(roles: _BasketTaskRoles) -> str:
    target_text = _generic_target_text(roles)
    return f"""### Atom Functions for UR5BreadBasket Dual-UR5 Placement

This is the Dual-UR5 configuration for the UR5BreadBasket task template. Use the
existing atomic action API and gripper semantics in this configuration.

Each atomic function returns a list of joint-space trajectories (`list[np.ndarray]`).
Use `robot_name="left_arm"` only for `{roles.left_target_runtime_uid}` and use
`robot_name="right_arm"` only for `{roles.right_target_runtime_uid}`.

The nominal task starts with simultaneous dual-arm grasping, followed by a
left-first placement with an overlapped handoff to the right arm:
- The first nominal edge must call `grasp` in both `left_arm_action` and
  `right_arm_action`.
- After both target objects are grasped, left-side placement steps put the
  actual placement action in `left_arm_action` and keep `right_arm_action`
  closed with `close_gripper(robot_name="right_arm", sample_num=20)` so the
  right arm keeps holding `{roles.right_target_runtime_uid}`.
- After the left arm releases `{roles.left_target_runtime_uid}`, first move it
  upward to clear the container.
- The next nominal edge must pair
  `back_to_initial_pose(robot_name="left_arm", sample_num=60)` in
  `left_arm_action` with the right arm's high-staging
  `move_relative_to_object` action in `right_arm_action`. Do not split this
  parallel handoff into separate edges.
- After the parallel handoff edge, the remaining right-side placement steps put
  the actual action in `right_arm_action` and set `left_arm_action` to null.
- `back_to_initial_pose` must never be used for an arm that is still holding a
  target object. The same arm must first execute `open_gripper` at the container
  release pose.

All atom functions are public-only wrappers backed by
`embodichain.lab.sim.atomic_actions.AtomicActionEngine`. Public action failures
raise immediately. Use only the parameters listed below.

Use the following functions exactly as defined. Do not invent new APIs or
parameters.

"grasp":
    def grasp(robot_name: str, obj_name: str, pre_grasp_dis: float, **kwargs) -> list[np.ndarray]

    Moves the selected UR5 to the target {target_text}, closes the gripper, and
    performs the public semantic post-grasp lift. Use:
    - `grasp(robot_name="left_arm", obj_name="{roles.left_target_runtime_uid}",
      pre_grasp_dis=0.08, sample_num=90)`
    - `grasp(robot_name="right_arm", obj_name="{roles.right_target_runtime_uid}",
      pre_grasp_dis=0.08, sample_num=90)`

"move_relative_to_object":
    def move_relative_to_object(robot_name: str,
                                obj_name: str,
                                x_offset=0.0,
                                y_offset=0.0,
                                z_offset=0.0,
                                **kwargs) -> list[np.ndarray]

    Moves the selected end-effector to a pose centered on a target object plus a
    world-frame xyz offset while preserving the current end-effector orientation.
    Use `obj_name="{roles.container_runtime_uid}"` twice for each placement:
    - First move to a high staging pose above the container:
      left arm uses `x_offset=0.0, y_offset=-0.04, z_offset=0.22`;
      right arm uses `x_offset=0.0, y_offset=0.04, z_offset=0.22`.
      The right-arm high-staging action must execute in the same graph edge as
      the left arm's `back_to_initial_pose` action.
    - Then lower to the release pose inside the container:
      left arm uses `x_offset=0.0, y_offset=-0.04, z_offset=0.12`;
      right arm uses `x_offset=0.0, y_offset=0.04, z_offset=0.12`.

"move_by_relative_offset":
    def move_by_relative_offset(robot_name: str,
                                dx=0.0,
                                dy=0.0,
                                dz=0.0,
                                mode="extrinsic",
                                **kwargs) -> list[np.ndarray]

    Moves the selected end-effector by a relative translation. The offset is
    applied along world axes when `mode="extrinsic"`. Use this for the short
    upward retreat after releasing a target object, before returning the arm to
    its initial pose.

"open_gripper":
    def open_gripper(robot_name: str, **kwargs) -> list[np.ndarray]

    Opens the selected gripper to release the held target object. For container
    release steps, always call this at the low release pose before any upward
    retreat or return-to-initial action:
    - `open_gripper(robot_name="left_arm", sample_num=30,
      open_threshold=-1.0, settle_steps=50)`
    - `open_gripper(robot_name="right_arm", sample_num=30,
      open_threshold=-1.0, settle_steps=50)`
    The negative `open_threshold` forces the open trajectory to be generated
    even if a stale cached gripper state incorrectly looks open. `settle_steps`
    keeps the gripper open and stationary after release so the object can fall
    out before the arm retreats.

"close_gripper":
    def close_gripper(robot_name: str, **kwargs) -> list[np.ndarray]

    Closes the selected gripper. Do not use this separately in the nominal
    graph, because `grasp` already closes the gripper.

"back_to_initial_pose":
    def back_to_initial_pose(robot_name: str, **kwargs) -> list[np.ndarray]

    Returns the selected UR5 arm to its initial joint configuration. In the
    nominal graph, use this after the selected arm releases and retreats upward.
    The left arm's return-to-initial action must execute concurrently with the
    right arm's high-staging action. Do not use this immediately after
    `move_relative_to_object`; the arm would still be holding the target object.
"""


def _validate_bundle(bundle: Mapping[str, Any], roles: _BasketTaskRoles) -> None:
    gym_config = bundle["gym_config"]
    if gym_config.get("id") != "AtomicActionsAgent-v3":
        raise ValueError("Generated gym config must use AtomicActionsAgent-v3.")
    if gym_config.get("robot", {}).get("uid") != "DualUR5":
        raise ValueError("Generated UR5 basket config must use DualUR5.")

    rigid_uids = {obj["uid"] for obj in gym_config.get("rigid_object", [])}
    required = {
        roles.left_target_runtime_uid,
        roles.right_target_runtime_uid,
        roles.container_runtime_uid,
    }
    if not required.issubset(rigid_uids):
        raise ValueError(
            f"Generated rigid objects missing: {sorted(required - rigid_uids)}"
        )

    success = gym_config["env"]["extensions"]["agent_success"]
    for term in success.get("terms", []):
        if (
            term.get("object") not in rigid_uids
            or term.get("container") not in rigid_uids
        ):
            raise ValueError(f"Invalid success term uid reference: {term}")


def _validate_relative_bundle(
    bundle: Mapping[str, Any],
    spec: _RelativePlacementSpec,
) -> None:
    gym_config = bundle["gym_config"]
    if gym_config.get("id") != "AtomicActionsAgent-v3":
        raise ValueError("Generated gym config must use AtomicActionsAgent-v3.")
    if gym_config.get("robot", {}).get("uid") != "DualUR5":
        raise ValueError("Generated relative placement config must use DualUR5.")

    rigid_uids = [obj["uid"] for obj in gym_config.get("rigid_object", [])]
    if len(rigid_uids) != len(set(rigid_uids)):
        raise ValueError(f"Duplicate rigid object runtime uid(s): {rigid_uids}")
    required = {spec.moved_runtime_uid, spec.reference_runtime_uid}
    missing = required - set(rigid_uids)
    if missing:
        raise ValueError(
            f"Generated relative config missing rigid object(s): {missing}"
        )

    _validate_success_uids(
        gym_config["env"]["extensions"]["agent_success"],
        set(rigid_uids),
    )
    registry = gym_config["env"]["events"]["register_info_to_env"]["params"]["registry"]
    registered = {entry["entity_cfg"]["uid"] for entry in registry}
    if not required.issubset(registered):
        raise ValueError(
            f"Relative config registry missing: {sorted(required - registered)}"
        )


def _validate_success_uids(success: Mapping[str, Any], rigid_uids: set[str]) -> None:
    if success.get("op") in {"all", "and", "any", "or"}:
        for term in success.get("terms", []):
            _validate_success_uids(term, rigid_uids)
        return

    success_type = str(success.get("type", success.get("func", ""))).lower()
    if success_type == "object_in_container":
        required_keys = ("object", "container")
    elif success_type in {"object_on_object", "object_on", "on_object"}:
        required_keys = ("object", "support")
    elif success_type in {
        "object_axis_offset_near",
        "object_relative_axis_near",
    }:
        required_keys = ("object", "reference")
    elif success_type in {"object_not_fallen", "not_fallen"}:
        required_keys = ("object",)
    else:
        raise ValueError(f"Unsupported generated success term: {success_type!r}.")

    for key in required_keys:
        uid = success.get(key)
        if uid not in rigid_uids:
            raise ValueError(f"Invalid success uid reference {key}={uid!r}.")


def _write_config_bundle(
    *,
    output_dir: Path,
    bundle: Mapping[str, Any],
    overwrite: bool,
) -> GeneratedUR5BasketConfigPaths:
    paths = GeneratedUR5BasketConfigPaths(
        output_dir=output_dir,
        gym_config=output_dir / "fast_gym_config.json",
        agent_config=output_dir / "agent_config.json",
        task_prompt=output_dir / "task_prompt.txt",
        basic_background=output_dir / "basic_background.txt",
        atom_actions=output_dir / "atom_actions.txt",
        summary=dict(bundle.get("summary", {})),
    )
    output_files = [
        paths.gym_config,
        paths.agent_config,
        paths.task_prompt,
        paths.basic_background,
        paths.atom_actions,
    ]
    existing = [path for path in output_files if path.exists()]
    if existing and not overwrite:
        existing_text = ", ".join(path.as_posix() for path in existing)
        raise FileExistsError(
            f"Generated file(s) already exist: {existing_text}. "
            "Pass overwrite=True or --overwrite to replace them."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(paths.gym_config, bundle["gym_config"])
    _write_json(paths.agent_config, bundle["agent_config"])
    _write_text(paths.task_prompt, bundle["task_prompt"])
    _write_text(paths.basic_background, bundle["basic_background"])
    _write_text(paths.atom_actions, bundle["atom_actions"])
    return paths


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=4) + "\n",
        encoding="utf-8",
    )


def _write_text(path: Path, content: str) -> None:
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _vector3(value: Any) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"Expected a 3-vector, got {value!r}.")
    return [float(item) for item in value]


def _clean_vector3(value: Any) -> list[float]:
    cleaned = []
    for item in _vector3(value):
        if abs(item - 1.0) < 1e-9:
            cleaned.append(1.0)
        elif abs(item) < 1e-12:
            cleaned.append(0.0)
        else:
            cleaned.append(item)
    return cleaned
