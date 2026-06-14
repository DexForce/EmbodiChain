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
import math
import re
import struct

from embodichain.gen_sim.action_agent_pipeline.generation.mesh_frame_normalization import (
    MeshFrameNormalizer,
)
from embodichain.gen_sim.action_agent_pipeline.generation.prompt_builders import (
    make_agent_config,
    make_basket_atom_actions_prompt,
    make_basket_basic_background,
    make_basket_task_prompt,
    make_relative_atom_actions_prompt,
    make_relative_basic_background,
    make_relative_task_prompt,
)

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
_TARGET_REPLACEMENT_MANIFEST_FILENAME = ".embodichain_replacement_manifest.json"

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
_ON_RELEASE_Z_OFFSET = 0.2
_DUAL_UR5_LEGACY_INIT_Z = 0.5
_DUAL_UR5_HIGH_TABLETOP_THRESHOLD = 1.0
_DUAL_UR5_HIGH_TABLETOP_INIT_Z = 0.8
_DUAL_UR5_ARM_COMPONENT_Z = 0.4
_DUAL_UR5_TABLETOP_CLEARANCE = 0.25
_DUAL_UR5_SIDE_AXIS_INDEX = 1
_BACKGROUND_MAX_CONVEX_HULL_NUM = 1
_TARGET_MAX_CONVEX_HULL_NUM = 16
_CONTAINER_MAX_CONVEX_HULL_NUM = 8
_EXTRA_RIGID_MAX_CONVEX_HULL_NUM = 1
_TABLETOP_OBJECT_CLEARANCE = 0.003
_GLB_JSON_CHUNK_TYPE = 0x4E4F534A
_GLB_BINARY_CHUNK_TYPE = 0x004E4942
_GLTF_COMPONENT_FORMATS = {
    5120: ("b", 1),
    5121: ("B", 1),
    5122: ("h", 2),
    5123: ("H", 2),
    5125: ("I", 4),
    5126: ("f", 4),
}
_GLTF_TYPE_COMPONENT_COUNTS = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT4": 16,
}

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
    reused: bool = False


@dataclass(frozen=True)
class _RelativePlacementStepSpec:
    moved_source_uid: str
    reference_source_uid: str
    moved_runtime_uid: str
    reference_runtime_uid: str
    relation: str
    active_side: str
    release_offset: list[float]
    high_offset: list[float]


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
    placements: tuple[_RelativePlacementStepSpec, ...]


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
    reuse_target_replacements: bool = True,
    prewarm_coacd_cache: bool = True,
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
        reuse_target_replacements: If true, reuse an existing replacement GLB
            at the expected output path when it matches the requested prompt.
        prewarm_coacd_cache: If true, precompute environment-side CoACD cache
            files referenced by the generated gym config before writing it.
        overwrite: If false, fail when generated files already exist.
        max_episodes: Value written to ``fast_gym_config.json``.
        max_episode_steps: Value written to ``fast_gym_config.json``.

    Returns:
        Paths of generated config files.
    """

    output_dir_path = Path(output_dir).expanduser().resolve()
    _raise_if_generated_files_exist(output_dir_path, overwrite)

    input_path = Path(gym_project).expanduser().resolve()
    gym_config_path = _resolve_gym_config_path(input_path)
    scene_dir = gym_config_path.parent
    source_config = _read_json(gym_config_path)
    project_name = _infer_project_name(input_path, scene_dir)
    replacement_specs = _normalize_target_replacements(target_replacements)
    mesh_normalizer = MeshFrameNormalizer(
        output_dir=output_dir_path / "mesh_assets" / "normalized"
    )

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
            mesh_normalizer=mesh_normalizer,
        )
        _validate_relative_bundle(bundle, spec)
        _attach_mesh_normalization_summary(bundle, mesh_normalizer)
        if prewarm_coacd_cache:
            _attach_coacd_cache_summary(bundle)
        return _write_config_bundle(
            output_dir=output_dir_path,
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
        reuse_target_replacements=reuse_target_replacements,
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
        mesh_normalizer=mesh_normalizer,
    )
    _validate_bundle(bundle, roles)
    _attach_mesh_normalization_summary(bundle, mesh_normalizer)
    if prewarm_coacd_cache:
        _attach_coacd_cache_summary(bundle)
    return _write_config_bundle(
        output_dir=output_dir_path,
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
    reuse_target_replacements: bool,
) -> tuple[_ResolvedTargetReplacement, ...]:
    resolved = []
    for replacement in replacement_specs:
        runtime_noun = _replacement_runtime_noun(replacement.prompt)
        output_root = scene_dir / "mesh_assets" / replacement.output_dir_name
        output_name = f"{runtime_noun}.glb"
        mesh_path = None
        reused = False
        if reuse_target_replacements:
            mesh_path = _resolve_reusable_target_replacement_mesh_path(
                output_root=output_root,
                prompt=replacement.prompt,
                output_name=output_name,
            )
            reused = mesh_path is not None
        if mesh_path is None:
            result = _run_prompt2geometry_replacement(
                prompt=replacement.prompt,
                output_root=output_root,
                output_name=output_name,
            )
            mesh_path = _resolve_prompt2geometry_mesh_path(result, output_root)
            _write_target_replacement_manifest(
                output_root=output_root,
                prompt=replacement.prompt,
                output_name=output_name,
                mesh_path=mesh_path,
            )
        elif reused:
            _write_target_replacement_manifest(
                output_root=output_root,
                prompt=replacement.prompt,
                output_name=output_name,
                mesh_path=mesh_path,
            )
        resolved.append(
            _ResolvedTargetReplacement(
                source_uid=replacement.source_uid,
                prompt=replacement.prompt,
                output_dir_name=replacement.output_dir_name,
                mesh_path=mesh_path,
                runtime_noun=runtime_noun,
                reused=reused,
            )
        )
    return tuple(resolved)


def _resolve_reusable_target_replacement_mesh_path(
    *,
    output_root: Path,
    prompt: str,
    output_name: str,
) -> Path | None:
    expected_mesh_path = (output_root / output_name).expanduser().resolve()
    if not expected_mesh_path.is_file():
        return None

    manifest_path = _target_replacement_manifest_path(output_root)
    if not manifest_path.is_file():
        return expected_mesh_path

    try:
        manifest = _read_json(manifest_path)
    except (OSError, json.JSONDecodeError):
        return None

    if manifest.get("prompt") != prompt or manifest.get("output_name") != output_name:
        return None

    manifest_mesh_path = Path(
        str(manifest.get("mesh_path", expected_mesh_path))
    ).expanduser()
    if not manifest_mesh_path.is_absolute():
        manifest_mesh_path = (output_root / manifest_mesh_path).resolve()
    else:
        manifest_mesh_path = manifest_mesh_path.resolve()
    if manifest_mesh_path.is_file():
        return manifest_mesh_path
    return expected_mesh_path


def _write_target_replacement_manifest(
    *,
    output_root: Path,
    prompt: str,
    output_name: str,
    mesh_path: Path,
) -> None:
    _write_json(
        _target_replacement_manifest_path(output_root),
        {
            "prompt": prompt,
            "output_name": output_name,
            "mesh_path": mesh_path.expanduser().resolve().as_posix(),
        },
    )


def _target_replacement_manifest_path(output_root: Path) -> Path:
    return output_root / _TARGET_REPLACEMENT_MANIFEST_FILENAME


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

    from embodichain.gen_sim.action_agent_pipeline.utils.llm_json import (
        extract_json_object,
    )
    from embodichain.gen_sim.action_agent_pipeline.utils.mllm import (
        create_chat_openai,
    )

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
    llm = create_chat_openai(
        temperature=0.0,
        model=model,
        usage_stage="config_generation.role_refinement",
    )
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

    from embodichain.gen_sim.action_agent_pipeline.utils.llm_json import (
        extract_json_object,
    )
    from embodichain.gen_sim.action_agent_pipeline.utils.mllm import (
        create_chat_openai,
    )

    prompt = (
        "Parse a simple Dual-UR5 tabletop relative-placement task and produce "
        "a constrained config-level JSON spec. This JSON is used to generate "
        "task_prompt.txt, basic_background.txt, atom_actions.txt, and "
        "agent_success; a second LLM will later read those prompts to generate "
        "the executable graph JSON.\n\n"
        "Return exactly one JSON object with this schema:\n"
        "{\n"
        '  "placements": [\n'
        "    {\n"
        '      "moved_object": "<source_uid from rigid_object>",\n'
        '      "reference_object": "<different source_uid from rigid_object>",\n'
        '      "goal_relation": '
        '"inside|on|left_of|right_of|front_of|behind",\n'
        '      "arm": "left|right|auto"\n'
        "    }\n"
        "  ],\n"
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
        "- Return one placement for a single-arm task and exactly two placements "
        "for a dual-arm task.\n"
        "- Treat the task as dual-arm when it explicitly says 双臂, 两臂, both "
        "arms, two arms, or when it describes separate work for the left arm and "
        "the right arm even if it does not literally say 双臂.\n"
        "- Do not invent a second placement when the task only moves one object.\n"
        "- moved_object is the object to grasp and move.\n"
        "- reference_object is the object used as the spatial reference, "
        "container, or support.\n"
        "- Within each placement, moved_object and reference_object must be "
        "different.\n"
        "- For dual-arm tasks, the placements must use two different moved_object "
        "values and one left arm plus one right arm. Use arm='auto' only when "
        "the user did not specify which arm handles that placement.\n"
        "- arm selects the single UR5 arm that should manipulate moved_object. "
        "Use arm='left' for explicit left-arm instructions such as 左臂, 左机械臂, "
        "left arm, or left UR5; use arm='right' for explicit right-arm "
        "instructions such as 右臂, 右机械臂, right arm, or right UR5; use "
        "arm='auto' when the task does not specify an arm.\n"
        "- For Chinese/English left/right/front/back, use the relation enums. "
        "front_of means negative world-x; behind means positive world-x; "
        "left_of means negative world-y; right_of means positive world-y.\n"
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
    llm = create_chat_openai(
        temperature=0.0,
        model=model,
        usage_stage="config_generation.relative_task",
    )
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
    by_uid = {obj.source_uid: obj for obj in rigid_objects}
    runtime_uids = _relative_runtime_uid_mapping(rigid_objects)

    placement_entries = _relative_placement_entries(response)
    if len(placement_entries) > 2:
        raise ValueError("Relative placement supports at most two arm placements.")

    forced_arm_sides = _relative_forced_arm_sides(
        placement_entries,
        by_uid=by_uid,
        rigid_objects=rigid_objects,
    )
    placements = tuple(
        _build_relative_placement_step(
            entry=entry,
            by_uid=by_uid,
            rigid_objects=rigid_objects,
            runtime_uids=runtime_uids,
            forced_side=forced_side,
        )
        for entry, forced_side in zip(placement_entries, forced_arm_sides)
    )
    _validate_relative_placements(placements)

    summary = str(response.get("task_prompt_summary", "")).strip()
    if not summary:
        summary = _default_relative_plan_summary(placements)
    background_notes = str(response.get("basic_background_notes", "")).strip()
    action_sketch = _string_list(response.get("action_sketch"))
    if not action_sketch:
        action_sketch = _default_relative_action_sketch(placements)

    primary = placements[0]

    return _RelativePlacementSpec(
        table_source_uid=table_source_uid,
        moved_source_uid=primary.moved_source_uid,
        reference_source_uid=primary.reference_source_uid,
        moved_runtime_uid=primary.moved_runtime_uid,
        reference_runtime_uid=primary.reference_runtime_uid,
        relation=primary.relation,
        active_side=primary.active_side,
        task_description=task_description,
        task_prompt_summary=summary,
        basic_background_notes=background_notes,
        action_sketch=action_sketch,
        release_offset=primary.release_offset,
        high_offset=primary.high_offset,
        placements=placements,
    )


def _relative_placement_entries(response: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    placements = response.get("placements")
    if placements is None:
        return [response]
    if not isinstance(placements, list) or not placements:
        raise ValueError("LLM response placements must be a non-empty list.")
    entries: list[Mapping[str, Any]] = []
    for index, placement in enumerate(placements):
        if not isinstance(placement, Mapping):
            raise ValueError(f"Placement {index} must be a JSON object.")
        entries.append(placement)
    return entries


def _relative_forced_arm_sides(
    placement_entries: list[Mapping[str, Any]],
    *,
    by_uid: Mapping[str, _SceneObject],
    rigid_objects: list[_SceneObject],
) -> list[str | None]:
    if len(placement_entries) != 2:
        return [None for _ in placement_entries]

    requested_sides = [
        _normalize_relative_arm(entry.get("arm")) for entry in placement_entries
    ]
    explicit_sides = [side for side in requested_sides if side != "auto"]
    if len(explicit_sides) == 2:
        return [None, None]
    if len(explicit_sides) == 1:
        complement = "right" if explicit_sides[0] == "left" else "left"
        return [
            requested_side if requested_side != "auto" else complement
            for requested_side in requested_sides
        ]

    moved_source_uids = [
        _resolve_rigid_source_uid(
            entry.get("moved_object"),
            rigid_objects,
            field_name="moved_object",
        )
        for entry in placement_entries
    ]
    positions = [
        _vector3(by_uid[source_uid].config.get("init_pos", [0.0, 0.0, 0.0]))
        for source_uid in moved_source_uids
    ]
    inferred_sides = [_arm_side_for_position(position) for position in positions]
    if set(inferred_sides) == {"left", "right"}:
        return inferred_sides

    side_values = [_position_side_axis_value(position) for position in positions]
    if side_values[0] <= side_values[1]:
        return ["left", "right"]
    return ["right", "left"]


def _build_relative_placement_step(
    *,
    entry: Mapping[str, Any],
    by_uid: Mapping[str, _SceneObject],
    rigid_objects: list[_SceneObject],
    runtime_uids: Mapping[str, str],
    forced_side: str | None,
) -> _RelativePlacementStepSpec:
    moved_source_uid = _resolve_rigid_source_uid(
        entry.get("moved_object"),
        rigid_objects,
        field_name="moved_object",
    )
    reference_source_uid = _resolve_rigid_source_uid(
        entry.get("reference_object"),
        rigid_objects,
        field_name="reference_object",
    )
    if moved_source_uid == reference_source_uid:
        raise ValueError(
            "Relative placement requires distinct moved/reference objects."
        )

    reference_obj = by_uid[reference_source_uid]
    relation = _normalize_relative_relation(entry.get("goal_relation"))
    if relation == "on" and _is_container_like(reference_obj):
        relation = "inside"

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
    requested_side = _normalize_relative_arm(entry.get("arm"))
    active_side = (
        forced_side
        if forced_side is not None
        else (
            _arm_side_for_position(moved_position)
            if requested_side == "auto"
            else requested_side
        )
    )

    return _RelativePlacementStepSpec(
        moved_source_uid=moved_source_uid,
        reference_source_uid=reference_source_uid,
        moved_runtime_uid=moved_runtime_uid,
        reference_runtime_uid=reference_runtime_uid,
        relation=relation,
        active_side=active_side,
        release_offset=release_offset,
        high_offset=high_offset,
    )


def _validate_relative_placements(
    placements: tuple[_RelativePlacementStepSpec, ...],
) -> None:
    if not placements:
        raise ValueError("Relative placement requires at least one placement.")
    moved_source_uids = [placement.moved_source_uid for placement in placements]
    if len(moved_source_uids) != len(set(moved_source_uids)):
        raise ValueError("Relative placements must use distinct moved_object values.")
    if len(placements) == 2:
        active_sides = {placement.active_side for placement in placements}
        if active_sides != {"left", "right"}:
            raise ValueError(
                "Dual-arm relative placement requires one left arm and one right arm."
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


def _normalize_relative_arm(value: Any) -> str:
    if value is None:
        return "auto"
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if text in {
        "",
        "auto",
        "automatic",
        "unspecified",
        "none",
        "null",
        "default",
        "自动",
        "默认",
        "未指定",
        "不指定",
    }:
        return "auto"
    if text in {
        "left",
        "left_arm",
        "left_ur5",
        "左",
        "左臂",
        "左机械臂",
        "左手",
        "左手臂",
    }:
        return "left"
    if text in {
        "right",
        "right_arm",
        "right_ur5",
        "右",
        "右臂",
        "右机械臂",
        "右手",
        "右手臂",
    }:
        return "right"
    raise ValueError(
        f"Unsupported relative placement arm {value!r}; expected 'left', "
        "'right', or 'auto'."
    )


def _relative_release_offset(relation: str) -> list[float]:
    relation = _normalize_relative_relation(relation)
    if relation == "inside":
        return [0.0, 0.0, _SIDE_RELEASE_Z_OFFSET]
    if relation == "on":
        return [0.0, 0.0, _ON_RELEASE_Z_OFFSET]
    if relation == "left_of":
        return [0.0, -_SIDE_RELATION_DISTANCE, _SIDE_RELEASE_Z_OFFSET]
    if relation == "right_of":
        return [0.0, _SIDE_RELATION_DISTANCE, _SIDE_RELEASE_Z_OFFSET]
    if relation == "front_of":
        return [-_SIDE_RELATION_DISTANCE, 0.0, _SIDE_RELEASE_Z_OFFSET]
    if relation == "behind":
        return [_SIDE_RELATION_DISTANCE, 0.0, _SIDE_RELEASE_Z_OFFSET]
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


def _default_relative_plan_summary(
    placements: Sequence[_RelativePlacementStepSpec],
) -> str:
    if len(placements) == 1:
        placement = placements[0]
        return _default_relative_task_summary(
            placement.moved_runtime_uid,
            placement.reference_runtime_uid,
            placement.relation,
        )
    placement_text = "; ".join(
        f"use the {placement.active_side} UR5 to move "
        f"`{placement.moved_runtime_uid}` "
        f"{_relative_relation_phrase(placement.relation)} "
        f"`{placement.reference_runtime_uid}`"
        for placement in placements
    )
    return f"Use both UR5 arms for a dual-arm relative placement: {placement_text}."


def _default_relative_action_sketch(
    placements: Sequence[_RelativePlacementStepSpec],
) -> list[str]:
    if len(placements) == 1:
        placement = placements[0]
        return [
            f"grasp {placement.moved_runtime_uid}",
            (
                f"move above the {placement.relation} release pose relative to "
                f"{placement.reference_runtime_uid}"
            ),
            "lower to the release pose",
            "open the gripper",
            "retreat upward",
        ]
    sketch = ["grasp both moved objects with their assigned arms"]
    for placement in placements:
        sketch.extend(
            [
                (
                    f"use {placement.active_side}_arm to move "
                    f"{placement.moved_runtime_uid} above the release pose relative "
                    f"to {placement.reference_runtime_uid}"
                ),
                f"lower and release {placement.moved_runtime_uid}",
                f"retreat {placement.active_side}_arm upward",
            ]
        )
    return sketch


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
    mesh_normalizer: MeshFrameNormalizer,
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
    table_config = _make_background_config(
        scene_dir,
        by_uid[roles.table_source_uid],
        mesh_normalizer,
    )
    table_top_z = _mesh_config_world_zmax(table_config)
    robot_init_z = _dual_ur5_init_z_from_table_top(table_top_z)

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
            table_config,
            _make_container_background_config(
                scene_dir,
                by_uid[roles.container_source_uid],
                roles.container_runtime_uid,
                container_scale,
                mesh_normalizer,
            ),
            *[
                _make_extra_background_config(scene_dir, obj, mesh_normalizer)
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
                mesh_normalizer,
            ),
            _make_target_object_config(
                scene_dir,
                by_uid[roles.left_target_source_uid],
                roles.left_target_runtime_uid,
                object_scale,
                replacement_by_source_uid.get(roles.left_target_source_uid),
                mesh_normalizer,
            ),
            *[
                _make_extra_rigid_object_config(
                    scene_dir,
                    obj,
                    _source_body_scale(obj),
                    mesh_normalizer,
                )
                for obj in extra_rigid_objects
            ],
        ],
    }
    _apply_tabletop_z_placement(gym_config, table_top_z)
    return {
        "gym_config": gym_config,
        "agent_config": make_agent_config(),
        "task_prompt": make_basket_task_prompt(task_name, project_name, roles),
        "basic_background": make_basket_basic_background(project_name, roles),
        "atom_actions": make_basket_atom_actions_prompt(roles),
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
                    "reused": replacement.reused,
                }
                for replacement in target_replacements
            ],
        },
    }


def _attach_coacd_cache_summary(bundle: dict[str, Any]) -> None:
    from embodichain.gen_sim.action_agent_pipeline.generation.coacd_cache import (
        prewarm_coacd_cache_for_gym_config,
    )

    bundle.setdefault("summary", {})["coacd_cache"] = (
        prewarm_coacd_cache_for_gym_config(bundle["gym_config"])
    )


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
    static_reference_source_uids = _static_relative_reference_source_uids(
        spec,
        by_uid,
    )
    dynamic_rigid_objects = [
        obj for obj in rigid_objects if obj.source_uid not in static_reference_source_uids
    ]
    static_reference_objects = [
        obj for obj in rigid_objects if obj.source_uid in static_reference_source_uids
    ]
    object_scale = _target_body_scale_vector(target_body_scale)
    robot_init_z = _estimate_dual_ur5_init_z(
        scene_dir,
        by_uid[spec.table_source_uid],
    )

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
                _make_container_background_config(
                    scene_dir,
                    obj,
                    runtime_uids[obj.source_uid],
                    _relative_object_body_scale(obj, target_scale=object_scale),
                )
                for obj in static_reference_objects
            ],
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
            for obj in dynamic_rigid_objects
        ],
    }
    return {
        "gym_config": gym_config,
        "agent_config": make_agent_config(),
        "task_prompt": make_relative_task_prompt(task_name, project_name, spec),
        "basic_background": make_relative_basic_background(project_name, spec),
        "atom_actions": make_relative_atom_actions_prompt(spec),
        "summary": _make_relative_summary(spec),
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


def _static_relative_reference_source_uids(
    spec: _RelativePlacementSpec,
    by_uid: Mapping[str, _SceneObject],
) -> set[str]:
    moved_source_uids = {placement.moved_source_uid for placement in spec.placements}
    return {
        placement.reference_source_uid
        for placement in spec.placements
        if placement.reference_source_uid not in moved_source_uids
        and _is_container_like(by_uid[placement.reference_source_uid])
    }


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


def _make_relative_summary(spec: _RelativePlacementSpec) -> dict[str, Any]:
    if len(spec.placements) == 1:
        return {
            "mode": "relative_placement",
            "moved_object": spec.moved_runtime_uid,
            "reference_object": spec.reference_runtime_uid,
            "relation": spec.relation,
            "active_arm": f"{spec.active_side}_arm",
            "release_offset": spec.release_offset,
        }
    return {
        "mode": "dual_arm_relative_placement",
        "placements": [
            {
                "moved_object": placement.moved_runtime_uid,
                "reference_object": placement.reference_runtime_uid,
                "relation": placement.relation,
                "active_arm": f"{placement.active_side}_arm",
                "release_offset": placement.release_offset,
            }
            for placement in spec.placements
        ],
    }


def _estimate_dual_ur5_init_z(scene_dir: Path, table_obj: _SceneObject) -> float:
    """Estimate robot root height from the table mesh top surface."""

    table_top_z = _resolve_table_mesh_world_zmax(scene_dir, table_obj)
    if table_top_z is None:
        return _DUAL_UR5_LEGACY_INIT_Z

    init_z = table_top_z + _DUAL_UR5_TABLETOP_CLEARANCE - _DUAL_UR5_ARM_COMPONENT_Z
    return round(max(_DUAL_UR5_LEGACY_INIT_Z, init_z), 6)


def _resolve_table_mesh_world_zmax(
    scene_dir: Path,
    table_obj: _SceneObject,
) -> float | None:
    shape = table_obj.config.get("shape", {})
    if not isinstance(shape, Mapping):
        return None
    if shape.get("shape_type") != "Mesh" or not shape.get("fpath"):
        return None

    mesh_path = _source_asset_path(scene_dir, str(shape["fpath"]))
    try:
        vertices = _load_mesh_vertices(mesh_path)
    except (
        OSError,
        ValueError,
        json.JSONDecodeError,
        UnicodeDecodeError,
        struct.error,
    ):
        return None
    if not vertices:
        return None

    world_matrix = _table_mesh_world_matrix(table_obj.config)
    return max(_transform_point(world_matrix, vertex)[2] for vertex in vertices)


def _source_asset_path(scene_dir: Path, fpath: str) -> Path:
    raw_path = Path(fpath)
    if raw_path.is_absolute():
        return raw_path.resolve()

    scene_candidate = (scene_dir / raw_path).resolve()
    if scene_candidate.exists():
        return scene_candidate

    repo_candidate = (_repo_root() / raw_path).resolve()
    if repo_candidate.exists():
        return repo_candidate
    return scene_candidate


def _load_mesh_vertices(mesh_path: Path) -> list[tuple[float, float, float]] | None:
    if mesh_path.suffix.lower() == ".glb":
        try:
            return list(_iter_glb_world_position_vertices(mesh_path))
        except (
            OSError,
            ValueError,
            json.JSONDecodeError,
            UnicodeDecodeError,
            struct.error,
        ):
            return _load_mesh_vertices_with_trimesh(mesh_path)
    return _load_mesh_vertices_with_trimesh(mesh_path)


def _load_mesh_vertices_with_trimesh(
    mesh_path: Path,
) -> list[tuple[float, float, float]] | None:
    try:
        import trimesh
    except ImportError:
        return None

    try:
        scene_or_mesh = trimesh.load(str(mesh_path), force="scene")
        try:
            mesh = scene_or_mesh.dump(concatenate=True)
        except AttributeError:
            mesh = scene_or_mesh
    except Exception:
        return None
    vertices = getattr(mesh, "vertices", None)
    if vertices is None or len(vertices) == 0:
        return None
    return [
        (float(vertex[0]), float(vertex[1]), float(vertex[2])) for vertex in vertices
    ]


def _iter_glb_world_position_vertices(
    mesh_path: Path,
):
    doc, binary_chunk = _read_glb(mesh_path)
    nodes = doc.get("nodes", [])
    if not isinstance(nodes, list):
        raise ValueError("GLB nodes must be a list.")

    scenes = doc.get("scenes", [])
    if scenes:
        scene_index = int(doc.get("scene", 0))
        root_node_ids = scenes[scene_index].get("nodes", [])
    else:
        root_node_ids = list(range(len(nodes)))

    stack = [(int(node_id), _identity_matrix4()) for node_id in root_node_ids]
    while stack:
        node_id, parent_matrix = stack.pop()
        node = nodes[node_id]
        node_matrix = _matrix_multiply(parent_matrix, _gltf_node_matrix(node))
        mesh_index = node.get("mesh")
        if mesh_index is not None:
            for vertex in _iter_gltf_mesh_position_vertices(
                doc,
                binary_chunk,
                int(mesh_index),
            ):
                yield _transform_point(node_matrix, vertex)
        for child_id in node.get("children", []) or []:
            stack.append((int(child_id), node_matrix))


def _read_glb(mesh_path: Path) -> tuple[dict[str, Any], bytes]:
    data = mesh_path.read_bytes()
    if len(data) < 20:
        raise ValueError("GLB file is too small.")

    magic, version, total_length = struct.unpack_from("<4sII", data, 0)
    if magic != b"glTF" or version != 2:
        raise ValueError("Only GLB version 2 files are supported.")
    if total_length > len(data):
        raise ValueError("GLB length header exceeds file size.")

    doc: dict[str, Any] | None = None
    binary_chunk = b""
    offset = 12
    while offset + 8 <= total_length:
        chunk_length, chunk_type = struct.unpack_from("<II", data, offset)
        offset += 8
        chunk_end = offset + chunk_length
        if chunk_end > total_length:
            raise ValueError("GLB chunk exceeds file size.")
        chunk = data[offset:chunk_end]
        offset = chunk_end
        if chunk_type == _GLB_JSON_CHUNK_TYPE:
            doc = json.loads(chunk.decode("utf-8").rstrip("\x00 "))
        elif chunk_type == _GLB_BINARY_CHUNK_TYPE:
            binary_chunk = chunk

    if doc is None:
        raise ValueError("GLB file does not contain a JSON chunk.")
    return doc, binary_chunk


def _iter_gltf_mesh_position_vertices(
    doc: Mapping[str, Any],
    binary_chunk: bytes,
    mesh_index: int,
):
    meshes = doc.get("meshes", [])
    accessors = doc.get("accessors", [])
    mesh = meshes[mesh_index]
    for primitive in mesh.get("primitives", []) or []:
        attributes = primitive.get("attributes", {})
        position_accessor = attributes.get("POSITION")
        if position_accessor is None:
            continue
        if int(position_accessor) >= len(accessors):
            raise ValueError("POSITION accessor index is out of range.")
        yield from _iter_gltf_accessor_vec3(doc, binary_chunk, int(position_accessor))


def _iter_gltf_accessor_vec3(
    doc: Mapping[str, Any],
    binary_chunk: bytes,
    accessor_index: int,
):
    accessor = doc["accessors"][accessor_index]
    if accessor.get("sparse"):
        raise ValueError("Sparse GLB accessors are not supported.")
    if accessor.get("type") != "VEC3":
        raise ValueError("POSITION accessor must be VEC3.")
    if "bufferView" not in accessor:
        raise ValueError("POSITION accessor must reference a bufferView.")

    component_type = int(accessor["componentType"])
    if component_type not in _GLTF_COMPONENT_FORMATS:
        raise ValueError(f"Unsupported GLB component type: {component_type}.")
    component_format, component_size = _GLTF_COMPONENT_FORMATS[component_type]
    component_count = _GLTF_TYPE_COMPONENT_COUNTS[accessor["type"]]
    buffer_view = doc["bufferViews"][int(accessor["bufferView"])]
    if int(buffer_view.get("buffer", 0)) != 0:
        raise ValueError("Only GLB embedded binary buffers are supported.")

    stride = int(buffer_view.get("byteStride", component_size * component_count))
    offset = int(buffer_view.get("byteOffset", 0)) + int(accessor.get("byteOffset", 0))
    element_format = "<" + component_format * component_count
    for index in range(int(accessor["count"])):
        values = struct.unpack_from(
            element_format,
            binary_chunk,
            offset + index * stride,
        )
        yield (float(values[0]), float(values[1]), float(values[2]))


def _table_mesh_world_matrix(table_config: Mapping[str, Any]) -> list[list[float]]:
    scale = _vector3(table_config.get("body_scale", [1.0, 1.0, 1.0]))
    init_local_pose = table_config.get("init_local_pose")
    if init_local_pose is not None:
        root_matrix = _matrix4(init_local_pose)
    else:
        root_matrix = _euler_xyz_degrees_matrix(
            _vector3(table_config.get("init_rot", [0.0, 0.0, 0.0])),
            _vector3(table_config.get("init_pos", [0.0, 0.0, 0.0])),
        )
    return _matrix_multiply(root_matrix, _scale_matrix4(scale))


def _gltf_node_matrix(node: Mapping[str, Any]) -> list[list[float]]:
    if "matrix" in node:
        values = [float(value) for value in node["matrix"]]
        if len(values) != 16:
            raise ValueError("GLB node matrix must contain 16 values.")
        return [[values[column * 4 + row] for column in range(4)] for row in range(4)]

    translation = [float(value) for value in node.get("translation", [0.0, 0.0, 0.0])]
    scale = [float(value) for value in node.get("scale", [1.0, 1.0, 1.0])]
    rotation = [float(value) for value in node.get("rotation", [0.0, 0.0, 0.0, 1.0])]
    if len(translation) != 3 or len(scale) != 3 or len(rotation) != 4:
        raise ValueError("Invalid GLB node TRS transform.")

    x, y, z, w = rotation
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    matrix = [
        [
            (1.0 - 2.0 * (yy + zz)) * scale[0],
            (2.0 * (xy - wz)) * scale[1],
            (2.0 * (xz + wy)) * scale[2],
            translation[0],
        ],
        [
            (2.0 * (xy + wz)) * scale[0],
            (1.0 - 2.0 * (xx + zz)) * scale[1],
            (2.0 * (yz - wx)) * scale[2],
            translation[1],
        ],
        [
            (2.0 * (xz - wy)) * scale[0],
            (2.0 * (yz + wx)) * scale[1],
            (1.0 - 2.0 * (xx + yy)) * scale[2],
            translation[2],
        ],
        [0.0, 0.0, 0.0, 1.0],
    ]
    return matrix


def _euler_xyz_degrees_matrix(
    rotation_deg: Sequence[float],
    translation: Sequence[float],
) -> list[list[float]]:
    rx, ry, rz = (math.radians(float(value)) for value in rotation_deg)
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    rot_x = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, cx, -sx, 0.0],
        [0.0, sx, cx, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    rot_y = [
        [cy, 0.0, sy, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-sy, 0.0, cy, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    rot_z = [
        [cz, -sz, 0.0, 0.0],
        [sz, cz, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    matrix = _matrix_multiply(_matrix_multiply(rot_z, rot_y), rot_x)
    matrix[0][3] = float(translation[0])
    matrix[1][3] = float(translation[1])
    matrix[2][3] = float(translation[2])
    return matrix


def _identity_matrix4() -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _scale_matrix4(scale: Sequence[float]) -> list[list[float]]:
    return [
        [float(scale[0]), 0.0, 0.0, 0.0],
        [0.0, float(scale[1]), 0.0, 0.0],
        [0.0, 0.0, float(scale[2]), 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _matrix4(value: Any) -> list[list[float]]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"Expected a 4x4 matrix, got {value!r}.")
    matrix = []
    for row in value:
        if not isinstance(row, (list, tuple)) or len(row) != 4:
            raise ValueError(f"Expected a 4x4 matrix, got {value!r}.")
        matrix.append([float(item) for item in row])
    return matrix


def _matrix_multiply(
    left: Sequence[Sequence[float]],
    right: Sequence[Sequence[float]],
) -> list[list[float]]:
    return [
        [
            sum(
                float(left[row][inner]) * float(right[inner][column])
                for inner in range(4)
            )
            for column in range(4)
        ]
        for row in range(4)
    ]


def _transform_point(
    matrix: Sequence[Sequence[float]],
    point: Sequence[float],
) -> tuple[float, float, float]:
    x, y, z = (float(point[0]), float(point[1]), float(point[2]))
    return (
        float(matrix[0][0]) * x
        + float(matrix[0][1]) * y
        + float(matrix[0][2]) * z
        + float(matrix[0][3]),
        float(matrix[1][0]) * x
        + float(matrix[1][1]) * y
        + float(matrix[1][2]) * z
        + float(matrix[1][3]),
        float(matrix[2][0]) * x
        + float(matrix[2][1]) * y
        + float(matrix[2][2]) * z
        + float(matrix[2][3]),
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
        "viewer_camera_uid": "cam_high",
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
        "viewer_camera_uid": "cam_high",
        "agent_success": _make_relative_success_spec(spec),
    }


def _make_relative_success_spec(spec: _RelativePlacementSpec) -> dict[str, Any]:
    if len(spec.placements) == 1:
        return _make_relative_placement_success_spec(spec.placements[0])
    return {
        "op": "all",
        "terms": [
            _make_relative_placement_success_spec(placement)
            for placement in spec.placements
        ],
    }


def _make_relative_placement_success_spec(
    placement: _RelativePlacementStepSpec,
) -> dict[str, Any]:
    if placement.relation == "inside":
        return _object_in_container_success(
            placement.moved_runtime_uid,
            placement.reference_runtime_uid,
        )
    if placement.relation == "on":
        return {
            "type": "object_on_object",
            "object": placement.moved_runtime_uid,
            "support": placement.reference_runtime_uid,
            "xy_radius": 0.08,
            "min_z_offset": 0.02,
            "max_z_offset": 0.35,
        }

    primary_axis, primary_offset, secondary_axis = _side_relation_axes(
        placement.relation
    )
    return {
        "op": "all",
        "terms": [
            {
                "type": "object_axis_offset_near",
                "object": placement.moved_runtime_uid,
                "reference": placement.reference_runtime_uid,
                "axis": primary_axis,
                "offset": primary_offset,
                "tolerance": 0.05,
            },
            {
                "type": "object_axis_offset_near",
                "object": placement.moved_runtime_uid,
                "reference": placement.reference_runtime_uid,
                "axis": secondary_axis,
                "offset": 0.0,
                "tolerance": 0.06,
            },
            {
                "type": "object_not_fallen",
                "object": placement.moved_runtime_uid,
                "max_tilt": 0.9,
            },
        ],
    }


def _side_relation_axes(relation: str) -> tuple[str, float, str]:
    if relation == "left_of":
        return "y", -_SIDE_RELATION_DISTANCE, "x"
    if relation == "right_of":
        return "y", _SIDE_RELATION_DISTANCE, "x"
    if relation == "front_of":
        return "x", -_SIDE_RELATION_DISTANCE, "y"
    if relation == "behind":
        return "x", _SIDE_RELATION_DISTANCE, "y"
    raise ValueError(f"Unsupported side relation: {relation!r}.")


def _make_relative_events_config(
    spec: _RelativePlacementSpec,
    rigid_runtime_uids: list[str],
) -> dict[str, Any]:
    return {
        "record_camera": _record_camera_event_config(),
        "validation_cameras": _validation_cameras_event_config(),
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
        "record_camera": _record_camera_event_config(),
        "validation_cameras": _validation_cameras_event_config(),
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


def _record_camera_event_config() -> dict[str, Any]:
    camera = _make_sensor_config()[0]
    extrinsics = camera["extrinsics"]
    return {
        "func": "record_camera_data",
        "mode": "interval",
        "interval_step": 1,
        "params": {
            "name": "record_cam_high",
            "resolution": [camera["width"], camera["height"]],
            "intrinsics": camera["intrinsics"],
            "eye": extrinsics["eye"],
            "target": extrinsics["target"],
            "up": extrinsics["up"],
        },
    }


def _validation_cameras_event_config() -> dict[str, Any]:
    return {
        "func": "validation_cameras",
        "mode": "trigger",
        "params": {},
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
                    "lang": _relative_dataset_instruction(spec),
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


def _relative_dataset_instruction(spec: _RelativePlacementSpec) -> str:
    if len(spec.placements) == 1:
        placement = spec.placements[0]
        return (
            f"Use the {placement.active_side} UR5 to move "
            f"{placement.moved_runtime_uid} "
            f"{_relative_relation_phrase(placement.relation)} "
            f"{placement.reference_runtime_uid}."
        )
    return " ".join(
        f"Use the {placement.active_side} UR5 to move "
        f"{placement.moved_runtime_uid} "
        f"{_relative_relation_phrase(placement.relation)} "
        f"{placement.reference_runtime_uid}."
        for placement in spec.placements
    )


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
            0,
            0,
            -1.57,
            -1.57,
            1.57,
            1.57,
            -1.57,
            -1.57,
            -1.57,
            -1.57,
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
            "intrinsics": [420, 420, 480, 270],
            "extrinsics": {
                "pos": [0.4, 0.0, 2.2],
                "eye": [0.6, 0.0, 3.3],
                "target": [0.0, 0.0, 0.75],
                "up": [1.0, 0.0, 0.0],
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
    shape = _make_shape_config(scene_dir, obj.config)
    return {
        "uid": "table",
        "shape": shape,
        "attrs": dict(_BACKGROUND_ATTRS),
        "body_scale": _clean_vector3(obj.config.get("body_scale", [1.0, 1.0, 1.0])),
        "body_type": "kinematic",
        "init_pos": _clean_vector3(obj.config.get("init_pos", [0.0, 0.0, 0.0])),
        "init_rot": _corrected_init_rot_for_shape(obj.config, shape),
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
    shape = _make_shape_config(scene_dir, obj.config)
    config = {
        "uid": _normalize_runtime_uid(obj.source_uid),
        "shape": shape,
        "attrs": copy.deepcopy(dict(obj.config.get("attrs", _BACKGROUND_ATTRS))),
        "body_scale": _clean_vector3(
            obj.config.get("body_scale", [1.0, 1.0, 1.0])
            if body_scale is None
            else body_scale
        ),
        "body_type": str(obj.config.get("body_type", "static")),
        "init_pos": _clean_vector3(obj.config.get("init_pos", [0.0, 0.0, 0.0])),
        "init_rot": _corrected_init_rot_for_shape(obj.config, shape),
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
        max_convex_hull_num=_TARGET_MAX_CONVEX_HULL_NUM,
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


def _make_container_background_config(
    scene_dir: Path,
    obj: _SceneObject,
    runtime_uid: str,
    body_scale: Any,
) -> dict[str, Any]:
    config = _make_container_object_config(
        scene_dir,
        obj,
        runtime_uid,
        body_scale,
    )
    config["body_type"] = "kinematic"
    config["init_rot"] = _corrected_init_rot_for_shape(obj.config, config["shape"])
    return config


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
    if max_convex_hull_num == _TARGET_MAX_CONVEX_HULL_NUM:
        resolved_max_convex_hull_num = max_convex_hull_num
    else:
        resolved_max_convex_hull_num = _role_limited_max_convex_hull_num(
            obj,
            max_convex_hull_num,
        )
    return _make_rigid_object_config(
        scene_dir,
        obj,
        runtime_uid,
        body_scale,
        max_convex_hull_num=resolved_max_convex_hull_num,
    )


def _make_rigid_object_config(
    scene_dir: Path,
    obj: _SceneObject,
    runtime_uid: str,
    body_scale: Any,
    max_convex_hull_num: int,
    mesh_fpath: str | Path | None = None,
) -> dict[str, Any]:
    shape = _make_shape_config(scene_dir, obj.config, mesh_fpath=mesh_fpath)
    config = {
        "uid": runtime_uid,
        "shape": shape,
        "attrs": dict(_RIGID_OBJECT_ATTRS),
        "init_pos": _clean_vector3(obj.config.get("init_pos", [0.0, 0.0, 0.0])),
        "init_rot": _corrected_init_rot_for_shape(obj.config, shape),
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
    for placement in spec.placements:
        if (
            placement.relation == "inside"
            and runtime_uid == placement.reference_runtime_uid
        ):
            return _CONTAINER_MAX_CONVEX_HULL_NUM
    task_uids = {
        uid
        for placement in spec.placements
        for uid in (placement.moved_runtime_uid, placement.reference_runtime_uid)
    }
    if runtime_uid in task_uids:
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


def _corrected_init_rot_for_shape(
    source_config: Mapping[str, Any],
    shape_config: Mapping[str, Any],
) -> list[float]:
    init_rot = _clean_vector3(source_config.get("init_rot", [0.0, 0.0, 0.0]))
    if not _is_glb_mesh_shape(shape_config):
        return init_rot

    from scipy.spatial.transform import Rotation

    source_rotation = Rotation.from_euler("XYZ", init_rot, degrees=True)
    correction = Rotation.from_euler(
        "X",
        _DEXSIM_041_GLB_LOCAL_X_CORRECTION_DEGREES,
        degrees=True,
    )
    corrected = source_rotation * correction
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Gimbal lock detected.*",
            category=UserWarning,
        )
        corrected_euler = corrected.as_euler("XYZ", degrees=True)
    return [float(value) for value in corrected_euler]


def _is_glb_mesh_shape(shape_config: Mapping[str, Any]) -> bool:
    if shape_config.get("shape_type") != "Mesh":
        return False
    fpath = shape_config.get("fpath")
    return isinstance(fpath, str) and Path(fpath).suffix.lower() == ".glb"


def _asset_path_for_config(scene_dir: Path, fpath: str) -> str:
    raw_path = Path(fpath)
    if raw_path.is_absolute():
        return raw_path.resolve().as_posix()
    return (scene_dir / raw_path).resolve().as_posix()


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "setup.py").exists() and (parent / "embodichain").exists():
            return parent
    return Path.cwd().resolve()


def _validate_bundle(bundle: Mapping[str, Any], roles: _BasketTaskRoles) -> None:
    gym_config = bundle["gym_config"]
    if gym_config.get("id") != "AtomicActionsAgent-v3":
        raise ValueError("Generated gym config must use AtomicActionsAgent-v3.")
    if gym_config.get("robot", {}).get("uid") != "DualUR5":
        raise ValueError("Generated UR5 basket config must use DualUR5.")

    rigid_uids = {obj["uid"] for obj in gym_config.get("rigid_object", [])}
    background_uids = {obj["uid"] for obj in gym_config.get("background", [])}
    scene_uids = rigid_uids | background_uids
    required_rigid = {
        roles.left_target_runtime_uid,
        roles.right_target_runtime_uid,
    }
    if not required_rigid.issubset(rigid_uids):
        raise ValueError(
            f"Generated rigid objects missing: {sorted(required_rigid - rigid_uids)}"
        )
    if roles.container_runtime_uid not in scene_uids:
        raise ValueError(
            f"Generated scene objects missing container: {roles.container_runtime_uid}"
        )

    success = gym_config["env"]["extensions"]["agent_success"]
    for term in success.get("terms", []):
        if (
            term.get("object") not in rigid_uids
            or term.get("container") not in scene_uids
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

    rigid_uid_list = [obj["uid"] for obj in gym_config.get("rigid_object", [])]
    if len(rigid_uid_list) != len(set(rigid_uid_list)):
        raise ValueError(f"Duplicate rigid object runtime uid(s): {rigid_uid_list}")
    rigid_uids = set(rigid_uid_list)
    background_uids = {obj["uid"] for obj in gym_config.get("background", [])}
    scene_uids = rigid_uids | background_uids
    moved_required = {placement.moved_runtime_uid for placement in spec.placements}
    missing_moved = moved_required - rigid_uids
    if missing_moved:
        raise ValueError(
            f"Generated relative config missing moved rigid object(s): {missing_moved}"
        )
    reference_required = {
        placement.reference_runtime_uid for placement in spec.placements
    }
    missing_reference = reference_required - scene_uids
    if missing_reference:
        raise ValueError(
            f"Generated relative config missing reference object(s): {missing_reference}"
        )

    _validate_success_uids(
        gym_config["env"]["extensions"]["agent_success"],
        rigid_uids=rigid_uids,
        scene_uids=scene_uids,
    )
    registry = gym_config["env"]["events"]["register_info_to_env"]["params"]["registry"]
    registered = {entry["entity_cfg"]["uid"] for entry in registry}
    required = moved_required | reference_required
    if not required.issubset(registered):
        raise ValueError(
            f"Relative config registry missing: {sorted(required - registered)}"
        )


def _validate_success_uids(
    success: Mapping[str, Any],
    *,
    rigid_uids: set[str],
    scene_uids: set[str],
) -> None:
    if success.get("op") in {"all", "and", "any", "or"}:
        for term in success.get("terms", []):
            _validate_success_uids(term, rigid_uids=rigid_uids, scene_uids=scene_uids)
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
        valid_uids = rigid_uids if key == "object" else scene_uids
        if uid not in valid_uids:
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
