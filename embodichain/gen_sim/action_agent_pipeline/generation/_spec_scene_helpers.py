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

"""Scene-summary and UID plumbing shared by the task-spec generators.

``arrangement_spec`` and ``stacking_spec`` both turn an LLM response into a
deterministic spec over the same ``SceneObject`` list. The helpers that read
object attributes, resolve mesh configs, build runtime UIDs, and summarize the
scene for the model are byte-identical or differ only by a route label. They
live here so the two routes cannot disagree on what a "resolved mesh" or a
"runtime UID" means.

Route-specific logic -- size scoring (max XYZ extent vs. footprint+height),
order-by aliases, anchor validation, orientation derivation -- stays in each
spec module and is injected where the shared helper needs it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from embodichain.gen_sim.action_agent_pipeline.generation.config_types import (
    SceneObject,
)
from embodichain.gen_sim.action_agent_pipeline.generation.naming import (
    _base_name,
    _normalize_runtime_uid,
)

__all__ = [
    "color_hint_for_object",
    "make_scene_summary",
    "object_attributes",
    "resolve_rigid_uid",
    "resolved_mesh_config",
    "rigid_runtime_uid_mapping",
]


def object_attributes(value: Any) -> dict[str, dict[str, str]]:
    """Normalize the LLM's ``object_attributes`` map into lowercase strings.

    Unknown shapes are skipped silently; the spec validator elsewhere rejects
    attributes attached to UIDs the router did not select.
    """
    if not isinstance(value, Mapping):
        return {}
    attributes: dict[str, dict[str, str]] = {}
    for source_uid, raw_attrs in value.items():
        if not isinstance(raw_attrs, Mapping):
            continue
        attributes[str(source_uid)] = {
            str(key): str(attr_value).strip().lower()
            for key, attr_value in raw_attrs.items()
            if str(attr_value).strip()
        }
    return attributes


def color_hint_for_object(obj: SceneObject) -> str | None:
    """Best-effort canonical color keyword from uid/description/mesh path.

    The LLM is allowed to omit color; this hint lets the prompt still express
    an ordering when the description mentions a color. Both EN and ZH aliases
    are matched because prompt2scene descriptions may be either language.
    """
    text = (
        f"{obj.source_uid} {obj.config.get('description', '')} "
        f"{obj.config.get('shape', {}).get('fpath', '')}"
    ).lower()
    color_aliases = {
        "red": ("red", "红"),
        "green": ("green", "绿"),
        "blue": ("blue", "蓝"),
        "yellow": ("yellow", "黄"),
        "orange": ("orange", "橙"),
        "purple": ("purple", "紫"),
        "black": ("black", "黑"),
        "white": ("white", "白"),
    }
    for canonical, aliases in color_aliases.items():
        if any(alias in text for alias in aliases):
            return canonical
    return None


def resolved_mesh_config(
    obj: SceneObject,
    *,
    scene_dir: Path,
) -> dict[str, Any]:
    """Return ``obj.config`` with a scene-relative mesh path resolved.

    Geometry helpers need an absolute path to read the GLB; source configs
    store paths relative to the scene directory. The original config is not
    mutated.
    """
    config = dict(obj.config)
    shape = dict(config.get("shape", {}) or {})
    fpath = shape.get("fpath")
    if isinstance(fpath, str):
        raw_path = Path(fpath)
        if not raw_path.is_absolute():
            shape["fpath"] = (scene_dir / raw_path).resolve().as_posix()
        config["shape"] = shape
    return config


def rigid_runtime_uid_mapping(
    rigid_objects: Sequence[SceneObject],
) -> dict[str, str]:
    """Map each source_uid to a stable runtime uid.

    When an object's base name is unique it becomes the runtime uid (readable
    in prompts); otherwise the source_uid is normalized to avoid collisions
    between two objects that share a generic name.
    """
    candidates = {obj.source_uid: _base_name(obj) for obj in rigid_objects}
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


def resolve_rigid_uid(
    value: str,
    rigid_by_uid: Mapping[str, SceneObject],
    *,
    field_name: str,
    route_label: str,
) -> str:
    """Resolve an LLM-returned uid against the rigid objects, fuzzily.

    Matches the exact source_uid first, then a normalized uid, then the base
    name -- so a model that says "cup" still resolves to "interact_cup_0".
    Exactly one match is required; zero or many are hard errors so a bad model
    answer fails generation rather than producing a wrong scene.
    """
    if value in rigid_by_uid:
        return value
    normalized = _normalize_runtime_uid(value)
    matches = [
        source_uid
        for source_uid, obj in rigid_by_uid.items()
        if _normalize_runtime_uid(source_uid) == normalized
        or _base_name(obj) == normalized
    ]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(f"LLM returned unknown {route_label} {field_name}: {value!r}.")
    raise ValueError(
        f"LLM returned ambiguous {route_label} {field_name}: {value!r}; "
        f"candidates: {matches}."
    )


def make_scene_summary(
    scene_objects: Sequence[SceneObject],
    *,
    scene_dir: Path,
    size_score_fn: Callable[[SceneObject, Path], float | None],
) -> list[dict[str, Any]]:
    """Build the per-object summary row list shown to the task-spec LLM.

    ``size_score_fn`` is route-specific: arrangement ranks by max XYZ extent,
    stacking by footprint-plus-height. Everything else about a row is identical
    across routes, so it lives here.
    """
    return [
        {
            "source_uid": obj.source_uid,
            "role": obj.source_role,
            "object_type": _base_name(obj),
            "description": str(obj.config.get("description", "")).strip(),
            "mesh": obj.config.get("shape", {}).get("fpath"),
            "init_pos": obj.config.get("init_pos"),
            "body_scale": obj.config.get("body_scale"),
            "color_hint": color_hint_for_object(obj),
            "size_score": size_score_fn(obj, scene_dir=scene_dir),
        }
        for obj in scene_objects
    ]
