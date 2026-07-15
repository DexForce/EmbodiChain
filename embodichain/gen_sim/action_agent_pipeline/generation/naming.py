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

from pathlib import Path
from typing import Any
import re

from embodichain.gen_sim.action_agent_pipeline.generation.config_types import (
    _BasketTaskRoles,
    _SceneObject,
)

__all__ = [
    "_base_name",
    "_candidate_relative_runtime_uid",
    "_container_runtime_uid",
    "_display_noun",
    "_generic_target_text",
    "_is_container_like",
    "_left_target_text",
    "_normalize_runtime_uid",
    "_object_text",
    "_plural",
    "_right_target_text",
    "_string_list",
    "_target_pair_text",
    "_target_plural_text",
    "_target_runtime_suffix",
    "_target_task_description_text",
]

_DIGIT_SUFFIX_RE = re.compile(r"_[0-9]+$")
_INVALID_UID_CHARS_RE = re.compile(r"[^0-9a-zA-Z_]+")
_CONTAINER_KEYWORDS = (
    "basket",
    "container",
    "bowl",
    "basin",
    "washbasin",
    "box",
    "bin",
    "tray",
    "crate",
    "pot",
    "pan",
    "bucket",
    "盆",
    "脸盆",
    "洗脸盆",
    "碗",
    "篮",
    "桶",
)
_DEFAULT_CONTAINER_RUNTIME_UID_ALIASES = {
    "basket": "wicker_basket",
}


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


def _container_runtime_uid(
    container: _SceneObject,
    aliases: dict[str, str] | None = None,
) -> str:
    """Return the runtime UID for the task container.

    The default aliases preserve legacy basket prompts, where any source name
    containing ``basket`` is exposed to the agent as ``wicker_basket``.
    """
    base = _base_name(container)
    for keyword, runtime_uid in (
        aliases or _DEFAULT_CONTAINER_RUNTIME_UID_ALIASES
    ).items():
        if keyword in base:
            return runtime_uid
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
