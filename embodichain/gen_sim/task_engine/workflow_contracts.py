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

"""Strict inputs for Task Engine cross-engine workflows."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Final, Literal, TypeAlias

__all__ = [
    "TASK_RUN_REQUEST_SCHEMA",
    "SceneInputKind",
    "TaskRunRequest",
    "scene_input_kind",
    "validate_task_run_request",
]

TASK_RUN_REQUEST_SCHEMA: Final = "embodichain.task-engine-run-request/v1"
TaskRunRequest: TypeAlias = dict[str, Any]
SceneInputKind = Literal["image", "gym_project"]

_REQUEST_KEYS = frozenset(
    {
        "schema_version",
        "task_id",
        "task_instruction",
        "image_path",
        "gym_project",
        "scene_edit_prompt",
        "output_dir",
    }
)


def validate_task_run_request(value: Mapping[str, Any]) -> TaskRunRequest:
    """Validate and detach one Task Engine run request.

    Version 1 deliberately has no ``scene_generation_prompt``. Image workflows
    use the image-only Scene Engine generation behavior and may apply one
    optional edit after that initial scene has been generated.
    """
    if not isinstance(value, Mapping):
        raise TypeError("TaskRunRequest must be a mapping.")
    result = deepcopy(dict(value))
    if set(result) != _REQUEST_KEYS:
        missing = sorted(_REQUEST_KEYS - set(result))
        extra = sorted(set(result) - _REQUEST_KEYS)
        raise ValueError(
            f"TaskRunRequest fields differ; missing={missing}, extra={extra}."
        )
    if result.get("schema_version") != TASK_RUN_REQUEST_SCHEMA:
        raise ValueError(
            "TaskRunRequest.schema_version must be " f"{TASK_RUN_REQUEST_SCHEMA!r}."
        )
    result["task_id"] = _nonempty(result.get("task_id"), "task_id")
    result["task_instruction"] = _nonempty(
        result.get("task_instruction"), "task_instruction"
    )
    result["output_dir"] = _path(result.get("output_dir"), "output_dir")

    image_path = _optional_path(result.get("image_path"), "image_path")
    gym_project = _optional_path(result.get("gym_project"), "gym_project")
    if (image_path is None) == (gym_project is None):
        raise ValueError(
            "TaskRunRequest requires exactly one of image_path or gym_project."
        )
    result["image_path"] = image_path
    result["gym_project"] = gym_project

    edit_prompt = result.get("scene_edit_prompt")
    if edit_prompt is not None:
        edit_prompt = _nonempty(edit_prompt, "scene_edit_prompt")
    result["scene_edit_prompt"] = edit_prompt
    _json_safe(result)
    return result


def scene_input_kind(request: Mapping[str, Any]) -> SceneInputKind:
    """Return the selected scene input kind after validating ``request``."""
    normalized = validate_task_run_request(request)
    return "image" if normalized["image_path"] is not None else "gym_project"


def _path(value: Any, field_name: str) -> str:
    text = _nonempty(value, field_name)
    return Path(text).expanduser().resolve().as_posix()


def _optional_path(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    return _path(value, field_name)


def _nonempty(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"TaskRunRequest.{field_name} must be a string.")
    result = value.strip()
    if not result:
        raise ValueError(f"TaskRunRequest.{field_name} must not be empty.")
    return result


def _json_safe(value: Any) -> None:
    try:
        json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("TaskRunRequest must contain strict JSON data.") from exc
