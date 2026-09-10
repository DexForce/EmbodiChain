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
    "validate_scene_history_root",
    "validate_scene_output_separation",
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
    if gym_project is not None:
        validate_scene_output_separation(gym_project, result["output_dir"])

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


def validate_scene_output_separation(
    gym_project: str | Path,
    output_dir: str | Path,
) -> None:
    """Reject output paths that could replace or modify a read-only source.

    Args:
        gym_project: Existing Gym project directory or configuration path.
        output_dir: Transactional output directory for the Task Engine run.

    Raises:
        ValueError: If either path contains the other or both paths are equal.
    """
    source = Path(gym_project).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError(
            "Task Engine output_dir and source Gym project must not overlap."
        )


def validate_scene_history_root(
    gym_project: str | Path,
    output_root: str | Path,
) -> None:
    """Protect a source project before reserving a history-directory child.

    A prior run may live below the same history root because every new run is
    published to a distinct timestamped child. The inverse remains unsafe:
    creating the history root at or below the source project would write a
    reservation and output artifacts into the read-only source tree.

    Args:
        gym_project: Existing Gym project directory or configuration path.
        output_root: Parent directory under which a new run will be reserved.

    Raises:
        ValueError: If the history root is equal to or contained by the source
            project boundary.
    """
    source = Path(gym_project).expanduser().resolve()
    protected_root = source.parent if source.is_file() else source
    history_root = Path(output_root).expanduser().resolve()
    if protected_root == history_root or protected_root in history_root.parents:
        raise ValueError(
            "Task Engine output_root must not be inside the read-only source "
            "Gym project."
        )


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
