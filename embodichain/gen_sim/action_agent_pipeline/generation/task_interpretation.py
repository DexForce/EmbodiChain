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

"""Single-call semantic interpretation for action-agent config generation.

The model selects both the task route and that route's semantic spec in one
response. Geometry, arm assignment, target poses, and task-graph construction
remain in the existing deterministic generators.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from embodichain.gen_sim.action_agent_pipeline.generation.config_types import (
    SceneObject,
)
from embodichain.gen_sim.action_agent_pipeline.generation.naming import (
    _base_name,
    _is_container_like,
)
from embodichain.gen_sim.action_agent_pipeline.generation.spec_llm import (
    SPEC_SYSTEM_MESSAGE,
    request_json_spec,
)
from embodichain.gen_sim.action_agent_pipeline.generation._spec_scene_helpers import (
    color_hint_for_object,
)
from embodichain.gen_sim.action_agent_pipeline.generation.task_router import (
    TaskRouteSpec,
    _TASK_ROUTE_ARRANGEMENT_LINE,
    _TASK_ROUTE_OBJECT_MANIPULATION,
    _TASK_ROUTE_STACKING,
    _TASK_ROUTE_UNSUPPORTED,
    _normalize_task_route,
    _normalize_task_route_response,
)

__all__ = [
    "TaskInterpretationSpec",
    "_call_task_interpretation_llm",
    "_interpret_task_with_llm",
    "_make_task_interpretation_scene_summary",
    "_normalize_task_interpretation_response",
]

_SCHEMA_VERSION = "task_interpretation_v1"
_REQUIRED_SPEC_FIELDS = {
    _TASK_ROUTE_ARRANGEMENT_LINE: {
        "objects",
        "order_by",
        "order_direction",
        "anchor",
    },
    _TASK_ROUTE_STACKING: {
        "objects",
        "stack_mode",
        "bottom_to_top",
        "order_by",
        "anchor",
    },
    # Accept the canonical ordered ``steps`` field and the pre-v2
    # ``manipulations`` spelling during the compatibility window.
    _TASK_ROUTE_OBJECT_MANIPULATION: set(),
}
_FORBIDDEN_SPEC_FIELDS = {
    _TASK_ROUTE_ARRANGEMENT_LINE: {
        "stack_mode",
        "bottom_to_top",
        "manipulations",
    },
    _TASK_ROUTE_STACKING: {
        "category_order",
        "object_categories",
        "line_axis",
        "manipulations",
    },
    _TASK_ROUTE_OBJECT_MANIPULATION: {
        "objects",
        "category_order",
        "object_categories",
        "line_axis",
        "stack_mode",
        "bottom_to_top",
    },
}


@dataclass(frozen=True)
class TaskInterpretationSpec:
    """Validated route decision paired with its route-specific semantic spec."""

    task_route: TaskRouteSpec
    spec: Mapping[str, Any]

    @property
    def route(self) -> str:
        """Return the canonical route discriminator."""
        return self.task_route.route


def _interpret_task_with_llm(
    *,
    scene_objects: Sequence[SceneObject],
    project_name: str,
    task_description: str,
    model: str | None,
    task_llm_caller: Callable[..., Mapping[str, Any]] | None = None,
) -> TaskInterpretationSpec:
    """Request and validate one combined route-and-semantics response."""
    if task_llm_caller is None:
        task_llm_caller = _call_task_interpretation_llm
    response = task_llm_caller(
        project_name=project_name,
        task_description=task_description,
        scene_summary=_make_task_interpretation_scene_summary(scene_objects),
        model=model,
    )
    return _normalize_task_interpretation_response(
        response,
        scene_objects=scene_objects,
        task_description=task_description,
    )


def _call_task_interpretation_llm(
    *,
    project_name: str,
    task_description: str,
    scene_summary: list[dict[str, Any]],
    model: str | None,
) -> dict[str, Any]:
    """Call the single config-generation semantic interpretation stage."""
    return request_json_spec(
        template_name="task_interpretation.txt",
        usage_stage="config_generation.task_interpretation",
        project_name=project_name,
        task_description=task_description,
        scene_summary=scene_summary,
        model=model,
        system_message=SPEC_SYSTEM_MESSAGE,
    )


def _make_task_interpretation_scene_summary(
    scene_objects: Sequence[SceneObject],
) -> list[dict[str, Any]]:
    """Build a route-neutral scene summary without deriving target geometry."""
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
            "is_container_like": _is_container_like(obj),
        }
        for obj in scene_objects
    ]


def _normalize_task_interpretation_response(
    response: Mapping[str, Any],
    *,
    scene_objects: Sequence[SceneObject],
    task_description: str,
) -> TaskInterpretationSpec:
    """Validate the combined response without issuing a repair LLM call."""
    if not isinstance(response, Mapping):
        raise ValueError("Task interpretation response must be a JSON object.")
    if response.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError(
            "Task interpretation schema_version must be " f"{_SCHEMA_VERSION!r}."
        )

    requested_route = _normalize_task_route(response.get("route"))
    task_route = _normalize_task_route_response(
        response,
        scene_objects=scene_objects,
        task_description=task_description,
    )
    if task_route.route != requested_route:
        raise ValueError(
            "Task interpretation route conflicts with deterministic task-text "
            f"validation: model returned {requested_route!r}, validation selected "
            f"{task_route.route!r}. Regenerate the config with a corrected response."
        )

    spec = response.get("spec")
    if not isinstance(spec, Mapping):
        raise ValueError("Task interpretation spec must be a JSON object.")
    if task_route.route == _TASK_ROUTE_UNSUPPORTED:
        if spec:
            raise ValueError("Unsupported task interpretation must use an empty spec.")
        return TaskInterpretationSpec(task_route=task_route, spec={})

    forbidden_fields = sorted(_FORBIDDEN_SPEC_FIELDS[task_route.route] & set(spec))
    if forbidden_fields:
        raise ValueError(
            f"Task interpretation route/spec mismatch for {task_route.route!r}; "
            f"unexpected field(s): {', '.join(forbidden_fields)}."
        )
    required_fields = _REQUIRED_SPEC_FIELDS[task_route.route]
    missing_fields = sorted(required_fields - set(spec))
    if missing_fields:
        raise ValueError(
            f"Task interpretation spec for {task_route.route!r} is missing "
            f"required field(s): {', '.join(missing_fields)}."
        )
    if task_route.route == _TASK_ROUTE_OBJECT_MANIPULATION:
        _validate_object_manipulation_steps(spec)
    return TaskInterpretationSpec(task_route=task_route, spec=dict(spec))


def _validate_object_manipulation_steps(spec: Mapping[str, Any]) -> None:
    """Require exactly one canonical or legacy ordered-step collection."""
    present = [key for key in ("steps", "manipulations") if key in spec]
    if not present:
        raise ValueError(
            "Task interpretation spec for 'object_manipulation' is missing "
            "required field 'steps'."
        )
    if len(present) > 1:
        raise ValueError(
            "Object-manipulation spec must not define both 'steps' and "
            "'manipulations'."
        )
    steps = spec[present[0]]
    if not isinstance(steps, list) or not steps:
        raise ValueError("Object-manipulation steps must be a non-empty list.")
