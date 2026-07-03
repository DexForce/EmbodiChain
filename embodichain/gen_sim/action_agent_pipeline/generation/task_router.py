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

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any
import json

from embodichain.gen_sim.action_agent_pipeline.generation.config_types import (
    _SceneObject,
)
from embodichain.gen_sim.action_agent_pipeline.generation.naming import (
    _base_name,
    _string_list,
)

__all__ = [
    "_TASK_ROUTE_ARRANGEMENT_LINE",
    "_TASK_ROUTE_OBJECT_MANIPULATION",
    "_TASK_ROUTE_STACKING",
    "_TASK_ROUTE_UNSUPPORTED",
    "_TaskRouteSpec",
    "_call_task_router_llm",
    "_make_task_router_scene_summary",
    "_route_task_with_llm",
]

_TASK_ROUTE_STACKING = "stacking"
_TASK_ROUTE_ARRANGEMENT_LINE = "arrangement_line"
_TASK_ROUTE_OBJECT_MANIPULATION = "object_manipulation"
_TASK_ROUTE_UNSUPPORTED = "unsupported"
_TASK_ROUTES = {
    _TASK_ROUTE_STACKING,
    _TASK_ROUTE_ARRANGEMENT_LINE,
    _TASK_ROUTE_OBJECT_MANIPULATION,
    _TASK_ROUTE_UNSUPPORTED,
}
_TASK_ROUTE_ALIASES = {
    "arrangement": _TASK_ROUTE_ARRANGEMENT_LINE,
    "line_arrangement": _TASK_ROUTE_ARRANGEMENT_LINE,
    "line": _TASK_ROUTE_ARRANGEMENT_LINE,
    "relative": _TASK_ROUTE_OBJECT_MANIPULATION,
    "relative_placement": _TASK_ROUTE_OBJECT_MANIPULATION,
    "manipulation": _TASK_ROUTE_OBJECT_MANIPULATION,
    "object": _TASK_ROUTE_OBJECT_MANIPULATION,
    "object_manipulation": _TASK_ROUTE_OBJECT_MANIPULATION,
    "stack": _TASK_ROUTE_STACKING,
    "stacking": _TASK_ROUTE_STACKING,
    "unsupported": _TASK_ROUTE_UNSUPPORTED,
}


@dataclass(frozen=True)
class _TaskRouteSpec:
    route: str
    confidence: float
    reason: str
    candidate_objects: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_summary(self) -> dict[str, Any]:
        return {
            "route": self.route,
            "confidence": self.confidence,
            "reason": self.reason,
            "candidate_objects": list(self.candidate_objects),
            "warnings": list(self.warnings),
        }


def _route_task_with_llm(
    *,
    scene_objects: Sequence[_SceneObject],
    project_name: str,
    task_description: str,
    model: str | None,
    task_router_llm_caller: Callable[..., Mapping[str, Any]] | None = None,
) -> _TaskRouteSpec:
    scene_summary = _make_task_router_scene_summary(scene_objects)
    if task_router_llm_caller is None:
        task_router_llm_caller = _call_task_router_llm
    response = task_router_llm_caller(
        project_name=project_name,
        task_description=task_description,
        scene_summary=scene_summary,
        model=model,
    )
    return _normalize_task_route_response(
        response,
        scene_objects=scene_objects,
    )


def _call_task_router_llm(
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
        "Classify a tabletop robot task into exactly one action-agent config "
        "generation route. This router only chooses the route; it must not "
        "generate atomic actions, task graphs, target poses, offsets, robot "
        "configs, or success specs.\n\n"
        "Return exactly one JSON object with this schema:\n"
        "{\n"
        '  "route": "stacking|arrangement_line|object_manipulation|unsupported",\n'
        '  "confidence": 0.0,\n'
        '  "reason": "<short route rationale>",\n'
        '  "candidate_objects": ["<source_uid from rigid_object>", "..."],\n'
        '  "warnings": ["<optional warning>", "..."]\n'
        "}\n\n"
        "Route rules:\n"
        "- Base the route primarily on the task description. Use scene objects "
        "only to understand available object names/counts and to add warnings.\n"
        "- Choose arrangement_line when multiple tabletop objects should form "
        "one global row, line, column, left-to-right order, color order, size "
        "order, or other one-dimensional arrangement. This includes Chinese "
        "phrases such as 摆成一排, 排成一行, 排列, 排序, 从左到右, 由大到小, "
        "and mixed object types such as bottles, blocks, boxes, or cans in one "
        "line.\n"
        "- Choose stacking when objects should be vertically stacked, piled, "
        "placed one on top of another as a stack, or nested into each other. "
        "This includes Chinese phrases such as 叠, 叠放, 堆叠, 摞.\n"
        "- Choose object_manipulation for one or two moved objects with a "
        "relative placement, insertion, support-surface placement, directional "
        "move, upright/lay-flat adjustment, pick-up, hold, or hover task.\n"
        "- Choose unsupported when none of the existing routes can represent "
        "the task, for example pouring, opening articulated objects, cutting, "
        "deformable manipulation, or long tool-use workflows.\n"
        "- candidate_objects should list source_uid values that appear central "
        "to the task. It may be empty if the object set is ambiguous.\n"
        "- Do not use markdown.\n\n"
        f"Project: {project_name}\n"
        f"Task description:\n{task_description}\n"
        f"Scene objects:\n{json.dumps(scene_summary, ensure_ascii=False, indent=2)}"
    )
    llm = create_chat_openai(
        temperature=0.0,
        model=model,
        usage_stage="config_generation.task_router",
    )
    response = llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are a strict JSON router for simulation config "
                    "generation. Return only the requested JSON object."
                )
            ),
            HumanMessage(content=prompt),
        ]
    )
    content = getattr(response, "content", response)
    return extract_json_object(content)


def _make_task_router_scene_summary(
    scene_objects: Sequence[_SceneObject],
) -> list[dict[str, Any]]:
    return [
        {
            "source_uid": obj.source_uid,
            "role": obj.source_role,
            "object_type": _base_name(obj),
            "description": str(obj.config.get("description", "")).strip(),
            "init_pos": obj.config.get("init_pos"),
        }
        for obj in scene_objects
    ]


def _normalize_task_route_response(
    response: Mapping[str, Any],
    *,
    scene_objects: Sequence[_SceneObject],
) -> _TaskRouteSpec:
    route = _normalize_task_route(response.get("route"))
    confidence = _normalize_confidence(response.get("confidence", 0.0))
    reason = str(response.get("reason", "")).strip()
    candidate_objects = tuple(_string_list(response.get("candidate_objects")))
    warnings = tuple(_string_list(response.get("warnings")))
    _validate_candidate_objects(candidate_objects, scene_objects)
    _validate_route_feasibility(route, scene_objects)
    if not reason:
        reason = f"Task router selected {route}."
    return _TaskRouteSpec(
        route=route,
        confidence=confidence,
        reason=reason,
        candidate_objects=candidate_objects,
        warnings=warnings,
    )


def _normalize_task_route(value: Any) -> str:
    route = str(value or "").strip().lower()
    route = _TASK_ROUTE_ALIASES.get(route, route)
    if route not in _TASK_ROUTES:
        raise ValueError(
            f"Unsupported task route {value!r}; expected one of "
            f"{', '.join(sorted(_TASK_ROUTES))}."
        )
    return route


def _normalize_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Task router confidence must be a number.") from exc
    if confidence < 0.0 or confidence > 1.0:
        raise ValueError("Task router confidence must be between 0.0 and 1.0.")
    return confidence


def _validate_candidate_objects(
    candidate_objects: Sequence[str],
    scene_objects: Sequence[_SceneObject],
) -> None:
    known_uids = {obj.source_uid for obj in scene_objects}
    unknown = sorted(set(candidate_objects) - known_uids)
    if unknown:
        raise ValueError(
            "Task router returned unknown candidate object(s): "
            f"{', '.join(unknown)}."
        )


def _validate_route_feasibility(
    route: str,
    scene_objects: Sequence[_SceneObject],
) -> None:
    rigid_count = sum(1 for obj in scene_objects if obj.source_role == "rigid_object")
    if (
        route in {_TASK_ROUTE_ARRANGEMENT_LINE, _TASK_ROUTE_STACKING}
        and rigid_count < 2
    ):
        raise ValueError(
            f"Task route {route!r} requires at least two movable rigid objects."
        )
    if route == _TASK_ROUTE_OBJECT_MANIPULATION and rigid_count < 1:
        raise ValueError(
            "Task route 'object_manipulation' requires at least one movable "
            "rigid object."
        )
