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

"""Route-free LLM planning boundary for Action Engine."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from string import Template
from typing import Any

from embodichain.gen_sim.action_engine.capabilities import build_default_registry
from embodichain.gen_sim.action_engine.domain import (
    TASK_AGENT_SCHEMA,
    validate_task_agent,
)

__all__ = ["plan_task"]

LLMCaller = Callable[..., Mapping[str, Any]]

_PROMPT_PATH = (
    Path(__file__).resolve().parents[4] / "texts" / "action_engine" / "task_planner.txt"
)
_GEN_CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "simready_pipeline"
    / "configs"
    / "gen_config.json"
)
_GEN_SIM_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
_UNSAFE_ID_RE = re.compile(r"[^0-9a-z]+")
_ORIENTATION_REQUEST_MARKERS = (
    "upright",
    "stand upright",
    "standing",
    "vertical",
    "lay flat",
    "lying flat",
    "orientation",
    "orient",
    "align",
    "aligned",
    "facing",
    "扶正",
    "竖直",
    "直立",
    "立起来",
    "放平",
    "平放",
    "躺平",
    "朝向",
    "对齐",
    "平行",
)
_ARRANGEMENT_WORLD_X_MARKERS = (
    "world_x",
    "world x",
    "x-axis",
    "x axis",
    "x轴",
    "x 轴",
    "x方向",
    "x 方向",
    "纵向",
    "前后排列",
    "前后摆放",
    "前后方向",
    "从前到后",
    "从前往后",
    "从后到前",
    "从后往前",
    "排成一列",
    "front-to-back",
    "front to back",
    "back-to-front",
    "back to front",
    "depth-wise",
    "depthwise",
    "longitudinal",
    "in a column",
)
_ARRANGEMENT_TABLE_LONG_AXIS_MARKERS = (
    "table_long_axis",
    "table long axis",
    "table's long axis",
    "table longest axis",
    "桌面长轴",
    "桌子的长轴",
    "桌子长轴",
)
_MODEL_STEP_KEYS = frozenset(
    {"id", "operator", "object", "objects", "actor", "goal", "depends_on"}
)

_MODEL_OUTPUT_SCHEMA: dict[str, Any] = {
    "title": "ActionEngineSemanticPlan",
    "type": "object",
    "additionalProperties": False,
    "required": ["semantic_steps", "allocation_groups"],
    "properties": {
        "semantic_steps": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["operator"],
                "properties": {
                    "id": {"type": "string"},
                    "operator": {"type": "string"},
                    "object": {"type": "string"},
                    "objects": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "actor": {"type": "object"},
                    "goal": {"type": "object"},
                    "depends_on": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
            },
        },
        "allocation_groups": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "semantic_step_ids", "arm_constraint"],
                "properties": {
                    "id": {"type": "string"},
                    "semantic_step_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "arm_constraint": {"const": "distinct_arms"},
                },
            },
        },
    },
}


def plan_task(
    task_description: str,
    scene_objects: Sequence[Mapping[str, Any]],
    *,
    task_name: str = "task",
    model: str | None = None,
    llm_caller: LLMCaller | None = None,
    deterministic_fallback: bool = False,
) -> dict[str, Any]:
    """Plan a natural-language task as route-free semantic steps.

    The model is intentionally prohibited from emitting atomic actions, graph
    edges, resources, target coordinates, or motion-policy parameters.
    ``compile_task_agent`` owns all of those deterministic decisions.

    Args:
        task_description: User goal in natural language.
        scene_objects: JSON-like scene inventory. ``runtime_uid`` is preferred
            over ``uid`` and ``source_uid`` for all generated references.
        task_name: Stable task identifier stored in the TaskAgent.
        model: Optional model-name override for the default LLM caller.
        llm_caller: Optional injected callable accepting ``prompt=`` and
            ``model=`` keyword arguments. It must return a mapping whose only
            top-level key is ``semantic_steps``.
        deterministic_fallback: If true, handle only unambiguous line-arrange
            and stack instructions without calling an LLM. This is intended for
            offline verification, not as a general natural-language parser.

    Returns:
        A validated ``action_engine_task_agent_v1`` mapping.
    """
    task_name = _nonempty(task_name, "task_name")
    task_description = _nonempty(task_description, "task_description")
    scene = _normalize_scene_objects(scene_objects)

    if deterministic_fallback:
        fallback_steps = _deterministic_semantic_steps(task_description, scene)
        if fallback_steps is not None:
            return _wrap_agent(
                task_name,
                task_description,
                fallback_steps,
                scene,
                allocation_groups=[],
            )

    prompt = _render_prompt(
        task_name=task_name,
        task_description=task_description,
        scene_objects=scene,
    )
    caller = llm_caller or _default_llm_caller
    response = caller(prompt=prompt, model=model)
    try:
        return _task_agent_from_response(
            response,
            task_name=task_name,
            task_description=task_description,
            scene=scene,
        )
    except (TypeError, ValueError) as first_error:
        # One bounded repair gives the model the verifier's exact complaint
        # without turning generation into an unbounded conversation.
        repair_prompt = (
            f"{prompt}\n\n"
            "Your previous JSON did not satisfy the TaskAgent contract.\n"
            f"Validation error: {first_error}\n"
            "Return one corrected JSON object. Do not explain the correction."
        )
        repaired = caller(prompt=repair_prompt, model=model)
        try:
            return _task_agent_from_response(
                repaired,
                task_name=task_name,
                task_description=task_description,
                scene=scene,
            )
        except (TypeError, ValueError) as second_error:
            raise ValueError(
                "Action Engine planner failed validation after one repair: "
                f"{second_error}"
            ) from second_error


def _task_agent_from_response(
    response: Any,
    *,
    task_name: str,
    task_description: str,
    scene: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Normalize and validate one model response as a TaskAgent."""
    if not isinstance(response, Mapping):
        raise ValueError("Action Engine planner output must be a JSON object.")
    allowed_fields = {"semantic_steps", "allocation_groups"}
    if not set(response) <= allowed_fields or "semantic_steps" not in response:
        raise ValueError(
            "Action Engine planner output may contain only 'semantic_steps' "
            "and 'allocation_groups'; "
            f"received fields {sorted(str(key) for key in response)}."
        )
    raw_steps = response["semantic_steps"]
    if not isinstance(raw_steps, Sequence) or isinstance(
        raw_steps, (str, bytes, bytearray)
    ):
        raise ValueError("Planner semantic_steps must be a list.")
    visible_operators = set(build_default_registry().operator_names())
    for index, step in enumerate(raw_steps):
        operator = step.get("operator") if isinstance(step, Mapping) else None
        if operator not in visible_operators:
            raise ValueError(
                f"Planner semantic_steps[{index}].operator must be one of "
                f"{sorted(visible_operators)}; got {operator!r}."
            )
    return _wrap_agent(
        task_name,
        task_description,
        raw_steps,
        scene,
        allocation_groups=response.get("allocation_groups", []),
    )


def _wrap_agent(
    task_name: str,
    task_description: str,
    raw_steps: Sequence[Any],
    scene: Sequence[Mapping[str, Any]],
    *,
    allocation_groups: Any,
) -> dict[str, Any]:
    steps = _normalize_semantic_steps(
        raw_steps,
        scene,
        task_description=task_description,
    )
    groups = _ensure_bilateral_allocation_group(
        task_description,
        steps,
        allocation_groups,
    )
    task_agent = validate_task_agent(
        {
            "schema_version": TASK_AGENT_SCHEMA,
            "task": task_name,
            "goal": task_description,
            "semantic_steps": steps,
            "allocation_groups": groups,
        },
        known_objects=[_scene_runtime_uid(item) for item in scene],
    )
    _validate_operator_contracts(task_agent)
    return task_agent


def _validate_operator_contracts(task_agent: Mapping[str, Any]) -> None:
    """Validate capability-specific step shapes inside the planner repair loop."""
    registry = build_default_registry()
    for step in task_agent["semantic_steps"]:
        operator = str(step["operator"])
        try:
            expanded = registry.operator(operator).expand(step)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Semantic step {step['id']!r} violates the {operator!r} "
                f"operator contract: {error}"
            ) from error
        if not expanded:
            raise ValueError(
                f"Semantic step {step['id']!r} produced no executable "
                f"{operator!r} operation."
            )


def _ensure_bilateral_allocation_group(
    task_description: str,
    steps: Sequence[Mapping[str, Any]],
    allocation_groups: Any,
) -> Any:
    """Preserve explicit or unambiguous two-sided upright arm intent."""
    if allocation_groups:
        return deepcopy(allocation_groups)
    normalized = task_description.casefold()
    bilateral = any(marker in normalized for marker in ("用双臂", "双臂", "both arms"))
    orient_steps = [
        step
        for step in steps
        if step.get("operator") == "orient_object"
        and not step.get("depends_on")
        and step.get("actor", {}).get("mode", "auto") == "auto"
    ]
    if not bilateral or len(orient_steps) != 2 or len(steps) != 2:
        return deepcopy(allocation_groups)
    return [
        {
            "id": "dual_arms_1",
            "semantic_step_ids": [step["id"] for step in orient_steps],
            "arm_constraint": "distinct_arms",
        }
    ]


def _normalize_semantic_steps(
    raw_steps: Sequence[Any],
    scene: Sequence[Mapping[str, Any]],
    *,
    task_description: str,
) -> list[dict[str, Any]]:
    if not raw_steps:
        raise ValueError("Planner semantic_steps must not be empty.")
    aliases = _scene_uid_aliases(scene)
    normalized: list[dict[str, Any]] = []
    known_ids: set[str] = set()
    previous_id: str | None = None

    for index, raw_step in enumerate(raw_steps, start=1):
        if not isinstance(raw_step, Mapping):
            raise ValueError(f"Planner semantic_steps[{index - 1}] must be an object.")
        step = deepcopy(dict(raw_step))
        unknown = sorted(set(step) - _MODEL_STEP_KEYS)
        if unknown:
            raise ValueError(
                f"Planner semantic_steps[{index - 1}] contains unsupported "
                f"fields: {unknown}."
            )
        operator = _nonempty(
            step.get("operator"),
            f"semantic_steps[{index - 1}].operator",
        )
        configured_id = str(step.get("id", "")).strip()
        step_id = configured_id or f"s{index:02d}_{_slug(operator)}"
        if step_id in known_ids:
            raise ValueError(
                f"Planner produced duplicate semantic step ID {step_id!r}."
            )
        known_ids.add(step_id)

        result: dict[str, Any] = {"id": step_id, "operator": operator}
        if "object" in step:
            result["object"] = _resolve_scene_uid(
                step["object"],
                aliases,
                f"semantic step {step_id!r} object",
            )
        if "objects" in step:
            objects = step["objects"]
            if not isinstance(objects, Sequence) or isinstance(
                objects, (str, bytes, bytearray)
            ):
                raise ValueError(f"Semantic step {step_id!r} objects must be a list.")
            result["objects"] = [
                _resolve_scene_uid(
                    object_uid,
                    aliases,
                    f"semantic step {step_id!r} objects",
                )
                for object_uid in objects
            ]

        actor = step.get("actor", {"mode": "auto"})
        if not isinstance(actor, Mapping):
            raise ValueError(f"Semantic step {step_id!r} actor must be an object.")
        result["actor"] = deepcopy(dict(actor))
        raw_goal = step.get("goal", {})
        if not isinstance(raw_goal, Mapping):
            raise ValueError(f"Semantic step {step_id!r} goal must be an object.")
        goal = deepcopy(dict(raw_goal))
        if operator == "arrange_line":
            # The model chooses semantics, but an unspecified line direction
            # has one stable robot-view default. Do not let sampling turn a
            # left-to-right row into a depth-wise layout with weaker reachability.
            goal["axis"] = _arrangement_line_axis(task_description)
            if not _requests_orientation_change(task_description):
                # A line-layout request does not imply reorientation. Silently
                # adding it can turn a reachable transport into an infeasible
                # fixed-grasp wrist flip.
                goal["orientation_goal"] = "preserve"
                goal["orientation_axis"] = "none"
        for key in (
            "anchor",
            "orientation_reference_object",
            "reference_object",
            "support_object",
        ):
            if key not in goal or goal[key] in {"table_center", "self"}:
                continue
            goal[key] = _resolve_scene_uid(
                goal[key],
                aliases,
                f"semantic step {step_id!r} goal.{key}",
            )
        result["goal"] = goal

        if "depends_on" in step:
            depends_on = step["depends_on"]
            if not isinstance(depends_on, Sequence) or isinstance(
                depends_on, (str, bytes, bytearray)
            ):
                raise ValueError(
                    f"Semantic step {step_id!r} depends_on must be a list."
                )
            result["depends_on"] = [str(value) for value in depends_on]
        else:
            # Sequential is the conservative default. The LLM must explicitly
            # emit an empty list when two semantic operations are independent.
            result["depends_on"] = [previous_id] if previous_id is not None else []
        normalized.append(result)
        previous_id = step_id
    return normalized


def _requests_orientation_change(task_description: str) -> bool:
    normalized = task_description.casefold()
    return any(marker in normalized for marker in _ORIENTATION_REQUEST_MARKERS)


def _arrangement_line_axis(task_description: str) -> str:
    """Resolve line direction from explicit intent, defaulting left-to-right."""
    normalized = task_description.casefold()
    if any(marker in normalized for marker in _ARRANGEMENT_TABLE_LONG_AXIS_MARKERS):
        return "table_long_axis"
    if any(marker in normalized for marker in _ARRANGEMENT_WORLD_X_MARKERS):
        return "world_x"
    return "world_y"


def _fuse_redundant_hold_place_steps(
    steps: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Remove a preparatory hold that a complete placement would repeat.

    ``place_relative`` already owns the pickup, transport, release, retreat,
    and home phases. A model may nevertheless emit ``hold_hover(object)``
    followed by ``place_relative(object)`` as if the operators were individual
    motion commands. The runtime cannot safely transfer that implicit held
    state between semantic steps, so normalize the unambiguous one-consumer
    pattern before TaskAgent validation.

    A hold with multiple consumers is intentionally left intact because it may
    reserve one arm while unrelated branches continue. Compilation rejects any
    later reuse of the held object rather than guessing an implicit handover.
    """
    result = [deepcopy(dict(step)) for step in steps]
    by_id = {step["id"]: step for step in result}
    dependents: dict[str, list[str]] = {step_id: [] for step_id in by_id}
    for step in result:
        for dependency in step["depends_on"]:
            if dependency in dependents:
                dependents[dependency].append(step["id"])

    removable: set[str] = set()
    claimed_places: set[str] = set()
    for hold in result:
        if hold["operator"] != "hold_hover":
            continue
        consumers = dependents[hold["id"]]
        if len(consumers) != 1:
            continue
        place = by_id[consumers[0]]
        if place["operator"] != "place_relative" or place.get("object") != hold.get(
            "object"
        ):
            continue
        if place["id"] in claimed_places:
            raise ValueError(
                f"Semantic step {place['id']!r} cannot consume more than one "
                "hold_hover state."
            )
        if not _is_default_hold_goal(hold):
            raise ValueError(
                f"Cannot fuse {hold['id']!r} into {place['id']!r}: a "
                "non-default hold_hover goal would be discarded."
            )

        place["actor"] = _merge_fused_actors(
            hold["actor"],
            place["actor"],
            hold_id=hold["id"],
            place_id=place["id"],
        )
        rewritten_dependencies: list[str] = []
        for dependency in place["depends_on"]:
            replacements = (
                hold["depends_on"] if dependency == hold["id"] else [dependency]
            )
            for replacement in replacements:
                if replacement not in rewritten_dependencies:
                    rewritten_dependencies.append(replacement)
        place["depends_on"] = rewritten_dependencies
        removable.add(hold["id"])
        claimed_places.add(place["id"])

    return [step for step in result if step["id"] not in removable]


def _is_default_hold_goal(hold: Mapping[str, Any]) -> bool:
    """Return whether removing a preparatory hover loses no requested state."""
    goal = hold["goal"]
    if set(goal) - {
        "orientation_axis",
        "orientation_goal",
        "reference_object",
        "reference_state",
    }:
        return False
    return (
        goal.get("orientation_axis", "none") == "none"
        and goal.get("orientation_goal", "preserve") == "preserve"
        and goal.get("reference_state", "initial") == "initial"
        and goal.get("reference_object", "self") in ("self", hold.get("object"))
    )


def _merge_fused_actors(
    hold_actor: Mapping[str, Any],
    place_actor: Mapping[str, Any],
    *,
    hold_id: str,
    place_id: str,
) -> dict[str, Any]:
    """Preserve an explicit arm requirement while fusing semantic steps."""
    hold = deepcopy(dict(hold_actor))
    place = deepcopy(dict(place_actor))
    hold_mode = hold.get("mode")
    place_mode = place.get("mode")
    hold_group = hold.get("allocation_group")
    place_group = place.get("allocation_group")
    if hold_group is not None and place_group is not None and hold_group != place_group:
        raise ValueError(
            f"Cannot fuse {hold_id!r} into {place_id!r}: conflicting "
            "allocation groups would lose explicit arm-allocation intent."
        )
    if hold_mode == "required" and place_mode == "required":
        if hold.get("arm") != place.get("arm"):
            raise ValueError(
                f"Cannot fuse {hold_id!r} into {place_id!r}: conflicting "
                "required arms would require an unsupported handover."
            )
        merged = place
    elif hold_mode == "required" and place_mode == "auto":
        merged = hold
    else:
        merged = place
    allocation_group = hold_group if hold_group is not None else place_group
    if allocation_group is not None:
        merged["allocation_group"] = allocation_group
    return merged


def _render_prompt(
    *,
    task_name: str,
    task_description: str,
    scene_objects: Sequence[Mapping[str, Any]],
) -> str:
    try:
        template_text = _PROMPT_PATH.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Action Engine planner prompt not found: {_PROMPT_PATH}"
        ) from exc
    capabilities = build_default_registry()
    return Template(template_text).substitute(
        task_name=task_name,
        task_description=task_description,
        scene_objects=json.dumps(
            list(scene_objects),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        operator_catalog=json.dumps(
            capabilities.operator_descriptions(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
    )


def _default_llm_caller(*, prompt: str, model: str | None) -> Mapping[str, Any]:
    """Invoke the configured OpenAI-compatible model with structured output."""
    # Heavy client imports remain lazy so validation and deterministic
    # compilation work in minimal simulation test environments.
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI

    settings = _load_llm_settings(model=model)
    kwargs: dict[str, Any] = {
        "api_key": settings["api_key"],
        "model": settings["model"],
        "temperature": 0,
    }
    if settings["base_url"]:
        kwargs["base_url"] = settings["base_url"]
    if settings["default_query"]:
        kwargs["default_query"] = settings["default_query"]
    client = ChatOpenAI(**kwargs)
    structured = (
        client.with_structured_output(_MODEL_OUTPUT_SCHEMA)
        if hasattr(client, "with_structured_output")
        else client
    )
    response = structured.invoke(
        [
            SystemMessage(
                content=(
                    "Return only the requested route-free semantic plan. "
                    "Never emit coordinates, atomic actions, or graph edges."
                )
            ),
            HumanMessage(content=prompt),
        ]
    )
    return _coerce_model_response(response)


def _load_llm_settings(*, model: str | None) -> dict[str, Any]:
    local_env = _load_env_file(_GEN_SIM_ENV_PATH)
    config: dict[str, Any] = {}
    if _GEN_CONFIG_PATH.exists():
        with _GEN_CONFIG_PATH.open("r", encoding="utf-8") as stream:
            raw = json.load(stream)
        if isinstance(raw, Mapping):
            llm = raw.get("llm", {})
            if isinstance(llm, Mapping):
                configured = llm.get("openai_compatible", {})
                if isinstance(configured, Mapping):
                    config = dict(configured)

    # Explicit process variables remain the highest-priority source. The local
    # file supplies project credentials without mutating os.environ, while the
    # JSON config continues to provide non-secret defaults.
    api_key = (
        _first_env_value(local_env, "OPENAI_API_KEY")
        or str(config.get("api_key", "")).strip()
    )
    selected_model = (
        (model.strip() if isinstance(model, str) else "")
        or _first_env_value(local_env, "OPENAI_MODEL", "LLM_MODEL")
        or str(config.get("model", "")).strip()
    )
    base_url = (
        _first_env_value(
            local_env,
            "OPENAI_BASE_URL",
            "OPENAI_API_BASE",
            "LLM_URL",
        )
        or str(config.get("base_url", "")).strip()
    ).rstrip("/")
    default_query = config.get("default_query", {}) or {}
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is required for Action Engine planning. Set it in "
            f"the process environment or {_GEN_SIM_ENV_PATH}."
        )
    if not selected_model:
        raise ValueError(
            "An LLM model is required through model=, OPENAI_MODEL, LLM_MODEL, "
            f"or {_GEN_CONFIG_PATH}."
        )
    if not isinstance(default_query, Mapping):
        raise ValueError("LLM default_query must be a mapping.")
    return {
        "api_key": api_key,
        "model": selected_model,
        "base_url": base_url,
        "default_query": dict(default_query),
    }


def _load_env_file(path: Path) -> dict[str, str]:
    """Read a local dotenv file without exporting credentials process-wide."""
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        key = key.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError(f"Invalid dotenv key at {path}:{line_number}.")
        value = raw_value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        values[key] = value
    return values


def _first_env_value(local_env: Mapping[str, str], *names: str) -> str | None:
    """Resolve aliases while keeping every shell value above local dotenv."""
    for source in (os.environ, local_env):
        for name in names:
            value = source.get(name)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _coerce_model_response(response: Any) -> Mapping[str, Any]:
    if isinstance(response, Mapping):
        return dict(response)
    content = getattr(response, "content", response)
    if isinstance(content, Mapping):
        return dict(content)
    if isinstance(content, list):
        content = "\n".join(
            str(item.get("text", ""))
            for item in content
            if isinstance(item, Mapping) and item.get("type") == "text"
        )
    if not isinstance(content, str):
        raise ValueError(
            f"Planner model output has unsupported type {type(content).__name__}."
        )
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:] if lines else lines
        lines = lines[:-1] if lines and lines[-1].startswith("```") else lines
        text = "\n".join(lines).strip()
    parsed = json.loads(text)
    if not isinstance(parsed, Mapping):
        raise ValueError("Planner model output must decode to a JSON object.")
    return dict(parsed)


def _normalize_scene_objects(
    scene_objects: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if not isinstance(scene_objects, Sequence) or isinstance(
        scene_objects, (str, bytes, bytearray)
    ):
        raise ValueError("scene_objects must be a list of mappings.")
    normalized: list[dict[str, Any]] = []
    runtime_uids: set[str] = set()
    for index, raw_object in enumerate(scene_objects):
        if not isinstance(raw_object, Mapping):
            raise ValueError(f"scene_objects[{index}] must be a mapping.")
        item = deepcopy(dict(raw_object))
        runtime_uid = _scene_runtime_uid(item)
        if runtime_uid in runtime_uids:
            raise ValueError(f"Duplicate scene runtime UID {runtime_uid!r}.")
        runtime_uids.add(runtime_uid)
        item["runtime_uid"] = runtime_uid
        normalized.append(item)
    if not normalized:
        raise ValueError("scene_objects must not be empty.")
    return normalized


def _scene_uid_aliases(
    scene_objects: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for item in scene_objects:
        runtime_uid = _scene_runtime_uid(item)
        for key in ("runtime_uid", "uid", "source_uid"):
            alias = item.get(key)
            if isinstance(alias, str) and alias:
                existing = aliases.get(alias)
                if existing is not None and existing != runtime_uid:
                    raise ValueError(f"Ambiguous scene object alias {alias!r}.")
                aliases[alias] = runtime_uid
    return aliases


def _resolve_scene_uid(value: Any, aliases: Mapping[str, str], context: str) -> str:
    uid = _nonempty(value, context)
    try:
        return aliases[uid]
    except KeyError as exc:
        raise ValueError(f"{context} references unknown scene object {uid!r}.") from exc


def _scene_runtime_uid(item: Mapping[str, Any]) -> str:
    for key in ("runtime_uid", "uid", "source_uid"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value
    raise ValueError("Every scene object requires runtime_uid, uid, or source_uid.")


def _deterministic_semantic_steps(
    task_description: str,
    scene: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]] | None:
    lowered = task_description.lower()
    line_requested = any(
        phrase in lowered
        for phrase in (
            "摆成一排",
            "排成一排",
            "排成一行",
            "arrange in a line",
            "one row",
        )
    )
    stack_requested = any(
        phrase in lowered for phrase in ("堆叠", "叠放", "摞起来", "stack", "pile")
    )
    if not line_requested and not stack_requested:
        return None

    movable = [
        item
        for item in scene
        if str(item.get("role", "")).lower() == "rigid_object"
        and _scene_runtime_uid(item) != "table"
    ]
    if line_requested and any(token in lowered for token in ("罐头", "易拉罐", "can")):
        cans = [
            item
            for item in movable
            if any(
                token
                in (
                    f"{item.get('uid', '')} {item.get('source_uid', '')} "
                    f"{item.get('description', '')}"
                ).lower()
                for token in ("can", "soda", "罐", "易拉罐")
            )
        ]
        if cans:
            movable = cans
    object_uids = [_scene_runtime_uid(item) for item in movable]
    if line_requested:
        if len(object_uids) < 2:
            raise ValueError("Deterministic arrange_line requires two movable objects.")
        return [
            {
                "id": "s01_arrange_line",
                "operator": "arrange_line",
                "objects": object_uids,
                "actor": {"mode": "auto"},
                "goal": {
                    "anchor": "table_center",
                    "axis": "world_y",
                    "order_by": "explicit",
                    "order_constraint": "free",
                    "order_direction": "given",
                    "orientation_axis": "none",
                    "orientation_goal": "preserve",
                },
                "depends_on": [],
            }
        ]
    if not object_uids:
        raise ValueError("Deterministic build_stack requires a movable object.")
    return [
        {
            "id": "s01_build_stack",
            "operator": "build_stack",
            "objects": object_uids,
            "actor": {"mode": "auto"},
            "goal": {
                "anchor": "table_center",
                "stack_mode": "on_top",
                "orientation_axis": "none",
                "orientation_goal": "preserve",
            },
            "depends_on": [],
        }
    ]


def _nonempty(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a non-empty string.")
    return value.strip()


def _slug(value: str) -> str:
    slug = _UNSAFE_ID_RE.sub("_", value.lower()).strip("_")
    return slug[:48].rstrip("_") or "step"
