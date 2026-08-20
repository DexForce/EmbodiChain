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
from embodichain.gen_sim.action_engine.orientation import (
    compile_orientation_constraint,
)

from .task_planner_prompt import TASK_PLANNER_PROMPT

__all__ = ["plan_task"]

LLMCaller = Callable[..., Mapping[str, Any]]

_GEN_CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "simready_pipeline"
    / "configs"
    / "gen_config.json"
)
_GEN_SIM_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
_UNSAFE_ID_RE = re.compile(r"[^0-9a-z]+")
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
    Returns:
        A validated ``action_engine_task_agent_v1`` mapping.
    """
    task_name = _nonempty(task_name, "task_name")
    task_description = _nonempty(task_description, "task_description")
    scene = _normalize_scene_objects(scene_objects)

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
    steps = _normalize_semantic_steps(raw_steps, scene)
    groups = deepcopy(allocation_groups)
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


def _normalize_semantic_steps(
    raw_steps: Sequence[Any],
    scene: Sequence[Mapping[str, Any]],
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
        "orientation_constraint",
        "orientation_axis",
        "orientation_directed",
        "orientation_goal",
        "reference_object",
        "reference_state",
    }:
        return False
    return (
        goal.get("orientation_axis", "none") == "none"
        and not compile_orientation_constraint(goal).terms
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
    capabilities = build_default_registry()
    return Template(TASK_PLANNER_PROMPT).substitute(
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
        "http_socket_options": (),
    }
    if settings["base_url"]:
        kwargs["base_url"] = settings["base_url"]
    if settings["default_query"]:
        kwargs["default_query"] = settings["default_query"]
    if _is_mimo_compatible(settings):
        # MiMo's OpenAI-compatible endpoint supports JSON mode but not the
        # OpenAI ``json_schema`` response format.  Disable hidden reasoning so
        # the bounded semantic response is not truncated to a few fields.
        kwargs.update(
            {
                "max_completion_tokens": _MIMO_MAX_COMPLETION_TOKENS,
                "extra_body": {"thinking": {"type": "disabled"}},
            }
        )
    client = ChatOpenAI(**kwargs)
    structured = _structured_output_runnable(
        client, _MODEL_OUTPUT_SCHEMA, settings=settings
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


_MIMO_MAX_COMPLETION_TOKENS = 4096


def _is_mimo_compatible(settings: Mapping[str, Any]) -> bool:
    """Identify MiMo models or regional compatible endpoints without secrets."""
    model = str(settings.get("model", "")).casefold()
    base_url = str(settings.get("base_url", "")).casefold()
    return "mimo" in model or "xiaomimimo.com" in base_url


def _structured_output_runnable(
    client: Any,
    schema: Mapping[str, Any],
    *,
    settings: Mapping[str, Any],
) -> Any:
    """Bind a portable JSON contract while retaining local strict validation.

    OpenAI-compatible providers do not share the same structured-output
    dialect.  MiMo documents ``json_object`` JSON mode rather than
    ``json_schema``; using the latter can return HTTP 200 with sparse nested
    objects.  The caller still validates the decoded object against its local
    schema after this transport-level binding.
    """
    if not hasattr(client, "with_structured_output"):
        return client
    method = "json_mode" if _is_mimo_compatible(settings) else "json_schema"
    try:
        return client.with_structured_output(schema, method=method)
    except (TypeError, ValueError):
        if method == "json_mode" and hasattr(client, "bind"):
            # Compatibility with older LangChain adapters that do not expose
            # the ``method`` keyword but do support response_format binding.
            from langchain_core.output_parsers import JsonOutputParser

            return (
                client.bind(response_format={"type": "json_object"})
                | JsonOutputParser()
            )
        # Preserve the historical adapter behavior for non-MiMo providers.
        return client.with_structured_output(schema)


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

    # A key and endpoint identify one provider transport and must not be mixed
    # across process, dotenv, and JSON configuration sources.
    api_key, base_url = _resolve_transport_settings(local_env, config)
    selected_model = (
        (model.strip() if isinstance(model, str) else "")
        or _first_env_value(
            local_env,
            "ACTION_ENGINE_LLM_MODEL",
            "OPENAI_MODEL",
            "LLM_MODEL",
        )
        or str(config.get("model", "")).strip()
    )
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


def _resolve_transport_settings(
    local_env: Mapping[str, str],
    config: Mapping[str, Any],
) -> tuple[str, str]:
    """Resolve an API key and endpoint from one configuration source."""
    transports = (
        (
            _mapping_value(os.environ, "OPENAI_API_KEY"),
            _mapping_value(
                os.environ,
                "OPENAI_BASE_URL",
                "OPENAI_API_BASE",
                "LLM_URL",
            ),
        ),
        (
            _mapping_value(local_env, "OPENAI_API_KEY"),
            _mapping_value(
                local_env,
                "OPENAI_BASE_URL",
                "OPENAI_API_BASE",
                "LLM_URL",
            ),
        ),
        (
            _mapping_value(config, "api_key"),
            _mapping_value(config, "base_url"),
        ),
    )
    for api_key, base_url in transports:
        if api_key and base_url:
            return api_key, base_url.rstrip("/")
    for api_key, base_url in transports:
        if api_key:
            return api_key, base_url.rstrip("/")
    return "", ""


def _mapping_value(source: Mapping[str, Any], *names: str) -> str:
    for name in names:
        value = source.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


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
    model_dump = getattr(response, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, Mapping):
            return dict(dumped)
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


def _nonempty(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a non-empty string.")
    return value.strip()


def _slug(value: str) -> str:
    slug = _UNSAFE_ID_RE.sub("_", value.lower()).strip("_")
    return slug[:48].rstrip("_") or "step"
