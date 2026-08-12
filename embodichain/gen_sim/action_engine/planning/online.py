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

"""Online planner producing a complete direct AtomicAction SeedGraph."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
import json
from time import perf_counter
from typing import Any

from embodichain.gen_sim.action_engine.capabilities import (
    AtomicCapabilityRegistry,
    build_atomic_capability_registry,
)
from embodichain.gen_sim.action_engine.domain import (
    public_task_spec,
    requested_visual_task_predicates,
    validate_public_task_spec,
    validate_task_spec,
)
from embodichain.gen_sim.action_engine.protocol import SEED_GRAPH_SCHEMA
from embodichain.gen_sim.action_engine.runtime.motion_policy import (
    resolve_motion_policy,
)

from .vision import (
    SceneObservation,
    _reject_live_fields as _reject_visual_live_fields,
    analyze_visual_scene,
    validate_visual_facts,
)
from .linker import link_seed_graph

__all__ = ["plan_online_seed_graph"]

GraphCaller = Callable[..., Mapping[str, Any]]

_GRAPH_OUTPUT_SCHEMA = {
    "title": "ActionEngineOnlineSeedGraphBody",
    "type": "object",
    "additionalProperties": False,
    "required": ["nodes", "task_groups", "success"],
    "properties": {
        "nodes": {"type": "array", "items": {"type": "object"}},
        "task_groups": {"type": "array", "items": {"type": "object"}},
        "success": {"type": "object"},
    },
}


def plan_online_seed_graph(
    task_spec: Mapping[str, Any],
    observation: SceneObservation,
    *,
    visual_facts: Mapping[str, Any] | None = None,
    vlm_model: str | None = None,
    fact_caller: GraphCaller | None = None,
    graph_caller: GraphCaller | None = None,
    registry: AtomicCapabilityRegistry | None = None,
    robot_profile: str = "dual_ur10",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Extract visual facts and produce one validated online SeedGraph."""
    started = perf_counter()
    task = (
        validate_public_task_spec(task_spec)
        if "task_instances" not in task_spec and task_spec.get("level") == "L4"
        else validate_task_spec(task_spec)
    )
    _reject_private_or_live_fields(public_task_spec(task), "online TaskSpec")
    capabilities = registry or build_atomic_capability_registry()
    _reject_visual_live_fields(observation.entities, "SceneObservation.entities")
    known_uids = {str(item["uid"]) for item in observation.entities}
    if len(known_uids) != len(observation.entities):
        raise ValueError("Online scene observation contains duplicate entity UIDs.")
    if not known_uids:
        raise ValueError("Online scene observation contains no simulator entities.")
    visual_call_counter = [0]
    allowed_task_predicates = requested_visual_task_predicates(task)
    facts = (
        validate_visual_facts(
            visual_facts,
            known_uids=known_uids,
            camera_uids={camera.uid for camera in observation.cameras},
            allowed_task_predicates=allowed_task_predicates,
        )
        if visual_facts is not None
        else analyze_visual_scene(
            observation,
            task,
            model=vlm_model,
            caller=fact_caller,
            call_counter=visual_call_counter,
        )
    )
    _validate_fact_information(facts)
    prompt = _prompt(task, facts, capabilities, robot_profile=robot_profile)
    if graph_caller is None:
        # Facts remain the auditable planner input, but the production VLM also
        # needs the same reset-time RGB/depth evidence to bind semantic TaskSpec
        # roles (for example, "the purple can") to the known simulator UIDs.
        # An injected graph caller keeps the compact facts-only contract used by
        # deterministic tests and alternative planners.
        def caller(**kwargs: Any) -> Mapping[str, Any]:
            return _default_graph_caller(observation=observation, **kwargs)

    else:
        caller = graph_caller
    first_error: Exception | None = None
    graph_call_count = 0
    for attempt in range(2):
        current_prompt = prompt
        if first_error is not None:
            current_prompt += (
                "\n\nThe previous graph was invalid. Correct only the JSON body. "
                f"Validation error: {first_error}"
            )
        graph_call_count += 1
        try:
            response = caller(
                prompt=current_prompt,
                schema=_GRAPH_OUTPUT_SCHEMA,
                model=vlm_model,
            )
            graph = _wrap_graph(response, task, capabilities)
            _reject_private_or_live_fields(graph, "online SeedGraph")
            graph = link_seed_graph(
                graph,
                registry=capabilities,
                task_order=[str(item["id"]) for item in task.get("task_instances", ())],
                known_objects=known_uids,
            )
            _validate_explicit_task_group_coverage(task, graph)
            for node in graph["nodes"]:
                capabilities.validate_binding(node)
                if capabilities.get(str(node["atomic_action"])).runtime_available:
                    resolve_motion_policy(
                        robot_profile,
                        node["atomic_action"],
                        node["motion_policy"],
                    )
            graph["metadata"].update(
                {
                    "planning_latency_seconds": perf_counter() - started,
                    "vlm_call_count": graph_call_count + visual_call_counter[0],
                    "visual_fact_call_count": visual_call_counter[0],
                    "graph_call_count": graph_call_count,
                }
            )
            return graph, facts
        except (TypeError, ValueError) as error:
            if attempt:
                raise ValueError(
                    "Online SeedGraph failed validation after one repair: " f"{error}"
                ) from error
            first_error = error
    raise AssertionError("unreachable")


def _prompt(
    task: Mapping[str, Any],
    facts: Mapping[str, Any],
    capabilities: AtomicCapabilityRegistry,
    *,
    robot_profile: str,
) -> str:
    from embodichain.gen_sim.action_engine.config import default_runtime_policy
    from embodichain.gen_sim.action_engine.tasks import task_capability_catalog

    runtime_policy = default_runtime_policy(robot_profile)
    motion_modifiers: dict[str, list[dict[str, str]]] = {
        action: [] for action in runtime_policy.motion_defaults
    }
    for modifier_type, modes in runtime_policy.motion_modifiers.items():
        for mode, action_patches in modes.items():
            for action in action_patches:
                motion_modifiers[action].append({"type": modifier_type, "mode": mode})
    grouping_instruction = (
        "Infer the necessary E TaskGroups from the abstract goal; the private "
        "reference task instances are intentionally hidden."
        if task["level"] == "L4"
        else "Every public TaskSpec task instance must correspond to exactly one TaskGroup."
    )
    return (
        "Produce the body of one coordinate-free direct AtomicAction SeedGraph. "
        f"{grouping_instruction} "
        "Nodes may contain only symbolic target bindings and scene UIDs; never "
        "emit world coordinates, poses, qpos, trajectories, or grasp poses. "
        "Do not emit Action Contracts or resource claims; the deterministic "
        "Contract Linker owns those fields. "
        "Use the supplied reset-time multi-view image evidence only to bind the "
        "public task semantics to known UIDs; use normalized visual constraints "
        "only when the facts justify them. "
        "Do not output reasoning. Planning-only actions may appear but must not "
        "be replaced with invented primitives.\n\n"
        f"Public TaskSpec:\n{json.dumps(public_task_spec(task), ensure_ascii=False, sort_keys=True)}\n\n"
        f"Visual facts:\n{json.dumps(facts, ensure_ascii=False, sort_keys=True)}\n\n"
        f"E1-E9 task semantics:\n{json.dumps(task_capability_catalog(), ensure_ascii=False, sort_keys=True)}\n\n"
        f"Atomic capabilities:\n{json.dumps(capabilities.catalog(), ensure_ascii=False, sort_keys=True)}\n\n"
        "Every node motion_policy must be an object with a modifiers list; "
        "the AtomicAction selects its base policy implicitly. Use only the "
        "typed modifiers supported by that action.\n"
        f"Allowed motion modifiers by AtomicAction:\n"
        f"{json.dumps(motion_modifiers, sort_keys=True)}"
    )


def _wrap_graph(
    response: Mapping[str, Any],
    task: Mapping[str, Any],
    capabilities: AtomicCapabilityRegistry,
) -> dict[str, Any]:
    if not isinstance(response, Mapping):
        raise TypeError("Online planner output must be a mapping.")
    if set(response) != {"nodes", "task_groups", "success"}:
        raise ValueError(
            "Online planner must return nodes, task_groups, and success only."
        )
    for index, node in enumerate(response.get("nodes", ())):
        if not isinstance(node, Mapping):
            raise TypeError(f"Online planner node {index} must be a mapping.")
        forbidden = sorted({"contract", "resources"} & set(node))
        if forbidden:
            raise ValueError(
                f"Online planner node {index} may not author linker-owned fields: "
                f"{forbidden}."
            )
    for index, group in enumerate(response.get("task_groups", ())):
        if not isinstance(group, Mapping):
            raise TypeError(f"Online planner TaskGroup {index} must be a mapping.")
        if "contract" in group:
            raise ValueError(
                f"Online planner TaskGroup {index} may not author its contract."
            )
    return {
        "schema_version": SEED_GRAPH_SCHEMA,
        "task_id": task["task_id"],
        "instruction": task["instruction"],
        "level": task["level"],
        "reasoning_type": task["reasoning_type"],
        "planner_route": "online",
        "nodes": deepcopy(response["nodes"]),
        "task_groups": deepcopy(response["task_groups"]),
        "success": deepcopy(response["success"]),
        "capability_catalog_hash": capabilities.catalog_hash(),
        "metadata": {
            "oracle_exposed": False,
            "visual_facts_used": True,
            "allocation_groups": deepcopy(
                task.get("metadata", {}).get("allocation_groups", [])
            ),
        },
    }


def _default_graph_caller(
    *,
    prompt: str,
    schema: Mapping[str, Any],
    model: str | None,
    observation: SceneObservation | None = None,
) -> Mapping[str, Any]:
    from .vision import _camera_evidence, _default_structured_caller, _vlm_model

    images: list[str] = []
    if observation is not None:
        _, images = _camera_evidence(observation)

    return _default_structured_caller(
        prompt=prompt,
        images=images,
        schema=schema,
        model=_vlm_model(model),
    )


def _validate_fact_information(facts: Mapping[str, Any]) -> None:
    """Reject low-information visual outputs before graph planning."""
    confidence = facts.get("confidence", 0.0)
    if float(confidence) < 0.5:
        raise ValueError(
            "VLM visual facts confidence is below the required 0.5 threshold."
        )
    entities = facts.get("entities", ())
    if not any(
        bool(item.get("visible", True)) and float(item.get("confidence", 0.0)) >= 0.5
        for item in entities
        if isinstance(item, Mapping)
    ):
        raise ValueError("VLM visual facts contain no reliable visible entity.")


def _validate_explicit_task_group_coverage(
    task: Mapping[str, Any], graph: Mapping[str, Any]
) -> None:
    """Reject an online graph that drops or invents an explicit L1-L3 step."""
    if task.get("level") == "L4":
        return
    expected = {
        str(item["id"])
        for item in task.get("task_instances", ())
        if isinstance(item, Mapping)
    }
    actual = {str(group["id"]) for group in graph.get("task_groups", ())}
    if expected != actual:
        raise ValueError(
            "Online SeedGraph TaskGroup coverage mismatch; "
            f"missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}."
        )


_PRIVATE_OR_LIVE_KEYS = frozenset(
    {
        "absolute_position",
        "coordinates",
        "grasp_pose",
        "joint_positions",
        "live_pose",
        "live_transform",
        "object_pose",
        "oracle",
        "pose",
        "positions",
        "qpos",
        "target_pose",
        "trajectory",
        "waypoints",
        "xpos",
    }
)


def _reject_private_or_live_fields(value: Any, context: str) -> None:
    """Reject private oracle and grounded simulator fields recursively."""
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).lower() in _PRIVATE_OR_LIVE_KEYS:
                raise ValueError(f"{context} contains private/live field {key!r}.")
            _reject_private_or_live_fields(child, f"{context}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _reject_private_or_live_fields(child, f"{context}[{index}]")
