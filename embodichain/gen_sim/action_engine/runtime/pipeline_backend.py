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

"""Adapter from Action Engine programs to the proven Action Agent runtime."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

import torch

from embodichain.gen_sim.action_agent_pipeline.domain.seed_task_graph import (
    MOTION_POLICY_VERSION,
    SEED_TASK_GRAPH_SCHEMA_VERSION,
    SEMANTIC_STEP_SCHEMA_VERSION,
    validate_seed_task_graph,
)
from embodichain.gen_sim.action_agent_pipeline.runtime.graph_compiler import (
    compile_agent_graph_spec,
)

from .models import ExecutionProgram

__all__ = ["execute_pipeline_program", "lower_to_pipeline_seed"]

_ROUTE_BY_OPERATOR = {
    "arrange_line": "arrangement_line",
    "build_stack": "stacking",
    "coordinated_transport": "object_manipulation",
    "hold_hover": "object_manipulation",
    "orient_object": "object_manipulation",
    "place_relative": "object_manipulation",
}
_POLICY_ALIASES = {
    "default_coordinated_transport": "default_transport",
}


def lower_to_pipeline_seed(
    program: ExecutionProgram,
    *,
    env: Any | None = None,
) -> dict[str, Any]:
    """Lower one Action Engine program into the mature Seed Graph contract.

    This is intentionally a strict migration adapter, not a second compiler.
    It preserves the compiled DAG and only translates schema vocabulary that
    differs between the two runtimes. Unsupported cross-route compositions or
    semantics fail before any simulator action is sent.

    Args:
        program: Validated Action Engine execution program.
        env: Live environment used only to resolve geometry-dependent symbolic
            axes such as ``auto`` and ``table_long_axis``.

    Returns:
        A validated ``seed_task_graph_v5`` mapping.

    Raises:
        ValueError: If the mature runtime cannot preserve program semantics.
    """
    route = _pipeline_route(program)
    steps = [_lower_step(step, env=env) for step in program.raw["semantic_steps"]]
    edges = [
        {
            "id": str(edge["id"]),
            "source": str(edge["source"]),
            "target": str(edge["target"]),
            "actions": [_lower_action(action) for action in edge["actions"]],
            "depends_on": list(edge.get("depends_on", [])),
            "resources": list(edge.get("resources", [])),
        }
        for edge in program.raw["edges"]
    ]
    seed = {
        "schema_version": SEED_TASK_GRAPH_SCHEMA_VERSION,
        "task": program.task,
        "route": route,
        "program": "action_engine_pipeline_adapter",
        "start": program.start,
        "goal": program.goal,
        "nodes": deepcopy(list(program.nodes)),
        "edges": edges,
        "semantic_step_schema_version": SEMANTIC_STEP_SCHEMA_VERSION,
        "semantic_steps": steps,
        "allocation_groups": deepcopy(list(program.allocation_groups)),
        "motion_policy_version": MOTION_POLICY_VERSION,
    }
    validate_seed_task_graph(seed, task_name=program.task)
    return seed


def execute_pipeline_program(
    program: ExecutionProgram,
    env: Any,
    *,
    run_id: str | None = None,
    episode_index: int = 0,
    runtime_graph_renderer: Any | None = None,
) -> Any:
    """Execute an Action Engine program through the mature pipeline runtime."""
    seed = lower_to_pipeline_seed(program, env=env)
    graph = compile_agent_graph_spec(seed, task_name=program.task)
    runtime_kwargs: dict[str, Any] = {
        "env": env,
        "runtime_run_id": run_id,
        "episode_index": int(episode_index),
        "runtime_graph_renderer": runtime_graph_renderer,
        "allow_grasp_annotation": True,
        "force_grasp_reannotate": False,
        "grasp_convex_decomposition_method": "vhacd",
        "strict_serial": bool(
            getattr(env, "action_engine_pipeline_strict_serial", False)
        ),
        "semantic_step_settle_steps": int(
            getattr(env, "action_engine_settle_steps", 10)
        ),
    }
    grasp_defaults = getattr(env, "agent_grasp_runtime_defaults", None)
    if isinstance(grasp_defaults, Mapping):
        for key, value in grasp_defaults.items():
            runtime_key = "grasp_finger_length" if key == "finger_length" else str(key)
            runtime_kwargs.setdefault(runtime_key, value)
    return graph.run(**runtime_kwargs)


def _pipeline_route(program: ExecutionProgram) -> str:
    routes = set()
    unsupported = set()
    for step in program.semantic_steps:
        route = _ROUTE_BY_OPERATOR.get(step.operator)
        if route is None:
            unsupported.add(step.operator)
        else:
            routes.add(route)
    if unsupported:
        raise ValueError(
            "Pipeline backend does not support Action Engine operator(s): "
            f"{sorted(unsupported)}. Use --runtime-backend independent only for "
            "experimental characterization."
        )
    if len(routes) != 1:
        raise ValueError(
            "Pipeline backend currently requires one runtime route family per "
            f"program, got {sorted(routes)}. Split the task at a closed-loop "
            "boundary instead of silently changing its semantics."
        )
    return next(iter(routes))


def _lower_step(step: Mapping[str, Any], *, env: Any | None) -> dict[str, Any]:
    operator = str(step["operator"])
    goal = deepcopy(dict(step["goal"]))
    object_uid = str(step["object"])

    if operator == "arrange_line":
        operator = "place_in_line"
        axis = str(goal["axis"])
        if axis == "table_long_axis":
            axis = _resolve_table_long_axis(env)
        goal = {
            key: value
            for key, value in goal.items()
            if key
            in {
                "anchor",
                "axis",
                "layout",
                "nominal_slot_index",
                "objects",
                "order_constraint",
                "order_by",
                "order_direction",
                "orientation_axis",
                "orientation_goal",
                "slot_constraint",
            }
        }
        goal["axis"] = axis
        postcondition = deepcopy(dict(step["postcondition"]))
    elif operator == "build_stack":
        operator = "place_on_stack"
        layer_index = int(goal["layer_index"])
        reference = goal.get("reference_object")
        goal = {
            "relation": "on",
            "reference_object": reference,
            "reference_state": "live" if reference is not None else "symbolic_anchor",
            "layer_index": layer_index,
            "stack_mode": str(goal["stack_mode"]),
            "orientation_goal": str(goal["orientation_goal"]),
            "orientation_axis": str(goal["orientation_axis"]),
        }
        postcondition = {
            "type": "stack_layer_supported",
            "layer_index": layer_index,
            "reference_object": reference,
        }
    elif operator == "orient_object":
        goal, postcondition = _lower_orient_goal(step, env=env)
        operator = "place_relative"
    elif operator == "coordinated_transport":
        goal, postcondition = _lower_coordinated_goal(step)
        operator = "coordinated_pickment"
    else:
        goal = _lower_relative_goal(step)
        postcondition = {
            "type": "semantic_goal",
            "operator": operator,
            "relation": str(goal["relation"]),
        }

    return {
        "id": str(step["id"]),
        "operator": operator,
        "object": object_uid,
        "actor": _lower_actor(step["actor"]),
        "goal": goal,
        "depends_on": list(step.get("depends_on", [])),
        "postcondition": postcondition,
        "edge_ids": list(step["edge_ids"]),
    }


def _lower_relative_goal(step: Mapping[str, Any]) -> dict[str, Any]:
    goal = dict(step["goal"])
    operator = str(step["operator"])
    fields = {
        "orientation_axis",
        "orientation_goal",
        "orientation_reference_object",
        "reference_object",
        "reference_state",
        "relation",
    }
    if operator == "hold_hover":
        fields.add("terminal_behavior")
    return {key: deepcopy(value) for key, value in goal.items() if key in fields}


def _lower_orient_goal(
    step: Mapping[str, Any],
    *,
    env: Any | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    goal = dict(step["goal"])
    if goal.get("orientation_goal") != "upright":
        raise ValueError(
            "Pipeline backend currently preserves orient_object only for "
            "orientation_goal='upright'."
        )
    if goal.get("position_anchor", "initial_xy") != "initial_xy":
        raise ValueError(
            "Pipeline backend cannot preserve orient_object position_anchor="
            f"{goal.get('position_anchor')!r}; expected 'initial_xy'."
        )
    axis = str(goal.get("upright_local_axis", "auto"))
    if axis in {"auto", "long_axis"}:
        axis = _resolve_object_long_axis(env, str(step["object"]))
    support = str(goal.get("support_object", "table"))
    lowered = {
        "relation": "on",
        "reference_object": support,
        "reference_state": "live",
        "orientation_goal": "upright",
        "orientation_axis": str(goal.get("orientation_axis", "none")),
        "placement_mode": "upright_in_place",
        "upright_local_axis": axis,
    }
    return lowered, {
        "type": "semantic_goal",
        "operator": "place_relative",
        "relation": "on",
    }


def _lower_coordinated_goal(
    step: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    goal = dict(step["goal"])
    if goal.get("payloads"):
        raise ValueError(
            "Pipeline backend cannot preserve coordinated_transport payloads yet."
        )
    terminal = str(goal.get("terminal_behavior", "hold"))
    reference = goal.get("reference_object")
    relation = goal.get("relation")
    if terminal == "hold" and reference is None:
        reference = str(step["object"])
        relation = "held_above_initial"
        reference_state = "initial"
    else:
        if not isinstance(reference, str) or not reference:
            raise ValueError(
                "Pipeline coordinated placement requires reference_object."
            )
        relation = str(relation or "on")
        reference_state = "live"
    lowered = {
        "relation": str(relation),
        "reference_object": str(reference),
        "reference_state": reference_state,
        "orientation_goal": str(goal.get("orientation_goal", "preserve")),
        "orientation_axis": str(goal.get("orientation_axis", "none")),
        "direction": str(goal.get("direction", "none")),
        "terminal_behavior": terminal,
    }
    return lowered, {
        "type": "semantic_goal",
        "operator": "coordinated_pickment",
        "relation": str(relation),
    }


def _lower_action(action: Mapping[str, Any]) -> dict[str, Any]:
    action_class = str(action["atomic_action_class"])
    if action_class not in {
        "CoordinatedPickment",
        "MoveEndEffector",
        "MoveHeldObject",
        "MoveJoints",
        "PickUp",
        "Place",
    }:
        raise ValueError(
            f"Pipeline backend does not support atomic action {action_class!r}."
        )
    binding = dict(action["target_binding"])
    kind = str(binding["kind"])
    allowed_fields = {
        "coordinated_goal": {"kind", "object"},
        "current_held_pose": {"kind"},
        "joint_state": {"kind", "source"},
        "object": {"affordance", "kind", "object"},
        "policy_pose": {"kind"},
        "semantic_goal": {"kind", "phase", "semantic_step"},
    }
    if kind not in allowed_fields:
        raise ValueError(f"Pipeline backend does not support binding {kind!r}.")
    return {
        "atomic_action_class": action_class,
        "actor": _lower_actor(action["actor"]),
        "control": str(action["control"]),
        "target_binding": {
            key: deepcopy(value)
            for key, value in binding.items()
            if key in allowed_fields[kind]
        },
        "motion_policy": _POLICY_ALIASES.get(
            str(action["motion_policy"]),
            str(action["motion_policy"]),
        ),
    }


def _lower_actor(actor: Mapping[str, Any]) -> dict[str, Any]:
    mode = str(actor["mode"])
    lowered = {"mode": mode}
    if mode == "required":
        lowered["arm"] = str(actor["arm"])
    elif mode == "coordinated":
        lowered["arms"] = [str(arm) for arm in actor["arms"]]
    return lowered


def _resolve_object_long_axis(env: Any | None, uid: str) -> str:
    if env is None:
        raise ValueError(
            f"Resolving upright_local_axis for {uid!r} requires a live environment."
        )
    entity = env.sim.get_rigid_object(uid)
    if entity is None:
        raise ValueError(f"Unknown rigid object {uid!r}.")
    vertices = _vertices(entity, env)
    axis_index = int(
        torch.argmax(vertices.max(dim=0).values - vertices.min(dim=0).values)
    )
    return ("x", "y", "z")[axis_index]


def _resolve_table_long_axis(env: Any | None) -> str:
    if env is None:
        raise ValueError("Resolving table_long_axis requires a live environment.")
    table = env.sim.get_rigid_object("table")
    if table is None:
        raise ValueError("Pipeline arrangement requires rigid object 'table'.")
    vertices = _vertices(table, env)
    pose = torch.as_tensor(
        table.get_local_pose(to_matrix=True),
        dtype=torch.float32,
        device=env.device,
    )
    if pose.ndim == 3:
        pose = pose[0]
    world = vertices @ pose[:3, :3].transpose(0, 1) + pose[:3, 3]
    axis_index = int(
        torch.argmax(world[:, :2].max(dim=0).values - world[:, :2].min(dim=0).values)
    )
    return ("world_x", "world_y")[axis_index]


def _vertices(entity: Any, env: Any) -> torch.Tensor:
    value = entity.get_vertices(env_ids=[0], scale=True)
    if isinstance(value, (list, tuple)):
        value = value[0]
    vertices = torch.as_tensor(value, dtype=torch.float32, device=env.device)
    if vertices.ndim == 3 and vertices.shape[0] == 1:
        vertices = vertices[0]
    if vertices.ndim != 2 or vertices.shape[-1] != 3 or vertices.numel() == 0:
        raise ValueError("Rigid-object mesh vertices must have shape (N, 3).")
    return vertices
