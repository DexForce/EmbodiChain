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

"""Render symbolic and compiled action-agent graphs as deterministic PNGs."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from functools import lru_cache
from io import BytesIO
import math
import textwrap
from typing import Any

import matplotlib

# Select a non-interactive backend before importing any rendering classes.
matplotlib.use("Agg", force=True)

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch
import networkx as nx

from embodichain.gen_sim.action_agent_pipeline.generation.seed_task_graph import (
    seed_task_graph_hash,
    validate_seed_task_graph,
)
from embodichain.gen_sim.action_agent_pipeline.protocol.actions import (
    LEFT_ARM_ACTION_KEY,
    RIGHT_ARM_ACTION_KEY,
)

__all__ = ["render_seed_task_graph_png", "render_task_graph_png"]

_BACKGROUND_COLOR = "#FFFFFF"
_TEXT_COLOR = "#17202A"
_MUTED_TEXT_COLOR = "#4D5B66"
_PROGRAM_COLOR = "#263746"
_STEP_COLOR = "#DCEBFA"
_ENTITY_COLOR = "#F1F3F5"
_POSTCONDITION_COLOR = "#DDF3E4"
_START_COLOR = "#34495E"
_GOAL_COLOR = "#59A96A"
_DEFAULT_NODE_COLOR = "#EEF1F4"
_LEFT_EDGE_COLOR = "#188977"
_RIGHT_EDGE_COLOR = "#D97706"
_DUAL_EDGE_COLOR = "#7C4DAD"
_NEUTRAL_EDGE_COLOR = "#7A8793"
_DEPENDENCY_EDGE_COLOR = "#3A6EA5"
_REFERENCE_EDGE_COLOR = "#876445"
_POSTCONDITION_EDGE_COLOR = "#3A8D5D"
_SEMANTIC_STEP_COLORS = (
    "#E8F1FB",
    "#FFF0DA",
    "#E8F5E9",
    "#F2EAF8",
    "#FCE8EC",
    "#E4F5F3",
)
_PNG_DPI = 160
_TASK_CHAIN_COLUMNS = 6


def render_seed_task_graph_png(seed_graph: Mapping[str, Any]) -> bytes:
    """Render one validated symbolic seed graph to PNG bytes.

    Args:
        seed_graph: Symbolic seed graph produced by deterministic generation.

    Returns:
        A complete PNG byte sequence.

    Raises:
        TypeError: If the seed graph does not follow the symbolic schema.
        ValueError: If the seed graph contains invalid dependencies or fields.
    """
    validate_seed_task_graph(seed_graph)
    display_graph = _build_seed_display_graph(seed_graph)
    positions = _seed_positions(display_graph)
    step_count = len(seed_graph["steps"])
    entity_count = sum(
        1
        for _, attributes in display_graph.nodes(data=True)
        if attributes["kind"] == "entity"
    )
    row_count = max(step_count, entity_count, 1)
    figure = _make_figure(width=18.0, height=max(7.0, row_count * 1.7 + 3.0))
    try:
        axis = figure.subplots()
        axis.set_axis_off()
        _draw_seed_edges(axis, display_graph, positions)
        _draw_nodes(axis, display_graph, positions, seed=True)
        _draw_seed_legend(axis)
        graph_hash = seed_task_graph_hash(seed_graph)[:12]
        title = (
            f"Seed Task Graph | task={seed_graph['task']} | "
            f"route={seed_graph['route']} | program={seed_graph['program']} | "
            f"steps={step_count} | hash={graph_hash}"
        )
        axis.set_title(
            title,
            color=_TEXT_COLOR,
            fontproperties=_font(size=12, weight="bold"),
            pad=22,
        )
        _fit_axis_to_positions(axis, positions, x_margin=0.75, y_margin=1.0)
        return _figure_png_bytes(figure)
    finally:
        figure.clear()


def render_task_graph_png(task_graph: Mapping[str, Any]) -> bytes:
    """Render one compiled task graph to PNG bytes.

    Args:
        task_graph: Deterministic atomic task graph.

    Returns:
        A complete PNG byte sequence.

    Raises:
        TypeError: If nodes or edges use unsupported types.
        ValueError: If identifiers, endpoints, or graph topology are invalid.
    """
    display_graph, node_records, edge_records = _validated_task_display_graph(
        task_graph
    )
    positions, is_chain = _task_positions(display_graph, task_graph)
    row_count = (
        math.ceil(len(node_records) / _TASK_CHAIN_COLUMNS)
        if is_chain
        else _topological_generation_count(display_graph)
    )
    column_count = (
        min(_TASK_CHAIN_COLUMNS, len(node_records))
        if is_chain
        else _max_topological_generation_width(display_graph)
    )
    figure = _make_figure(
        width=max(11.0, min(24.0, column_count * 3.5 + 1.5)),
        height=max(7.0, min(30.0, row_count * 3.4 + 3.0)),
    )
    try:
        axis = figure.subplots()
        axis.set_axis_off()
        edge_step_ids = _edge_semantic_step_ids(task_graph, edge_records)
        _draw_task_edges(axis, edge_records, positions, edge_step_ids)
        node_step_ids = _node_semantic_step_ids(edge_records, edge_step_ids)
        _draw_task_nodes(
            axis,
            node_records,
            positions,
            task_graph=task_graph,
            node_step_ids=node_step_ids,
        )
        _draw_task_legend(axis)
        graph_hash = str(task_graph.get("seed_graph_hash", "unavailable"))[:12]
        semantic_steps = task_graph.get("semantic_steps") or []
        title = (
            f"Compiled Task Graph | task={task_graph.get('task', 'unknown')} | "
            f"nodes={len(node_records)} | edges={len(edge_records)} | "
            f"semantic_steps={len(semantic_steps)} | "
            f"seed_hash={graph_hash}"
        )
        axis.set_title(
            title,
            color=_TEXT_COLOR,
            fontproperties=_font(size=12, weight="bold"),
            pad=22,
        )
        _fit_axis_to_positions(axis, positions, x_margin=1.0, y_margin=1.15)
        return _figure_png_bytes(figure)
    finally:
        figure.clear()


def _build_seed_display_graph(
    seed_graph: Mapping[str, Any],
) -> nx.MultiDiGraph:
    """Build the semantic display model without mutating the source graph."""
    graph = nx.MultiDiGraph()
    graph.add_node(
        "program",
        kind="program",
        label=_wrapped_label(
            str(seed_graph["task"]),
            str(seed_graph["route"]),
            str(seed_graph["program"]),
            width=25,
        ),
        order=0,
    )
    entity_order: dict[str, int] = {}
    steps = seed_graph["steps"]
    for step_index, step in enumerate(steps):
        step_id = str(step["id"])
        step_node = f"step:{step_id}"
        actor_text = _seed_actor_text(step["actor"])
        graph.add_node(
            step_node,
            kind="semantic_step",
            label=_wrapped_label(
                _short_identifier(step_id),
                str(step["operator"]),
                actor_text,
                width=28,
            ),
            order=step_index,
            step_id=step_id,
        )
        dependencies = step["depends_on"]
        if dependencies:
            for dependency in dependencies:
                graph.add_edge(
                    f"step:{dependency}",
                    step_node,
                    relation="depends_on",
                    label="depends_on",
                )
        else:
            graph.add_edge(
                "program",
                step_node,
                relation="starts",
                label="starts",
            )

        object_uid = str(step["object"])
        if object_uid != "__arrangement__":
            entity_node = _add_seed_entity(graph, entity_order, object_uid)
            graph.add_edge(
                step_node,
                entity_node,
                relation="acts_on",
                label="acts_on",
            )

        goal = step["goal"]
        for reference_key in ("reference_object", "orientation_reference_object"):
            reference_uid = goal.get(reference_key)
            if isinstance(reference_uid, str) and reference_uid:
                entity_node = _add_seed_entity(
                    graph,
                    entity_order,
                    reference_uid,
                )
                graph.add_edge(
                    step_node,
                    entity_node,
                    relation="references",
                    label=(
                        "references"
                        if reference_key == "reference_object"
                        else "orientation reference"
                    ),
                )

        members = goal.get("objects")
        if isinstance(members, Sequence) and not isinstance(members, (str, bytes)):
            ordered = goal.get("order_constraint") != "free"
            for member_index, member in enumerate(members, start=1):
                if not isinstance(member, str) or not member:
                    continue
                entity_node = _add_seed_entity(graph, entity_order, member)
                graph.add_edge(
                    step_node,
                    entity_node,
                    relation="member",
                    label=f"member #{member_index}" if ordered else "member",
                )

        postcondition = step["postcondition"]
        postcondition_node = f"post:{step_id}"
        graph.add_node(
            postcondition_node,
            kind="postcondition",
            label=_postcondition_label(postcondition),
            order=step_index,
        )
        graph.add_edge(
            step_node,
            postcondition_node,
            relation="verifies",
            label="verifies",
        )
    return graph


def _add_seed_entity(
    graph: nx.MultiDiGraph,
    entity_order: dict[str, int],
    uid: str,
) -> str:
    node_id = f"entity:{uid}"
    if node_id not in graph:
        entity_order[uid] = len(entity_order)
        graph.add_node(
            node_id,
            kind="entity",
            label=_wrap_and_truncate(uid, width=24, limit=58),
            order=entity_order[uid],
        )
    return node_id


def _seed_positions(graph: nx.MultiDiGraph) -> dict[str, tuple[float, float]]:
    nodes_by_kind: dict[str, list[str]] = defaultdict(list)
    for node_id, attributes in graph.nodes(data=True):
        nodes_by_kind[attributes["kind"]].append(node_id)
    for nodes in nodes_by_kind.values():
        nodes.sort(key=lambda node_id: graph.nodes[node_id]["order"])

    row_count = max(
        len(nodes_by_kind["semantic_step"]),
        len(nodes_by_kind["entity"]),
        1,
    )
    positions: dict[str, tuple[float, float]] = {
        "program": (0.0, -0.8 * (row_count - 1))
    }
    positions.update(
        _column_positions(
            nodes_by_kind["semantic_step"],
            x=4.2,
            row_count=row_count,
        )
    )
    positions.update(
        _column_positions(
            nodes_by_kind["entity"],
            x=9.0,
            row_count=row_count,
        )
    )
    step_y = {
        graph.nodes[node_id]["order"]: positions[node_id][1]
        for node_id in nodes_by_kind["semantic_step"]
    }
    for node_id in nodes_by_kind["postcondition"]:
        order = graph.nodes[node_id]["order"]
        positions[node_id] = (13.5, step_y.get(order, 0.0))
    return positions


def _column_positions(
    node_ids: Sequence[str],
    *,
    x: float,
    row_count: int,
) -> dict[str, tuple[float, float]]:
    if not node_ids:
        return {}
    if len(node_ids) == 1:
        return {node_ids[0]: (x, -0.8 * (row_count - 1))}
    total_height = 1.6 * (row_count - 1)
    interval = total_height / (len(node_ids) - 1)
    return {node_id: (x, -index * interval) for index, node_id in enumerate(node_ids)}


def _validated_task_display_graph(
    task_graph: Mapping[str, Any],
) -> tuple[nx.DiGraph, list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    if not isinstance(task_graph, Mapping):
        raise TypeError("Task graph must be a mapping.")
    nodes = task_graph.get("nodes")
    edges = task_graph.get("edges")
    if not isinstance(nodes, list) or not nodes:
        raise ValueError("Task graph requires a non-empty nodes list.")
    if not isinstance(edges, list) or not edges:
        raise ValueError("Task graph requires a non-empty edges list.")

    graph = nx.DiGraph()
    node_records: list[Mapping[str, Any]] = []
    node_ids: set[str] = set()
    for index, node in enumerate(nodes):
        if not isinstance(node, Mapping):
            raise TypeError(f"Task graph node {index} must be a mapping.")
        node_id = node.get("id")
        if not isinstance(node_id, str) or not node_id:
            raise ValueError(f"Task graph node {index} requires a non-empty id.")
        if node_id in node_ids:
            raise ValueError(f"Duplicate task graph node id: {node_id!r}.")
        node_ids.add(node_id)
        node_records.append(node)
        graph.add_node(node_id)

    edge_records: list[Mapping[str, Any]] = []
    edge_ids: set[str] = set()
    for index, edge in enumerate(edges):
        if not isinstance(edge, Mapping):
            raise TypeError(f"Task graph edge {index} must be a mapping.")
        edge_id = edge.get("id")
        source = edge.get("source")
        target = edge.get("target")
        if not isinstance(edge_id, str) or not edge_id:
            raise ValueError(f"Task graph edge {index} requires a non-empty id.")
        if edge_id in edge_ids:
            raise ValueError(f"Duplicate task graph edge id: {edge_id!r}.")
        if source not in node_ids or target not in node_ids:
            raise ValueError(
                f"Task graph edge {edge_id!r} references an unknown endpoint."
            )
        if graph.has_edge(source, target):
            raise ValueError(
                f"Task graph contains parallel edges from {source!r} to {target!r}."
            )
        edge_ids.add(edge_id)
        edge_records.append(edge)
        graph.add_edge(source, target, edge_id=edge_id)

    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("Task graph must be a directed acyclic graph.")
    return graph, node_records, edge_records


def _task_positions(
    graph: nx.DiGraph,
    task_graph: Mapping[str, Any],
) -> tuple[dict[str, tuple[float, float]], bool]:
    if _is_single_chain(graph, task_graph):
        ordered_nodes = list(nx.topological_sort(graph))
        positions: dict[str, tuple[float, float]] = {}
        for index, node_id in enumerate(ordered_nodes):
            row = index // _TASK_CHAIN_COLUMNS
            column = index % _TASK_CHAIN_COLUMNS
            if row % 2:
                column = _TASK_CHAIN_COLUMNS - 1 - column
            positions[node_id] = (column * 3.6, -row * 3.4)
        return positions, True

    positions = {}
    generations = list(nx.topological_generations(graph))
    for generation_index, generation in enumerate(generations):
        ordered_generation = sorted(generation)
        generation_width = len(ordered_generation)
        for column, node_id in enumerate(ordered_generation):
            centered_column = column - (generation_width - 1) / 2.0
            positions[node_id] = (centered_column * 3.6, -generation_index * 3.4)
    return positions, False


def _is_single_chain(graph: nx.DiGraph, task_graph: Mapping[str, Any]) -> bool:
    if graph.number_of_edges() != graph.number_of_nodes() - 1:
        return False
    if not nx.is_weakly_connected(graph):
        return False
    if any(graph.in_degree(node_id) > 1 for node_id in graph):
        return False
    if any(graph.out_degree(node_id) > 1 for node_id in graph):
        return False
    starts = [node_id for node_id in graph if graph.in_degree(node_id) == 0]
    goals = [node_id for node_id in graph if graph.out_degree(node_id) == 0]
    if len(starts) != 1 or len(goals) != 1:
        return False
    configured_start = task_graph.get("start")
    configured_goal = task_graph.get("goal")
    return configured_start in (None, starts[0]) and configured_goal in (None, goals[0])


def _edge_semantic_step_ids(
    task_graph: Mapping[str, Any],
    edge_records: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    known_edge_ids = {str(edge["id"]) for edge in edge_records}
    result: dict[str, str] = {}
    semantic_steps = task_graph.get("semantic_steps", [])
    if semantic_steps is None:
        return result
    if not isinstance(semantic_steps, list):
        raise TypeError("Task graph semantic_steps must be a list when present.")
    for index, step in enumerate(semantic_steps):
        if not isinstance(step, Mapping):
            raise TypeError(f"Task graph semantic step {index} must be a mapping.")
        step_id = step.get("id")
        edge_ids = step.get("edge_ids")
        if not isinstance(step_id, str) or not step_id:
            raise ValueError(f"Task graph semantic step {index} requires an id.")
        if not isinstance(edge_ids, list):
            raise TypeError(
                f"Task graph semantic step {step_id!r} edge_ids must be a list."
            )
        for edge_id in edge_ids:
            if edge_id not in known_edge_ids:
                raise ValueError(
                    f"Semantic step {step_id!r} references unknown edge {edge_id!r}."
                )
            if edge_id in result:
                raise ValueError(
                    f"Task graph edge {edge_id!r} belongs to multiple semantic steps."
                )
            result[edge_id] = step_id
    return result


def _node_semantic_step_ids(
    edge_records: Sequence[Mapping[str, Any]],
    edge_step_ids: Mapping[str, str],
) -> dict[str, str]:
    result: dict[str, str] = {}
    for edge in edge_records:
        step_id = edge_step_ids.get(str(edge["id"]))
        if step_id is not None:
            result[str(edge["target"])] = step_id
    return result


def _draw_seed_edges(
    axis: Any,
    graph: nx.MultiDiGraph,
    positions: Mapping[str, tuple[float, float]],
) -> None:
    relation_styles = {
        "starts": (_DEPENDENCY_EDGE_COLOR, 0.0),
        "depends_on": (_DEPENDENCY_EDGE_COLOR, 0.22),
        "acts_on": (_NEUTRAL_EDGE_COLOR, 0.0),
        "references": (_REFERENCE_EDGE_COLOR, -0.12),
        "member": (_NEUTRAL_EDGE_COLOR, 0.08),
        "verifies": (_POSTCONDITION_EDGE_COLOR, 0.0),
    }
    for source, target, _, attributes in graph.edges(keys=True, data=True):
        relation = str(attributes["relation"])
        color, curvature = relation_styles[relation]
        _draw_arrow(
            axis,
            positions[source],
            positions[target],
            color=color,
            label=str(attributes["label"]),
            curvature=curvature,
            font_size=6.5,
        )


def _draw_nodes(
    axis: Any,
    graph: nx.MultiDiGraph,
    positions: Mapping[str, tuple[float, float]],
    *,
    seed: bool,
) -> None:
    colors = {
        "program": _PROGRAM_COLOR,
        "semantic_step": _STEP_COLOR,
        "entity": _ENTITY_COLOR,
        "postcondition": _POSTCONDITION_COLOR,
    }
    for node_id, attributes in graph.nodes(data=True):
        kind = str(attributes["kind"])
        dark = kind == "program"
        axis.text(
            *positions[node_id],
            str(attributes["label"]),
            ha="center",
            va="center",
            color="#FFFFFF" if dark else _TEXT_COLOR,
            fontproperties=_font(
                size=7.4 if seed else 7.0,
                weight="bold" if kind in {"program", "semantic_step"} else "normal",
            ),
            bbox={
                "boxstyle": "round,pad=0.55,rounding_size=0.18",
                "facecolor": colors[kind],
                "edgecolor": "#60717E",
                "linewidth": 1.0,
            },
            zorder=3,
        )


def _draw_task_edges(
    axis: Any,
    edge_records: Sequence[Mapping[str, Any]],
    positions: Mapping[str, tuple[float, float]],
    edge_step_ids: Mapping[str, str],
) -> None:
    for index, edge in enumerate(edge_records):
        source = str(edge["source"])
        target = str(edge["target"])
        color = _task_edge_color(edge)
        label = _task_edge_label(edge, edge_step_ids.get(str(edge["id"])))
        same_row = math.isclose(positions[source][1], positions[target][1])
        curvature = 0.0 if same_row else (0.08 if index % 2 == 0 else -0.08)
        _draw_arrow(
            axis,
            positions[source],
            positions[target],
            color=color,
            label=label,
            curvature=curvature,
            font_size=5.8,
        )


def _draw_task_nodes(
    axis: Any,
    node_records: Sequence[Mapping[str, Any]],
    positions: Mapping[str, tuple[float, float]],
    *,
    task_graph: Mapping[str, Any],
    node_step_ids: Mapping[str, str],
) -> None:
    start_id = task_graph.get("start")
    goal_id = task_graph.get("goal")
    step_color_indices: dict[str, int] = {}
    for node in node_records:
        node_id = str(node["id"])
        step_id = node_step_ids.get(node_id)
        if step_id is not None and step_id not in step_color_indices:
            step_color_indices[step_id] = len(step_color_indices)
        if node_id == start_id:
            face_color = _START_COLOR
            text_color = "#FFFFFF"
        elif node_id == goal_id:
            face_color = _GOAL_COLOR
            text_color = "#FFFFFF"
        elif step_id is not None:
            face_color = _SEMANTIC_STEP_COLORS[
                step_color_indices[step_id] % len(_SEMANTIC_STEP_COLORS)
            ]
            text_color = _TEXT_COLOR
        else:
            face_color = _DEFAULT_NODE_COLOR
            text_color = _TEXT_COLOR

        semantic = str(node.get("semantic", ""))
        label = _wrapped_label(
            _short_identifier(node_id),
            _wrap_and_truncate(semantic, width=22, limit=52),
            width=22,
        )
        axis.text(
            *positions[node_id],
            label,
            ha="center",
            va="center",
            color=text_color,
            fontproperties=_font(size=6.8, weight="bold"),
            bbox={
                "boxstyle": "round,pad=0.5,rounding_size=0.16",
                "facecolor": face_color,
                "edgecolor": "#60717E",
                "linewidth": 1.0,
            },
            zorder=3,
        )


def _draw_arrow(
    axis: Any,
    source: tuple[float, float],
    target: tuple[float, float],
    *,
    color: str,
    label: str,
    curvature: float,
    font_size: float,
) -> None:
    arrow = FancyArrowPatch(
        source,
        target,
        arrowstyle="-|>",
        mutation_scale=11,
        linewidth=1.35,
        color=color,
        connectionstyle=f"arc3,rad={curvature}",
        shrinkA=48,
        shrinkB=48,
        zorder=1,
    )
    axis.add_patch(arrow)
    midpoint_x = (source[0] + target[0]) / 2.0
    midpoint_y = (source[1] + target[1]) / 2.0
    if math.isclose(source[1], target[1]):
        midpoint_y += 0.28
    else:
        midpoint_x += 0.22 if curvature >= 0 else -0.22
    axis.text(
        midpoint_x,
        midpoint_y,
        label,
        ha="center",
        va="center",
        color=_MUTED_TEXT_COLOR,
        fontproperties=_font(size=font_size),
        bbox={
            "boxstyle": "round,pad=0.15",
            "facecolor": "#FFFFFF",
            "edgecolor": "none",
            "alpha": 0.92,
        },
        zorder=2,
    )


def _draw_seed_legend(axis: Any) -> None:
    handles = [
        Patch(facecolor=_PROGRAM_COLOR, edgecolor="#60717E", label="Program"),
        Patch(facecolor=_STEP_COLOR, edgecolor="#60717E", label="Semantic step"),
        Patch(facecolor=_ENTITY_COLOR, edgecolor="#60717E", label="Object/reference"),
        Patch(
            facecolor=_POSTCONDITION_COLOR,
            edgecolor="#60717E",
            label="Postcondition",
        ),
    ]
    axis.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=4,
        frameon=False,
        prop=_font(size=7),
    )


def _draw_task_legend(axis: Any) -> None:
    handles = [
        Patch(facecolor=_START_COLOR, edgecolor="#60717E", label="Start"),
        Patch(facecolor=_GOAL_COLOR, edgecolor="#60717E", label="Goal"),
        Line2D([0], [0], color=_LEFT_EDGE_COLOR, lw=2, label="Left arm"),
        Line2D([0], [0], color=_RIGHT_EDGE_COLOR, lw=2, label="Right arm"),
        Line2D([0], [0], color=_DUAL_EDGE_COLOR, lw=2, label="Dual arm"),
    ]
    axis.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.09),
        ncol=5,
        frameon=False,
        prop=_font(size=7),
    )


def _task_edge_color(edge: Mapping[str, Any]) -> str:
    has_left = isinstance(edge.get(LEFT_ARM_ACTION_KEY), Mapping)
    has_right = isinstance(edge.get(RIGHT_ARM_ACTION_KEY), Mapping)
    if has_left and has_right:
        return _DUAL_EDGE_COLOR
    if has_left:
        return _LEFT_EDGE_COLOR
    if has_right:
        return _RIGHT_EDGE_COLOR
    return _NEUTRAL_EDGE_COLOR


def _task_edge_label(
    edge: Mapping[str, Any],
    semantic_step_id: str | None,
) -> str:
    lines = [_short_identifier(str(edge["id"]))]
    if semantic_step_id is not None:
        lines.append(_short_identifier(semantic_step_id))
    for prefix, action_key in (
        ("[L]", LEFT_ARM_ACTION_KEY),
        ("[R]", RIGHT_ARM_ACTION_KEY),
    ):
        action = edge.get(action_key)
        if not isinstance(action, Mapping):
            continue
        action_class = str(action.get("atomic_action_class", "unknown"))
        lines.append(f"{prefix} {_wrap_and_truncate(action_class, width=20, limit=28)}")
        target_uid = _action_target_uid(action)
        if target_uid is not None:
            lines.append(
                f"target: {_wrap_and_truncate(target_uid, width=20, limit=34)}"
            )
    if len(lines) == 1:
        lines.append("[?] no arm action")
    return "\n".join(lines)


def _action_target_uid(action: Mapping[str, Any]) -> str | None:
    target_object = action.get("target_object")
    if not isinstance(target_object, Mapping):
        return None
    object_name = target_object.get("obj_name")
    if isinstance(object_name, str) and object_name:
        return object_name
    return None


def _seed_actor_text(actor: Mapping[str, Any]) -> str:
    mode = str(actor.get("mode", "unknown"))
    if mode in {"required", "assigned"}:
        arm = str(actor.get("arm", "unknown"))
        return f"{mode}: {arm}"
    if mode == "coordinated":
        arms = actor.get("arms", [])
        if isinstance(arms, Sequence) and not isinstance(arms, (str, bytes)):
            return f"coordinated: {', '.join(str(arm) for arm in arms)}"
    return mode


def _postcondition_label(postcondition: Mapping[str, Any]) -> str:
    condition_type = postcondition.get("type", postcondition.get("op", "condition"))
    relation = postcondition.get("relation")
    details = [str(condition_type)]
    if relation is not None:
        details.append(f"relation: {relation}")
    if "layer_index" in postcondition:
        details.append(f"layer: {postcondition['layer_index']}")
    return _wrapped_label("postcondition", *details, width=24)


def _short_identifier(identifier: str) -> str:
    prefix = identifier.split("_", 1)[0]
    if prefix and len(prefix) <= 12:
        return prefix
    return _wrap_and_truncate(identifier, width=18, limit=24)


def _wrapped_label(*parts: str, width: int) -> str:
    return "\n".join(
        _wrap_and_truncate(part, width=width, limit=max(width * 2, 36))
        for part in parts
        if part
    )


def _wrap_and_truncate(value: str, *, width: int, limit: int) -> str:
    normalized = " ".join(str(value).replace("`", "").split())
    if len(normalized) > limit:
        normalized = normalized[: max(1, limit - 3)].rstrip() + "..."
    return "\n".join(
        textwrap.wrap(
            normalized,
            width=width,
            break_long_words=True,
            break_on_hyphens=False,
        )
    )


def _make_figure(*, width: float, height: float) -> Figure:
    figure = Figure(
        figsize=(width, height),
        facecolor=_BACKGROUND_COLOR,
        constrained_layout=False,
    )
    FigureCanvasAgg(figure)
    return figure


def _figure_png_bytes(figure: Figure) -> bytes:
    buffer = BytesIO()
    figure.savefig(
        buffer,
        format="png",
        dpi=_PNG_DPI,
        bbox_inches="tight",
        facecolor=_BACKGROUND_COLOR,
        metadata={"Software": "EmbodiChain action-agent pipeline"},
    )
    return buffer.getvalue()


def _fit_axis_to_positions(
    axis: Any,
    positions: Mapping[str, tuple[float, float]],
    *,
    x_margin: float,
    y_margin: float,
) -> None:
    x_values = [position[0] for position in positions.values()]
    y_values = [position[1] for position in positions.values()]
    axis.set_xlim(min(x_values) - x_margin, max(x_values) + x_margin)
    axis.set_ylim(min(y_values) - y_margin, max(y_values) + y_margin)


def _topological_generation_count(graph: nx.DiGraph) -> int:
    return max(1, sum(1 for _ in nx.topological_generations(graph)))


def _max_topological_generation_width(graph: nx.DiGraph) -> int:
    return max(
        1,
        max(len(generation) for generation in nx.topological_generations(graph)),
    )


def _font(*, size: float, weight: str = "normal") -> FontProperties:
    return FontProperties(family=_font_family(), size=size, weight=weight)


@lru_cache(maxsize=1)
def _font_family() -> str:
    available = {font.name for font in fontManager.ttflist}
    for candidate in (
        "Noto Sans CJK SC",
        "Noto Sans CJK TC",
        "Noto Sans CJK JP",
        "WenQuanYi Micro Hei",
        "DejaVu Sans",
    ):
        if candidate in available:
            return candidate
    return "sans-serif"
