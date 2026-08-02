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

"""Headless PNG rendering for Action Engine execution programs.

The renderer deliberately consumes the same validated, coordinate-free
``action_engine_execution_program_v1`` document as runtime. It does not define
another graph schema and never grounds symbolic targets. Seed programs use a
compact folded timeline when they are a single chain; genuine DAGs use stable
actor swimlanes so forks, joins, and dependency constraints remain visible.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from typing import Any

import matplotlib

# Select the non-interactive backend before importing any canvas primitives.
matplotlib.use("Agg", force=True)

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.figure import Figure
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
import networkx as nx

from embodichain.gen_sim.action_engine.domain import (
    EXECUTION_PROGRAM_SCHEMA,
    validate_execution_program,
)

__all__ = ["render_seed_task_graph_png", "render_task_graph_png"]

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_EXECUTION_KEYS = frozenset(
    {
        "schema_version",
        "task",
        "goal_description",
        "start",
        "goal",
        "nodes",
        "edges",
        "semantic_steps",
        "allocation_groups",
        "motion_policy_version",
    }
)

_BACKGROUND = "#F8FAFB"
_INK = "#17212B"
_MUTED = "#66727D"
_BORDER = "#CBD4DC"
_LEFT = "#168A78"
_RIGHT = "#D97706"
_AUTO = "#59636D"
_COORDINATED = "#7652A5"
_DEPENDENCY = "#3973B7"
_SUCCESS = "#25834B"
_FAILED = "#C43E3E"
_SKIPPED = "#8B949C"
_LANE_COLORS = {
    "left": _LEFT,
    "auto": _AUTO,
    "right": _RIGHT,
    "coordinated": _COORDINATED,
}
_LANE_BACKGROUNDS = {
    "left": "#EAF6F3",
    "auto": "#F0F3F5",
    "right": "#FFF4E6",
}
_LANE_LABELS = {
    "left": "LEFT ARM  [L]",
    "auto": "WORLD / AUTO / COORDINATED",
    "right": "RIGHT ARM  [R]",
}
_STATUS_COLORS = {
    "success": _SUCCESS,
    "executed": _SUCCESS,
    "failed": _FAILED,
    "aborted": _FAILED,
    "skipped": _SKIPPED,
}
_STATUS_BADGES = {
    "success": "OK",
    "executed": "OK",
    "failed": "FAIL",
    "aborted": "ABORT",
    "skipped": "SKIP",
}


@dataclass(frozen=True)
class _RuntimeOverlay:
    """Execution annotations kept separate from the immutable seed program."""

    edge_status: Mapping[str, str]
    edge_arm: Mapping[str, str]
    step_status: Mapping[str, str]
    graph_status: str | None = None


@dataclass(frozen=True)
class _GraphData:
    """Validated program plus indices shared by both layout strategies."""

    program: Mapping[str, Any]
    graph: nx.MultiDiGraph
    node_by_id: Mapping[str, Mapping[str, Any]]
    edge_by_id: Mapping[str, Mapping[str, Any]]
    step_by_id: Mapping[str, Mapping[str, Any]]
    lane_override: Mapping[str, str]
    runtime: _RuntimeOverlay


def render_seed_task_graph_png(seed_graph: Mapping[str, Any]) -> bytes:
    """Render an ``action_engine_execution_program_v1`` through headless Agg."""
    program = validate_execution_program(seed_graph)
    return _render(program, _RuntimeOverlay({}, {}, {}))


def render_task_graph_png(task_graph: Mapping[str, Any]) -> bytes:
    """Render an execution program with optional runtime event annotations.

    A bare program is accepted. Runtime events may be stored in its ``runtime``
    envelope, or beside a nested ``execution_program``, ``program``, or
    ``seed_task_graph``. A record alone is rejected because it omits topology.
    """
    program = _extract_execution_program(task_graph)
    runtime = _extract_runtime_overlay(task_graph)
    return _render(program, runtime)


def _render(
    program: Mapping[str, Any],
    runtime: _RuntimeOverlay,
) -> bytes:
    data = _graph_data(program, runtime)
    if _is_single_chain(data):
        return _render_chain(data)
    return _render_dag(data)


def _extract_execution_program(document: Mapping[str, Any]) -> dict[str, Any]:
    """Find and validate the execution program embedded in a display document."""
    if not isinstance(document, Mapping):
        raise ValueError("Task graph visualization input must be a mapping.")

    if document.get("schema_version") == EXECUTION_PROGRAM_SCHEMA:
        # A runtime artifact may preserve the program fields and add annotations.
        candidate = {key: document[key] for key in _EXECUTION_KEYS if key in document}
        return validate_execution_program(candidate)

    for key in ("execution_program", "program", "seed_task_graph"):
        candidate = document.get(key)
        if isinstance(candidate, Mapping):
            return validate_execution_program(candidate)

    # Supporting a full program plus a runtime schema at the top level keeps
    # visualization useful for simple JSON joins without weakening validation.
    if {"nodes", "edges", "semantic_steps"}.issubset(document):
        candidate = {key: document[key] for key in _EXECUTION_KEYS if key in document}
        return validate_execution_program(candidate)

    raise ValueError(
        "Runtime records do not contain graph topology. Provide the matching "
        "ExecutionProgram under 'execution_program', 'program', or "
        "'seed_task_graph'."
    )


def _extract_runtime_overlay(document: Mapping[str, Any]) -> _RuntimeOverlay:
    """Reduce a runtime record to the small set of display-only annotations."""
    record = document.get("runtime")
    if record is None:
        record = document.get("runtime_record", document)
    if not isinstance(record, Mapping):
        raise ValueError("runtime_record must be a mapping.")
    raw_events = record.get("events", document.get("events", []))
    if not isinstance(raw_events, Sequence) or isinstance(
        raw_events, (str, bytes, bytearray)
    ):
        raise ValueError("Runtime events must be a list.")

    edge_status: dict[str, str] = {}
    edge_arm: dict[str, str] = {}
    step_status: dict[str, str] = {}
    for index, event in enumerate(raw_events):
        if not isinstance(event, Mapping):
            raise ValueError(f"Runtime events[{index}] must be a mapping.")
        event_kind = event.get("event")
        status = _optional_text(event.get("status"))
        if event_kind == "edge":
            edge_id = _optional_text(event.get("edge_id"))
            if edge_id and status:
                edge_status[edge_id] = status.lower()
            arm = _optional_text(event.get("arm"))
            if edge_id and arm:
                edge_arm[edge_id] = arm
        elif event_kind == "semantic_step":
            step_id = _optional_text(event.get("semantic_step_id"))
            if step_id and status:
                step_status[step_id] = status.lower()

    graph_status = _optional_text(record.get("status"))
    return _RuntimeOverlay(
        edge_status=edge_status,
        edge_arm=edge_arm,
        step_status=step_status,
        graph_status=graph_status.lower() if graph_status else None,
    )


def _graph_data(
    program: Mapping[str, Any],
    runtime: _RuntimeOverlay,
) -> _GraphData:
    node_by_id = {str(node["id"]): node for node in program["nodes"]}
    edge_by_id = {str(edge["id"]): edge for edge in program["edges"]}
    step_by_id = {str(step["id"]): step for step in program["semantic_steps"]}
    graph = nx.MultiDiGraph()
    graph.add_nodes_from(node_by_id)
    for edge in program["edges"]:
        source = str(edge["source"])
        target = str(edge["target"])
        graph.add_edge(source, target, edge_id=str(edge["id"]))
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("ExecutionProgram node topology must be a directed DAG.")

    return _GraphData(
        program=program,
        graph=graph,
        node_by_id=node_by_id,
        edge_by_id=edge_by_id,
        step_by_id=step_by_id,
        lane_override=_allocation_lane_overrides(program),
        runtime=runtime,
    )


def _allocation_lane_overrides(
    program: Mapping[str, Any],
) -> dict[str, str]:
    """Give auto actors stable lanes when a distinct-arm group is declared."""
    result: dict[str, str] = {}
    for group in program.get("allocation_groups", []):
        if group.get("arm_constraint") != "distinct_arms":
            continue
        members = group.get("semantic_step_ids", [])
        for index, step_id in enumerate(members):
            result[str(step_id)] = "left" if index % 2 == 0 else "right"
    return result


def _is_single_chain(data: _GraphData) -> bool:
    graph = data.graph
    if graph.number_of_edges() != graph.number_of_nodes() - 1:
        return False
    if any(graph.in_degree(node) > 1 for node in graph):
        return False
    if any(graph.out_degree(node) > 1 for node in graph):
        return False
    return (
        graph.in_degree(str(data.program["start"])) == 0
        and graph.out_degree(str(data.program["goal"])) == 0
        and nx.is_weakly_connected(graph)
    )


def _ordered_chain_edges(data: _GraphData) -> list[Mapping[str, Any]]:
    current = str(data.program["start"])
    result: list[Mapping[str, Any]] = []
    while current != str(data.program["goal"]):
        outgoing = list(data.graph.out_edges(current, data=True))
        if len(outgoing) != 1:
            raise ValueError("ExecutionProgram chain has an incomplete path.")
        _, target, attrs = outgoing[0]
        result.append(data.edge_by_id[str(attrs["edge_id"])])
        current = str(target)
    if len(result) != len(data.edge_by_id):
        raise ValueError("ExecutionProgram chain does not cover every edge.")
    return result


def _render_chain(data: _GraphData) -> bytes:
    """Render a long linear program as a bounded, folded state timeline."""
    edges = _ordered_chain_edges(data)
    nodes = [str(data.program["start"])]
    nodes.extend(str(edge["target"]) for edge in edges)

    slots_per_row = 5
    row_count = (len(nodes) + slots_per_row - 1) // slots_per_row
    width = 16.0
    height = max(4.8, 2.6 + row_count * 2.1)
    figure, axis = _new_figure(width, height)
    try:
        _draw_header(axis, data, width)
        left, right, first_y = 1.0, width - 1.0, 2.15
        spacing = (right - left) / (slots_per_row - 1)
        positions: dict[str, tuple[float, float]] = {}
        for index, node_id in enumerate(nodes):
            row, column = divmod(index, slots_per_row)
            visual_column = column if row % 2 == 0 else slots_per_row - 1 - column
            positions[node_id] = (
                left + visual_column * spacing,
                first_y + row * 2.05,
            )

        for edge in edges:
            source = positions[str(edge["source"])]
            target = positions[str(edge["target"])]
            lane = _edge_lane(edge, data)
            color = _edge_color(str(edge["id"]), lane, data.runtime)
            midpoint = _midpoint(source, target)
            vertical = abs(source[0] - target[0]) < 0.1
            _draw_labeled_edge(
                axis,
                source,
                target,
                color=color,
                label=_edge_label(edge, data),
                label_position=(
                    (midpoint[0] - 1.48, midpoint[1])
                    if vertical
                    else (midpoint[0], midpoint[1] - 0.38)
                ),
                font_size=5.4,
            )

        for index, node_id in enumerate(nodes):
            _draw_state_node(
                axis,
                node_id,
                data.node_by_id[node_id],
                positions[node_id],
                start=node_id == str(data.program["start"]),
                goal=node_id == str(data.program["goal"]),
                fork=False,
                join=False,
                index=index,
            )

        return _figure_png_bytes(figure)
    finally:
        figure.clear()


def _render_dag(data: _GraphData) -> bytes:
    """Render forks and joins against persistent actor swimlanes."""
    levels = _dag_levels(data.graph)
    maximum_level = max(levels.values(), default=0)
    width = 15.6
    height = max(6.0, 3.5 + maximum_level * 2.15)
    figure, axis = _new_figure(width, height)
    try:
        _draw_header(axis, data, width)
        lane_centers = {"left": 2.6, "auto": 7.8, "right": 13.0}
        _draw_swimlanes(axis, width, height, lane_centers)
        positions = _dag_positions(data, levels, lane_centers)

        # Dependency arrows are drawn first and remain visibly distinct from
        # physical state transitions through color and dash pattern.
        edge_midpoints = {
            edge_id: _midpoint(
                positions[str(edge["source"])],
                positions[str(edge["target"])],
            )
            for edge_id, edge in data.edge_by_id.items()
        }
        for prerequisite_id, dependent_id in _dependency_pairs(data):
            _draw_dependency_arrow(
                axis,
                edge_midpoints[prerequisite_id],
                edge_midpoints[dependent_id],
            )

        pair_groups: defaultdict[tuple[str, str], list[str]] = defaultdict(list)
        for edge in data.edge_by_id.values():
            pair_groups[(str(edge["source"]), str(edge["target"]))].append(
                str(edge["id"])
            )
        for edge in data.edge_by_id.values():
            edge_id = str(edge["id"])
            source_id = str(edge["source"])
            target_id = str(edge["target"])
            lane = _edge_lane(edge, data)
            midpoint = _midpoint(positions[source_id], positions[target_id])
            direction = -1.0 if midpoint[0] < 7.8 else 1.0
            if abs(positions[source_id][0] - positions[target_id][0]) < 0.4:
                direction = 1.0
            parallel_ids = pair_groups[(source_id, target_id)]
            parallel_index = parallel_ids.index(edge_id)
            curvature = (parallel_index - (len(parallel_ids) - 1) / 2.0) * 0.20
            _draw_labeled_edge(
                axis,
                positions[source_id],
                positions[target_id],
                color=_edge_color(edge_id, lane, data.runtime),
                label=_edge_label(edge, data),
                label_position=(
                    midpoint[0] + 0.52 * direction + curvature * 3.4,
                    midpoint[1] - 0.08,
                ),
                font_size=5.15,
                curvature=curvature,
            )

        for index, node_id in enumerate(nx.topological_sort(data.graph)):
            _draw_state_node(
                axis,
                str(node_id),
                data.node_by_id[str(node_id)],
                positions[str(node_id)],
                start=str(node_id) == str(data.program["start"]),
                goal=str(node_id) == str(data.program["goal"]),
                fork=data.graph.out_degree(node_id) > 1,
                join=data.graph.in_degree(node_id) > 1,
                index=index,
            )

        return _figure_png_bytes(figure)
    finally:
        figure.clear()


def _dag_levels(graph: nx.MultiDiGraph) -> dict[str, int]:
    """Assign the longest-path depth so dependencies always flow downward."""
    levels: dict[str, int] = {}
    for node in nx.topological_sort(graph):
        predecessors = list(graph.predecessors(node))
        levels[str(node)] = (
            max(levels[str(parent)] for parent in predecessors) + 1
            if predecessors
            else 0
        )
    return levels


def _dag_positions(
    data: _GraphData,
    levels: Mapping[str, int],
    lane_centers: Mapping[str, float],
) -> dict[str, tuple[float, float]]:
    """Place branch nodes in actor lanes and structural fork/join nodes centrally."""
    base: dict[str, tuple[str, int]] = {}
    for node_id in data.node_by_id:
        incoming = list(data.graph.in_edges(node_id, data=True))
        outgoing = list(data.graph.out_edges(node_id, data=True))
        if (
            node_id in {str(data.program["start"]), str(data.program["goal"])}
            or len(incoming) > 1
            or len(outgoing) > 1
        ):
            lane = "auto"
        elif incoming:
            edge = data.edge_by_id[str(incoming[0][2]["edge_id"])]
            lane = _edge_lane(edge, data)
        elif outgoing:
            edge = data.edge_by_id[str(outgoing[0][2]["edge_id"])]
            lane = _edge_lane(edge, data)
        else:
            lane = "auto"
        if lane == "coordinated":
            lane = "auto"
        base[node_id] = (lane, levels[node_id])

    groups: defaultdict[tuple[str, int], list[str]] = defaultdict(list)
    for node_id, lane_level in base.items():
        groups[lane_level].append(node_id)

    result: dict[str, tuple[float, float]] = {}
    for (lane, level), node_ids in groups.items():
        ordered = sorted(node_ids)
        center = lane_centers[lane]
        # Small symmetric offsets prevent same-level nodes from hiding each
        # other while keeping every node visibly inside its actor lane.
        offsets = [
            (index - (len(ordered) - 1) / 2.0) * 0.72 for index in range(len(ordered))
        ]
        for node_id, offset in zip(ordered, offsets, strict=True):
            result[node_id] = (center + offset, 2.35 + level * 2.15)
    return result


def _dependency_pairs(data: _GraphData) -> list[tuple[str, str]]:
    """Return explicit edge dependencies plus missing semantic dependencies."""
    result: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for edge in data.edge_by_id.values():
        dependent_id = str(edge["id"])
        for prerequisite_id in edge.get("depends_on", []):
            pair = (str(prerequisite_id), dependent_id)
            if pair not in seen:
                seen.add(pair)
                result.append(pair)

    for step in data.step_by_id.values():
        dependent_edges = step.get("edge_ids", [])
        if not dependent_edges:
            continue
        for prerequisite_step_id in step.get("depends_on", []):
            prerequisite = data.step_by_id[str(prerequisite_step_id)]
            pair = (
                str(prerequisite["edge_ids"][-1]),
                str(dependent_edges[0]),
            )
            if pair not in seen:
                seen.add(pair)
                result.append(pair)
    return result


def _edge_lane(edge: Mapping[str, Any], data: _GraphData) -> str:
    edge_id = str(edge["id"])
    observed_arm = data.runtime.edge_arm.get(edge_id)
    if observed_arm:
        return _arm_lane(observed_arm)

    action_lanes = {
        _actor_lane(action.get("actor", {})) for action in edge.get("actions", [])
    }
    action_lanes.discard("auto")
    if action_lanes == {"left"}:
        return "left"
    if action_lanes == {"right"}:
        return "right"
    if "coordinated" in action_lanes or action_lanes == {"left", "right"}:
        return "coordinated"
    return data.lane_override.get(str(edge["semantic_step_id"]), "auto")


def _actor_lane(actor: Any) -> str:
    if not isinstance(actor, Mapping):
        return "auto"
    mode = str(actor.get("mode", "auto")).lower()
    if mode == "required":
        return _arm_lane(str(actor.get("arm", "")))
    if mode == "coordinated":
        return "coordinated"
    return "auto"


def _arm_lane(arm: str) -> str:
    normalized = arm.strip().lower()
    if "left" in normalized:
        return "left"
    if "right" in normalized:
        return "right"
    if normalized in {"both", "coordinated", "dual_arm", "dual"}:
        return "coordinated"
    return "auto"


def _edge_color(
    edge_id: str,
    lane: str,
    runtime: _RuntimeOverlay,
) -> str:
    status = runtime.edge_status.get(edge_id)
    return _STATUS_COLORS.get(status or "", _LANE_COLORS[lane])


def _edge_label(edge: Mapping[str, Any], data: _GraphData) -> str:
    edge_id = str(edge["id"])
    step = data.step_by_id[str(edge["semantic_step_id"])]
    lane = _edge_lane(edge, data)
    badge = {
        "left": "L",
        "right": "R",
        "coordinated": "LR",
        "auto": "A",
    }[lane]
    status = data.runtime.edge_status.get(edge_id) or data.runtime.step_status.get(
        str(step["id"])
    )
    status_badge = f" [{_STATUS_BADGES.get(status, status.upper())}]" if status else ""
    action_names = [
        str(action.get("atomic_action_class", "action"))
        for action in edge.get("actions", [])
    ]
    action_text = " + ".join(action_names[:2])
    if len(action_names) > 2:
        action_text += f" +{len(action_names) - 2}"
    action = edge["actions"][0]
    binding = _binding_summary(action.get("target_binding", {}))
    policy = _clip(str(action.get("motion_policy", "")), 28)
    semantic = f"{step['operator']} : {step['object']}"
    return "\n".join(
        (
            _clip(f"{edge_id} [{badge}]{status_badge}", 34),
            _clip(f"{action_text} | {semantic}", 42),
            _clip(f"{binding} | {policy}", 42),
        )
    )


def _binding_summary(value: Any) -> str:
    if not isinstance(value, Mapping):
        return "symbolic target"
    kind = str(value.get("kind", "target"))
    details: list[str] = []
    for key in (
        "object",
        "reference_object",
        "support_object",
        "relation",
        "phase",
        "slot",
        "layer",
    ):
        if key in value:
            details.append(f"{key}={value[key]}")
        if len(details) == 2:
            break
    return f"{kind} ({', '.join(details)})" if details else kind


def _draw_header(axis: Any, data: _GraphData, width: float) -> None:
    status = data.runtime.graph_status
    status_text = f"  [{status.upper()}]" if status else ""
    axis.text(
        0.55,
        0.45,
        _clip(f"ACTION ENGINE / {data.program['task']}{status_text}", 84),
        ha="left",
        va="center",
        color=_INK,
        fontproperties=_font(13.0, "bold"),
        zorder=20,
    )
    axis.text(
        0.55,
        0.90,
        _clip(str(data.program["goal_description"]), 115),
        ha="left",
        va="top",
        color=_MUTED,
        fontproperties=_font(7.6),
        linespacing=1.25,
        zorder=20,
    )
    axis.plot(
        [0.55, width - 0.55],
        [1.35, 1.35],
        color=_BORDER,
        linewidth=0.8,
        zorder=19,
    )


def _draw_swimlanes(
    axis: Any,
    width: float,
    height: float,
    centers: Mapping[str, float],
) -> None:
    boundaries = {
        "left": (0.55, 5.15),
        "auto": (5.25, 10.35),
        "right": (10.45, width - 0.55),
    }
    for lane in ("left", "auto", "right"):
        left, right = boundaries[lane]
        axis.add_patch(
            FancyBboxPatch(
                (left, 1.50),
                right - left,
                height - 2.0,
                boxstyle="round,pad=0.0,rounding_size=0.05",
                facecolor=_LANE_BACKGROUNDS[lane],
                edgecolor=_BORDER,
                linewidth=0.7,
                zorder=-10,
            )
        )
        axis.plot(
            [left, right],
            [1.50, 1.50],
            color=_LANE_COLORS[lane],
            linewidth=2.3,
            zorder=-9,
        )
        axis.text(
            centers[lane],
            1.76,
            _LANE_LABELS[lane],
            ha="center",
            va="center",
            color=_LANE_COLORS[lane],
            fontproperties=_font(7.0, "bold"),
            zorder=10,
        )


def _draw_labeled_edge(
    axis: Any,
    source: tuple[float, float],
    target: tuple[float, float],
    *,
    color: str,
    label: str,
    label_position: tuple[float, float],
    font_size: float,
    curvature: float = 0.0,
) -> None:
    """Draw one solid state transition and its compact symbolic label."""
    axis.add_patch(
        FancyArrowPatch(
            source,
            target,
            arrowstyle="-|>",
            mutation_scale=11,
            color=color,
            linewidth=1.7,
            shrinkA=17,
            shrinkB=17,
            connectionstyle=f"arc3,rad={curvature}",
            zorder=3,
        )
    )
    axis.text(
        *label_position,
        label,
        ha="center",
        va="center",
        color=_INK,
        fontproperties=_font(font_size),
        linespacing=1.12,
        bbox={
            "boxstyle": "round,pad=0.22",
            "facecolor": "#FFFFFF",
            "edgecolor": color,
            "linewidth": 0.55,
            "alpha": 0.96,
        },
        zorder=8,
    )


def _draw_dependency_arrow(
    axis: Any,
    source: tuple[float, float],
    target: tuple[float, float],
) -> None:
    if source == target:
        return
    axis.add_patch(
        FancyArrowPatch(
            source,
            target,
            arrowstyle="-|>",
            mutation_scale=8,
            color=_DEPENDENCY,
            linewidth=1.25,
            linestyle=(0, (2.2, 2.2)),
            shrinkA=5,
            shrinkB=5,
            connectionstyle="arc3,rad=-0.17",
            alpha=0.95,
            zorder=1,
        )
    )
    midpoint = _midpoint(source, target)
    axis.text(
        midpoint[0],
        midpoint[1] + 0.22,
        "DEP",
        ha="center",
        va="center",
        color=_DEPENDENCY,
        fontproperties=_font(4.8, "bold"),
        zorder=2,
    )


def _draw_state_node(
    axis: Any,
    node_id: str,
    node: Mapping[str, Any],
    center: tuple[float, float],
    *,
    start: bool,
    goal: bool,
    fork: bool,
    join: bool,
    index: int,
) -> None:
    fill = "#DDEFEA" if start else ("#E7F2DD" if goal else "#FFFFFF")
    edge = _SUCCESS if goal else (_LEFT if start else _INK)
    radius = 0.29 if (start or goal or fork or join) else 0.24
    axis.add_patch(
        Circle(
            center,
            radius=radius,
            facecolor=fill,
            edgecolor=edge,
            linewidth=1.6,
            zorder=12,
        )
    )
    axis.text(
        center[0],
        center[1],
        str(index),
        ha="center",
        va="center",
        color=_INK,
        fontproperties=_font(6.2, "bold"),
        zorder=13,
    )
    role = (
        "START"
        if start
        else ("GOAL" if goal else ("FORK" if fork else "JOIN" if join else ""))
    )
    semantic = _clip(str(node.get("semantic", node_id)), 27)
    axis.text(
        center[0],
        center[1] + 0.43,
        "\n".join(part for part in (role, semantic) if part),
        ha="center",
        va="top",
        color=edge if role else _MUTED,
        fontproperties=_font(5.3, "bold" if role else "normal"),
        linespacing=1.08,
        zorder=13,
    )


def _new_figure(width: float, height: float) -> tuple[Figure, Any]:
    figure = Figure(figsize=(width, height), dpi=150, facecolor=_BACKGROUND)
    axis = figure.subplots()
    axis.set_facecolor(_BACKGROUND)
    axis.set_axis_off()
    axis.set_xlim(0.0, width)
    axis.set_ylim(height, 0.0)
    return figure, axis


def _figure_png_bytes(figure: Figure) -> bytes:
    buffer = BytesIO()
    FigureCanvasAgg(figure).print_png(buffer)
    payload = buffer.getvalue()
    if not payload.startswith(_PNG_SIGNATURE):
        raise RuntimeError("Matplotlib did not produce a valid PNG payload.")
    return payload


@lru_cache(maxsize=1)
def _font_family() -> str:
    """Prefer a CJK-capable font while retaining a portable fallback."""
    available = {font.name for font in fontManager.ttflist}
    for family in (
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "Source Han Sans CN",
        "WenQuanYi Micro Hei",
        "Microsoft YaHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ):
        if family in available:
            return family
    return "sans-serif"


def _font(size: float, weight: str = "normal") -> FontProperties:
    return FontProperties(family=_font_family(), size=size, weight=weight)


def _optional_text(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _clip(value: str, length: int) -> str:
    return value if len(value) <= length else f"{value[: max(1, length - 3)]}..."


def _midpoint(
    first: tuple[float, float],
    second: tuple[float, float],
) -> tuple[float, float]:
    return ((first[0] + second[0]) / 2.0, (first[1] + second[1]) / 2.0)
