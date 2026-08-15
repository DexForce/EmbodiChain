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

"""Headless PNG rendering for direct AtomicAction SeedGraphs.

The renderer consumes the same validated coordinate-free v3 graph as runtime,
then builds an internal display view without grounding symbolic targets. E
TaskGroups remain the semantic grouping labels over the rendered action nodes.
Single chains use a folded timeline; DAGs use stable actor swimlanes.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from math import hypot
from typing import Any

import matplotlib

# Select the non-interactive backend before importing any canvas primitives.
matplotlib.use("Agg", force=True)

from matplotlib import patheffects
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.figure import Figure
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
import networkx as nx

from embodichain.gen_sim.action_engine.domain import (
    EXECUTION_PROGRAM_SCHEMA,
    validate_execution_program,
)
from embodichain.gen_sim.action_engine.protocol import SEED_GRAPH_SCHEMA

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
_DEPENDENCY = "#8A94A0"

# The figures are designed at this display width in inches; every type size
# below is chosen to stay readable when the PNG is shown at exactly this size.
_TARGET_WIDTH = 8.0
_DPI = 300
_LEVEL_STEP = 1.15
_NODE_RADIUS = 0.16
_SPECIAL_NODE_RADIUS = 0.20
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
    """Render a v3 SeedGraph or package-owned legacy program through Agg."""
    program = _display_program(seed_graph)
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

    if document.get("schema_version") in {EXECUTION_PROGRAM_SCHEMA, SEED_GRAPH_SCHEMA}:
        # A runtime artifact may preserve the program fields and add annotations.
        if document.get("schema_version") == SEED_GRAPH_SCHEMA:
            candidate = dict(document)
            candidate.pop("runtime", None)
            candidate.pop("runtime_record", None)
            return _display_program(candidate)
        candidate = {key: document[key] for key in _EXECUTION_KEYS if key in document}
        return validate_execution_program(candidate)

    for key in ("execution_program", "program", "seed_task_graph"):
        candidate = document.get(key)
        if isinstance(candidate, Mapping):
            return _display_program(candidate)

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


def _display_program(value: Mapping[str, Any]) -> dict[str, Any]:
    if value.get("schema_version") == SEED_GRAPH_SCHEMA:
        from embodichain.gen_sim.action_engine.compiler import (
            seed_graph_to_execution_program,
        )

        return seed_graph_to_execution_program(value, require_executable=False)
    return validate_execution_program(value)


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

    slots_per_row = 4
    row_count = (len(nodes) + slots_per_row - 1) // slots_per_row
    width = _TARGET_WIDTH
    height = max(3.4, 2.0 + row_count * 1.55)
    figure, axis = _new_figure(width, height)
    try:
        _draw_header(axis, data, width)
        left, right, first_y = 0.7, width - 0.7, 2.05
        spacing = (right - left) / (slots_per_row - 1)
        positions: dict[str, tuple[float, float]] = {}
        for index, node_id in enumerate(nodes):
            row, column = divmod(index, slots_per_row)
            visual_column = column if row % 2 == 0 else slots_per_row - 1 - column
            positions[node_id] = (
                left + visual_column * spacing,
                first_y + row * 1.55,
            )

        for edge in edges:
            source = positions[str(edge["source"])]
            target = positions[str(edge["target"])]
            lane = _edge_lane(edge, data)
            color = _edge_color(str(edge["id"]), lane, data.runtime)
            label_position, label_align = _edge_label_position(source, target, width)
            _draw_labeled_edge(
                axis,
                source,
                target,
                color=color,
                label=_edge_label(edge, data),
                label_position=label_position,
                label_align=label_align,
            )

        for index, node_id in enumerate(nodes):
            _draw_state_node(
                axis,
                positions[node_id],
                start=node_id == str(data.program["start"]),
                goal=node_id == str(data.program["goal"]),
                fork=False,
                join=False,
                index=index,
            )

        _draw_legend(axis, width, height - 0.28)
        return _figure_png_bytes(figure)
    finally:
        figure.clear()


def _render_dag(data: _GraphData) -> bytes:
    """Render forks and joins against persistent actor swimlanes."""
    levels = _dag_levels(data.graph)
    maximum_level = max(levels.values(), default=0)
    width = _TARGET_WIDTH
    height = max(5.2, 2.15 + maximum_level * _LEVEL_STEP + 1.35)
    figure, axis = _new_figure(width, height)
    try:
        _draw_header(axis, data, width)
        boundaries, lane_centers = _lane_geometry(width)
        _draw_swimlanes(axis, height, boundaries, lane_centers)
        positions = _dag_positions(data, levels, lane_centers)

        # Dependency arrows are drawn first and stay visually subordinate to
        # physical state transitions; only constraints not already implied by
        # the state topology are shown.
        for source_id, target_id in _visible_dependencies(data):
            _draw_dependency_arrow(
                axis,
                positions[source_id],
                positions[target_id],
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
            parallel_ids = pair_groups[(source_id, target_id)]
            parallel_index = parallel_ids.index(edge_id)
            curvature = (parallel_index - (len(parallel_ids) - 1) / 2.0) * 0.20
            label_position, label_align = _edge_label_position(
                positions[source_id],
                positions[target_id],
                width,
            )
            _draw_labeled_edge(
                axis,
                positions[source_id],
                positions[target_id],
                color=_edge_color(edge_id, lane, data.runtime),
                label=_edge_label(edge, data),
                label_position=label_position,
                label_align=label_align,
                curvature=curvature,
            )

        for index, node_id in enumerate(nx.topological_sort(data.graph)):
            _draw_state_node(
                axis,
                positions[str(node_id)],
                start=str(node_id) == str(data.program["start"]),
                goal=str(node_id) == str(data.program["goal"]),
                fork=data.graph.out_degree(node_id) > 1,
                join=data.graph.in_degree(node_id) > 1,
                index=index,
            )

        _draw_legend(axis, width, height - 0.28)
        return _figure_png_bytes(figure)
    finally:
        figure.clear()


def _lane_geometry(
    width: float,
) -> tuple[dict[str, tuple[float, float]], dict[str, float]]:
    """Even thirds for lane boundaries with derived actor centers."""
    margin = 0.35
    area = width - 2 * margin
    first = margin + area / 3.0
    second = margin + 2 * area / 3.0
    boundaries = {
        "left": (margin, first),
        "auto": (first, second),
        "right": (second, width - margin),
    }
    centers = {lane: (left + right) / 2.0 for lane, (left, right) in boundaries.items()}
    return boundaries, centers


def _edge_label_position(
    source: tuple[float, float],
    target: tuple[float, float],
    width: float,
) -> tuple[tuple[float, float], str]:
    """Place halo labels beside arrows instead of boxing them on the edge."""
    midpoint = _midpoint(source, target)
    dx = target[0] - source[0]
    dy = target[1] - source[1]
    if abs(dx) < 0.3:
        # Keep vertical-arrow labels inside the canvas: right side on the left
        # half of the figure, left side on the right half.
        if midpoint[0] > width / 2.0:
            return (midpoint[0] - 0.14, midpoint[1]), "right"
        return (midpoint[0] + 0.14, midpoint[1]), "left"
    length = hypot(dx, dy) or 1.0
    normal_x, normal_y = dy / length, -dx / length
    if normal_x < 0:
        normal_x, normal_y = -normal_x, -normal_y
    if abs(normal_x) < 0.2 and normal_y > 0:
        # Horizontal arrows keep their label above the line in both directions.
        normal_x, normal_y = -normal_x, -normal_y
    return (
        (midpoint[0] + normal_x * 0.16, midpoint[1] + normal_y * 0.16),
        "center",
    )


def _visible_dependencies(data: _GraphData) -> list[tuple[str, str]]:
    """Node anchors for dependencies not implied by state continuity."""
    result: list[tuple[str, str]] = []
    for prerequisite_id, dependent_id in _dependency_pairs(data):
        source = str(data.edge_by_id[prerequisite_id]["target"])
        target = str(data.edge_by_id[dependent_id]["source"])
        if source == target or nx.has_path(data.graph, source, target):
            continue
        result.append((source, target))
    return result


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
            (index - (len(ordered) - 1) / 2.0) * 0.55 for index in range(len(ordered))
        ]
        for node_id, offset in zip(ordered, offsets, strict=True):
            result[node_id] = (center + offset, 2.15 + level * _LEVEL_STEP)
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
    """One-line semantic phrase; execution details live in the JSON artifacts."""
    edge_id = str(edge["id"])
    step = data.step_by_id[str(edge["semantic_step_id"])]
    status = data.runtime.edge_status.get(edge_id) or data.runtime.step_status.get(
        str(step["id"])
    )
    status_badge = f" [{_STATUS_BADGES.get(status, status.upper())}]" if status else ""
    return _clip(f"{step['operator']}: {step['object']}", 40) + status_badge


def _draw_header(axis: Any, data: _GraphData, width: float) -> None:
    status = data.runtime.graph_status
    status_text = f"  [{status.upper()}]" if status else ""
    axis.text(
        0.4,
        0.42,
        _clip(f"ACTION ENGINE / {data.program['task']}{status_text}", 84),
        ha="left",
        va="center",
        color=_INK,
        fontproperties=_font(10.0, "bold"),
        zorder=20,
    )
    axis.text(
        0.4,
        0.80,
        _clip(str(data.program["goal_description"]), 115),
        ha="left",
        va="top",
        color=_MUTED,
        fontproperties=_font(7.0),
        linespacing=1.25,
        zorder=20,
    )
    axis.plot(
        [0.4, width - 0.4],
        [1.28, 1.28],
        color=_BORDER,
        linewidth=0.7,
        zorder=19,
    )


def _draw_swimlanes(
    axis: Any,
    height: float,
    boundaries: Mapping[str, tuple[float, float]],
    centers: Mapping[str, float],
) -> None:
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
                linewidth=0.6,
                zorder=-10,
            )
        )
        axis.plot(
            [left, right],
            [1.50, 1.50],
            color=_LANE_COLORS[lane],
            linewidth=1.1,
            zorder=-9,
        )
        axis.text(
            centers[lane],
            1.74,
            _LANE_LABELS[lane],
            ha="center",
            va="center",
            color=_LANE_COLORS[lane],
            fontproperties=_font(6.8, "bold"),
            zorder=10,
        )


def _draw_legend(axis: Any, width: float, y: float) -> None:
    """Single-row edge-type legend; START/GOAL labels are self-explanatory."""
    entries = (
        ("left", "left action", False),
        ("right", "right action", False),
        ("coordinated", "coordinated", False),
        ("auto", "auto / world", False),
        ("dependency", "dependency", True),
    )
    slot = 1.32
    start = (width - slot * len(entries)) / 2.0
    for index, (key, label, dashed) in enumerate(entries):
        x = start + index * slot
        color = _DEPENDENCY if dashed else _LANE_COLORS[key]
        axis.add_patch(
            FancyArrowPatch(
                (x, y),
                (x + 0.3, y),
                arrowstyle="-|>",
                mutation_scale=7,
                color=color,
                linewidth=1.0,
                linestyle=(0, (3.0, 2.6)) if dashed else "-",
                zorder=20,
            )
        )
        axis.text(
            x + 0.38,
            y,
            label,
            ha="left",
            va="center",
            color=_MUTED,
            fontproperties=_font(6.2),
            zorder=20,
        )


def _draw_labeled_edge(
    axis: Any,
    source: tuple[float, float],
    target: tuple[float, float],
    *,
    color: str,
    label: str,
    label_position: tuple[float, float],
    label_align: str = "center",
    curvature: float = 0.0,
) -> None:
    """Draw one solid state transition and its halo-backed one-line label."""
    axis.add_patch(
        FancyArrowPatch(
            source,
            target,
            arrowstyle="-|>",
            mutation_scale=9,
            color=color,
            linewidth=1.15,
            shrinkA=12,
            shrinkB=12,
            connectionstyle=f"arc3,rad={curvature}",
            zorder=3,
        )
    )
    axis.text(
        *label_position,
        label,
        ha=label_align,
        va="center",
        color=_INK,
        fontproperties=_font(6.5),
        path_effects=[patheffects.withStroke(linewidth=1.7, foreground=_BACKGROUND)],
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
            mutation_scale=7,
            color=_DEPENDENCY,
            linewidth=0.9,
            linestyle=(0, (3.0, 2.6)),
            shrinkA=12,
            shrinkB=12,
            connectionstyle="arc3,rad=-0.2",
            alpha=0.9,
            zorder=1,
        )
    )


def _draw_state_node(
    axis: Any,
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
    radius = _SPECIAL_NODE_RADIUS if (start or goal or fork or join) else _NODE_RADIUS
    axis.add_patch(
        Circle(
            center,
            radius=radius,
            facecolor=fill,
            edgecolor=edge,
            linewidth=1.1,
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
        fontproperties=_font(6.5, "bold"),
        zorder=13,
    )
    role = (
        "START"
        if start
        else ("GOAL" if goal else ("FORK" if fork else "JOIN" if join else ""))
    )
    if role:
        axis.text(
            center[0],
            center[1] + radius + 0.12,
            role,
            ha="center",
            va="top",
            color=edge,
            fontproperties=_font(6.0, "bold"),
            zorder=13,
        )


def _new_figure(width: float, height: float) -> tuple[Figure, Any]:
    figure = Figure(figsize=(width, height), dpi=_DPI, facecolor=_BACKGROUND)
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
