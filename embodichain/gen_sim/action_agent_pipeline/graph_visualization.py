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

"""Render executable Seed and grounded runtime Task graphs as PNGs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import networkx as nx

from embodichain.gen_sim.action_agent_pipeline.domain.seed_task_graph import (
    seed_task_graph_hash,
    validate_seed_task_graph,
)
from embodichain.gen_sim.action_agent_pipeline.protocol.actions import (
    LEFT_ARM_ACTION_KEY,
    RIGHT_ARM_ACTION_KEY,
)

__all__ = ["render_seed_task_graph_png", "render_task_graph_png"]

_BACKGROUND = "#FAFBFC"
_TEXT = "#17202A"
_MUTED = "#5A6976"
_BORDER = "#7B8994"
_STATE = "#FFFFFF"
_START = "#34495E"
_GOAL = "#56A766"
_LEFT = "#168A78"
_RIGHT = "#D97706"
_DUAL = "#7C4DAD"
_AUTO = "#64748B"
_DEPENDENCY = "#3973B7"
_SUCCESS = "#28744A"
_FAILED = "#C0392B"
_SKIPPED = "#94A3B8"
_PENDING = "#CBD5E1"
_PNG_DPI = 150
_MAX_EDGES_PER_ROW = 5


@dataclass(frozen=True)
class _DisplayAction:
    """One concise arm action shown on an executable edge."""

    arm: str
    action_class: str
    primary: str | None
    detail: str | None
    status: str | None = None


@dataclass(frozen=True)
class _DisplayEdge:
    """One executable edge after applying the visualization information budget."""

    edge_id: str
    source: str
    target: str
    actions: tuple[_DisplayAction, ...]
    semantic_step_id: str | None


@dataclass(frozen=True)
class _DisplayNode:
    """One compact execution state derived from its incoming atomic action."""

    node_id: str
    state: str
    detail: str | None
    is_start: bool = False
    is_goal: bool = False
    runtime_status: str | None = None


@dataclass(frozen=True)
class _TaskGroup:
    """One semantic or deterministically inferred group of executable edges."""

    group_id: str
    operator: str
    object_uid: str | None
    actor_text: str
    goal: Mapping[str, Any] | None
    postcondition: Mapping[str, Any] | None
    edges: tuple[Mapping[str, Any], ...]
    derived: bool
    runtime: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class _TimelineRow:
    """A bounded group fragment occupying one row in the execution timeline."""

    group: _TaskGroup
    edges: tuple[Mapping[str, Any], ...]
    continued: bool
    is_last: bool


def render_seed_task_graph_png(seed_graph: Mapping[str, Any]) -> bytes:
    """Render every Seed v2 state and symbolic action in execution order."""
    validate_seed_task_graph(seed_graph)
    return _render_graph_timeline(seed_graph)


def render_task_graph_png(task_graph: Mapping[str, Any]) -> bytes:
    """Render one environment's grounded runtime graph and execution state."""
    return _render_graph_timeline(task_graph)


def _render_graph_timeline(task_graph: Mapping[str, Any]) -> bytes:
    """Render one complete state-action graph using a stable shared layout."""
    graph, node_records, edge_records = _validated_task_display_graph(task_graph)
    if not _is_single_chain(graph, task_graph):
        return _render_task_dag(task_graph, graph, node_records, edge_records)

    ordered_edges = _ordered_chain_edges(graph, task_graph, edge_records)
    groups = _task_groups(task_graph, ordered_edges)
    group_by_edge = _group_by_edge(groups)
    rows = _timeline_rows(ordered_edges, group_by_edge)
    display_edges = {
        str(edge["id"]): _display_edge(
            edge,
            group_by_edge[str(edge["id"])],
        )
        for edge in ordered_edges
    }
    display_nodes = _display_chain_nodes(
        task_graph,
        ordered_edges,
        group_by_edge,
    )

    row_height = 2.25
    top = 1.25
    canvas_height = max(5.4, top + len(rows) * row_height + 0.6)
    canvas_width = 14.8
    figure = _make_figure(canvas_width, canvas_height)
    try:
        axis = figure.subplots()
        axis.set_axis_off()
        axis.set_xlim(0.0, canvas_width)
        axis.set_ylim(canvas_height, 0.0)
        _draw_task_header(
            axis,
            task_graph,
            node_count=len(node_records),
            edge_count=len(edge_records),
            group_count=len(groups),
        )

        previous_endpoint: tuple[float, float] | None = None
        for row_index, row in enumerate(rows):
            y = top + row_index * row_height
            direction = 1 if row_index % 2 == 0 else -1
            previous_endpoint = _draw_timeline_row(
                axis,
                row,
                display_edges=display_edges,
                display_nodes=display_nodes,
                direction=direction,
                y=y,
                canvas_width=canvas_width,
                previous_endpoint=previous_endpoint,
            )

        axis.text(
            canvas_width - 0.45,
            canvas_height - 0.2,
            _graph_footer(task_graph),
            ha="right",
            va="bottom",
            color=_MUTED,
            fontproperties=_font(size=6.2),
        )
        return _figure_png_bytes(figure)
    finally:
        figure.clear()


def _draw_seed_header(axis: Any, seed_graph: Mapping[str, Any]) -> None:
    axis.text(
        1.0,
        0.42,
        f"{seed_graph['task']}  |  Executable Seed Graph",
        ha="left",
        va="center",
        color=_TEXT,
        fontproperties=_font(size=13, weight="bold"),
    )
    axis.text(
        1.0,
        0.85,
        (
            f"{seed_graph['route']}  ·  {seed_graph['program']}  ·  "
            f"{len(seed_graph['semantic_steps'])} semantic steps"
        ),
        ha="left",
        va="center",
        color=_MUTED,
        fontproperties=_font(size=7.4),
    )


def _draw_seed_step(
    axis: Any,
    step: Mapping[str, Any],
    *,
    index: int,
    x: float,
    y: float,
    width: float,
    height: float,
) -> None:
    arm_label, accent = _seed_actor_badge(step.get("actor", {}))
    axis.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.015,rounding_size=0.08",
            facecolor="#FFFFFF",
            edgecolor=_BORDER,
            linewidth=1.0,
            zorder=2,
        )
    )
    axis.plot(
        [x + 0.04, x + 0.04],
        [y + 0.12, y + height - 0.12],
        color=accent,
        linewidth=4.0,
        solid_capstyle="round",
        zorder=3,
    )
    axis.text(
        x + 0.25,
        y + 0.25,
        f"{_compact_id(str(step['id']))}  ·  {str(step['operator']).upper()}",
        ha="left",
        va="center",
        color=_TEXT,
        fontproperties=_font(size=7.3, weight="bold"),
        zorder=4,
    )
    axis.text(
        x + width - 0.22,
        y + 0.25,
        arm_label,
        ha="right",
        va="center",
        color="#FFFFFF",
        bbox={
            "boxstyle": "round,pad=0.22",
            "facecolor": accent,
            "edgecolor": accent,
            "linewidth": 0.8,
        },
        fontproperties=_font(size=6.4, weight="bold"),
        zorder=4,
    )
    relation, metadata = _seed_relation_summary(step)
    axis.text(
        x + 0.25,
        y + 0.68,
        relation,
        ha="left",
        va="center",
        color=_TEXT,
        fontproperties=_font(size=9.0, weight="bold"),
        zorder=4,
    )
    if metadata:
        axis.text(
            x + 0.25,
            y + 1.02,
            metadata,
            ha="left",
            va="center",
            color=_MUTED,
            fontproperties=_font(size=6.5),
            zorder=4,
        )
    axis.text(
        x + width - 0.22,
        y + 1.03,
        f"✓ {_postcondition_summary(step['postcondition'])}",
        ha="right",
        va="center",
        color=_SUCCESS,
        fontproperties=_font(size=6.4, weight="bold"),
        zorder=4,
    )


def _draw_seed_dependencies(
    axis: Any,
    steps: Sequence[Mapping[str, Any]],
    positions: Mapping[str, float],
    *,
    x: float,
    width: float,
    step_height: float,
) -> None:
    index_by_id = {str(step["id"]): index for index, step in enumerate(steps)}
    for target_step in steps:
        target_id = str(target_step["id"])
        target_index = index_by_id[target_id]
        dependencies = list(target_step.get("depends_on", []))
        for dependency_index, source_id in enumerate(dependencies):
            source_index = index_by_id[str(source_id)]
            source_y = positions[str(source_id)]
            target_y = positions[target_id]
            if target_index == source_index + 1 and len(dependencies) == 1:
                center_x = x + width / 2.0
                axis.add_patch(
                    FancyArrowPatch(
                        (center_x, source_y + step_height + 0.04),
                        (center_x, target_y - 0.04),
                        arrowstyle="-|>",
                        mutation_scale=10,
                        linewidth=1.4,
                        color=_DEPENDENCY,
                        zorder=1,
                    )
                )
                axis.text(
                    center_x + 0.14,
                    (source_y + step_height + target_y) / 2.0,
                    "depends_on",
                    ha="left",
                    va="center",
                    color=_DEPENDENCY,
                    fontproperties=_font(size=5.8, weight="bold"),
                    zorder=1,
                )
                continue

            rail_x = x + width + 0.35 + dependency_index * 0.18
            source_center = source_y + step_height / 2.0
            target_center = target_y + step_height / 2.0
            axis.plot(
                [x + width, rail_x, rail_x],
                [source_center, source_center, target_center],
                color=_DEPENDENCY,
                linewidth=1.1,
                zorder=1,
            )
            axis.add_patch(
                FancyArrowPatch(
                    (rail_x, target_center),
                    (x + width + 0.03, target_center),
                    arrowstyle="-|>",
                    mutation_scale=9,
                    linewidth=1.1,
                    color=_DEPENDENCY,
                    zorder=1,
                )
            )
            axis.text(
                rail_x + 0.06,
                (source_center + target_center) / 2.0,
                "dep",
                ha="left",
                va="center",
                color=_DEPENDENCY,
                fontproperties=_font(size=5.5, weight="bold"),
                zorder=1,
            )


def _seed_relation_summary(step: Mapping[str, Any]) -> tuple[str, str]:
    goal = step["goal"]
    operator = str(step["operator"])
    if operator == "arrange_in_line":
        objects = [_humanize_uid(str(uid)) for uid in goal["objects"]]
        order = "  →  ".join(objects)
        if goal.get("order_constraint") != "ordered":
            order = "  ·  ".join(objects)
        relation = f"LINE  |  {_truncate(order, 92)}"
        metadata = (
            f"axis {goal.get('axis', 'auto')}  ·  anchor {goal.get('anchor', 'auto')}"
            f"  ·  {goal.get('order_constraint', 'free')} order"
        )
        return relation, metadata

    object_name = _humanize_uid(str(step["object"]))
    relation = str(goal.get("relation", operator)).replace("_", " ").upper()
    reference = goal.get("reference_object")
    if reference:
        expression = (
            f"{object_name}   ── {relation} ──▶   " f"{_humanize_uid(str(reference))}"
        )
    else:
        expression = f"{object_name}   ── {relation}"
    metadata_parts = []
    if goal.get("reference_state"):
        metadata_parts.append(f"{goal['reference_state']} reference")
    if goal.get("orientation_goal"):
        metadata_parts.append(f"{goal['orientation_goal']} orientation")
    if "layer_index" in goal:
        metadata_parts.append(f"layer {goal['layer_index']}")
    return expression, "  ·  ".join(metadata_parts)


def _seed_goal_lines(step: Mapping[str, Any]) -> list[str]:
    """Return the exact concise goal lines used by the seed renderer."""
    relation, metadata = _seed_relation_summary(step)
    return [relation, metadata] if metadata else [relation]


def _seed_actor_badge(actor: Any) -> tuple[str, str]:
    if not isinstance(actor, Mapping):
        return "[AUTO]", _AUTO
    arm = actor.get("arm")
    arms = actor.get("arms")
    if arm == "left_arm":
        return "[L]", _LEFT
    if arm == "right_arm":
        return "[R]", _RIGHT
    if isinstance(arms, list) and {"left_arm", "right_arm"}.issubset(set(arms)):
        return "[L+R]", _DUAL
    return "[AUTO]", _AUTO


def _draw_task_header(
    axis: Any,
    task_graph: Mapping[str, Any],
    *,
    node_count: int,
    edge_count: int,
    group_count: int,
) -> None:
    is_seed = task_graph.get("schema_version") == "seed_task_graph_v2"
    title = (
        "Executable Seed State-Action Timeline"
        if is_seed
        else "Runtime Grounded State-Action Timeline"
    )
    axis.text(
        0.65,
        0.38,
        f"{task_graph.get('task', 'unknown')}  |  {title}",
        ha="left",
        va="center",
        color=_TEXT,
        fontproperties=_font(size=13, weight="bold"),
    )
    seed_hash = (
        seed_task_graph_hash(task_graph)[:12]
        if is_seed
        else str(task_graph.get("seed_graph_hash", "unavailable"))[:12]
    )
    axis.text(
        0.65,
        0.83,
        (
            f"{node_count} states  ·  {edge_count} actions  ·  "
            f"{group_count} semantic groups  ·  seed {seed_hash}"
        ),
        ha="left",
        va="center",
        color=_MUTED,
        fontproperties=_font(size=7.2),
    )


def _draw_timeline_row(
    axis: Any,
    row: _TimelineRow,
    *,
    display_edges: Mapping[str, _DisplayEdge],
    display_nodes: Mapping[str, _DisplayNode],
    direction: int,
    y: float,
    canvas_width: float,
    previous_endpoint: tuple[float, float] | None,
) -> tuple[float, float]:
    edge_count = len(row.edges)
    color = _row_color(row.edges)
    actor = _row_actor_badge(row.edges)
    group_name = _compact_group_name(row.group)
    object_name = (
        _humanize_uid(row.group.object_uid) if row.group.object_uid else "unresolved"
    )
    group_label = f"{group_name}  ·  {object_name}  ·  {actor}"
    if row.group.derived:
        group_label += "  ·  inferred"
    if row.continued:
        group_label += "  ·  continued"
    axis.plot(
        [0.65, canvas_width - 0.65],
        [y + 0.12, y + 0.12],
        color="#D7DEE4",
        linewidth=0.8,
        zorder=0,
    )
    axis.plot(
        [0.65, 0.98],
        [y + 0.12, y + 0.12],
        color=color,
        linewidth=4.0,
        solid_capstyle="round",
        zorder=2,
    )
    axis.text(
        1.08,
        y + 0.12,
        group_label,
        ha="left",
        va="center",
        color=_TEXT,
        bbox={
            "boxstyle": "square,pad=0.12",
            "facecolor": _BACKGROUND,
            "edgecolor": "none",
        },
        fontproperties=_font(size=7.3, weight="bold"),
        zorder=3,
    )
    if row.group.goal:
        axis.text(
            canvas_width - 0.7,
            y + 0.12,
            _compiled_goal_summary(row.group.object_uid, row.group.goal),
            ha="right",
            va="center",
            color=_MUTED,
            bbox={
                "boxstyle": "square,pad=0.12",
                "facecolor": _BACKGROUND,
                "edgecolor": "none",
            },
            fontproperties=_font(size=6.2),
            zorder=3,
        )

    left = 1.0
    right = canvas_width - 1.0
    slot_width = (right - left) / edge_count
    if direction > 0:
        x_positions = [left + index * slot_width for index in range(edge_count + 1)]
    else:
        x_positions = [right - index * slot_width for index in range(edge_count + 1)]
    baseline = y + 1.25

    if previous_endpoint is not None:
        _draw_row_continuation(
            axis,
            source=previous_endpoint,
            target=(x_positions[0], baseline),
            color=color,
        )
    else:
        source_id = str(row.edges[0]["source"])
        _draw_state_node(
            axis,
            display_nodes[source_id],
            center=(x_positions[0], baseline),
        )

    for index, edge_record in enumerate(row.edges):
        display_edge = display_edges[str(edge_record["id"])]
        source_x = x_positions[index]
        target_x = x_positions[index + 1]
        _draw_action_edge(
            axis,
            display_edge,
            source=(source_x, baseline),
            target=(target_x, baseline),
        )
        target_node = display_nodes[display_edge.target]
        _draw_state_node(
            axis,
            target_node,
            center=(target_x, baseline),
        )

    if row.group.postcondition and row.is_last:
        postcondition_runtime = (
            row.group.runtime.get("postcondition")
            if isinstance(row.group.runtime, Mapping)
            else None
        )
        postcondition_success = (
            postcondition_runtime.get("success")
            if isinstance(postcondition_runtime, Mapping)
            else None
        )
        if postcondition_runtime is None:
            marker = "EXPECT"
            marker_color = _MUTED
        elif postcondition_success is True:
            marker = "✓"
            marker_color = _SUCCESS
        elif postcondition_success is False:
            marker = "FAILED"
            marker_color = _FAILED
        else:
            marker = "PENDING"
            marker_color = _PENDING
        axis.text(
            canvas_width - 0.7,
            y + 2.02,
            f"{marker} {_postcondition_summary(row.group.postcondition)}",
            ha="right",
            va="center",
            color=marker_color,
            fontproperties=_font(size=6.1, weight="bold"),
            zorder=3,
        )
    return x_positions[-1], baseline


def _draw_state_node(
    axis: Any,
    node: _DisplayNode,
    *,
    center: tuple[float, float],
) -> None:
    width = 1.22
    height = 0.82
    if node.is_start:
        face_color, text_color = _START, "#FFFFFF"
    elif node.runtime_status == "failed":
        face_color, text_color = _FAILED, "#FFFFFF"
    elif node.runtime_status == "skipped":
        face_color, text_color = _SKIPPED, "#FFFFFF"
    elif node.runtime_status == "pending":
        face_color, text_color = "#F1F5F9", _MUTED
    elif node.is_goal:
        face_color, text_color = _GOAL, "#FFFFFF"
    else:
        face_color, text_color = _STATE, _TEXT
    axis.add_patch(
        FancyBboxPatch(
            (center[0] - width / 2.0, center[1] - height / 2.0),
            width,
            height,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=face_color,
            edgecolor=_BORDER,
            linewidth=1.0,
            zorder=5,
        )
    )
    lines = [_compact_id(node.node_id), node.state]
    if node.detail:
        lines.append(_truncate(node.detail, 22))
    axis.text(
        center[0],
        center[1],
        "\n".join(lines),
        ha="center",
        va="center",
        color=text_color,
        fontproperties=_font(size=5.8, weight="bold"),
        linespacing=1.04,
        zorder=6,
    )


def _draw_action_edge(
    axis: Any,
    edge: _DisplayEdge,
    *,
    source: tuple[float, float],
    target: tuple[float, float],
) -> None:
    color = _display_edge_color(edge)
    direction = 1 if target[0] > source[0] else -1
    node_half_width = 0.61
    start = (source[0] + direction * node_half_width, source[1])
    end = (target[0] - direction * node_half_width, target[1])
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.8,
            color=color,
            zorder=2,
        )
    )
    label = "\n".join(_display_edge_lines(edge))
    axis.text(
        (source[0] + target[0]) / 2.0,
        source[1] - 0.56,
        label,
        ha="center",
        va="center",
        color=_TEXT,
        bbox={
            "boxstyle": "round,pad=0.24",
            "facecolor": "#FFFFFF",
            "edgecolor": color,
            "linewidth": 0.9,
        },
        fontproperties=_font(size=5.8, weight="bold"),
        linespacing=1.08,
        zorder=4,
    )


def _draw_row_continuation(
    axis: Any,
    *,
    source: tuple[float, float],
    target: tuple[float, float],
    color: str,
) -> None:
    middle_y = (source[1] + target[1]) / 2.0
    axis.plot(
        [source[0], source[0], target[0]],
        [source[1] + 0.42, middle_y, middle_y],
        color=color,
        linewidth=1.2,
        linestyle=(0, (3, 2)),
        zorder=1,
    )
    axis.add_patch(
        FancyArrowPatch(
            (target[0], middle_y),
            (target[0], target[1] - 0.1),
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=1.2,
            color=color,
            linestyle=(0, (3, 2)),
            zorder=1,
        )
    )


def _display_edge(
    edge: Mapping[str, Any],
    group: _TaskGroup,
) -> _DisplayEdge:
    actions = []
    symbolic_actions = edge.get("actions")
    if isinstance(symbolic_actions, list):
        for action in symbolic_actions:
            if not isinstance(action, Mapping):
                continue
            actions.append(
                _display_action(
                    action,
                    arm=_symbolic_action_arm(action),
                    context_object=group.object_uid,
                    goal=group.goal,
                )
            )
    for arm, action_key in (("L", LEFT_ARM_ACTION_KEY), ("R", RIGHT_ARM_ACTION_KEY)):
        action = edge.get(action_key)
        if isinstance(action, Mapping):
            actions.append(
                _display_action(
                    action,
                    arm=arm,
                    context_object=group.object_uid,
                    goal=group.goal,
                )
            )
    return _DisplayEdge(
        edge_id=str(edge["id"]),
        source=str(edge["source"]),
        target=str(edge["target"]),
        actions=tuple(actions),
        semantic_step_id=None if group.derived else group.group_id,
    )


def _display_action(
    action: Mapping[str, Any],
    *,
    arm: str,
    context_object: str | None,
    goal: Mapping[str, Any] | None = None,
) -> _DisplayAction:
    action_class = str(action.get("atomic_action_class", "UnknownAction"))
    primary = None
    detail = None
    target = action.get("target_object")
    if isinstance(target, Mapping):
        uid = target.get("obj_name")
        if uid:
            primary = _humanize_uid(str(uid))
        affordance = target.get("affordance")
        if affordance:
            detail = f"grasp: {affordance}"

    binding = action.get("target_binding")
    if isinstance(binding, Mapping):
        kind = str(binding.get("kind", "binding"))
        bound_object = binding.get("object")
        primary = (
            _humanize_uid(str(bound_object)) if bound_object else kind.replace("_", " ")
        )
        policy = action.get("motion_policy")
        detail = f"{kind} · {policy}" if policy else kind

    runtime = action.get("runtime")
    if isinstance(runtime, Mapping):
        assigned_arm = runtime.get("assigned_arm")
        target_position = runtime.get("resolved_target_position")
        status = runtime.get("status")
        object_position = _runtime_pose_position(runtime.get("observed_object_pose"))
        reference_position = _runtime_pose_position(
            runtime.get("observed_reference_pose")
        )
        if assigned_arm:
            arm = {
                "left_arm": "L",
                "right_arm": "R",
                "coordinated": "L+R",
            }.get(str(assigned_arm), "A")
        if isinstance(target_position, Sequence) and len(target_position) >= 3:
            target_text = "target " + ", ".join(
                f"{float(value):.3f}" for value in target_position[:3]
            )
            primary = (
                f"obj {_compact_xyz(object_position)} -> {target_text}"
                if object_position is not None
                else target_text
            )
        elif object_position is not None:
            primary = "obj " + _compact_xyz(object_position)
        failure_reason = runtime.get("failure_reason")
        if failure_reason:
            details = [str(failure_reason)]
        else:
            details = []
            if reference_position is not None:
                details.append("ref " + _compact_xyz(reference_position))
            policy = runtime.get("resolved_motion_policy")
            if isinstance(policy, Mapping):
                policy_text = _compact_runtime_policy(policy)
                if policy_text:
                    details.append(policy_text)
            elif detail:
                details.append(detail)
            if status:
                details.insert(0, str(status))
        if details:
            detail = " · ".join(details)

    pose = action.get("target_object_pose")
    if not isinstance(pose, Mapping):
        pose = action.get("target_pose")
    if isinstance(pose, Mapping):
        reference = pose.get("support") or pose.get("obj_name") or pose.get("reference")
        relation = _pose_relation(action_class, pose, goal=goal)
        if reference:
            primary = f"{relation} {_humanize_uid(str(reference))}".strip()
        elif relation and relation != "to":
            primary = relation

    if action_class == "MoveJoints":
        target_qpos = action.get("target_qpos")
        if isinstance(target_qpos, Mapping) and target_qpos.get("source"):
            source = str(target_qpos["source"])
            primary = "home" if source == "initial" else source
    elif primary is None and context_object:
        primary = _humanize_uid(context_object)

    return _DisplayAction(
        arm=arm,
        action_class=action_class,
        primary=primary,
        detail=detail,
        status=(
            str(runtime.get("status"))
            if isinstance(runtime, Mapping) and runtime.get("status")
            else None
        ),
    )


def _display_edge_lines(edge: _DisplayEdge) -> list[str]:
    edge_id = _compact_id(edge.edge_id)
    if not edge.actions:
        return [f"{edge_id}  [–] UnknownAction"]
    if len(edge.actions) == 1:
        action = edge.actions[0]
        lines = [f"{edge_id}  [{action.arm}] {action.action_class}"]
        if action.primary:
            lines.append(_truncate(action.primary, 30))
        if action.detail:
            lines.append(_truncate(action.detail, 30))
        return lines[:3]

    lines = [f"{edge_id}  [L+R]"]
    for action in edge.actions:
        text = f"{action.arm} {action.action_class}"
        if action.primary:
            text += f" · {action.primary}"
        lines.append(_truncate(text, 32))
    return lines[:3]


def _action_lines(
    action: Mapping[str, Any],
    *,
    prefix: str,
    context_object: str | None,
) -> list[str]:
    """Expose the concise action label policy for focused unit tests."""
    display = _display_action(
        action,
        arm=prefix,
        context_object=context_object,
    )
    lines = [f"{display.arm} · {display.action_class}"]
    if display.primary:
        lines.append(display.primary)
    if display.detail:
        lines.append(display.detail)
    return lines


def _display_chain_nodes(
    task_graph: Mapping[str, Any],
    ordered_edges: Sequence[Mapping[str, Any]],
    group_by_edge: Mapping[str, _TaskGroup],
) -> dict[str, _DisplayNode]:
    start_id = str(ordered_edges[0]["source"])
    goal_id = str(task_graph.get("goal", ordered_edges[-1]["target"]))
    node_semantics = {
        str(node["id"]): str(node.get("semantic", "")).strip()
        for node in task_graph.get("nodes", [])
        if isinstance(node, Mapping) and isinstance(node.get("id"), str)
    }
    node_status = {
        str(node["id"]): node.get("runtime_status")
        for node in task_graph.get("nodes", [])
        if isinstance(node, Mapping) and isinstance(node.get("id"), str)
    }
    nodes = {
        start_id: _DisplayNode(
            node_id=start_id,
            state="START",
            detail=node_semantics.get(start_id) or "initial",
            is_start=True,
            is_goal=start_id == goal_id,
            runtime_status=node_status.get(start_id),
        )
    }
    for edge in ordered_edges:
        edge_id = str(edge["id"])
        group = group_by_edge[edge_id]
        display_edge = _display_edge(edge, group)
        derived_node = _state_after_edge(
            display_edge,
            context_object=group.object_uid,
            is_goal=display_edge.target == goal_id,
            runtime_status=node_status.get(display_edge.target),
        )
        semantic = node_semantics.get(display_edge.target)
        nodes[display_edge.target] = (
            _DisplayNode(
                node_id=derived_node.node_id,
                state=_semantic_state_title(semantic),
                detail=semantic,
                is_start=derived_node.is_start,
                is_goal=derived_node.is_goal,
                runtime_status=derived_node.runtime_status,
            )
            if semantic
            else derived_node
        )
    return nodes


def _semantic_state_title(semantic: str) -> str:
    """Map an explicit Seed state description to a compact node heading."""
    normalized = semantic.lower()
    if "initial state" in normalized or "at initial" in normalized:
        return "HOME"
    if "retreated" in normalized:
        return "RETREATED"
    if "released" in normalized:
        return "RELEASED"
    if "holding" in normalized or " held " in f" {normalized} ":
        return "HOLDING"
    if "complete" in normalized:
        return "COMPLETE"
    return "STATE"


def _state_after_edge(
    edge: _DisplayEdge,
    *,
    context_object: str | None,
    is_goal: bool,
    runtime_status: str | None,
) -> _DisplayNode:
    object_name = _humanize_uid(context_object) if context_object else None
    if not edge.actions:
        return _DisplayNode(
            edge.target,
            "STATE",
            None,
            is_goal=is_goal,
            runtime_status=runtime_status,
        )
    if len(edge.actions) > 1:
        return _DisplayNode(
            edge.target,
            "DUAL ACTION",
            "completed",
            is_goal=is_goal,
            runtime_status=runtime_status,
        )

    action = edge.actions[0]
    action_class = action.action_class
    if action_class == "PickUp":
        return _DisplayNode(
            edge.target,
            "HOLDING",
            action.primary or object_name,
            is_goal=is_goal,
            runtime_status=runtime_status,
        )
    if action_class in {"Place", "MoveHeldObject"}:
        state = "PLACED" if action_class == "Place" else "MOVED"
        detail_parts = []
        if object_name:
            detail_parts.append(object_name)
        if action.primary and action.primary != object_name:
            detail_parts.append(action.primary)
        return _DisplayNode(
            edge.target,
            state,
            " · ".join(detail_parts) or action.primary,
            is_goal=is_goal,
            runtime_status=runtime_status,
        )
    if action_class == "MoveJoints":
        return _DisplayNode(
            edge.target,
            f"{'LEFT' if action.arm == 'L' else 'RIGHT'} ARM",
            (action.primary or "moved").upper(),
            is_goal=is_goal,
            runtime_status=runtime_status,
        )
    return _DisplayNode(
        edge.target,
        _split_camel_case(action_class).upper(),
        action.primary or object_name,
        is_goal=is_goal,
        runtime_status=runtime_status,
    )


def _timeline_rows(
    ordered_edges: Sequence[Mapping[str, Any]],
    group_by_edge: Mapping[str, _TaskGroup],
) -> list[_TimelineRow]:
    segments: list[tuple[_TaskGroup, list[Mapping[str, Any]]]] = []
    for edge in ordered_edges:
        group = group_by_edge[str(edge["id"])]
        if not segments or segments[-1][0] is not group:
            segments.append((group, []))
        segments[-1][1].append(edge)

    rows = []
    for group, segment_edges in segments:
        for offset in range(0, len(segment_edges), _MAX_EDGES_PER_ROW):
            chunk_end = min(offset + _MAX_EDGES_PER_ROW, len(segment_edges))
            rows.append(
                _TimelineRow(
                    group=group,
                    edges=tuple(segment_edges[offset:chunk_end]),
                    continued=offset > 0,
                    is_last=chunk_end == len(segment_edges),
                )
            )
    return rows


def _task_groups(
    task_graph: Mapping[str, Any],
    edge_records: Sequence[Mapping[str, Any]],
) -> list[_TaskGroup]:
    edge_by_id = {str(edge["id"]): edge for edge in edge_records}
    semantic_steps = task_graph.get("semantic_steps")
    if isinstance(semantic_steps, list) and semantic_steps:
        groups = []
        assigned: set[str] = set()
        for step in semantic_steps:
            edge_ids = [str(edge_id) for edge_id in step.get("edge_ids", [])]
            group_edges = tuple(edge_by_id[edge_id] for edge_id in edge_ids)
            assigned.update(edge_ids)
            groups.append(
                _TaskGroup(
                    group_id=str(step["id"]),
                    operator=str(step.get("operator", "semantic_step")),
                    object_uid=_optional_string(step.get("object")),
                    actor_text=_actor_text(step.get("actor", {})),
                    goal=(
                        step.get("goal")
                        if isinstance(step.get("goal"), Mapping)
                        else None
                    ),
                    postcondition=(
                        step.get("postcondition")
                        if isinstance(step.get("postcondition"), Mapping)
                        else None
                    ),
                    edges=group_edges,
                    derived=False,
                    runtime=(
                        step.get("runtime")
                        if isinstance(step.get("runtime"), Mapping)
                        else None
                    ),
                )
            )
        unassigned = tuple(
            edge for edge in edge_records if str(edge["id"]) not in assigned
        )
        if unassigned:
            groups.extend(_inferred_groups(unassigned, start_index=len(groups) + 1))
        return groups
    return _inferred_groups(tuple(edge_records), start_index=1)


def _inferred_groups(
    edges: tuple[Mapping[str, Any], ...],
    *,
    start_index: int,
) -> list[_TaskGroup]:
    segments: list[list[Mapping[str, Any]]] = []
    current: list[Mapping[str, Any]] = []
    for edge in edges:
        if current and _edge_has_action_class(edge, "PickUp"):
            segments.append(current)
            current = []
        current.append(edge)
    if current:
        segments.append(current)
    return [
        _inferred_group(tuple(segment), start_index + index)
        for index, segment in enumerate(segments)
    ]


def _inferred_group(
    edges: tuple[Mapping[str, Any], ...],
    index: int,
) -> _TaskGroup:
    object_uid = None
    arms = set()
    for edge in edges:
        for arm, action_key in (
            ("left_arm", LEFT_ARM_ACTION_KEY),
            ("right_arm", RIGHT_ARM_ACTION_KEY),
        ):
            action = edge.get(action_key)
            if not isinstance(action, Mapping):
                continue
            arms.add(arm)
            target = action.get("target_object")
            if (
                object_uid is None
                and isinstance(target, Mapping)
                and isinstance(target.get("obj_name"), str)
            ):
                object_uid = str(target["obj_name"])
    actor_text = " + ".join(sorted(arms)) if arms else "unresolved"
    return _TaskGroup(
        group_id=f"segment_{index:02d}",
        operator="atomic_sequence",
        object_uid=object_uid,
        actor_text=actor_text,
        goal=None,
        postcondition=None,
        edges=edges,
        derived=True,
        runtime=None,
    )


def _group_by_edge(groups: Sequence[_TaskGroup]) -> dict[str, _TaskGroup]:
    result = {}
    for group in groups:
        for edge in group.edges:
            result[str(edge["id"])] = group
    return result


def _ordered_chain_edges(
    graph: nx.DiGraph,
    task_graph: Mapping[str, Any],
    edge_records: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    edge_by_id = {str(edge["id"]): edge for edge in edge_records}
    starts = [node_id for node_id in graph if graph.in_degree(node_id) == 0]
    current = str(task_graph.get("start") or starts[0])
    ordered = []
    while graph.out_degree(current):
        target = next(iter(graph.successors(current)))
        edge_id = str(graph[current][target]["edge_id"])
        ordered.append(edge_by_id[edge_id])
        current = str(target)
    if len(ordered) != len(edge_records):
        raise ValueError("Task graph chain traversal did not cover every edge.")
    return ordered


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
    node_records = []
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

    edge_records = []
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
    _validate_semantic_edge_ownership(task_graph, edge_ids)
    return graph, node_records, edge_records


def _validate_semantic_edge_ownership(
    task_graph: Mapping[str, Any],
    edge_ids: set[str],
) -> None:
    semantic_steps = task_graph.get("semantic_steps")
    if semantic_steps is None:
        return
    if not isinstance(semantic_steps, list):
        raise TypeError("Task graph semantic_steps must be a list when present.")
    assigned: set[str] = set()
    step_ids: set[str] = set()
    for index, step in enumerate(semantic_steps):
        if not isinstance(step, Mapping):
            raise TypeError(f"Task graph semantic step {index} must be a mapping.")
        step_id = step.get("id")
        owned_edges = step.get("edge_ids")
        if not isinstance(step_id, str) or not step_id:
            raise ValueError(f"Task graph semantic step {index} requires an id.")
        if step_id in step_ids:
            raise ValueError(f"Duplicate task graph semantic step id: {step_id!r}.")
        step_ids.add(step_id)
        if not isinstance(owned_edges, list):
            raise TypeError(
                f"Task graph semantic step {step_id!r} edge_ids must be a list."
            )
        for edge_id in owned_edges:
            if edge_id not in edge_ids:
                raise ValueError(
                    f"Semantic step {step_id!r} references unknown edge {edge_id!r}."
                )
            if edge_id in assigned:
                raise ValueError(
                    f"Task graph edge {edge_id!r} belongs to multiple semantic steps."
                )
            assigned.add(str(edge_id))


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
    return (
        len(starts) == 1
        and len(goals) == 1
        and task_graph.get("start") in (None, starts[0])
        and task_graph.get("goal") in (None, goals[0])
    )


def _render_task_dag(
    task_graph: Mapping[str, Any],
    graph: nx.DiGraph,
    nodes: Sequence[Mapping[str, Any]],
    edges: Sequence[Mapping[str, Any]],
) -> bytes:
    generations = list(nx.topological_generations(graph))
    width = max(11.0, max(len(generation) for generation in generations) * 3.0)
    height = max(6.0, len(generations) * 2.0 + 1.5)
    figure = _make_figure(min(width, 25.0), min(height, 28.0))
    try:
        axis = figure.subplots()
        axis.set_axis_off()
        positions = {}
        for generation_index, generation in enumerate(generations):
            ordered = sorted(generation)
            for column, node_id in enumerate(ordered):
                positions[str(node_id)] = (
                    column - (len(ordered) - 1) / 2.0,
                    -generation_index,
                )
        nx.draw_networkx_nodes(
            graph,
            positions,
            ax=axis,
            node_color="#FFFFFF",
            edgecolors=_BORDER,
            node_size=2200,
        )
        nx.draw_networkx_edges(
            graph,
            positions,
            ax=axis,
            edge_color=[_edge_color(edge) for edge in edges],
            arrows=True,
            arrowsize=14,
            width=1.6,
        )
        node_labels = {
            str(node["id"]): (
                f"{_compact_id(str(node['id']))}\n"
                f"{_truncate(str(node.get('semantic', 'state')), 28)}"
            )
            for node in nodes
        }
        edge_labels = {
            (str(edge["source"]), str(edge["target"])): "\n".join(
                _display_edge_lines(_display_edge(edge, _inferred_group((edge,), 1)))
            )
            for edge in edges
        }
        nx.draw_networkx_labels(
            graph,
            positions,
            labels=node_labels,
            ax=axis,
            font_size=6,
            font_family=_font_family(),
        )
        nx.draw_networkx_edge_labels(
            graph,
            positions,
            edge_labels=edge_labels,
            ax=axis,
            font_size=5,
            font_family=_font_family(),
            rotate=False,
        )
        axis.set_title(
            f"{task_graph.get('task', 'unknown')} | Compiled State–Action DAG",
            fontproperties=_font(size=13, weight="bold"),
            color=_TEXT,
        )
        return _figure_png_bytes(figure)
    finally:
        figure.clear()


def _row_color(edges: Sequence[Mapping[str, Any]]) -> str:
    symbolic_arms = {
        _symbolic_action_arm(action)
        for edge in edges
        for action in (
            edge.get("actions") if isinstance(edge.get("actions"), list) else []
        )
        if isinstance(action, Mapping)
    }
    if {"L", "R"}.issubset(symbolic_arms) or "L+R" in symbolic_arms:
        return _DUAL
    if symbolic_arms == {"L"}:
        return _LEFT
    if symbolic_arms == {"R"}:
        return _RIGHT
    has_left = any(isinstance(edge.get(LEFT_ARM_ACTION_KEY), Mapping) for edge in edges)
    has_right = any(
        isinstance(edge.get(RIGHT_ARM_ACTION_KEY), Mapping) for edge in edges
    )
    if has_left and has_right:
        return _DUAL
    if has_left:
        return _LEFT
    if has_right:
        return _RIGHT
    return _AUTO


def _row_actor_badge(edges: Sequence[Mapping[str, Any]]) -> str:
    color = _row_color(edges)
    if color == _DUAL:
        return "[L+R]"
    if color == _LEFT:
        return "[L]"
    if color == _RIGHT:
        return "[R]"
    return "[AUTO]"


def _display_edge_color(edge: _DisplayEdge) -> str:
    statuses = {action.status for action in edge.actions if action.status}
    if "failed" in statuses:
        return _FAILED
    if statuses == {"skipped"}:
        return _SKIPPED
    if statuses == {"pending"}:
        return _PENDING
    arms = {action.arm for action in edge.actions}
    if arms == {"L", "R"}:
        return _DUAL
    if arms == {"L"}:
        return _LEFT
    if arms == {"R"}:
        return _RIGHT
    return _AUTO


def _edge_color(edge: Mapping[str, Any]) -> str:
    symbolic = edge.get("actions")
    if isinstance(symbolic, list):
        arms = {
            _symbolic_action_arm(action)
            for action in symbolic
            if isinstance(action, Mapping)
        }
        if {"L", "R"}.issubset(arms) or "L+R" in arms:
            return _DUAL
        if arms == {"L"}:
            return _LEFT
        if arms == {"R"}:
            return _RIGHT
        return _AUTO
    has_left = isinstance(edge.get(LEFT_ARM_ACTION_KEY), Mapping)
    has_right = isinstance(edge.get(RIGHT_ARM_ACTION_KEY), Mapping)
    if has_left and has_right:
        return _DUAL
    if has_left:
        return _LEFT
    if has_right:
        return _RIGHT
    return _AUTO


def _edge_has_action_class(edge: Mapping[str, Any], action_class: str) -> bool:
    symbolic = edge.get("actions")
    if isinstance(symbolic, list) and any(
        isinstance(action, Mapping)
        and action.get("atomic_action_class") == action_class
        for action in symbolic
    ):
        return True
    return any(
        isinstance(edge.get(key), Mapping)
        and edge[key].get("atomic_action_class") == action_class
        for key in (LEFT_ARM_ACTION_KEY, RIGHT_ARM_ACTION_KEY)
    )


def _symbolic_action_arm(action: Mapping[str, Any]) -> str:
    runtime = action.get("runtime")
    if isinstance(runtime, Mapping):
        assigned = runtime.get("assigned_arm")
        if assigned == "left_arm":
            return "L"
        if assigned == "right_arm":
            return "R"
    actor = action.get("actor")
    if not isinstance(actor, Mapping):
        return "A"
    if actor.get("mode") == "coordinated":
        return "L+R"
    arm = actor.get("arm")
    if arm == "left_arm":
        return "L"
    if arm == "right_arm":
        return "R"
    return "A"


def _graph_footer(graph: Mapping[str, Any]) -> str:
    if graph.get("schema_version") == "seed_task_graph_v2":
        return "Symbolic bindings and named policies only; no runtime coordinates."
    return "One environment and episode; coordinates and execution status are grounded."


def _runtime_pose_position(value: Any) -> Sequence[Any] | None:
    if not isinstance(value, Mapping):
        return None
    position = value.get("position")
    if isinstance(position, Sequence) and len(position) >= 3:
        return position
    return None


def _compact_xyz(position: Sequence[Any]) -> str:
    return ",".join(f"{float(value):.3f}" for value in position[:3])


def _compact_runtime_policy(policy: Mapping[str, Any]) -> str:
    preferred_keys = (
        "relation_distance",
        "line_spacing",
        "surface_clearance",
        "pre_grasp_distance",
        "lift_height",
        "retreat_height",
        "postcondition_tolerance",
        "sample_interval",
    )
    values = []
    for key in preferred_keys:
        if key not in policy:
            continue
        short_key = {
            "postcondition_tolerance": "tol",
            "pre_grasp_distance": "pre",
            "relation_distance": "dist",
            "surface_clearance": "clear",
            "sample_interval": "samples",
            "retreat_height": "retreat",
            "line_spacing": "spacing",
            "lift_height": "lift",
        }[key]
        value = policy[key]
        values.append(
            f"{short_key}={float(value):.3f}"
            if isinstance(value, float)
            else f"{short_key}={value}"
        )
        if len(values) == 2:
            break
    return "policy " + ",".join(values) if values else ""


def _pose_relation(
    action_class: str,
    pose: Mapping[str, Any],
    *,
    goal: Mapping[str, Any] | None,
) -> str:
    if goal and goal.get("relation"):
        return str(goal["relation"]).replace("_", " ")
    if pose.get("support") or pose.get("z_policy") == "surface_release":
        return "on"
    if action_class == "Place":
        return "at"
    return "to"


def _compiled_goal_summary(
    object_uid: str | None,
    goal: Mapping[str, Any],
) -> str:
    object_name = _humanize_uid(object_uid) if object_uid else "object"
    relation = str(goal.get("relation", goal.get("layout", "goal"))).replace("_", " ")
    reference = goal.get("reference_object")
    if reference:
        return f"goal: {object_name} {relation} " f"{_humanize_uid(str(reference))}"
    return f"goal: {object_name} {relation}"


def _postcondition_summary(postcondition: Mapping[str, Any]) -> str:
    condition_type = postcondition.get("type")
    if condition_type:
        parts = [str(condition_type).replace("_", " ")]
        if "relation" in postcondition:
            parts.append(str(postcondition["relation"]).replace("_", " "))
        if "layer_index" in postcondition:
            parts.append(f"layer {postcondition['layer_index']}")
        return " · ".join(parts)
    terms = postcondition.get("terms")
    if isinstance(terms, list):
        return f"{len(terms)} verification terms"
    return "postcondition"


def _compact_group_name(group: _TaskGroup) -> str:
    if group.derived:
        return group.group_id.replace("_", " ").upper()
    return _compact_id(group.group_id)


def _actor_text(actor: Any) -> str:
    if not isinstance(actor, Mapping):
        return "unresolved"
    mode = str(actor.get("mode", "auto"))
    arm = actor.get("arm")
    if arm:
        return f"{mode} {arm}"
    arms = actor.get("arms")
    if isinstance(arms, list):
        return f"{mode} {' + '.join(str(value) for value in arms)}"
    return mode


def _optional_string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _compact_id(identifier: str) -> str:
    prefix = str(identifier).split("_", 1)[0]
    return prefix if prefix else str(identifier)


def _humanize_uid(uid: str | None) -> str:
    if not uid:
        return "unresolved"
    text = str(uid)
    if text.startswith("interact_"):
        text = text[len("interact_") :]
    return text.replace("_", " ")


def _split_camel_case(value: str) -> str:
    result = []
    for character in value:
        if result and character.isupper() and result[-1].islower():
            result.append(" ")
        result.append(character)
    return "".join(result)


def _truncate(value: str, limit: int) -> str:
    normalized = " ".join(str(value).replace("`", "").split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(1, limit - 3)].rstrip() + "..."


def _make_figure(width: float, height: float) -> Figure:
    safe_width = min(max(width, 6.0), 28.0)
    safe_height = min(max(height, 4.0), 32.0)
    return Figure(
        figsize=(safe_width, safe_height),
        dpi=_PNG_DPI,
        facecolor=_BACKGROUND,
        constrained_layout=False,
    )


def _figure_png_bytes(figure: Figure) -> bytes:
    buffer = BytesIO()
    FigureCanvasAgg(figure).print_png(buffer)
    return buffer.getvalue()


@lru_cache(maxsize=1)
def _font_family() -> str:
    preferred = (
        "Noto Sans CJK SC",
        "Noto Sans CJK TC",
        "Noto Sans CJK JP",
        "Source Han Sans SC",
        "WenQuanYi Micro Hei",
        "DejaVu Sans",
    )
    available = {font.name for font in fontManager.ttflist}
    for family in preferred:
        if family in available:
            return family
    return "sans-serif"


def _font(*, size: float, weight: str = "normal") -> FontProperties:
    return FontProperties(family=_font_family(), size=size, weight=weight)
