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

"""Headless PNG views for the current ``semantic_task_graph/v1`` contract."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import textwrap
from typing import Any, Final, Literal

from .reporting import validate_execution_report
from .semantic_graph import SemanticTaskGraph, validate_semantic_task_graph

__all__ = [
    "SemanticGraphView",
    "ATOMIC_SKILL_MAP",
    "render_semantic_task_graph_png",
    "write_semantic_task_graph_png",
]

SemanticGraphView = Literal["groups", "calls"]

ATOMIC_SKILL_MAP: Final[dict[str, str]] = {
    "E1": "pick_up → place / stack_place",
    "E2": "pick_up → move_held_object → place",
    "E3": "pick_up → move_held_object → pour → place",
    "E4": "pick_up → hand_over",
    "E5": "coordinated_pickment → coordinated_transport",
    "E6": "slide → move_joints",
    "E9": "press",
}

_BACKGROUND: Final = "#F7F9FB"
_INK: Final = "#17212B"
_MUTED: Final = "#66727D"
_BORDER: Final = "#C8D1D9"
_DEPENDENCY: Final = "#8A94A0"
_ARM_COLORS: Final = {
    "left": "#168A78",
    "right": "#D97706",
    "coordinated": "#7652A5",
    "auto": "#59636D",
}
_STATUS_COLORS: Final = {
    "success": "#25834B",
    "failed": "#C43E3E",
    "skipped": "#8B949C",
    "running": "#2563A8",
    "unknown": _BORDER,
}
_TASK_COLORS: Final = {
    "E1": "#EAF2FF",
    "E2": "#EAF6F3",
    "E3": "#FFF4E6",
    "E4": "#F1EBFA",
    "E5": "#F7EAF3",
    "E6": "#E9F1F7",
    "E9": "#FFF1E6",
}
_PNG_SIGNATURE: Final = b"\x89PNG\r\n\x1a\n"


def render_semantic_task_graph_png(
    graph: Mapping[str, Any],
    execution_report: Mapping[str, Any] | None = None,
    *,
    view: SemanticGraphView = "groups",
) -> bytes:
    """Render a current semantic graph as a deterministic PNG.

    Args:
        graph: A validated or JSON-compatible ``semantic_task_graph/v1`` value.
        execution_report: Optional Task Program execution report used only for
            success/failure overlays.
        view: ``"groups"`` for the long-horizon overview or ``"calls"`` for
            one card per Semantic Call.

    Returns:
        PNG bytes produced by the non-interactive Matplotlib backend.

    Raises:
        TypeError: If either input is not a mapping.
        ValueError: If the graph, report, or view is invalid.
    """
    if view not in {"groups", "calls"}:
        raise ValueError("view must be 'groups' or 'calls'.")
    selected = validate_semantic_task_graph(graph)
    report = (
        None
        if execution_report is None
        else validate_execution_report(execution_report)
    )
    statuses = _runtime_statuses(selected, report)
    if view == "groups":
        return _render_groups(selected, statuses)
    return _render_calls(selected, statuses)


def write_semantic_task_graph_png(
    graph: Mapping[str, Any],
    output: str | Path,
    execution_report: Mapping[str, Any] | None = None,
    *,
    view: SemanticGraphView = "groups",
) -> Path:
    """Render a semantic graph and atomically write its PNG file.

    Args:
        graph: A ``semantic_task_graph/v1`` mapping.
        output: Destination PNG path.
        execution_report: Optional runtime report for status overlays.
        view: ``"groups"`` or ``"calls"``.

    Returns:
        The resolved output path.
    """
    payload = render_semantic_task_graph_png(
        graph,
        execution_report,
        view=view,
    )
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    return destination


def _runtime_statuses(
    graph: SemanticTaskGraph,
    report: Mapping[str, Any] | None,
) -> dict[str, str]:
    statuses = {str(node["id"]): "unknown" for node in graph["nodes"]}
    if report is None:
        return statuses
    runtime = report.get("runtime_result")
    segments = runtime.get("segments", []) if isinstance(runtime, Mapping) else []
    node_ids = list(statuses)
    for index, segment in enumerate(segments):
        if not isinstance(segment, Mapping):
            continue
        segment_id = segment.get("name")
        if not isinstance(segment_id, str) or segment_id not in statuses:
            metadata = segment.get("metadata")
            call_indices = (
                metadata.get("semantic_call_indices", [])
                if isinstance(metadata, Mapping)
                else []
            )
            segment_id = (
                node_ids[int(call_indices[0])]
                if call_indices and int(call_indices[0]) < len(node_ids)
                else node_ids[index] if index < len(node_ids) else None
            )
        if segment_id is not None:
            statuses[segment_id] = (
                "success" if segment.get("success") is True else "failed"
            )
    return statuses


def _render_groups(graph: SemanticTaskGraph, statuses: Mapping[str, str]) -> bytes:
    """Render one horizontal timeline with independent control-part lanes."""
    groups = list(graph["task_groups"])
    node_by_id = {str(node["id"]): node for node in graph["nodes"]}
    group_nodes = {
        str(group["id"]): [node_by_id[str(node_id)] for node_id in group["node_ids"]]
        for group in groups
    }
    levels = _group_levels(groups)
    max_level = max(levels.values(), default=0)
    width = max(23.0, 4.0 + (max_level + 1) * 2.55)
    height = 10.3
    figure, axis = _figure(width, height)
    _draw_header(axis, graph, width, "Unified Timeline")

    left_y, handover_y, right_y = 3.35, 4.55, 5.75
    axis.text(
        0.55, left_y, "LEFT ARM", color=_ARM_COLORS["left"], fontsize=9, weight="bold"
    )
    axis.text(
        0.55,
        handover_y,
        "HANDOVER",
        color=_ARM_COLORS["coordinated"],
        fontsize=7.5,
        weight="bold",
    )
    axis.text(
        0.55,
        right_y,
        "RIGHT ARM",
        color=_ARM_COLORS["right"],
        fontsize=9,
        weight="bold",
    )
    axis.plot(
        [1.8, width - 0.55], [left_y, left_y], color=_ARM_COLORS["left"], alpha=0.3
    )
    axis.plot(
        [1.8, width - 0.55],
        [handover_y, handover_y],
        color=_ARM_COLORS["coordinated"],
        alpha=0.18,
        linestyle=(0, (4, 3)),
    )
    axis.plot(
        [1.8, width - 0.55], [right_y, right_y], color=_ARM_COLORS["right"], alpha=0.3
    )

    positions: dict[str, tuple[float, str]] = {}
    occupied: dict[tuple[int, str], int] = {}
    for group in groups:
        group_id = str(group["id"])
        lane = _group_lane(group_nodes[group_id])
        level = levels[group_id]
        slot = occupied.get((level, lane), 0)
        occupied[(level, lane)] = slot + 1
        x = 2.4 + level * (width - 4.0) / max(max_level, 1) + slot * 1.55
        positions[group_id] = (x, lane)
    for group in groups:
        group_id = str(group["id"])
        x, lane = positions[group_id]
        for dependency in group["depends_on"]:
            source_x, source_lane = positions[str(dependency)]
            _arrow(
                axis,
                (source_x, _lane_y(source_lane, left_y, handover_y, right_y)),
                (x, _lane_y(lane, left_y, handover_y, right_y)),
                color=_DEPENDENCY,
                dashed=True,
            )
    for group in groups:
        group_id = str(group["id"])
        x, lane = positions[group_id]
        nodes = group_nodes[group_id]
        if lane == "handover":
            _draw_handover_group(axis, group, nodes, x, left_y, right_y, statuses)
        else:
            _draw_timeline_group(
                axis,
                group,
                nodes,
                x,
                _lane_y(lane, left_y, handover_y, right_y),
                lane,
                statuses,
            )
    _draw_skill_table(axis, width, height)
    return _png(figure)


def _render_calls(graph: SemanticTaskGraph, statuses: Mapping[str, str]) -> bytes:
    nodes = list(graph["nodes"])
    columns = 4
    rows = (len(nodes) + columns - 1) // columns
    width = 15.0
    height = max(3.2, 1.0 + rows * 1.35)
    figure, axis = _figure(width, height)
    positions: dict[str, tuple[float, float]] = {}
    left = 1.6
    right = width - 1.6
    for index, node in enumerate(nodes):
        row, column = divmod(index, columns)
        positions[str(node["id"])] = (
            left + column * (right - left) / max(columns - 1, 1),
            1.45 + row * 1.35,
        )
    _draw_header(axis, graph, width, "Semantic Calls")
    by_id = {str(node["id"]): node for node in nodes}
    for node in nodes:
        target = positions[str(node["id"])]
        for dependency in node["depends_on"]:
            source = positions[str(dependency)]
            _arrow(axis, source, target, color=_DEPENDENCY, dashed=False)
    for node in nodes:
        _draw_call(axis, by_id[str(node["id"])], positions[str(node["id"])], statuses)
    return _png(figure)


def _figure(width: float, height: float) -> tuple[Any, Any]:
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(width, height), dpi=160, facecolor=_BACKGROUND)
    FigureCanvasAgg(figure)
    axis = figure.add_axes([0.0, 0.0, 1.0, 1.0])
    axis.set_facecolor(_BACKGROUND)
    axis.set_xlim(0.0, width)
    axis.set_ylim(height, 0.0)
    axis.set_axis_off()
    return figure, axis


def _font(size: float) -> Any:
    from matplotlib.font_manager import FontProperties, findSystemFonts, fontManager

    for path in findSystemFonts():
        if any(
            token in path.lower()
            for token in ("wqy-zenhei", "noto sans cjk", "sourcehan")
        ):
            return FontProperties(fname=path, size=size)

    available = {font.name for font in fontManager.ttflist}
    family = next(
        (
            name
            for name in (
                "Noto Sans CJK SC",
                "Source Han Sans CN",
                "WenQuanYi Zen Hei",
                "DejaVu Sans",
            )
            if name in available
        ),
        "sans-serif",
    )
    return FontProperties(family=family, size=size)


def _draw_header(axis: Any, graph: SemanticTaskGraph, width: float, view: str) -> None:
    axis.text(0.45, 0.28, str(graph["task_id"]), color=_INK, fontsize=13, weight="bold")
    axis.text(
        0.45,
        0.60,
        f"{view}  ·  {len(graph['nodes'])} calls  ·  {len(graph['task_groups'])} groups",
        color=_MUTED,
        fontsize=8,
    )
    instruction = str(graph.get("instruction", ""))
    if instruction:
        axis.text(
            0.45,
            0.90,
            "\n".join(textwrap.wrap(instruction, width=125)),
            color=_INK,
            fontsize=7.3,
            fontproperties=_font(7.3),
            va="top",
        )
    axis.text(
        width - 0.45, 0.28, "SemanticTaskGraph v1", ha="right", color=_MUTED, fontsize=7
    )


def _draw_group(
    axis: Any,
    group: Mapping[str, Any],
    nodes: Sequence[Mapping[str, Any]],
    box: tuple[float, float, float, float],
    statuses: Mapping[str, str],
) -> None:
    x, y, width, height = box
    task_type = str(group["task_type"])
    face = _TASK_COLORS.get(task_type, "#FFFFFF")
    _box(axis, x, y, width, height, face, _BORDER, radius=0.08)
    axis.text(
        x + 0.18,
        y + 0.23,
        f"{group['id']}  ·  {task_type}",
        color=_INK,
        fontsize=8,
        weight="bold",
    )
    axis.text(
        x + width - 0.18,
        y + 0.23,
        _group_status(nodes, statuses),
        ha="right",
        color=_MUTED,
        fontsize=7,
    )
    card_width = (width - 0.45) / 4.0
    card_y = y + 0.38
    for index, node in enumerate(nodes):
        column = index % 4
        row = index // 4
        _draw_call(
            axis,
            node,
            (
                x + 0.12 + column * card_width + card_width / 2,
                card_y + row * 0.4 + 0.16,
            ),
            statuses,
            width=card_width - 0.08,
        )


def _group_levels(groups: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    levels: dict[str, int] = {}
    for group in groups:
        group_id = str(group["id"])
        levels[group_id] = max(
            (levels[str(dependency)] + 1 for dependency in group["depends_on"]),
            default=0,
        )
    return levels


def _group_lane(nodes: Sequence[Mapping[str, Any]]) -> str:
    primary_nodes = [node for node in nodes if node.get("role") == "primary"]
    selected_nodes = primary_nodes or list(nodes)
    resources: set[str] = set()
    for node in selected_nodes:
        call_resources = node["call"].get("resources", {})
        if isinstance(call_resources, Mapping):
            resources.update(str(value) for value in call_resources.values())
    if "left" in resources and "right" in resources:
        return "handover"
    if "left" in resources:
        return "left"
    return "right"


def _lane_y(lane: str, left: float, handover: float, right: float) -> float:
    return {"left": left, "handover": handover, "right": right}[lane]


def _draw_timeline_group(
    axis: Any,
    group: Mapping[str, Any],
    nodes: Sequence[Mapping[str, Any]],
    x: float,
    y: float,
    lane: str,
    statuses: Mapping[str, str],
) -> None:
    edge = _ARM_COLORS[lane]
    summary, object_id, detail = _group_summary(group, nodes)
    _timeline_card(axis, x, y, edge, group, summary, object_id, detail, statuses)


def _draw_handover_group(
    axis: Any,
    group: Mapping[str, Any],
    nodes: Sequence[Mapping[str, Any]],
    x: float,
    left_y: float,
    right_y: float,
    statuses: Mapping[str, str],
) -> None:
    summary, object_id, detail = _group_summary(group, nodes)
    _timeline_card(
        axis,
        x,
        left_y,
        _ARM_COLORS["left"],
        group,
        summary,
        object_id,
        detail,
        statuses,
    )
    _timeline_card(
        axis,
        x,
        right_y,
        _ARM_COLORS["right"],
        group,
        summary,
        object_id,
        detail,
        statuses,
    )
    _arrow(
        axis,
        (x, left_y + 0.34),
        (x, right_y - 0.34),
        color=_ARM_COLORS["coordinated"],
        dashed=False,
    )
    axis.text(
        x,
        (left_y + right_y) / 2,
        "transfer",
        ha="center",
        va="center",
        fontsize=5.5,
        color=_ARM_COLORS["coordinated"],
        weight="bold",
    )


def _timeline_card(
    axis: Any,
    x: float,
    y: float,
    edge: str,
    group: Mapping[str, Any],
    summary: str,
    object_id: str | None,
    detail: str,
    statuses: Mapping[str, str],
) -> None:
    import matplotlib.patches as patches

    width, height = 1.5, 0.78
    status = _group_status(
        [{"id": node_id} for node_id in group["node_ids"]],
        {node_id: statuses.get(node_id, "unknown") for node_id in group["node_ids"]},
    )
    status_edge = _STATUS_COLORS.get("success" if status == "OK" else "unknown", edge)
    axis.add_patch(
        patches.FancyBboxPatch(
            (x - width / 2, y - height / 2),
            width,
            height,
            boxstyle="round,pad=.03,rounding_size=.07",
            facecolor=_TASK_COLORS.get(str(group["task_type"]), "#FFFFFF"),
            edgecolor=edge,
            linewidth=1.25,
            zorder=4,
        )
    )
    axis.text(
        x,
        y + 0.18,
        f"{group['id'].replace('step_', 'S')} · {group['task_type']}",
        ha="center",
        va="center",
        fontsize=6.6,
        weight="bold",
        color=_INK,
        zorder=5,
    )
    if object_id:
        axis.text(
            x,
            y - 0.01,
            _clip(object_id, 25),
            ha="center",
            va="center",
            fontsize=5.8,
            weight="bold",
            color=edge,
            zorder=5,
        )
    axis.text(
        x,
        y - 0.20,
        detail,
        ha="center",
        va="center",
        fontsize=5.4,
        color=_MUTED,
        zorder=5,
    )
    from matplotlib.patches import Circle

    axis.add_patch(
        Circle(
            (x + width / 2 - 0.12, y - height / 2 + 0.12),
            0.045,
            facecolor=_STATUS_COLORS["success"] if status == "OK" else status_edge,
            edgecolor="none",
            zorder=6,
        )
    )


def _group_summary(
    group: Mapping[str, Any], nodes: Sequence[Mapping[str, Any]]
) -> tuple[str, str | None, str]:
    object_id = None
    relation = None
    call_names: list[str] = []
    for node in nodes:
        call = node["call"]
        args = call.get("arguments", call)
        if isinstance(args, Mapping):
            object_id = object_id or args.get("object")
            relation = relation or args.get("relation")
        call_names.append(
            str(call.get("call_id", call.get("kind", "call"))).replace("gen_sim.", "")
        )
    if str(group["task_type"]) == "E4":
        summary = "handover"
    elif relation:
        summary = str(relation)
    else:
        summary = str(group["task_type"])
    return summary, None if object_id is None else str(object_id), _clip(summary, 20)


def _draw_skill_table(axis: Any, width: float, height: float) -> None:
    axis.text(
        0.55, 7.20, "Atomic Skill Mapping", fontsize=8.5, color=_INK, weight="bold"
    )
    entries = list(ATOMIC_SKILL_MAP.items())
    columns = 2
    row_height = 0.36
    col_width = (width - 1.1) / columns
    for index, (task_type, skills) in enumerate(entries):
        row, column = divmod(index, columns)
        x = 0.55 + column * col_width
        y = 7.45 + row * row_height
        axis.text(x, y, f"{task_type}: {skills}", fontsize=6.7, color=_INK, va="top")


def _draw_call(
    axis: Any,
    node: Mapping[str, Any],
    center: tuple[float, float],
    statuses: Mapping[str, str],
    *,
    width: float = 2.7,
) -> None:
    import matplotlib.patches as patches

    x, y = center
    call = node["call"]
    arm = _arm(call)
    status = statuses.get(str(node["id"]), "unknown")
    label = _call_label(node)
    height = 0.28 if width < 1.0 else 0.52
    axis.add_patch(
        patches.FancyBboxPatch(
            (x - width / 2, y - height / 2),
            width,
            height,
            boxstyle="round,pad=0.03,rounding_size=0.05",
            facecolor=_TASK_COLORS.get(str(node["task_type"]), "#FFFFFF"),
            edgecolor=_STATUS_COLORS.get(status, _BORDER),
            linewidth=1.2 if status != "unknown" else 0.8,
            zorder=3,
        )
    )
    axis.text(
        x,
        y,
        label,
        ha="center",
        va="center",
        color=_INK,
        fontsize=5.5 if width < 1.0 else 7,
    )
    axis.plot(
        x - width / 2 + 0.08,
        y,
        marker="o",
        markersize=3.5,
        color=_ARM_COLORS[arm],
        zorder=4,
    )


def _draw_group_dependencies(
    axis: Any,
    groups: Sequence[Mapping[str, Any]],
    boxes: Mapping[str, tuple[float, float, float, float]],
) -> None:
    for group in groups:
        target = boxes[str(group["id"])]
        for dependency in group["depends_on"]:
            source = boxes[str(dependency)]
            _arrow(
                axis,
                (source[0] + source[2] / 2, source[1] + source[3]),
                (target[0] + target[2] / 2, target[1]),
                color=_DEPENDENCY,
                dashed=True,
            )


def _arrow(
    axis: Any,
    source: tuple[float, float],
    target: tuple[float, float],
    *,
    color: str,
    dashed: bool,
) -> None:
    from matplotlib.patches import FancyArrowPatch

    axis.add_patch(
        FancyArrowPatch(
            source,
            target,
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=0.8,
            linestyle="--" if dashed else "-",
            color=color,
            shrinkA=4,
            shrinkB=4,
            zorder=1,
        )
    )


def _box(
    axis: Any,
    x: float,
    y: float,
    width: float,
    height: float,
    face: str,
    edge: str,
    *,
    radius: float,
) -> None:
    from matplotlib.patches import FancyBboxPatch

    axis.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle=f"round,pad=0.02,rounding_size={radius}",
            facecolor=face,
            edgecolor=edge,
            linewidth=0.8,
            zorder=2,
        )
    )


def _group_status(
    nodes: Sequence[Mapping[str, Any]], statuses: Mapping[str, str]
) -> str:
    values = [statuses.get(str(node["id"]), "unknown") for node in nodes]
    if values and all(value == "success" for value in values):
        return "OK"
    if "failed" in values:
        return "FAIL"
    if "running" in values:
        return "RUN"
    return ""


def _arm(call: Mapping[str, Any]) -> str:
    resources = call.get("resources", {})
    if not isinstance(resources, Mapping):
        return "auto"
    values = [
        str(resources.get(key, ""))
        for key in ("primary", "destination", "source", "left", "right")
    ]
    if "left" in values and "right" in values:
        return "coordinated"
    if "left" in values:
        return "left"
    if "right" in values:
        return "right"
    return "auto"


def _call_label(node: Mapping[str, Any]) -> str:
    call = node["call"]
    if call.get("kind") == "registered":
        name = str(call.get("call_id", "registered"))
    else:
        name = str(call.get("kind", "call"))
    arguments = call.get("arguments", call)
    if isinstance(arguments, Mapping):
        subject = arguments.get("object") or call.get("object")
        relation = arguments.get("relation")
        if subject:
            name = f"{name}: {subject}"
        if relation:
            name = f"{name} [{relation}]"
    return _clip(name.replace("gen_sim.", ""), 34)


def _clip(value: str, limit: int) -> str:
    return value if len(value) <= limit else value[: limit - 1] + "…"


def _png(figure: Any) -> bytes:
    from io import BytesIO

    from matplotlib.backends.backend_agg import FigureCanvasAgg

    buffer = BytesIO()
    FigureCanvasAgg(figure).print_png(buffer)
    payload = buffer.getvalue()
    if not payload.startswith(_PNG_SIGNATURE):
        raise RuntimeError("Semantic graph renderer did not produce a PNG.")
    return payload
