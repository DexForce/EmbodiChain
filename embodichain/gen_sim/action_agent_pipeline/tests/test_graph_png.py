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

from io import BytesIO

from PIL import Image, ImageStat
import pytest

from embodichain.gen_sim.action_agent_pipeline.generation.seed_task_graph import (
    seed_task_graph_hash,
)
from embodichain.gen_sim.action_agent_pipeline.generation.visualization.graph_png import (
    _build_seed_display_graph,
    _edge_semantic_step_ids,
    _task_positions,
    _validated_task_display_graph,
    render_seed_task_graph_png,
    render_task_graph_png,
)

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def test_seed_graph_reuses_entities_and_preserves_semantic_relations() -> None:
    seed_graph = _relative_seed_graph()

    display_graph = _build_seed_display_graph(seed_graph)

    assert "entity:object_a" in display_graph
    assert "entity:object_c" in display_graph
    assert (
        sum(
            1
            for _, target, attributes in display_graph.edges(data=True)
            if target == "entity:object_a" and attributes["relation"] == "acts_on"
        )
        == 2
    )
    assert (
        sum(
            1
            for _, target, attributes in display_graph.edges(data=True)
            if target == "entity:object_c" and attributes["relation"] == "references"
        )
        == 2
    )
    assert display_graph.has_edge("step:s01_move_left", "step:s02_move_right")


def test_seed_renderer_produces_nonblank_png_with_stable_dimensions() -> None:
    seed_graph = _relative_seed_graph(task_name="中文可视化任务")

    first = render_seed_task_graph_png(seed_graph)
    second = render_seed_task_graph_png(seed_graph)

    first_image = _assert_nonblank_png(first)
    second_image = _assert_nonblank_png(second)
    assert first_image.size == second_image.size


def test_arrangement_members_are_displayed_without_virtual_object() -> None:
    seed_graph = _arrangement_seed_graph(order_constraint="ordered")

    display_graph = _build_seed_display_graph(seed_graph)
    member_labels = [
        attributes["label"]
        for _, _, attributes in display_graph.edges(data=True)
        if attributes["relation"] == "member"
    ]

    assert "entity:__arrangement__" not in display_graph
    assert member_labels == ["member #1", "member #2", "member #3"]


def test_task_graph_uses_serpentine_layout_for_long_single_chain() -> None:
    task_graph = _task_graph(edge_count=30)
    display_graph, _, _ = _validated_task_display_graph(task_graph)

    positions, is_chain = _task_positions(display_graph, task_graph)

    assert is_chain
    assert positions["v0"] == (0.0, 0.0)
    assert positions["v5"][0] > positions["v0"][0]
    assert positions["v6"][0] == positions["v5"][0]
    assert positions["v7"][0] < positions["v6"][0]
    assert len(set(positions.values())) == 31


def test_task_renderer_supports_semantic_groups_and_dual_arm_edges() -> None:
    task_graph = _task_graph(edge_count=2, dual_arm=True)
    task_graph["semantic_steps"] = [
        {
            "id": "s01_move",
            "edge_ids": ["e01"],
        },
        {
            "id": "s02_finish",
            "edge_ids": ["e02"],
        },
    ]
    task_graph["seed_graph_hash"] = seed_task_graph_hash(_relative_seed_graph())

    display_graph, _, edge_records = _validated_task_display_graph(task_graph)
    semantic_steps = _edge_semantic_step_ids(task_graph, edge_records)
    image = render_task_graph_png(task_graph)

    assert display_graph.number_of_edges() == 2
    assert semantic_steps == {"e01": "s01_move", "e02": "s02_finish"}
    _assert_nonblank_png(image)


@pytest.mark.parametrize(
    ("mutation", "error_match"),
    [
        (lambda graph: graph.update(nodes=[]), "non-empty nodes"),
        (
            lambda graph: graph["nodes"].append({"id": "v0", "semantic": "duplicate"}),
            "Duplicate task graph node",
        ),
        (
            lambda graph: graph["edges"][0].update(target="unknown"),
            "unknown endpoint",
        ),
        (
            lambda graph: graph["edges"].append(
                {
                    "id": "cycle",
                    "source": "v2",
                    "target": "v0",
                    "left_arm_action": {"atomic_action_class": "MoveJoints"},
                    "right_arm_action": None,
                }
            ),
            "directed acyclic",
        ),
    ],
)
def test_task_renderer_rejects_invalid_graphs(mutation, error_match: str) -> None:
    task_graph = _task_graph(edge_count=2)
    mutation(task_graph)

    with pytest.raises(ValueError, match=error_match):
        render_task_graph_png(task_graph)


def test_seed_renderer_rejects_non_prior_dependency() -> None:
    seed_graph = _relative_seed_graph()
    seed_graph["steps"][0]["depends_on"] = ["s02_move_right"]

    with pytest.raises(ValueError, match="non-prior"):
        render_seed_task_graph_png(seed_graph)


def _assert_nonblank_png(content: bytes) -> Image.Image:
    assert content.startswith(_PNG_SIGNATURE)
    image = Image.open(BytesIO(content))
    image.load()
    assert image.mode in {"RGB", "RGBA"}
    assert image.width >= 640
    assert image.height >= 480
    assert any(
        channel_extrema[0] != channel_extrema[1]
        for channel_extrema in image.getextrema()
    )
    assert max(ImageStat.Stat(image.convert("RGB")).var) > 0.0
    return image


def _relative_seed_graph(task_name: str = "relative_visualization") -> dict:
    first_step = _relative_seed_step(
        step_id="s01_move_left",
        arm="left_arm",
        relation="left_of",
        dependencies=[],
    )
    second_step = _relative_seed_step(
        step_id="s02_move_right",
        arm="right_arm",
        relation="right_of",
        dependencies=["s01_move_left"],
    )
    return {
        "schema_version": "seed_task_graph_v1",
        "task": task_name,
        "route": "object_manipulation",
        "program": "place_relative",
        "steps": [first_step, second_step],
    }


def _relative_seed_step(
    *,
    step_id: str,
    arm: str,
    relation: str,
    dependencies: list[str],
) -> dict:
    return {
        "id": step_id,
        "operator": "place_relative",
        "object": "object_a",
        "actor": {"mode": "required", "arm": arm},
        "goal": {
            "relation": relation,
            "reference_object": "object_c",
            "reference_state": "live",
            "orientation_goal": "preserve",
            "orientation_axis": "none",
        },
        "depends_on": dependencies,
        "postcondition": {
            "type": "semantic_goal",
            "operator": "place_relative",
            "relation": relation,
        },
    }


def _arrangement_seed_graph(*, order_constraint: str) -> dict:
    return {
        "schema_version": "seed_task_graph_v1",
        "task": "arrangement_visualization",
        "route": "arrangement_line",
        "program": "arrange_in_line",
        "steps": [
            {
                "id": "s01_arrange",
                "operator": "arrange_in_line",
                "object": "__arrangement__",
                "actor": {"mode": "auto"},
                "goal": {
                    "layout": "line",
                    "objects": ["object_a", "object_b", "object_c"],
                    "order_constraint": order_constraint,
                    "axis": "world_y",
                    "anchor": "table_center",
                    "order_by": "explicit",
                    "order_direction": "given",
                },
                "depends_on": [],
                "postcondition": {
                    "type": (
                        "objects_in_ordered_line"
                        if order_constraint == "ordered"
                        else "objects_in_line"
                    ),
                },
            }
        ],
    }


def _task_graph(*, edge_count: int, dual_arm: bool = False) -> dict:
    nodes = [
        {
            "id": f"v{index}",
            "semantic": f"Node {index} with a deliberately descriptive label",
        }
        for index in range(edge_count + 1)
    ]
    edges = []
    for index in range(1, edge_count + 1):
        left_action = {
            "atomic_action_class": "PickUp" if index == 1 else "MoveHeldObject",
            "target_object": {"obj_name": f"object_{index}"},
        }
        right_action = (
            {
                "atomic_action_class": "MoveHeldObject",
                "target_object": {"obj_name": f"object_{index}"},
            }
            if dual_arm and index == 1
            else None
        )
        edges.append(
            {
                "id": f"e{index:02d}",
                "source": f"v{index - 1}",
                "target": f"v{index}",
                "left_arm_action": left_action,
                "right_arm_action": right_action,
            }
        )
    return {
        "task": "compiled_visualization",
        "start": "v0",
        "goal": f"v{edge_count}",
        "nodes": nodes,
        "edges": edges,
    }
