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

from copy import deepcopy
from io import BytesIO
from types import SimpleNamespace

from PIL import Image, ImageStat
import pytest

from embodichain.gen_sim.action_agent_pipeline.generation.seed_task_graph import (
    make_arrangement_seed_task_graph,
    make_relative_seed_task_graph,
    validate_seed_task_graph,
)
from embodichain.gen_sim.action_agent_pipeline.graph_visualization import (
    _compact_node_link_layout,
    _display_edge,
    _display_edge_lane,
    _display_edge_lines,
    _node_link_edge_lines,
    _task_groups,
    _validated_task_display_graph,
    render_seed_task_graph_png,
    render_task_graph_png,
)

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def test_seed_v2_renderer_shows_complete_symbolic_topology() -> None:
    seed = _relative_seed_graph()

    graph, nodes, edges = _validated_task_display_graph(seed)
    groups = _task_groups(seed, edges)
    first_label = _display_edge_lines(_display_edge(edges[0], groups[0]))

    assert graph.number_of_nodes() == len(seed["nodes"])
    assert graph.number_of_edges() == len(seed["edges"])
    assert len(nodes) == len(seed["nodes"])
    assert "PickUp" in first_label[0]
    assert any("object a" in line for line in first_label)
    assert any("object" in line and "default_pickup" in line for line in first_label)
    assert _display_edge_lane(_display_edge(edges[0], groups[0])) == "left"
    assert _display_edge_lane(_display_edge(edges[5], groups[1])) == "right"
    _assert_nonblank_png(render_seed_task_graph_png(seed))


def test_seed_renderer_is_stable_and_supports_chinese_task_name() -> None:
    seed = _relative_seed_graph(task_name="中文可视化任务")

    first = _assert_nonblank_png(render_seed_task_graph_png(seed))
    second = _assert_nonblank_png(render_seed_task_graph_png(seed))

    assert first.size == second.size


def test_seed_node_link_layout_folds_each_semantic_step_deterministically() -> None:
    seed = _relative_seed_graph()
    _, _, edges = _validated_task_display_graph(seed)
    groups = _task_groups(seed, edges)
    display_edges = {
        edge["id"]: _display_edge(edge, group)
        for group in groups
        for edge in group.edges
    }

    positions, layouts = _compact_node_link_layout(
        groups,
        display_edges=display_edges,
        center_x=7.3,
    )

    assert set(positions) == {node["id"] for node in seed["nodes"]}
    assert positions[seed["start"]][0] == pytest.approx(7.3)
    assert all(positions[edge["target"]][0] < 7.3 for edge in groups[0].edges[:-1])
    assert all(positions[edge["target"]][0] > 7.3 for edge in groups[1].edges[:-1])
    assert len({positions[edge["target"]][0] for edge in groups[0].edges[:-1]}) == 2
    assert positions[groups[0].edges[-1]["target"]][0] == pytest.approx(7.3)
    assert layouts[0].bottom == pytest.approx(layouts[1].top)


def test_seed_auto_actor_stays_visually_unassigned() -> None:
    seed = _relative_seed_graph()
    first_step = seed["semantic_steps"][0]
    first_step["actor"] = {"mode": "auto"}
    edge_by_id = {edge["id"]: edge for edge in seed["edges"]}
    for edge_id in first_step["edge_ids"]:
        edge_by_id[edge_id]["actions"][0]["actor"] = {"mode": "auto"}

    _, _, edges = _validated_task_display_graph(seed)
    groups = _task_groups(seed, edges)
    group = groups[0]
    display_edges = {
        edge["id"]: _display_edge(edge, candidate_group)
        for candidate_group in groups
        for edge in candidate_group.edges
    }
    positions, _ = _compact_node_link_layout(
        groups,
        display_edges=display_edges,
        center_x=7.3,
    )

    assert _display_edge_lane(_display_edge(edges[0], group)) == "auto"
    assert 5.15 < positions[edges[0]["target"]][0] < 9.45
    _assert_nonblank_png(render_seed_task_graph_png(seed))


def test_coordinated_edge_labels_each_arm_action_and_motion_policy() -> None:
    seed = _relative_seed_graph()
    edge = seed["edges"][0]
    right_action = deepcopy(edge["actions"][0])
    right_action["actor"] = {"mode": "required", "arm": "right_arm"}
    edge["actions"].append(right_action)
    _, _, edges = _validated_task_display_graph(seed)
    group = _task_groups(seed, edges)[0]

    lines = _node_link_edge_lines(_display_edge(edges[0], group))

    assert "[L] PickUp" in lines[1]
    assert "[R] PickUp" in lines[2]
    assert all("default_pickup" in line for line in lines[1:])


def test_large_seed_uses_a_bounded_vertical_node_link_canvas() -> None:
    steps = tuple(
        SimpleNamespace(
            runtime_uid=f"can_{index}",
            slot_index=index,
            orientation_goal="preserve",
            orientation_axis="none",
        )
        for index in range(6)
    )
    seed = make_arrangement_seed_task_graph(
        "large_arrangement",
        SimpleNamespace(
            task_description="arrange cans by size",
            order_by="size",
            order_direction="ascending",
            axis="world_x",
            anchor="center",
            semantic_order=tuple(step.runtime_uid for step in steps),
            steps=steps,
        ),
    )

    image = _assert_nonblank_png(render_seed_task_graph_png(seed))

    assert len(seed["nodes"]) == 31
    assert image.height > image.width
    assert image.height <= 8000


def test_runtime_renderer_displays_grounded_arm_position_and_status() -> None:
    runtime = _runtime_graph()
    _, _, edges = _validated_task_display_graph(runtime)
    groups = _task_groups(runtime, edges)
    group = groups[0]

    lines = _display_edge_lines(_display_edge(edges[0], group))
    display_edges = {
        edge["id"]: _display_edge(edge, candidate_group)
        for candidate_group in groups
        for edge in candidate_group.edges
    }
    positions, _ = _compact_node_link_layout(
        groups,
        display_edges=display_edges,
        center_x=7.3,
    )

    assert lines[0].endswith("[R] PickUp")
    assert _display_edge_lane(_display_edge(edges[0], group)) == "right"
    assert any("executed" in line for line in lines)
    assert all(positions[edge["target"]][0] > 7.3 for edge in group.edges[:-1])
    _assert_nonblank_png(render_task_graph_png(runtime))


def test_seed_rejects_runtime_geometry_leakage() -> None:
    seed = _relative_seed_graph()
    seed["edges"][0]["actions"][0]["target_binding"]["position"] = [1.0, 2.0, 3.0]

    with pytest.raises(ValueError, match="grounded field"):
        validate_seed_task_graph(seed)

    seed = _relative_seed_graph()
    seed["semantic_steps"][0]["goal"]["sample_interval"] = 30
    with pytest.raises(ValueError, match="grounded field"):
        validate_seed_task_graph(seed)


def test_seed_rejects_invalid_symbolic_action_contract() -> None:
    seed = _relative_seed_graph()
    seed["edges"][0]["actions"][0]["target_binding"]["unexpected"] = "value"

    with pytest.raises(ValueError, match="invalid fields"):
        validate_seed_task_graph(seed)

    seed = _relative_seed_graph()
    seed["edges"][0]["actions"][0]["motion_policy"] = "default_home"
    with pytest.raises(ValueError, match="requires motion policy"):
        validate_seed_task_graph(seed)


def test_seed_rejects_v1_with_regeneration_message() -> None:
    with pytest.raises(ValueError, match="--overwrite"):
        validate_seed_task_graph({"schema_version": "seed_task_graph_v1"})


def test_seed_rejects_invalid_dependency_and_edge_coverage() -> None:
    seed = _relative_seed_graph()
    seed["semantic_steps"][0]["depends_on"] = ["unknown"]
    with pytest.raises(ValueError, match="non-prior"):
        validate_seed_task_graph(seed)

    seed = _relative_seed_graph()
    seed["semantic_steps"][0]["edge_ids"] = seed["semantic_steps"][0]["edge_ids"][:-1]
    with pytest.raises(ValueError, match="cover every Seed edge"):
        validate_seed_task_graph(seed)

    seed = _relative_seed_graph()
    seed["edges"][0]["target"], seed["edges"][2]["target"] = (
        seed["edges"][2]["target"],
        seed["edges"][0]["target"],
    )
    with pytest.raises(ValueError, match="topology"):
        validate_seed_task_graph(seed)


def _relative_seed_graph(task_name: str = "relative_visualization") -> dict:
    placements = (
        SimpleNamespace(
            intent="place_relative",
            moved_runtime_uid="object_a",
            reference_runtime_uid="object_c",
            relation="left_of",
            reference_is_initial_pose=False,
            orientation_goal="preserve",
            orientation_axis="none",
            orientation_align_to_runtime_uid=None,
            arm_request="left",
            step_id="s01_move_left",
            depends_on=(),
        ),
        SimpleNamespace(
            intent="place_relative",
            moved_runtime_uid="object_a",
            reference_runtime_uid="object_c",
            relation="right_of",
            reference_is_initial_pose=False,
            orientation_goal="preserve",
            orientation_axis="none",
            orientation_align_to_runtime_uid=None,
            arm_request="right",
            step_id="s02_move_right",
            depends_on=("s01_move_left",),
        ),
    )
    return make_relative_seed_task_graph(
        task_name,
        SimpleNamespace(
            intent="place_relative",
            placements=placements,
            coordinated_direction=None,
            coordinated_terminal_behavior=None,
        ),
    )


def _runtime_graph() -> dict:
    seed = _relative_seed_graph()
    first_seed_step = seed["semantic_steps"][0]
    first_seed_step["actor"] = {"mode": "auto"}
    edge_by_id = {edge["id"]: edge for edge in seed["edges"]}
    for edge_id in first_seed_step["edge_ids"]:
        edge_by_id[edge_id]["actions"][0]["actor"] = {"mode": "auto"}
    runtime = deepcopy(seed)
    runtime["schema_version"] = "runtime_task_graph_v1"
    runtime["seed_graph_schema_version"] = "seed_task_graph_v2"
    runtime["seed_graph_hash"] = "a" * 64
    runtime["run_id"] = "20260729T000000.000000Z"
    runtime["episode_index"] = 0
    runtime["env_id"] = 0
    runtime["robot_profile"] = "dual_franka"
    runtime["semantic_steps"][0]["runtime"] = {
        "status": "executed",
        "assigned_arm": "right_arm",
        "postcondition": {"evaluated": True, "success": True},
    }
    runtime_edge_by_id = {edge["id"]: edge for edge in runtime["edges"]}
    for edge_id in runtime["semantic_steps"][0]["edge_ids"]:
        runtime_edge_by_id[edge_id]["actions"][0]["runtime"] = {
            "assigned_arm": "right_arm",
            "status": "executed",
        }
    first_runtime = runtime["edges"][0]["actions"][0]["runtime"]
    first_runtime.update(
        {
            "assigned_arm": "right_arm",
            "resolved_target_position": [0.1, 0.2, 0.3],
            "status": "executed",
        }
    )
    return runtime


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
