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
    make_relative_seed_task_graph,
    validate_seed_task_graph,
)
from embodichain.gen_sim.action_agent_pipeline.graph_visualization import (
    _display_edge,
    _display_edge_lines,
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
    _assert_nonblank_png(render_seed_task_graph_png(seed))


def test_seed_renderer_is_stable_and_supports_chinese_task_name() -> None:
    seed = _relative_seed_graph(task_name="中文可视化任务")

    first = _assert_nonblank_png(render_seed_task_graph_png(seed))
    second = _assert_nonblank_png(render_seed_task_graph_png(seed))

    assert first.size == second.size


def test_runtime_renderer_displays_grounded_arm_position_and_status() -> None:
    runtime = _runtime_graph()
    _, _, edges = _validated_task_display_graph(runtime)
    group = _task_groups(runtime, edges)[0]

    lines = _display_edge_lines(_display_edge(edges[0], group))

    assert lines[0].endswith("[R] PickUp")
    assert any("executed" in line for line in lines)
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
    runtime = deepcopy(seed)
    runtime["schema_version"] = "runtime_task_graph_v1"
    runtime["seed_graph_schema_version"] = "seed_task_graph_v2"
    runtime["seed_graph_hash"] = "a" * 64
    runtime["run_id"] = "20260729T000000.000000Z"
    runtime["episode_index"] = 0
    runtime["env_id"] = 0
    runtime["robot_profile"] = "dual_franka"
    first_runtime = runtime["edges"][0]["actions"][0].setdefault("runtime", {})
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
