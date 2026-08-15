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

from embodichain.gen_sim.action_engine.compiler import (
    compile_task_agent,
    compile_task_agent_v2,
)
from embodichain.gen_sim.action_engine.domain import (
    EXECUTION_PROGRAM_SCHEMA,
    MOTION_POLICY_VERSION,
    TASK_AGENT_SCHEMA,
    validate_execution_program,
)
from embodichain.gen_sim.action_engine.graph_visualization import (
    _RuntimeOverlay,
    _dag_levels,
    _dag_positions,
    _dependency_pairs,
    _graph_data,
    render_seed_task_graph_png,
    render_task_graph_png,
)

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _image(payload: bytes) -> Image.Image:
    assert payload.startswith(_PNG_SIGNATURE)
    image = Image.open(BytesIO(payload)).convert("RGB")
    extrema = ImageStat.Stat(image).extrema
    assert any(low != high for low, high in extrema)
    return image


def _contains_color(
    image: Image.Image,
    color: str,
    *,
    minimum_pixels: int = 8,
    tolerance: int = 4,
) -> bool:
    target = tuple(bytes.fromhex(color.removeprefix("#")))
    matches = 0
    payload = image.tobytes()
    for offset in range(0, len(payload), 3):
        pixel = payload[offset : offset + 3]
        if all(
            abs(channel - expected) <= tolerance
            for channel, expected in zip(pixel, target)
        ):
            matches += 1
            if matches >= minimum_pixels:
                return True
    return False


def _chain_program() -> dict[str, object]:
    return compile_task_agent(
        {
            "schema_version": TASK_AGENT_SCHEMA,
            "task": "中文单链任务",
            "goal": "Pick up the cup and keep it hovering.",
            "semantic_steps": [
                {
                    "id": "s01_hover",
                    "operator": "hold_hover",
                    "object": "cup",
                    "actor": {"mode": "required", "arm": "left_arm"},
                    "goal": {},
                    "depends_on": [],
                }
            ],
        }
    )


def _action(
    action_class: str,
    arm: str | None,
    target: str,
) -> dict[str, object]:
    actor = {"mode": "auto"} if arm is None else {"mode": "required", "arm": arm}
    return {
        "atomic_action_class": action_class,
        "actor": actor,
        "control": "arm",
        "target_binding": {"kind": "object", "object": target},
        "motion_policy": {"modifiers": []},
    }


def _fork_join_program() -> dict[str, object]:
    program = {
        "schema_version": EXECUTION_PROGRAM_SCHEMA,
        "task": "fork_join_demo",
        "goal_description": "Move two objects in parallel, then finish.",
        "start": "v_start",
        "goal": "v_goal",
        "nodes": [
            {"id": "v_start", "semantic": "ready"},
            {"id": "v_left", "semantic": "left branch active"},
            {"id": "v_right", "semantic": "right branch active"},
            {"id": "v_join", "semantic": "branches complete"},
            {"id": "v_goal", "semantic": "task complete"},
        ],
        "edges": [
            {
                "id": "e_left_pick",
                "source": "v_start",
                "target": "v_left",
                "semantic_step_id": "s_left",
                "actions": [_action("PickUp", "left_arm", "left_object")],
                "depends_on": [],
                "resources": ["arm:left_arm"],
            },
            {
                "id": "e_right_pick",
                "source": "v_start",
                "target": "v_right",
                "semantic_step_id": "s_right",
                "actions": [_action("PickUp", "right_arm", "right_object")],
                "depends_on": [],
                "resources": ["arm:right_arm"],
            },
            {
                "id": "e_left_join",
                "source": "v_left",
                "target": "v_join",
                "semantic_step_id": "s_left",
                "actions": [_action("MoveHeldObject", "left_arm", "left_object")],
                "depends_on": ["e_left_pick"],
                "resources": ["arm:left_arm"],
            },
            {
                "id": "e_right_join",
                "source": "v_right",
                "target": "v_join",
                "semantic_step_id": "s_right",
                "actions": [_action("MoveHeldObject", "right_arm", "right_object")],
                "depends_on": ["e_right_pick"],
                "resources": ["arm:right_arm"],
            },
            {
                "id": "e_finish",
                "source": "v_join",
                "target": "v_goal",
                "semantic_step_id": "s_finish",
                "actions": [_action("MoveJoints", None, "home")],
                "depends_on": ["e_left_join", "e_right_join"],
                "resources": ["arm:auto"],
            },
        ],
        "semantic_steps": [
            {
                "id": "s_left",
                "parent_step_id": "s_left",
                "operator": "place_relative",
                "object": "left_object",
                "actor": {"mode": "required", "arm": "left_arm"},
                "goal": {"relation": "on"},
                "depends_on": [],
                "postcondition": {"type": "semantic_goal"},
                "edge_ids": ["e_left_pick", "e_left_join"],
            },
            {
                "id": "s_right",
                "parent_step_id": "s_right",
                "operator": "place_relative",
                "object": "right_object",
                "actor": {"mode": "required", "arm": "right_arm"},
                "goal": {"relation": "on"},
                "depends_on": [],
                "postcondition": {"type": "semantic_goal"},
                "edge_ids": ["e_right_pick", "e_right_join"],
            },
            {
                "id": "s_finish",
                "parent_step_id": "s_finish",
                "operator": "hold_hover",
                "object": "home",
                "actor": {"mode": "auto"},
                "goal": {},
                "depends_on": ["s_left", "s_right"],
                "postcondition": {"type": "semantic_goal"},
                "edge_ids": ["e_finish"],
            },
        ],
        "allocation_groups": [
            {
                "id": "g_parallel",
                "semantic_step_ids": ["s_left", "s_right"],
                "arm_constraint": "distinct_arms",
                "execution_policy": "parallel_if_feasible",
                "parallel_action_classes": ["PickUp"],
                "workspace_policy": "shared_target_serial",
            }
        ],
        "motion_policy_version": MOTION_POLICY_VERSION,
    }
    return validate_execution_program(program)


def test_seed_renderer_produces_a_compact_headless_png() -> None:
    first = _image(render_seed_task_graph_png(_chain_program()))
    second = _image(render_seed_task_graph_png(_chain_program()))

    assert first.size == second.size
    assert first.width > first.height
    assert first.height < 1_200


def test_fork_join_layout_uses_actor_lanes_and_dependency_links() -> None:
    program = _fork_join_program()
    data = _graph_data(program, _RuntimeOverlay({}, {}, {}))
    levels = _dag_levels(data.graph)
    positions = _dag_positions(
        data,
        levels,
        {"left": 2.6, "auto": 7.8, "right": 13.0},
    )

    assert positions["v_left"][0] < 5.15
    assert positions["v_right"][0] > 10.45
    assert positions["v_start"][0] == pytest.approx(7.8)
    assert positions["v_join"][0] == pytest.approx(7.8)
    assert ("e_left_join", "e_finish") in _dependency_pairs(data)
    assert ("e_right_join", "e_finish") in _dependency_pairs(data)

    image = _image(render_seed_task_graph_png(program))
    assert image.width > image.height
    assert _contains_color(image, "#168A78")
    assert _contains_color(image, "#D97706")
    assert _contains_color(image, "#3973B7")


def test_parallel_single_phase_edges_are_rendered_as_a_multigraph() -> None:
    program = compile_task_agent(
        {
            "schema_version": TASK_AGENT_SCHEMA,
            "task": "parallel_press",
            "goal": "Press both independent buttons.",
            "semantic_steps": [
                {
                    "id": "s_left",
                    "operator": "press",
                    "object": "left_button",
                    "actor": {"mode": "required", "arm": "left_arm"},
                    "goal": {},
                    "depends_on": [],
                },
                {
                    "id": "s_right",
                    "operator": "press",
                    "object": "right_button",
                    "actor": {"mode": "required", "arm": "right_arm"},
                    "goal": {},
                    "depends_on": [],
                },
            ],
        }
    )
    assert {(edge["source"], edge["target"]) for edge in program["edges"]} == {
        ("v0_start", "v_goal")
    }

    image = _image(render_seed_task_graph_png(program))

    assert _contains_color(image, "#168A78")
    assert _contains_color(image, "#D97706")


def test_runtime_renderer_overlays_observed_statuses() -> None:
    program = _fork_join_program()
    runtime = {
        **program,
        "runtime": {
            "schema_version": "action_engine_runtime_record_v1",
            "status": "failed",
            "events": [
                {
                    "event": "edge",
                    "edge_id": "e_left_pick",
                    "arm": "left_arm",
                    "status": "executed",
                },
                {
                    "event": "edge",
                    "edge_id": "e_right_pick",
                    "arm": "right_arm",
                    "status": "failed",
                },
            ],
        },
    }

    image = _image(render_task_graph_png(runtime))

    assert _contains_color(image, "#25834B")
    assert _contains_color(image, "#C43E3E")


def test_runtime_renderer_accepts_v2_seed_graph_envelope() -> None:
    task_agent = {
        "schema_version": TASK_AGENT_SCHEMA,
        "task": "v2_runtime_overlay",
        "goal": "Hold the cup.",
        "semantic_steps": [
            {
                "id": "hold",
                "operator": "hold_hover",
                "object": "cup",
                "actor": {"mode": "required", "arm": "left_arm"},
                "goal": {},
                "depends_on": [],
            }
        ],
    }
    seed = compile_task_agent_v2(task_agent)
    document = {
        **seed,
        "runtime": {
            "schema_version": "action_engine_runtime_record_v2",
            "status": "success",
            "events": [],
        },
    }

    image = _image(render_task_graph_png(document))

    assert _contains_color(image, "#25834B")


def test_runtime_record_without_program_is_rejected() -> None:
    with pytest.raises(ValueError, match="do not contain graph topology"):
        render_task_graph_png(
            {
                "schema_version": "action_engine_runtime_record_v1",
                "events": [],
            }
        )
