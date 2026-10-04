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
from pathlib import Path

from PIL import Image

from embodichain.gen_sim.task_engine.semantic_graph_visualization import (
    render_semantic_task_graph_png,
    write_semantic_task_graph_png,
)
from embodichain.gen_sim.task_engine.cli import build_parser

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _graph() -> dict[str, object]:
    return {
        "schema_version": "semantic_task_graph/v1",
        "task_id": "visual_demo",
        "instruction": "Pick and place the cube",
        "planner_route": "offline",
        "integration_fingerprint": "0" * 64,
        "targets": {},
        "nodes": [
            {
                "id": "pick_cube",
                "call": {"kind": "pick", "object": "cube"},
                "depends_on": [],
                "task_instance_id": "pick_group",
                "task_type": "E1",
                "role": "primary",
            },
            {
                "id": "place_cube",
                "call": {
                    "kind": "place",
                    "object": "cube",
                    "inside": "tray_inside",
                },
                "depends_on": ["pick_cube"],
                "task_instance_id": "place_group",
                "task_type": "E1",
                "role": "primary",
            },
        ],
        "task_groups": [
            {
                "id": "pick_group",
                "task_type": "E1",
                "node_ids": ["pick_cube"],
                "depends_on": [],
                "success": {"kind": "call_completed"},
            },
            {
                "id": "place_group",
                "task_type": "E1",
                "node_ids": ["place_cube"],
                "depends_on": ["pick_group"],
                "success": {"kind": "call_completed"},
            },
        ],
        "success": {"kind": "all_task_groups"},
    }


def _report() -> dict[str, object]:
    return {
        "schema_version": "task_program_execution_report/v1",
        "status": "succeeded",
        "task_id": "visual_demo",
        "semantic_call_count": 2,
        "integration_fingerprint": "0" * 64,
        "record_dir": "/tmp/trajectory",
        "environments": [
            {
                "env_id": 0,
                "success": True,
                "terminal_reason": "success",
                "semantic_success": {"pick_group": True, "place_group": True},
            }
        ],
        "runtime_result": {
            "segments": [
                {"name": "pick_cube", "success": True},
                {"name": "place_cube", "success": True},
            ]
        },
        "failure": None,
    }


def test_semantic_graph_renderer_produces_group_and_call_pngs(tmp_path: Path) -> None:
    graph = _graph()
    group_png = render_semantic_task_graph_png(graph, _report(), view="groups")
    call_png = render_semantic_task_graph_png(graph, _report(), view="calls")

    assert group_png.startswith(_PNG_SIGNATURE)
    assert call_png.startswith(_PNG_SIGNATURE)
    assert Image.open(BytesIO(group_png)).size[0] > 0
    assert Image.open(BytesIO(call_png)).size[1] > 0

    output = write_semantic_task_graph_png(
        graph, tmp_path / "semantic_task_graph.png", _report(), view="groups"
    )
    assert output.is_file()
    assert output.read_bytes().startswith(_PNG_SIGNATURE)


def test_task_engine_cli_exposes_visualize_command() -> None:
    args = build_parser().parse_args(
        ["visualize", "--graph", "graph.json", "--output", "graph.png"]
    )
    assert args.command == "visualize"
    assert args.view == "groups"
