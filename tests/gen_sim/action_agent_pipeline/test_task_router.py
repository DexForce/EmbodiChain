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

import pytest

from embodichain.gen_sim.action_agent_pipeline.generation.config_types import (
    _SceneObject,
)
from embodichain.gen_sim.action_agent_pipeline.generation.task_router import (
    _TASK_ROUTE_ARRANGEMENT_LINE,
    _TASK_ROUTE_OBJECT_MANIPULATION,
    _route_task_with_llm,
)


def test_task_router_uses_strict_route_response_and_scene_summary() -> None:
    scene_objects = _router_scene_objects()
    calls = []

    def fake_router_llm(**kwargs):
        calls.append(kwargs)
        assert kwargs["project_name"] == "router_project"
        assert kwargs["task_description"] == "将三个方块摆成一排"
        rigid_items = [
            item for item in kwargs["scene_summary"] if item["role"] == "rigid_object"
        ]
        assert [item["source_uid"] for item in rigid_items] == [
            "wood_block_0",
            "wood_block_1",
            "cardboard_box_0",
        ]
        return {
            "route": "arrangement_line",
            "confidence": 0.94,
            "reason": "The task asks for a global row arrangement.",
            "candidate_objects": [
                "wood_block_0",
                "wood_block_1",
                "cardboard_box_0",
            ],
            "warnings": ["cardboard_box_0 is treated as the third block-like object."],
        }

    route = _route_task_with_llm(
        scene_objects=scene_objects,
        project_name="router_project",
        task_description="将三个方块摆成一排",
        model="router-model",
        task_router_llm_caller=fake_router_llm,
    )

    assert len(calls) == 1
    assert route.route == _TASK_ROUTE_ARRANGEMENT_LINE
    assert route.confidence == pytest.approx(0.94)
    assert route.candidate_objects == (
        "wood_block_0",
        "wood_block_1",
        "cardboard_box_0",
    )
    assert route.to_summary()["route"] == _TASK_ROUTE_ARRANGEMENT_LINE


def test_task_router_normalizes_route_aliases() -> None:
    route = _route_task_with_llm(
        scene_objects=_router_scene_objects(),
        project_name="router_project",
        task_description="move the block next to the box",
        model=None,
        task_router_llm_caller=lambda **_: {
            "route": "relative_placement",
            "confidence": 0.8,
            "reason": "Relative placement.",
            "candidate_objects": ["wood_block_0"],
        },
    )

    assert route.route == _TASK_ROUTE_OBJECT_MANIPULATION


def test_task_router_rejects_unknown_candidate_object() -> None:
    with pytest.raises(ValueError, match="unknown candidate object"):
        _route_task_with_llm(
            scene_objects=_router_scene_objects(),
            project_name="router_project",
            task_description="arrange the blocks",
            model=None,
            task_router_llm_caller=lambda **_: {
                "route": "arrangement_line",
                "confidence": 0.9,
                "reason": "Arrange objects.",
                "candidate_objects": ["missing_block"],
            },
        )


def test_task_router_rejects_unfeasible_arrangement_route() -> None:
    with pytest.raises(ValueError, match="requires at least two movable"):
        _route_task_with_llm(
            scene_objects=[
                _SceneObject(
                    source_uid="table",
                    source_role="background",
                    config={"description": "table"},
                )
            ],
            project_name="router_project",
            task_description="arrange the blocks",
            model=None,
            task_router_llm_caller=lambda **_: {
                "route": "arrangement_line",
                "confidence": 0.9,
                "reason": "Arrange objects.",
            },
        )


def _router_scene_objects() -> list[_SceneObject]:
    return [
        _SceneObject(
            source_uid="table",
            source_role="background",
            config={
                "description": "A rectangular table.",
                "shape": {"fpath": "mesh_assets/table/table.glb"},
            },
        ),
        _SceneObject(
            source_uid="wood_block_0",
            source_role="rigid_object",
            config={
                "description": "A cubic wooden block.",
                "shape": {"fpath": "mesh_assets/wood_block_0/block.glb"},
            },
        ),
        _SceneObject(
            source_uid="wood_block_1",
            source_role="rigid_object",
            config={
                "description": "A cubic wooden block.",
                "shape": {"fpath": "mesh_assets/wood_block_1/block.glb"},
            },
        ),
        _SceneObject(
            source_uid="cardboard_box_0",
            source_role="rigid_object",
            config={
                "description": "A rectangular cardboard box.",
                "shape": {"fpath": "mesh_assets/cardboard_box_0/box.glb"},
            },
        ),
    ]
