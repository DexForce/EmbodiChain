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

from collections.abc import Mapping
from typing import Any

import pytest

from embodichain.gen_sim.action_agent_pipeline.generation.config_types import (
    SceneObject,
)
from embodichain.gen_sim.action_agent_pipeline.generation import task_interpretation
from embodichain.gen_sim.action_agent_pipeline.generation.task_interpretation import (
    _call_task_interpretation_llm,
    _interpret_task_with_llm,
)


def _scene_objects() -> list[SceneObject]:
    return [
        SceneObject(
            source_uid="table",
            source_role="background",
            config={"description": "table"},
        ),
        SceneObject(
            source_uid="can_1",
            source_role="rigid_object",
            config={"description": "red can", "init_pos": [0.0, 0.2, 0.1]},
        ),
        SceneObject(
            source_uid="can_2",
            source_role="rigid_object",
            config={"description": "blue can", "init_pos": [0.0, -0.2, 0.1]},
        ),
        SceneObject(
            source_uid="bucket",
            source_role="rigid_object",
            config={"description": "popcorn bucket", "init_pos": [0.2, 0.0, 0.1]},
        ),
    ]


def _response(route: str) -> dict[str, Any]:
    specs = {
        "arrangement_line": {
            "objects": ["can_1", "can_2"],
            "category_order": ["can"],
            "object_categories": {"can_1": "can", "can_2": "can"},
            "order_by": "explicit",
            "order_direction": "given",
            "ordered_attributes": [],
            "object_attributes": {},
            "anchor": "table_center",
            "line_axis": "world_y",
        },
        "stacking": {
            "objects": ["can_1", "can_2"],
            "stack_mode": "on_top",
            "bottom_to_top": ["can_1", "can_2"],
            "order_by": "explicit",
            "object_attributes": {},
            "anchor": {"type": "object", "object": "bucket"},
        },
        "object_manipulation": {
            "manipulations": [
                {
                    "intent": "place_relative",
                    "moved_object": "can_1",
                    "arm": "auto",
                    "reference_object": "bucket",
                    "goal_relation": "inside",
                }
            ]
        },
    }
    return {
        "schema_version": "task_interpretation_v1",
        "route": route,
        "confidence": 0.95,
        "reason": f"Selected {route}.",
        "candidate_objects": ["can_1", "can_2"],
        "warnings": [],
        "spec": specs[route],
    }


@pytest.mark.parametrize(
    ("route", "task_description"),
    [
        ("arrangement_line", "将两个罐头摆成一排"),
        ("stacking", "把两个罐头叠放起来"),
        ("object_manipulation", "把红色罐头放进爆米花桶"),
    ],
)
def test_task_interpretation_calls_model_once_for_each_route(
    route: str,
    task_description: str,
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_llm(**kwargs: Any) -> Mapping[str, Any]:
        calls.append(kwargs)
        return _response(route)

    result = _interpret_task_with_llm(
        scene_objects=_scene_objects(),
        project_name="task",
        task_description=task_description,
        model="mock-model",
        task_llm_caller=fake_llm,
    )

    assert result.route == route
    assert result.spec == _response(route)["spec"]
    assert len(calls) == 1
    assert calls[0]["scene_summary"][1]["color_hint"] == "red"


def test_task_interpretation_does_not_retry_after_llm_json_failure() -> None:
    call_count = 0

    def failing_llm(**kwargs: Any) -> Mapping[str, Any]:
        nonlocal call_count
        call_count += 1
        raise ValueError("malformed JSON")

    with pytest.raises(ValueError, match="malformed JSON"):
        _interpret_task_with_llm(
            scene_objects=_scene_objects(),
            project_name="task",
            task_description="将两个罐头摆成一排",
            model=None,
            task_llm_caller=failing_llm,
        )

    assert call_count == 1


def test_task_interpretation_rejects_route_spec_mismatch_without_retry() -> None:
    response = _response("arrangement_line")
    response["spec"] = _response("stacking")["spec"]
    call_count = 0

    def fake_llm(**kwargs: Any) -> Mapping[str, Any]:
        nonlocal call_count
        call_count += 1
        return response

    with pytest.raises(ValueError, match="route/spec mismatch"):
        _interpret_task_with_llm(
            scene_objects=_scene_objects(),
            project_name="task",
            task_description="将两个罐头摆成一排",
            model=None,
            task_llm_caller=fake_llm,
        )

    assert call_count == 1


def test_task_interpretation_rejects_unknown_candidate_without_retry() -> None:
    response = _response("arrangement_line")
    response["candidate_objects"] = ["missing_object"]
    call_count = 0

    def fake_llm(**kwargs: Any) -> Mapping[str, Any]:
        nonlocal call_count
        call_count += 1
        return response

    with pytest.raises(ValueError, match="unknown candidate"):
        _interpret_task_with_llm(
            scene_objects=_scene_objects(),
            project_name="task",
            task_description="将两个罐头摆成一排",
            model=None,
            task_llm_caller=fake_llm,
        )

    assert call_count == 1


def test_task_interpretation_rejects_keyword_route_conflict() -> None:
    response = _response("object_manipulation")

    with pytest.raises(ValueError, match="route conflicts"):
        _interpret_task_with_llm(
            scene_objects=_scene_objects(),
            project_name="task",
            task_description="把两个罐头叠放起来",
            model=None,
            task_llm_caller=lambda **kwargs: response,
        )


def test_task_interpretation_uses_single_usage_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_request(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return _response("arrangement_line")

    monkeypatch.setattr(task_interpretation, "request_json_spec", fake_request)

    result = _call_task_interpretation_llm(
        project_name="task",
        task_description="将两个罐头摆成一排",
        scene_summary=[],
        model=None,
    )

    assert result["route"] == "arrangement_line"
    assert captured["template_name"] == "task_interpretation.txt"
    assert captured["usage_stage"] == "config_generation.task_interpretation"
