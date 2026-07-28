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
    RelativePlacementStepSpec,
    SceneObject,
)
from embodichain.gen_sim.action_agent_pipeline.generation.relative_spec import (
    _apply_relative_task_response,
    _build_object_manipulation_spec_with_llm,
    _canonicalize_flat_coordinated_transport_entries,
    _coordinated_payload_entries,
    _normalize_relative_relation,
    _order_relative_placements_by_dependency,
    _relative_scene_runtime_uid_mapping,
)


def _scene_objects() -> list[SceneObject]:
    return [
        SceneObject(
            source_uid="table_src",
            source_role="background",
            config={"uid": "table", "init_pos": [0.0, 0.0, 0.0]},
        ),
        SceneObject(
            source_uid="can_a",
            source_role="rigid_object",
            config={"description": "can", "init_pos": [0.0, 0.3, 0.1]},
        ),
        SceneObject(
            source_uid="can_b",
            source_role="rigid_object",
            config={"description": "can", "init_pos": [0.0, -0.3, 0.1]},
        ),
        SceneObject(
            source_uid="tray",
            source_role="rigid_object",
            config={"description": "flat tray", "init_pos": [0.2, 0.0, 0.1]},
        ),
    ]


def _apply_response(response: dict, task_description: str = "Place the can") -> object:
    scene_objects = _scene_objects()
    return _apply_relative_task_response(
        response=response,
        table_source_uid="table_src",
        scene_objects=scene_objects,
        rigid_objects=scene_objects[1:],
        task_description=task_description,
        release_offset_fn=lambda relation: (
            {
                "on": [0.0, 0.0, 0.02],
                "inside": [0.0, 0.0, 0.03],
                "left_of": [0.0, 0.15, 0.02],
            }.get(relation, [0.15, 0.0, 0.02])
        ),
        staging_z_delta=0.2,
        pose_sensitive_staging_z_delta=0.1,
    )


def test_object_manipulation_builder_populates_scene_summary() -> None:
    captured: dict = {}

    def fake_task_llm(**kwargs):
        captured.update(kwargs)
        return {
            "intent": "place_relative",
            "moved_object": "can_a",
            "reference_object": "tray",
            "goal_relation": "on",
        }

    spec = _build_object_manipulation_spec_with_llm(
        scene_objects=_scene_objects(),
        project_name="task2_2",
        task_description="Use both arms to stand the objects upright.",
        model=None,
        release_offset_fn=lambda relation: [0.0, 0.0, 0.03],
        staging_z_delta=0.2,
        pose_sensitive_staging_z_delta=0.1,
        task_llm_caller=fake_task_llm,
    )

    assert spec.moved_source_uid == "can_a"
    assert captured["project_name"] == "task2_2"
    assert captured["scene_summary"][1]["object_type"] == "can_a"
    assert captured["scene_summary"][3]["is_container_like"] is True


def test_single_relative_response_preserves_normalized_plan() -> None:
    spec = _apply_response(
        {
            "intent": "place",
            "moved_object": "can_a",
            "reference_object": "tray",
            "goal_relation": "on_top_of",
            "arm": "auto",
            "task_prompt_summary": "Place can A on the tray.",
        }
    )

    assert spec.intent == "place_relative"
    assert spec.moved_source_uid == "can_a"
    assert spec.reference_source_uid == "tray"
    assert spec.moved_runtime_uid == "can_a"
    assert spec.reference_runtime_uid == "target_tray"
    assert spec.relation == "inside"
    assert spec.active_side == "left"
    assert spec.release_offset == [0.0, 0.0, 0.03]
    assert spec.high_offset == [0.0, 0.0, 0.23]
    assert spec.orientation_goal == "preserve"
    assert len(spec.placements) == 1


def test_hold_hover_preserves_self_reference_and_height() -> None:
    spec = _apply_response(
        {
            "intent": "hold",
            "moved_object": "can_b",
            "hover_height": 0.17,
        },
        task_description="Pick up can B and hold it.",
    )

    placement = spec.placements[0]
    assert spec.intent == "hold_hover"
    assert placement.reference_is_initial_pose is True
    assert placement.reference_source_uid == "can_b"
    assert placement.release_offset == [0.0, 0.0, 0.17]
    assert placement.high_offset == [0.0, 0.0, 0.17]


def test_upright_in_place_normalizes_orientation_and_table_reference() -> None:
    spec = _apply_response(
        {
            "intent": "place_relative",
            "moved_object": "can_a",
            "reference_object": "self",
            "goal_relation": "on",
            "orientation_goal": "upright",
        },
        task_description="Stand can A upright in place.",
    )

    placement = spec.placements[0]
    assert placement.upright_in_place is True
    assert placement.reference_source_uid == "table_src"
    assert placement.reference_runtime_uid == "table"
    assert placement.reference_is_initial_pose is False
    assert placement.orientation_goal == "upright"
    assert placement.orientation_axis == "none"


def test_flat_coordinated_payloads_fold_into_one_entry() -> None:
    entries = _canonicalize_flat_coordinated_transport_entries(
        [
            {
                "intent": "place_relative",
                "moved_object": "can_a",
                "reference_object": "tray",
                "goal_relation": "on",
                "arm": "left",
            },
            {
                "intent": "place_relative",
                "moved_object": "can_b",
                "reference_object": "tray",
                "goal_relation": "on",
                "arm": "right",
            },
            {
                "intent": "coordinated_pickment",
                "moved_object": "tray",
                "goal_relation": "front_of",
            },
        ],
        rigid_objects=_scene_objects()[1:],
    )

    assert len(entries) == 1
    assert entries[0]["direction"] == "front"
    assert entries[0]["payloads"] == [
        {"object": "can_a", "arm": "left", "slot": "left"},
        {"object": "can_b", "arm": "right", "slot": "right"},
    ]


def test_nested_coordinated_payloads_preserve_terminal_contract() -> None:
    spec = _apply_response(
        {
            "intent": "coordinated_pickment",
            "moved_object": "tray",
            "payloads": [
                {"object": "can_a", "slot": "left"},
                {"object": "can_b", "slot": "right"},
            ],
            "direction": "front-left",
            "terminal_behavior": "hold",
        },
        task_description="Use both arms to carry the loaded tray.",
    )

    assert spec.intent == "coordinated_pickment"
    assert spec.coordinated_direction == "front_left"
    assert spec.coordinated_terminal_behavior == "hold"
    assert [placement.moved_source_uid for placement in spec.placements] == [
        "can_a",
        "can_b",
        "tray",
    ]
    assert [placement.active_side for placement in spec.placements[:2]] == [
        "left",
        "right",
    ]


def test_coordinated_payload_count_limit_is_rejected() -> None:
    scene_objects = _scene_objects()
    with pytest.raises(ValueError, match="at most 4 objects"):
        _coordinated_payload_entries(
            {
                "moved_object": "tray",
                "payloads": [{"object": "can_a"}] * 5,
            },
            by_uid={obj.source_uid: obj for obj in scene_objects},
            rigid_objects=scene_objects[1:],
        )


def test_dependency_order_and_cycle_detection_remain_deterministic() -> None:
    first = RelativePlacementStepSpec(
        intent="place_relative",
        moved_source_uid="can_a",
        reference_source_uid="can_b",
        moved_runtime_uid="can_a",
        reference_runtime_uid="can_b",
        relation="on",
        active_side="left",
        release_offset=[0.0, 0.0, 0.02],
        high_offset=[0.0, 0.0, 0.2],
    )
    second = RelativePlacementStepSpec(
        intent="place_relative",
        moved_source_uid="can_b",
        reference_source_uid="tray",
        moved_runtime_uid="can_b",
        reference_runtime_uid="container",
        relation="on",
        active_side="right",
        release_offset=[0.0, 0.0, 0.02],
        high_offset=[0.0, 0.0, 0.2],
    )

    ordered = _order_relative_placements_by_dependency((first, second))
    assert [placement.moved_source_uid for placement in ordered] == ["can_b", "can_a"]

    cyclic_second = RelativePlacementStepSpec(
        **{
            **second.__dict__,
            "reference_source_uid": "can_a",
            "reference_runtime_uid": "can_a",
        }
    )
    with pytest.raises(ValueError, match="cyclic object dependency"):
        _order_relative_placements_by_dependency((first, cyclic_second))


def test_relative_identity_and_relation_aliases_remain_stable() -> None:
    runtime_uids = _relative_scene_runtime_uid_mapping(
        _scene_objects(),
        table_source_uid="table_src",
    )

    assert runtime_uids == {
        "table_src": "table",
        "can_a": "can_a",
        "can_b": "can_b",
        "tray": "target_tray",
    }
    assert _normalize_relative_relation("front-left") == "front_left_of"
    assert _normalize_relative_relation("放入") == "inside"
