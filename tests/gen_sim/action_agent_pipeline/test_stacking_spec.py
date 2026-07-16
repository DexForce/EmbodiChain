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

from embodichain.gen_sim.action_agent_pipeline.generation import stacking_spec
from embodichain.gen_sim.action_agent_pipeline.generation.config_types import (
    _RelativePlacementSpec,
    _RelativePlacementStepSpec,
    _SceneObject,
    _StackingSpec,
    _StackingStepSpec,
)
from embodichain.gen_sim.action_agent_pipeline.generation.prompt_builders import (
    make_relative_task_graph,
    make_stacking_task_graph,
)
from embodichain.gen_sim.action_agent_pipeline.generation.relative_spec import (
    _order_relative_placements_by_dependency,
)
from embodichain.gen_sim.action_agent_pipeline.generation.success_specs import (
    _make_stacking_success_spec,
)
from embodichain.gen_sim.action_agent_pipeline.generation.task_router import (
    _normalize_task_route_response,
)


def test_explicit_stacking_goal_overrides_object_manipulation_route() -> None:
    route = _normalize_task_route_response(
        {
            "route": "object_manipulation",
            "confidence": 0.8,
            "reason": "Two relative placements.",
        },
        scene_objects=[
            _scene_object("block_a"),
            _scene_object("block_b"),
        ],
        task_description="把 A 叠放到 B 上",
    )

    assert route.route == "stacking"


def test_negated_stacking_goal_does_not_override_route() -> None:
    route = _normalize_task_route_response(
        {
            "route": "object_manipulation",
            "confidence": 0.8,
            "reason": "Keep the objects separate.",
        },
        scene_objects=[
            _scene_object("block_a"),
            _scene_object("block_b"),
        ],
        task_description="不要堆叠 A 和 B，把 A 放到 B 左边",
    )

    assert route.route == "object_manipulation"


def test_existing_stack_description_does_not_override_route() -> None:
    route = _normalize_task_route_response(
        {
            "route": "object_manipulation",
            "confidence": 0.8,
            "reason": "Remove one object from the existing stack.",
        },
        scene_objects=[
            _scene_object("block_a"),
            _scene_object("block_b"),
        ],
        task_description="把 A 从已经叠放好的 B 上拿走",
    )

    assert route.route == "object_manipulation"


def test_object_anchored_stack_builds_support_chain(tmp_path) -> None:
    scene_objects = [
        _scene_object("table", role="background"),
        _scene_object("headphones"),
        _scene_object("paper_cup"),
        _scene_object("popcorn_bucket"),
    ]

    spec = stacking_spec._apply_stacking_task_response(
        response={
            "objects": ["paper_cup", "headphones"],
            "stack_mode": "on_top",
            "bottom_to_top": ["paper_cup", "headphones"],
            "order_by": "explicit",
            "anchor": {"type": "object", "object": "popcorn_bucket"},
        },
        table_source_uid="table",
        scene_objects=scene_objects,
        rigid_objects=scene_objects[1:],
        scene_dir=tmp_path,
        task_description="把耳机叠放到纸杯上，再把纸杯叠放到爆米花桶上",
    )

    assert spec.anchor == "object"
    assert spec.anchor_source_uid == "popcorn_bucket"
    assert [step.source_uid for step in spec.steps] == ["paper_cup", "headphones"]
    assert [step.support_runtime_uid for step in spec.steps] == [
        spec.anchor_runtime_uid,
        spec.steps[0].runtime_uid,
    ]


def test_object_anchor_allows_one_moved_stack_layer(tmp_path) -> None:
    scene_objects = [
        _scene_object("table", role="background"),
        _scene_object("block_a"),
        _scene_object("block_b"),
    ]

    spec = stacking_spec._apply_stacking_task_response(
        response={
            "objects": ["block_a"],
            "stack_mode": "on_top",
            "bottom_to_top": ["block_a"],
            "order_by": "explicit",
            "anchor": {"type": "object", "object": "block_b"},
        },
        table_source_uid="table",
        scene_objects=scene_objects,
        rigid_objects=scene_objects[1:],
        scene_dir=tmp_path,
        task_description="把 A 叠放到 B 上",
    )

    assert len(spec.steps) == 1
    assert spec.steps[0].support_runtime_uid == spec.anchor_runtime_uid


def test_object_anchored_stack_uses_dynamic_support_targets() -> None:
    bottom = _stacking_step("paper_cup", 0, support="popcorn_bucket")
    top = _stacking_step("headphones", 1, support="paper_cup")
    spec = _stacking_spec(
        (bottom, top),
        anchor="object",
        anchor_source_uid="popcorn_bucket_source",
        anchor_runtime_uid="popcorn_bucket",
    )

    graph = make_stacking_task_graph("anchored", spec)
    place_actions = [
        edge["left_arm_action"] or edge["right_arm_action"]
        for edge in graph["edges"]
        if (edge["left_arm_action"] or edge["right_arm_action"])["atomic_action_class"]
        == "Place"
    ]
    place_targets = [action["target_object_pose"] for action in place_actions]

    assert [target["obj_name"] for target in place_targets] == [
        "popcorn_bucket",
        "paper_cup",
    ]
    assert all(target["offset"][:2] == [0.0, 0.0] for target in place_targets)
    assert all(target["z_policy"] == "surface_release" for target in place_targets)
    assert all(
        action["cfg"]["max_approach_retract_z"] == pytest.approx(0.8)
        for action in place_actions
    )


def test_object_anchored_stack_success_uses_direct_supports() -> None:
    bottom = _stacking_step("paper_cup", 0, support="popcorn_bucket")
    top = _stacking_step("headphones", 1, support="paper_cup")
    spec = _stacking_spec(
        (bottom, top),
        anchor="object",
        anchor_source_uid="popcorn_bucket_source",
        anchor_runtime_uid="popcorn_bucket",
    )

    success = _make_stacking_success_spec(spec)
    support_pairs = {
        (term["object"], term["support"])
        for term in success["terms"]
        if term["type"] == "object_on_object"
    }

    assert support_pairs == {
        ("paper_cup", "popcorn_bucket"),
        ("headphones", "paper_cup"),
    }


def test_object_anchored_nested_stack_centers_each_inner_container() -> None:
    outer = _stacking_step("left_cup", 0, support="popcorn_bucket")
    inner = _stacking_step("right_cup", 1, support="left_cup")
    spec = _stacking_spec(
        (outer, inner),
        anchor="object",
        anchor_source_uid="popcorn_bucket_source",
        anchor_runtime_uid="popcorn_bucket",
        stack_mode="nested",
    )

    graph = make_stacking_task_graph("nested", spec)
    place_targets = [
        (edge["left_arm_action"] or edge["right_arm_action"])["target_object_pose"]
        for edge in graph["edges"]
        if (edge["left_arm_action"] or edge["right_arm_action"])["atomic_action_class"]
        == "Place"
    ]
    success = _make_stacking_success_spec(spec)

    assert [target["obj_name"] for target in place_targets] == [
        "popcorn_bucket",
        "left_cup",
    ]
    assert all(target["offset"][:2] == [0.0, 0.0] for target in place_targets)
    assert all("z_policy" not in target for target in place_targets)
    assert {
        (term["object"], term["container"])
        for term in success["terms"]
        if term["type"] == "object_in_container"
    } == {
        ("left_cup", "popcorn_bucket"),
        ("right_cup", "left_cup"),
    }


@pytest.mark.parametrize(
    ("blocked_count", "expected_direction"),
    [
        (0, [0.0, 0.0]),
        (1, [0.0, -1.0]),
        (2, [0.0, 1.0]),
        (3, [0.0, -1.0]),
    ],
)
def test_stacking_anchor_uses_fixed_table_axis_candidate_order(
    monkeypatch: pytest.MonkeyPatch,
    blocked_count: int,
    expected_direction: list[float],
) -> None:
    table = {"uid": "table"}
    obstacle = {"uid": "cup"}
    offset = stacking_spec._ANCHOR_OFFSET
    candidate_order = (
        [0.0, 0.0],
        [0.0, -offset],
        [0.0, offset],
    )
    blocked = candidate_order[:blocked_count]
    monkeypatch.setattr(
        stacking_spec,
        "_mesh_config_world_xy_center",
        lambda config: [0.0, 0.0],
    )
    monkeypatch.setattr(
        stacking_spec,
        "_mesh_config_world_xy_axes",
        lambda config: ([0.0, 1.0], [-1.0, 0.0]),
    )
    monkeypatch.setattr(
        stacking_spec,
        "_mesh_config_world_xy_bounds",
        lambda config: (
            ([-1.0, -1.0], [1.0, 1.0])
            if config["uid"] == "table"
            else ([0.0, 0.0], [0.0, 0.0])
        ),
    )
    monkeypatch.setattr(
        stacking_spec,
        "_xy_point_to_bounds_distance",
        lambda point, bounds: (0.0 if point in blocked else float("inf")),
    )

    anchor = stacking_spec._generated_stacking_anchor_xy(
        table,
        [0.0, 0.0],
        object_configs={"table": table, "cup": obstacle},
    )

    expected_anchor = [offset * component for component in expected_direction]
    assert anchor == pytest.approx(expected_anchor)


def test_stacking_anchor_treats_task_objects_as_obstacles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = {"uid": "table"}
    task_object = {"uid": "cube"}
    monkeypatch.setattr(
        stacking_spec,
        "_mesh_config_world_xy_center",
        lambda config: [0.0, 0.0],
    )
    monkeypatch.setattr(
        stacking_spec,
        "_mesh_config_world_xy_axes",
        lambda config: ([1.0, 0.0], [0.0, 1.0]),
    )
    monkeypatch.setattr(
        stacking_spec,
        "_mesh_config_world_xy_bounds",
        lambda config: (
            ([-1.0, -1.0], [1.0, 1.0])
            if config["uid"] == "table"
            else ([0.0, 0.0], [0.0, 0.0])
        ),
    )

    anchor = stacking_spec._generated_stacking_anchor_xy(
        table,
        [0.0, 0.0],
        object_configs={"table": table, "cube": task_object},
    )

    assert anchor == pytest.approx([-stacking_spec._ANCHOR_OFFSET, 0.0])


def test_stacking_anchor_uses_bounds_clearance_not_point_occupancy() -> None:
    distance = stacking_spec._xy_point_to_bounds_distance(
        [0.0, 0.0],
        ([0.19, -0.01], [0.21, 0.01]),
    )

    assert distance == pytest.approx(0.19)


def test_stacking_anchor_falls_back_to_back_without_clearance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = {"uid": "table"}
    obstacle = {"uid": "cube"}
    obstacle_half_extent = stacking_spec._ANCHOR_OFFSET
    monkeypatch.setattr(
        stacking_spec,
        "_mesh_config_world_xy_center",
        lambda config: [0.0, 0.0],
    )
    monkeypatch.setattr(
        stacking_spec,
        "_mesh_config_world_xy_axes",
        lambda config: ([1.0, 0.0], [0.0, 1.0]),
    )
    monkeypatch.setattr(
        stacking_spec,
        "_mesh_config_world_xy_bounds",
        lambda config: (
            ([-1.0, -1.0], [1.0, 1.0])
            if config["uid"] == "table"
            else (
                [-obstacle_half_extent, -obstacle_half_extent],
                [obstacle_half_extent, obstacle_half_extent],
            )
        ),
    )

    anchor = stacking_spec._generated_stacking_anchor_xy(
        table,
        [0.0, 0.0],
        object_configs={"table": table, "cube": obstacle},
    )

    assert anchor == pytest.approx([-stacking_spec._ANCHOR_OFFSET, 0.0])


def test_relative_placements_are_ordered_by_moved_object_dependency() -> None:
    lower = _relative_step("lower", "base", side="right")
    upper = _relative_step("upper", "lower", side="left")

    ordered = _order_relative_placements_by_dependency((upper, lower))

    assert [placement.moved_source_uid for placement in ordered] == [
        "lower",
        "upper",
    ]


def test_relative_placements_reject_cyclic_dependencies() -> None:
    first = _relative_step("first", "second", side="left")
    second = _relative_step("second", "first", side="right")

    with pytest.raises(ValueError, match="cyclic object dependency"):
        _order_relative_placements_by_dependency((first, second))


def test_dependent_relative_graph_is_serial_in_dependency_order() -> None:
    lower = _relative_step("lower", "base", side="right")
    upper = _relative_step("upper", "lower", side="left")
    spec = _relative_spec((lower, upper))

    graph = make_relative_task_graph("dependent", spec)
    action_classes = [
        (edge["left_arm_action"] or edge["right_arm_action"])["atomic_action_class"]
        for edge in graph["edges"]
    ]

    assert action_classes == [
        "PickUp",
        "Place",
        "MoveJoints",
        "PickUp",
        "Place",
        "MoveJoints",
    ]


def test_stacking_preserve_uses_direct_place_without_move_held_object() -> None:
    step = _StackingStepSpec(
        source_uid="cube_source",
        runtime_uid="cube",
        layer_index=0,
        active_side="left",
        target_position=[0.0, 0.0, 0.2],
        high_position=[0.0, 0.0, 0.3],
    )
    spec = _StackingSpec(
        table_source_uid="table_source",
        task_description="stack cube",
        task_prompt_summary="Stack cube.",
        basic_background_notes="",
        stack_mode="on_top",
        order_by="explicit",
        anchor="table_center",
        anchor_xy=[0.0, 0.0],
        steps=(step,),
    )

    graph = make_stacking_task_graph("stack", spec)
    actions = [edge["left_arm_action"] for edge in graph["edges"]]

    assert [action["atomic_action_class"] for action in actions] == [
        "PickUp",
        "Place",
        "MoveJoints",
    ]
    assert actions[1]["cfg"]["max_approach_retract_z"] == pytest.approx(0.8)


def test_elongated_stacking_object_does_not_request_axis_alignment() -> None:
    orientation = stacking_spec._stacking_config_orientation(
        {
            "uid": "elongated_block",
            "body_scale": [4.0, 1.0, 1.0],
        },
        stack_mode="on_top",
    )

    assert orientation == ("preserve", "none")


def test_pose_sensitive_relative_graph_uses_one_final_move() -> None:
    placement = _relative_step("bottle", "table", side="left")
    placement = _RelativePlacementStepSpec(
        **{
            **placement.__dict__,
            "orientation_goal": "upright",
            "release_position": [0.1, 0.2, 0.3],
            "high_position": [0.1, 0.2, 0.55],
        }
    )
    graph = make_relative_task_graph("upright", _relative_spec((placement,)))
    move_targets = [
        edge["left_arm_action"]["target_object_pose"]
        for edge in graph["edges"]
        if edge["left_arm_action"] is not None
        and edge["left_arm_action"]["atomic_action_class"] == "MoveHeldObject"
    ]

    assert len(move_targets) == 1
    assert move_targets[0]["orientation_goal"] == "upright"
    assert move_targets[0]["offset"] == pytest.approx([0.0, 0.0, 0.2])


def _relative_step(
    moved_uid: str,
    reference_uid: str,
    *,
    side: str,
) -> _RelativePlacementStepSpec:
    return _RelativePlacementStepSpec(
        intent="place_relative",
        moved_source_uid=moved_uid,
        reference_source_uid=reference_uid,
        moved_runtime_uid=moved_uid,
        reference_runtime_uid=reference_uid,
        relation="on",
        active_side=side,
        release_offset=[0.0, 0.0, 0.2],
        high_offset=[0.0, 0.0, 0.3],
    )


def _relative_spec(
    placements: tuple[_RelativePlacementStepSpec, ...],
) -> _RelativePlacementSpec:
    primary = placements[0]
    return _RelativePlacementSpec(
        intent="place_relative",
        table_source_uid="table",
        moved_source_uid=primary.moved_source_uid,
        reference_source_uid=primary.reference_source_uid,
        moved_runtime_uid=primary.moved_runtime_uid,
        reference_runtime_uid=primary.reference_runtime_uid,
        relation=primary.relation,
        active_side=primary.active_side,
        task_description="stack objects",
        task_prompt_summary="Stack objects.",
        basic_background_notes="",
        action_sketch=[],
        release_offset=primary.release_offset,
        high_offset=primary.high_offset,
        placements=placements,
        release_position=primary.release_position,
        high_position=primary.high_position,
        orientation_goal=primary.orientation_goal,
        orientation_axis=primary.orientation_axis,
    )


def _scene_object(uid: str, *, role: str = "rigid_object") -> _SceneObject:
    return _SceneObject(
        source_uid=uid,
        source_role=role,
        config={"uid": uid, "init_pos": [0.0, 0.0, 0.0]},
    )


def _stacking_step(
    uid: str,
    layer_index: int,
    *,
    support: str | None,
) -> _StackingStepSpec:
    return _StackingStepSpec(
        source_uid=f"{uid}_source",
        runtime_uid=uid,
        layer_index=layer_index,
        active_side="left" if layer_index % 2 == 0 else "right",
        target_position=[0.0, 0.0, 0.2 + 0.1 * layer_index],
        high_position=[0.0, 0.0, 0.3 + 0.1 * layer_index],
        support_runtime_uid=support,
    )


def _stacking_spec(
    steps: tuple[_StackingStepSpec, ...],
    *,
    anchor: str,
    anchor_source_uid: str | None = None,
    anchor_runtime_uid: str | None = None,
    stack_mode: str = "on_top",
) -> _StackingSpec:
    return _StackingSpec(
        table_source_uid="table_source",
        task_description="stack objects",
        task_prompt_summary="Stack objects.",
        basic_background_notes="",
        stack_mode=stack_mode,
        order_by="explicit",
        anchor=anchor,
        anchor_xy=[0.0, 0.0],
        steps=steps,
        anchor_source_uid=anchor_source_uid,
        anchor_runtime_uid=anchor_runtime_uid,
    )
