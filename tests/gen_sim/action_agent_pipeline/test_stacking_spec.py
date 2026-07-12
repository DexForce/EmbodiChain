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


def test_stacking_anchor_uses_fixed_table_axis_candidate_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = {"uid": "table"}
    obstacle = {"uid": "cup"}
    bounds_by_uid = {
        "table": ([-1.0, -1.0], [1.0, 1.0]),
        "cup": ([-0.01, -0.01], [0.01, 0.01]),
    }
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
        lambda config: bounds_by_uid[config["uid"]],
    )

    anchor = stacking_spec._generated_stacking_anchor_xy(
        table,
        [0.0, 0.0],
        object_configs={"table": table, "cup": obstacle},
        ignored_runtime_uids=set(),
    )

    assert anchor == pytest.approx([0.0, 0.15])


def test_stacking_anchor_ignores_all_task_objects(
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
        lambda config: ([-1.0, -1.0], [1.0, 1.0]),
    )

    anchor = stacking_spec._generated_stacking_anchor_xy(
        table,
        [0.0, 0.0],
        object_configs={"table": table, "cube": task_object},
        ignored_runtime_uids={"cube"},
    )

    assert anchor == pytest.approx([0.0, 0.0])


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


def test_pose_sensitive_relative_graph_rotates_at_high_staging() -> None:
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

    assert [target["orientation_goal"] for target in move_targets] == [
        "preserve",
        "upright",
        "upright",
    ]
    assert move_targets[0]["offset"] == pytest.approx([0.0, 0.0, 0.3])
    assert move_targets[-1]["offset"] == pytest.approx([0.0, 0.0, 0.2])


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
