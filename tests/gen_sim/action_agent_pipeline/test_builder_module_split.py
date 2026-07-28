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

from embodichain.gen_sim.action_agent_pipeline.contracts import (
    LEFT_ARM_ACTION_KEY,
    RIGHT_ARM_ACTION_KEY,
)
from embodichain.gen_sim.action_agent_pipeline.generation import prompt_builders
from embodichain.gen_sim.action_agent_pipeline.generation.action_spec_builders import (
    _compact_json,
)
from embodichain.gen_sim.action_agent_pipeline.generation.arrangement_diagnostics import (
    make_arrangement_task_prompt,
)
from embodichain.gen_sim.action_agent_pipeline.generation.config_types import (
    ArrangementLineSpec,
    ArrangementLineStepSpec,
    RelativePlacementSpec,
    RelativePlacementStepSpec,
    StackingSpec,
    StackingStepSpec,
)
from embodichain.gen_sim.action_agent_pipeline.generation.task_graph_builders import (
    make_arrangement_task_graph,
    make_relative_task_graph,
    make_stacking_task_graph,
)


def _action_classes(graph: dict) -> list[str]:
    classes = []
    for edge in graph["edges"]:
        for slot in (LEFT_ARM_ACTION_KEY, RIGHT_ARM_ACTION_KEY):
            action = edge[slot]
            if action is not None:
                classes.append(action["atomic_action_class"])
    return classes


@pytest.mark.parametrize(
    ("orientation_goal", "orientation_axis", "expected_count"),
    [
        ("preserve", "none", 6),
        ("axis_align", "x", 7),
    ],
)
def test_arrangement_graph_preserves_edge_sequence(
    orientation_goal: str,
    orientation_axis: str,
    expected_count: int,
) -> None:
    step = ArrangementLineStepSpec(
        source_uid="can",
        runtime_uid="can_0",
        slot_index=0,
        active_side="left",
        target_xy=[0.1, 0.2],
        release_position=[0.1, 0.2, 0.3],
        high_position=[0.1, 0.2, 0.5],
        orientation_goal=orientation_goal,
        orientation_axis=orientation_axis,
    )
    spec = ArrangementLineSpec(
        table_source_uid="table",
        task_description="Arrange the cans",
        task_prompt_summary="Arrange cans in a line",
        basic_background_notes="",
        order_by="size",
        order_direction="ascending",
        axis="world_x",
        anchor="center",
        steps=(step,),
        line_origin_xy=[0.1, 0.2],
        spacing=0.1,
        layout_clearance=0.02,
    )

    graph = make_arrangement_task_graph("arrange", spec)

    assert len(graph["edges"]) == expected_count
    expected = ["PickUp", "MoveHeldObject"]
    if orientation_goal != "preserve":
        expected.append("MoveHeldObject")
    expected.extend(["MoveHeldObject", "Place", "MoveEndEffector", "MoveJoints"])
    assert _action_classes(graph) == expected

    # The diagnostic record must quote the same first action emitted by the graph.
    prompt = make_arrangement_task_prompt("arrange", "gym_export", spec)
    first_action = graph["edges"][0][LEFT_ARM_ACTION_KEY]
    assert _compact_json(first_action) in prompt


@pytest.mark.parametrize(
    ("anchor", "orientation_goal", "orientation_axis", "expected_count"),
    [
        ("center", "preserve", "none", 3),
        ("center", "upright", "none", 7),
        ("object", "preserve", "none", 3),
    ],
)
def test_stacking_graph_preserves_anchor_and_orientation_routes(
    anchor: str,
    orientation_goal: str,
    orientation_axis: str,
    expected_count: int,
) -> None:
    step = StackingStepSpec(
        source_uid="cup",
        runtime_uid="cup_0",
        layer_index=0,
        active_side="right",
        target_position=[0.2, 0.0, 0.3],
        high_position=[0.2, 0.0, 0.5],
        support_runtime_uid="bowl_0" if anchor == "object" else None,
        orientation_goal=orientation_goal,
        orientation_axis=orientation_axis,
    )
    spec = StackingSpec(
        table_source_uid="table",
        task_description="Stack the cups",
        task_prompt_summary="Stack cups",
        basic_background_notes="",
        stack_mode="on_top",
        order_by="size",
        anchor=anchor,
        anchor_xy=[0.2, 0.0],
        steps=(step,),
        anchor_source_uid="bowl" if anchor == "object" else None,
        anchor_runtime_uid="bowl_0" if anchor == "object" else None,
    )

    graph = make_stacking_task_graph("stack", spec)

    assert len(graph["edges"]) == expected_count
    if orientation_goal == "preserve":
        assert _action_classes(graph) == ["PickUp", "Place", "MoveJoints"]
    else:
        assert _action_classes(graph) == [
            "PickUp",
            "MoveHeldObject",
            "MoveHeldObject",
            "MoveHeldObject",
            "Place",
            "MoveEndEffector",
            "MoveJoints",
        ]
    if anchor == "object":
        target = graph["edges"][1][RIGHT_ARM_ACTION_KEY]["target_object_pose"]
        assert target["support"] == "bowl_0"
        assert target["z_policy"] == "surface_release"


def _relative_step(
    *,
    moved: str,
    reference: str,
    side: str,
    intent: str = "place_relative",
    orientation_goal: str = "preserve",
    reference_source: str | None = None,
) -> RelativePlacementStepSpec:
    return RelativePlacementStepSpec(
        intent=intent,
        moved_source_uid=moved,
        reference_source_uid=reference_source or reference,
        moved_runtime_uid=f"{moved}_0",
        reference_runtime_uid=f"{reference}_0",
        relation="on",
        active_side=side,
        release_offset=[0.0, 0.0, 0.02],
        high_offset=[0.0, 0.0, 0.2],
        orientation_goal=orientation_goal,
        orientation_axis="none",
    )


def _relative_spec(
    placements: tuple[RelativePlacementStepSpec, ...],
    *,
    intent: str = "place_relative",
    primary_index: int = 0,
    terminal_behavior: str | None = None,
) -> RelativePlacementSpec:
    primary = placements[primary_index]
    return RelativePlacementSpec(
        intent=intent,
        table_source_uid="table",
        moved_source_uid=primary.moved_source_uid,
        reference_source_uid=primary.reference_source_uid,
        moved_runtime_uid=primary.moved_runtime_uid,
        reference_runtime_uid=primary.reference_runtime_uid,
        relation=primary.relation,
        active_side=primary.active_side,
        task_description="Move objects",
        task_prompt_summary="Move objects relative to references",
        basic_background_notes="",
        action_sketch=["pick", "move", "release"],
        release_offset=primary.release_offset,
        high_offset=primary.high_offset,
        placements=placements,
        orientation_goal=primary.orientation_goal,
        orientation_axis=primary.orientation_axis,
        coordinated_direction="left",
        coordinated_terminal_behavior=terminal_behavior,
    )


@pytest.mark.parametrize(
    ("orientation_goal", "expected_classes"),
    [
        ("preserve", ["PickUp", "Place", "MoveJoints"]),
        (
            "upright",
            ["PickUp", "MoveHeldObject", "Place", "MoveEndEffector", "MoveJoints"],
        ),
    ],
)
def test_single_relative_graph_routes(
    orientation_goal: str,
    expected_classes: list[str],
) -> None:
    step = _relative_step(
        moved="can",
        reference="plate",
        side="left",
        orientation_goal=orientation_goal,
    )
    graph = make_relative_task_graph("relative", _relative_spec((step,)))

    assert _action_classes(graph) == expected_classes


def test_dual_relative_parallel_and_serial_dependencies() -> None:
    first = _relative_step(moved="apple", reference="plate", side="left")
    parallel_second = _relative_step(
        moved="orange",
        reference="bowl",
        side="right",
    )
    serial_second = _relative_step(
        moved="orange",
        reference="apple",
        reference_source="apple",
        side="right",
    )

    parallel = make_relative_task_graph(
        "parallel",
        _relative_spec((first, parallel_second)),
    )
    serial = make_relative_task_graph(
        "serial",
        _relative_spec((first, serial_second)),
    )

    assert len(parallel["edges"]) == 5
    assert parallel["edges"][0][LEFT_ARM_ACTION_KEY] is not None
    assert parallel["edges"][0][RIGHT_ARM_ACTION_KEY] is not None
    assert len(serial["edges"]) == 6
    assert all(
        (edge[LEFT_ARM_ACTION_KEY] is None) != (edge[RIGHT_ARM_ACTION_KEY] is None)
        for edge in serial["edges"]
    )


def test_hold_hover_and_coordinated_routes() -> None:
    left = _relative_step(moved="apple", reference="table", side="left")
    right = _relative_step(moved="orange", reference="table", side="right")
    hold_hover = make_relative_task_graph(
        "hover",
        _relative_spec((left, right), intent="hold_hover"),
    )

    coordinated = _relative_step(
        moved="tray",
        reference="table",
        side="left",
        intent="coordinated_pickment",
    )
    simple = make_relative_task_graph(
        "coordinated",
        _relative_spec((coordinated,), intent="coordinated_pickment"),
    )

    assert len(hold_hover["edges"]) == 3
    assert len(simple["edges"]) == 3
    assert (
        simple["edges"][0][LEFT_ARM_ACTION_KEY]["atomic_action_class"]
        == "CoordinatedPickment"
    )


@pytest.mark.parametrize(
    ("terminal_behavior", "expected_count"), [("hold", 4), ("place", 8)]
)
def test_coordinated_transport_routes(
    terminal_behavior: str,
    expected_count: int,
) -> None:
    payload = _relative_step(moved="can", reference="tray", side="left")
    carrier = _relative_step(
        moved="tray",
        reference="table",
        side="left",
        intent="coordinated_pickment",
    )
    spec = _relative_spec(
        (payload, carrier),
        intent="coordinated_pickment",
        primary_index=1,
        terminal_behavior=terminal_behavior,
    )

    graph = make_relative_task_graph("transport", spec)

    assert len(graph["edges"]) == expected_count
    coordinated_actions = [
        action
        for edge in graph["edges"]
        for action in (edge[LEFT_ARM_ACTION_KEY], edge[RIGHT_ARM_ACTION_KEY])
        if action is not None and action["atomic_action_class"] == "CoordinatedPickment"
    ]
    assert coordinated_actions[0]["target_object"]["payloads"] == ["can_0"]


def test_prompt_builder_facade_reexports_graph_builders() -> None:
    assert prompt_builders.make_arrangement_task_graph is make_arrangement_task_graph
    assert prompt_builders.make_stacking_task_graph is make_stacking_task_graph
    assert prompt_builders.make_relative_task_graph is make_relative_task_graph
