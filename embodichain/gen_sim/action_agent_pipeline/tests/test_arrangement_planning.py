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

from pathlib import Path

from embodichain.gen_sim.action_agent_pipeline.generation import arrangement_layout
from embodichain.gen_sim.action_agent_pipeline.generation.arrangement_execution import (
    _arrangement_initial_occupancy_schedule,
    _arrangement_slot_allowed_sides,
)
from embodichain.gen_sim.action_agent_pipeline.generation.arrangement_layout import (
    _ArrangementFootprint,
)
from embodichain.gen_sim.action_agent_pipeline.generation.arrangement_spec import (
    _apply_arrangement_task_response,
    _arrangement_line_slot_positions,
)
from embodichain.gen_sim.action_agent_pipeline.generation.config_types import (
    ArrangementLineSpec,
    ArrangementLineStepSpec,
    SceneObject,
)


def test_arrangement_response_normalization_preserves_resolved_spec() -> None:
    scene_objects = [
        SceneObject(
            source_uid="table_src",
            source_role="background",
            config={"uid": "table", "init_pos": [1.0, 2.0, 0.0]},
        ),
        SceneObject(
            source_uid="can_a",
            source_role="rigid_object",
            config={"uid": "a", "description": "red can", "init_pos": [0.0, 0.3, 0.1]},
        ),
        SceneObject(
            source_uid="can_b",
            source_role="rigid_object",
            config={
                "uid": "b",
                "description": "blue can",
                "init_pos": [0.0, -0.3, 0.2],
            },
        ),
    ]

    spec = _apply_arrangement_task_response(
        response={
            "objects": ["can_a", "can_b"],
            "object_categories": {"can_a": "can", "can_b": "can"},
            "category_order": ["can"],
            "order_by": "explicit",
            "order_direction": "given",
            "anchor": "table_center",
            "task_prompt_summary": "Arrange cans",
        },
        table_source_uid="table_src",
        scene_objects=scene_objects,
        rigid_objects=scene_objects[1:],
        scene_dir=Path("."),
        task_description="Arrange cans in one line",
    )

    assert spec == ArrangementLineSpec(
        table_source_uid="table_src",
        task_description="Arrange cans in one line",
        task_prompt_summary="Arrange cans",
        basic_background_notes="",
        order_by="explicit",
        order_direction="given",
        axis="world_y",
        anchor="table_center",
        steps=(
            ArrangementLineStepSpec(
                source_uid="can_a",
                runtime_uid="can_a",
                slot_index=0,
                active_side="left",
                target_xy=[1.0, 1.96],
                release_position=[1.0, 1.96, 0.12],
                high_position=[1.0, 1.96, 0.325],
                category="can",
            ),
            ArrangementLineStepSpec(
                source_uid="can_b",
                runtime_uid="can_b",
                slot_index=1,
                active_side="right",
                target_xy=[1.0, 2.04],
                release_position=[1.0, 2.04, 0.22],
                high_position=[1.0, 2.04, 0.425],
                category="can",
            ),
        ),
        line_origin_xy=[1.0, 2.0],
        spacing=0.08,
        layout_clearance=0.025,
        category_order=("can",),
    )


def test_arrangement_line_slots_resolve_table_long_axis_deterministically() -> None:
    slots = _arrangement_line_slot_positions(
        anchor_xy=[0.5, -0.5],
        count=3,
        spacing=0.2,
        line_axis="table_long_axis",
        table_bounds=([-1.0, -2.0], [1.0, 2.0]),
    )

    assert slots == [[0.5, -0.7], [0.5, -0.5], [0.5, -0.3]]


def test_arrangement_occupancy_schedule_moves_blocker_first() -> None:
    steps = [
        ArrangementLineStepSpec(
            source_uid="a",
            runtime_uid="a",
            slot_index=0,
            active_side="left",
            target_xy=[0.4, 0.0],
            release_position=[0.4, 0.0, 0.1],
            high_position=[0.4, 0.0, 0.3],
        ),
        ArrangementLineStepSpec(
            source_uid="b",
            runtime_uid="b",
            slot_index=1,
            active_side="left",
            target_xy=[0.8, 0.0],
            release_position=[0.8, 0.0, 0.1],
            high_position=[0.8, 0.0, 0.3],
        ),
    ]
    rigid_configs = {
        "a": {"init_pos": [0.0, 0.0, 0.1]},
        "b": {"init_pos": [0.4, 0.0, 0.1]},
    }
    footprint_by_uid = {
        "a": _ArrangementFootprint(
            xy_bounds=([-0.05, -0.05], [0.05, 0.05]),
            half_extent=0.05,
        ),
        "b": _ArrangementFootprint(
            xy_bounds=([0.35, -0.05], [0.45, 0.05]),
            half_extent=0.05,
        ),
    }

    scheduled = _arrangement_initial_occupancy_schedule(
        steps,
        rigid_configs=rigid_configs,
        footprint_by_uid=footprint_by_uid,
        clearance=0.0,
    )

    assert scheduled is not None
    execution_steps, blockers, conflict_count = scheduled
    assert [step.runtime_uid for step in execution_steps] == ["b", "a"]
    assert blockers == {"a": ("b",), "b": ()}
    assert conflict_count == 1


def test_arrangement_outer_slots_keep_deterministic_arm_constraints() -> None:
    allowed_sides = [_arrangement_slot_allowed_sides(index, 5) for index in range(5)]

    assert allowed_sides == [
        frozenset({"right"}),
        frozenset({"left", "right"}),
        frozenset({"left", "right"}),
        frozenset({"left", "right"}),
        frozenset({"left"}),
    ]


def test_arrangement_spec_preserves_line_slot_helper_import() -> None:
    assert (
        _arrangement_line_slot_positions
        is arrangement_layout._arrangement_line_slot_positions
    )
