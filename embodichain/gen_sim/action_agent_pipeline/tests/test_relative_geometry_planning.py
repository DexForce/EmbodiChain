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
    RelativePlacementSpec,
    RelativePlacementStepSpec,
)
from embodichain.gen_sim.action_agent_pipeline.generation.relative_geometry import (
    _coordinated_payload_grid_offsets,
    _make_relative_summary,
    _with_coordinated_side_release_height_offsets,
    _with_coordinated_transport_geometry,
    _with_final_auto_arm_sides,
    _with_inside_container_slot_offsets,
    _with_self_relative_absolute_targets,
)
from embodichain.gen_sim.action_agent_pipeline.generation import (
    relative_surface_geometry,
    relative_transport_geometry,
)


def _placement(
    moved: str,
    reference: str,
    *,
    side: str,
    relation: str = "inside",
    intent: str = "place_relative",
    self_reference: bool = False,
) -> RelativePlacementStepSpec:
    return RelativePlacementStepSpec(
        intent=intent,
        moved_source_uid=moved,
        reference_source_uid=reference,
        moved_runtime_uid=moved,
        reference_runtime_uid=reference,
        relation=relation,
        active_side=side,
        release_offset=[0.0, 0.0, 0.02],
        high_offset=[0.0, 0.0, 0.22],
        reference_is_initial_pose=self_reference,
    )


def _spec(
    placements: tuple[RelativePlacementStepSpec, ...],
    *,
    intent: str = "place_relative",
    primary_index: int = 0,
    direction: str | None = None,
    terminal: str | None = None,
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
        task_prompt_summary="Move objects",
        basic_background_notes="",
        action_sketch=["pick", "move", "place"],
        release_offset=primary.release_offset,
        high_offset=primary.high_offset,
        placements=placements,
        reference_is_initial_pose=primary.reference_is_initial_pose,
        coordinated_direction=direction,
        coordinated_terminal_behavior=terminal,
    )


def test_self_relative_targets_use_generated_initial_position() -> None:
    placement = _placement(
        "can_a",
        "can_a",
        side="left",
        relation="left_of",
        self_reference=True,
    )

    resolved = _with_self_relative_absolute_targets(
        _spec((placement,)),
        {"rigid_object": [{"uid": "can_a", "init_pos": [0.4, -0.2, 0.1]}]},
    )

    assert resolved.release_position == [0.4, -0.2, 0.12]
    assert resolved.high_position == [0.4, -0.2, 0.32]


def test_inside_container_slots_follow_arm_order_with_default_geometry() -> None:
    left = _placement("can_a", "container", side="left")
    right = _placement("can_b", "container", side="right")

    resolved = _with_inside_container_slot_offsets(
        _spec((left, right)),
        {
            "rigid_object": [
                {"uid": "can_a", "init_pos": [0.0, 0.2, 0.1]},
                {"uid": "can_b", "init_pos": [0.0, -0.2, 0.1]},
                {"uid": "container", "init_pos": [0.0, 0.0, 0.1]},
            ]
        },
    )

    assert resolved.placements[0].release_offset == [0.0, 0.04, 0.12]
    assert resolved.placements[1].release_offset == [0.0, -0.04, 0.12]


def test_payload_grid_is_deterministic_without_mesh_bounds() -> None:
    placements = tuple(
        _placement(f"can_{index}", "tray", side="left" if index < 2 else "right")
        for index in range(4)
    )
    object_configs = {
        placement.moved_runtime_uid: {
            "uid": placement.moved_runtime_uid,
            "init_pos": [0.0, float(index), 0.1],
        }
        for index, placement in enumerate(placements)
    }

    offsets = _coordinated_payload_grid_offsets(
        [0, 1, 2, 3],
        placements=placements,
        object_configs=object_configs,
        container_config=None,
    )

    assert offsets == {
        0: [-0.05, -0.05, 0.12],
        1: [0.05, -0.05, 0.12],
        2: [-0.05, 0.05, 0.12],
        3: [0.05, 0.05, 0.12],
    }


def test_coordinated_transport_uses_fallback_distance_without_meshes() -> None:
    payload = _placement("can_a", "tray", side="left", relation="on")
    carrier = _placement(
        "tray",
        "table",
        side="left",
        relation="on",
        intent="coordinated_pickment",
    )
    resolved = _with_coordinated_transport_geometry(
        _spec(
            (payload, carrier),
            intent="coordinated_pickment",
            primary_index=1,
            direction="front_left",
            terminal="hold",
        ),
        {
            "background": [{"uid": "table", "init_pos": [0.0, 0.0, 0.0]}],
            "rigid_object": [{"uid": "tray", "init_pos": [0.2, -0.1, 0.1]}],
        },
    )

    assert resolved.release_offset == [0.106066, 0.106066, 0.0]
    assert resolved.release_position == [0.306066, 0.006066, 0.1]


def test_coordinated_transport_rejects_target_beyond_table_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        relative_transport_geometry,
        "_mesh_config_world_xy_bounds",
        lambda _: ((-0.5, -0.5), (0.5, 0.5)),
    )
    monkeypatch.setattr(
        relative_transport_geometry,
        "_mesh_config_world_xy_extents",
        lambda _: (0.1, 0.1),
    )

    with pytest.raises(ValueError, match="inside the table boundary"):
        relative_transport_geometry._coordinated_safe_transport_distance(
            initial_position=[0.42, 0.0, 0.1],
            direction=(1.0, 0.0),
            carrier_config={},
            table_config={},
        )


def test_side_relation_release_preserves_generated_height_difference() -> None:
    placement = _placement(
        "can_a",
        "can_b",
        side="left",
        relation="left_of",
    )

    resolved = _with_coordinated_side_release_height_offsets(
        _spec((placement,)),
        {
            "rigid_object": [
                {"uid": "can_a", "init_pos": [0.0, 0.2, 0.3]},
                {"uid": "can_b", "init_pos": [0.0, -0.2, 0.1]},
            ]
        },
    )

    assert resolved.release_offset[2] == 0.2
    assert resolved.high_offset[2] == 0.3


def test_surface_release_uses_support_top_and_rotated_object_bottom(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    placement = _placement("can_a", "tray", side="left", relation="on")
    monkeypatch.setattr(
        relative_surface_geometry,
        "_mesh_config_world_zmax",
        lambda _: 0.4,
    )
    monkeypatch.setattr(
        relative_surface_geometry,
        "_target_local_zmin_for_orientation",
        lambda *_: -0.1,
    )

    resolved = relative_surface_geometry._with_on_surface_release_offset(
        placement,
        {
            "rigid_object": [
                {"uid": "can_a", "init_pos": [0.0, 0.2, 0.3]},
                {"uid": "tray", "init_pos": [0.0, 0.0, 0.1]},
            ]
        },
    )

    assert resolved.release_offset[2] == 0.45
    assert resolved.high_offset[2] == 0.55


def test_auto_arm_and_summary_shape_remain_stable() -> None:
    placement = _placement("can_a", "tray", side="right", relation="on")
    resolved = _with_final_auto_arm_sides(
        _spec((placement,)),
        {"rigid_object": [{"uid": "can_a", "init_pos": [0.0, 0.4, 0.1]}]},
    )

    assert resolved.active_side == "left"
    assert _make_relative_summary(resolved) == {
        "mode": "object_manipulation",
        "intent": "place_relative",
        "moved_object": "can_a",
        "reference_object": "tray",
        "relation": "on",
        "active_arm": "left_arm",
        "release_offset": [0.0, 0.0, 0.02],
        "hover_height": 0.1,
        "orientation_goal": "preserve",
        "orientation_axis": "none",
        "orientation_align_to": None,
        "surface_clearance": 0.05,
    }
