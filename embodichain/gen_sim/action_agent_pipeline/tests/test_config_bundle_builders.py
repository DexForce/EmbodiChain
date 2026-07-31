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
from types import SimpleNamespace

import pytest

from embodichain.gen_sim.action_agent_pipeline.generation import (
    config_bundle_builders as builders,
)
from embodichain.gen_sim.action_agent_pipeline.generation.config_io import (
    validate_config_bundle,
)
from embodichain.gen_sim.action_agent_pipeline.generation.config_types import (
    ArrangementLineSpec,
    ArrangementLineStepSpec,
    RelativePlacementSpec,
    RelativePlacementStepSpec,
    SceneObject,
    StackingSpec,
    StackingStepSpec,
)
from embodichain.gen_sim.action_agent_pipeline.generation.seed_task_graph import (
    seed_task_graph_hash,
)


def test_arrangement_builder_publishes_complete_bundle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    robot_profile = _patch_builder_dependencies(monkeypatch)
    spec = ArrangementLineSpec(
        table_source_uid="table",
        task_description="Arrange both cans by size.",
        task_prompt_summary="Arrange cans.",
        basic_background_notes="",
        order_by="size",
        order_direction="ascending",
        axis="world_x",
        anchor="center",
        steps=(
            _arrangement_step("can_a", 0),
            _arrangement_step("can_b", 1),
        ),
        line_origin_xy=[0.0, 0.0],
        spacing=0.1,
        layout_clearance=0.02,
    )

    bundle = builders._build_arrangement_line_bundle(
        scene_dir=tmp_path,
        source_config={},
        spec=spec,
        project_name="project",
        task_name="arrangement",
        robot_profile=robot_profile,
        target_body_scale=1.0,
        max_episodes=1,
        max_episode_steps=100,
        mesh_normalizer=SimpleNamespace(),
        source_scene_body_scale_mode=None,
        preserve_source_scene_geometry=False,
        source_scene_z_rotation_degrees=0.0,
        arrangement_debug_visualization=False,
        load_template_material=False,
    )

    _assert_bundle_contract(bundle)


def test_stacking_builder_publishes_complete_bundle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    robot_profile = _patch_builder_dependencies(monkeypatch)
    spec = StackingSpec(
        table_source_uid="table",
        task_description="Stack both cans on the basket.",
        task_prompt_summary="Stack cans.",
        basic_background_notes="",
        stack_mode="on_top",
        order_by="explicit",
        anchor="object",
        anchor_xy=[0.0, 0.0],
        steps=(
            _stacking_step("can_a", 0, "basket"),
            _stacking_step("can_b", 1, "can_a"),
        ),
        anchor_source_uid="basket",
        anchor_runtime_uid="basket",
    )

    bundle = builders._build_stacking_bundle(
        scene_dir=tmp_path,
        source_config={},
        spec=spec,
        project_name="project",
        task_name="stacking",
        robot_profile=robot_profile,
        target_body_scale=1.0,
        max_episodes=1,
        max_episode_steps=100,
        mesh_normalizer=SimpleNamespace(),
        source_scene_body_scale_mode=None,
        preserve_source_scene_geometry=False,
        source_scene_z_rotation_degrees=0.0,
        load_template_material=False,
    )

    _assert_bundle_contract(bundle)


def test_relative_builder_publishes_complete_bundle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    robot_profile = _patch_builder_dependencies(monkeypatch)
    placement = RelativePlacementStepSpec(
        intent="place_relative",
        moved_source_uid="can_a",
        reference_source_uid="basket",
        moved_runtime_uid="can_a",
        reference_runtime_uid="basket",
        relation="inside",
        active_side="left",
        release_offset=[0.0, 0.0, 0.1],
        high_offset=[0.0, 0.0, 0.2],
        arm_request="left",
        step_id="s01_place_can_a_inside_basket",
    )
    spec = RelativePlacementSpec(
        intent=placement.intent,
        table_source_uid="table",
        moved_source_uid=placement.moved_source_uid,
        reference_source_uid=placement.reference_source_uid,
        moved_runtime_uid=placement.moved_runtime_uid,
        reference_runtime_uid=placement.reference_runtime_uid,
        relation=placement.relation,
        active_side=placement.active_side,
        task_description="Put the can in the basket.",
        task_prompt_summary="Put away the can.",
        basic_background_notes="",
        action_sketch=[],
        release_offset=list(placement.release_offset),
        high_offset=list(placement.high_offset),
        placements=(placement,),
    )

    bundle = builders._build_relative_placement_bundle(
        scene_dir=tmp_path,
        source_config={},
        spec=spec,
        project_name="project",
        task_name="relative",
        robot_profile=robot_profile,
        target_body_scale=1.0,
        preserve_source_target_body_scale=False,
        source_target_body_scale_multiplier=None,
        source_scene_body_scale_mode=None,
        max_episodes=1,
        max_episode_steps=100,
        mesh_normalizer=SimpleNamespace(),
        preserve_source_scene_geometry=False,
        source_scene_z_rotation_degrees=0.0,
        inside_container_slot_distance_scale=1.0,
        surface_release_clearance=0.01,
        load_template_material=False,
    )

    _assert_bundle_contract(bundle)


def _patch_builder_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> SimpleNamespace:
    """Replace geometry and simulator factories while exercising real builders."""
    scene_objects = [
        SceneObject("table", "background", {}),
        SceneObject("can_a", "rigid_object", {}),
        SceneObject("can_b", "rigid_object", {}),
        SceneObject("basket", "rigid_object", {}),
    ]
    runtime_uids = {obj.source_uid: obj.source_uid for obj in scene_objects}
    robot_profile = SimpleNamespace(
        make_robot_config=lambda table_top_z: {
            "uid": "robot",
            "table_top_z": table_top_z,
        },
        summary=lambda: {"id": "test_robot"},
        robot_meta_type="test_robot",
    )

    monkeypatch.setattr(
        builders, "_collect_scene_objects", lambda config: scene_objects
    )
    monkeypatch.setattr(
        builders,
        "_relative_scene_runtime_uid_mapping",
        lambda objects, **kwargs: dict(runtime_uids),
    )
    monkeypatch.setattr(
        builders,
        "_make_background_config",
        lambda *args, **kwargs: {"uid": "table"},
    )
    monkeypatch.setattr(builders, "_mesh_config_world_zmax", lambda config: 0.75)
    monkeypatch.setattr(
        builders,
        "_make_sensor_config_factory_for_robot",
        lambda config: lambda: [{"uid": "camera"}],
    )
    monkeypatch.setattr(
        builders,
        "_make_relative_rigid_object_config",
        lambda *, runtime_uid, **kwargs: {"uid": runtime_uid},
    )
    monkeypatch.setattr(
        builders,
        "_make_arrangement_events_config",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(builders, "_make_observations_config", lambda config: {})
    monkeypatch.setattr(builders, "_make_light_config", lambda: [])
    monkeypatch.setattr(
        builders, "make_runtime_agent_config", lambda: {"TaskAgent": {}}
    )
    monkeypatch.setattr(builders, "_source_body_scale", lambda obj: 1.0)
    monkeypatch.setattr(
        builders,
        "_source_scene_body_scale_override",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        builders,
        "_relative_generated_object_body_scale",
        lambda *args, **kwargs: 1.0,
    )
    monkeypatch.setattr(
        builders,
        "_relative_rigid_object_max_convex_hull_num",
        lambda *args, **kwargs: 1,
    )
    monkeypatch.setattr(
        builders,
        "_moved_rigid_object_max_convex_hull_num",
        lambda obj: 1,
    )
    monkeypatch.setattr(
        builders,
        "_container_rigid_object_max_convex_hull_num",
        lambda obj: 1,
    )

    # Geometry mutation is tested in its owner modules. Builder tests keep these
    # seams inert so failures identify bundle assembly rather than mesh fixtures.
    for name in (
        "_apply_scene_z_rotation",
        "_apply_source_scene_transforms",
        "_maybe_apply_source_scene_body_scale",
        "_maybe_apply_source_scene_xy_scale",
        "_maybe_apply_tabletop_z_placement",
        "_maybe_preserve_source_scene_vertical_contacts",
    ):
        monkeypatch.setattr(builders, name, lambda *args, **kwargs: None)
    for name in (
        "_with_arrangement_generated_pose_targets",
        "_with_coordinated_side_release_height_offsets",
        "_with_coordinated_transport_geometry",
        "_with_final_auto_arm_sides",
        "_with_inside_container_slot_offsets",
        "_with_on_surface_release_offsets",
        "_with_self_relative_absolute_targets",
        "_with_stacking_generated_targets",
    ):
        monkeypatch.setattr(
            builders,
            name,
            lambda spec, *args, **kwargs: spec,
        )
    monkeypatch.setattr(
        builders,
        "_source_objects_by_runtime_uid",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        builders,
        "_make_arrangement_extensions_config",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        builders,
        "_make_relative_extensions_config",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        builders,
        "_make_stacking_extensions_config",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        builders,
        "_make_arrangement_dataset_config",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        builders,
        "_make_relative_dataset_config",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        builders,
        "_make_stacking_dataset_config",
        lambda *args, **kwargs: {},
    )
    return robot_profile


def _assert_bundle_contract(bundle: dict) -> None:
    validate_config_bundle(bundle)
    assert set(bundle) == {
        "agent_config",
        "gym_config",
        "seed_task_graph",
        "summary",
    }
    assert bundle["summary"]["seed_graph_hash"] == seed_task_graph_hash(
        bundle["seed_task_graph"]
    )


def _arrangement_step(runtime_uid: str, slot_index: int) -> ArrangementLineStepSpec:
    return ArrangementLineStepSpec(
        source_uid=runtime_uid,
        runtime_uid=runtime_uid,
        slot_index=slot_index,
        active_side="left",
        target_xy=[0.0, float(slot_index) * 0.1],
        release_position=[0.0, float(slot_index) * 0.1, 0.1],
        high_position=[0.0, float(slot_index) * 0.1, 0.2],
    )


def _stacking_step(
    runtime_uid: str,
    layer_index: int,
    support_runtime_uid: str,
) -> StackingStepSpec:
    return StackingStepSpec(
        source_uid=runtime_uid,
        runtime_uid=runtime_uid,
        layer_index=layer_index,
        active_side="left",
        target_position=[0.0, 0.0, 0.1 + layer_index * 0.1],
        high_position=[0.0, 0.0, 0.2 + layer_index * 0.1],
        support_runtime_uid=support_runtime_uid,
    )
