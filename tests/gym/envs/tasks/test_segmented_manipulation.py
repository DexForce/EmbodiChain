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

"""CPU checks for measured, configuration-defined manipulation experts."""

from __future__ import annotations

from pathlib import Path

import pytest

from embodichain.lab.gym.envs import EmbodiedEnv
from embodichain.lab.gym.utils._component_composition import (
    _resolve_gym_components,
    _validate_scene_binding_targets,
)
from embodichain.lab.gym.utils.gym_utils import config_to_cfg
from embodichain.lab.gym.utils.registration import REGISTERED_ENVS
from embodichain.lab.task_program import load_task_program
from embodichain.lab.task_program.integrations._configured_composition import (
    _load_configured_task_program_deployment,
)
from embodichain.utils.utility import load_config

_CONFIG_ROOT = (
    Path(__file__).resolve().parents[4]
    / "extensions/embodichain_tasks/configs/tasks/manipulation/tableware"
)


def _deployment(task_name: str):
    """Compose the production deployment and strictly preflight its program."""
    path = _CONFIG_ROOT / task_name / "task.cobotmagic.yaml"
    physical = _resolve_gym_components(load_config(path), base_dir=path.parent)
    deployment = _load_configured_task_program_deployment(
        task_program=physical.config["task_program"],
        skill_profile=physical.embodiment_skill_profile,
        base_dir=path.parent,
    )
    _validate_scene_binding_targets(
        deployment.scene_binding, simulation=physical.scene_config
    )
    program = load_task_program(
        deployment.program_path,
        integration=deployment.selection,
        validation_context=deployment.integration.registration.catalog,
    )
    compiled = deployment.integration.registration.catalog.preflight(program)
    return deployment, program, compiled


@pytest.mark.parametrize(
    ("task_name", "environment_id", "segment_names"),
    (
        (
            "stack_blocks_two",
            "StackBlocksTwo-v1",
            ("pick_block_2", "place_block_2_on_block_1"),
        ),
        (
            "blocks_ranking_rgb",
            "BlocksRankingRGB-v1",
            (
                "pick_red_block",
                "place_red_block",
                "pick_blue_block",
                "place_blue_block",
            ),
        ),
    ),
)
def test_manipulation_experts_use_common_environment_and_preserve_semantic_segments(
    task_name: str, environment_id: str, segment_names: tuple[str, ...], monkeypatch
) -> None:
    """The configured experts keep stable IDs without custom environment classes."""
    monkeypatch.setattr("embodichain.data.get_data_path", lambda value: value)
    path = _CONFIG_ROOT / task_name / "task.cobotmagic.yaml"
    cfg = config_to_cfg(load_config(path), source_path=path)
    _, program, compiled = _deployment(task_name)

    assert cfg.task_program.program_id == program.program_id
    assert REGISTERED_ENVS[environment_id].cls is EmbodiedEnv
    assert tuple(segment.name for segment in compiled.iter_segments()) == segment_names
    assert all(segment.instruction for segment in program.program.items)
    assert tuple(
        call.call.semantic_id
        for segment in compiled.iter_segments()
        for call in segment.calls
    ) == (
        ("pick", "place")
        if task_name == "stack_blocks_two"
        else ("pick", "place", "pick", "place")
    )


def test_stack_program_keeps_measured_lift_and_relative_upright_stack_gates() -> None:
    """Randomized base poses remain authoritative at placement and acceptance."""
    deployment, program, _ = _deployment("stack_blocks_two")
    picking, placing = program.program.items
    (lift,) = picking.validators
    assert (lift.kind, lift.object, lift.minimum_lift) == (
        "object_lifted",
        "block_2",
        0.05,
    )
    position, base_upright, top_upright = placing.validators
    assert (position.object, position.reference) == ("block_2", "block_1")
    assert position.position_offset == (0.0, 0.0, 0.05)
    assert position.position_tolerance == (0.025, 0.025, 0.012)
    assert (base_upright.object, top_upright.object) == ("block_1", "block_2")
    assert base_upright.maximum_tilt_degrees == top_upright.maximum_tilt_degrees == 45.0
    assert {policy.entity for policy in placing.post} == {"block_1", "block_2"}
    (support,) = deployment.integration.registration.scene_binding.support_surfaces
    assert support.parent_id == "block_1"
    assert support.object_target_frame == "world_axes"
    assert support.object_target_pose[11] == 0.05


def test_rgb_program_keeps_sequential_arm_ownership_and_world_relative_targets() -> (
    None
):
    """Red uses the right arm, blue uses the left, and final acceptance rechecks red."""
    deployment, program, _ = _deployment("blocks_ranking_rgb")
    red_pick, red_place, blue_pick, blue_place = program.program.items
    assert (
        red_pick.steps.call.resources
        == red_place.steps.call.resources
        == {"primary": "primary_manipulator"}
    )
    assert (
        blue_pick.steps.call.resources
        == blue_place.steps.call.resources
        == {"primary": "secondary_manipulator"}
    )
    resources = {
        resource.resource_id: resource
        for resource in deployment.integration.registration.robot_profile_binding.resources
    }
    assert resources["primary_manipulator"].endpoints[0].control_part == "right_arm"
    assert resources["secondary_manipulator"].endpoints[0].control_part == "left_arm"
    surfaces = deployment.integration.registration.scene_binding.support_surfaces
    assert [support.parent_id for support in surfaces] == ["block_2", "block_2"]
    assert [support.object_target_frame for support in surfaces] == [
        "world_axes",
        "world_axes",
    ]
    assert [support.object_target_pose[3] for support in surfaces] == [-0.08, 0.08]
    assert [support.object_target_pose[11] for support in surfaces] == [0.08, 0.08]
    assert red_place.validators[0].position_tolerance == (0.035, 0.03, None)
    assert [validator.object for validator in blue_place.validators] == [
        "block_3",
        "block_1",
    ]
    assert red_place.post[0].entity == "block_1"
    assert blue_place.post[0].entity == "block_3"


@pytest.mark.parametrize("task_name", ("stack_blocks_two", "blocks_ranking_rgb"))
def test_manipulation_grasp_dwell_and_placement_use_shared_atomic_options(
    task_name: str,
) -> None:
    """Grasp preload lives in PickUp options and does not mutate object dynamics."""
    deployment, _, _ = _deployment(task_name)
    (preset,) = deployment.integration.registration.robot_profile_binding.presets
    options = preset.skill_option_templates
    assert options["pick"].grasp_settle_steps == 45
    assert options["pick"].hand_interp_steps == 10
    assert options["place"].preserve_current_object_orientation is True
    assert preset.motion_policy.sample_count == 90
