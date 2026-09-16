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

"""Production configuration contracts for Press, Twist, and OpenDoor tasks."""

from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path

import pytest

from embodichain.lab.gym.utils._component_composition import (
    _ResolvedGymComponents,
    _resolve_gym_components,
    _validate_scene_binding_targets,
)
from embodichain.lab.sim.atomic_actions import (
    OpenDoorOptions,
    PressOptions,
    TwistOptions,
)
from embodichain.lab.sim.spawn.descriptors import _joint_property_matches
from embodichain.lab.task_program import load_task_program
from embodichain.lab.task_program.integrations._configured_composition import (
    _ConfiguredTaskProgramDeployment,
    _load_configured_task_program_deployment,
)
from embodichain.utils.utility import load_config

_TASK_ROOT = Path(__file__).parents[4] / "embodichain_tasks/configs/tasks/manipulation"
_TASKS = ("press", "twist", "open_door")
_TARGETS = {
    "press": ("microwave_button", "button_cap"),
    "twist": ("microwave_knob", "cap_1"),
    "open_door": ("microwave_handle", "door_handle"),
}
_OPTION_TYPES = {
    "press": PressOptions,
    "twist": TwistOptions,
    "open_door": OpenDoorOptions,
}
_MICROWAVE_JOINTS = (
    "door_hinge",
    "power_knob_rotation",
    "start_button_press",
    "timer_knob_rotation",
)


def _deployment(
    task: str,
) -> tuple[_ResolvedGymComponents, _ConfiguredTaskProgramDeployment]:
    """Resolve real packaged components through the same loader as run-env."""
    path = _TASK_ROOT / task / "task.ur5.yaml"
    physical = _resolve_gym_components(load_config(path), base_dir=path.parent)
    assert physical.embodiment_skill_profile is not None
    assert physical.scene_config is not None
    deployment = _load_configured_task_program_deployment(
        task_program=physical.config["task_program"],
        skill_profile=physical.embodiment_skill_profile,
        base_dir=path.parent,
    )
    _validate_scene_binding_targets(
        deployment.scene_binding, simulation=physical.scene_config
    )
    return physical, deployment


@pytest.mark.parametrize("task", _TASKS)
def test_articulation_program_preflights_with_its_registered_target(task: str) -> None:
    """Each configuration exposes a single supported, embodiment-independent call."""
    _, deployment = _deployment(task)
    catalog = deployment.integration.registration.catalog
    program = load_task_program(
        deployment.program_path,
        integration=deployment.selection,
        validation_context=catalog,
    )
    compiled = catalog.preflight(program)
    calls = tuple(
        invocation.call
        for segment in compiled.iter_segments()
        for invocation in segment.calls
    )

    assert compiled.segment_count == 1
    assert len(calls) == 1
    assert calls[0].semantic_id == f"simulation.articulation_link_{task}"
    target_key = "handle" if task == "open_door" else "target"
    expected_arguments = {target_key: _TARGETS[task][0]}
    if task == "open_door":
        # The asset's hinge interval is [0, 2.00713] rad; the goal is 60 degrees.
        expected_arguments["open_fraction"] = pytest.approx(math.pi / 3.0 / 2.00713)
    assert dict(calls[0].arguments) == expected_arguments


@pytest.mark.parametrize("task", _TASKS)
def test_articulation_scene_and_service_select_the_same_native_link(task: str) -> None:
    """Physical UID, semantic parent, and lowerer geometry must share one target."""
    physical, deployment = _deployment(task)
    registration = deployment.integration.registration
    root = registration.scene_binding.articulations[0]
    link = registration.scene_binding.links[0]
    factories = registration.registered_semantic_lowerer_factories

    assert physical.config["physics"] == "default"
    assert len(factories) == 1
    assert root.entity_id == link.articulation_id == factories[0].articulation_id
    assert (
        root.simulation_uid == factories[0].articulation_simulation_uid == "microwave"
    )
    assert physical.scene_config["articulation"][0]["uid"] == root.simulation_uid
    assert link.entity_id == factories[0].link_entity_id == _TARGETS[task][0]
    assert link.native_link_name == _TARGETS[task][1]
    assert factories[0].target_pose_mode == "snapshot"


@pytest.mark.parametrize("task", _TASKS)
def test_articulation_policy_preserves_typed_options_and_phase_budget(
    task: str,
) -> None:
    """Task options compose into a sufficiently long trajectory on the shared robot."""
    _, deployment = _deployment(task)
    profile = deployment.integration.registration.robot_profile_binding
    preset = profile.presets[0]
    options = preset.action_option_templates[f"simulation.articulation_link_{task}"]

    assert type(options) is _OPTION_TYPES[task]
    assert preset.motion_policy.strategy == "ik_interp"
    if task == "press":
        assert options.hand_interp_steps == 12
        assert options.approach_distance == pytest.approx(0.12)
        assert options.press_distance == pytest.approx(0.03)
        assert preset.motion_policy.sample_count >= options.hand_interp_steps + 8
    elif task == "twist":
        assert options.twist_angle == pytest.approx(-math.pi / 4.0)
        assert options.pre_grasp_distance == pytest.approx(0.12)
        assert preset.motion_policy.sample_count >= 2 * options.hand_interp_steps + 8
    else:
        assert options.approach_distance == pytest.approx(0.1)
        assert options.retract_distance == pytest.approx(0.1)
        minimum = 2 * options.hand_interp_steps + options.door_waypoint_count + 7
        assert preset.motion_policy.sample_count >= minimum


@pytest.mark.parametrize("task", _TASKS)
def test_articulation_deployment_rejects_a_missing_physical_root(task: str) -> None:
    """A semantic microwave cannot pass composition without its physical object."""
    physical, deployment = _deployment(task)
    scene = deepcopy(physical.scene_config)
    scene["articulation"] = []

    with pytest.raises(ValueError, match="'microwave'.*not declared"):
        _validate_scene_binding_targets(deployment.scene_binding, simulation=scene)


def test_door_uses_dense_handle_sampling_without_changing_other_tasks() -> None:
    """Handle contact sampling and cache refresh belong to the task override."""
    _, door = _deployment("open_door")
    _, press = _deployment("press")
    door_factory = (
        door.integration.adapter_factory.delegate._grasp_pose_generator_factories[
            "hand"
        ]
    )
    press_factory = (
        press.integration.adapter_factory.delegate._grasp_pose_generator_factories[
            "hand"
        ]
    )

    assert door_factory.opening_margin == pytest.approx(0.03)
    assert door_factory.sample_count == 10000
    assert door_factory.force_refresh is True
    assert press_factory.opening_margin == pytest.approx(0.002)
    assert press_factory.sample_count == 1000
    assert press_factory.force_refresh is None


def test_twist_joint_rules_keep_target_damped_without_a_return_spring() -> None:
    """Per-joint drive rules resolve uniquely and leave the knob position unforced."""
    physical, _ = _deployment("twist")
    props = physical.scene_config["articulation"][0]["joint_drive_props"]
    resolved = {
        field: dict(
            _joint_property_matches(
                props[field],
                list(_MICROWAVE_JOINTS),
                property_name=field,
                numeric_only=field != "target_mode",
                control_parts=None,
            )
        )
        for field in ("target_mode", "stiffness", "damping", "max_effort")
    }

    assert props["drive_type"] == "force"
    assert resolved["target_mode"]["power_knob_rotation"] == "velocity"
    assert resolved["stiffness"]["power_knob_rotation"] == 0.0
    assert resolved["damping"]["power_knob_rotation"] > 0.0
    assert resolved["max_effort"]["power_knob_rotation"] > 0.0
    for joint in set(_MICROWAVE_JOINTS) - {"power_knob_rotation"}:
        assert resolved["target_mode"][joint] == "position_velocity"
        assert resolved["stiffness"][joint] > 0.0


__all__: list[str] = []
