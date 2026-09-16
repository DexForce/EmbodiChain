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

"""Production configuration contracts for the cube Pick-and-Pour deployment."""

from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path

import pytest
import torch

from embodichain.lab.gym.utils._component_composition import (
    _ResolvedGymComponents,
    _resolve_gym_components,
    _validate_scene_binding_targets,
)
from embodichain.lab.sim.atomic_actions import PickUpOptions, PourOptions
from embodichain.lab.task_program import load_task_program
from embodichain.lab.task_program.integrations._configured_composition import (
    _ConfiguredTaskProgramDeployment,
    _load_configured_task_program_deployment,
)
from embodichain.utils.utility import load_config

_DEPLOYMENT_PATH = (
    Path(__file__).parents[4]
    / "embodichain_tasks/configs/tasks/manipulation/pour/task.ur5.yaml"
)
_POUR_CALL_ID = "simulation.pour"
_CUBE_SIZE = (0.05, 0.05, 0.05)
_CUBE_MASS = 0.05


@pytest.fixture
def pour_deployment() -> (
    tuple[_ResolvedGymComponents, _ConfiguredTaskProgramDeployment]
):
    """Resolve the packaged deployment through the production component loader."""
    physical = _resolve_gym_components(
        load_config(_DEPLOYMENT_PATH), base_dir=_DEPLOYMENT_PATH.parent
    )
    assert physical.embodiment_skill_profile is not None
    assert physical.scene_config is not None
    deployment = _load_configured_task_program_deployment(
        task_program=physical.config["task_program"],
        skill_profile=physical.embodiment_skill_profile,
        base_dir=_DEPLOYMENT_PATH.parent,
    )
    _validate_scene_binding_targets(
        deployment.scene_binding, simulation=physical.scene_config
    )
    return physical, deployment


def test_pour_program_picks_before_tilting_the_same_cube(
    pour_deployment: tuple[_ResolvedGymComponents, _ConfiguredTaskProgramDeployment],
) -> None:
    """Both calls preflight against the composed, registered production catalog."""
    _, deployment = pour_deployment
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

    assert [call.semantic_id for call in calls] == ["pick", _POUR_CALL_ID]
    assert calls[0].object.entity_id == "cube"
    assert dict(calls[1].arguments) == {"object": "cube"}


def test_pour_preserves_axis_aware_grasp_and_signed_tilt_options(
    pour_deployment: tuple[_ResolvedGymComponents, _ConfiguredTaskProgramDeployment],
) -> None:
    """Pour receives the held cube's local axis and an explicit 45-degree tilt."""
    _, deployment = pour_deployment
    registration = deployment.integration.registration
    grasp = registration.scene_binding.antipodal_grasps[0]
    options = registration.robot_profile_binding.presets[0].action_option_templates

    assert grasp.object_id == "cube"
    assert grasp.internal_axis == pytest.approx((1.0, 0.0, 0.0))
    assert type(options["pick"]) is PickUpOptions
    assert torch.allclose(
        options["pick"].approach_direction, torch.tensor([-0.707, 0.0, -0.707])
    )
    assert options["pick"].pre_grasp_distance == pytest.approx(0.15)
    assert options["pick"].lift_height == pytest.approx(0.16)
    assert type(options[_POUR_CALL_ID]) is PourOptions
    assert options[_POUR_CALL_ID].rotate_angle == pytest.approx(math.pi / 4.0)
    factories = registration.registered_semantic_lowerer_factories
    assert len(factories) == 1
    assert factories[0].call_id == _POUR_CALL_ID
    assert factories[0].object_id == grasp.object_id


def test_pour_cube_is_dynamic_and_starts_above_the_ground(
    pour_deployment: tuple[_ResolvedGymComponents, _ConfiguredTaskProgramDeployment],
) -> None:
    """The physical scene uses ordinary dynamics with no replay-clearing event."""
    physical, _ = pour_deployment
    cube = physical.scene_config["rigid_object"][0]

    assert cube["body_type"] == "dynamic"
    assert cube["shape"]["size"] == list(_CUBE_SIZE)
    assert cube["init_pos"][2] == pytest.approx(_CUBE_SIZE[2] / 2.0)
    assert cube["attrs"]["mass_props"]["mass"] == pytest.approx(_CUBE_MASS)
    assert physical.config["env"]["events"] == {}


def test_pour_rejects_a_missing_physical_cube_before_runtime(
    pour_deployment: tuple[_ResolvedGymComponents, _ConfiguredTaskProgramDeployment],
) -> None:
    """An unbound semantic object cannot pass the deployment's physical boundary."""
    physical, deployment = pour_deployment
    scene = deepcopy(physical.scene_config)
    scene["rigid_object"] = []

    with pytest.raises(ValueError, match="'cube'.*not declared"):
        _validate_scene_binding_targets(deployment.scene_binding, simulation=scene)


__all__: list[str] = []
