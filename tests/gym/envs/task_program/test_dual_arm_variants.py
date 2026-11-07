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

"""Production deployment coverage for the two supported dual-arm embodiments."""

from __future__ import annotations

from copy import deepcopy
import importlib
from pathlib import Path

import pytest

from embodichain.lab.gym.utils._component_composition import _resolve_gym_components
from embodichain.lab.gym.utils.gym_utils import config_to_cfg
from embodichain.lab.gym.utils.registration import get_env_spec
from embodichain.utils.utility import load_config

_CONFIG_ROOT = (
    Path(__file__).resolve().parents[4] / "extensions/embodichain_tasks/configs"
)
_VARIANTS = ("dual_ur5_dh_pgi_140_80", "tianji_marvin")
_BASELINES = (
    "blocks_ranking_size",
    "match_object_container",
    "place_object_drawer",
    "stack_cups",
)
_TASKS = (
    "hand_over",
    "repeated_pick_place",
    "tableware/pour_water",
    "tableware/blocks_ranking_rgb",
    "tableware/stack_blocks_two",
    *(f"tableware/{name}" for name in _BASELINES),
)


def _deployment_path(task: str, variant: str) -> Path:
    if task == "hand_over" and variant == "dual_ur5_dh_pgi_140_80":
        return _CONFIG_ROOT / "tasks/manipulation/hand_over/task.dual_ur5.yaml"
    return _CONFIG_ROOT / "tasks/manipulation" / task / f"task.{variant}.yaml"


@pytest.mark.parametrize("task", _TASKS)
@pytest.mark.parametrize("variant", _VARIANTS)
def test_dual_arm_deployment_decodes_and_registers(task: str, variant: str) -> None:
    """Every catalog variant resolves its actual robot, sensors and Gym class."""
    path = _deployment_path(task, variant)
    payload = load_config(path)
    if task.split("/")[-1] in _BASELINES:
        importlib.import_module(
            f"embodichain_tasks.manipulation.{task.replace('/', '.')}"
        )
    cfg = config_to_cfg(payload, source_path=path)
    cfg.robot.validate()
    assert get_env_spec(payload["id"]).cls is not None
    assert {
        "left_arm",
        "right_arm",
        "left_eef",
        "right_eef",
    } <= cfg.robot.control_parts.keys()
    sensors = [*cfg.sensor, *cfg.robot.sensors]
    assert "cam_high" in {sensor.uid for sensor in sensors}
    assert len({sensor.uid for sensor in sensors}) == len(sensors)
    if task.startswith("tableware/") and task != "tableware/pour_water":
        observation = cfg.observations.norm_robot_eef_joint
        assert observation.params == {"control_parts": ["left_eef", "right_eef"]}


def test_variant_catalogs_cover_all_tasks_with_unique_ids() -> None:
    """Discovery exposes both embodiments without shadowing other deployments."""
    identifiers = []
    for task in _TASKS:
        directory = _CONFIG_ROOT / "tasks/manipulation" / task
        catalog = load_config(directory / "catalog.yaml")
        for variant in _VARIANTS:
            filename = catalog["deployments"][variant]["config"]
            expected = _deployment_path(task, variant)
            assert directory / filename == expected
            identifiers.append(load_config(expected)["id"])
    assert len(identifiers) == len(set(identifiers))


@pytest.mark.parametrize("variant", _VARIANTS)
def test_resource_aliases_preserve_bimanual_physical_claims(variant: str) -> None:
    """Generic manipulator aliases retain the same physical DAG leaves and endpoints."""
    path = _CONFIG_ROOT / "components/embodiments" / f"{variant}.yaml"
    physical = _resolve_gym_components(
        {"embodiment": {"component": str(path)}}, base_dir=path.parent
    )
    profile = physical.embodiment_skill_profile
    assert profile is not None
    resources = {resource["resource_id"]: resource for resource in profile["resources"]}
    for alias, side in (
        ("primary_manipulator", "right"),
        ("secondary_manipulator", "left"),
    ):
        assert resources[alias]["members"] == [side]
        assert resources[alias]["endpoints"] == resources[side]["endpoints"]
    assert profile["contract_id"] == "dual_arm_parallel_gripper"


@pytest.mark.parametrize(
    "task", ("tableware/blocks_ranking_rgb", "tableware/stack_blocks_two")
)
def test_variants_reuse_program_and_measured_acceptance(task: str) -> None:
    """Changing the robot keeps the same program, segment gates and reset scene."""
    paths = [_deployment_path(task, variant) for variant in _VARIANTS]
    paths.append(paths[0].parent / "task.cobotmagic.yaml")
    payloads = [load_config(path) for path in paths]
    assert len({payload["task_program"]["program"] for payload in payloads}) == 1
    assert len({payload["environment"]["component"] for payload in payloads}) == 1
    template = deepcopy(
        load_config(paths[0].parent / payloads[0]["task_program"]["program"])
    )
    assert all(segment.get("validators") for segment in template["program"]["items"])
