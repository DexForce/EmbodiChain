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

"""Tests for task-first official task discovery and registration."""

from __future__ import annotations

import importlib
from importlib.util import find_spec
from pathlib import Path

from embodichain.lab.gym.utils.registration import (
    REGISTERED_ENVS,
    discover_task_packages,
)
from embodichain.lab.sim.cfg import physics_cfg_for_backend
from embodichain.learning.rl.env import get_registered_learning_env_names
from embodichain.utils.utility import load_config
from embodichain_tasks.classic_control.point_mass import PointMassEnv

EXPECTED_IMPORT_REGISTERED_TASK_MODULES = {
    "BlocksRankingRGB-v1": "embodichain_tasks.manipulation.tableware.blocks_ranking_rgb",
    "BlocksRankingSize-v1": "embodichain_tasks.manipulation.tableware.blocks_ranking_size",
    "CartPoleRL": "embodichain_tasks.classic_control.cart_pole",
    "MatchObjectContainer-v1": "embodichain_tasks.manipulation.tableware.match_object_container",
    "PlaceObjectDrawer-v1": "embodichain_tasks.manipulation.tableware.place_object_drawer",
    "PushCubeRL": "embodichain_tasks.manipulation.push_cube",
    "ScoopIce-v1": "embodichain_tasks.manipulation.tableware.scoop_ice",
    "SimpleTask-v1": "embodichain_tasks.special.simple_task",
    "StackBlocksTwo-v1": "embodichain_tasks.manipulation.tableware.stack_blocks_two",
    "StackCups-v1": "embodichain_tasks.manipulation.tableware.stack_cups",
    "StayStillSave-v1": "embodichain_tasks.special.stay_still_save",
}
REMOVED_AGENT_ENV_IDS = {"PourWaterAgent-v3", "RearrangementAgent-v3"}
CONFIG_DEFINED_TASK_PROGRAM_TASKS = {"pour_water"}
RL_SIMULATOR_ENV_IDS = {"CartPoleRL", "PushCubeRL"}
TABLEWARE_CONFIG_TASKS = {
    "blocks_ranking_rgb",
    "blocks_ranking_size",
    "match_object_container",
    "place_object_drawer",
    "pour_water",
    "scoop_ice",
    "stack_blocks_two",
    "stack_cups",
}
TASK_CONFIG_ROOT = (
    Path(__file__).resolve().parents[3] / "embodichain_tasks/configs/tasks"
)


def test_official_gym_configs_own_one_explicit_physics_backend() -> None:
    """Every runnable config resolves one backend-aligned physics mapping."""
    for config_path in sorted(TASK_CONFIG_ROOT.rglob("*")):
        if config_path.suffix.lower() not in {".json", ".yaml", ".yml"}:
            continue
        config = load_config(config_path)
        if type(config) is not dict or not config.get("id"):
            continue

        environment = config.get("environment")
        if type(environment) is dict and "component" in environment:
            assert "physics" not in config, config_path
            assert "physics_config" not in config, config_path
            owner_path = config_path.parent / environment["component"]
            owner = load_config(owner_path)
        else:
            owner_path = config_path
            owner = config

        backend = owner.get("physics")
        assert backend in {"default", "newton"}, owner_path
        physics_config = owner.get("physics_config", {})
        assert type(physics_config) is dict, owner_path
        config_type = type(physics_cfg_for_backend(backend))
        config_type(**physics_config)


def test_import_registered_gym_ids_resolve_to_flat_task_modules() -> None:
    """Import-registered Gym IDs resolve to task-named Python modules."""
    discover_task_packages()

    actual_modules = {
        env_id: REGISTERED_ENVS[env_id].cls.__module__
        for env_id in EXPECTED_IMPORT_REGISTERED_TASK_MODULES
    }
    assert actual_modules == EXPECTED_IMPORT_REGISTERED_TASK_MODULES
    assert all(
        not hasattr(importlib.import_module(module_name), "__path__")
        for module_name in set(EXPECTED_IMPORT_REGISTERED_TASK_MODULES.values())
    )


def test_point_mass_registration_uses_classic_control_task_module() -> None:
    """The lightweight RL task remains registered outside an RL code silo."""
    discover_task_packages()

    assert "pointmassrl" in get_registered_learning_env_names()
    assert PointMassEnv.__module__ == "embodichain_tasks.classic_control.point_mass"


def test_simulator_rl_tasks_declare_rl_capability() -> None:
    """Simulator tasks with bundled RL configs expose that capability."""
    discover_task_packages()

    actual = {
        env_id
        for env_id in EXPECTED_IMPORT_REGISTERED_TASK_MODULES
        if REGISTERED_ENVS[env_id].supports_rl
    }
    assert actual == RL_SIMULATOR_ENV_IDS


def test_legacy_tableware_agent_envs_are_not_registered() -> None:
    """Removed BaseAgentEnv variants do not remain in the task registry."""
    discover_task_packages()

    assert REMOVED_AGENT_ENV_IDS.isdisjoint(REGISTERED_ENVS)
    assert find_spec("embodichain_tasks.tableware") is None
    assert find_spec("embodichain_tasks.manipulation.tableware.base_agent_env") is None


def test_tableware_configs_are_nested_under_manipulation() -> None:
    """Tableware configs mirror their Python subdomain hierarchy."""
    tableware_root = TASK_CONFIG_ROOT / "manipulation/tableware"

    assert {path.name for path in tableware_root.iterdir() if path.is_dir()} == (
        TABLEWARE_CONFIG_TASKS
    )
    assert not (TASK_CONFIG_ROOT / "tableware").exists()


def test_config_defined_task_programs_do_not_need_python_task_modules() -> None:
    """Pure Task Programs keep their runtime and workflow in task-local config."""
    for task_name in CONFIG_DEFINED_TASK_PROGRAM_TASKS:
        module_name = f"embodichain_tasks.manipulation.tableware.{task_name}"
        config_root = TASK_CONFIG_ROOT / "manipulation/tableware" / task_name

        assert find_spec(module_name) is None
        environment_path = config_root / "env.yaml"
        task_path = config_root / "task.cobotmagic.yaml"
        assert environment_path.is_file()
        assert task_path.is_file()
        environment = load_config(environment_path)
        task = load_config(task_path)
        assert environment["environment_id"] == task_name
        assert environment["physics"] in {"default", "newton"}
        assert "task_program" not in environment
        assert task["environment"] == {"component": "env.yaml"}
        assert (config_root / "task_program/program.yaml").is_file()
        integration_path = config_root / "task_program/integration.yaml"
        assert integration_path.is_file()
        assert "scene_binding" in load_config(integration_path)
        assert not (config_root / "task_program/scene_binding.yaml").exists()
