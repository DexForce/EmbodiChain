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

from embodichain.lab.gym.utils.registration import (
    REGISTERED_ENVS,
    discover_task_packages,
)
from embodichain.learning.rl.env import get_registered_learning_env_names
from embodichain_tasks.classic_control.point_mass import PointMassEnv

EXPECTED_IMPORT_REGISTERED_TASK_MODULES = {
    "BlocksRankingRGB-v1": "embodichain_tasks.tableware.blocks_ranking_rgb.task",
    "BlocksRankingSize-v1": "embodichain_tasks.tableware.blocks_ranking_size.task",
    "CartPoleRL": "embodichain_tasks.classic_control.cart_pole.task",
    "MatchObjectContainer-v1": "embodichain_tasks.tableware.match_object_container.task",
    "PlaceObjectDrawer-v1": "embodichain_tasks.tableware.place_object_drawer.task",
    "PourWater-v3": "embodichain_tasks.tableware.pour_water.task",
    "PourWaterAgent-v3": "embodichain_tasks.tableware.pour_water.task",
    "PushCubeRL": "embodichain_tasks.manipulation.push_cube.task",
    "Rearrangement-v3": "embodichain_tasks.tableware.rearrangement.task",
    "RearrangementAgent-v3": "embodichain_tasks.tableware.rearrangement.task",
    "ScoopIce-v1": "embodichain_tasks.tableware.scoop_ice.task",
    "SimpleTask-v1": "embodichain_tasks.special.simple_task.task",
    "StackBlocksTwo-v1": "embodichain_tasks.tableware.stack_blocks_two.task",
    "StackCups-v1": "embodichain_tasks.tableware.stack_cups.task",
    "StayStillSave-v1": "embodichain_tasks.special.stay_still_save.task",
}


def test_import_registered_gym_ids_resolve_to_task_modules() -> None:
    """Import-registered Gym IDs resolve to classes owned by task.py."""
    discover_task_packages()

    actual_modules = {
        env_id: REGISTERED_ENVS[env_id].cls.__module__
        for env_id in EXPECTED_IMPORT_REGISTERED_TASK_MODULES
    }
    assert actual_modules == EXPECTED_IMPORT_REGISTERED_TASK_MODULES


def test_point_mass_registration_uses_classic_control_task_module() -> None:
    """The lightweight RL task remains registered outside an RL code silo."""
    discover_task_packages()

    assert "pointmassrl" in get_registered_learning_env_names()
    assert PointMassEnv.__module__ == (
        "embodichain_tasks.classic_control.point_mass.task"
    )
