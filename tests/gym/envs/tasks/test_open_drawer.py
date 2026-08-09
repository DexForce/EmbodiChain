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

"""Tests for the official Open Drawer task environment."""

from __future__ import annotations

import json

import pytest
import torch
from dexsim.types import DriveType

from embodichain.lab.gym.envs import EmbodiedEnv
from embodichain.lab.gym.utils.gym_utils import config_to_cfg
from embodichain.lab.gym.utils.registration import (
    REGISTERED_ENVS,
    discover_task_packages,
    make,
)

discover_task_packages()

from embodichain_tasks.configs import get_config_path  # noqa: E402
from embodichain_tasks.tableware.open_drawer import OpenDrawerEnv  # noqa: E402

CONFIG_PATH = get_config_path("gym/open_drawer/cobot_magic_3cam.json")


def _load_config() -> dict:
    """Load a fresh copy because ``config_to_cfg`` resolves paths in place."""
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


class TestOpenDrawerEnv:
    """Registration, configuration, and simulation tests for OpenDrawerEnv."""

    def test_registered_with_gym_id(self) -> None:
        """The migrated task is discoverable through the official registry."""
        assert "OpenDrawer-v1" in REGISTERED_ENVS
        spec = REGISTERED_ENVS["OpenDrawer-v1"]
        assert spec.cls is OpenDrawerEnv
        assert spec.max_episode_steps == 300
        assert issubclass(OpenDrawerEnv, EmbodiedEnv)

    def test_config_preserves_entity_specific_drive_defaults(self) -> None:
        """A partial drawer drive config stays none while the robot stays force."""
        config = _load_config()
        drawer_dict = config["articulation"][0]
        assert "drive_type" not in drawer_dict["drive_pros"]

        cfg = config_to_cfg(config)

        assert cfg.articulation[0].drive_pros.drive_type == "none"
        assert cfg.robot.drive_pros.drive_type == "force"

    @pytest.mark.requires_sim
    def test_built_environment_drive_types_and_demo(self) -> None:
        """The built task has correct drives and generates valid expert actions."""
        config = _load_config()
        config["headless"] = True
        cfg = config_to_cfg(config)

        env = make("OpenDrawer-v1", cfg=cfg)
        try:
            drawer = env.sim.get_articulation("drawer")
            assert drawer.get_joint_drive_type() == [[DriveType.NONE] * drawer.dof]
            assert env.robot.get_joint_drive_type(
                joint_ids=env.robot.active_joint_ids
            ) == [[DriveType.FORCE] * len(env.robot.active_joint_ids)]

            actions = env.create_demo_action_list()
            assert actions.ndim == 2
            assert actions.shape[0] > 0
            assert actions.shape[1] == len(env.active_joint_ids)
            assert torch.isfinite(actions).all()
        finally:
            env.close()
