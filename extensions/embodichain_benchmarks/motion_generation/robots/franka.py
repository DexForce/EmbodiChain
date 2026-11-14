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

"""Built-in Franka robot providers."""

from __future__ import annotations

from embodichain.lab.sim.cfg import RobotCfg
from .._atomic_fixtures import (
    _create_franka_pgi_cfg,
    _create_franka_native_hand_cfg,
    _create_franka_robotiq_cfg,
)

from ..registry import register_robot_provider
from .base import RobotProvider

__all__ = ["FrankaPandaProvider", "FrankaPgiProvider", "FrankaRobotiqProvider"]


class FrankaPandaProvider(RobotProvider):
    """Build the stock Franka Panda with its native two-finger hand and TCP."""

    def build_cfg(self) -> RobotCfg:
        """Build a stock Panda while applying optional suite overrides."""
        values = {
            "uid": "benchmark_franka_panda",
            **dict(self.spec.config),
        }
        values.pop(
            "robot_type", None
        )  # Legacy suite selector; the embodiment selects FR3.
        return _create_franka_native_hand_cfg(values)


class FrankaPgiProvider(RobotProvider):
    """Build the legacy Franka + PGI assembly for custom benchmark suites.

    The suites were authored with the base rotated 180 degrees, as in the
    atomic-skill tutorial scenes, so the provider keeps that rotation; suite
    overrides apply on top.
    """

    def build_cfg(self) -> RobotCfg:
        """Build the Franka + PGI benchmark configuration."""
        return _create_franka_pgi_cfg(
            {
                "uid": "benchmark_franka_pgi",
                "init_rot": [0.0, 0.0, 180.0],
                **dict(self.spec.config),
            }
        )


class FrankaRobotiqProvider(RobotProvider):
    """Build the Franka arm with Robotiq 2F-140 for Atomic Skill suites.

    The seven-joint arm ends at ``fr3_link8``; its TCP carries the mounted
    hand's rotation and 0.21 m tool offset. The six-joint ``eef`` group follows
    the authored Robotiq linkage order. Suite overrides apply last.
    """

    def build_cfg(self) -> RobotCfg:
        """Build an independent assembly with the suites' rotated base.

        Returns:
            Robot configuration with arm solver, gripper drives and contacts.
        """
        return _create_franka_robotiq_cfg(
            {
                "uid": "benchmark_franka_robotiq",
                "init_rot": [0.0, 0.0, 180.0],
                **dict(self.spec.config),
            }
        )


register_robot_provider("franka_robotiq", FrankaRobotiqProvider)
register_robot_provider("franka_panda", FrankaPandaProvider)
register_robot_provider("franka_pgi", FrankaPgiProvider)
