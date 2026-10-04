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

"""UR5 robot provider for motion-generation benchmarks."""

from __future__ import annotations

from embodichain.lab.sim.cfg import RobotCfg
from embodichain.lab.sim.robots import URRobotCfg
from embodichain.lab.sim.utility.cfg_utils import merge_robot_cfg
from scripts.tutorials.atomic_action.tutorial_utils import create_ur5_gripper_robot_cfg

from ..registry import register_robot_provider
from .base import RobotProvider

__all__ = ["UR5Provider", "UR5PgiProvider"]


class UR5Provider(RobotProvider):
    """Build the fixed-base UR5 used by the NMG free-space benchmark."""

    def build_cfg(self) -> RobotCfg:
        """Build a stock UR5 while applying optional suite overrides.

        Returns:
            The configured UR5 robot.
        """
        values = {
            "uid": "benchmark_ur5",
            "robot_type": "ur5",
            **dict(self.spec.config),
        }
        return URRobotCfg.from_dict(values)


class UR5PgiProvider(RobotProvider):
    """Build the UR5 with the tutorial PGI gripper for Atomic Tasks."""

    def build_cfg(self) -> RobotCfg:
        """Use the tutorial UR5 arm, gripper, and TCP configuration.

        Returns:
            The configured UR5 with PGI gripper.
        """
        cfg = create_ur5_gripper_robot_cfg()
        return merge_robot_cfg(
            cfg,
            {
                "uid": "benchmark_ur5_pgi",
                **dict(self.spec.config),
            },
        )


register_robot_provider("ur5", UR5Provider)
register_robot_provider("ur5_pgi", UR5PgiProvider)
