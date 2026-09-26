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
from embodichain.lab.sim.robots import FrankaPandaCfg
from embodichain.lab.sim.utility.cfg_utils import merge_robot_cfg
from scripts.tutorials.atomic_action.tutorial_utils import (
    create_franka_panda_robot_cfg,
)

from ..registry import register_robot_provider
from .base import RobotProvider

__all__ = ["FrankaPandaProvider", "FrankaPgiProvider"]


class FrankaPandaProvider(RobotProvider):
    """Build the stock Franka Panda used by the free-space track."""

    def build_cfg(self) -> RobotCfg:
        """Build a stock Panda while applying optional suite overrides."""
        values = {
            "uid": "benchmark_franka_panda",
            "robot_type": "panda",
            **dict(self.spec.config),
        }
        return FrankaPandaCfg.from_dict(values)


class FrankaPgiProvider(RobotProvider):
    """Build a Franka arm assembled with the tutorial PGI gripper.

    The configuration mirrors the Franka compatibility in
    ``scripts/tutorials/atomic_action``: an arm-only Panda URDF, the shared
    ``DH_PGI_140_80`` component, a 180-degree base rotation, and the PGI TCP.
    """

    def build_cfg(self) -> RobotCfg:
        """Build the Franka + PGI benchmark configuration."""
        cfg = create_franka_panda_robot_cfg()
        return merge_robot_cfg(
            cfg,
            {
                "uid": "benchmark_franka_pgi",
                **dict(self.spec.config),
            },
        )


register_robot_provider("franka_panda", FrankaPandaProvider)
register_robot_provider("franka_pgi", FrankaPgiProvider)
