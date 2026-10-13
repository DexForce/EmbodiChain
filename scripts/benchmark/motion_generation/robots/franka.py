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
from embodichain.lab.sim.robots import FrankaPandaCfg, FrankaPandaPgiCfg

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
    """Build the library Franka + PGI assembly used by the Atomic Skill suites.

    The suites were authored with the base rotated 180 degrees, as in the
    atomic-action tutorial scenes, so the provider keeps that rotation; suite
    overrides apply on top.
    """

    def build_cfg(self) -> RobotCfg:
        """Build the Franka + PGI benchmark configuration."""
        return FrankaPandaPgiCfg.from_dict(
            {
                "uid": "benchmark_franka_pgi",
                "init_rot": [0.0, 0.0, 180.0],
                **dict(self.spec.config),
            }
        )


register_robot_provider("franka_panda", FrankaPandaProvider)
register_robot_provider("franka_pgi", FrankaPgiProvider)
