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

"""Robot-provider contract for planner benchmark suites."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from embodichain.lab.sim import SimulationManager
    from embodichain.lab.sim.cfg import RobotCfg
    from embodichain.lab.sim.objects import Robot

    from ..config import RobotSpecCfg

__all__ = ["RobotProvider"]


class RobotProvider(ABC):
    """Build one benchmark embodiment behind a stable suite identifier."""

    control_part: str = "arm"

    def __init__(self, spec: "RobotSpecCfg") -> None:
        self.spec = spec

    @abstractmethod
    def build_cfg(self) -> "RobotCfg":
        """Build the robot configuration without mutating a simulation."""

    def add_robot(self, simulation: "SimulationManager") -> "Robot":
        """Add the configured robot to a simulation."""
        return simulation.add_robot(cfg=self.build_cfg())
