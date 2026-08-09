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

"""Scenario provider contract for motion-generation tracks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import Robot

    from ..config import SuiteCfg, TrackCfg
    from ..models import BenchmarkCase

__all__ = ["ScenarioProvider"]


class ScenarioProvider(ABC):
    """Generate fixed cases for one registered scenario kind."""

    required_capabilities: frozenset[str] = frozenset()

    @abstractmethod
    def batch_sizes(self, suite: "SuiteCfg", track: "TrackCfg") -> list[int]:
        """Return simulator batch sizes required by this track."""

    @abstractmethod
    def generate_cases(
        self,
        suite: "SuiteCfg",
        track: "TrackCfg",
        robot: "Robot",
        control_part: str,
        batch_size: int,
    ) -> list["BenchmarkCase"]:
        """Build the frozen case manifest for one batch size."""
