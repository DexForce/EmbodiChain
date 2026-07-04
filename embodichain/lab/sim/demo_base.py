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

"""Optional base class for simulation demos."""

from __future__ import annotations

import argparse
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from embodichain.lab.sim.utility.demo_utils import shutdown_sim

if TYPE_CHECKING:
    from embodichain.lab.sim import SimulationManager


__all__ = ["DemoBase"]


class DemoBase(ABC):
    """Lightweight lifecycle base class for simulation demos.

    Subclasses implement :meth:`setup` and :meth:`run`; the base class handles
    argument injection and guaranteed cleanup.

    Args:
        args: Parsed command-line arguments.
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.sim: SimulationManager | None = None

    @abstractmethod
    def setup(self) -> None:
        """Create simulation, robot, cameras, etc."""

    @abstractmethod
    def run(self) -> None:
        """Execute the demo logic."""

    def cleanup(self) -> None:
        """Release simulation resources. Called automatically by :meth:`main`."""
        if self.sim is not None:
            shutdown_sim(self.sim)

    def main(self) -> None:
        """Run the full demo lifecycle."""
        self.setup()
        try:
            self.run()
        finally:
            self.cleanup()
