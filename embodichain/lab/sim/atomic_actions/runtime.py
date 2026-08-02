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

"""Engine-owned planning resources shared by atomic actions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .core import resolve_runtime_device
from .trajectory import TrajectoryBuilder

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import Robot
    from embodichain.lab.sim.planners import MotionGenerator


class ActionPlanningServices:
    """Planning resources exclusively owned by one atomic-action engine.

    An action may borrow these resources after the engine binds it, but callers
    never pass a motion generator to individual actions. Keeping the generator
    and trajectory builder here gives one engine a single planner backend,
    robot, device, cache, and collision-world owner.

    Args:
        motion_generator: Motion generator owned by the engine.
    """

    def __init__(self, motion_generator: MotionGenerator) -> None:
        self._motion_generator = motion_generator
        self._robot: Robot = motion_generator.robot
        self._device = resolve_runtime_device(motion_generator.device)
        self._trajectory_builder = TrajectoryBuilder(motion_generator)

    @property
    def motion_generator(self) -> MotionGenerator:
        """Return the single motion generator owned by the engine."""
        return self._motion_generator

    @property
    def robot(self) -> Robot:
        """Return the robot planned by this service set."""
        return self._robot

    @property
    def device(self) -> torch.device:
        """Return the concrete device used for planning."""
        return self._device

    @property
    def trajectory_builder(self) -> TrajectoryBuilder:
        """Return the shared stateless trajectory builder."""
        return self._trajectory_builder

    @property
    def planner_name(self) -> str:
        """Return the configured planner backend name."""
        planner_cfg = getattr(
            getattr(self._motion_generator, "planner", None), "cfg", None
        )
        planner_name = getattr(planner_cfg, "planner_type", None)
        return "unknown" if planner_name is None else str(planner_name)


__all__ = ["ActionPlanningServices"]
