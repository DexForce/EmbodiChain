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

"""Typed physical drawer planning inputs for canonical Atomic Skills."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable
from embodichain.lab.sim.scene.drawer_observation import DrawerPayloadObservation

if TYPE_CHECKING:
    from .planning_probes import RetainedCommandAuthority
    from embodichain.lab.sim.evidence import RobotCommandObservation
    from .invocation import ResolvedActionRequest
    from .state import PlanningContext
    from embodichain.lab.sim.objects import Robot
    from embodichain.lab.sim.motion.mesh_collision import (
        RobotArticulationCollisionSnapshot,
    )

__all__ = ["DrawerPlanningInputs", "DrawerPlanningProvider"]


@dataclass(frozen=True, slots=True)
class DrawerPlanningInputs:
    """Explicit target binding and borrowed live physical payload readers."""

    target_id: str
    link_entity_id: str
    observation: DrawerPayloadObservation
    other_payloads: tuple[DrawerPayloadObservation, ...] = ()


@runtime_checkable
class DrawerPlanningProvider(Protocol):
    """Supply declared occurrence geometry without choosing an action algorithm."""

    @property
    def robot(self) -> Robot:
        """Return the exact borrowed engine robot."""
        ...

    def inputs(
        self, request: ResolvedActionRequest, context: PlanningContext
    ) -> tuple[DrawerPlanningInputs, ...]:
        """Resolve explicit occurrence bindings and validate their native lifetime."""
        ...

    def prepare_articulation_query(
        self, request: ResolvedActionRequest, context: PlanningContext
    ) -> RobotArticulationCollisionSnapshot:
        """Capture one owned robot/articulation-only planning query."""
        ...
