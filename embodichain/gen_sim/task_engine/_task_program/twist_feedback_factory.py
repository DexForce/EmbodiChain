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
"""Immutable declaration borrowing the existing E8 action-owned controller."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from .twist_binding import TwistRoute

__all__: list[str] = []


@dataclass(frozen=True, slots=True)
class TwistFeedbackFactory:
    route: TwistRoute
    controller_id: ClassVar[str] = "gen_sim.twist.precontact"
    revision: ClassVar[str] = "1"
    supported_skill_ids: ClassVar[tuple[str, ...]] = ("twist",)

    def create(
        self, *, simulation: Any, robot: Any, scene_registry: Any, engine: Any
    ) -> Any:
        from .twist_feedback import TwistFeedbackState
        from .twist_runtime import GenSimTwist

        action = engine.actions.get("twist")
        if not isinstance(action, GenSimTwist) or engine.robot is not robot:
            raise ValueError("E8 feedback requires the exact registered action/robot.")
        state = action._feedback_state
        if (
            not isinstance(state, TwistFeedbackState)
            or state.route != self.route
            or state.robot is not robot
            or state.simulation is not simulation
        ):
            raise ValueError("E8 feedback runtime ownership differs from registration.")
        return state
