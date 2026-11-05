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

"""Declared stack descent targets using the canonical shared Place skill."""

from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Any, ClassVar
import math
import torch
from embodichain.lab.sim.atomic_actions.goals import resolve_pose_goal
from embodichain.lab.sim.atomic_actions.primitives.place import PlaceGoal
from embodichain.lab.sim.atomic_actions import AtomicActionEngine
from embodichain.lab.task_program.compiler.lowering import (
    SemanticLowering,
    RegisteredSemanticLowerer,
)
from embodichain.lab.task_program.semantics import RegisteredSemanticCall, SceneRegistry
from .placement import (
    ObservedRelativePlaceLowererFactory,
    ObservedRelativePlaceBinding,
    _TerminalReleasePlaceLowerer,
)

__all__ = ["StackPlaceLowererFactory"]
STACK_PLACE_CALL = "gen_sim.stack_place"


class _StackPlaceLowerer(_TerminalReleasePlaceLowerer):
    """Reuse the original goal/effect binding without overriding action options."""

    call_id: ClassVar[str] = STACK_PLACE_CALL

    def __init__(
        self,
        routes: tuple[Any, ...],
        *,
        approach: float,
        release: float,
        nominal: float,
    ) -> None:
        super().__init__(routes)
        self._approach = approach
        self._release = release
        self._nominal = nominal

    def lower(
        self,
        call: RegisteredSemanticCall,
        *,
        context: Any,
        bound: Any,
        option_template: Any,
    ) -> SemanticLowering:
        result = super().lower(
            call, context=context, bound=bound, option_template=option_template
        )
        # Freeze this observed, stable support for a precision descent. The
        # final task policy still checks the actually observed support relation.
        release = resolve_pose_goal(
            result.goal.xpos, context, name="stack_release"
        ).clone()
        if release.ndim == 2:
            release = release.unsqueeze(0).expand(context.batch_size, -1, -1).clone()
        approach = release.clone()
        approach[:, 2, 3] += self._approach - self._nominal
        release[:, 2, 3] += self._release - self._nominal
        return replace(
            result,
            goal=PlaceGoal(
                xpos=torch.stack((approach, release), dim=1),
                tcp_symmetry="z_roll_180",
            ),
        )

    def pick_lookahead_targets(
        self, call: RegisteredSemanticCall, **kwargs: Any
    ) -> Any:
        targets = super().pick_lookahead_targets(call, **kwargs)
        if targets is None:
            return None
        return tuple(
            replace(
                target,
                pose=replace(
                    target.pose,
                    world_displacement=(
                        target.pose.world_displacement
                        + target.pose.world_displacement.new_tensor(
                            [0.0, 0.0, self._release - self._nominal]
                        )
                    ),
                ),
            )
            for target in targets
        )


@dataclass(frozen=True, slots=True)
class StackPlaceLowererFactory(ObservedRelativePlaceLowererFactory):
    """Bind source-qualified support routes to exact approach/release targets.

    Route validation reuses the shared static reference checks; runtime lowering
    intentionally retains the verified attachment transform, as in the source
    precision-descent path. This does not opt into observed-grasp rebasing.
    """

    call_id: ClassVar[str] = STACK_PLACE_CALL
    revision: ClassVar[str] = "terminal-4"
    lowerer_type: ClassVar[type[_TerminalReleasePlaceLowerer]] = _StackPlaceLowerer
    approach_clearance: float = 0.02
    release_clearance: float = 0.003
    nominal_clearance: float = 0.01

    def __post_init__(self) -> None:
        ObservedRelativePlaceLowererFactory.__post_init__(self)
        if any(
            type(v) not in (int, float) or not math.isfinite(v) or v <= 0
            for v in (
                self.approach_clearance,
                self.release_clearance,
                self.nominal_clearance,
            )
        ):
            raise ValueError("Stack clearances must be finite positive lengths.")
        if self.release_clearance >= self.approach_clearance:
            raise ValueError("Stack approach must be above its release target.")

    def create(
        self,
        *,
        simulation: object,
        robot: object,
        scene_registry: SceneRegistry,
        engine: AtomicActionEngine,
    ) -> RegisteredSemanticLowerer:
        """Create one precision-descent lowerer from declared shared routes.

        Args:
            simulation: Borrowed owner; unused by pure target binding.
            robot: Exact engine robot.
            scene_registry: Authoritative source/support registry.
            engine: Existing canonical Atomic engine.

        Returns:
            Fresh lowerer preserving verified-grasp target and release effect.
        """
        if engine.robot is not robot:
            raise ValueError("Stack Place requires the engine's exact robot.")
        return _StackPlaceLowerer(
            self._canonical_routes(scene_registry),
            approach=self.approach_clearance,
            release=self.release_clearance,
            nominal=self.nominal_clearance,
        )
