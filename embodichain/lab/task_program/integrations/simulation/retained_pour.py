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

"""Retained-contact Pour lowering through the canonical shared skill."""

from __future__ import annotations
from dataclasses import dataclass, replace
from typing import ClassVar
from embodichain.lab.sim.atomic_actions import (
    ActionOptions,
    AtomicActionEngine,
    PlanningContext,
)
from embodichain.lab.task_program.compiler.lowering import (
    RegisteredHeldObjectEffect,
    RegisteredSemanticEffect,
    RegisteredSemanticLowerer,
    SemanticLowering,
    SemanticObjectTarget,
)
from embodichain.lab.task_program.semantics import (
    BoundSemanticCall,
    HeldObjectRelation,
    RegisteredSemanticCall,
    SceneObjectRef,
    SceneRegistry,
    SemanticEffectKind,
)
from .._configured_services import _PourLowerer, _PourLowererFactory, _PourTargetBinding
from embodichain.lab.sim.scene.pour_inputs import configured_pour_gravity
from .pour_projection import SimulationPourProjectionProvider

__all__ = ["RetainedPourFactory"]


class _RetainedPourLowerer(_PourLowerer):
    effect_contract_kind: ClassVar[SemanticEffectKind] = SemanticEffectKind.ATTACH

    def __init__(
        self,
        object_id: str,
        projection: SimulationPourProjectionProvider | None = None,
    ) -> None:
        super().__init__(object_id)
        self._projection = projection

    def lower_occurrence(
        self,
        call: RegisteredSemanticCall,
        *,
        context: PlanningContext,
        bound: BoundSemanticCall,
        option_template: ActionOptions,
        invocation_id: str,
        revision: int,
    ) -> SemanticLowering:
        del revision
        lowered = self.lower(
            call, context=context, bound=bound, option_template=option_template
        )
        if self._projection is None:
            return lowered
        goal = self._projection.prepare_pour_goal(
            invocation_id,
            context,
            options=option_template,
            motion_policy=bound.preset.motion_policy,
        )
        if goal is None:
            return lowered
        if goal.source_entity_id != self._object_id:
            raise ValueError(
                "Retained Pour source differs from its declared occurrence."
            )
        return replace(lowered, goal=goal)

    def lower(
        self,
        call: RegisteredSemanticCall,
        *,
        context: PlanningContext,
        bound: BoundSemanticCall,
        option_template: ActionOptions,
    ) -> SemanticLowering:
        lowered = super().lower(
            call, context=context, bound=bound, option_template=option_template
        )
        return replace(
            lowered,
            registered_effect=RegisteredSemanticEffect(
                effect_kind=SemanticEffectKind.ATTACH,
                held_objects=(
                    RegisteredHeldObjectEffect(
                        expectation_id="primary",
                        relation=HeldObjectRelation.ATTACHED,
                        object_id=self._object_id,
                        slot_id="primary",
                        acquire_before_segment="pour",
                        retain_during_segments=("pour",),
                    ),
                ),
            ),
        )

    def pick_lookahead_targets(
        self,
        call: RegisteredSemanticCall,
        *,
        picked_object: SceneObjectRef,
        bound: BoundSemanticCall,
        previous_target: SemanticObjectTarget | None,
    ) -> tuple[SemanticObjectTarget, ...] | None:
        self._validate_call(call)
        return () if picked_object.entity_id == self._object_id else None


@dataclass(frozen=True, slots=True)
class RetainedPourFactory(_PourLowererFactory):
    """Declare measured retention during the canonical Pour phase.

    This opt-in factory retains the source Default/B1 qualification and creates
    neither a planner nor a sensor. The public call ID stays ``simulation.pour``;
    its phase effect is separate from any TaskSpec poured-fluid predicate.
    """

    revision: ClassVar[str] = "retained-pour-1"

    def __post_init__(self) -> None:
        super(RetainedPourFactory, self).__post_init__()
        if self.target is not None:
            raise ValueError("Retained source Pour does not accept an external target.")

    def create(
        self,
        *,
        simulation: object,
        robot: object,
        scene_registry: SceneRegistry,
        engine: AtomicActionEngine,
    ) -> RegisteredSemanticLowerer:
        """Validate the declared source and bind its retained phase effect.

        Args:
            simulation: Exact Default/B1 owner with explicit nonzero gravity.
            robot: Borrowed engine robot.
            scene_registry: Existing canonical source/grasp registry.
            engine: Existing ordinary Atomic engine.

        Returns:
            A fresh canonical Pour lowerer with monitored attachment semantics.
        """
        configured_pour_gravity(simulation)
        super(RetainedPourFactory, self).create(
            simulation=simulation,
            robot=robot,
            scene_registry=scene_registry,
            engine=engine,
        )
        provider = getattr(
            getattr(engine, "planning_services", None),
            "pour_grasp_projection_provider",
            None,
        )
        return _RetainedPourLowerer(
            self.object_id,
            (
                provider
                if isinstance(provider, SimulationPourProjectionProvider)
                else None
            ),
        )
