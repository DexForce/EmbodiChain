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

"""Explicit occurrence-bound source jaw Pour through the canonical shared skill."""

from __future__ import annotations
from dataclasses import dataclass, replace
from typing import ClassVar
from embodichain.lab.task_program.compiler import CompiledTaskProgram
from embodichain.lab.task_program.semantics import SceneManifest
from embodichain.lab.sim.atomic_actions import (
    ActionOptions,
    AtomicActionEngine,
    PlanningContext,
)
from embodichain.lab.sim.atomic_actions.primitives.pour import JawPourGoal
from embodichain.lab.task_program.compiler.lowering import (
    RegisteredSemanticLowerer,
    SemanticLowering,
    SemanticObjectTarget,
)
from embodichain.lab.task_program.semantics import (
    BoundSemanticCall,
    RegisteredSemanticCall,
    SceneObjectRef,
    SceneRegistry,
)
from .._configured_services import _PourLowererFactory, _PourTargetBinding
from embodichain.lab.sim.scene.pour_inputs import configured_pour_gravity

__all__ = ["JawPourFactory"]


class _JawPourLowerer(RegisteredSemanticLowerer):
    call_id = _PourLowererFactory.call_id
    target_descriptor = _PourLowererFactory.target_descriptor

    def __init__(
        self,
        delegate: RegisteredSemanticLowerer,
        receivers: tuple[tuple[str, str], ...],
    ) -> None:
        self._delegate = delegate
        self._receivers = dict(receivers)

    def lower(
        self,
        call: RegisteredSemanticCall,
        *,
        context: PlanningContext,
        bound: BoundSemanticCall,
        option_template: ActionOptions,
    ) -> SemanticLowering:
        raise ValueError("Source jaw Pour requires an exact execution occurrence.")

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
        if invocation_id not in self._receivers:
            raise ValueError("Source jaw Pour occurrence has no declared receiver.")
        lowered = self._delegate.lower(
            call, context=context, bound=bound, option_template=option_template
        )
        return replace(
            lowered, goal=JawPourGoal(receiver_entity_id=self._receivers[invocation_id])
        )

    def pick_lookahead_targets(
        self,
        call: RegisteredSemanticCall,
        *,
        picked_object: SceneObjectRef,
        bound: BoundSemanticCall,
        previous_target: SemanticObjectTarget | None,
    ) -> tuple[SemanticObjectTarget, ...] | None:
        return self._delegate.pick_lookahead_targets(
            call,
            picked_object=picked_object,
            bound=bound,
            previous_target=previous_target,
        )


@dataclass(frozen=True, slots=True)
class JawPourFactory(_PourLowererFactory):
    """Declare unchanged source jaw arcs with exact occurrence-local receivers.

    Public Semantic Call identity remains ``simulation.pour``. The default
    shared Pour implementation and mouth-controlled goals remain unchanged.
    """

    revision: ClassVar[str] = "source-jaw-pour-1"
    receivers: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        super(JawPourFactory, self).__post_init__()
        if self.target is not None or not self.receivers:
            raise ValueError("Jaw Pour requires receiver bindings and no mouth target.")
        if type(self.receivers) is not tuple or any(
            type(pair) is not tuple
            or len(pair) != 2
            or any(
                type(value) is not str or not value or value != value.strip()
                for value in pair
            )
            for pair in self.receivers
        ):
            raise ValueError(
                "Jaw Pour receivers require canonical occurrence/entity pairs."
            )
        if len(dict(self.receivers)) != len(self.receivers) or any(
            receiver == self.object_id for _, receiver in self.receivers
        ):
            raise ValueError(
                "Jaw Pour occurrences must be unique and receivers distinct."
            )

    def validate_program(
        self, program: CompiledTaskProgram, scene: SceneManifest
    ) -> None:
        """Reject missing, stale or non-Pour occurrence bindings before simulation.

        Args:
            program: Compiled program with stable serial occurrence identities.
            scene: Declared canonical scene manifest.
        """
        calls = {}
        for segment in program.iter_segments():
            for item in segment.calls:
                call = item.call
                if (
                    isinstance(call, RegisteredSemanticCall)
                    and call.call_id == self.call_id
                ):
                    if dict(call.arguments) != {"object": self.object_id}:
                        raise ValueError(
                            "Jaw Pour call differs from its declared source."
                        )
                    calls[
                        f"{program.program_id}/{segment.segment_id}:{item.segment_call_index}"
                    ] = call
        if set(calls) != set(dict(self.receivers)):
            raise ValueError(
                "Jaw Pour receiver bindings must cover exact Pour occurrences."
            )
        for _, receiver in self.receivers:
            scene.resolve(receiver, expected_type=SceneObjectRef)

    def create(
        self,
        *,
        simulation: object,
        robot: object,
        scene_registry: SceneRegistry,
        engine: AtomicActionEngine,
    ) -> RegisteredSemanticLowerer:
        """Bind an ordinary shared lowerer to declared source jaw-goal receivers.

        Args:
            simulation: Exact Default/B1 physical owner.
            robot: Borrowed robot owning the existing Atomic engine.
            scene_registry: Canonical runtime entities.
            engine: Existing shared Atomic engine.

        Returns:
            Fresh lowerer preserving original source lookahead and call identity.
        """
        configured_pour_gravity(simulation)
        for _, receiver in self.receivers:
            scene_registry.resolve(receiver, expected_type=SceneObjectRef)
        delegate = super(JawPourFactory, self).create(
            simulation=simulation,
            robot=robot,
            scene_registry=scene_registry,
            engine=engine,
        )
        return _JawPourLowerer(delegate, self.receivers)
