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

"""Configured typed control calls; all geometry remains runtime-owned."""

from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar
from embodichain.lab.sim.atomic_actions import (
    ActionOptions,
    AtomicActionEngine,
    ObjectSemantics,
    PlanningContext,
    SceneEntityPose,
    SkillDescriptor,
    Press,
    PressGoal,
    PressOptions,
    Twist,
    TwistGoal,
    TwistOptions,
    OpenDoor,
    OpenDoorGoal,
    OpenDoorOptions,
)
from embodichain.lab.sim.atomic_actions.affordance import (
    PressAffordance,
    TwistAffordance,
    OpenDoorAffordance,
)
from embodichain.lab.task_program.compiler.lowering import (
    RegisteredSemanticLowerer,
    SemanticLowering,
)
from embodichain.lab.task_program.semantics import (
    BoundSemanticCall,
    RegisteredSemanticCall,
    SceneRegistry,
)
from embodichain.lab.task_program.language.semantic_contracts import (
    validate_semantic_call_payload,
)
from .extensions import RegisteredSemanticLowererFactory

__all__: list[str] = []
_TYPES = {
    "simulation.press": (Press, PressGoal, PressOptions, PressAffordance),
    "simulation.twist": (Twist, TwistGoal, TwistOptions, TwistAffordance),
    "simulation.open_door": (
        OpenDoor,
        OpenDoorGoal,
        OpenDoorOptions,
        OpenDoorAffordance,
    ),
}


class _ControlLowerer(RegisteredSemanticLowerer):
    preserves_symbolic_state: ClassVar[bool] = True

    def __init__(self, call_id: str, semantics: dict[str, ObjectSemantics]) -> None:
        self.call_id = call_id
        self.target_descriptor = _TYPES[call_id][0].descriptor()
        self._semantics = semantics

    def lower(
        self,
        call: RegisteredSemanticCall,
        *,
        context: PlanningContext,
        bound: BoundSemanticCall,
        option_template: ActionOptions,
    ) -> SemanticLowering:
        resource = bound.binding.resources.get("primary")
        if resource is None or "motion" not in resource.endpoints:
            raise ValueError("Control call requires a bound primary motion endpoint")
        state_key = resource.endpoints["motion"].task_state_key
        if context.task.get_held_object(state_key) is not None:
            raise ValueError("Control call requires the selected hand to be free")
        if call.semantic_id != self.call_id:
            raise ValueError("Control call identity differs from its lowerer")
        validate_semantic_call_payload(
            {
                "kind": "registered",
                "call_id": self.call_id,
                "arguments": dict(call.arguments),
                "resources": dict(call.resources),
            }
        )
        _, goal_type, options_type, _ = _TYPES[self.call_id]
        if type(option_template) is not options_type:
            raise TypeError("Control call has incompatible options")
        object_id = call.arguments["object"]
        if object_id not in self._semantics:
            raise ValueError(f"Control object {object_id!r} is not configured")
        kwargs = dict(
            semantics=self._semantics[object_id], target_pose=SceneEntityPose(object_id)
        )
        if self.call_id == "simulation.open_door":
            kwargs["open_fraction"] = 1.0 if call.arguments["state"] == "open" else 0.0
        return SemanticLowering(goal=goal_type(**kwargs))


@dataclass(frozen=True, slots=True)
class _ControlLowererFactory(RegisteredSemanticLowererFactory):
    object_affordances: tuple[tuple[str, str], ...]
    call_id: ClassVar[str]
    revision: ClassVar[str] = "1"

    def __post_init__(self) -> None:
        if type(self.object_affordances) is not tuple or not self.object_affordances:
            raise ValueError(
                "Control calls require immutable object/affordance bindings"
            )
        seen = set()
        for pair in self.object_affordances:
            if (
                type(pair) is not tuple
                or len(pair) != 2
                or any(not isinstance(v, str) or not v.strip() for v in pair)
                or pair[0] in seen
            ):
                raise ValueError(
                    "Control object/affordance bindings must be unique nonempty pairs"
                )
            seen.add(pair[0])

    def create(
        self,
        *,
        simulation: object,
        robot: object,
        scene_registry: SceneRegistry,
        engine: AtomicActionEngine,
    ) -> RegisteredSemanticLowerer:
        del simulation
        if engine.robot is not robot:
            raise ValueError("Control factory requires the engine's exact robot")
        semantics = {}
        for object_id, affordance_id in self.object_affordances:
            value = scene_registry.object_semantics(object_id, affordance=affordance_id)
            if type(value.affordance) is not _TYPES[self.call_id][3]:
                raise TypeError("Control affordance does not match the Atomic Skill")
            semantics[object_id] = value
        return _ControlLowerer(self.call_id, semantics)


@dataclass(frozen=True, slots=True)
class _PressLowererFactory(_ControlLowererFactory):
    call_id: ClassVar[str] = "simulation.press"
    target_descriptor: ClassVar[SkillDescriptor] = Press.descriptor()


@dataclass(frozen=True, slots=True)
class _TwistLowererFactory(_ControlLowererFactory):
    call_id: ClassVar[str] = "simulation.twist"
    target_descriptor: ClassVar[SkillDescriptor] = Twist.descriptor()


@dataclass(frozen=True, slots=True)
class _OpenDoorLowererFactory(_ControlLowererFactory):
    call_id: ClassVar[str] = "simulation.open_door"
    target_descriptor: ClassVar[SkillDescriptor] = OpenDoor.descriptor()
