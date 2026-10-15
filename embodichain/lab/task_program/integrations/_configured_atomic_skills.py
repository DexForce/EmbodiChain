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

"""Allowlisted typed lowerers for the remaining configured Atomic Skills."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import ClassVar

import torch

from embodichain.lab.sim.atomic_actions import (
    ActionOptions,
    AtomicActionEngine,
    AxisAlign,
    AxisAlignAffordance,
    AxisAlignGoal,
    AxisAlignOptions,
    JointPositionGoal,
    MoveJoints,
    MoveJointsOptions,
    ObjectSemantics,
    OpenDoor,
    OpenDoorAffordance,
    OpenDoorGoal,
    OpenDoorOptions,
    PlanningContext,
    Press,
    PressAffordance,
    PressGoal,
    PressOptions,
    SkillDescriptor,
    Twist,
    TwistAffordance,
    TwistGoal,
    TwistOptions,
    sample_initial_articulation_geometry,
)
from embodichain.lab.task_program.compiler.lowering import (
    RegisteredSemanticLowerer,
    SemanticLowering,
)
from embodichain.lab.task_program.semantics import (
    BoundSemanticCall,
    GRASP_AFFORDANCE_CAPABILITY,
    RegisteredSemanticCall,
    SceneArticulationRef,
    SceneLinkRef,
    SceneObjectRef,
    SceneRegistry,
)
from .extensions import RegisteredSemanticLowererFactory

__all__: list[str] = []


def _identifier(value: object) -> str:
    """Validate one immutable configured scene identifier."""
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(
            "Configured entity IDs must be nonempty strings without outer whitespace."
        )
    return value


class _LinkSkillLowerer(RegisteredSemanticLowerer):
    """Ground a declared link against the current planning observation."""

    options_type: ClassVar[type[ActionOptions]]

    def __init__(self, semantics: ObjectSemantics) -> None:
        self._semantics = semantics

    def lower(
        self,
        call: RegisteredSemanticCall,
        *,
        context: PlanningContext,
        bound: BoundSemanticCall,
        option_template: ActionOptions,
    ) -> SemanticLowering:
        """Ground a typed goal while retaining preset-owned options.

        Args:
            call: Registered call with its closed declarative arguments.
            context: Current batch observation used for goal grounding.
            bound: Catalog-owned resource and preset binding.
            option_template: Owned option template from that preset.

        Returns:
            Typed goal without an action-option override.
        """
        del bound
        if type(option_template) is not self.options_type:
            raise TypeError(
                f"{self.call_id} requires exact {self.options_type.__name__} options."
            )
        arguments = dict(call.arguments)
        # Resolve from this call's observation, then keep the contact path fixed
        # while the skill intentionally moves the target articulation link.
        target_pose = context.scene.entities[self._semantics.entity_id].pose.clone()
        if self.call_id == "simulation.open_door":
            if (
                set(arguments) != {"handle", "open_angle"}
                or arguments["handle"] != self._semantics.entity_id
            ):
                raise ValueError(
                    "OpenDoor arguments must select the configured handle and open_angle."
                )
            angle = arguments["open_angle"]
            if type(angle) not in (int, float) or not math.isfinite(angle):
                raise ValueError("open_angle must be a finite real scalar in radians.")
            affordance = self._semantics.affordance
            assert isinstance(affordance, OpenDoorAffordance)
            lower, upper = affordance.joint_limits
            closed, opened = (
                (lower, upper) if affordance.opening_direction > 0 else (upper, lower)
            )
            fraction = (angle - closed) / (opened - closed)
            if not 0.0 <= fraction <= 1.0:
                raise ValueError("open_angle must lie within the hinge's legal limits.")
            return SemanticLowering(
                goal=OpenDoorGoal(
                    self._semantics,
                    target_pose,
                    fraction,
                )
            )
        if arguments != {"target": self._semantics.entity_id}:
            raise ValueError(
                f"{self.call_id} arguments must select only the configured target."
            )
        goal_type = PressGoal if self.call_id == "simulation.press" else TwistGoal
        return SemanticLowering(
            goal=goal_type(
                self._semantics,
                target_pose,
            )
        )


class _PressLowerer(_LinkSkillLowerer):
    call_id = "simulation.press"
    target_descriptor = Press.descriptor()
    options_type = PressOptions


class _TwistLowerer(_LinkSkillLowerer):
    call_id = "simulation.twist"
    target_descriptor = Twist.descriptor()
    options_type = TwistOptions


class _OpenDoorLowerer(_LinkSkillLowerer):
    call_id = "simulation.open_door"
    target_descriptor = OpenDoor.descriptor()
    options_type = OpenDoorOptions


@dataclass(frozen=True, slots=True)
class _LinkSkillLowererFactory(RegisteredSemanticLowererFactory):
    """Resolve link geometry without introducing task-specific scene code."""

    articulation_id: str
    articulation_simulation_uid: str
    link_entity_id: str
    lowerer_type: ClassVar[type[_LinkSkillLowerer]]

    def __post_init__(self) -> None:
        for value in (
            self.articulation_id,
            self.articulation_simulation_uid,
            self.link_entity_id,
        ):
            _identifier(value)

    def create(
        self,
        *,
        simulation: object,
        robot: object,
        scene_registry: SceneRegistry,
        engine: AtomicActionEngine,
    ) -> RegisteredSemanticLowerer:
        """Create a fresh lowerer against the selected live scene.

        Args:
            simulation: Host that owns the declared articulation.
            robot: Robot selected by the configured embodiment.
            scene_registry: Validated logical-to-native scene registrations.
            engine: Exact Atomic engine receiving the lowered goal.

        Returns:
            Fresh lowerer with geometry owned by this assembly.
        """
        if engine.robot is not robot:
            raise ValueError("Configured lowerers require the engine's exact robot.")
        registration = scene_registry.lookup(
            self.link_entity_id, expected_type=SceneLinkRef
        )
        if registration.parent != SceneArticulationRef(self.articulation_id):
            raise ValueError("Configured link belongs to a different articulation.")
        articulation = simulation.get_articulation(self.articulation_simulation_uid)
        if articulation is None:
            raise ValueError("Configured articulation is absent from the simulation.")
        native_name = registration.native_name
        if native_name not in articulation.link_names:
            raise ValueError(f"Configured articulation has no link {native_name!r}.")
        geometry = {}
        if self.call_id == "simulation.open_door":
            affordance = OpenDoorAffordance.from_articulation(articulation, native_name)
        else:
            geometry = sample_initial_articulation_geometry(
                articulation,
                native_name,
                initial_qpos=articulation.cfg.init_qpos,
                initial_qpos_joint_names=articulation.joint_names,
                body_scale=articulation.cfg.body_scale,
            ).to_object_geometry()
            if self.call_id == "simulation.press":
                affordance = PressAffordance()
            else:
                vertices, _ = articulation.get_link_vert_face(native_name)
                center = torch.as_tensor(vertices).mean(dim=0).tolist()
                affordance = TwistAffordance(
                    grasp_position=tuple(center),
                    grasp_roll_range=(-math.pi, math.pi),
                )
        return self.lowerer_type(
            ObjectSemantics(
                entity_id=self.link_entity_id,
                label=registration.semantic_type or "articulation_link",
                geometry=geometry,
                affordance=affordance,
            )
        )


@dataclass(frozen=True, slots=True)
class _PressLowererFactory(_LinkSkillLowererFactory):
    call_id: ClassVar[str] = "simulation.press"
    revision: ClassVar[str] = "1"
    target_descriptor: ClassVar[SkillDescriptor] = Press.descriptor()
    lowerer_type = _PressLowerer


@dataclass(frozen=True, slots=True)
class _TwistLowererFactory(_LinkSkillLowererFactory):
    call_id: ClassVar[str] = "simulation.twist"
    revision: ClassVar[str] = "1"
    target_descriptor: ClassVar[SkillDescriptor] = Twist.descriptor()
    lowerer_type = _TwistLowerer


@dataclass(frozen=True, slots=True)
class _OpenDoorLowererFactory(_LinkSkillLowererFactory):
    call_id: ClassVar[str] = "simulation.open_door"
    revision: ClassVar[str] = "1"
    target_descriptor: ClassVar[SkillDescriptor] = OpenDoor.descriptor()
    lowerer_type = _OpenDoorLowerer


class _AxisAlignLowerer(RegisteredSemanticLowerer):
    call_id = "simulation.axis_align"
    target_descriptor = AxisAlign.descriptor()

    def __init__(self, semantics: ObjectSemantics) -> None:
        self._semantics = semantics

    def lower(
        self,
        call: RegisteredSemanticCall,
        *,
        context: PlanningContext,
        bound: BoundSemanticCall,
        option_template: ActionOptions,
    ) -> SemanticLowering:
        """Ground a typed goal while retaining preset-owned options.

        Args:
            call: Registered call with its closed declarative arguments.
            context: Current batch observation used for goal grounding.
            bound: Catalog-owned resource and preset binding.
            option_template: Owned option template from that preset.

        Returns:
            Typed goal without an action-option override.
        """
        del context, bound
        if type(option_template) is not AxisAlignOptions:
            raise TypeError("AxisAlign requires exact AxisAlignOptions.")
        if dict(call.arguments) != {"object": self._semantics.entity_id}:
            raise ValueError(
                "AxisAlign arguments must select only the configured object."
            )
        return SemanticLowering(goal=AxisAlignGoal(self._semantics))


@dataclass(frozen=True, slots=True)
class _AxisAlignLowererFactory(RegisteredSemanticLowererFactory):
    call_id: ClassVar[str] = "simulation.axis_align"
    revision: ClassVar[str] = "1"
    target_descriptor: ClassVar[SkillDescriptor] = AxisAlign.descriptor()
    object_id: str

    def __post_init__(self) -> None:
        _identifier(self.object_id)

    def create(
        self,
        *,
        simulation: object,
        robot: object,
        scene_registry: SceneRegistry,
        engine: AtomicActionEngine,
    ) -> RegisteredSemanticLowerer:
        """Create a fresh lowerer for the selected embodiment.

        Args:
            simulation: Host simulation used by the adapter.
            robot: Robot selected by the configured embodiment.
            scene_registry: Validated logical scene registrations.
            engine: Exact Atomic engine receiving the lowered goal.

        Returns:
            Fresh typed lowerer with no shared live state.
        """
        del simulation
        if engine.robot is not robot:
            raise ValueError("AxisAlign requires the engine's exact robot.")
        ref = scene_registry.resolve(self.object_id, expected_type=SceneObjectRef)
        grasp = scene_registry.resolve_affordance(
            ref, capability=GRASP_AFFORDANCE_CAPABILITY
        )
        semantics = scene_registry.object_semantics(ref, affordance=grasp)
        if not isinstance(semantics.affordance, AxisAlignAffordance):
            raise TypeError("AxisAlign requires an axis-aware grasp affordance.")
        return _AxisAlignLowerer(semantics)


class _MoveJointsLowerer(RegisteredSemanticLowerer):
    call_id = "simulation.move_joints"
    target_descriptor = MoveJoints.descriptor()

    def lower(
        self,
        call: RegisteredSemanticCall,
        *,
        context: PlanningContext,
        bound: BoundSemanticCall,
        option_template: ActionOptions,
    ) -> SemanticLowering:
        """Ground a typed goal while retaining preset-owned options.

        Args:
            call: Registered call with its closed declarative arguments.
            context: Current batch observation used for goal grounding.
            bound: Catalog-owned resource and preset binding.
            option_template: Owned option template from that preset.

        Returns:
            Typed goal without an action-option override.
        """
        del bound
        if type(option_template) is not MoveJointsOptions:
            raise TypeError("MoveJoints requires exact MoveJointsOptions.")
        arguments = dict(call.arguments)
        if set(arguments) != {"qpos"}:
            raise ValueError("MoveJoints arguments must contain one qpos target.")
        values = arguments["qpos"]
        if (
            not isinstance(values, (list, tuple))
            or not values
            or any(
                type(value) not in (int, float) or not math.isfinite(value)
                for value in values
            )
        ):
            raise ValueError("qpos must contain finite real joint coordinates.")
        return SemanticLowering(
            goal=JointPositionGoal(
                torch.tensor(
                    values,
                    device=context.robot.qpos.device,
                    dtype=context.robot.qpos.dtype,
                )
            )
        )


@dataclass(frozen=True, slots=True)
class _MoveJointsLowererFactory(RegisteredSemanticLowererFactory):
    call_id: ClassVar[str] = "simulation.move_joints"
    revision: ClassVar[str] = "1"
    target_descriptor: ClassVar[SkillDescriptor] = MoveJoints.descriptor()

    def create(
        self,
        *,
        simulation: object,
        robot: object,
        scene_registry: SceneRegistry,
        engine: AtomicActionEngine,
    ) -> RegisteredSemanticLowerer:
        """Create a fresh lowerer for the selected embodiment.

        Args:
            simulation: Host simulation used by the adapter.
            robot: Robot selected by the configured embodiment.
            scene_registry: Validated logical scene registrations.
            engine: Exact Atomic engine receiving the lowered goal.

        Returns:
            Fresh typed lowerer with no shared live state.
        """
        del simulation, scene_registry
        if engine.robot is not robot:
            raise ValueError("MoveJoints requires the engine's exact robot.")
        return _MoveJointsLowerer()
