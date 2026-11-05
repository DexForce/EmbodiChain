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

"""Explicit retained-object alignment lowering into the canonical shared skill."""

from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar
from collections.abc import Sequence
import math
from embodichain.compute.geometry.alignment import held_alignment_targets
from embodichain.lab.sim.atomic_actions import (
    AtomicActionEngine,
    MoveHeldObject,
    MoveHeldObjectOptions,
    HeldObjectPoseGoal,
    PlanningContext,
    ActionOptions,
    SkillDescriptor,
)
from embodichain.lab.task_program.compiler.lowering import (
    RegisteredSemanticLowerer,
    RegisteredSemanticEffect,
    RegisteredHeldObjectEffect,
    SemanticLowering,
)
from embodichain.lab.task_program.semantics import (
    BoundSemanticCall,
    RegisteredSemanticCall,
    SceneObjectRef,
    SceneManifest,
    SceneRegistry,
    SemanticEffectKind,
    HeldObjectRelation,
)
from ..extensions import RegisteredSemanticLowererFactory
from .targets import _identifier

__all__ = ["HeldAlignmentRoute", "HeldAlignmentLowererFactory"]


@dataclass(frozen=True, slots=True)
class HeldAlignmentRoute:
    """Exact source alignment operands, separate from skill and motion policies."""

    object_id: str
    target_id: str
    preserve_yaw: bool
    axis_direction_free: bool
    local_axis: tuple[float, float, float]
    position: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        for name in ("object_id", "target_id"):
            _identifier(getattr(self, name), field_name=name)
        if (
            type(self.preserve_yaw) is not bool
            or type(self.axis_direction_free) is not bool
        ):
            raise TypeError("Alignment yaw/axis choices must be explicit booleans.")
        for name in ("local_axis", "position"):
            value = getattr(self, name)
            if name == "position" and value is None:
                continue
            if (
                not isinstance(value, (tuple, list))
                or len(value) != 3
                or any(
                    type(v) not in (float, int) or not math.isfinite(v) for v in value
                )
            ):
                raise ValueError(f"{name} must contain three finite values.")
            object.__setattr__(self, name, tuple(value))
        if not any(self.local_axis):
            raise ValueError("Alignment axis must be nonzero.")
        if (self.target_id == "current_object_pose") != (self.position is None):
            raise ValueError(
                "Alignment current-pose selector and explicit destination disagree."
            )

    @property
    def selector(self) -> tuple[str, str, bool, bool]:
        """Return the exact declaration selection key."""
        return (
            self.object_id,
            self.target_id,
            self.preserve_yaw,
            self.axis_direction_free,
        )


class _HeldAlignmentLowerer(RegisteredSemanticLowerer):
    call_id: ClassVar[str] = "gen_sim.align_held"
    target_descriptor: ClassVar[SkillDescriptor] = MoveHeldObject.descriptor()
    preserves_symbolic_state: ClassVar[bool] = True

    def __init__(self, routes: tuple[HeldAlignmentRoute, ...], robot: object) -> None:
        self._routes = {route.selector: route for route in routes}
        self._robot = robot

    def lower(
        self,
        call: RegisteredSemanticCall,
        *,
        context: PlanningContext,
        bound: BoundSemanticCall,
        option_template: ActionOptions,
    ) -> SemanticLowering:
        if type(option_template) is not MoveHeldObjectOptions:
            raise TypeError("Held alignment requires MoveHeldObjectOptions.")
        args = dict(call.arguments)
        if (
            set(args) - {"axis_direction_free"} != {"object", "target", "preserve_yaw"}
            or type(args["preserve_yaw"]) is not bool
            or type(args.get("axis_direction_free", False)) is not bool
        ):
            raise ValueError(
                "Held alignment requires references and a declared yaw policy."
            )
        route = self._routes[
            (
                args["object"],
                args["target"],
                args["preserve_yaw"],
                args.get("axis_direction_free", False),
            )
        ]
        binding = bound.binding.action_binding
        motion, grasp = (
            binding.endpoint("primary", name) for name in ("motion", "grasp")
        )
        key = motion.task_state_key
        if type(key) is not str or not key or grasp.task_state_key != key:
            raise ValueError("Held alignment requires one exact shared resource.")
        held = context.task.get_held_object(key)
        if held is None or not context.task.exclusive_held_object_mask(key).all():
            raise ValueError("Held alignment requires an exclusive held object.")
        if held.semantics.entity_id != route.object_id:
            raise ValueError(
                "Held alignment object does not match the verified attachment."
            )
        observed = context.scene.entities[route.object_id]
        if observed.confidence <= 0:
            raise ValueError("Held alignment requires a current object observation.")
        pose = observed.pose.clone()
        if pose.shape == (4, 4):
            pose = pose.unsqueeze(0).expand(context.batch_size, -1, -1).clone()
        root = None
        if not route.preserve_yaw:
            part = (
                bound.binding.resources["primary"]
                .endpoints["motion"]
                .runtime_target.control_part
            )
            root_name = self._robot.cfg.solver_cfg[part].root_link_name
            root = self._robot.get_link_pose(
                link_name=root_name, env_ids=context.env_ids.tolist(), to_matrix=True
            ).to(pose)
        target = held_alignment_targets(
            pose,
            route.local_axis,
            position=route.position,
            direction_free=route.axis_direction_free,
            preserve_yaw=route.preserve_yaw,
            object_vertices=held.semantics.affordance.mesh_vertices,
            root_pose=root,
            object_to_eef=held.object_to_eef,
        )
        effect = (
            RegisteredSemanticEffect(
                effect_kind=SemanticEffectKind.ATTACH,
                held_objects=(
                    RegisteredHeldObjectEffect(
                        expectation_id="primary",
                        relation=HeldObjectRelation.ATTACHED,
                        object_id=route.object_id,
                        slot_id="primary",
                    ),
                ),
            )
            if self.effect_contract_kind is SemanticEffectKind.ATTACH
            else None
        )
        return SemanticLowering(
            goal=HeldObjectPoseGoal(target, world_yaw_free=not route.preserve_yaw),
            registered_effect=effect,
        )


class _RetainedHeldAlignmentLowerer(_HeldAlignmentLowerer):
    effect_contract_kind: ClassVar[SemanticEffectKind] = SemanticEffectKind.ATTACH
    preserves_symbolic_state: ClassVar[bool] = False


@dataclass(frozen=True, slots=True)
class HeldAlignmentLowererFactory(RegisteredSemanticLowererFactory):
    """Expose the stable alignment alias using explicit source-qualified operands."""

    call_id: ClassVar[str] = "gen_sim.align_held"
    revision: ClassVar[str] = "12"
    target_descriptor: ClassVar[SkillDescriptor] = MoveHeldObject.descriptor()
    routes: tuple[HeldAlignmentRoute, ...]
    verify_retention: bool = False

    def __post_init__(self) -> None:
        if type(self.verify_retention) is not bool:
            raise TypeError("verify_retention must be bool.")
        if (
            type(self.routes) is not tuple
            or not self.routes
            or any(type(route) is not HeldAlignmentRoute for route in self.routes)
        ):
            raise TypeError("Alignment requires explicit immutable typed routes.")
        if len({route.selector for route in self.routes}) != len(self.routes):
            raise ValueError("Alignment routes must be unique.")

    def validate_scene_declaration(self, scene: SceneManifest) -> None:
        """Validate exact object references before runtime assembly.

        Args:
            scene: Immutable canonical declaration.
        """
        for route in self.routes:
            scene.resolve(route.object_id, expected_type=SceneObjectRef)

    def create(
        self,
        *,
        simulation: object,
        robot: object,
        scene_registry: SceneRegistry,
        engine: AtomicActionEngine,
    ) -> RegisteredSemanticLowerer:
        """Bind already declared alignment operands without planning motion.

        Args:
            simulation: Borrowed physical owner; no native reads occur here.
            robot: Exact engine robot for active-root observation during lowering.
            scene_registry: Existing authoritative object registry.
            engine: Existing ordinary Atomic engine.

        Returns:
            A fresh lowerer with the explicit retention effect contract.
        """
        if engine.robot is not robot:
            raise ValueError("Held alignment must bind the factory's robot.")
        self.validate_scene_declaration(scene_registry)
        return (
            _RetainedHeldAlignmentLowerer
            if self.verify_retention
            else _HeldAlignmentLowerer
        )(self.routes, robot)
