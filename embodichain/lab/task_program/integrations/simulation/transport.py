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

"""Configured object-specific transport routes into the shared MoveHeldObject skill."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch
from embodichain.lab.sim.atomic_skills import (
    SceneEntityPose,
    SkillOptions,
    AtomicSkillEngine,
    MoveHeldObject,
    MoveHeldObjectOptions,
    PlanningContext,
    SkillDescriptor,
)
from embodichain.lab.task_program.semantics import (
    BoundSemanticCall,
    RegisteredSemanticCall,
    SceneObjectRef,
    SceneRegistry,
    SceneManifest,
    SemanticEffectKind,
    SemanticPose,
)
from embodichain.lab.task_program.compiler.lowering import (
    RegisteredSemanticLowerer,
    SemanticLowering,
    SemanticObjectTarget,
    _held_transport_lowering,
)
from embodichain.lab.task_program.integrations.extensions import (
    RegisteredSemanticLowererFactory,
)
from embodichain.lab.sim.atomic_skills.goals import resolve_pose_goal
from .targets import (
    AbsolutePoseTarget,
    SceneEntityTarget,
    resolve_pose_target,
    _identifier,
)

__all__ = ["HeldObjectTransportRoute", "RoutedMoveHeldObjectLowererFactory"]

_MOVE_HELD_OBJECT_CALL_ID = "simulation.move_held_object"


@dataclass(frozen=True, slots=True)
class HeldObjectTransportRoute:
    """One exact declared object target, separate from planning options.

    Args:
        object_id: Canonical carried-object identity.
        target_id: Named target selected by registered call arguments.
        pose: Immutable absolute or scene-relative pose declaration.
        preserve_current_object_orientation: Retain the observed carried-object rotation.
        waypoints: Optional object-pose waypoints visited before the final pose.
    """

    object_id: str
    target_id: str
    pose: AbsolutePoseTarget | SceneEntityTarget
    preserve_current_object_orientation: bool = False
    waypoints: tuple[AbsolutePoseTarget | SceneEntityTarget, ...] = ()

    def __post_init__(self) -> None:
        _identifier(self.object_id, field_name="object_id")
        _identifier(self.target_id, field_name="target_id")
        if type(self.preserve_current_object_orientation) is not bool:
            raise TypeError("preserve_current_object_orientation must be a bool.")
        resolve_pose_target(self.pose)
        object.__setattr__(self, "pose", self.pose.snapshot())
        if type(self.waypoints) is not tuple:
            raise TypeError("waypoints must be an exact tuple.")
        for waypoint in self.waypoints:
            resolve_pose_target(waypoint)
        object.__setattr__(
            self, "waypoints", tuple(value.snapshot() for value in self.waypoints)
        )


class _RoutedMoveHeldObjectLowerer(RegisteredSemanticLowerer):
    """Bind absolute or scene-relative targets to the shared transport goal."""

    call_id: ClassVar[str] = _MOVE_HELD_OBJECT_CALL_ID
    target_descriptor: ClassVar[SkillDescriptor] = MoveHeldObject.descriptor()
    effect_contract_kind: ClassVar[SemanticEffectKind] = SemanticEffectKind.ATTACH

    def __init__(self, routes: tuple[HeldObjectTransportRoute, ...]) -> None:
        self._routes = {route.target_id: route for route in routes}
        if not routes or len(self._routes) != len(routes):
            raise ValueError("MoveHeldObject targets must be non-empty and unique.")

    def _route(self, call: RegisteredSemanticCall) -> HeldObjectTransportRoute:
        arguments = dict(call.arguments)
        from embodichain.lab.task_program.language.semantic_contracts import (
            validate_semantic_call_payload,
        )

        validate_semantic_call_payload(
            {
                "kind": "registered",
                "call_id": self.call_id,
                "arguments": arguments,
                "resources": dict(call.resources),
            }
        )
        route = self._routes.get(arguments["target"])
        if route is None or arguments.get("object", route.object_id) != route.object_id:
            raise ValueError(
                "MoveHeldObject call does not match a configured object-target route."
            )
        if "reference" in arguments and (
            type(route.pose) is not SceneEntityTarget
            or arguments["reference"] != route.pose.entity_id
        ):
            raise ValueError(
                "MoveHeldObject reference does not match the configured target."
            )
        return route

    def lower(
        self,
        call: RegisteredSemanticCall,
        *,
        context: PlanningContext,
        bound: BoundSemanticCall,
        option_template: SkillOptions,
    ) -> SemanticLowering:
        """Bind an exact target without observing the robot or solving IK."""
        del bound
        if type(option_template) is not MoveHeldObjectOptions:
            raise TypeError(
                "Configured transport requires exact MoveHeldObjectOptions."
            )
        route = self._route(call)
        declarations = (*route.waypoints, route.pose)
        targets = [resolve_pose_target(value) for value in declarations]
        if route.preserve_current_object_orientation or route.waypoints:
            targets = [
                resolve_pose_goal(value, context, name="transport target").clone()
                for value in targets
            ]
            targets = [
                (
                    value.unsqueeze(0).expand(context.batch_size, -1, -1).clone()
                    if value.ndim == 2
                    else value
                )
                for value in targets
            ]
        if route.preserve_current_object_orientation:
            observed = resolve_pose_goal(
                SceneEntityPose(route.object_id), context, name="transport object"
            )
            for value in targets:
                value[..., :3, :3] = observed[..., :3, :3]
        target = torch.stack(targets, dim=1) if route.waypoints else targets[0]
        return _held_transport_lowering(route.object_id, target)

    def pick_lookahead_targets(
        self,
        call: RegisteredSemanticCall,
        *,
        picked_object: SceneObjectRef,
        bound: BoundSemanticCall,
        previous_target: SemanticObjectTarget | None,
    ) -> tuple[SemanticObjectTarget, ...] | None:
        """Conservatively screen the nominal target, which is always permitted."""
        del bound, previous_target
        route = self._route(call)
        if picked_object.entity_id != route.object_id:
            return None
        return tuple(
            SemanticObjectTarget(
                pose=(
                    SemanticPose(pose.position, pose.quaternion_xyzw)
                    if type(pose) is AbsolutePoseTarget
                    else resolve_pose_target(pose)
                ),
                preserve_current_object_orientation=route.preserve_current_object_orientation,
            )
            for pose in (*route.waypoints, route.pose)
        )


@dataclass(frozen=True, slots=True)
class RoutedMoveHeldObjectLowererFactory(RegisteredSemanticLowererFactory):
    """Create routed transport lowering into the existing MoveHeldObject skill.

    Args:
        routes: Nonempty immutable routes with unique target IDs. They declare
            retained attachment; the original single-target configuration keeps
            its existing effect contract and factory identity.
    """

    call_id: ClassVar[str] = _MOVE_HELD_OBJECT_CALL_ID
    revision: ClassVar[str] = "routed-3"
    target_descriptor: ClassVar[SkillDescriptor] = MoveHeldObject.descriptor()
    routes: tuple[HeldObjectTransportRoute, ...]

    def __post_init__(self) -> None:
        if (
            type(self.routes) is not tuple
            or not self.routes
            or not all(type(route) is HeldObjectTransportRoute for route in self.routes)
        ):
            raise TypeError(
                "routes must be a non-empty tuple of HeldObjectTransportRoute values."
            )
        if len({route.target_id for route in self.routes}) != len(self.routes):
            raise ValueError("MoveHeldObject target IDs must be unique.")

    def create(
        self,
        *,
        simulation: object,
        robot: object,
        scene_registry: SceneRegistry,
        engine: AtomicSkillEngine,
    ) -> RegisteredSemanticLowerer:
        """Validate canonical references and retain only inert declarations.

        Args:
            simulation: Owning simulation; no live state is read.
            robot: Exact robot selected by the existing engine.
            scene_registry: Registry containing the declared canonical identities.
            engine: Existing AtomicSkillEngine; no skill registration is changed.

        Returns:
            A fresh lowerer with no retained simulator or robot handle.
        """
        del simulation
        if engine.robot is not robot:
            raise ValueError("MoveHeldObject requires the engine's exact robot.")
        self.validate_scene_declaration(SceneManifest.from_registry(scene_registry))
        return _RoutedMoveHeldObjectLowerer(self.routes)

    def validate_scene_declaration(self, scene: SceneManifest) -> None:
        """Require canonical object and reference IDs without observing providers.

        Args:
            scene: Immutable provider-free scene metadata.
        """
        for route in self.routes:
            if (
                scene.resolve(route.object_id, expected_type=SceneObjectRef).entity_id
                != route.object_id
            ):
                raise ValueError(
                    "Transport objects require canonical scene identities."
                )
            for pose in (*route.waypoints, route.pose):
                if (
                    type(pose) is SceneEntityTarget
                    and scene.resolve(pose.entity_id).entity_id != pose.entity_id
                ):
                    raise ValueError(
                        "Transport references require canonical scene identities."
                    )
