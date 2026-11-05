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

from dataclasses import dataclass, replace
from typing import ClassVar
from embodichain.lab.sim.atomic_actions import (
    ActionOptions,
    AtomicActionEngine,
    MoveHeldObject,
    MoveHeldObjectOptions,
    HeldObjectPoseGoal,
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
    SkillPolicyPreset,
)
from embodichain.lab.task_program.compiler.lowering import (
    RegisteredSemanticLowerer,
    SemanticLowering,
    SemanticObjectTarget,
)
from embodichain.lab.task_program.integrations.extensions import (
    RegisteredSemanticLowererFactory,
)
from .effect_contracts import TransportEffectContract, _attachment_effect
from .targets import (
    AbsolutePoseTarget,
    SceneEntityTarget,
    resolve_pose_target,
    _identifier,
)

__all__ = [
    "HeldObjectTransportRoute",
    "RoutedMoveHeldObjectLowererFactory",
    "TerminalAttachmentTransportLowererFactory",
    "DrawerTransportLowererFactory",
]

_MOVE_HELD_OBJECT_CALL_ID = "simulation.move_held_object"


@dataclass(frozen=True, slots=True)
class HeldObjectTransportRoute:
    """One exact declared object target, separate from planning options.

    Args:
        object_id: Canonical carried-object identity.
        target_id: Named target selected by registered call arguments.
        pose: Immutable absolute or scene-relative pose declaration.
    """

    object_id: str
    target_id: str
    pose: AbsolutePoseTarget | SceneEntityTarget

    def __post_init__(self) -> None:
        _identifier(self.object_id, field_name="object_id")
        _identifier(self.target_id, field_name="target_id")
        resolve_pose_target(self.pose)
        object.__setattr__(self, "pose", self.pose.snapshot())


class _RoutedMoveHeldObjectLowerer(RegisteredSemanticLowerer):
    """Bind absolute or scene-relative targets to the shared transport goal."""

    call_id: ClassVar[str] = _MOVE_HELD_OBJECT_CALL_ID
    effect_contract: ClassVar[TransportEffectContract] = (
        TransportEffectContract.GUARDED_ATTACHMENT
    )
    target_descriptor: ClassVar[SkillDescriptor] = MoveHeldObject.descriptor()
    effect_contract_kind: ClassVar[SemanticEffectKind] = SemanticEffectKind.ATTACH

    def __init__(self, routes: tuple[HeldObjectTransportRoute, ...]) -> None:
        self._routes = {route.target_id: route for route in routes}
        if not routes or len(self._routes) != len(routes):
            raise ValueError("MoveHeldObject targets must be non-empty and unique.")

    def _route(self, call: RegisteredSemanticCall) -> HeldObjectTransportRoute:
        arguments = dict(call.arguments)
        if "target" not in arguments or set(arguments) - {
            "target",
            "object",
            "reference",
        }:
            raise ValueError("MoveHeldObject arguments must select a declared target.")
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
        option_template: ActionOptions,
    ) -> SemanticLowering:
        """Bind an exact target without observing the robot or solving IK."""
        del context, bound
        if type(option_template) is not MoveHeldObjectOptions:
            raise TypeError(
                "Configured transport requires exact MoveHeldObjectOptions."
            )
        if (
            self.effect_contract is TransportEffectContract.TERMINAL_ATTACHMENT
            and not option_template.report_retained_attachment
        ):
            raise ValueError(
                "Terminal attachment requires report_retained_attachment=True."
            )
        route = self._route(call)
        return SemanticLowering(
            goal=HeldObjectPoseGoal(
                resolve_pose_target(route.pose),
                object_id=(
                    route.object_id
                    if self.effect_contract
                    is TransportEffectContract.GUARDED_ATTACHMENT
                    else None
                ),
            ),
            registered_effect=_attachment_effect(route.object_id, self.effect_contract),
        )

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
        return (
            SemanticObjectTarget(
                pose=(
                    SemanticPose(route.pose.position, route.pose.quaternion_xyzw)
                    if type(route.pose) is AbsolutePoseTarget
                    else resolve_pose_target(route.pose)
                )
            ),
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
    revision: ClassVar[str] = "routed-2"
    target_descriptor: ClassVar[SkillDescriptor] = MoveHeldObject.descriptor()
    routes: tuple[HeldObjectTransportRoute, ...]
    _lowerer_type: ClassVar[type[_RoutedMoveHeldObjectLowerer]] = (
        _RoutedMoveHeldObjectLowerer
    )

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
        engine: AtomicActionEngine,
    ) -> RegisteredSemanticLowerer:
        """Validate canonical references and retain only inert declarations.

        Args:
            simulation: Owning simulation; no live state is read.
            robot: Exact robot selected by the existing engine.
            scene_registry: Registry containing the declared canonical identities.
            engine: Existing AtomicActionEngine; no skill registration is changed.

        Returns:
            A fresh lowerer with no retained simulator or robot handle.
        """
        del simulation
        if engine.robot is not robot:
            raise ValueError("MoveHeldObject requires the engine's exact robot.")
        self.validate_scene_declaration(SceneManifest.from_registry(scene_registry))
        return self._lowerer_type(self.routes)

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
            if type(route.pose) is SceneEntityTarget:
                if (
                    scene.resolve(route.pose.entity_id).entity_id
                    != route.pose.entity_id
                ):
                    raise ValueError(
                        "Transport references require canonical scene identities."
                    )


class _TerminalAttachmentTransportLowerer(_RoutedMoveHeldObjectLowerer):
    """Keep the original terminal-only attachment proof and identical motion goal."""

    effect_contract: ClassVar[TransportEffectContract] = (
        TransportEffectContract.TERMINAL_ATTACHMENT
    )


@dataclass(frozen=True, slots=True)
class TerminalAttachmentTransportLowererFactory(RoutedMoveHeldObjectLowererFactory):
    """Select the complete source terminal ATTACH contract explicitly.

    The goal and shared action remain unchanged. The selected preset must retain
    ``report_retained_attachment=True`` and verified terminal effect monitoring;
    phase-qualified generic transport remains the ordinary default.

    Args:
        routes: Complete immutable object/target routes using the shared transport skill.
    """

    revision: ClassVar[str] = "terminal-attachment-1"
    _lowerer_type: ClassVar[type[_RoutedMoveHeldObjectLowerer]] = (
        _TerminalAttachmentTransportLowerer
    )

    @classmethod
    def validate_preset(cls, preset: SkillPolicyPreset) -> None:
        """Reject a terminal contract lacking its actual configured verification.

        Args:
            preset: Exact default or occurrence-selected policy before live assembly.

        Raises:
            ValueError: Attachment reporting or verified terminal monitoring is absent.
        """
        from embodichain.lab.task_program.semantics import EffectAssurance

        options = preset.action_option_template(cls.call_id)
        if (
            type(options) is not MoveHeldObjectOptions
            or not options.report_retained_attachment
        ):
            raise ValueError(
                "Terminal attachment requires report_retained_attachment=True."
            )
        if (
            preset.effect_assurance is not EffectAssurance.VERIFIED
            or preset.effect_monitors.get(cls.call_id) is None
        ):
            raise ValueError(
                "Terminal attachment requires verified terminal monitoring."
            )


class _DrawerTransportLowerer(_RoutedMoveHeldObjectLowerer):
    """Retain the original effectless drawer transport call contract."""

    call_id: ClassVar[str] = "gen_sim.drawer_transport"
    effect_contract_kind: ClassVar[SemanticEffectKind | None] = None
    preserves_symbolic_state: ClassVar[bool] = True

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
            goal=replace(lowered.goal, object_id=None),
            registered_effect=None,
        )


@dataclass(frozen=True, slots=True)
class DrawerTransportLowererFactory(RoutedMoveHeldObjectLowererFactory):
    """Declare source drawer targets without claiming generic transport equivalence.

    The configured registration rejects execution until its drawer-specific
    planning service is declared. Direct declaration inspection remains pure.
    """

    call_id: ClassVar[str] = "gen_sim.drawer_transport"
    revision: ClassVar[str] = "drawer-2"

    def create(
        self,
        *,
        simulation: object,
        robot: object,
        scene_registry: SceneRegistry,
        engine: AtomicActionEngine,
    ) -> RegisteredSemanticLowerer:
        """Validate exact owners and construct the original effectless lowerer.

        Args:
            simulation: Borrowed simulation; no state is read.
            robot: Exact engine robot.
            scene_registry: Canonical object/target registry.
            engine: Existing shared Atomic engine.

        Returns:
            A fresh effectless transport lowerer without runtime handles.
        """
        RoutedMoveHeldObjectLowererFactory.create(
            self,
            simulation=simulation,
            robot=robot,
            scene_registry=scene_registry,
            engine=engine,
        )
        return _DrawerTransportLowerer(self.routes)
