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

"""Observed relative placement bindings that preserve existing release evidence."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, ClassVar
import torch

from embodichain.lab.sim.atomic_actions import (
    ActionOptions,
    AtomicActionEngine,
    Place,
    PlanningContext,
    SkillDescriptor,
)
from embodichain.lab.task_program.compiler.lowering import (
    RegisteredSemanticLowerer,
    SemanticLowering,
    RegisteredSemanticEffect,
)
from embodichain.lab.task_program.semantics import (
    BoundSemanticCall,
    RegisteredSemanticCall,
    SceneObjectRef,
    SceneArticulationRef,
    SceneLinkRef,
    SceneRegistry,
    SceneManifest,
)
from ..extensions import RegisteredSemanticLowererFactory
from .._configured_services import (
    _RelativePlaceLowerer,
    _identifier,
    _world_displacement,
)

__all__ = [
    "ObservedRelativePlaceBinding",
    "ObservedRelativePlaceLowererFactory",
    "UprightPlaceLowererFactory",
]
_PLACE_RELATIVE_CALL_ID = "simulation.place_relative"


@dataclass(frozen=True, slots=True)
class ObservedRelativePlaceBinding:
    """One source-qualified relative placement target.

    Args:
        object_id: Canonical manipulated object identity.
        reference_entity_id: Declared rigid-object or articulation reference.
        relation: Declared semantic spatial relation.
        world_displacement: Explicit displacement in the world frame.
        world_yaw_offset: Explicit additional world yaw in radians.
        stage_id: Optional semantic stage selector.
        reference_link_id: Optional qualified link under the reference articulation.
        reference_part: Optional source part selector bound to that link.
    """

    object_id: str
    reference_entity_id: str
    relation: str
    world_displacement: tuple[float, float, float]
    world_yaw_offset: float = 0.0
    stage_id: str = ""
    reference_link_id: str | None = None
    reference_part: str | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "object_id",
            "reference_entity_id",
            "relation",
        ):
            object.__setattr__(
                self,
                field_name,
                _identifier(getattr(self, field_name), field_name=field_name),
            )
        if type(self.stage_id) is not str or self.stage_id != self.stage_id.strip():
            raise ValueError("stage_id must be a string without outer whitespace.")
        if self.relation not in {
            "front_left_of",
            "front_right_of",
            "back_left_of",
            "back_right_of",
            "above",
            "behind",
            "front_of",
            "left_of",
            "on",
            "right_of",
        }:
            raise ValueError(
                "Relative Place relation must be one of above, behind, front_of, "
                "left_of, on, right_of, or a front/back left/right diagonal."
            )
        object.__setattr__(
            self,
            "world_displacement",
            _world_displacement(self.world_displacement),
        )
        if (
            isinstance(self.world_yaw_offset, bool)
            or not isinstance(self.world_yaw_offset, (int, float))
            or not math.isfinite(self.world_yaw_offset)
        ):
            raise ValueError("world_yaw_offset must be finite.")
        object.__setattr__(self, "world_yaw_offset", float(self.world_yaw_offset))
        for name in ("reference_link_id", "reference_part"):
            if getattr(self, name) is not None:
                object.__setattr__(
                    self, name, _identifier(getattr(self, name), field_name=name)
                )
        if self.reference_part is not None and self.reference_link_id is None:
            raise ValueError("A reference part requires its qualified link.")

    @property
    def selector(self) -> tuple[str, str, str, str]:
        """Return the semantic arguments selecting this immutable route."""
        return self.stage_id, self.object_id, self.reference_entity_id, self.relation


class _TerminalReleasePlaceLowerer(_RelativePlaceLowerer):
    """Declare the original complete terminal release independently of phases."""

    def _release_effect(self, object_id: str) -> RegisteredSemanticEffect:
        from .effect_contracts import PlacementEffectContract, _release_effect

        return _release_effect(object_id, PlacementEffectContract.TERMINAL_RELEASE)


class _ObservedRelativePlaceLowerer(_TerminalReleasePlaceLowerer):
    """Correct placement geometry without rebasing the release evidence."""

    def __init__(
        self, routes: tuple[ObservedRelativePlaceBinding, ...], robot: Any
    ) -> None:
        super().__init__(routes)
        self._robot = robot

    def _resolve_route(
        self, call: RegisteredSemanticCall
    ) -> ObservedRelativePlaceBinding:
        arguments = dict(call.arguments)
        part = arguments.pop("reference_part", None)
        route = super()._resolve_route(replace(call, arguments=arguments))
        if part != route.reference_part:
            raise ValueError(
                "Relative Place reference part differs from its declared route."
            )
        return replace(
            route,
            reference_entity_id=route.reference_link_id or route.reference_entity_id,
        )

    def lower(
        self,
        call: RegisteredSemanticCall,
        *,
        context: PlanningContext,
        bound: BoundSemanticCall,
        option_template: ActionOptions,
    ) -> SemanticLowering:
        result = super().lower(
            call, context=context, bound=bound, option_template=option_template
        )
        route = self._resolve_route(call)
        object_pose = self._observed_pose(context, route.object_id)
        motion = bound.binding.resources["primary"].endpoints["motion"].runtime_target
        eef = self._robot.compute_fk(
            qpos=context.robot.qpos[:, list(motion.joint_ids)],
            name=motion.control_part,
            env_ids=context.env_ids.tolist(),
            to_matrix=True,
        ).to(object_pose)
        observed_grasp = torch.linalg.solve(object_pose, eef)
        world_orientation = result.goal.xpos.world_orientation
        if route.world_yaw_offset:
            if world_orientation is None:
                raise ValueError(
                    "Relative Place world yaw requires preserved object orientation."
                )
            angle = object_pose.new_tensor(route.world_yaw_offset)
            yaw = torch.eye(3, dtype=object_pose.dtype, device=object_pose.device)
            yaw[0, 0] = torch.cos(angle)
            yaw[0, 1] = -torch.sin(angle)
            yaw[1, 0] = torch.sin(angle)
            yaw[1, 1] = torch.cos(angle)
            world_orientation = yaw @ world_orientation
        return replace(
            result,
            goal=replace(
                result.goal,
                xpos=replace(
                    result.goal.xpos,
                    relative_pose=observed_grasp,
                    world_orientation=world_orientation,
                ),
            ),
        )


@dataclass(frozen=True, slots=True)
class ObservedRelativePlaceLowererFactory(RegisteredSemanticLowererFactory):
    """Create observed-grasp placement lowering into the existing Place skill.

    Args:
        routes: Immutable source-qualified relative placement declarations.
    """

    call_id: ClassVar[str] = _PLACE_RELATIVE_CALL_ID
    revision: ClassVar[str] = "observed-terminal-2"
    target_descriptor: ClassVar[SkillDescriptor] = Place.descriptor()

    routes: tuple[ObservedRelativePlaceBinding, ...]
    lowerer_type: ClassVar[type[_ObservedRelativePlaceLowerer]] = (
        _ObservedRelativePlaceLowerer
    )

    def __post_init__(self) -> None:
        if type(self.routes) is not tuple or not self.routes:
            raise ValueError("Relative Place routes must be a non-empty exact tuple.")
        if not all(
            type(route) is ObservedRelativePlaceBinding for route in self.routes
        ):
            raise TypeError(
                "Relative Place routes must be ObservedRelativePlaceBinding values."
            )
        if len({route.selector for route in self.routes}) != len(self.routes):
            raise ValueError("Relative Place routes must be unique.")

    def create(
        self,
        *,
        simulation: object,
        robot: object,
        scene_registry: SceneRegistry,
        engine: AtomicActionEngine,
    ) -> RegisteredSemanticLowerer:
        """Bind canonical references to a fresh runtime lowerer.

        Args:
            simulation: Owning simulation, not read during factory construction.
            robot: Exact robot whose captured joint state supplies runtime FK.
            scene_registry: Registry owning all declared references.
            engine: Existing canonical Atomic engine for that robot.

        Returns:
            A lowerer that updates goals without rebasing held-state evidence.
        """
        del simulation
        if engine.robot is not robot:
            raise ValueError(
                "Relative Place lowerer requires the engine's exact robot."
            )
        return self.lowerer_type(self._canonical_routes(scene_registry), robot)

    def validate_scene_declaration(self, scene: SceneManifest) -> None:
        """Validate source references using provider-free scene metadata.

        Args:
            scene: The immutable declared scene used before live assembly.
        """
        self._canonical_routes(scene)

    def _canonical_routes(
        self, scene: SceneManifest | SceneRegistry
    ) -> tuple[ObservedRelativePlaceBinding, ...]:
        routes: list[ObservedRelativePlaceBinding] = []
        for route in self.routes:
            object_ref = scene.resolve(
                route.object_id,
                expected_type=SceneObjectRef,
            )
            reference_ref = scene.resolve(
                route.reference_entity_id,
            )
            if type(reference_ref) not in {SceneObjectRef, SceneArticulationRef}:
                raise TypeError(
                    "Relative Place reference must be a rigid or articulation root."
                )
            if isinstance(reference_ref, SceneArticulationRef) and route.relation in {
                "on",
                "above",
            }:
                raise ValueError(
                    "Articulated supports require qualified link bindings."
                )
            if route.reference_link_id is not None:
                link = scene.lookup(route.reference_link_id, expected_type=SceneLinkRef)
                if link.parent != reference_ref:
                    raise ValueError(
                        "Relative Place link belongs to a different articulation."
                    )
            routes.append(
                ObservedRelativePlaceBinding(
                    object_id=object_ref.entity_id,
                    reference_entity_id=reference_ref.entity_id,
                    relation=route.relation,
                    world_displacement=route.world_displacement,
                    world_yaw_offset=route.world_yaw_offset,
                    stage_id=route.stage_id,
                    reference_link_id=route.reference_link_id,
                    reference_part=route.reference_part,
                )
            )
        return tuple(routes)


class _UprightPlaceLowerer(_ObservedRelativePlaceLowerer):
    """Preserve the registered upright alias while shared Place owns execution."""

    call_id: ClassVar[str] = "gen_sim.place_upright"


@dataclass(frozen=True, slots=True)
class UprightPlaceLowererFactory(ObservedRelativePlaceLowererFactory):
    """Bind the public upright release alias to observed relative placement.

    Routes retain the same stage-qualified displacement and yaw declarations as
    ordinary observed placement. Upright acceptance remains a declared post-policy;
    this factory does not rotate objects, plan paths or issue success evidence.
    """

    call_id: ClassVar[str] = "gen_sim.place_upright"
    lowerer_type: ClassVar[type[_ObservedRelativePlaceLowerer]] = _UprightPlaceLowerer
