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

"""Configured coordinated retention using the ordinary shared skill."""

from __future__ import annotations
from dataclasses import dataclass, replace
from typing import ClassVar
from embodichain.lab.sim.atomic_actions import AtomicActionEngine, ObjectSemantics
from embodichain.lab.task_program.semantics import (
    SceneRegistry,
    SceneManifest,
    SceneObjectRef,
    GRASP_AFFORDANCE_CAPABILITY,
)
from embodichain.lab.task_program.compiler.lowering import RegisteredSemanticLowerer
from .._configured_services import (
    _CoordinatedTransportRoute,
    _identifier,
    _world_displacement,
)
from embodichain.lab.task_program.semantics import SemanticEffectKind
from .._configured_services import (
    _CoordinatedTransportLowerer,
    _CoordinatedTransportLowererFactory,
)

__all__ = [
    "CoordinatedHoldLowererFactory",
    "CoordinatedTargetRoute",
    "CoordinatedPlacementLowererFactory",
]


class _CoordinatedHoldLowerer(_CoordinatedTransportLowerer):
    call_id: ClassVar[str] = "simulation.coordinated_hold"
    effect_contract_kind: ClassVar[SemanticEffectKind] = SemanticEffectKind.ATTACH
    release: ClassVar[bool] = False


@dataclass(frozen=True, slots=True)
class CoordinatedHoldLowererFactory(_CoordinatedTransportLowererFactory):
    """Bind ordered configured transport targets while retaining both grasps.

    The routes use the existing relative-pose or fresh world-displacement form.
    Typed action options must explicitly retain the object; this factory never
    rewrites options or infers successful dual attachment from a target pose.
    """

    call_id: ClassVar[str] = "simulation.coordinated_hold"
    revision: ClassVar[str] = "hold-1"
    lowerer_type: ClassVar[type[_CoordinatedTransportLowerer]] = _CoordinatedHoldLowerer


@dataclass(frozen=True, slots=True)
class CoordinatedTargetRoute(_CoordinatedTransportRoute):
    """Source-qualified coordinated on-support or zero-net-displacement route.

    Relative supports require an explicit distinct rigid reference and world
    offset. A zero world offset means retain the measured object pose; it is not
    converted into an invented motion or filtered out as an empty request.
    """

    relation: str | None = None

    def __post_init__(self) -> None:
        if self.relation is not None:
            if self.relation != "on" or self.relative_pose is not None:
                raise ValueError(
                    "Coordinated relative placement only supports on with a world offset."
                )
            for name in ("object_id", "target_id", "reference_entity_id"):
                _identifier(getattr(self, name), field_name=name)
            if self.object_id == self.reference_entity_id:
                raise ValueError("An object cannot be its own placement support.")
            object.__setattr__(
                self, "world_displacement", _world_displacement(self.world_displacement)
            )
            return
        if (
            type(self.world_displacement) is tuple
            and len(self.world_displacement) == 3
            and all(
                type(value) in (int, float) and value == 0
                for value in self.world_displacement
            )
            and self.reference_entity_id is None
            and self.relative_pose is None
        ):
            _identifier(self.object_id, field_name="object_id")
            _identifier(self.target_id, field_name="target_id")
            return
        _CoordinatedTransportRoute.__post_init__(self)


class _CoordinatedPlacementLowerer(_CoordinatedTransportLowerer):
    def __init__(
        self,
        routes: tuple[CoordinatedTargetRoute, ...],
        semantics: tuple[ObjectSemantics, ...],
    ) -> None:
        if len(routes) != len(semantics):
            raise ValueError(
                "Coordinated target routes require matching object semantics."
            )
        self._routes = {
            (route.object_id, route.target_id): (route, sem)
            for route, sem in zip(routes, semantics, strict=True)
        }


@dataclass(frozen=True, slots=True)
class CoordinatedPlacementLowererFactory(_CoordinatedTransportLowererFactory):
    """Resolve source-qualified support routes through canonical dual-arm transport.

    This explicit extension leaves the original shared route class and default
    factory identity unchanged. Source mesh/root lookup and candidate selection
    stay with Scene and Atomic; lowering only binds the declared target frame.
    """

    revision: ClassVar[str] = "support-1"
    routes: tuple[CoordinatedTargetRoute, ...]
    lowerer_type: ClassVar[type[_CoordinatedTransportLowerer]] = (
        _CoordinatedPlacementLowerer
    )

    def __post_init__(self) -> None:
        if (
            type(self.routes) is not tuple
            or not self.routes
            or any(type(route) is not CoordinatedTargetRoute for route in self.routes)
        ):
            raise TypeError("Coordinated placement requires explicit typed routes.")
        if len({(route.object_id, route.target_id) for route in self.routes}) != len(
            self.routes
        ):
            raise ValueError("Coordinated placement routes must be unique.")

    def _canonical_routes(
        self, scene: SceneManifest | SceneRegistry
    ) -> tuple[CoordinatedTargetRoute, ...]:
        result = []
        for route in self.routes:
            obj = scene.resolve(route.object_id, expected_type=SceneObjectRef)
            reference = (
                scene.resolve(route.reference_entity_id, expected_type=SceneObjectRef)
                if route.relation is not None
                else (
                    scene.resolve(route.reference_entity_id)
                    if route.reference_entity_id is not None
                    else None
                )
            )
            result.append(
                replace(
                    route,
                    object_id=obj.entity_id,
                    reference_entity_id=(
                        None if reference is None else reference.entity_id
                    ),
                )
            )
        return tuple(result)

    def validate_scene_declaration(self, scene: SceneManifest) -> None:
        """Reject unknown or non-rigid support identities before runtime assembly.

        Args:
            scene: Provider-free authoritative scene declaration.
        """
        self._canonical_routes(scene)

    def create(
        self,
        *,
        simulation: object,
        robot: object,
        scene_registry: SceneRegistry,
        engine: AtomicActionEngine,
    ) -> RegisteredSemanticLowerer:
        """Resolve immutable target/grasp bindings without planning or native reads.

        Args:
            simulation: Borrowed owner, unused by declaration binding.
            robot: Exact engine robot.
            scene_registry: Canonical object and grasp registry.
            engine: Existing Atomic engine.

        Returns:
            Fresh lowerer whose target offset retains live object orientation.
        """
        if engine.robot is not robot:
            raise ValueError("Coordinated transport requires the engine's exact robot.")
        routes = self._canonical_routes(scene_registry)
        semantics = []
        for route in routes:
            obj = scene_registry.resolve(route.object_id, expected_type=SceneObjectRef)
            grasp = scene_registry.resolve_affordance(
                obj, capability=GRASP_AFFORDANCE_CAPABILITY
            )
            semantics.append(scene_registry.object_semantics(obj, affordance=grasp))
        return self.lowerer_type(routes, tuple(semantics))


class _CoordinatedRetainedTargetLowerer(_CoordinatedPlacementLowerer):
    call_id: ClassVar[str] = "simulation.coordinated_hold"
    effect_contract_kind: ClassVar[SemanticEffectKind] = SemanticEffectKind.ATTACH
    release: ClassVar[bool] = False


@dataclass(frozen=True, slots=True)
class _CoordinatedRetainedTargetFactory(CoordinatedPlacementLowererFactory):
    call_id: ClassVar[str] = "simulation.coordinated_hold"
    lowerer_type: ClassVar[type[_CoordinatedTransportLowerer]] = (
        _CoordinatedRetainedTargetLowerer
    )

    def __post_init__(self) -> None:
        CoordinatedPlacementLowererFactory.__post_init__(self)
        if any(route.relation is not None for route in self.routes):
            raise ValueError("Coordinated on placement requires release.")
