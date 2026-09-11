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

"""Task-owned immutable route declarations and skill binding services."""

from __future__ import annotations
from dataclasses import dataclass, make_dataclass
import hashlib
import math
from typing import Any, ClassVar
import torch
from embodichain.lab.task_program.integrations.extensions import (
    RegisteredSemanticLowererFactory,
)
from embodichain.lab.sim.atomic_actions import (
    ActionOptions,
    AtomicActionEngine,
    CoordinatedPickGoal,
    CoordinatedPickment,
    CoordinatedPickmentOptions,
    HeldObjectPoseGoal,
    MoveHeldObject,
    MoveHeldObjectOptions,
    ObjectSemantics,
    Place,
    PlanningContext,
    SceneEntityPose,
    SkillDescriptor,
    GraspGoal,
    PickUp,
    PickUpOptions,
)
from embodichain.lab.task_program.semantics import (
    BoundSemanticCall,
    GRASP_AFFORDANCE_CAPABILITY,
    HeldObjectRelation,
    RegisteredSemanticCall,
    SceneObjectRef,
    SceneRegistry,
    SemanticEffectKind,
    SemanticPose,
)
from embodichain.lab.task_program.compiler.lowering import (
    RegisteredHeldObjectEffect,
    RegisteredPhaseProtection,
    RegisteredPhaseProtectionKind,
    RegisteredSemanticLowerer,
    RegisteredSemanticEffect,
    SemanticLowering,
    SemanticObjectTarget,
)

from embodichain.lab.task_program.integrations._configured_services import (
    _CoordinatedTransportRoute,
    _RelativePlaceLowerer,
    _coordinated_transport_route,
    _identifier,
    _pose,
    _world_displacement,
)

from .grasp_filter import TaskGraspPoseGenerator

__all__: list[str] = []

_COORDINATED_TRANSPORT_CALL_ID = "simulation.coordinated_transport"


_MOVE_HELD_OBJECT_CALL_ID = "simulation.move_held_object"


_PLACE_RELATIVE_CALL_ID = "simulation.place_relative"


_COORDINATED_HOLD_CALL_ID = "simulation.coordinated_hold"


_PICK_CALL_ID = "simulation.pick"


def _point(
    value: tuple[float, float, float],
    *,
    field_name: str,
) -> tuple[float, float, float]:
    """Validate one finite local-frame point."""
    if type(value) is not tuple or len(value) != 3:
        raise TypeError(f"{field_name} must be an exact three-value tuple.")
    normalized = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in normalized):
        raise ValueError(f"{field_name} must contain three finite values.")
    return normalized


@dataclass(frozen=True, slots=True)
class _SceneEntityTarget:
    """Deeply immutable scene-relative target stored in an integration."""

    entity_id: str
    relative_pose: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        _identifier(self.entity_id, field_name="entity_id")
        if self.relative_pose is not None:
            object.__setattr__(self, "relative_pose", _pose(self.relative_pose))

    def snapshot(self) -> _SceneEntityTarget:
        """Return the immutable declaration, which owns no runtime tensors."""
        return self


@dataclass(frozen=True, slots=True)
class _AbsolutePoseTarget:
    """Tensor-free absolute pose stored in a fingerprinted declaration."""

    position: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "position", _point(self.position, field_name="position")
        )
        quaternion = self.quaternion_wxyz
        if (
            type(quaternion) is not tuple
            or len(quaternion) != 4
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                for value in quaternion
            )
        ):
            raise ValueError("quaternion_wxyz must contain four finite real values.")
        norm = math.hypot(*quaternion)
        if norm <= 1.0e-8:
            raise ValueError("quaternion_wxyz must be non-zero.")
        object.__setattr__(
            self, "quaternion_wxyz", tuple(value / norm for value in quaternion)
        )

    def snapshot(self) -> _AbsolutePoseTarget:
        """Return the deeply immutable target declaration."""
        return self

    def to_matrix(self) -> torch.Tensor:
        """Create an independently owned runtime pose."""
        return SemanticPose(self.position, self.quaternion_wxyz).to_matrix()


def _configured_goal_pose(
    value: _AbsolutePoseTarget | _SceneEntityTarget,
) -> torch.Tensor | SceneEntityPose:
    """Convert an inert configured pose without observing or choosing a target."""
    if type(value) is _AbsolutePoseTarget:
        return value.to_matrix()
    if type(value) is _SceneEntityTarget:
        return SceneEntityPose(
            value.entity_id,
            relative_pose=(
                None
                if value.relative_pose is None
                else torch.tensor(value.relative_pose, dtype=torch.float32).reshape(
                    4, 4
                )
            ),
        )
    raise TypeError(
        "Configured targets must be _AbsolutePoseTarget or _SceneEntityTarget."
    )


@dataclass(frozen=True, slots=True)
class _PickRoute:
    """One named set of target-dependent pickup constraints."""

    object_id: str
    target_id: str
    release_clearance_object_pose: _AbsolutePoseTarget | _SceneEntityTarget | None = (
        None
    )
    release_clearance_plane_z: float | None = None
    release_clearance_safety_margin: float = 0.0
    grasp_region: str | None = None

    def __post_init__(self) -> None:
        _identifier(self.object_id, field_name="object_id")
        _identifier(self.target_id, field_name="target_id")
        if (self.release_clearance_object_pose is None) != (
            self.release_clearance_plane_z is None
        ):
            raise ValueError(
                "Release-clearance pose and plane must be provided together."
            )
        if self.release_clearance_object_pose is not None:
            _configured_goal_pose(self.release_clearance_object_pose)
            object.__setattr__(
                self,
                "release_clearance_object_pose",
                self.release_clearance_object_pose.snapshot(),
            )
        for name in ("release_clearance_plane_z", "release_clearance_safety_margin"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be a finite real number.")
        if self.release_clearance_safety_margin < 0.0:
            raise ValueError("release_clearance_safety_margin must be non-negative.")
        if self.grasp_region not in {None, "upper_half"}:
            raise ValueError("Task grasp_region must be upper_half or omitted.")
        if (
            self.release_clearance_object_pose is not None
            and type(self.release_clearance_object_pose) is not _AbsolutePoseTarget
        ):
            raise ValueError(
                "Task release-clearance filtering requires an absolute pose."
            )


class _PickLowerer(RegisteredSemanticLowerer):
    """Bind configured target constraints to the shared pickup goal."""

    call_id: ClassVar[str] = _PICK_CALL_ID
    target_descriptor: ClassVar[SkillDescriptor] = PickUp.descriptor()
    effect_contract_kind: ClassVar[SemanticEffectKind] = SemanticEffectKind.ATTACH
    phase_protection_kind: ClassVar[RegisteredPhaseProtectionKind] = (
        RegisteredPhaseProtectionKind.ACQUIRE
    )

    def __init__(
        self,
        routes: tuple[_PickRoute, ...],
        semantics: tuple[ObjectSemantics, ...],
        grasp_generators: Any = None,
        call_id: str | None = None,
    ) -> None:
        self._routes = {(route.object_id, route.target_id): route for route in routes}
        self._semantics = {item.entity_id: item for item in semantics}
        if call_id is not None:
            self.call_id = call_id
        self._grasp_generators = (
            {} if grasp_generators is None else dict(grasp_generators)
        )
        if not routes or len(self._routes) != len(routes):
            raise ValueError("Configured Pick routes must be non-empty and unique.")
        if len(self._semantics) != len(semantics) or any(
            route.object_id not in self._semantics for route in routes
        ):
            raise ValueError(
                "Configured Pick routes require unique matching object semantics."
            )

    def lower(
        self,
        call: RegisteredSemanticCall,
        *,
        context: PlanningContext,
        bound: BoundSemanticCall,
        option_template: ActionOptions,
    ) -> SemanticLowering:
        """Bind task declarations to baseline goal/options and the installed sampler."""
        if type(option_template) is not PickUpOptions:
            raise TypeError("Configured Pick requires exact PickUpOptions.")
        arguments = dict(call.arguments)
        if set(arguments) != {"object", "target"}:
            raise ValueError(
                "Configured Pick arguments must contain only object and target."
            )
        key = (arguments["object"], arguments["target"])
        route = self._routes.get(key)
        if route is None:
            raise ValueError(f"Configured Pick has no route {key!r}.")
        semantics = self._semantics[route.object_id]
        if (
            route.grasp_region is not None
            or route.release_clearance_object_pose is not None
        ):
            if (
                option_template.pick_object_part != "center"
                or option_template.rotate_upright is not None
                or option_template.fixed_object_to_eef is not None
            ):
                raise ValueError(
                    "Constrained Pick requires unrotated, unrestricted baseline sampling."
                )
            target = (
                bound.binding.resources["primary"].endpoints["grasp"].runtime_target
            )
            sampler = self._grasp_generators.get(target.target_id)
            if not isinstance(sampler, TaskGraspPoseGenerator):
                raise ValueError(
                    "Constrained Pick requires the task-owned grasp filter."
                )
            sampler.require_rule(
                route.object_id,
                route.target_id,
                semantics.affordance.mesh_vertices,
                semantics.affordance.mesh_triangles,
            )
        return SemanticLowering(
            goal=GraspGoal(semantics=semantics),
            phase_protection=RegisteredPhaseProtection(
                kind=RegisteredPhaseProtectionKind.ACQUIRE,
                held_object=RegisteredHeldObjectEffect(
                    expectation_id="primary",
                    relation=HeldObjectRelation.ATTACHED,
                    object_id=route.object_id,
                    slot_id="primary",
                ),
                active_segments=("lift",),
                gate_segment="lift",
            ),
            registered_effect=RegisteredSemanticEffect(
                effect_kind=SemanticEffectKind.ATTACH,
                held_objects=(
                    RegisteredHeldObjectEffect(
                        expectation_id="primary",
                        relation=HeldObjectRelation.ATTACHED,
                        object_id=route.object_id,
                        slot_id="primary",
                    ),
                ),
            ),
        )


@dataclass(frozen=True, slots=True)
class _PickLowererFactory(RegisteredSemanticLowererFactory):
    """Resolve canonical objects for configured pickup targets."""

    revision: ClassVar[str] = "2"
    target_descriptor: ClassVar[SkillDescriptor] = PickUp.descriptor()
    routes: tuple[_PickRoute, ...]
    call_id: ClassVar[str] = _PICK_CALL_ID
    lowerer_type: ClassVar[type[_PickLowerer]] = _PickLowerer

    def __post_init__(self) -> None:
        if (
            type(self.routes) is not tuple
            or not self.routes
            or not all(type(route) is _PickRoute for route in self.routes)
        ):
            raise TypeError("routes must be a non-empty tuple of _PickRoute values.")
        if len({(route.object_id, route.target_id) for route in self.routes}) != len(
            self.routes
        ):
            raise ValueError("Configured Pick routes must be unique.")

    def create(
        self,
        *,
        simulation: object,
        robot: object,
        scene_registry: SceneRegistry,
        engine: AtomicActionEngine,
    ) -> RegisteredSemanticLowerer:
        """Resolve grasp semantics without choosing grasp regions or directions."""
        del simulation
        if engine.robot is not robot:
            raise ValueError("Configured Pick requires the engine's exact robot.")
        semantics = []
        for object_id in dict.fromkeys(route.object_id for route in self.routes):
            ref = scene_registry.resolve(object_id, expected_type=SceneObjectRef)
            grasp = scene_registry.resolve_affordance(
                ref, capability=GRASP_AFFORDANCE_CAPABILITY
            )
            semantics.append(scene_registry.object_semantics(ref, affordance=grasp))
        return self.lowerer_type(
            self.routes,
            tuple(semantics),
            engine.grasp_pose_generators,
        )


def make_pick_factory(
    routes: tuple[_PickRoute, ...], call_id: str
) -> _PickLowererFactory:
    """Bind a task policy alias to the baseline's class-level factory identity.

    Only an inert identifier is specialized; the lowerer implementation and
    target Atomic Skill remain unchanged. No executable code comes from data.
    """
    if call_id == _PICK_CALL_ID:
        return _PickLowererFactory(routes)
    if not call_id.startswith("gen_sim.pick."):
        raise ValueError("Task Pick aliases must use the gen_sim.pick namespace.")
    suffix = hashlib.sha256(call_id.encode("utf-8")).hexdigest()[:16]
    lowerer_type = type(
        f"_TaskPickLowerer_{suffix}",
        (_PickLowerer,),
        {"call_id": call_id, "__module__": __name__, "__slots__": ()},
    )
    factory_type = make_dataclass(
        f"_TaskPickFactory_{suffix}",
        [],
        bases=(_PickLowererFactory,),
        namespace={
            "call_id": call_id,
            "lowerer_type": lowerer_type,
            "__module__": __name__,
        },
        frozen=True,
        slots=True,
    )
    return factory_type(routes)


@dataclass(frozen=True, slots=True)
class _MoveHeldObjectRoute:
    """One exact declared object target."""

    object_id: str
    target_id: str
    pose: _AbsolutePoseTarget | _SceneEntityTarget

    def __post_init__(self) -> None:
        _identifier(self.object_id, field_name="object_id")
        _identifier(self.target_id, field_name="target_id")
        _configured_goal_pose(self.pose)
        object.__setattr__(self, "pose", self.pose.snapshot())


class _MoveHeldObjectLowerer(RegisteredSemanticLowerer):
    """Bind absolute or scene-relative targets to the shared transport goal."""

    call_id: ClassVar[str] = _MOVE_HELD_OBJECT_CALL_ID
    target_descriptor: ClassVar[SkillDescriptor] = MoveHeldObject.descriptor()
    preserves_symbolic_state: ClassVar[bool] = True

    def __init__(self, routes: tuple[_MoveHeldObjectRoute, ...]) -> None:
        self._routes = {route.target_id: route for route in routes}
        if not routes or len(self._routes) != len(routes):
            raise ValueError("MoveHeldObject targets must be non-empty and unique.")

    def _route(self, call: RegisteredSemanticCall) -> _MoveHeldObjectRoute:
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
            type(route.pose) is not _SceneEntityTarget
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
        route = self._route(call)
        return SemanticLowering(
            goal=HeldObjectPoseGoal(_configured_goal_pose(route.pose)),
            phase_protection=(
                RegisteredPhaseProtection(
                    kind=RegisteredPhaseProtectionKind.RETAIN,
                    held_object=RegisteredHeldObjectEffect(
                        expectation_id="primary",
                        relation=HeldObjectRelation.ATTACHED,
                        object_id=route.object_id,
                        slot_id="primary",
                    ),
                    active_segments=("transport",),
                )
                if self.phase_protection_kind is not None
                else None
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
        """Conservatively screen the nominal target, which is always permitted."""
        del bound, previous_target
        route = self._route(call)
        if picked_object.entity_id != route.object_id:
            return None
        return (
            SemanticObjectTarget(
                pose=(
                    SemanticPose(route.pose.position, route.pose.quaternion_wxyz)
                    if type(route.pose) is _AbsolutePoseTarget
                    else _configured_goal_pose(route.pose)
                )
            ),
        )


class _ProtectedMoveHeldObjectLowerer(_MoveHeldObjectLowerer):
    """Opt-in transport with a compiler-bound measured held-object guard."""

    phase_protection_kind: ClassVar[RegisteredPhaseProtectionKind] = (
        RegisteredPhaseProtectionKind.RETAIN
    )


@dataclass(frozen=True, slots=True)
class _MoveHeldObjectLowererFactory(RegisteredSemanticLowererFactory):
    """Validate canonical references for configured transport goals."""

    call_id: ClassVar[str] = _MOVE_HELD_OBJECT_CALL_ID
    revision: ClassVar[str] = "2"
    target_descriptor: ClassVar[SkillDescriptor] = MoveHeldObject.descriptor()
    routes: tuple[_MoveHeldObjectRoute, ...]
    phase_protection: str | None = None

    def __post_init__(self) -> None:
        if self.phase_protection is not None and (
            type(self.phase_protection) is not str
            or self.phase_protection != "held_object_v1"
        ):
            raise ValueError("phase_protection must be 'held_object_v1' or omitted.")
        if (
            type(self.routes) is not tuple
            or not self.routes
            or not all(type(route) is _MoveHeldObjectRoute for route in self.routes)
        ):
            raise TypeError(
                "routes must be a non-empty tuple of _MoveHeldObjectRoute values."
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
        """Retain only inert target declarations, never a live robot."""
        del simulation
        if engine.robot is not robot:
            raise ValueError("MoveHeldObject requires the engine's exact robot.")
        for route in self.routes:
            scene_registry.resolve(route.object_id, expected_type=SceneObjectRef)
            if type(route.pose) is _SceneEntityTarget:
                scene_registry.lookup(route.pose.entity_id)
        lowerer_type = (
            _ProtectedMoveHeldObjectLowerer
            if self.phase_protection is not None
            else _MoveHeldObjectLowerer
        )
        return lowerer_type(self.routes)


@dataclass(frozen=True, slots=True)
class _RelativePlaceRoute:
    """One configured object relation expressed in the world frame."""

    object_id: str
    reference_entity_id: str
    relation: str
    world_displacement: tuple[float, float, float]

    def __post_init__(self) -> None:
        for field_name in ("object_id", "reference_entity_id", "relation"):
            object.__setattr__(
                self,
                field_name,
                _identifier(getattr(self, field_name), field_name=field_name),
            )
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

    @property
    def selector(self) -> tuple[str, str, str]:
        """Return the semantic arguments selecting this immutable route."""
        return self.object_id, self.reference_entity_id, self.relation


@dataclass(frozen=True, slots=True)
class _RelativePlaceLowererFactory(RegisteredSemanticLowererFactory):
    """Create fresh relative-placement lowerers from canonical scene refs."""

    call_id: ClassVar[str] = _PLACE_RELATIVE_CALL_ID
    revision: ClassVar[str] = "1"
    target_descriptor: ClassVar[SkillDescriptor] = Place.descriptor()

    routes: tuple[_RelativePlaceRoute, ...]

    def __post_init__(self) -> None:
        if type(self.routes) is not tuple or not self.routes:
            raise ValueError("Relative Place routes must be a non-empty exact tuple.")
        if not all(type(route) is _RelativePlaceRoute for route in self.routes):
            raise TypeError("Relative Place routes must be _RelativePlaceRoute values.")
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
        """Canonicalize all object/reference IDs before constructing a lowerer."""
        del simulation
        if engine.robot is not robot:
            raise ValueError(
                "Relative Place lowerer requires the engine's exact robot."
            )
        routes: list[_RelativePlaceRoute] = []
        for route in self.routes:
            object_ref = scene_registry.resolve(
                route.object_id,
                expected_type=SceneObjectRef,
            )
            reference_ref = scene_registry.resolve(
                route.reference_entity_id,
                expected_type=SceneObjectRef,
            )
            routes.append(
                _RelativePlaceRoute(
                    object_id=object_ref.entity_id,
                    reference_entity_id=reference_ref.entity_id,
                    relation=route.relation,
                    world_displacement=route.world_displacement,
                )
            )
        return _RelativePlaceLowerer(tuple(routes))


class _CoordinatedTransportLowerer(RegisteredSemanticLowerer):
    """Lower one configured dual-arm object transport and release route."""

    call_id: ClassVar[str] = _COORDINATED_TRANSPORT_CALL_ID
    target_descriptor: ClassVar[SkillDescriptor] = CoordinatedPickment.descriptor()
    effect_contract_kind: ClassVar[SemanticEffectKind] = SemanticEffectKind.RELEASE
    release: ClassVar[bool] = True

    def __init__(
        self,
        routes: tuple[
            _CoordinatedTransportRoute | tuple[str, str, str, tuple[float, ...]],
            ...,
        ],
        semantics: tuple[ObjectSemantics, ...],
    ) -> None:
        if len(routes) != len(semantics):
            raise ValueError(
                "Coordinated transport routes and semantics must have equal length."
            )
        normalized = tuple(
            _coordinated_transport_route(route, index=index)
            for index, route in enumerate(routes)
        )
        self._routes = {
            (route.object_id, route.target_id): (route, object_semantics)
            for route, object_semantics in zip(normalized, semantics, strict=True)
        }

    def lower(
        self,
        call: RegisteredSemanticCall,
        *,
        context: PlanningContext,
        bound: BoundSemanticCall,
        option_template: ActionOptions,
    ) -> SemanticLowering:
        """Construct one late-bound coordinated object target."""
        del bound
        if type(option_template) is not CoordinatedPickmentOptions:
            raise TypeError(
                "Configured coordinated transport requires an exact "
                "CoordinatedPickmentOptions template."
            )
        if option_template.release is not self.release:
            raise ValueError(
                f"Configured {self.call_id} requires release={self.release}."
            )
        arguments = dict(call.arguments)
        if set(arguments) not in (
            {"object", "target"},
            {"object", "target", "world_displacement"},
        ):
            raise ValueError(
                f"{self.call_id} arguments must contain object, target, and an "
                "optional world_displacement."
            )
        route = (arguments["object"], arguments["target"])
        resolved = self._routes.get(route)
        if resolved is None:
            raise ValueError(
                f"{self.call_id} does not declare object-target route {route!r}."
            )
        route_cfg, semantics = resolved
        if "world_displacement" in arguments:
            declared = tuple(float(value) for value in arguments["world_displacement"])
            if declared != route_cfg.world_displacement:
                raise ValueError(
                    f"{self.call_id} world_displacement does not match its "
                    "configured route."
                )
        if route_cfg.world_displacement is not None:
            try:
                observed = context.scene.entities[route_cfg.object_id]
            except KeyError as exc:
                raise KeyError(
                    "Coordinated transport world displacement references "
                    f"unobserved object {route_cfg.object_id!r}."
                ) from exc
            if observed.confidence <= 0.0:
                raise ValueError(
                    "Coordinated transport requires positive observation "
                    f"confidence for {route_cfg.object_id!r}."
                )
            object_pose = observed.pose.to(
                device=context.robot.qpos.device,
                dtype=context.robot.qpos.dtype,
            )
            if object_pose.dim() == 2:
                object_pose = object_pose.unsqueeze(0).expand(
                    context.batch_size,
                    -1,
                    -1,
                )
            object_target_pose: torch.Tensor | SceneEntityPose = object_pose.clone()
            displacement = torch.tensor(
                route_cfg.world_displacement,
                dtype=object_target_pose.dtype,
                device=object_target_pose.device,
            )
            object_target_pose[:, :3, 3] += displacement
        else:
            assert route_cfg.reference_entity_id is not None
            assert route_cfg.relative_pose is not None
            object_target_pose = SceneEntityPose(
                route_cfg.reference_entity_id,
                relative_pose=torch.tensor(
                    route_cfg.relative_pose,
                    dtype=torch.float32,
                ).reshape(4, 4),
            )
        return SemanticLowering(
            goal=CoordinatedPickGoal(
                semantics=semantics,
                object_target_pose=object_target_pose,
            ),
            registered_effect=RegisteredSemanticEffect(
                effect_kind=(
                    SemanticEffectKind.RELEASE
                    if self.release
                    else SemanticEffectKind.ATTACH
                ),
                held_objects=tuple(
                    RegisteredHeldObjectEffect(
                        expectation_id=slot_id,
                        relation=(
                            HeldObjectRelation.DETACHED
                            if self.release
                            else HeldObjectRelation.ATTACHED
                        ),
                        object_id=route_cfg.object_id,
                        slot_id=slot_id,
                        allow_missing_detached_baseline=self.release,
                    )
                    for slot_id in ("left", "right")
                ),
            ),
        )


class _CoordinatedHoldLowerer(_CoordinatedTransportLowerer):
    """Lower one coordinated transport that retains both verified grasps."""

    call_id: ClassVar[str] = _COORDINATED_HOLD_CALL_ID
    effect_contract_kind: ClassVar[SemanticEffectKind] = SemanticEffectKind.ATTACH
    release: ClassVar[bool] = False


@dataclass(frozen=True, slots=True)
class _CoordinatedTransportLowererFactory(RegisteredSemanticLowererFactory):
    """Create configured dual-arm transport routes from canonical scene refs."""

    call_id: ClassVar[str] = _COORDINATED_TRANSPORT_CALL_ID
    revision: ClassVar[str] = "1"
    target_descriptor: ClassVar[SkillDescriptor] = CoordinatedPickment.descriptor()
    lowerer_type: ClassVar[type[_CoordinatedTransportLowerer]] = (
        _CoordinatedTransportLowerer
    )

    routes: tuple[
        _CoordinatedTransportRoute | tuple[str, str, str, tuple[float, ...]],
        ...,
    ]

    def __post_init__(self) -> None:
        if type(self.routes) is not tuple or not self.routes:
            raise ValueError(
                "Coordinated transport routes must be a non-empty exact tuple."
            )
        normalized = [
            _coordinated_transport_route(route, index=index)
            for index, route in enumerate(self.routes)
        ]
        selectors = [(route.object_id, route.target_id) for route in normalized]
        if len(set(selectors)) != len(selectors):
            raise ValueError("Coordinated transport routes must be unique.")
        object.__setattr__(self, "routes", tuple(normalized))

    def create(
        self,
        *,
        simulation: object,
        robot: object,
        scene_registry: SceneRegistry,
        engine: AtomicActionEngine,
    ) -> RegisteredSemanticLowerer:
        """Resolve grasp semantics and validate all live target references."""
        del simulation
        if engine.robot is not robot:
            raise ValueError(
                "Coordinated transport lowerer requires the engine's exact robot."
            )
        canonical_routes: list[_CoordinatedTransportRoute] = []
        semantics: list[ObjectSemantics] = []
        for route in self.routes:
            object_ref = scene_registry.resolve(
                route.object_id,
                expected_type=SceneObjectRef,
            )
            grasp_ref = scene_registry.resolve_affordance(
                object_ref,
                capability=GRASP_AFFORDANCE_CAPABILITY,
            )
            if route.world_displacement is not None:
                canonical_routes.append(
                    _CoordinatedTransportRoute(
                        object_id=object_ref.entity_id,
                        target_id=route.target_id,
                        world_displacement=route.world_displacement,
                    )
                )
            else:
                assert route.reference_entity_id is not None
                assert route.relative_pose is not None
                reference = scene_registry.lookup(route.reference_entity_id)
                canonical_routes.append(
                    _CoordinatedTransportRoute(
                        object_id=object_ref.entity_id,
                        target_id=route.target_id,
                        reference_entity_id=reference.ref.entity_id,
                        relative_pose=route.relative_pose,
                    )
                )
            semantics.append(
                scene_registry.object_semantics(
                    object_ref,
                    affordance=grasp_ref,
                )
            )
        return self.lowerer_type(
            tuple(canonical_routes),
            tuple(semantics),
        )


@dataclass(frozen=True, slots=True)
class _CoordinatedHoldLowererFactory(_CoordinatedTransportLowererFactory):
    """Create configured coordinated hold routes."""

    call_id: ClassVar[str] = _COORDINATED_HOLD_CALL_ID
    lowerer_type: ClassVar[type[_CoordinatedTransportLowerer]] = _CoordinatedHoldLowerer
