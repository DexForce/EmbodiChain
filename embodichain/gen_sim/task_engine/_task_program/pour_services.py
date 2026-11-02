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
"""Explicit GenSim pour transport and retained-contact service declarations."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from types import MappingProxyType
from typing import ClassVar, Mapping

import numpy as np
import torch

from embodichain.lab.sim.atomic_actions import (
    ActionOptions,
    AtomicActionEngine,
    HeldObjectPoseGoal,
    JointPositionTarget,
    PlanningContext,
    SceneEntityPose,
)
from embodichain.lab.task_program.compiler.lowering import (
    RegisteredHeldObjectEffect,
    RegisteredSemanticLowerer,
    RegisteredSemanticEffect,
    SemanticLowering,
    SemanticObjectTarget,
)
from embodichain.lab.task_program.integrations._configured_services import (
    _PourLowerer,
    _PourLowererFactory,
    _identifier,
)
from embodichain.lab.task_program.semantics import (
    BoundSemanticCall,
    HeldObjectRelation,
    RegisteredSemanticCall,
    SceneObjectRef,
    SceneRegistry,
    SemanticEffectKind,
)
from .services import (
    _MoveHeldObjectLowerer,
    _MoveHeldObjectLowererFactory,
    _MoveHeldObjectRoute,
    _SceneEntityTarget,
)
from .pour_region import PourRegionGeometry, _position_budget

__all__ = [
    "PourStageRoute",
    "PourTransportFactory",
    "PourRegionTransportFactory",
    "PourRegionGeometry",
    "RetainedPourFactory",
    "native_vertices",
    "gravity_from_simulation",
    "active_arm_origin",
]


def _readonly(value: np.ndarray) -> np.ndarray:
    result = value.copy()
    result.setflags(write=False)
    return result


def native_vertices(scene_registry: SceneRegistry, entity_id: str) -> np.ndarray:
    """Read one B1 native body-local mesh, applying body scale exactly once."""
    registration = scene_registry.lookup(entity_id, expected_type=SceneObjectRef)
    provider = registration.state_provider
    entity = getattr(provider, "entity", None)
    getter = getattr(entity, "get_vertices", None)
    if not callable(getter):
        raise TypeError("Pour geometry requires a native rigid-object state provider.")
    if getattr(entity, "num_instances", None) != 1:
        raise ValueError("Pour native geometry currently requires one native instance.")
    vertices = getter(env_ids=[0], scale=True)
    if (
        not isinstance(vertices, torch.Tensor)
        or vertices.ndim != 3
        or vertices.shape[0] != 1
        or vertices.shape[1] == 0
        or vertices.shape[2] != 3
        or not vertices.is_floating_point()
        or not torch.isfinite(vertices).all()
    ):
        raise ValueError("Pour native vertices must be finite floating (1, N, 3).")
    return _readonly(vertices[0].detach().cpu().numpy())


def gravity_from_simulation(simulation: object) -> tuple[float, float, float]:
    """Read the actual Default/B1 configuration; never invent missing gravity."""
    if (
        getattr(simulation, "physics_backend", None) != "default"
        or getattr(simulation, "num_envs", None) != 1
    ):
        raise ValueError("Explicit pour services currently require Default/B1.")
    config = getattr(getattr(simulation, "sim_config", None), "physics_cfg", None)
    gravity = np.asarray(getattr(config, "gravity", None), dtype=float)
    if (
        gravity.shape != (3,)
        or not np.isfinite(gravity).all()
        or np.linalg.norm(gravity) <= 0
    ):
        raise ValueError("Pour services require a finite nonzero configured gravity.")
    return tuple(float(value) for value in gravity)


def active_arm_origin(
    robot: object, motion: JointPositionTarget, context: PlanningContext
) -> np.ndarray:
    """Read the compiled arm solver's current root, never guess the robot base."""
    if context.batch_size != 1 or context.env_ids.shape != (1,):
        raise ValueError("Pour region arm observation requires an exact B1 row.")
    solver = robot.get_solver(motion.control_part)
    name = getattr(solver, "root_link_name", None)
    if not isinstance(name, str) or not name:
        raise ValueError("Pour region requires the bound arm solver's root link.")
    pose = robot.get_link_pose(link_name=name, env_ids=context.env_ids, to_matrix=True)
    if not isinstance(pose, torch.Tensor) or pose.shape != (1, 4, 4):
        raise ValueError("Pour region requires an observed B1 native arm-root pose.")
    from .pour_geometry import _pose

    return _pose(pose[0].detach().cpu().numpy())[:3, 3].copy()


@dataclass(frozen=True, slots=True)
class PourStageRoute:
    """One exact receiver and nonnegative clearance for an ordinary pour stage."""

    object_id: str
    target_id: str
    receiver_id: str
    clearance_m: float

    def __post_init__(self) -> None:
        for name in ("object_id", "target_id", "receiver_id"):
            _identifier(getattr(self, name), field_name=name)
        if self.object_id == self.receiver_id:
            raise ValueError("A pour source cannot be its own receiver.")
        if (
            type(self.clearance_m) not in (int, float)
            or not math.isfinite(self.clearance_m)
            or self.clearance_m < 0
        ):
            raise ValueError("clearance_m must be finite and nonnegative.")


class _PourTransportLowerer(_MoveHeldObjectLowerer):
    def __init__(
        self,
        routes: tuple[_MoveHeldObjectRoute, ...],
        staging_routes: tuple[PourStageRoute, ...],
        vertices: dict[str, np.ndarray],
        identifiers: dict[str, str],
        gravity: tuple[float, float, float],
    ) -> None:
        super().__init__(routes)
        self._stages = {route.target_id: route for route in staging_routes}
        self._vertices, self._identifiers = vertices, identifiers
        self._gravity = gravity

    @property
    def staging_routes(self) -> tuple[PourStageRoute, ...]:
        """Return the immutable configured stage declarations."""
        return tuple(self._stages.values())

    @property
    def scene_vertices(self) -> Mapping[str, np.ndarray]:
        """Return detached read-only native-scaled local geometry copies."""
        return MappingProxyType(
            {key: _readonly(value) for key, value in self._vertices.items()}
        )

    @property
    def gravity_world(self) -> tuple[float, float, float]:
        """Return the actual configured world-frame gravity."""
        return self._gravity

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
        route = self._route(call)
        stage = self._stages.get(route.target_id)
        if stage is None:
            return lowered
        if context.batch_size != 1 or context.env_ids.tolist() != [0]:
            raise ValueError("Pour staging currently supports only Default/B1 row 0.")
        poses = []
        for identifier in (stage.object_id, stage.receiver_id):
            state = context.scene.entities[self._identifiers[identifier]]
            if state.confidence <= 0:
                raise ValueError("Pour staging requires observed source and receiver.")
            pose = state.pose[0] if state.pose.ndim == 3 else state.pose
            poses.append(pose.detach().cpu().numpy())
        target = self._staging_target(stage, poses, bound, context)
        matrix = torch.as_tensor(
            target, dtype=context.robot.qpos.dtype, device=context.robot.qpos.device
        )
        receiver = torch.as_tensor(poses[1], dtype=matrix.dtype, device=matrix.device)
        return replace(
            lowered,
            goal=HeldObjectPoseGoal(
                SceneEntityPose(
                    self._identifiers[stage.receiver_id],
                    world_orientation=matrix[:3, :3],
                    world_displacement=matrix[:3, 3] - receiver[:3, 3],
                )
            ),
        )

    def _staging_target(
        self,
        stage: PourStageRoute,
        poses: list[np.ndarray],
        bound: BoundSemanticCall,
        context: PlanningContext,
    ) -> np.ndarray:
        del bound, context
        from .pour_geometry import staging_pose

        return staging_pose(
            self._vertices[stage.object_id],
            self._vertices[stage.receiver_id],
            *poses,
            clearance_m=stage.clearance_m,
            gravity_world=self._gravity,
        )

    def pick_lookahead_targets(
        self,
        call: RegisteredSemanticCall,
        *,
        picked_object: SceneObjectRef,
        bound: BoundSemanticCall,
        previous_target: SemanticObjectTarget | None,
    ) -> tuple[SemanticObjectTarget, ...] | None:
        route = self._route(call)
        if route.target_id in self._stages:
            return () if picked_object.entity_id == route.object_id else None
        return super().pick_lookahead_targets(
            call,
            picked_object=picked_object,
            bound=bound,
            previous_target=previous_target,
        )


@dataclass(frozen=True, slots=True)
class PourTransportFactory(_MoveHeldObjectLowererFactory):
    """Opt into measured-entry staging without changing ordinary return routes."""

    revision: ClassVar[str] = "pour-transport-1"
    staging_routes: tuple[PourStageRoute, ...]

    def __post_init__(self) -> None:
        _MoveHeldObjectLowererFactory.__post_init__(self)
        if (
            type(self.staging_routes) is not tuple
            or not self.staging_routes
            or not all(type(route) is PourStageRoute for route in self.staging_routes)
        ):
            raise TypeError(
                "staging_routes must be a nonempty tuple of PourStageRoute."
            )
        routes = {route.target_id: route for route in self.routes}
        if len({route.target_id for route in self.staging_routes}) != len(
            self.staging_routes
        ):
            raise ValueError("Pour staging target IDs must be unique.")
        for stage in self.staging_routes:
            route = routes.get(stage.target_id)
            if (
                route is None
                or route.object_id != stage.object_id
                or type(route.pose) is not _SceneEntityTarget
                or route.pose.entity_id != stage.receiver_id
            ):
                raise ValueError(
                    "Pour staging must match its exact configured object/receiver route."
                )

    def create(
        self,
        *,
        simulation: object,
        robot: object,
        scene_registry: SceneRegistry,
        engine: AtomicActionEngine,
    ) -> RegisteredSemanticLowerer:
        """Capture only exact native geometry and gravity, never plans or grasps."""
        _MoveHeldObjectLowererFactory.create(
            self,
            simulation=simulation,
            robot=robot,
            scene_registry=scene_registry,
            engine=engine,
        )
        gravity = gravity_from_simulation(simulation)
        identifiers = {
            identifier: scene_registry.resolve(
                identifier, expected_type=SceneObjectRef
            ).entity_id
            for route in self.staging_routes
            for identifier in (route.object_id, route.receiver_id)
        }
        vertices = {
            identifier: native_vertices(scene_registry, identifier)
            for identifier in identifiers
        }
        return _PourTransportLowerer(
            self.routes, self.staging_routes, vertices, identifiers, gravity
        )


class _PourRegionTransportLowerer(_PourTransportLowerer):
    def __init__(
        self,
        source: _PourTransportLowerer,
        region_geometry: PourRegionGeometry,
        robot: object,
    ) -> None:
        super().__init__(
            tuple(source._routes.values()),
            source.staging_routes,
            source._vertices,
            source._identifiers,
            source.gravity_world,
        )
        self.region_geometry, self._robot = region_geometry, robot

    def _staging_target(
        self,
        stage: PourStageRoute,
        poses: list[np.ndarray],
        bound: BoundSemanticCall,
        context: PlanningContext,
    ) -> np.ndarray:
        if bound is None or bound.binding.action_binding is None:
            raise ValueError("Pour region requires its compiled motion binding.")
        motion = bound.binding.action_binding.endpoint(
            "primary", "motion"
        ).require_target(JointPositionTarget)
        return self.region_geometry.staging_pose(
            stage.object_id,
            stage.receiver_id,
            *poses,
            active_arm_origin(self._robot, motion, context),
            clearance_m=stage.clearance_m,
        )


@dataclass(frozen=True, slots=True)
class PourRegionTransportFactory(PourTransportFactory):
    """Fresh explicit inset-region policy; legacy transport identity is unchanged."""

    revision: ClassVar[str] = "pour-region-transport-1"
    proxy_position_tolerance_m: float

    def __post_init__(self) -> None:
        PourTransportFactory.__post_init__(self)
        _position_budget(self.proxy_position_tolerance_m)

    def create(
        self,
        *,
        simulation: object,
        robot: object,
        scene_registry: SceneRegistry,
        engine: AtomicActionEngine,
    ) -> RegisteredSemanticLowerer:
        """Capture paired native vertices/faces once, without plans or IK."""
        source = PourTransportFactory.create(
            self,
            simulation=simulation,
            robot=robot,
            scene_registry=scene_registry,
            engine=engine,
        )
        faces = {}
        for identifier in dict.fromkeys(
            route.receiver_id for route in self.staging_routes
        ):
            registration = scene_registry.lookup(
                identifier, expected_type=SceneObjectRef
            )
            entity = getattr(registration.state_provider, "entity", None)
            getter = getattr(entity, "get_triangles", None)
            if not callable(getter):
                raise TypeError("Pour region requires native receiver triangles.")
            value = getter(env_ids=[0])
            if (
                not isinstance(value, torch.Tensor)
                or value.ndim != 3
                or value.shape[0] != 1
                or value.shape[2] != 3
                or value.dtype == torch.bool
                or value.is_floating_point()
                or value.is_complex()
            ):
                raise ValueError(
                    "Pour native receiver faces must be integer (1, F, 3)."
                )
            faces[identifier] = value[0].detach().cpu().numpy()
        geometry = PourRegionGeometry(
            source._vertices,
            faces,
            source.gravity_world,
            self.proxy_position_tolerance_m,
        )
        return _PourRegionTransportLowerer(source, geometry, robot)


class _RetainedPourLowerer(_PourLowerer):
    effect_contract_kind: ClassVar[SemanticEffectKind] = SemanticEffectKind.ATTACH

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
    """Explicitly guard the already-held source during the canonical pour phase."""

    revision: ClassVar[str] = "retained-pour-1"

    def create(
        self,
        *,
        simulation: object,
        robot: object,
        scene_registry: SceneRegistry,
        engine: AtomicActionEngine,
    ) -> RegisteredSemanticLowerer:
        """Keep legacy source-semantic validation and add no private grasp state."""
        gravity_from_simulation(simulation)
        _PourLowererFactory.create(
            self,
            simulation=simulation,
            robot=robot,
            scene_registry=scene_registry,
            engine=engine,
        )
        return _RetainedPourLowerer(self.object_id)
