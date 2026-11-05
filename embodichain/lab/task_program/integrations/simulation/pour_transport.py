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

"""Configured observed Pour staging through shared geometry and motion owners."""

from __future__ import annotations
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import ClassVar, Mapping
import math
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
    RegisteredSemanticLowerer,
    SemanticLowering,
    SemanticObjectTarget,
)
from embodichain.lab.task_program.semantics import (
    BoundSemanticCall,
    RegisteredSemanticCall,
    SceneObjectRef,
    SceneRegistry,
    SceneManifest,
)
from embodichain.lab.sim.scene.pour_inputs import (
    capture_pour_vertices,
    capture_pour_faces,
    configured_pour_gravity,
    observed_pour_arm_origin,
)
from embodichain.compute.geometry.pour_region import _position_budget
from embodichain.lab.sim.scene.pour_region import (
    PourRegionGeometry,
    PourGeometryProvider,
)
from .transport import (
    _RoutedMoveHeldObjectLowerer,
    RoutedMoveHeldObjectLowererFactory,
    HeldObjectTransportRoute,
)
from .targets import SceneEntityTarget, _identifier

__all__ = ["PourStageRoute", "PourTransportFactory", "PourRegionTransportFactory"]


def _geometry_provider(engine: AtomicActionEngine) -> PourGeometryProvider | None:
    provider = getattr(
        getattr(engine, "planning_services", None),
        "pour_grasp_projection_provider",
        None,
    )
    return provider if isinstance(provider, PourGeometryProvider) else None


def _readonly(value: np.ndarray) -> np.ndarray:
    result = value.copy()
    result.setflags(write=False)
    return result


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


class _PourTransportLowerer(_RoutedMoveHeldObjectLowerer):
    def __init__(
        self,
        routes: tuple[HeldObjectTransportRoute, ...],
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
        from embodichain.compute.geometry.pouring import (
            pour_staging_pose as staging_pose,
        )

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
class PourTransportFactory(RoutedMoveHeldObjectLowererFactory):
    """Opt into measured-entry staging without changing ordinary return routes."""

    revision: ClassVar[str] = "pour-transport-1"
    staging_routes: tuple[PourStageRoute, ...]

    def __post_init__(self) -> None:
        RoutedMoveHeldObjectLowererFactory.__post_init__(self)
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
                or type(route.pose) is not SceneEntityTarget
                or route.pose.entity_id != stage.receiver_id
            ):
                raise ValueError(
                    "Pour staging must match its exact configured object/receiver route."
                )

    def validate_scene_declaration(self, scene: SceneManifest) -> None:
        """Require exact rigid stage participants before any native capture.

        Args:
            scene: Canonical immutable scene declarations.
        """
        RoutedMoveHeldObjectLowererFactory.validate_scene_declaration(self, scene)
        for stage in self.staging_routes:
            for entity_id in (stage.object_id, stage.receiver_id):
                scene.resolve(entity_id, expected_type=SceneObjectRef)

    def create(
        self,
        *,
        simulation: object,
        robot: object,
        scene_registry: SceneRegistry,
        engine: AtomicActionEngine,
    ) -> RegisteredSemanticLowerer:
        """Capture only exact native geometry and gravity, never plans or grasps."""
        RoutedMoveHeldObjectLowererFactory.create(
            self,
            simulation=simulation,
            robot=robot,
            scene_registry=scene_registry,
            engine=engine,
        )
        gravity = configured_pour_gravity(simulation)
        identifiers = {
            identifier: scene_registry.resolve(
                identifier, expected_type=SceneObjectRef
            ).entity_id
            for route in self.staging_routes
            for identifier in (route.object_id, route.receiver_id)
        }
        provider = _geometry_provider(engine)
        cached = {} if provider is None else provider.scene_vertices
        if provider is not None and tuple(provider.gravity_world) != gravity:
            raise ValueError("Cached Pour geometry has a different configured gravity.")
        vertices = {
            identifier: (
                _readonly(cached[identifier])
                if identifier in cached
                else capture_pour_vertices(
                    scene_registry.lookup(
                        identifier, expected_type=SceneObjectRef
                    ).state_provider.entity
                )
            )
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
            observed_pour_arm_origin(self._robot, motion.control_part, context.env_ids),
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
        provider = _geometry_provider(engine)
        cached_regions = (
            []
            if provider is None
            else [
                provider.region_geometry(
                    stage.object_id, stage.receiver_id, self.proxy_position_tolerance_m
                )
                for stage in self.staging_routes
            ]
        )
        faces = {}
        for stage, geometry in zip(self.staging_routes, cached_regions, strict=False):
            if geometry is None:
                continue
            if (
                type(geometry) is not PourRegionGeometry
                or geometry.proxy_position_tolerance_m
                != self.proxy_position_tolerance_m
                or geometry.gravity_world != source.gravity_world
                or stage.receiver_id not in geometry.faces
                or any(
                    key not in geometry.vertices
                    or not np.array_equal(geometry.vertices[key], source._vertices[key])
                    for key in (stage.object_id, stage.receiver_id)
                )
            ):
                raise ValueError(
                    "Cached Pour region differs from its declared participants or budget."
                )
            faces[stage.receiver_id] = geometry.faces[stage.receiver_id]
        for identifier in dict.fromkeys(
            route.receiver_id for route in self.staging_routes
        ):
            if identifier in faces:
                continue
            registration = scene_registry.lookup(
                identifier, expected_type=SceneObjectRef
            )
            faces[identifier] = capture_pour_faces(registration.state_provider.entity)
        geometry = PourRegionGeometry(
            source._vertices,
            faces,
            source.gravity_world,
            self.proxy_position_tolerance_m,
        )
        return _PourRegionTransportLowerer(source, geometry, robot)
