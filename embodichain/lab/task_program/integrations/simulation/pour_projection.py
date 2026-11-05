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

"""Resolve declared Pour candidate inputs through shared Scene and Compute APIs."""

from __future__ import annotations
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import TYPE_CHECKING
import numpy as np
import torch
from embodichain.compute.geometry.pouring import pour_staging_pose
from embodichain.lab.sim.atomic_actions.bindings import JointPositionTarget
from embodichain.lab.sim.atomic_actions.pour_projection import PourGraspProjectionInputs
from embodichain.lab.sim.evidence import PhysicalObservationStamp
from embodichain.lab.sim.scene.pour_region import PourRegionGeometry
from embodichain.lab.sim.scene.pour_inputs import observed_pour_arm_origin
from embodichain.lab.sim.utility.grasp_geometry import ordered_pad_line_world

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import Robot
    from embodichain.lab.sim.atomic_actions import (
        PlanningContext,
        ResolvedActionRequest,
        PourOptions,
        MotionPolicy,
        RetainedPourGoal,
    )
    from embodichain.lab.sim.utility.grasp_geometry import RobotHandGeometryCfg
__all__ = ["ResolvedPourProjectionRoute", "SimulationPourProjectionProvider"]


@dataclass(frozen=True, slots=True, eq=False)
class ResolvedPourProjectionRoute:
    """Runtime operands resolved from exact compiled calls and selected presets.

    These values are not another independently configurable execution policy.
    The host resolves angle/sample count from the future Pour's effective options
    and MotionPolicy before constructing this value. Raw poses retain their
    source tensor dtype for the bounded candidate audit.
    """

    pick_id: str
    stage_id: str
    pour_id: str
    object_id: str
    receiver_id: str
    resource: str
    return_pose: torch.Tensor
    clearance_m: float
    angle_rad: float
    sample_count: int
    proxy_position_tolerance_m: float | None = None

    def __post_init__(self) -> None:
        for name in (
            "pick_id",
            "stage_id",
            "pour_id",
            "object_id",
            "receiver_id",
            "resource",
        ):
            value = getattr(self, name)
            if type(value) is not str or not value or value != value.strip():
                raise ValueError(f"{name} requires an exact identity.")
        if (
            len({self.pick_id, self.stage_id, self.pour_id}) != 3
            or self.object_id == self.receiver_id
        ):
            raise ValueError(
                "Pour projection participants and occurrences must be distinct."
            )
        if (
            not isinstance(self.return_pose, torch.Tensor)
            or self.return_pose.shape != (4, 4)
            or not self.return_pose.is_floating_point()
            or not torch.isfinite(self.return_pose).all()
        ):
            raise ValueError(
                "Pour projection requires an exact finite return-pose tensor."
            )
        object.__setattr__(self, "return_pose", self.return_pose.detach().cpu().clone())
        if (
            type(self.clearance_m) not in (int, float)
            or not math.isfinite(self.clearance_m)
            or self.clearance_m < 0
        ):
            raise ValueError("Pour staging clearance must be finite and nonnegative.")
        if (
            type(self.angle_rad) not in (int, float)
            or not math.isfinite(self.angle_rad)
            or self.angle_rad == 0
        ):
            raise ValueError("Resolved Pour angle must be finite and nonzero.")
        if type(self.sample_count) is not int or self.sample_count < 3:
            raise ValueError("Resolved Pour sample count must be at least three.")
        budget = self.proxy_position_tolerance_m
        if budget is not None and (
            type(budget) not in (int, float) or not math.isfinite(budget) or budget <= 0
        ):
            raise ValueError("Pour region budget must be finite and positive.")


class SimulationPourProjectionProvider:
    """Own geometry snapshots and adapt one fresh observed Pick into numeric inputs."""

    def __init__(
        self,
        robot: Robot,
        *,
        routes: tuple[ResolvedPourProjectionRoute, ...],
        vertices: Mapping[str, np.ndarray],
        gravity_world: tuple[float, float, float],
        hand_geometry_cfgs: Mapping[str, RobotHandGeometryCfg],
        region_geometries: Mapping[str, PourRegionGeometry] | None = None,
        read_source_stamp: Callable[[], PhysicalObservationStamp],
    ) -> None:
        """Borrow the engine robot and own explicitly resolved source data.

        Args:
            robot: Exact assembled engine robot.
            routes: Runtime routes already resolved from program/policy declarations.
            vertices: Cached source/receiver native-scaled local arrays in source order.
            gravity_world: Actual configured gravity, read by its Scene owner.
            hand_geometry_cfgs: Existing explicit engine hand calibrations.
            region_geometries: Exact Pick occurrence to owned inset geometry.
            read_source_stamp: Guarded reader for the current observed native source.
        """
        if type(routes) is not tuple or any(
            type(route) is not ResolvedPourProjectionRoute for route in routes
        ):
            raise TypeError("Pour projection routes must be immutable typed operands.")
        self.robot = robot
        self._routes = {route.pick_id: deepcopy(route) for route in routes}
        if len(self._routes) != len(routes):
            raise ValueError("Duplicate Pour projection Pick occurrence.")
        owned = {}
        for name, value in vertices.items():
            value = np.array(value, copy=True)
            if (
                type(name) is not str
                or not name
                or value.ndim != 2
                or value.shape[1] != 3
                or not len(value)
                or not np.isfinite(value).all()
            ):
                raise ValueError(
                    "Pour projection requires explicit finite source geometry."
                )
            value.setflags(write=False)
            owned[name] = value
        self._vertices = MappingProxyType(owned)
        gravity = np.asarray(gravity_world)
        if (
            gravity.shape != (3,)
            or not np.isfinite(gravity).all()
            or gravity[0] != 0
            or gravity[1] != 0
            or gravity[2] >= 0
        ):
            raise ValueError(
                "Pour projection requires actual finite negative-Z gravity."
            )
        self._gravity = tuple(float(value) for value in gravity)
        self._hands = deepcopy(dict(hand_geometry_cfgs))
        self._regions = MappingProxyType(dict(region_geometries or {}))
        needed = {
            route.pick_id
            for route in routes
            if route.proxy_position_tolerance_m is not None
        }
        if set(self._regions) != needed or any(
            type(value) is not PourRegionGeometry
            or value.proxy_position_tolerance_m
            != self._routes[key].proxy_position_tolerance_m
            or value.gravity_world != self._gravity
            for key, value in self._regions.items()
        ):
            raise ValueError(
                "Pour region data must match exact occurrence, gravity and budget."
            )
        if any(
            route.object_id not in owned or route.receiver_id not in owned
            for route in routes
        ):
            raise ValueError("Pour projection has missing source/receiver geometry.")
        for key, geometry in self._regions.items():
            route = self._routes[key]
            if route.receiver_id not in geometry.faces or any(
                identity not in geometry.vertices
                or not np.array_equal(
                    np.asarray(owned[identity], dtype=np.float64),
                    geometry.vertices[identity],
                )
                for identity in (route.object_id, route.receiver_id)
            ):
                raise ValueError(
                    "Pour projection region differs from its declared participant geometry."
                )

        if not callable(read_source_stamp):
            raise TypeError("Pour projection requires an observed source reader.")
        self._read_stamp = read_source_stamp

    @property
    def scene_vertices(self) -> Mapping[str, np.ndarray]:
        """Return read-only owned array copies for other shared service consumers."""
        result = {name: value.copy() for name, value in self._vertices.items()}
        for value in result.values():
            value.setflags(write=False)
        return MappingProxyType(result)

    @property
    def gravity_world(self) -> tuple[float, float, float]:
        """Return the unchanged configured world-gravity vector."""
        return self._gravity

    @property
    def region_geometries(self) -> Mapping[str, PourRegionGeometry]:
        """Return exact occurrence-owned immutable inset snapshots."""
        return self._regions

    def region_geometry(
        self, object_id: str, receiver_id: str, proxy_position_tolerance_m: float
    ) -> PourRegionGeometry | None:
        """Return the declared pair/budget snapshot without native reads.

        Args:
            object_id: Canonical source participant.
            receiver_id: Canonical receiver participant.
            proxy_position_tolerance_m: Exact declared inset budget.

        Returns:
            Existing immutable snapshot, or None only for an unselected pair.
            A known pair with a different staging mode/budget rejects instead of
            authorizing another native capture under changed declarations.
        """
        declared = False
        for key, route in self._routes.items():
            if route.object_id != object_id or route.receiver_id != receiver_id:
                continue
            declared = True
            geometry = self._regions.get(key)
            if (
                geometry is not None
                and geometry.proxy_position_tolerance_m == proxy_position_tolerance_m
            ):
                return geometry
        if declared:
            raise ValueError(
                "Pour geometry lookup conflicts with its declared staging mode or budget."
            )
        return None

    def _check_stamp(self, stamp: PhysicalObservationStamp) -> None:
        if (
            type(stamp) is not PhysicalObservationStamp
            or stamp.source_kind != "default.contacts"
            or self._read_stamp() != stamp
        ):
            raise ValueError(
                "Pour projection observation source changed or is unavailable."
            )

    def prepare_pour_goal(
        self,
        invocation_id: str,
        context: PlanningContext,
        *,
        options: PourOptions,
        motion_policy: MotionPolicy,
    ) -> RetainedPourGoal | None:
        """Resolve the selected Pour using the same owned geometry as its Pick.

        Args:
            invocation_id: Exact current execution occurrence.
            context: Synchronized B1/env0 physical observation.
            options: Effective options of this occurrence.
            motion_policy: Effective motion policy of this occurrence.

        Returns:
            Owned retained goal for a declared Pour, or None for an unbound call.
            Policy or observation drift raises before Atomic planning.
        """
        from embodichain.lab.sim.atomic_actions import RetainedPourGoal

        route = next(
            (
                value
                for value in self._routes.values()
                if value.pour_id == invocation_id
            ),
            None,
        )
        if route is None:
            return None
        if (
            options.rotate_angle != route.angle_rad
            or motion_policy.sample_count != route.sample_count
        ):
            raise ValueError("Pour execution differs from its resolved future policy.")
        if (
            context.batch_size != 1
            or context.env_ids.tolist() != [0]
            or context.robot.timestamp != context.scene.timestamp
        ):
            raise ValueError(
                "Pour execution requires a synchronized B1/env0 observation."
            )
        self._check_stamp(context.world_source_stamp)
        return RetainedPourGoal(
            source_entity_id=route.object_id,
            receiver_entity_id=route.receiver_id,
            source_vertices=torch.from_numpy(self._vertices[route.object_id].copy()),
            receiver_vertices=torch.from_numpy(
                self._vertices[route.receiver_id].copy()
            ),
            return_pose=route.return_pose,
            gravity_world=self._gravity,
            proxy_position_tolerance_m=route.proxy_position_tolerance_m,
        )

    def prepare(
        self, request: ResolvedActionRequest, context: PlanningContext
    ) -> PourGraspProjectionInputs | None:
        """Resolve geometry, poses and ordered jaw vector for one exact occurrence.

        Args:
            request: Bound ordinary Pour-purpose Pick invocation.
            context: Current B1/env0 robot and scene observations.

        Returns:
            Owned source-compatible numeric operands and audit identity, or None
            only for an undeclared occurrence. Required-but-unavailable operands
            raise, retaining the declared five-target qualification constraint.
        """
        route = self._routes.get(request.invocation_id)
        if route is None:
            return None
        if (
            request.skill_id != "pick_up"
            or request.goal.semantics.entity_id != route.object_id
            or request.skill_options.adaptive_grasp != "pour"
            or len(request.skill_options.downstream_object_target_poses) != 2
        ):
            raise ValueError(
                "Pour projection requires its exact source Pick and two-target suffix."
            )
        if (
            context.batch_size != 1
            or context.env_ids.tolist() != [0]
            or context.robot.timestamp != context.scene.timestamp
        ):
            raise ValueError(
                "Pour projection requires a synchronized B1/env0 observation."
            )
        self._check_stamp(context.world_source_stamp)

        def pose(entity_id):
            entity = context.scene.entities.get(entity_id)
            if entity is None or entity.pose is None:
                raise ValueError(
                    "Pour projection requires an observed exact rigid entity."
                )
            value = entity.pose
            if value.ndim == 2:
                value = value[None]
            if value.shape != (1, 4, 4):
                raise ValueError("Pour projection requires B1 observed poses.")
            return value[0].detach().cpu().numpy()

        source_pose, receiver_pose = pose(route.object_id), pose(route.receiver_id)
        motion = request.binding.endpoint("primary", "motion").require_target(
            JointPositionTarget
        )
        hand = request.binding.endpoint("primary", "grasp").require_target(
            JointPositionTarget
        )
        cfg = self._hands.get(hand.target_id)
        if (
            cfg is None
            or cfg.motion_part != motion.control_part
            or cfg.hand_part != hand.control_part
        ):
            raise ValueError(
                "Pour projection requires its exact declared hand calibration."
            )
        if route.pick_id in self._regions:
            geometry = self._regions[route.pick_id]
            staging = geometry.staging_pose(
                route.object_id,
                route.receiver_id,
                source_pose,
                receiver_pose,
                observed_pour_arm_origin(
                    self.robot, motion.control_part, context.env_ids
                ),
                clearance_m=route.clearance_m,
            )
            source, receiver = (
                geometry.vertices[route.object_id],
                geometry.vertices[route.receiver_id],
            )
        else:
            source, receiver = (
                self._vertices[route.object_id],
                self._vertices[route.receiver_id],
            )
            staging = pour_staging_pose(
                source,
                receiver,
                source_pose,
                receiver_pose,
                clearance_m=route.clearance_m,
                gravity_world=self._gravity,
            )
        tcp = self.robot.compute_fk(
            qpos=context.robot.qpos[:, list(motion.joint_ids)],
            name=motion.control_part,
            to_matrix=True,
        )
        jaw = ordered_pad_line_world(self.robot, cfg=cfg)
        local = (tcp[:, :3, :3].transpose(1, 2) @ jaw[..., None])[0, :, 0]
        self._check_stamp(context.world_source_stamp)
        return PourGraspProjectionInputs(
            pick_id=route.pick_id,
            stage_id=route.stage_id,
            pour_id=route.pour_id,
            object_id=route.object_id,
            receiver_id=route.receiver_id,
            resource=route.resource,
            source_vertices=source,
            receiver_vertices=receiver,
            staging=staging,
            receiver_pose=receiver_pose,
            return_pose=route.return_pose,
            pad_line_tcp=local,
            gravity_world=self._gravity,
            clearance_m=route.clearance_m,
            angle_rad=route.angle_rad,
            sample_count=route.sample_count,
            proxy_position_tolerance_m=route.proxy_position_tolerance_m,
        )
