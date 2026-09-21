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

"""Live drawer placement preparation without predictive acceptance gates."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import torch

from embodichain.lab.sim.atomic_actions import (
    AtomicActionEngine,
    GraspGoal,
    HeldObjectPoseGoal,
    JointPositionTarget,
    PickUpOptions,
    PlaceGoal,
    SceneEntityPose,
)
from .drawer_binding import DrawerRoute, ENTRY_RIM_CLEARANCE
from .drawer_geometry import placement_pose

__all__: list[str] = []


class DrawerObservation:
    """Keep placement geometry local; read actual link/object poses on every plan."""

    def __init__(self, route: DrawerRoute, simulation: Any) -> None:
        self.route = route
        self.art = simulation.get_articulation(route.binding.object_id)
        self.obj = simulation.get_rigid_object(route.object_id)
        self.vertices = self.obj.get_vertices(scale=True)[0].detach().cpu().numpy()

    def poses(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.obj.get_local_pose(to_matrix=True),
            self.art.get_link_pose(self.route.binding.link, to_matrix=True),
        )

    def target_pose(self) -> torch.Tensor:
        """Aim at the current drawer, without requiring the payload to fit."""
        obj, link = self.poses()
        poses = [
            placement_pose(
                self.vertices,
                obj[i, :3, :3].detach().cpu().numpy(),
                link[i].detach().cpu().numpy(),
                np.asarray(self.route.lower),
                np.asarray(self.route.upper),
            )
            for i in range(len(obj))
        ]
        target = torch.as_tensor(np.stack(poses), dtype=obj.dtype, device=obj.device)
        outward = (
            -link.new_tensor(self.route.binding.axis) * self.route.binding.axis_sign
        )
        depth = (
            (link.new_tensor(self.route.upper) - link.new_tensor(self.route.lower))
            * outward.abs()
        ).sum()
        index = list(self.art.joint_names).index(self.route.binding.joint)
        opening = (
            (self.art.get_qpos()[:, index] - self.route.binding.target("closed"))
            .abs()
            .to(link)
        )
        # Keep the reachable exposed-floor bias.  A deeper target makes the
        # held payload collide with the front panel before release.
        shift = (depth - opening).clamp_min(0) / 2
        direction = link[:, :3, :3] @ outward
        target[:, :3, 3] += direction * shift[:, None]
        return target

    def entry_poses(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Cross the front rim and release above it without forcing the hand inside."""
        target = self.target_pose()
        _, link = self.poses()
        outward = (
            -link.new_tensor(self.route.binding.axis) * self.route.binding.axis_sign
        )
        vertices = link.new_tensor(self.vertices)
        local_rotation = link[:, :3, :3].transpose(-1, -2) @ target[:, :3, :3]
        points = vertices @ local_rotation.transpose(-1, -2)
        low, high = link.new_tensor(self.route.lower), link.new_tensor(self.route.upper)
        local_target = (
            link[:, :3, :3].transpose(-1, -2)
            @ (target[:, :3, 3] - link[:, :3, 3]).unsqueeze(-1)
        ).squeeze(-1)
        rim = target.clone()
        # Clearance belongs to crossing the front board, not the inside endpoint.
        rise = (
            high[2] + 0.025 - (local_target[:, 2] + points[:, :, 2].amin(1))
        ).clamp_min(0)
        rim[:, :3, 3] += link[:, :3, 2] * rise[:, None]
        rim[:, :3, 3] += link[:, :3, 2] * ENTRY_RIM_CLEARANCE
        front = torch.maximum(low * outward, high * outward).sum()
        back_extent = (points @ outward).amin(1)
        distance = (front + 0.04 - back_extent - (local_target @ outward)).clamp_min(0)
        outside = rim.clone()
        outside[:, :3, 3] += (link[:, :3, :3] @ outward) * distance[:, None]
        # Open above the rim.  The payload can fall toward the drawer while the
        # hand avoids pushing the front panel or the upper drawer stack.
        release = rim.clone()
        release[:, :3, 3] += link[:, :3, 2] * 0.12
        return outside, rim, release

    def transport_pose(self) -> torch.Tensor:
        return self.entry_poses()[0]


def prepare_place(
    request: Any,
    context: Any,
    observations: tuple[DrawerObservation, ...],
) -> Any:
    """Refresh the drawer target and reuse the standard Place path generation."""
    if not isinstance(request.goal, PlaceGoal) or not isinstance(
        request.goal.xpos, SceneEntityPose
    ):
        return request
    observation = next(
        (
            obs
            for obs in observations
            if obs.route.affordance == request.goal.xpos.entity_id
        ),
        None,
    )
    if observation is None:
        return request
    endpoint = request.binding.endpoint("primary", "motion")
    held = context.task.get_held_object(endpoint.task_state_key)
    if held is None or held.semantics.entity_id != observation.route.object_id:
        raise ValueError("Drawer Place requires the declared object to be held.")
    object_to_eef = held.object_to_eef.to(context.robot.qpos)
    target = (
        torch.stack(observation.entry_poses(), dim=1).to(context.robot.qpos)
        @ object_to_eef[:, None]
    )
    return replace(
        request,
        goal=PlaceGoal(
            target,
            tcp_symmetry=request.goal.tcp_symmetry,
        ),
        skill_options=replace(
            request.skill_options,
            lift_height=0.0,
        ),
    )


class DrawerPlacementEngine(AtomicActionEngine):
    """Reuse normal planning/execution; only refresh the live drawer placement goal."""

    def __init__(
        self,
        *args: Any,
        drawer_observations: tuple[DrawerObservation, ...],
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._drawer_observations = drawer_observations

    def _plan_request(self, request: Any, context: Any = None) -> Any:
        current = self.initial_context() if context is None else context
        if isinstance(request.goal, HeldObjectPoseGoal) and isinstance(
            request.goal.object_target_pose, SceneEntityPose
        ):
            observation = next(
                (
                    obs
                    for obs in self._drawer_observations
                    if obs.route.affordance == request.goal.object_target_pose.entity_id
                ),
                None,
            )
            if observation is not None:
                endpoint = request.binding.endpoint("primary", "motion")
                held = current.task.get_held_object(endpoint.task_state_key)
                if (
                    held is None
                    or held.semantics.entity_id != observation.route.object_id
                ):
                    raise ValueError(
                        "Drawer transport requires the declared held object."
                    )
                target = observation.transport_pose().to(current.robot.qpos)
                prepared = replace(
                    request,
                    goal=HeldObjectPoseGoal(target),
                    motion_policy=replace(request.motion_policy, strategy="motion_gen"),
                )
                motion = endpoint.require_target(JointPositionTarget)
                with self.motion_generator.transport(observation, motion, current):
                    plan = super()._plan_request(prepared, current)
                return replace(
                    plan,
                    scene_dependencies=tuple(
                        dict.fromkeys(
                            (
                                *plan.scene_dependencies,
                                observation.route.binding.link_id,
                            )
                        )
                    ),
                    diagnostics=replace(
                        plan.diagnostics,
                        backend="curobo",
                        metadata={
                            **plan.diagnostics.metadata,
                            "drawer_transport": self.motion_generator.transport_metadata,
                        },
                    ),
                )
        prepared = prepare_place(request, current, self._drawer_observations)
        if (
            isinstance(request.goal, GraspGoal)
            and isinstance(request.skill_options, PickUpOptions)
            and any(
                obs.route.object_id == request.goal.semantics.entity_id
                for obs in self._drawer_observations
            )
        ):
            # Attempt acquisition first; placement is planned from the acquired grasp.
            prepared = replace(
                request,
                skill_options=replace(
                    request.skill_options, downstream_object_target_poses=()
                ),
            )
        if prepared is not request and isinstance(request.goal, PlaceGoal):
            with self.motion_generator.placement_path(prepared.goal.xpos):
                plan = super()._plan_request(prepared, current)
        else:
            plan = super()._plan_request(prepared, current)
        if (
            prepared is request
            or not isinstance(request.goal, PlaceGoal)
            or not plan.plan_success.any()
        ):
            return plan
        # Do not turn contact-induced drawer motion into a diagnostic replan.
        # The physical outcome, including whether the drawer can close, is the
        # result of this direct attempt.
        return plan
