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
    PickUpOptions,
    PlaceGoal,
    SceneEntityPose,
)
from .drawer_binding import DrawerRoute
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
        return torch.as_tensor(np.stack(poses), dtype=obj.dtype, device=obj.device)


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
    target = observation.target_pose().to(context.robot.qpos) @ held.object_to_eef.to(
        context.robot.qpos
    )
    return replace(
        request,
        goal=PlaceGoal(
            target,
            tcp_symmetry=request.goal.tcp_symmetry,
        ),
        skill_options=replace(
            request.skill_options,
            # Clear the released object rather than waiting with the hand nearby.
            lift_height=max(request.skill_options.lift_height, 0.12),
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
        plan = super()._plan_request(prepared, current)
        if (
            prepared is request
            or not isinstance(request.goal, PlaceGoal)
            or not plan.plan_success.any()
        ):
            return plan
        observation = next(
            obs
            for obs in self._drawer_observations
            if obs.route.affordance == request.goal.xpos.entity_id
        )
        release = next(
            segment.start for segment in plan.segments if segment.name == "release"
        )
        dependencies = (observation.route.affordance, observation.route.binding.link_id)
        return replace(
            plan,
            scene_dependencies=tuple(
                dict.fromkeys((*plan.scene_dependencies, *dependencies))
            ),
            scene_dependency_monitor_until={
                **plan.scene_dependency_monitor_until,
                **{name: release for name in dependencies},
            },
        )
