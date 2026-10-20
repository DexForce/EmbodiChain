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

"""GenSim-owned drawer placement preparation and bounded candidate planning."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import torch

from embodichain.lab.sim.atomic_actions import (
    ActionPlan,
    GraspGoal,
    HeldObjectPoseGoal,
    JointPositionTarget,
    PickUpOptions,
    PlanTransform,
    PlanningFailure,
    PlaceGoal,
    SceneEntityPose,
    SlideGoal,
)
from .drawer_binding import DrawerRoute, ENTRY_RIM_CLEARANCE
from .drawer_geometry import (
    drawer_containment_report,
    placement_pose_candidates,
)
from .invocation_policy import GenSimActionEngine

__all__: list[str] = []

DRAWER_MAX_FULL_PLAN_CANDIDATES = 2
_DRAWER_PREFILTER_STAGES = ("outside", "rim", "release")


def _prefilter_place_candidates(
    request: Any, context: Any, candidates: tuple[Any, ...], robot: Any
) -> tuple[tuple[tuple[int, Any], ...], list[dict[str, Any]]]:
    """Rank drawer candidates with sequential endpoint IK before full planning.

    This is deliberately kinematic only: it does not reject by payload size,
    collision, velocity, or physical containment.  A failed prefilter is a
    ranking signal, not proof that the complete planner cannot find a path.
    """
    if context.batch_size != 1:
        raise ValueError(
            "GenSim drawer candidate planning currently requires batch_size=1."
        )
    endpoint = request.binding.endpoint("primary", "motion")
    motion = endpoint.require_target(JointPositionTarget)
    seed = context.robot.qpos[:, list(motion.joint_ids)].clone()
    records: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        candidate_seed = seed.clone()
        solved = 0
        failure_stage: str | None = None
        failure_reason: str | None = None
        failure_detail: str | None = None
        joint_distance = 0.0
        target = candidate.goal.xpos
        for stage_index, stage_name in enumerate(_DRAWER_PREFILTER_STAGES):
            try:
                result = robot.compute_ik(
                    pose=target[:, stage_index],
                    name=motion.control_part,
                    joint_seed=candidate_seed,
                )
                if result is None:
                    raise RuntimeError("IK solver is unavailable")
                success, qpos = result
                success = torch.as_tensor(
                    success, dtype=torch.bool, device=candidate_seed.device
                ).reshape(-1)
                qpos = torch.as_tensor(
                    qpos, dtype=candidate_seed.dtype, device=candidate_seed.device
                )
                if qpos.ndim == 3:
                    qpos = qpos[:, 0]
                if qpos.shape != candidate_seed.shape:
                    raise ValueError(
                        "IK prefilter returned qpos with shape "
                        f"{tuple(qpos.shape)}, expected {tuple(candidate_seed.shape)}"
                    )
                if not bool(success[0]) or not torch.isfinite(qpos).all():
                    failure_stage = stage_name
                    failure_reason = "ik_failed"
                    break
                joint_distance += float(
                    torch.linalg.vector_norm(qpos - candidate_seed).item()
                )
                candidate_seed = qpos
                solved += 1
            except Exception as exc:  # pragma: no cover - backend-specific failure
                failure_stage = stage_name
                failure_reason = f"ik_error:{type(exc).__name__}"
                failure_detail = str(exc)
                break
        records.append(
            {
                "candidate_id": f"drawer_candidate_{index}",
                "index": index,
                "prefilter": "endpoint_ik",
                "solved_stages": solved,
                "accepted": solved == len(_DRAWER_PREFILTER_STAGES),
                "failure_stage": failure_stage,
                "failure_reason": failure_reason,
                "failure_detail": failure_detail,
                "joint_distance": joint_distance,
            }
        )

    ranked = sorted(
        range(len(candidates)),
        key=lambda index: (
            not records[index]["accepted"],
            -records[index]["solved_stages"],
            records[index]["joint_distance"],
            index,
        ),
    )
    survivors = [index for index in ranked if records[index]["accepted"]]
    selected = survivors[:DRAWER_MAX_FULL_PLAN_CANDIDATES]
    if not selected:
        # Preserve one complete-planner attempt even when the cheap screen has
        # no survivor; endpoint IK is intentionally not a hard rejection gate.
        selected = ranked[:1]
    return tuple((index, candidates[index]) for index in selected), records


def _validate_drawer_plan_velocity(plan: Any, robot: Any) -> Any:
    """Apply the final full-robot velocity gate to a drawer Place plan."""
    from .planning_probe import _validate_final_plan_velocity

    velocity = _validate_final_plan_velocity(plan, robot)
    accepted = plan.plan_success & velocity["valid_mask"]
    if accepted.all():
        return plan
    failure = plan.diagnostics.failure or PlanningFailure(
        "drawer_place_velocity_limit", retryable=True
    )
    return replace(
        plan,
        plan_success=accepted,
        diagnostics=replace(
            plan.diagnostics,
            failure=failure,
            metadata={
                **plan.diagnostics.metadata,
                "drawer_final_velocity": {
                    "valid_mask": velocity["valid_mask"].detach().cpu().tolist(),
                    "diagnostics": velocity["diagnostics"],
                },
            },
        ),
    )


def _apply_drawer_plan_transform(
    engine: Any,
    request: Any,
    context: Any,
    plan: Any,
    plan_transform: PlanTransform | None,
) -> Any:
    """Apply an expansion transform once, after candidate selection."""
    if plan_transform is None:
        return plan
    transformed = plan_transform(request, context, plan.snapshot())
    if not isinstance(transformed, ActionPlan):
        raise TypeError("plan_transform must return an ActionPlan.")
    owned = transformed.snapshot()
    engine._validate_plan(owned, context, request)
    return owned


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
        """Return the first geometric payload target for compatibility."""
        return self._target_poses()[0]

    def _target_poses(self) -> tuple[torch.Tensor, ...]:
        """Build bounded payload targets from live drawer and object poses."""
        obj, link = self.poses()
        targets: list[torch.Tensor] = []
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
        for row in range(len(obj)):
            poses = placement_pose_candidates(
                self.vertices,
                obj[row, :3, :3].detach().cpu().numpy(),
                link[row].detach().cpu().numpy(),
                np.asarray(self.route.lower),
                np.asarray(self.route.upper),
            )
            row_targets: list[torch.Tensor] = []
            for pose in poses:
                target = torch.as_tensor(pose, dtype=obj.dtype, device=obj.device)
                target[:3, 3] += direction[row] * shift[row]
                row_targets.append(target)
            if not row_targets:
                raise ValueError(
                    f"Drawer payload {self.route.object_id!r} produced no "
                    "finite placement candidate."
                )
            targets.append(torch.stack(row_targets))
        width = max(item.shape[0] for item in targets)
        padded = obj.new_zeros((len(targets), width, 4, 4))
        padded[:] = torch.eye(4, dtype=obj.dtype, device=obj.device)
        for row, values in enumerate(targets):
            padded[row, : values.shape[0]] = values
            if values.shape[0] < width:
                padded[row, values.shape[0] :] = values[-1]
        return tuple(padded[:, index] for index in range(width))

    def entry_poses(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Cross the front rim and release above it without forcing the hand inside."""
        target = self.target_pose()
        return self._entry_poses_for_target(target)

    def _entry_poses_for_target(
        self, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build outside/rim/release poses for one payload target."""
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

    def entry_pose_candidates(self) -> tuple[tuple[torch.Tensor, ...], ...]:
        """Return finite outside/rim/release candidates for each environment."""
        return tuple(
            self._entry_poses_for_target(target) for target in self._target_poses()
        )

    def containment_report(self) -> dict[str, object]:
        """Read a live link-local mesh containment report for the payload."""
        obj, link = self.poses()
        report = drawer_containment_report(
            self.vertices,
            obj[0].detach().cpu().numpy(),
            link[0].detach().cpu().numpy(),
            np.asarray(self.route.lower),
            np.asarray(self.route.upper),
        )
        return report

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
    target = torch.stack(observation.entry_poses(), dim=1).to(context.robot.qpos)
    target = target @ object_to_eef[:, None]
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


def prepare_place_candidates(
    request: Any,
    context: Any,
    observations: tuple[DrawerObservation, ...],
) -> tuple[Any, ...]:
    """Create bounded drawer Place requests without touching shared Place."""
    if not isinstance(request.goal, PlaceGoal) or not isinstance(
        request.goal.xpos, SceneEntityPose
    ):
        return (request,)
    observation = next(
        (
            obs
            for obs in observations
            if obs.route.affordance == request.goal.xpos.entity_id
        ),
        None,
    )
    if observation is None:
        return (request,)
    endpoint = request.binding.endpoint("primary", "motion")
    held = context.task.get_held_object(endpoint.task_state_key)
    if held is None or held.semantics.entity_id != observation.route.object_id:
        raise ValueError("Drawer Place requires the declared object to be held.")
    object_to_eef = held.object_to_eef.to(context.robot.qpos)
    requests = []
    for entry in observation.entry_pose_candidates():
        target = torch.stack(entry, dim=1).to(context.robot.qpos)
        requests.append(
            replace(
                request,
                goal=PlaceGoal(
                    target @ object_to_eef[:, None],
                    tcp_symmetry=request.goal.tcp_symmetry,
                ),
                skill_options=replace(request.skill_options, lift_height=0.0),
            )
        )
    return tuple(requests)


class DrawerPlacementEngine(GenSimActionEngine):
    """Reuse normal planning/execution; only refresh the live drawer placement goal."""

    def __init__(
        self,
        *args: Any,
        drawer_observations: tuple[DrawerObservation, ...],
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._drawer_observations = drawer_observations

    def _plan_request(
        self,
        request: Any,
        context: Any = None,
        *,
        plan_transform: PlanTransform | None = None,
    ) -> Any:
        current = self.initial_context() if context is None else context
        if (
            self._drawer_observations
            and isinstance(request.goal, (PlaceGoal, HeldObjectPoseGoal, SlideGoal))
            and getattr(current, "batch_size", 1) != 1
        ):
            raise ValueError(
                "GenSim drawer runtime currently supports batch_size=1 only."
            )
        if (
            isinstance(request.goal, SlideGoal)
            and request.goal.joint_target is not None
        ):
            target = request.goal.joint_target
            for observation in self._drawer_observations:
                binding = observation.route.binding
                if (
                    binding.object_id == target.articulation_id
                    and binding.joint == target.joint_name
                    and abs(target.position - binding.target("closed"))
                    <= max(target.tolerance, 0.01)
                ):
                    report = observation.containment_report()
                    if not report["accepted"]:
                        raise ValueError(
                            "Drawer close rejected: payload containment failed; "
                            f"report={report}"
                        )
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
                    plan = super()._plan_request(
                        prepared, current, plan_transform=plan_transform
                    )
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
        prepared = request
        if (
            isinstance(request.goal, GraspGoal)
            and isinstance(request.skill_options, PickUpOptions)
            and any(
                obs.route.object_id == request.goal.semantics.entity_id
                and any(
                    isinstance(target, SceneEntityPose)
                    and target.entity_id == obs.route.affordance
                    for target in request.skill_options.downstream_object_target_poses
                )
                for obs in self._drawer_observations
            )
        ):
            # Only this Pick's explicit drawer continuation defers its place probe.
            # A later drawer task on the same object must not alter ordinary picks.
            prepared = replace(
                request,
                skill_options=replace(
                    request.skill_options, downstream_object_target_poses=()
                ),
            )
        candidate_mode = isinstance(request.goal, PlaceGoal) and any(
            obs.route.affordance == request.goal.xpos.entity_id
            for obs in self._drawer_observations
            if isinstance(request.goal.xpos, SceneEntityPose)
        )
        if not candidate_mode:
            return super()._plan_request(
                prepared, current, plan_transform=plan_transform
            )

        all_prepared_candidates = prepare_place_candidates(
            request, current, self._drawer_observations
        )
        if not all_prepared_candidates:
            raise ValueError("Drawer placement produced no candidate targets.")
        selected_candidates, prefilter_results = _prefilter_place_candidates(
            request, current, all_prepared_candidates, self.robot
        )
        plans = []
        candidate_results = []
        for index, prepared in selected_candidates:
            with self.motion_generator.placement_path(prepared.goal.xpos):
                plan = super()._plan_request(prepared, current, plan_transform=None)
            plan = _validate_drawer_plan_velocity(plan, self.robot)
            plan = replace(
                plan,
                diagnostics=replace(
                    plan.diagnostics,
                    metadata={
                        **plan.diagnostics.metadata,
                        "drawer_placement_candidate": index,
                        "drawer_placement_candidate_count": len(
                            all_prepared_candidates
                        ),
                    },
                ),
            )
            candidate_results.append(
                {
                    "candidate_id": f"drawer_candidate_{index}",
                    "index": index,
                    "plan_success": plan.plan_success.detach().cpu().tolist(),
                    "failure": (
                        None
                        if plan.diagnostics.failure is None
                        else plan.diagnostics.failure.code
                    ),
                }
            )
            plans.append(plan)
            if bool(plan.plan_success.all().item()):
                break

        plan = plans[-1]
        selected_index = candidate_results[-1]["index"]
        selected_request = all_prepared_candidates[selected_index]
        plan = _apply_drawer_plan_transform(
            self, selected_request, current, plan, plan_transform
        )
        plan = _validate_drawer_plan_velocity(plan, self.robot)
        plan = replace(
            plan,
            diagnostics=replace(
                plan.diagnostics,
                metadata={
                    **plan.diagnostics.metadata,
                    "drawer_placement_prefilter": prefilter_results,
                    "drawer_placement_candidates": candidate_results,
                    "drawer_candidate_target_positions": [
                        prepared.goal.xpos[:, :, :3, 3].detach().cpu().tolist()
                        for prepared in all_prepared_candidates
                    ],
                },
            ),
        )
        return plan
