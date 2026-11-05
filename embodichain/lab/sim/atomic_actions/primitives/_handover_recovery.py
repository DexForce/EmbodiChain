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

"""Atomic-owned receiving-direction selection and bounded transfer retry."""

from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Any, TYPE_CHECKING
from contextlib import ExitStack
import torch
from embodichain.compute.geometry.gripper import (
    adaptive_grasp_approaches,
    opposite_grasp_end,
)
from embodichain.lab.sim.motion.search import record_search
from ..plans import ActionPlan, PlannerDiagnostics, PlanningFailure
from ..effects import StateDelta
from ..state import TaskState
from ._grasp_sampling import observed_axis, select_grasp
from ._manipulation_checks import inspect_robot_motion

if TYPE_CHECKING:
    from ._handover_scene import HandOverEnvironmentScreen

__all__: list[str] = []


@dataclass
class ReceivePlanningState:
    resources: Any
    held: Any
    alternate: torch.Tensor
    sampled: bool = False
    screen: HandOverEnvironmentScreen | None = None
    exchange_pose: torch.Tensor | None = None
    root_poses: tuple[torch.Tensor, torch.Tensor] | None = None

    def sample(self, action, request, context, affordance, object_pose):
        resources, held = self.resources, self.held
        axis = observed_axis(affordance, object_pose)
        source = object_pose @ held.object_to_eef.to(object_pose)
        destination = action.robot.compute_fk(
            qpos=context.robot.qpos[:, list(resources.second.arm.joint_ids)],
            name=resources.second.arm.control_part,
            to_matrix=True,
        )
        positive = opposite_grasp_end(axis, object_pose, source, destination)
        choices = adaptive_grasp_approaches(
            axis,
            object_pose,
            destination,
            ends=True,
            positive=positive,
            purpose="handover_receive",
        )
        choice = replace(
            choices[0],
            direction=torch.where(
                self.alternate[:, None], choices[1].direction, choices[0].direction
            ),
        )
        candidate_report = None
        raw_pool = None

        def observe_pool(pool):
            nonlocal raw_pool
            raw_pool = pool

        def feasible(poses, _direction):
            nonlocal candidate_report
            if raw_pool is None or raw_pool.poses is not poses:
                raise ValueError(
                    "HandOver candidate screen requires its original pool."
                )
            eligible = (
                raw_pool.valid
                & torch.isfinite(raw_pool.costs)
                & torch.isfinite(poses).all(dim=(-1, -2))
            )
            lengths = action._compute_existing_hold_segment_lengths(
                request.motion_policy.sample_count,
                request.skill_options,
            )
            valid, evidence = self.screen.candidate_mask(
                poses,
                eligible,
                resources=resources,
                approach_samples=lengths["approach"],
                pre_grasp_distance=request.skill_options.pre_grasp_distance,
            )
            selected_poses = poses
            if (
                not valid.any()
                and eligible.any()
                and self.exchange_pose is not None
                and self.root_poses is not None
            ):
                from ._handover_exchange import select_exchange

                station, selected_poses, valid, station_evidence = select_exchange(
                    self.screen,
                    action.robot,
                    request,
                    context,
                    resources,
                    held,
                    self.exchange_pose,
                    *self.root_poses,
                    raw_pool,
                    valid,
                    lengths["approach"],
                    request.skill_options.pre_grasp_distance,
                )
                self.exchange_pose = station
                evidence = {**evidence, "exchange_selection": station_evidence}
            candidate_report = {
                **evidence,
                "raw_valid_count": raw_pool.valid.sum(-1).tolist(),
                "sampler_eligible_count": eligible.sum(-1).tolist(),
                "original_candidates": [
                    {
                        "env_row": row,
                        "pool_index": column,
                        "pose": poses[row, column].detach().cpu().tolist(),
                        "cost": float(raw_pool.costs[row, column]),
                    }
                    for row, column in eligible.nonzero(as_tuple=False).tolist()
                ],
            }
            return selected_poses, valid

        screen_candidates = (
            self.screen is not None and context.affordance_sampling is None
        )
        sample, _ = select_grasp(
            affordance,
            action.planning_services.grasp_pose_generator(
                resources.second.hand.target_id
            ),
            object_pose,
            (choice,),
            context,
            (request.invocation_id or action.skill_id) + ":receive_grasp",
            feasible=feasible if screen_candidates else None,
            raw_candidate_observer=observe_pool if screen_candidates else None,
            axis_region_policy=request.skill_options.receive_region,
        )
        self.sampled = True
        return replace(
            sample,
            metadata={
                **sample.metadata,
                **(
                    {"reference_poses": self.exchange_pose.detach().cpu().tolist()}
                    if self.exchange_pose is not None
                    else {}
                ),
                **(
                    {"handover_candidate_environment": candidate_report}
                    if candidate_report is not None
                    else {}
                ),
                "adaptive_grasp": {
                    **sample.metadata["adaptive_grasp"],
                    "strategy_indices": torch.where(
                        sample.success, self.alternate.long(), -1
                    ).tolist(),
                },
            },
        )


def _qualify(action, request, plan, state):
    exit_check = request.skill_options.source_retreat_mode == "root_separation"
    if not plan.plan_success.any() or (state.screen is None and not exit_check):
        return plan
    try:
        evidence = (
            state.screen.inspect(plan) if state.screen is not None else {"valid": True}
        )
        if exit_check:
            checked = inspect_robot_motion(
                action.robot,
                plan.joint_trajectory.positions[0, -1:],
                state.resources.second.arm.control_part,
            )
            evidence = {
                **evidence,
                "post_handover_exit": checked,
                "valid": evidence.get("valid") is True and checked.get("valid") is True,
            }
    except (AttributeError, KeyError, TypeError, ValueError, RuntimeError) as error:
        evidence = {"valid": False, "unavailable": str(error)}
    valid = evidence.get("valid") is True
    return replace(
        plan,
        plan_success=(
            plan.plan_success if valid else torch.zeros_like(plan.plan_success)
        ),
        diagnostics=replace(
            plan.diagnostics,
            metadata={**plan.diagnostics.metadata, "handover_environment": evidence},
            failure=(
                plan.diagnostics.failure
                if valid
                else PlanningFailure(
                    "handover_receiver_environment_unqualified", retryable=False
                )
            ),
        ),
    )


def plan_receive(action, request, context, resources, held):
    if not request.skill_options.require_receiver_scene:
        return _plan_receive(action, request, context, resources, held, None)
    from ._handover_scene import HandOverEnvironmentScreen
    from embodichain.lab.sim.scene.native_hand_scene import NativeHandSceneQuery

    query = None
    with ExitStack() as stack:
        try:
            query = action.planning_services.handover_scene_provider.prepare(
                request, context
            )
            if type(query) is not NativeHandSceneQuery:
                raise TypeError("HandOver provider must return NativeHandSceneQuery.")
            stack.enter_context(query)
            screen = HandOverEnvironmentScreen(action.robot, request, context, query)
        except (
            AttributeError,
            ImportError,
            KeyError,
            TypeError,
            ValueError,
            RuntimeError,
        ) as error:
            if type(query) is NativeHandSceneQuery:
                query.close()
            return action.failed_plan(
                request,
                context,
                message=f"HandOver receiver environment geometry unavailable: {error}",
            )
        return _plan_receive(action, request, context, resources, held, screen)


def _plan_receive(action, request, context, resources, held, screen):
    state = ReceivePlanningState(
        resources,
        held,
        torch.zeros(context.batch_size, dtype=torch.bool, device=action.device),
        screen=screen,
    )

    def build():
        record_search("handover_direction_attempts", budget_checkpoint=False)
        return _qualify(
            action,
            request,
            action._build_existing_hold(
                request, context, resources, held, receive_state=state
            ),
            state,
        )

    first = build()
    if first.plan_success.all() or not state.sampled:
        return first
    state.alternate = ~first.plan_success
    devices = [context.robot.qpos.device.index] if context.robot.qpos.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        second = build()
    metadata = {
        "receive_direction_attempts": [
            {
                "plan_success": p.plan_success.tolist(),
                "metadata": dict(p.diagnostics.metadata),
            }
            for p in (first, second)
        ],
        "selected_receive_attempt": torch.where(
            first.plan_success, 0, torch.where(second.plan_success, 1, -1)
        ).tolist(),
    }
    if not (second.plan_success & ~first.plan_success).any():
        return replace(
            first,
            diagnostics=replace(
                first.diagnostics, metadata={**first.diagnostics.metadata, **metadata}
            ),
        )
    merged = _merge_receive_plans(action, request, context, first, second, metadata)
    return _qualify(action, request, merged, state)


def _merge_receive_plans(action, request, context, first, second, metadata):
    """Retain successful rows and shared HandOver's planned effect contracts."""
    keep = first.plan_success
    if first.segments != second.segments:
        raise ValueError("HandOver direction retries must preserve phase timing.")
    trajectory = replace(
        second.joint_trajectory,
        positions=torch.where(
            keep[:, None, None],
            first.joint_trajectory.positions,
            second.joint_trajectory.positions,
        ),
        dt=torch.where(
            keep[:, None], first.joint_trajectory.dt, second.joint_trajectory.dt
        ),
    )

    def effects(name):
        previous, retry = getattr(first, name), getattr(second, name)
        hypothetical = TaskState(
            batch_size=context.batch_size, device=context.robot.qpos.device
        )
        hypothetical = previous.apply(hypothetical, keep)
        hypothetical = retry.apply(hypothetical, ~keep & second.plan_success)
        return StateDelta(
            held_object_updates={
                key: hypothetical.get_held_object(key)
                for key in previous.held_object_updates.keys()
                | retry.held_object_updates.keys()
            }
        )

    return action.build_plan(
        request,
        context,
        success=keep | second.plan_success,
        trajectory=trajectory,
        expected_effects=effects("expected_effects"),
        effect_candidates=effects("effect_candidates"),
        diagnostics=PlannerDiagnostics(
            backend=first.diagnostics.backend,
            metadata=metadata,
        ),
        segment_lengths={
            segment.name: segment.stop - segment.start for segment in first.segments
        },
        scene_dependency_monitor_until=first.scene_dependency_monitor_until,
    )
