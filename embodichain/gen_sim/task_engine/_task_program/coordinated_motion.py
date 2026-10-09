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

"""GenSim E5 continuation through the shared coordinated motion planner."""

from __future__ import annotations

from typing import ClassVar

import torch

from embodichain.lab.sim.atomic_actions.core import AtomicAction
from embodichain.lab.sim.atomic_actions.effects import StateDelta
from embodichain.lab.sim.atomic_actions.goals import resolve_pose_goal
from embodichain.lab.sim.atomic_actions.invocation import ResolvedActionRequest
from embodichain.lab.sim.atomic_actions.plans import ActionPlan, TimedTrajectory
from embodichain.lab.sim.atomic_actions.primitives.coordinated_pickment import (
    CoordinatedPickGoal,
    CoordinatedPickment,
    CoordinatedPickmentOptions,
)
from embodichain.lab.sim.atomic_actions.state import HeldObjectState, PlanningContext
from embodichain.lab.sim.atomic_actions.trajectory_ops import translate_pose_world

__all__: list[str] = []

COORDINATED_MOTION_REVISION = 1


class GenSimCoordinatedPickment(
    AtomicAction[CoordinatedPickGoal, CoordinatedPickmentOptions]
):
    """Reuse verified grasps for continuation; empty hands retain ordinary pickup.

    The composed shared action supplies the binding and synchronized trajectory
    helpers. There is no local executor, live IK loop, or physical-state commit.
    """

    skill_id: ClassVar[str] = CoordinatedPickment.skill_id
    GoalType = CoordinatedPickGoal
    OptionsType = CoordinatedPickmentOptions
    binding_contract = CoordinatedPickment.binding_contract
    agent_visible = CoordinatedPickment.agent_visible
    open_loop = CoordinatedPickment.open_loop

    def __init__(
        self, default_options: CoordinatedPickmentOptions | None = None
    ) -> None:
        super().__init__(default_options)
        self._pickup = CoordinatedPickment()

    def _scene_dependencies(
        self,
        request: ResolvedActionRequest[CoordinatedPickGoal, CoordinatedPickmentOptions],
    ) -> tuple[str, ...]:
        return self._pickup._scene_dependencies(request)

    def _plan(
        self,
        request: ResolvedActionRequest[CoordinatedPickGoal, CoordinatedPickmentOptions],
        context: PlanningContext,
    ) -> ActionPlan:
        helper = self._pickup
        helper._bind(self.planning_services)
        resources = helper._resolve_resources(request)
        keys = (resources.left_task_state_key, resources.right_task_state_key)
        masks = [context.task.held_object_mask(key) for key in keys]
        if not bool((masks[0] | masks[1]).any()):
            return helper.plan(request, context)

        held = [context.task.get_held_object(key) for key in keys]
        entity = request.goal.semantics.entity_id
        if entity is None or any(
            item is None or item.semantics.entity_id != entity for item in held
        ):
            return self.failed_plan(
                request,
                context,
                message="E5 continuation requires both hands to hold the requested object.",
            )
        success = masks[0] & masks[1]
        if not success.any():
            return self.failed_plan(
                request,
                context,
                message="E5 continuation has no jointly verified held rows.",
            )
        if (
            request.motion_policy.strategy == "motion_gen"
            and self.motion_generator.planner.cfg.planner_type == "curobo"
        ):
            raise ValueError(
                "Coordinated dual-arm planning is not supported by the cuRobo backend."
            )

        options = request.skill_options
        initial = helper._resolve_object_initial_pose(request.goal, context)
        target = helper._resolve_pose(
            resolve_pose_goal(
                request.goal.object_target_pose, context, name="object_target_pose"
            ),
            "object_target_pose",
        )
        offsets = [
            helper._resolve_pose(item.object_to_eef, "verified_object_to_eef")
            for item in held
        ]
        left_start, right_start = helper._resolve_dual_arm_start(context, resources)
        counts = {
            "move": 0,
            "hold": options.hold_steps,
            "release": max(2, options.release_steps) if options.release else 0,
            "retreat": max(2, options.retreat_steps) if options.release else 0,
        }
        counts["move"] = request.motion_policy.sample_count - sum(counts.values())
        if counts["move"] < 2:
            raise ValueError("E5 continuation requires at least two move waypoints.")
        object_path = helper._interpolate_object_pose(
            initial, target, counts["move"], include_orientation=True
        )
        success, left_move, right_move = helper._plan_synchronized_object_motion(
            left_start,
            right_start,
            object_path,
            offsets[0],
            offsets[1],
            success,
            resources,
            options,
        )
        left_final, right_final = left_move[:, -1], right_move[:, -1]
        pieces = [
            helper._assemble_segment(
                context,
                left_move,
                right_move,
                helper._repeat_qpos(resources.left_hand_close_qpos, counts["move"]),
                helper._repeat_qpos(resources.right_hand_close_qpos, counts["move"]),
                resources=resources,
            )
        ]
        if counts["hold"]:
            pieces.append(
                helper._assemble_segment(
                    context,
                    helper._repeat_qpos(left_final, counts["hold"]),
                    helper._repeat_qpos(right_final, counts["hold"]),
                    helper._repeat_qpos(resources.left_hand_close_qpos, counts["hold"]),
                    helper._repeat_qpos(
                        resources.right_hand_close_qpos, counts["hold"]
                    ),
                    resources=resources,
                )
            )
        endpoints = [target @ offset for offset in offsets]
        if options.release:
            pieces.append(
                helper._assemble_segment(
                    context,
                    helper._repeat_qpos(left_final, counts["release"]),
                    helper._repeat_qpos(right_final, counts["release"]),
                    helper._interpolate_qpos(
                        resources.left_hand_close_qpos,
                        resources.left_hand_open_qpos,
                        counts["release"],
                    ),
                    helper._interpolate_qpos(
                        resources.right_hand_close_qpos,
                        resources.right_hand_open_qpos,
                        counts["release"],
                    ),
                    resources=resources,
                )
            )
            retreat = target.new_tensor([0.0, 0.0, options.retreat_distance])
            success, left_retreat = helper._plan_masked_arm_trajectory(
                resources.left_arm.control_part,
                left_final,
                translate_pose_world(endpoints[0], retreat)[:, None],
                counts["retreat"],
                success,
            )
            success, right_retreat = helper._plan_masked_arm_trajectory(
                resources.right_arm.control_part,
                right_final,
                translate_pose_world(endpoints[1], retreat)[:, None],
                counts["retreat"],
                success,
            )
            pieces.append(
                helper._assemble_segment(
                    context,
                    left_retreat,
                    right_retreat,
                    helper._repeat_qpos(
                        resources.left_hand_open_qpos, counts["retreat"]
                    ),
                    helper._repeat_qpos(
                        resources.right_hand_open_qpos, counts["retreat"]
                    ),
                    resources=resources,
                )
            )
        effects = StateDelta(
            held_object_updates={
                key: (
                    None
                    if options.release
                    else HeldObjectState(item.semantics, offset, endpoint)
                )
                for key, item, offset, endpoint in zip(
                    keys, held, offsets, endpoints, strict=True
                )
            }
        )
        return self.build_plan(
            request,
            context,
            success=success,
            trajectory=TimedTrajectory.from_uniform_step(
                torch.cat(pieces, dim=1),
                env_ids=context.env_ids,
                step_dt=context.require_control_dt(),
            ),
            expected_effects=effects,
            segment_lengths=counts,
            # This object's motion is commanded, not an external target change.
            # Any independent destination dependencies remain fully monitored.
            scene_dependency_monitor_until=(
                {entity: 0} if entity in self._scene_dependencies(request) else {}
            ),
        )
