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

"""Action Engine-specific adapters for mainline atomic actions."""

from __future__ import annotations

from dataclasses import dataclass, replace

from embodichain.lab.sim.atomic_actions import (
    ActionPlan,
    JointPositionGoal,
    MoveHeldObject,
    MoveHeldObjectOptions,
    MoveJoints,
    MoveJointsOptions,
    PlanningContext,
    ResolvedActionRequest,
    StateDelta,
)

__all__ = [
    "ActionEngineMoveJoints",
    "ActionEngineMoveJointsOptions",
    "ExactTargetMoveHeldObject",
    "ExactTargetMoveHeldObjectOptions",
]


@dataclass(frozen=True, slots=True, eq=False)
class ExactTargetMoveHeldObjectOptions(MoveHeldObjectOptions):
    """Action Engine transport options for a grounded object target."""


class ExactTargetMoveHeldObject(MoveHeldObject):
    """Action Engine marker for mainline exact-target transport."""

    OptionsType = ExactTargetMoveHeldObjectOptions
    binding_contract = MoveHeldObject.binding_contract


@dataclass(frozen=True, slots=True, eq=False)
class ActionEngineMoveJointsOptions(MoveJointsOptions):
    """Joint motion with an explicit optional single-arm release effect."""

    single_release: bool = False
    """Whether a successful gripper-open command releases the held object."""

    def __post_init__(self) -> None:
        if type(self.single_release) is not bool:
            raise TypeError("single_release must be a boolean.")


class ActionEngineMoveJoints(MoveJoints):
    """Preserve ordinary joint motion and commit explicit release nodes."""

    OptionsType = ActionEngineMoveJointsOptions
    binding_contract = MoveJoints.binding_contract

    def _plan(
        self,
        request: ResolvedActionRequest[
            JointPositionGoal,
            ActionEngineMoveJointsOptions,
        ],
        context: PlanningContext,
    ) -> ActionPlan:
        endpoint = request.binding.endpoint("primary", "motion")
        task_state_key = endpoint.task_state_key
        if request.skill_options.single_release:
            if not isinstance(task_state_key, str) or not task_state_key:
                raise ValueError(
                    "Single-arm release requires a non-empty task-state key."
                )
            if context.task.get_held_object(task_state_key) is None:
                return self.failed_plan(
                    request,
                    context,
                    message=(
                        "Single-arm release requires an object held by task-state "
                        f"resource {task_state_key!r}."
                    ),
                )

        plan = super()._plan(request, context)
        if not request.skill_options.single_release:
            return plan
        return replace(
            plan,
            expected_effects=StateDelta(
                held_object_updates={task_state_key: None},
            ),
        )
