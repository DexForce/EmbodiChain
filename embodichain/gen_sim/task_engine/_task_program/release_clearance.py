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

"""Bind a released hand's declared clearance route to the shared Cartesian skill."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, ClassVar

import torch

from embodichain.lab.sim.atomic_actions import (
    EndEffectorPoseGoal,
    MoveEndEffector,
    MoveEndEffectorOptions,
)
from embodichain.lab.task_program.compiler.lowering import (
    RegisteredSemanticLowerer,
    SemanticLowering,
)
from embodichain.lab.task_program.semantics import (
    RegisteredSemanticCall,
    SceneObjectRef,
    SemanticCallDescriptor,
    SkillPolicyPreset,
)

__all__: list[str] = []

CLEAR_RELEASED_CALL = "gen_sim.clear_released"


def _clearance_poses(
    current: torch.Tensor,
    root: torch.Tensor,
    *,
    height: float,
    retreat: float,
) -> torch.Tensor:
    """First clear vertically, then move toward the owning arm's root."""
    raised = current.clone()
    raised[:, 2, 3] = torch.clamp_min(raised[:, 2, 3], height)
    direction = root[:, :2, 3] - current[:, :2, 3]
    direction = torch.nn.functional.normalize(direction, dim=-1)
    withdrawn = raised.clone()
    withdrawn[:, :2, 3] += retreat * direction
    return torch.stack((raised, withdrawn), dim=1)


class _ClearReleasedLowerer(RegisteredSemanticLowerer):
    call_id: ClassVar[str] = CLEAR_RELEASED_CALL
    target_descriptor = MoveEndEffector.descriptor()
    preserves_symbolic_state: ClassVar[bool] = True

    def __init__(
        self,
        routes: tuple[tuple[str, str, float], ...],
        robot: Any,
        *,
        retreat_distance: float,
    ) -> None:
        self._routes = {(obj, target): height for obj, target, height in routes}
        self._robot = robot
        self._retreat = retreat_distance

    def lower(
        self,
        call: RegisteredSemanticCall,
        *,
        context: Any,
        bound: Any,
        option_template: Any,
    ) -> SemanticLowering:
        if type(option_template) is not MoveEndEffectorOptions:
            raise TypeError("Released-hand clearance requires MoveEndEffectorOptions.")
        arguments = dict(call.arguments)
        if set(arguments) != {"object", "target"}:
            raise ValueError("Released-hand clearance requires only object and target.")
        selector = (arguments["object"], arguments["target"])
        if selector not in self._routes:
            raise ValueError("Released-hand clearance has no declared route.")
        endpoint = bound.binding.resources["primary"].endpoints["motion"]
        held = context.task.get_held_object(endpoint.task_state_key)
        if held is not None and held.active_mask.any():
            raise ValueError("Released-hand clearance cannot move a held object.")
        motion = endpoint.runtime_target
        current = self._robot.compute_fk(
            qpos=context.robot.qpos[:, list(motion.joint_ids)],
            name=motion.control_part,
            to_matrix=True,
        )
        root = self._robot.get_link_pose(
            link_name=self._robot.cfg.solver_cfg[motion.control_part].root_link_name,
            env_ids=context.env_ids.tolist(),
            to_matrix=True,
        ).to(current)
        return SemanticLowering(
            goal=EndEffectorPoseGoal(
                _clearance_poses(
                    current,
                    root,
                    height=self._routes[selector],
                    retreat=self._retreat,
                )
            )
        )


@dataclass(frozen=True, slots=True)
class _ClearReleasedFactory:
    call_id: ClassVar[str] = CLEAR_RELEASED_CALL
    revision: ClassVar[str] = "1"
    target_descriptor = MoveEndEffector.descriptor()
    routes: tuple[tuple[str, str, float], ...]
    retreat_distance: float = 0.10

    def create(
        self, *, simulation: Any, robot: Any, scene_registry: Any, engine: Any
    ) -> _ClearReleasedLowerer:
        if engine.robot is not robot:
            raise ValueError("Released-hand clearance must bind the factory's robot.")
        for obj, _, _ in self.routes:
            scene_registry.resolve(obj, expected_type=SceneObjectRef)
        return _ClearReleasedLowerer(
            self.routes, robot, retreat_distance=self.retreat_distance
        )


def with_release_clearance(registration: Any, *, program: dict[str, Any]) -> Any:
    """Register only generated clearance routes, with no execution overrides."""
    routes = {}
    for item in program["program"]["items"]:
        call = item["steps"]["call"]
        if call.get("call_id") != CLEAR_RELEASED_CALL:
            continue
        args = call["arguments"]
        if set(args) != {"object", "target"}:
            raise ValueError("Released-hand clearance requires only object and target.")
        values = program["targets"][args["target"]]["values"]
        if len(values) != 1:
            raise ValueError(
                "Released-hand clearance requires one exact staging target."
            )
        height = values[0]["position"][2]
        if (
            isinstance(height, bool)
            or not isinstance(height, (float, int))
            or not math.isfinite(height)
        ):
            raise ValueError("Released-hand clearance height must be finite.")
        routes[(args["object"], args["target"])] = (
            args["object"],
            args["target"],
            float(height),
        )
    if not routes:
        return registration
    factory = _ClearReleasedFactory(tuple(routes.values()))
    catalog = registration.call_catalog.with_descriptor(
        SemanticCallDescriptor(
            call_id=CLEAR_RELEASED_CALL,
            spec_type=RegisteredSemanticCall,
            target_descriptor=factory.target_descriptor,
        )
    )
    presets = []
    for preset in registration.robot_profile_binding.presets:
        options = dict(preset.action_option_templates)
        options[CLEAR_RELEASED_CALL] = MoveEndEffectorOptions()
        presets.append(
            SkillPolicyPreset(
                preset_id=preset.preset_id,
                required_planner=preset.required_planner,
                action_option_templates=options,
                effect_monitors=preset.effect_monitors,
                effect_assurance=preset.effect_assurance,
                motion_policy=preset.motion_policy,
                tracking_policy=preset.tracking_policy,
                recovery_policy=preset.recovery_policy,
                workflow_recovery_policy=preset.workflow_recovery_policy,
                runner_cfg=preset.runner_cfg,
            )
        )
    return replace(
        registration,
        call_catalog=catalog,
        robot_profile_binding=replace(
            registration.robot_profile_binding, presets=tuple(presets)
        ),
        registered_semantic_lowerer_factories=(
            *registration.registered_semantic_lowerer_factories,
            factory,
        ),
    )
