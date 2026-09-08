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

"""Bind an observed held-object axis to an existing pose-motion skill."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, ClassVar

import torch

from embodichain.lab.sim.atomic_actions.primitives.move_held_object import (
    HeldObjectPoseGoal,
    MoveHeldObject,
    MoveHeldObjectOptions,
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
from embodichain.utils.math import axis_angle_to_rotation_matrix

__all__: list[str] = []

ALIGN_HELD_CALL = "gen_sim.align_held"


class _AlignHeldLowerer(RegisteredSemanticLowerer):
    call_id: ClassVar[str] = ALIGN_HELD_CALL
    target_descriptor = MoveHeldObject.descriptor()
    preserves_symbolic_state: ClassVar[bool] = True

    def __init__(self, routes: tuple[tuple[Any, ...], ...], robot: Any) -> None:
        self._routes = {
            (obj, target, preserve): (axis, position)
            for obj, target, preserve, axis, position in routes
        }
        self._robot = robot

    def lower(
        self,
        call: RegisteredSemanticCall,
        *,
        context: Any,
        bound: Any,
        option_template: Any,
    ) -> SemanticLowering:
        if type(option_template) is not MoveHeldObjectOptions:
            raise TypeError("Held alignment requires MoveHeldObjectOptions.")
        args = dict(call.arguments)
        if (
            set(args) != {"object", "target", "preserve_yaw"}
            or type(args["preserve_yaw"]) is not bool
        ):
            raise ValueError(
                "Held alignment requires references and a declared yaw policy."
            )
        axis, position = self._routes[
            (args["object"], args["target"], args["preserve_yaw"])
        ]
        resource = bound.binding.resources["primary"]
        key = resource.endpoints["motion"].task_state_key
        held = context.task.get_held_object(key)
        if held is None or held.semantics.entity_id != args["object"]:
            raise ValueError("Held alignment requires the declared object to be held.")
        observed = context.scene.entities[args["object"]]
        if observed.confidence <= 0:
            raise ValueError("Held alignment requires a current object observation.")
        pose = observed.pose.clone()
        if pose.shape == (4, 4):
            pose = pose.unsqueeze(0).expand(context.batch_size, -1, -1).clone()
        source = torch.nn.functional.normalize(
            pose[:, :3, :3] @ pose.new_tensor(axis), dim=-1
        )
        target = torch.zeros_like(source)
        target[:, 2] = 1
        cross = torch.cross(source, target, dim=-1)
        sine = torch.linalg.vector_norm(cross, dim=-1)
        cosine = (source * target).sum(-1).clamp(-1, 1)
        rotation_axis = cross / sine.clamp_min(1e-8)[:, None]
        # Opposite axes need a deterministic perpendicular, not an arbitrary yaw.
        basis = torch.eye(3, device=pose.device, dtype=pose.dtype)[
            source.abs().argmin(-1)
        ]
        perpendicular = torch.nn.functional.normalize(
            torch.cross(source, basis, dim=-1), dim=-1
        )
        rotation_axis = torch.where(
            (sine < 1e-8)[:, None], perpendicular, rotation_axis
        )
        correction = axis_angle_to_rotation_matrix(
            rotation_axis * torch.atan2(sine, cosine)[:, None]
        )
        pose[:, :3, :3] = correction @ pose[:, :3, :3]
        if position is not None:
            pose[:, :3, 3] = pose.new_tensor(position)
        if args["preserve_yaw"]:
            return SemanticLowering(goal=HeldObjectPoseGoal(pose))
        # Upright leaves yaw free. Prefer a TCP approaching from its arm root
        # rather than forcing the wrist to point back toward the robot base.
        motion = resource.endpoints["motion"].runtime_target
        part = motion.control_part
        root_name = self._robot.cfg.solver_cfg[part].root_link_name
        root_pose = self._robot.get_link_pose(
            link_name=root_name,
            env_ids=context.env_ids.tolist(),
            to_matrix=True,
        ).to(device=pose.device, dtype=pose.dtype)
        toward_object = pose[:, :2, 3] - root_pose[..., :2, 3]
        object_to_eef = held.object_to_eef.to(device=pose.device, dtype=pose.dtype)
        approach = (pose[:, :3, :3] @ object_to_eef[..., :3, :3])[:, :2, 2]
        flip = (torch.linalg.vector_norm(approach, dim=-1) > 0.1) & (
            (approach * toward_object).sum(-1) < 0
        )
        half_turn = torch.diag(pose.new_tensor([-1.0, -1.0, 1.0]))
        pose[:, :3, :3] = torch.where(
            flip[:, None, None],
            half_turn @ pose[:, :3, :3],
            pose[:, :3, :3],
        )
        return SemanticLowering(goal=HeldObjectPoseGoal(pose))


@dataclass(frozen=True, slots=True)
class _AlignHeldFactory:
    call_id: ClassVar[str] = ALIGN_HELD_CALL
    revision: ClassVar[str] = "3"
    target_descriptor = MoveHeldObject.descriptor()
    routes: tuple[
        tuple[str, str, bool, tuple[float, ...], tuple[float, ...] | None], ...
    ]

    def create(
        self, *, simulation: Any, robot: Any, scene_registry: Any, engine: Any
    ) -> _AlignHeldLowerer:
        if engine.robot is not robot:
            raise ValueError("Held alignment must bind the factory's robot.")
        for obj, _, _, _, _ in self.routes:
            scene_registry.resolve(obj, expected_type=SceneObjectRef)
        return _AlignHeldLowerer(self.routes, robot)


def with_held_alignment(
    registration: Any, *, program: dict[str, Any], constraints: Any
) -> Any:
    axes = {
        cfg.entity: cfg.local_axis
        for cfg in constraints.values()
        if cfg.local_axis is not None
    }
    routes = {}
    for item in program["program"]["items"]:
        call = item["steps"]["call"]
        if call.get("call_id") != ALIGN_HELD_CALL:
            continue
        args = call["arguments"]
        obj, target = args["object"], args["target"]
        preserve = args.get("preserve_yaw")
        if type(preserve) is not bool:
            raise ValueError(
                "Held alignment needs an explicit preserve_yaw policy; regenerate the bundle."
            )
        values = (
            None
            if target == "current_object_pose"
            else program["targets"][target]["values"]
        )
        if (values is not None and len(values) != 1) or axes.get(obj) is None:
            raise ValueError(
                "Held alignment needs an exact staging target and declared axis."
            )
        routes[(obj, target, preserve)] = (
            obj,
            target,
            preserve,
            axes[obj],
            None if values is None else tuple(values[0]["position"]),
        )
    if not routes:
        return registration
    factory = _AlignHeldFactory(tuple(routes.values()))
    catalog = registration.call_catalog.with_descriptor(
        SemanticCallDescriptor(
            call_id=ALIGN_HELD_CALL,
            spec_type=RegisteredSemanticCall,
            target_descriptor=factory.target_descriptor,
        )
    )
    presets = []
    for preset in registration.robot_profile_binding.presets:
        options = dict(preset.action_option_templates)
        options[ALIGN_HELD_CALL] = MoveHeldObjectOptions()
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
