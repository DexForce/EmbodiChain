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
import math
import json
from typing import Any, ClassVar

import torch
from embodichain.utils import logger
from embodichain.lab.sim.motion.motion_generator import MotionGenOptions
from embodichain.lab.sim.motion.planners.utils import MoveType, PlanState

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

    def __init__(
        self,
        routes: tuple[tuple[Any, ...], ...],
        robot: Any,
        yaw_offsets: tuple[tuple[str, str, float], ...] = (),
        descent_positions: tuple[tuple[str, str, tuple[float, ...]], ...] = (),
        motion_generator: Any = None,
    ) -> None:
        self._routes = {
            (obj, target, preserve): (axis, position)
            for obj, target, preserve, axis, position in routes
        }
        self._robot = robot
        self._motion_generator = motion_generator
        self._yaw_offsets = {(obj, target): yaw for obj, target, yaw in yaw_offsets}
        self._descent_positions = {
            (obj, target): position for obj, target, position in descent_positions
        }

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
        # Complete the placement yaw while the payload is at staging height.
        yaw = self._yaw_offsets.get((args["object"], args["target"]), 0.0)
        if yaw:
            pose[:, :3, :3] = (
                axis_angle_to_rotation_matrix(pose.new_tensor([0.0, 0.0, yaw]))
                @ pose[:, :3, :3]
            )
        release = self._descent_positions.get((args["object"], args["target"]))
        if release is not None:
            pose = _select_upright_yaw(
                self._robot,
                self._motion_generator,
                pose,
                object_to_eef,
                context.robot.qpos[:, list(motion.joint_ids)],
                part,
                context.env_ids.tolist(),
                release,
            )
        return SemanticLowering(goal=HeldObjectPoseGoal(pose))


def _select_upright_yaw(
    robot: Any,
    motion_generator: Any,
    pose: torch.Tensor,
    object_to_eef: torch.Tensor,
    seed: torch.Tensor,
    part: str,
    env_ids: list[int],
    release: tuple[float, ...],
) -> torch.Tensor:
    """Rank free-yaw targets; full trajectory and execution gates remain mandatory."""
    limits = robot.get_qpos_limits(name=part, env_ids=env_ids).to(pose)
    low, high = limits[..., 0], limits[..., 1]
    best_score = pose.new_full((len(pose),), -float("inf"))
    best_pose = pose.clone()
    evidence = []
    for degrees in (0, -30, 30, -60, 60, -90, 90):
        candidate = pose.clone()
        candidate[:, :3, :3] = (
            axis_angle_to_rotation_matrix(
                pose.new_tensor([0.0, 0.0, math.radians(degrees)])
            )
            @ pose[:, :3, :3]
        )
        targets = []
        for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
            target = candidate.clone()
            target[:, :3, 3] = torch.lerp(
                candidate[:, :3, 3], pose.new_tensor(release), fraction
            )
            targets.append(
                PlanState(move_type=MoveType.EEF_MOVE, xpos=target @ object_to_eef)
            )
        # Coarse ranking plans are never executed or substituted for the real plan.
        planned = motion_generator.generate(
            targets,
            options=MotionGenOptions(
                strategy="ik_interp",
                start_qpos=seed,
                control_part=part,
                sample_count=6,
                preserve_cartesian_samples=True,
                is_linear=True,
                interpolation_dt=2.0,
            ),
        )
        qpos = planned.positions[:, 1:]
        delta = planned.positions[:, 1:] - planned.positions[:, :-1]
        feasible = planned.success & torch.isfinite(qpos).all(-1).all(-1)
        feasible &= (delta[:, 1:].abs().amax(-1) < 0.5).all(-1)
        travel = torch.linalg.vector_norm(delta, dim=-1).sum(-1)
        margin = (
            torch.minimum(
                (qpos - low[:, None]) / (high - low)[:, None],
                (high[:, None] - qpos) / (high - low)[:, None],
            )
            .amin(-1)
            .amin(-1)
        )
        score = torch.where(feasible, margin - 0.01 * travel, -float("inf"))
        best_pose = torch.where(
            (score > best_score)[:, None, None], candidate, best_pose
        )
        best_score = torch.maximum(best_score, score)
        evidence.append(
            {
                "yaw_delta_degrees": degrees,
                "feasible": feasible.tolist(),
                "minimum_joint_margin": [
                    value if math.isfinite(value) else None for value in margin.tolist()
                ],
                "joint_travel": [
                    value if math.isfinite(value) else None for value in travel.tolist()
                ],
            }
        )
    logger.log_info("GenSim upright yaw candidates: " + json.dumps(evidence))
    return best_pose


@dataclass(frozen=True, slots=True)
class _AlignHeldFactory:
    call_id: ClassVar[str] = ALIGN_HELD_CALL
    revision: ClassVar[str] = "5"
    target_descriptor = MoveHeldObject.descriptor()
    routes: tuple[
        tuple[str, str, bool, tuple[float, ...], tuple[float, ...] | None], ...
    ]
    yaw_offsets: tuple[tuple[str, str, float], ...] = ()
    descent_positions: tuple[tuple[str, str, tuple[float, ...]], ...] = ()

    def create(
        self, *, simulation: Any, robot: Any, scene_registry: Any, engine: Any
    ) -> _AlignHeldLowerer:
        if engine.robot is not robot:
            raise ValueError("Held alignment must bind the factory's robot.")
        for obj, _, _, _, _ in self.routes:
            scene_registry.resolve(obj, expected_type=SceneObjectRef)
        return _AlignHeldLowerer(
            self.routes,
            robot,
            self.yaw_offsets,
            self.descent_positions,
            engine.motion_generator,
        )


def with_held_alignment(
    registration: Any, *, program: dict[str, Any], constraints: Any
) -> Any:
    axes = {
        cfg.entity: cfg.local_axis
        for cfg in constraints.values()
        if cfg.local_axis is not None
    }
    routes = {}
    yaw_offsets = {}
    pickup_targets = {}
    descent_positions = {}
    for item in program["program"]["items"]:
        call = item["steps"]["call"]
        if call.get("call_id", "").startswith("gen_sim.pick."):
            pickup_targets[call["arguments"]["object"]] = call["arguments"]["target"]
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
        if values is not None and not preserve and abs(axes[obj][2]) < 1.0 - 1e-6:
            arm = call.get("resources", {}).get("primary", "left")
            yaw_offsets[(obj, target)] = (
                math.pi / 3.0 if arm == "right" else -math.pi / 3.0
            )
            if obj in pickup_targets:
                release_values = program["targets"][pickup_targets[obj]]["values"]
                if len(release_values) == 1:
                    descent_positions[(obj, target)] = tuple(
                        release_values[0]["position"]
                    )
    if not routes:
        return registration
    factory = _AlignHeldFactory(
        tuple(routes.values()),
        tuple((obj, target, yaw) for (obj, target), yaw in yaw_offsets.items()),
        tuple((obj, target, pos) for (obj, target), pos in descent_positions.items()),
    )
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
        relation_grounders=(),
        robot_profile_binding=replace(
            registration.robot_profile_binding, presets=tuple(presets)
        ),
        registered_semantic_lowerer_factories=(
            *registration.registered_semantic_lowerer_factories,
            factory,
        ),
    )
