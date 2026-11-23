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


"""Ground the two Tool Hang pickup programs in live poses and native pad contacts."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, replace
from importlib.resources import files
from typing import ClassVar

import torch
from tensordict import TensorDict

from embodichain.compute.geometry.grip_contacts import classify_contacts
from embodichain.lab.gym.envs import DemoSegment, EmbodiedEnv
from embodichain.lab.gym.envs.types import ControllerAction
from embodichain.lab.sim.atomic_skills import (
    EndEffectorPoseGoal,
    MoveEndEffector,
    MoveEndEffectorOptions,
    MotionPolicy,
    PickUp,
    PickUpOptions,
    RecoveryPolicy,
)
from embodichain.lab.sim.atomic_skills.primitives.contact import (
    ContactPhase,
    ContactPickUpGoal,
    ResolvedContactPhase,
)
from embodichain.lab.task_program.compiler.lowering import (
    RegisteredSemanticLowerer,
    SemanticLowering,
)
from embodichain.lab.task_program.integrations.catalog import (
    SimulationTaskProgramRegistration,
)
from embodichain.lab.task_program.integrations.simulation.bindings import (
    ControlPartCommandPreset,
    ControlPartEndpointBinding,
    ControlPartResourceBinding,
    SimulationArticulationBinding,
    SimulationRobotSkillProfileBinding,
    SimulationSceneBinding,
)
from embodichain.lab.task_program.integrations.simulation.environment import (
    create_simulation_task_program_adapter,
)
from embodichain.lab.task_program.language.loader import load_task_program
from embodichain.lab.task_program.language.schema import TaskProgramIntegrationCfg
from embodichain.lab.task_program.language.semantic_contracts import (
    SemanticCallContract,
)
from embodichain.lab.task_program.semantics.calls import (
    RegisteredSemanticCall,
    SemanticCallDescriptor,
    builtin_semantic_call_catalog,
)
from embodichain.lab.task_program.semantics.profiles import (
    EffectAssurance,
    SkillPolicyPreset,
)

__all__: list[str] = []


class _Approach(RegisteredSemanticLowerer):
    """Approach the observed grip from above with the open hand."""

    call_id: ClassVar[str] = "tool_hang.approach_grip"
    target_descriptor: ClassVar = MoveEndEffector.descriptor()
    requires_unoccupied_slots: ClassVar = ("primary",)

    def lower(self, call, *, context, bound, option_template):
        target = call.arguments["object"]
        pose = context.scene.entities[target].pose
        return SemanticLowering(
            goal=EndEffectorPoseGoal(_target(pose, target, approach=True))
        )


def _target(pose: torch.Tensor, target: str, *, approach: bool) -> torch.Tensor:
    """Return an arena-frame TCP with horizontal closing axis through the handle."""
    if target == "frame":
        offset = (0.04375, 0.0, 0.068)
        column, sign, height = 0, -1.0, 0.07 if approach else -0.003
    elif target == "tool":
        offset = (-0.004 if approach else -0.026, 0.0, 0.0)
        column, sign, height = 1, 1.0, 0.14 if approach else -0.006
    else:
        raise ValueError(f"Unknown Tool Hang pickup object {target!r}.")
    closing = sign * pose[:, :3, column].clone()
    closing[:, 2] = 0
    closing = torch.nn.functional.normalize(closing, dim=-1)
    down = torch.zeros_like(closing)
    down[:, 2] = -1
    goal = pose.clone()
    goal[:, :3, :3] = torch.stack(
        (torch.linalg.cross(closing, down), closing, down), dim=-1
    )
    goal[:, :3, 3] += pose[:, :3, :3] @ pose.new_tensor(offset)
    goal[:, 2, 3] += height
    return goal


class _PickupContact:
    """Supply the grasp TCP and verify that the physically pinched body followed it."""

    source_id = "tool_hang.pinch"

    def __init__(self, simulation, robot):
        self.sensor = simulation.get_sensor("grasp_contacts")
        self.target_actors = {
            uid: self.sensor.get_actor_ids(uid, [uid])[0].item()
            for uid in ("frame", "tool")
        }
        self.pads = tuple(
            self.sensor.get_actor_ids(robot.uid, ["fr3_leftpad", "fr3_rightpad"])[
                0
            ].tolist()
        )

    def resolve(self, phase, request, context, active_mask):
        pose = context.scene.entities[phase.target].pose
        goal = _target(pose, phase.target, approach=False)
        return ResolvedContactPhase(
            values=goal,
            held_qpos=context.robot.qpos,
            valid=active_mask,
            data=(
                (
                    "closed_qpos",
                    request.binding.endpoint("primary", "grasp").joint_positions(
                        "grasp",
                        num_envs=context.batch_size,
                        device=pose.device,
                        dtype=context.robot.qpos.dtype,
                    ),
                ),
                ("grip_z", goal[:, 2, 3]),
            ),
        )

    def hold_steps(self, target):
        raise ValueError(f"Unknown Tool Hang hold {target!r}.")

    def begin(self, cursor, context, active_mask):
        """No contact or object state changes when the plan starts."""

    def finish(self, cursor, context, active_mask):
        name = cursor.phase.target
        goal = _target(context.scene.entities[name].pose, name, approach=False)
        lifted = goal[:, 2, 3] >= dict(cursor.resolved.data)["grip_z"] + 0.009
        self.sensor.update()
        data = self.sensor.get_data()
        target = self.target_actors[name]
        valid = data["is_valid"][0] & data["impulse_valid"][0]
        pinched, available = classify_contacts(
            data["user_ids"][0, valid],
            data["normal"][0, valid],
            data["impulse"][0, valid],
            target=target,
            pads=self.pads,
        )
        ok = (
            active_mask
            & lifted
            & pinched
            & available
            & ~self.sensor.dropped_contacts_mask
        )
        if bool(ok.all()):
            return ok, ()
        return ok, (
            f"Pickup of {name!r} failed: lift={bool(lifted.item())}, "
            f"opposed_pad_contact={pinched}, contact_available={available}, "
            f"dropped_contacts={self.sensor.dropped_contacts_mask.item()}.",
        )


class _Pick(RegisteredSemanticLowerer):
    """Keep the perception source alive for this engine's entire physical grasp."""

    def __init__(self, source):
        self.source = source

    call_id: ClassVar = "tool_hang.pick"
    target_descriptor: ClassVar = PickUp.descriptor()
    requires_unoccupied_slots: ClassVar = ("primary",)

    def lower(self, call, *, context, bound, option_template):
        return SemanticLowering(
            goal=ContactPickUpGoal(
                self.source.source_id,
                (ContactPhase("grasp", call.arguments["object"]),),
            )
        )


@dataclass(frozen=True, slots=True)
class _ApproachFactory:
    call_id: ClassVar = _Approach.call_id
    revision: ClassVar = "1"
    target_descriptor: ClassVar = _Approach.target_descriptor

    @property
    def argument_contract(self):
        return SemanticCallContract(required={"object": ("frame", "tool")}, optional={})

    def create(self, *, simulation, robot, scene_registry, engine):
        return _Approach()


@dataclass(frozen=True, slots=True)
class _PickFactory(_ApproachFactory):
    call_id: ClassVar = _Pick.call_id
    target_descriptor: ClassVar = _Pick.target_descriptor

    def create(self, *, simulation, robot, scene_registry, engine):
        source = _PickupContact(simulation, robot)
        engine.planning_services.register_contact_source(source.source_id, source)
        return _Pick(source)


def _finger_effort(
    actions: Iterable[ControllerAction], joints: list[int], effort: float
) -> Iterator[ControllerAction]:
    """Add Panda preload on closure to each stock controller position/velocity command."""
    for action in actions:
        value = action.value
        if isinstance(value, torch.Tensor):
            value = TensorDict({"qpos": value}, batch_size=value.shape[:1])
        else:
            value = value.clone()
        qpos = value["qpos"]
        force = torch.zeros_like(qpos)
        force[:, joints] = torch.where(qpos[:, joints] < 0.014, -effort, 0.0)
        value["qf"] = force
        yield ControllerAction(value, metadata=action.metadata)


def _registration() -> SimulationTaskProgramRegistration:
    factories = (_ApproachFactory(), _PickFactory())
    catalog = builtin_semantic_call_catalog()
    for factory in factories:
        catalog = catalog.with_descriptor(
            SemanticCallDescriptor(
                call_id=factory.call_id,
                spec_type=RegisteredSemanticCall,
                target_descriptor=factory.target_descriptor,
            )
        )
    return SimulationTaskProgramRegistration(
        scene_binding=SimulationSceneBinding(
            registry_id="tool_hang_pickup",
            articulations=tuple(
                SimulationArticulationBinding(uid, uid) for uid in ("frame", "tool")
            ),
        ),
        robot_profile_binding=SimulationRobotSkillProfileBinding(
            profile_id="tool_hang_panda",
            resources=(
                ControlPartResourceBinding(
                    "panda",
                    (
                        ControlPartEndpointBinding(
                            "motion",
                            "arm",
                            frozenset(
                                {
                                    "motion.cartesian_pose",
                                    "kinematics.forward",
                                    "kinematics.inverse",
                                    "kinematics.batch_inverse",
                                }
                            ),
                        ),
                        ControlPartEndpointBinding(
                            "grasp",
                            "eef",
                            frozenset({"interaction.grasp"}),
                            command_preset="fingers",
                        ),
                    ),
                ),
            ),
            command_presets=(
                ControlPartCommandPreset(
                    "fingers", "eef", {"open": (0.04, 0.04), "grasp": (0.0, 0.0)}
                ),
            ),
            defaults={
                "move_end_effector": {"primary": "panda"},
                "pick_up": {"primary": "panda"},
            },
            presets=(
                SkillPolicyPreset(
                    "pickup",
                    effect_assurance=EffectAssurance.VERIFIED,
                    motion_policy=MotionPolicy(
                        sample_count=26,
                        dynamic_collision_mode="off",
                        acceleration_limit=None,
                        interpolation_mode="fixed_grid",
                        ik_seed_mode="sequential",
                    ),
                    recovery_policy=RecoveryPolicy(max_replans=0, max_skill_retries=0),
                    skill_option_templates={
                        _Approach.call_id: MoveEndEffectorOptions(),
                        _Pick.call_id: PickUpOptions(
                            pre_grasp_distance=0.06,
                            hand_interp_steps=3,
                            grasp_settle_steps=16,
                            lift_height=0.018,
                        ),
                    },
                ),
            ),
            default_preset="pickup",
        ),
        call_catalog=catalog,
        registered_semantic_lowerer_factories=factories,
    )


def pickup_segments(env: EmbodiedEnv, object_id: str) -> Iterator[DemoSegment]:
    """Compile and execute the approach and contact-qualified pickup on the live robot."""
    if object_id not in {"frame", "tool"}:
        raise ValueError(f"Unknown Tool Hang pickup object {object_id!r}.")
    program = load_task_program(
        files("embodichain.extensions.tasks").joinpath(
            "configs/tasks/manipulation/assembly/tool_hang/task_program",
            f"pick_{object_id}.yaml",
        ),
        integration=TaskProgramIntegrationCfg(
            robot_profile="tool_hang_panda",
            scene_registry="tool_hang_pickup",
            runtime_preset="pickup",
        ),
    )
    adapter = create_simulation_task_program_adapter(env, registration=_registration())
    try:
        compiled = adapter.compile(program)
        bridge = adapter.create_bridge(compiled)
        for segment in bridge.iter_segments():
            if segment.name.startswith("pick_"):
                yield replace(
                    segment,
                    actions=_finger_effort(
                        segment.actions,
                        env.robot.get_joint_ids("eef"),
                        12.0 if object_id == "frame" else 19.0,
                    ),
                )
            else:
                yield segment
    finally:
        adapter.close()
