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

"""Independent Atomic benchmark scenes built from the shared sample assets.

Run: embodichain benchmark atomic-action --skill pickup --robot ur5
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import math
from typing import Any

import torch

from embodichain.lab.sim.atomic_actions import (
    ActionInvocation,
    AtomicActionEngine,
    AxisAlignGoal,
    AxisAlignOptions,
    ControlPartCommandProfile,
    GraspGoal,
    HandOverGoal,
    HandOverOptions,
    JointPositionGoal,
    MotionPolicy,
    ObservedArticulationJointState,
    OpenDoorGoal,
    OpenDoorOptions,
    PickUpOptions,
    PlaceGoal,
    PlaceOptions,
    PourGoal,
    PourOptions,
    PressGoal,
    PressOptions,
    SceneSnapshot,
    TwistGoal,
    TwistOptions,
    create_simulation_atomic_action_engine,
)
from embodichain.lab.scripts._atomic_action import (
    axis_align,
    open_door,
    pickup,
    place,
    pour,
    press,
    slide,
    twist,
)
from embodichain.lab.scripts._atomic_action import support

__all__ = ["AtomicBenchmarkScene", "build_scene"]


@dataclass
class AtomicBenchmarkScene:
    """One live scene and its direct Atomic invocations, with no task runtime."""

    sim: Any
    robot: Any
    engine: AtomicActionEngine
    invocations: tuple[ActionInvocation, ...]
    target: Any = None
    rigid_object: Any = None
    link_name: str | None = None
    joint_index: int | None = None
    initial_scene: SceneSnapshot | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def observe(self) -> dict[str, torch.Tensor]:
        """Read actual simulation state rather than commanded or projected state."""
        robot = self.robot
        parts = (
            ("left_arm", "right_arm")
            if self.metadata["skill"] == "handover"
            else ("arm",)
        )
        observation = {"qpos": robot.get_qpos(target=False).detach().cpu().clone()}
        for part in parts:
            observation[part] = (
                robot.compute_fk(
                    robot.get_qpos(name=part, target=False), name=part, to_matrix=True
                )
                .detach()
                .cpu()
                .clone()
            )
        if self.rigid_object is not None:
            observation["object"] = (
                self.rigid_object.get_local_pose(to_matrix=True).detach().cpu().clone()
            )
        if self.joint_index is not None:
            observation["joint"] = (
                self.target.get_qpos(target=False)[:, self.joint_index]
                .detach()
                .cpu()
                .clone()
            )
        if self.link_name is not None:
            observation["link"] = (
                self.target.get_link_pose(self.link_name, to_matrix=True)
                .detach()
                .cpu()
                .clone()
            )
        return observation


def _create_robot(sim: Any, args: argparse.Namespace) -> Any:
    """Retain sample mounts and the run-env Handover Robotiq grippers."""
    skill = args._config["skill"]
    if skill == "handover":
        from embodichain.lab.scripts._atomic_action import handover

        return handover.create_dual_robot(sim, args.robot)
    kwargs: dict[str, Any] = {"tcp_point": "grasp"}
    if skill in {"move_joint", "press", "twist"}:
        kwargs["tcp_point"] = "default"
    if skill in {"press", "twist", "slide", "open_door"}:
        kwargs["init_qpos"] = [0.0, -1.57, 1.57, -3.14, -1.57, 0.0, 0.0, 0.0]
    return support.add_tutorial_robot(sim, args.robot, **kwargs)


def build_scene(args: argparse.Namespace) -> AtomicBenchmarkScene:
    """Build a simulation and goals directly from shared Atomic sample helpers.

    Args:
        args: Resolved sample arguments with robot, planner and scene options.

    Returns:
        Live scene with independent Atomic invocations and measured-state access.
    """
    skill = args._config["skill"]
    sim = support.create_tutorial_simulation(
        args, arena_space=3.0 if skill == "handover" else 2.5
    )
    robot = _create_robot(sim, args)

    target = obj = semantics = target_pose = None
    link = joint_index = None
    scene_snapshot = None
    metadata: dict[str, Any] = {
        "skill": skill,
        "support_z": 0.0,
        "gripper_urdfs": {
            name: component["urdf_path"]
            for name, component in robot.cfg.urdf_cfg.components.items()
            if name == "hand" or name.endswith("_hand")
        },
    }
    if skill in {"pickup", "place"}:
        obj = (pickup if skill == "pickup" else place).create_pick_object(sim)
        semantics = support.create_antipodal_semantics(obj, label="cube")
    elif skill in {"pour", "axis_align"}:
        obj = axis_align.create_align_object(
            sim, obj_position=pour.OBJ_POSITION if skill == "pour" else None
        )
        internal, target_axis = (
            (pour.POUR_INTERNAL_AXIS, None)
            if skill == "pour"
            else axis_align.ALIGNMENT_AXES[args.alignment]
        )
        semantics = axis_align.create_axis_align_semantics(obj, internal)
        metadata.update(internal_axis=list(internal), target_axis=target_axis)
    elif skill == "handover":
        from embodichain.lab.scripts._atomic_action import handover

        handover.create_support_surface(sim)
        obj = handover.create_handover_object(sim, args)
        semantics = support.create_antipodal_semantics(obj, label="handover")
        metadata.update(
            support_z=handover.SUPPORT_SURFACE_Z,
            target_position=list(handover.FINAL_OBJECT_XYZ),
        )
    elif skill == "press":
        target = press.create_microwave(sim)
        semantics, target_pose = press.create_button_semantics(target)
        link, joint_index = press.BUTTON_LINK_NAME, target.joint_names.index(
            "button_joint"
        )
        limits = target.get_qpos_limits()[0, joint_index]
        metadata.update(
            joint_target=float(limits[1]), press_distance=args.press_distance
        )
    elif skill == "twist":
        target = twist.create_microwave(sim)
        semantics, target_pose = twist.create_knob_semantics(
            target,
            grasp_roll_range=(
                (-math.pi, math.pi) if args.affordance_branches > 1 else None
            ),
        )
        link, joint_index = twist.KNOB_LINK_NAME, target.joint_names.index("knob_joint")
        metadata.update(
            axis_world=(
                target_pose[:, :3, :3]
                @ semantics.affordance.twist_axis.to(sim.device)[:, None]
            )
            .squeeze(-1)
            .cpu()
            .tolist(),
            twist_angle=args.twist_angle,
        )
    elif skill == "slide":
        target = slide.create_drawer(sim)
        semantics = slide.create_drawer_semantics(target)
        link = slide.HANDLE_LINK_NAME
        joint_index = 0
        target_pose = target.get_link_pose(link, to_matrix=True)
        metadata["joint_target"] = args.translation_distance
    elif skill == "open_door":
        target = open_door.create_microwave(sim)
        semantics = open_door.create_door_handle_semantics(target)
        link = open_door.HANDLE_LINK_NAME
        joint_index = target.joint_names.index(semantics.affordance.joint_name)
        target_pose = target.get_link_pose(link, to_matrix=True)
        metadata["joint_target"] = args.open_angle

    sim.prepare()
    if skill == "handover":
        sim.update(step=500)
        support.clone_local_pose_from_first_env(obj)
        obj.clear_dynamics()
    close = 0.040 if skill in {"press", "open_door"} else None
    if skill == "handover":
        hands = {
            name: support.get_hand_open_close_qpos(
                robot, hand_control_part=name, close_qpos=handover.HAND_CLOSE_QPOS
            )
            for name in ("left_eef", "right_eef")
        }
    elif skill != "move_joint":
        hands = {
            "eef": support.get_hand_open_close_qpos(
                robot, **({"close_qpos": close} if close is not None else {})
            )
        }
        if obj is not None:
            support.initialize_pre_pick_robot_pose(robot, obj, hands["eef"][0])
    else:
        hands = {}
    profiles = {
        name: ControlPartCommandProfile.joint_positions(open=values[0], grasp=values[1])
        for name, values in hands.items()
    }
    generators = {
        name: support.create_parallel_jaw_grasp_pose_generator(
            n_sample=args.n_sample,
            force_refresh=args.force_reannotate,
            **(
                {"gripper_model": handover.HANDOVER_GRIPPER_MODEL}
                if skill == "handover"
                else {}
            ),
        )
        for name in hands
    }
    engine_options = {
        "motion_generator": support.create_tutorial_motion_generator(
            robot, planner=args.planner
        ),
        "control_profiles": profiles,
        "grasp_pose_generators": generators,
    }
    engine = (
        AtomicActionEngine(**engine_options)
        if obj is None
        else create_simulation_atomic_action_engine(
            scene_entities=(obj,), **engine_options
        )
    )
    controls = {"primary": {"motion": "arm", "grasp": "eef"}}

    def invocation(
        name: str,
        goal: Any,
        count: int,
        options: Any = None,
        *,
        strategy: str = "motion_gen",
        parts: Any = None,
    ) -> ActionInvocation:
        return engine.make_invocation(
            name,
            goal,
            control_parts=controls if parts is None else parts,
            motion_policy=MotionPolicy(strategy=strategy, sample_count=count),
            **({"skill_options": options} if options is not None else {}),
        )

    if skill in {"pickup", "place", "pour"}:
        pick_options = PickUpOptions(
            pre_grasp_distance=0.15,
            lift_height=0.16,
            hand_interp_steps=12,
            **(
                {
                    "approach_direction": pickup.resolve_approach_direction(
                        args, sim.device
                    )
                }
                if skill == "pickup"
                else (
                    {
                        "approach_direction": torch.tensor(
                            pour.APPROACH_DIRECTION, device=sim.device
                        )
                    }
                    if skill == "pour"
                    else {}
                )
            ),
        )
        calls = [invocation("pick_up", GraspGoal(semantics), 120, pick_options)]
        if skill == "place":
            poses = place.make_place_eef_poses(sim.device)
            # Open the gripper close to the ground to limit landing drift.
            poses[-1, 2, 3] = 0.03
            metadata["release_pose"] = poses[-1].cpu().tolist()
            calls.append(
                invocation(
                    "place",
                    PlaceGoal(
                        support.broadcast_waypoint_pose_batch(poses, args.num_envs)
                    ),
                    120,
                    PlaceOptions(
                        lift_height=place.PLACE_LIFT_HEIGHT, hand_interp_steps=12
                    ),
                )
            )
        elif skill == "pour":
            metadata["pour_angle"] = args.rotate_angle
            calls.append(
                invocation(
                    "pour", PourGoal(), 80, PourOptions(rotate_angle=args.rotate_angle)
                )
            )
    elif skill == "press":
        calls = [
            invocation(
                "press",
                PressGoal(semantics, target_pose),
                256,
                PressOptions(
                    hand_interp_steps=12,
                    approach_distance=0.12,
                    press_distance=args.press_distance,
                ),
                strategy="ik_interp",
            )
        ]
    elif skill == "twist":
        calls = [
            invocation(
                "twist",
                TwistGoal(semantics, target_pose),
                256,
                TwistOptions(
                    hand_interp_steps=20,
                    pre_grasp_distance=0.12,
                    twist_angle=args.twist_angle,
                ),
            )
        ]
    elif skill == "slide":
        calls = [
            slide.create_invocation(
                engine,
                semantics,
                target_pose,
                direction="pull",
                approach_distance=args.approach_distance,
                translation_distance=args.translation_distance,
            )
        ]
    elif skill == "axis_align":
        calls = [
            invocation(
                "axis_align",
                AxisAlignGoal(semantics),
                180,
                AxisAlignOptions(
                    target_axis=torch.tensor(
                        metadata["target_axis"], device=sim.device
                    ),
                    approach_direction=torch.tensor(
                        [0.0, 0.0, -1.0], device=sim.device
                    ),
                    pre_grasp_distance=0.15,
                    lift_height=0.16,
                    hand_interp_steps=12,
                ),
            )
        ]
    elif skill == "open_door":
        affordance = semantics.affordance
        lower, upper = affordance.joint_limits
        closed, opened = (
            (lower, upper) if affordance.opening_direction > 0 else (upper, lower)
        )
        calls = [
            invocation(
                "open_door",
                OpenDoorGoal(
                    semantics,
                    target_pose,
                    open_fraction=(args.open_angle - closed) / (opened - closed),
                ),
                300,
                OpenDoorOptions(
                    hand_interp_steps=30,
                    door_waypoint_count=args.door_waypoint_count,
                    approach_distance=args.approach_distance,
                    retract_distance=args.retract_distance,
                ),
                strategy="ik_interp",
            )
        ]
        scene_snapshot = SceneSnapshot(
            timestamp=0.0,
            version=0,
            articulation_joints={
                (
                    open_door.MICROWAVE_SCENE_ENTITY_ID,
                    affordance.joint_name,
                ): ObservedArticulationJointState(
                    target.get_qpos(target=False)[:, joint_index : joint_index + 1]
                )
            },
        )
    elif skill == "handover":
        pose = torch.eye(4, device=sim.device)
        pose[:3, 3] = torch.tensor(handover.FINAL_OBJECT_XYZ, device=sim.device)
        calls = [
            invocation(
                "hand_over",
                HandOverGoal(semantics, target_pose=pose),
                220,
                HandOverOptions(
                    pre_grasp_distance=handover.HANDOVER_PRE_GRASP_DISTANCE,
                    lift_height=handover.HANDOVER_LIFT_HEIGHT,
                    hand_interp_steps=handover.HANDOVER_HAND_INTERP_STEPS,
                ),
                parts={
                    "source": {"motion": "left_arm", "grasp": "left_eef"},
                    "destination": {"motion": "right_arm", "grasp": "right_eef"},
                },
            )
        ]
    else:
        home = robot.get_qpos(name="arm")[0].clone()
        limits = robot.get_qpos_limits(name="arm")[0]
        targets = []
        for offsets in (
            (0.35, 0.37, -0.27, -0.08, 0.0, 0.20),
            (0.15, 0.17, -0.12, -0.03, 0.0, 0.10),
            (),
        ):
            value = home.clone()
            value[: len(offsets)] += torch.tensor(offsets, device=sim.device)
            targets.append(
                torch.minimum(torch.maximum(value, limits[:, 0]), limits[:, 1])
            )
        metadata["joint_targets"] = [value.detach().cpu().tolist() for value in targets]
        calls = [
            invocation(
                "move_joints",
                JointPositionGoal(value),
                120,
                parts={"primary": {"motion": "arm"}},
            )
            for value in targets
        ]

    metadata.update(
        hand_joint_ids={name: list(robot.get_joint_ids(name=name)) for name in hands},
        hand_open={
            name: values[0].detach().cpu().tolist() for name, values in hands.items()
        },
        hand_close={
            name: values[1].detach().cpu().tolist() for name, values in hands.items()
        },
        arm_joint_ids=list(
            robot.get_joint_ids(name="left_arm" if skill == "handover" else "arm")
        ),
    )
    support.prepare_tutorial_scene(sim, args, "")
    return AtomicBenchmarkScene(
        sim,
        robot,
        engine,
        tuple(calls),
        target,
        obj,
        link,
        joint_index,
        scene_snapshot,
        metadata,
    )
