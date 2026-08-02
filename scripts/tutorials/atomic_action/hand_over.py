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

"""Demonstrate a dual-arm handover with a single textured mesh object.

The left arm picks the object up by its top part, hands it to the right arm at a
middle handover pose, the right arm grasps the bottom part, the left arm
releases, and the right arm carries the object to the other side.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch

from embodichain.lab.gym.utils.gym_utils import add_env_launcher_args_to_parser
from embodichain.lab.sim import SimulationManager
from embodichain.lab.sim.atomic_actions import (
    ActionBinding,
    ActionInvocation,
    GraspGoal,
    AtomicActionEngine,
    ControlPartCommandProfile,
    HandOver,
    HandOverOptions,
    PickUp,
    PickUpOptions,
    MotionPolicy,
)
from embodichain.lab.sim.cfg import (
    JointDrivePropertiesCfg,
    RigidBodyAttributesCfg,
    RigidObjectCfg,
    RobotCfg,
    URDFCfg,
)
from embodichain.data import get_data_path
from embodichain.lab.sim.objects import RigidObject, Robot
from embodichain.lab.sim.shapes import CubeCfg, MeshCfg
from embodichain.lab.sim.solvers import URSolverCfg
from embodichain.utils import logger
from scripts.tutorials.atomic_action.tutorial_utils import (
    create_antipodal_semantics,
    create_toppra_motion_generator,
    create_tutorial_simulation,
    get_hand_open_close_qpos,
    clone_local_pose_from_first_env,
    prepare_tutorial_scene,
    publish_tutorial_scene,
    replay_trajectory,
    serve_tutorial_scene,
)

ARM_URDF_PATH = "UniversalRobots/UR5/UR5.urdf"
GRIPPER_URDF_PATH = "DH_PGI_140_80/DH_PGI_140_80.urdf"
OBJECT_MESH_PATH = get_data_path("SodaCan/simple_cola_can.obj")
GRIPPER_TCP_Z = 0.155
ROBOT_INIT_POS = (1.95, 0.0, 0.1)
ROBOT_INIT_ROT = (0.0, 0.0, -90.0)
LEFT_ARM_HOME = (0.0, 0.0, -1.57, -1.57, 1.57, 1.57)
RIGHT_ARM_HOME = (-1.57, -1.57, -1.57, -1.57, 0.0, 0.0)
SUPPORT_SURFACE_Z = 0.50
SUPPORT_SURFACE_SIZE = (0.8, 1.2, 0.02)
SUPPORT_SURFACE_CENTER = (
    0.0,
    0.0,
    SUPPORT_SURFACE_Z - 0.5 * SUPPORT_SURFACE_SIZE[2],
)

# --- Adjustable scene placeholders -----------------------------------------
# The object starts on the left side, is handed over at a lifted middle pose,
# and is delivered to the right side. Tweak these to match the mesh geometry
# and the dual-UR5 reach.
OBJECT_INIT_XY = (0.0, 0.02)
MIDDLE_OBJECT_XYZ = (0.0, 0.02, 0.82)
MIDDLE_OBJECT_YAW_DEG = 0.0
FINAL_OBJECT_XYZ = (0.22, 0.02, 0.72)
FINAL_OBJECT_YAW_DEG = 0.0
# ---------------------------------------------------------------------------

HAND_CLOSE_QPOS = 0.026
PICKUP_SAMPLE_INTERVAL = 80
PICKUP_HAND_INTERP_STEPS = 5
PICKUP_PRE_GRASP_DISTANCE = 0.08
PICKUP_LIFT_HEIGHT = 0.1
HANDOVER_SAMPLE_INTERVAL = 140
HANDOVER_HAND_INTERP_STEPS = 10
HANDOVER_HOLD_STEPS = 4
HANDOVER_RETREAT_STEPS = 28
HANDOVER_PRE_GRASP_DISTANCE = 0.08
HANDOVER_LIFT_HEIGHT = 0.08
TRAJECTORY_SIM_STEPS = 4
HANDOVER_RECORD_LOOK_AT = (
    (-0.25, 0.02, 2.5),
    (0.0, 0.02, 0.75),
    (0.0, 0.0, 1.0),
)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the demo."""
    parser = argparse.ArgumentParser(description="Dual-arm handover demo")
    add_env_launcher_args_to_parser(parser)
    parser.set_defaults(device="cpu", renderer="hybrid")
    parser.add_argument(
        "--diagnose_plan",
        action="store_true",
        help="Plan and print diagnostics without playing the trajectory.",
    )
    parser.add_argument(
        "--auto_play",
        action="store_true",
        help="Run the viewer demo without waiting for keyboard input.",
    )
    parser.add_argument(
        "--headless_play",
        action="store_true",
        help="Execute planned trajectories without opening the viewer window.",
    )
    return parser.parse_args()


def rotation_z(yaw: float) -> np.ndarray:
    """Build a 3x3 yaw rotation matrix."""
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    return np.array(
        [
            [cos_yaw, -sin_yaw, 0.0],
            [sin_yaw, cos_yaw, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def make_transform(xyz: tuple[float, float, float], yaw: float) -> np.ndarray:
    """Build a homogeneous transform from translation and yaw."""
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = rotation_z(yaw)
    transform[:3, 3] = np.asarray(xyz, dtype=np.float32)
    return transform


def create_dual_ur5_robot(sim: SimulationManager) -> Robot:
    """Create a dual-UR5 robot with one PGI gripper on each arm."""
    arm_urdf_path = ARM_URDF_PATH
    gripper_urdf_path = GRIPPER_URDF_PATH
    tcp = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, GRIPPER_TCP_Z],
        [0.0, 0.0, 0.0, 1.0],
    ]
    cfg = RobotCfg(
        uid="DualUR5HandOver",
        urdf_cfg=URDFCfg(
            components=[
                {
                    "component_type": "left_arm",
                    "urdf_path": arm_urdf_path,
                    "transform": make_transform((-0.3, -1.45, 0.4), np.pi / 2),
                },
                {
                    "component_type": "right_arm",
                    "urdf_path": arm_urdf_path,
                    "transform": make_transform((0.3, -1.45, 0.4), np.pi / 2),
                },
                {"component_type": "left_hand", "urdf_path": gripper_urdf_path},
                {"component_type": "right_hand", "urdf_path": gripper_urdf_path},
            ],
            fname="dual_ur5_hand_over",
            name_case={"joint": "upper", "link": "lower"},
        ),
        drive_pros=JointDrivePropertiesCfg(
            stiffness={
                "LEFT_JOINT[0-9]": 1e4,
                "RIGHT_JOINT[0-9]": 1e4,
                "LEFT_GRIPPER_FINGER[1-2]_JOINT_1": 1e2,
                "RIGHT_GRIPPER_FINGER[1-2]_JOINT_1": 1e2,
            },
            damping={
                "LEFT_JOINT[0-9]": 1e3,
                "RIGHT_JOINT[0-9]": 1e3,
                "LEFT_GRIPPER_FINGER[1-2]_JOINT_1": 1e1,
                "RIGHT_GRIPPER_FINGER[1-2]_JOINT_1": 1e1,
            },
            max_effort={
                "LEFT_JOINT[0-9]": 1e5,
                "RIGHT_JOINT[0-9]": 1e5,
                "LEFT_GRIPPER_FINGER[1-2]_JOINT_1": 1e3,
                "RIGHT_GRIPPER_FINGER[1-2]_JOINT_1": 1e3,
            },
            drive_type="force",
        ),
        control_parts={
            "left_arm": ["LEFT_JOINT[0-9]"],
            "right_arm": ["RIGHT_JOINT[0-9]"],
            "dual_arm": ["LEFT_JOINT[0-9]", "RIGHT_JOINT[0-9]"],
            "left_hand": ["LEFT_GRIPPER_FINGER1_JOINT_1"],
            "right_hand": ["RIGHT_GRIPPER_FINGER1_JOINT_1"],
        },
        solver_cfg={
            "left_arm": URSolverCfg(
                ur_type="ur5",
                tcp=tcp,
                end_link_name="left_ee_link",
                root_link_name="left_base_link",
                ik_nearest_weight=[1.0, 4.0, 1.0, 1.0, 1.0, 1.0],
            ),
            "right_arm": URSolverCfg(
                ur_type="ur5",
                tcp=tcp,
                end_link_name="right_ee_link",
                root_link_name="right_base_link",
                ik_nearest_weight=[1.0, 4.0, 1.0, 1.0, 1.0, 1.0],
            ),
        },
        init_pos=list(ROBOT_INIT_POS),
        init_rot=list(ROBOT_INIT_ROT),
        init_qpos=list(LEFT_ARM_HOME) + list(RIGHT_ARM_HOME) + [0.0, 0.0, 0.0, 0.0],
    )
    return sim.add_robot(cfg=cfg)


def create_support_surface(sim: SimulationManager) -> RigidObject:
    """Create a compact support slab under the staged object."""
    return sim.add_rigid_object(
        cfg=RigidObjectCfg(
            uid="support_surface",
            shape=CubeCfg(size=list(SUPPORT_SURFACE_SIZE)),
            attrs=RigidBodyAttributesCfg(
                mass=10.0,
                dynamic_friction=0.9,
                static_friction=0.95,
                restitution=0.01,
            ),
            body_type="static",
            init_pos=list(SUPPORT_SURFACE_CENTER),
            init_rot=[0.0, 0.0, 0.0],
        )
    )


def create_handover_object(sim: SimulationManager) -> RigidObject:
    """Create the textured mesh object on the support surface."""
    return sim.add_rigid_object(
        cfg=RigidObjectCfg(
            uid="handover_object",
            shape=MeshCfg(fpath=OBJECT_MESH_PATH, compute_uv=False),
            attrs=RigidBodyAttributesCfg(
                mass=0.01,
                dynamic_friction=0.97,
                static_friction=0.99,
                angular_damping=1.0,
                linear_damping=0.5,
                contact_offset=0.001,
                rest_offset=0.0,
                restitution=0.01,
                min_position_iters=32,
                min_velocity_iters=8,
                max_depenetration_velocity=2.0,
            ),
            max_convex_hull_num=1,
            init_pos=[OBJECT_INIT_XY[0], OBJECT_INIT_XY[1], SUPPORT_SURFACE_Z + 0.12],
            init_rot=[90.0, 0.0, 0.0],
            body_scale=(0.56, 0.56, 0.56),
        )
    )


def settle_object(sim: SimulationManager, obj: RigidObject, step: int = 5) -> None:
    """Settle an object before planning."""
    if sim.device.type == "cuda":
        sim.init_gpu_physics()
    obj.reset()
    if step > 0:
        sim.update(step=step)
    obj.clear_dynamics()


def run_handover_demo(
    args: argparse.Namespace,
    sim: SimulationManager,
    robot: Robot,
) -> None:
    """Plan and optionally execute a pick-up followed by a handover."""
    create_support_surface(sim)
    obj = create_handover_object(sim)
    settle_object(sim, obj, step=0)
    clone_local_pose_from_first_env(obj)
    obj.clear_dynamics()
    publish_tutorial_scene(sim, args)
    object_semantics = create_antipodal_semantics(
        obj, label="handover", n_sample=10000, force_reannotate=False
    )
    motion_gen = create_toppra_motion_generator(robot)

    left_open, left_close = get_hand_open_close_qpos(
        robot, hand_control_part="left_hand", close_qpos=HAND_CLOSE_QPOS
    )
    right_open, right_close = get_hand_open_close_qpos(
        robot, hand_control_part="right_hand", close_qpos=HAND_CLOSE_QPOS
    )

    middle_pose = torch.as_tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0, 0.7],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )

    final_pose = torch.as_tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, -0.2],
            [0.0, 1.0, 0.0, 0.7],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )

    # Step 1 - the left arm picks the object up by its top part.
    pick_up_action = PickUp(
        default_options=PickUpOptions(
            pick_object_part="top",
            pre_grasp_distance=PICKUP_PRE_GRASP_DISTANCE,
            lift_height=PICKUP_LIFT_HEIGHT,
            hand_interp_steps=PICKUP_HAND_INTERP_STEPS,
            approach_direction=torch.as_tensor(
                [0.0, -707106781, -707106781], dtype=torch.float32
            ),
        ),
    )
    # Step 2 - hand the object from the left arm to the right arm.
    handover_action = HandOver(
        default_options=HandOverOptions(
            receive_pick_object_part="bottom",
            middle_object_pose=middle_pose,
            final_object_pose=final_pose,
            pre_grasp_distance=HANDOVER_PRE_GRASP_DISTANCE,
            lift_height=HANDOVER_LIFT_HEIGHT,
            hand_interp_steps=HANDOVER_HAND_INTERP_STEPS,
            hold_steps=HANDOVER_HOLD_STEPS,
            retreat_steps=HANDOVER_RETREAT_STEPS,
            receive_approach_direction=torch.as_tensor(
                [0.0, 707106781, -707106781], dtype=torch.float32
            ),
        ),
    )
    engine = AtomicActionEngine(
        motion_generator=motion_gen,
        control_profiles={
            "left_hand": ControlPartCommandProfile.joint_positions(
                open=left_open,
                grasp=left_close,
            ),
            "right_hand": ControlPartCommandProfile.joint_positions(
                open=right_open,
                grasp=right_close,
            ),
        },
    )
    engine.register(pick_up_action)
    engine.register(handover_action)

    wait_for_user = prepare_tutorial_scene(
        sim, args, "Inspect the scene, then press Enter to plan the handover..."
    )
    # wait for object to drop
    for _ in range(20):
        sim.update(step=10)
    compiled = engine.compile(
        (
            ActionInvocation(
                "pick_up",
                GraspGoal(object_semantics),
                ActionBinding(
                    manipulators={"primary": "left_arm"},
                    end_effectors={"primary": "left_hand"},
                ),
                MotionPolicy(sample_count=PICKUP_SAMPLE_INTERVAL),
            ),
            ActionInvocation(
                "hand_over",
                GraspGoal(object_semantics),
                ActionBinding(
                    manipulators={
                        "source": "left_arm",
                        "destination": "right_arm",
                    },
                    end_effectors={
                        "source": "left_hand",
                        "destination": "right_hand",
                    },
                ),
                MotionPolicy(sample_count=HANDOVER_SAMPLE_INTERVAL),
            ),
        )
    )
    success = compiled.plan_success
    traj = compiled.trajectory.positions

    if not success.all():
        logger.log_warning("Failed to plan the full pick-up + handover trajectory.")
        return

    if args.diagnose_plan:
        logger.log_info(f"Planned full trajectory with {traj.shape[1]} waypoints.")
        return

    if wait_for_user:
        input("Press Enter to execute the handover...")

    replay_trajectory(
        sim,
        robot,
        traj,
        args,
        video_prefix="handover_auto_play",
        hold_steps=0,
        trajectory_sim_steps=TRAJECTORY_SIM_STEPS,
        look_at=HANDOVER_RECORD_LOOK_AT,
    )
    if wait_for_user:
        input("Press Enter to exit the simulation...")


def main() -> None:
    """Run the dual-arm handover demo."""
    args = parse_arguments()
    sim = create_tutorial_simulation(
        args,
        arena_space=3.0,
        light_pos=(0.0, -0.4, 3.0),
    )
    robot = create_dual_ur5_robot(sim)
    try:
        run_handover_demo(args, sim, robot)
        serve_tutorial_scene(sim, args)
    finally:
        sim.destroy()


if __name__ == "__main__":
    main()
