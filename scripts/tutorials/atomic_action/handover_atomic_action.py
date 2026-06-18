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

"""
This script demonstrates a dual-arm atomic handover skill.

The right UR10 first picks up a bottle, then hands it to the left UR10 in the air
using HandoverAction.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch

from embodichain.lab.gym.utils.gym_utils import add_env_launcher_args_to_parser
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.atomic_actions import (
    AntipodalAffordance,
    HandoverAction,
    HandoverActionCfg,
    ObjectSemantics,
    PickUpAction,
    PickUpActionCfg,
)
from embodichain.lab.sim.cfg import (
    JointDrivePropertiesCfg,
    LightCfg,
    RenderCfg,
    RigidBodyAttributesCfg,
    RigidObjectCfg,
    RobotCfg,
    URDFCfg,
)
from embodichain.lab.sim.objects import RigidObject, Robot
from embodichain.lab.sim.planners import MotionGenCfg, MotionGenerator, ToppraPlannerCfg
from embodichain.lab.sim.shapes import CubeCfg, MeshCfg
from embodichain.lab.sim.solvers import PytorchSolverCfg
from embodichain.toolkits.graspkit.pg_grasp.antipodal_generator import (
    AntipodalSamplerCfg,
    GraspGeneratorCfg,
)
from embodichain.toolkits.graspkit.pg_grasp.gripper_collision_checker import (
    GripperCollisionCfg,
)
from embodichain.utils import logger


ARM_URDF_PATH = "UniversalRobots/UR10/UR10.urdf"
GRIPPER_URDF_PATH = "DH_PGI_140_80/DH_PGI_140_80.urdf"
BOTTLE_MESH_PATH = "ScannedBottle/yibao.ply"
BOTTLE_LABEL = "bottle"
GRIPPER_MAX_OPEN_WIDTH = 0.080
GRIPPER_FINGER_LENGTH = 0.088
GRIPPER_ROOT_Z_WIDTH = 0.096
GRIPPER_Y_THICKNESS = 0.040
GRIPPER_TCP_Z = 0.121
BOTTLE_GRASP_COLLISION_THRESHOLD = -0.004
PICK_SAMPLE_INTERVAL = 100
HANDOVER_SAMPLE_INTERVAL = 100
ROBOT_INIT_POS = (2.4, 0.0, 0.1)
ROBOT_INIT_ROT = (0.0, 0.0, -90.0)
TABLE_TOP_Z = 0.65
TABLE_SIZE = (1.6, 1.2, 0.02)
TABLE_CENTER = (-0.45, 0.0, TABLE_TOP_Z - TABLE_SIZE[2] * 0.5)
BOTTLE_FOOTPRINT_CENTER_XY = (0.0, -0.15)
BOTTLE_TABLE_CLEARANCE = 0.0
BOTTLE_INIT_ROT = (180.0, 0.0, -90.0)
BOTTLE_LOCAL_FOOTPRINT_CENTER_AFTER_INIT_ROT = (
    0.141664481,
    0.025152357,
)
BOTTLE_LOCAL_Z_MIN_AFTER_INIT_ROT = -0.14948709716796874
BOTTLE_LOCAL_CENTER = (
    0.02234459877014161,
    0.1391264892578125,
    0.07730183506011963,
)
BOTTLE_LOCAL_AXIS_MIN = 0.005116572952270508
BOTTLE_LOCAL_AXIS_MAX = 0.14948709716796876
HANDOVER_BOTTLE_CENTER = (-0.35, -0.05, 1.0)
HANDOVER_RECEIVER_AXIS_MARGIN = 0.035


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the demo."""
    parser = argparse.ArgumentParser(description="Dual-arm bottle handover demo")
    add_env_launcher_args_to_parser(parser)
    parser.add_argument(
        "--n_sample",
        type=int,
        default=10000,
        help="Number of surface samples for antipodal grasp generation.",
    )
    parser.add_argument(
        "--force_reannotate",
        action="store_true",
        help="Force grasp region re-annotation for the bottle.",
    )
    parser.add_argument(
        "--diagnose_plan",
        action="store_true",
        help="Plan and print diagnostics without playing the trajectory.",
    )
    parser.add_argument(
        "--debug_hand_state",
        action="store_true",
        help="Log hand targets and object pose during execution.",
    )
    parser.add_argument(
        "--auto_play",
        action="store_true",
        help="Run the viewer demo without waiting for keyboard input.",
    )
    return parser.parse_args()


def get_cached_data_path(data_path: str) -> str:
    """Resolve an asset path from the local cache before importing data helpers."""
    if os.path.isabs(data_path):
        return data_path

    data_root = Path(
        os.environ.get(
            "EMBODICHAIN_DATA_ROOT",
            str(Path.home() / ".cache" / "embodichain_data"),
        )
    )
    candidates = (
        data_root / data_path,
        data_root / "extract" / data_path,
    )
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    from embodichain.data import get_data_path

    return get_data_path(data_path)


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


def initialize_simulation(args: argparse.Namespace) -> SimulationManager:
    """Create the simulation manager and a light."""
    sim = SimulationManager(
        SimulationManagerCfg(
            headless=True,
            sim_device=args.device,
            render_cfg=RenderCfg(renderer=args.renderer),
            physics_dt=1.0 / 100.0,
            arena_space=3.0,
        )
    )
    sim.add_light(
        cfg=LightCfg(
            uid="main_light",
            color=(0.6, 0.6, 0.6),
            intensity=30.0,
            init_pos=(0.0, -0.4, 3.0),
        )
    )
    return sim


def create_dual_ur10_robot(sim: SimulationManager) -> Robot:
    """Create a dual-UR10 robot with one PGI gripper on each arm."""
    arm_urdf_path = get_cached_data_path(ARM_URDF_PATH)
    gripper_urdf_path = get_cached_data_path(GRIPPER_URDF_PATH)
    tcp = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, GRIPPER_TCP_Z],
        [0.0, 0.0, 0.0, 1.0],
    ]
    left_arm_home = [0.0, 0.0, -1.57, -1.57, 1.57, 1.57]
    right_arm_home = [-1.57, -1.57, -1.57, -1.57, 0.0, 0.0]
    cfg = RobotCfg(
        uid="DualUR10Handover",
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
            fname="dual_ur10_handover",
        ),
        drive_pros=JointDrivePropertiesCfg(
            stiffness={
                "LEFT_JOINT[0-9]": 1e4,
                "RIGHT_JOINT[0-9]": 1e4,
                "LEFT_GRIPPER_FINGER[1-2]_JOINT_1": 1e3,
                "RIGHT_GRIPPER_FINGER[1-2]_JOINT_1": 1e3,
            },
            damping={
                "LEFT_JOINT[0-9]": 1e3,
                "RIGHT_JOINT[0-9]": 1e3,
                "LEFT_GRIPPER_FINGER[1-2]_JOINT_1": 1e2,
                "RIGHT_GRIPPER_FINGER[1-2]_JOINT_1": 1e2,
            },
            max_effort={
                "LEFT_JOINT[0-9]": 1e5,
                "RIGHT_JOINT[0-9]": 1e5,
                "LEFT_GRIPPER_FINGER[1-2]_JOINT_1": 1e4,
                "RIGHT_GRIPPER_FINGER[1-2]_JOINT_1": 1e4,
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
            "left_arm": PytorchSolverCfg(
                end_link_name="left_ee_link",
                root_link_name="left_base_link",
                tcp=tcp,
                num_samples=30,
            ),
            "right_arm": PytorchSolverCfg(
                end_link_name="right_ee_link",
                root_link_name="right_base_link",
                tcp=tcp,
                num_samples=30,
            ),
        },
        init_pos=list(ROBOT_INIT_POS),
        init_rot=list(ROBOT_INIT_ROT),
        init_qpos=left_arm_home + right_arm_home + [0.0, 0.0, 0.0, 0.0],
    )
    return sim.add_robot(cfg=cfg)


def create_support_plane(sim: SimulationManager) -> RigidObject:
    """Create a static support plane."""
    return sim.add_rigid_object(
        cfg=RigidObjectCfg(
            uid="support_plane",
            shape=CubeCfg(size=list(TABLE_SIZE)),
            attrs=RigidBodyAttributesCfg(
                dynamic_friction=0.97,
                static_friction=0.99,
            ),
            body_type="static",
            init_pos=list(TABLE_CENTER),
        )
    )


def create_bottle(sim: SimulationManager) -> RigidObject:
    """Create the bottle to be picked and handed off."""
    bottle_scale = 0.0008
    bottle_init_xy = [
        BOTTLE_FOOTPRINT_CENTER_XY[axis]
        - BOTTLE_LOCAL_FOOTPRINT_CENTER_AFTER_INIT_ROT[axis]
        for axis in range(2)
    ]
    bottle_init_z = (
        TABLE_TOP_Z
        + BOTTLE_TABLE_CLEARANCE
        - BOTTLE_LOCAL_Z_MIN_AFTER_INIT_ROT
    )
    return sim.add_rigid_object(
        cfg=RigidObjectCfg(
            uid="bottle",
            shape=MeshCfg(fpath=get_cached_data_path(BOTTLE_MESH_PATH)),
            attrs=RigidBodyAttributesCfg(
                mass=0.02,
                dynamic_friction=0.97,
                static_friction=0.99,
                angular_damping=2.0,
                linear_damping=1.0,
                contact_offset=0.001,
                rest_offset=0.0,
                min_position_iters=32,
                min_velocity_iters=4,
                max_depenetration_velocity=2.0,
            ),
            max_convex_hull_num=16,
            init_pos=[bottle_init_xy[0], bottle_init_xy[1], bottle_init_z],
            init_rot=list(BOTTLE_INIT_ROT),
            body_scale=(bottle_scale, bottle_scale, bottle_scale),
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


def build_grasp_generator_cfg(args: argparse.Namespace) -> GraspGeneratorCfg:
    """Build antipodal grasp generator config."""
    return GraspGeneratorCfg(
        viser_port=11802,
        antipodal_sampler_cfg=AntipodalSamplerCfg(
            n_sample=args.n_sample,
            max_length=GRIPPER_MAX_OPEN_WIDTH,
            min_length=0.003,
        ),
        is_partial_annotate=False,
        is_filter_ground_collision=False,
    )


def build_gripper_collision_cfg() -> GripperCollisionCfg:
    """Build gripper collision config for grasp generation."""
    return GripperCollisionCfg(
        max_open_length=GRIPPER_MAX_OPEN_WIDTH,
        finger_length=GRIPPER_FINGER_LENGTH,
        y_thickness=GRIPPER_Y_THICKNESS,
        root_z_width=GRIPPER_ROOT_Z_WIDTH,
        open_check_margin=0.002,
        point_sample_dense=0.012,
    )


def create_object_semantics(
    obj: RigidObject, args: argparse.Namespace
) -> ObjectSemantics:
    """Create bottle semantics for PickUpAction."""
    return ObjectSemantics(
        label=BOTTLE_LABEL,
        geometry={
            "mesh_vertices": obj.get_vertices(env_ids=[0], scale=True)[0],
            "mesh_triangles": obj.get_triangles(env_ids=[0])[0],
        },
        affordance=AntipodalAffordance(
            object_label=BOTTLE_LABEL,
            force_reannotate=args.force_reannotate,
            custom_config={
                "gripper_collision_cfg": build_gripper_collision_cfg(),
                "generator_cfg": build_grasp_generator_cfg(args),
            },
        ),
        entity=obj,
    )


def get_hand_open_close_qpos(
    robot: Robot, hand_control_part: str, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get open and close qpos for a PGI gripper control part."""
    limits = robot.get_qpos_limits(name=hand_control_part)[0].to(
        device=device, dtype=torch.float32
    )
    hand_open = limits[:, 0]
    hand_close = torch.minimum(limits[:, 1], torch.full_like(limits[:, 1], 0.030))
    return hand_open, hand_close


def format_tensor(tensor: torch.Tensor) -> str:
    """Format tensor values for compact logging."""
    rounded = (tensor.detach().cpu() * 10000.0).round() / 10000.0
    return str(rounded.tolist())


def build_horizontal_bottle_pose(device: torch.device) -> torch.Tensor:
    """Build a hand-tuned horizontal bottle pose for the mid-air handover."""
    pose = torch.eye(4, dtype=torch.float32, device=device)
    pose[:3, 0] = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=device)
    pose[:3, 1] = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device=device)
    pose[:3, 2] = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=device)

    bottle_center = torch.tensor(
        HANDOVER_BOTTLE_CENTER, dtype=torch.float32, device=device
    )
    local_center = torch.tensor(BOTTLE_LOCAL_CENTER, dtype=torch.float32, device=device)
    pose[:3, 3] = bottle_center - pose[:3, :3] @ local_center
    return pose


def build_object_aware_handover_target(
    robot: Robot, obj: RigidObject, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build dual-arm handover poses from the current held-object geometry."""
    obj_pose = obj.get_local_pose(to_matrix=True).to(device=device, dtype=torch.float32)
    giver_tcp = robot.compute_fk(
        qpos=robot.get_qpos(name="right_arm"), name="right_arm", to_matrix=True
    )
    obj_to_giver = torch.bmm(torch.linalg.inv(obj_pose), giver_tcp)

    handover_obj_pose = build_horizontal_bottle_pose(device).unsqueeze(0).repeat(
        obj_pose.shape[0], 1, 1
    )
    giver_handover_pose = torch.bmm(handover_obj_pose, obj_to_giver)

    axis_min = BOTTLE_LOCAL_AXIS_MIN + HANDOVER_RECEIVER_AXIS_MARGIN
    axis_max = BOTTLE_LOCAL_AXIS_MAX - HANDOVER_RECEIVER_AXIS_MARGIN
    axis_center = 0.5 * (BOTTLE_LOCAL_AXIS_MIN + BOTTLE_LOCAL_AXIS_MAX)
    receiver_obj_pose = obj_to_giver.clone()
    giver_axis = obj_to_giver[:, 2, 3]
    receiver_axis = torch.where(
        giver_axis < axis_center,
        torch.full_like(giver_axis, axis_max),
        torch.full_like(giver_axis, axis_min),
    )
    receiver_obj_pose[:, 2, 3] = receiver_axis
    receiver_handover_pose = torch.bmm(handover_obj_pose, receiver_obj_pose)

    handover_target = torch.stack(
        [giver_handover_pose[0], receiver_handover_pose[0]], dim=0
    )
    return handover_target, handover_obj_pose[0]


def log_handover_geometry(target: torch.Tensor, bottle_pose: torch.Tensor) -> None:
    """Log the object-aware handover geometry."""
    tcp_delta = target[0, :3, 3] - target[1, :3, 3]
    tcp_distance = torch.linalg.norm(tcp_delta)
    bottle_center = torch.tensor(
        HANDOVER_BOTTLE_CENTER, dtype=torch.float32, device=target.device
    )
    logger.log_info(
        "handover target geometry: "
        f"giver_tcp={format_tensor(target[0, :3, 3])}, "
        f"receiver_tcp={format_tensor(target[1, :3, 3])}, "
        f"tcp_distance={format_tensor(tcp_distance)}, "
        f"bottle_center={format_tensor(bottle_center)}, "
        f"bottle_axis={format_tensor(bottle_pose[:3, 2])}"
    )


def log_tcp_alignment(
    robot: Robot,
    traj: torch.Tensor,
    joint_ids: list[int],
    target: torch.Tensor,
    action: HandoverAction,
) -> None:
    """Log TCP alignment at the end of the approach phase."""
    segments = action.get_segment_lengths()
    approach_idx = segments["approach"] - 1
    joint_id_to_col = {joint_id: col for col, joint_id in enumerate(joint_ids)}
    giver_cols = [joint_id_to_col[joint_id] for joint_id in action.giver_arm_joint_ids]
    receiver_cols = [
        joint_id_to_col[joint_id] for joint_id in action.receiver_arm_joint_ids
    ]
    giver_tcp = robot.compute_fk(
        qpos=traj[:, approach_idx, giver_cols],
        name=action.cfg.giver_arm_control_part,
        to_matrix=True,
    )
    receiver_tcp = robot.compute_fk(
        qpos=traj[:, approach_idx, receiver_cols],
        name=action.cfg.receiver_arm_control_part,
        to_matrix=True,
    )
    for label, tcp, target_pose in (
        ("giver", giver_tcp, target[None, 0]),
        ("receiver", receiver_tcp, target[None, 1]),
    ):
        pos_error = torch.norm(tcp[:, :3, 3] - target_pose[:, :3, 3], dim=1)
        rot_delta = torch.bmm(tcp[:, :3, :3].transpose(1, 2), target_pose[:, :3, :3])
        trace = rot_delta[:, 0, 0] + rot_delta[:, 1, 1] + rot_delta[:, 2, 2]
        rot_error = torch.acos(((trace - 1.0) * 0.5).clamp(-1.0, 1.0))
        logger.log_info(
            f"{label} handover tcp pos={format_tensor(tcp[0, :3, 3])}, "
            f"target={format_tensor(target_pose[0, :3, 3])}, "
            f"pos_error={format_tensor(pos_error)}, "
            f"rot_error_rad={format_tensor(rot_error)}"
        )


def log_action_plan(
    robot: Robot,
    action_name: str,
    traj: torch.Tensor,
    joint_ids: list[int],
    segments: dict[str, int] | None = None,
) -> None:
    """Log common action plan details."""
    joint_names = [robot.joint_names[joint_id] for joint_id in joint_ids]
    logger.log_info(f"{action_name} joint ids: {joint_ids}")
    logger.log_info(f"{action_name} joint names: {joint_names}")
    logger.log_info(f"{action_name} trajectory shape: {tuple(traj.shape)}")
    if segments is not None:
        logger.log_info(f"{action_name} trajectory segments: {segments}")


def log_execution_state(
    robot: Robot,
    obj: RigidObject,
    step_idx: int,
    total_steps: int,
) -> None:
    """Log hand and bottle state during execution."""
    obj_pose = obj.get_local_pose(to_matrix=True)
    left_hand = robot.get_qpos(name="left_hand")
    right_hand = robot.get_qpos(name="right_hand")
    logger.log_info(
        f"step={step_idx}/{total_steps - 1}, "
        f"left_hand={format_tensor(left_hand[0])}, "
        f"right_hand={format_tensor(right_hand[0])}, "
        f"bottle_pos={format_tensor(obj_pose[0, :3, 3])}"
    )


def execute_trajectory(
    sim: SimulationManager,
    robot: Robot,
    traj: torch.Tensor,
    joint_ids: list[int],
    obj: RigidObject,
    debug_hand_state: bool,
) -> None:
    """Play a planned trajectory in simulation."""
    total_steps = traj.shape[1]
    log_stride = max(1, total_steps // 10)
    for i in range(total_steps):
        robot.set_qpos(traj[:, i, :], joint_ids=joint_ids)
        sim.update(step=4)
        if debug_hand_state and (i % log_stride == 0 or i == total_steps - 1):
            log_execution_state(robot, obj, i, total_steps)
        time.sleep(1e-2)


def run_handover_demo(
    args: argparse.Namespace, sim: SimulationManager, robot: Robot
) -> None:
    """Plan and optionally execute pick-up plus handover."""
    create_support_plane(sim)
    bottle = create_bottle(sim)
    settle_object(sim, bottle, step=0)
    semantics = create_object_semantics(bottle, args)
    motion_gen = MotionGenerator(
        cfg=MotionGenCfg(planner_cfg=ToppraPlannerCfg(robot_uid=robot.uid))
    )

    right_open, right_close = get_hand_open_close_qpos(
        robot, "right_hand", sim.device
    )
    left_open, left_close = get_hand_open_close_qpos(robot, "left_hand", sim.device)
    pick_action = PickUpAction(
        motion_generator=motion_gen,
        cfg=PickUpActionCfg(
            control_part="right_arm",
            hand_control_part="right_hand",
            hand_open_qpos=right_open,
            hand_close_qpos=right_close,
            approach_direction=torch.tensor(
                [0.0, 0.0, -1.0], dtype=torch.float32, device=sim.device
            ),
            pre_grasp_distance=0.15,
            lift_height=0.12,
            sample_interval=PICK_SAMPLE_INTERVAL,
            hand_interp_steps=10,
        ),
    )
    handover_action = HandoverAction(
        motion_generator=motion_gen,
        cfg=HandoverActionCfg(
            control_part="dual_arm",
            giver_arm_control_part="right_arm",
            receiver_arm_control_part="left_arm",
            giver_hand_control_part="right_hand",
            receiver_hand_control_part="left_hand",
            giver_hand_open_qpos=right_open,
            giver_hand_close_qpos=right_close,
            receiver_hand_open_qpos=left_open,
            receiver_hand_close_qpos=left_close,
            sample_interval=HANDOVER_SAMPLE_INTERVAL,
            hand_interp_steps=10,
            handover_hold_steps=6,
            retreat_steps=16,
            pre_handover_distance=0.10,
            giver_retreat_distance=0.08,
            receiver_retreat_distance=0.00,
        ),
    )

    if not args.diagnose_plan:
        sim.open_window()
        if not args.auto_play:
            input("Inspect the scene, then press Enter to plan pick-up...")

    start_time = time.time()
    pick_success, pick_traj, pick_joint_ids = pick_action.execute(semantics)
    logger.log_info(f"Plan pick-up cost time: {time.time() - start_time:.2f} seconds")
    if not pick_success:
        logger.log_warning("Failed to plan pick-up trajectory.")
        return
    log_action_plan(robot, "pick_up", pick_traj, pick_joint_ids)

    if args.diagnose_plan:
        robot.set_qpos(pick_traj[:, -1, :], joint_ids=pick_joint_ids)
    else:
        if not args.auto_play:
            input("Press Enter to execute pick-up...")
        execute_trajectory(
            sim, robot, pick_traj, pick_joint_ids, bottle, args.debug_hand_state
        )
    bottle.clear_dynamics()

    handover_target, handover_bottle_pose = build_object_aware_handover_target(
        robot, bottle, sim.device
    )
    log_handover_geometry(handover_target, handover_bottle_pose)
    start_time = time.time()
    handover_success, handover_traj, handover_joint_ids = handover_action.execute(
        handover_target
    )
    logger.log_info(
        f"Plan handover cost time: {time.time() - start_time:.2f} seconds"
    )
    if not handover_success:
        logger.log_warning("Failed to plan handover trajectory.")
        return
    log_action_plan(
        robot,
        "handover",
        handover_traj,
        handover_joint_ids,
        handover_action.get_segment_lengths(),
    )
    log_tcp_alignment(
        robot, handover_traj, handover_joint_ids, handover_target, handover_action
    )

    if args.diagnose_plan:
        return

    if not args.auto_play:
        input("Press Enter to execute handover...")
    execute_trajectory(
        sim,
        robot,
        handover_traj,
        handover_joint_ids,
        bottle,
        args.debug_hand_state,
    )
    if not args.auto_play:
        input("Press Enter to exit the simulation...")


def main() -> None:
    """Run the dual-arm handover demo."""
    args = parse_arguments()
    sim = initialize_simulation(args)
    robot = create_dual_ur10_robot(sim)
    run_handover_demo(args, sim, robot)


if __name__ == "__main__":
    main()
