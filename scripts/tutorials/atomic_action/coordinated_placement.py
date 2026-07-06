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

"""Demonstrate dual-arm coordinated placement with bread and pan meshes.

The left UR5 picks up bread. The right UR5 picks up a pan and moves it to the
lower alignment pose. The left UR5 places the bread above the pan and releases
it while the right hand keeps holding the pan.
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
from scipy.spatial.transform import Rotation as SciRotation

from embodichain.lab.gym.utils.gym_utils import add_env_launcher_args_to_parser
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.atomic_actions import (
    Affordance,
    CoordinatedPlacement,
    CoordinatedPlacementCfg,
    CoordinatedPlacementTarget,
    HeldObjectState,
    ObjectSemantics,
    TrajectoryBuilder,
    WorldState,
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
from embodichain.lab.sim.planners import (
    MotionGenCfg,
    MotionGenerator,
    MoveType,
    PlanState,
    ToppraPlannerCfg,
)
from embodichain.lab.sim.shapes import MeshCfg
from embodichain.lab.sim.solvers import URSolverCfg
from embodichain.utils import logger
from scripts.tutorials.atomic_action.tutorial_utils import (
    broadcast_pose_batch,
    clone_local_pose_from_first_env,
    draw_axis_marker,
    get_tutorial_window_size,
    make_ur5_solver_cfg,
    should_open_tutorial_window,
    should_wait_for_tutorial_input,
    start_auto_play_recording,
    stop_auto_play_recording,
)

DEFAULT_MESH_FRAME_CORRECTION_EULER_DEG = (-90.0, 0.0, 0.0)
# DexSim imports this pan GLB with gym raw +Z mapped to local -Y.  The +90deg X
# correction therefore makes the pan opening point upward in world Z.
PAN_MESH_FRAME_CORRECTION_EULER_DEG = (90.0, 0.0, 0.0)
PAN_WORLD_YAW_CORRECTION_DEG = 270.0


def transform_baseline_pose(
    init_pos: tuple[float, float, float],
    init_rot: tuple[float, float, float],
    *,
    z_offset: float = 0.0,
    mesh_frame_correction_euler_deg: tuple[float, float, float] = (
        DEFAULT_MESH_FRAME_CORRECTION_EULER_DEG
    ),
    world_yaw_correction_deg: float = 0.0,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Apply mesh-frame correction while preserving baseline world placement."""
    pos = np.asarray(init_pos, dtype=np.float64)
    pos[2] += z_offset
    rot = (
        SciRotation.from_euler("Z", world_yaw_correction_deg, degrees=True)
        * SciRotation.from_euler("XYZ", init_rot, degrees=True)
        * SciRotation.from_euler("XYZ", mesh_frame_correction_euler_deg, degrees=True)
    ).as_euler("XYZ", degrees=True)
    return tuple(float(value) for value in pos), tuple(float(value) for value in rot)


ARM_URDF_PATH = "UniversalRobots/UR5/UR5.urdf"
GRIPPER_URDF_PATH = "DH_PGI_140_80/DH_PGI_140_80.urdf"
PLACEMENT_ASSET_ROOT = "CoordinatedPlacementAndPickment"
TABLE_MESH_PATH = f"{PLACEMENT_ASSET_ROOT}/table.glb"
BREAD_MESH_PATH = f"{PLACEMENT_ASSET_ROOT}/bread.glb"
PAN_MESH_PATH = f"{PLACEMENT_ASSET_ROOT}/pan.glb"
BREAD_LABEL = "bread"
PAN_LABEL = "pan"
GRIPPER_TCP_Z = 0.121
PICK_SAMPLE_INTERVAL = 100
COORDINATED_SAMPLE_INTERVAL = 120
ROBOT_INIT_POS = (1.85, 0.0, 0.1)
ROBOT_INIT_ROT = (0.0, 0.0, -90.0)
LEFT_ARM_HOME = (0.0, 0.0, -1.57, -1.57, 1.57, 1.57)
RIGHT_ARM_HOME = (-1.57, -1.57, -1.57, -1.57, 0.0, 0.0)
TABLE_TOP_Z = 0.65
BASELINE_TABLE_TOP_Z = 0.3621708124799265
SCENE_Z_OFFSET = TABLE_TOP_Z - BASELINE_TABLE_TOP_Z
BASELINE_TABLE_INIT_POS = (
    0.00014585733079742588,
    0.00023304896730074557,
    -0.019599792839044783,
)
BASELINE_TABLE_INIT_ROT = (
    0.0001074673904926984,
    0.00865572768366991,
    -90.6562109309317,
)
BASELINE_BREAD_INIT_POS = (
    0.007266042530919159,
    0.17218712515099063,
    0.38805152145807564,
)
BASELINE_BREAD_INIT_ROT = (
    179.93952112929065,
    -0.12776179446053365,
    85.59207565132371,
)
BASELINE_PAN_INIT_POS = (
    0.0009683294205463406,
    -0.14189524793277888,
    0.38900474548025743,
)
BASELINE_PAN_INIT_ROT = (
    -179.23950670370294,
    -0.4795764805552328,
    98.19364391929443,
)
TABLE_INIT_POS, TABLE_INIT_ROT = transform_baseline_pose(
    BASELINE_TABLE_INIT_POS,
    BASELINE_TABLE_INIT_ROT,
    z_offset=SCENE_Z_OFFSET,
)
BREAD_INIT_POS, BREAD_INIT_ROT = transform_baseline_pose(
    BASELINE_BREAD_INIT_POS,
    BASELINE_BREAD_INIT_ROT,
    z_offset=SCENE_Z_OFFSET,
)
PAN_INIT_POS, PAN_INIT_ROT = transform_baseline_pose(
    BASELINE_PAN_INIT_POS,
    BASELINE_PAN_INIT_ROT,
    z_offset=SCENE_Z_OFFSET,
    mesh_frame_correction_euler_deg=PAN_MESH_FRAME_CORRECTION_EULER_DEG,
    world_yaw_correction_deg=PAN_WORLD_YAW_CORRECTION_DEG,
)
PAN_INIT_POS = (PAN_INIT_POS[0], PAN_INIT_POS[1], TABLE_TOP_Z + 0.001)
PAN_TARGET_CENTER_XY = (-0.06, 0.0)
PAN_TARGET_Z_LIFT = 0.06
BREAD_PLACE_TARGET_OFFSET_XY = (-0.06, -0.16)
BREAD_ON_PAN_CLEARANCE = 0.006
BREAD_GRASP_Z_CLEARANCE = 0.018
PAN_GRASP_Z_CLEARANCE = 0.0
PAN_HANDLE_LOCAL_Z_MIN = 0.04
PAN_BASIN_LOCAL_Z_MAX = 0.04
PAN_HANDLE_ROOT_OFFSET = 0.035
PAN_HANDLE_CLOSE_QPOS = 0.045
PAN_PICK_SAMPLE_INTERVAL = 130
PAN_PICK_HAND_INTERP_STEPS = 32
BREAD_TARGET_WORLD_YAW_DEG = 0.0
BREAD_TARGET_HEIGHT_OFFSET = 0.1
SUPPORT_TARGET_HEIGHT_OFFSET = 0.0
PICK_APPROACH_DISTANCE = 0.12
PLACE_LIFT_HEIGHT = 0.10
TRAJECTORY_SIM_STEPS = 8


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the demo."""
    parser = argparse.ArgumentParser(description="Dual-arm coordinated placement demo")
    add_env_launcher_args_to_parser(parser)
    parser.set_defaults(device="cuda", renderer="hybrid")
    parser.add_argument(
        "--diagnose_plan",
        action="store_true",
        help="Plan and print diagnostics without playing the trajectory.",
    )
    parser.add_argument(
        "--debug_state",
        action="store_true",
        help="Log hand targets and object poses during execution.",
    )
    parser.add_argument(
        "--auto_play",
        action="store_true",
        help="Run the viewer demo without waiting for keyboard input.",
    )
    parser.add_argument(
        "--no_vis_eef_axis",
        action="store_true",
        help="Do not draw coordinated placement target coordinate frames.",
    )
    parser.add_argument(
        "--headless_play",
        action="store_true",
        help="Execute planned trajectories without opening the viewer window.",
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
    width, height = get_tutorial_window_size(args)
    sim = SimulationManager(
        SimulationManagerCfg(
            width=width,
            height=height,
            headless=True,
            num_envs=args.num_envs,
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


def make_prefixed_ur5_solver_cfg(prefix: str) -> URSolverCfg:
    """Create a UR5 solver cfg for a prefixed arm in the assembled robot."""
    cfg = make_ur5_solver_cfg(GRIPPER_TCP_Z)
    cfg.root_link_name = f"{prefix}_base_link"
    cfg.end_link_name = f"{prefix}_ee_link"
    return cfg


def create_dual_ur5_robot(sim: SimulationManager) -> Robot:
    """Create a dual-UR5 robot with one PGI gripper on each arm."""
    arm_urdf_path = get_cached_data_path(ARM_URDF_PATH)
    gripper_urdf_path = get_cached_data_path(GRIPPER_URDF_PATH)
    cfg = RobotCfg(
        uid="DualUR5CoordinatedPlacement",
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
            fname="dual_ur5_coordinated_placement",
        ),
        drive_pros=JointDrivePropertiesCfg(
            stiffness={
                "left_joint[0-9]": 1e4,
                "right_joint[0-9]": 1e4,
                "left_gripper_finger[1-2]_joint_1": 1e3,
                "right_gripper_finger[1-2]_joint_1": 1e3,
            },
            damping={
                "left_joint[0-9]": 1e3,
                "right_joint[0-9]": 1e3,
                "left_gripper_finger[1-2]_joint_1": 1e2,
                "right_gripper_finger[1-2]_joint_1": 1e2,
            },
            max_effort={
                "left_joint[0-9]": 1e5,
                "right_joint[0-9]": 1e5,
                "left_gripper_finger[1-2]_joint_1": 1e4,
                "right_gripper_finger[1-2]_joint_1": 1e4,
            },
            drive_type="force",
        ),
        control_parts={
            "left_arm": ["left_joint[0-9]"],
            "right_arm": ["right_joint[0-9]"],
            "dual_arm": ["left_joint[0-9]", "right_joint[0-9]"],
            "left_hand": ["left_gripper_finger1_joint_1"],
            "right_hand": ["right_gripper_finger1_joint_1"],
        },
        solver_cfg={
            "left_arm": make_prefixed_ur5_solver_cfg("left"),
            "right_arm": make_prefixed_ur5_solver_cfg("right"),
        },
        init_pos=list(ROBOT_INIT_POS),
        init_rot=list(ROBOT_INIT_ROT),
        init_qpos=list(LEFT_ARM_HOME) + list(RIGHT_ARM_HOME) + [0.0, 0.0, 0.0, 0.0],
    )
    return sim.add_robot(cfg=cfg)


def create_table(sim: SimulationManager) -> RigidObject:
    """Create the table mesh from the bread-pan gym project."""
    return sim.add_rigid_object(
        cfg=RigidObjectCfg(
            uid="table",
            shape=MeshCfg(fpath=get_cached_data_path(TABLE_MESH_PATH)),
            attrs=RigidBodyAttributesCfg(
                mass=10.0,
                dynamic_friction=0.9,
                static_friction=0.95,
                restitution=0.01,
            ),
            body_type="kinematic",
            init_pos=list(TABLE_INIT_POS),
            init_rot=list(TABLE_INIT_ROT),
        )
    )


def create_bread(sim: SimulationManager) -> RigidObject:
    """Create the bread mesh to be placed by the left arm."""
    return sim.add_rigid_object(
        cfg=RigidObjectCfg(
            uid="bread",
            shape=MeshCfg(
                fpath=get_cached_data_path(BREAD_MESH_PATH), compute_uv=False
            ),
            attrs=RigidBodyAttributesCfg(
                mass=0.01,
                contact_offset=0.003,
                rest_offset=0.001,
                restitution=0.01,
                min_position_iters=32,
                min_velocity_iters=8,
                max_depenetration_velocity=10.0,
            ),
            body_scale=(1.75, 1.75, 1.75),
            max_convex_hull_num=8,
            init_pos=list(BREAD_INIT_POS),
            init_rot=list(BREAD_INIT_ROT),
        )
    )


def create_pan(sim: SimulationManager) -> RigidObject:
    """Create the pan mesh held below the bread by the right arm."""
    return sim.add_rigid_object(
        cfg=RigidObjectCfg(
            uid="pan",
            shape=MeshCfg(fpath=get_cached_data_path(PAN_MESH_PATH), compute_uv=False),
            attrs=RigidBodyAttributesCfg(
                mass=0.01,
                dynamic_friction=0.97,
                static_friction=0.99,
                angular_damping=2.0,
                linear_damping=1.0,
                contact_offset=0.001,
                rest_offset=0.0,
                restitution=0.01,
                min_position_iters=32,
                min_velocity_iters=8,
                max_depenetration_velocity=2.0,
            ),
            body_scale=(1.75, 1.75, 1.75),
            max_convex_hull_num=16,
            init_pos=list(PAN_INIT_POS),
            init_rot=list(PAN_INIT_ROT),
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


def create_object_semantics(obj: RigidObject, label: str) -> ObjectSemantics:
    """Create minimal object semantics for manually specified grasps."""
    return ObjectSemantics(
        label=label,
        geometry={},
        affordance=Affordance(object_label=label),
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


def get_pan_handle_open_close_qpos(
    robot: Robot, hand_control_part: str, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get right hand qpos tuned for holding the thin pan handle."""
    limits = robot.get_qpos_limits(name=hand_control_part)[0].to(
        device=device, dtype=torch.float32
    )
    hand_open = limits[:, 0]
    hand_close = torch.clamp(
        torch.full_like(limits[:, 1], PAN_HANDLE_CLOSE_QPOS),
        min=limits[:, 0],
        max=limits[:, 1],
    )
    return hand_open, hand_close


def invert_pose(pose: torch.Tensor) -> torch.Tensor:
    """Invert batched homogeneous transforms."""
    inv_pose = pose.clone()
    rot_t = pose[:, :3, :3].transpose(1, 2)
    inv_pose[:, :3, :3] = rot_t
    inv_pose[:, :3, 3] = -torch.bmm(rot_t, pose[:, :3, 3:4]).squeeze(-1)
    return inv_pose


def get_local_vertices(obj: RigidObject) -> torch.Tensor:
    """Get scaled local mesh vertices."""
    return obj.get_vertices(env_ids=[0], scale=True)[0]


def compute_local_bounds(vertices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute local mesh AABB from scaled vertices."""
    return vertices.min(dim=0).values, vertices.max(dim=0).values


def transform_points(pose: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Transform local points by a homogeneous pose."""
    return points @ pose[:3, :3].transpose(0, 1) + pose[:3, 3]


def compute_world_bounds(
    object_pose: torch.Tensor,
    local_vertices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute world AABB from transformed local mesh vertices."""
    world_vertices = transform_points(object_pose, local_vertices)
    return world_vertices.min(dim=0).values, world_vertices.max(dim=0).values


def rotate_pose_about_world_z(pose: torch.Tensor, yaw_deg: float) -> torch.Tensor:
    """Rotate pose orientation about world Z while preserving translation."""
    yaw = math.radians(yaw_deg)
    rot = torch.eye(3, dtype=pose.dtype, device=pose.device)
    rot[0, 0] = math.cos(yaw)
    rot[0, 1] = -math.sin(yaw)
    rot[1, 0] = math.sin(yaw)
    rot[1, 1] = math.cos(yaw)
    rotated_pose = pose.clone()
    rotated_pose[:3, :3] = rot @ pose[:3, :3]
    return rotated_pose


def get_pan_basin_vertices(pan_vertices: torch.Tensor) -> torch.Tensor:
    """Select pan vertices that belong to the basin instead of the long handle."""
    basin_vertices = pan_vertices[pan_vertices[:, 2] <= PAN_BASIN_LOCAL_Z_MAX]
    if basin_vertices.numel() == 0:
        logger.log_warning("Pan basin vertex mask is empty; falling back to full mesh.")
        return pan_vertices
    return basin_vertices


def build_top_down_tcp_pose(
    position: torch.Tensor, device: torch.device
) -> torch.Tensor:
    """Build a simple top-down TCP pose for manually grasping flat objects."""
    pose = torch.eye(4, dtype=torch.float32, device=device)
    pose[:3, :3] = torch.tensor(
        [
            [-0.0539, -0.9985, -0.0022],
            [-0.9977, 0.0540, -0.0401],
            [0.0401, 0.0000, -0.9992],
        ],
        dtype=torch.float32,
        device=device,
    )
    pose[:3, 3] = position
    return pose


def build_flat_object_grasp_pose(
    object_pose: torch.Tensor,
    local_vertices: torch.Tensor,
    local_min: torch.Tensor,
    local_max: torch.Tensor,
    device: torch.device,
    *,
    world_xy_offset: tuple[float, float] = (0.0, 0.0),
    z_clearance: float = 0.02,
) -> torch.Tensor:
    """Build a hand-tuned top-down grasp TCP pose over a flat object."""
    local_center = 0.5 * (local_min + local_max)
    local_center = local_center.to(device=device, dtype=torch.float32)
    grasp_position = object_pose[:3, 3] + object_pose[:3, :3] @ local_center
    _, world_max = compute_world_bounds(object_pose, local_vertices)
    grasp_position[0] += world_xy_offset[0]
    grasp_position[1] += world_xy_offset[1]
    grasp_position[2] = world_max[2] + z_clearance
    return build_top_down_tcp_pose(grasp_position, device)


def normalize_vector(vector: torch.Tensor, fallback: torch.Tensor) -> torch.Tensor:
    """Normalize a vector with a deterministic fallback for degenerate cases."""
    norm = torch.linalg.norm(vector)
    if norm < 1e-6:
        return fallback.to(device=vector.device, dtype=vector.dtype)
    return vector / norm


def build_pan_handle_grasp_pose(
    pan_pose: torch.Tensor,
    pan_vertices: torch.Tensor,
    device: torch.device,
    *,
    z_clearance: float = 0.006,
) -> torch.Tensor:
    """Build a top-down TCP pose that pinches the pan handle."""
    handle_vertices = pan_vertices[pan_vertices[:, 2] > PAN_HANDLE_LOCAL_Z_MIN]
    if handle_vertices.numel() == 0:
        logger.log_warning(
            "Pan handle vertex mask is empty; falling back to full mesh."
        )
        handle_vertices = pan_vertices

    handle_world = transform_points(pan_pose, handle_vertices)
    handle_min = handle_world.min(dim=0).values
    handle_max = handle_world.max(dim=0).values
    grasp_position = 0.5 * (handle_min + handle_max)

    pan_world = transform_points(pan_pose, pan_vertices)
    pan_center_xy = pan_world[:, :2].mean(dim=0)
    handle_dir_xy = grasp_position[:2] - pan_center_xy
    handle_axis = normalize_vector(
        torch.tensor(
            [handle_dir_xy[0], handle_dir_xy[1], 0.0],
            dtype=torch.float32,
            device=device,
        ),
        torch.tensor([1.0, -0.2, 0.0], dtype=torch.float32, device=device),
    )
    grasp_position[:2] -= handle_axis[:2] * PAN_HANDLE_ROOT_OFFSET
    grasp_position[2] = handle_max[2] + z_clearance

    y_axis = handle_axis
    z_axis = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device=device)
    x_axis = normalize_vector(
        torch.cross(y_axis, z_axis, dim=0),
        torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=device),
    )
    y_axis = normalize_vector(
        torch.cross(z_axis, x_axis, dim=0),
        y_axis,
    )

    pose = torch.eye(4, dtype=torch.float32, device=device)
    pose[:3, 0] = x_axis
    pose[:3, 1] = y_axis
    pose[:3, 2] = z_axis
    pose[:3, 3] = grasp_position
    return pose


def build_support_object_target_pose(
    pan_pose: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Build target pose for the support pan."""
    pose = pan_pose.clone().to(device=device, dtype=torch.float32)
    pose[0, 3] = PAN_TARGET_CENTER_XY[0]
    pose[1, 3] = PAN_TARGET_CENTER_XY[1]
    pose[2, 3] = TABLE_TOP_Z + 0.001 + PAN_TARGET_Z_LIFT
    return pose


def build_placing_object_target_pose(
    bread_pose: torch.Tensor,
    bread_vertices: torch.Tensor,
    pan_vertices: torch.Tensor,
    support_target_pose: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Build target bread pose aligned above the pan."""
    pose = rotate_pose_about_world_z(
        bread_pose.clone().to(device=device, dtype=torch.float32),
        BREAD_TARGET_WORLD_YAW_DEG,
    )
    pan_basin_world = transform_points(
        support_target_pose, get_pan_basin_vertices(pan_vertices)
    )
    basin_center_xy = 0.5 * (
        pan_basin_world[:, :2].min(dim=0).values
        + pan_basin_world[:, :2].max(dim=0).values
    )
    pose[0, 3] = basin_center_xy[0] + BREAD_PLACE_TARGET_OFFSET_XY[0]
    pose[1, 3] = basin_center_xy[1] + BREAD_PLACE_TARGET_OFFSET_XY[1]
    pan_top_z = pan_basin_world[:, 2].max()
    bread_bottom_z = compute_world_bounds(pose, bread_vertices)[0][2]
    pose[2, 3] += pan_top_z + BREAD_ON_PAN_CLEARANCE - bread_bottom_z
    return pose


def format_tensor(tensor: torch.Tensor) -> str:
    """Format tensor values for compact logging."""
    rounded = (tensor.detach().cpu() * 10000.0).round() / 10000.0
    return str(rounded.tolist())


def compute_actual_held_state(
    robot: Robot,
    semantics: ObjectSemantics,
    object_pose: torch.Tensor,
    arm_control_part: str,
    device: torch.device,
) -> HeldObjectState:
    """Build held-object state from current object pose and current TCP FK."""
    arm_qpos = robot.get_qpos(name=arm_control_part).to(device=device)
    tcp_pose = robot.compute_fk(
        arm_qpos,
        name=arm_control_part,
        to_matrix=True,
    ).to(device=device, dtype=torch.float32)
    object_pose = broadcast_pose_batch(
        object_pose.to(device=device, dtype=torch.float32),
        num_envs=tcp_pose.shape[0],
    )
    object_to_eef = torch.bmm(invert_pose(object_pose), tcp_pose)
    return HeldObjectState(
        semantics=semantics,
        object_to_eef=object_to_eef,
        grasp_xpos=tcp_pose,
    )


def plan_manual_pick(
    motion_gen: MotionGenerator,
    robot: Robot,
    *,
    arm_control_part: str,
    hand_control_part: str,
    grasp_pose: torch.Tensor,
    hand_open_qpos: torch.Tensor,
    hand_close_qpos: torch.Tensor,
    sample_interval: int,
    hand_interp_steps: int,
    pre_grasp_distance: float,
    lift_height: float,
    start_state: WorldState,
) -> tuple[bool, torch.Tensor, WorldState]:
    """Plan a demo-local pick trajectory from a hand-authored TCP grasp pose."""
    builder = TrajectoryBuilder(motion_gen)
    device = start_state.last_qpos.device
    n_envs = robot.get_qpos().shape[0]
    robot_dof = robot.dof
    arm_joint_ids = robot.get_joint_ids(name=arm_control_part)
    hand_joint_ids = robot.get_joint_ids(name=hand_control_part)
    arm_dof = len(arm_joint_ids)

    grasp_xpos = builder.resolve_pose_target(grasp_pose, n_envs=n_envs)
    grasp_z = grasp_xpos[:, :3, 2]
    pre_grasp_xpos = builder.apply_local_offset(
        grasp_xpos, -grasp_z * pre_grasp_distance
    )
    lift_xpos = builder.apply_local_offset(
        grasp_xpos,
        torch.tensor([0.0, 0.0, lift_height], device=device),
    )
    start_arm_qpos = builder.resolve_start_qpos(
        start_state.last_qpos[:, arm_joint_ids],
        n_envs=n_envs,
        arm_dof=arm_dof,
        control_part=arm_control_part,
    )
    n_approach, n_close, n_lift = builder.split_three_phase(
        sample_interval,
        hand_interp_steps,
        first_phase_name="approach",
        third_phase_name="lift",
    )

    target_states_list = [
        [
            PlanState(xpos=pre_grasp_xpos[i], move_type=MoveType.EEF_MOVE),
            PlanState(xpos=grasp_xpos[i], move_type=MoveType.EEF_MOVE),
        ]
        for i in range(n_envs)
    ]
    ok, approach_arm = builder.plan_arm_traj(
        target_states_list,
        start_arm_qpos,
        n_approach,
        control_part=arm_control_part,
        arm_dof=arm_dof,
    )
    if not bool(torch.all(ok)):
        logger.log_warning(f"Failed to plan {arm_control_part} manual pick approach.")
        return (
            False,
            torch.empty((n_envs, 0, robot_dof), dtype=torch.float32, device=device),
            start_state,
        )

    grasp_arm_qpos = approach_arm[:, -1, :]
    target_states_list = [
        [PlanState(xpos=lift_xpos[i], move_type=MoveType.EEF_MOVE)]
        for i in range(n_envs)
    ]
    ok, lift_arm = builder.plan_arm_traj(
        target_states_list,
        grasp_arm_qpos,
        n_lift,
        control_part=arm_control_part,
        arm_dof=arm_dof,
    )
    if not bool(torch.all(ok)):
        logger.log_warning(f"Failed to plan {arm_control_part} manual pick lift.")
        return (
            False,
            torch.empty((n_envs, 0, robot_dof), dtype=torch.float32, device=device),
            start_state,
        )

    hand_open = builder.expand_hand_qpos(
        hand_open_qpos, n_envs=n_envs, hand_dof=len(hand_joint_ids)
    )
    hand_close = builder.expand_hand_qpos(
        hand_close_qpos, n_envs=n_envs, hand_dof=len(hand_joint_ids)
    )
    hand_close_path = builder.interpolate_hand_qpos(
        hand_open, hand_close, n_waypoints=n_close
    )

    full = start_state.last_qpos.unsqueeze(1).repeat(1, sample_interval, 1).clone()
    full[:, :n_approach, arm_joint_ids] = approach_arm
    full[:, :n_approach, hand_joint_ids] = hand_open.unsqueeze(1)
    full[:, n_approach : n_approach + n_close, arm_joint_ids] = (
        grasp_arm_qpos.unsqueeze(1)
    )
    full[:, n_approach : n_approach + n_close, hand_joint_ids] = hand_close_path
    full[:, n_approach + n_close :, arm_joint_ids] = lift_arm
    full[:, n_approach + n_close :, hand_joint_ids] = hand_close.unsqueeze(1)
    return True, full, WorldState(last_qpos=full[:, -1, :].clone(), held_object=None)


def log_scene_targets(
    bread_pose: torch.Tensor,
    pan_pose: torch.Tensor,
    support_target_pose: torch.Tensor | None = None,
    placing_target_pose: torch.Tensor | None = None,
) -> None:
    """Log compact object and target positions for diagnosis."""
    logger.log_info(
        "scene objects: "
        f"bread_origin={format_tensor(bread_pose[:3, 3])}, "
        f"pan_origin={format_tensor(pan_pose[:3, 3])}"
    )
    if support_target_pose is not None and placing_target_pose is not None:
        logger.log_info(
            "coordinated targets: "
            f"support_pan_origin={format_tensor(support_target_pose[:3, 3])}, "
            f"placing_bread_origin={format_tensor(placing_target_pose[:3, 3])}"
        )


def draw_coordinated_axes(
    sim: SimulationManager,
    support_target_pose: torch.Tensor,
    placing_target_pose: torch.Tensor,
    num_envs: int,
) -> None:
    """Draw coordinate-frame markers for coordinated placement targets."""
    draw_axis_marker(
        sim,
        "support_pan_target_axis",
        broadcast_pose_batch(support_target_pose, num_envs=num_envs),
        axis_len=0.08,
        axis_size=0.004,
    )
    draw_axis_marker(
        sim,
        "placing_bread_target_axis",
        broadcast_pose_batch(placing_target_pose, num_envs=num_envs),
        axis_len=0.08,
        axis_size=0.004,
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
    bread: RigidObject,
    pan: RigidObject,
    step_idx: int,
    total_steps: int,
) -> None:
    """Log hand and object state during execution."""
    bread_pose = bread.get_local_pose(to_matrix=True)
    pan_pose = pan.get_local_pose(to_matrix=True)
    left_hand = robot.get_qpos(name="left_hand")
    right_hand = robot.get_qpos(name="right_hand")
    logger.log_info(
        f"step={step_idx}/{total_steps - 1}, "
        f"left_hand={format_tensor(left_hand[0])}, "
        f"right_hand={format_tensor(right_hand[0])}, "
        f"bread_pos={format_tensor(bread_pose[0, :3, 3])}, "
        f"pan_pos={format_tensor(pan_pose[0, :3, 3])}"
    )


def execute_trajectory(
    sim: SimulationManager,
    robot: Robot,
    traj: torch.Tensor,
    joint_ids: list[int],
    bread: RigidObject,
    pan: RigidObject,
    debug_state: bool,
) -> None:
    """Play a planned trajectory in simulation."""
    total_steps = traj.shape[1]
    log_stride = max(1, total_steps // 10)
    for i in range(total_steps):
        robot.set_qpos(traj[:, i, :], joint_ids=joint_ids)
        sim.update(step=TRAJECTORY_SIM_STEPS)
        if debug_state and (i % log_stride == 0 or i == total_steps - 1):
            log_execution_state(robot, bread, pan, i, total_steps)
        time.sleep(1e-2)


def run_coordinated_placement_demo(
    args: argparse.Namespace, sim: SimulationManager, robot: Robot
) -> None:
    """Plan and optionally execute pick-up and coordinated placement."""
    create_table(sim)
    bread = create_bread(sim)
    pan = create_pan(sim)
    settle_object(sim, bread, step=0)
    settle_object(sim, pan, step=0)
    bread_pose_batch = clone_local_pose_from_first_env(bread)
    pan_pose_batch = clone_local_pose_from_first_env(pan)
    bread.clear_dynamics()
    pan.clear_dynamics()
    bread_pose = bread_pose_batch[0].to(device=sim.device, dtype=torch.float32)
    pan_pose = pan_pose_batch[0].to(device=sim.device, dtype=torch.float32)
    n_envs = bread_pose_batch.shape[0]
    bread_vertices = get_local_vertices(bread)
    pan_vertices = get_local_vertices(pan)
    bread_local_min, bread_local_max = compute_local_bounds(bread_vertices)
    log_scene_targets(bread_pose, pan_pose)
    bread_semantics = create_object_semantics(bread, BREAD_LABEL)
    pan_semantics = create_object_semantics(pan, PAN_LABEL)
    motion_gen = MotionGenerator(
        cfg=MotionGenCfg(planner_cfg=ToppraPlannerCfg(robot_uid=robot.uid))
    )

    right_open, right_close = get_pan_handle_open_close_qpos(
        robot, "right_hand", sim.device
    )
    left_open, left_close = get_hand_open_close_qpos(robot, "left_hand", sim.device)
    coordinated_action = CoordinatedPlacement(
        motion_generator=motion_gen,
        cfg=CoordinatedPlacementCfg(
            control_part="dual_arm",
            placing_arm_control_part="left_arm",
            support_arm_control_part="right_arm",
            placing_hand_control_part="left_hand",
            support_hand_control_part="right_hand",
            placing_hand_open_qpos=left_open,
            placing_hand_close_qpos=left_close,
            support_hand_close_qpos=right_close,
            release=True,
            placing_height_offset=BREAD_TARGET_HEIGHT_OFFSET,
            support_height_offset=SUPPORT_TARGET_HEIGHT_OFFSET,
            lift_height=PLACE_LIFT_HEIGHT,
            sample_interval=COORDINATED_SAMPLE_INTERVAL,
            hand_interp_steps=10,
            hold_steps=6,
            retreat_steps=18,
        ),
    )
    full_joint_ids = list(range(robot.dof))
    state = WorldState(last_qpos=robot.get_qpos().clone())

    wait_for_user = should_wait_for_tutorial_input(args)
    if should_open_tutorial_window(args):
        sim.open_window()
    if wait_for_user and not args.diagnose_plan:
        input("Inspect the scene, then press Enter to plan left pick-up...")

    bread_grasp_pose = build_flat_object_grasp_pose(
        bread_pose,
        bread_vertices,
        bread_local_min,
        bread_local_max,
        sim.device,
        z_clearance=BREAD_GRASP_Z_CLEARANCE,
    )
    start_time = time.time()
    left_pick_success, left_pick_traj, state = plan_manual_pick(
        motion_gen,
        robot,
        arm_control_part="left_arm",
        hand_control_part="left_hand",
        grasp_pose=bread_grasp_pose,
        hand_open_qpos=left_open,
        hand_close_qpos=left_close,
        sample_interval=PICK_SAMPLE_INTERVAL,
        hand_interp_steps=10,
        pre_grasp_distance=PICK_APPROACH_DISTANCE,
        lift_height=0.12,
        start_state=state,
    )
    logger.log_info(
        f"Plan left bread pick-up cost time: {time.time() - start_time:.2f} seconds"
    )
    if not left_pick_success:
        logger.log_warning("Failed to plan left bread pick-up trajectory.")
        return
    log_action_plan(robot, "left_pick_up", left_pick_traj, full_joint_ids)
    bread_object_to_eef = torch.bmm(
        broadcast_pose_batch(invert_pose(bread_pose.unsqueeze(0)), num_envs=n_envs),
        broadcast_pose_batch(bread_grasp_pose, num_envs=n_envs),
    )
    bread_held_state = HeldObjectState(
        semantics=bread_semantics,
        object_to_eef=bread_object_to_eef,
        grasp_xpos=broadcast_pose_batch(bread_grasp_pose, num_envs=n_envs),
    )

    pan_grasp_pose = build_pan_handle_grasp_pose(
        pan_pose,
        pan_vertices,
        sim.device,
        z_clearance=PAN_GRASP_Z_CLEARANCE,
    )
    start_time = time.time()
    right_pick_success, right_pick_traj, state = plan_manual_pick(
        motion_gen,
        robot,
        arm_control_part="right_arm",
        hand_control_part="right_hand",
        grasp_pose=pan_grasp_pose,
        hand_open_qpos=right_open,
        hand_close_qpos=right_close,
        sample_interval=PAN_PICK_SAMPLE_INTERVAL,
        hand_interp_steps=PAN_PICK_HAND_INTERP_STEPS,
        pre_grasp_distance=PICK_APPROACH_DISTANCE,
        lift_height=0.10,
        start_state=state,
    )
    logger.log_info(
        f"Plan right pan pick-up cost time: {time.time() - start_time:.2f} seconds"
    )
    if not right_pick_success:
        logger.log_warning("Failed to plan right pan pick-up trajectory.")
        return
    log_action_plan(robot, "right_pick_up", right_pick_traj, full_joint_ids)
    pan_object_to_eef = torch.bmm(
        broadcast_pose_batch(invert_pose(pan_pose.unsqueeze(0)), num_envs=n_envs),
        broadcast_pose_batch(pan_grasp_pose, num_envs=n_envs),
    )
    pan_held_state = HeldObjectState(
        semantics=pan_semantics,
        object_to_eef=pan_object_to_eef,
        grasp_xpos=broadcast_pose_batch(pan_grasp_pose, num_envs=n_envs),
    )

    if args.diagnose_plan:
        robot.set_qpos(state.last_qpos, joint_ids=full_joint_ids)
    else:
        if wait_for_user:
            input("Press Enter to execute both pick-up trajectories...")
        execute_trajectory(
            sim,
            robot,
            left_pick_traj,
            full_joint_ids,
            bread,
            pan,
            args.debug_state,
        )
        bread.clear_dynamics()
        execute_trajectory(
            sim,
            robot,
            right_pick_traj,
            full_joint_ids,
            bread,
            pan,
            args.debug_state,
        )
        pan.clear_dynamics()
        bread_pose_batch = clone_local_pose_from_first_env(bread).to(
            device=sim.device, dtype=torch.float32
        )
        pan_pose_batch = clone_local_pose_from_first_env(pan).to(
            device=sim.device, dtype=torch.float32
        )
        bread.clear_dynamics()
        pan.clear_dynamics()
        bread_pose = bread_pose_batch[0]
        pan_pose = pan_pose_batch[0]
        bread_held_state = compute_actual_held_state(
            robot,
            bread_semantics,
            bread_pose_batch,
            "left_arm",
            sim.device,
        )
        pan_held_state = compute_actual_held_state(
            robot,
            pan_semantics,
            pan_pose_batch,
            "right_arm",
            sim.device,
        )

    support_target_pose = build_support_object_target_pose(pan_pose, sim.device)
    placing_target_pose = build_placing_object_target_pose(
        bread_pose,
        bread_vertices,
        pan_vertices,
        support_target_pose,
        sim.device,
    )
    log_scene_targets(
        bread_pose,
        pan_pose,
        support_target_pose,
        placing_target_pose,
    )
    if not args.auto_play and not args.no_vis_eef_axis:
        draw_coordinated_axes(
            sim,
            support_target_pose,
            placing_target_pose,
        )
    coordinated_target = CoordinatedPlacementTarget(
        placing_object_target_pose=broadcast_pose_batch(
            placing_target_pose, num_envs=n_envs
        ),
        support_object_target_pose=broadcast_pose_batch(
            support_target_pose, num_envs=n_envs
        ),
        placing_held_object=bread_held_state,
        support_held_object=pan_held_state,
        placing_height_offset=BREAD_TARGET_HEIGHT_OFFSET,
        support_height_offset=SUPPORT_TARGET_HEIGHT_OFFSET,
        release=True,
    )
    start_time = time.time()
    coordinated_result = coordinated_action.execute(coordinated_target, state)
    coordinated_success = coordinated_result.success
    coordinated_traj = coordinated_result.trajectory
    state = coordinated_result.next_state
    logger.log_info(
        "Plan coordinated placement cost time: "
        f"{time.time() - start_time:.2f} seconds"
    )
    if not coordinated_success:
        logger.log_warning("Failed to plan coordinated placement trajectory.")
        return
    log_action_plan(
        robot,
        "coordinated_placement",
        coordinated_traj,
        full_joint_ids,
        coordinated_action._compute_segment_lengths(coordinated_action.cfg.release),
    )

    if args.diagnose_plan:
        return

    recording_started = start_auto_play_recording(
        sim,
        args,
        video_prefix="coordinated_placement_auto_play",
        look_at=(
            (-0.25, 0.0, 2.5),
            (-0.05, 0.0, 0.72),
            (0.0, 0.0, 1.0),
        ),
    )
    try:
        if args.auto_play and not args.no_vis_eef_axis:
            draw_coordinated_axes(
                sim,
                support_target_pose,
                placing_target_pose,
                num_envs=n_envs,
            )
        if wait_for_user:
            input("Press Enter to execute coordinated placement...")
        execute_trajectory(
            sim,
            robot,
            coordinated_traj,
            full_joint_ids,
            bread,
            pan,
            args.debug_state,
        )
        for _ in range(80):
            robot.set_qpos(state.last_qpos, joint_ids=full_joint_ids)
            sim.update(step=2)
            time.sleep(1e-2)
    finally:
        stop_auto_play_recording(sim, recording_started)
    if wait_for_user:
        input("Press Enter to exit the simulation...")


def main() -> None:
    """Run the coordinated placement demo."""
    args = parse_arguments()
    sim = initialize_simulation(args)
    robot = create_dual_ur5_robot(sim)
    run_coordinated_placement_demo(args, sim, robot)


if __name__ == "__main__":
    main()
