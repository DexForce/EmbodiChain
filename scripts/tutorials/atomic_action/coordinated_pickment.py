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

"""Demonstrate dual-arm coordinated pickment with selectable object meshes.

The two UR10 arms pinch opposite sides of one object, lift it together, and move
the object to an object-centric target pose while both grippers stay closed.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass
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
    CoordinatedPickment,
    CoordinatedPickmentCfg,
    CoordinatedPickmentTarget,
    ObjectSemantics,
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
from embodichain.lab.sim.planners import MotionGenCfg, MotionGenerator, ToppraPlannerCfg
from embodichain.lab.sim.shapes import MeshCfg
from embodichain.lab.sim.solvers import PytorchSolverCfg
from embodichain.utils import logger
from embodichain.utils.math import matrix_from_euler
from scripts.tutorials.atomic_action.tutorial_utils import (
    get_tutorial_window_size,
    start_auto_play_recording,
    stop_auto_play_recording,
)

DEFAULT_MESH_FRAME_CORRECTION_EULER_DEG = (-90.0, 0.0, 0.0)


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


ARM_URDF_PATH = "UniversalRobots/UR10/UR10.urdf"
GRIPPER_URDF_PATH = "DH_PGI_140_80/DH_PGI_140_80.urdf"
SCRIPT_DIR = Path(__file__).resolve().parent
TABLE_MESH_PATH = SCRIPT_DIR / "table.glb"
GRIPPER_TCP_Z = 0.121
ROBOT_INIT_POS = (2.25, 0.0, 0.1)
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
TABLE_INIT_POS, TABLE_INIT_ROT = transform_baseline_pose(
    BASELINE_TABLE_INIT_POS,
    BASELINE_TABLE_INIT_ROT,
    z_offset=SCENE_Z_OFFSET,
)


@dataclass(frozen=True)
class PickmentObjectPreset:
    """Configuration for an object used by the coordinated pickment demo."""

    label: str
    mesh_path: Path
    init_xy: tuple[float, float]
    init_rot: tuple[float, float, float]
    table_clearance: float
    body_scale: tuple[float, float, float]
    grasp_end_margin_ratio: float
    grasp_z_clearance: float
    target_translation: tuple[float, float, float]
    target_world_yaw_deg: float
    hand_close_qpos: float
    grasp_z_ratio: float | None = None


OBJECT_PRESETS = {
    "pencil": PickmentObjectPreset(
        label="pencil",
        mesh_path=SCRIPT_DIR / "pencil.glb",
        init_xy=(-0.02, 0.02),
        # Rotate the imported pencil from its default upright orientation to a tabletop pose.
        init_rot=(90.0, 0.0, 0.0),
        table_clearance=0.008,
        body_scale=(2.0, 2.0, 2.0),
        grasp_end_margin_ratio=0.12,
        grasp_z_clearance=0.015,
        target_translation=(-0.22, -0.04, 0.16),
        target_world_yaw_deg=0.0,
        hand_close_qpos=0.026,
    ),
    "pot": PickmentObjectPreset(
        label="pot",
        mesh_path=SCRIPT_DIR / "pot.glb",
        init_xy=(-0.02, 0.02),
        init_rot=(-90.0, 90.0, 0.0),
        table_clearance=0.008,
        body_scale=(2.0, 2.0, 2.0),
        grasp_end_margin_ratio=0.08,
        grasp_z_clearance=0.01,
        target_translation=(-0.12, -0.03, 0.12),
        target_world_yaw_deg=0.0,
        hand_close_qpos=0.026,
        grasp_z_ratio=0.55,
    ),
}
PICKMENT_SAMPLE_INTERVAL = 96
PICKMENT_OBJECT_MOTION_KEYFRAMES = 6
PICKMENT_PRE_GRASP_DISTANCE = 0.11
PICKMENT_LIFT_HEIGHT = 0.10
PICKMENT_HAND_INTERP_STEPS = 10
PICKMENT_HOLD_STEPS = 4
TRAJECTORY_SIM_STEPS = 4


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the demo."""
    parser = argparse.ArgumentParser(description="Dual-arm coordinated pickment demo")
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
        "--headless_play",
        action="store_true",
        help="Execute planned trajectories without opening the viewer window.",
    )
    parser.add_argument(
        "--object",
        choices=sorted(OBJECT_PRESETS),
        default="pencil",
        help="Object mesh to grasp in the coordinated pickment demo.",
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
    cfg = RobotCfg(
        uid="DualUR10CoordinatedPickment",
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
            fname="dual_ur10_coordinated_pickment",
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
        init_qpos=list(LEFT_ARM_HOME) + list(RIGHT_ARM_HOME) + [0.0, 0.0, 0.0, 0.0],
    )
    return sim.add_robot(cfg=cfg)


def create_table(sim: SimulationManager) -> RigidObject:
    """Create the table mesh used by the pickment scene."""
    return sim.add_rigid_object(
        cfg=RigidObjectCfg(
            uid="table",
            shape=MeshCfg(fpath=str(TABLE_MESH_PATH)),
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


def create_pickment_object(
    sim: SimulationManager,
    preset: PickmentObjectPreset,
) -> RigidObject:
    """Create the selected object mesh on the table."""
    obj = sim.add_rigid_object(
        cfg=RigidObjectCfg(
            uid=preset.label,
            shape=MeshCfg(fpath=str(preset.mesh_path), compute_uv=False),
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
            max_convex_hull_num=16,
            init_pos=[preset.init_xy[0], preset.init_xy[1], TABLE_TOP_Z],
            init_rot=list(preset.init_rot),
            body_scale=preset.body_scale,
        )
    )
    obj.cfg.init_pos = compute_tabletop_init_pos(obj, preset)
    obj.reset()
    return obj


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
    robot: Robot,
    hand_control_part: str,
    device: torch.device,
    close_qpos: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get open and close qpos for a PGI gripper control part."""
    limits = robot.get_qpos_limits(name=hand_control_part)[0].to(
        device=device, dtype=torch.float32
    )
    hand_open = limits[:, 0]
    hand_close = torch.clamp(
        torch.full_like(limits[:, 1], close_qpos),
        min=limits[:, 0],
        max=limits[:, 1],
    )
    return hand_open, hand_close


def get_local_vertices(obj: RigidObject) -> torch.Tensor:
    """Get scaled local mesh vertices."""
    return obj.get_vertices(env_ids=[0], scale=True)[0]


def compute_local_bounds(vertices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute local mesh AABB from scaled vertices."""
    return vertices.min(dim=0).values, vertices.max(dim=0).values


def compute_tabletop_init_pos(
    obj: RigidObject,
    preset: PickmentObjectPreset,
) -> tuple[float, float, float]:
    """Place an object so its rotated mesh bottom sits on the table."""
    vertices = get_local_vertices(obj)
    rot = torch.as_tensor(preset.init_rot, dtype=torch.float32, device=vertices.device)
    rot = rot.unsqueeze(0) * torch.pi / 180.0
    upright_rot = matrix_from_euler(rot, "XYZ")[0]
    rotated_vertices = vertices @ upright_rot.T
    bottom_z = rotated_vertices[:, 2].min().item()
    z = TABLE_TOP_Z + preset.table_clearance - bottom_z
    return (preset.init_xy[0], preset.init_xy[1], z)


def invert_pose(pose: torch.Tensor) -> torch.Tensor:
    """Invert batched homogeneous transforms."""
    inv_pose = pose.clone()
    rot_t = pose[:, :3, :3].transpose(1, 2)
    inv_pose[:, :3, :3] = rot_t
    inv_pose[:, :3, 3] = -torch.bmm(rot_t, pose[:, :3, 3:4]).squeeze(-1)
    return inv_pose


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


def normalize_vector(vector: torch.Tensor, fallback: torch.Tensor) -> torch.Tensor:
    """Normalize a vector with a deterministic fallback for degenerate cases."""
    norm = torch.linalg.norm(vector)
    if norm < 1e-6:
        return fallback.to(device=vector.device, dtype=vector.dtype)
    return vector / norm


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


def build_object_grasp_poses(
    object_pose: torch.Tensor,
    local_vertices: torch.Tensor,
    preset: PickmentObjectPreset,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build left/right TCP poses that pinch opposite sides of the object."""
    local_min, local_max = compute_local_bounds(local_vertices)
    extents = local_max - local_min
    long_axis_idx = int(torch.argmax(extents).item())
    axis_local = torch.zeros(3, dtype=torch.float32, device=device)
    axis_local[long_axis_idx] = 1.0
    long_axis = normalize_vector(
        object_pose[:3, :3] @ axis_local,
        torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=device),
    )

    local_center = 0.5 * (local_min + local_max)
    margin = extents[long_axis_idx] * preset.grasp_end_margin_ratio
    left_local = local_center.clone()
    right_local = local_center.clone()
    left_local[long_axis_idx] = local_min[long_axis_idx] + margin
    right_local[long_axis_idx] = local_max[long_axis_idx] - margin

    world_min, world_max = compute_world_bounds(object_pose, local_vertices)
    left_position = object_pose[:3, 3] + object_pose[:3, :3] @ left_local.to(device)
    right_position = object_pose[:3, 3] + object_pose[:3, :3] @ right_local.to(device)
    if preset.grasp_z_ratio is None:
        grasp_z = world_max[2] + preset.grasp_z_clearance
    else:
        grasp_z = (
            world_min[2]
            + (world_max[2] - world_min[2]) * preset.grasp_z_ratio
            + preset.grasp_z_clearance
        )
    left_position[2] = grasp_z
    right_position[2] = grasp_z

    z_axis = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device=device)
    x_axis = normalize_vector(
        torch.cross(long_axis, z_axis, dim=0),
        torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=device),
    )
    y_axis = normalize_vector(torch.cross(z_axis, x_axis, dim=0), long_axis)

    left_pose = torch.eye(4, dtype=torch.float32, device=device)
    left_pose[:3, 0] = x_axis
    left_pose[:3, 1] = y_axis
    left_pose[:3, 2] = z_axis
    left_pose[:3, 3] = left_position

    right_pose = torch.eye(4, dtype=torch.float32, device=device)
    right_pose[:3, 0] = -x_axis
    right_pose[:3, 1] = -y_axis
    right_pose[:3, 2] = z_axis
    right_pose[:3, 3] = right_position
    return left_pose, right_pose


def build_object_target_pose(
    object_pose: torch.Tensor,
    object_vertices: torch.Tensor,
    preset: PickmentObjectPreset,
    device: torch.device,
) -> torch.Tensor:
    """Build the target pose for the whole object."""
    pose = rotate_pose_about_world_z(
        object_pose.clone().to(device=device, dtype=torch.float32),
        preset.target_world_yaw_deg,
    )
    pose[:3, 3] += torch.tensor(
        preset.target_translation, dtype=torch.float32, device=device
    )
    bottom_z = compute_world_bounds(pose, object_vertices)[0][2]
    pose[2, 3] += TABLE_TOP_Z + preset.table_clearance + 0.10 - bottom_z
    return pose


def format_tensor(tensor: torch.Tensor) -> str:
    """Format tensor values for compact logging."""
    rounded = (tensor.detach().cpu() * 10000.0).round() / 10000.0
    return str(rounded.tolist())


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


def log_scene_targets(
    object_label: str,
    object_pose: torch.Tensor,
    target_pose: torch.Tensor,
    left_grasp_pose: torch.Tensor,
    right_grasp_pose: torch.Tensor,
) -> None:
    """Log compact object and grasp target positions."""
    logger.log_info(
        "pickment scene: "
        f"object={object_label}, "
        f"object_origin={format_tensor(object_pose[:3, 3])}, "
        f"target_origin={format_tensor(target_pose[:3, 3])}, "
        f"left_grasp={format_tensor(left_grasp_pose[:3, 3])}, "
        f"right_grasp={format_tensor(right_grasp_pose[:3, 3])}"
    )


def log_execution_state(
    robot: Robot,
    obj: RigidObject,
    step_idx: int,
    total_steps: int,
) -> None:
    """Log hand and object state during execution."""
    object_pose = obj.get_local_pose(to_matrix=True)
    left_hand = robot.get_qpos(name="left_hand")
    right_hand = robot.get_qpos(name="right_hand")
    logger.log_info(
        f"step={step_idx}/{total_steps - 1}, "
        f"left_hand={format_tensor(left_hand[0])}, "
        f"right_hand={format_tensor(right_hand[0])}, "
        f"{obj.uid}_pos={format_tensor(object_pose[0, :3, 3])}"
    )


def execute_trajectory(
    sim: SimulationManager,
    robot: Robot,
    traj: torch.Tensor,
    obj: RigidObject,
    debug_state: bool,
) -> None:
    """Play a planned trajectory in simulation."""
    total_steps = traj.shape[1]
    log_stride = max(1, total_steps // 10)
    for i in range(total_steps):
        robot.set_qpos(traj[:, i, :])
        sim.update(step=TRAJECTORY_SIM_STEPS)
        if debug_state and (i % log_stride == 0 or i == total_steps - 1):
            log_execution_state(robot, obj, i, total_steps)
        time.sleep(1e-2)


def run_coordinated_pickment_demo(
    args: argparse.Namespace,
    sim: SimulationManager,
    robot: Robot,
) -> None:
    """Plan and optionally execute coordinated object pickment."""
    preset = OBJECT_PRESETS[args.object]
    create_table(sim)
    obj = create_pickment_object(sim, preset)
    settle_object(sim, obj, step=0)
    object_pose = obj.get_local_pose(to_matrix=True)[0].to(
        device=sim.device, dtype=torch.float32
    )
    object_vertices = get_local_vertices(obj)
    object_semantics = create_object_semantics(obj, preset.label)
    motion_gen = MotionGenerator(
        cfg=MotionGenCfg(planner_cfg=ToppraPlannerCfg(robot_uid=robot.uid))
    )

    left_open, left_close = get_hand_open_close_qpos(
        robot, "left_hand", sim.device, preset.hand_close_qpos
    )
    right_open, right_close = get_hand_open_close_qpos(
        robot, "right_hand", sim.device, preset.hand_close_qpos
    )
    pickment_action = CoordinatedPickment(
        motion_generator=motion_gen,
        cfg=CoordinatedPickmentCfg(
            control_part="dual_arm",
            left_arm_control_part="left_arm",
            right_arm_control_part="right_arm",
            left_hand_control_part="left_hand",
            right_hand_control_part="right_hand",
            left_hand_open_qpos=left_open,
            left_hand_close_qpos=left_close,
            right_hand_open_qpos=right_open,
            right_hand_close_qpos=right_close,
            pre_grasp_distance=PICKMENT_PRE_GRASP_DISTANCE,
            lift_height=PICKMENT_LIFT_HEIGHT,
            sample_interval=PICKMENT_SAMPLE_INTERVAL,
            hand_interp_steps=PICKMENT_HAND_INTERP_STEPS,
            hold_steps=PICKMENT_HOLD_STEPS,
            object_motion_keyframes=PICKMENT_OBJECT_MOTION_KEYFRAMES,
        ),
    )

    left_grasp_pose, right_grasp_pose = build_object_grasp_poses(
        object_pose,
        object_vertices,
        preset,
        sim.device,
    )
    target_pose = build_object_target_pose(
        object_pose,
        object_vertices,
        preset,
        sim.device,
    )
    log_scene_targets(
        preset.label,
        object_pose,
        target_pose,
        left_grasp_pose,
        right_grasp_pose,
    )

    left_object_to_eef = torch.bmm(
        invert_pose(object_pose.unsqueeze(0)),
        left_grasp_pose.unsqueeze(0),
    )
    right_object_to_eef = torch.bmm(
        invert_pose(object_pose.unsqueeze(0)),
        right_grasp_pose.unsqueeze(0),
    )
    pickment_target = CoordinatedPickmentTarget(
        object_target_pose=target_pose,
        object_semantics=object_semantics,
        left_object_to_eef=left_object_to_eef,
        right_object_to_eef=right_object_to_eef,
        object_initial_pose=object_pose,
    )

    wait_for_user = not args.auto_play and not args.headless_play
    if not args.diagnose_plan and not args.headless_play:
        sim.open_window()
        if wait_for_user:
            input("Inspect the scene, then press Enter to plan pickment...")

    start_time = time.time()
    result = pickment_action.execute(
        pickment_target,
        WorldState(last_qpos=robot.get_qpos().clone()),
    )
    logger.log_info(
        f"Plan coordinated pickment cost time: {time.time() - start_time:.2f} seconds"
    )
    if not result.success:
        logger.log_warning("Failed to plan coordinated pickment trajectory.")
        return
    traj = result.trajectory
    joint_ids = list(range(robot.dof))
    log_action_plan(
        robot,
        "coordinated_pickment",
        traj,
        joint_ids,
        pickment_action.get_segment_lengths(),
    )

    if args.diagnose_plan:
        return

    if wait_for_user:
        input("Press Enter to execute coordinated pickment...")
    recording_started = start_auto_play_recording(
        sim,
        args,
        video_prefix=f"coordinated_pickment_{args.object}_auto_play",
    )
    try:
        execute_trajectory(
            sim,
            robot,
            traj,
            obj,
            args.debug_state,
        )
    finally:
        stop_auto_play_recording(sim, recording_started)
    if wait_for_user:
        input("Press Enter to exit the simulation...")


def main() -> None:
    """Run the coordinated pickment demo."""
    args = parse_arguments()
    sim = initialize_simulation(args)
    robot = create_dual_ur10_robot(sim)
    run_coordinated_pickment_demo(args, sim, robot)


if __name__ == "__main__":
    main()
