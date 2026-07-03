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

"""Demonstrate Press on the center of a regular wooden block."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from embodichain.data import get_data_path
from embodichain.lab.gym.utils.gym_utils import add_env_launcher_args_to_parser
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.atomic_actions import (
    AtomicActionEngine,
    EndEffectorPoseTarget,
    MoveEndEffector,
    MoveEndEffectorCfg,
    Press,
    PressCfg,
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
from embodichain.lab.sim.material import VisualMaterialCfg
from embodichain.lab.sim.objects import RigidObject, Robot
from embodichain.lab.sim.planners import MotionGenerator, MotionGenCfg, ToppraPlannerCfg
from embodichain.lab.sim.shapes import CubeCfg
from embodichain.utils import logger
from scripts.tutorials.atomic_action.tutorial_utils import (
    draw_axis_marker,
    get_tutorial_window_size,
    make_ur5_solver_cfg,
    start_auto_play_recording,
    stop_auto_play_recording,
)

GRIPPER_URDF_PATH = "DH_PGI_140_80/DH_PGI_140_80.urdf"
GRIPPER_HAND_JOINT_PATTERN = "GRIPPER_FINGER1_JOINT_1"
GRIPPER_TCP_Z = 0.15

MOVE_SAMPLE_INTERVAL = 60
PRESS_SAMPLE_INTERVAL = 90
HAND_INTERP_STEPS = 12
POST_TRAJECTORY_STEPS = 180
TABLE_SIZE = [1.0, 1.4, 0.05]
TABLE_TOP_Z = -0.045
BLOCK_SIZE = [0.12, 0.12, 0.06]
BLOCK_CENTER = [-0.30, -0.12, TABLE_TOP_Z + 0.5 * BLOCK_SIZE[2]]
PRESS_CLEARANCE = 0.13
PRESS_SURFACE_OFFSET = 0.003
DEFAULT_PRESS_TOLERANCE = 0.01


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Demonstrate Press on the center of a regular wooden block."
    )
    add_env_launcher_args_to_parser(parser)
    parser.add_argument(
        "--auto_play",
        action="store_true",
        help="Run the viewer demo without waiting for keyboard input.",
    )
    parser.add_argument(
        "--debug_state",
        action="store_true",
        help="Log the block pose during replay.",
    )
    parser.add_argument(
        "--press_tolerance",
        type=float,
        default=DEFAULT_PRESS_TOLERANCE,
        help="XY tolerance in meters for checking whether press reaches block center.",
    )
    parser.add_argument(
        "--block_pos",
        type=float,
        nargs=2,
        default=BLOCK_CENTER[:2],
        metavar=("X", "Y"),
        help="Initial XY position of the wooden block center.",
    )
    parser.add_argument(
        "--no_vis_eef_axis",
        action="store_true",
        help="Do not draw the press target coordinate frame before planning.",
    )
    return parser.parse_args()


def initialize_simulation(args: argparse.Namespace) -> SimulationManager:
    width, height = get_tutorial_window_size(args)
    cfg = SimulationManagerCfg(
        width=width,
        height=height,
        headless=True,
        sim_device=args.device,
        render_cfg=RenderCfg(renderer=args.renderer),
        physics_dt=1.0 / 100.0,
        arena_space=2.5,
    )
    sim = SimulationManager(cfg)
    sim.add_light(
        cfg=LightCfg(
            uid="main_light",
            color=(0.6, 0.6, 0.6),
            intensity=30.0,
            init_pos=(1.0, 0.0, 3.0),
        )
    )
    return sim


def create_robot(sim: SimulationManager, position=(0.0, 0.0, 0.0)) -> Robot:
    ur5_urdf_path = get_data_path("UniversalRobots/UR5/UR5.urdf")
    gripper_urdf_path = get_data_path(GRIPPER_URDF_PATH)
    cfg = RobotCfg(
        uid="UR5",
        urdf_cfg=URDFCfg(
            components=[
                {"component_type": "arm", "urdf_path": ur5_urdf_path},
                {"component_type": "hand", "urdf_path": gripper_urdf_path},
            ]
        ),
        drive_pros=JointDrivePropertiesCfg(
            stiffness={"JOINT[0-9]": 1e4, GRIPPER_HAND_JOINT_PATTERN: 1e3},
            damping={"JOINT[0-9]": 1e3, GRIPPER_HAND_JOINT_PATTERN: 1e2},
            max_effort={"JOINT[0-9]": 1e5, GRIPPER_HAND_JOINT_PATTERN: 1e4},
            drive_type="force",
        ),
        control_parts={
            "arm": ["JOINT[0-9]"],
            "hand": [GRIPPER_HAND_JOINT_PATTERN],
        },
        solver_cfg={"arm": make_ur5_solver_cfg(GRIPPER_TCP_Z)},
        init_qpos=[0.0, -1.57, 1.57, -1.57, -1.57, 0.0, 0.0, 0.0],
        init_pos=position,
    )
    return sim.add_robot(cfg=cfg)


def create_table(sim: SimulationManager) -> RigidObject:
    cfg = RigidObjectCfg(
        uid="table",
        shape=CubeCfg(size=TABLE_SIZE),
        body_type="static",
        attrs=RigidBodyAttributesCfg(
            dynamic_friction=0.8,
            static_friction=0.9,
        ),
        init_pos=[-0.30, 0.10, TABLE_TOP_Z - 0.5 * TABLE_SIZE[2]],
    )
    return sim.add_rigid_object(cfg=cfg)


def create_wooden_block(
    sim: SimulationManager,
    block_center: list[float],
) -> RigidObject:
    cfg = RigidObjectCfg(
        uid="wooden_block",
        shape=CubeCfg(
            size=BLOCK_SIZE,
            visual_material=VisualMaterialCfg(
                uid="wooden_block_mat",
                base_color=[0.58, 0.32, 0.14, 1.0],
                roughness=0.85,
            ),
        ),
        body_type="static",
        attrs=RigidBodyAttributesCfg(
            dynamic_friction=0.8,
            static_friction=0.9,
        ),
        init_pos=block_center,
    )
    return sim.add_rigid_object(cfg=cfg)


def settle_object(sim: SimulationManager, obj: RigidObject, step: int = 5) -> None:
    if sim.device.type == "cuda":
        sim.init_gpu_physics()
    obj.reset()
    sim.update(step=step)
    obj.clear_dynamics()


def get_hand_close_qpos(robot: Robot, device: torch.device) -> torch.Tensor:
    hand_limits = robot.get_qpos_limits(name="hand")[0].to(
        device=device, dtype=torch.float32
    )
    return hand_limits[:, 1]


def make_top_down_eef_pose(position: torch.Tensor) -> torch.Tensor:
    pose = torch.eye(4, dtype=torch.float32, device=position.device)
    pose[:3, :3] = torch.tensor(
        [
            [-0.0539, -0.9985, -0.0022],
            [-0.9977, 0.0540, -0.0401],
            [0.0401, 0.0000, -0.9992],
        ],
        dtype=torch.float32,
        device=position.device,
    )
    pose[:3, 3] = position
    return pose


def format_tensor(tensor: torch.Tensor) -> str:
    rounded = (tensor.detach().cpu() * 10000.0).round() / 10000.0
    return str(rounded.tolist())


def log_block_state(block: RigidObject, label: str) -> None:
    block_pose = block.get_local_pose(to_matrix=True)
    logger.log_info(
        f"{label}: pos={format_tensor(block_pose[0, :3, 3])}, "
        f"z_axis={format_tensor(block_pose[0, :3, 2])}"
    )


def draw_press_target_axis(sim: SimulationManager, press_target: torch.Tensor) -> None:
    if press_target.dim() == 2:
        press_target = press_target.unsqueeze(0)
    draw_axis_marker(sim, "press_target_axis", press_target)


def compute_press_center_check(
    robot: Robot,
    traj: torch.Tensor,
    block: RigidObject,
    tolerance: float,
) -> tuple[bool, float, int, torch.Tensor, torch.Tensor]:
    arm_joint_ids = robot.get_joint_ids(name="arm")
    press_segment_start = MOVE_SAMPLE_INTERVAL + HAND_INTERP_STEPS
    press_segment_end = MOVE_SAMPLE_INTERVAL + PRESS_SAMPLE_INTERVAL
    arm_traj = traj[:, press_segment_start:press_segment_end, arm_joint_ids]
    fk_pose = torch.stack(
        [
            robot.compute_fk(
                qpos=waypoint.unsqueeze(0),
                name="arm",
                to_matrix=True,
            )[0]
            for waypoint in arm_traj[0]
        ],
        dim=0,
    )

    block_pose = block.get_local_pose(to_matrix=True)
    block_center = block_pose[0, :3, 3]
    block_top_z = block_center[2] + 0.5 * BLOCK_SIZE[2]
    target_xy = block_center[:2]
    target_z = block_top_z + PRESS_SURFACE_OFFSET

    xy_error = torch.linalg.norm(fk_pose[:, :2, 3] - target_xy, dim=1)
    z_error = torch.abs(fk_pose[:, 2, 3] - target_z)
    combined_error = xy_error + z_error
    best_idx = int(torch.argmin(combined_error).item())
    best_pos = fk_pose[best_idx, :3, 3]
    center_error = float(torch.linalg.norm(best_pos[:2] - target_xy).item())
    return (
        center_error <= tolerance,
        center_error,
        press_segment_start + best_idx,
        best_pos,
        torch.tensor(
            [target_xy[0], target_xy[1], target_z],
            dtype=torch.float32,
            device=traj.device,
        ),
    )


def run_press_demo(args: argparse.Namespace) -> None:
    sim = initialize_simulation(args)
    robot = create_robot(sim)
    create_table(sim)
    block_center = [
        args.block_pos[0],
        args.block_pos[1],
        TABLE_TOP_Z + 0.5 * BLOCK_SIZE[2],
    ]
    block = create_wooden_block(sim, block_center)

    settle_object(sim, block, step=5)
    motion_gen = MotionGenerator(
        cfg=MotionGenCfg(planner_cfg=ToppraPlannerCfg(robot_uid=robot.uid))
    )
    hand_close = get_hand_close_qpos(robot, sim.device)

    atomic_engine = AtomicActionEngine(motion_generator=motion_gen)
    atomic_engine.register(
        MoveEndEffector(
            motion_gen,
            cfg=MoveEndEffectorCfg(
                control_part="arm",
                sample_interval=MOVE_SAMPLE_INTERVAL,
            ),
        )
    )
    atomic_engine.register(
        Press(
            motion_gen,
            cfg=PressCfg(
                control_part="arm",
                hand_control_part="hand",
                hand_close_qpos=hand_close,
                sample_interval=PRESS_SAMPLE_INTERVAL,
                hand_interp_steps=HAND_INTERP_STEPS,
            ),
        )
    )

    if not args.headless:
        sim.open_window()
    if not args.auto_play:
        input("Inspect the wooden block, then press Enter to plan...")

    block_pose = block.get_local_pose(to_matrix=True)
    block_top_z = block_pose[0, 2, 3] + 0.5 * BLOCK_SIZE[2]
    press_position = block_pose[0, :3, 3].clone()
    press_position[2] = block_top_z + PRESS_SURFACE_OFFSET
    move_position = press_position.clone()
    move_position[2] = block_top_z + PRESS_CLEARANCE

    move_target = make_top_down_eef_pose(move_position)
    press_target = make_top_down_eef_pose(press_position)
    if not args.no_vis_eef_axis:
        draw_press_target_axis(sim, press_target)

    logger.log_info("Planning MoveEndEffector -> Press")
    start_time = time.time()
    is_success, traj, _ = atomic_engine.run(
        steps=[
            ("move_end_effector", EndEffectorPoseTarget(xpos=move_target)),
            ("press", EndEffectorPoseTarget(xpos=press_target)),
        ]
    )
    cost_time = time.time() - start_time
    logger.log_info(f"Plan trajectory cost time: {cost_time:.2f} seconds")
    if not is_success:
        logger.log_warning("Failed to plan Press demo trajectory.")
        return

    is_center_hit, center_error, hit_step, hit_pos, expected_pos = (
        compute_press_center_check(
            robot=robot,
            traj=traj,
            block=block,
            tolerance=args.press_tolerance,
        )
    )
    logger.log_info(
        "Press center check: "
        f"success={is_center_hit}, "
        f"xy_error={center_error:.4f} m, "
        f"hit_step={hit_step}, "
        f"hit_pos={format_tensor(hit_pos)}, "
        f"expected={format_tensor(expected_pos)}"
    )
    if not is_center_hit:
        logger.log_warning(
            "Press planned trajectory did not reach the block center within "
            f"{args.press_tolerance:.4f} m."
        )
        return

    if not args.auto_play:
        input("Press Enter to replay the Press demo...")

    recording_started = start_auto_play_recording(
        sim, args, video_prefix="press_auto_play"
    )
    try:
        log_stride = max(1, traj.shape[1] // 10)
        for i in range(traj.shape[1]):
            robot.set_qpos(traj[:, i, :])
            sim.update(step=4)
            if args.debug_state and (i % log_stride == 0 or i == traj.shape[1] - 1):
                log_block_state(block, f"replay step {i}/{traj.shape[1] - 1}")
            time.sleep(1e-2)

        logger.log_info("Press returned the end-effector to the starting pose.")
        final_qpos = traj[:, -1, :]
        for i in range(POST_TRAJECTORY_STEPS):
            robot.set_qpos(final_qpos)
            sim.update(step=2)
            if args.debug_state and i % max(1, POST_TRAJECTORY_STEPS // 5) == 0:
                log_block_state(block, f"post step {i}/{POST_TRAJECTORY_STEPS - 1}")
            time.sleep(1e-2)
    finally:
        stop_auto_play_recording(sim, recording_started)

    if not args.auto_play:
        input("Press Enter to exit the simulation...")


def main() -> None:
    args = parse_arguments()
    run_press_demo(args)


if __name__ == "__main__":
    main()
