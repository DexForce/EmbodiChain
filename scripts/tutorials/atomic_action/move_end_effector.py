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

"""Demonstrate MoveEndEffector with a multi-waypoint pose trajectory."""

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
)
from embodichain.lab.sim.cfg import (
    JointDrivePropertiesCfg,
    LightCfg,
    RenderCfg,
    RobotCfg,
    URDFCfg,
)
from embodichain.lab.sim.objects import Robot
from embodichain.lab.sim.planners import MotionGenerator, MotionGenCfg, ToppraPlannerCfg
from embodichain.lab.sim.solvers import PytorchSolverCfg
from embodichain.utils import logger
from scripts.tutorials.atomic_action.tutorial_utils import (
    draw_axis_marker,
    get_tutorial_window_size,
    start_auto_play_recording,
    stop_auto_play_recording,
)

GRIPPER_URDF_PATH = "DH_PGI_140_80/DH_PGI_140_80.urdf"
GRIPPER_HAND_JOINT_PATTERN = "GRIPPER_FINGER1_JOINT_1"
GRIPPER_TCP_Z = 0.15

MOVE_SAMPLE_INTERVAL = 80
POST_TRAJECTORY_STEPS = 120


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Demonstrate MoveEndEffector with a multi-waypoint pose trajectory."
    )
    add_env_launcher_args_to_parser(parser)
    parser.add_argument(
        "--auto_play",
        action="store_true",
        help="Run the viewer demo without waiting for keyboard input.",
    )
    parser.add_argument(
        "--no_vis_eef_axis",
        action="store_true",
        help="Do not draw the current end-effector/TCP coordinate frame before planning.",
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
        solver_cfg={
            "arm": PytorchSolverCfg(
                end_link_name="ee_link",
                root_link_name="base_link",
                tcp=[
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, GRIPPER_TCP_Z],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            )
        },
        init_qpos=[0.0, -1.57, 1.57, -1.57, -1.57, 0.0, 0.0, 0.0],
        init_pos=position,
    )
    return sim.add_robot(cfg=cfg)


def make_top_down_eef_pose(device: torch.device) -> torch.Tensor:
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
    pose[:3, 3] = torch.tensor([0.30, -0.20, 0.36], dtype=torch.float32, device=device)
    return pose


def make_side_eef_pose(device: torch.device) -> torch.Tensor:
    """A second waypoint offset from the top-down pose for the multi-waypoint demo."""
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
    pose[:3, 3] = torch.tensor([0.45, 0.10, 0.30], dtype=torch.float32, device=device)
    return pose


def format_tensor(tensor: torch.Tensor) -> str:
    rounded = (tensor.detach().cpu() * 10000.0).round() / 10000.0
    return str(rounded.tolist())


def main() -> None:
    """Move the robot end effector through a multi-waypoint pose trajectory."""
    args = parse_arguments()

    # ------------------------------------------------------------------ #
    # Step 1: Set up simulation and robot                                 #
    # ------------------------------------------------------------------ #
    sim = initialize_simulation(args)
    robot = create_robot(sim)

    # ------------------------------------------------------------------ #
    # Step 2: Create a MotionGenerator for the robot                      #
    # ------------------------------------------------------------------ #
    motion_gen = MotionGenerator(
        cfg=MotionGenCfg(planner_cfg=ToppraPlannerCfg(robot_uid=robot.uid))
    )

    # ------------------------------------------------------------------ #
    # Step 3: Configure the MoveEndEffector atomic action                 #
    # ------------------------------------------------------------------ #
    move_cfg = MoveEndEffectorCfg(
        control_part="arm",
        sample_interval=MOVE_SAMPLE_INTERVAL,
    )

    # ------------------------------------------------------------------ #
    # Step 4: Build the AtomicActionEngine                                #
    # ------------------------------------------------------------------ #
    atomic_engine = AtomicActionEngine(motion_generator=motion_gen)
    atomic_engine.register(MoveEndEffector(motion_gen, cfg=move_cfg))

    # ------------------------------------------------------------------ #
    # Step 5: Define and visualize the end-effector target                #
    # ------------------------------------------------------------------ #
    target_pose = make_top_down_eef_pose(sim.device)
    side_pose = make_side_eef_pose(sim.device)
    if not args.headless:
        sim.open_window()
    if not args.no_vis_eef_axis:
        draw_axis_marker(sim, "move_end_effector_target_axis", target_pose)
        draw_axis_marker(sim, "move_end_effector_side_axis", side_pose)
    if not args.auto_play:
        input("Inspect the robot, then press Enter to plan MoveEndEffector...")

    # ------------------------------------------------------------------ #
    # Step 6: Plan the declared (name, typed_target) sequence             #
    # ------------------------------------------------------------------ #
    # Pass a multi-waypoint trajectory (n_envs, n_waypoint, 4, 4): the
    # end-effector visits `target_pose` then `side_pose` in a single plan.
    n_envs = robot.get_qpos().shape[0]
    multi_waypoint_xpos = (
        torch.stack([target_pose, side_pose], dim=0)
        .unsqueeze(0)
        .repeat(n_envs, 1, 1, 1)
    )
    logger.log_info(
        "Planning MoveEndEffector through multi-waypoint trajectory: "
        f"xpos0={format_tensor(target_pose[:3, 3])} -> "
        f"xpos1={format_tensor(side_pose[:3, 3])}"
    )
    is_success, traj, _ = atomic_engine.run(
        steps=[
            (
                "move_end_effector",
                EndEffectorPoseTarget(xpos=multi_waypoint_xpos),
            )
        ]
    )
    if not is_success:
        logger.log_warning("Failed to plan MoveEndEffector demo trajectory.")
        return

    if not args.auto_play:
        input("Press Enter to replay the MoveEndEffector demo...")

    # ------------------------------------------------------------------ #
    # Step 7: Replay the planned trajectory                               #
    # ------------------------------------------------------------------ #
    recording_started = start_auto_play_recording(
        sim, args, video_prefix="move_end_effector_auto_play"
    )
    try:
        for i in range(traj.shape[1]):
            robot.set_qpos(traj[:, i, :])
            sim.update(step=4)
            time.sleep(1e-2)

        final_qpos = traj[:, -1, :]
        for _ in range(POST_TRAJECTORY_STEPS):
            robot.set_qpos(final_qpos)
            sim.update(step=2)
            time.sleep(1e-2)
    finally:
        stop_auto_play_recording(sim, recording_started)

    if not args.auto_play:
        input("Press Enter to exit the simulation...")


if __name__ == "__main__":
    main()
