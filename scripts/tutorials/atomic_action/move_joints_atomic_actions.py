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

"""Demonstrate MoveJoints with named and explicit joint-space targets."""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
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
    JointPositionTarget,
    MoveJoints,
    MoveJointsCfg,
    NamedJointPositionTarget,
)
from embodichain.lab.sim.cfg import (
    JointDrivePropertiesCfg,
    LightCfg,
    MarkerCfg,
    RenderCfg,
    RobotCfg,
    URDFCfg,
)
from embodichain.lab.sim.objects import Robot
from embodichain.lab.sim.planners import MotionGenerator, MotionGenCfg, ToppraPlannerCfg
from embodichain.lab.sim.solvers import PytorchSolverCfg
from embodichain.utils import logger

GRIPPER_URDF_PATH = "DH_PGI_140_80/DH_PGI_140_80.urdf"
GRIPPER_HAND_JOINT_PATTERN = "GRIPPER_FINGER1_JOINT_1"
GRIPPER_TCP_Z = 0.15

MOVE_JOINTS_SAMPLE_INTERVAL = 80
POST_TRAJECTORY_STEPS = 120
DEFAULT_AUTO_PLAY_LOOK_AT = (
    (-1.6, -1.5, 1.2),
    (0.0, 0.0, 0.25),
    (0.0, 0.0, 1.0),
)
RECORD_WIDTH = 640
RECORD_HEIGHT = 480
VIEWER_WIDTH = 1600
VIEWER_HEIGHT = 900
AUTO_PLAY_RECORD_FPS = 20
AUTO_PLAY_RECORD_MAX_MEMORY = 2048
EEF_AXIS_LEN = 0.06
EEF_AXIS_SIZE = 0.003


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Demonstrate MoveJoints with named and explicit qpos targets."
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
        help="Log robot qpos during replay.",
    )
    parser.add_argument(
        "--no_vis_eef_axis",
        action="store_true",
        help="Do not draw the current end-effector/TCP coordinate frame before planning.",
    )
    return parser.parse_args()


def get_tutorial_window_size(args: argparse.Namespace) -> tuple[int, int]:
    """Return the viewer window size used by this tutorial."""
    return VIEWER_WIDTH, VIEWER_HEIGHT


def start_auto_play_recording(
    sim: SimulationManager,
    args: argparse.Namespace,
    video_prefix: str,
    look_at: tuple[
        Sequence[float],
        Sequence[float],
        Sequence[float],
    ] = DEFAULT_AUTO_PLAY_LOOK_AT,
) -> bool:
    """Start recording for ``--auto_play`` tutorial runs."""
    if not getattr(args, "auto_play", False):
        return False

    original_width = sim.sim_config.width
    original_height = sim.sim_config.height
    try:
        sim.sim_config.width = RECORD_WIDTH
        sim.sim_config.height = RECORD_HEIGHT
        if not sim.start_window_record(
            fps=AUTO_PLAY_RECORD_FPS,
            max_memory=AUTO_PLAY_RECORD_MAX_MEMORY,
            video_prefix=video_prefix,
            look_at=look_at,
            use_sim_time=True,
        ):
            raise RuntimeError("Failed to start auto_play recording.")
    finally:
        sim.sim_config.width = original_width
        sim.sim_config.height = original_height
    return True


def stop_auto_play_recording(
    sim: SimulationManager,
    recording_started: bool,
) -> None:
    """Stop recording and wait until the mp4 has been written."""
    if not recording_started:
        return

    if sim.is_window_recording():
        sim.stop_window_record()
    sim.wait_window_record_saves()


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


def make_arm_qpos(values: list[float], device: torch.device) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32, device=device)


def format_tensor(tensor: torch.Tensor) -> str:
    rounded = (tensor.detach().cpu() * 10000.0).round() / 10000.0
    return str(rounded.tolist())


def draw_current_eef_axis(sim: SimulationManager, robot: Robot) -> None:
    eef_pose = robot.compute_fk(
        qpos=robot.get_qpos(name="arm"),
        name="arm",
        to_matrix=True,
    )
    sim.draw_marker(
        cfg=MarkerCfg(
            name="current_eef_axis",
            marker_type="axis",
            axis_xpos=eef_pose,
            axis_size=EEF_AXIS_SIZE,
            axis_len=EEF_AXIS_LEN,
        )
    )


def run_move_joints_demo(args: argparse.Namespace) -> None:
    sim = initialize_simulation(args)
    robot = create_robot(sim)
    motion_gen = MotionGenerator(
        cfg=MotionGenCfg(planner_cfg=ToppraPlannerCfg(robot_uid=robot.uid))
    )

    ready_qpos = make_arm_qpos([0.35, -1.20, 1.30, -1.65, -1.57, 0.20], sim.device)
    home_qpos = make_arm_qpos([0.0, -1.57, 1.57, -1.57, -1.57, 0.0], sim.device)
    move_joints_cfg = MoveJointsCfg(
        control_part="arm",
        sample_interval=MOVE_JOINTS_SAMPLE_INTERVAL,
        named_joint_positions={"ready": ready_qpos},
    )

    atomic_engine = AtomicActionEngine(motion_generator=motion_gen)
    atomic_engine.register(MoveJoints(motion_gen, cfg=move_joints_cfg))

    if not args.headless:
        sim.open_window()
    if not args.no_vis_eef_axis:
        draw_current_eef_axis(sim, robot)
    if not args.auto_play:
        input("Inspect the robot, then press Enter to plan MoveJoints...")

    logger.log_info(
        "Planning MoveJoints: NamedJointPositionTarget('ready') -> explicit home qpos"
    )
    is_success, traj, _ = atomic_engine.run(
        steps=[
            ("move_joints", NamedJointPositionTarget(name="ready")),
            ("move_joints", JointPositionTarget(qpos=home_qpos)),
        ]
    )
    if not is_success:
        logger.log_warning("Failed to plan MoveJoints demo trajectory.")
        return

    if not args.auto_play:
        input("Press Enter to replay the MoveJoints demo...")

    recording_started = start_auto_play_recording(
        sim, args, video_prefix="move_joints_auto_play"
    )
    try:
        log_stride = max(1, traj.shape[1] // 10)
        for i in range(traj.shape[1]):
            robot.set_qpos(traj[:, i, :])
            sim.update(step=4)
            if args.debug_state and (i % log_stride == 0 or i == traj.shape[1] - 1):
                logger.log_info(
                    f"replay step {i}/{traj.shape[1] - 1}: "
                    f"arm_qpos={format_tensor(traj[0, i, :6])}"
                )
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


def main() -> None:
    args = parse_arguments()
    run_move_joints_demo(args)


if __name__ == "__main__":
    main()
