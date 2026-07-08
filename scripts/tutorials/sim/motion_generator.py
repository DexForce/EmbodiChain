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

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence

import numpy as np
import torch

from embodichain.lab.gym.utils.gym_utils import add_env_launcher_args_to_parser
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import RenderCfg
from embodichain.lab.sim.objects import Robot
from embodichain.lab.sim.planners import (
    MotionGenCfg,
    MotionGenOptions,
    MotionGenerator,
    PlanState,
    ToppraPlanOptions,
    ToppraPlannerCfg,
)
from embodichain.lab.sim.planners.utils import TrajectorySampleMethod
from embodichain.lab.sim.robots import CobotMagicCfg

RECORD_WIDTH = 1920
RECORD_HEIGHT = 1080
DEFAULT_ARENA_SPACE = 3.0
DEFAULT_RECORD_TARGET_Z = 0.95
DEFAULT_RECORD_MAX_MEMORY = 2048


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the motion-generator tutorial."""
    parser = argparse.ArgumentParser(
        description="Generate and replay MotionGenerator trajectories for one or more environments."
    )
    add_env_launcher_args_to_parser(parser)
    parser.add_argument(
        "--arena-space",
        type=float,
        default=DEFAULT_ARENA_SPACE,
        help="Spacing between replicated tutorial environments.",
    )
    parser.add_argument(
        "--step-delay",
        type=float,
        default=0.1,
        help="Seconds to wait between trajectory waypoints during playback.",
    )
    parser.add_argument(
        "--record-fps",
        type=int,
        default=20,
        help="Output video FPS for headless recording.",
    )
    parser.add_argument(
        "--record-save-path",
        type=str,
        default=None,
        help="Optional mp4 output path for headless recording.",
    )
    parser.add_argument(
        "--disable-record",
        action="store_true",
        help="Disable automatic whole-scene recording in headless mode.",
    )
    return parser.parse_args()


def compute_record_look_at(
    num_envs: int,
    arena_space: float,
) -> tuple[
    tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]
]:
    """Return a fixed camera pose that frames the full replicated arena grid."""
    if num_envs <= 0:
        raise ValueError(f"num_envs must be positive, got {num_envs}.")

    scene_grid_length = int(np.ceil(np.sqrt(num_envs)))
    scene_grid_rows = int(np.ceil(num_envs / scene_grid_length))
    span_x = float(max(scene_grid_length - 1, 0) * arena_space)
    span_y = float(max(scene_grid_rows - 1, 0) * arena_space)
    scene_extent = 0.5 * max(span_x, span_y)

    target = (
        0.5 * span_x,
        0.5 * span_y,
        DEFAULT_RECORD_TARGET_Z,
    )
    eye = (
        2.6 + target[0] + scene_extent,
        -2.2 - scene_extent,
        1.6 + 0.4 * scene_extent,
    )
    return eye, target, (0.0, 0.0, 1.0)


def move_robot_along_trajectory(
    sim: SimulationManager,
    robot: Robot,
    arm_name: str,
    qpos_trajectory: torch.Tensor | Sequence[torch.Tensor],
) -> None:
    """Play back a planned joint trajectory for one or more environments.

    This function assumes the simulation is in manual-update mode and calls
    :meth:`SimulationManager.update` after each waypoint so physics advances.

    Args:
        sim: Simulation manager instance.
        robot: Robot instance.
        arm_name: Name of the robot arm to control.
        qpos_trajectory: Joint positions shaped ``(B, N, DOF)``, ``(N, DOF)``,
            or a sequence of waypoint tensors.
        delay: Time delay between each step in seconds.
    """
    if isinstance(qpos_trajectory, Sequence):
        qpos_steps = list(qpos_trajectory)
        if not qpos_steps:
            return
        if qpos_steps[0].dim() == 1:
            qpos_trajectory = torch.stack(qpos_steps, dim=0).unsqueeze(0)
        else:
            qpos_trajectory = torch.stack(qpos_steps, dim=1)
    if qpos_trajectory.dim() == 2:
        qpos_trajectory = qpos_trajectory.unsqueeze(0)
    if qpos_trajectory.dim() != 3:
        raise ValueError(
            "qpos_trajectory must have shape (B, N, DOF) or (N, DOF), "
            f"got {tuple(qpos_trajectory.shape)}."
        )

    joint_ids = robot.get_joint_ids(arm_name)
    for qpos_step in qpos_trajectory.transpose(0, 1):
        robot.set_qpos(qpos=qpos_step, joint_ids=joint_ids)
        sim.update(step=4)


def create_demo_trajectory(
    robot: Robot,
    arm_name: str,
    num_envs: int = 1,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Generate a three-point batched demo trajectory for the requested env count."""
    if num_envs <= 0:
        raise ValueError(f"num_envs must be positive, got {num_envs}.")

    tensor_kwargs = {"dtype": torch.float32}
    robot_device = getattr(robot, "device", None)
    if isinstance(robot_device, (str, torch.device)):
        tensor_kwargs["device"] = robot_device

    qpos_fk = torch.tensor(
        [[0.0, np.pi / 4, -np.pi / 4, 0.0, np.pi / 4, 0.0]],
        **tensor_kwargs,
    ).repeat(num_envs, 1)
    xpos_begin = robot.compute_fk(name=arm_name, qpos=qpos_fk, to_matrix=True)
    xpos_mid = xpos_begin.clone()
    xpos_mid[:, 2, 3] -= 0.1
    xpos_final = xpos_mid.clone()
    xpos_final[:, 0, 3] += 0.2

    qpos_begin = robot.compute_ik(pose=xpos_begin, name=arm_name)[1]
    qpos_mid = robot.compute_ik(pose=xpos_mid, name=arm_name)[1]
    qpos_final = robot.compute_ik(pose=xpos_final, name=arm_name)[1]
    return [qpos_begin, qpos_mid, qpos_final], [xpos_begin, xpos_mid, xpos_final]


def start_headless_recording(
    sim: SimulationManager,
    args: argparse.Namespace,
) -> bool:
    """Start headless viewer recording with a whole-scene camera."""
    if not args.headless or args.disable_record:
        return False

    look_at = compute_record_look_at(
        num_envs=sim.num_envs,
        arena_space=sim.sim_config.arena_space,
    )
    if not sim.start_window_record(
        save_path=args.record_save_path,
        fps=args.record_fps,
        max_memory=DEFAULT_RECORD_MAX_MEMORY,
        video_prefix="motion_generator_headless",
        look_at=look_at,
        use_sim_time=False,
    ):
        raise RuntimeError("Failed to start headless recording")

    print("[INFO]: Headless recording enabled.")
    print(
        "[INFO]: The output path is reported by `SimulationManager.start_window_record()`."
    )
    return True


def main() -> None:
    """Run the motion-generator tutorial."""
    args = parse_args()

    np.set_printoptions(precision=5, suppress=True)
    torch.set_printoptions(precision=5, sci_mode=False)

    sim = SimulationManager(
        SimulationManagerCfg(
            width=RECORD_WIDTH,
            height=RECORD_HEIGHT,
            headless=True,
            physics_dt=1.0 / 100.0,
            sim_device=args.device,
            render_cfg=RenderCfg(renderer=args.renderer),
            num_envs=args.num_envs,
            arena_space=args.arena_space,
        )
    )

    robot: Robot = sim.add_robot(cfg=CobotMagicCfg.from_dict({"uid": "CobotMagic"}))
    arm_name = "left_arm"

    if sim.is_use_gpu_physics:
        sim.init_gpu_physics()

    if not args.headless:
        sim.open_window()

    print(
        f"[INFO]: Running motion generator tutorial with {sim.num_envs} environment(s)"
    )

    recording_started = start_headless_recording(sim, args)
    try:
        qpos_list, xpos_list = create_demo_trajectory(
            robot=robot,
            arm_name=arm_name,
            num_envs=sim.num_envs,
        )

        motion_generator = MotionGenerator(
            cfg=MotionGenCfg(
                planner_cfg=ToppraPlannerCfg(
                    robot_uid=robot.uid,
                )
            )
        )

        options = MotionGenOptions(
            control_part=arm_name,
            start_qpos=qpos_list[0],
            is_interpolate=True,
            is_linear=False,
            plan_opts=ToppraPlanOptions(
                constraints={
                    "velocity": 0.2,
                    "acceleration": 0.5,
                },
                sample_method=TrajectorySampleMethod.QUANTITY,
                sample_interval=20,
            ),
        )

        joint_plan = motion_generator.generate(
            target_states=[PlanState.from_qpos(qpos) for qpos in qpos_list],
            options=options,
        )
        if joint_plan.positions is None:
            raise RuntimeError("Joint-space planning did not produce any positions.")
        move_robot_along_trajectory(
            sim=sim,
            robot=robot,
            arm_name=arm_name,
            qpos_trajectory=joint_plan.positions,
        )

        options.is_linear = True
        cartesian_plan = motion_generator.generate(
            target_states=[PlanState.from_xpos(xpos) for xpos in xpos_list],
            options=options,
        )
        if cartesian_plan.positions is None:
            raise RuntimeError(
                "Cartesian-space planning did not produce any positions."
            )
        sim.reset()
        move_robot_along_trajectory(
            sim=sim,
            robot=robot,
            arm_name=arm_name,
            qpos_trajectory=cartesian_plan.positions,
        )
    finally:
        if sim.is_window_recording():
            sim.stop_window_record()
            sim.wait_window_record_saves()
        sim.destroy()


if __name__ == "__main__":
    main()
