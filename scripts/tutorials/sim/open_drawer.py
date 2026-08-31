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

"""Use a Franka Panda and MotionGenerator to open a passive drawer."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import torch

from embodichain.data import get_data_path
from embodichain.lab.gym.utils.gym_utils import add_env_launcher_args_to_parser
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import (
    ArticulationCfg,
    JointDrivePropertiesCfg,
    NewtonPhysicsCfg,
    RenderCfg,
    RigidBodyAttributesCfg,
    physics_cfg_for_backend,
)
from embodichain.lab.sim.objects import Articulation, Robot
from embodichain.lab.sim.planners import (
    MotionGenCfg,
    MotionGenerator,
    MotionGenOptions,
    PlanState,
    ToppraPlannerCfg,
    ToppraPlanOptions,
    TrajectorySampleMethod,
)
from embodichain.lab.sim.robots import FrankaPandaCfg
from embodichain.lab.visualization import visualization_cfg_from_args

__all__ = [
    "create_scene",
    "generate_arm_trajectory",
    "get_handle_grasp_pose",
    "main",
    "move_gripper",
    "open_drawer",
    "play_arm_trajectory",
    "solve_ik_waypoints",
]

ARM_NAME = "arm"
HAND_NAME = "hand"
HANDLE_LINK_NAME = "handle_xpos"
DRAWER_ASSET = "SlidingBoxDrawer/SlidingBoxDrawer.urdf"

APPROACH_DISTANCE = 0.10
PULL_DISTANCE = 0.16
NEWTON_PULL_DISTANCE = 0.20
NEWTON_PUSH_DISTANCE_SCALE = 0.4
DRAWER_SUCCESS_THRESHOLD = 0.10
NEWTON_DRAWER_SUCCESS_THRESHOLD = 0.04
HALF_OPEN_FRACTION = 0.5
HALF_OPEN_TOLERANCE = 0.02
NEWTON_HALF_OPEN_TOLERANCE = 0.04
RECORD_WIDTH = 1280
RECORD_HEIGHT = 720
RECORD_LOOK_AT = (
    (-0.72, -1.05, 1.0),
    (0.45, 0.0, 0.52),
    (0.0, 0.0, 1.0),
)


def create_scene(sim: SimulationManager) -> tuple[Robot, Articulation]:
    """Add a Franka Panda and a passive sliding drawer to the scene.

    Args:
        sim: Simulation manager that owns the scene.

    Returns:
        The Franka robot and drawer articulation.

    Raises:
        RuntimeError: If the robot could not be added.
    """
    # Add the existing Franka configuration. Higher contact friction helps the
    # fingertips retain the narrow drawer handle during the pull phase.
    robot_cfg = FrankaPandaCfg.from_dict(
        {
            "uid": "tutorial_franka",
            "robot_type": "panda",
            "attrs": {
                "static_friction": 1.0,
                "dynamic_friction": 1.0,
            },
        }
    )
    if sim.is_newton_backend:
        robot_cfg.joint_drive_props.damping["fr3_finger_joint[1-2]"] = 10.0
    robot = sim.add_robot(cfg=robot_cfg)
    if robot is None:
        raise RuntimeError("Failed to add the Franka Panda robot.")

    # Keep the drawer base fixed while leaving its prismatic joint passive. The
    # 180-degree yaw makes the drawer's opening direction point toward Franka.
    drawer = sim.add_articulation(
        cfg=ArticulationCfg(
            uid="drawer",
            fpath=get_data_path(DRAWER_ASSET),
            asset_physics_mode="overlay",
            init_pos=(0.72, 0.0, 0.42),
            init_rot=(0.0, 0.0, 180.0),
            fix_base=True,
            joint_drive_props=JointDrivePropertiesCfg(drive_type="none"),
            attrs=RigidBodyAttributesCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
        )
    )
    return robot, drawer


def solve_ik_waypoints(
    robot: Robot,
    target_poses: Sequence[torch.Tensor],
    start_qpos: torch.Tensor,
) -> list[torch.Tensor]:
    """Solve sparse Cartesian waypoints with the previous solution as the seed.

    Args:
        robot: Robot whose arm solver is used.
        target_poses: Batched target TCP poses, each shaped ``(B, 4, 4)``.
        start_qpos: Batched initial arm positions shaped ``(B, arm_dof)``.

    Returns:
        Batched arm-joint waypoints in the same order as ``target_poses``.

    Raises:
        RuntimeError: If IK fails for any environment.
    """
    qpos_seed = start_qpos
    qpos_waypoints: list[torch.Tensor] = []
    for waypoint_index, target_pose in enumerate(target_poses):
        success, qpos = robot.compute_ik(
            pose=target_pose,
            joint_seed=qpos_seed,
            name=ARM_NAME,
        )
        failed_env_ids = (
            torch.nonzero(~success.bool(), as_tuple=False).flatten().cpu().tolist()
        )
        if failed_env_ids:
            raise RuntimeError(
                f"IK failed at waypoint {waypoint_index} for environments "
                f"{failed_env_ids}."
            )
        qpos_waypoints.append(qpos)
        qpos_seed = qpos
    return qpos_waypoints


def generate_arm_trajectory(
    motion_generator: MotionGenerator,
    qpos_waypoints: Sequence[torch.Tensor],
    start_qpos: torch.Tensor,
    sample_count: int,
) -> torch.Tensor:
    """Time-parameterize arm waypoints with MotionGenerator and TOPPRA.

    Args:
        motion_generator: Motion generator bound to the Franka robot.
        qpos_waypoints: Batched arm-joint targets.
        start_qpos: Batched starting arm positions.
        sample_count: Number of trajectory samples returned by TOPPRA.

    Returns:
        Joint positions shaped ``(B, sample_count, arm_dof)``.

    Raises:
        ValueError: If no target waypoint is supplied.
        RuntimeError: If trajectory generation fails.
    """
    if not qpos_waypoints:
        raise ValueError("qpos_waypoints must contain at least one target.")

    result = motion_generator.generate(
        target_states=[PlanState.from_qpos(qpos) for qpos in qpos_waypoints],
        options=MotionGenOptions(
            control_part=ARM_NAME,
            start_qpos=start_qpos,
            is_interpolate=True,
            is_linear=False,
            interpolate_nums=8,
            plan_opts=ToppraPlanOptions(
                constraints={
                    "velocity": 0.35,
                    "acceleration": 0.75,
                },
                sample_method=TrajectorySampleMethod.QUANTITY,
                sample_interval=sample_count,
            ),
        ),
    )
    if result.positions is None or not result.is_all_success():
        raise RuntimeError("MotionGenerator failed to produce an arm trajectory.")
    return result.positions


def play_arm_trajectory(
    sim: SimulationManager,
    robot: Robot,
    trajectory: torch.Tensor,
    *,
    physics_steps_per_waypoint: int = 4,
) -> None:
    """Send a planned arm trajectory to the robot's position drives.

    Args:
        sim: Simulation manager to advance.
        robot: Franka robot to control.
        trajectory: Batched joint positions shaped ``(B, N, arm_dof)``.
        physics_steps_per_waypoint: Physics updates between consecutive targets.
    """
    for qpos in trajectory.unbind(dim=1):
        robot.set_qpos(qpos=qpos, name=ARM_NAME)
        sim.update(step=physics_steps_per_waypoint)


def move_gripper(
    sim: SimulationManager,
    robot: Robot,
    target_qpos: torch.Tensor,
    *,
    num_steps: int = 40,
) -> None:
    """Interpolate the gripper from its current position to a target.

    Args:
        sim: Simulation manager to advance.
        robot: Franka robot to control.
        target_qpos: Batched gripper target shaped ``(B, hand_dof)``.
        num_steps: Number of interpolation samples.
    """
    start_qpos = robot.get_qpos(name=HAND_NAME)
    interpolation = torch.linspace(
        0.0,
        1.0,
        steps=num_steps,
        dtype=start_qpos.dtype,
        device=start_qpos.device,
    )
    for alpha in interpolation:
        robot.set_qpos(
            qpos=torch.lerp(start_qpos, target_qpos, alpha),
            name=HAND_NAME,
        )
        sim.update(step=4)


def get_handle_grasp_pose(drawer: Articulation) -> torch.Tensor:
    """Return the handle frame with the gripper rolled 90 degrees.

    The rotation is applied around the TCP's local Z axis, preserving the
    approach and pull direction while rotating the finger-closing direction.

    Args:
        drawer: Drawer articulation that owns the handle link.

    Returns:
        Batched grasp poses shaped ``(B, 4, 4)``.
    """
    grasp_pose = drawer.get_link_pose(HANDLE_LINK_NAME, to_matrix=True)
    quarter_turn_about_tcp_z = grasp_pose.new_tensor(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    grasp_pose[:, :3, :3] = grasp_pose[:, :3, :3] @ quarter_turn_about_tcp_z
    return grasp_pose


def open_drawer(
    sim: SimulationManager,
    robot: Robot,
    drawer: Articulation,
    motion_generator: MotionGenerator,
    *,
    wait_for_input: bool = True,
) -> torch.Tensor:
    """Pull the drawer open, then push it halfway closed.

    Args:
        sim: Simulation manager to advance.
        robot: Franka robot used for manipulation.
        drawer: Passive drawer articulation.
        motion_generator: Motion generator bound to ``robot``.
        wait_for_input: Whether to wait for Enter before executing trajectories.

    Returns:
        Final drawer joint positions shaped ``(B, drawer_dof)``.

    Raises:
        RuntimeError: If the drawer does not open or return halfway as expected.
    """
    hand_limits = robot.get_qpos_limits(name=HAND_NAME)
    hand_open_qpos = hand_limits[..., 1]
    hand_closed_qpos = hand_limits[..., 0]
    move_gripper(sim, robot, hand_open_qpos, num_steps=20)

    # Finish tool initialization before establishing the task's initial state.
    # This also clears any startup contact impulse from opening the fingers.
    drawer.reset()
    sim.update(step=5)

    # Roll the asset's handle frame 90 degrees around TCP Z. Its approach axis
    # stays unchanged while the fingers rotate to close vertically on the handle.
    grasp_pose = get_handle_grasp_pose(drawer)
    approach_pose = grasp_pose.clone()
    approach_pose[:, :3, 3] -= grasp_pose[:, :3, 2] * APPROACH_DISTANCE

    start_qpos = robot.get_qpos(name=ARM_NAME)
    approach_waypoints = solve_ik_waypoints(
        robot,
        target_poses=[approach_pose, grasp_pose],
        start_qpos=start_qpos,
    )
    approach_trajectory = generate_arm_trajectory(
        motion_generator,
        qpos_waypoints=approach_waypoints,
        start_qpos=start_qpos,
        sample_count=60,
    )
    if wait_for_input:
        input("[READY]: Trajectory planned. Press Enter to start execution...")
    play_arm_trajectory(sim, robot, approach_trajectory)

    # Close around the handle, then allow contacts to settle before pulling.
    move_gripper(sim, robot, hand_closed_qpos)
    sim.update(step=100 if sim.is_newton_backend else 10)

    # Re-read the live handle frame after grasping. Pulling along its -Z axis
    # follows the drawer's prismatic joint toward Franka.
    grasped_handle_pose = get_handle_grasp_pose(drawer)
    pull_pose = grasped_handle_pose.clone()
    pull_distance = NEWTON_PULL_DISTANCE if sim.is_newton_backend else PULL_DISTANCE
    pull_pose[:, :3, 3] -= grasped_handle_pose[:, :3, 2] * pull_distance

    pull_start_qpos = robot.get_qpos(name=ARM_NAME)
    pull_waypoints = solve_ik_waypoints(
        robot,
        target_poses=[pull_pose],
        start_qpos=pull_start_qpos,
    )
    pull_trajectory = generate_arm_trajectory(
        motion_generator,
        qpos_waypoints=pull_waypoints,
        start_qpos=pull_start_qpos,
        sample_count=80,
    )
    play_arm_trajectory(
        sim,
        robot,
        pull_trajectory,
        physics_steps_per_waypoint=5,
    )
    sim.update(step=50)

    pulled_opening = drawer.get_qpos()[:, 0].clone()
    print(
        "[INFO]: Drawer opening after pull (m): "
        f"{pulled_opening.detach().cpu().tolist()}",
        flush=True,
    )
    success_threshold = (
        NEWTON_DRAWER_SUCCESS_THRESHOLD
        if sim.is_newton_backend
        else DRAWER_SUCCESS_THRESHOLD
    )
    if not torch.all(pulled_opening >= success_threshold).item():
        raise RuntimeError(
            "The drawer did not open far enough through gripper contact. "
            f"Expected at least {success_threshold:.2f} m."
        )

    # Push the drawer back by half of its measured opening. Moving along the
    # handle frame's +Z axis reverses the pull while the gripper stays closed.
    half_open_target = pulled_opening * HALF_OPEN_FRACTION
    push_distance = pulled_opening - half_open_target
    if sim.is_newton_backend:
        push_distance *= NEWTON_PUSH_DISTANCE_SCALE
    pushed_handle_pose = get_handle_grasp_pose(drawer)
    push_pose = pushed_handle_pose.clone()
    push_pose[:, :3, 3] += pushed_handle_pose[:, :3, 2] * push_distance.unsqueeze(-1)

    push_start_qpos = robot.get_qpos(name=ARM_NAME)
    push_waypoints = solve_ik_waypoints(
        robot,
        target_poses=[push_pose],
        start_qpos=push_start_qpos,
    )
    push_trajectory = generate_arm_trajectory(
        motion_generator,
        qpos_waypoints=push_waypoints,
        start_qpos=push_start_qpos,
        sample_count=50,
    )
    play_arm_trajectory(
        sim,
        robot,
        push_trajectory,
        physics_steps_per_waypoint=5,
    )
    sim.update(step=50)

    drawer_qpos = drawer.get_qpos()
    final_opening = drawer_qpos[:, 0]
    print(
        "[INFO]: Drawer opening after half push (m): "
        f"{final_opening.detach().cpu().tolist()}",
        flush=True,
    )
    half_open_tolerance = (
        NEWTON_HALF_OPEN_TOLERANCE if sim.is_newton_backend else HALF_OPEN_TOLERANCE
    )
    if not torch.all(
        torch.abs(final_opening - half_open_target) <= half_open_tolerance
    ).item():
        raise RuntimeError(
            "The drawer did not return to half of its pulled opening. "
            f"Expected an error no greater than {half_open_tolerance:.2f} m."
        )
    return drawer_qpos


def main() -> None:
    """Run the Franka drawer-manipulation tutorial."""
    parser = argparse.ArgumentParser(
        description="Use a Franka Panda and MotionGenerator to open a drawer."
    )
    add_env_launcher_args_to_parser(parser)
    parser.add_argument(
        "--hold-steps",
        type=int,
        default=100,
        help="Physics steps to hold the final open-drawer pose before exiting.",
    )
    parser.add_argument(
        "--auto-start",
        action="store_true",
        help="Execute trajectories without waiting for Enter.",
    )
    parser.add_argument(
        "--record-save-path",
        type=str,
        default=None,
        help="Optional MP4 path for recording from a fixed headless camera.",
    )
    parser.add_argument(
        "--record-fps",
        type=int,
        default=30,
        help="Frames per second for headless recording.",
    )
    args = parser.parse_args()
    if args.num_envs < 1:
        parser.error("--num_envs must be at least 1")
    if args.hold_steps < 0:
        parser.error("--hold-steps must be non-negative")
    if args.record_fps < 1:
        parser.error("--record-fps must be at least 1")
    if args.record_save_path is not None and not args.headless:
        parser.error("--record-save-path requires --headless")

    # PytorchSolver samples multiple IK seeds; make the tutorial trajectory
    # reproducible across repeated runs of the same backend.
    torch.manual_seed(0)

    physics_cfg = physics_cfg_for_backend(args.physics)
    if isinstance(physics_cfg, NewtonPhysicsCfg):
        # The Franka, drawer, and their contacts need larger MuJoCo-Warp
        # constraint buffers than the lightweight scene defaults.
        physics_cfg.solver_cfg = {
            "solver_type": "mujoco_warp",
            "njmax": 8192,
            "nconmax": 8192,
        }

    sim = SimulationManager(
        SimulationManagerCfg(
            width=RECORD_WIDTH,
            height=RECORD_HEIGHT,
            headless=args.headless,
            sim_device=args.device,
            num_envs=args.num_envs,
            arena_space=args.arena_space,
            physics_dt=1.0 / 100.0,
            physics_cfg=physics_cfg,
            render_cfg=RenderCfg(renderer=args.renderer),
            visualization=visualization_cfg_from_args(args),
        )
    )

    try:
        robot, drawer = create_scene(sim)

        sim.prepare()
        if not args.headless and not args.viser:
            sim.open_window()

        sim.update(step=5)
        motion_generator = MotionGenerator(
            cfg=MotionGenCfg(
                planner_cfg=ToppraPlannerCfg(
                    robot_uid=robot.uid,
                    # Keep this small tutorial deterministic across platforms.
                    max_workers=1,
                ),
            )
        )

        if args.record_save_path is not None:
            if not sim.start_window_record(
                save_path=args.record_save_path,
                fps=args.record_fps,
                max_memory=2048,
                video_prefix="open_drawer_headless",
                look_at=RECORD_LOOK_AT,
                use_sim_time=True,
            ):
                raise RuntimeError("Failed to start headless recording.")

        print(
            f"[INFO]: Opening drawers in {sim.num_envs} environment(s).",
            flush=True,
        )
        open_drawer(
            sim,
            robot,
            drawer,
            motion_generator,
            wait_for_input=not args.auto_start,
        )
        if args.hold_steps:
            sim.update(step=args.hold_steps)
    finally:
        if sim.is_window_recording():
            sim.stop_window_record()
        sim.wait_window_record_saves()
        sim.destroy()


if __name__ == "__main__":
    main()
