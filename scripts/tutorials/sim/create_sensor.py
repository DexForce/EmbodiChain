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
This script demonstrates how to create and simulate a camera sensor attached to a robot using SimulationManager.
It shows how to configure a camera sensor, attach it to the robot's end-effector, and visualize the sensor's output during simulation.
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np
import torch

from scipy.spatial.transform import Rotation as R

from embodichain.lab.sim import SimulationManager
from embodichain.lab.sim.sensors import Camera, CameraCfg
from embodichain.lab.sim.objects import Robot
from embodichain.lab.sim.cfg import (
    JointDrivePropertiesCfg,
    RobotCfg,
    URDFCfg,
    RigidObjectCfg,
)
from embodichain.lab.sim.shapes import CubeCfg
from embodichain.data import get_data_path
from embodichain.lab.sim.utility.demo_utils import (
    DemoRecording,
    add_demo_args,
    create_default_sim,
    maybe_init_gpu_physics,
    maybe_open_window,
    resolve_demo_steps,
    run_simulation_loop,
    setup_print_options,
    shutdown_sim,
)

ACTION_SWITCH_INTERVAL = 100
ACTION_CYCLE_STEPS = 2 * ACTION_SWITCH_INTERVAL


def mask_to_color_map(mask, user_ids):
    """
    Convert instance mask into color map.
    :param mask: Instance mask map.
    :param user_ids: List of unique user IDs in the mask.
    :return: Color map.
    """
    # Create a blank RGB image
    color_map = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    # Generate deterministic colors based on user_id values
    colors = []
    for user_id in user_ids:
        # Use the user_id as seed to generate deterministic color
        np.random.seed(user_id)
        color = np.random.choice(range(256), size=3)
        colors.append(color)

    for idx, color in enumerate(colors):
        # Assign color to the instances of each class
        color_map[mask == user_ids[idx]] = color

    return color_map


def main():
    """Main function to demonstrate robot sensor simulation."""

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Create and simulate a robot in SimulationManager"
    )
    add_demo_args(parser)
    parser.add_argument(
        "--attach_sensor",
        "--attach-sensor",
        action="store_true",
        help="Attach sensor to robot end-effector",
    )
    args = parser.parse_args()
    setup_print_options()

    # Initialize simulation
    print("Creating simulation...")
    sim = create_default_sim(
        args,
        arena_space=3.0,
        num_envs=args.num_envs,
        add_default_light=False,
    )

    # Create robot configuration
    robot = create_robot(sim)

    sensor = create_sensor(sim, args)

    # Add a cube to the scene
    cube_cfg = RigidObjectCfg(
        uid="cube",
        shape=CubeCfg(size=[0.05, 0.05, 0.05]),  # Use CubeCfg for a cube
        init_pos=[1.2, -0.2, 0.1],
        init_rot=[0, 0, 0],
    )
    sim.add_rigid_object(cfg=cube_cfg)

    # Initialize GPU physics if using CUDA
    maybe_init_gpu_physics(sim)

    # Open visualization window if not headless
    maybe_open_window(sim, args)

    # Run simulation loop
    run_simulation(sim, robot, sensor, args)


def create_sensor(sim: SimulationManager, args):
    # intrinsics params
    intrinsics = (600, 600, 320.0, 240.0)
    width = 640
    height = 480

    # extrinsics params
    pos = [0.09, 0.05, 0.04]
    quat_xyzw = R.from_euler("xyz", [-35, 135, 0], degrees=True).as_quat()

    # If attach_sensor is True, attach to robot end-effector; otherwise, place it in the scene
    if args.attach_sensor:
        parent = "ee_link"
    else:
        parent = None
        pos = [1.2, -0.2, 1.5]
        quat_xyzw = R.from_euler("xyz", [0, 180, 0], degrees=True).as_quat()

    # CameraCfg uses (w, x, y, z), while SciPy returns (x, y, z, w).
    quat = [quat_xyzw[3], *quat_xyzw[:3]]

    # create camera sensor and attach to robot end-effector
    camera: Camera = sim.add_sensor(
        sensor_cfg=CameraCfg(
            width=width,
            height=height,
            intrinsics=intrinsics,
            extrinsics=CameraCfg.ExtrinsicsCfg(
                parent=parent,
                pos=pos,
                quat=quat,
            ),
            near=0.01,
            far=10.0,
            enable_color=True,
            enable_depth=True,
            enable_mask=True,
            enable_normal=True,
        )
    )
    return camera


def create_robot(sim):
    """Create and configure a robot in the simulation."""

    print("Loading robot...")

    # Get SR5 URDF path
    sr5_urdf_path = get_data_path("Rokae/SR5/SR5.urdf")

    # Get hand URDF path
    hand_urdf_path = get_data_path(
        "BrainCoHandRevo1/BrainCoLeftHand/BrainCoLeftHand.urdf"
    )

    # Define control parts for the robot
    # Joint names in control_parts can be regex patterns
    CONTROL_PARTS = {
        "arm": [
            "joint[1-6]",  # Matches JOINT1, JOINT2, ..., JOINT6
        ],
        "hand": ["LEFT_.*"],  # Matches all joints starting with L_
    }

    # Define transformation for hand attachment
    hand_attach_xpos = np.eye(4)
    hand_attach_xpos[:3, :3] = R.from_rotvec([90, 0, 0], degrees=True).as_matrix()
    hand_attach_xpos[2, 3] = 0.02

    cfg = RobotCfg(
        uid="sr5_with_brainco",
        urdf_cfg=URDFCfg(
            components=[
                {
                    "component_type": "arm",
                    "urdf_path": sr5_urdf_path,
                },
                {
                    "component_type": "hand",
                    "urdf_path": hand_urdf_path,
                    "transform": hand_attach_xpos,
                },
            ]
        ),
        control_parts=CONTROL_PARTS,
        drive_pros=JointDrivePropertiesCfg(
            stiffness={"joint[1-6]": 1e4, "LEFT_.*": 1e3},
            damping={"joint[1-6]": 1e3, "LEFT_.*": 1e2},
        ),
    )

    # Add robot to simulation
    robot: Robot = sim.add_robot(cfg=cfg)

    print(f"Robot created successfully with {robot.dof} joints")

    return robot


def get_sensor_image(camera: Camera, headless=False, step_count=0):
    """
    Get color, depth, mask, and normals views from the camera,
    and visualize them in a 2x2 grid (or save if headless).
    """
    import matplotlib.pyplot as plt

    camera.update()
    data = camera.get_data()
    # Get four views
    rgba = data["color"].cpu().numpy()[0, :, :, :3]  # (H, W, 3)
    depth = data["depth"].squeeze().cpu().numpy()  # (H, W)
    mask = data["mask"].squeeze().cpu().numpy()  # (H, W)
    normals = data["normal"].cpu().numpy()[0]  # (H, W, 3)

    # Normalize for visualization
    depth_vis = (depth - depth.min()) / (np.ptp(depth) + 1e-8)
    depth_vis = (depth_vis * 255).astype("uint8")
    mask_vis = mask_to_color_map(mask, user_ids=np.unique(mask))
    normals_vis = ((normals + 1) / 2 * 255).astype("uint8")

    # Prepare titles and images for display
    titles = ["Color", "Depth", "Mask", "Normals"]
    images = [
        cv2.cvtColor(rgba, cv2.COLOR_RGB2BGR),
        cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR),
        mask_vis,
        cv2.cvtColor(normals_vis, cv2.COLOR_RGB2BGR),
    ]

    if not headless:
        # Concatenate images for 2x2 grid display using OpenCV
        top = np.hstack([images[0], images[1]])
        bottom = np.hstack([images[2], images[3]])
        grid = np.vstack([top, bottom])
        cv2.imshow("Sensor Views (Color / Depth / Mask / Normals)", grid)
        cv2.waitKey(1)
    else:
        # Save the 2x2 grid as an image using matplotlib
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        for ax, img, title in zip(axs.flatten(), images, titles):
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.set_title(title)
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(f"sensor_views_{step_count}.png")
        plt.close(fig)


def run_simulation(
    sim: SimulationManager,
    robot: Robot,
    camera: Camera,
    args: argparse.Namespace,
) -> None:
    """Run the simulation loop with robot and camera sensor control."""

    print("Starting simulation...")
    print("Robot will move through different poses")
    print("Press Ctrl+C to stop")

    arm_joint_ids = robot.get_joint_ids("arm")
    # Define some target joint positions for demonstration

    arm_position1 = (
        torch.tensor(
            [0.0, 0.5, -1.5, 0.3, -0.5, 0], dtype=torch.float32, device=sim.device
        )
        .unsqueeze_(0)
        .repeat(sim.num_envs, 1)
    )

    arm_position2 = (
        torch.tensor(
            [0.0, 0.5, -1.5, -0.3, -0.5, 0], dtype=torch.float32, device=sim.device
        )
        .unsqueeze_(0)
        .repeat(sim.num_envs, 1)
    )

    def update_target(step: int) -> None:
        """Move the robot and capture images at each target switch."""
        cycle_step = (step - 1) % ACTION_CYCLE_STEPS

        if cycle_step == 0:
            robot.set_qpos(qpos=arm_position1, joint_ids=arm_joint_ids)
            print("Moving to arm position 1")

            get_sensor_image(camera, headless=args.headless, step_count=step)

        if cycle_step == ACTION_SWITCH_INTERVAL:
            robot.set_qpos(qpos=arm_position2, joint_ids=arm_joint_ids)
            print("Moving to arm position 2")

            get_sensor_image(camera, headless=args.headless, step_count=step)

    try:
        with DemoRecording(sim, args, prefix="create_sensor"):
            run_simulation_loop(
                sim,
                max_steps=resolve_demo_steps(args),
                on_step=update_target,
            )
    finally:
        print("Cleaning up...")
        shutdown_sim(sim)


if __name__ == "__main__":
    main()
