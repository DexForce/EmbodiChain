# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
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

import argparse
import cv2
import numpy as np
import time
import torch

torch.set_printoptions(precision=4, sci_mode=False)

from scipy.spatial.transform import Rotation as R

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
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


def mask_to_color_map(mask, user_ids, fix_seed=True):
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
    parser.add_argument(
        "--num_envs", type=int, default=1, help="Number of environments to simulate"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run simulation on",
    )
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    parser.add_argument(
        "--enable_rt", action="store_true", help="Enable ray tracing rendering"
    )
    parser.add_argument(
        "--attach_sensor",
        action="store_true",
        help="Attach sensor to robot end-effector",
    )
    args = parser.parse_args()

    # Initialize simulation
    print("Creating simulation...")
    config = SimulationManagerCfg(
        headless=True,
        sim_device=args.device,
        arena_space=3.0,
        enable_rt=args.enable_rt,
        physics_dt=1.0 / 100.0,
        num_envs=args.num_envs,
    )
    sim = SimulationManager(config)

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
    if sim.is_use_gpu_physics:
        sim.init_gpu_physics()

    # Open visualization window if not headless
    if not args.headless:
        sim.open_window()

    # Run simulation loop
    run_simulation(sim, robot, sensor)


def create_sensor(sim: SimulationManager, args):
    # intrinsics params
    intrinsics = (600, 600, 320.0, 240.0)
    width = 640
    height = 480

    # extrinsics params
    pos = [0.09, 0.05, 0.04]
    quat = R.from_euler("xyz", [-35, 135, 0], degrees=True).as_quat().tolist()

    # If attach_sensor is True, attach to robot end-effector; otherwise, place it in the scene
    if args.attach_sensor:
        parent = "ee_link"
    else:
        parent = None
        pos = [1.2, -0.2, 1.5]
        quat = R.from_euler("xyz", [0, 180, 0], degrees=True).as_quat().tolist()
        quat = [quat[3], quat[0], quat[1], quat[2]]  # Convert to (w, x, y, z)

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
            "JOINT[1-6]",  # Matches JOINT1, JOINT2, ..., JOINT6
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
            stiffness={"JOINT[1-6]": 1e4, "LEFT_.*": 1e3},
            damping={"JOINT[1-6]": 1e3, "LEFT_.*": 1e2},
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
    depth = data["depth"].squeeze_().cpu().numpy()  # (H, W)
    mask = data["mask"].squeeze_().cpu().numpy()  # (H, W)
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


def run_simulation(sim: SimulationManager, robot: Robot, camera: Camera):
    """Run the simulation loop with robot and camera sensor control."""

    print("Starting simulation...")
    print("Robot will move through different poses")
    print("Press Ctrl+C to stop")

    step_count = 0

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

    try:
        while True:
            # Update physics
            sim.update(step=1)

            if step_count % 1001 == 0:
                robot.set_qpos(qpos=arm_position1, joint_ids=arm_joint_ids)
                print(f"Moving to arm position 1")

                # Refresh and get image from sensor
                get_sensor_image(camera)

            if step_count % 2003 == 0:
                robot.set_qpos(qpos=arm_position2, joint_ids=arm_joint_ids)
                print(f"Moving to arm position 2")

                # Refresh and get image from sensor
                get_sensor_image(camera)

            step_count += 1

    except KeyboardInterrupt:
        print("Stopping simulation...")
    finally:
        print("Cleaning up...")
        sim.destroy()


if __name__ == "__main__":
    main()
