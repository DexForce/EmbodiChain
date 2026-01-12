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
This script demonstrates how to use the Gizmo class for interactive camera control.
It shows how to create a gizmo attached to a camera for real-time pose manipulation.
"""

import argparse
import cv2
import numpy as np
import time
import torch

torch.set_printoptions(precision=4, sci_mode=False)

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.sensors import Camera, CameraCfg
from embodichain.lab.sim.cfg import RigidObjectCfg, RigidBodyAttributesCfg
from embodichain.lab.sim.shapes import CubeCfg
from embodichain.utils import logger


def main():
    """Main function to demonstrate camera gizmo manipulation."""

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Create and simulate a camera with gizmo in SimulationManager"
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
        "--enable_rt",
        action="store_true",
        default=False,
        help="Enable ray tracing for better visuals",
    )
    args = parser.parse_args()

    # Configure the simulation
    sim_cfg = SimulationManagerCfg(
        width=1920,
        height=1080,
        physics_dt=1.0 / 100.0,
        sim_device=args.device,
        enable_rt=args.enable_rt,
    )

    # Create simulation context
    sim = SimulationManager(sim_cfg)
    sim.set_manual_update(False)

    # Add some objects to the scene for camera to observe
    for i in range(5):
        cube_cfg = RigidObjectCfg(
            uid=f"cube_{i}",
            shape=CubeCfg(size=[0.1, 0.1, 0.1]),
            body_type="dynamic",
            attrs=RigidBodyAttributesCfg(
                mass=1.0,
                dynamic_friction=0.5,
                static_friction=0.5,
                restitution=0.3,
            ),
            init_pos=[0.5 + i * 0.3, 0.0, 0.5],
        )
        sim.add_rigid_object(cfg=cube_cfg)

    # Create camera configuration
    camera_cfg = CameraCfg(
        uid="gizmo_camera",
        width=640,
        height=480,
        intrinsics=(320, 320, 320, 240),  # fx, fy, cx, cy
        near=0.1,
        far=10.0,
        enable_color=True,
        enable_depth=True,
        extrinsics=CameraCfg.ExtrinsicsCfg(
            eye=(2.0, 2.0, 2.0),
            target=(0.0, 0.0, 0.0),
            up=(0.0, 0.0, 1.0),
        ),
    )

    # Add camera to simulation
    camera = sim.add_sensor(sensor_cfg=camera_cfg)

    # Wait for initialization
    time.sleep(0.2)

    # Enable gizmo for interactive camera control using the new unified API
    sim.enable_gizmo(uid="gizmo_camera")
    if not sim.has_gizmo("gizmo_camera"):
        logger.log_error("Failed to enable gizmo for camera!")
        return

    # Open simulation window (if not headless)
    if not args.headless:
        sim.open_window()

    logger.log_info("Gizmo-Camera tutorial started!")
    logger.log_info(
        "Use the gizmo to interactively control the camera position and orientation"
    )
    logger.log_info(
        "The camera will follow the gizmo pose for dynamic viewpoint control"
    )
    logger.log_info("Press Ctrl+C to stop the simulation")

    # Run simulation loop
    run_simulation(sim, camera)


def run_simulation(sim, camera):
    """Run the simulation loop with gizmo updates."""
    step_count = 0
    last_time = time.time()
    last_step = 0

    logger.log_info("Camera view window will open. Press Ctrl+C or 'q' to exit")
    logger.log_info(
        "Use the gizmo in the 3D view to control camera position and orientation"
    )

    try:
        while True:
            # Update all gizmos managed by sim (including camera gizmo)
            sim.update_gizmos()

            # Update camera to get latest sensor data
            camera.update()

            # Refresh camera data if method available
            if hasattr(camera, "refresh"):
                camera.refresh()

            step_count += 1

            # Display camera view in separate window
            if step_count % 5 == 0:  # Update display every 5 steps for performance
                data = camera.get_data()
                if "color" in data:
                    # Get RGB image and convert for OpenCV display
                    rgb_image = data["color"].cpu().numpy()[0, :, :, :3]  # (H, W, 3)
                    # Convert RGB to BGR for OpenCV
                    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

                # Add text overlay
                cv2.putText(
                    bgr_image,
                    "Press 'h' to toggle camera gizmo visibility",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

                # Display the image
                cv2.imshow("Gizmo Camera View", bgr_image)

                # Check for key press
                key = cv2.waitKey(1) & 0xFF
                if key == ord("h"):
                    # Toggle the camera gizmo visibility using SimulationManager API
                    sim.toggle_gizmo_visibility("gizmo_camera")

            # Example: Destroy gizmo after certain steps to test cleanup
            if step_count == 30000 and sim.has_gizmo("gizmo_camera"):
                logger.log_info("Disabling gizmo at step 30000 (demonstration)")
                sim.disable_gizmo("gizmo_camera")

            # Print simulation statistics and camera info
            if step_count % 1000 == 0:
                current_time = time.time()
                elapsed = current_time - last_time
                fps = (
                    sim.num_envs * (step_count - last_step) / elapsed
                    if elapsed > 0
                    else 0
                )

                # Get camera pose for debugging
                if sim.has_gizmo("gizmo_camera"):
                    camera_pose = camera.get_local_pose(to_matrix=True)[0]
                    camera_pos = camera_pose[:3, 3]
                    logger.log_info(
                        f"Step: {step_count}, FPS: {fps:.2f}, Camera pos: [{camera_pos[0]:.2f}, {camera_pos[1]:.2f}, {camera_pos[2]:.2f}]"
                    )
                else:
                    logger.log_info(f"Step: {step_count}, FPS: {fps:.2f}")

                last_time = current_time
                last_step = step_count

    except KeyboardInterrupt:
        logger.log_info("\nStopping simulation...")
    finally:
        # Clean up resources
        cv2.destroyAllWindows()
        # Disable gizmo if it exists
        if sim.has_gizmo("gizmo_camera"):
            sim.disable_gizmo("gizmo_camera")
        sim.destroy()
        logger.log_info("Simulation terminated successfully")


if __name__ == "__main__":
    main()
