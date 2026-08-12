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
Gizmo Scene Example: Interactive scene with both robot and rigid object gizmos

This example demonstrates how to create an interactive simulation scene with:
- A UR10 robot with gizmo control for end-effector manipulation
- A rigid object (cube) with gizmo control for direct manipulation
Both objects can be interactively controlled through their respective gizmos.
"""

from __future__ import annotations

import time
import torch
import numpy as np
import argparse
import cv2

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.visualization import visualization_cfg_from_args
from embodichain.lab.sim.cfg import (
    RenderCfg,
    RobotCfg,
    URDFCfg,
    JointDrivePropertiesCfg,
    RigidObjectCfg,
    RigidBodyAttributesCfg,
)
from embodichain.lab.gym.utils.gym_utils import add_env_launcher_args_to_parser
from embodichain.lab.sim.shapes import CubeCfg
from embodichain.lab.sim.sensors import CameraCfg
from embodichain.lab.sim.solvers import PinkSolverCfg
from embodichain.lab.sim.objects import create_robot_ik_gizmo_controller
from embodichain.data import get_data_path
from embodichain.utils import logger


def main():
    """Main function to create and run the simulation scene."""

    parser = argparse.ArgumentParser(
        description="Create a simulation scene with SimulationManager"
    )
    add_env_launcher_args_to_parser(parser)
    args = parser.parse_args()

    # Configure the simulation
    sim_cfg = SimulationManagerCfg(
        width=1920,
        height=1080,
        headless=True,
        physics_dt=1.0 / 100.0,
        sim_device=args.device,
        render_cfg=RenderCfg(renderer=args.renderer),
        visualization=visualization_cfg_from_args(args),
    )

    sim = SimulationManager(sim_cfg)
    sim.set_manual_update(False)

    # Get DexForce W1 URDF path
    urdf_path = get_data_path("DexforceW1V021/DexforceW1_v02_1.urdf")

    # Create DexForce W1 robot
    robot_cfg = RobotCfg(
        uid="w1_gizmo_test",
        urdf_cfg=URDFCfg(
            components=[{"component_type": "humanoid", "urdf_path": urdf_path}]
        ),
        control_parts={"left_arm": ["LEFT_J[1-7]"], "right_arm": ["RIGHT_J[1-7]"]},
        solver_cfg={
            "left_arm": PinkSolverCfg(
                urdf_path=urdf_path,
                end_link_name="left_ee",
                root_link_name="left_arm_base",
                pos_eps=1e-2,
                rot_eps=5e-2,
                max_iterations=300,
                dt=0.1,
            ),
            "right_arm": PinkSolverCfg(
                urdf_path=urdf_path,
                end_link_name="right_ee",
                root_link_name="right_arm_base",
                pos_eps=1e-2,
                rot_eps=5e-2,
                max_iterations=300,
                dt=0.1,
            ),
        },
        drive_pros=JointDrivePropertiesCfg(
            stiffness={"LEFT_J[1-7]": 1e4, "RIGHT_J[1-7]": 1e4},
            damping={"LEFT_J[1-7]": 1e3, "RIGHT_J[1-7]": 1e3},
        ),
    )
    robot = sim.add_robot(cfg=robot_cfg)

    # Set initial joint positions for both arms
    left_arm_qpos = torch.tensor(
        [
            [0, 0, -np.pi / 4, np.pi / 4, -np.pi / 2, 0.0, np.pi / 4, 0.0]
        ],  # WAIST + LEFT_J[1-7]
        dtype=torch.float32,
        device="cpu",
    )
    right_arm_qpos = torch.tensor(
        [
            [0, 0, np.pi / 4, -np.pi / 4, np.pi / 2, 0.0, -np.pi / 4, 0.0]
        ],  # WAIST + RIGHT_J[1-7]
        dtype=torch.float32,
        device="cpu",
    )

    left_joint_ids = robot.get_joint_ids("left_arm")
    right_joint_ids = robot.get_joint_ids("right_arm")

    robot.set_qpos(qpos=left_arm_qpos, joint_ids=left_joint_ids)
    robot.set_qpos(qpos=right_arm_qpos, joint_ids=right_joint_ids)

    # Create a rigid object (cube) positioned to the side of the robot
    cube_cfg = RigidObjectCfg(
        uid="interactive_cube",
        shape=CubeCfg(size=[0.1, 0.1, 0.1]),
        body_type="kinematic",
        attrs=RigidBodyAttributesCfg(
            mass=1.0,
            dynamic_friction=0.5,
            static_friction=0.5,
            restitution=0.1,
        ),
        init_pos=[1.0, 0.0, 0.5],  # Position to the side of the robot
    )
    cube = sim.add_rigid_object(cube_cfg)

    camera_cfg = CameraCfg(
        uid="scene_camera",
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
    camera = sim.add_sensor(sensor_cfg=camera_cfg)

    native_window_opened = False
    if not args.headless:
        native_window_opened = sim.open_window()

    native_controls = []
    if native_window_opened:
        sim.enable_entity_gizmo()
        for control_part in ("left_arm", "right_arm"):
            native_controls.append(
                create_robot_ik_gizmo_controller(
                    robot,
                    control_part=control_part,
                    world=sim.get_world(),
                )
            )
    elif args.viser:
        sim.enable_gizmo(
            uid="w1_gizmo_test",
            control_part="left_arm",
        )
        if not sim.has_gizmo("w1_gizmo_test", control_part="left_arm"):
            logger.log_error("Failed to enable left arm gizmo!")
            return

        sim.enable_gizmo(
            uid="w1_gizmo_test",
            control_part="right_arm",
        )
        if not sim.has_gizmo("w1_gizmo_test", control_part="right_arm"):
            logger.log_error("Failed to enable right arm gizmo!")
            return

        sim.enable_gizmo(
            uid="interactive_cube",
        )
        if not sim.has_gizmo("interactive_cube"):
            logger.log_error("Failed to enable gizmo for cube!")
            return

        sim.enable_gizmo(
            uid="scene_camera",
        )
        if not sim.has_gizmo("scene_camera"):
            logger.log_error("Failed to enable gizmo for camera!")
            return
    else:
        logger.log_warning(
            "Gizmo interaction is disabled in headless mode without Viser."
        )

    logger.log_info("Gizmo Scene example started!")
    if args.viser:
        logger.log_info("Four gizmos are active in the scene:")
        logger.log_info(
            "1. Left arm gizmo - Use to drag the left arm end-effector (EE)"
        )
        logger.log_info(
            "2. Right arm gizmo - Use to drag the right arm end-effector (EE)"
        )
        logger.log_info("3. Cube gizmo - Use to drag and position the cube")
        logger.log_info("4. Camera gizmo - Use to drag and orient the camera")
    elif native_window_opened:
        logger.log_info("Press I to show or hide each robot TCP IK Gizmo.")
        logger.log_info("Select a scene entity and press G to manipulate its root.")
    logger.log_info("Press Ctrl+C to stop the simulation")

    run_simulation(
        sim,
        native_controls=native_controls,
        show_camera_window=native_window_opened,
    )


def run_simulation(
    sim: SimulationManager,
    *,
    native_controls=(),
    show_camera_window: bool,
) -> None:
    step_count = 0
    # Get the camera instance by uid
    camera = sim.get_sensor("scene_camera")
    try:
        last_time = time.time()
        last_step = 0
        while True:
            time.sleep(0.033)  # 30Hz
            for controller, _ in native_controls:
                controller.update()
            sim.update_gizmos()
            sim.capture_visualization_safely()
            step_count += 1

            # Display camera view in a window every 5 steps
            if show_camera_window and camera is not None and step_count % 5 == 0:
                camera.update()
                data = camera.get_data()
                if "color" in data:
                    rgb_image = data["color"].cpu().numpy()[0, :, :, :3]
                    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
                    cv2.putText(
                        bgr_image,
                        "Camera sensor preview",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )
                    cv2.imshow("Camera Sensor View", bgr_image)
                    cv2.waitKey(1)

            if step_count % 100 == 0:
                current_time = time.time()
                elapsed = current_time - last_time
                fps = (
                    sim.num_envs * (step_count - last_step) / elapsed
                    if elapsed > 0
                    else 0
                )
                logger.log_info(f"Simulation step: {step_count}, FPS: {fps:.2f}")
                last_time = current_time
                last_step = step_count
    except KeyboardInterrupt:
        logger.log_info("\nStopping simulation...")
    finally:
        if show_camera_window:
            cv2.destroyAllWindows()
        sim.destroy()
        logger.log_info("Simulation terminated successfully")


if __name__ == "__main__":
    main()
