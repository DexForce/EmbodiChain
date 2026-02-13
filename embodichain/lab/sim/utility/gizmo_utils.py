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
Gizmo utility functions for EmbodiSim.

This module provides utility functions for creating gizmo transform callbacks.
"""

from typing import Callable
from typing import TYPE_CHECKING
from dexsim.types import TransformMask

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import Robot


def create_gizmo_callback() -> Callable:
    """Create a standard gizmo transform callback function.

    This callback handles basic translation and rotation operations for gizmo controls.
    It applies transformations directly to the node when gizmo controls are manipulated.

    Returns:
        Callable: A callback function that can be used with gizmo.node.set_flush_transform_callback()
    """

    def gizmo_transform_callback(node, translation, rotation, flag):
        if node is not None:
            if flag == (TransformMask.TRANSFORM_LOCAL | TransformMask.TRANSFORM_T):
                # Handle translation changes
                node.set_translation(translation)
            elif flag == (TransformMask.TRANSFORM_LOCAL | TransformMask.TRANSFORM_R):
                # Handle rotation changes
                node.set_rotation_rpy(rotation)

    return gizmo_transform_callback


def run_gizmo_robot_control_loop(
    robot: "Robot", control_part: str = "arm", end_link_name: str | None = None
):
    """Run a control loop for testing gizmo controls on a robot.

    This function implements a control loop that allows users to manipulate a robot
    using gizmo controls with keyboard input for additional commands.

    Args:
        robot (Robot): The robot to control with the gizmo.
        control_part (str, optional): The part of the robot to control. Defaults to "arm".
        end_link_name (str | None, optional): The name of the end link for FK calculations. Defaults to None.

    Keyboard Controls:
        Q/ESC: Exit the control loop
        P: Print current robot state (joint positions, end-effector pose)
        G: Toggle gizmo visibility
        R: Reset robot to initial pose
        I: Print control information
    """
    import select
    import sys
    import tty
    import termios
    import time
    import numpy as np

    np.set_printoptions(precision=5, suppress=True)

    from embodichain.lab.sim import SimulationManager
    from embodichain.lab.sim.objects import Robot
    from embodichain.lab.sim.solvers import PinkSolverCfg

    from embodichain.utils.logger import log_info, log_warning, log_error

    sim = SimulationManager.get_instance()

    # Enter auto-update mode.
    sim.set_manual_update(False)

    # Replace robot's default solver with PinkSolver for gizmo control.
    robot_solver = robot.get_solver(name=control_part)
    control_part_link_names = robot.get_control_part_link_names(name=control_part)
    end_link_name = (
        control_part_link_names[-1] if end_link_name is None else end_link_name
    )
    pink_solver_cfg = PinkSolverCfg(
        urdf_path=robot.cfg.fpath,
        end_link_name=end_link_name,
        root_link_name=robot_solver.root_link_name,
        pos_eps=1e-2,
        rot_eps=5e-2,
        max_iterations=300,
        dt=0.1,
    )
    robot.init_solver(cfg={control_part: pink_solver_cfg})

    # Enable gizmo for the robot
    gizmo = sim.enable_gizmo(uid=robot.uid, control_part=control_part)

    # Store initial robot configuration
    initial_qpos = robot.get_qpos(name=control_part)

    gizmo_visible = True

    log_info("\n=== Gizmo Robot Control ===")
    log_info("Gizmo Controls:")
    log_info("  Use the 3D gizmo to drag and manipulate the robot")
    log_info("\nKeyboard Controls:")
    log_info("  Q/ESC: Exit control loop")
    log_info("  P: Print current robot state")
    log_info("  G: Toggle gizmo visibility")
    log_info("  R: Reset robot to initial pose")
    log_info("  I: Print this information again")

    # Save terminal settings
    old_settings = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())

    def get_key():
        """Non-blocking keyboard input."""
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None

    try:
        while True:
            time.sleep(0.033)  # ~30Hz
            sim.update_gizmos()

            # Check for keyboard input
            key = get_key()

            if key:
                # Exit controls
                if key in ["q", "Q", "\x1b"]:  # Q or ESC
                    log_info("Exiting gizmo control loop...")
                    sim.disable_gizmo(uid=robot.uid, control_part=control_part)
                    if robot_solver:
                        robot.init_solver(
                            cfg={control_part: robot_solver.cfg}
                        )  # Restore original solver
                    break

                # Print robot state
                elif key in ["p", "P"]:
                    current_qpos = robot.get_qpos(name=control_part)
                    eef_pose = robot.compute_fk(name=control_part, qpos=current_qpos)
                    log_info(f"\n=== Robot State ===")
                    log_info(f"Control part: {control_part}")
                    log_info(f"Joint positions: {current_qpos.squeeze().tolist()}")
                    log_info(f"End-effector pose:\n{eef_pose.squeeze().numpy()}")

                    if eef_pose is None:
                        log_info(
                            "End-effector pose unavailable: compute_fk returned None "
                            f"for control part '{control_part}'."
                        )
                    else:
                        eef_pose_np = eef_pose.detach().cpu().numpy().squeeze()
                        log_info(f"End-effector pose:\n{eef_pose_np}")
                elif key in ["g", "G"]:
                    if gizmo_visible:
                        sim.set_gizmo_visibility(
                            uid=robot.uid, control_part=control_part, visible=False
                        )
                        log_info("Gizmo hidden")
                        gizmo_visible = False
                    else:
                        sim.set_gizmo_visibility(
                            uid=robot.uid, control_part=control_part, visible=True
                        )
                        log_info("Gizmo shown")
                        gizmo_visible = True

                # Reset to initial pose
                elif key in ["r", "R"]:
                    # TODO: Workaround for reset. Gizmo pose should be fixed in the future.
                    sim.disable_gizmo(uid=robot.uid, control_part=control_part)
                    robot.clear_dynamics()
                    robot.set_qpos(qpos=initial_qpos, name=control_part, target=False)
                    sim.enable_gizmo(uid=robot.uid, control_part=control_part)
                    log_info("Robot reset to initial pose")

                # Print info
                elif key in ["i", "I"]:
                    log_info("\n=== Gizmo Robot Control ===")
                    log_info("Gizmo Controls:")
                    log_info("  Use the 3D gizmo to drag and manipulate the robot")
                    log_info("\nKeyboard Controls:")
                    log_info("  Q/ESC: Exit control loop")
                    log_info("  P: Print current robot state")
                    log_info("  G: Toggle gizmo visibility")
                    log_info("  R: Reset robot to initial pose")
                    log_info("  I: Print this information again")

    except KeyboardInterrupt:
        sim.disable_gizmo(uid=robot.uid, control_part=control_part)
        if robot_solver:
            robot.init_solver(
                cfg={control_part: robot_solver.cfg}
            )  # Restore original solver
        log_info("\nControl loop interrupted by user (Ctrl+C)")

    finally:
        try:
            # Restore terminal settings
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        except:
            pass
        log_info("Gizmo control loop terminated")
