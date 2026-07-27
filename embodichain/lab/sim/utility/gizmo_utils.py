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

"""Gizmo utility functions for EmbodiChain.

This module provides utility functions for creating gizmo transform callbacks.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import Robot

__all__ = ["create_gizmo_callback", "run_gizmo_robot_control_loop"]


def create_gizmo_callback() -> Callable[[Any, Any, Any], None]:
    """Create a standard gizmo transform callback function.

    This callback handles local pose for gizmo controls.
    It applies transformations directly to the node when gizmo controls are manipulated.

    Returns:
        A callback compatible with dexsim's gizmo local-pose flush hook.
    """

    def gizmo_transform_callback(node: Any, local_pose: Any, flag: Any) -> None:
        if node is not None:
            node.set_transform(local_pose, flag)

    return gizmo_transform_callback


def run_gizmo_robot_control_loop(
    robot: Robot | str,
    control_part: str = "arm",
    end_link_name: str | None = None,
) -> None:
    """Run a control loop for testing gizmo controls on a robot.

    This function implements a control loop that allows users to manipulate a robot
    using gizmo controls with keyboard input for additional commands.

    Args:
        robot (Robot | str): The robot to control with the gizmo.
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
    from embodichain.lab.sim.objects import GizmoCfg

    from embodichain.utils.logger import log_error, log_info

    sim = SimulationManager.get_instance()

    if isinstance(robot, str):
        robot_uid = robot
        robot = sim.get_robot(uid=robot_uid)
        if robot is None:
            log_error(f"Robot {robot_uid!r} was not found.")
            return

    # Enter auto-update mode.
    sim.set_manual_update(False)

    # Resolve only the chain metadata. dexsim owns the Newton IK solver and
    # writes its drive targets back through the EmbodiChain Robot API.
    robot_solver = (
        robot.get_solver(name=control_part)
        if robot.cfg.solver_cfg is not None
        else None
    )
    control_part_link_names = robot.get_control_part_link_names(name=control_part)
    if not control_part_link_names:
        raise ValueError(f"Control part {control_part!r} has no links.")
    root_link_name = (
        robot_solver.root_link_name
        if robot_solver is not None
        else control_part_link_names[0]
    )
    end_link_name = (
        (
            robot_solver.end_link_name
            if robot_solver is not None
            else control_part_link_names[-1]
        )
        if end_link_name is None
        else end_link_name
    )
    tcp_pose = robot_solver.get_tcp() if robot_solver is not None else None
    gizmo_cfg = GizmoCfg(
        ik_root_link_name=root_link_name,
        ik_end_link_name=end_link_name,
        ik_tcp_pose=tcp_pose,
    )

    # Enable gizmo for the robot
    gizmo = sim.enable_gizmo(
        uid=robot.uid,
        control_part=control_part,
        gizmo_cfg=gizmo_cfg,
    )
    if gizmo is None:
        log_error(f"Failed to enable gizmo for control part {control_part!r}.")
        return

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

    def get_key() -> str | None:
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
                    break

                # Print robot state
                elif key in ["p", "P"]:
                    current_qpos = robot.get_qpos(name=control_part)
                    eef_pose = robot.get_link_pose(end_link_name, to_matrix=True)
                    if tcp_pose is not None:
                        tcp_tensor = np.asarray(tcp_pose, dtype=np.float32)
                        eef_pose = eef_pose @ eef_pose.new_tensor(tcp_tensor)
                    log_info(f"\n=== Robot State ===")
                    log_info(f"Control part: {control_part}")
                    log_info(f"Joint positions: {current_qpos.squeeze().tolist()}")
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
                    sim.enable_gizmo(
                        uid=robot.uid,
                        control_part=control_part,
                        gizmo_cfg=gizmo_cfg,
                    )
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
        log_info("\nControl loop interrupted by user (Ctrl+C)")

    finally:
        try:
            # Restore terminal settings
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        except:
            pass
        log_info("Gizmo control loop terminated")
