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

import select
import sys
import tty
import termios
import time
import cv2
import torch
import numpy as np

from scipy.spatial.transform import Rotation as R

from embodichain.lab.sim.sensors import Camera
from embodichain.utils.logger import log_info, log_error, log_warning


def run_keyboard_control_for_camera(
    sensor: Camera | str,
    trans_step: float = 0.01,
    rot_step: float = 1.0,
    vis_pose: bool = False,
) -> None:
    """Run keyboard control loop for camera pose adjustment.

    Args:
        sensor (Camera | str): Camera sensor or name of the camera to control.
        trans_step (float, optional): Translation step size. Defaults to 0.01.
        rot_step (float, optional): Rotation step size in degrees. Defaults to 1.0.
        vis_pose (bool, optional): Whether to visualize the camera pose in axis form. Defaults to False.
    """
    from embodichain.lab.sim import SimulationManager

    sim = SimulationManager.get_instance()

    if vis_pose and sim.is_rt_enabled:
        log_warning(
            "'vis_pose' is not fully supported with ray tracing enabled. Will be fixed in future updates."
        )
        return

    if isinstance(sensor, str):
        sensor = sim.get_sensor(uid=sensor)

    if sensor.num_instances > 1:
        log_warning(
            "Multiple sensor instances detected. Keyboard control will only work for one instance."
        )
        return

    log_info("\n=== Camera Pose Control ===")
    log_info("Translation controls:")
    log_info("  W/S: Move forward/backward (Z-axis)")
    log_info("  A/D: Move left/left (Y-axis)")
    log_info("  Q/E: Move up/down (X-axis)")
    log_info("\nRotation controls:")
    log_info("  I/K: Pitch up/down (X-rotation)")
    log_info("  J/L: Yaw left/left (Z-rotation)")
    log_info("  U/O: Roll left/left (Y-rotation)")
    log_info("\nOther controls:")
    log_info("  R: Reset to initial pose")
    log_info("  P: Print current pose")
    log_info("  ESC: Exit control mode")

    init_pose = sensor.get_local_pose(to_matrix=True).squeeze().numpy()

    marker = None
    if vis_pose:
        from embodichain.lab.sim.cfg import MarkerCfg

        init_axis_pose = sensor.get_arena_pose(to_matrix=True).squeeze().numpy()

        marker = sim.draw_marker(
            cfg=MarkerCfg(
                name="camera_axis",
                marker_type="axis",
                axis_xpos=[init_axis_pose],
                axis_size=0.002,
                axis_len=0.05,
            )
        )

        # TODO: We may add node to BatchEntity object.
        marker[0].node.attach_node(sensor._entities[0].get_node())

    try:
        while True:
            current_pose = sensor.get_local_pose(to_matrix=True).squeeze().numpy()

            sensor.update()
            image = sensor.get_data()["color"].squeeze(0).cpu().numpy()
            image_vis = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.putText(
                image_vis,
                "Press keys to control camera pose",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                image_vis,
                "ESC to exit control mode",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.imshow("cam view", image_vis)
            key = cv2.waitKey(1) & 0xFF

            if key == 255:
                continue
            elif key == 27:
                if vis_pose:
                    sim.remove_marker("camera_axis")
                log_info("Exiting keyboard control mode...")
                break

            pose_changed = False
            new_pose = current_pose.copy()

            # controlling translation
            if key == ord("w"):
                new_pose[2, 3] += trans_step
                pose_changed = True
                log_info(f"Moving forward: Z += {trans_step}")
            elif key == ord("s"):
                new_pose[2, 3] -= trans_step
                pose_changed = True
                log_info(f"Moving backward: Z -= {trans_step}")
            elif key == ord("a"):
                new_pose[1, 3] -= trans_step
                pose_changed = True
                log_info(f"Moving left: Y -= {trans_step}")
            elif key == ord("d"):
                new_pose[1, 3] += trans_step
                pose_changed = True
                log_info(f"Moving left: Y += {trans_step}")
            elif key == ord("q"):
                new_pose[0, 3] += trans_step
                pose_changed = True
                log_info(f"Moving up: X += {trans_step}")
            elif key == ord("e"):
                new_pose[0, 3] -= trans_step
                pose_changed = True
                log_info(f"Moving down: X -= {trans_step}")

            # controlling rotation
            elif key == ord("i"):
                current_rotation = R.from_matrix(new_pose[:3, :3])
                delta_rotation = R.from_euler("x", rot_step, degrees=True)
                new_rotation = delta_rotation * current_rotation
                new_pose[:3, :3] = new_rotation.as_matrix()
                pose_changed = True
                log_info(f"Pitch up: X rotation += {rot_step}°")
            elif key == ord("k"):
                current_rotation = R.from_matrix(new_pose[:3, :3])
                delta_rotation = R.from_euler("x", -rot_step, degrees=True)
                new_rotation = delta_rotation * current_rotation
                new_pose[:3, :3] = new_rotation.as_matrix()
                pose_changed = True
                log_info(f"Pitch down: X rotation -= {rot_step}°")
            elif key == ord("j"):
                current_rotation = R.from_matrix(new_pose[:3, :3])
                delta_rotation = R.from_euler("z", rot_step, degrees=True)
                new_rotation = delta_rotation * current_rotation
                new_pose[:3, :3] = new_rotation.as_matrix()
                pose_changed = True
                log_info(f"Yaw left: Z rotation += {rot_step}°")
            elif key == ord("l"):
                current_rotation = R.from_matrix(new_pose[:3, :3])
                delta_rotation = R.from_euler("z", -rot_step, degrees=True)
                new_rotation = delta_rotation * current_rotation
                new_pose[:3, :3] = new_rotation.as_matrix()
                pose_changed = True
                log_info(f"Yaw left: Z rotation -= {rot_step}°")
            elif key == ord("u"):
                current_rotation = R.from_matrix(new_pose[:3, :3])
                delta_rotation = R.from_euler("y", rot_step, degrees=True)
                new_rotation = delta_rotation * current_rotation
                new_pose[:3, :3] = new_rotation.as_matrix()
                pose_changed = True
                log_info(f"Roll left: Y rotation += {rot_step}°")
            elif key == ord("o"):
                current_rotation = R.from_matrix(new_pose[:3, :3])
                delta_rotation = R.from_euler("y", -rot_step, degrees=True)
                new_rotation = delta_rotation * current_rotation
                new_pose[:3, :3] = new_rotation.as_matrix()
                pose_changed = True
                log_info(f"Roll left: Y rotation -= {rot_step}°")

            # other controls
            elif key == ord("r"):
                new_pose = init_pose.copy()
                pose_changed = True
                log_info("Reset to initial pose")
            elif key == ord("p"):
                new_pose_print = new_pose.copy()
                if sensor.is_attached is False:
                    new_pose_print[:3, 1] = -new_pose_print[:3, 1]
                    new_pose_print[:3, 2] = -new_pose_print[:3, 2]
                translation = new_pose_print[:3, 3]
                rot = R.from_matrix(new_pose_print[:3, :3])
                quaternion = rot.as_quat()
                log_info("Current Camera pose:")
                log_info(f"Translation: {translation}")
                quat_wxyz = [quaternion[3], quaternion[0], quaternion[1], quaternion[2]]
                log_info(f"Quaternion (w, x, y, z): {quat_wxyz}")

                rotation_euler = rot.as_euler("xyz", degrees=True)
                log_info(f"Rotation (XYZ Euler, degrees): {rotation_euler}")

            if pose_changed:
                cam_pose = new_pose.copy()
                cam_pose = torch.as_tensor(cam_pose, dtype=torch.float32).unsqueeze_(0)
                sensor.set_local_pose(cam_pose)

                if vis_pose:
                    sim.update(step=1)

    except KeyboardInterrupt:
        if vis_pose:
            sim.remove_marker("camera_axis")
        log_error("Keyboard control interrupted by user. Exiting control mode...")
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception as e:
            log_warning(f"cv2.destroyAllWindows() failed: {e}")


def run_keyboard_control_for_light(
    light: object | str,
    trans_step: float = 0.01,
    intensity_step: float = 1.0,
    falloff_step: float = 1.0,
    color_step: float = 0.05,
    vis_pose: bool = False,
) -> None:
    """Run keyboard control loop for light adjustment.

    Args:
        light (Light | str): Light object or name of the light to control.
        trans_step (float, optional): Translation step size. Defaults to 0.01.
        intensity_step (float, optional): Intensity adjustment step. Defaults to 0.1.
        falloff_step (float, optional): Falloff/radius adjustment step. Defaults to 0.1.
        color_step (float, optional): Color channel adjustment step. Defaults to 0.05.
        vis_pose (bool, optional): Whether to visualize the light position with a marker. Defaults to False.
    """
    from embodichain.lab.sim.objects import Light
    from embodichain.lab.sim import SimulationManager

    sim = SimulationManager.get_instance()

    if vis_pose and sim.is_rt_enabled:
        log_warning(
            "'vis_pose' is not fully supported with ray tracing enabled. Will be fixed in future updates."
        )
        return

    if isinstance(light, str):
        light: Light = sim.get_light(uid=light)

    if light.num_instances > 1:
        log_warning(
            "Multiple light instances detected. Keyboard control will only work for one instance."
        )
        return

    log_info("\n=== Light Control ===")
    log_info("Translation controls:")
    log_info("  W/S: Move forward/backward (Z-axis)")
    log_info("  A/D: Move left/right (Y-axis)")
    log_info("  Q/E: Move up/down (X-axis)")
    log_info("\nIntensity controls:")
    log_info("  I/K: Increase/decrease intensity")
    log_info("\nFalloff controls:")
    log_info("  U/O: Increase/decrease falloff radius")
    log_info("\nColor controls:")
    log_info("  T/Y: Increase/decrease red channel")
    log_info("  G/H: Increase/decrease green channel")
    log_info("  B/N: Increase/decrease blue channel")
    log_info("\nOther controls:")
    log_info("  R: Reset to initial values")
    log_info("  P: Print current light properties")
    log_info("  ESC: Exit control mode")

    # Store initial values from config
    init_pose = light.get_local_pose()
    init_color = torch.as_tensor(light.cfg.color).clone()
    init_intensity = float(light.cfg.intensity)
    init_falloff = float(light.cfg.radius)

    # Current values
    current_color = init_color.clone()
    current_intensity = init_intensity
    current_falloff = init_falloff

    marker = None
    if vis_pose:
        from embodichain.lab.sim import SimulationManager
        from embodichain.lab.sim.cfg import MarkerCfg

        init_marker_pose = light.get_local_pose(to_matrix=True).squeeze().numpy()

        sim = SimulationManager.get_instance()
        marker = sim.draw_marker(
            cfg=MarkerCfg(
                name="light_marker",
                marker_type="axis",
                axis_xpos=[init_marker_pose],
                axis_size=0.002,
                axis_len=0.05,
            )
        )

        # TODO: We may add node to BatchEntity object.
        marker[0].node.attach_node(light._entities[0].get_node())

    log_info("\nLight control active. Press keys to adjust light properties...")

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
            current_pose = light.get_local_pose().squeeze().numpy()

            # Non-blocking key input
            key = get_key()

            if key is None:
                continue
            elif key in ["\x1b"]:  # Q or ESC
                if vis_pose:
                    sim.remove_marker("light_marker")
                log_info("Exiting light control mode...")
                break

            property_changed = False
            new_pose = current_pose.copy()

            # Translation controls
            if key in ["w", "W"]:
                new_pose[2] += trans_step
                property_changed = True
                log_info(f"Moving forward: Z += {trans_step}")
            elif key in ["s", "S"]:
                new_pose[2] -= trans_step
                property_changed = True
                log_info(f"Moving backward: Z -= {trans_step}")
            elif key in ["a", "A"]:
                new_pose[1] -= trans_step
                property_changed = True
                log_info(f"Moving left: Y -= {trans_step}")
            elif key in ["d", "D"]:
                new_pose[1] += trans_step
                property_changed = True
                log_info(f"Moving right: Y += {trans_step}")
            elif key in ["q", "Q"]:
                new_pose[0] += trans_step
                property_changed = True
                log_info(f"Moving up: X += {trans_step}")
            elif key in ["e", "E"]:
                new_pose[0] -= trans_step
                property_changed = True
                log_info(f"Moving down: X -= {trans_step}")

            # Intensity controls
            elif key in ["i", "I"]:
                current_intensity += intensity_step
                current_intensity = max(0.0, current_intensity)
                light.set_intensity(torch.tensor(current_intensity))
                property_changed = True
                log_info(f"Intensity increased to: {current_intensity:.2f}")
            elif key in ["k", "K"]:
                current_intensity -= intensity_step
                current_intensity = max(0.0, current_intensity)
                light.set_intensity(torch.tensor(current_intensity))
                property_changed = True
                log_info(f"Intensity decreased to: {current_intensity:.2f}")

            # Falloff controls
            elif key in ["u", "U"]:
                current_falloff += falloff_step
                current_falloff = max(0.0, current_falloff)
                light.set_falloff(torch.tensor(current_falloff))
                property_changed = True
                log_info(f"Falloff increased to: {current_falloff:.2f}")
            elif key in ["o", "O"]:
                current_falloff -= falloff_step
                current_falloff = max(0.0, current_falloff)
                light.set_falloff(torch.tensor(current_falloff))
                property_changed = True
                log_info(f"Falloff decreased to: {current_falloff:.2f}")

            # Color controls - Red channel
            elif key in ["t", "T"]:
                current_color[0] += color_step
                current_color[0] = torch.clamp(current_color[0], 0.0, 1.0)
                light.set_color(current_color)
                property_changed = True
                log_info(f"Red channel increased to: {current_color[0]:.2f}")
            elif key in ["y", "Y"]:
                current_color[0] -= color_step
                current_color[0] = torch.clamp(current_color[0], 0.0, 1.0)
                light.set_color(current_color)
                property_changed = True
                log_info(f"Red channel decreased to: {current_color[0]:.2f}")

            # Color controls - Green channel
            elif key in ["g", "G"]:
                current_color[1] += color_step
                current_color[1] = torch.clamp(current_color[1], 0.0, 1.0)
                light.set_color(current_color)
                property_changed = True
                log_info(f"Green channel increased to: {current_color[1]:.2f}")
            elif key in ["h", "H"]:
                current_color[1] -= color_step
                current_color[1] = torch.clamp(current_color[1], 0.0, 1.0)
                light.set_color(current_color)
                property_changed = True
                log_info(f"Green channel decreased to: {current_color[1]:.2f}")

            # Color controls - Blue channel
            elif key in ["b", "B"]:
                current_color[2] += color_step
                current_color[2] = torch.clamp(current_color[2], 0.0, 1.0)
                light.set_color(current_color)
                property_changed = True
                log_info(f"Blue channel increased to: {current_color[2]:.2f}")
            elif key in ["n", "N"]:
                current_color[2] -= color_step
                current_color[2] = torch.clamp(current_color[2], 0.0, 1.0)
                light.set_color(current_color)
                property_changed = True
                log_info(f"Blue channel decreased to: {current_color[2]:.2f}")

            # Reset control
            elif key in ["r", "R"]:
                current_color = init_color.clone()
                current_intensity = init_intensity
                current_falloff = init_falloff
                light.set_local_pose(init_pose)
                light.set_color(current_color)
                light.set_intensity(torch.tensor(current_intensity))
                light.set_falloff(torch.tensor(current_falloff))
                property_changed = True
                log_info("Reset to initial light properties")

            # Print current properties
            elif key in ["p", "P"]:
                translation = current_pose[:3]
                log_info("\n=== Current Light Properties ===")
                log_info(f"Position: {translation}")
                log_info(f"Color (RGB): {current_color.numpy()}")
                log_info(f"Intensity: {current_intensity:.2f}")
                log_info(f"Falloff: {current_falloff:.2f}")

            # Update pose if translation changed
            if property_changed and not np.allclose(new_pose, current_pose):
                light_pose = torch.as_tensor(new_pose, dtype=torch.float32).unsqueeze_(
                    0
                )
                light.set_local_pose(light_pose)

            # Update simulation if any property changed
            if property_changed and vis_pose:
                sim.update(step=1)

    except KeyboardInterrupt:
        if vis_pose:
            sim.remove_marker("light_marker")
        log_info("\nControl loop interrupted by user (Ctrl+C)")
    except Exception as e:
        log_error(f"Error in light control loop: {e}")
    finally:
        try:
            # Restore terminal settings
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        except:
            pass
        log_info("Light control loop terminated")
