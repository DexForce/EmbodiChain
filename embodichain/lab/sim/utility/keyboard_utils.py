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

import cv2
import torch
import numpy as np

from scipy.spatial.transform import Rotation as R

from embodichain.lab.sim.sensors import Camera
from embodichain.utils.logger import log_info, log_error, log_warning


def run_keyboard_control_for_camera(
    sensor: Camera,
    trans_step: float = 0.01,
    rot_step: float = 1.0,
    vis_pose: bool = False,
) -> None:
    """Run keyboard control loop for camera pose adjustment.

    Args:
        sensor (Camera): Camera sensor to control.
        trans_step (float, optional): Translation step size. Defaults to 0.01.
        rot_step (float, optional): Rotation step size in degrees. Defaults to 1.0.
        vis_pose (bool, optional): Whether to visualize the camera pose in axis form. Defaults to False.
    """
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
        from embodichain.lab.sim import SimulationManager
        from embodichain.lab.sim.cfg import MarkerCfg

        init_axis_pose = sensor.get_arena_pose(to_matrix=True).squeeze().numpy()

        sim = SimulationManager.get_instance()
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
                translation = new_pose[:3, 3]
                rot = R.from_matrix(new_pose[:3, :3])
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
