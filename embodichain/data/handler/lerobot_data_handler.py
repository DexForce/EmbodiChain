from typing import Dict, Any, List, Union, Optional
from copy import deepcopy
from pathlib import Path
import traceback

import numpy as np
import torch

from embodichain.lab.gym.envs import EmbodiedEnv
from embodichain.data.enum import (
    HandQposNormalizer,
    Modality,
    JointType,
)
from embodichain.utils.utility import get_right_name
from embodichain.lab.gym.utils.misc import is_stereocam, data_key_to_control_part
from embodichain.utils import logger
from embodichain.data.enum import (
    SUPPORTED_PROPRIO_TYPES,
    SUPPORTED_ACTION_TYPES,
)
from tqdm import tqdm

# Optional LeRobot imports (for convert to lerobot format functionality)
from lerobot.datasets.lerobot_dataset import LeRobotDataset, HF_LEROBOT_HOME


class LerobotDataHandler:
    def __init__(
        self,
        env: EmbodiedEnv,
    ):
        self.env = env

    def _build_lerobot_features(self, use_videos: bool = True) -> Dict:
        """
        Build LeRobot features dict from environment metadata.

        Args:
            use_videos (bool): Whether to use video encoding. Defaults to True.

        Returns:
            Dict: Features dictionary compatible with LeRobot format.
        """
        features = {}
        robot_meta_config = self.env.metadata["dataset"]["robot_meta"]
        extra_vision_config = robot_meta_config["observation"]["vision"]

        # Add image features
        for camera_name in extra_vision_config.keys():
            sensor = self.env.get_sensor(camera_name)
            is_stereo = is_stereocam(sensor)

            # Get image shape from sensor
            img_shape = (
                sensor.cfg.height,
                sensor.cfg.width,
                3,
            )

            features[camera_name] = {
                "dtype": "video" if use_videos else "image",
                "shape": img_shape,
                "names": ["height", "width", "channel"],
            }

            if is_stereo:
                features[get_right_name(camera_name)] = {
                    "dtype": "video" if use_videos else "image",
                    "shape": img_shape,
                    "names": ["height", "width", "channel"],
                }

        # Add state features (proprio)
        qpos = self.env.robot.get_qpos()
        state_dim = qpos.shape[1]

        if state_dim > 0:
            features["observation.state"] = {
                "dtype": "float32",
                "shape": (state_dim,),
                "names": ["state"],
            }

        # Add action features
        action_dim = robot_meta_config.get("arm_dofs", 7)
        features["action"] = {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": ["action"],
        }

        return features

    def _convert_frame_to_lerobot(
        self, obs: Dict[str, Any], action: Dict[str, Any], task: str
    ) -> List[Dict]:
        """
        Convert frames from all environments to LeRobot format.

        Args:
            obs (Dict): Observation dict from environment (batched).
            action (Dict): Action dict from environment (batched).
            task (str): Task description.

        Returns:
            List[Dict]: List of frames in LeRobot format, one per environment.
        """
        robot_meta_config = self.env.metadata["dataset"]["robot_meta"]
        extra_vision_config = robot_meta_config["observation"]["vision"]
        robot = self.env.robot
        arm_dofs = robot_meta_config.get("arm_dofs", 7)

        # Determine batch size from qpos
        qpos = obs["robot"][JointType.QPOS.value]

        for control_part in robot.control_parts:
            indices = robot.get_joint_ids(control_part, remove_mimic=True)
            qpos_data = qpos[0][indices].cpu().numpy()
            qpos_data = HandQposNormalizer.normalize_hand_qpos(
                qpos_data, control_part, robot=robot
            )
            # state_list.append(qpos_data)
        num_envs = qpos.shape[0]

        frames = []

        all_qpos = robot.get_qpos()

        # Process each environment
        for env_idx in range(num_envs):
            frame = {"task": task}

            # Add images
            for camera_name in extra_vision_config.keys():
                if camera_name in obs["sensor"]:
                    is_stereo = is_stereocam(self.env.get_sensor(camera_name))

                    # Process left/main camera image
                    color_data = obs["sensor"][camera_name]["color"]
                    if isinstance(color_data, torch.Tensor):
                        color_img = color_data[env_idx][:, :, :3].cpu().numpy()
                    else:
                        color_img = np.array(color_data)[env_idx][:, :, :3]

                    # Ensure uint8 format (0-255 range)
                    if color_img.dtype == np.float32 or color_img.dtype == np.float64:
                        color_img = (color_img * 255).astype(np.uint8)

                    frame[camera_name] = color_img

                    # Process right camera image if stereo
                    if is_stereo:
                        color_right_data = obs["sensor"][camera_name]["color_right"]
                        if isinstance(color_right_data, torch.Tensor):
                            color_right_img = (
                                color_right_data[env_idx][:, :, :3].cpu().numpy()
                            )
                        else:
                            color_right_img = np.array(color_right_data)[env_idx][
                                :, :, :3
                            ]

                        # Ensure uint8 format
                        if (
                            color_right_img.dtype == np.float32
                            or color_right_img.dtype == np.float64
                        ):
                            color_right_img = (color_right_img * 255).astype(np.uint8)

                        frame[get_right_name(camera_name)] = color_right_img

            # Add state (proprio)
            # robot.get_qpos() returns shape (num_envs, num_joints)
            state = all_qpos[env_idx].cpu().numpy().astype(np.float32)

            frame["observation.state"] = state

            # Add actions
            # Handle different action types
            if isinstance(action, torch.Tensor):
                action_data = action[env_idx, :arm_dofs].cpu().numpy()
            elif isinstance(action, np.ndarray):
                action_data = action[env_idx, :arm_dofs]
            elif isinstance(action, dict):
                # If action is a dict, try to extract the actual action data
                action_data = action.get("action", action.get("arm_action", action))
                if isinstance(action_data, torch.Tensor):
                    action_data = action_data[env_idx, :arm_dofs].cpu().numpy()
                elif isinstance(action_data, np.ndarray):
                    action_data = action_data[env_idx, :arm_dofs]
            else:
                # Fallback: try to convert to numpy
                action_data = np.array(action)[env_idx, :arm_dofs]

            frame["action"] = action_data

            frames.append(frame)

        return frames
