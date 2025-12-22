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
        save_path: str = None,
        compression_opts: int = 9,
    ):
        self.env = env
        self.save_path = save_path
        self.data = {}

        # save all supported proprio and action types.
        robot_meta_config = deepcopy(self.env.metadata["dataset"]["robot_meta"])
        robot_meta_config["observation"][
            Modality.STATES.value
        ] = SUPPORTED_PROPRIO_TYPES
        robot_meta_config[Modality.ACTIONS.value] = SUPPORTED_ACTION_TYPES

        self.compression_opts = compression_opts

    def extract_to_lerobot(
        self,
        obs_list: List[Dict[str, Any]],
        action_list: List[Dict[str, Any]],
        repo_id: str,
        fps: int = 30,
        use_videos: bool = True,
        image_writer_threads: int = 4,
        image_writer_processes: int = 0,
    ) -> "LeRobotDataset":
        """
        Extract data and save in LeRobot format.

        Args:
            obs_list (List[Dict]): List of observation dicts.
            action_list (List[Dict]): List of action dicts.
            repo_id (str): Repository ID for the LeRobot dataset.
            fps (int): Frames per second. Defaults to 30.
            use_videos (bool): Whether to use video encoding. Defaults to True.
            image_writer_threads (int): Number of threads for image writing. Defaults to 4.
            image_writer_processes (int): Number of processes for image writing. Defaults to 0.

        Returns:
            LeRobotDataset: The created LeRobot dataset instance.
        """
        # Build features dict from environment metadata
        features = self._build_lerobot_features(use_videos=use_videos)

        # Get robot type
        robot_type = self.env.metadata["dataset"]["robot_meta"].get(
            "robot_type", "unknown"
        )

        # Create LeRobot dataset
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            robot_type=robot_type,
            fps=fps,
            features=features,
            use_videos=use_videos,
            image_writer_threads=image_writer_threads,
            image_writer_processes=image_writer_processes,
        )

        # Get task/instruction
        task = self.env.metadata["dataset"]["instruction"].get("lang", "unknown_task")

        # Convert and add frames
        logger.log_info(f"Converting {len(obs_list)} frames to LeRobot format...")
        for i, (obs, action) in enumerate(zip(obs_list, action_list)):
            frame = self._convert_frame_to_lerobot(obs, action, task)
            dataset.add_frame(frame)

        return dataset

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
        state_dim = 0
        for proprio_name in SUPPORTED_PROPRIO_TYPES:
            robot = self.env.robot
            part = data_key_to_control_part(
                robot=robot,
                control_parts=robot_meta_config.get("control_parts", []),
                data_key=proprio_name,
            )
            if part:
                indices = robot.get_joint_ids(part, remove_mimic=True)
                state_dim += len(indices)

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
    ) -> Dict:
        """
        Convert a single frame to LeRobot format.

        Args:
            obs (Dict): Observation dict from environment.
            action (Dict): Action dict from environment.
            task (str): Task description.

        Returns:
            Dict: Frame in LeRobot format.
        """
        frame = {"task": task}
        robot_meta_config = self.env.metadata["dataset"]["robot_meta"]
        extra_vision_config = robot_meta_config["observation"]["vision"]

        # Add images
        for camera_name in extra_vision_config.keys():
            if camera_name in obs["sensor"]:
                is_stereo = is_stereocam(self.env.get_sensor(camera_name))

                # Process left/main camera image
                color_data = obs["sensor"][camera_name]["color"]
                if isinstance(color_data, torch.Tensor):
                    color_img = color_data.squeeze(0)[:, :, :3].cpu().numpy()
                else:
                    color_img = np.array(color_data).squeeze(0)[:, :, :3]

                # Ensure uint8 format (0-255 range)
                if color_img.dtype == np.float32 or color_img.dtype == np.float64:
                    color_img = (color_img * 255).astype(np.uint8)

                frame[camera_name] = color_img

                # Process right camera image if stereo
                if is_stereo:
                    color_right_data = obs["sensor"][camera_name]["color_right"]
                    if isinstance(color_right_data, torch.Tensor):
                        color_right_img = (
                            color_right_data.squeeze(0)[:, :, :3].cpu().numpy()
                        )
                    else:
                        color_right_img = np.array(color_right_data).squeeze(0)[
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
        state_list = []
        robot = self.env.robot
        qpos = obs["robot"][JointType.QPOS.value]
        for proprio_name in SUPPORTED_PROPRIO_TYPES:
            part = data_key_to_control_part(
                robot=robot,
                control_parts=robot_meta_config.get("control_parts", []),
                data_key=proprio_name,
            )
            if part:
                indices = robot.get_joint_ids(part, remove_mimic=True)
                qpos_data = qpos[0][indices].cpu().numpy()
                qpos_data = HandQposNormalizer.normalize_hand_qpos(
                    qpos_data, part, robot=robot
                )
                state_list.append(qpos_data)

        if state_list:
            frame["observation.state"] = np.concatenate(state_list)

        # Add actions
        robot = self.env.robot
        arm_dofs = robot_meta_config.get("arm_dofs", 7)

        # Handle different action types
        if isinstance(action, torch.Tensor):
            action_data = action[0, :arm_dofs].cpu().numpy()
        elif isinstance(action, np.ndarray):
            action_data = action[0, :arm_dofs]
        elif isinstance(action, dict):
            # If action is a dict, try to extract the actual action data
            # This depends on your action dict structure
            action_data = action.get("action", action.get("arm_action", action))
            if isinstance(action_data, torch.Tensor):
                action_data = action_data[0, :arm_dofs].cpu().numpy()
            elif isinstance(action_data, np.ndarray):
                action_data = action_data[0, :arm_dofs]
        else:
            # Fallback: try to convert to numpy
            action_data = np.array(action)[0, :arm_dofs]

        frame["action"] = action_data

        return frame


def save_to_lerobot_format(
    env: EmbodiedEnv,
    obs_list: List[Dict[str, Any]],
    action_list: List[Dict[str, Any]],
    repo_id: str,
    fps: int = 30,
    use_videos: bool = True,
    push_to_hub: bool = False,
    image_writer_threads: int = 4,
    image_writer_processes: int = 0,
) -> Optional[str]:
    """
    Save episode data to LeRobot format.

    Args:
        env (EmbodiedEnv): Environment instance.
        obs_list (List[Dict]): List of observation dicts (without last obs).
        action_list (List[Dict]): List of action dicts.
        repo_id (str): Repository ID for LeRobot dataset (e.g., "username/dataset_name").
        fps (int): Frames per second. Defaults to 30.
        use_videos (bool): Whether to encode images as videos. Defaults to True.
        push_to_hub (bool): Whether to push to Hugging Face Hub. Defaults to False.
        image_writer_threads (int): Number of threads for image writing. Defaults to 4.
        image_writer_processes (int): Number of processes for image writing. Defaults to 0.

    Returns:
        Optional[str]: Path to saved dataset, or None if failed.

    Example:
        >>> save_to_lerobot_format(
        ...     env=env,
        ...     obs_list=env.episode_obs_list[:-1],
        ...     action_list=env.episode_action_list,
        ...     repo_id="my_username/my_robot_dataset",
        ...     fps=30,
        ...     use_videos=True,
        ...     push_to_hub=False,
        ... )
    """

    if len(obs_list) == 0 or len(action_list) == 0:
        logger.log_error("obs_list and action_list cannot be empty")
        return None

    if len(obs_list) != len(action_list):
        logger.log_error(
            f"obs_list and action_list must have same length, got {len(obs_list)} and {len(action_list)}"
        )
        return None

    try:
        extractor = LerobotDataHandler(env)

        # Build features
        features = extractor._build_lerobot_features(use_videos=use_videos)

        # Get robot type
        robot_type = env.metadata["dataset"]["robot_meta"].get("robot_type", "unknown")

        # Get or create dataset
        if HF_LEROBOT_HOME is not None:
            dataset_path = Path(HF_LEROBOT_HOME) / repo_id
        else:
            # Fallback to default path
            dataset_path = Path.home() / ".cache" / "huggingface" / "lerobot" / repo_id

        # Check if dataset already exists
        if dataset_path.exists():
            logger.log_info(f"Loading existing LeRobot dataset from {dataset_path}")
            dataset = LeRobotDataset(repo_id=repo_id)
        else:
            logger.log_info(f"Creating new LeRobot dataset at {dataset_path}")
            dataset = LeRobotDataset.create(
                repo_id=repo_id,
                robot_type=robot_type,
                fps=fps,
                features=features,
                use_videos=use_videos,
                image_writer_threads=image_writer_threads,
                image_writer_processes=image_writer_processes,
            )

        # Get task
        task = env.metadata["dataset"]["instruction"].get("lang", "unknown_task")

        # Add frames
        logger.log_info(f"Adding {len(obs_list)} frames to LeRobot dataset...")
        for obs, action in tqdm(zip(obs_list, action_list), total=len(obs_list)):
            frame = extractor._convert_frame_to_lerobot(obs, action, task)
            dataset.add_frame(frame)

        # Save episode
        logger.log_info("Saving episode to LeRobot dataset...")
        dataset.save_episode()

        # Optionally push to hub
        if push_to_hub:
            logger.log_info(f"Pushing dataset to Hugging Face Hub: {repo_id}")
            dataset.push_to_hub(
                tags=[robot_type, "imitation"],
                private=False,
                push_videos=use_videos,
                license="apache-2.0",
            )

        logger.log_info(f"Successfully saved episode to {dataset_path}")
        return str(dataset_path)

    except Exception as e:
        logger.log_error(f"Failed to save to LeRobot format: {e}")
        traceback.print_exc()
        return None
