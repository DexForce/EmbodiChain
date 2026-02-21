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

"""Dataset functors for collecting and saving episode data."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

import numpy as np
import gymnasium as gym
import torch
import tqdm

from embodichain.utils import logger
from embodichain.data.constants import EMBODICHAIN_DEFAULT_DATASET_ROOT
from embodichain.lab.gym.utils.misc import is_stereocam
from embodichain.utils.utility import get_right_name
from embodichain.data.enum import JointType
from .manager_base import Functor
from .cfg import DatasetFunctorCfg

if TYPE_CHECKING:
    from embodichain.lab.gym.envs import EmbodiedEnv

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    LEROBOT_AVAILABLE = True
    __all__ = ["LeRobotRecorder"]
except ImportError:
    LEROBOT_AVAILABLE = False
    __all__ = []


class LeRobotRecorder(Functor):
    """Functor for recording episodes in LeRobot format.

    This functor handles:
    - Recording observation-action pairs during episodes
    - Converting data to LeRobot format
    - Saving episodes when they complete
    """

    def __init__(self, cfg: DatasetFunctorCfg, env: EmbodiedEnv):
        """Initialize the LeRobot dataset recorder.

        Args:
            cfg: Functor configuration containing params:
                - save_path: Root directory for saving datasets
                - robot_meta: Robot metadata for dataset
                - instruction: Optional task instruction
                - extra: Optional extra metadata
                - use_videos: Whether to save videos
                - image_writer_threads: Number of threads for image writing
                - image_writer_processes: Number of processes for image writing
            env: The environment instance
        """
        if not LEROBOT_AVAILABLE:
            logger.log_error(
                "LeRobot is not installed. Please install it with: pip install lerobot"
            )

        super().__init__(cfg, env)

        # Extract parameters from cfg.params
        params = cfg.params

        # Required parameters
        self.lerobot_data_root = params.get(
            "save_path", EMBODICHAIN_DEFAULT_DATASET_ROOT
        )
        self.robot_meta = params.get("robot_meta", {})

        # Optional parameters
        self.instruction = params.get("instruction", None)
        self.extra = params.get("extra", {})

        # Experimental parameters for extra episode info saving.
        self.extra_episode_info = self.extra.get("episode_info", {})
        self.extra_episode_info_buffer = {}
        self.use_videos = params.get("use_videos", False)

        # LeRobot dataset instance
        self.dataset: Optional[LeRobotDataset] = None
        self.dataset_full_path: Optional[Path] = None

        # Tracking
        self.total_time: float = 0.0
        self.curr_episode: int = 0

        # Initialize dataset
        self._initialize_dataset()

    @property
    def dataset_path(self) -> str:
        """Path to the dataset directory."""
        return (
            str(self.dataset_full_path) if self.dataset_full_path else "Not initialized"
        )

    def __call__(
        self,
        env: EmbodiedEnv,
        env_ids: Union[torch.Tensor, None],
        save_path: Optional[str] = None,
        robot_meta: Optional[Dict] = None,
        instruction: Optional[str] = None,
        extra: Optional[Dict] = None,
        use_videos: bool = False,
    ) -> None:
        """Main entry point for the recorder functor.

        This method is called by DatasetManager.apply(mode="save") to save completed episodes.
        It reads data from the environment's episode buffers.

        Args:
            env: The environment instance.
            env_ids: Environment IDs to save. If None, attempts to save all environments.
        """
        # If env_ids is None, check all environments for completed episodes
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=env.device)
        elif isinstance(env_ids, (list, range)):
            env_ids = torch.tensor(list(env_ids), device=env.device)

        # Save episodes for specified environments
        if len(env_ids) > 0:
            self._save_episodes(env_ids)

    def _save_episodes(
        self,
        env_ids: torch.Tensor,
    ) -> None:
        """Save completed episodes for specified environments."""
        task = self.instruction.get("lang", "unknown_task")

        # Process each environment
        for env_id in env_ids.cpu().tolist():
            # Get buffer for this environment (already contains single-env data)
            obs_list = self._env.episode_obs_buffer[env_id]
            action_list = self._env.episode_action_buffer[env_id]

            if len(obs_list) == 0:
                logger.log_warning(f"No episode data to save for env {env_id}")
                continue

            # Align obs and action
            if len(obs_list) > len(action_list):
                obs_list = obs_list[:-1]

            # Update metadata
            extra_info = self.extra.copy() if self.extra else {}
            fps = self.dataset.meta.info.get("fps", 30)
            current_episode_time = len(obs_list) / fps if fps > 0 else 0

            episode_extra_info = extra_info.copy()
            self.total_time += current_episode_time
            episode_extra_info["total_time"] = self.total_time
            self._save_extra_episode_meta_info(env_id)

            try:
                for obs, action in tqdm.tqdm(
                    zip(obs_list, action_list),
                    total=len(obs_list),
                    desc=f"Converting env {env_id} episode to LeRobot format",
                ):
                    frame = self._convert_frame_to_lerobot(obs, action, task)
                    self.dataset.add_frame(frame)

                self.dataset.save_episode()

                logger.log_info(
                    f"[LeRobotRecorder] Saved dataset to: {self.dataset_path}\n"
                    f"  Episode {self.curr_episode} (env {env_id}): {len(obs_list)} frames"
                )

                self.curr_episode += 1
            except Exception as e:
                logger.log_error(f"Failed to save episode {env_id}: {e}")

    def _save_extra_episode_meta_info(self, env_id: int) -> None:
        """Save extra episode meta info for a specific environment ID."""

        curr_extra_episode_info = {}
        if self.extra_episode_info:
            for key, attr_list in self.extra_episode_info.items():
                if key == "rigid_object_physics_attributes":
                    rigid_obj_list = self._env.sim.get_rigid_object_uid_list()
                    for obj_uid in rigid_obj_list:
                        curr_extra_episode_info[obj_uid] = {}
                        obj = self._env.sim.get_rigid_object(obj_uid)
                        for attr in attr_list:
                            if attr == "mass":
                                curr_extra_episode_info[obj_uid]["mass"] = round(
                                    obj.get_mass(env_ids=[env_id]).squeeze_().item(), 5
                                )

        self.extra_episode_info_buffer[self.curr_episode] = curr_extra_episode_info
        self._update_dataset_info(
            {"extra_episode_info": self.extra_episode_info_buffer}
        )

    def finalize(self) -> Optional[str]:
        """Finalize the dataset."""
        # Save any remaining episodes
        env_ids_with_data = []
        for env_id in range(self.num_envs):
            if len(self._env.episode_obs_buffer[env_id]) > 0:
                env_ids_with_data.append(env_id)

        if env_ids_with_data:
            active_env_ids = torch.tensor(env_ids_with_data, device=self.device)
            self._save_episodes(active_env_ids)

        try:
            if self.dataset is not None:
                self.dataset.finalize()
                logger.log_info(
                    f"[LeRobotRecorder] Dataset finalized successfully\n"
                    f"  Path: {self.dataset_path}\n"
                    f"  Total episodes: {self.curr_episode}\n"
                    f"  Total time: {self.total_time:.2f}s"
                )
                return self.dataset_path
        except Exception as e:
            logger.log_error(f"[LeRobotRecorder] Failed to finalize dataset: {e}")

        return None

    def _initialize_dataset(self) -> None:
        """Initialize the LeRobot dataset."""
        robot_type = self.robot_meta.get("robot_type", "robot")
        scene_type = self.extra.get("scene_type", "scene")
        task_description = self.extra.get("task_description", "task")

        robot_type = str(robot_type).lower().replace(" ", "_")
        task_description = str(task_description).lower().replace(" ", "_")

        # Use lerobot_data_root from __init__
        lerobot_data_root = Path(self.lerobot_data_root)

        # Generate dataset folder name with auto-incrementing suffix
        base_name = f"{robot_type}_{scene_type}_{task_description}"

        # Find the next available sequence number by checking existing folders
        existing_dirs = list(lerobot_data_root.glob(f"{base_name}_*"))
        if not existing_dirs:
            dataset_id = 0
        else:
            # Extract sequence numbers from existing directories
            max_id = -1
            for dir_path in existing_dirs:
                suffix = dir_path.name[len(base_name) + 1 :]  # +1 for underscore
                if suffix.isdigit():
                    max_id = max(max_id, int(suffix))
            dataset_id = max_id + 1

        # Format dataset name with zero-padding (3 digits: 000, 001, 002, ...)
        dataset_name = f"{base_name}_{dataset_id:03d}"

        # LeRobot's root parameter is the COMPLETE dataset path (not parent directory)
        self.dataset_full_path = lerobot_data_root / dataset_name

        fps = self.robot_meta.get("control_freq", 30)
        features = self._build_features()

        self.dataset = LeRobotDataset.create(
            repo_id=dataset_name,
            fps=fps,
            root=str(self.dataset_full_path),
            robot_type=robot_type,
            features=features,
            use_videos=self.use_videos,
        )
        logger.log_info(f"Created LeRobot dataset at: {self.dataset_full_path}")

    def _build_features(self) -> Dict:
        """Build LeRobot features dict."""
        features = {}

        # Setup robot joint state features based on control_parts or all joints if not specified.
        control_parts = self.robot_meta.get("control_parts", None)
        if control_parts is not None:
            self._joint_ids = []
            for part in control_parts:
                part_joint_ids = self._env.robot.get_joint_ids(part, remove_mimic=True)
                self._joint_ids.extend(part_joint_ids)
        else:
            self._joint_ids = self._env.robot.get_joint_ids(remove_mimic=True)

        state_dim = len(self._joint_ids)
        # Create joint names.
        joint_names = [self._env.robot.joint_names[i] for i in self._joint_ids]

        features["observation.qpos"] = {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": joint_names,
        }
        features["observation.qvel"] = {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": joint_names,
        }
        features["observation.qf"] = {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": joint_names,
        }

        # Use full qpos dimension for action (includes gripper)
        action_dim = len(self._joint_ids)
        features["action"] = {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": joint_names,
        }

        # Setup sensor observation features based env.observation.sensor
        if self._env.has_sensors:
            sensor_obs_space: dict = self._env.single_observation_space["sensor"]

            for sensor_name, value in sensor_obs_space.items():
                sensor = self._env.get_sensor(sensor_name)
                is_stereo = is_stereocam(sensor)

                for frame_name, space in value.items():
                    # TODO: Support depth (uint16) and mask (also uint16 or uint8)
                    if frame_name not in ["color", "color_right"]:
                        logger.log_error(
                            f"Only support 'color' frame for vision sensors, but got '{frame_name}' in sensor '{sensor_name}'"
                        )

                    features[f"{sensor_name}.{frame_name}"] = {
                        "dtype": "video" if self.use_videos else "image",
                        "shape": (sensor.cfg.height, sensor.cfg.width, 3),
                        "names": ["height", "width", "channel"],
                    }

                    if is_stereo:
                        features[f"{sensor_name}.{frame_name}_right"] = {
                            "dtype": "video" if self.use_videos else "image",
                            "shape": (sensor.cfg.height, sensor.cfg.width, 3),
                            "names": ["height", "width", "channel"],
                        }

        # TODO: The extra observation features are supposed to be defined in a flattened way in the observation space.
        # Lerobot requires a flat feature dict, so we may need to support nested dicts to flatten dict conversion in the future.
        # Add any extra features specified in observation space excluding 'robot' and 'sensor'
        for key, space in self._env.single_observation_space.items():
            if key in ["robot", "sensor"]:
                continue

            if isinstance(space, gym.spaces.Dict):
                logger.log_warning(
                    f"Nested Dict observation space for key '{key}' is not directly supported. "
                    f"Please flatten it or specify features manually. Skipping '{key}'."
                )
                continue

            names = key
            if "vel" in key:
                names = ["lin_x", "lin_y", "lin_z", "ang_x", "ang_y", "ang_z"]
            elif "pose" in key:
                names = ["x", "y", "z", "qw", "qx", "qy", "qz"]

            features[f"observation.{key}"] = {
                "dtype": str(space.dtype),
                "shape": space.shape,
                "names": names,
            }

        return features

    def _convert_frame_to_lerobot(
        self, obs: Dict[str, Any], action: Any, task: str
    ) -> Dict:
        """Convert a single frame to LeRobot format.

        Args:
            obs: Single environment observation (already extracted from batch)
            action: Single environment action (already extracted from batch)
            task: Task name

        Returns:
            Frame dict in LeRobot format with numpy arrays
        """
        frame = {"task": task}

        if self._env.has_sensors:
            sensor_obs_space: dict = self._env.single_observation_space["sensor"]

            # Add images
            for sensor_name, value in sensor_obs_space.items():
                sensor = self._env.get_sensor(sensor_name)
                is_stereo = is_stereocam(sensor)

                color_data = obs["sensor"][sensor_name]["color"]
                color_img = color_data[:, :, :3].cpu()
                frame[f"{sensor_name}.color"] = color_img

                if is_stereo:
                    color_right_data = obs["sensor"][sensor_name]["color_right"]
                    color_right_img = color_right_data[:, :, :3].cpu()
                    frame[f"{sensor_name}.color_right"] = color_right_img

        # Add state
        frame["observation.qpos"] = obs["robot"]["qpos"][self._joint_ids].cpu()
        frame["observation.qvel"] = obs["robot"]["qvel"][self._joint_ids].cpu()
        frame["observation.qf"] = obs["robot"]["qf"][self._joint_ids].cpu()

        # Add extra observation features if they exist
        for key in obs:
            if key in ["robot", "sensor"]:
                continue

            frame[f"observation.{key}"] = obs[key].cpu()

        # Add action.
        action = action[self._joint_ids]
        if isinstance(action, torch.Tensor):
            action_data = action.cpu()
        elif isinstance(action, dict):
            # Extract qpos from action dict
            action_tensor = action.get(
                "qpos", action.get("delta_qpos", action.get("action", None))
            )
            if action_tensor is None:
                # Fallback to first tensor value
                for v in action.values():
                    if isinstance(v, (torch.Tensor, np.ndarray)):
                        action_tensor = v
                        break

            if isinstance(action_tensor, torch.Tensor):
                action_data = action_tensor.cpu()

        frame["action"] = action_data

        return frame

    def _update_dataset_info(self, updates: dict) -> bool:
        """Update dataset metadata."""
        if self.dataset is None:
            logger.log_error("LeRobotDataset not initialized.")
            return False

        try:
            self.dataset.meta.info.update(updates)
            return True
        except Exception as e:
            logger.log_error(f"Failed to update dataset info: {e}")
            return False
