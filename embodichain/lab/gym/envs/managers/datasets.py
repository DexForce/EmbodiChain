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

from tensordict import TensorDict

from embodichain.utils import logger
from embodichain.data.constants import EMBODICHAIN_DEFAULT_DATASET_ROOT
from embodichain.lab.gym.utils.misc import is_stereocam
from embodichain.lab.sim.sensors import Camera, ContactSensor
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
            obs_list = self._env.rollout_buffer["obs"][
                env_id, : self._env.current_rollout_step
            ]
            action_list = self._env.rollout_buffer["actions"][
                env_id, : self._env.current_rollout_step
            ]

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

    def finalize(self) -> Optional[str]:
        """Finalize the dataset."""
        # Save any remaining episodes
        if self._env.current_rollout_step > 0:
            active_env_ids = torch.arange(self._env.num_envs, device=self._env.device)
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
            metadata_buffer_size=1,
        )
        logger.log_info(f"Created LeRobot dataset at: {self.dataset_full_path}")

    def _build_features(self) -> Dict:
        """Build LeRobot features dict."""
        features = {}

        state_dim = len(self._env.active_joint_ids)
        # Create joint names.
        joint_names = [
            self._env.robot.joint_names[i] for i in self._env.active_joint_ids
        ]

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
        action_dim = state_dim
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

                if isinstance(sensor, Camera):
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
                elif isinstance(sensor, ContactSensor):
                    for frame_name, space in value.items():
                        features[f"{sensor_name}.{frame_name}"] = {
                            "dtype": str(space.dtype),
                            "shape": space.shape,
                            "names": frame_name,
                        }

        # Add any extra features specified in observation space excluding 'robot' and 'sensor'
        for key, space in self._env.single_observation_space.items():
            if key in ["robot", "sensor"]:
                continue

            if isinstance(space, gym.spaces.Dict):
                # Handle nested Dict observation spaces (e.g., physics attributes)
                self._add_nested_features(features, key, space)
                continue

            features[key] = {
                "dtype": str(space.dtype),
                "shape": space.shape,
                "names": key,
            }

        self._modify_feature_names(features)
        return features

    def _add_nested_features(
        self, features: Dict, key: str, space: gym.spaces.Dict
    ) -> None:
        """Add features from nested Dict observation space.

        This recursively processes nested observation spaces and adds them to the features dict.
        For example, physics attributes stored as 'object_physics' with sub-keys
        (mass, friction, damping, inertia, body_scale) will be flattened to:
        - observation.object_physics.mass
        - observation.object_physics.friction
        - observation.object_physics.damping
        - observation.object_physics.inertia
        - observation.object_physics.body_scale

        Args:
            features: The features dict to update.
            key: The top-level key of the nested space.
            space: The nested Dict observation space.
        """
        for sub_key, sub_space in space.spaces.items():
            if isinstance(sub_space, gym.spaces.Dict):
                # Recursively handle deeper nesting
                self._add_nested_features(features, f"{key}.{sub_key}", sub_space)
            else:
                feature_name = f"observation.{key}.{sub_key}"
                # Handle empty shapes for scalar values (e.g., mass, friction, damping)
                # LeRobot requires non-empty shapes, so convert () to (1,)
                shape = sub_space.shape if sub_space.shape else (1,)
                features[feature_name] = {
                    "dtype": str(sub_space.dtype),
                    "shape": shape,
                    "names": sub_key,
                }

    def _modify_feature_names(self, features: dict[str, Any]) -> None:
        """Get feature names for an observation based on its functor config.

        Note:
            The `space` parameter is kept for API consistency but not used
            directly, as the feature names are derived from the functor config
            and entity properties.

        For observations generated by `get_object_uid`, returns meaningful names:
        - RigidObject: object UID names
        - Articulation/Robot: link names

        Args:
            key: The observation space key.
            space: The observation space.

        Returns:
            A list of feature names for the observation.
        """
        from embodichain.lab.gym.envs.managers.observations import get_object_uid
        from embodichain.lab.sim.objects import RigidObject, Articulation, Robot

        # Change the features shape if is ()
        for key, feature in features.items():
            if feature["shape"] == ():
                features[key]["shape"] = (1,)

        for functor_name in self._env.observation_manager.active_functors["add"]:
            functor_cfg = self._env.observation_manager.get_functor_cfg(
                functor_name=functor_name
            )
            if functor_cfg.func == get_object_uid:
                obs_key = functor_cfg.name
                asset_uid = functor_cfg.params["entity_cfg"].uid
                asset = self._env.sim.get_asset(asset_uid)
                if isinstance(asset, RigidObject):
                    features[obs_key]["names"] = asset_uid
                elif isinstance(asset, (Articulation, Robot)):
                    link_names = asset.link_names
                    features[obs_key]["names"] = link_names
                else:
                    logger.log_warning(
                        f"Asset with UID '{asset_uid}' is not RigidObject, Articulation or Robot. Cannot assign feature names based on asset properties."
                    )

    def _convert_frame_to_lerobot(
        self, obs: TensorDict, action: TensorDict | torch.Tensor, task: str
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

                if isinstance(sensor, Camera):
                    is_stereo = is_stereocam(sensor)

                    color_data = obs["sensor"][sensor_name]["color"]
                    color_img = color_data[:, :, :3].cpu()
                    frame[f"{sensor_name}.color"] = color_img

                    if is_stereo:
                        color_right_data = obs["sensor"][sensor_name]["color_right"]
                        color_right_img = color_right_data[:, :, :3].cpu()
                        frame[f"{sensor_name}.color_right"] = color_right_img
                elif isinstance(sensor, ContactSensor):
                    for frame_name in value.keys():
                        frame[f"{sensor_name}.{frame_name}"] = obs["sensor"][
                            sensor_name
                        ][
                            frame_name
                        ].cpu()  # Debug here to inspect contact sensor data
                else:
                    logger.log_warning(
                        f"Unsupported sensor type for '{sensor_name}' when converting to LeRobot format. Currently only support Camera and ContactSensor."
                    )

        # Add state
        frame["observation.qpos"] = obs["robot"]["qpos"].cpu()
        frame["observation.qvel"] = obs["robot"]["qvel"].cpu()
        frame["observation.qf"] = obs["robot"]["qf"].cpu()

        # Add extra observation features if they exist
        for key in obs.keys():
            if key in ["robot", "sensor"]:
                continue

            value = obs[key]
            if isinstance(value, TensorDict):
                # Handle nested TensorDict (e.g., physics attributes)
                self._add_nested_obs_to_frame(frame, key, value)
            else:
                if value.shape == ():
                    value = value.unsqueeze(0)
                frame[key] = value.cpu()

        # Add action.
        if isinstance(action, torch.Tensor):
            action_data = action.cpu()
        elif isinstance(action, TensorDict):
            # Extract qpos from action dict
            action_tensor = action.get("qpos", None)
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

    def _add_nested_obs_to_frame(
        self, frame: Dict, key: str, nested_obs: TensorDict
    ) -> None:
        """Add nested observation data to frame dict.

        This recursively processes nested TensorDict observations and adds them to the frame dict.
        For example, physics attributes stored as 'object_physics' with sub-keys
        (mass, friction, damping, inertia, body_scale) will be flattened to:
        - observation.object_physics.mass
        - observation.object_physics.friction
        - observation.object_physics.damping
        - observation.object_physics.inertia
        - observation.object_physics.body_scale

        Args:
            frame: The frame dict to update.
            key: The top-level key of nested observation.
            nested_obs: The nested TensorDict observation.
        """
        for sub_key, sub_value in nested_obs.items():
            if isinstance(sub_value, TensorDict):
                # Recursively handle deeper nesting
                self._add_nested_obs_to_frame(frame, f"{key}.{sub_key}", sub_value)
            else:
                value = sub_value.cpu()
                # Handle 0D tensors (scalars) - convert to 1D for LeRobot compatibility
                if isinstance(value, torch.Tensor) and value.ndim == 0:
                    value = value.unsqueeze(0)
                frame[f"observation.{key}.{sub_key}"] = value

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
