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

"""Dataset manager for collecting and saving episode data."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
import torch

from embodichain.utils import logger
from embodichain.lab.sim.types import EnvObs, EnvAction
from embodichain.lab.gym.utils.misc import is_stereocam
from embodichain.utils.utility import get_right_name
from embodichain.data.enum import JointType
from .manager_base import ManagerBase
from .cfg import DatasetCfg

if TYPE_CHECKING:
    from embodichain.lab.gym.envs import EmbodiedEnv

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, HF_LEROBOT_HOME
    LEROBOT_AVAILABLE = True
except ImportError:
    LEROBOT_AVAILABLE = False


class DatasetManager(ManagerBase):
    """Manager for collecting and saving episode data in LeRobot format.
    
    The dataset manager is responsible for:
    - Recording observations and actions during episode rollouts
    - Converting data to LeRobot format
    - **Automatically saving episodes when they complete (via on_episode_end)**
    - Managing dataset metadata
    
    The manager automatically saves episodes when:
    1. Episode completes (done=True) via on_episode_end()
    2. Environment is closed via finalize()
    
    Episodes are NOT saved on reset() - only on explicit episode completion.
    """

    _env: EmbodiedEnv
    """The environment instance."""

    def __init__(self, cfg: DatasetCfg, env: EmbodiedEnv):
        """Initialize the dataset manager.
        
        Args:
            cfg: Configuration for the dataset manager.
            env: The environment instance.
        """
        if not LEROBOT_AVAILABLE:
            raise ImportError(
                "LeRobot is not available. Install it with: pip install lerobot"
            )
        
        super().__init__(cfg, env)
        
        # Episode data buffers - store batched data (shared across all envs)
        self.episode_obs_list: List[Dict] = []  # Single list for batched obs
        self.episode_action_list: List[Any] = []  # Single list for batched actions
        
        # LeRobot dataset instance
        self.dataset: Optional[LeRobotDataset] = None
        
        # Track total time across all episodes
        self.total_time: float = 0.0
        
        # Track current episode count
        self.curr_episode: int = 0
        
        # Initialize the dataset
        self._initialize_dataset()
        
        logger.log_info(
            f"DatasetManager initialized with LeRobot format at: {self.dataset_path}"
        )

    """
    Properties.
    """

    @property
    def active_functors(self) -> list[str]:
        """Name of active functors.
        
        DatasetManager doesn't use functors like other managers, 
        so this returns an empty list.
        """
        return []

    @property
    def __str__(self) -> str:
        """Returns: A string representation for dataset manager."""
        msg = "<DatasetManager> for LeRobot format\n"
        msg += f"  Dataset path: {self.dataset_path}\n"
        msg += f"  Robot type: {self.cfg.robot_meta.get('robot_type', 'unknown')}\n"
        msg += f"  Control freq: {self.cfg.robot_meta.get('control_freq', 30)} Hz\n"
        msg += f"  Export success only: {self.cfg.export_success_only}\n"
        return msg

    @property
    def dataset_path(self) -> str:
        """Path to the dataset directory."""
        return str(Path(self.lerobot_data_root) / self.repo_id)

    def record_step(self, obs: EnvObs, action: EnvAction) -> None:
        """Record a step (observation-action pair) for all environments.
        
        Called by the environment's step() method to record data.
        Stores batched data efficiently (one copy for all environments).
        
        Args:
            obs: Observation from the environment (batched for all envs).
            action: Action applied to the environment (batched for all envs).
        """
        self.episode_obs_list.append(obs)
        self.episode_action_list.append(action)
    
    def on_episode_end(
        self,
        env_ids: torch.Tensor,
        terminateds: torch.Tensor,
        info: Dict[str, Any]
    ) -> None:
        """Handle episode completion and automatically save data.
        
        Called by the environment when episodes complete (done=True).
        This is the key method that makes saving automatic!
        
        Args:
            env_ids: Environment IDs that have completed episodes.
            terminateds: Termination flags (success/fail).
            info: Info dict containing success/fail information.
        """
        if len(env_ids) == 0:
            return
        
        logger.log_info(f"DatasetManager.on_episode_end() called for env_ids: {env_ids.cpu().tolist()}")
        
        # Auto-save completed episodes
        self._auto_save_episodes(
            env_ids, 
            terminateds=terminateds,
            info=info
        )

    def finalize(self) -> Optional[str]:
        """Finalize the dataset and return the save path.
        
        Called when the environment is closed. Saves any remaining episodes
        and finalizes the dataset.
        
        Returns:
            Path to the saved dataset, or None if failed.
        """
        # Save any remaining episodes if there's data
        if len(self.episode_obs_list) > 0:
            # Create dummy env_ids for all environments
            active_env_ids = torch.arange(self.num_envs, device=self.device)
            self._auto_save_episodes(active_env_ids)
        
        try:
            if self.dataset is not None:
                self.dataset.finalize()
                logger.log_info(f"Dataset finalized at: {self.dataset_path}")
                return self.dataset_path
        except Exception as e:
            logger.log_error(f"Failed to finalize dataset: {e}")
        
        return None

    def _reset_buffer(self) -> None:
        """(Internal) Reset episode buffers (clears all batched data)."""
        self.episode_obs_list.clear()
        self.episode_action_list.clear()
        logger.log_info("Reset buffers (cleared all batched data)")

    def _auto_save_episodes(
        self,
        env_ids: torch.Tensor,
        terminateds: Optional[torch.Tensor] = None,
        info: Optional[Dict[str, Any]] = None
    ) -> None:
        """Automatically save episodes for specified environments.
        
        This is the core auto-save logic that runs without manual intervention.
        Processes batched data and saves each environment as a separate episode.
        
        Args:
            env_ids: Environment IDs to save.
            terminateds: Termination flags (for determining success).
            info: Info dict containing success information.
        """
        # Check if episode has data
        if len(self.episode_obs_list) == 0:
            logger.log_warning("No episode data to save")
            return
        
        obs_list = self.episode_obs_list
        action_list = self.episode_action_list
        
        # Align obs and action (remove last obs if needed)
        if len(obs_list) > len(action_list):
            obs_list = obs_list[:-1]
        
        # Get task description
        task = self.cfg.instruction.get("lang", "unknown_task")
        
        # Prepare extra info (calculate total time for all episodes)
        extra_info = self.cfg.extra.copy() if self.cfg.extra else {}
        fps = self.dataset.meta.info.get("fps", 30)
        current_episode_time = (len(obs_list) * len(env_ids)) / fps if fps > 0 else 0
        
        episode_extra_info = extra_info.copy()
        self.total_time += current_episode_time
        episode_extra_info["total_time"] = self.total_time
        episode_extra_info["data_type"] = "sim"
        self.update_dataset_info({"extra": episode_extra_info})
        
        # Process each environment as a separate episode
        for env_id in env_ids.cpu().tolist():
            # Determine if episode was successful
            is_success = False
            if info is not None and 'success' in info:
                success_tensor = info['success']
                if isinstance(success_tensor, torch.Tensor):
                    is_success = success_tensor[env_id].item()
                else:
                    is_success = success_tensor
            elif terminateds is not None:
                is_success = terminateds[env_id].item()
            
            # Skip failed episodes if configured
            logger.log_info(f"Episode {env_id} success: {is_success}")
            if self.cfg.export_success_only and not is_success:
                logger.log_info(f"Skipping failed episode for env {env_id}")
                continue
            
            # Convert and save episode
            try:
                # Add frames for this specific environment
                for obs, action in zip(obs_list, action_list):
                    frame = self._convert_frame_to_lerobot(obs, action, task, env_id)
                    self.dataset.add_frame(frame)
                
                # Save episode for this environment
                self.dataset.save_episode()
                
                status = "successful" if is_success else "failed"
                logger.log_info(
                    f"Auto-saved {status} episode {self.curr_episode} for env {env_id} with {len(obs_list)} frames"
                )
                self.curr_episode += 1
                
            except Exception as e:
                logger.log_error(f"Failed to auto-save episode {env_id}: {e}")
        
        # Clear buffer after saving all episodes
        self._reset_buffer()

    def _initialize_dataset(self) -> None:
        """Initialize the LeRobot dataset."""
        # Extract naming components from config
        robot_type = self.cfg.robot_meta.get("robot_type", "robot")
        scene_type = self.cfg.extra.get("scene_type", "scene")
        task_description = self.cfg.extra.get("task_description", "task")

        robot_type = str(robot_type).lower().replace(" ", "_")
        task_description = str(task_description).lower().replace(" ", "_")

        # Determine lerobot data root directory
        lerobot_data_root = self.cfg.save_path
        if lerobot_data_root is None:
            lerobot_data_root = Path(HF_LEROBOT_HOME)
        else:
            lerobot_data_root = Path(lerobot_data_root)

        # Auto-increment id until the repo_id subdirectory does not exist
        dataset_id = self.cfg.id
        while True:
            repo_id = f"{robot_type}_{scene_type}_{task_description}_v{dataset_id}"
            dataset_dir = lerobot_data_root / repo_id
            if not dataset_dir.exists():
                break
            dataset_id += 1

        # Store computed values
        self.repo_id = repo_id
        self.dataset_id = dataset_id
        self.lerobot_data_root = str(lerobot_data_root)

        # Get dataset configuration
        fps = self.cfg.robot_meta.get("control_freq", 30)
        use_videos = self.cfg.use_videos
        image_writer_threads = self.cfg.image_writer_threads
        image_writer_processes = self.cfg.image_writer_processes

        # Build features
        features = self._build_features()

        try:
            # Try to create new dataset
            self.dataset = LeRobotDataset.create(
                repo_id=repo_id,
                fps=fps,
                root=str(lerobot_data_root),
                robot_type=robot_type,
                features=features,
                use_videos=use_videos,
                image_writer_threads=image_writer_threads,
                image_writer_processes=image_writer_processes,
            )
            logger.log_info(f"Created LeRobot dataset at: {lerobot_data_root / repo_id}")
        except FileExistsError:
            # Dataset already exists, load it instead
            logger.log_info(f"Dataset {repo_id} already exists at {lerobot_data_root}, loading it...")
            self.dataset = LeRobotDataset(
                repo_id=repo_id,
                root=str(lerobot_data_root),
            )
            logger.log_info(f"Loaded existing LeRobot dataset at: {lerobot_data_root / repo_id}")
        except Exception as e:
            logger.log_error(f"Failed to create/load LeRobot dataset: {e}")
            raise

    def _build_features(self) -> Dict:
        """Build LeRobot features dict from environment metadata."""
        features = {}
        extra_vision_config = self.cfg.robot_meta.get("observation", {}).get("vision", {})

        # Add image features
        for camera_name in extra_vision_config.keys():
            sensor = self._env.get_sensor(camera_name)
            is_stereo = is_stereocam(sensor)

            # Get image shape from sensor
            img_shape = (sensor.cfg.height, sensor.cfg.width, 3)

            features[camera_name] = {
                "dtype": "video" if self.cfg.use_videos else "image",
                "shape": img_shape,
                "names": ["height", "width", "channel"],
            }

            if is_stereo:
                features[get_right_name(camera_name)] = {
                    "dtype": "video" if self.cfg.use_videos else "image",
                    "shape": img_shape,
                    "names": ["height", "width", "channel"],
                }

        # Add state features (proprio)
        qpos = self._env.robot.get_qpos()
        state_dim = qpos.shape[1]

        if state_dim > 0:
            features["observation.state"] = {
                "dtype": "float32",
                "shape": (state_dim,),
                "names": ["state"],
            }

        # Add action features
        action_dim = self.cfg.robot_meta.get("arm_dofs", 7)
        features["action"] = {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": ["action"],
        }

        return features

    def _convert_frame_to_lerobot(
        self, obs: Dict[str, Any], action: Any, task: str, env_id: int
    ) -> Dict:
        """Convert a single frame from one environment to LeRobot format.
        
        Args:
            obs: Observation dict from environment (batched).
            action: Action from environment (batched).
            task: Task description.
            env_id: Environment index to extract data for.
            
        Returns:
            Single frame in LeRobot format for the specified environment.
        """
        frame = {"task": task}
        
        extra_vision_config = self.cfg.robot_meta.get("observation", {}).get("vision", {})
        arm_dofs = self.cfg.robot_meta.get("arm_dofs", 7)
        
        # Add images
        for camera_name in extra_vision_config.keys():
            if camera_name in obs.get("sensor", {}):
                sensor = self._env.get_sensor(camera_name)
                is_stereo = is_stereocam(sensor)
                
                # Process left/main camera image
                color_data = obs["sensor"][camera_name]["color"]
                if isinstance(color_data, torch.Tensor):
                    color_img = color_data[env_id][:, :, :3].cpu().numpy()
                else:
                    color_img = np.array(color_data)[env_id][:, :, :3]
                
                # Ensure uint8 format (0-255 range)
                if color_img.dtype in [np.float32, np.float64]:
                    color_img = (color_img * 255).astype(np.uint8)
                
                frame[camera_name] = color_img
                
                # Process right camera image if stereo
                if is_stereo:
                    color_right_data = obs["sensor"][camera_name]["color_right"]
                    if isinstance(color_right_data, torch.Tensor):
                        color_right_img = color_right_data[env_id][:, :, :3].cpu().numpy()
                    else:
                        color_right_img = np.array(color_right_data)[env_id][:, :, :3]
                    
                    if color_right_img.dtype in [np.float32, np.float64]:
                        color_right_img = (color_right_img * 255).astype(np.uint8)
                    
                    frame[get_right_name(camera_name)] = color_right_img
        
        # Add state (proprio)
        qpos = obs["robot"][JointType.QPOS.value]
        if isinstance(qpos, torch.Tensor):
            state_data = qpos[env_id].cpu().numpy().astype(np.float32)
        else:
            state_data = np.array(qpos)[env_id].astype(np.float32)
        
        frame["observation.state"] = state_data
        
        # Add action
        if isinstance(action, torch.Tensor):
            action_data = action[env_id, :arm_dofs].cpu().numpy()
        elif isinstance(action, np.ndarray):
            action_data = action[env_id, :arm_dofs]
        elif isinstance(action, dict):
            action_data = action.get("action", action.get("arm_action", action))
            if isinstance(action_data, torch.Tensor):
                action_data = action_data[env_id, :arm_dofs].cpu().numpy()
            elif isinstance(action_data, np.ndarray):
                action_data = action_data[env_id, :arm_dofs]
        else:
            action_data = np.array(action)[env_id, :arm_dofs]
        
        frame["action"] = action_data
        
        return frame

    def update_dataset_info(self, updates: dict) -> bool:
        """Update the LeRobot dataset's meta.info with custom key-value pairs.
        
        Args:
            updates: Dictionary of key-value pairs to add or update in meta.info.
        
        Returns:
            True if successful, False otherwise.
        
        Example:
            >>> dataset_manager.update_dataset_info({
            ...     "author": "DexForce",
            ...     "date_collected": "2025-12-22",
            ...     "custom_key": "custom_value"
            ... })
        """
        if self.dataset is None:
            logger.log_error(
                "LeRobotDataset not initialized. Cannot update dataset info."
            )
            return False

        try:
            self.dataset.meta.info.update(updates)
            logger.log_info(
                f"Successfully updated dataset info with keys: {list(updates.keys())}"
            )
            return True
        except Exception as e:
            logger.log_error(f"Failed to update dataset info: {e}")
            return False

    def _prepare_functors(self):
        pass
