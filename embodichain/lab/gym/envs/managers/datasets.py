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

"""Dataset functors for collecting and saving episode data."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import numpy as np
import torch

from embodichain.utils import logger
from embodichain.data.constants import EMBODICHAIN_DEFAULT_DATASET_ROOT
from embodichain.lab.sim.types import EnvObs, EnvAction
from embodichain.lab.gym.utils.misc import is_stereocam
from embodichain.utils.utility import get_right_name
from embodichain.data.enum import JointType
from .manager_base import Functor
from .cfg import DatasetFunctorCfg

if TYPE_CHECKING:
    from embodichain.lab.gym.envs import EmbodiedEnv

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, HF_LEROBOT_HOME

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
                - export_success_only: Whether to export only successful episodes
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
        self.use_videos = params.get("use_videos", False)
        self.export_success_only = params.get("export_success_only", False)

        # Episode data buffers
        self.episode_obs_list: List[Dict] = []
        self.episode_action_list: List[Any] = []

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

    def reset(self, env_ids: Optional[torch.Tensor] = None) -> None:
        """Reset the recorder buffers.

        Args:
            env_ids: Environment IDs to reset (currently clears all data).
        """
        self._reset_buffer()

    def __call__(
        self,
        env: EmbodiedEnv,
        env_ids: Union[torch.Tensor, None],
        obs: EnvObs,
        action: EnvAction,
        dones: torch.Tensor,
        terminateds: torch.Tensor,
        info: Dict[str, Any],
        save_path: Optional[str] = None,
        id: Optional[str] = None,
        robot_meta: Optional[Dict] = None,
        instruction: Optional[str] = None,
        extra: Optional[Dict] = None,
        use_videos: bool = False,
        export_success_only: bool = False,
    ) -> None:
        """Main entry point for the recorder functor.

        This method is called by DatasetManager.apply(mode="save") with runtime arguments
        as positional parameters and configuration parameters from cfg.params.

        Args:
            env: The environment instance.
            env_ids: Environment IDs (for consistency with EventManager pattern).
            obs: Observation from the environment.
            action: Action applied to the environment.
            dones: Boolean tensor indicating which envs completed episodes.
            terminateds: Termination flags (success/fail).
            info: Info dict containing success/fail information.
            save_path: Root directory (already set in __init__).
            id: Dataset identifier (already set in __init__).
            robot_meta: Robot metadata (already set in __init__).
            instruction: Task instruction (already set in __init__).
            extra: Extra metadata (already set in __init__).
            use_videos: Whether to save videos (already set in __init__).
            export_success_only: Whether to export only successful episodes (already set in __init__).
        """
        # Always record the step
        self._record_step(obs, action)

        # Check if any episodes are done and save them
        done_env_ids = dones.nonzero(as_tuple=False).squeeze(-1)
        if len(done_env_ids) > 0:
            # Save completed episodes
            self._save_episodes(done_env_ids, terminateds, info)

    def _record_step(self, obs: EnvObs, action: EnvAction) -> None:
        """Record a single step."""
        self.episode_obs_list.append(obs)
        self.episode_action_list.append(action)

    def _save_episodes(
        self,
        env_ids: torch.Tensor,
        terminateds: Optional[torch.Tensor] = None,
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save completed episodes."""
        if len(self.episode_obs_list) == 0:
            logger.log_warning("No episode data to save")
            return

        obs_list = self.episode_obs_list
        action_list = self.episode_action_list

        # Align obs and action
        if len(obs_list) > len(action_list):
            obs_list = obs_list[:-1]

        task = self.instruction.get("lang", "unknown_task")

        # Update metadata
        extra_info = self.extra.copy() if self.extra else {}
        fps = self.dataset.meta.info.get("fps", 30)
        current_episode_time = (len(obs_list) * len(env_ids)) / fps if fps > 0 else 0

        episode_extra_info = extra_info.copy()
        self.total_time += current_episode_time
        episode_extra_info["total_time"] = self.total_time
        self._update_dataset_info({"extra": episode_extra_info})

        # Process each environment
        for env_id in env_ids.cpu().tolist():
            is_success = False
            if info is not None and "success" in info:
                success_tensor = info["success"]
                if isinstance(success_tensor, torch.Tensor):
                    is_success = success_tensor[env_id].item()
                else:
                    is_success = success_tensor
            elif terminateds is not None:
                is_success = terminateds[env_id].item()

            if self.export_success_only and not is_success:
                logger.log_info(f"Skipping failed episode for env {env_id}")
                continue

            try:
                for obs, action in zip(obs_list, action_list):
                    frame = self._convert_frame_to_lerobot(obs, action, task, env_id)
                    self.dataset.add_frame(frame)

                self.dataset.save_episode()
                logger.log_info(
                    f"Auto-saved {'successful' if is_success else 'failed'} "
                    f"episode {self.curr_episode} for env {env_id} with {len(obs_list)} frames"
                )
                self.curr_episode += 1
            except Exception as e:
                logger.log_error(f"Failed to save episode {env_id}: {e}")

        self._reset_buffer()

    def finalize(self) -> Optional[str]:
        """Finalize the dataset."""
        if len(self.episode_obs_list) > 0:
            active_env_ids = torch.arange(self.num_envs, device=self.device)
            self._save_episodes(active_env_ids)

        try:
            if self.dataset is not None:
                self.dataset.finalize()
                logger.log_info(f"Dataset finalized at: {self.dataset_path}")
                return self.dataset_path
        except Exception as e:
            logger.log_error(f"Failed to finalize dataset: {e}")

        return None

    def _reset_buffer(self) -> None:
        """Reset episode buffers."""
        self.episode_obs_list.clear()
        self.episode_action_list.clear()
        logger.log_info("Reset buffers (cleared all batched data)")

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
        extra_vision_config = self.robot_meta.get("observation", {}).get("vision", {})

        for camera_name in extra_vision_config.keys():
            sensor = self._env.get_sensor(camera_name)
            is_stereo = is_stereocam(sensor)
            img_shape = (sensor.cfg.height, sensor.cfg.width, 3)

            features[camera_name] = {
                "dtype": "video" if self.use_videos else "image",
                "shape": img_shape,
                "names": ["height", "width", "channel"],
            }

            if is_stereo:
                features[get_right_name(camera_name)] = {
                    "dtype": "video" if self.use_videos else "image",
                    "shape": img_shape,
                    "names": ["height", "width", "channel"],
                }

        qpos = self._env.robot.get_qpos()
        state_dim = qpos.shape[1]

        if state_dim > 0:
            features["observation.state"] = {
                "dtype": "float32",
                "shape": (state_dim,),
                "names": ["state"],
            }

        action_dim = self.robot_meta.get("arm_dofs", 7)
        features["action"] = {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": ["action"],
        }

        return features

    def _convert_frame_to_lerobot(
        self, obs: Dict[str, Any], action: Any, task: str, env_id: int
    ) -> Dict:
        """Convert a single frame to LeRobot format."""
        frame = {"task": task}
        extra_vision_config = self.robot_meta.get("observation", {}).get("vision", {})
        arm_dofs = self.robot_meta.get("arm_dofs", 7)

        # Add images
        for camera_name in extra_vision_config.keys():
            if camera_name in obs.get("sensor", {}):
                sensor = self._env.get_sensor(camera_name)
                is_stereo = is_stereocam(sensor)

                color_data = obs["sensor"][camera_name]["color"]
                if isinstance(color_data, torch.Tensor):
                    color_img = color_data[env_id][:, :, :3].cpu().numpy()
                else:
                    color_img = np.array(color_data)[env_id][:, :, :3]

                if color_img.dtype in [np.float32, np.float64]:
                    color_img = (color_img * 255).astype(np.uint8)

                frame[camera_name] = color_img

                if is_stereo:
                    color_right_data = obs["sensor"][camera_name]["color_right"]
                    if isinstance(color_right_data, torch.Tensor):
                        color_right_img = (
                            color_right_data[env_id][:, :, :3].cpu().numpy()
                        )
                    else:
                        color_right_img = np.array(color_right_data)[env_id][:, :, :3]

                    if color_right_img.dtype in [np.float32, np.float64]:
                        color_right_img = (color_right_img * 255).astype(np.uint8)

                    frame[get_right_name(camera_name)] = color_right_img

        # Add state
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
