#!/usr/bin/env python3
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

"""
Script to replay LeRobot dataset trajectories in EmbodiedEnv.

This script loads a LeRobot dataset and replays the recorded trajectories
in the EmbodiedEnv environment. It focuses on trajectory replay and uses
sensor configurations from the environment config file.

Usage:
    python replay_dataset.py --dataset_path /path/to/dataset --config /path/to/gym_config.json
    python replay_dataset.py --dataset_path outputs/commercial_cobotmagic_pour_water_001 --config configs/gym/pour_water/gym_config.json --episode 0
"""

import os
import argparse
import gymnasium
import torch
import numpy as np
from pathlib import Path

from embodichain.utils.logger import log_warning, log_info, log_error
from embodichain.utils.utility import load_json
from embodichain.lab.gym.envs import EmbodiedEnvCfg
from embodichain.lab.gym.utils.gym_utils import (
    config_to_cfg,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Replay LeRobot dataset in EmbodiedEnv")
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the LeRobot dataset directory"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the gym config JSON file (for environment setup)"
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=None,
        help="Specific episode index to replay (default: replay all episodes)"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode without rendering"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Frames per second for replay (default: use dataset fps)"
    )
    parser.add_argument(
        "--save_video",
        action="store_true",
        help="Save replay as video"
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default="./replay_videos",
        help="Path to save replay videos"
    )
    return parser.parse_args()


def load_lerobot_dataset(dataset_path):
    """Load LeRobot dataset from the given path.
    
    Args:
        dataset_path: Path to the LeRobot dataset directory
        
    Returns:
        LeRobotDataset instance
    """
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as e:
        log_error(
            f"Failed to import LeRobot: {e}. "
            "Please install lerobot: pip install lerobot"
        )
        return None
    
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        log_error(f"Dataset path does not exist: {dataset_path}")
        return None

    # Get repo_id from the dataset path (last directory name)
    repo_id = dataset_path.name
    # root = str(dataset_path.parent)

    log_info(f"Loading LeRobot dataset: {repo_id} from {dataset_path}")

    try:
        dataset = LeRobotDataset(repo_id=repo_id, root=dataset_path)
        log_info(f"Dataset loaded successfully:")
        log_info(f"  - Total episodes: {dataset.meta.info.get('total_episodes', 'N/A')}")
        log_info(f"  - Total frames: {dataset.meta.info.get('total_frames', 'N/A')}")
        log_info(f"  - FPS: {dataset.meta.info.get('fps', 'N/A')}")
        log_info(f"  - Robot type: {dataset.meta.info.get('robot_type', 'N/A')}")
        return dataset
    except Exception as e:
        log_error(f"Failed to load dataset: {e}")
        return None


def create_replay_env(config_path, headless=False):
    """Create EmbodiedEnv for replay based on config.
    
    Args:
        config_path: Path to the gym config JSON file
        headless: Whether to run in headless mode
        
    Returns:
        Gymnasium environment instance
    """
    # Load configuration
    gym_config = load_json(config_path)
    
    # Disable dataset recording during replay
    if "dataset" in gym_config.get("env", {}):
        gym_config["env"]["dataset"] = None
    
    # Convert config to dataclass
    cfg: EmbodiedEnvCfg = config_to_cfg(gym_config)
    
    # Set render mode
    if not headless:
        cfg.render_mode = "human"
    else:
        cfg.render_mode = None
    
    # Create environment
    log_info(f"Creating environment: {gym_config['id']}")
    env = gymnasium.make(id=gym_config["id"], cfg=cfg)
    
    return env


def replay_episode(env, dataset, episode_idx, fps=None, save_video=False, video_path=None):
    """Replay a single episode from the dataset.
    
    Args:
        env: EmbodiedEnv instance
        dataset: LeRobotDataset instance
        episode_idx: Episode index to replay
        fps: Frames per second for replay
        save_video: Whether to save replay as video
        video_path: Path to save video
        
    Returns:
        True if replay was successful, False otherwise
    """
    # Get episode data
    try:
        ep_meta = dataset.meta.episodes[episode_idx]
        start_idx = ep_meta["dataset_from_index"]
        end_idx = ep_meta["dataset_to_index"]
        episode_data = [dataset[i] for i in range(start_idx, end_idx)]
        log_info(f"Replaying episode {episode_idx} with {len(episode_data)} frames")
    except Exception as e:
        log_error(f"Failed to load episode {episode_idx}: {e}")
        return False
    
    # Reset environment
    obs, info = env.reset()
    
    # Setup video recording if needed
    if save_video and video_path:
        os.makedirs(video_path, exist_ok=True)
        video_file = os.path.join(video_path, f"episode_{episode_idx:04d}.mp4")
        # TODO: Implement video recording
        log_warning("Video recording is not yet implemented")
    
    # Replay trajectory
    for frame_idx in range(len(episode_data)):
        # Get action from dataset
        frame = episode_data[frame_idx]
        
        # Extract action based on dataset action space
        # The action format depends on the dataset's robot configuration
        if "action" in frame:
            action = frame["action"]
            if isinstance(action, torch.Tensor):
                action = action.cpu().numpy()
        else:
            log_warning(f"No action found in frame {frame_idx}, skipping")
            continue
        
        # Step environment with recorded action
        obs, reward, done, truncated, info = env.step(action)
        
        # Optional: Add delay to match FPS
        if fps:
            import time
            time.sleep(1.0 / fps)
        
        # Check if episode ended
        if done or truncated:
            log_info(f"Episode ended at frame {frame_idx}/{len(episode_data)}")
            break

    log_info(f"Successfully replayed episode {episode_idx}")
    return True


def main():
    """Main function to replay LeRobot dataset."""
    args = parse_args()
    
    # Load dataset
    dataset = load_lerobot_dataset(args.dataset_path)
    if dataset is None:
        return
    
    # Create replay environment
    env = create_replay_env(args.config, headless=args.headless)
    
    # Determine FPS
    fps = args.fps if args.fps else dataset.meta.info.get("fps", 30)
    log_info(f"Replay FPS: {fps}")
    
    # Replay episodes
    if args.episode is not None:
        # Replay single episode
        log_info(f"Replaying single episode: {args.episode}")
        success = replay_episode(
            env, 
            dataset, 
            args.episode, 
            fps=fps,
            save_video=args.save_video,
            video_path=args.video_path
        )
        if not success:
            log_error(f"Failed to replay episode {args.episode}")
    else:
        # Replay all episodes
        total_episodes = dataset.meta.info.get("total_episodes", 0)
        log_info(f"Replaying all {total_episodes} episodes")

        for episode_idx in range(total_episodes):
            log_info(f"\n{'='*60}")
            log_info(f"Episode {episode_idx + 1}/{total_episodes}")
            log_info(f"{'='*60}")

            success = replay_episode(
                env,
                dataset,
                episode_idx,
                fps=fps,
                save_video=args.save_video,
                video_path=args.video_path
            )
            
            if not success:
                log_warning(f"Skipping episode {episode_idx} due to errors")
                continue
    
    # Cleanup
    env.close()
    log_info("Replay completed successfully!")


if __name__ == "__main__":
    main()
