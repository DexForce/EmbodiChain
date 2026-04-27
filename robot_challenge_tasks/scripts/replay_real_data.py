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

"""Script to replay RobotChallenge official real data in simulation environment."""

import argparse
import json
from pathlib import Path
import torch
import numpy as np

import gymnasium as gym
import robot_challenge_tasks

from embodichain.lab.sim.objects import Robot
from embodichain.lab.gym.utils.gym_utils import (
    add_env_launcher_args_to_parser,
    build_env_cfg_from_args,
)
from embodichain.lab.scripts.run_env import main as run_env_main


def map_real_data_to_env_actions(real_data, robot: Robot) -> torch.Tensor:
    """Map the real data to environment actions.

    Args:
        real_data (torch.Tensor): State tensor of shape (num_steps, dof) where dof includes
                                  joint positions and gripper state for both left and right arms.
                                  For a 6-DOF arm with 1-DOF gripper per arm:
                                  dof = 14 (7 for left arm + 7 for right arm).
        robot (Robot): The robot object to get the action mapping.

    Returns:
        List[Dict]: A list of action dictionaries that can be executed in the environment.
    """
    actions_new = torch.zeros((len(real_data), robot.dof), dtype=torch.float32)
    left_arm_indices = robot.get_joint_ids("left_arm")
    left_eef_indices = robot.get_joint_ids("left_eef", remove_mimic=True)
    right_arm_indices = robot.get_joint_ids("right_arm")
    right_eef_indices = robot.get_joint_ids("right_eef", remove_mimic=True)
    actions_new[:, left_arm_indices] = real_data[:, :6]
    actions_new[:, left_eef_indices] = real_data[:, [6]]
    actions_new[:, right_arm_indices] = real_data[:, 7:13]
    actions_new[:, right_eef_indices] = real_data[:, [13]]

    return actions_new


def read_real_data(data_path: str, episode_id: int):
    """Read the real data from the specified path and episode ID.

    Args:
        data_path (str): The path to the real data.
        episode_id (int): The episode ID to read.

    Returns:
        torch.Tensor: State tensor of shape (num_steps, dof) where dof includes
                      joint positions and gripper state for both left and right arms.
                      For a 6-DOF arm with 1-DOF gripper per arm:
                      dof = 14 (7 for left arm + 7 for right arm).
    """

    # TODO: Currently only work for dual-arm aloha.

    # Construct paths to left and right state files
    data_dir = Path(data_path) / "data" / f"episode_{episode_id:06d}" / "states"
    left_file = data_dir / "left_states.jsonl"
    right_file = data_dir / "right_states.jsonl"

    # Read left arm states
    left_states = []
    if left_file.exists():
        with open(left_file, "r") as f:
            for line in f:
                if line.strip():
                    state = json.loads(line)
                    # Extract joint positions (qpos) and gripper state
                    qpos = state.get("qpos", [])
                    gripper = state.get("gripper", 0.0)
                    left_states.append(qpos + [gripper])

    # Read right arm states
    right_states = []
    if right_file.exists():
        with open(right_file, "r") as f:
            for line in f:
                if line.strip():
                    state = json.loads(line)
                    # Extract joint positions (qpos) and gripper state
                    qpos = state.get("qpos", [])
                    gripper = state.get("gripper", 0.0)
                    right_states.append(qpos + [gripper])

    # Combine left and right states
    combined_states = []
    num_steps = max(len(left_states), len(right_states))

    for i in range(num_steps):
        left_state = left_states[i] if i < len(left_states) else [0.0] * 7
        right_state = right_states[i] if i < len(right_states) else [0.0] * 7
        combined_states.append(left_state + right_state)

    # Convert to torch tensor
    states_tensor = torch.tensor(combined_states, dtype=torch.float32)

    return states_tensor


if __name__ == "__main__":
    np.set_printoptions(precision=5, suppress=True)
    torch.set_printoptions(precision=5, sci_mode=False)

    parser = argparse.ArgumentParser()

    add_env_launcher_args_to_parser(parser)

    parser.add_argument(
        "--data_path",
        help="The path to the real data.",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--episode_id",
        help="The episode ID to replay.",
        required=True,
        type=int,
    )

    args = parser.parse_args()

    args.preview = True
    args.filter_visual_rand = True

    env_cfg, gym_config, action_config = build_env_cfg_from_args(args)

    env = gym.make(id=gym_config["id"], cfg=env_cfg, **action_config)
    _, _ = env.reset()

    robot = env.get_wrapper_attr("robot")

    actions = read_real_data(args.data_path, args.episode_id)
    env_actions = map_real_data_to_env_actions(actions, robot)

    for action in env_actions:
        obs, reward, terminated, truncated, info = env.step(action)

    from IPython import embed

    embed()
