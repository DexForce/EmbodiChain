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

from __future__ import annotations

import torch
from tensordict import TensorDict

from ..utils.helper import dict_to_tensordict
from .base import BaseCollector


class SyncCollector(BaseCollector):
    """Synchronous data collector for standard RL training.

    Collects a complete rollout of specified length, then returns it.
    Used with RolloutBuffer for standard PPO training.

    Usage:
        collector = SyncCollector(env, policy, device)
        rollout = collector.collect(num_steps=2048)
        buffer.add(rollout)
    """

    def collect(self, num_steps: int) -> TensorDict:
        """Collect a synchronous rollout.

        Args:
            num_steps: Number of steps to collect

        Returns:
            TensorDict with batch_size=[T, N] containing full rollout
        """
        self.policy.train()
        current_td = self.obs_tensordict
        rollout_list = []

        for t in range(num_steps):
            # Policy forward: adds "action", "sample_log_prob", "value" to tensordict
            self.policy.forward(current_td)

            # Extract action for environment step
            action = current_td["action"]
            action_type = getattr(self.env, "action_type", "delta_qpos")
            action_dict = {action_type: action}

            # Environment step - returns tuple (env returns dict, not TensorDict)
            next_obs, reward, terminated, truncated, env_info = self.env.step(
                action_dict
            )

            # Convert env dict observation to TensorDict at boundary
            next_obs_td = dict_to_tensordict(next_obs, self.device)

            # Build "next" TensorDict
            done = terminated | truncated
            next_obs_for_td = next_obs_td["observation"]

            # Ensure batch_size consistency - use next_obs_td's batch_size
            batch_size = next_obs_td.batch_size[0]

            next_td = TensorDict(
                {
                    "observation": next_obs_for_td,
                    "reward": (
                        reward.float().unsqueeze(-1)
                        if reward.dim() == 1
                        else reward.float()
                    ),
                    "done": (
                        done.bool().unsqueeze(-1) if done.dim() == 1 else done.bool()
                    ),
                    "terminated": (
                        terminated.bool().unsqueeze(-1)
                        if terminated.dim() == 1
                        else terminated.bool()
                    ),
                    "truncated": (
                        truncated.bool().unsqueeze(-1)
                        if truncated.dim() == 1
                        else truncated.bool()
                    ),
                },
                batch_size=torch.Size([batch_size]),
                device=self.device,
            )

            # Compute next value for GAE (bootstrap value)
            with torch.no_grad():
                next_value_td = TensorDict(
                    {"observation": next_obs_for_td},
                    batch_size=next_td.batch_size,
                    device=self.device,
                )
                self.policy.get_value(next_value_td)
                next_td["value"] = next_value_td["value"]

            # Add "next" to current tensordict
            current_td["next"] = next_td

            # Store complete transition
            rollout_list.append(current_td.clone())

            # Callback for statistics and logging
            if self.on_step_callback is not None:
                self.on_step_callback(current_td, env_info)

            # Prepare next iteration - use the converted TensorDict
            current_td = next_obs_td

        # Update observation for next collection
        self.obs_tensordict = current_td

        # Stack into [T, N, ...] TensorDict
        rollout = torch.stack(rollout_list, dim=0)

        return rollout
