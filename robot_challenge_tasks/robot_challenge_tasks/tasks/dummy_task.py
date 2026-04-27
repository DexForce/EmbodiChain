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

import torch
from typing import List, Dict, Any

from embodichain.lab.gym.utils.registration import register_env
from embodichain.lab.gym.envs import EmbodiedEnv, EmbodiedEnvCfg


@register_env("DummyTask-v1", max_episode_steps=100)
class DummyTaskEnv(EmbodiedEnv):
    """
    A simple dummy task for testing and template purposes.

    This is a basic demonstration task that shows how to construct an environment
    and create a sequence of demonstration actions.
    """

    def __init__(self, cfg: EmbodiedEnvCfg = None, **kwargs):
        super().__init__(cfg, **kwargs)

    def create_demo_action_list(self, *args, **kwargs) -> List[Dict[str, Any]]:
        """Create a list of demonstration actions.

        This method generates a simple sequence of actions that can be used
        to demonstrate the task or serve as a baseline trajectory.

        Returns:
            List of action dictionaries, each containing action parameters.
        """
        demo_actions = []

        # Sample a simple trajectory from limits (num_envs, num_joints, 2)
        for i in range(100):
            action = self.action_space.sample() * 0.05  # Sample a random action

            action = torch.as_tensor(action, device=self.device)  # Convert to tensor

            demo_actions.append(action)

        return demo_actions
