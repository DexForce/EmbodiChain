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

from abc import ABC, abstractmethod
from typing import Callable, Optional
import torch
from tensordict import TensorDict

from ..utils.helper import dict_to_tensordict


class BaseCollector(ABC):
    """Abstract base class for data collectors.

    Defines the interface that all collectors must implement.
    """

    def __init__(
        self,
        env,
        policy,
        device: torch.device,
        on_step_callback: Optional[Callable] = None,
    ):
        """Initialize base collector.

        Args:
            env: Environment to collect from
            policy: Policy for action selection
            device: Device for tensor operations
            on_step_callback: Optional callback(tensordict, env_info) called after each step
        """
        self.env = env
        self.policy = policy
        self.device = device
        self.on_step_callback = on_step_callback
        if hasattr(self.policy, "bind_env"):
            self.policy.bind_env(self.env)

        # Initialize observation
        obs_dict, _ = self.env.reset()
        self.obs_tensordict = dict_to_tensordict(obs_dict, self.device)

    def _format_env_action(self, action: torch.Tensor):
        """Format policy action for the target environment.

        When an ActionManager is configured, the environment expects a mapping
        keyed by the active action term name. Otherwise, the environment expects
        a raw tensor that is applied directly as joint-space command.
        """
        action_manager = getattr(self.env, "action_manager", None)
        if action_manager is not None:
            return {action_manager.action_type: action}
        return action

    @abstractmethod
    def collect(self, num_steps: int | None = None, **kwargs) -> TensorDict:
        """Collect data from environment.

        Args:
            num_steps: Number of steps to collect (required by SyncCollector,
                ignored by AsyncCollector which collects continuously).

        Returns:
            TensorDict with collected data
        """
        pass
