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

"""RSL-RL 5.4 VecEnv interface for the reference task deployments."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import torch
import warp as wp
from tensordict import TensorDict

if TYPE_CHECKING:
    from .runtime import IsaacVelocityEnv

__all__ = ["ReferenceRslRlEnv"]


@contextmanager
def _simulation_stream(device: torch.device):
    """Run a policy call on the simulation stream with ordered handoffs."""
    if device.type != "cuda":
        yield
        return
    caller = torch.cuda.current_stream(device)
    stream = wp.stream_to_torch(str(device))
    same_stream = caller.cuda_stream == stream.cuda_stream
    if not same_stream:
        stream.wait_stream(caller)
    try:
        with torch.cuda.stream(stream):
            yield
    finally:
        if not same_stream:
            caller.wait_stream(stream)


class ReferenceRslRlEnv:
    """Expose reference policy observations and timeout bootstrapping to RSL-RL."""

    def __init__(self, env: IsaacVelocityEnv) -> None:
        self.env = env
        self.num_envs = env.num_envs
        self.num_actions = env.velocity_task_config.action_dim
        self.max_episode_length = env.max_episode_steps
        self.device = env.device
        self.cfg = env.cfg

    @property
    def episode_length_buf(self) -> torch.Tensor:
        """Current control-step counters used by the RSL-RL runner."""
        return self.env._elapsed_steps

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor) -> None:
        self.env._elapsed_steps.copy_(value)
        self.env._episode_step.copy_(value)

    def get_observations(self) -> TensorDict:
        """Return the shared actor/critic reference policy group."""
        with _simulation_stream(self.device):
            obs = self.env.get_obs()
        return TensorDict({"policy": obs["policy"]}, batch_size=[self.num_envs])

    def step(
        self, actions: torch.Tensor
    ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Forward unbounded actions and keep time limits separate from failures."""
        with _simulation_stream(self.device):
            obs, rewards, terminated, truncated, info = self.env.step(
                actions, policy_observations_only=True
            )
        return (
            TensorDict({"policy": obs["policy"]}, batch_size=[self.num_envs]),
            rewards,
            terminated | truncated,
            {"time_outs": truncated},
        )

    def close(self) -> None:
        """Close the owned simulation environment."""
        self.env.close()
