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

from __future__ import annotations

import math

import torch
from tensordict import TensorDict

__all__ = ["RolloutBuffer"]


class RolloutBuffer:
    """Single-rollout buffer backed by a preallocated TensorDict."""

    def __init__(
        self,
        num_envs: int,
        rollout_len: int,
        obs_dim: int,
        action_dim: int,
        device: torch.device,
    ) -> None:
        self.num_envs = num_envs
        self.rollout_len = rollout_len
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        self._rollout = self._allocate_rollout()
        self._is_full = False

    def start_rollout(self) -> TensorDict:
        """Return the shared rollout TensorDict for collector write-in."""
        if self._is_full:
            raise RuntimeError("RolloutBuffer already contains a rollout.")
        self._clear_dynamic_fields()
        return self._rollout

    def add(self, rollout: TensorDict) -> None:
        """Mark the shared rollout as ready for consumption."""
        if rollout is not self._rollout:
            raise ValueError(
                "RolloutBuffer only accepts its shared rollout TensorDict."
            )
        if tuple(rollout.batch_size) != (self.num_envs, self.rollout_len):
            raise ValueError(
                "Rollout batch size does not match buffer allocation: "
                f"expected ({self.num_envs}, {self.rollout_len}), got {tuple(rollout.batch_size)}."
            )
        self._is_full = True

    def get(self, flatten: bool = True) -> TensorDict:
        """Return the stored rollout and clear the buffer."""
        if not self._is_full:
            raise RuntimeError("RolloutBuffer is empty.")

        rollout = self._rollout
        self._is_full = False

        if not flatten:
            return rollout

        total_batch = math.prod(rollout.batch_size)
        return rollout.reshape(total_batch)

    def is_full(self) -> bool:
        """Return whether a rollout is waiting to be consumed."""
        return self._is_full

    def _allocate_rollout(self) -> TensorDict:
        """Preallocate rollout storage with batch shape `[num_envs, time]`."""
        return TensorDict(
            {
                "observation": torch.empty(
                    self.num_envs,
                    self.rollout_len,
                    self.obs_dim,
                    dtype=torch.float32,
                    device=self.device,
                ),
                "action": torch.empty(
                    self.num_envs,
                    self.rollout_len,
                    self.action_dim,
                    dtype=torch.float32,
                    device=self.device,
                ),
                "sample_log_prob": torch.empty(
                    self.num_envs,
                    self.rollout_len,
                    dtype=torch.float32,
                    device=self.device,
                ),
                "value": torch.empty(
                    self.num_envs,
                    self.rollout_len,
                    dtype=torch.float32,
                    device=self.device,
                ),
                "next": {
                    "observation": torch.empty(
                        self.num_envs,
                        self.rollout_len,
                        self.obs_dim,
                        dtype=torch.float32,
                        device=self.device,
                    ),
                    "reward": torch.empty(
                        self.num_envs,
                        self.rollout_len,
                        dtype=torch.float32,
                        device=self.device,
                    ),
                    "done": torch.empty(
                        self.num_envs,
                        self.rollout_len,
                        dtype=torch.bool,
                        device=self.device,
                    ),
                    "terminated": torch.empty(
                        self.num_envs,
                        self.rollout_len,
                        dtype=torch.bool,
                        device=self.device,
                    ),
                    "truncated": torch.empty(
                        self.num_envs,
                        self.rollout_len,
                        dtype=torch.bool,
                        device=self.device,
                    ),
                    "value": torch.empty(
                        self.num_envs,
                        self.rollout_len,
                        dtype=torch.float32,
                        device=self.device,
                    ),
                },
            },
            batch_size=[self.num_envs, self.rollout_len],
            device=self.device,
        )

    def _clear_dynamic_fields(self) -> None:
        """Drop algorithm-added fields before reusing the shared rollout."""
        for key in ("advantage", "return", "seq_mask", "seq_return", "entropy"):
            if key in self._rollout.keys():
                del self._rollout[key]
