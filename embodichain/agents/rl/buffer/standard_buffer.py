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
from typing import Optional


class RolloutBuffer:
    """Standard on-policy rollout buffer for PPO (matches mainstream implementations).

    Unlike VLA buffer which accumulates multiple rollouts with FIFO eviction,
    this buffer follows standard PPO pattern:
    - Stores exactly ONE rollout at a time
    - After training, buffer is cleared (on-policy: use once and discard)
    - Simple and efficient for normal-sized models

    Interface compatible with VLABuffer for easy switching.
    """

    def __init__(self, buffer_size: int, device: torch.device):
        """Initialize standard rollout buffer.

        Args:
            buffer_size: Not used (kept for interface compatibility)
            device: Device to store tensors on
        """
        self.device = device
        self._rollout: Optional[TensorDict] = None

    def add(self, rollout: TensorDict) -> None:
        """Add a rollout to buffer, replacing any existing rollout.

        Args:
            rollout: TensorDict with batch_size=[T, N, ...]
        """
        # Standard PPO: replace existing rollout (not accumulate)
        self._rollout = rollout.to(self.device)

    def get(self, flatten: bool = True) -> TensorDict:
        """Get rollout from buffer and clear it (standard PPO behavior).

        Args:
            flatten: If True, flatten to [batch_size, ...].
                    If False, return as [T, N, ...].

        Returns:
            TensorDict with rollout data
        """
        if self._rollout is None:
            raise ValueError("Buffer is empty")

        rollout = self._rollout

        # Clear after retrieval (on-policy: use once)
        self._rollout = None

        if flatten:
            # Flatten [T, N, ...] -> [T*N, ...]
            return rollout.reshape(-1)
        else:
            return rollout

    def clear(self) -> None:
        """Clear buffer."""
        self._rollout = None

    def is_full(self) -> bool:
        """Check if buffer has a rollout ready for training.

        Returns:
            True if buffer contains a rollout
        """
        return self._rollout is not None

    def __len__(self) -> int:
        """Return 1 if buffer has data, 0 otherwise."""
        return 1 if self._rollout is not None else 0

    def get_num_rollouts(self) -> int:
        """Return current number of rollouts in buffer (0 or 1)."""
        return 1 if self._rollout is not None else 0

    def get_num_transitions(self) -> int:
        """Return total number of transitions stored."""
        if self._rollout is None:
            return 0
        return self._rollout.batch_size[0] * self._rollout.batch_size[1]

    def get_stats(self) -> dict:
        """Get buffer statistics for logging.

        Returns:
            Dict with buffer stats
        """
        return {
            "buffer_size": 1 if self._rollout is not None else 0,
            "buffer_capacity": 1,
            "total_transitions": self.get_num_transitions(),
            "buffer_usage": 1.0 if self._rollout is not None else 0.0,
        }
