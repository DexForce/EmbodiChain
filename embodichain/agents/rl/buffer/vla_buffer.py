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

import threading
import torch
from tensordict import TensorDict
from typing import Optional


class VLABuffer:
    """Rollout buffer for VLA RL with (B, T) batch-first layout.

    Stores complete rollouts to ensure correct GAE computation (GAE requires
    sequential timesteps within the same trajectory). Async collector accumulates
    T steps per env, then adds the full rollout.

    Key characteristics:
    - Rollout-level storage: Collect full rollout [T, N] before adding
    - Batch-first layout: Stores and returns [N, T, ...] for VLA training
    - Thread-safe: Async collector writes, main thread reads
    - Single rollout: When full, one rollout ready for training

    Storage layout: [N, T, ...] - batch (env) first, time second.
    """

    def __init__(
        self,
        buffer_size: int,
        device: torch.device,
        num_envs: int,
    ):
        """Initialize VLA buffer.

        Args:
            buffer_size: Total transitions per rollout (T * N)
            device: Device for tensors
            num_envs: Number of parallel environments (N)
        """
        self.buffer_size = buffer_size
        self.device = device
        self.num_envs = num_envs
        self.rollout_length = buffer_size // num_envs  # T
        if self.rollout_length * num_envs != buffer_size:
            raise ValueError(
                f"buffer_size ({buffer_size}) must be divisible by num_envs ({num_envs})"
            )

        self._rollout: Optional[TensorDict] = None  # [N, T, ...]
        self._lock = threading.Lock()

    def add_rollout(self, rollout: TensorDict) -> None:
        """Add a complete rollout. Fixed layout: [N, T] (batch-first).

        GAE requires same-trajectory timesteps; we only accept full rollouts.

        Args:
            rollout: TensorDict with batch_size=[N, T, ...]
        """
        with self._lock:
            if (
                rollout.batch_size[0] != self.num_envs
                or rollout.batch_size[1] != self.rollout_length
            ):
                raise ValueError(
                    f"Rollout shape {rollout.batch_size} does not match "
                    f"expected (N={self.num_envs}, T={self.rollout_length})"
                )
            self._rollout = rollout.to(self.device)

    def add_batch(self, transitions: TensorDict) -> None:
        """Deprecated: Use add_rollout. Batch must be a complete rollout [N, T]."""
        if len(transitions.batch_size) >= 2:
            self.add_rollout(transitions)
        else:
            raise NotImplementedError(
                "VLABuffer requires full rollout. Use add_rollout(rollout) with [N, T]."
            )

    def get(self, flatten: bool = True) -> TensorDict:
        """Get rollout from buffer (thread-safe).

        Args:
            flatten: If True, flatten to [N*T, ...] for minibatch sampling.

        Returns:
            TensorDict with batch_size=[N, T] or [N*T] when flatten=True
        """
        with self._lock:
            if self._rollout is None:
                raise ValueError("Buffer is empty")

            rollout = self._rollout
            self._rollout = None

        if flatten:
            return rollout.reshape(-1)
        return rollout

    def clear(self) -> None:
        """Clear buffer."""
        with self._lock:
            self._rollout = None

    def __len__(self) -> int:
        """Return 1 if has rollout, 0 otherwise."""
        with self._lock:
            return 1 if self._rollout is not None else 0

    def is_full(self) -> bool:
        """True when one complete rollout is ready."""
        with self._lock:
            return self._rollout is not None

    def get_num_rollouts(self) -> int:
        """Return 1 if has rollout, 0 otherwise."""
        with self._lock:
            return 1 if self._rollout is not None else 0

    def get_stats(self) -> dict:
        """Get buffer statistics."""
        with self._lock:
            has_data = self._rollout is not None
        return {
            "buffer_size": self.rollout_length * self.num_envs,
            "rollout_length": self.rollout_length,
            "num_envs": self.num_envs,
            "layout": "batch_first",
            "has_rollout": has_data,
        }
