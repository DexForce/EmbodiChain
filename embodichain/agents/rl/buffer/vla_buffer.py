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


class VLABuffer:
    """FIFO rollout buffer for VLA RL with pre-allocated TensorDict storage.

    Uses a single pre-allocated TensorDict with circular indexing for efficient
    high-frequency transition writes. Designed for async VLA scenarios where
    model inference is slow but training is fast.

    Key characteristics:
    - Pre-allocated memory: Zero-copy writes via direct indexing
    - FIFO eviction: Circular buffer automatically overwrites oldest data
    - Transition-level storage: Each step is a separate entry
    - High-frequency writes: Optimized for async collection (no TensorDict creation overhead)

    Storage layout: Single TensorDict with shape [buffer_size, ...]
    """

    def __init__(self, buffer_size: int, device: torch.device):
        """Initialize VLA buffer with lazy allocation.

        Args:
            buffer_size: Maximum number of transitions to store
            device: Device to store tensors on
        """
        self.buffer_size = buffer_size
        self.device = device
        self.buffer: Optional[TensorDict] = None  # Lazy init on first add
        self.write_pos = 0  # Current write position (circular)
        self.size = 0  # Current valid data count
        self._total_added = 0
        self._initialized = False

    def _initialize_buffer(self, template: TensorDict) -> None:
        """Initialize buffer structure from first transition template.

        Args:
            template: First transition TensorDict to infer structure from
        """
        if self._initialized:
            return

        # Pre-allocate buffer with buffer_size
        # Template should be a single transition [key: shape]
        self.buffer = template.expand(self.buffer_size).clone()
        self._initialized = True

    def add(self, transition: TensorDict) -> None:
        """Add a single transition to buffer (high-frequency async writes).

        Args:
            transition: Single transition TensorDict (no batch dimension)
        """
        # Lazy initialization on first add
        if not self._initialized:
            self._initialize_buffer(transition.to(self.device))

        # Ensure transition is on correct device
        transition = transition.to(self.device)

        # Direct index assignment (zero-copy write)
        self.buffer[self.write_pos] = transition

        # Update circular index
        self.write_pos = (self.write_pos + 1) % self.buffer_size

        # Update size (saturates at buffer_size)
        self.size = min(self.size + 1, self.buffer_size)
        self._total_added += 1

    def add_batch(self, transitions: TensorDict) -> None:
        """Add multiple transitions at once (batch write).

        Args:
            transitions: Batch of transitions with shape [batch_size, ...]
        """
        batch_size = transitions.batch_size[0]

        # Lazy initialization
        if not self._initialized:
            self._initialize_buffer(transitions[0].to(self.device))

        transitions = transitions.to(self.device)

        # Handle circular write
        for i in range(batch_size):
            self.buffer[self.write_pos] = transitions[i]
            self.write_pos = (self.write_pos + 1) % self.buffer_size
            self.size = min(self.size + 1, self.buffer_size)
            self._total_added += 1

    def get(self, flatten: bool = True) -> TensorDict:
        """Get valid data from buffer.

        Args:
            flatten: If True, return flattened [size, ...]. Currently only supports True.

        Returns:
            TensorDict with batch_size=[size, ...] containing valid data
        """
        if not self._initialized or self.size == 0:
            raise ValueError("Buffer is empty")

        if not flatten:
            raise NotImplementedError("Only flatten=True is supported for VLABuffer")

        # Return first 'size' elements (valid data)
        # Note: Data is in insertion order up to write_pos, then wraps
        if self.size < self.buffer_size:
            # Buffer not yet full, data is [0:size]
            return self.buffer[: self.size]
        else:
            # Buffer full, need to rearrange to maintain temporal order
            # Oldest data is at write_pos, newest at write_pos-1
            indices = (
                torch.arange(
                    self.write_pos,
                    self.write_pos + self.buffer_size,
                    device=self.device,
                )
                % self.buffer_size
            )
            return self.buffer[indices]

    def clear(self) -> None:
        """Clear buffer (reset pointers, keep pre-allocated memory)."""
        self.write_pos = 0
        self.size = 0
        # Keep buffer allocated for reuse

    def __len__(self) -> int:
        """Return current number of valid transitions."""
        return self.size

    def is_full(self) -> bool:
        """Check if buffer is at full buffer_size."""
        return self.size >= self.buffer_size

    def get_num_rollouts(self) -> int:
        """Return 1 (buffer stores transitions, not rollouts)."""
        return 1 if self.size > 0 else 0

    def get_stats(self) -> dict:
        """Get buffer statistics for logging."""
        return {
            "buffer_size": self.size,
            "buffer_capacity": self.buffer_size,
            "total_transitions": self.size,
            "total_added": self._total_added,
            "buffer_usage": (
                self.size / self.buffer_size if self.buffer_size > 0 else 0.0
            ),
            "write_pos": self.write_pos,
        }
