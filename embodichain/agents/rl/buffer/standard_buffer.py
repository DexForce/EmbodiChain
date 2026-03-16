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

import torch
from tensordict import TensorDict

from .utils import transition_view

__all__ = ["RolloutBuffer"]


class RolloutBuffer:
    """Single-rollout buffer backed by a preallocated TensorDict.

    The shared rollout uses a uniform `[num_envs, time + 1]` layout. For
    transition-only fields such as `action`, `reward`, and `done`, the final
    time index is reused as padding so the collector, environment, and
    algorithms can share a single TensorDict batch shape.
    """

    def __init__(
        self,
        num_envs: int,
        rollout_len: int,
        obs_dim: int,
        action_dim: int,
        device: torch.device,
        use_raw_obs: bool = False,
    ) -> None:
        self.num_envs = num_envs
        self.rollout_len = rollout_len
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        self.use_raw_obs = use_raw_obs
        self._rollout = self._allocate_rollout()
        self._is_full = False

    def start_rollout(self) -> TensorDict:
        """Return the shared rollout TensorDict for collector write-in."""
        if self._is_full:
            raise RuntimeError("RolloutBuffer already contains a rollout.")
        self._clear_dynamic_fields()
        if self.use_raw_obs:
            self._rollout.raw_obs = [None] * (self.rollout_len + 1)
        return self._rollout

    def add(self, rollout: TensorDict) -> None:
        """Mark the shared rollout as ready for consumption."""
        if rollout is not self._rollout:
            raise ValueError(
                "RolloutBuffer only accepts its shared rollout TensorDict."
            )
        if tuple(rollout.batch_size) != (self.num_envs, self.rollout_len + 1):
            raise ValueError(
                "Rollout batch size does not match buffer allocation: "
                f"expected ({self.num_envs}, {self.rollout_len + 1}), got {tuple(rollout.batch_size)}."
            )
        self._validate_rollout_layout(rollout)
        self._is_full = True

    def get(self, flatten: bool = True) -> TensorDict:
        """Return the stored rollout and clear the buffer.

        When `flatten` is True, the rollout is first converted to a transition
        view that drops the padded final slot from transition-only fields.
        """
        if not self._is_full:
            raise RuntimeError("RolloutBuffer is empty.")

        rollout = self._rollout
        self._is_full = False

        if not flatten:
            return rollout

        return transition_view(rollout, flatten=True)

    def is_full(self) -> bool:
        """Return whether a rollout is waiting to be consumed."""
        return self._is_full

    def _allocate_rollout(self) -> TensorDict:
        """Preallocate rollout storage with uniform `[num_envs, time + 1]` shape."""
        td = TensorDict(
            {
                "obs": torch.empty(
                    self.num_envs,
                    self.rollout_len + 1,
                    self.obs_dim,
                    dtype=torch.float32,
                    device=self.device,
                ),
                "action": torch.empty(
                    self.num_envs,
                    self.rollout_len + 1,
                    self.action_dim,
                    dtype=torch.float32,
                    device=self.device,
                ),
                "sample_log_prob": torch.empty(
                    self.num_envs,
                    self.rollout_len + 1,
                    dtype=torch.float32,
                    device=self.device,
                ),
                "value": torch.empty(
                    self.num_envs,
                    self.rollout_len + 1,
                    dtype=torch.float32,
                    device=self.device,
                ),
                "reward": torch.empty(
                    self.num_envs,
                    self.rollout_len + 1,
                    dtype=torch.float32,
                    device=self.device,
                ),
                "done": torch.empty(
                    self.num_envs,
                    self.rollout_len + 1,
                    dtype=torch.bool,
                    device=self.device,
                ),
                "terminated": torch.empty(
                    self.num_envs,
                    self.rollout_len + 1,
                    dtype=torch.bool,
                    device=self.device,
                ),
                "truncated": torch.empty(
                    self.num_envs,
                    self.rollout_len + 1,
                    dtype=torch.bool,
                    device=self.device,
                ),
            },
            batch_size=[self.num_envs, self.rollout_len + 1],
            device=self.device,
        )
        return td

    def _clear_dynamic_fields(self) -> None:
        """Drop algorithm-added fields before reusing the shared rollout."""
        for key in ("advantage", "return", "seq_mask", "seq_return", "entropy"):
            if key in self._rollout.keys():
                del self._rollout[key]
        if self.use_raw_obs and hasattr(self._rollout, "raw_obs"):
            delattr(self._rollout, "raw_obs")
        if hasattr(self._rollout, "chunk_step"):
            delattr(self._rollout, "chunk_step")
        self._reset_padding_slot()

    def _reset_padding_slot(self) -> None:
        """Reset the last transition-only slot reused as padding."""
        last_idx = self.rollout_len
        self._rollout["action"][:, last_idx].zero_()
        self._rollout["sample_log_prob"][:, last_idx].zero_()
        self._rollout["reward"][:, last_idx].zero_()
        self._rollout["done"][:, last_idx].fill_(False)
        self._rollout["terminated"][:, last_idx].fill_(False)
        self._rollout["truncated"][:, last_idx].fill_(False)

    def _validate_rollout_layout(self, rollout: TensorDict) -> None:
        """Validate the expected tensor shapes for the shared rollout."""
        expected_shapes = {
            "obs": (self.num_envs, self.rollout_len + 1, self.obs_dim),
            "action": (self.num_envs, self.rollout_len + 1, self.action_dim),
            "sample_log_prob": (self.num_envs, self.rollout_len + 1),
            "value": (self.num_envs, self.rollout_len + 1),
            "reward": (self.num_envs, self.rollout_len + 1),
            "done": (self.num_envs, self.rollout_len + 1),
            "terminated": (self.num_envs, self.rollout_len + 1),
            "truncated": (self.num_envs, self.rollout_len + 1),
        }
        for key, expected_shape in expected_shapes.items():
            actual_shape = tuple(rollout[key].shape)
            if actual_shape != expected_shape:
                raise ValueError(
                    f"Rollout field '{key}' shape mismatch: expected {expected_shape}, "
                    f"got {actual_shape}."
                )
