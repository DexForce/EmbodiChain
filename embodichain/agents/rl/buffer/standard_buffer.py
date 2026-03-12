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

from tensordict import TensorDict

__all__ = ["RolloutBuffer"]


class RolloutBuffer:
    """Single-rollout buffer backed by a TensorDict."""

    def __init__(self) -> None:
        self._rollout: TensorDict | None = None

    def add(self, rollout: TensorDict) -> None:
        """Store a single rollout with batch shape `[num_envs, time]`."""
        if self._rollout is not None:
            raise RuntimeError("RolloutBuffer already contains a rollout.")
        self._rollout = rollout.clone()

    def get(self, flatten: bool = True) -> TensorDict:
        """Return the stored rollout and clear the buffer."""
        if self._rollout is None:
            raise RuntimeError("RolloutBuffer is empty.")

        rollout = self._rollout
        self._rollout = None

        if not flatten:
            return rollout

        total_batch = math.prod(rollout.batch_size)
        return rollout.reshape(total_batch)

    def is_full(self) -> bool:
        """Return whether a rollout is waiting to be consumed."""
        return self._rollout is not None
