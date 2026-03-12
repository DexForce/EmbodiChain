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

from collections.abc import Iterator

import torch
from tensordict import TensorDict

__all__ = ["iterate_minibatches"]


def iterate_minibatches(
    rollout: TensorDict, batch_size: int, device: torch.device
) -> Iterator[TensorDict]:
    """Yield shuffled minibatches from a flattened rollout."""
    total = rollout.batch_size[0]
    indices = torch.randperm(total, device=device)
    for start in range(0, total, batch_size):
        yield rollout[indices[start : start + batch_size]]
