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

__all__ = ["iterate_minibatches", "transition_view"]


def transition_view(rollout: TensorDict, flatten: bool = False) -> TensorDict:
    """Build a transition-aligned TensorDict from a rollout.

    The shared rollout uses a uniform `[num_envs, time + 1]` layout. For
    transition-only fields such as `action`, `reward`, and `done`, the final
    slot is reserved as padding so that all rollout fields share the same batch
    shape. This helper drops that padded slot and exposes the valid transition
    slices as a TensorDict with batch shape `[num_envs, time]`.

    Args:
        rollout: Rollout TensorDict with root batch shape `[num_envs, time + 1]`.
        flatten: If True, return a flattened `[num_envs * time]` view.

    Returns:
        TensorDict containing transition-aligned fields.
    """
    action = rollout["action"][:, :-1]
    num_envs, time_dim = action.shape[:2]
    transition_fields = {
        "action": action,
        "sample_log_prob": rollout["sample_log_prob"][:, :-1],
        "value": rollout["value"][:, :-1],
        "next_value": rollout["value"][:, 1:],
        "reward": rollout["reward"][:, :-1],
        "done": rollout["done"][:, :-1],
        "terminated": rollout["terminated"][:, :-1],
        "truncated": rollout["truncated"][:, :-1],
    }
    if "obs" in rollout.keys():
        transition_fields["obs"] = rollout["obs"][:, :-1]
    td = TensorDict(
        transition_fields,
        batch_size=[num_envs, time_dim],
        device=rollout.device,
    )

    for key in (
        "advantage",
        "return",
        "seq_mask",
        "seq_return",
        "entropy",
        "step_repeat",
        "execute_full_chunk",
    ):
        if key in rollout.keys():
            td[key] = rollout[key][:, :-1]

    if hasattr(rollout, "chunk_step") and rollout.chunk_step is not None:
        td["chunk_step"] = rollout.chunk_step

    if "action_chunk" in rollout.keys():
        td["action_chunk"] = rollout["action_chunk"][:, :-1]

    if flatten:
        return td.reshape(num_envs * time_dim)
    return td


def iterate_minibatches(
    rollout: TensorDict, batch_size: int, device: torch.device
) -> Iterator[TensorDict]:
    """Yield shuffled minibatches from a flattened rollout."""
    total = rollout.batch_size[0]
    idx_device = rollout.device if rollout.device is not None else device
    indices = torch.randperm(total, device=idx_device)
    for start in range(0, total, batch_size):
        batch_indices = indices[start : start + batch_size]
        batch = rollout[batch_indices].clone()
        batch["_indices"] = batch_indices
        yield batch
