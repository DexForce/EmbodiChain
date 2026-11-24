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

from embodichain.learning.rl.buffer import transition_view
from embodichain.learning.rl.policies.policy import _TrainingInputSpec

__all__ = ["compute_gae"]


def _resolve_training_input_spec(policy: torch.nn.Module) -> _TrainingInputSpec:
    """Read the policy's optional contract once for the current learner update."""
    policy_module = getattr(policy, "module", policy)
    get_spec = getattr(policy_module, "_training_input_spec", None)
    spec = _TrainingInputSpec() if get_spec is None else get_spec()
    if not isinstance(spec, _TrainingInputSpec):
        raise TypeError("_training_input_spec() must return _TrainingInputSpec.")
    return spec


def _training_transition_view(
    rollout: TensorDict, input_spec: _TrainingInputSpec, fields: tuple[str, ...]
) -> TensorDict:
    """Flatten declared policy/algorithm fields, preserving undeclared payloads.

    Removing the bootstrap slot makes environment-major storage noncontiguous;
    flattening therefore copies each field. Select before that copy and before
    subsequent minibatch gathers. Policies without a field declaration retain
    the complete historical payload, including derived transition fields.
    """
    if input_spec.fields is None:
        return transition_view(rollout, flatten=True)
    names = {*input_spec.fields, *fields}
    if "next_value" in names:
        # The transition API owns this derived field, including precedence over
        # any same-named rollout extra. Select before flattening its views.
        transitions = transition_view(rollout)
        return transitions.select(
            *(name for name in transitions.keys() if name in names)
        ).reshape(-1)
    # Keep source order for deterministic field traversal and retain missing-key
    # checks at their existing consumers.
    selected = rollout.select(*(name for name in rollout.keys() if name in names))
    return selected[:, :-1].reshape(-1)


@torch.no_grad()
def _update_observation_normalization(
    policy,
    rollout: TensorDict,
    input_spec: _TrainingInputSpec,
    flat_transitions: TensorDict | None = None,
) -> None:
    """Update policy normalizers after the rollout optimization is complete."""
    policy_module = getattr(policy, "module", policy)
    update_normalization = getattr(policy_module, "update_normalization", None)
    if update_normalization is None or input_spec.normalization_fields == ():
        return

    fields = {}
    names = input_spec.normalization_fields
    reuse_flat = (
        names is not None
        and flat_transitions is not None
        and all(name in flat_transitions.keys() for name in names)
    )
    for name in ("obs", "critic_obs") if names is None else names:
        if name not in rollout.keys():
            continue
        observations = flat_transitions[name] if reuse_flat else rollout[name][:, :-1]
        fields[name] = observations.reshape(-1, observations.shape[-1])
    observations = fields["obs"] if names is None else next(iter(fields.values()))
    update_normalization(
        TensorDict(
            fields,
            batch_size=[observations.shape[0]],
            device=observations.device,
        )
    )


def compute_gae(
    rollout: TensorDict, gamma: float, gae_lambda: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE over a rollout stored as `[num_envs, time + 1]`.

    Args:
        rollout: Rollout TensorDict where `value[:, -1]` stores the bootstrap
            value for the final observation and transition-only fields reserve
            their last slot as padding.
        gamma: Discount factor.
        gae_lambda: GAE lambda coefficient.

    Returns:
        Tuple of `(advantages, returns)`, both shaped `[num_envs, time]`.
    """
    values = rollout["value"].float()
    rewards = rollout["reward"][:, :-1].float()
    dones = rollout["done"][:, :-1].bool()
    truncated = rollout["truncated"][:, :-1].bool()

    # Native environments preserve the terminal observation before auto-reset.
    # Retain the historical fallback for external environments that do not yet
    # publish terminal observations; never bootstrap from a reset state.
    timeout_values = values[:, :-1]
    if "timeout_value" in rollout.keys():
        terminal_values = rollout["timeout_value"][:, :-1]
        timeout_values = torch.where(
            torch.isfinite(terminal_values), terminal_values, timeout_values
        )
    truncated = truncated & ~rollout["terminated"][:, :-1].bool()
    rewards = rewards + gamma * timeout_values * truncated

    if rewards.ndim != 2:
        raise ValueError(
            f"Expected reward tensor with shape [num_envs, time], got {rewards.shape}."
        )

    num_envs, time_dim = rewards.shape
    if values.shape != (num_envs, time_dim + 1):
        raise ValueError(
            "Expected value tensor with shape [num_envs, time + 1], got "
            f"{values.shape} for rewards shape {rewards.shape}."
        )
    advantages = torch.zeros_like(rollout["reward"].float())
    last_advantage = torch.zeros(num_envs, device=rewards.device, dtype=rewards.dtype)
    not_done = (~dones).float()
    deltas = rewards + gamma * values[:, 1:] * not_done - values[:, :-1]
    continuation = gamma * gae_lambda * not_done

    for t in reversed(range(time_dim)):
        last_advantage = deltas[:, t] + continuation[:, t] * last_advantage
        advantages[:, t] = last_advantage

    returns = torch.zeros_like(advantages)
    returns[:, :-1] = advantages[:, :-1] + values[:, :-1]
    rollout["advantage"] = advantages
    rollout["return"] = returns
    return advantages[:, :-1], returns[:, :-1]
