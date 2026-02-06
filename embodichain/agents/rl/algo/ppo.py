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

import torch
from tensordict import TensorDict
from embodichain.agents.rl.utils import AlgorithmCfg, compute_gae
from embodichain.utils import configclass
from .base import BaseAlgorithm


def _print_tensordict_tree(td, prefix="", is_last=True, name="TensorDict"):
    """Recursively print TensorDict structure in tree format."""
    connector = "└── " if is_last else "├── "

    # Print current node
    batch_info = (
        f"batch_size={list(td.batch_size)}" if hasattr(td, "batch_size") else ""
    )
    device_info = f"device={td.device}" if hasattr(td, "device") else ""
    meta_info = ", ".join(filter(None, [batch_info, device_info]))
    print(f"{prefix}{connector}{name}: TensorDict ({meta_info})")

    # Prepare prefix for children
    extension = "    " if is_last else "│   "
    new_prefix = prefix + extension

    # Get all keys
    keys = sorted(td.keys()) if hasattr(td, "keys") else []

    for i, key in enumerate(keys):
        is_last_child = i == len(keys) - 1
        value = td[key]

        if isinstance(value, TensorDict):
            # Recursively print nested TensorDict
            _print_tensordict_tree(value, new_prefix, is_last_child, name=key)
        elif isinstance(value, torch.Tensor):
            # Print tensor info
            child_connector = "└── " if is_last_child else "├── "
            shape_str = "x".join(map(str, value.shape))
            dtype_str = str(value.dtype).replace("torch.", "")
            print(
                f"{new_prefix}{child_connector}{key}: Tensor([{shape_str}], {dtype_str})"
            )
        else:
            # Print other types
            child_connector = "└── " if is_last_child else "├── "
            print(f"{new_prefix}{child_connector}{key}: {type(value).__name__}")


@configclass
class PPOCfg(AlgorithmCfg):
    """Configuration for the PPO algorithm."""

    n_epochs: int = 10
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5


class PPO(BaseAlgorithm):
    """PPO algorithm using TensorDict for all data flow.
    Data collection is handled by Collector classes (SyncCollector/AsyncCollector).
    """

    def __init__(self, cfg: PPOCfg, policy):
        self.cfg = cfg
        self.policy = policy
        self.device = torch.device(cfg.device)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.learning_rate)

    def update(self, rollout: TensorDict) -> dict:
        """Update the policy using collected rollout TensorDict (TorchRL style).

        Args:
            rollout: TensorDict with batch_size=[T, N] from collect_rollout()
                    OR [size] from VLA buffer

        Returns:
            Dictionary of training metrics
        """
        # Ensure 2D format [T, N] for GAE computation
        if len(rollout.batch_size) == 1:
            rollout = rollout.unsqueeze(1)  # [size] -> [size, 1]

        # Compute GAE advantages and returns
        rollout = compute_gae(
            rollout, gamma=self.cfg.gamma, gae_lambda=self.cfg.gae_lambda
        )

        # Flatten to [T*N, ...] for training
        flat_data = rollout.reshape(-1)
        total_samples = flat_data.batch_size[0]

        # Normalize advantages globally
        advantages = flat_data["advantage"]
        advantages_normalized = (advantages - advantages.mean()) / (
            advantages.std() + 1e-8
        )
        flat_data["advantage"] = advantages_normalized

        total_actor_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_steps = 0
        total_clip_fraction = 0.0
        total_approx_kl = 0.0

        for epoch in range(self.cfg.n_epochs):
            # Shuffle data each epoch
            indices = torch.randperm(total_samples, device=self.device)
            shuffled_data = flat_data[indices]

            # Iterate over minibatches
            num_minibatches = total_samples // self.cfg.batch_size
            for i in range(num_minibatches):
                start_idx = i * self.cfg.batch_size
                end_idx = start_idx + self.cfg.batch_size
                batch_td = shuffled_data[start_idx:end_idx]

                # Extract data from TensorDict batch
                old_logprobs = batch_td["sample_log_prob"]
                returns = batch_td["value_target"]
                advantages = batch_td[
                    "advantage"
                ]  # Note: advantages are already normalized globally before shuffling

                # Evaluate actions with current policy
                self.policy.evaluate_actions(batch_td)

                # Get updated values
                logprobs = batch_td["sample_log_prob"]
                entropy = batch_td["entropy"]
                values = batch_td["value"]

                # Ensure shapes match (squeeze if needed)
                if old_logprobs.dim() > 1:
                    old_logprobs = old_logprobs.squeeze(-1)
                if logprobs.dim() > 1:
                    logprobs = logprobs.squeeze(-1)
                if values.dim() > 1:
                    values = values.squeeze(-1)
                if returns.dim() > 1:
                    returns = returns.squeeze(-1)
                if advantages.dim() > 1:
                    advantages = advantages.squeeze(-1)
                if entropy.dim() > 1:
                    entropy = entropy.squeeze(-1)

                # PPO loss computation
                ratio = (logprobs - old_logprobs).exp()
                surr1 = ratio * advantages
                surr2 = (
                    torch.clamp(
                        ratio, 1.0 - self.cfg.clip_coef, 1.0 + self.cfg.clip_coef
                    )
                    * advantages
                )
                actor_loss = -torch.min(surr1, surr2).mean()
                value_loss = torch.nn.functional.mse_loss(values, returns)
                entropy_loss = -entropy.mean()

                # Diagnostics
                with torch.no_grad():
                    clip_fraction = (
                        ((ratio - 1.0).abs() > self.cfg.clip_coef).float().mean()
                    )
                    approx_kl = ((ratio - 1.0) - (logprobs - old_logprobs)).mean()

                loss = (
                    actor_loss
                    + self.cfg.vf_coef * value_loss
                    + self.cfg.ent_coef * entropy_loss
                )

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.cfg.max_grad_norm
                )
                self.optimizer.step()

                bs = batch_td.batch_size[0]
                total_actor_loss += actor_loss.item() * bs
                total_value_loss += value_loss.item() * bs
                total_entropy += (-entropy_loss.item()) * bs
                total_clip_fraction += clip_fraction.item() * bs
                total_approx_kl += approx_kl.item() * bs
                total_steps += bs

        return {
            "actor_loss": total_actor_loss / max(1, total_steps),
            "value_loss": total_value_loss / max(1, total_steps),
            "entropy": total_entropy / max(1, total_steps),
            "clip_fraction": total_clip_fraction / max(1, total_steps),
            "approx_kl": total_approx_kl / max(1, total_steps),
        }
