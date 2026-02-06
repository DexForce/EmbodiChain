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
from typing import Dict, Any, Callable

from tensordict import TensorDict
from embodichain.agents.rl.utils import AlgorithmCfg, compute_gae, dict_to_tensordict
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

    Following TorchRL conventions: no custom buffer class, just TensorDict operations.
    All data I/O uses TensorDict - no tensor fallback.
    """

    def __init__(self, cfg: PPOCfg, policy):
        self.cfg = cfg
        self.policy = policy
        self.device = torch.device(cfg.device)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.learning_rate)

    def collect_rollout(
        self,
        env,
        policy,
        tensordict: TensorDict,
        buffer_size: int,
        on_step_callback: Callable | None = None,
    ) -> TensorDict:
        """Collect a rollout using TensorDict data flow.

        Args:
            env: Environment to collect from
            policy: Policy to use for action selection
            tensordict: Initial TensorDict with "observation" key
            buffer_size: Number of steps to collect
            on_step_callback: Optional callback called after each step

        Returns:
            TensorDict with batch_size=[T, N] containing full rollout data
        """
        policy.train()
        current_td = tensordict
        rollout_list = []

        for t in range(buffer_size):
            # Policy forward: adds "action", "sample_log_prob", "value" to tensordict
            policy.forward(current_td)

            # Extract action for environment step
            action = current_td["action"]
            action_type = getattr(env, "action_type", "delta_qpos")
            action_dict = {action_type: action}

            # Environment step - returns tuple (env returns dict, not TensorDict)
            next_obs, reward, terminated, truncated, env_info = env.step(action_dict)

            # Convert env dict observation to TensorDict at boundary
            next_obs_td = dict_to_tensordict(next_obs, self.device)

            # Build "next" TensorDict
            done = terminated | truncated
            next_obs_for_td = next_obs_td["observation"]

            # Ensure batch_size consistency - use next_obs_td's batch_size
            batch_size = next_obs_td.batch_size[0]

            next_td = TensorDict(
                {
                    "observation": next_obs_for_td,
                    "reward": (
                        reward.float().unsqueeze(-1)
                        if reward.dim() == 1
                        else reward.float()
                    ),
                    "done": (
                        done.bool().unsqueeze(-1) if done.dim() == 1 else done.bool()
                    ),
                    "terminated": (
                        terminated.bool().unsqueeze(-1)
                        if terminated.dim() == 1
                        else terminated.bool()
                    ),
                    "truncated": (
                        truncated.bool().unsqueeze(-1)
                        if truncated.dim() == 1
                        else truncated.bool()
                    ),
                },
                batch_size=torch.Size([batch_size]),
                device=self.device,
            )

            # Compute next value for GAE (bootstrap value)
            with torch.no_grad():
                next_value_td = TensorDict(
                    {"observation": next_obs_for_td},
                    batch_size=next_td.batch_size,
                    device=self.device,
                )
                policy.get_value(next_value_td)
                next_td["value"] = next_value_td["value"]

            # Add "next" to current tensordict
            current_td["next"] = next_td

            # Store complete transition
            rollout_list.append(current_td.clone())

            # Debug: Print TensorDict structure on first step
            if len(rollout_list) == 1:
                print("\n" + "=" * 80)
                print("[DEBUG] Step 0 TensorDict Structure (Tree View)")
                print("=" * 80)
                _print_tensordict_tree(current_td, prefix="", is_last=True)
                print("=" * 80 + "\n")

            # Callback for statistics and logging
            if on_step_callback is not None:
                on_step_callback(current_td, env_info)

            # Prepare next iteration - use the converted TensorDict
            current_td = next_obs_td

        # Stack into [T, N, ...] TensorDict
        rollout = torch.stack(rollout_list, dim=0)

        print("\n" + "=" * 80)
        print(
            f"[DEBUG] Stacked Rollout (T={rollout.batch_size[0]}, N={rollout.batch_size[1]})"
        )
        print("=" * 80)
        _print_tensordict_tree(rollout, prefix="", is_last=True)
        print("=" * 80 + "\n")

        # Compute GAE advantages and returns
        rollout = compute_gae(
            rollout, gamma=self.cfg.gamma, gae_lambda=self.cfg.gae_lambda
        )

        return rollout

    def update(self, rollout: TensorDict) -> dict:
        """Update the policy using collected rollout TensorDict (TorchRL style).

        Args:
            rollout: TensorDict with batch_size=[T, N] from collect_rollout()

        Returns:
            Dictionary of training metrics
        """
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
