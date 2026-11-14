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

"""Pathwise actor optimization with terminal values and detached lambda targets."""

from __future__ import annotations

from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
import time
from typing import Any, Iterator

import torch
from tensordict import TensorDict

from embodichain.learning.rl.collector import DifferentiableRollout
from embodichain.learning.rl.utils import (
    LRSchedulerCfg,
    OptimizerCfg,
    build_optimizer,
    coerce_optimizer_cfg,
)
from embodichain.utils import configclass
from .apg import APG, APGCfg

__all__ = ["APGActorCritic", "APGActorCriticCfg", "differentiable_lambda_returns"]


def differentiable_lambda_returns(
    rewards: torch.Tensor,
    next_values: torch.Tensor,
    continuation: torch.Tensor,
    done: torch.Tensor,
    *,
    gamma: float,
    gae_lambda: float,
) -> torch.Tensor:
    """Compute time-major lambda returns with episode-local bootstrap.

    Args:
        rewards: Rewards shaped ``[time, environments]``.
        next_values: Values of successors captured before auto-reset.
        continuation: Soft continuation times hard-termination and validity masks.
        done: Episode boundaries. Truncations bootstrap but stop the recursion.
        gamma: Discount per control step.
        gae_lambda: Mixing coefficient between one-step and longer targets.

    Returns:
        Returns with the same shape as rewards. Detachment is caller-owned.
    """
    if (
        rewards.shape != next_values.shape
        or rewards.shape != continuation.shape
        or rewards.shape != done.shape
    ):
        raise ValueError("Lambda-return fields must have matching shapes.")
    if rewards.ndim != 2 or rewards.shape[0] == 0:
        raise ValueError("Lambda returns require a non-empty time-major batch.")
    future = next_values[-1]
    returns = []
    for index in range(rewards.shape[0] - 1, -1, -1):
        future = torch.where(done[index], next_values[index], future)
        future = rewards[index] + gamma * continuation[index] * (
            (1.0 - gae_lambda) * next_values[index] + gae_lambda * future
        )
        returns.append(future)
    return torch.stack(returns[::-1])


@configclass
class APGActorCriticCfg(APGCfg):
    """Actor and critic optimization settings for short differentiable rollouts.

    ``normalize_return`` divides the actor objective by the collected horizon.
    Critic targets retain unscaled discounted reward units. Critic normalization
    statistics can be updated after actor backpropagation, before forming targets.
    """

    critic_optimizer: OptimizerCfg = OptimizerCfg(learning_rate=2e-4)
    critic_epochs: int = 5
    critic_minibatches: int = 4
    critic_max_grad_norm: float = 1.0
    actor_grad_value_clip: float = 0.0
    normalize_return: bool = True
    update_critic_normalization: bool = False


@dataclass
class _CriticBatch:
    inputs: torch.Tensor
    successors: torch.Tensor
    rewards: torch.Tensor
    continuation: torch.Tensor
    done: torch.Tensor
    weights: torch.Tensor


class APGActorCritic(APG):
    """Optimize an actor through rewards and a frozen terminal-value function.

    Policies expose disjoint actor/critic ``optimization_parameter_groups()``
    and ``get_value(TensorDict)``. Each segment bootstraps its final successor;
    complete rollouts ignore rows after their first episode boundary. Critic
    fitting uses detached inputs, lambda targets and soft-survival weights.
    """

    def __init__(self, cfg: APGActorCriticCfg, policy: torch.nn.Module) -> None:
        if not hasattr(policy, "optimization_parameter_groups"):
            raise TypeError(
                "APGActorCritic requires actor and critic parameter groups."
            )
        actor, critic = policy.optimization_parameter_groups()
        self._actor_parameters, self._critic_parameters = tuple(actor), tuple(critic)
        if not actor or not critic or set(map(id, actor)) & set(map(id, critic)):
            raise ValueError(
                "Actor and critic parameter groups must be non-empty and disjoint."
            )
        if not 0.0 <= cfg.gamma <= 1.0 or not 0.0 <= cfg.gae_lambda <= 1.0:
            raise ValueError("gamma and gae_lambda must lie in [0, 1].")
        if cfg.critic_epochs < 1 or cfg.critic_minibatches < 1:
            raise ValueError("Critic epochs and minibatches must be positive.")
        if cfg.critic_max_grad_norm <= 0 or cfg.actor_grad_value_clip < 0:
            raise ValueError("Invalid actor or critic gradient clipping limit.")
        if cfg.update_critic_normalization and not callable(
            getattr(policy, "update_critic_normalization", None)
        ):
            raise TypeError("Policy must expose update_critic_normalization().")
        super().__init__(cfg, policy)
        cfg.critic_optimizer = coerce_optimizer_cfg(cfg.critic_optimizer)
        self.critic_optimizer = build_optimizer(
            self._critic_parameters, cfg.critic_optimizer
        )
        self._critic_batches: list[_CriticBatch] = []

    def _optimization_parameters(self) -> tuple[torch.nn.Parameter, ...]:
        return self._actor_parameters

    def rollout_context(self) -> AbstractContextManager[None]:
        """Keep critic parameters and statistics fixed during actor collection."""
        return self._frozen_critic()

    @contextmanager
    def _frozen_critic(self) -> Iterator[None]:
        flags = [parameter.requires_grad for parameter in self._critic_parameters]
        modules = []
        for name in ("critic", "critic_obs_normalizer"):
            module = getattr(self.policy, name, None)
            if isinstance(module, torch.nn.Module):
                modules.extend(module.modules())
        modes = [(module, module.training) for module in modules]
        try:
            for module, _ in modes:
                module.training = False
            for parameter in self._critic_parameters:
                parameter.requires_grad_(False)
            yield
        finally:
            for module, mode in modes:
                module.training = mode
            for parameter, flag in zip(self._critic_parameters, flags):
                parameter.requires_grad_(flag)

    def _value(self, observation: torch.Tensor) -> torch.Tensor:
        fields = TensorDict(
            {"obs": observation, "critic_obs": observation},
            batch_size=observation.shape[:-1],
        )
        return self.policy.get_value(fields)["value"]

    def begin_update(self) -> None:
        super().begin_update()
        self.critic_optimizer.zero_grad(set_to_none=True)
        self._critic_batches = []

    def accumulate_segment(self, rollout: DifferentiableRollout) -> None:
        """Accumulate a bootstrapped segment, restarting weights after reset."""
        self._accumulate(
            rollout, complete=False, objective_scale=1.0, accumulation_scale=1.0
        )

    def accumulate_complete_rollout(
        self,
        rollout: DifferentiableRollout,
        *,
        objective_scale: float | torch.Tensor = 1.0,
        accumulation_scale: float = 1.0,
    ) -> None:
        """Accumulate one independent rollout through its first episode boundary."""
        self._accumulate(
            rollout,
            complete=True,
            objective_scale=objective_scale,
            accumulation_scale=accumulation_scale,
        )

    def _accumulate(
        self,
        rollout: DifferentiableRollout,
        *,
        complete: bool,
        objective_scale: float | torch.Tensor,
        accumulation_scale: float,
    ) -> None:
        if not self._update_active:
            raise RuntimeError("Call begin_update() before accumulating a rollout.")
        if not rollout.num_steps or accumulation_scale <= 0:
            raise ValueError(
                "A non-empty rollout and positive accumulation scale are required."
            )
        transitions = rollout.transitions
        observations, successors = [], []
        missing_terminal = [
            step.truncated
            for step in transitions
            if step.terminal_critic_observation is None
        ]
        if missing_terminal and bool(torch.stack(missing_terminal).any()):
            raise ValueError(
                "Truncation requires a pre-reset terminal_critic_observation."
            )
        for step in transitions:
            observations.append(
                step.critic_observation
                if step.critic_observation is not None
                else step.observation
            )
            successor = (
                step.next_critic_observation
                if step.next_critic_observation is not None
                else step.next_observation
            )
            if step.terminal_critic_observation is not None:
                successor = torch.where(
                    step.done.unsqueeze(-1), step.terminal_critic_observation, successor
                )
            successors.append(successor)
        inputs = torch.stack([observation.detach() for observation in observations])
        critic_successors = torch.stack(
            [successor.detach() for successor in successors]
        )
        rewards = rollout.rewards
        ones = torch.ones_like(rewards[0])
        valid = torch.stack(
            [
                ones if step.valid is None else step.valid.to(ones)
                for step in transitions
            ]
        ).detach()
        next_valid = torch.stack(
            [
                ones if step.next_valid is None else step.next_valid.to(ones)
                for step in transitions
            ]
        ).detach()
        soft = torch.stack(
            [
                ones if step.continuation is None else step.continuation.to(ones)
                for step in transitions
            ]
        )
        terminated = torch.stack([step.terminated for step in transitions])
        done = torch.stack([step.done for step in transitions])
        continuation = soft * (~terminated).to(soft) * next_valid
        rewards = rewards * next_valid
        scale = torch.as_tensor(
            objective_scale, device=rewards.device, dtype=rewards.dtype
        ).detach()
        if (
            scale.ndim > 1
            or (scale.ndim == 1 and scale.shape != ones.shape)
            or not bool(torch.isfinite(scale).all())
        ):
            raise ValueError(
                "objective_scale must be finite and scalar or one value per environment."
            )
        with self._frozen_critic():
            # Only episode boundaries and the final step bootstrap the actor.
            # Other successor values enter detached critic targets below.
            if not bool(done[:-1].any()):
                values = torch.cat(
                    (torch.zeros_like(rewards[:-1]), self._value(successors[-1])[None])
                )
            else:
                bootstrap = done.clone()
                bootstrap[-1] = True
                values = torch.zeros_like(rewards).masked_scatter(
                    bootstrap, self._value(torch.stack(successors)[bootstrap])
                )
            returns, entropy = torch.zeros_like(ones), torch.zeros_like(ones)
            weight, discount = ones, ones
            critic_weights = []
            for index, step in enumerate(transitions):
                weight = weight * valid[index]
                critic_weights.append(weight.detach())
                returns = returns + discount * weight * rewards[index]
                if "entropy" in step.policy_output.keys():
                    entropy = (
                        entropy + discount * weight * step.policy_output["entropy"]
                    )
                boundary = (
                    step.done
                    if index + 1 < rollout.num_steps
                    else torch.ones_like(step.done)
                )
                returns = (
                    returns
                    + discount
                    * self.cfg.gamma
                    * weight
                    * continuation[index]
                    * boundary
                    * values[index]
                )
                weight = weight * continuation[index]
                discount = discount * self.cfg.gamma
                if complete:
                    weight = weight * (~step.done)
                else:
                    weight = torch.where(step.done, ones, weight)
                    discount = torch.where(step.done, ones, discount)
            divisor = rollout.num_steps if self.cfg.normalize_return else 1
            objective = (returns * scale).mean() / divisor
            entropy = (entropy * scale).mean() / divisor
            loss = (-objective - self.cfg.ent_coef * entropy) * accumulation_scale
            self._critic_batches.append(
                _CriticBatch(
                    inputs.detach(),
                    critic_successors,
                    rewards.detach(),
                    continuation.detach(),
                    done,
                    torch.stack(critic_weights),
                )
            )
            self._accumulate_loss(
                rollout,
                loss=loss,
                objective=objective * accumulation_scale,
                entropy=entropy * accumulation_scale,
            )

    def finish_update(self) -> dict[str, float]:
        """Apply the actor update, then regress the critic on detached targets."""
        metrics = super().finish_update()
        if metrics["skipped_update"]:
            self._critic_batches = []
            metrics["critic_seconds"] = 0.0
            return metrics
        critic_started = time.perf_counter()
        inputs = torch.cat(
            [batch.inputs.flatten(0, 1) for batch in self._critic_batches]
        )
        if self.cfg.update_critic_normalization:
            # Count valid current states once. Add bootstrapped successors only
            # when they are absent from the next valid current-state row.
            samples = []
            for batch in self._critic_batches:
                active = batch.weights > 0
                samples.append(batch.inputs[active])
                missing = torch.ones_like(batch.done)
                missing[:-1] = batch.done[:-1] | ~active[1:]
                bootstrap = active & (batch.continuation > 0) & missing
                samples.append(batch.successors[bootstrap])
            samples = torch.cat(samples)
            if samples.shape[0]:
                self.policy.update_critic_normalization(samples)
        with torch.no_grad(), self._frozen_critic():
            targets = torch.cat(
                [
                    differentiable_lambda_returns(
                        batch.rewards,
                        self._value(batch.successors),
                        batch.continuation,
                        batch.done,
                        gamma=self.cfg.gamma,
                        gae_lambda=self.cfg.gae_lambda,
                    ).flatten()
                    for batch in self._critic_batches
                ]
            )
        weights = torch.cat([batch.weights.flatten() for batch in self._critic_batches])
        # Empty-weight rows must not create Adam steps: momentum can update
        # parameters even when the current minibatch gradient is zero.
        target_mean, weight_mean = float(targets.mean()), float(weights.mean())
        active = weights > 0
        inputs, targets, weights = inputs[active], targets[active], weights[active]
        totals = inputs.new_zeros(3)
        updates = 0
        for _ in range(self.cfg.critic_epochs if inputs.shape[0] else 0):
            indices = torch.randperm(inputs.shape[0], device=inputs.device)
            for batch in indices.tensor_split(
                min(self.cfg.critic_minibatches, inputs.shape[0])
            ):
                predictions = self._value(inputs[batch])
                loss = (
                    (predictions - targets[batch]).square() * weights[batch]
                ).sum() / weights[batch].sum().clamp_min(1.0)
                self.critic_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(
                    self._critic_parameters, self.cfg.critic_max_grad_norm
                )
                if bool(torch.isfinite(loss) & torch.isfinite(norm)):
                    self.critic_optimizer.step()
                    totals[0] += loss.detach()
                    totals[1] += norm.detach()
                    updates += 1
                else:
                    self.critic_optimizer.zero_grad(set_to_none=True)
                    totals[2] += 1
                    if not self.cfg.skip_nonfinite_updates:
                        raise FloatingPointError(
                            "APGActorCritic produced non-finite critic gradients."
                        )
        metrics.update(
            critic_loss=float(totals[0]) / max(updates, 1),
            critic_grad_norm=float(totals[1]) / max(updates, 1),
            critic_updates=float(updates),
            critic_skipped_updates=float(totals[2]),
            critic_target_mean=target_mean,
            critic_weight_mean=weight_mean,
        )
        self._critic_batches = []
        metrics["critic_seconds"] = time.perf_counter() - critic_started
        return metrics

    def _after_gradient_clip(self) -> None:
        if self.cfg.actor_grad_value_clip > 0:
            torch.nn.utils.clip_grad_value_(
                self._actor_parameters, self.cfg.actor_grad_value_clip
            )

    def cancel_update(self) -> None:
        super().cancel_update()
        self.critic_optimizer.zero_grad(set_to_none=True)
        self._critic_batches = []

    def state_dict(self) -> dict[str, Any]:
        """Include both optimizers in trainer checkpoints."""
        return {
            **super().state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
        }

    def validate_state_dict(
        self, state: dict[str, Any], *, resume_scheduler: str = "restore"
    ) -> None:
        """Validate both optimizer layouts before restoring live state."""
        super().validate_state_dict(state, resume_scheduler=resume_scheduler)
        saved = state["critic_optimizer"]
        groups = saved["param_groups"]
        if len(groups) != len(self.critic_optimizer.param_groups):
            raise ValueError("Incompatible critic optimizer parameter groups.")
        for group, current in zip(groups, self.critic_optimizer.param_groups):
            if len(group["params"]) != len(current["params"]):
                raise ValueError("Incompatible critic optimizer parameter count.")
            for identifier, parameter in zip(group["params"], current["params"]):
                for value in saved["state"].get(identifier, {}).values():
                    if (
                        isinstance(value, torch.Tensor)
                        and value.ndim > 0
                        and value.shape != parameter.shape
                    ):
                        raise ValueError("Incompatible critic optimizer state shape.")

    def _load_state_dict(self, state: dict[str, Any], *, resume_scheduler: str) -> None:
        super()._load_state_dict(state, resume_scheduler=resume_scheduler)
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
