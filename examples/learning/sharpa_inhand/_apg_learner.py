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

"""SuperMimic learner whose actor updates use EmbodiChain APG.

The pinned upstream HORA teacher, world model, replay, sampling and optimizers
are retained. ``value_bootstrap`` routes the original terminal-value actor
objective through EmbodiChain APG; otherwise the learned value is omitted.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sharpa_rl_unilab.algos.supermimic.learner import SharpaSuperMimicLearner
from sharpa_rl_unilab.algos.supermimic.state import unflatten_state
from sharpa_rl_unilab.tasks.sharpa_inhand.protocol import ACTOR_HISTORY
from tensordict import TensorDict

from embodichain.learning.rl.algo.apg import APG, APGCfg
from embodichain.learning.rl.collector import (
    DifferentiableRollout,
    DifferentiableTransition,
)

__all__: list[str] = []


class _APGLearner(SharpaSuperMimicLearner):
    def __init__(
        self, *args: Any, value_bootstrap: bool = False, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.value_bootstrap = value_bootstrap
        # AOTAutograd's donated buffers require one backward per compiled graph.
        # Validate the independent reduction in the eager smoke; preserve the
        # reference compiled training path without retaining a second backward.
        self._verify_actor_gradient = value_bootstrap and not bool(
            self.cfg.algo.compile
        )
        self.apg = APG(
            APGCfg(
                device=str(self.device),
                gamma=float(self.cfg.algo.gamma),
                max_grad_norm=1.0,
            ),
            self.actor,
        )
        # Preserve the reference mixed Muon/Adam optimizer and its parameters.
        self.apg.optimizer = self.opt_pi

    def policy_update(self) -> dict[str, float]:
        start, history = self.buffer.sample_starts(
            int(self.cfg.algo.pi_batch), ACTOR_HISTORY
        )
        state, context = unflatten_state(start["s"]), start["xi"]
        noisy, clean, priv = self._initial_history(history)
        packed = torch.cat((noisy.flatten(-2), priv[:, -1]), -1)
        initial = packed
        survival = torch.ones(packed.shape[0], device=self.device)
        transitions = []
        features = [self._critic_features(clean, priv)]
        rewards, soft_dones = [], []
        flags = torch.zeros_like(survival, dtype=torch.bool)
        for module in (self.wm, self.critic):
            module.requires_grad_(False)
        try:
            for _ in range(int(self.cfg.algo.horizon)):
                before = packed
                state, packed, noisy, clean, priv, feature, reward, done = (
                    self._imagination_step_fn(
                        state, packed, context, noisy, clean, priv
                    )
                )
                # Original differentiable soft survival, including its gradient.
                # No imagined autoreset or hard threshold replaces this owner.
                transitions.append(
                    DifferentiableTransition(
                        observation=before,
                        policy_output=TensorDict({}, batch_size=[packed.shape[0]]),
                        reward=survival * reward,
                        terminated=flags,
                        truncated=flags,
                        next_observation=packed,
                        info={},
                    )
                )
                survival = survival * (1 - done)
                if self.value_bootstrap:
                    features.append(feature)
                    rewards.append(reward.detach())
                    soft_dones.append(done.detach())
            rollout = DifferentiableRollout(initial, tuple(transitions))
            terminal_value = None
            expected_gradients = None
            if self.value_bootstrap:
                terminal_value = survival * self.critic(
                    TensorDict({"critic": features[-1]}, batch_size=[packed.shape[0]])
                ).squeeze(-1)
                if self._verify_actor_gradient:
                    # Independent reduction of the pinned upstream objective:
                    # sum gamma^t * survival_t * reward_t + gamma^H * survival_H * V_H.
                    gamma = float(self.cfg.algo.gamma)
                    objective = torch.zeros_like(survival)
                    for index, transition in enumerate(transitions):
                        objective = objective + gamma**index * transition.reward
                    objective = objective + gamma ** len(transitions) * terminal_value
                    expected_gradients = torch.autograd.grad(
                        -objective.mean(),
                        tuple(self.actor.parameters()),
                        retain_graph=True,
                        allow_unused=True,
                    )
            self.apg.begin_update()
            self.apg.accumulate_complete_rollout(rollout, terminal_value=terminal_value)
            gradient_error = 0.0
            gradient_relative_error = 0.0
            if expected_gradients is not None:
                error_squared = torch.zeros((), device=self.device)
                expected_squared = torch.zeros_like(error_squared)
                for parameter, expected in zip(
                    self.actor.parameters(), expected_gradients, strict=True
                ):
                    if expected is None:
                        assert parameter.grad is None
                    else:
                        difference = parameter.grad - expected
                        gradient_error = max(
                            gradient_error, float(difference.abs().max())
                        )
                        if not bool(self.cfg.algo.use_amp):
                            torch.testing.assert_close(
                                parameter.grad, expected, rtol=1e-5, atol=1e-6
                            )
                        error_squared += difference.float().square().sum()
                        expected_squared += expected.float().square().sum()
                gradient_relative_error = float(
                    (error_squared / expected_squared.clamp_min(1e-30)).sqrt()
                )
                # Equivalent return reductions can traverse BF16 backward
                # accumulation in different orders. Bound the whole-gradient
                # relative L2 error below BF16 epsilon; retain FP32 checks above.
                if bool(self.cfg.algo.use_amp) and (
                    not np.isfinite(gradient_relative_error)
                    or gradient_relative_error > 1e-3
                ):
                    raise AssertionError(
                        f"AMP actor gradient relative error {gradient_relative_error}"
                    )
                self._verify_actor_gradient = False
            # Match both original gradient clipping stages. APG's subsequent
            # norm bound is idempotent after these stricter bounds.
            norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            torch.nn.utils.clip_grad_value_(self.actor.parameters(), 0.1)
            metrics = self.apg.finish_update()
            if self.actor_avg is not None and not metrics["skipped_update"]:
                self.actor_avg.update_parameters(self.actor)
            metrics = {
                **{f"apg/{key}": value for key, value in metrics.items()},
                "loss/actor_grad_norm": float(norm),
                "imagination/survival": float(survival.detach().mean()),
                "parity/actor_gradient_checked": float(expected_gradients is not None),
            }
            if expected_gradients is not None:
                metrics["parity/actor_gradient_max_error"] = gradient_error
                metrics["parity/actor_gradient_relative_l2_error"] = (
                    gradient_relative_error
                )
        except BaseException:
            self.apg.cancel_update()
            raise
        finally:
            for module in (self.wm, self.critic):
                module.requires_grad_(True)
        if self.value_bootstrap:
            metrics["loss/critic"] = self._fit_values(features, rewards, soft_dones)
        return metrics

    def _fit_values(
        self,
        features: list[torch.Tensor],
        rewards: list[torch.Tensor],
        soft_dones: list[torch.Tensor],
    ) -> float:
        """Fit the original imagined lambda-return value targets."""
        inputs = torch.stack(features).detach()
        self.critic.obs_normalizer.update(inputs.flatten(0, 1))
        with torch.no_grad():
            values = self.critic(
                TensorDict(
                    {"critic": inputs.flatten(0, 1)},
                    batch_size=[inputs.shape[0] * inputs.shape[1]],
                )
            ).view(*inputs.shape[:2])
            target = values[-1]
            targets = []
            gamma, lam = float(self.cfg.algo.gamma), float(self.cfg.algo.lam)
            for index in reversed(range(len(rewards))):
                target = rewards[index] + gamma * (1 - soft_dones[index]) * (
                    (1 - lam) * values[index + 1] + lam * target
                )
                targets.append(target)
            targets = torch.stack(targets[::-1])
        flat_inputs, flat_targets = inputs[:-1].flatten(0, 1), targets.flatten()
        count = len(flat_inputs)
        minibatches = int(self.cfg.algo.critic_minibatches)
        size = count // minibatches
        if size == 0:
            raise ValueError("critic_minibatches exceeds imagined transition count")
        for _ in range(int(self.cfg.algo.critic_epochs)):
            order = torch.randperm(count, device=self.device)
            for index in range(minibatches):
                rows = order[index * size : (index + 1) * size]
                prediction = self.critic(
                    TensorDict({"critic": flat_inputs[rows]}, batch_size=[len(rows)])
                ).squeeze(-1)
                loss = (prediction - flat_targets[rows]).square().mean()
                self._optimize(loss, self.opt_v, self.critic)
        return float(loss.detach())
