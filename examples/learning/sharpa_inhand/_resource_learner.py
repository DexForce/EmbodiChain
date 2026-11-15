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

"""Reference learner with host replay and recomputed world-model activations.

Replay capacity, batch sizes, objective, precision and optimizer updates are
unchanged. Host replay retains the reference CPU sampling laws; its RNG backend
is CPU rather than CUDA. Activation recomputation preserves module state keys
and parameter identity so original checkpoints/optimizers remain loadable.
"""

from __future__ import annotations

from typing import Any

import torch
from _apg_learner import _APGLearner
from omegaconf import OmegaConf
from sharpa_rl_unilab.algos.supermimic.replay import ReplayBuffer
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    checkpoint_wrapper,
)

from embodichain.learning.rl.experimental._host_replay import _HostReplayAdapter

__all__: list[str] = []


class _ResourceAPGLearner(_APGLearner):
    def __init__(
        self,
        env: Any,
        cfg: Any,
        actor: torch.nn.Module,
        critic: torch.nn.Module,
        mdp: Any,
        capture: Any,
        *,
        value_bootstrap: bool = True,
    ) -> None:
        if bool(cfg.algo.compile):
            raise ValueError(
                "Host-replay activation recomputation currently qualifies eager execution only"
            )
        local = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        steps = int(local.algo.buffer_steps)
        local.algo.buffer_steps = min(steps, 16)
        super().__init__(
            env, local, actor, critic, mdp, capture, value_bootstrap=value_bootstrap
        )
        self.cfg.algo.buffer_steps = steps
        self.buffer = _HostReplayAdapter(
            ReplayBuffer(steps, env.num_envs, "cpu"), self.device
        )
        keys = tuple(self.wm.state_dict())
        parameters = tuple(self.wm.parameters())
        self.wm.net = checkpoint_wrapper(
            self.wm.net,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            preserve_rng_state=True,
        )
        if tuple(self.wm.state_dict()) != keys or any(
            a is not b for a, b in zip(parameters, self.wm.parameters(), strict=True)
        ):
            raise RuntimeError(
                "Activation wrapper changed checkpoint keys or parameter identity"
            )
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
