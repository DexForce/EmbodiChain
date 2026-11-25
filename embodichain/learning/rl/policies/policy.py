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

"""Policy interface and training-input contracts for RL algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch.nn as nn

import torch
from tensordict import TensorDict

__all__ = ["Policy"]


@dataclass(frozen=True)
class _TrainingInputSpec:
    """Policy-owned inputs for action evaluation and auxiliary objectives.

    ``fields=None`` keeps the full transition payload. ``mutates_inputs``
    describes action evaluation: algorithms protect their reference targets
    before that call. Auxiliary objectives may require additional fields; this
    flag does not declare their mutation behavior. ``normalization_fields=None``
    preserves the complete historical normalization payload; an empty tuple
    declares that no observation statistics consume this rollout.
    """

    fields: tuple[str, ...] | None = None
    mutates_inputs: bool = True
    normalization_fields: tuple[str, ...] | None = None


class Policy(nn.Module, ABC):
    """Expose action sampling, action evaluation and value inference to algorithms."""

    device: torch.device
    """Device where the policy parameters are located."""

    def __init__(self) -> None:
        super().__init__()

    def _training_input_spec(self) -> _TrainingInputSpec:
        """Keep undeclared/custom evaluators on the full, defensive input path."""
        return _TrainingInputSpec()

    def get_action(
        self, tensordict: TensorDict, deterministic: bool = False
    ) -> TensorDict:
        """Sample actions into the provided TensorDict without gradients.

        Args:
            tensordict: Input TensorDict containing ``obs`` and, for policies
                with a privileged critic, ``critic_obs``.
            deterministic: If True, return the mean action; otherwise sample

        Returns:
            TensorDict with `action`, `sample_log_prob`, and `value` populated.
        """
        with torch.no_grad():
            return self.forward(tensordict, deterministic=deterministic)

    def get_differentiable_action(
        self, tensordict: TensorDict, deterministic: bool = False
    ) -> TensorDict:
        """Sample actions while preserving gradients to policy parameters.

        Stochastic implementations must use a reparameterized sample such as
        :meth:`torch.distributions.Distribution.rsample`. The base
        implementation fails explicitly to prevent a non-reparameterized
        ``sample()`` call from silently producing zero pathwise gradients.

        Args:
            tensordict: Input TensorDict containing ``obs`` and, for policies
                with a privileged critic, ``critic_obs``.
            deterministic: If True, return a differentiable deterministic
                action.

        Returns:
            TensorDict with differentiable policy outputs populated.

        Raises:
            NotImplementedError: If the policy has no differentiable sampling
                implementation.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement differentiable action sampling."
        )

    @abstractmethod
    def forward(
        self, tensordict: TensorDict, deterministic: bool = False
    ) -> TensorDict:
        """Write sampled actions and value estimates into the TensorDict."""
        raise NotImplementedError

    @abstractmethod
    def get_value(self, tensordict: TensorDict) -> TensorDict:
        """Write value estimate for the given observations into the TensorDict.

        Args:
            tensordict: Input TensorDict containing ``obs`` and, for policies
                with a privileged critic, ``critic_obs``.

        Returns:
            TensorDict with `value` populated.
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate_actions(self, tensordict: TensorDict) -> TensorDict:
        """Evaluate actions and return current policy outputs.

        Args:
            tensordict: TensorDict containing ``obs``, ``action``, and
                optional ``critic_obs``.

        Returns:
            A new TensorDict containing `sample_log_prob`, `entropy`, and `value`.
        """
        raise NotImplementedError
