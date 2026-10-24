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

"""Single-device optimizer updates shared by policy and dynamics fitting."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from contextlib import AbstractContextManager, nullcontext
from typing import Any

import torch
from torch import nn

from .types import SupervisedBatch, SupervisedObjective

__all__ = ["SupervisedLearner"]


class SupervisedLearner:
    """Own one optimizer while objectives own mathematical and graph semantics.

    Args:
        model: Caller-constructed model on its intended device.
        objective: Loss builder, including any sequence unrolling or integration.
        optimizer: Optimizer over the model's trainable parameters.
        max_grad_norm: Positive clipping bound, or zero to measure without clipping.
        amp_dtype: Optional bfloat16 autocast for the forward objective. Numerical
            objectives may explicitly leave autocast for FP32 state integration.
        clip_grad: Optional numerical-profile-specific clipping implementation.
        numerical_context: Context factory spanning forward, backward and clipping.

    Sampling, normalizer updates, learning-rate schedules and model publication
    belong to callers. An update is committed only after optimizer.step succeeds.
    """

    def __init__(
        self,
        model: nn.Module,
        objective: SupervisedObjective,
        optimizer: torch.optim.Optimizer,
        *,
        max_grad_norm: float = 0.0,
        amp_dtype: torch.dtype | None = None,
        clip_grad: (
            Callable[[Iterable[nn.Parameter], float], torch.Tensor] | None
        ) = None,
        numerical_context: Callable[[], AbstractContextManager[Any]] | None = None,
    ) -> None:
        if not math.isfinite(max_grad_norm) or max_grad_norm < 0:
            raise ValueError("max_grad_norm must be finite and nonnegative")
        if amp_dtype not in (None, torch.bfloat16):
            raise ValueError("Only optional bfloat16 autocast is supported")
        self.model, self.objective, self.optimizer = model, objective, optimizer
        self.max_grad_norm, self.amp_dtype = float(max_grad_norm), amp_dtype
        self.clip_grad = clip_grad
        self.numerical_context = numerical_context or nullcontext
        self.num_updates = 0

    def update(self, batch: SupervisedBatch) -> dict[str, float | int]:
        """Backpropagate one objective and commit one finite parameter update.

        Args:
            batch: Owned training inputs and supervision on the model's device.

        Returns:
            Loss, pre-clipping gradient norm, effective weight, update count and
            objective metrics. Returned statistics do not retain the graph.

        Raises:
            FloatingPointError: Loss or gradient norm is not finite; optimizer
                parameters and the committed update counter are not advanced.
        """
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        parameter = next(self.model.parameters())
        autocast = (
            torch.autocast(parameter.device.type, dtype=self.amp_dtype)
            if self.amp_dtype is not None
            else nullcontext()
        )
        with self.numerical_context():
            with autocast:
                result = self.objective.compute_loss(self.model, batch)
            if result.loss.ndim != 0:
                raise ValueError("A supervised objective must return a scalar loss")
            if not math.isfinite(float(result.weight)) or result.weight <= 0:
                raise ValueError("Objective weight must be finite and positive")
            result.loss.backward()
            parameters = list(self.model.parameters())
            if self.clip_grad is not None:
                norm = self.clip_grad(parameters, self.max_grad_norm)
            else:
                norm = torch.nn.utils.clip_grad_norm_(
                    parameters,
                    self.max_grad_norm or float("inf"),
                    error_if_nonfinite=True,
                )
            observed = (
                torch.stack(
                    (
                        result.loss.detach(),
                        torch.as_tensor(norm, device=result.loss.device).detach(),
                    )
                )
                .cpu()
                .tolist()
            )
            if not all(math.isfinite(value) for value in observed):
                raise FloatingPointError("Non-finite supervised loss or gradient norm")
        self.optimizer.step()
        self.num_updates += 1
        metrics = {name: float(value) for name, value in result.metrics.items()}
        metrics.update(
            loss=observed[0],
            grad_norm=observed[1],
            weight=float(result.weight),
            num_updates=self.num_updates,
        )
        return metrics

    def state_dict(self) -> dict[str, Any]:
        """Return model, optimizer and committed update state.

        Returns:
            A PyTorch-style state mapping; callers serialize it before mutation.
        """
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "num_updates": self.num_updates,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore an update boundary after checking model tensor compatibility.

        Args:
            state: State produced by :meth:`state_dict`.
        """
        current = self.model.state_dict()
        saved = state["model"]
        if current.keys() != saved.keys():
            raise ValueError("Supervised checkpoint model keys differ")
        for key, value in current.items():
            if isinstance(value, torch.Tensor) and (
                not isinstance(saved[key], torch.Tensor)
                or value.shape != saved[key].shape
                or value.dtype != saved[key].dtype
            ):
                raise ValueError(f"Supervised checkpoint model tensor differs: {key}")
            elif not isinstance(value, torch.Tensor) and value != saved[key]:
                raise ValueError(f"Supervised checkpoint model metadata differs: {key}")
        count = state["num_updates"]
        if type(count) is not int or count < 0:
            raise ValueError("Invalid committed update count")
        self.model.load_state_dict(saved)
        self.optimizer.load_state_dict(state["optimizer"])
        self.num_updates = count
