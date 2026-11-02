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

"""Single-device and distributed optimizer updates for policy and dynamics fitting."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from contextlib import AbstractContextManager, nullcontext
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from .distributed import _DistributedContext
from .types import SupervisedBatch, SupervisedObjective

__all__ = ["SupervisedLearner"]


class _ObjectiveModule(nn.Module):
    """Wrap the complete graph while keeping model-specific APIs available."""

    def __init__(self, model: nn.Module, objective: SupervisedObjective) -> None:
        super().__init__()
        self.model = model
        object.__setattr__(self, "objective", objective)

    def forward(
        self, batch: SupervisedBatch
    ) -> tuple[torch.Tensor, Mapping[str, torch.Tensor | float], float]:
        result = self.objective.compute_loss(self.model, batch)
        return result.loss, result.metrics, result.weight


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
        ddp_find_unused_parameters: Enable graph traversal for objectives that
            omit trainable parameters. Applies only with an initialized group.
        ddp_broadcast_buffers: Broadcast rank-zero buffers before forwards.
            Defaults to the existing rank-local policy; explicit reference
            recipes can request the PyTorch DDP broadcast policy.

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
        ddp_find_unused_parameters: bool = False,
        ddp_broadcast_buffers: bool = False,
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
        self.ddp_find_unused_parameters = ddp_find_unused_parameters
        self.ddp_broadcast_buffers = ddp_broadcast_buffers
        self._ddp: DistributedDataParallel | None = None
        self._distributed_metrics: tuple[str, ...] | None = None

    def _configure_distributed(self) -> _DistributedContext:
        context = _DistributedContext(next(self.model.parameters()).device)
        if context.enabled and self._ddp is None:
            self._ddp = DistributedDataParallel(
                _ObjectiveModule(self.model, self.objective),
                device_ids=(
                    [context.device.index] if context.device.type == "cuda" else None
                ),
                broadcast_buffers=self.ddp_broadcast_buffers,
                find_unused_parameters=self.ddp_find_unused_parameters,
            )
        return context

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
        context = self._configure_distributed()
        if context.enabled:
            return self._distributed_update(batch, context)
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

    def _distributed_update(
        self, batch: SupervisedBatch, context: _DistributedContext
    ) -> dict[str, float | int]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        autocast = (
            torch.autocast(context.device.type, dtype=self.amp_dtype)
            if self.amp_dtype is not None
            else nullcontext()
        )
        with self.numerical_context():

            def forward() -> tuple[torch.Tensor, dict[str, float], float]:
                assert self._ddp is not None
                with autocast:
                    loss, values, weight = self._ddp(batch)
                if loss.ndim != 0 or not loss.requires_grad:
                    raise ValueError(
                        "A supervised objective must return a scalar loss with gradients"
                    )
                weight = float(weight)
                if not math.isfinite(weight) or weight <= 0:
                    raise ValueError(
                        "Objective weight must be finite and positive on every rank"
                    )
                if not torch.isfinite(loss.detach()):
                    raise FloatingPointError("Non-finite supervised loss")
                metrics = {name: float(value) for name, value in values.items()}
                if not all(math.isfinite(value) for value in metrics.values()):
                    raise FloatingPointError("Non-finite supervised metric")
                names = tuple(sorted(metrics))
                if (
                    self._distributed_metrics is not None
                    and names != self._distributed_metrics
                ):
                    raise ValueError("DDP objective metric names must remain constant")
                return loss, metrics, weight

            loss, local_metrics, weight = context.run(forward, "objective")
            names = tuple(sorted(local_metrics))
            if self._distributed_metrics is None:
                context.agree(names, "objective metric names")
                self._distributed_metrics = names
            # Native DDP averages parameter gradients. Scale each local mean by
            # W*w/sum(w) to obtain the mean over valid supervision, not ranks.
            observations = torch.tensor(
                [weight, float(loss.detach()) * weight]
                + [local_metrics[name] * weight for name in names],
                dtype=torch.float64,
                device=context.control_device,
            )
            dist.all_reduce(observations)
            if not torch.isfinite(observations).all():
                raise FloatingPointError(
                    "Non-finite globally reduced supervised statistics"
                )
            global_weight = float(observations[0])
            (loss * (context.world_size * weight / global_weight)).backward()

            def clip() -> float:
                parameters = list(self.model.parameters())
                norm = (
                    self.clip_grad(parameters, self.max_grad_norm)
                    if self.clip_grad is not None
                    else torch.nn.utils.clip_grad_norm_(
                        parameters, self.max_grad_norm or float("inf")
                    )
                )
                value = float(norm)
                if not math.isfinite(value):
                    raise FloatingPointError("Non-finite supervised gradient norm")
                return value

            norm = context.run(clip, "gradient clipping")
        # A failed optimizer/communication operation aborts the job; no attempt
        # is made to reuse a partially updated replica or broken process group.
        context.run(self.optimizer.step, "optimizer update")
        self.num_updates += 1
        reduced = observations.cpu().tolist()
        metrics = {
            name: reduced[index + 2] / global_weight for index, name in enumerate(names)
        }
        metrics.update(
            loss=reduced[1] / global_weight,
            grad_norm=norm,
            weight=global_weight,
            num_updates=self.num_updates,
        )
        return metrics

    def _optimizer_layout(self) -> list[list[str]]:
        names = {
            id(parameter): name for name, parameter in self.model.named_parameters()
        }
        try:
            return [
                [names[id(parameter)] for parameter in group["params"]]
                for group in self.optimizer.param_groups
            ]
        except KeyError as error:
            raise ValueError(
                "Optimizer parameters must belong to the learner model"
            ) from error

    def state_dict(self) -> dict[str, Any]:
        """Return model, optimizer and committed update state.

        Returns:
            A PyTorch-style state mapping; callers serialize it before mutation.
        """
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "optimizer_type": f"{type(self.optimizer).__module__}.{type(self.optimizer).__qualname__}",
            "optimizer_layout": self._optimizer_layout(),
            "num_updates": self.num_updates,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore an update boundary after checking model tensor compatibility.

        Args:
            state: State produced by :meth:`state_dict`.
        """
        self._validate_state_dict(state)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.num_updates = state["num_updates"]

    def _validate_state_dict(self, state: Mapping[str, Any]) -> None:
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
        optimizer_type = (
            f"{type(self.optimizer).__module__}.{type(self.optimizer).__qualname__}"
        )
        if (
            state.get("optimizer_type") != optimizer_type
            or state.get("optimizer_layout") != self._optimizer_layout()
        ):
            raise ValueError(
                "Supervised checkpoint optimizer type or parameter layout differs"
            )
        groups = state["optimizer"].get("param_groups", [])
        if len(groups) != len(self.optimizer.param_groups) or any(
            len(group.get("params", [])) != len(actual["params"])
            for group, actual in zip(groups, self.optimizer.param_groups)
        ):
            raise ValueError("Supervised checkpoint optimizer parameter groups differ")
