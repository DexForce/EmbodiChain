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

"""Multi-step RWM supervision and replay-window adaptation."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from embodichain.learning.sl import LossResult, SupervisedBatch
from .kinematic import KinematicWorldModel
from .model import RobotWorldModel

__all__ = ["RobotWorldModelObjective", "replay_supervised_batch"]


def replay_supervised_batch(
    windows: Mapping[str, torch.Tensor],
    *,
    source_id: str = "transition-replay",
    complete_windows: bool = False,
) -> SupervisedBatch:
    """Expose an owned transition window as initial state and future supervision.

    Args:
        windows: Output of ``TransitionReplay.sample_windows``. ``state``,
            ``action``, ``next_state``, ``context`` and ``valid`` are required.
        source_id: Stable identity supplied by the replay's owning workflow.
        complete_windows: Retain only full-horizon episodes; also exclude rows
            with false ``usable`` flags when that auxiliary field is present.

    Returns:
        Batch with the measured starting state and actions as inputs. Future
        measured states enter targets only; their values never teacher-force
        the autoregressive rollout.
    """
    if complete_windows:
        selected = windows["valid"].all(1)
        if "usable" in windows:
            selected &= windows["usable"].all(1)
        if not bool(selected.any()):
            raise ValueError("No complete usable replay windows")
        windows = {name: value[selected] for name, value in windows.items()}
    return SupervisedBatch(
        inputs={
            "initial_state": windows["state"][:, 0],
            "actions": windows["action"],
            "contexts": windows["context"],
        },
        targets=windows["next_state"],
        masks={"valid": windows["valid"]},
        metadata={"source_id": source_id},
    )


class RobotWorldModelObjective:
    """Fit a complete autoregressive window using normalized state errors.

    Args:
        error_floor: Minimum physical increment scale, or None to use the model scale.
        temporal_reduction: ``mean`` averages valid steps; ``sum`` sums time then
            averages windows, matching a fixed-horizon autoregressive objective.
        error_weights: Optional nonnegative weights in increment feature order.
        compile_mode: Optional Torch compilation mode for complete fixed-shape
            windows. Padded windows retain the masked eager path.

    Loss averages normalized squared error over increment features and applies
    the configured temporal reduction. Invalid suffixes are excluded before physical-error
    arithmetic and may contain nonfinite padding. This supervised objective
    does not create rewards, continuation probabilities or policy gradients.
    The last-step diagnostic weights each window by its valid transition count.
    """

    def __init__(
        self,
        error_floor: float | None = 1e-2,
        *,
        temporal_reduction: str = "mean",
        error_weights: tuple[float, ...] = (),
        compile_mode: str | None = None,
    ) -> None:
        if error_floor is not None and not 0 < error_floor < float("inf"):
            raise ValueError("error_floor must be finite and positive")
        if temporal_reduction not in ("mean", "sum"):
            raise ValueError("temporal_reduction must be mean or sum")
        if error_weights and (
            any(not 0 <= w < float("inf") for w in error_weights)
            or not any(error_weights)
        ):
            raise ValueError(
                "Error weights must be finite, nonnegative and not all zero"
            )
        self.error_floor = error_floor
        self.temporal_reduction, self.error_weights = temporal_reduction, error_weights
        self._compiled_complete = (
            None
            if compile_mode is None
            else torch.compile(self._complete_loss, mode=compile_mode, fullgraph=True)
        )

    def compute_loss(
        self, model: RobotWorldModel | KinematicWorldModel, batch: SupervisedBatch
    ) -> LossResult:
        """Build an untruncated multi-step prediction graph.

        Args:
            model: Configured robot world model.
            batch: ``initial_state[B,D]``, ``actions[B,T,A]`` and optional
                ``contexts[B,T,C]`` inputs; canonical ``targets[B,T,D]`` and
                a Boolean prefix mask ``valid[B,T]``.

        Returns:
            Connected normalized loss and detached aggregation-ready diagnostics.
        """
        state = batch.inputs["initial_state"]
        actions = batch.inputs["actions"]
        contexts = batch.inputs.get("contexts")
        target = batch.targets
        valid = batch.masks["valid"]
        if (
            actions.ndim != 3
            or actions.shape[:1] != state.shape[:1]
            or actions.shape[1] < 1
            or state.ndim != 2
            or target.shape != actions.shape[:2] + (model.state_dim,)
            or valid.shape != actions.shape[:2]
            or valid.dtype != torch.bool
            or not valid[:, 0].all()
            or (valid[:, 1:] & ~valid[:, :-1]).any()
        ):
            raise ValueError(
                "RWM supervision requires aligned windows with valid prefix masks"
            )
        if contexts is not None and contexts.shape != actions.shape[:2] + (
            model.cfg.context_dim,
        ):
            raise ValueError("RWM context must align with every supervised transition")
        if self._compiled_complete is not None and bool(valid.all()):
            loss, first, last, position = self._compiled_complete(
                model, state, actions, contexts, target
            )
            count = state.shape[0] * (
                actions.shape[1] if self.temporal_reduction == "mean" else 1
            )
            return LossResult(
                loss,
                metrics={
                    "first_step_mse": first.detach(),
                    "last_step_mse": last.detach(),
                    "position_error_m": position.detach(),
                },
                weight=float(count),
            )
        predictions = []
        for index in range(actions.shape[1]):
            active = valid[:, index, None]
            # Padding must not enter nonlinear model arithmetic: masking only
            # the final loss can still produce NaN gradients from invalid rows.
            action = torch.where(active, actions[:, index], 0)
            context = (
                None if contexts is None else torch.where(active, contexts[:, index], 0)
            )
            next_state = model(
                state,
                action,
                context,
            )
            state = torch.where(active, next_state, state)
            predictions.append(state)
        predicted = torch.stack(predictions, 1)[valid]
        target = target[valid]
        # A learner's autocast covers the objective, so explicitly keep error
        # transport, logarithms and normalization in physical FP32 arithmetic.
        with torch.autocast(state.device.type, enabled=False):
            error = model.state_error(predicted, target)
            scale = model.error_scale()
            if self.error_floor is not None:
                scale = scale.clamp_min(self.error_floor)
            per_feature = (error / scale).square()
            squared = per_feature.mean(-1)
            if self.error_weights:
                if len(self.error_weights) != error.shape[-1]:
                    raise ValueError("Error weights must match the increment width")
                weighted = (per_feature * error.new_tensor(self.error_weights)).mean(-1)
            else:
                weighted = squared
            count = valid.sum() if self.temporal_reduction == "mean" else valid.shape[0]
            loss = weighted.sum() / count
            lengths = valid.sum(1)
            # Boolean selection is row-major; each prefix ends at its cumulative
            # valid count, so last-step diagnostics never inspect padded values.
            final = squared[lengths.cumsum(0) - 1]
            final = (final * lengths).sum() / lengths.sum()
            position_error = (predicted[..., :3] - target[..., :3]).norm(dim=-1)
        return LossResult(
            loss,
            metrics={
                "first_step_mse": squared[
                    torch.cat((lengths.new_zeros(1), lengths.cumsum(0)[:-1]))
                ]
                .mean()
                .detach(),
                "last_step_mse": final.detach(),
                "position_error_m": position_error.mean().detach(),
            },
            weight=float(count),
        )

    def _complete_loss(
        self,
        model: RobotWorldModel | KinematicWorldModel,
        state: torch.Tensor,
        actions: torch.Tensor,
        contexts: torch.Tensor | None,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compile the full autoregression and backward without masked indexing."""
        predictions = []
        for index in range(actions.shape[1]):
            state = model(
                state,
                actions[:, index],
                None if contexts is None else contexts[:, index],
            )
            predictions.append(state)
        predicted = torch.stack(predictions, 1).flatten(0, 1)
        target = target.flatten(0, 1)
        with torch.autocast(state.device.type, enabled=False):
            error = model.state_error(predicted, target)
            scale = model.error_scale()
            if self.error_floor is not None:
                scale = scale.clamp_min(self.error_floor)
            per_feature = (error / scale).square()
            squared = per_feature.mean(-1)
            if self.error_weights:
                if len(self.error_weights) != error.shape[-1]:
                    raise ValueError("Error weights must match the increment width")
                weighted = (per_feature * error.new_tensor(self.error_weights)).mean(-1)
            else:
                weighted = squared
            count = actions.shape[0] * (
                actions.shape[1] if self.temporal_reduction == "mean" else 1
            )
            loss = weighted.sum() / count
            by_step = squared.unflatten(0, actions.shape[:2])
            position_error = (predicted[..., :3] - target[..., :3]).norm(dim=-1).mean()
        return loss, by_step[:, 0].mean(), by_step[:, -1].mean(), position_error
