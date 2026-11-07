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
from .robot_state import robot_state_error
from .model import RobotWorldModel

__all__ = ["RobotWorldModelObjective", "replay_supervised_batch"]


def replay_supervised_batch(
    windows: Mapping[str, torch.Tensor], *, source_id: str = "transition-replay"
) -> SupervisedBatch:
    """Expose an owned transition window as initial state and future supervision.

    Args:
        windows: Output of ``TransitionReplay.sample_windows``. ``state``,
            ``action``, ``next_state``, ``context`` and ``valid`` are required.
        source_id: Stable identity supplied by the replay's owning workflow.

    Returns:
        Batch with the measured starting state and actions as inputs. Future
        measured states enter targets only; their values never teacher-force
        the autoregressive rollout.
    """
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
        error_floor: Minimum physical increment scale for near-constant fields.

    Loss is the mean squared normalized error over valid transitions and state
    increment features. Invalid suffixes are excluded before physical-error
    arithmetic and may contain nonfinite padding. This supervised objective
    does not create rewards, continuation probabilities or policy gradients.
    The last-step diagnostic weights each window by its valid transition count.
    """

    def __init__(self, error_floor: float = 1e-2) -> None:
        if not 0 < error_floor < float("inf"):
            raise ValueError("error_floor must be finite and positive")
        self.error_floor = error_floor

    def compute_loss(
        self, model: RobotWorldModel, batch: SupervisedBatch
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
            error = robot_state_error(predicted, target)
            scale = model.output_normalizer.standard_deviation.clamp_min(
                self.error_floor
            )
            squared = (error / scale).square().mean(-1)
            loss = squared.mean()
            lengths = valid.sum(1)
            # Boolean selection is row-major; each prefix ends at its cumulative
            # valid count, so last-step diagnostics never inspect padded values.
            final = squared[lengths.cumsum(0) - 1]
            final = (final * lengths).sum() / lengths.sum()
            position_error = (predicted[..., :3] - target[..., :3]).norm(dim=-1)
        return LossResult(
            loss,
            metrics={
                "last_step_mse": final.detach(),
                "position_error_m": position_error.mean().detach(),
            },
            weight=float(valid.sum()),
        )
