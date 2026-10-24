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

"""Multi-step RWM supervision and graph-preserving prediction state."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import torch

from embodichain.compute.dynamics.robot_state import robot_state_error
from embodichain.learning.models.rwm import RobotWorldModel
from .types import LossResult, SupervisedBatch

if TYPE_CHECKING:
    from .trainer import SupervisedTrainer

__all__ = [
    "RobotWorldModelObjective",
    "RWMPredictor",
    "replay_supervised_batch",
    "build_rwm_trainer",
]


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
    increment features. Invalid suffixes are excluded. This supervised objective
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
            state = model(
                state,
                actions[:, index],
                None if contexts is None else contexts[:, index],
            )
            predictions.append(state)
        predicted = torch.stack(predictions, 1)
        # A learner's autocast covers the objective, so explicitly keep error
        # transport, logarithms and normalization in physical FP32 arithmetic.
        with torch.autocast(state.device.type, enabled=False):
            error = robot_state_error(predicted, target)
            scale = model.output_normalizer.standard_deviation.clamp_min(
                self.error_floor
            )
            squared = (error / scale).square().mean(-1)
            loss = squared[valid].mean()
            final_index = valid.sum(1) - 1
            final = squared[torch.arange(len(state), device=state.device), final_index]
            final = (final * valid.sum(1)).sum() / valid.sum()
            position_error = (predicted[..., :3] - target[..., :3]).norm(dim=-1)
        return LossResult(
            loss,
            metrics={
                "last_step_mse": final.detach(),
                "position_error_m": position_error[valid].mean().detach(),
            },
            weight=float(valid.sum()),
        )


class RWMPredictor:
    """Retain a batched prediction state without native simulation buffers.

    Args:
        model: Caller-owned world model.
        initial_state: Initial canonical FP32 state batch.

    Calls allocate new states, preserving outstanding results and unselected
    environment graphs. A caller performing policy optimization should use
    ``frozen_world_model`` around the rollout and dependent backward work.
    """

    def __init__(self, model: RobotWorldModel, initial_state: torch.Tensor) -> None:
        self.model = model
        self._state = initial_state.clone()
        if self._state.ndim != 2 or self._state.shape[-1] != model.state_dim:
            raise ValueError("Predictor initial state must be a canonical state batch")

    @property
    def state(self) -> torch.Tensor:
        """Current graph-connected state; callers must not mutate its storage."""
        return self._state

    def step(
        self, action: torch.Tensor, context: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Advance one transition and retain the returned graph-connected state.

        Args:
            action: One aligned control per environment.
            context: Optional physical context.

        Returns:
            Newly allocated canonical states.
        """
        self._state = self.model(self._state, action, context)
        return self._state

    def reset(
        self, initial_state: torch.Tensor, env_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Replace all or selected states without overwriting old graph tensors.

        Args:
            initial_state: Full replacement batch, or one state per selected id.
            env_ids: Optional unique one-dimensional environment indices.

        Returns:
            Current state after replacement. Unselected rows retain their graph.
        """
        if (
            initial_state.dtype != self._state.dtype
            or initial_state.device != self._state.device
        ):
            raise ValueError("Predictor reset state dtype and device must match")
        if env_ids is None:
            if initial_state.shape != self._state.shape:
                raise ValueError("Full predictor reset must preserve batch layout")
            self._state = initial_state.clone()
        else:
            if (
                env_ids.ndim != 1
                or env_ids.dtype != torch.int64
                or initial_state.shape != (len(env_ids), self.model.state_dim)
            ):
                raise ValueError(
                    "Selective predictor reset requires aligned int64 indices"
                )
            if env_ids.unique().numel() != env_ids.numel():
                raise ValueError("Selective predictor reset indices must be unique")
            self._state = self._state.index_copy(0, env_ids, initial_state)
        return self._state

    def detach_state(self) -> torch.Tensor:
        """Cut the current prediction graph while preserving every state value.

        Returns:
            The detached state. No reset, resampling or statistics update occurs.
        """
        self._state = self._state.detach()
        return self._state

    def state_dict(self) -> dict[str, Any]:
        """Return an owned graph-free prediction-state snapshot.

        Returns:
            Canonical state and model identity; model weights remain owned by
            the learning checkpoint.
        """
        return {
            "format": "rwm-predictor-v1",
            "contract": self.model.get_extra_state(),
            "state": self._state.detach().clone(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore prediction state under the same physical/action contract.

        Args:
            state: Snapshot from :meth:`state_dict`.
        """
        value = state["state"]
        if (
            state.get("format") != "rwm-predictor-v1"
            or state.get("contract") != self.model.get_extra_state()
            or value.shape != self._state.shape
            or value.dtype != self._state.dtype
        ):
            raise ValueError("Predictor snapshot contract or state layout differs")
        self._state = value.detach().to(self._state.device).clone()


def build_rwm_trainer(config: Mapping[str, Any]) -> SupervisedTrainer:
    """Build the standalone offline RWM recipe for ``train-sl``.

    Args:
        config: Standard ``model/data/objective/optimizer/trainer/evaluation``
            mapping. ``data.path`` is a tensor-only PyTorch file containing
            ``inputs``, ``targets``, ``masks`` and ``metadata``. Metadata must
            declare ``split``, ``action_semantics`` and ``step_dt``. Optional
            ``data.validation_path`` must identify a validation split. Inputs
            follow :meth:`RobotWorldModelObjective.compute_loss`.

    Returns:
        Shared supervised trainer. Initial normalization uses only valid
        measured transitions in the training corpus. This factory does not
        collect data, initialize simulation or construct a policy learner.
    """
    from embodichain.learning.models.rwm import RobotWorldModelCfg
    from .learner import SupervisedLearner
    from .sources import TensorBatchSource
    from .trainer import SupervisedTrainer, SupervisedTrainerCfg

    trainer_cfg = SupervisedTrainerCfg(**dict(config.get("trainer", {})))
    torch.manual_seed(trainer_cfg.seed)
    model = RobotWorldModel(RobotWorldModelCfg(**dict(config.get("model", {}))))

    def load_source(path: str, split: str) -> TensorBatchSource:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        metadata = payload["metadata"]
        if metadata.get("split") != split:
            raise ValueError(f"RWM corpus must declare split={split}")
        if (
            metadata.get("action_semantics") != model.cfg.action_semantics
            or metadata.get("step_dt") != model.cfg.step_dt
        ):
            raise ValueError("RWM corpus action semantics or control period differ")
        return TensorBatchSource(
            SupervisedBatch(
                payload["inputs"], payload["targets"], payload["masks"], metadata
            ),
            source_id=f"rwm-{split}",
        )

    data_cfg = config["data"]
    source = load_source(data_cfg["path"], "train")
    corpus = source.batch
    valid = corpus.masks["valid"]
    measured = torch.cat(
        (corpus.inputs["initial_state"][:, None], corpus.targets[:, :-1]), 1
    )
    contexts = corpus.inputs.get("contexts")
    model.update_normalization(
        measured[valid],
        corpus.inputs["actions"][valid],
        corpus.targets[valid],
        None if contexts is None else contexts[valid],
    )
    model.to(trainer_cfg.device)
    objective = RobotWorldModelObjective(**dict(config.get("objective", {})))
    optimizer_cfg = dict(config.get("optimizer", {}))
    kind = optimizer_cfg.pop("type", "adam")
    if kind not in {"adam", "adamw"}:
        raise ValueError("RWM optimizer must be adam or adamw")
    learning_rate = optimizer_cfg.pop("learning_rate", 1e-3)
    maximum_norm = optimizer_cfg.pop("max_grad_norm", 1.0)
    optimizer_class = torch.optim.Adam if kind == "adam" else torch.optim.AdamW
    optimizer = optimizer_class(model.parameters(), lr=learning_rate, **optimizer_cfg)
    learner = SupervisedLearner(model, objective, optimizer, max_grad_norm=maximum_norm)
    validation = (
        load_source(data_cfg["validation_path"], "validation")
        if data_cfg.get("validation_path")
        else None
    )

    def evaluate(current: SupervisedLearner) -> Mapping[str, float]:
        assert validation is not None
        sums: dict[str, float] = {}
        weight = 0.0
        was_training = current.model.training
        current.model.eval()
        try:
            with torch.no_grad():
                for batch in validation.iter_batches(trainer_cfg.batch_size):
                    result = objective.compute_loss(
                        current.model, batch.to(trainer_cfg.device)
                    )
                    values = {
                        "loss": float(result.loss),
                        **{key: float(value) for key, value in result.metrics.items()},
                    }
                    for name, value in values.items():
                        sums[name] = sums.get(name, 0.0) + value * result.weight
                    weight += result.weight
        finally:
            current.model.train(was_training)
        return {name: value / weight for name, value in sums.items()}

    return SupervisedTrainer(
        learner,
        source,
        trainer_cfg,
        signature={
            "recipe": "robot-world-model-v1",
            "model": model.get_extra_state(),
            "objective": dict(config.get("objective", {})),
            "optimizer": dict(config.get("optimizer", {})),
            "validation_source": None if validation is None else validation.source_id,
        },
        evaluate=evaluate if validation is not None else None,
    )
