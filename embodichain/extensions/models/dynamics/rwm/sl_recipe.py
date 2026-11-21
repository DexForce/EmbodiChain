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

"""Installable offline RWM recipe for the common supervised trainer."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from embodichain.learning.sl import (
    SupervisedBatch,
    SupervisedLearner,
    SupervisedTrainer,
    SupervisedTrainerCfg,
    TensorBatchSource,
)
from .model import RobotWorldModel, RobotWorldModelCfg
from .supervised import RobotWorldModelObjective

__all__ = ["build_rwm_trainer"]


def build_rwm_trainer(config: Mapping[str, Any]) -> SupervisedTrainer:
    """Build the standalone offline RWM recipe for ``train-sl``.

    Args:
        config: Standard ``model/data/objective/optimizer/trainer/evaluation``
            mapping. ``data.path`` is a tensor-only PyTorch file containing
            ``inputs``, ``targets``, ``masks`` and ``metadata``. Metadata must
            declare ``split``, ``action_semantics`` and ``step_dt``. Named model
            joints require the same ordered ``joint_names`` in metadata. Optional
            ``data.validation_path`` must identify a validation split. Inputs
            follow :meth:`RobotWorldModelObjective.compute_loss`.

    Returns:
        Shared supervised trainer. Initial normalization uses only valid
        measured transitions in the training corpus. This factory does not
        collect data, initialize simulation or construct a policy learner.
    """
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
        declared_joints = metadata.get("joint_names", ())
        if not isinstance(declared_joints, (list, tuple)) or tuple(
            declared_joints
        ) != tuple(model.cfg.joint_names):
            raise ValueError(
                "RWM corpus ordered joint_names differ from model contract"
            )
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
