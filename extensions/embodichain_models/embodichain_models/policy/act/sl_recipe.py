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

"""Offline ACT recipe for the shared supervised-learning trainer."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from embodichain.data_pipeline.datasets import LeRobotDataSource, LeRobotImageReader
from embodichain.learning.sl import (
    SupervisedLearner,
    SupervisedTrainer,
    SupervisedTrainerCfg,
)
from .data import ACTSequenceSource
from .model import ACTCfg, ACTObjective, ACTPolicy

__all__ = ["build_act_trainer"]


def build_act_trainer(config: Mapping[str, Any]) -> SupervisedTrainer:
    """Create an offline visual ACT trainer from explicitly described datasets.

    Args:
        config: Standard supervised recipe mapping. ``data`` declares
            ``train_path``, ``validation_path``, ``schema`` and ``image_key``;
            both datasets use the same observation/action contract. The schema
            includes ordered feature names, action meaning and ``step_dt``.

    Returns:
        Trainer using train-only frozen normalization and optional validation.
        Dataset acquisition and physical policy evaluation remain host-owned.
    """
    trainer_cfg = SupervisedTrainerCfg(**dict(config.get("trainer", {})))
    model_cfg = ACTCfg(**dict(config.get("model", {})))
    data_cfg = dict(config["data"])
    schema = data_cfg["schema"]
    image_key = data_cfg.get("image_key", "observation.images.cam_high")
    train_path = Path(data_cfg["train_path"]).resolve()
    validation_path = (
        Path(data_cfg["validation_path"]).resolve()
        if data_cfg.get("validation_path")
        else None
    )
    if validation_path == train_path:
        raise ValueError("Training and validation datasets must be distinct")
    if (
        len(schema["observation"]["names"]) != model_cfg.state_dim
        or len(schema["action"]["names"]) != model_cfg.action_dim
    ):
        raise ValueError("ACT data schema widths do not match the model")

    def make_source(
        path: Path, split: str
    ) -> tuple[LeRobotDataSource, ACTSequenceSource]:
        repo_id = path.name
        numeric = LeRobotDataSource(
            path,
            schema=schema,
            source_id=f"act-{split}",
            repo_id=repo_id,
            memory_limit_bytes=int(
                data_cfg.get("numeric_memory_limit_bytes", 512 * 1024**2)
            ),
        )
        images = LeRobotImageReader(
            path,
            image_key,
            repo_id=repo_id,
            cache_bytes=int(data_cfg.get("image_cache_bytes", 512 * 1024**2)),
        )
        return numeric, ACTSequenceSource(numeric, images, model_cfg.horizon)

    numeric, source = make_source(train_path, "train")
    validation = (
        None
        if validation_path is None
        else make_source(validation_path, "validation")[1]
    )
    torch.manual_seed(trainer_cfg.seed)
    model = ACTPolicy(model_cfg)
    valid = numeric.data["valid"]
    states = numeric.data["obs"][valid].float()
    actions = numeric.data["actions"][valid].float()
    model.set_normalization(
        states.mean(0),
        states.std(0, unbiased=False).clamp_min(1e-3),
        actions.mean(0),
        actions.std(0, unbiased=False).clamp_min(1e-3),
    )
    model.to(trainer_cfg.device)
    objective = ACTObjective(**dict(config.get("objective", {})))
    optimizer_cfg = dict(config.get("optimizer", {}))
    learning_rate = float(optimizer_cfg.pop("learning_rate", 1e-4))
    backbone_lr = float(optimizer_cfg.pop("backbone_learning_rate", 1e-5))
    weight_decay = float(optimizer_cfg.pop("weight_decay", 1e-4))
    max_grad_norm = float(optimizer_cfg.pop("max_grad_norm", 1.0))
    if optimizer_cfg:
        raise ValueError(f"Unknown ACT optimizer settings: {sorted(optimizer_cfg)}")
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not n.startswith("backbone.")
                ]
            },
            {"params": list(model.backbone.parameters()), "lr": backbone_lr},
        ],
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    learner = SupervisedLearner(
        model, objective, optimizer, max_grad_norm=max_grad_norm
    )

    def evaluate(current: SupervisedLearner) -> Mapping[str, float]:
        assert validation is not None
        generator = torch.Generator(device="cpu").manual_seed(8128)
        values: dict[str, float] = {}
        for _ in range(4):
            batch = validation.sample(trainer_cfg.batch_size, generator=generator).to(
                trainer_cfg.device
            )
            result = objective.compute_loss(current.model, batch)
            for name, value in {"loss": result.loss, **result.metrics}.items():
                values[name] = values.get(name, 0.0) + float(value) / 4
        return values

    return SupervisedTrainer(
        learner,
        source,
        trainer_cfg,
        signature={
            "recipe": "visual-act-v1",
            "model": model_cfg.to_dict(),
            "schema": schema,
            "image_key": image_key,
            "image_preprocessing": "rgb-imagenet-v1",
            "objective": dict(config.get("objective", {})),
            "optimizer": dict(config.get("optimizer", {})),
            "task_identity": config.get("task_identity"),
            "scene_seeds": config.get("scene_seeds"),
            "validation_source": None if validation is None else validation.source_id,
        },
        evaluate=evaluate if validation is not None else None,
    )
