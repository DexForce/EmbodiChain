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

"""NeRD window sampling and standalone recipes for ``train-sl``."""

from __future__ import annotations

import random
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch

from embodichain.compute.dynamics import NeRDStateCodec
from embodichain.data_pipeline.dynamics.shards import DynamicsWindowDataset
from embodichain.learning.models.nerd import NeRDModel, NeRDModelCfg
from embodichain.learning.sl import SupervisedBatch, SupervisedLearner
from embodichain.learning.sl.trainer import SupervisedTrainer, SupervisedTrainerCfg

from ._numerics import _attention_context, _clip_grad_norm_, _validate_numerical_profile
from .supervised import NeRDObjective, make_nerd_batch
from .trainer import (
    _PREPROCESSING_EXECUTION,
    compute_statistics,
    encode_batch,
    evaluate_model,
    load_statistics,
)

__all__ = ["NeRDWindowSource", "build_supervised_trainer"]


class NeRDWindowSource:
    """Adapt an immutable NeRD window reader without owning sampling RNG.

    Args:
        dataset: Caller-owned window dataset, including optional resident cache.
        codec: Physical codec matching that dataset.
        device: Encoding device, preserving the reference preprocessing policy.
    """

    def __init__(
        self,
        dataset: DynamicsWindowDataset,
        codec: NeRDStateCodec,
        device: str | torch.device = "cpu",
    ) -> None:
        self.dataset, self.codec, self.device = dataset, codec, device
        self.source_id = (
            f"{dataset.fingerprint}:{dataset.split}:"
            f"history={dataset.history_length}:prefix={dataset.include_prefix}"
        )

    def sample(self, batch_size: int, *, generator: torch.Generator) -> SupervisedBatch:
        """Sample copied windows using an explicit CPU generator.

        Args:
            batch_size: Positive number of windows sampled with replacement.
            generator: Caller-owned sampling generator, committed after updates.

        Returns:
            Encoded inputs, physical targets and valid-position mask.
        """
        indices = torch.randint(len(self.dataset), (batch_size,), generator=generator)
        return make_nerd_batch(self.dataset.get_batch(indices), self.codec, self.device)


def build_supervised_trainer(config: dict[str, Any]) -> SupervisedTrainer:
    """Build a simulation-independent NeRD recipe for the shared trainer.

    Args:
        config: ``train-sl`` mapping. ``model`` is a NeRD model configuration;
            ``data.path`` identifies committed shards and optional
            ``data.statistics`` supplies shared frozen training statistics.
            ``objective`` accepts ``include_prefix`` and ``numerical_profile``.
            ``optimizer`` accepts Adam rates, clipping and interval/offset LR
            scheduling. ``trainer`` uses ``SupervisedTrainerCfg``.

    Returns:
        Common trainer with a NeRD objective and owned window reader references.
        Its checkpoints use the common SL format; the existing ``dynamics`` CLI
        continues reading and writing historical NeRD training checkpoints.
    """
    cfg = SupervisedTrainerCfg(**config.get("trainer", {}))
    cfg.validate()
    if cfg.total_updates < 1:
        raise ValueError("NeRD recipes require a positive update budget")
    torch.manual_seed(cfg.seed)
    model = NeRDModel(NeRDModelCfg(**config["model"]))
    data = config["data"]
    objective_cfg = config.get("objective", {})
    include_prefix = objective_cfg.get("include_prefix", False)
    profile = _validate_numerical_profile(
        objective_cfg.get("numerical_profile", "default")
    )
    if profile == "reference":
        model.cfg.attention_backend = "math"
    train = DynamicsWindowDataset(
        data["path"],
        "train",
        model.cfg.history_length,
        include_prefix=include_prefix,
        verify=True,
    )
    validation = DynamicsWindowDataset(
        data["path"],
        "validation",
        model.cfg.history_length,
        include_prefix=include_prefix,
    )
    codec = NeRDStateCodec(**train.manifest["metadata"]["codec"])
    statistics = (
        load_statistics(data["statistics"])
        if data.get("statistics")
        else compute_statistics(train, codec, cfg.batch_size)
    )
    if statistics["dataset_fingerprint"] != train.fingerprint:
        raise ValueError("NeRD statistics/dataset fingerprint mismatch")
    model.set_normalization(
        {
            key: (value["mean"], value["var"])
            for key, value in statistics["inputs"].items()
        },
        statistics["output"]["mean"],
        statistics["output"]["var"],
    )
    model.to(cfg.device)
    encode_batch(train.get_batch([0]), codec, cfg.device)
    cache_bytes = data.get("resident_cache_bytes", 0)
    if cache_bytes:
        used = train.enable_device_cache(cfg.device, cache_bytes)
        validation.enable_device_cache(cfg.device, cache_bytes - used)
    if torch.device(cfg.device).type == "cuda":
        raw = train.get_batch([0] * cfg.batch_size)
        for _ in range(3):
            encode_batch(raw, codec, cfg.device)
    optimizer_cfg = config.get("optimizer", {})
    max_grad_norm = float(optimizer_cfg.get("max_grad_norm", 1.0))
    rate = float(optimizer_cfg.get("learning_rate", 1e-3))
    final_rate = float(optimizer_cfg.get("final_learning_rate", 1e-4))
    interval = int(optimizer_cfg.get("lr_update_interval", 1))
    offset = int(optimizer_cfg.get("lr_offset", 0))
    if interval < 1 or offset < 0 or rate <= 0 or final_rate < 0 or max_grad_norm <= 0:
        raise ValueError("Invalid NeRD optimizer/schedule settings")
    optimizer = torch.optim.Adam(model.parameters(), lr=rate)

    def schedule(step: int) -> float:
        ratio = (step // interval + offset) / (cfg.total_updates / interval + offset)
        return ((1 - ratio) * rate + ratio * final_rate) / rate

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    learner = SupervisedLearner(
        model,
        NeRDObjective(include_prefix=include_prefix),
        optimizer,
        max_grad_norm=max_grad_norm,
        clip_grad=lambda parameters, maximum: _clip_grad_norm_(
            parameters,
            maximum,
            profile=profile,
            error_if_nonfinite=profile == "default",
        ),
        numerical_context=lambda: _attention_context(profile),
    )
    evaluation = config.get("evaluation", {})

    def evaluate(current: SupervisedLearner) -> dict[str, float]:
        with _attention_context(profile):
            report = evaluate_model(
                current.model,
                codec,
                validation,
                cfg.batch_size,
                cfg.device,
                max_batches=evaluation.get("max_batches"),
            )
        return {
            key: float(value)
            for key, value in report.items()
            if isinstance(value, (int, float))
        }

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    return SupervisedTrainer(
        learner,
        NeRDWindowSource(train, codec, cfg.device),
        cfg,
        signature={
            "model": model.cfg.to_dict(),
            "objective": {
                "include_prefix": include_prefix,
                "numerical_profile": profile,
            },
            "optimizer": optimizer_cfg,
            "physics": train.manifest["metadata"],
            "preprocessing_execution": _PREPROCESSING_EXECUTION,
            "statistics_sha256": (
                hashlib.sha256(Path(data["statistics"]).read_bytes()).hexdigest()
                if data.get("statistics")
                else None
            ),
        },
        scheduler=scheduler,
        evaluate=evaluate,
    )
