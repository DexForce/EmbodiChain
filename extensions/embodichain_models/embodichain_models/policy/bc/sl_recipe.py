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

"""Installable offline BC recipes for the common supervised trainer."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch

from embodichain.learning.sl import SupervisedLearner
from embodichain.learning.sl.trainer import SupervisedTrainer, SupervisedTrainerCfg

from .data import BCWindowDataset
from .model import ChunkBCModel, ChunkBCModelCfg
from .supervised import ChunkBCObjective, HDF5BCSource, PrefetchedBCSource

__all__ = ["build_supervised_trainer"]


def build_supervised_trainer(config: dict[str, Any]) -> SupervisedTrainer:
    """Build a simulator-free state or RGB BC recipe from stored physical targets.

    Args:
        config: ``train-sl`` mapping. ``model`` uses ``ChunkBCModelCfg``;
            ``data.manifest`` selects fixed HDF5 episodes and
            ``data.normalization`` is a tensor-only file containing the four
            frozen statistics accepted by ``model.set_normalization``.
            ``data.reader`` optionally supplies observation/target field names,
            handle capacity, checksums and an observation slice as
            ``[start, stop, step]``. Its explicit ``include_partial_targets``
            option defaults to false; true includes tail conditioning with
            finite right padding and a valid-target mask. Targets are
            precomputed by the host by default. A programmatic host may pass
            a pure target encoder/fields/version through the reader mapping;
            the model package never imports the host's task/controller.
            ``data.prefetch_batches`` is zero or one. ``optimizer`` accepts
            AdamW ``learning_rate``, ``weight_decay`` and ``max_grad_norm``;
            ``objective`` accepts only masked normalized target MSE.
            ``trainer`` uses the shared ``SupervisedTrainerCfg``.

    Returns:
        Common trainer with a caller-closed source and recipe-bound recovery.
        Task collection, target construction and physical evaluation remain
        host-owned; this factory makes no task qualification claim.
    """
    cfg = SupervisedTrainerCfg(**config.get("trainer", {}))
    cfg.validate()
    data = config["data"]
    prefetch = data.get("prefetch_batches", 1)
    if type(prefetch) is not int or prefetch not in (0, 1):
        raise ValueError("BC recipes support zero or one prefetched batch")
    objective = config.get("objective", {"type": "masked_normalized_target_mse"})
    if objective != {"type": "masked_normalized_target_mse"}:
        raise ValueError("Unknown BC objective; expected masked normalized target MSE")
    optimizer_cfg = config.get("optimizer", {})
    if set(optimizer_cfg) - {"learning_rate", "weight_decay", "max_grad_norm"}:
        raise ValueError("Unknown BC AdamW optimizer setting")
    rate = float(optimizer_cfg.get("learning_rate", 3e-4))
    decay = float(optimizer_cfg.get("weight_decay", 1e-6))
    maximum = float(optimizer_cfg.get("max_grad_norm", 1.0))
    if not (
        0 < rate < float("inf")
        and 0 <= decay < float("inf")
        and 0 < maximum < float("inf")
    ):
        raise ValueError(
            "BC optimizer values must be finite with positive rate/clipping"
        )
    torch.manual_seed(cfg.seed)
    model = ChunkBCModel(ChunkBCModelCfg(**config["model"]))
    statistics_path = Path(data["normalization"])
    model.set_normalization(
        **torch.load(statistics_path, map_location="cpu", weights_only=True)
    )
    model.to(device=cfg.device, dtype=torch.float32)
    reader = dict(data.get("reader", {}))
    if "observation_slice" in reader:
        reader["observation_slice"] = slice(*reader["observation_slice"])
    dataset = BCWindowDataset(
        data["manifest"],
        history_length=model.cfg.history_length,
        prediction_horizon=model.cfg.prediction_horizon,
        **reader,
    )
    try:
        if dataset.cameras != model.cfg.cameras:
            raise ValueError("BC dataset camera order differs from the model contract")
        optimizer = torch.optim.AdamW(model.parameters(), lr=rate, weight_decay=decay)
        learner = SupervisedLearner(
            model, ChunkBCObjective(), optimizer, max_grad_norm=maximum
        )
        source = HDF5BCSource(dataset)
        partial_signature = (
            {
                "partial_targets": {
                    "include_partial_targets": True,
                    "padding": "finite-zero-right-padding-valid-mask@1",
                }
            }
            if dataset.include_partial_targets
            else {}
        )
        return SupervisedTrainer(
            learner,
            PrefetchedBCSource(source) if prefetch else source,
            cfg,
            signature={
                "recipe": "chunk-bc-stored-targets-v1",
                "model": model.get_extra_state(),
                "objective": objective,
                "optimizer": {
                    "type": "AdamW",
                    "learning_rate": rate,
                    "weight_decay": decay,
                    "max_grad_norm": maximum,
                },
                "normalization_sha256": hashlib.sha256(
                    statistics_path.read_bytes()
                ).hexdigest(),
                "data_source": dataset.source_id,
                **partial_signature,
            },
        )
    except BaseException:
        dataset.close()
        raise
