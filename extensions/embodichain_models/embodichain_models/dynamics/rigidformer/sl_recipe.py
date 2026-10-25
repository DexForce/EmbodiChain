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

"""Standalone RigidFormer geometry recipe for the shared SL trainer."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import torch

from embodichain.learning.sl import SupervisedBatch

from .supervised import RigidFormerObjective, rigidformer_batch

if TYPE_CHECKING:
    from embodichain.learning.sl.trainer import SupervisedTrainer

__all__ = ["build_rigidformer_trainer"]


class _RigidFormerSource:
    def __init__(self, source: Any) -> None:
        self.source = source
        self.source_id = source.source_id

    def sample(self, batch_size: int, *, generator: torch.Generator) -> SupervisedBatch:
        raw = self.source.sample_batch(batch_size, generator=generator)
        return rigidformer_batch(raw, metadata={"source_id": self.source_id})


def build_rigidformer_trainer(config: Mapping[str, Any]) -> SupervisedTrainer:
    """Build an offline geometry recipe for the unified ``train-sl`` command.

    Args:
        config: Recipe mapping with ``data.path`` pointing to a tensor-only
            PyTorch archive containing ``positions``, ``properties``, and
            ``step_dt``. Optional geometry lengths use their source field names.
            The ``model``, ``optimizer``, and ``trainer`` mappings configure the
            upstream network, AdamW, and shared trainer, respectively.

    Returns:
        Shared trainer consuming bounded causal geometry windows.
    """
    from pathlib import Path
    import hashlib

    from .data import RigidBodySequenceSource
    from .model import RigidFormerCfg, RigidFormerModel
    from embodichain.learning.sl import SupervisedLearner
    from embodichain.learning.sl.trainer import SupervisedTrainer, SupervisedTrainerCfg

    data_cfg = dict(config["data"])
    path = Path(data_cfg.pop("path"))
    source_id = data_cfg.pop("source_id", None)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    fingerprint = digest.hexdigest()
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    source = RigidBodySequenceSource(
        payload["positions"],
        payload["properties"],
        step_dt=payload["step_dt"],
        source_id=source_id or fingerprint,
        episode_lengths=payload.get("episode_lengths"),
        object_lens=payload.get("object_lens"),
        object_point_lens=payload.get("object_point_lens"),
        **data_cfg,
    )
    trainer_cfg = SupervisedTrainerCfg(**config.get("trainer", {}))
    torch.manual_seed(trainer_cfg.seed)
    model = RigidFormerModel(RigidFormerCfg(**config.get("model", {}))).to(
        trainer_cfg.device
    )
    optimizer_cfg = {"lr": 3e-4, "weight_decay": 1e-2, **config.get("optimizer", {})}
    learner = SupervisedLearner(
        model,
        RigidFormerObjective(),
        torch.optim.AdamW(model.parameters(), **optimizer_cfg),
        max_grad_norm=1.0,
    )
    return SupervisedTrainer(
        learner,
        _RigidFormerSource(source),
        trainer_cfg,
        signature={
            "geometry_sha256": fingerprint,
            "model": model.cfg.to_dict(),
            "optimizer": optimizer_cfg,
            "data": data_cfg,
            "objective": "rigidformer-reference-v1",
        },
    )
