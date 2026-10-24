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

"""RigidFormer objective and conversion of explicit geometry supervision."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from embodichain.learning.sl import LossResult, SupervisedBatch

if TYPE_CHECKING:
    from embodichain.learning.sl.trainer import SupervisedTrainer

__all__ = ["RigidFormerObjective", "rigidformer_batch", "build_rigidformer_trainer"]

_INPUTS = (
    "object_pos",
    "object_pos_prev",
    "object_first_frame_pos",
    "vertex_properties",
    "delta_times",
)
_OPTIONAL = ("anchor_indices", "object_point_lens", "object_lens")


def rigidformer_batch(
    values: Mapping[str, torch.Tensor], *, metadata: Mapping[str, Any] | None = None
) -> SupervisedBatch:
    """Separate next-frame targets from causal RigidFormer inputs.

    Args:
        values: Owned geometry tensors in the upstream keyword layout.
        metadata: Optional source identity and frame provenance.

    Returns:
        A supervised batch with target geometry outside the input mapping.
    """
    missing = set((*_INPUTS, "object_pos_next")) - values.keys()
    if missing:
        raise ValueError(f"Missing geometry fields: {sorted(missing)}")
    inputs = {key: values[key] for key in (*_INPUTS, *_OPTIONAL) if key in values}
    context = dict(metadata or {})
    context.update(
        {key: values[key] for key in ("episode_index", "frame_index") if key in values}
    )
    return SupervisedBatch(
        inputs=inputs,
        targets=values["object_pos_next"],
        metadata=context,
    )


class RigidFormerObjective:
    """Preserve upstream anchor position/acceleration losses and their weights."""

    def compute_loss(self, model: nn.Module, batch: SupervisedBatch) -> LossResult:
        """Compute loss without placing the next-frame target in model inputs.

        Args:
            model: RigidFormer-compatible model returning loss and breakdown.
            batch: Explicit geometry inputs and next-frame targets.

        Returns:
            Scalar loss and upstream position/acceleration diagnostic terms.
        """
        if batch.masks:
            raise ValueError(
                "RigidFormer masks use object_lens/object_point_lens inputs; "
                "arbitrary temporal loss masks are unsupported"
            )
        loss, terms = model(**batch.inputs, object_pos_next=batch.targets)
        return LossResult(
            loss=loss,
            metrics={"acceleration": terms.acceleration, "position": terms.position},
            weight=float(batch.targets.shape[0]),
        )


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

    from embodichain.data_pipeline.datasets.rigid_body import RigidBodySequenceSource
    from embodichain.learning.models.rigidformer import RigidFormerCfg, RigidFormerModel
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
