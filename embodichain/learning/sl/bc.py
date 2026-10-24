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

"""Behavior-cloning objectives shared by numeric and visual target predictors."""

from __future__ import annotations

import torch

from embodichain.data_pipeline.datasets.bc_hdf5 import BCWindowDataset
from embodichain.learning.models.chunk_bc import ChunkBCModel
from .types import LossResult, SupervisedBatch

__all__ = ["ChunkBCObjective", "HDF5BCSource"]


class ChunkBCObjective:
    """Fit valid current/future targets from causal measured input histories.

    The data adapter supplies physical target vectors. This objective uses the
    model's frozen normalization and averages over valid target positions and
    features. Target construction and controller decoding remain task-owned.
    """

    def compute_loss(self, model: ChunkBCModel, batch: SupervisedBatch) -> LossResult:
        """Build the masked normalized target MSE.

        Args:
            model: State-only or visual chunk predictor.
            batch: ``history`` and optional ``images`` inputs, physical target
                tensor ``[B,F,D]`` and Boolean ``target_valid[B,F]`` mask.

        Returns:
            Mean loss with valid target count as evaluation aggregation weight.
        """
        predicted = model(batch.inputs["history"], batch.inputs.get("images"))
        target = batch.targets
        valid = batch.masks["target_valid"]
        if (
            target.shape != predicted.shape
            or target.dtype != torch.float32
            or valid.dtype != torch.bool
            or valid.shape != predicted.shape[:2]
            or not valid.any()
        ):
            raise ValueError(
                "BC supervision requires aligned targets and a nonempty valid mask"
            )
        normalized = (target - model.target_mean) / model.target_std
        loss = (predicted - normalized).square()[valid].mean()
        return LossResult(
            loss, {"normalized_mse": loss.detach()}, weight=float(valid.sum())
        )


class HDF5BCSource:
    """Adapt fixed HDF5 windows to the shared supervised batch interface.

    Args:
        dataset: Caller-owned bounded episode reader. The caller also owns
            closing its file handles after training or failure.
    """

    def __init__(self, dataset: BCWindowDataset) -> None:
        self.dataset = dataset
        self.source_id = dataset.source_id

    def sample(self, batch_size: int, *, generator: torch.Generator) -> SupervisedBatch:
        """Draw a batch with explicit sampling RNG and source identity.

        Args:
            batch_size: Number of complete history/target windows.
            generator: Caller-owned CPU sampling RNG.

        Returns:
            Owned numeric/image histories and physical target supervision.
        """
        return SupervisedBatch(**self.dataset.sample(batch_size, generator=generator))

    def close(self) -> None:
        """Close cached dataset handles."""
        self.dataset.close()
