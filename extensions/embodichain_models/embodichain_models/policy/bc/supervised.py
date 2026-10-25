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

from concurrent.futures import Future, ThreadPoolExecutor

import torch

from embodichain_models.policy.bc.data import BCWindowDataset
from embodichain_models.policy.bc.model import ChunkBCModel
from embodichain.learning.sl.types import LossResult, SupervisedBatch

__all__ = ["ChunkBCObjective", "HDF5BCSource", "PrefetchedBCSource"]


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


class PrefetchedBCSource:
    """Prepare at most one future fixed-corpus batch with private sampling RNG.

    Args:
        source: Fixed, deterministic HDF5 adapter with no random augmentation.

    A worker owns all sampling and HDF5 access. Its future RNG state is published
    to the supplied generator only when that batch is delivered. Checkpoints
    therefore preserve the committed sampler cursor; an unused future batch is
    reproducible from that cursor after restart. Changing the requested cursor
    or batch size discards an incompatible pending batch. Closing joins the
    worker before releasing HDF5 handles. This adapter does not acquire data.
    """

    def __init__(self, source: HDF5BCSource) -> None:
        self.source = source
        self.source_id = source.source_id
        self._pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="bc-offline-read"
        )
        self._future: Future[tuple[SupervisedBatch, torch.Tensor]] | None = None
        self._expected: torch.Tensor | None = None
        self._batch_size = 0
        self._closed = False

    def _submit(self, batch_size: int, state: torch.Tensor) -> None:
        self._expected = state.clone()
        self._batch_size = batch_size

        def draw() -> tuple[SupervisedBatch, torch.Tensor]:
            rng = torch.Generator(device="cpu")
            rng.set_state(state)
            return self.source.sample(batch_size, generator=rng), rng.get_state()

        self._future = self._pool.submit(draw)

    def sample(self, batch_size: int, *, generator: torch.Generator) -> SupervisedBatch:
        """Deliver one ordered batch and prepare one bounded successor.

        Args:
            batch_size: Number of complete history/target windows.
            generator: Caller-owned committed CPU sampling state.

        Returns:
            The same batch and next RNG state as direct synchronous sampling.
        """
        if self._closed:
            raise RuntimeError("BC prefetch source is closed")
        state = generator.get_state()
        if self._future is not None and (
            batch_size != self._batch_size or not torch.equal(state, self._expected)
        ):
            self._future.cancel()
            if not self._future.cancelled():
                self._future.result()
            self._future = None
        if self._future is None:
            self._submit(batch_size, state)
        batch, after = self._future.result()
        generator.set_state(after)
        self._submit(batch_size, after)
        return batch

    def close(self) -> None:
        """Drain/cancel the single pending batch and then close file handles."""
        if not self._closed:
            self._closed = True
            self._pool.shutdown(wait=True, cancel_futures=True)
            self.source.close()
