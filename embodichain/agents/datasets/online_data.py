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

from __future__ import annotations

from typing import Callable, Iterator, List, Optional

from tensordict import TensorDict
from torch.utils.data import IterableDataset

from embodichain.agents.engine.data import OnlineDataEngine
from embodichain.agents.datasets.sampler import ChunkSizeSampler


__all__ = [
    "OnlineDataset",
]


class OnlineDataset(IterableDataset):
    """Infinite IterableDataset backed by a live OnlineDataEngine shared buffer.

    Two sampling modes are supported depending on the ``batch_size`` argument:

    **Item mode** (``batch_size=None``, default)
        ``__iter__`` yields one ``TensorDict`` of shape ``[chunk_size]`` per step.
        Use with a standard ``DataLoader(dataset, batch_size=B)`` so the
        DataLoader handles collation and worker sharding.

    **Batch mode** (``batch_size=N``)
        ``__iter__`` yields one pre-batched ``TensorDict`` of shape
        ``[N, chunk_size]`` per step by calling
        ``engine.sample_batch(N, chunk_size)`` directly.
        Use with ``DataLoader(dataset, batch_size=None)`` to skip DataLoader
        collation and leverage the engine's bulk-sampling efficiency.

    **Dynamic chunk sizes**
        Pass a :class:`ChunkSizeSampler` as ``chunk_size`` to draw a fresh
        chunk length on every iteration step.  In batch mode the size is
        sampled once per step and applied uniformly to all trajectories in
        the batch, ensuring a consistent ``[batch_size, chunk_size]`` shape.
        Two built-in samplers are provided:

        - :class:`UniformChunkSampler` — uniform discrete distribution over
          ``[low, high]``.
        - :class:`GMMChunkSampler` — Gaussian Mixture Model, useful for
          multi-modal chunk-length curricula.

    .. note::
        ``__len__`` is intentionally absent — ``IterableDataset`` does not
        require it and the stream is infinite.

    .. note::
        Multi-worker DataLoader: each worker gets its own iterator; since
        sampling is independent random draws from shared memory, this is safe.

    Args:
        engine: A started OnlineDataEngine whose shared buffer is used for
            sampling.
        chunk_size: Fixed number of consecutive timesteps per chunk (``int``),
            or a :class:`ChunkSizeSampler` that returns a fresh size on every
            iteration step.
        batch_size: If ``None``, yield single chunks of shape ``[chunk_size]``
            (item mode). If an int, yield pre-batched TensorDicts of shape
            ``[batch_size, chunk_size]`` (batch mode).
        transform: Optional ``(TensorDict) -> TensorDict`` applied to each
            yielded item/batch before returning.

    Example — fixed chunk size, item mode::

        dataset = OnlineDataset(engine, chunk_size=64)
        loader  = DataLoader(dataset, batch_size=32, num_workers=4,
                             collate_fn=OnlineDataset.collate_fn)
        for batch in loader:
            # batch has shape [32, 64, ...]
            train_step(batch)

    Example — fixed chunk size, batch mode::

        dataset = OnlineDataset(engine, chunk_size=64, batch_size=32)
        loader  = DataLoader(dataset, batch_size=None,
                             collate_fn=OnlineDataset.passthrough_collate_fn)
        for batch in loader:
            # batch has shape [32, 64, ...]
            train_step(batch)

    Example — dynamic chunk size with uniform sampler::

        sampler = UniformChunkSampler(low=16, high=64)
        dataset = OnlineDataset(engine, chunk_size=sampler)
        loader  = DataLoader(dataset, batch_size=32)
        for batch in loader:
            # chunk dimension varies each batch
            train_step(batch)

    Example — dynamic chunk size with GMM sampler::

        sampler = GMMChunkSampler(
            means=[16.0, 64.0], stds=[4.0, 8.0], weights=[0.6, 0.4],
            low=8, high=96,
        )
        dataset = OnlineDataset(engine, chunk_size=sampler, batch_size=32)
        loader  = DataLoader(dataset, batch_size=None)
        for batch in loader:
            train_step(batch)
    """

    def __init__(
        self,
        engine: OnlineDataEngine,
        chunk_size: int | ChunkSizeSampler,
        batch_size: Optional[int] = None,
        transform: Optional[Callable[[TensorDict], TensorDict]] = None,
    ) -> None:
        if isinstance(chunk_size, int):
            if chunk_size < 1:
                raise ValueError(f"chunk_size must be ≥ 1, got {chunk_size}.")
        elif not isinstance(chunk_size, ChunkSizeSampler):
            raise TypeError(
                f"chunk_size must be an int or a ChunkSizeSampler, got {type(chunk_size).__name__}."
            )
        self._engine = engine
        self._chunk_size = chunk_size
        self._batch_size = batch_size
        self._transform = transform

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _next_chunk_size(self) -> int:
        """Return the chunk size for the current iteration step.

        For fixed ``int`` chunk sizes this is a no-op attribute read.
        For :class:`ChunkSizeSampler` instances the sampler is called to draw
        a fresh value.

        Returns:
            Positive integer chunk size.
        """
        if isinstance(self._chunk_size, int):
            return self._chunk_size
        return self._chunk_size()

    # ------------------------------------------------------------------
    # IterableDataset interface
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[TensorDict]:
        """Yield trajectory chunks indefinitely from the shared buffer.

        In item mode each call to ``next()`` draws one chunk of shape
        ``[chunk_size]``.  In batch mode each call draws a full batch of
        shape ``[batch_size, chunk_size]``.  When a :class:`ChunkSizeSampler`
        is used, ``chunk_size`` is re-sampled once per yielded item/batch.

        Yields:
            TensorDict sampled from the engine's shared buffer, optionally
            post-processed by ``transform``.
        """
        if self._batch_size is None:
            # In item mode, keep chunk_size fixed per iterator to preserve
            # consistent shapes for DataLoader collation.
            chunk_size = self._next_chunk_size()

            while True:
                # Item mode: draw one trajectory and remove the outer batch dim.
                raw = self._engine.sample_batch(batch_size=1, chunk_size=chunk_size)
                sample: TensorDict = raw[0]

                if self._transform is not None:
                    sample = self._transform(sample)

                yield sample

        while True:
            chunk_size = self._next_chunk_size()

            # Batch mode: draw a full pre-batched TensorDict.
            sample = self._engine.sample_batch(
                batch_size=self._batch_size, chunk_size=chunk_size
            )

            if self._transform is not None:
                sample = self._transform(sample)

            yield sample

    @staticmethod
    def collate_fn(batch: List[TensorDict]) -> TensorDict:
        """Collate a list of TensorDicts into a single batched TensorDict.

        Pass this as ``collate_fn`` to ``DataLoader`` when using item mode
        (``batch_size`` not None on the DataLoader side) to avoid the default
        collation failure with TensorDict objects.

        Args:
            batch: List of TensorDicts, each of shape ``[chunk_size, ...]``.

        Returns:
            Stacked TensorDict of shape ``[len(batch), chunk_size, ...]``.
        """
        import torch

        return torch.stack(batch)

    @staticmethod
    def passthrough_collate_fn(batch: TensorDict) -> TensorDict:
        """Collate function for batch-mode DataLoaders.

        When the dataset is in batch mode it already yields pre-batched
        TensorDicts.  With ``batch_size=None``, PyTorch's DataLoader skips
        auto-batching and passes each item directly to ``collate_fn`` as-is
        (not wrapped in a list).  This function returns the TensorDict
        unchanged.

        Pass this as ``collate_fn`` to ``DataLoader`` when using batch mode
        (``batch_size=None`` on the DataLoader side) to avoid the default
        collation failure with TensorDict objects.

        Args:
            batch: A pre-batched TensorDict of shape
                ``[batch_size, chunk_size, ...]`` passed directly by the
                DataLoader.

        Returns:
            The pre-batched TensorDict unchanged.
        """
        return batch
