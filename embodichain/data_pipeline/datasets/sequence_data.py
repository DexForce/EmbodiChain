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

"""PyTorch iteration shared by resident and live causal data sources."""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterator

import torch
from tensordict import TensorDict
from torch.utils.data import IterableDataset, get_worker_info

from ._sampling import _pad_context
from .sequence import BatchSource, SequenceSpecCfg

__all__ = ["SequenceDataset"]


class SequenceDataset(IterableDataset):
    """Iterate owned causal sequences from a live or resident data source.

    Args:
        source: BatchSource, such as OnlineDataSource, LeRobotDataSource or
            TensorDataSource. The caller owns its loading/production lifecycle.
        sequence: Causal window and boundary requirements.
        batch_size: None yields one context for DataLoader collation; a positive
            integer yields complete batches (use DataLoader batch_size=None).
        transform: Optional batch/item transform preserving axes and masks.
        generator: Optional CPU sampling RNG. Explicit generators require
            num_workers=0 to prevent copied workers repeating the same stream.
        max_batches: Optional positive draws per iterator. Stored cardinality
            does not imply iteration length: random sampling otherwise repeats.
            Finite iteration requires num_workers=0.

    There is no inferred epoch length or automatic refill. ``max_batches`` is
    a bound on each iterator, not a source quota or a checkpointed update count.
    Sampling charges delivery before any transform or optimizer update; a
    failed consumer does not refund it. Prefetched batches also count.

    Quota exhaustion and source errors propagate immediately. Budgeted sources
    reject unsupported worker copies; this wrapper never clones allowances,
    retries permanent failures or guesses an epoch length.
    """

    def __init__(
        self,
        source: BatchSource,
        sequence: SequenceSpecCfg,
        *,
        batch_size: int | None = None,
        transform: Callable[[TensorDict], TensorDict] | None = None,
        generator: torch.Generator | None = None,
        max_batches: int | None = None,
    ) -> None:
        super().__init__()
        if batch_size is not None and (type(batch_size) is not int or batch_size < 1):
            raise ValueError("batch_size must be None or a positive integer")
        if max_batches is not None and (
            type(max_batches) is not int or max_batches < 1
        ):
            raise ValueError("max_batches must be None or a positive integer")
        sequence._check()
        self.source = source
        self.sequence = copy.deepcopy(sequence)
        self.batch_size = batch_size
        self.transform = transform
        self.generator = generator
        self.max_batches = max_batches

    def __iter__(self) -> Iterator[TensorDict]:
        """Yield copied contexts until the explicit iteration limit or an error.

        Returns:
            Iterator of TensorDicts, prebatched when batch_size is specified.
        """
        if get_worker_info() is not None and (
            self.generator is not None or self.max_batches is not None
        ):
            raise RuntimeError(
                "Explicit generators and finite iteration limits require owner-process Dataset consumption"
            )
        index = 0
        while self.max_batches is None or index < self.max_batches:
            batch = self.source.sample_context(
                self.batch_size or 1, self.sequence, generator=self.generator
            )
            sample = batch[0] if self.batch_size is None else batch
            if self.transform is not None:
                axes = sample.batch_size
                sample = self.transform(sample)
                if not isinstance(sample, TensorDict) or sample.batch_size != axes:
                    raise ValueError(
                        "Context transforms must preserve TensorDict batch axes"
                    )
                _pad_context(sample, sample["valid"], sample["loss_mask"])
            yield sample
            index += 1

    def resolve_language(self, batch: TensorDict) -> dict[str, object]:
        """Resolve source-local language IDs when the source supplies dictionaries.

        Args:
            batch: Copied numeric batch containing task/subtask indices.

        Returns:
            Overall and segment text with the batch's nested shape.
        """
        resolver = getattr(self.source, "resolve_language", None)
        if resolver is None:
            raise ValueError("Source does not provide language dictionaries")
        return resolver(batch)

    @staticmethod
    def collate_fn(batch: list[TensorDict]) -> TensorDict:
        """Stack contexts for item-mode DataLoader use.

        Args:
            batch: Individual owned contexts.

        Returns:
            TensorDict with a new leading batch axis.
        """
        return torch.stack(batch)

    @staticmethod
    def passthrough_collate_fn(batch: TensorDict) -> TensorDict:
        """Pass through an already batched item.

        Args:
            batch: Source-owned copied minibatch.

        Returns:
            The supplied TensorDict unchanged.
        """
        return batch
