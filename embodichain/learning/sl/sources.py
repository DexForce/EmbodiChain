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

"""Adapters for fixed CPU tensor corpora without simulator dependencies."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
import hashlib
import json
import copy
from typing import Any, Protocol
import torch
from .types import SupervisedBatch, _map_tensors

__all__ = ["SupervisedSource", "TensorBatchSource"]


class SupervisedSource(Protocol):
    """Owned batches drawn with an explicit caller-owned CPU sampling RNG."""

    source_id: str

    def sample(self, batch_size: int, *, generator: torch.Generator) -> SupervisedBatch:
        """Draw a batch without hidden waits, production or lifecycle changes.

        Args:
            batch_size: Number of training examples.
            generator: Caller-owned sampling generator.

        Returns:
            An independently owned batch with source-specific tensor axes.
        """
        ...


class TensorBatchSource:
    """Fixed tensor-tree examples with explicit content identity.

    Args:
        batch: Resident CPU examples; every tensor has the same first batch axis.
        source_id: Optional durable corpus identity. A content hash is always
            retained so reusing an ID cannot silently replace training data.

    Construction takes a detached owned snapshot. This source has no simulator,
    action-schema, sequence-length or image-layout assumption.
    """

    def __init__(self, batch: SupervisedBatch, *, source_id: str = "tensor") -> None:
        sizes = []
        digest = hashlib.sha256()

        def snapshot(tensor: torch.Tensor) -> torch.Tensor:
            if tensor.device.type != "cpu" or tensor.ndim == 0:
                raise ValueError("Tensor corpora require batched CPU tensors")
            sizes.append(tensor.shape[0])
            value = tensor.detach().clone().contiguous()
            digest.update(f"{tuple(value.shape)}:{value.dtype}".encode())
            digest.update(value.view(torch.uint8).numpy().tobytes())
            return value

        self.batch = SupervisedBatch(
            _map_tensors(batch.inputs, snapshot),
            _map_tensors(batch.targets, snapshot),
            _map_tensors(batch.masks, snapshot),
            copy.deepcopy(dict(batch.metadata)),
        )
        if not sizes or min(sizes) < 1 or len(set(sizes)) != 1:
            raise ValueError("All corpus tensors require the same nonempty batch axis")

        def key_signature(key: Any) -> Any:
            if isinstance(key, tuple):
                return {"kind": "tuple", "items": [key_signature(item) for item in key]}
            if key is None or type(key) in (str, int, float, bool):
                return {"kind": type(key).__name__, "value": key}
            raise ValueError(
                "Tensor-tree mapping keys must be scalar primitives or tuples"
            )

        def structure(value: Any) -> Any:
            if isinstance(value, torch.Tensor):
                return {"shape": list(value.shape), "dtype": str(value.dtype)}
            if isinstance(value, Mapping):
                return {
                    "kind": "mapping",
                    "items": [
                        [key_signature(key), structure(item)]
                        for key, item in value.items()
                    ],
                }
            if isinstance(value, (tuple, list)):
                return {
                    "kind": "tuple" if isinstance(value, tuple) else "list",
                    "items": [structure(item) for item in value],
                }
            return value

        digest.update(
            json.dumps(
                {
                    "inputs": structure(batch.inputs),
                    "targets": structure(batch.targets),
                    "masks": structure(batch.masks),
                    "metadata": batch.metadata,
                },
                sort_keys=True,
                allow_nan=False,
            ).encode()
        )
        self.size = sizes[0]
        self.source_id = f"{source_id}:{digest.hexdigest()}"

    def _gather(self, indices: torch.Tensor) -> SupervisedBatch:
        gather = lambda tensor: tensor[indices]
        return SupervisedBatch(
            _map_tensors(self.batch.inputs, gather),
            _map_tensors(self.batch.targets, gather),
            _map_tensors(self.batch.masks, gather),
            {
                **copy.deepcopy(self.batch.metadata),
                "source_id": self.source_id,
                "indices": indices.clone(),
            },
        )

    def sample(self, batch_size: int, *, generator: torch.Generator) -> SupervisedBatch:
        """Sample examples with replacement using an explicit RNG.

        Args:
            batch_size: Positive number of examples.
            generator: CPU sampling RNG.

        Returns:
            Owned tensor trees and sampled source indices.
        """
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be positive")
        return self._gather(
            torch.randint(self.size, (batch_size,), generator=generator)
        )

    def iter_batches(self, batch_size: int) -> Iterator[SupervisedBatch]:
        """Visit all examples once in stable source order, retaining the tail.

        Args:
            batch_size: Positive maximum batch size.

        Yields:
            Owned consecutive batches, including the final partial batch.
        """
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be positive")
        for start in range(0, self.size, batch_size):
            yield self._gather(torch.arange(start, min(start + batch_size, self.size)))
