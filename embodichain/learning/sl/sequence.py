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

"""Optional causal-data bridge with explicit encoding and recovery capability."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import copy
import hashlib
import json
from typing import Any

import torch
from tensordict import TensorDict

from embodichain.data_pipeline.datasets.schema import data_schema_fingerprint
from embodichain.data_pipeline.datasets.sequence import BatchSource, SequenceSpecCfg
from embodichain.data_pipeline.datasets.sources import TensorDataSource

from .types import SupervisedBatch

__all__ = ["SequenceBatchSource"]


def _json(value: Any) -> str:
    # Normalize JSON's integer mapping keys (notably language dictionaries)
    # before sorting; storage diagnostics do not enter this representation.
    return json.dumps(
        json.loads(json.dumps(value, allow_nan=False)),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _hash_tensor(digest: Any, value: torch.Tensor) -> None:
    """Visit logical C-order bytes with at most 1 MiB of contiguous staging."""
    limit = 1024 * 1024
    if value.numel() * value.element_size() <= limit:
        chunk = value.detach().contiguous().reshape(-1).view(torch.uint8)
        digest.update(memoryview(chunk.numpy()))
        return
    row_bytes = value[0].numel() * value.element_size()
    if row_bytes > limit:
        for index in range(value.shape[0]):
            _hash_tensor(digest, value[index])
    else:
        count = max(1, limit // row_bytes)
        for start in range(0, value.shape[0], count):
            _hash_tensor(digest, value[start : start + count])


class SequenceBatchSource:
    """Encode one caller-owned causal source without managing its production.

    Args:
        source: Live or resident ``BatchSource``. It owns sampling, continuity,
            padding, delivery quotas and copied window storage.
        sequence: Frozen history/future, segment and generation selection.
        encoder: Explicit pure function receiving one owned canonical context
            and returning ``SupervisedBatch``. Select measured inputs, targets
            and objective masks deliberately; future observations are never
            assigned to inputs by this adapter. Returned tensors must belong
            to this context or newly allocated storage, not a reused cache.
        encoder_version: Persisted identity of the encoding, including semantic
            checks, units, feature order and any captured configuration. Callable
            code is not inspected; the caller must version changes truthfully.
        immutable: Explicit promise that resident data and semantic metadata
            remain unchanged. Exact recovery additionally requires a
            ``TensorDataSource`` (including resident ``LeRobotDataSource``)
            without delivery quotas. The default refuses exact recovery.

    The reserved ``metadata['sequence_source']`` contains source identity,
    schema, sequence and ``context`` (canonical validity/loss masks and all
    provenance, excluding observations/actions). Optional ``storage`` is copied
    for diagnostics and excluded from identity. Encoder metadata cannot use
    this reserved key. Non-storage source metadata must be finite JSON data.
    No model/action meaning is inferred from tensor widths.

    Construction hashes immutable CPU payloads once, without a corpus copy,
    using bounded staging. Sampling checks tensor versions/structure and
    semantic declarations; recovery rehashes content. Do not mutate resident
    storage, including through aliases or untracked ``.data`` writes. Online
    buffers and delivery credits are not checkpointed: live/budgeted runs can
    train and save optimizer snapshots, but :meth:`validate_resume` rejects
    exact continuation. Start/stop, refill, waits and error recovery remain
    caller-owned. A delivered context is not refunded if encoding/update fails.
    """

    def __init__(
        self,
        source: BatchSource,
        sequence: SequenceSpecCfg,
        *,
        encoder: Callable[[TensorDict], SupervisedBatch],
        encoder_version: str,
        immutable: bool = False,
    ) -> None:
        if not callable(encoder):
            raise TypeError("Sequence encoding requires an explicit callable")
        if not isinstance(encoder_version, str) or not encoder_version.strip():
            raise ValueError("encoder_version must be a nonempty persisted identity")
        if type(immutable) is not bool:
            raise ValueError("immutable must be an explicit boolean")
        if not isinstance(source.source_id, str) or not source.source_id:
            raise ValueError("Sequence source requires a nonempty source_id")
        self.source = source
        self._sequence = copy.deepcopy(sequence)
        self._sequence.validate()
        self._sequence._check()
        self._encoder = encoder
        self._encoder_version = encoder_version
        self._immutable = immutable
        self._schema = copy.deepcopy(dict(source.schema))
        self._declaration = self._describe()
        self._content: str | None = None
        self._versions: list[Any] = []
        if immutable:
            if not isinstance(source, TensorDataSource):
                raise ValueError(
                    "Immutable recovery requires a resident TensorDataSource"
                )
            self._check_quota()
            self._content = self._hash_content()
            self._versions = self._tensor_versions()
        self.source_id = (
            "sequence:"
            + hashlib.sha256(
                _json([self._declaration, self._content]).encode()
            ).hexdigest()
        )

    def _describe(self) -> str:
        metadata = getattr(self.source, "metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError("Sequence source metadata must be a mapping")
        return _json(
            {
                "format": "supervised-sequence-v1",
                "source_type": f"{type(self.source).__module__}.{type(self.source).__qualname__}",
                "source_id": self.source.source_id,
                "schema": data_schema_fingerprint(self.source.schema),
                "sequence": self._sequence.to_dict(),
                "encoder_version": self._encoder_version,
                "immutable": self._immutable,
                "metadata": {
                    key: value for key, value in metadata.items() if key != "storage"
                },
            }
        )

    def _leaves(self) -> list[tuple[Any, torch.Tensor]]:
        assert isinstance(self.source, TensorDataSource)
        return sorted(
            self.source.data.items(include_nested=True, leaves_only=True),
            key=lambda item: (item[0],) if isinstance(item[0], str) else item[0],
        )

    def _hash_content(self) -> str:
        assert isinstance(self.source, TensorDataSource)
        digest = hashlib.sha256(_json(list(self.source.data.batch_size)).encode())
        for key, value in self._leaves():
            if value.device.type != "cpu" or value.layout != torch.strided:
                raise ValueError("Immutable sequence data must be strided CPU tensors")
            digest.update(_json([key, list(value.shape), str(value.dtype)]).encode())
            _hash_tensor(digest, value)
        return digest.hexdigest()

    def _tensor_versions(self) -> list[Any]:
        assert isinstance(self.source, TensorDataSource)
        return [tuple(self.source.data.batch_size)] + [
            (
                key,
                tuple(value.shape),
                value.dtype,
                value.stride(),
                value.data_ptr(),
                value._version,
            )
            for key, value in self._leaves()
        ]

    def _check_quota(self) -> None:
        assert isinstance(self.source, TensorDataSource)
        if self.source.get_sampling_budget_snapshot()["enabled"]:
            raise ValueError("Immutable exact recovery requires an unbudgeted source")

    def _check_source(self) -> None:
        if self._describe() != self._declaration:
            raise ValueError(
                "Sequence source identity, schema or semantic metadata changed"
            )
        if self._immutable:
            self._check_quota()
            if self._tensor_versions() != self._versions:
                raise ValueError("Immutable sequence source tensors changed")

    def sample(self, batch_size: int, *, generator: torch.Generator) -> SupervisedBatch:
        """Deliver one encoded owned window batch without retries or refunds.

        Args:
            batch_size: Number of contexts requested from the underlying source.
            generator: Explicit caller-owned CPU sampling generator.

        Returns:
            Encoded input/target/mask trees and aligned host-side source context.
        """
        self._check_source()
        context = self.source.sample_context(
            batch_size, copy.deepcopy(self._sequence), generator=generator
        )
        encoded = self._encoder(context)
        if not isinstance(encoded, SupervisedBatch):
            raise TypeError("Sequence encoder must return SupervisedBatch")
        if "sequence_source" in encoded.metadata:
            raise ValueError("Encoder metadata cannot use reserved sequence_source")
        metadata = {
            "source_id": self.source_id,
            "schema": copy.deepcopy(self._schema),
            "sequence": self._sequence.to_dict(),
            "context": context.exclude("obs", "actions"),
        }
        storage = getattr(self.source, "metadata", {}).get("storage")
        if storage is not None:
            metadata["storage"] = copy.deepcopy(storage)
        return SupervisedBatch(
            encoded.inputs,
            encoded.targets,
            encoded.masks,
            {**encoded.metadata, "sequence_source": metadata},
        )

    def validate_resume(self) -> None:
        """Reject non-reproducible source recovery before trainer state changes.

        Raises:
            ValueError: Source is live/unverified/budgeted, its declarations or
                resident payload changed, or its immutable promise was broken.
        """
        if not self._immutable:
            raise ValueError(
                "Exact resume requires an explicitly immutable unbudgeted resident source; "
                "live or quota-managed delivery state is not restored"
            )
        self._check_source()
        if self._hash_content() != self._content:
            raise ValueError("Immutable sequence source content changed")
