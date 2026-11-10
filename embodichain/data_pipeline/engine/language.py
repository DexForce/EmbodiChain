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

"""Append-only shared language dictionaries for online trajectory consumers."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
from multiprocessing.context import BaseContext
from typing import Any, Literal

import torch

__all__ = ["SharedLanguageRegistry"]

_LanguageKind = Literal["task", "subtask"]


class SharedLanguageRegistry:
    """Share immutable text indices without a multiprocessing manager server.

    Task and subtask indices have separate dictionaries. New descriptions append
    UTF-8 JSON records to a bounded shared byte buffer; previous indices never
    change, even when their trajectory slots are overwritten. Each process keeps
    its own incremental lookup cache and uses the shared length lock to publish
    or read complete records.

    Args:
        capacity_bytes: Maximum encoded registry size for the engine's lifetime.
        context: Multiprocessing context used by the producer and consumers.

    Raises:
        ValueError: If capacity is not a positive integer.
    """

    def __init__(
        self,
        capacity_bytes: int = 1024 * 1024,
        context: BaseContext | None = None,
    ) -> None:
        if type(capacity_bytes) is not int or capacity_bytes < 1:
            raise ValueError("language_buffer_bytes must be a positive integer")
        context = context or mp.get_context("forkserver")
        self._buffer = context.RawArray("B", capacity_bytes)
        self._length = context.Value("Q", 0)
        self._reset_cache()

    def _reset_cache(self) -> None:
        self._cache_pid = os.getpid()
        self._offset = 0
        self._texts: dict[str, list[str]] = {"task": [], "subtask": []}
        self._indices: dict[str, dict[str, int]] = {"task": {}, "subtask": {}}

    @staticmethod
    def _validate_kind(kind: str) -> None:
        if kind not in {"task", "subtask"}:
            raise ValueError("language kind must be task or subtask")

    def _sync_locked(self) -> None:
        if self._cache_pid != os.getpid():
            self._reset_cache()
        end = self._length.value
        if end == self._offset:
            return
        payload = bytes(self._buffer[self._offset : end])
        for line in payload.splitlines():
            kind, description = json.loads(line)
            index = len(self._texts[kind])
            self._texts[kind].append(description)
            self._indices[kind][description] = index
        self._offset = end

    def intern(self, description: str, *, kind: _LanguageKind = "task") -> int:
        """Return an existing text index or append one immutable dictionary entry.

        Args:
            description: Nonempty language snapshot, preserved exactly.
            kind: The task or subtask dictionary to update.

        Returns:
            Stable zero-based index in the selected dictionary.

        Raises:
            ValueError: If the description or dictionary kind is invalid.
            OverflowError: If a new description exceeds the shared capacity.
                Existing descriptions remain resolvable after overflow.
        """
        self._validate_kind(kind)
        if not isinstance(description, str) or not description.strip():
            raise ValueError("language description must be nonempty text")
        with self._length.get_lock():
            self._sync_locked()
            existing = self._indices[kind].get(description)
            if existing is not None:
                return existing
            payload = (
                json.dumps(
                    [kind, description], ensure_ascii=False, separators=(",", ":")
                )
                + "\n"
            ).encode("utf-8")
            end = self._length.value + len(payload)
            if end > len(self._buffer):
                raise OverflowError(
                    "Online language registry exhausted language_buffer_bytes="
                    f"{len(self._buffer)}; increase the capacity and create a new engine."
                )
            self._buffer[self._length.value : end] = payload
            self._length.value = end
            self._sync_locked()
            return self._indices[kind][description]

    def resolve(self, indices: torch.Tensor, *, kind: _LanguageKind = "task") -> Any:
        """Decode a tensor of indices while preserving its nested batch shape.

        Args:
            indices: Integer task/subtask indices from a copied trajectory batch.
            kind: Dictionary that produced these indices.

        Returns:
            A string for a scalar index, otherwise nested Python lists of strings
            with the same shape as ``indices``.

        Raises:
            TypeError: If indices are not an integer tensor.
            ValueError: If any index is unknown, including invalid padding ``-1``.
        """
        self._validate_kind(kind)
        if not isinstance(indices, torch.Tensor) or indices.dtype not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }:
            raise TypeError("language indices must be an integer tensor")
        values = indices.detach().cpu().tolist()
        with self._length.get_lock():
            self._sync_locked()
            texts = self._texts[kind]

            def decode(value: Any) -> Any:
                if isinstance(value, list):
                    return [decode(item) for item in value]
                if not 0 <= value < len(texts):
                    raise ValueError(f"Unknown online {kind} index {value}")
                return texts[value]

            return decode(values)

    def __getstate__(self) -> dict[str, Any]:
        """Share storage handles, rebuilding process-local caches after spawn."""
        return {"_buffer": self._buffer, "_length": self._length}

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore the shared handles with an independent lookup cache."""
        self.__dict__.update(state)
        self._reset_cache()
