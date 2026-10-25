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

"""Budgeted immutable transition storage for long single-process training runs."""

from __future__ import annotations

import math
import os
from typing import Any

import numpy as np
import torch

__all__: list[str] = []


class _ResidentWindows:
    """Own contiguous raw fields; gather fresh bounded windows without HDF5 I/O."""

    def __init__(
        self, dataset: Any, device: str | torch.device, max_bytes: int
    ) -> None:
        if type(max_bytes) is not int or max_bytes < 1:
            raise ValueError("Resident cache budget must be a positive byte count")
        if not len(dataset):
            raise ValueError("Resident cache requires eligible trajectory windows")
        self.device = torch.device(device)
        self.pid = os.getpid()
        self.rows = dataset._rows
        self.ends = np.asarray(dataset._ends, dtype=np.int64)
        self.history = dataset.history_length
        self.include_prefix = dataset.include_prefix
        self.offsets = {}
        self.fields: dict[str, dict[str, torch.Tensor]] = {}
        required = 0
        for kind, fields in dataset.manifest["fields"].items():
            sizes = [row["length"] + int(kind == "states") for row in self.rows]
            self.offsets[kind] = np.concatenate(([0], np.cumsum(sizes, dtype=np.int64)))
            for spec in fields.values():
                required += (
                    int(self.offsets[kind][-1])
                    * math.prod(spec["shape"])
                    * np.dtype(spec["dtype"]).itemsize
                )
        if required > max_bytes:
            raise ValueError(
                f"Resident fields require {required} bytes, exceeding cache budget {max_bytes}"
            )
        self.nbytes = required
        for kind, fields in dataset.manifest["fields"].items():
            self.fields[kind] = {}
            for name, spec in fields.items():
                dtype = torch.from_numpy(
                    np.empty((), dtype=np.dtype(spec["dtype"]))
                ).dtype
                self.fields[kind][name] = torch.empty(
                    (int(self.offsets[kind][-1]), *spec["shape"]),
                    dtype=dtype,
                    device=self.device,
                )
        # Host staging remains bounded by the existing read-page size even for
        # very long trajectories. Copying never changes the persisted precision.
        for row_index, row in enumerate(self.rows):
            source = dataset._handle(row["file"])[row["group"]]
            for kind, fields in dataset.manifest["fields"].items():
                count = row["length"] + int(kind == "states")
                base = int(self.offsets[kind][row_index])
                for name, spec in fields.items():
                    for start in range(0, count, dataset.read_page_steps):
                        end = min(count, start + dataset.read_page_steps)
                        block = dataset._read_block(
                            source[kind],
                            name,
                            start,
                            end,
                            tuple(spec["shape"]),
                            np.dtype(spec["dtype"]),
                        )
                        self.fields[kind][name][base + start : base + end].copy_(
                            torch.from_numpy(block)
                        )

    def get_batch(self, requests: list[int]) -> dict[str, Any]:
        if os.getpid() != self.pid:
            raise RuntimeError("Resident dynamics caches are owned by one process")
        indices = np.asarray(requests, dtype=np.int64)
        indices = np.where(indices < 0, indices + self.ends[-1], indices)
        if np.any(indices < 0) or np.any(indices >= self.ends[-1]):
            raise IndexError("Resident dynamics window index out of range")
        rows = np.searchsorted(self.ends, indices, side="right")
        first = np.concatenate(([0], self.ends[:-1]))
        offsets = indices - first[rows]
        ends = offsets + (1 if self.include_prefix else self.history)
        starts = np.maximum(0, ends - self.history)
        lengths = ends - starts
        result = {
            "trajectory_id": [self.rows[index]["id"] for index in rows],
            "start": torch.from_numpy(starts),
            "length": torch.from_numpy(lengths),
            "valid_mask": torch.arange(self.history)[None, :]
            < torch.from_numpy(lengths)[:, None],
        }
        for kind, fields in self.fields.items():
            extra = int(kind == "states")
            take = (
                self.offsets[kind][rows, None]
                + starts[:, None]
                + np.minimum(
                    np.arange(self.history + extra)[None, :],
                    (lengths - 1 + extra)[:, None],
                )
            )
            take = torch.as_tensor(take, device=self.device)
            # Advanced indexing returns independent tensors, including repeated
            # rows. Later seeks/close cannot mutate a returned learner batch.
            result[kind] = {name: value[take] for name, value in fields.items()}
        return result
