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

"""Bounded storage for exact shared Torch minibatch sampling streams.

Store RNG states at fixed block boundaries instead of materializing every
sample index. The emitted stream remains identical to the original matrix.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

__all__: list[str] = []
_FORMAT = "embodichain.nerd.sample_order.v1"
_OFFICIAL_FORMAT = "embodichain.nerd.official_shuffle_order.v1"
_MAX_PERMUTATION_BYTES = 32 * 1024**2


def _official_dimensions(
    dataset_size: int, steps: int, batch_size: int, epoch_steps: int
) -> tuple[int, int, int]:
    """Bound the source iterator schedule and return pass dimensions."""
    for value in (dataset_size, steps, batch_size, epoch_steps):
        if type(value) is not int or value < 1:
            raise ValueError("Official shuffle dimensions must be positive integers")
    if dataset_size * 8 > _MAX_PERMUTATION_BYTES:
        raise ValueError("Official shuffle permutation exceeds 32 MiB")
    batches = dataset_size // batch_size
    if not batches or epoch_steps < batches:
        raise ValueError(
            "Official shuffle requires an exhausted source iterator each epoch"
        )
    passes_per_epoch = (epoch_steps + batches - 1) // batches
    epochs, tail = divmod(steps, epoch_steps)
    passes = epochs * passes_per_epoch + ((tail + batches - 1) // batches)
    return batches, passes_per_epoch, passes


def _permutation_sha256(value: np.ndarray) -> str:
    """Hash native little-endian contiguous int64 indices without a full copy."""
    if (
        sys.byteorder != "little"
        or value.dtype != np.int64
        or not value.flags.c_contiguous
    ):
        raise ValueError(
            "Official permutation hash requires contiguous little-endian int64"
        )
    return hashlib.sha256(memoryview(value).cast("B")).hexdigest()


def _write_official_shuffle_order(
    path: Path,
    *,
    dataset_size: int,
    steps: int,
    batch_size: int,
    epoch_steps: int,
    pass_seeds: np.ndarray | Sequence[int],
    permutation_sha256: Sequence[str],
    provenance: dict[str, Any],
) -> tuple[Path, Path, Path]:
    """Persist captured original sampler seeds and full permutation digests.

    The caller must establish original global-RNG event provenance. This writer
    never invents seeds or equates an independent shuffle with official replay.
    Epoch boundaries discard the unfinished final permutation, matching the
    pinned source's exhausted outer iterator. Hashes bind each actual CPU
    randperm so another Torch version must prove its output before returning a
    minibatch. Old with-replacement descriptors retain their original format.
    """
    _, _, passes = _official_dimensions(dataset_size, steps, batch_size, epoch_steps)
    seeds = np.asarray(pass_seeds)
    if seeds.dtype != np.int64 or seeds.shape != (passes,) or np.any(seeds < 0):
        raise ValueError(
            "Official shuffle requires one nonnegative int64 seed per pass"
        )
    hashes = list(permutation_sha256)
    if len(hashes) != passes or any(
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
        for value in hashes
    ):
        raise ValueError("Official shuffle requires one complete SHA256 per pass")
    if not isinstance(provenance, dict) or not provenance:
        raise ValueError("Official shuffle requires source RNG provenance")
    provenance = json.loads(json.dumps(provenance, allow_nan=False))
    path = Path(path)
    if path.suffix != ".json":
        raise ValueError("Official shuffle descriptor must use .json")
    seeds_path = path.with_name(path.stem + "-seeds.npy")
    hashes_path = path.with_name(path.stem + "-permutation-sha256.npy")
    if any(value.exists() for value in (path, seeds_path, hashes_path)):
        raise FileExistsError(path)
    with seeds_path.open("xb") as stream:
        np.save(stream, seeds, allow_pickle=False)
    with hashes_path.open("xb") as stream:
        np.save(stream, np.asarray(hashes, dtype="S64"), allow_pickle=False)
    descriptor = {
        "format": _OFFICIAL_FORMAT,
        "dataset_size": dataset_size,
        "steps": steps,
        "batch_size": batch_size,
        "epoch_steps": epoch_steps,
        "drop_last": True,
        "discard_epoch_tail": True,
        "permutation_algorithm": "torch.randperm-cpu-int64-little-endian",
        "seeds_file": seeds_path.name,
        "seeds_sha256": _digest(seeds_path),
        "permutation_hashes_file": hashes_path.name,
        "permutation_hashes_sha256": _digest(hashes_path),
        "provenance": provenance,
    }
    with path.open("x") as stream:
        json.dump(descriptor, stream, sort_keys=True, indent=2, allow_nan=False)
    return path, seeds_path, hashes_path


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _write_checkpointed_order(
    path: Path,
    *,
    dataset_size: int,
    steps: int,
    batch_size: int,
    seed: int,
    block_steps: int = 1024,
) -> tuple[Path, Path]:
    """Freeze the same continuous randint stream using block RNG checkpoints."""
    for name, value in (
        ("dataset_size", dataset_size),
        ("steps", steps),
        ("batch_size", batch_size),
        ("block_steps", block_steps),
    ):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if block_steps * batch_size > 4_194_304:
        raise ValueError("sample-order block exceeds 32 MiB")
    path = Path(path)
    states_path = path.with_name(path.stem + "-rng.npy")
    if path.exists() or states_path.exists():
        raise FileExistsError(path)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    state_width = generator.get_state().numel()
    blocks = (steps + block_steps - 1) // block_steps
    states = np.lib.format.open_memmap(
        states_path, mode="w+", dtype=np.uint8, shape=(blocks, state_width)
    )
    for block, start in enumerate(range(0, steps, block_steps)):
        states[block] = generator.get_state().numpy()
        # Exactly the same call shape/order as the previous matrix writer.
        torch.randint(
            dataset_size,
            (min(block_steps, steps - start), batch_size),
            generator=generator,
        )
    states.flush()
    del states
    descriptor = {
        "format": _FORMAT,
        "dataset_size": dataset_size,
        "steps": steps,
        "batch_size": batch_size,
        "seed": seed,
        "block_steps": block_steps,
        "torch_version": str(torch.__version__),
        "states_file": states_path.name,
        "states_sha256": _digest(states_path),
    }
    with path.open("x") as stream:
        json.dump(descriptor, stream, sort_keys=True, indent=2)
    return path, states_path


class _CheckpointedOrder:
    """Read-only random access with at most one materialized index block."""

    def __init__(self, path: Path) -> None:
        path = Path(path).resolve()
        spec = json.loads(path.read_text())
        expected_keys = {
            "format",
            "dataset_size",
            "steps",
            "batch_size",
            "seed",
            "block_steps",
            "torch_version",
            "states_file",
            "states_sha256",
        }
        if set(spec) != expected_keys or spec["format"] != _FORMAT:
            raise ValueError("Unsupported checkpointed sample order")
        if spec["torch_version"] != str(torch.__version__):
            raise ValueError("Sample-order Torch version mismatch")
        for key in ("dataset_size", "steps", "batch_size", "block_steps"):
            if type(spec[key]) is not int or spec[key] < 1:
                raise ValueError("Invalid sample-order dimensions")
        if type(spec["seed"]) is not int or spec["seed"] < 0:
            raise ValueError("Invalid sample-order seed")
        if spec["block_steps"] * spec["batch_size"] > 4_194_304:
            raise ValueError("sample-order block exceeds 32 MiB")
        states_path = (path.parent / spec["states_file"]).resolve()
        if states_path.parent != path.parent or not states_path.is_file():
            raise ValueError("Sample-order RNG file must be beside its descriptor")
        if _digest(states_path) != spec["states_sha256"]:
            raise ValueError("Sample-order RNG checksum mismatch")
        self.states = np.load(states_path, mmap_mode="r", allow_pickle=False)
        generator = torch.Generator(device="cpu").manual_seed(spec["seed"])
        blocks = (spec["steps"] + spec["block_steps"] - 1) // spec["block_steps"]
        if (
            self.states.dtype != np.uint8
            or self.states.shape != (blocks, generator.get_state().numel())
            or not np.array_equal(self.states[0], generator.get_state().numpy())
        ):
            raise ValueError("Invalid sample-order RNG checkpoints")
        self.spec = spec
        self.shape = (spec["steps"], spec["batch_size"])
        self.dataset_size = spec["dataset_size"]
        self._block_id: int | None = None
        self._block: np.ndarray | None = None

    def __getitem__(self, index: int) -> np.ndarray:
        if not 0 <= index < self.shape[0]:
            raise IndexError(index)
        block_id, offset = divmod(index, self.spec["block_steps"])
        if self._block_id != block_id:
            generator = torch.Generator(device="cpu")
            generator.set_state(torch.from_numpy(self.states[block_id].copy()))
            start = block_id * self.spec["block_steps"]
            count = min(self.spec["block_steps"], self.shape[0] - start)
            self._block = torch.randint(
                self.dataset_size, (count, self.shape[1]), generator=generator
            ).numpy()
            self._block_id = block_id
        # Returned rows must survive later seeks and may be changed by callers.
        return self._block[offset].copy()


class _OfficialShuffleOrder:
    """Replay captured source passes with one bounded, hash-checked permutation."""

    def __init__(self, path: Path) -> None:
        path = Path(path).resolve()
        spec = json.loads(path.read_text())
        keys = {
            "format",
            "dataset_size",
            "steps",
            "batch_size",
            "epoch_steps",
            "drop_last",
            "discard_epoch_tail",
            "permutation_algorithm",
            "seeds_file",
            "seeds_sha256",
            "permutation_hashes_file",
            "permutation_hashes_sha256",
            "provenance",
        }
        if set(spec) != keys or spec["format"] != _OFFICIAL_FORMAT:
            raise ValueError("Unsupported official shuffle order")
        if (
            spec["drop_last"] is not True
            or spec["discard_epoch_tail"] is not True
            or spec["permutation_algorithm"] != "torch.randperm-cpu-int64-little-endian"
            or not isinstance(spec["provenance"], dict)
            or not spec["provenance"]
            or sys.byteorder != "little"
        ):
            raise ValueError("Invalid official shuffle contract")
        batches, per_epoch, passes = _official_dimensions(
            spec["dataset_size"], spec["steps"], spec["batch_size"], spec["epoch_steps"]
        )
        arrays = {}
        for key, hash_key in (
            ("seeds_file", "seeds_sha256"),
            ("permutation_hashes_file", "permutation_hashes_sha256"),
        ):
            if not isinstance(spec[key], str):
                raise ValueError("Official shuffle payload must be a sibling file")
            payload = (path.parent / spec[key]).resolve()
            if payload.parent != path.parent or not payload.is_file():
                raise ValueError("Official shuffle payload must be a sibling file")
            if _digest(payload) != spec[hash_key]:
                raise ValueError("Official shuffle payload checksum mismatch")
            arrays[key] = np.load(payload, mmap_mode="r", allow_pickle=False)
        seeds, hashes = arrays["seeds_file"], arrays["permutation_hashes_file"]
        if seeds.dtype != np.int64 or seeds.shape != (passes,) or np.any(seeds < 0):
            raise ValueError("Invalid official shuffle pass seeds")
        if hashes.dtype != np.dtype("S64") or hashes.shape != (passes,):
            raise ValueError("Invalid official shuffle permutation hashes")
        if any(
            len(value) != 64 or any(char not in "0123456789abcdef" for char in value)
            for value in (bytes(item).decode("ascii") for item in hashes)
        ):
            raise ValueError("Invalid official shuffle permutation hashes")
        self.spec, self.seeds, self.permutation_hashes = spec, seeds, hashes
        self.shape = (spec["steps"], spec["batch_size"])
        self.dataset_size = spec["dataset_size"]
        self._batches_per_pass, self._passes_per_epoch = batches, per_epoch
        self._pass_id: int | None = None
        self._permutation: np.ndarray | None = None

    def __getitem__(self, index: int) -> np.ndarray:
        if not 0 <= index < self.shape[0]:
            raise IndexError(index)
        epoch, within_epoch = divmod(index, self.spec["epoch_steps"])
        within_pass, row = divmod(within_epoch, self._batches_per_pass)
        pass_id = epoch * self._passes_per_epoch + within_pass
        if self._pass_id != pass_id:
            # Rows are owned copies; dropping the previous cache cannot
            # invalidate a previously returned minibatch or double live storage.
            self._permutation, self._pass_id = None, None
            generator = torch.Generator(device="cpu").manual_seed(
                int(self.seeds[pass_id])
            )
            permutation = torch.randperm(
                self.dataset_size, generator=generator, dtype=torch.int64
            ).numpy()
            expected = bytes(self.permutation_hashes[pass_id]).decode("ascii")
            if _permutation_sha256(permutation) != expected:
                raise ValueError(
                    "Official shuffle permutation checksum mismatch; Torch RNG algorithm differs"
                )
            self._permutation, self._pass_id = permutation, pass_id
        start = row * self.shape[1]
        return self._permutation[start : start + self.shape[1]].copy()


def _open_order(
    path: str | Path,
) -> np.ndarray | _CheckpointedOrder | _OfficialShuffleOrder:
    """Open a new RNG-checkpoint stream or a backward-compatible index matrix."""
    path = Path(path)
    if path.suffix == ".json":
        if json.loads(path.read_text()).get("format") == _OFFICIAL_FORMAT:
            return _OfficialShuffleOrder(path)
        return _CheckpointedOrder(path)
    order = np.load(path, mmap_mode="r", allow_pickle=False)
    if order.dtype.kind not in "iu":
        raise ValueError("Memory-mapped sample order requires integer indices")
    return order
