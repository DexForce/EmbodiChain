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

"""Compare individual and grouped dynamics window I/O on real and stress data.

Run: python -m scripts.benchmark.learning.dynamics_input --dataset DATASET
     --stress-dir /tmp/nerd-input-stress --output /tmp/input-benchmark.json

CPU read, collation and encoding are timed together; index generation, source
verification and correctness checks are excluded. Stress data repeats an actual
trajectory only to exercise many-shard I/O; it is not a physics accuracy dataset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import statistics
import time
import tracemalloc
from typing import Any

__all__ = ["benchmark_inputs", "create_stress_dataset"]


def _assert_equal(left: Any, right: Any) -> None:
    import torch

    if isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_equal(left[key], right[key])
    elif isinstance(left, torch.Tensor):
        assert left.dtype == right.dtype and torch.equal(left, right)
    else:
        assert left == right


def _tensor_bytes(value: Any) -> int:
    import torch

    if isinstance(value, dict):
        return sum(_tensor_bytes(item) for item in value.values())
    return (
        value.numel() * value.element_size() if isinstance(value, torch.Tensor) else 0
    )


def create_stress_dataset(
    source: str | Path,
    output: str | Path,
    *,
    trajectories: int = 512,
    trajectories_per_shard: int = 4,
) -> Path:
    """Repeat one real trajectory into independent shards for bounded-I/O stress."""
    import h5py
    from embodichain.data_pipeline.dynamics.shards import (
        load_manifest,
        DynamicsShardWriter,
        dataset_fingerprint,
    )

    if min(trajectories, trajectories_per_shard) < 1:
        raise ValueError("Stress sizes must be positive")
    source = Path(source)
    output = Path(output)
    manifest = load_manifest(source, verify=True)
    first_shard = manifest["shards"][0]
    first = first_shard["trajectories"][0]
    with h5py.File(source / first_shard["file"], "r") as file:
        root = file[first["group"]]
        arrays = {
            kind: {name: value[()] for name, value in root[kind].items()}
            for kind in ("states", "transitions")
        }
    with DynamicsShardWriter(
        output,
        {
            **manifest["metadata"],
            "purpose": "repeated_real_trajectory_io_stress_only",
            "source_fingerprint": dataset_fingerprint(manifest),
        },
        trajectories_per_shard=trajectories_per_shard,
    ) as writer:
        for index in range(trajectories):
            writer.add_trajectory(
                f"stress-{index}",
                arrays["states"],
                arrays["transitions"],
                split="train",
            )
    return output


def benchmark_inputs(
    root: str | Path,
    *,
    batch_sizes: tuple[int, ...] = (1, 16, 512),
    history: int = 10,
    warmup: int = 2,
    repeats: int = 10,
    max_open_shards: int = 4,
    read_page_steps: int = 256,
    unique_trajectories: bool = False,
) -> dict[str, Any]:
    """Measure identical batches using two readers with identical cache limits."""
    import h5py
    import numpy as np
    import torch
    from torch.utils.data import default_collate
    from embodichain.compute.dynamics import NeRDStateCodec
    from embodichain.data_pipeline.dynamics.shards import DynamicsWindowDataset
    from embodichain.learning.dynamics.trainer import encode_batch

    if (
        min(*batch_sizes, history, warmup, repeats, max_open_shards, read_page_steps)
        < 1
    ):
        raise ValueError("Benchmark workload sizes must be positive")
    torch.set_num_threads(1)
    readers = {
        name: DynamicsWindowDataset(
            root,
            "train",
            history,
            max_open_shards=max_open_shards,
            read_page_steps=read_page_steps,
            verify=True,
        )
        for name in ("individual", "grouped")
    }
    codec = NeRDStateCodec(**readers["individual"].manifest["metadata"]["codec"])
    reference = readers["individual"]
    records = []
    try:
        for batch_size in batch_sizes:
            generator = torch.Generator().manual_seed(42)
            selections = []
            for _ in range(warmup + repeats):
                if unique_trajectories:
                    if batch_size > len(reference._rows):
                        raise ValueError(
                            "Stress needs at least one distinct trajectory per requested sample"
                        )
                    rows = torch.randperm(len(reference._rows), generator=generator)[
                        :batch_size
                    ].tolist()
                    indices = []
                    for row in rows:
                        start = reference._ends[row - 1] if row else 0
                        width = reference._ends[row] - start
                        indices.append(
                            start + int(torch.randint(width, (), generator=generator))
                        )
                else:
                    indices = torch.randint(
                        len(reference), (batch_size,), generator=generator
                    ).tolist()
                selections.append(indices)
            expected = default_collate([reference[index] for index in selections[0]])
            actual = readers["grouped"].get_batch(selections[0])
            _assert_equal(actual, expected)
            # Compare actual encoder results explicitly; no tolerance changes.
            encoded_old = encode_batch(expected, codec, "cpu")
            encoded_new = encode_batch(actual, codec, "cpu")
            for old, new in zip(encoded_old, encoded_new):
                _assert_equal(old, new)
            del actual, expected, encoded_old, encoded_new
            for name, dataset in readers.items():

                def load(indices: list[int]) -> dict[str, Any]:
                    return (
                        dataset.get_batch(indices)
                        if name == "grouped"
                        else default_collate([dataset[index] for index in indices])
                    )

                samples = []
                for iteration, indices in enumerate(selections):
                    start = time.perf_counter()
                    batch = load(indices)
                    encoded = encode_batch(batch, codec, "cpu")
                    elapsed = time.perf_counter() - start
                    if iteration >= warmup:
                        samples.append(elapsed)
                    del batch, encoded
                # Separate memory/I/O instrumentation from latency samples.
                dataset.close()
                original_handle = dataset._handle
                opens = 0

                def observed_handle(filename: str) -> Any:
                    nonlocal opens
                    if filename not in dataset._handles:
                        opens += 1
                    return original_handle(filename)

                dataset._handle = observed_handle
                tracemalloc.start()
                batch = load(selections[-1])
                peak = tracemalloc.get_traced_memory()[1]
                tracemalloc.stop()
                output_bytes = _tensor_bytes(batch)
                del batch
                dataset._handle = original_handle
                records.append(
                    {
                        "implementation": name,
                        "batch_size": batch_size,
                        "history": history,
                        "median_seconds": statistics.median(samples),
                        "min_seconds": min(samples),
                        "max_seconds": max(samples),
                        "samples_seconds": samples,
                        "python_numpy_peak_bytes_cold_handles": peak,
                        "returned_tensor_bytes": output_bytes,
                        "cold_batch_file_opens": opens,
                        "retained_file_handles": len(dataset._handles),
                        "exact_output_and_encoding_equal": True,
                    }
                )
        return {
            "dataset": str(Path(root).resolve()),
            "fingerprint": reference.fingerprint,
            "robot": codec.robot,
            "scope": "CPU read+collate+encode; excludes indices,verification,training",
            "unique_trajectories": unique_trajectories,
            "trajectory_count": len(reference._rows),
            "shard_count": len(reference.manifest["shards"]),
            "max_open_shards": max_open_shards,
            "read_page_steps": read_page_steps,
            "max_source_frames_per_field_read": read_page_steps + history,
            "cache_policy": "bounded file LRU only; no retained trajectory/data blocks",
            "memory_scope": "tracemalloc Python/NumPy allocations, excludes native HDF5 caches/Torch allocator",
            "dtype": "preserved source dtype; codec float32",
            "cpu_threads": 1,
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "h5py": h5py.__version__,
            "numpy": np.__version__,
            "warmup": warmup,
            "repeats": repeats,
            "runs": records,
        }
    finally:
        for dataset in readers.values():
            dataset.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, action="append", required=True)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 16, 512])
    parser.add_argument("--history", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--stress-dir", type=Path)
    parser.add_argument("--stress-trajectories", type=int, default=512)
    parser.add_argument("--stress-shard-size", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = [
        benchmark_inputs(
            path,
            batch_sizes=tuple(args.batch_sizes),
            history=args.history,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        for path in args.dataset
    ]
    if args.stress_dir:
        stress = create_stress_dataset(
            args.dataset[-1],
            args.stress_dir,
            trajectories=args.stress_trajectories,
            trajectories_per_shard=args.stress_shard_size,
        )
        results.append(
            benchmark_inputs(
                stress,
                batch_sizes=tuple(args.batch_sizes),
                history=args.history,
                warmup=args.warmup,
                repeats=args.repeats,
                unique_trajectories=True,
            )
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, allow_nan=False))
    for result in results:
        print(result["dataset"], "unique", result["unique_trajectories"])
        for record in result["runs"]:
            print(
                {
                    key: value
                    for key, value in record.items()
                    if key != "samples_seconds"
                }
            )


if __name__ == "__main__":
    main()
