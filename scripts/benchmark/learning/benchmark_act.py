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

"""Measure bounded visual ACT sampling, optimizer updates and inference.

Example: python -m scripts.benchmark.learning.benchmark_act --report /tmp/act-train/report.json
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import statistics
import time
from pathlib import Path
from typing import Callable

import torch

from embodichain.data_pipeline.datasets import LeRobotDataSource, LeRobotImageReader
from embodichain.learning.sl import SupervisedLearner
from embodichain_models.policy.act import (
    ACTCfg,
    ACTObjective,
    ACTPolicy,
    ACTSequenceSource,
)


def _measure(
    operation: Callable[[], object],
    synchronize: Callable[[], None],
    *,
    warmup: int,
    repeats: int,
) -> dict[str, float]:
    for _ in range(warmup):
        operation()
        synchronize()
    times = []
    for _ in range(repeats):
        synchronize()
        start = time.perf_counter()
        operation()
        synchronize()
        times.append(1000 * (time.perf_counter() - start))
    ordered = sorted(times)
    return {
        "median_ms": statistics.median(times),
        "p90_ms": ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))],
        "repeats": repeats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    torch.set_num_threads(1)
    report = json.loads(args.report.read_text())
    root = Path(report["dataset"])
    start = time.perf_counter()
    numeric = LeRobotDataSource(
        root,
        schema=report["schema"],
        source_id="act-benchmark",
        repo_id=root.name,
        memory_limit_bytes=512 * 1024**2,
    )
    readers = {
        size: LeRobotImageReader(
            root, "observation.images.cam_high", repo_id=root.name, cache_bytes=size
        )
        for size in (0, 512 * 1024**2)
    }
    sources = {
        size: ACTSequenceSource(numeric, reader, 30) for size, reader in readers.items()
    }
    setup_seconds = time.perf_counter() - start
    output: dict[str, object] = {
        "hardware": (
            torch.cuda.get_device_name(args.device)
            if args.device.startswith("cuda")
            else platform.processor()
        ),
        "device": args.device,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "image": [3, 224, 224],
        "horizon": 30,
        "cpu_threads": torch.get_num_threads(),
        "setup_seconds_including_hashes": setup_seconds,
        "sampling": {},
        "inference": {},
    }
    for size, source in sources.items():
        for batch_size in (1, 32):
            generator = torch.Generator().manual_seed(12345)
            output["sampling"][f"cache_{size}_batch_{batch_size}"] = _measure(
                lambda: source.sample(batch_size, generator=generator),
                lambda: None,
                warmup=5,
                repeats=args.repeats,
            )
    model = ACTPolicy(ACTCfg(pretrained_backbone=False)).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    learner = SupervisedLearner(model, ACTObjective(), optimizer, max_grad_norm=1.0)
    sync = (
        (lambda: torch.cuda.synchronize(args.device))
        if args.device.startswith("cuda")
        else (lambda: None)
    )
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(args.device)
    for batch_size in (1, 32):
        batch = (
            sources[512 * 1024**2]
            .sample(batch_size, generator=torch.Generator().manual_seed(3))
            .to(args.device)
        )
        model.eval()

        def inference() -> torch.Tensor:
            with torch.inference_mode():
                return model.predict(batch.inputs["rgb"], batch.inputs["state"])

        output["inference"][str(batch_size)] = _measure(
            inference, sync, warmup=5, repeats=args.repeats
        )
        if batch_size == 32:
            output["update_batch_32"] = _measure(
                lambda: learner.update(batch), sync, warmup=3, repeats=args.repeats
            )
    output["peak_gpu_allocated_bytes"] = (
        torch.cuda.max_memory_allocated(args.device)
        if args.device.startswith("cuda")
        else None
    )
    output["peak_gpu_reserved_bytes"] = (
        torch.cuda.max_memory_reserved(args.device)
        if args.device.startswith("cuda")
        else None
    )
    output["peak_process_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    encoded = json.dumps(output, indent=2)
    print(encoded)
    if args.output:
        args.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
