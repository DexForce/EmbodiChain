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
"""Measure CPU shared-buffer quota selection, locking and copying.

Run: python -m scripts.benchmark.data_pipeline.benchmark_ods_sampling_budget \
    --output /tmp/ods-sampling-budget

Uses synthetic 51-float observations, seven-float actions and realistic valid
trajectory lengths. It invokes the real public context sampler without a live
simulation producer. Disabled/high quotas use ABBA order; near-exhaustion resets
fresh source identities outside timing. No GPU work or universal speed claim.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any

import numpy as np
import psutil
import torch
from tensordict import TensorDict

from embodichain.data_pipeline.engine.data import (
    OnlineDataEngine,
    OnlineDataEngineCfg,
    OnlineDataEngineState,
    _populate_provenance,
)

__all__ = ["main"]


class _SamplerFixture(OnlineDataEngine):
    """Build actual shared sampling buffers, with no simulation subprocess."""

    def _create_buffer(self) -> TensorDict:
        rows, steps = self.cfg.buffer_size, self.cfg.max_episode_steps
        valid = torch.arange(steps)[None] < (320 + torch.arange(rows) % 24)[:, None]
        payload = {
            "obs": torch.arange(rows * steps * 51, dtype=torch.float32).reshape(
                rows, steps, 51
            )
            / 1000,
            "actions": torch.zeros(rows, steps, 7),
            "valid": valid,
            "segment_accepted": valid.clone(),
            "continuity_id": torch.where(valid, 0, -1),
            "segment_id": torch.where(valid, 0, -1),
        }
        for key in (
            "task_index",
            "subtask_index",
            "ods_run_id",
            "ods_episode_id",
            "ods_generation",
            "ods_frame_index",
            "ods_accepted_at_ns",
            "ods_published_at_monotonic_ns",
        ):
            payload[key] = torch.full((rows, steps), -1, dtype=torch.int64)
        return TensorDict(
            payload, batch_size=[rows, steps], device="cpu"
        ).share_memory_()


def _publish(engine: _SamplerFixture, episode_start: int, generation: int) -> None:
    """Reset synthetic source identities outside the timed sampling operation."""
    with engine._lock_index.get_lock():
        _populate_provenance(
            engine.shared_buffer,
            run_id=engine.run_id,
            episode_start=episode_start,
            generation=generation,
            accepted_at_ns=time.time_ns(),
            published_monotonic_ns=time.monotonic_ns(),
        )
        if engine._sampling_budget is not None:
            engine._sampling_budget.publish(
                0, engine.buffer_size, episode_start, generation
            )


def _measure(
    capacity: int, batch_size: int, mode: str, iterations: int, warmup: int
) -> dict[str, Any]:
    """Measure sampler service only; exclude setup, quota reset and validation."""
    quota = (
        None
        if mode == "disabled"
        else (
            math.ceil(batch_size / capacity)
            if mode == "near_exhaustion"
            else 1_000_000_000
        )
    )
    engine = _SamplerFixture(
        OnlineDataEngineCfg(
            buffer_size=capacity,
            max_episode_steps=600,
            state_dim=9,
            action_dim=7,
            gym_config={"num_envs": 1},
            refill_policy="manual",
            episode_sample_budget=quota,
            episode_sample_hard_limit=quota,
        )
    )
    # This fixture has only a consumer; no lifecycle action targets a fake worker.
    engine._owner_pid = -1
    engine._set_state(OnlineDataEngineState.READY)
    engine._lock_index[:] = [-1, -1]
    if mode != "near_exhaustion":
        _publish(engine, 0, 1)
    process = psutil.Process()
    before = process.memory_info().rss
    pss_samples = []
    timings = []
    accepted = 0
    torch.manual_seed(17001)
    for iteration in range(warmup + iterations):
        if mode == "near_exhaustion":
            _publish(engine, iteration * capacity, iteration + 1)
        started = time.perf_counter_ns()
        batch = engine.sample_batch_with_context(batch_size, 2, 64)
        duration_ms = (time.perf_counter_ns() - started) / 1e6
        # Independent output/ledger checks are not included in service timing.
        assert batch.batch_size == torch.Size([batch_size, 65])
        assert batch["valid"][:, 1].all()
        assert not (batch["loss_mask"] & ~batch["valid"]).any()
        if iteration >= warmup:
            timings.append(duration_ms)
            accepted += 1
            if iteration % 10 == 0:
                pss_samples.append(getattr(process.memory_full_info(), "pss", None))
        if mode == "near_exhaustion":
            snapshot = engine.get_sampling_budget_snapshot()
            assert sum(row["delivered"] for row in snapshot["rows"]) == batch_size
            assert all(row["delivered"] <= row["assigned"] for row in snapshot["rows"])
    snapshot = engine.get_sampling_budget_snapshot()
    if mode != "disabled":
        assert snapshot["total_delivered_chunks"] == (warmup + iterations) * batch_size
    after = process.memory_info().rss
    result = {
        "capacity": capacity,
        "batch_size": batch_size,
        "mode": mode,
        "p50_ms": float(np.percentile(timings, 50)),
        "p95_ms": float(np.percentile(timings, 95)),
        "mean_ms": float(np.mean(timings)),
        "chunks_per_second": batch_size / (np.mean(timings) / 1000),
        "cpu_delta_mb": (after - before) / 1024**2,
        "host_pss_sampled_peak_bytes": max(
            (value for value in pss_samples if value is not None), default=None
        ),
        "gpu_delta_mb": None,
        "peak_gpu_mb": None,
        "gpu_memory_scope": "not used; CPU-only workload",
        "validation_success_rate": accepted / iterations,
        "quota_state_payload_bytes": 0 if quota is None else capacity * 4 * 8 + 2 * 8,
        "delivered_chunks": snapshot["total_delivered_chunks"],
        "service_times_ms": timings,
    }
    del batch, engine
    gc.collect()
    return result


def _report(path: Path, result: dict[str, Any]) -> None:
    """Write the three component-benchmark tables with explicit timing scope."""
    rows = result["rows"]
    lines = [
        "# ODS per-source quota sampler benchmark",
        "",
        result["workload"],
        "",
        "Setup, source replacement, validation, memory sampling and logging are excluded from service time. "
        "The actual copy lock and TensorDict clone/padding are included. No real producer contention or GPU training is measured.",
        "",
        "## Time & Memory",
        "",
        "| B | Mode | Repeat | cost_time_ms (p50) | p95_ms | cpu_delta_mb | sampled PSS peak bytes | gpu_delta_mb | peak_gpu_mb |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['batch_size']} | {row['mode']} | {row['repeat']} | {row['p50_ms']:.4f} | {row['p95_ms']:.4f} | {row['cpu_delta_mb']:.3f} | {row['host_pss_sampled_peak_bytes']} | not used | not used |"
        )
    lines += [
        "",
        "## Success & Other Metrics",
        "",
        "| B | Mode | Repeat | Validation success | chunks/s | Quota state bytes |",
        "| ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['batch_size']} | {row['mode']} | {row['repeat']} | {row['validation_success_rate']} | {row['chunks_per_second']:.1f} | {row['quota_state_payload_bytes']} |"
        )
    lines += [
        "",
        "## Leaderboard",
        "",
        "All validated modes are listed; compare service timings only at the same batch size. "
        "Near-exhaustion is a distinct binding-cap stress case, not identical work to an unrestricted sampler.",
        "",
        "| Validation rank | B | Mode | Mean validation success |",
        "| ---: | ---: | --- | ---: |",
    ]
    groups = {(row["batch_size"], row["mode"]) for row in rows}
    averages = [
        (
            batch,
            mode,
            float(
                np.mean(
                    [
                        r["validation_success_rate"]
                        for r in rows
                        if r["batch_size"] == batch and r["mode"] == mode
                    ]
                )
            ),
        )
        for batch, mode in sorted(groups)
    ]
    scores = sorted({score for _, _, score in averages}, reverse=True)
    for batch, mode, score in sorted(
        averages, key=lambda row: (-row[2], row[0], row[1])
    ):
        lines.append(f"| {scores.index(score) + 1} | {batch} | {mode} | {score} |")
    lines += [
        "",
        "PSS is sampled after returned batches and is not an exact peak of internal temporaries. RSS deltas include allocator caching. "
        "Per-source state is O(C); temporary selection storage is O(B+C), independent of the numerical quota magnitude.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    """Run the explicit CPU workload matrix and save raw samples plus report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capacity", type=int, default=32)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[64, 256, 1024])
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if (
        min(args.capacity, args.iterations, args.repeats, *args.batch_sizes) < 1
        or args.warmup < 0
    ):
        parser.error("Workload counts must be positive; warmup must be nonnegative")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    rows = []
    for batch_size in args.batch_sizes:
        for repeat in range(args.repeats):
            for mode in (
                "disabled",
                "high_quota",
                "high_quota",
                "disabled",
                "near_exhaustion",
            ):
                row = _measure(
                    args.capacity, batch_size, mode, args.iterations, args.warmup
                )
                row["repeat"] = repeat
                rows.append(row)
                print(
                    json.dumps(
                        {
                            key: value
                            for key, value in row.items()
                            if key != "service_times_ms"
                        }
                    ),
                    flush=True,
                )
    source = (
        Path(__file__).resolve().parents[3] / "embodichain/data_pipeline/engine/data.py"
    )
    result = {
        "workload": "CPU shared buffer C32 by default, T600, valid lengths320–343, B64/256/1024, H2/F64, 51-float observations and 7-float actions plus provenance.",
        "argv": sys.argv,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "order_per_repeat": [
            "disabled",
            "high_quota",
            "high_quota",
            "disabled",
            "near_exhaustion",
        ],
        "real_producer": False,
        "gpu_used": False,
        "rows": rows,
    }
    (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    _report(args.output / "REPORT.md", result)
    print(f"Report: {args.output / 'REPORT.md'}")


if __name__ == "__main__":
    main()
