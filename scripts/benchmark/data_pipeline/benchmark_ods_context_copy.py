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

"""Measure owned ODS CPU sampling copies without simulation or a learner.

Run: python scripts/benchmark/data_pipeline/benchmark_ods_context_copy.py \
    --implementation baseline=/tmp/embodichain-baseline \
    --implementation optimized=. --output-dir outputs/benchmarks/ods-copy

Each checkout is imported in a separate subprocess. Input construction,
correctness hashes, output release and a separate sampled-RSS pass are outside
latency samples. This isolates synchronous sampling, not training throughput.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import threading
import time
from typing import Any

__all__: list[str] = []


def _hash_tensor(value: Any) -> str:
    """Hash every byte without introducing a second full image copy."""
    return hashlib.sha256(memoryview(value.contiguous().numpy())).hexdigest()


def _digest(batch: Any) -> dict[str, str]:
    """Describe and hash all returned TensorDict leaves outside timing."""
    return {
        str(key): f"{tuple(value.shape)}:{value.dtype}:{_hash_tensor(value)}"
        for key, value in batch.items(include_nested=True, leaves_only=True)
    }


def _fixture(cameras: int) -> Any:
    """Construct seeded numeric and RGBA payloads with dirty tails and boundaries."""
    import torch
    from tensordict import TensorDict

    generator = torch.Generator(device="cpu").manual_seed(317)
    shape = (8, 600)
    frames = torch.arange(shape[1])[None].expand(shape)
    rows = torch.arange(shape[0])[:, None].expand(shape)
    lengths = 600 - torch.arange(shape[0]) * 7
    valid = frames < lengths[:, None]
    continuity = (frames >= 300).to(torch.int64)
    continuity[~valid] = -1
    obs = {"state": torch.randn(*shape, 59, generator=generator)}
    for camera in range(cameras):
        obs[f"rgba_{camera}"] = torch.randint(
            0, 256, (*shape, 256, 256, 4), dtype=torch.uint8, generator=generator
        )
    data = TensorDict(
        {
            "obs": TensorDict(obs, batch_size=shape),
            "actions": torch.randn(*shape, 7, generator=generator),
            "rewards": torch.randn(*shape, 1, generator=generator),
            "valid": valid,
            "continuity_id": continuity,
            "segment_id": (frames >= 150).to(torch.int64),
            "segment_accepted": valid.clone(),
            "task_index": torch.zeros(shape, dtype=torch.int64),
            "subtask_index": torch.ones(shape, dtype=torch.int64),
            "ods_run_id": torch.full(shape, 17, dtype=torch.int64),
            "ods_episode_id": rows.clone(),
            "ods_frame_index": frames.clone(),
            "ods_generation": torch.ones(shape, dtype=torch.int64),
            "ods_accepted_at_ns": torch.full(shape, 12345, dtype=torch.int64),
            "ods_published_at_monotonic_ns": torch.full(
                shape, 23456, dtype=torch.int64
            ),
        },
        batch_size=shape,
        device="cpu",
    )
    # The actual sampler owns these independent CPU shared-memory tensors.
    return data.share_memory_()


def _run_worker(args: argparse.Namespace) -> None:
    """Time one checkout and vision shape with real engine/source methods."""
    import psutil
    import torch

    from embodichain.data_pipeline.datasets import OnlineDataSource, SequenceSpecCfg
    from embodichain.data_pipeline.engine.data import (
        OnlineDataEngine,
        OnlineDataEngineCfg,
        OnlineDataEngineState,
    )

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    imported = Path(sys.modules[OnlineDataEngine.__module__].__file__).resolve()
    if not imported.is_relative_to(Path(args.root).resolve()):
        raise RuntimeError(f"Wrong checkout imported: {imported}")
    prepared = _fixture(args.cameras)

    class FixtureEngine(OnlineDataEngine):
        """Retain normal synchronization and sampling, substituting only allocation."""

        def _create_buffer(self) -> Any:
            """Supply the prefilled shared fixture instead of constructing a simulator."""
            return prepared

    engine = FixtureEngine(
        OnlineDataEngineCfg(
            gym_config={"num_envs": 1},
            buffer_size=8,
            max_episode_steps=600,
            state_dim=59,
            action_dim=7,
            include_provenance=True,
            refill_policy="manual",
        )
    )
    # Consumer identity bypasses owner-only simulation liveness, not copy locks.
    engine._owner_pid = -1
    engine.run_id = 17
    engine._set_state(OnlineDataEngineState.READY)
    engine._init_signal.set()
    with engine._lock_index.get_lock():
        engine._lock_index[:] = [-1, -1]
    source = OnlineDataSource(
        engine,
        schema={
            "action": {
                "names": [f"joint_{index}" for index in range(7)],
                "semantics": "expert_controller_command",
                "step_dt": 1 / 30,
            }
        },
    )
    spec = SequenceSpecCfg(2, 32)
    rng = torch.Generator(device="cpu")
    process = psutil.Process()
    rows: list[dict[str, Any]] = []
    for batch_size in (1, 32):
        for api in ("chunk", "context", "source"):

            def reset(seed: int) -> None:
                """Restore equivalent local/global sampling randomness outside timing."""
                rng.manual_seed(seed)
                torch.manual_seed(seed)

            def sample() -> Any:
                """Call one public synchronous production sampling method."""
                if api == "chunk":
                    return engine.sample_batch(batch_size, 33)
                if api == "context":
                    return engine.sample_batch_with_context(
                        batch_size, 2, 32, generator=rng
                    )
                return source.sample_context(batch_size, spec, generator=rng)

            for repeat in range(args.warmup):
                reset(1000 + repeat)
                result = sample()
                del result
            samples: list[float] = []
            for repeat in range(args.repeats):
                reset(2000 + repeat)
                start = time.perf_counter()
                result = sample()
                samples.append((time.perf_counter() - start) * 1000)
                del result

            # Separate warmed memory observation; sampling is approximate, not a
            # true allocator peak. Retain precisely one returned batch in all arms.
            gc.collect()
            reset(3000)
            before = process.memory_info().rss
            memory_samples = [before]
            finished = threading.Event()

            def poll_rss() -> None:
                """Observe this subprocess RSS every millisecond in the memory pass."""
                while not finished.wait(0.001):
                    memory_samples.append(process.memory_info().rss)

            observer = threading.Thread(target=poll_rss)
            observer.start()
            result = sample()
            after = process.memory_info().rss
            memory_samples.append(after)
            finished.set()
            observer.join()
            del result

            # Ensure context checks actually cover padding, including B=1.
            correctness_seed = 4000
            while True:
                reset(correctness_seed)
                result = sample()
                if api == "chunk" or not result["valid"].all():
                    break
                del result
                correctness_seed += 1
            digest = _digest(result)
            rng_hash = _hash_tensor(
                torch.get_rng_state() if api == "chunk" else rng.get_state()
            )
            global_preserved = api == "chunk" or torch.equal(
                torch.get_rng_state(),
                torch.Generator().manual_seed(correctness_seed).get_state(),
            )
            invalid = ~result["valid"]
            padding_ok = True
            for key, value in result.items(include_nested=True, leaves_only=True):
                if key in {"valid", "loss_mask"}:
                    continue
                identifier = isinstance(key, str) and (
                    key.startswith("ods_")
                    or key
                    in {
                        "frame_ids",
                        "generations",
                        "task_index",
                        "subtask_index",
                        "accepted_at_ns",
                        "published_at_monotonic_ns",
                    }
                )
                padding_ok &= bool((value[invalid] == (-1 if identifier else 0)).all())
            # Replacement and later sampling must leave every returned leaf owned.
            with engine._lock_index.get_lock():
                for value in prepared["obs"].values():
                    (
                        value.bitwise_xor_(255)
                        if value.dtype == torch.uint8
                        else value.neg_()
                    )
                prepared["actions"].neg_()
                prepared["ods_episode_id"].add_(8)
                prepared["ods_generation"].add_(1)
            reset(correctness_seed)
            replacement = sample()
            owned = _digest(result) == digest
            del replacement
            with engine._lock_index.get_lock():
                for value in prepared["obs"].values():
                    (
                        value.bitwise_xor_(255)
                        if value.dtype == torch.uint8
                        else value.neg_()
                    )
                prepared["actions"].neg_()
                prepared["ods_episode_id"].sub_(8)
                prepared["ods_generation"].sub_(1)
            rows.append(
                {
                    "api": api,
                    "batch_size": batch_size,
                    "cameras": args.cameras,
                    "samples_ms": samples,
                    "cost_time_ms": statistics.median(samples),
                    "min_ms": min(samples),
                    "max_ms": max(samples),
                    "cpu_delta_mb": (after - before) / 1024**2,
                    "sampled_peak_delta_mb": (max(memory_samples) - before) / 1024**2,
                    "rss_baseline_mb": before / 1024**2,
                    "output_bytes": sum(
                        v.numel() * v.element_size()
                        for _, v in result.items(include_nested=True, leaves_only=True)
                    ),
                    "correctness_seed": correctness_seed,
                    "padding_frames": int(invalid.sum()),
                    "padding_ok": padding_ok,
                    "owned_after_replacement": owned,
                    "global_rng_preserved": global_preserved,
                    "output_hashes": digest,
                    "rng_hash": rng_hash,
                }
            )
            del result
            print(
                f"{args.label}: cameras={args.cameras} B={batch_size} {api}: {rows[-1]['cost_time_ms']:.3f} ms",
                flush=True,
            )
    engine._set_state(OnlineDataEngineState.CREATED)
    engine._owner_pid = os.getpid()
    engine.stop()
    packet = {
        "label": args.label,
        "root": args.root,
        "commit": subprocess.check_output(
            ["git", "-C", args.root, "rev-parse", "HEAD"], text=True
        ).strip(),
        "production_source_sha256": {
            str(path.relative_to(Path(args.root))): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in (
                Path(args.root) / "embodichain/data_pipeline/engine/data.py",
                Path(args.root) / "embodichain/data_pipeline/datasets/sources.py",
                Path(args.root) / "embodichain/data_pipeline/datasets/_sampling.py",
            )
        },
        "hardware": {
            "platform": platform.platform(),
            "cpu": Path("/proc/cpuinfo")
            .read_text()
            .split("model name\t: ")[1]
            .splitlines()[0],
            "affinity": process.cpu_affinity(),
            "total_memory_bytes": psutil.virtual_memory().total,
        },
        "software": {
            "python": sys.version,
            "torch": torch.__version__,
            "tensordict": __import__("tensordict").__version__,
        },
        "method": {
            "warmup": args.warmup,
            "repeats": args.repeats,
            "torch_threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(),
            "device": "cpu",
            "cache_shape": [8, 600],
            "state_width": 59,
            "action_width": 7,
            "image_shape": [256, 256, 4],
            "image_dtype": "uint8",
            "history": 2,
            "future": 32,
            "persistent_cache_bytes": sum(
                value.numel() * value.element_size()
                for _, value in prepared.items(include_nested=True, leaves_only=True)
            ),
        },
        "rows": rows,
    }
    Path(args.worker_output).write_text(json.dumps(packet, indent=2) + "\n")


def _write_report(packets: list[dict[str, Any]], output: Path) -> None:
    """Write exactly three tables with correctness and scoped timing evidence."""
    reference = {
        (r["cameras"], r["batch_size"], r["api"]): r
        for p in packets
        if p["label"] == packets[0]["label"]
        for r in p["rows"]
    }
    lines = [
        "# ODS CPU sampling-copy benchmark",
        "",
        "Synchronous production methods; no simulation, LeRobot reader, learner or GPU. Returned output release, correctness and memory observation are excluded from latency. Repeated samples support median and observed range; no percentile-tail claim is made.",
        "",
        "## Time & Memory",
        "",
        "| workload | implementation | cost_time_ms | min–max ms | cpu_delta_mb | sampled_peak_delta_mb | gpu_delta_mb | peak_gpu_mb |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    quality: list[tuple[str, dict[str, Any], bool]] = []
    for packet in packets:
        for row in packet["rows"]:
            baseline = reference[(row["cameras"], row["batch_size"], row["api"])]
            equivalent = (
                row["output_hashes"] == baseline["output_hashes"]
                and row["rng_hash"] == baseline["rng_hash"]
            )
            quality.append((packet["label"], row, equivalent))
            lines.append(
                f"| camera={row['cameras']} B={row['batch_size']} {row['api']} | {packet['label']} | {row['cost_time_ms']:.3f} | {row['min_ms']:.3f}–{row['max_ms']:.3f} | {row['cpu_delta_mb']:.1f} | {row['sampled_peak_delta_mb']:.1f} | N/A | N/A |"
            )
    lines += [
        "",
        "## Success & Other Metrics",
        "",
        "| workload | implementation | success_rate | all-leaf and RNG equivalence | padding frames | padding correct | owned after replacement |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for label, row, equivalent in quality:
        passed = (
            equivalent
            and row["padding_ok"]
            and row["owned_after_replacement"]
            and row["global_rng_preserved"]
        )
        lines.append(
            f"| camera={row['cameras']} B={row['batch_size']} {row['api']} | {label} | {int(passed):.0%} | {equivalent} | {row['padding_frames']} | {row['padding_ok']} | {row['owned_after_replacement']} |"
        )
    ranking = []
    for label in dict.fromkeys(p["label"] for p in packets):
        checks = [
            equivalent
            and row["padding_ok"]
            and row["owned_after_replacement"]
            and row["global_rng_preserved"]
            for name, row, equivalent in quality
            if name == label
        ]
        ranking.append((sum(checks) / len(checks), label, len(checks)))
    lines += [
        "",
        "## Leaderboard",
        "",
        "| rank | algorithm | overall_success_rate | cases |",
        "| --- | --- | --- | --- |",
    ]
    for rank, (success, label, count) in enumerate(
        sorted(ranking, key=lambda entry: -entry[0]), 1
    ):
        lines.append(f"| {rank} | {label} | {success:.0%} | {count} |")
    lines += [
        "",
        "Memory is a separate warmed pass: CPU retained RSS delta and maximum observed increment from 1 ms process-RSS polling, not an allocator peak. Persistent cache bytes and software/hardware/affinity are recorded in the JSON. Shared-memory fixture construction and dirty-tail/continuity setup are excluded. No worker writes during timing, so producer contention and training overlap remain unmeasured. Implementations use the same script and seeds; checkout order alternates by camera count. First-implementation hashes are the comparison reference, not independent semantic proof.",
        "",
    ]
    output.write_text("\n".join(lines))


def run_all_benchmarks() -> None:
    """Run isolated checkout comparisons and emit raw JSON plus one Markdown report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--implementation", action="append", default=[], metavar="LABEL=PATH"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/benchmarks/ods-copy")
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--cameras", type=int, default=0)
    parser.add_argument("--root")
    parser.add_argument("--label")
    parser.add_argument("--worker-output")
    args = parser.parse_args()
    if args.warmup < 1 or args.repeats < 1:
        parser.error("warmup and repeats must be positive")
    if args.worker_output:
        _run_worker(args)
        return
    if not args.implementation:
        parser.error("at least one --implementation LABEL=PATH is required")
    implementations = [
        (label, str(Path(root).resolve()))
        for label, root in (item.split("=", 1) for item in args.implementation)
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    packets = []
    for cameras in (0, 1, 2):
        for label, root in (
            implementations if cameras % 2 == 0 else implementations[::-1]
        ):
            destination = args.output_dir / f"{label}-camera-{cameras}.json"
            environment = {
                **os.environ,
                "PYTHONPATH": root,
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
            }
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--root",
                    root,
                    "--label",
                    label,
                    "--cameras",
                    str(cameras),
                    "--warmup",
                    str(args.warmup),
                    "--repeats",
                    str(args.repeats),
                    "--worker-output",
                    str(destination.resolve()),
                ],
                env=environment,
                check=True,
            )
            packets.append(json.loads(destination.read_text()))
    # Stable baseline-first packet order, independent of alternating execution.
    packets.sort(
        key=lambda packet: (
            [label for label, _ in implementations].index(packet["label"]),
            packet["rows"][0]["cameras"],
        )
    )
    (args.output_dir / "raw.json").write_text(json.dumps(packets, indent=2) + "\n")
    report = args.output_dir / "report.md"
    _write_report(packets, report)
    print(f"Markdown report saved: {report.resolve()}")


if __name__ == "__main__":
    run_all_benchmarks()
