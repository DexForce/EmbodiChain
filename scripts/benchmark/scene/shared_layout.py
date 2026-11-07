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

"""Measure shared layout preparation, v4 persistence, consumption and native application.

Run: python -m scripts.benchmark.scene.shared_layout --workspace <scene-output>
--policy <v4-policy.json>. Native application uses four Default environments,
real source assets and no renderer or physics stepping. GPU timing includes
transfers and the canonical CPU certificate; it is not kernel-only latency.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from datetime import datetime
import gc
import json
from pathlib import Path
import platform
import statistics
import tempfile
import time
from typing import Any

import psutil
import torch

from embodichain.gen_sim.scene_engine.randomization.adapter import (
    scene_layout_input_from_export,
)
from embodichain.lab.sim.scene.expansion import (
    prepare_scene_randomization,
    generate_scene_variants,
    save_scene_variants,
    load_scene_variants,
    apply_scene_variants,
)


def _timed(operation: Callable[[], Any], device: str) -> tuple[Any, float]:
    """Return completed-call wall milliseconds with explicit CUDA synchronization."""
    synchronize = lambda: (
        torch.cuda.synchronize(device) if device.startswith("cuda") else None
    )
    synchronize()
    start = time.perf_counter()
    value = operation()
    synchronize()
    return value, (time.perf_counter() - start) * 1000


def _generation_memory(operation: Callable[[], Any], device: str) -> dict:
    """Measure a separate warmed call; retain one numeric result until sampled."""
    gc.collect()
    cuda = device.startswith("cuda")
    if cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    process = psutil.Process()
    rss = process.memory_info().rss
    baseline = torch.cuda.memory_allocated(device) if cuda else 0
    value = operation()
    if cuda:
        torch.cuda.synchronize(device)
    result = {
        "cpu_delta_mb": (process.memory_info().rss - rss) / 1024**2,
        "gpu_delta_mb": (
            (torch.cuda.memory_allocated(device) - baseline) / 1024**2 if cuda else None
        ),
        "peak_gpu_mb": (
            torch.cuda.max_memory_allocated(device) / 1024**2 if cuda else None
        ),
    }
    del value
    return result


def _consume(original: Any, saved: Any) -> None:
    """Stream every typed variant and compare complete poses and identities."""
    for index in range(len(original)):
        if original[index] != saved[index]:
            raise ValueError("Roundtrip changed a complete SceneVariant.")


def _apply_all(sim: Any, variants: list) -> None:
    """Publish all variants in groups of four, including a partial final group."""
    permitted = tuple(change.entity_uid for change in variants[0].pose_changes)
    for start in range(0, len(variants), 4):
        batch = variants[start : start + 4]
        apply_scene_variants(
            sim, batch, env_ids=list(range(len(batch))), movable_entity_ids=permitted
        )


def _report(output: Path, rows: list[dict], metadata: dict) -> Path:
    """Write one Markdown report with exactly three tables and a JSON companion."""
    output.mkdir(parents=True, exist_ok=True)
    stem = output / ("shared_layout_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    stem.with_suffix(".json").write_text(
        json.dumps({"metadata": metadata, "runs": rows}, indent=2)
    )
    lines = [
        "# Shared scene-layout benchmark",
        "",
        json.dumps(metadata),
        "",
        "## Time & Memory",
        "",
        "| algorithm | layouts | stage | repeats | cost_time_ms (median) | cpu_delta_mb | gpu_delta_mb | peak_gpu_mb |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    keys = list(dict.fromkeys((r["device"], r["count"]) for r in rows))
    for device, count in keys:
        group = [r for r in rows if (r["device"], r["count"]) == (device, count)]
        for stage in group[0]["stages"]:
            memory = group[0]["memory"] if stage == "generate" else {}
            cells = [
                "N/A" if memory.get(key) is None else f"{memory[key]:.2f}"
                for key in ("cpu_delta_mb", "gpu_delta_mb", "peak_gpu_mb")
            ]
            lines.append(
                f"| {device}/packed | {count} | {stage} | {len(group)} | {statistics.median(r['stages'][stage] for r in group):.3f} | "
                + " | ".join(cells)
                + " |"
            )
    lines += [
        "",
        "## Success & Other Metrics",
        "",
        "| algorithm | layouts | success_rate | accepted / attempts | full warm pipeline ms | identity/pose equality |",
        "| --- | ---: | ---: | --- | ---: | --- |",
    ]
    for device, count in keys:
        group = [r for r in rows if (r["device"], r["count"]) == (device, count)]
        good = sum(r["generated"] for r in group)
        lines.append(
            f"| {device}/packed | {count} | {good / (count * len(group)):.2%} | {good}/{sum(r['attempts'] for r in group)} | {statistics.median(sum(r['stages'].values()) for r in group):.3f} | complete |"
        )
    lines += [
        "",
        "## Leaderboard",
        "",
        "| rank | algorithm | overall_success_rate | full pipeline layouts/s |",
        "| ---: | --- | ---: | ---: |",
    ]
    ranking = []
    for device in dict.fromkeys(r["device"] for r in rows):
        group = [r for r in rows if r["device"] == device]
        good, requested = sum(r["generated"] for r in group), sum(
            r["count"] for r in group
        )
        ranking.append(
            (
                good / requested,
                good * 1000 / sum(sum(r["stages"].values()) for r in group),
                device,
            )
        )
    for index, (rate, throughput, device) in enumerate(
        sorted(ranking, reverse=True), 1
    ):
        lines.append(f"| {index} | {device}/packed | {rate:.2%} | {throughput:.2f} |")
    lines += [
        "",
        "Warm generation excludes imports, mesh loading, support extraction, rasterization and simulator construction. Preparation costs are recorded separately in metadata. Each size/backend is warmed once; all sizes use the configured repeat count. Latency samples exclude the separate generation memory pass.",
        "",
        "Consumption constructs and compares all generated and loaded typed variants. Application uses preconstructed variants and four native environments, includes all preflight checks and pose publication, and advances no physics time. Pipeline totals include both stages once.",
        "",
        "CPU memory is retained process RSS delta; CUDA memory covers the Torch allocator only. Other stages and native allocations are unmeasured (N/A). Cache sizes and prepared geometry are persistent outside the sampled delta. Geometric success does not establish task success.",
        "",
    ]
    path = stem.with_suffix(".md")
    path.write_text("\n".join(lines))
    return path


def main() -> None:
    """Benchmark one export through production shared APIs on CPU and CUDA."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[20, 500, 5000])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/benchmarks"))
    args = parser.parse_args()
    if min(*args.sizes, args.repeats, args.threads) < 1:
        parser.error("sizes, repeats and threads must be positive")
    torch.set_num_threads(args.threads)
    source, extraction_ms = _timed(
        lambda: scene_layout_input_from_export(args.workspace / "scene_export"), "cpu"
    )
    policy = json.loads(args.policy.read_text())
    devices = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])
    if len(devices) == 1:
        print("CUDA unavailable: GPU measurements skipped.", flush=True)
    from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
    from embodichain.gen_sim.scene_engine.pipeline.utils.scene_usd import (
        load_scene_export_into_sim,
    )

    cfg = SimulationManagerCfg(
        num_envs=4, device="cpu", headless=True, startup_summary="off"
    )
    cfg.render_cfg.renderer = "no-render"
    sim = SimulationManager(cfg)
    rows, preparation = [], {}
    try:
        load_scene_export_into_sim(
            sim=sim, output_root=args.workspace, force_static_rigids=True
        )
        sim.prepare()
        with tempfile.TemporaryDirectory(prefix="shared-layout-bench-") as temporary:
            root = Path(temporary)
            for device in devices:
                cache = root / ("cache_" + device.replace(":", "_"))
                selected_policy = {**policy, "device": device}
                prepared, cold = _timed(
                    lambda: prepare_scene_randomization(
                        source, selected_policy, cache_dir=cache
                    ),
                    "cpu",
                )
                prepared, warm = _timed(
                    lambda: prepare_scene_randomization(
                        source, selected_policy, cache_dir=cache
                    ),
                    "cpu",
                )
                preparation[device] = {
                    "cold_ms": cold,
                    "cached_ms": warm,
                    "cache_bytes": sum(p.stat().st_size for p in cache.iterdir()),
                }
                for count in args.sizes:
                    operation = lambda: generate_scene_variants(
                        prepared,
                        candidate_count=count,
                        max_attempts=max(200000, count * 500),
                        seed=42,
                    )
                    operation()
                    memory = _generation_memory(operation, device)
                    for repeat in range(args.repeats):
                        result, generation_ms = _timed(
                            lambda: generate_scene_variants(
                                prepared,
                                candidate_count=count,
                                max_attempts=max(200000, count * 500),
                                seed=42 + repeat,
                            ),
                            device,
                        )
                        run = root / f"run_{device.replace(':', '_')}_{count}_{repeat}"
                        _, save_ms = _timed(
                            lambda: save_scene_variants(result, run), device
                        )
                        loaded, reload_ms = _timed(
                            lambda: load_scene_variants(run), device
                        )
                        _, consume_ms = _timed(
                            lambda: _consume(result.variants, loaded.variants), device
                        )
                        typed = list(loaded.variants)
                        _, apply_ms = _timed(lambda: _apply_all(sim, typed), device)
                        if result.status != "completed":
                            raise RuntimeError("Whole-candidate budget exhausted.")
                        row = {
                            "device": device,
                            "count": count,
                            "seed": 42 + repeat,
                            "generated": len(typed),
                            "attempts": result.attempts,
                            "stages": {
                                "generate": generation_ms,
                                "save": save_ms,
                                "reload": reload_ms,
                                "consume_all": consume_ms,
                                "apply_all": apply_ms,
                            },
                            "memory": memory,
                        }
                        rows.append(row)
                        print(json.dumps(row), flush=True)
                        del typed, loaded, result
    finally:
        sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()
    metadata = {
        "workspace": str(args.workspace.resolve()),
        "source_sha256": source.fingerprint,
        "dtype": "float64",
        "cpu": (
            next(
                (
                    line.split(":", 1)[1].strip()
                    for line in Path("/proc/cpuinfo").read_text().splitlines()
                    if line.startswith("model name")
                ),
                platform.processor(),
            )
            if Path("/proc/cpuinfo").exists()
            else platform.processor()
        ),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "geometry_extraction_ms": extraction_ms,
        "preparation": preparation,
        "policy": policy,
        "application_envs": 4,
        "application_backend": "default/no-render, static source rigids, preconstructed variants",
        "physics_steps": 0,
    }
    print(
        f"Markdown report saved: {_report(args.output_dir, rows, metadata).resolve()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
