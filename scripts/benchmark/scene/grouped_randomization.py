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

"""Compare complete grouped-generation, durable-save and validated-reload costs.

Run: python -m scripts.benchmark.scene.grouped_randomization --workspace <scene>
Small workloads repeat across seeds; workloads >=5,000 run once. Exactly one
Markdown report records costs, geometric completion and all tested algorithms.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from datetime import datetime
import gc
import json
from pathlib import Path
import statistics
import shutil
import tempfile
import time
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import psutil
import torch

from embodichain.gen_sim.scene_engine.randomization.baseline import (
    prepare_scene_baseline,
)
from embodichain.gen_sim.scene_engine.randomization.cfg import (
    SceneRandomizationCfg,
    load_randomization_config,
)
from embodichain.gen_sim.scene_engine.randomization.generation import randomize_scene
from embodichain.gen_sim.scene_engine.randomization.loading import load_layouts
from embodichain.gen_sim.scene_engine.randomization.persistence import save_layouts


def _measure(function: Callable[[], Any]) -> tuple[Any, dict[str, float]]:
    """Time one completed operation with process RSS and PyTorch VRAM deltas."""
    gc.collect()
    cuda = torch.cuda.is_initialized()
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    process = psutil.Process()
    rss = process.memory_info().rss
    gpu = torch.cuda.memory_allocated() if cuda else 0
    start = time.perf_counter()
    result = function()
    if cuda:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return result, {
        "cost_time_ms": elapsed * 1000,
        "cpu_delta_mb": (process.memory_info().rss - rss) / 1024**2,
        "gpu_delta_mb": (
            (torch.cuda.memory_allocated() - gpu) / 1024**2 if cuda else 0.0
        ),
        "peak_gpu_mb": torch.cuda.max_memory_allocated() / 1024**2 if cuda else 0.0,
    }


def _trial(
    cfg: SceneRandomizationCfg,
    storage: str,
    destination: Path,
    *,
    consume_all: bool = False,
) -> dict:
    """Measure production APIs, then consume representative complete candidates."""
    result, generate = _measure(lambda: randomize_scene(cfg))
    run, save = _measure(
        lambda: save_layouts(result, output_root=destination, layout_format=storage)
    )
    loaded, reload = _measure(lambda: load_layouts(run / "layouts"))
    if result.status != "completed" or len(loaded.variants) != cfg.candidate_count:
        raise RuntimeError(f"Incomplete benchmark run: {result.status}")
    start = time.perf_counter()
    indices = sorted({0, cfg.candidate_count // 2, cfg.candidate_count - 1})
    for index in indices:
        candidate = result.candidates[index]
        variant = loaded.variants[index]
        if candidate.variant != variant:
            raise RuntimeError("Saved/reloaded candidate identity or poses changed.")
        entities = {entity.uid: entity for entity in candidate.fragment.entities}
        for change in variant.pose_changes:
            np.testing.assert_allclose(
                entities[change.entity_uid].init_local_pose,
                change.arena_pose,
                rtol=0,
                atol=1e-12,
            )
    materialize_ms = (time.perf_counter() - start) * 1000
    consume_stats = None
    if consume_all:

        def consume():
            for index in range(cfg.candidate_count):
                # Release each complete fragment immediately, as a streaming host does.
                candidate = result.candidates[index]
                variant = loaded.variants[index]
                if candidate.variant != variant:
                    raise RuntimeError("Consumed variant differs after roundtrip.")

        _, consume_stats = _measure(consume)
    return {
        "count": cfg.candidate_count,
        "seed": cfg.seed,
        "generated": len(result.candidates),
        "attempts": result.attempts,
        "status": result.status,
        "generator_version": result.algorithm_version,
        "roundtrip_verified": True,
        "stages": {
            "generate": generate,
            "save": save,
            "reload": reload,
            **({"consume_all": consume_stats} if consume_all else {}),
        },
        "end_to_end_ms": sum(
            stage["cost_time_ms"] for stage in (generate, save, reload)
        ),
        "full_pipeline_ms": (
            (
                sum(stage["cost_time_ms"] for stage in (generate, save, reload))
                + consume_stats["cost_time_ms"]
            )
            if consume_stats is not None
            else None
        ),
        "consume_sample_ms": materialize_ms,
    }


def _report(output: Path, rows: list[dict], metadata: dict) -> Path:
    """Write the required cost, correctness and complete leaderboard tables."""
    timestamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
    stem = output / f"grouped_randomization_{timestamp}"
    stem.with_suffix(".json").write_text(
        json.dumps({"metadata": metadata, "runs": rows}, indent=2) + "\n"
    )
    lines = [
        "# Grouped randomization end-to-end benchmark",
        "",
        json.dumps(metadata, ensure_ascii=False),
        "",
        "## Time & Memory",
        "",
        "| algorithm | layouts | stage | repeats | cost_time_ms (median) | cpu_delta_mb | gpu_delta_mb | peak_gpu_mb |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    keys = list(dict.fromkeys((row["algorithm"], row["count"]) for row in rows))
    for algorithm, count in keys:
        group = [
            row
            for row in rows
            if row["algorithm"] == algorithm and row["count"] == count
        ]
        for stage in group[0]["stages"]:
            values = {
                name: statistics.median(row["stages"][stage][name] for row in group)
                for name in (
                    "cost_time_ms",
                    "cpu_delta_mb",
                    "gpu_delta_mb",
                    "peak_gpu_mb",
                )
            }
            lines.append(
                f"| {algorithm} | {count} | {stage} | {len(group)} | {values['cost_time_ms']:.3f} | {values['cpu_delta_mb']:.2f} | {values['gpu_delta_mb']:.2f} | {values['peak_gpu_mb']:.2f} |"
            )
    lines += [
        "",
        "## Success & Other Metrics",
        "",
        "| algorithm | requested layouts | completed / requested | accepted / attempts | generation/save/reload ms (median) | full pipeline incl. consumption ms | all files/identities checked |",
        "| --- | ---: | --- | --- | ---: | ---: | --- |",
    ]
    for algorithm, count in keys:
        group = [
            row
            for row in rows
            if row["algorithm"] == algorithm and row["count"] == count
        ]
        completed, attempts = sum(row["generated"] for row in group), sum(
            row["attempts"] for row in group
        )
        median = statistics.median(row["end_to_end_ms"] for row in group)
        full = [
            row["full_pipeline_ms"]
            for row in group
            if row.get("full_pipeline_ms") is not None
        ]
        full_label = f"{statistics.median(full):.3f}" if full else "N/A"
        lines.append(
            f"| {algorithm} | {count} | {completed}/{count*len(group)} | {completed}/{attempts} | {median:.3f} | {full_label} | yes |"
        )
    rankings = []
    for algorithm in dict.fromkeys(row["algorithm"] for row in rows):
        group = [row for row in rows if row["algorithm"] == algorithm]
        generated, requested = sum(row["generated"] for row in group), sum(
            row["count"] for row in group
        )
        rankings.append(
            (
                generated / requested,
                generated
                / (
                    sum(
                        row.get("full_pipeline_ms") or row["end_to_end_ms"]
                        for row in group
                    )
                    / 1000
                ),
                algorithm,
            )
        )
    lines += [
        "",
        "## Leaderboard",
        "",
        "| rank | algorithm | overall request completion rate | end-to-end layouts/s |",
        "| ---: | --- | ---: | ---: |",
    ]
    for rank, (rate, throughput, algorithm) in enumerate(
        sorted(rankings, reverse=True), 1
    ):
        lines.append(f"| {rank} | {algorithm} | {rate:.2%} | {throughput:.2f} |")
    lines += [
        "",
        "Notes:",
        "- Timed operations include baseline-cache/source validation, candidate generation, fsynced persistence and full provenance/geometry/identity validation on reload. JSON end_to_end_ms retains the numeric pipeline cost; full_pipeline_ms additionally includes all paired generated-fragment/reloaded-variant consumption. Leaderboard uses the full cost when measured.",
        "- Python imports, asset copying and CUDA warmup are outside steady-state timings. No renderer or physical simulation is constructed. Cold/warm support cache preparation is measured once and reported in metadata; v3 consume_all measures complete streaming object consumption separately.",
        "- Batch results defer runtime configuration/typed-pose construction. Three representative candidates are fully materialized and compared after each trial; this cost is recorded in JSON, outside the numeric pipeline timing.",
        "- SLSQP repairs proposals; batch rejection discards invalid proposals. Acceptance fractions differ and are reported separately from request completion.",
        "- CPU memory is process RSS delta; GPU memory is PyTorch allocated VRAM, not native renderer memory. CUDA timing is synchronized.",
        "- Small cases repeat across seeds 42 onward; workloads >=5,000 run once. Inputs use the same asset baseline and configured group margin recorded in metadata.",
        "- These measurements establish geometric layouts and reproducible saved identities, not physical stability or task execution success.",
        "",
    ]
    path = stem.with_suffix(".md")
    path.write_text("\n".join(lines))
    return path


def main() -> None:
    """Benchmark original and batch pipelines against one existing scene export."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        help="Optional v3 intra-group policy; compares CPU/CUDA with identical scope.",
    )
    parser.add_argument("--sizes", type=int, nargs="+", default=[20, 500, 5000])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/benchmarks"))
    args = parser.parse_args()
    if min(*args.sizes, args.repeats) <= 0:
        parser.error("sizes and repeats must be positive")
    workspace = args.workspace.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cfg = SceneRandomizationCfg(
        scene_path=workspace / "scene_export",
        base_data_path=workspace / "scene_export/scene_randomization_base.yaml",
    )
    if args.config:
        cfg = load_randomization_config(args.config, output_root=workspace)
    cfg.validate()
    baseline = prepare_scene_baseline(cfg)
    algorithms = [
        ("slsqp/json", "slsqp", "cpu", "json"),
        ("batched/cpu/packed", "batched", "cpu", "packed"),
    ]
    support_preparation = None
    if cfg.intra_group and cfg.intra_group.targets:
        from embodichain.gen_sim.scene_engine.randomization._intra_geometry import (
            _prepare_geometry,
        )

        algorithms = [("hierarchical/cpu/packed", "batched", "cpu", "packed")]
        # An isolated copy measures a genuinely cold cache without deleting saved-run evidence.
        with tempfile.TemporaryDirectory(prefix="support-cache-cold-") as temporary:
            export = Path(temporary) / "scene_export"
            shutil.copytree(
                cfg.scene_path,
                export,
                ignore=shutil.ignore_patterns("scene_randomization_geometry"),
            )
            cold_cfg = cfg.replace(
                scene_path=export,
                base_data_path=export / "scene_randomization_base.yaml",
            )
            cold_baseline = prepare_scene_baseline(cold_cfg)
            _, cold = _measure(
                lambda: _prepare_geometry(cold_baseline, cfg.intra_group)
            )
        _prepare_geometry(baseline, cfg.intra_group)
        _, warm = _measure(lambda: _prepare_geometry(baseline, cfg.intra_group))
        support_preparation = {"cold": cold, "warm": warm}
    warmup_ms = None
    if torch.cuda.is_available():
        start = time.perf_counter()
        randomize_scene(
            cfg.replace(backend="batched", device="cuda:0"),
            candidate_count=1,
            max_attempts=200000 if cfg.intra_group else 100,
        )
        torch.cuda.synchronize()
        warmup_ms = (time.perf_counter() - start) * 1000
        algorithms.append(
            (
                (
                    "hierarchical/cuda/packed"
                    if cfg.intra_group
                    else "batched/cuda/packed"
                ),
                "batched",
                "cuda:0",
                "packed",
            )
        )
    else:
        print("CUDA benchmark skipped: CUDA is unavailable.", flush=True)
    rows = []
    with tempfile.TemporaryDirectory(prefix="grouped-randomization-") as temporary:
        destination = Path(temporary)
        for count in args.sizes:
            for algorithm, backend, device, storage in algorithms:
                for repeat in range(1 if count >= 5000 else args.repeats):
                    trial_cfg = cfg.replace(
                        backend=backend,
                        device=device,
                        candidate_count=count,
                        max_attempts=(
                            max(200000, count * 500)
                            if cfg.intra_group
                            else max(1000, count * 50)
                        ),
                        seed=42 + repeat,
                    )
                    row = _trial(
                        trial_cfg,
                        storage,
                        destination,
                        consume_all=bool(cfg.intra_group),
                    )
                    row["algorithm"] = algorithm
                    rows.append(row)
                    print(json.dumps(row), flush=True)
    cpu = next(
        line.split(":", 1)[1].strip()
        for line in Path("/proc/cpuinfo").read_text().splitlines()
        if line.startswith("model name")
    )
    metadata = {
        "workspace": str(workspace),
        "cpu": cpu,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "groups": len(baseline.data["groups"]),
        "margin_m": cfg.collision_margin_m,
        "generator_versions": sorted({row["generator_version"] for row in rows}),
        "cuda_warmup_ms": warmup_ms,
        "support_preparation": support_preparation,
        "consumption": (
            "all layouts streamed"
            if cfg.intra_group
            else "three representative layouts"
        ),
        "time": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
    }
    path = _report(args.output_dir, rows, metadata)
    print(f"Markdown report saved: {path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
