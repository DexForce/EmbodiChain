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

"""Render measured data-stream results without rerunning acquisition or decoding.

Run: python -m examples.ods.data_stream_demo.report OUTPUT_ROOT --animation
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median
from typing import Any

__all__ = ["write_report"]

_ARMS = {
    "lerobot": "Decoded LeRobot / K",
    "streaming": "ODS streaming / unlimited",
    "protected": "ODS protected / K",
    "lerobot_unbounded": "Decoded LeRobot / unlimited",
    "frozen": "ODS frozen / C / K",
    "lerobot_same_cache": "Decoded LeRobot / C / K",
}
_MAIN = ("lerobot", "streaming", "protected")
_VISIONS = ("state", "single", "dual")
_COLORS = ("#4676ac", "#dc8b36", "#23977c", "#9657a4", "#c36b72", "#667d3c")
_METRICS = (
    "first_batch_from_launch_seconds",
    "first_batch_from_reader_seconds",
    "delivered_chunks_per_second",
    "request_latency_p50_seconds",
    "request_latency_p95_seconds",
    "service_p50_seconds",
    "service_p95_seconds",
    "accepted_episodes_per_second",
    "data_wait_seconds",
    "throttle_seconds",
    "cache_bytes",
    "peak_pss_bytes",
    "peak_vram_bytes",
    "final_disk_bytes",
    "peak_temporary_disk_bytes",
    "logical_payload_bytes_per_second",
    "quality_psnr_db",
    "quality_mae",
    "cap_violations",
    "delivered_chunks",
    "window_delivered_chunks",
    "post_window_chunks",
    "unconsumed_prefetch_chunks",
)


def _number(value: Any) -> float | None:
    return (
        float(value)
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        else None
    )


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object in {path}")
    return value


def _events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as error:
            raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
        if not isinstance(row, dict):
            raise ValueError(f"Expected an object at {path}:{line_number}")
        rows.append(row)
    return rows


def _load(root: Path) -> list[dict[str, Any]]:
    paths = sorted(root.rglob("summary.json")) if root.is_dir() else [root]
    runs = []
    for path in paths:
        summary = _json(path)
        config = summary.get("config", {})
        if config.get("arm") not in _ARMS or config.get("vision") not in _VISIONS:
            raise ValueError(f"Unknown experiment arm/vision in {path}")
        if summary.get("status") not in {"complete", "failed", "unsupported"}:
            raise ValueError(f"Unknown run status in {path}")
        runs.append(
            {
                "path": str(path.resolve()),
                "summary": summary,
                "config": config,
                "batches": _events(path.parent / "batches.jsonl"),
                "lifecycle": _events(path.parent / "lifecycle.jsonl"),
            }
        )
    if not runs:
        raise ValueError(f"No run summaries found in {root}")
    return sorted(
        runs,
        key=lambda r: (
            r["config"]["arm"],
            r["config"]["vision"],
            str(r["config"].get("seed")),
        ),
    )


def _qualified(run: dict[str, Any]) -> bool:
    return (
        run["summary"]["status"] == "complete"
        and run["config"].get("suite") in {"report", "appendix"}
        and run["summary"].get("metrics", {}).get("qualification_passed") is True
    )


def _metric(run: dict[str, Any], key: str) -> float | None:
    return _number(run["summary"].get("metrics", {}).get(key))


def _group(runs: list[dict[str, Any]], arm: str, vision: str) -> list[dict[str, Any]]:
    return [
        r for r in runs if r["config"]["arm"] == arm and r["config"]["vision"] == vision
    ]


def _stats(runs: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    values = [_metric(r, metric) for r in runs if _qualified(r)]
    values = [v for v in values if v is not None]
    return {
        "n": len(values),
        "median": median(values) if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def _formatted(stats: dict[str, Any], scale: float = 1.0) -> str:
    if not stats["n"]:
        return "unknown (n=0)"
    values = [stats[k] / scale for k in ("median", "min", "max")]
    return f"{values[0]:.3g} [{values[1]:.3g}, {values[2]:.3g}]; n={stats['n']}"


def _cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(_cell(v) for v in row) + " |" for row in rows),
    ]


def _save(figure: Any, output: Path, name: str) -> None:
    for extension in ("png", "svg", "pdf"):
        figure.savefig(output / f"{name}.{extension}", dpi=170, bbox_inches="tight")


def _plot_points(
    axis: Any,
    runs: list[dict[str, Any]],
    metric: str,
    x: float,
    color: str,
    *,
    scale: float = 1.0,
) -> None:
    for i, run in enumerate(runs):
        value = _metric(run, metric)
        if value is not None:
            axis.scatter(
                x + (i - (len(runs) - 1) / 2) * 0.045,
                value / scale,
                s=23,
                facecolors=color if _qualified(run) else "none",
                edgecolors=color,
                linewidths=1,
                zorder=3,
            )
    stats = _stats(runs, metric)
    if stats["n"]:
        center = stats["median"] / scale
        axis.errorbar(
            x,
            center,
            yerr=[[center - stats["min"] / scale], [stats["max"] / scale - center]],
            fmt="_",
            color=color,
            markersize=14,
            capsize=3,
            zorder=4,
        )


def _timeline(runs: list[dict[str, Any]], plt: Any, output: Path) -> Any:
    figure, axes = plt.subplots(3, 1, figsize=(13, 11), squeeze=False)
    for vision, axis in zip(_VISIONS, axes[:, 0]):
        selected = [
            r
            for r in runs
            if r["config"]["vision"] == vision and r["config"]["arm"] in _MAIN
        ]
        for row, run in enumerate(selected):
            cache = run["config"].get("cache_episodes", 1)
            cache = cache if isinstance(cache, int) and cache > 0 else 1
            end_times = [
                _number(e.get("time_seconds"))
                for e in run["lifecycle"] + run["batches"]
            ]
            end = max((t for t in end_times if t is not None), default=0.0)
            first = _metric(run, "first_batch_from_launch_seconds")
            ready = _metric(run, "source_ready_seconds")
            offline = run["config"]["arm"].startswith("lerobot")
            cache_end = min(end, ready) if offline and ready is not None else end
            if offline and ready is not None:
                duration = _metric(run, "actual_measurement_seconds")
                files_end = max(end, ready + duration) if duration is not None else end
                axis.hlines(
                    row, ready, files_end, colors="#596874", linewidth=3, alpha=0.65
                )
            if ready is not None:
                axis.plot(ready, row - 0.46, marker="|", color="#777777", markersize=9)
            if first is not None:
                axis.plot(first, row - 0.46, marker="*", color="#333333", markersize=6)
            active = {}
            for event in sorted(
                run["lifecycle"], key=lambda e: _number(e.get("time_seconds")) or 0
            ):
                time_value, slot = _number(event.get("time_seconds")), event.get("slot")
                if time_value is None or not isinstance(slot, int):
                    continue
                if slot in active:
                    start, generation = active.pop(slot)
                    axis.broken_barh(
                        [(start, max(0, min(time_value, cache_end) - start))],
                        (row - 0.37 + 0.6 * slot / cache, 0.6 / cache),
                        facecolors=plt.get_cmap("tab10")(int(generation) % 10),
                        alpha=0.4,
                    )
                if event.get("kind") == "published":
                    active[slot] = (time_value, event.get("generation", 0))
            for slot, (start, generation) in active.items():
                axis.broken_barh(
                    [(start, max(0, cache_end - start))],
                    (row - 0.37 + 0.6 * slot / cache, 0.6 / cache),
                    facecolors=plt.get_cmap("tab10")(int(generation) % 10),
                    alpha=0.4,
                )
            for batch in run["batches"]:
                time_value = _number(batch.get("time_seconds"))
                if time_value is None:
                    continue
                for generation, count in batch.get("generations", {}).items():
                    if _number(count) is not None and count > 0:
                        axis.scatter(
                            time_value,
                            row + 0.3,
                            s=min(24, 2 + count),
                            color=plt.get_cmap("tab10")(int(generation) % 10),
                            marker="x" if batch.get("within_window") is False else "o",
                            linewidths=(
                                0.8 if batch.get("within_window") is False else 0
                            ),
                        )
            if not run["lifecycle"]:
                axis.text(
                    0.01,
                    row,
                    "no lifecycle records",
                    transform=axis.get_yaxis_transform(),
                    fontsize=7,
                    va="center",
                )
        axis.set_yticks(
            range(len(selected)),
            [
                f"{_ARMS[r['config']['arm']]} / {r['config'].get('seed', '?')} / {r['summary']['status']}"
                for r in selected
            ],
            fontsize=7,
        )
        axis.set_title(
            f"{vision}: cache residency (bands) and delivered generations (dots; x = late)"
        )
        axis.set_xlabel(
            "Seconds from process launch; | = reader start, * = first batch"
        )
        axis.grid(axis="x", alpha=0.2)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    generations = sorted(
        {
            int(e["generation"])
            for r in runs
            for e in r["lifecycle"]
            if _number(e.get("generation")) is not None
        }
        | {int(g) for r in runs for b in r["batches"] for g in b.get("generations", {})}
    )
    if generations:
        figure.legend(
            handles=[
                Patch(color=plt.get_cmap("tab10")(g % 10), label=f"Generation {g}")
                for g in generations
            ]
            + [
                Line2D(
                    [0],
                    [0],
                    color="#596874",
                    linewidth=3,
                    label="Finalized corpus on disk",
                )
            ],
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=min(6, len(generations) + 1),
        )
    figure.tight_layout()
    _save(figure, output, "01_timeline")
    return figure


def _reuse(runs: list[dict[str, Any]], plt: Any, output: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(14, 4), squeeze=False)
    for vision, axis in zip(_VISIONS, axes[0]):
        caps = set()
        for a, arm in enumerate(_MAIN):
            selected = _group(runs, arm, vision)
            for s, run in enumerate(selected):
                usage = run["summary"].get("episode_usage", [])
                for i, episode in enumerate(usage):
                    value = _number(episode.get("delivered_chunks"))
                    if value is None:
                        continue
                    evicted = episode.get("evicted") is True
                    axis.scatter(
                        a + (s - 1) * 0.16 + ((i % 7) - 3) * 0.012,
                        value,
                        marker="D" if evicted else "o",
                        s=12,
                        facecolors=(
                            "none" if evicted or not _qualified(run) else _COLORS[a]
                        ),
                        edgecolors=_COLORS[a],
                        alpha=0.7,
                        linewidths=0.7,
                    )
                cap = _number(run["config"].get("quota"))
                if cap is not None:
                    caps.add(cap)
        for cap in caps:
            axis.axhline(
                cap, color="#555555", linestyle="--", linewidth=1, label=f"K={cap:g}"
            )
        axis.set_xticks(range(3), ["LeRobot / K", "ODS / no K", "ODS / K"], rotation=15)
        axis.set_title(vision)
        axis.set_ylabel("Delivered chunks per immutable source")
        axis.set_yscale("symlog", linthresh=10)
        axis.set_ylim(bottom=0)
        axis.grid(axis="y", alpha=0.2)
        if caps:
            axis.legend(fontsize=8)
    figure.suptitle("All identities; hollow diamonds mark collector-cache eviction")
    figure.tight_layout()
    _save(figure, output, "02_reuse")
    plt.close(figure)


def _latency(runs: list[dict[str, Any]], plt: Any, output: Path) -> None:
    figure, axes = plt.subplots(3, 3, figsize=(13, 10), squeeze=False)
    for v, vision in enumerate(_VISIONS):
        for a, arm in enumerate(_MAIN):
            selected = _group(runs, arm, vision)
            _plot_points(
                axes[v, 0],
                selected,
                "first_batch_from_launch_seconds",
                a - 0.12,
                _COLORS[a],
            )
            _plot_points(
                axes[v, 0],
                selected,
                "first_batch_from_reader_seconds",
                a + 0.12,
                "#777777",
            )
            _plot_points(
                axes[v, 1],
                selected,
                "request_latency_p95_seconds",
                a,
                _COLORS[a],
                scale=0.001,
            )
            for s, run in enumerate(selected):
                demand = _number(run["config"].get("requested_chunks_per_second"))
                delivered = _metric(run, "delivered_chunks_per_second")
                if demand is not None and demand > 0 and delivered is not None:
                    axes[v, 2].scatter(
                        a + (s - 1) * 0.045,
                        delivered / demand,
                        facecolors=_COLORS[a] if _qualified(run) else "none",
                        edgecolors=_COLORS[a],
                        s=23,
                    )
            ratios = [
                _metric(r, "delivered_chunks_per_second")
                / r["config"]["requested_chunks_per_second"]
                for r in selected
                if _qualified(r)
                and _metric(r, "delivered_chunks_per_second") is not None
                and (_number(r["config"].get("requested_chunks_per_second")) or 0) > 0
            ]
            if ratios:
                middle = median(ratios)
                axes[v, 2].errorbar(
                    a,
                    middle,
                    yerr=[[middle - min(ratios)], [max(ratios) - middle]],
                    fmt="_",
                    markersize=14,
                    capsize=3,
                    color=_COLORS[a],
                )
        for axis, title, label in zip(
            axes[v],
            (
                "First batch: launch (color) / reader (gray)",
                "Consumer-visible p95 batch latency",
                "Delivered rate / nominal demand",
            ),
            ("Seconds", "Milliseconds", "Ratio"),
        ):
            axis.set_title(f"{vision}: {title}", fontsize=9)
            axis.set_ylabel(label)
            axis.set_xticks(
                range(3), ["LeRobot / K", "ODS / no K", "ODS / K"], rotation=15
            )
            axis.grid(axis="y", alpha=0.2)
        axes[v, 2].axhline(1, color="#555555", linestyle="--", linewidth=1)
    figure.suptitle("Seed points; median/range use complete, qualified runs only")
    figure.tight_layout()
    _save(figure, output, "03_latency_demand")
    plt.close(figure)


def _visual_load(runs: list[dict[str, Any]], plt: Any, output: Path) -> None:
    fields = (
        ("delivered_chunks_per_second", "Delivered chunks/s", 1),
        ("accepted_episodes_per_second", "Accepted episodes/s", 1),
        ("logical_payload_bytes_per_second", "Logical returned MiB/s", 1024**2),
        ("peak_pss_bytes", "Sampled process-tree peak PSS (GiB)", 1024**3),
        ("peak_vram_bytes", "Sampled VRAM (GiB; declared scope)", 1024**3),
        ("final_disk_bytes", "Final dataset storage (GiB)", 1024**3),
    )
    figure, axes = plt.subplots(2, 3, figsize=(13, 8), squeeze=False)
    for axis, (field, title, scale) in zip(axes.flat, fields):
        for a, arm in enumerate(_MAIN):
            medians, positions = [], []
            for v, vision in enumerate(_VISIONS):
                selected = _group(runs, arm, vision)
                x = v + (a - 1) * 0.17
                _plot_points(axis, selected, field, x, _COLORS[a], scale=scale)
                stats = _stats(selected, field)
                if stats["n"]:
                    positions.append(x)
                    medians.append(stats["median"] / scale)
            axis.plot(
                positions,
                medians,
                color=_COLORS[a],
                linewidth=1,
                alpha=0.65,
                label=_ARMS[arm],
            )
        axis.set_xticks(range(3), _VISIONS)
        axis.set_title(title, fontsize=10)
        axis.grid(axis="y", alpha=0.2)
    axes[0, 0].legend(fontsize=7)
    figure.suptitle(
        "Visual workload: filled=qualified; hollow=incomplete/unqualified; absent=unknown"
    )
    figure.tight_layout()
    _save(figure, output, "04_visual_load")
    plt.close(figure)


def _animate(runs: list[dict[str, Any]], plt: Any, output: Path) -> None:
    from matplotlib.animation import FuncAnimation, PillowWriter

    selected = [r for r in runs if r["config"]["arm"] == "protected"]
    if not selected:
        return
    seed = min(str(r["config"].get("seed")) for r in selected)
    selected = [r for r in selected if str(r["config"].get("seed")) == seed]
    figure, axes = plt.subplots(
        len(selected), 1, figsize=(9, 3 * len(selected)), squeeze=False
    )
    cursors = []
    end = 0.0
    for run, axis in zip(selected, axes[:, 0]):
        points = sorted(
            (
                (_number(b.get("time_seconds")), _number(b.get("chunks")))
                for b in run["batches"]
                if _number(b.get("time_seconds")) is not None
                and _number(b.get("chunks")) is not None
            )
        )
        total = 0.0
        times, counts = [0.0], [0.0]
        for time_value, chunks in points:
            total += chunks
            times.append(time_value)
            counts.append(total)
        end = max(end, max(times))
        axis.step(times, counts, where="post", color="#23977c")
        axis.set(
            title=f"Protected / {run['config']['vision']} / seed {seed}",
            xlabel="Seconds from launch",
            ylabel="Lifetime delivered chunks",
        )
        cursors.append(axis.axvline(0, color="#dc8b36"))
    figure.tight_layout()

    def update(frame: int) -> list[Any]:
        for cursor in cursors:
            cursor.set_xdata([end * frame / 119] * 2)
        return cursors

    animation = FuncAnimation(figure, update, frames=120, blit=False)
    animation.save(output / "data_flow.gif", writer=PillowWriter(fps=6))
    plt.close(figure)


def write_report(
    input_root: Path, output_dir: Path | None = None, *, animation: bool = False
) -> Path:
    """Write one report and four figure sets from retained experiment logs.

    Unknown measurements remain unknown. Aggregates exclude failed, unsupported
    and unqualified runs, whose summaries and any measurements remain visible.
    No performance winner or learning conclusion is inferred.

    Args:
        input_root: Root containing run directories, or one summary JSON file.
        output_dir: Artifact directory, defaulting to ``input_root/report``.
        animation: Also render a posthoc GIF from recorded protected-run batches.

    Returns:
        Path to the generated Markdown report.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = _load(Path(input_root))
    output = (
        Path(output_dir)
        if output_dir is not None
        else (
            Path(input_root).parent / "report"
            if Path(input_root).is_file()
            else Path(input_root) / "report"
        )
    )
    output.mkdir(parents=True, exist_ok=True)
    groups = []
    for arm in _ARMS:
        for vision in _VISIONS:
            selected = _group(runs, arm, vision)
            if selected:
                groups.append(
                    {
                        "arm": arm,
                        "vision": vision,
                        "attempted": len(selected),
                        "qualified": sum(_qualified(r) for r in selected),
                        "metrics": {key: _stats(selected, key) for key in _METRICS},
                    }
                )
    serialized = {
        "aggregation": "suite=report|appendix, complete and qualification_passed=true only; median [min,max]; unknown is null",
        "groups": groups,
        "runs": [{"summary_path": r["path"], **r["summary"]} for r in runs],
    }
    (output / "report.json").write_text(
        json.dumps(serialized, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    figure = _timeline(runs, plt, output)
    plt.close(figure)
    _reuse(runs, plt, output)
    _latency(runs, plt, output)
    _visual_load(runs, plt, output)
    if animation:
        _animate(runs, plt, output)
    lines = [
        "# ODS data-stream capability report",
        "",
        "This report measures data delivery, not training or policy quality. It does not assume ODS is faster than decoded LeRobot.",
        "",
        "All attempted runs are retained below. Filled plot markers and aggregate median [min,max] include only complete, explicitly qualified report/appendix runs. Hollow markers preserve measurements from incomplete, unqualified, smoke or calibration runs; missing measurements are unknown, never zero. Appendix results are one-seed diagnostics when only one seed is present.",
        "",
        "## Time and memory",
        "",
    ]
    lines += _table(
        [
            "Group / vision",
            "Attempted / qualified",
            "First batch from launch (s)",
            "First batch from reader (s)",
            "Batch p95 (s)",
            "Chunks/s",
            "Peak PSS (GiB)",
            "Final disk (GiB)",
        ],
        [
            [
                f"{'Main' if g['arm'] in _MAIN else 'Appendix'}: {_ARMS[g['arm']]} / {g['vision']}",
                f"{g['attempted']} / {g['qualified']}",
                *(
                    _formatted(g["metrics"][key], scale)
                    for key, scale in (
                        ("first_batch_from_launch_seconds", 1),
                        ("first_batch_from_reader_seconds", 1),
                        ("request_latency_p95_seconds", 1),
                        ("delivered_chunks_per_second", 1),
                        ("peak_pss_bytes", 1024**3),
                        ("final_disk_bytes", 1024**3),
                    )
                ),
            ]
            for g in groups
        ],
    )
    lines += [
        "",
        "Launch latency includes the collect-N LeRobot barrier versus the initial-C ODS barrier. Reader latency begins with source-ready access; LeRobot still performs real video decoding. These scopes overlap and must not be added. Final-batch latency includes waiting and output conversion; worker item-service time is a separate component measurement, not decoder kernel time.",
        "",
        "## Integrity and exposure",
        "",
    ]
    lines += _table(
        [
            "Group / vision",
            "Delivered chunks",
            "Within window",
            "After window",
            "K violations",
            "Unconsumed prefetch",
            "PSNR (dB)",
            "MAE",
        ],
        [
            [
                f"{_ARMS[g['arm']]} / {g['vision']}",
                *(
                    _formatted(g["metrics"][key])
                    for key in (
                        "delivered_chunks",
                        "window_delivered_chunks",
                        "post_window_chunks",
                        "cap_violations",
                        "unconsumed_prefetch_chunks",
                        "quality_psnr_db",
                        "quality_mae",
                    )
                ),
            ]
            for g in groups
        ],
    )
    lines += [
        "",
        "One credit is one anchor-source chunk, independent of camera count or context length. Delivered chunks and reuse are lifetime counts, including any complete batch returned after the observation deadline; late batches are excluded from window throughput and latency metrics. Source counts include evicted identities. A cap bounds delivery rather than optimizer updates. AV1 video is lossy; speed must be considered alongside the reported raw-reference quality, codec and recording configuration. Infinite PSNR for exact equality requires an explicit equality count in the run metadata; it is not replaced with a finite score.",
        "",
        "## Run qualification",
        "",
    ]
    lines += _table(
        ["Run", "Suite", "Seed", "Status", "Qualified", "Reason / error"],
        [
            [
                f"{_ARMS[r['config']['arm']]} / {r['config']['vision']}",
                r["config"].get("suite", "unknown"),
                r["config"].get("seed", "unknown"),
                r["summary"]["status"],
                r["summary"].get("metrics", {}).get("qualification_passed", "unknown"),
                r["summary"].get("error")
                or r["summary"].get("qualification")
                or "See retained summary",
            ]
            for r in runs
        ],
    )
    lines += ["", "## Figures", ""]
    for name, caption in (
        (
            "01_timeline",
            "Availability, cache residency and delivered-generation timelines. Offline collector bands stop at reader start (after engine shutdown); gray lines show finalized files on disk independently of remaining K. Unrecorded write locks are not inferred.",
        ),
        (
            "02_reuse",
            "Per-source reuse; hollow diamonds mean collector-cache eviction. For LeRobot, all finalized corpus files remain available despite collector eviction. The K line is a protection setting, not an optimal learning budget.",
        ),
        (
            "03_latency_demand",
            "Cold versus source-ready first batch, consumer p95 and served demand. Quota waits and service bottlenecks must be interpreted using the retained counters.",
        ),
        (
            "04_visual_load",
            "State/single/dual workload costs. Logical MiB/s is returned tensor payload, not measured bus or disk bandwidth. Memory coverage follows each run's instrumentation declaration.",
        ),
    ):
        lines += [
            f"![{caption}]({name}.png)",
            "",
            caption + f" [SVG]({name}.svg) · [PDF]({name}.pdf)",
            "",
        ]
    if (output / "data_flow.gif").exists():
        lines += [
            "![Posthoc recorded delivery animation](data_flow.gif)",
            "",
            "Animation replays logged deliveries; it is not a recording of simulation or a separate measurement.",
            "",
        ]
    lines += [
        "## Interpretation limits",
        "",
        "A new source identity does not establish broader state coverage. Fixed cache tensor bytes do not establish constant total process memory. Protected waiting can intentionally reduce delivered throughput; it must not be presented as a copy/decoder bottleneck. This finalized local LeRobot path does not establish that the LeRobot ecosystem cannot stream. No universal speedup, learning-efficiency or policy-performance claim follows from these measurements.",
        "",
        "Raw summaries and input paths, including configuration, qualification failures and cleanup records, are preserved in [report.json](report.json). Re-run this renderer after interrupted experiments have written their final summaries; partial JSONL lines fail visibly rather than being discarded.",
    ]
    report_path = output / "report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--animation", action="store_true")
    arguments = parser.parse_args()
    print(
        write_report(
            arguments.input_root, arguments.output, animation=arguments.animation
        )
    )
