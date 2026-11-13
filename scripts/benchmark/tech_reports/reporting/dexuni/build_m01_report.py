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
"""Summarize the five formal M01 episodes per platform into results and REPORT.md.

Run: .venv/bin/python scripts/benchmark/tech_reports/reporting/dexuni/build_m01_report.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

import numpy as np
from scripts.benchmark.tech_reports.physics.experiments.dexuni.m01_common import OUTPUT

PLATFORMS = {
    "embodichain": "EmbodiChain / DexUni two-way, VBD 12",
    "isaaclab": "Isaac Lab / CouplerProxy MuJoCo-Warp + VBD 12",
}
OBSERVATIONS = (
    "reference_sag_mm",
    "hold_sag_mm",
    "extra_hold_sag_mm",
    "hold_end_distance_mm",
    "hold_lowest_node_above_table_mm",
    "carry_sag_mm",
    "final_residual_sag_mm",
    "final_centre_x_displacement_mm",
)
__all__ = ["main"]


def _load(platform: str) -> list[dict]:
    runs = []
    for directory in sorted((OUTPUT / "measurement" / platform).glob("r0*")):
        manifest = json.loads((directory / "manifest.json").read_text())
        with (directory / "timing.csv").open() as stream:
            steps = [float(row["step_s"]) for row in csv.DictReader(stream)]
        runs.append({"repeat": directory.name, "manifest": manifest, "steps": steps})
    return runs


def main() -> None:
    """Write episodes.csv, steps.csv, summary.json, a chart and REPORT.md."""
    results = OUTPUT / "measurement/results"
    results.mkdir(parents=True, exist_ok=True)
    data = {p: _load(p) for p in PLATFORMS}
    episodes, summary = [], {}
    with (results / "steps.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["platform", "repeat", "frame", "step_s"])
        for platform, runs in data.items():
            for run in runs:
                for frame, value in enumerate(run["steps"], 1):
                    writer.writerow([platform, run["repeat"], frame, value])
    for platform, runs in data.items():
        rows = []
        for run in runs:
            m = run["manifest"]
            ok = m["status"] == "completed" and all(m["performance_checks"].values())
            row = {
                "platform": platform,
                "repeat": run["repeat"],
                "valid": ok,
                "mean_ms": m["summary"]["cost_time_ms"],
                "median_ms": m["summary"]["median_ms"],
                "p95_ms": m["summary"]["p95_ms"],
                "init_s": m["init_s"],
                "gpu_process_peak_mib": m["gpu_memory"].get("gpu_process_peak_mib"),
                "cuda_graph": m["effective"]["cuda_graph_status"],
                "vbd_iterations": m["effective"]["vbd_iterations"],
                **{k: m["observations"][k] for k in OBSERVATIONS},
            }
            rows.append(row)
        episodes += rows
        mean = np.array([r["mean_ms"] for r in rows])
        summary[platform] = {
            "label": PLATFORMS[platform],
            "episodes": len(rows),
            "valid": sum(r["valid"] for r in rows),
            "mean_ms": float(mean.mean()),
            "sd_ms": float(mean.std(ddof=1)) if len(mean) > 1 else 0.0,
            "p95_ms_mean": float(np.mean([r["p95_ms"] for r in rows])),
            "init_s_mean": float(np.mean([r["init_s"] for r in rows])),
            "gpu_process_peak_mib_max": max(
                (r["gpu_process_peak_mib"] or 0) for r in rows
            ),
            **{
                k: {
                    "mean": float(np.mean([r[k] for r in rows])),
                    "min": float(np.min([r[k] for r in rows])),
                    "max": float(np.max([r[k] for r in rows])),
                }
                for k in OBSERVATIONS
            },
            "versions": runs[0]["manifest"]["environment"].get("versions"),
        }
    with (results / "episodes.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(episodes[0]))
        writer.writeheader()
        writer.writerows(episodes)
    ratio = summary["isaaclab"]["mean_ms"] / summary["embodichain"]["mean_ms"]
    summary["isaaclab_over_embodichain_step_time"] = ratio
    (results / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for i, platform in enumerate(PLATFORMS):
        values = [r["mean_ms"] for r in episodes if r["platform"] == platform]
        s = summary[platform]
        axes[0].bar(i, s["mean_ms"], yerr=s["sd_ms"], capsize=6, alpha=0.6)
        axes[0].scatter([i] * len(values), values, color="black", s=12, zorder=3)
        axes[1].plot(
            range(1, len(values) + 1), values, marker="o", label=PLATFORMS[platform]
        )
    axes[0].set_xticks(range(len(PLATFORMS)), list(PLATFORMS.values()), fontsize=8)
    axes[0].set(ylabel="Mean outer step (ms)", title="Five episodes: mean ± SD")
    axes[1].set(xlabel="Repeat", ylabel="Mean outer step (ms)", title="Per episode")
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(results / "performance_comparison.png", dpi=150)
    plt.close(fig)

    def fmt(platform: str, key: str) -> str:
        v = summary[platform][key]
        return f"{v['mean']:.1f} ({v['min']:.1f}–{v['max']:.1f})"

    ec, il = summary["embodichain"], summary["isaaclab"]
    order = sorted(PLATFORMS, key=lambda p: summary[p]["mean_ms"])
    lines = [
        "# M01 W1 双臂软梁弯曲、搬运与恢复：正式性能",
        "",
        "同一冻结四面体软梁（1225 节点 / 3840 四面体，泡沫材料）、W1 关节指令与 1/60 s 外步、10 子步 × 12 VBD 迭代；两边 CUDA Graph 开启，无相机。双向耦合 + 动态关节：两平台用各自的关节驱动跟踪同一 DexSim 关节目标（DexUni two_way；Isaac Lab CouplerProxy：MuJoCo-Warp 机器人 + Newton VBD，staggered 代理）。",
        "",
        f"每平台 5 次独立运行，交替执行；每次 {ec_steps(data)} 外步、16.0 s，前 30 步预热。每帧关节目标在计时前写入。± 为五次运行均值的样本标准差。GPU 时钟未锁定。",
        "",
        "## Time & Memory",
        "",
        "| 平台 | 外步均值 ± SD (ms) | 每次 p95 的均值 (ms) | 进程显存采样最大值 (MiB) | 初始化 (s) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for p in PLATFORMS:
        s = summary[p]
        lines.append(
            f"| {s['label']} | {s['mean_ms']:.3f} ± {s['sd_ms']:.3f} | {s['p95_ms_mean']:.3f} | {s['gpu_process_peak_mib_max']:.1f} | {s['init_s_mean']:.2f} |"
        )
    lines += [
        "",
        "外步计时等待 Warp GPU 工作完成，包含平台推进（关节驱动、耦合与 VBD）；目标写入、状态读回、检查、显存采样和保存在计时之外。显存为进程追踪器的 nvidia-smi 采样，不是瞬态峰值。",
        "",
        "## Success & Other Metrics",
        "",
        "均值（最小–最大），单位 mm。下垂为梁中部节点组相对两端节点组连线的距离（向下为正）；参考为刚抬起（6.0–6.5 s），保持为弯曲保持段（8.5–10.0 s），搬运为 12.0–13.5 s，残余为结束帧。",
        "",
        "| 平台 | 有效 / 尝试 | 参考下垂 | 保持下垂 | 额外下垂 | 保持端距 | 搬运下垂 | 残余下垂 | 梁中心 X 位移 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for p in PLATFORMS:
        s = summary[p]
        lines.append(
            f"| {s['label']} | {s['valid']}/{s['episodes']} | {fmt(p, 'reference_sag_mm')} | {fmt(p, 'hold_sag_mm')} | {fmt(p, 'extra_hold_sag_mm')} | {fmt(p, 'hold_end_distance_mm')} | {fmt(p, 'carry_sag_mm')} | {fmt(p, 'final_residual_sag_mm')} | {fmt(p, 'final_centre_x_displacement_mm')} |"
        )
    lines += [
        "",
        "未设通过阈值（计划为指引）；所有有效运行状态全程有限。",
        "",
        "## Leaderboard",
        "",
        "| 排序（按外步耗时） | 平台 | 外步均值 (ms) |",
        "| --- | --- | --- |",
    ]
    for rank, p in enumerate(order, 1):
        lines.append(
            f"| {rank} | {summary[p]['label']} | {summary[p]['mean_ms']:.3f} |"
        )
    lines += [
        "",
        f"Isaac / EC 外步耗时比为 {ratio:.3f}。两边迭代数相同，但机器人求解（DexUni MuJoCo 关节 vs MuJoCo-Warp）与耦合实现不同，比值包含这些差异，不是相同算法的单独加速倍数，也不代表带渲染的端到端吞吐。",
        "",
        "[对照视频](videos/m01_comparison.mp4) · [协议与源码哈希](measurement/protocol.json) · [逐次统计](measurement/results/episodes.csv) · [全部外步计时](measurement/results/steps.csv) · [汇总](measurement/results/summary.json) · [性能对比图](measurement/results/performance_comparison.png)",
        "",
        f"软件版本：EC Newton {(ec['versions'] or {}).get('newton')}；Isaac Lab Newton {(il['versions'] or {}).get('newton')}。Newton 版本不同，未做统一。",
    ]
    if (OUTPUT / "measurement/invalid_contention").exists():
        lines += [
            "",
            "EC 第 1 次首轮尝试与另一会话的 DexSim M03 预览在 GPU 上重叠约 12 s，按协议作废并移至 [measurement/invalid_contention](measurement/invalid_contention/README.md)，GPU 空闲后重测；其他九份未受影响（驱动在每次运行前等待其他仿真结束，并记录运行中出现的重叠，见 measurement/logs/gpu_contention.log）。",
        ]
    (OUTPUT / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps({p: (summary[p]["mean_ms"], summary[p]["sd_ms"]) for p in PLATFORMS})
    )


def ec_steps(data: dict) -> int:
    return data["embodichain"][0]["manifest"]["outer_steps"]


if __name__ == "__main__":
    main()
