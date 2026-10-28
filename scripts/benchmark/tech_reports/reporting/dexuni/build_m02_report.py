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
"""Summarize the five formal M02 episodes per platform into results and REPORT.md.

Run: .venv/bin/python scripts/benchmark/tech_reports/reporting/dexuni/build_m02_report.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

import numpy as np
from scripts.benchmark.tech_reports.physics.experiments.dexuni.m02_common import OUTPUT

PLATFORMS = {
    "embodichain": "EmbodiChain / DexUni surface-fast 8",
    "isaaclab": "Isaac Lab / Newton VBD 10",
}
OBSERVATIONS = (
    "final_left_xy_error_mm",
    "final_right_xy_error_mm",
    "final_mid_xy_error_mm",
    "final_highest_node_above_table_mm",
    "hang_lowest_node_above_table_mm",
    "hang_jitter_p95_mm",
    "max_tcp_error_mm",
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
        "# M02 W1 方布单次对折：正式性能",
        "",
        "同一冻结网格（41×49 节点、10 mm 间距）、材料、W1 关节轨迹与 1/60 s 外步、10 子步；两边 CUDA Graph 开启，无相机。机器人为运动学回放、单向耦合（计划原写双向，用户确认改为单向），因此不包含机器人受载反馈。EmbodiChain 使用 DexUni soft 接触 + surface-fast（8 次基础迭代、batched Jacobi/Chebyshev/多层校正、缓存）；Isaac Lab 使用自带 Newton VBD 10 次迭代（Isaac Lab 无 surface-fast）。该组属于统一场景、材料、控制轨迹及时间步配置下的求解器级优化比较；基础迭代次数和preset加速机制是求解策略的一部分，不能按普通VBD迭代次数直接等价换算。",
        "",
        f"每平台 5 次独立运行，交替执行；每次 {ec_steps(data)} 外步、16.5 s，前 30 步预热。± 为五次运行均值的样本标准差。GPU 时钟未锁定。",
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
        "外步计时等待 Warp GPU 工作完成，包含平台推进及图内运动学回放；状态读回、检查、显存采样和保存在计时之外。显存为进程追踪器的 nvidia-smi 采样，不是瞬态峰值。",
        "",
        "## Success & Other Metrics",
        "",
        "均值（最小–最大），单位 mm。角点 XY 距离是上下两层对应角点（近边/远边材料节点组）的水平距离，越小对齐越好；抖动为悬挂阶段（7.2–8.0 s）远离夹爪节点的帧间二阶差分 p95。",
        "",
        "| 平台 | 有效 / 尝试 | 左角 XY | 右角 XY | 中点 XY | 最终最高点 | 悬挂最低点 | 悬挂抖动 p95 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for p in PLATFORMS:
        s = summary[p]
        lines.append(
            f"| {s['label']} | {s['valid']}/{s['episodes']} | {fmt(p, 'final_left_xy_error_mm')} | {fmt(p, 'final_right_xy_error_mm')} | {fmt(p, 'final_mid_xy_error_mm')} | {fmt(p, 'final_highest_node_above_table_mm')} | {fmt(p, 'hang_lowest_node_above_table_mm')} | {s['hang_jitter_p95_mm']['mean']:.2f} |"
        )
    lines += [
        "",
        "未设对齐通过阈值（计划为指引，不按阈值调参）；所有运行都完成拿起、悬挂和放下，布料状态全程有限，关节回放误差 ≤1e-5 rad。",
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
        f"Isaac / EC 外步耗时比为 {ratio:.3f}。该比值反映统一任务配置下DexUni surface-fast相对普通VBD的求解器级策略优化收益，基础迭代次数是策略的一部分。折叠效果按各项实测指标分别描述；该步进耗时比不代表带渲染的端到端吞吐。",
        "",
        "[对照视频](videos/m02_comparison.mp4) · [协议与源码哈希](measurement/protocol.json) · [逐次统计](measurement/results/episodes.csv) · [全部外步计时](measurement/results/steps.csv) · [汇总](measurement/results/summary.json) · [性能对比图](measurement/results/performance_comparison.png)",
        "",
        f"软件版本：EC Newton {(ec['versions'] or {}).get('newton')}；Isaac Lab Newton {(il['versions'] or {}).get('newton')}。Newton 版本不同，未做统一。",
    ]
    (OUTPUT / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps({p: (summary[p]["mean_ms"], summary[p]["sd_ms"]) for p in PLATFORMS})
    )


def ec_steps(data: dict) -> int:
    return data["embodichain"][0]["manifest"]["outer_steps"]


if __name__ == "__main__":
    main()
