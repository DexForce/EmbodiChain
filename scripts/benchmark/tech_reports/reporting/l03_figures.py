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

"""Render L-03 figures from formal samples only, without importing a simulator."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

__all__ = ["main"]


def main() -> None:
    """Keep plotted values, run IDs, ranges and export settings beside each figure."""
    base = Path("outputs/benchmarks/L-03/three-robots-v1")
    summary = json.loads((base / "analysis/summary.json").read_text())
    rows = json.loads((base / "analysis/runs.json").read_text())
    assert len(summary) == 36 and all(s["repeats"] == 3 for s in summary)
    lookup = {
        (s["robot"], s["num_envs"], s["cache_mode"], s["platform"]): s for s in summary
    }
    by_run = {s["run_id"]: s for s in rows}
    output = base / "figures"
    output.mkdir(exist_ok=True)
    font = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    font_manager.fontManager.addfont(str(font))
    family = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update(
        {
            "font.family": family,
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.titleweight": "bold",
        }
    )
    robots = {"cartpole": "Cartpole", "franka": "Franka Panda", "go2": "Unitree Go2"}
    platforms = {
        "embodichain": ("EmbodiChain / DexUni", "#007d82"),
        "isaaclab": ("Isaac Lab / PhysX", "#c56b24"),
    }
    scales = [1, 100, 1000]
    ready_ymax = max(s["ready_max_s"] for s in summary) * 1.08

    def save(figure, name: str) -> None:
        for extension in ("png", "svg", "pdf"):
            figure.savefig(
                output / (name + "." + extension),
                dpi=170,
                bbox_inches="tight",
                facecolor="white",
            )
        plt.close(figure)

    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.2), sharex=True, sharey=True)
    for i, cache in enumerate(("C", "H")):
        for j, (robot, title) in enumerate(robots.items()):
            ax = axes[i, j]
            for platform, (label, color) in platforms.items():
                data = [lookup[(robot, n, cache, platform)] for n in scales]
                y = np.array([s["ready_median_s"] for s in data])
                error = [
                    y - [s["ready_min_s"] for s in data],
                    [s["ready_max_s"] for s in data] - y,
                ]
                ax.errorbar(
                    scales,
                    y,
                    yerr=error,
                    label=label,
                    color=color,
                    marker="o",
                    capsize=4,
                    linewidth=1.8,
                )
            ax.set_title(f"{title} · {cache}")
            ax.set_xscale("log")
            ax.set_xticks(scales, labels=[str(n) for n in scales])
            ax.set_ylim(0, ready_ymax)
            ax.grid(axis="y", alpha=0.2)
            if i == 1:
                ax.set_xlabel("环境数 num_envs（对数轴）")
            if j == 0:
                ax.set_ylabel("就绪时间（s）")
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        names,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.0),
    )
    fig.subplots_adjust(top=0.89, bottom=0.13, hspace=0.28, wspace=0.15)
    fig.text(
        0.5,
        0.015,
        "每点 3 次：中位数与最小–最大范围。C/H 均为新进程；H 另有同配置预载，C 并非磁盘冷启动。",
        ha="center",
        fontsize=10,
    )
    save(fig, "loading-time")

    stages = {
        "engine_s": ("运行时启动", "#305d78"),
        "declare_s": ("声明 / 复制", "#37908a"),
        "prepare_s": ("物理准备", "#d6a443"),
        "initialize_first_readback_s": ("初始化 / 首步 / 读回", "#b46d70"),
    }
    representative = []
    names = []
    for robot, title in robots.items():
        for platform in platforms:
            representative.append(
                by_run[lookup[(robot, 1000, "H", platform)]["representative_run_id"]]
            )
            names.append(
                title
                + " · "
                + ("EmbodiChain" if platform == "embodichain" else "Isaac Lab")
            )
    fig, ax = plt.subplots(figsize=(12.5, 5.7))
    ypos = np.arange(6)
    left = np.zeros(6)
    for key, (label, color) in stages.items():
        values = np.array([s[key] for s in representative])
        ax.barh(ypos, values, left=left, label=label, color=color, height=0.65)
        for i, value in enumerate(values):
            if value >= 0.45:
                ax.text(
                    left[i] + value / 2,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=9,
                )
        left += values
    for i, value in enumerate(left):
        ax.text(value + 0.10, i, f"{value:.3f} s", va="center", fontsize=10)
    ax.set_yticks(ypos, names)
    ax.invert_yaxis()
    ax.set_xlabel("阶段耗时（s）")
    ax.set_xlim(0, left.max() * 1.16)
    ax.grid(axis="x", alpha=0.15)
    ax.set_axisbelow(True)
    fig.legend(loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.6, 0.99))
    fig.subplots_adjust(left=0.235, right=0.97, top=0.86, bottom=0.18)
    fig.text(
        0.6,
        0.03,
        "N=1000、H；每组取总就绪时间居中的实际运行。分段边界按平台接口定义，不能视为纯解析器对比。",
        ha="center",
        fontsize=9,
    )
    save(fig, "loading-stages")

    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.2), sharex=True, sharey="row")
    for i, (metric, label) in enumerate(
        (("ram", "进程树 RAM 峰值（MiB）"), ("vram", "进程显存峰值（MiB）"))
    ):
        memory_ymax = (
            max(s[metric + "_max_mib"] for s in summary if s["cache_mode"] == "H")
            * 1.08
        )
        for j, (robot, title) in enumerate(robots.items()):
            ax = axes[i, j]
            for platform, (platform_label, color) in platforms.items():
                data = [lookup[(robot, n, "H", platform)] for n in scales]
                y = np.array([s[metric + "_median_mib"] for s in data])
                error = [
                    y - [s[metric + "_min_mib"] for s in data],
                    [s[metric + "_max_mib"] for s in data] - y,
                ]
                ax.errorbar(
                    scales,
                    y,
                    yerr=error,
                    color=color,
                    marker="o",
                    capsize=4,
                    linewidth=1.8,
                    label=platform_label,
                )
            ax.set_title(title + " · H")
            ax.set_xscale("log")
            ax.set_xticks(scales, labels=[str(n) for n in scales])
            ax.set_ylim(0, memory_ymax)
            ax.grid(axis="y", alpha=0.2)
            if i == 1:
                ax.set_xlabel("环境数 num_envs（对数轴）")
            if j == 0:
                ax.set_ylabel(label)
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        names,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1),
    )
    fig.subplots_adjust(top=0.89, bottom=0.13, hspace=0.28, wspace=0.15)
    fig.text(
        0.5,
        0.015,
        "仅统计入口至 ready 窗口；目标采样间隔 20 ms。每点为 3 次进程峰值的中位数，误差线为最小–最大值。",
        ha="center",
        fontsize=10,
    )
    save(fig, "loading-memory")
    (output / "plot-data.json").write_text(
        json.dumps(
            dict(
                scope="108 resource-isolated formal samples; preloads, pilots and media excluded",
                summary=summary,
                stage_representatives=representative,
                parameters=dict(
                    font=str(font),
                    dpi=170,
                    scales=scales,
                    time_units="seconds",
                    memory_units="MiB",
                    error_bars="min to max across 3 fresh processes, not confidence intervals",
                    representative="observed run with median ready_s",
                    colors=platforms,
                ),
            ),
            indent=2,
        )
        + "\n"
    )
    shutil.copy2(__file__, output / "plot-source.py")
    print("Wrote 3 figures, each in PNG/SVG/PDF, with source and plot-data.json")


if __name__ == "__main__":
    main()
