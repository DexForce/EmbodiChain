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
"""Publication figures from retained per-process benchmark statistics."""

from __future__ import annotations

import base64
import math
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, findfont

__all__ = ["build_performance_figures"]


def build_performance_figures(tasks: list[dict], directory: Path) -> dict:
    """Export absolute/normalized paired means as SVG and PNG.

    Args:
        tasks: Retained task summaries with per-process mean and sample SD.
        directory: Destination for standalone publication figures.

    Returns:
        Figure variants keyed by unit and ordering, with embedded SVG content.
    """
    directory.mkdir(parents=True, exist_ok=True)
    font = findfont(
        FontProperties(
            family=[
                "Noto Sans CJK JP",
                "Noto Sans CJK SC",
                "Microsoft YaHei",
                "DejaVu Sans",
            ]
        )
    )
    fp = FontProperties(fname=font)
    ec, il, ink, muted = "#087f78", "#82939e", "#18383e", "#687b80"
    formal = [t for t in tasks if t["kind"] == "formal"]
    result = {}
    with plt.rc_context(
        {"svg.fonttype": "path", "font.size": 10, "axes.unicode_minus": False}
    ):
        for unit in ("absolute", "relative"):
            for order in ("task", "reduction"):
                rows = (
                    sorted(formal, key=lambda t: -t["ec_latency_reduction_percent"])
                    if order == "reduction"
                    else formal
                )
                fig = plt.figure(figsize=(12.4, 6.9), facecolor="white")
                ax = fig.add_axes((0.235, 0.17, 0.48, 0.68))
                side = fig.add_axes((0.755, 0.17, 0.225, 0.68), sharey=ax)
                maximum = (
                    math.ceil(
                        max(
                            t["stats"]["isaaclab"]["mean_ms"]
                            + t["stats"]["isaaclab"]["sd_ms"]
                            for t in rows
                        )
                        / 20
                    )
                    * 20
                    if unit == "absolute"
                    else 100
                )
                ax.set_xlim(0, maximum * 1.015)
                ax.set_ylim(len(rows) - 0.45, -0.55)
                ax.set_xticks([maximum * i / 4 for i in range(5)])
                ax.set_yticks(
                    range(len(rows)),
                    [t["code"] + "  " + t["name"] for t in rows],
                    fontproperties=fp,
                    fontsize=10,
                    color=ink,
                )
                ax.tick_params(axis="y", length=0, pad=16)
                ax.tick_params(axis="x", colors=muted, labelsize=9, length=0, pad=10)
                ax.grid(axis="x", color="#e5ece9", linewidth=0.8)
                for spine in ax.spines.values():
                    spine.set_visible(False)
                ax.set_xlabel(
                    (
                        "物理外步耗时（ms / 步） · 越低越好"
                        if unit == "absolute"
                        else "相对耗时（%） · 每项 Isaac 均值 = 100"
                    ),
                    fontproperties=fp,
                    fontsize=10,
                    color=muted,
                    labelpad=14,
                )
                side.set_xlim(0, 3)
                side.axis("off")
                for x, label, color in (
                    (0.1, "EC / ms", ec),
                    (1.1, "Isaac / ms", il),
                    (2.25, "耗时降低", ec),
                ):
                    side.text(
                        x,
                        -1.08,
                        label,
                        fontproperties=fp,
                        fontsize=9,
                        color=color,
                        ha="center",
                    )
                for y, t in enumerate(rows):
                    e, i = t["stats"]["embodichain"], t["stats"]["isaaclab"]
                    scale = 1 if unit == "absolute" else 100 / i["mean_ms"]
                    if t["ec_latency_reduction_percent"] >= 50:
                        ax.axhspan(y - 0.43, y + 0.43, color="#f0f8f4", zorder=0)
                    ax.plot(
                        [e["mean_ms"] * scale, i["mean_ms"] * scale],
                        [y, y],
                        color="#c4dcd4",
                        lw=4,
                        solid_capstyle="round",
                        zorder=2,
                    )
                    for stats, color, marker in ((i, il, "D"), (e, ec, "o")):
                        ax.errorbar(
                            stats["mean_ms"] * scale,
                            y,
                            xerr=stats["sd_ms"] if unit == "absolute" else None,
                            fmt=marker,
                            color=color,
                            markersize=7,
                            capsize=3,
                            elinewidth=1.5,
                            markeredgecolor="white",
                            markeredgewidth=1,
                            zorder=3,
                        )
                    side.text(
                        0.1,
                        y,
                        f'{e["mean_ms"]:.2f}',
                        ha="center",
                        va="center",
                        fontsize=10,
                        color=ec,
                        fontweight="bold",
                    )
                    side.text(
                        1.1,
                        y,
                        f'{i["mean_ms"]:.2f}',
                        ha="center",
                        va="center",
                        fontsize=10,
                        color=muted,
                    )
                    side.text(
                        2.25,
                        y,
                        f'−{t["ec_latency_reduction_percent"]:.1f}%',
                        ha="center",
                        va="center",
                        fontsize=11,
                        color=ec,
                        fontweight="bold",
                    )
                fig.text(
                    0.035,
                    0.95,
                    "物理推进性能 / EmbodiChain vs Isaac Lab",
                    fontproperties=fp,
                    fontsize=17,
                    color=ink,
                )
                fig.text(
                    0.035,
                    0.901,
                    "10 项双平台正式测量 · 每侧 5 个进程 · 无渲染 · GPU 同步",
                    fontproperties=fp,
                    fontsize=10,
                    color=muted,
                )
                fig.text(
                    0.73,
                    0.915,
                    "● EmbodiChain",
                    fontproperties=fp,
                    fontsize=10,
                    color=ec,
                )
                fig.text(
                    0.86, 0.915, "◆ Isaac Lab", fontproperties=fp, fontsize=10, color=il
                )
                fig.text(
                    0.035,
                    0.045,
                    "点为进程均值的平均值；绝对视图误差线为进程间 SD。不同求解策略及任务效果需结合正文解读。",
                    fontproperties=fp,
                    fontsize=9,
                    color=muted,
                )
                key = f"{unit}_{order}"
                svg = directory / f"physics_latency_{key}.svg"
                png = svg.with_suffix(".png")
                fig.savefig(svg, metadata={"Date": None})
                fig.savefig(png, dpi=200)
                plt.close(fig)
                result[key] = {
                    "svg_data": "data:image/svg+xml;base64,"
                    + base64.b64encode(svg.read_bytes()).decode(),
                    "svg_name": svg.name,
                    "png_name": png.name,
                    "tasks": [t["code"] for t in rows],
                }
    return result
