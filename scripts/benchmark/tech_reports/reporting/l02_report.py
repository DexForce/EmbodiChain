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

"""Render L-02 evidence into a three-chapter draft and the required three tables."""

from __future__ import annotations
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    root = Path("outputs/benchmarks/L-02")
    analysis = root / "analysis"
    reports = root / "reports"
    reports.mkdir(exist_ok=True)
    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    rows = json.loads((analysis / "runs.json").read_text())
    summary = json.loads((analysis / "summary.json").read_text())
    media_index = root / "videos/validation.json"
    media = (
        json.loads(media_index.read_text())["records"] if media_index.exists() else []
    )
    media = [r for r in media if r.get("selected", True)]
    main = [r for r in summary if r["purpose"] == "main"]
    capacity = [r for r in summary if r["purpose"] == "capacity"]
    capacity_plan = root / "plans/capacity_v2.json"
    capacity_pending = (
        any(r["status"] == "pending" for r in json.loads(capacity_plan.read_text()))
        if capacity_plan.exists()
        else True
    )
    labels = {
        "embodichain": "EmbodiChain / DexUni",
        "isaaclab": "Isaac Lab / PhysX TGS",
    }
    colors = {"embodichain": "#007a87", "isaaclab": "#cc681b"}
    shapes = {"cube": "立方体", "sphere": "球体"}
    done = sum(r["completed"] for r in main)
    eligible = sum(r["eligible"] for r in main)
    sample_gap_ms = 1000 * max(
        (
            r.get("memory_sample_interval_max_s", 0)
            for r in rows
            if r["purpose"] in ["main", "capacity"]
        ),
        default=0,
    )
    grouped = {
        tuple(
            r[k]
            for k in ["shape", "object_count", "asset_mode", "cache_mode", "platform"]
        ): r
        for r in main
    }

    def fmt(row: dict, key: str, digits: int = 2) -> str:
        x = row.get(key)
        return (
            f"{x['median']:.{digits}f} [{x['min']:.{digits}f}, {x['max']:.{digits}f}]"
            if x
            else "未取得"
        )

    def table(data: list[dict]) -> list[str]:
        lines = [
            "| 形状 | 数量 | 资产 | 缓存 | 完成／尝试 | 有效计时 | T_ready（s） | T_frame（s） | RAM 峰值（MiB） | GPU 进程峰值（MiB） |",
            "|---|---:|---|---|---:|---:|---:|---:|---:|---:|",
        ]
        for r in data:
            lines.append(
                f"| {shapes[r['shape']]} | {r['object_count']} | {r['asset_mode']} | {r['cache_mode']} | {r['completed']}/{r['attempts']} | {r['eligible']} | {fmt(r,'T_ready_s')} | {fmt(r,'T_frame_s')} | {fmt(r,'ram_peak_mib',0)} | {fmt(r,'vram_peak_mib',0)} |"
            )
        return lines

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.2,
            "svg.fonttype": "none",
        }
    )
    for cache, suffix in [("H", ""), ("C", "-C")]:
        fig, axs = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
        for i, shape in enumerate(["cube", "sphere"]):
            for platform in labels:
                for mode, style in [("S", "-o"), ("U", "--s")]:
                    data = sorted(
                        [
                            r
                            for r in main
                            if r["shape"] == shape
                            and r["platform"] == platform
                            and r["asset_mode"] == mode
                            and r["cache_mode"] == cache
                            and r["eligible"]
                        ],
                        key=lambda r: r["object_count"],
                    )
                    for j, key in enumerate(
                        ["T_frame_s", "ram_peak_mib", "vram_peak_mib"]
                    ):
                        points = [r for r in data if key in r]
                        if points:
                            x = [r["object_count"] for r in points]
                            y = [r[key]["median"] for r in points]
                            err = np.array(
                                [
                                    [r[key]["median"] - r[key]["min"] for r in points],
                                    [r[key]["max"] - r[key]["median"] for r in points],
                                ]
                            )
                            axs[i, j].errorbar(
                                x,
                                y,
                                yerr=err,
                                fmt=style,
                                color=colors[platform],
                                capsize=3,
                                label=(
                                    "EmbodiChain"
                                    if platform == "embodichain"
                                    else "Isaac Lab"
                                )
                                + " / "
                                + mode,
                                linewidth=1.4,
                                markersize=4,
                            )
            for j, label in enumerate(
                [
                    "Entry to first frame (s)",
                    "Process RAM peak (MiB)",
                    "Process GPU peak (MiB)",
                ]
            ):
                axs[i, j].set(
                    xscale="log",
                    xlabel="Dynamic object count",
                    ylabel=label,
                    title=shape.capitalize(),
                )
                axs[i, j].set_xticks([1, 100, 1000], labels=["1", "100", "1000"])
        if main:
            axs[0, 0].legend(fontsize=8)
        for extension in ["png", "svg", "pdf"]:
            fig.savefig(figures / f"l02-scaling{suffix}.{extension}", dpi=160)
        plt.close(fig)
    (figures / "l02-scaling-data.json").write_text(json.dumps(main, indent=2) + "\n")
    if capacity:
        fig, axs = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        for ax, platform in zip(axs, labels):
            for mode, style in [("S", "-o"), ("U", "--s")]:
                data = sorted(
                    [
                        r
                        for r in capacity
                        if r["platform"] == platform and r["asset_mode"] == mode
                    ],
                    key=lambda r: r["object_count"],
                )
                points = [r for r in data if r.get("T_frame_s")]
                if points:
                    ax.errorbar(
                        [r["object_count"] for r in points],
                        [r["T_frame_s"]["median"] for r in points],
                        yerr=np.array(
                            [
                                [
                                    r["T_frame_s"]["median"] - r["T_frame_s"]["min"]
                                    for r in points
                                ],
                                [
                                    r["T_frame_s"]["max"] - r["T_frame_s"]["median"]
                                    for r in points
                                ],
                            ]
                        ),
                        fmt=style,
                        color=colors[platform],
                        capsize=3,
                        label=f"{mode}: completed repetitions",
                    )
                for r in data:
                    failed = [
                        x
                        for x in rows
                        if x["run_id"] in r["run_ids"]
                        and x["status"] in ["timeout", "failed"]
                        and not x["gpu_competitor_pids"]
                    ]
                    if any(x["timeout"] and x["T_frame_s"] is None for x in failed):
                        ax.scatter(
                            [r["object_count"]],
                            [180],
                            marker="^",
                            facecolors="none",
                            edgecolors="#a43838",
                        )
                        ax.annotate(
                            f"{mode}: timeout (n={len(failed)})",
                            (r["object_count"], 180),
                            xytext=(0, 6 if mode == "S" else -15),
                            textcoords="offset points",
                            ha="center",
                            fontsize=8,
                        )
                    elif failed:
                        ax.text(
                            r["object_count"],
                            0.95 if mode == "S" else 0.87,
                            f"{mode}: incomplete",
                            transform=ax.get_xaxis_transform(),
                            ha="center",
                            fontsize=8,
                            color="#a43838",
                        )
            ax.axhline(180, color="#a43838", linewidth=0.8, linestyle=":")
            ax.set(
                xscale="log",
                xlabel="Dynamic cube count",
                ylabel="Entry to first frame (s)",
                title=labels[platform].split(" / ")[0],
                ylim=(0, 205),
            )
            ax.set_xticks(
                [2000, 4000, 8000, 16000], labels=["2000", "4000", "8000", "16000"]
            )
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=8, loc="upper left")
        for extension in ["png", "svg", "pdf"]:
            fig.savefig(figures / f"l02-capacity.{extension}", dpi=160)
        plt.close(fig)
        (figures / "l02-capacity-data.json").write_text(
            json.dumps(
                dict(
                    summary=capacity,
                    runs=[r for r in rows if r["purpose"] == "capacity"],
                ),
                indent=2,
            )
            + "\n"
        )
    lines = [
        "# L-02：基础物体加载与资产复用",
        "",
        f"> 实验初稿 · 主测完成 {done}/148 次；其中 {eligible} 次通过内容核验，且计时窗口采样未检测到其他 GPU 计算进程。结果来自同一张 RTX 5090 D v2。",
        "",
        "## 第一章　对比与结论",
        "",
    ]
    ec = grouped.get(("cube", 1000, "S", "H", "embodichain"))
    il = grouped.get(("cube", 1000, "S", "H", "isaaclab"))
    if ec and il and ec["eligible"] == il["eligible"] == 10:
        lines += [
            f"**在本轮共同场景和公开加载路径下，1000 个共享立方体的首帧等待时间为 EmbodiChain {ec['T_frame_s']['median']:.2f} s、Isaac Lab {il['T_frame_s']['median']:.2f} s；EmbodiChain 较慢，同时 RAM 采样峰值中位数低 {100*(1-ec['ram_peak_mib']['median']/il['ram_peak_mib']['median']):.1f}%，GPU 进程采样峰值中位数低 {100*(1-ec['vram_peak_mib']['median']/il['vram_peak_mib']['median']):.1f}%。** 这体现加载等待与资源占用的取舍，不代表物理求解吞吐或其他复杂场景的排名。",
            "",
        ]
    lines += [
        "本项测量从 Python 脚本入口到物理状态可用，以及第一张 RGB 图像可用的完整等待时间，包含引擎导入与启动。两端读取相同单物体 USD：50 mm 立方体或球体、0.1 kg、解析碰撞；零重力，无接触，640×480 固定相机。S 为共享一份源资产，U 为尺寸与颜色各不相同的资产。采用各自原生渲染器，首帧画质不等价；内存也包含各自渲染栈的占用。",
        "",
        "**缓存范围：**C 为部分受控的新进程加载，H 为同配置完整预载后的新进程加载。解析碰撞不需要网格 cooking；OS、通用 shader／kernel 及未控制的原生渲染缓存保留。本轮不能据 C/H 差值声称“完全冷启动加速”。",
        "",
        "以下为 H 组代表结果，数字是独立进程重复的中位数 [最小值, 最大值]；完整 C/H 数据在后两章。",
        "",
        "| 场景 | 平台 | 完成／尝试 | 至首张图像（s） | RAM 峰值（MiB） | GPU 进程峰值（MiB） |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in main:
        if (
            r["cache_mode"] == "H"
            and r["shape"] == "cube"
            and r["asset_mode"] == "S"
            and r["object_count"] in [1, 1000]
        ):
            lines.append(
                f"| {shapes[r['shape']]} {r['object_count']}／{r['asset_mode']} | {labels[r['platform']]} | {r['completed']}/{r['attempts']} | {fmt(r,'T_frame_s')} | {fmt(r,'ram_peak_mib',0)} | {fmt(r,'vram_peak_mib',0)} |"
            )
    lines += [
        "",
        "少量物体时，固定启动开销占主要部分；1000 个物体时，EmbodiChain 的准备阶段主导等待。共享与不同资产在本轮轻量网格中的加载时间接近，重复引用源文件并未显示明显的端到端加速。",
        "",
        "![L-02 热加载数量—时间与内存曲线](../figures/l02-scaling.png)",
        "",
        "*H 为完成同配置预载后的独立进程；误差棒是最小／最大值，三次重复不构成显著性结论。S/U 改变整体几何与材质输入，本项不单独归因某一项缓存。*",
        "",
        "[原生场景视频与截图](../videos/index.md) · [逐次运行与原始数据](data-index.md)",
        "",
    ]
    geometry_ratios = []
    for platform in labels:
        cube = grouped.get(("cube", 1000, "S", "H", platform))
        sphere = grouped.get(("sphere", 1000, "S", "H", platform))
        if cube and sphere and cube["eligible"] and sphere["eligible"]:
            geometry_ratios.append(
                f"{labels[platform]} {sphere['T_frame_s']['median']/cube['T_frame_s']['median']:.3f} 倍"
            )
    if len(geometry_ratios) == 2:
        lines += [
            "1000 个共享球体的三角形总数是立方体的 44 倍，首帧时间比为："
            + "、".join(geometry_ratios)
            + "。这说明本轮轻量网格的等待成本不能只按面数估算；复杂贴图和碰撞网格不在本项覆盖范围内。",
            "",
        ]
    if capacity and not capacity_pending:
        descriptions = []
        for platform in labels:
            tested = [r for r in capacity if r["platform"] == platform]
            counts = sorted({r["object_count"] for r in tested})
            both_complete = [
                n
                for n in counts
                if all(
                    any(
                        r["object_count"] == n
                        and r["asset_mode"] == mode
                        and r["completed"] == r["attempts"] == 3
                        for r in tested
                    )
                    for mode in ["S", "U"]
                )
            ]
            failed = [r for r in tested if r["completed"] < r["attempts"]]
            prefix = f"{labels[platform].split(' / ')[0]}："
            prefix += (
                f"S/U 两组均完整完成 {max(both_complete)} 个物体；"
                if both_complete
                else "主测 1000 个物体已完成；"
            )
            if failed:
                first = min(r["object_count"] for r in failed)
                outcomes = "、".join(
                    f"{r['asset_mode']} {r['completed']}/{r['attempts']}"
                    for r in tested
                    if r["object_count"] == first
                )
                prefix += f"首次出现未完成的规模为 {first}（{outcomes}）"
            descriptions.append(prefix)
        lines += [
            "**容量扩展（每进程 180 s 预算）：**"
            + "；".join(descriptions)
            + "。这些是本轮观察到的运行结果，未测得引擎的绝对容量；失败发生阶段和原始内存记录见后两章。",
            "",
        ]
        interference = sum(
            r["purpose"] == "capacity" and bool(r["gpu_competitor_pids"]) for r in rows
        )
        if interference:
            lines += [
                f"容量组有 {interference} 次尝试检测到其他 GPU 计算进程，已单列；这些记录不用于独占条件下的性能判断。148 次主测均未检测到这类干扰。",
                "",
            ]
    for chapter, platform in [("第二章", "embodichain"), ("第三章", "isaaclab")]:
        lines += [f"## {chapter}　{labels[platform]} 详细报告", ""]
        if platform == "embodichain":
            lines += [
                "使用 EmbodiChain `f179b555`，以及 P-03 保存的原 DexSim 安装基线＋MR !1501 USD parser 修复副本；Newton 1.6.0，Warp 1.17.0。通过 `SimulationManager` 与 `RigidObjectGroupCfg` 集中声明对象，随后 `prepare()`、首步和原生相机读回。未修改求解器源码。",
                "",
                "物理为 DexUni VBD 10 次迭代、睡眠开启、5 ms；图形为 Hybrid、每像素 1 样本、关闭降噪／超分辨率，固定曝光。共享资产指同一个文件被重复引用；不把输入去重当成已实现 GPU 实例化。",
                "",
                "共享组在计时后的只读核验中仍读到每个物体各自的原生材质 ID。该数量反映材质绑定，不直接等于独立显存分配量；源文件复用、内部资源复用和端到端时间收益应分别判断。",
                "",
            ]
        else:
            lines += [
                "使用 Isaac Lab `ae37b028`／Isaac Sim 6.1，GPU PhysX TGS 4/1。通过 `RigidObjectCollectionCfg` 逐物体引用同一组源 USD，保留完整材质与物理属性；不是提前打包完整场景的 USD。图形为原生 RTX Realtime，关闭 AA、DLSS、降噪、自动曝光和帧生成。",
                "",
                "该安装版本的公开 `reset()` 包含两次内部物理初始化调用，成本计入 T_ready。之后执行一个显式 5 ms 步并取得全量状态；没有额外预热物理步后重新开始计时。不同原生渲染器的外观不在本项作质量排名。首帧仅指场景 RGB 已返回，不代表图像已收敛；关闭降噪、每像素一次采样的原片存在噪声。",
                "",
            ]
        lines += (
            ["### 主测结果", ""]
            + table([r for r in main if r["platform"] == platform])
            + ["", "### 容量试探", ""]
        )
        if any(r["platform"] == platform for r in capacity):
            if capacity_pending:
                lines += ["容量批次仍在运行，以下为重建报告时已有的记录。", ""]
            lines += table([r for r in capacity if r["platform"] == platform])
            lines += [""]
            for mode in ["S", "U"]:
                tested = sorted(
                    [
                        r
                        for r in capacity
                        if r["platform"] == platform and r["asset_mode"] == mode
                    ],
                    key=lambda r: r["object_count"],
                )
                complete = [r for r in tested if r["completed"] == r["attempts"] == 3]
                failed = [
                    r
                    for r in tested
                    if any(
                        x["run_id"] in r["run_ids"]
                        and x["status"] in ["timeout", "failed"]
                        for x in rows
                    )
                ]
                if complete:
                    lines += [
                        f"{mode} 组：本轮三次均完成的最高已测规模为 **{complete[-1]['object_count']}**。"
                    ]
                if failed:
                    lines += [
                        f"{mode} 组首次出现未完成记录的规模为 {failed[0]['object_count']}；本组更高规模不再继续。"
                    ]
            failed_rows = [
                r
                for r in rows
                if r["platform"] == platform
                and r["purpose"] in ["prepare", "capacity"]
                and r["object_count"] > 1000
                and r["status"] in ["timeout", "failed"]
            ]
            if failed_rows:
                lines += [
                    "",
                    "| 未完成运行 | 用途／资产 | 最后完成阶段 | 首帧时间（s） | 首帧或中止前 RAM／GPU 峰值（MiB） | 原因 |",
                    "|---|---|---|---:|---:|---|",
                ]
                for r in failed_rows:
                    frame = (
                        f"{r['T_frame_s']:.2f}"
                        if r["T_frame_s"] is not None
                        else "尚未取得"
                    )
                    ram = (
                        f"{r['ram_peak_mib']:.0f}"
                        if r["ram_peak_mib"] is not None
                        else "缺失"
                    )
                    gpu = (
                        f"{r['vram_peak_mib']:.0f}"
                        if r["vram_peak_mib"] is not None
                        else "缺失"
                    )
                    last = {
                        "engine_initialized": "引擎初始化",
                        "declared": "对象／相机声明",
                        "prepared": "最终准备",
                        "ready": "首步状态",
                        "frame": "首帧",
                    }.get(r.get("last_stage"), "入口")
                    reason = (
                        (
                            "首帧已返回，审计或退出超过 180 s"
                            if r["T_frame_s"] is not None
                            else "180 s 内尚未返回首帧"
                        )
                        if r["timeout"]
                        else r.get("failure_reason", "进程异常")
                    )
                    if r["gpu_competitor_pids"]:
                        reason += "；检测到其他 GPU 计算进程"
                    attempt = (
                        "预载"
                        if r["purpose"] == "prepare"
                        else f"第 {r['repeat']+1} 次"
                    )
                    purpose = "预载" if r["purpose"] == "prepare" else "正式"
                    lines.append(
                        f"| [{r['object_count']}／{attempt}](../runs/{r['run_id']}/console.log) | {purpose}／{r['asset_mode']} | {last} | {frame} | {ram}／{gpu} | {reason} |"
                    )
        else:
            lines += ["尚未完成。"]
        lines += [
            "",
            "容量从 2000、4000、8000 至 16000 个立方体逐级试探，S/U 分开。每进程上限 180 s，失败后停止该平台／资产方式的更高规模；另一平台继续。超时只能说明本次等待预算内没有完成，不能称为显存或引擎的绝对上限。预载也超时的规模没有建立完整 H 条件，后续三次作为加载尝试保留，不进入热加载时间比。预载失败、内容核验或退出失败另列于原始索引。",
            "",
        ]
        observations = analysis / "capacity-process-observations.jsonl"
        if platform == "embodichain" and observations.exists():
            obs = json.loads(observations.read_text().splitlines()[0])
            lines += [
                f"一条 2000 物体运行在启动 {obs['elapsed_since_launch_s']:.1f} s 时的只读采样显示：CPU 为 {obs['cpu_percent_1s_one_core_is_100']:.1f}%（100% 为一个逻辑核），GPU 利用率 {obs['gpu_query'].split(',')[1].strip()}，进程 RAM {obs['rss_mib']:.0f} MiB。它提示优先检查 CPU 侧准备工作或同步等待；单次采样没有提供调用栈，尚不能唯一定位原因。原始[进程观察](../analysis/capacity-process-observations.jsonl)保留。",
                "",
            ]
        if platform == "isaaclab":
            smaller = next(
                (
                    r
                    for r in capacity
                    if r["platform"] == platform
                    and r["asset_mode"] == "S"
                    and r["object_count"] == 2000
                    and r["eligible"] == 3
                ),
                None,
            )
            larger = next(
                (
                    r
                    for r in capacity
                    if r["platform"] == platform
                    and r["asset_mode"] == "S"
                    and r["object_count"] == 4000
                    and r["eligible"] == 3
                ),
                None,
            )
            if smaller and larger:
                lines += [
                    f"在相同容量布局规则与机位下，共享物体从 2000 增至 4000，首帧时间中位数从 {smaller['T_frame_s']['median']:.2f} s 增至 {larger['T_frame_s']['median']:.2f} s，增加为 {larger['T_frame_s']['median']/smaller['T_frame_s']['median']:.2f} 倍。本轮这两个点的等待时间增长明显超过数量增幅，不宜按物体数线性外推；没有据此拟合渐近复杂度。",
                    "",
                ]
        stage_data = [
            r
            for r in main
            if r["platform"] == platform
            and r["cache_mode"] == "H"
            and r["shape"] == "cube"
            and r["asset_mode"] == "S"
            and r["eligible"]
        ]
        lines += [
            "### 时间主要花在哪里",
            "",
            "以下为共享立方体 H 组的阶段中位数，各列由逐次差值分别统计，不能把中位数相加当成总时间。两端 prepare 承担的工作不同，仅用于定位各自入口的成本。",
            "",
            "| 数量 | 导入／引擎初始化（s） | 声明对象／相机（s） | 最终准备（s） | 首步／状态读回（s） | 首张 RGB 读回（s） |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
        for r in stage_data:
            values = [
                f"{r[k]['median']:.3f}"
                for k in [
                    "engine_initialized_s",
                    "scene_declaration_s",
                    "scene_preparation_s",
                    "first_step_readback_s",
                    "first_rgb_readback_s",
                ]
            ]
            lines.append(f"| {r['object_count']} | " + " | ".join(values) + " |")
        if platform == "embodichain":
            large = next((r for r in stage_data if r["object_count"] == 1000), None)
            if large:
                lines += [
                    "",
                    f"1000 个共享立方体的最终准备阶段中位数为 {large['scene_preparation_s']['median']:.2f} s，是当前入口中主要的等待来源。[源码核对](../analysis/prepare-api-inspection.json)表明 `prepare()` 涵盖源解析、场景／模型实例化、状态绑定和渲染状态发布。后续优化可先剖析这些边界；本轮没有内部子阶段计时，不能据此归因某个求解器内核。",
                ]
        else:
            large = next((r for r in stage_data if r["object_count"] == 1000), None)
            if large:
                lines += [
                    "",
                    f"1000 个共享立方体 H 组的首帧时间仍在 {large['T_frame_s']['min']:.2f}–{large['T_frame_s']['max']:.2f} s 间波动，全部重复均保留。延迟主要出现在准备阶段；该窗口未检测到其他 GPU 计算进程，但本轮没有进一步隔离 CPU 调度、图形资源与缓存因素，不能给出唯一原因。",
                ]
        reuse = []
        for shape in shapes:
            shared = grouped.get((shape, 1000, "S", "H", platform))
            unique = grouped.get((shape, 1000, "U", "H", platform))
            if shared and unique and shared["eligible"] and unique["eligible"]:
                reuse.append(
                    f"{shapes[shape]} {unique['T_frame_s']['median']/shared['T_frame_s']['median']:.3f}"
                )
        lines += [
            "",
            "1000 物体的 U/S 首帧时间比："
            + "；".join(reuse)
            + "。本比值只描述不同资产相对共享源文件的观测成本；输入几何、颜色同时变化，且重复数有限，不能单独归因网格缓存或材质缓存。",
            "",
        ]
        subset = [
            r
            for r in rows
            if r["platform"] == platform
            and r["purpose"] == "main"
            and r.get("content_complete")
        ]
        if subset:
            lines += [
                "### 内容核验",
                "",
                f"全部对象状态、质量、惯量、网格数量、三角形数、碰撞数量及材质颜色在计时后核对。已完成主测中，初态位置最大残差 {max(r['position_error_max_m'] for r in subset)*1000:.6f} mm，质量最大差 {max(r['mass_error_max_kg'] for r in subset):.3g} kg，惯量最大分量差 {max(r['inertia_error_max_kg_m2'] for r in subset):.3g} kg·m²，visual 尺寸最大差 {max(r['visual_extent_error_max_m'] for r in subset):.3g} m。残差直接报告，不另设人为精度合格线。",
                "",
            ]
        native = next(
            (r for r in media if r["platform"] == platform and r["shape"] == "cube"),
            None,
        )
        if native:
            path = "../runs/" + native["run_id"] + "/native_camera"
            lines += [
                "### 原生运行画面",
                "",
                f"![1000 个共享立方体：{labels[platform]} 原生相机]({path}/frame_0029.png)",
                "",
                f"1000 个共享立方体、相同机位的实际仿真画面。[3 s／1× 原片]({path}/native_1x.mp4)。展示录像配置 8 spp 和平台原生降噪，Isaac 补录另用 DLSS 质量模式；物理参数与主测相同，图中不是测速首帧。球体录像和全部帧、时刻、状态核验见[素材索引](../videos/index.md)。",
                "",
            ]
    if capacity:
        lines += [
            "### 容量曲线",
            "",
            "![容量试探：完成时间与超时记录](../figures/l02-capacity.png)",
            "",
            "圆点／方点仅为已完成且计时条件有效的重复，误差棒是范围；空心三角表示 180 s 预算结束仍未取得首帧，不是加载完成时间。标注次数只统计未检测到其他 GPU 计算进程的尝试。未执行的更高规模没有数据点，完整跳过原因保留在[容量计划](../plans/capacity_v2.json)。",
            "",
        ]
        contaminated = [
            r for r in rows if r["purpose"] == "capacity" and r["gpu_competitor_pids"]
        ]
        if contaminated:
            descriptions = [
                f"{labels[r['platform']]}／{r['object_count']}／{r['asset_mode']}／第 {r['repeat']+1} 次"
                for r in contaminated
            ]
            lines += [
                "容量尝试中，"
                + "、".join(descriptions)
                + "检测到其他 GPU 计算进程。这些失败记录保留，但不作为独占 GPU 条件下的耗时或容量证据；其余重复的状态单独保留。",
                "",
            ]
    lines += [
        "### 共同计时、输入与复现",
        "",
        "硬件为 AMD Ryzen 9 9950X（16 核／32 线程）、约 60.5 GiB 系统内存、RTX 5090 D v2（24455 MiB），驱动 580.159.04；资产位于 ZHITAI Ti600 2TB NVMe 的 ext4 分区。两端 OMP／OpenBLAS 线程上限均为 4，其他线程池保持引擎默认；这不等于整个进程统一限制到 4 核，也不代表各平台最优线程数。完整硬件与版本见[环境快照](../analysis/environment.json)。",
        "",
        "t0 在入口第一段代码中记录，包含参数读取、引擎导入、启动、全部场景对象／相机／灯光声明和首次资源准备。tP 为完成首步、同步 GPU 并读回全部状态；tF 为取得首张包含场景物体的 RGB 图像。截图保存、模型审计和录像在 tF 后。prepare 与首步读回后同步 Warp／CUDA；原生图形队列由相机读回闭合。各平台 prepare 的工作范围不同，阶段差额不能当成相同内部操作的耗时；主比较采用 T_ready 与 T_frame。",
        "",
        f"RAM 为进程树 RSS；显存由 NVML 同时检查计算／图形进程，保留整卡基线和增量作复核。采样请求间隔 20 ms，已测主测／容量记录的最长相邻采样空隙为 {sample_gap_ms:.1f} ms；实际间隔及逐点读数均保存。这里报告采样峰值，可能漏过瞬时尖峰。正常完成记录只截取入口至首帧，退出与审计用量排除；尚未返回首帧的失败记录列到进程终止的观测峰值，不作为已加载场景的内存占用排名。",
        "",
        "[C 组完整数量曲线](../figures/l02-scaling-C.png)与 H 组使用同一指标；原始图表数据另存 JSON。主测输入为 10×10×10 网格，间距 0.1 m。不同资产的缩放为 0.9–1.1，质量固定，惯量随尺寸重算；立方体 12 三角形，球体 528 三角形。容量组使用另一份固定 25×25×26 网格及固定相机，密度相同。两端都有一张隐藏的静态解析地面。全部源文件、布局、配置和代码快照保留。",
        "",
        "[原始 CSV](../analysis/runs.csv) · [逐次 JSON](../analysis/runs.json) · [148 次主测冻结记录](../analysis/main-v2-frozen.json) · [统计定义](../analysis/method.json) · [输入清单](../prepared/assets-v1/manifest.json) · [三表运行摘要](benchmark-summary.md)。每次运行目录含 config、源码快照、launch、阶段日志、退出状态、20 ms 请求间隔的内存记录、first_state、model 和 first_frame。所有预检和失败记录保留，主测只选择 protocol-v2 且满足相同计时条件的记录。",
        "",
        "在项目根目录依次执行 `python -m scripts.benchmark.tech_reports.reporting.l02_review`、`l02_validate`、`l02_media_review`、`l02_report`（后三者沿用同一模块前缀）可离线重建。运行环境使用已有 embodichain Conda；新增仿真入口和完整参数见项目 README。",
        "",
    ]
    (reports / "L-02.md").write_text("\n".join(lines))
    lines = [
        "# L-02 Benchmark Summary",
        "",
        "## Time & Memory",
        "",
        "| 配置 | 计时样本／尝试 | cost_time_ms（至首帧） | cpu_delta_mb | gpu_delta_mb | peak_gpu_mb |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in summary:
        name = f"{r['purpose']} / {r['platform']} / {r['shape']} / {r['object_count']} / {r['asset_mode']} / {r['cache_mode']}"
        ms = f"{1000*r['T_frame_s']['median']:.1f}" if r.get("T_frame_s") else "未取得"
        lines.append(
            f"| {name} | {r['eligible']}/{r['attempts']} | {ms} | {fmt(r,'cpu_delta_mb',0)} | {fmt(r,'gpu_delta_mb',0)} | {fmt(r,'vram_peak_mib',0)} |"
        )
    lines += [
        "",
        "## Success & Other Metrics",
        "",
        "完成定义为首步、有效 RGB、完整对象／碰撞／网格／材质读回并正常退出；没有人为精度 pass/fail。全部预检、准备、超时、失败和运行中记录见数据索引。",
        "",
        "| 配置 | 完成／尝试 | 数量 | 资产方式 | 数据 |",
        "|---|---:|---:|---|---|",
    ]
    for r in summary:
        name = f"{r['purpose']} / {r['platform']} / {r['shape']} / {r['cache_mode']}"
        lines.append(
            f"| {name} | {r['completed']}/{r['attempts']} | {r['object_count']} | {r['asset_mode']} | [逐次数据](../analysis/runs.csv) |"
        )
    lines += [
        "",
        "## Leaderboard",
        "",
        "按同条件保留全部配置，不设综合总分；相同任务全完成时没有成功率名次差异。不能配对、缓存未完全受控或缺少有效重复时注明原因。",
        "",
        "| 场景 | EmbodiChain 完成／尝试 | Isaac 完成／尝试 | T_ready 比（Isaac / EC） | T_frame 比（Isaac / EC） |",
        "|---|---:|---:|---:|---:|",
    ]
    pairs = defaultdict(dict)
    for r in summary:
        pairs[
            tuple(
                r[k]
                for k in [
                    "purpose",
                    "shape",
                    "object_count",
                    "asset_mode",
                    "cache_mode",
                ]
            )
        ][r["platform"]] = r
    for key, pair in sorted(pairs.items()):
        a = pair.get("embodichain", {})
        b = pair.get("isaaclab", {})
        ratios = []
        for metric in ["T_ready_s", "T_frame_s"]:
            ratios.append(
                f"{b[metric]['median']/a[metric]['median']:.3f}"
                if a.get(metric)
                and b.get(metric)
                and a.get("completed") == a.get("attempts")
                and b.get("completed") == b.get("attempts")
                and a.get("eligible") == a.get("attempts")
                and b.get("eligible") == b.get("attempts")
                else "不作直接比值"
            )
        a_count = f"{a['completed']}/{a['attempts']}" if a else "未执行"
        b_count = f"{b['completed']}/{b['attempts']}" if b else "未执行"
        lines.append(
            f"| {' / '.join(map(str,key))} | {a_count} | {b_count} | {ratios[0]} | {ratios[1]} |"
        )
    lines += [
        "",
        "内存单位按 MiB；表头沿用上游字段名。C 只部分控制缓存，不称为完全冷启动。详细配置与限制见 [技术稿](L-02.md)。",
    ]
    (reports / "benchmark-summary.md").write_text("\n".join(lines) + "\n")
    lines = [
        "# L-02 原始数据索引",
        "",
        "保留全部尝试，包括预检、热加载准备、正式主测、容量试探和素材补录。输出中的绝对本机路径仅用于复现，未上传或发布。",
        "",
        "进程被超时终止时，入口写入的 run.json 可能仍显示 running，阶段字段也可能尚未回写；最终状态取 exit.json，已完成时间点从 console.log 恢复。下表和 analysis/runs.json 已按此规则整理，不将这些记录视为仍在运行。主测指标以离线双精度重算为准。",
        "",
        "交付冻结后，[文件哈希清单](../analysis/delivery-manifest.json)记录原始运行、输入、报告、图表、素材和分析源码；[最终核验](../analysis/final-validation.json)记录矩阵、依赖完整性、媒体及文档链接检查。清单是本地证据，不代表公开发布或许可授权。",
        "",
        "| 运行 | 用途／配置 | 退出与内容状态 | 最后阶段／原因 | 原始数据 |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        path = "../runs/" + r["run_id"]
        available = [
            f"[{name}]({path}/{name})"
            for name in [
                "config.json",
                "launch.json",
                "run.json",
                "exit.json",
                "console.log",
                "memory.jsonl",
                "first_state.npy",
                "model.npz",
                "model-audit.json",
                "first_frame.png",
            ]
            if (root / "runs" / r["run_id"] / name).exists()
        ]
        lines.append(
            f"| {r['run_id']} | {r['purpose']}／{r['shape']}／{r['object_count']}／{r['asset_mode']}／{r['cache_mode']} | {r['status']}{'；其他计算进程干扰' if r['gpu_competitor_pids'] else ''} | {r.get('last_stage') or '入口'}／{r.get('failure_reason') or '—'} | {' · '.join(available)} |"
        )
    (reports / "data-index.md").write_text("\n".join(lines) + "\n")
    print(reports / "L-02.md", f"main {done}/148, eligible {eligible}")


if __name__ == "__main__":
    main()
