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
"""Build M06 comparison media and the fixed-five formal report from retained data.

Run: .venv/bin/python scripts/benchmark/tech_reports/reporting/dexuni/build_m06_report.py --preview
Omit --preview to rebuild the completed measurement report without simulation.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO))

from scripts.benchmark.tech_reports.physics.experiments.dexuni.s04_measurement import (
    file_hashes,
    summarize_steps,
    write_json,
)
from scripts.benchmark.tech_reports.physics.platforms.dexuni.dexsim.m06 import (
    DEFAULT_OUTPUT as TASK,
    digest,
)

PLATFORMS = ("embodichain", "isaaclab")
LABELS = {
    "embodichain": "EmbodiChain / DexUni fast 8 + adaptive",
    "isaaclab": "Isaac Lab / Newton VBD 15",
}
__all__ = ["build_preview", "build_report", "main"]


def build_preview() -> None:
    """Compose timestamp-matched perspective/overhead views without stretching them."""
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont, ImageOps

    font = ImageFont.truetype("DejaVuSans.ttf", 21)
    preview = TASK / "preview"
    paths = {p: preview / p / f"{p}.mp4" for p in PLATFORMS}
    readers = {p: imageio.get_reader(str(path)) for p, path in paths.items()}
    times = {
        p: json.loads((preview / p / "frame_times.json").read_text())["actual_time"]
        for p in PLATFORMS
    }
    if any(len(t) != 453 for t in times.values()) or not np.allclose(
        times["embodichain"], times["isaaclab"], rtol=0, atol=1e-8
    ):
        raise ValueError("M06 video timestamps differ")
    if (preview / "comparison.mp4").exists():
        raise FileExistsError("Archive superseded comparison before rebuilding")
    stages = []
    try:
        with imageio.get_writer(
            str(preview / "comparison.mp4"),
            fps=30,
            codec="libx264",
            quality=8,
            macro_block_size=2,
        ) as writer:
            for frame in range(453):
                panel = Image.new("RGB", (1440, 1080))
                for col, p in enumerate(PLATFORMS):
                    source = Image.fromarray(readers[p].get_data(frame))
                    for row, box in enumerate(((0, 0, 960, 720), (960, 0, 1680, 720))):
                        view = ImageOps.contain(source.crop(box), (720, 540))
                        panel.paste(
                            view,
                            (
                                col * 720 + (720 - view.width) // 2,
                                row * 540 + (540 - view.height) // 2,
                            ),
                        )
                    draw = ImageDraw.Draw(panel)
                    draw.rectangle((col * 720, 0, (col + 1) * 720, 36), fill="black")
                    draw.text((col * 720 + 8, 6), LABELS[p], font=font, fill="white")
                    if p == "embodichain" and frame in (59, 179, 239, 299, 359, 452):
                        stages.append(source.resize((840, 360)))
                writer.append_data(np.asarray(panel))
        montage = Image.new("RGB", (1680, 1080))
        for i, source in enumerate(stages):
            montage.paste(source, ((i % 2) * 840, (i // 2) * 360))
        montage.save(preview / "embodichain/stages.jpg")
    finally:
        for reader in readers.values():
            reader.close()
    write_json(
        preview / "comparison_frames.json",
        {
            "frames": 453,
            "fps": 30,
            "size": [1440, 1080],
            "actual_time": times["embodichain"],
            "ec_video_sha256": digest(paths["embodichain"]),
            "isaaclab_video_sha256": digest(paths["isaaclab"]),
            "comparison_sha256": digest(preview / "comparison.mp4"),
            "solver_policies": LABELS,
            "human_video_approved": False,
            "measurement_configuration_authorized": True,
        },
    )
    ec = json.loads((preview / "embodichain/manifest.json").read_text())
    write_json(
        preview / "embodichain/VALIDATION.json",
        {
            "status": ec["status"],
            "full_trajectory": ec["full_trajectory"],
            "effective": ec["effective"],
            "observations": ec["observations"],
            "max_tcp_error_m": ec["max_sampled_tcp_error_m"],
            "video_frames": ec["video_frames"],
            "source_sha256": ec["source_sha256"],
            "video_sha256": digest(paths["embodichain"]),
            "human_video_approved": False,
        },
    )
    print(f"M06 current comparison ready: {preview / 'comparison.mp4'}", flush=True)


def _stat(values: list[float]) -> dict[str, float]:
    """Aggregate independent episode measurements with sample standard deviation."""
    a = np.asarray(values, dtype=float)
    return {
        "mean": float(a.mean()),
        "sample_std": float(a.std(ddof=1)),
        "min": float(a.min()),
        "max": float(a.max()),
    }


def build_report() -> None:
    """Retain every complete sample and report task success separately from latency."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    measurement = TASK / "measurement"
    attempts = json.loads((measurement / "attempts.json").read_text())
    protocol = json.loads((measurement / "protocol.json").read_text())
    replaced = [a for a in attempts if a.get("retest", False)]
    replacement_note = (
        "EmbodiChain 第 5 次已按用户要求重测替换，原本地记录已移除；其他九份不变。"
        "当前保存样本的检查通过数不用于估计未经替换的首轮成功率。"
        if replaced
        else "未按最终衣服形态筛选或补跑。"
    )
    if len(attempts) != 10 or not all(a["performance_valid"] for a in attempts):
        raise ValueError("The full fixed-five batch must be complete before ranking")
    if file_hashes(TASK / "inputs") != protocol["input_sha256"]:
        raise ValueError("Frozen inputs changed")
    result_dir = measurement / "results"
    result_dir.mkdir(exist_ok=True)
    summaries: dict[str, Any] = {}
    rows, all_steps, curves = [], [], []
    for attempt in attempts:
        run = TASK / attempt["run"]
        manifest = json.loads((run / "manifest.json").read_text())
        if manifest["protocol_sha256"] != digest(measurement / "protocol.json"):
            raise ValueError("Recorded protocol identity differs")
        with (run / "timing.csv").open() as stream:
            timings = list(csv.DictReader(stream))
        seconds = np.asarray([float(t["step_s"]) for t in timings])
        metrics = summarize_steps(seconds)
        if metrics != manifest["summary"]:
            raise ValueError("Raw timings and recorded summary differ")
        for row in timings:
            if (row["steady"] == "True") != (int(row["frame"]) > 30):
                raise ValueError("Warmup membership changed")
            all_steps.append(
                dict(platform=attempt["platform"], repeat=attempt["repeat"], **row)
            )
        gpu = manifest["gpu_memory"]
        memory = (
            gpu["gpu_process_peak_mib"]
            if gpu["gpu_tracking_available"] and gpu["gpu_process_peak_mib"] > 0
            else None
        )
        observation = manifest["observations"]
        row = dict(
            platform=attempt["platform"],
            repeat=attempt["repeat"],
            mean_ms=metrics["cost_time_ms"],
            median_ms=metrics["median_ms"],
            p95_ms=metrics["p95_ms"],
            readback_ms=float(
                np.mean([float(t["readback_record_s"]) for t in timings[30:]]) * 1000
            ),
            process_gpu_sampled_max_mib=memory,
            init_s=manifest["init_s"],
            serialization_s=manifest["serialization_s"],
            task_success=manifest["task_success"],
            lift_m=observation["left_sleeve_peak_lift_m"],
            rightward_motion_m=observation["left_sleeve_final_rightward_motion_m"],
            max_tcp_error_m=manifest["max_tcp_error_m"],
            graph=manifest["effective"]["cuda_graph_status"],
        )
        rows.append(row)
        with np.load(run / "states.npz") as states:
            with np.load(TASK / "inputs/cloth_mesh.npz") as mesh:
                left = states["particles"][:, mesh["left_ids"]].mean(1)
            curves.append(
                (attempt["platform"], attempt["repeat"], states["time"].copy(), left)
            )
    for name, values in (("episodes.csv", rows), ("steps.csv", all_steps)):
        with (result_dir / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    for p in PLATFORMS:
        selected = [r for r in rows if r["platform"] == p]
        if len(selected) != 5:
            raise ValueError("Expected exactly five unfiltered episodes per platform")
        summaries[p] = {
            name: _stat([r[name] for r in selected])
            for name in (
                "mean_ms",
                "median_ms",
                "p95_ms",
                "readback_ms",
                "init_s",
                "serialization_s",
                "lift_m",
                "rightward_motion_m",
            )
        }
        memory = [r["process_gpu_sampled_max_mib"] for r in selected]
        summaries[p]["process_gpu_sampled_max_mib"] = (
            _stat(memory) if all(v is not None for v in memory) else None
        )
        summaries[p].update(
            repeats=5,
            task_successes=sum(r["task_success"] for r in selected),
            finite_complete_episodes=5,
        )
    ratio = (
        summaries["isaaclab"]["mean_ms"]["mean"]
        / summaries["embodichain"]["mean_ms"]["mean"]
    )
    summary = {
        "task": "M06",
        "platforms": summaries,
        "isaac_to_ec_latency_ratio": ratio,
        "statistics": "mean and sample standard deviation of five retained independent episode means; replacement status disclosed separately",
        "protocol_sha256": digest(measurement / "protocol.json"),
    }
    summary["replacement_note"] = replacement_note
    summary["requested_replacements"] = [
        {
            "platform": a["platform"],
            "repeat": a["repeat"],
            "authorization": a["authorization"],
        }
        for a in replaced
    ]
    write_json(result_dir / "summary.json", summary)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    curve_rows = []
    colors = {"embodichain": "tab:blue", "isaaclab": "tab:orange"}
    for p, repeat, times, left in curves:
        for axis, component in zip(axes, (0, 2)):
            axis.plot(
                times,
                left[:, component],
                color=colors[p],
                alpha=0.5,
                label=p if repeat == 1 else None,
            )
        curve_rows.extend(
            {
                "platform": p,
                "repeat": repeat,
                "actual_time_s": float(t),
                "left_x_m": float(node[0]),
                "left_z_m": float(node[2]),
            }
            for t, node in zip(times, left)
        )
    for axis, title in zip(axes, ("Left sleeve x", "Left sleeve z")):
        axis.set(
            xlabel="Actual simulation time (s)", ylabel="Position (m)", title=title
        )
        axis.grid(alpha=0.25)
        axis.legend()
    fig.savefig(result_dir / "sleeve_curves.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    fig.subplots_adjust(bottom=0.24, top=0.82, wspace=0.3)
    for col, p in enumerate(PLATFORMS):
        selected = sorted(
            (row for row in rows if row["platform"] == p),
            key=lambda row: row["repeat"],
        )
        values = [row["mean_ms"] for row in selected]
        stats = summaries[p]["mean_ms"]
        axes[0].bar(
            col,
            stats["mean"],
            yerr=stats["sample_std"],
            color=colors[p],
            alpha=0.35,
            capsize=6,
        )
        axes[0].scatter(
            col + np.linspace(-0.12, 0.12, 5),
            values,
            color=colors[p],
            s=30,
            zorder=3,
        )
        axes[0].text(
            col,
            stats["mean"] + stats["sample_std"] + 1.8,
            f"{stats['mean']:.3f} ± {stats['sample_std']:.3f}",
            ha="center",
            fontsize=10,
        )
        axes[1].plot(
            [row["repeat"] for row in selected],
            values,
            marker="o",
            color=colors[p],
            label=p,
        )
    for attempt in replaced:
        row = next(
            row
            for row in rows
            if row["platform"] == attempt["platform"]
            and row["repeat"] == attempt["repeat"]
        )
        axes[1].scatter(
            row["repeat"],
            row["mean_ms"],
            marker="*",
            s=170,
            color=colors[row["platform"]],
            edgecolor="black",
            zorder=4,
        )
        axes[1].annotate(
            "EC r05 retest",
            (row["repeat"], row["mean_ms"]),
            xytext=(-72, -27),
            textcoords="offset points",
            fontsize=9,
        )
    axes[0].set(
        xticks=[0, 1],
        xticklabels=["EmbodiChain\nDexUni fast", "Isaac Lab\nNewton VBD"],
        title="Mean ± sample SD, with individual runs",
        ylabel="Completed outer-step latency (ms)",
    )
    axes[1].set(
        xticks=range(1, 6),
        xlabel="Retained repeat",
        title="Each retained run",
        ylabel="Completed outer-step latency (ms)",
    )
    axes[1].legend(loc="upper left")
    for axis in axes:
        axis.set_ylim(0, max(row["mean_ms"] for row in rows) * 1.22)
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
    fig.suptitle("M06 completed outer-step latency — lower is better", fontsize=13)
    note = "EC: surface-fast, 8 base sweeps + adaptive fallback; Isaac Lab: Newton VBD, 15 sweeps."
    note += "\nFive retained independent runs per platform; error bars are sample SD."
    if replaced:
        note += " EC r05 was retested and replaced on request."
    fig.text(0.06, 0.04, note, fontsize=9)
    fig.savefig(result_dir / "performance_comparison.png", dpi=180)
    plt.close(fig)
    with (result_dir / "sleeve_curves.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(curve_rows[0]))
        writer.writeheader()
        writer.writerows(curve_rows)
    text = "# M06 Franka T 恤单次折叠：正式性能\n\n同一冻结网格、材料、关节轨迹与 1/60 s 外步、10 子步；两边 CUDA Graph 开启。EmbodiChain 使用 DexUni surface-fast：8 次基础迭代、Jacobi/Chebyshev/多层校正、缓存、默认 5 微米/子步位移阈值及最多 20 次自适应回退；Isaac Lab 使用原生 Newton VBD 15 次迭代。该组属于统一场景、材料、控制轨迹及时间步配置下的求解器级优化比较；fast基础8次、自适应回退与普通VBD15次是各自求解策略的组成部分，一次迭代的计算内容不能等价换算。\n\n每平台当前保留五份独立无相机运行，原矩阵顺序交替；每次 906 外步、15.1 s，前 30 步预热，876 步进入稳态。以下 ± 为五次独立运行的样本标准差。GPU 时钟未锁定，原矩阵按冻结顺序执行，保存样本的替换情况单独说明。\n\n## Time & Memory\n\n| 平台 | 外步均值 ± SD (ms) | 每次 p95 的均值 (ms) | 状态读回 (ms) | 进程显存采样最大值 (MiB) | 初始化 (s) | 保存 (s) |\n| --- | --- | --- | --- | --- | --- | --- |\n"
    for p in PLATFORMS:
        s = summaries[p]
        m = s["mean_ms"]
        memory = s["process_gpu_sampled_max_mib"]
        text += f"| {LABELS[p]} | {m['mean']:.3f} ± {m['sample_std']:.3f} | {s['p95_ms']['mean']:.3f} | {s['readback_ms']['mean']:.3f} | {memory['mean'] if memory else '不可用'} | {s['init_s']['mean']:.3f} | {s['serialization_s']['mean']:.3f} |\n"
    text += "\n外步计时等待 Warp GPU 工作完成，包含平台推进及图内控制/FK；状态读回、物理检查、内存采样和保存位于计时之外。显存由进程追踪器采样；本批次因 pynvml 不可用，实际采用 nvidia-smi 回退，不能当作瞬态峰值；另有每次一条独立未计时的 warmed memory call，Torch 分配器峰值仅覆盖 Torch。\n\n## Success & Other Metrics\n\n| 平台 | 完整有限 / 尝试 | 当前保存样本折叠检查通过 | 袖口抬升均值 (mm) | 向右位移均值 (mm) | Graph |\n| --- | --- | --- | --- | --- | --- |\n"
    for p in PLATFORMS:
        s = summaries[p]
        text += f"| {LABELS[p]} | 5/5 | {s['task_successes']}/5 | {s['lift_m']['mean']*1000:.1f} | {s['rightward_motion_m']['mean']*1000:.1f} | 5/5 captured |\n"
    text += "\n折叠检查沿用场景既有条件：TCP 误差 ≤0.5 mm、袖口抬升 ≥50 mm、向右位移 ≥150 mm。折叠检查和性能有效性分别统计，用户指定的样本替换见下文；这些指标不证明精确对折、无自相交或两平台逐点等价。\n\n## Leaderboard\n\n| 排序（通过率，再按耗时） | 平台 | 当前保存样本折叠检查通过 | 外步均值 (ms) |\n| --- | --- | --- | --- |\n"
    order = sorted(
        PLATFORMS,
        key=lambda p: (
            -summaries[p]["task_successes"],
            summaries[p]["mean_ms"]["mean"],
        ),
    )
    for rank, p in enumerate(order, 1):
        text += f"| {rank} | {LABELS[p]} | {summaries[p]['task_successes']}/5 | {summaries[p]['mean_ms']['mean']:.3f} |\n"
    text += f"\n本组 Isaac / EC 外步耗时比为 {ratio:.3f}。此比值反映统一任务配置下DexUni surface-fast相对普通VBD的求解器级策略优化收益。当前保存样本两平台均通过既有折叠检查，效果以这些检查及原始形态指标为准；步进耗时比不代表带渲染与数据保存的端到端吞吐。\n\n[当前对照视频](preview/comparison.mp4) · [原始尝试](measurement/attempts.json) · [协议与源码哈希](measurement/protocol.json) · [逐次统计](measurement/results/episodes.csv) · [全部外步计时](measurement/results/steps.csv) · [汇总](measurement/results/summary.json) · [袖口曲线](measurement/results/sleeve_curves.png) · [实现记录](README.md)。当前视频复现本次配置，未单独标为人工观看批准。\n"
    environments = {}
    for p in PLATFORMS:
        run = TASK / next(a["run"] for a in attempts if a["platform"] == p)
        environment = json.loads((run / "manifest.json").read_text())["environment"]
        environments[p] = {
            "python": environment["python"],
            "versions": environment["versions"],
            "gpu": environment["gpu"],
            "driver": environment["driver"],
        }
    ec_versions = environments["embodichain"]["versions"]
    il_versions = environments["isaaclab"]["versions"]
    text += "\n" + replacement_note + "\n"
    text += "\n[性能对比图](measurement/results/performance_comparison.png)：左侧为五份保存样本的均值、样本标准差及单次点，右侧为逐次耗时；EC 第 5 次的重测替换已标注。绘图数据见 episodes.csv。\n"
    text += f"\n软件版本按各进程记录的发行包元数据：EC Newton {ec_versions['newton']}、DexSim {ec_versions['dexsim']}；Isaac Lab Newton {il_versions['newton']}；两边 Warp {ec_versions['warp-lang']}，GPU {environments['embodichain']['gpu']}。Newton 版本不同，本组未做环境升级或版本统一。\n"
    failed = [a for a in attempts if not a["task_success"]]
    if failed:
        text += "\n未通过折叠检查的完整样本："
        for a in failed:
            manifest = json.loads((TASK / a["run"] / "manifest.json").read_text())
            distance = manifest["observations"]["left_sleeve_final_rightward_motion_m"]
            text += f"{a['platform']} r{a['repeat']:02d}，袖口向右 {distance*100:.2f} cm，检查 {manifest['task_checks']}。"
        text += "速度收益成立，但该版不能视为已达到相同折叠稳定性。\n"
    (TASK / "REPORT.md").write_text(text)
    write_json(result_dir / "environments.json", environments)
    write_json(
        result_dir / "report_source.json",
        {
            "path": str(Path(__file__)),
            "sha256": digest(Path(__file__)),
            "frozen_measurement_report_source_sha256": protocol["source_sha256"][
                str(Path(__file__))
            ],
            "post_measurement_change": "postprocessing discloses actual memory backend, dependency versions and the user-requested r05 replacement; timing/physics sources unchanged; the other nine raw runs preserved",
        },
    )
    raw = {}
    for attempt in attempts:
        run = TASK / attempt["run"]
        raw.update(
            {
                str((run / name).relative_to(TASK)): sha
                for name, sha in file_hashes(run).items()
            }
        )
        raw[attempt["log"]] = digest(TASK / attempt["log"])
    write_json(measurement / "raw_hashes.json", raw)
    print(f"M06 formal report ready: {TASK / 'REPORT.md'}", flush=True)


def main() -> None:
    """Build preview media or rebuild existing formal results without simulation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    build_preview() if args.preview else build_report()


if __name__ == "__main__":
    main()
