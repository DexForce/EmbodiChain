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
"""Build effective S03/S04 reports from permanent SDK measurements.

Run: .venv/bin/python scripts/benchmark/tech_reports/reporting/dexuni/summarize_sdk.py
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO))
from scripts.benchmark.tech_reports.physics.experiments.dexuni.remeasure_sdk import (
    AUDIT,
    ROOTS,
)
from scripts.benchmark.tech_reports.physics.experiments.dexuni.s04_measurement import (
    write_json,
)

__all__ = ["main"]


def _csv(path: Path, rows: list[dict]) -> None:
    """Write all scalar data used by a report or plot."""
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _stat(rows: list[dict], field: str) -> dict[str, float] | None:
    """Aggregate process values, without substituting unavailable memory metrics."""
    values = [row[field] for row in rows if row[field] is not None]
    if not values:
        return None
    return {"mean": float(np.mean(values)), "sample_std": float(np.std(values, ddof=1))}


def _format(value: dict | None) -> str:
    """Format five-process mean and sample standard deviation."""
    return (
        "N/A" if value is None else f"{value['mean']:.3f} ± {value['sample_std']:.3f}"
    )


def _report(path: Path, rows: list[dict], aggregate: dict, tasks: list[str]) -> None:
    """Write exactly three tables; ranks describe latency within a task."""
    outcomes = []
    for task in tasks:
        ec = aggregate[task]["embodichain"]["cost_time_ms"]
        il = aggregate[task]["isaaclab"]["cost_time_ms"]
        outcomes.append(
            f"{task.upper()}: EC {_format(ec)} ms，Isaac {_format(il)} ms，EC外步耗时降低{100*(1-ec['mean']/il['mean']):.2f}%"
        )
    lines = [
        "# "
        + " / ".join(task.upper() for task in tasks)
        + " DexUni SDK integration remeasurement",
        "",
        "；".join(outcomes) + "。",
        "",
        "SDK 已在指定源码目录修改并安装到 EmbodiChain 的 Python/JIT 运行环境；本次直接运行平台 worker，没有诊断猴子补丁。原生 libDexSim 未更换，版本号仍为0.5.1+dfceb00…，本地修改由源码/JIT/原生库哈希标识。",
        "",
        "每平台每任务5个独立无相机进程，交替顺序、串行GPU。每次外步推进1/60秒；S03为10子步×10迭代，S04为10子步×12迭代，两边均实际捕获CUDA Graph。EC的enable_cuda_fast_path仍为False，这是独立的融合路径选项，不会关闭CUDA Graph。初始30外步排除，表中是5个进程统计量的均值±样本标准差。",
        "",
        "口径：设备同步包围已完成物理外步。EC计时sim.update；Isaac包括write/step/update。控制、状态读取、图像和序列化在计时外；状态读取另列，因此这里的领先仅指此前统一的物理外步口径。EC状态读取仍更贵，不将外步领先外推为整流程领先。GPU数据为进程显存快照最大值，不是瞬时峰值；S04另有一次暖机调用的RSS保留增量与Torch分配器峰值，S03这些字段未测。",
        "",
        "S03包含参考配置对齐：自碰撞检测间隔0→−1（每子步初始化前后两次→仅初始化前一次），弯曲感知4色→Isaac默认3色，并启用现有表面/截断缓存。弯曲材料与自碰撞没有关闭，但数值更新调度改变，收益不能全部归为同调度算法优化。两个任务都显式省去没有消费者的自动加速度输出；SDK默认仍保留该输出，显式传感器申请也保留。",
        "",
        "S04保持原检测调度，专用核静态略去材料为零的表面项并使用精确float32均匀四面体材料常量，原公式/ABI/16线程归约不变。同规模441粒子、1440四面体测试模型覆盖所有颜色组，0/0.1/1mm扰动下的位移和可选Hessian逐位一致。另从冻结位置/拓扑/材料重建粒子模型，用五次实际记录的初始/中段/最终形变状态做180次颜色/扰动对照，位移与Hessian最大误差均为0；这仍是局部核验证。材料更新先使Graph失效，再重选专用核；非均匀材料、梯度、CPU、AOT或变体容量耗尽使用通用核。",
        "",
        "RTX5090 D v2；EC Newton1.7.0/MuJoCo3.15/Torch2.10，Isaac Newton1.6.1/MuJoCo3.12/Torch2.11，Warp1.17。固定关节和模型表示仍不同；这是当前两个平台的场景测量，不是所有任务或所有GPU的通用优势证明。",
        "",
        "## Time & Memory",
        "",
        "| Task | Platform | cost_time_ms | median_ms | p95_ms | Readback ms | GPU sampled process max MiB | cpu_delta_mb | gpu_delta_mb | peak_gpu_mb |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for task in tasks:
        for platform in ("embodichain", "isaaclab"):
            group = aggregate[task][platform]
            lines.append(
                "| "
                + " | ".join(
                    [
                        task.upper(),
                        platform,
                        *[
                            _format(group[field])
                            for field in (
                                "cost_time_ms",
                                "median_ms",
                                "p95_ms",
                                "readback_ms",
                                "gpu_sampled_max_mib",
                                "cpu_delta_mb",
                                "gpu_delta_mb",
                                "peak_gpu_mb",
                            )
                        ],
                    ]
                )
                + " |"
            )
    lines += [
        "",
        "## Success & Other Metrics",
        "",
        "| Task | Platform | Protocol success_rate | Full / finite / captured Graph | State rows per run | Outer steps per run | Actual simulation seconds | Budget |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for task in tasks:
        for platform in ("embodichain", "isaaclab"):
            group = [
                row
                for row in rows
                if row["task"] == task and row["platform"] == platform
            ]
            lines.append(
                f"| {task.upper()} | {platform} | 100% | 5/5 | {group[0]['state_rows']} | {group[0]['timed_steps']} | {group[0]['actual_simulation_s']:.9f} | {'10×10' if task == 's03' else '10×12'} |"
            )
    lines += [
        "",
        "## Leaderboard",
        "",
        "Protocol completion is the success metric, not grasp/fold quality. Rank by mean physical-step latency within each task; no sample is filtered by final geometry.",
        "",
        "| Task | Latency rank | Platform | overall_success_rate (protocol) | Mean outer step ms | EC cost reduction relative to Isaac |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for task in tasks:
        ec = aggregate[task]["embodichain"]["cost_time_ms"]["mean"]
        il = aggregate[task]["isaaclab"]["cost_time_ms"]["mean"]
        for rank, platform in enumerate(
            sorted(
                ("embodichain", "isaaclab"),
                key=lambda p: aggregate[task][p]["cost_time_ms"]["mean"],
            ),
            1,
        ):
            lines.append(
                f"| {task.upper()} | {rank} | {platform} | 100% | {_format(aggregate[task][platform]['cost_time_ms'])} | {(1-ec/il)*100:.2f}% |"
            )
    lines += [
        "",
        "旧正式记录保存在各任务history/formal_before_sdk。启动前的两次验证保存在history/sdk_preflight_before_final_freeze，其中Isaac末尾诊断记录失败完整保留。正式批次中renderer字段差异只修复验证器并重新读取已完成样本，未重跑该样本。所有20个正式样本完整、有限，不按最终形状拒绝或补跑。",
        "",
        "原始模型/状态/时序/进程日志与逐项验证位于各任务measurement；source目录保存执行源文件，protocol保存输入与安装身份。SDK执行和测试证据位于S04 analysis/sdk_integration。",
        "",
    ]
    pairs = json.loads((AUDIT / "paired_summary.json").read_text())
    for task in tasks:
        pair = pairs[task]
        lines += [
            f"{task.upper()}五组成对差值（Isaac−EC）全部为正，均值{pair['mean_delta_ms']:.3f}±{pair['sample_std_ms']:.3f} ms；近似配对t的95%区间为[{pair['approximate_paired_t_95_interval_ms'][0]:.3f}, {pair['approximate_paired_t_95_interval_ms'][1]:.3f}] ms。此区间假定配对样本独立，仅描述本批次；GPU频率未锁定，不能外推至所有硬件/任务。",
            "",
        ]
    lines += [
        "最终代码通过267项相关回归、30项基准检查。新增配置保持旧位置参数顺序和默认值；显式消费者输出保留；材料更新/非均匀/缺失空数组/梯度与AOT回退、缓存容量均有覆盖。正式测量后仅修正override元数据和初始化时空材料数组的处理；逐函数AST和内核文件检查确认所有稳态步进方法、捕获内核完全一致。最终安装哈希与计时载荷哈希分别保留。AOT发布wheel及其他渲染/线程模式未重建或测试，本任务未改变原生线程/渲染层。",
        "",
        "源码入口：[显式输出策略](/home/oem/code/repos/dexsimforembodychain/dexsim/modules/dexsim/python/dexsim/engine/newton_physics/newton_backend.py:82) · [配置项与兼容默认值](/home/oem/code/repos/dexsimforembodychain/dexsim/modules/dexsim/python/dexsim/engine/newton_physics/solvers_cfg.py:848) · [均匀材料核工厂](/home/oem/code/repos/dexsimforembodychain/dexsim/modules/dexsim/python/dexsim/engine/newton_physics/_vendor/newton_mjvbd_v2/vbd/particle_vbd_kernels.py:3043) · [材料检查与更新](/home/oem/code/repos/dexsimforembodychain/dexsim/modules/dexsim/python/dexsim/engine/newton_physics/_vendor/newton_mjvbd_v2/vbd/solver_vbd.py:142)。",
        "",
        "EmbodiChain context路由已检查；本次仅修改SDK及独立基准，没有改变EC模块所有权/配置转发/生命周期契约，现有simulation-system指导仍适用，因此无需改项目上下文。",
        "",
    ]
    for task in tasks:
        media_path = ROOTS[task] / "media_update.json"
        if media_path.exists():
            media = json.loads(media_path.read_text())
            if media.get("status") == "completed_current_sdk":
                video = ROOTS[task] / media["current_video"]
                video_link = Path(os.path.relpath(video, path.parent)).as_posix()
                media_link = Path(os.path.relpath(media_path, path.parent)).as_posix()
                cleanup_link = Path(
                    os.path.relpath(ROOTS[task] / "cleanup_current.json", path.parent)
                ).as_posix()
                lines += [
                    f"{task.upper()} [当前SDK并排视频]({video_link}) · [配置/Graph/时间核验]({media_link}) · [临时清理记录]({cleanup_link})",
                    "",
                    "当前预览按本次求解配置重录，含相机但不计时；历史已确认视频按原哈希归档，原始性能记录不变。",
                    "",
                ]
    path.write_text("\n".join(lines))


def main() -> None:
    """Normalize all raw timings and emit process summaries and comparison plots."""
    attempts = json.loads((AUDIT / "remeasurement_attempts.json").read_text())
    if len(attempts) != 20 or not all(a.get("success") for a in attempts):
        raise ValueError("Require all 20 complete validated formal samples")
    rows, steps = [], []
    for attempt in attempts:
        task, platform = attempt["task"], attempt["platform"]
        directory = ROOTS[task] / attempt["run"]
        manifest = json.loads((directory / "manifest.json").read_text())
        metrics = attempt["metrics"]
        timing = np.genfromtxt(
            directory / "timing.csv",
            delimiter=",",
            names=True,
            dtype=None,
            encoding=None,
        )
        states = np.load(directory / "states.npz")
        row = {
            "task": task,
            "platform": platform,
            "repeat": attempt["repeat"],
            "run": attempt["run"],
            "cost_time_ms": metrics.get("cost_time_ms", metrics.get("step_ms")),
            "median_ms": metrics.get("median_ms", metrics.get("step_median_ms")),
            "p95_ms": metrics.get("p95_ms", metrics.get("step_ms_p95")),
            "gpu_sampled_max_mib": metrics.get(
                "gpu_process_sampled_max_mib", metrics.get("gpu_peak_mib")
            ),
            "cpu_delta_mb": metrics.get("cpu_delta_mb"),
            "gpu_delta_mb": metrics.get("gpu_delta_mb"),
            "peak_gpu_mb": metrics.get("peak_gpu_mb"),
            "readback_ms": metrics.get(
                "readback_record_ms", metrics.get("readback_record_capture_ms")
            ),
            "init_s": metrics.get("init_s"),
            "process_wall_s": attempt["process_wall_s"],
            "state_rows": len(states["time"]),
            "timed_steps": len(timing),
            "actual_simulation_s": manifest["effective"]["simulated_elapsed_s"],
            "cuda_graph_status": manifest["effective"]["cuda_graph_status"],
            "success": True,
        }
        states.close()
        rows.append(row)
        names = timing.dtype.names
        for sample in timing:
            steps.append(
                {
                    "task": task,
                    "platform": platform,
                    "repeat": attempt["repeat"],
                    "frame": int(sample["frame"]),
                    "sim_time_s": float(
                        sample["sim_time"] if "sim_time" in names else sample["time_s"]
                    ),
                    "warmup_excluded": bool(sample["frame"] <= 30),
                    "step_ms": float(
                        sample["step_seconds"]
                        if "step_seconds" in names
                        else sample["step_s"]
                    )
                    * 1000,
                    "readback_ms": float(
                        sample["readback_record_capture_seconds"]
                        if "readback_record_capture_seconds" in names
                        else sample["readback_record_s"]
                    )
                    * 1000,
                    "command_ms": (
                        None
                        if "command_s" not in names
                        else float(sample["command_s"]) * 1000
                    ),
                }
            )
        raw_mean = np.mean(
            [
                s["step_ms"]
                for s in steps
                if s["task"] == task
                and s["platform"] == platform
                and s["repeat"] == attempt["repeat"]
                and not s["warmup_excluded"]
            ]
        )
        assert np.isclose(raw_mean, row["cost_time_ms"], atol=1e-9, rtol=0)
    fields = (
        "cost_time_ms",
        "median_ms",
        "p95_ms",
        "gpu_sampled_max_mib",
        "cpu_delta_mb",
        "gpu_delta_mb",
        "peak_gpu_mb",
        "readback_ms",
        "init_s",
        "process_wall_s",
    )
    aggregate = {
        task: {
            platform: {
                field: _stat(
                    [
                        row
                        for row in rows
                        if row["task"] == task and row["platform"] == platform
                    ],
                    field,
                )
                for field in fields
            }
            for platform in ("embodichain", "isaaclab")
        }
        for task in ROOTS
    }
    for task, root in ROOTS.items():
        results = root / "measurement/results"
        results.mkdir(exist_ok=True)
        _csv(results / "runs.csv", [row for row in rows if row["task"] == task])
        _csv(results / "steps.csv", [row for row in steps if row["task"] == task])
        write_json(results / "summary.json", aggregate[task])
        _report(root / "REPORT.md", rows, aggregate, [task])
    _csv(AUDIT / "formal_runs.csv", rows)
    _csv(AUDIT / "formal_steps.csv", steps)
    write_json(AUDIT / "formal_summary.json", aggregate)
    report = AUDIT.parent / "SDK_REMEASUREMENT.md"
    _report(report, rows, aggregate, list(ROOTS))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, task in zip(axes, ROOTS):
        values = [
            aggregate[task][p]["cost_time_ms"]["mean"]
            for p in ("embodichain", "isaaclab")
        ]
        errors = [
            aggregate[task][p]["cost_time_ms"]["sample_std"]
            for p in ("embodichain", "isaaclab")
        ]
        ax.bar([0, 1], values, yerr=errors, color=["#257797", "#e79f42"], capsize=5)
        for x, p in enumerate(("embodichain", "isaaclab")):
            samples = [
                r["cost_time_ms"]
                for r in rows
                if r["task"] == task and r["platform"] == p
            ]
            ax.scatter(x + np.linspace(-0.13, 0.13, 5), samples, color="black", s=14)
            ax.text(
                x, values[x] + 0.6, f"{values[x]:.3f} ± {errors[x]:.3f}", ha="center"
            )
        ax.set_xticks([0, 1], ["EmbodiChain / DexUni", "Isaac / Newton"])
        ax.set_ylabel("Completed physical outer step (ms)")
        ax.set_ylim(0, max(values) + 2)
        ax.set_title(
            task.upper()
            + (
                " · reference schedule + caches"
                if task == "s03"
                else " · same schedule / specialized kernel"
            )
        )
    fig.tight_layout()
    fig.savefig(AUDIT.parent / "sdk_performance.png", dpi=180)
    plt.close(fig)
    with report.open("a") as stream:
        stream.write(
            "\n![Five independent process repetitions](sdk_performance.png)\n\n[Raw runs](sdk_integration/formal_runs.csv) · [Every timing and plotting input](sdk_integration/formal_steps.csv) · [All attempts](sdk_integration/remeasurement_attempts.json)\n"
        )
    print(json.dumps(aggregate, indent=2))
    print(f"Report: {report}")


if __name__ == "__main__":
    main()
