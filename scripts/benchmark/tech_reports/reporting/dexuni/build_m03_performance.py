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
"""Rebuild the accepted M03 fixed-five performance report without simulation.

Run: .venv/bin/python scripts/benchmark/tech_reports/reporting/dexuni/build_m03_performance.py
"""

from __future__ import annotations
import csv
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
from scripts.benchmark.tech_reports.physics.experiments.dexuni.m03_common import (
    OUTPUT,
    digest,
    write_json,
)
from scripts.benchmark.tech_reports.physics.experiments.dexuni.s04_measurement import (
    file_hashes,
    summarize_steps,
)

__all__ = ["build_report"]
PLATFORMS = ("embodichain", "isaaclab")


def stats(values: list[float]) -> dict[str, float]:
    """Summarize independent run means with sample standard deviation."""
    a = np.asarray(values, dtype=float)
    return dict(
        mean=float(a.mean()), std=float(a.std(ddof=1)), median=float(np.median(a))
    )


def build_report() -> None:
    """Verify all ten raw episodes and report latency separately from final shape."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    measurement = OUTPUT / "measurement"
    protocol = json.loads((measurement / "protocol.json").read_text())
    attempts = json.loads((measurement / "attempts.json").read_text())
    if len(attempts) != 10 or not all(a["performance_valid"] for a in attempts):
        raise ValueError("All ten planned processes must finish before aggregation")
    if file_hashes(OUTPUT / "inputs") != protocol["input_sha256"]:
        raise ValueError("Accepted inputs changed")
    for p, h in protocol["preview_sha256"].items():
        if digest(OUTPUT / p) != h:
            raise ValueError(f"Accepted media changed: {p}")
    results = {}
    plot_data = {}
    for p in PLATFORMS:
        manifests = []
        timings = []
        reads = []
        for i in range(1, 6):
            root = measurement / p
            m = json.loads((root / f"r{i:02d}_manifest.json").read_text())
            if m["status"] != "completed" or not all(m["performance_checks"].values()):
                raise ValueError("Invalid timing episode")
            if m["protocol_sha256"] != digest(measurement / "protocol.json"):
                raise ValueError("Protocol changed")
            with (root / f"r{i:02d}_timing.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            seconds = np.array([float(r["step_s"]) for r in rows])
            read = np.array([float(r["readback_s"]) for r in rows])
            summary = summarize_steps(seconds)
            if not np.isclose(
                summary["cost_time_ms"],
                m["summary"]["cost_time_ms"],
                rtol=0,
                atol=1e-10,
            ):
                raise ValueError("Raw timings differ")
            with np.load(root / f"r{i:02d}_states.npz") as state:
                assert state["particles"].shape == (1651, 4, 961, 3)
                assert all(np.isfinite(state[k]).all() for k in state.files)
                assert np.allclose(
                    state["time"], np.arange(1651) / 60, rtol=0, atol=1e-8
                )
            manifests.append(m)
            timings.append(seconds * 1000)
            reads.append(read * 1000)
        means = [m["summary"]["cost_time_ms"] for m in manifests]
        memory = {
            key: stats([m["memory_call"][key] for m in manifests])
            for key in ("cpu_delta_mb", "gpu_delta_mb", "peak_gpu_mb")
        }
        process_peaks = [
            m["gpu_memory"]["gpu_process_peak_mib"]
            for m in manifests
            if m["gpu_memory"]["gpu_tracking_available"]
        ]
        results[p] = dict(
            latency_ms=stats(means),
            run_means_ms=means,
            run_medians_ms=[m["summary"]["median_ms"] for m in manifests],
            pooled_step_median_ms=float(np.median(np.array(timings)[:, 30:])),
            pooled_step_p95_ms=float(np.percentile(np.array(timings)[:, 30:], 95)),
            readback_ms=stats([m["readback_mean_ms"] for m in manifests]),
            memory=memory,
            process_gpu_sampled_max_mib=(
                stats(process_peaks) if len(process_peaks) == 5 else None
            ),
            initialized_process_gpu_mib=stats(
                [m["gpu_memory"]["gpu_process_initial_mib"] for m in manifests]
            ),
            physical_lift_checks_passed=sum(
                m["observations"]["all_four_clear"] for m in manifests
            ),
            physical_placement_checks_passed=sum(
                m["observations"]["all_four_candidate_placement"] for m in manifests
            ),
            performance_valid=5,
            environment=manifests[0]["environment"],
        )
        plot_data[p + "_step_ms"] = np.array(timings)
        plot_data[p + "_readback_ms"] = np.array(reads)
    a = results["embodichain"]["latency_ms"]["mean"]
    b = results["isaaclab"]["latency_ms"]["mean"]
    result = dict(
        task="M03",
        scope="unified scene/materials/control/time stepping; solver-level optimized strategy: DexUni surface-fast base8+adaptive versus ordinary VBD20",
        interpretation="本组实测：DexUni fast基础8次（保留自适应回退）外步耗时降低72.7%，对应约3.67倍步进吞吐；四层抬升检查均为5/5，放置检查为EC 5/5、Isaac Lab 0/5。因而在本任务中，优化后的求解策略同时取得更低耗时和更好的目标放置结果。这里的效果结论指本任务的放置指标，不代表所有布形指标或其它场景的普遍排序。",
        results=results,
        ec_latency_reduction_percent=(b - a) / b * 100,
        isaac_over_ec_time_ratio=b / a,
        protocol_sha256=digest(measurement / "protocol.json"),
        attempts_sha256=digest(measurement / "attempts.json"),
    )
    write_json(measurement / "results.json", result)
    np.savez_compressed(measurement / "plot_data.npz", **plot_data)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for p in PLATFORMS:
        axes[0].plot(np.arange(1, 6), results[p]["run_means_ms"], "o-", label=p)
        axes[1].plot(
            np.arange(1650)[30:] / 60,
            np.mean(plot_data[p + "_step_ms"][:, 30:], axis=0),
            label=p,
            alpha=0.8,
            lw=0.7,
        )
    axes[0].set(xlabel="Independent process repeat", ylabel="Completed outer step / ms")
    axes[1].set(xlabel="Simulation time / s", ylabel="Five-run mean per step / ms")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.2)
    fig.savefig(measurement / "performance.png", dpi=160)
    plt.close(fig)
    text = "# M03 实际性能：已确认场景，固定参数各5次\n\n"
    text += "[已确认视频](preview/comparison.mp4) · [逐次数据](measurement/results.json) · [耗时曲线](measurement/performance.png) · [绘图原始数据](measurement/plot_data.npz) · [冻结协议](measurement/protocol.json)\n\n"
    text += "同一套27.5 s场景、4×961动态布节点、24命名关节轨迹、60 Hz、10子步、CUDA Graph。机器人为one_way / kinematic。EC使用surface-fast基础8次及preset自适应回退；Isaac Lab使用普通VBD20次。这是统一场景、材料、控制轨迹、时间步和子步配置下的求解器级优化比较；fast基础8次与普通VBD20次是各自求解策略的一部分，单次迭代的计算内容不同。\n\n"
    text += "本组实测：DexUni fast基础8次（保留自适应回退）外步耗时降低72.7%，对应约3.67倍步进吞吐；四层抬升检查均为5/5，放置检查为EC 5/5、Isaac Lab 0/5。因而在本任务中，优化后的求解策略同时取得更低耗时和更好的目标放置结果。这里的效果结论指本任务的放置指标，不代表所有布形指标或其它场景的普遍排序。\n\n"
    text += "Time & Memory\n\n| 平台 | cost_time_ms 均值±样本标准差 | 外步中位数 / P95 ms | 进程显存采样最大值 MiB | cpu_delta_mb | gpu_delta_mb | peak_gpu_mb |\n|---|---|---|---|---|---|---|\n"
    for p in PLATFORMS:
        r = results[p]
        lat = r["latency_ms"]
        mem = r["memory"]
        peak = r["process_gpu_sampled_max_mib"]
        peak_text = f"{peak['mean']:.1f}" if peak else "未测得"
        text += f"| {p} | {lat['mean']:.3f} ± {lat['std']:.3f} | {r['pooled_step_median_ms']:.3f} / {r['pooled_step_p95_ms']:.3f} | {peak_text} | {mem['cpu_delta_mb']['mean']:.3f} | {mem['gpu_delta_mb']['mean']:.3f} | {mem['peak_gpu_mb']['mean']:.3f} |\n"
    text += "\n耗时是独立进程的稳态外步均值再取平均，±为5次均值的样本标准差。每次完整1650步，前30步预热不纳入主指标，后1620步均保留，共8100步/平台。GPU在起止同步，计时包括平台物理外步和末尾同步，不含状态读取、检查、渲染、内存采样或序列化。外步中位数/P95汇总8100个稳态样本，不是进程均值的分位数。\n\n"
    text += "显存主列为nvidia-smi进程显存离散采样最大值的5次平均（本批实际后端），不是瞬时峰值。cpu_delta_mb为单次额外暖态调用保留输出后的RSS增量；gpu_delta_mb、peak_gpu_mb仅覆盖Torch分配器，不能代表Warp/原生总显存。内存调用在1650步之后单独执行，不属于轨迹或时间样本。\n\n"
    text += "Success & Other Metrics\n\n| 平台 | 性能协议检查 | 抬升检查 | 放置候选线 | 单次状态读取均值 ms | success_rate |\n|---|---|---|---|---|---|\n"
    for p in PLATFORMS:
        r = results[p]
        text += f"| {p} | {r['performance_valid']}/5 | {r['physical_lift_checks_passed']}/5 | {r['physical_placement_checks_passed']}/5 | {r['readback_ms']['mean']:.3f} | 不作物理成功率比较 |\n"
    text += "\n物理检查仅描述这5次回放，性能统计不按最终布形或20 mm放置线筛选。有限状态、真实时间、命名关节、TCP、Graph、迭代预算及无相机均独立核验。状态读取包含传回CPU和末尾同步，不含Python记录与验证；未测量有渲染端到端FPS。\n\n"
    text += "Leaderboard（仅按已确认配置的外步耗时；不排名物理质量）\n\n| 顺序 | 平台 | 外步均值 ms | 较慢平台耗时 ÷ 本平台耗时 |\n|---|---|---|---|\n"
    for rank, p in enumerate(
        sorted(PLATFORMS, key=lambda x: results[x]["latency_ms"]["mean"]), 1
    ):
        mean = results[p]["latency_ms"]["mean"]
        text += f"| {rank} | {p} | {mean:.3f} | {max(a,b)/mean:.3f}× |\n"
    env = results["embodichain"]["environment"]
    text += f"\n硬件：{env['gpu']}，驱动{env['driver']}，CUDA {env['torch_cuda']}。逐进程的软件版本、CPU线程、初始化时间及原始1651帧状态见measurement下各平台的r01–r05 manifest/CSV/NPZ。原预览与inputs哈希全部保持不变。仅本批两平台各5次，无DexSim性能数据。\n"
    (OUTPUT / "REPORT.md").write_text(text)
    (OUTPUT / "README.md").write_text(
        "# M03 当前唯一一套\n\n[已确认视频](preview/comparison.mp4) · [性能与场景记录](REPORT.md) · [测量结果](measurement/results.json)\n\n当前场景已获用户确认；两平台各5次测量完成，输入和预览保持不变。\n"
    )
    print(OUTPUT / "REPORT.md", flush=True)


if __name__ == "__main__":
    build_report()
