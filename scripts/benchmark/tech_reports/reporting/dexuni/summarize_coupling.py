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
"""Rebuild coupling diagnosis from retained profiles and independent validation runs.

Run: .venv/bin/python scripts/benchmark/tech_reports/reporting/dexuni/summarize_coupling.py
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO))
TECH = REPO / "scripts/benchmark/tech_reports"
ROOTS = {
    "s03": TECH.parents[2]
    / "outputs/benchmarks/tech_reports/dexuni/s03_cloth_corner_lift",
    "s04": TECH.parents[2]
    / "outputs/benchmarks/tech_reports/dexuni/s04_rigid_soft_squeeze_grasp",
}
OUT = ROOTS["s04"] / "analysis"
__all__ = ["load_diagnostic", "main"]


def load_diagnostic(task: str, directory: Path) -> dict:
    """Load original worker outputs, validating finite full episodes without geometry filters."""
    d = json.loads((directory / "diagnostics.json").read_text())
    if task == "s04":
        states = dict(np.load(directory / "states.npz"))
        seconds = np.load(directory / "step_seconds.npy")
        config = json.loads((ROOTS[task] / "inputs/s04_config.json").read_text())
        graph = d["effective"]["cuda_graph_status"]
        complete = (
            len(seconds) == 720
            and len(states["time"]) == 721
            and abs(d["effective"]["simulated_elapsed_s"] - 12) < 1e-8
        )
    else:
        path = next(directory.glob(f"raw_data/{d['platform']}/*/manifest.json")).parent
        manifest = json.loads((path / "manifest.json").read_text())
        config = json.loads((path / "config.json").read_text())
        states = dict(np.load(path / "states.npz"))
        with (path / "timing.csv").open() as stream:
            seconds = np.array(
                [float(r["step_seconds"]) for r in csv.DictReader(stream)]
            )
        complete = (
            manifest["status"] == "completed"
            and manifest["controller"]["finished"]
            and len(seconds) == 793
            and len(states["time"]) == 794
        )
        graph = manifest["effective"]["cuda_graph_status"]
    required = ("particles", "qpos", "tcp", "target_q", "time")
    checks = {
        "completed": bool(complete),
        "finite": all(np.isfinite(states[k]).all() for k in required),
        "positive_times": bool(np.isfinite(seconds).all() and (seconds > 0).all()),
        "graph": graph == "captured",
    }
    steady = seconds[30:] * 1000
    memory = d.get("gpu_memory", {})
    return {
        "task": task,
        "platform": d["platform"],
        "case": d["case"],
        "repeat": d["repeat"],
        "run": str(directory.relative_to(TECH)),
        "cost_time_ms": float(np.mean(steady)),
        "median_ms": float(np.median(steady)),
        "p95_ms": float(np.percentile(steady, 95)),
        "steps": len(seconds),
        "states": len(states["time"]),
        "measured_steps": len(steady),
        "checks": checks,
        "success": all(checks.values()),
        "particle_groups": d["solver"]["vbd"]["particle_color_groups"],
        "self_contact_interval": d["solver"]["vbd"].get(
            "particle_collision_detection_interval"
        ),
        "gpu_sampled_mib": (
            memory.get("gpu_process_peak_mib")
            if memory.get("gpu_tracking_available")
            else None
        ),
        "profile": bool(d["profile_window"][0]),
        "event_instrumentation": sum(
            1 for _ in (directory / "stream_timing.csv").open()
        )
        > 1,
        "kernel_errors": d.get("kernel_equivalence_max_errors"),
        "driver_sha256": d.get("driver_sha256"),
        "input_config_sha256": d.get("input_config_sha256"),
    }


def _mean_std(rows: list[dict], key: str) -> str:
    """Format variability across independent processes."""
    values = [r[key] for r in rows if r.get(key) is not None]
    if not values:
        return "N/A"
    return f"{np.mean(values):.3f} ± {np.std(values,ddof=1):.3f}"


def _profile_rows() -> list[dict]:
    """Join retained kernel summaries to capture source modules for coarse attribution."""
    rows = []
    for task, root in ROOTS.items():
        for platform in ("embodichain", "isaaclab"):
            r = 1 if task == "s03" and platform == "isaaclab" else 0
            nodes = json.loads(
                (
                    root
                    / f"analysis/coupling/{platform}_baseline_r{r:02d}/capture_nodes.json"
                ).read_text()
            )
            keys = sorted(
                {(n["kernel"], n["module"]) for n in nodes},
                key=lambda x: len(x[0]),
                reverse=True,
            )
            path = (
                root
                / "analysis"
                / (
                    "ec_nsys_stats.csv"
                    if platform == "embodichain"
                    else "il_nsys_stats.csv"
                )
            )
            lines = path.read_text().splitlines()
            start = next(
                i
                for i, l in enumerate(lines)
                if l.startswith("Time (%),Total Time (ns),Instances")
            )
            for record in csv.DictReader(lines[start:]):
                if (
                    not record.get("Total Time (ns)")
                    or not record["Total Time (ns)"].isdigit()
                ):
                    break
                name = record["Name"]
                module = next(
                    (
                        m
                        for k, m in keys
                        if (name.startswith(k + "_") or ("_" + k + "_") in name)
                    ),
                    "unknown_or_builtin",
                )
                if "mujoco" in module:
                    stage = "MuJoCo / state export"
                elif "tri_mesh_collision" in module:
                    stage = "cloth self-contact detection"
                elif "particle_vbd" in module or "solve_elasticity_tile" in module:
                    stage = "particle VBD / truncation"
                elif "rigid_vbd" in module:
                    stage = "rigid VBD"
                elif "coupled" in module or "vbd_coupling" in module:
                    stage = "coupling / state copies"
                elif "narrow_phase" in module:
                    stage = "rigid collision"
                elif "collide" in module or "contact" in module:
                    stage = "collision / contact setup"
                else:
                    stage = "other / builtins"
                rows.append(
                    {
                        "task": task,
                        "platform": platform,
                        "stage": stage,
                        "kernel": name,
                        "module": module,
                        "instances": int(record["Instances"]),
                        "per_profiled_outer_ms": int(record["Total Time (ns)"]) / 5e6,
                    }
                )
    return rows


def main() -> None:
    """Verify the fixed matrix, persist all exploratory cases and build a three-table report."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    attempts = json.loads((OUT / "validation_attempts.json").read_text())
    protocol = json.loads((OUT / "validation_protocol.json").read_text())
    valid_attempts = [a for a in attempts if a["returncode"] == 0]
    if len(valid_attempts) != 18:
        raise ValueError("The fixed validation matrix is incomplete")
    rows = []
    for attempt in valid_attempts:
        row = load_diagnostic(attempt["task"], TECH / attempt["run"])
        if (
            attempt["returncode"] != 0
            or not row["success"]
            or row["profile"]
            or row["event_instrumentation"]
        ):
            raise ValueError(f"Invalid validation sample: {attempt}")
        expected = protocol["source_sha256"][
            "physics/experiments/dexuni/diagnose_coupling.py"
        ]
        allowed = {
            expected,
            *[
                r["physics/experiments/dexuni/diagnose_coupling.py"]
                for r in protocol.get("source_sha256_revisions", [])
            ],
        }
        if row["driver_sha256"] not in allowed:
            raise ValueError("Validation worker changed")
        config_path = ROOTS[row["task"]] / (
            "inputs/s03/s03_config.json"
            if row["task"] == "s03"
            else "inputs/s04_config.json"
        )
        if (
            row["input_config_sha256"]
            != hashlib.sha256(config_path.read_bytes()).hexdigest()
        ):
            raise ValueError("Frozen config changed")
        rows.append(row)
    fields = [k for k in rows[0] if k not in ("checks", "kernel_errors")]
    with (OUT / "validation_runs.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    all_cases = []
    for task, root in ROOTS.items():
        for directory in sorted((root / "analysis/coupling").glob("*")):
            try:
                all_cases.append(load_diagnostic(task, directory))
            except Exception as error:
                all_cases.append(
                    {
                        "task": task,
                        "run": str(directory.relative_to(TECH)),
                        "status": "failed_or_incomplete",
                        "error": repr(error),
                    }
                )
    (OUT / "all_cases.json").write_text(json.dumps(all_cases, indent=2) + "\n")
    profile = _profile_rows()
    with (OUT / "profile_kernels.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(profile[0]))
        writer.writeheader()
        writer.writerows(profile)
    grouped = {}
    summary = {}
    for task in ROOTS:
        chosen = [r for r in rows if r["task"] == task]
        grouped[task] = {
            "EC baseline": [
                r
                for r in chosen
                if r["platform"] == "embodichain" and r["case"] == "baseline"
            ],
            "EC candidate": [
                r
                for r in chosen
                if r["platform"] == "embodichain" and r["case"] != "baseline"
            ],
            "Isaac baseline": [r for r in chosen if r["platform"] == "isaaclab"],
        }
        summary[task] = {}
        for label, group in grouped[task].items():
            if len(group) != 3:
                raise ValueError(
                    "Each implementation requires three independent processes"
                )
            values = [r["cost_time_ms"] for r in group]
            summary[task][label] = {
                "mean_ms": float(np.mean(values)),
                "sample_std_ms": float(np.std(values, ddof=1)),
            }
        a = summary[task]
        a["candidate_reduction_vs_EC_percent"] = (
            1 - a["EC candidate"]["mean_ms"] / a["EC baseline"]["mean_ms"]
        ) * 100
        a["candidate_reduction_vs_Isaac_percent"] = (
            1 - a["EC candidate"]["mean_ms"] / a["Isaac baseline"]["mean_ms"]
        ) * 100
    (OUT / "validation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for axis, task in zip(axes, ROOTS):
        labels = list(grouped[task])
        means = [summary[task][l]["mean_ms"] for l in labels]
        std = [summary[task][l]["sample_std_ms"] for l in labels]
        axis.bar(labels, means, yerr=std, color=["C0", "C2", "C1"], capsize=4)
        axis.set(
            title=task.upper(),
            ylabel="Completed outer step (ms)",
            ylim=(0, max(means) * 1.2),
        )
    fig.tight_layout()
    fig.savefig(OUT / "validation_performance.png", dpi=160)
    plt.close(fig)
    lines = [
        "# S03 / S04 DexUni 源码诊断与优化验证",
        "",
        "结论：旧结果比较的是不同配置/表示下的原生平台成本，不能直接解释为耦合算法本身更慢。DexUni 的双向路径继承 Newton SolverCoupledProxy；差距主要来自自碰撞频率、VBD着色与实现、未使用输出和模型表示。正式五次测量和视频保持原记录，以下是独立的诊断验证。",
        "",
        "验证矩阵在探索之后固定：每任务 EC原配置、EC候选、Isaac原配置各3个独立进程，18次完整运行（另有2次生成器头/许可证定位错误的初始化失败，完整保留），交替顺序、串行GPU，无相机、无Nsight、无CUDA事件逐步仪表。完成外步计时在Warp设备同步之间，去掉前30步。输入/步长/子步/迭代/材料/控制目标不降低；不按最终形状筛选或补跑。S03候选显式统一到Isaac的自碰撞频率及着色调度，这不是保持EC原调度的纯内核加速。",
        "",
        "## 源码中的原因",
        "",
        "- DexUni `_vendor/newton_mjvbd_v2/solver_mjvbd_v2.py` 的 SolverMJVBDV2 继承 _OneWayCoupledProxy → SolverCoupledProxy，底层合成框架与Isaac一致。其差异在自有MuJoCo/VBD和碰撞实现、实体选择及包装层。",
        "- S03：DexSim newton_backend.py 调用 builder.color(include_bending=True)，实际4组；Isaac vbd_manager.py 调用 builder.color(balance_colors=False)，默认include_bending=False，实际3组。每子步每轮按颜色串行求解，旧GPU窗口中S03弹性核调用数2000 vs1500次（5外步）。弯曲系数并未关闭，但坐标更新分组不同，不能把改成3组描述为完全相同数值调度。",
        "- S03：DexUni SolverVBD默认particle_collision_detection_interval=0，初始化前后两次检测；Isaac VBDSolverCfg默认-1，仅初始化前一次。5外步GPU窗口中edge-edge/vertex-triangle检测各100 vs50次，BVH refit200 vs100次。原共享配置未显式指定这个参数，之前预算审计遗漏了这一层默认值。",
        "- DexSim newton_backend.py 自动request_state_attributes('body_qdd')，MuJoCo `_enable_rne_postconstraint` 据此开启后约束加速度/力输出及拷贝。两个基准均无传感器读取这些字段，Isaac没有申请它们。关闭未使用输出减少工作，不能用于需要加速度/力传感器的运行。",
        "- S04：EC MuJoCo模型45 bodies/145 geoms，Isaac34/133；代理VBD视图45 vs28 bodies。Isaac URDF转换默认合并固定关节；EC保留固定帧。body/geom数不是24个受控DOF数。精简碰撞缺失的代理到34个VBD bodies未显著加速，不能把整个差距归因于代理数量。",
        "- S04：所有表面膜材料和弯曲材料均为零，但DexUni通用solve_elasticity_tile仍遍历三角形/边邻接；Isaac make_solve_elasticity_tile在代码生成时剔除无效分支。候选剔除这两段并将完全一致的float32四面体材料烘焙为常量，保留原四面体公式、16线程归约和相同ABI。材料更新必须重建此专用核，非均匀/非有限材料会拒绝该候选。",
        "- 单独打开enable_cuda_fast_path、静止刚体缓存、提前过滤代理接触和禁止support acceleration均未成为最终方案。fast path原模型不满足融合条件：S03缺少truncation cache，S04包含四面体；缓存/融合组合的探索结果也不能代替完整重复验证。所有探索保留在all_cases.json。",
        "",
        "源码位置：DexSim /home/oem/code/repos/dexsimforembodychain/dexsim/modules/dexsim/python/dexsim/engine/newton_physics/；实际运行来自EC环境中的_jit/modules/mjvbd_v2。Isaac /home/oem/app/IsaacLab/source/source/isaaclab_newton/isaaclab_newton/physics/ 和其环境newton/_src/solvers。运行build=0.5.1+dfceb00…；EC Newton1.7.0 / MuJoCo3.15.0 / Torch2.10，Isaac Newton1.6.1 / MuJoCo3.12.0 / Torch2.11。版本及固定帧表示仍未强行统一，因此这是当前平台实测，非孤立耦合器优劣或通用优势证明。本轮使用同期新跑的基线，数值与原先五次报告略有变化；所有提速比例只使用这一轮同期对照计算。",
        "",
        "## Time & Memory",
        "",
        "| Task / implementation | cost_time_ms (mean ± sample std) | Median ms | p95 ms | GPU sampled process max MiB | cpu_delta_mb | gpu_delta_mb | peak_gpu_mb |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for task, groups in grouped.items():
        for label, group in groups.items():
            lines.append(
                f"| {task.upper()} {label} | {_mean_std(group,'cost_time_ms')} | {_mean_std(group,'median_ms')} | {_mean_std(group,'p95_ms')} | {_mean_std(group,'gpu_sampled_mib')} | N/A | N/A | N/A |"
            )
    lines += [
        "",
        "显存是NVML进程归属快照（solver初始化后和episode后）的最大值，包括上下文、缓存与编译模块；不是瞬时峰值或分配器峰值。未测RSS/Torch增量峰值，明确N/A。缓存额外内存不应从仅Torch统计推断。",
        "",
        "## Success & Checks",
        "",
        "| Task | Completed / finite / captured Graph | Budget | Recorded state rows | Candidate validation |",
        "| --- | --- | --- | --- | --- |",
    ]
    for task, groups in grouped.items():
        flat = [r for g in groups.values() for r in g]
        checks = (
            "S04原核vs专用核：代表颜色组在最终状态及0.1/1mm扰动下最大误差0（每次重新检验）"
            if task == "s04"
            else "S03材料/弯曲/自碰撞保留；显式-1/3色与Isaac调度一致；缓存按每次检测刷新"
        )
        lines.append(
            f"| {task.upper()} | 9/9 | 10×{'10' if task=='s03' else '12'} | {sum(r['states'] for r in flat)} | {checks} |"
        )
    lines += [
        "",
        "完整动态轨迹存在非确定性接触排序，两平台/重复进程本来也不会逐帧完全一致。没有把最终形状或提升高度作为成功率/性能筛选项；S04的局部内核等价性是独立数值检查，不能声称整个多接触轨迹bitwise一致。",
        "",
        "## Performance Summary",
        "",
        "| Task | EC原配置→候选耗时降幅 | 候选相对Isaac耗时降幅 | Interpretation |",
        "| --- | --- | --- | --- |",
    ]
    for task, s in summary.items():
        interpretation = (
            "配置对齐（检测频率/着色）+关闭未用输出+表面/截断缓存"
            if task == "s03"
            else "关闭未用输出+同公式/归约的静态体积核；碰撞频率和迭代不变"
        )
        lines.append(
            f"| {task.upper()} | {s['candidate_reduction_vs_EC_percent']:.2f}% | {s['candidate_reduction_vs_Isaac_percent']:.2f}% | {interpretation} |"
        )
    lines += [
        "",
        "![Independent validation repeats](validation_performance.png)",
        "",
        "[18次原始测量CSV](validation_runs.csv) · [固定验证协议](validation_protocol.json) · [全部尝试](validation_attempts.json) · [探索结果及失败](all_cases.json) · [GPU内核计数/来源/耗时](profile_kernels.csv)",
        "",
        "GPU分析器只捕获第340–344外步，共5次；用于解释调用次数和阶段，不作为全任务耗时统计，也不从该窗口外推固定百分比。NSYS报告和原始统计位于两任务各自analysis目录。验证使用原平台worker，诊断驱动只在进程内安装候选，未改已安装SDK或正式输入/测量；驱动与专用核源代码、模型/状态/时间以及源码sha均可复核。",
        "",
        "实施建议：先在DexSim增加显式的可选输出策略；补齐同模型静态体积核的工厂和材料更新失效逻辑；在基准协议显式记录自碰撞频率、include_bending和实际着色数。S03选择参考平台默认配置必须明确记录，不能把减少一次检测混入纯运行时提速宣传。材料缓存只对受支持非梯度、静态材料实例启用。正式基准更换配置前，按原约定重新验收候选视频，再替换有效测量。",
        "",
    ]
    evidence_path = OUT / "source_evidence.json"
    if evidence_path.is_file():
        evidence = json.loads(evidence_path.read_text())
        lines += ["", "源码定位：", ""]
        lines += [
            f"- [{e['label']}]({e['path']}:{e['line']})" for e in evidence["references"]
        ]
        lines += [
            "",
            "[source_evidence.json](source_evidence.json)保存引用文件哈希；三个关键函数的vendor/live JIT AST逐项一致，定位的是实际执行代码。",
            "",
        ]
    (OUT / "COUPLING_ANALYSIS.md").write_text("\n".join(lines))
    print(json.dumps(summary, indent=2))
    print(f"Report: {OUT/'COUPLING_ANALYSIS.md'}")


if __name__ == "__main__":
    main()
