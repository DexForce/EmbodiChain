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

"""Review the separately recorded L-02 sleep-off capacity diagnostic.

Run: python -m scripts.benchmark.tech_reports.reporting.l02_sleep_review
"""

from __future__ import annotations

import os

from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

from .l02_multi_report import table

__all__ = ["main"]
ROOT = Path(
    os.environ.get("SIM_REPORT_ROOT", Path(__file__).resolve().parents[4])
).resolve()
BASE = ROOT / "outputs/benchmarks/L-02/sleep-off-v1"
REFERENCE = ROOT / "outputs/benchmarks/L-02/multi-arena-v1"


def main() -> None:
    """Recompute diagnostics, verify raw evidence and write a three-table report."""
    plans = [
        job
        for p in sorted((BASE / "plans").glob("*.json"))
        for job in json.loads(p.read_text())
    ]
    assert plans and all(j["status"] != "pending" for j in plans)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.benchmark.tech_reports.reporting.l02_multi_report",
            "--partial",
            "--input-root",
            str(BASE),
        ],
        cwd=ROOT,
        check=True,
    )
    analysis = BASE / "analysis"
    rows = json.loads((analysis / "runs.json").read_text())
    summary = json.loads((analysis / "summary.json").read_text())
    checks = json.loads((analysis / "content-validation.json").read_text())
    references = json.loads((REFERENCE / "analysis/summary.json").read_text())
    refs = {
        (r["platform"], r["num_envs"]): r
        for r in references
        if r["shape"] == "cube" and r["cache"] == "H"
    }
    for row in rows:
        folder = BASE / "runs" / row["run_id"]
        cfg = json.loads((folder / "config.json").read_text())
        assert cfg["rigid_enable_sleep"] is False and not cfg["record"]
        assert cfg["platform"] == "embodichain" and cfg["shape"] == "cube"
        if row["status"] == "completed":
            audit = json.loads((folder / "model-audit.json").read_text())
            assert audit["rigid_enable_sleep"] is False
            assert audit["body_count"] == cfg["num_envs"]
    for record in json.loads((analysis / "sleep-construction-source.json").read_text()):
        assert (
            hashlib.sha256(Path(record["source"]).read_bytes()).hexdigest()
            == record["sha256"]
        )
    baseline = json.loads((REFERENCE / "analysis/evidence-manifest.json").read_text())
    for record in baseline:
        assert (
            hashlib.sha256((ROOT / record["path"]).read_bytes()).hexdigest()
            == record["sha256"]
        )
    validation = {
        "attempts": len(rows),
        "statuses": dict(Counter(r["status"] for r in rows)),
        "plan_statuses": dict(Counter(j["status"] for j in plans)),
        "audited_completed_runs": len(checks),
        "baseline_raw_files_verified": len(baseline),
        "baseline_mismatches": 0,
        "solver_sources_unchanged": True,
        "position_error_max_m": max(c["local_position_error_max_m"] for c in checks),
        "inertia_error_max_kg_m2": max(c["inertia_error_max_kg_m2"] for c in checks),
        "contended_run_ids": [r["run_id"] for r in rows if r["foreign_compute_pids"]],
    }
    (analysis / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    reports = BASE / "reports"
    reports.mkdir(exist_ok=True)
    completed = [r for r in rows if r["status"] == "completed"]
    largest = max(r["num_envs"] for r in completed)
    largest_repeats = sum(
        r["num_envs"] == largest and r["purpose"] == "capacity" for r in completed
    )
    capacity_counts = sorted(
        {r["num_envs"] for r in rows if r["purpose"] == "capacity"}
    )
    text = f"""# L-02 补充：关闭睡眠后的多 arena 加载

2026-10-27 · 独立诊断 · 每 arena 一个立方体 · 主测仍保留睡眠开启

**关闭睡眠后，本轮已完成 {largest} 个 arena 的加载、首步和全部状态读回，其中独立容量重复 {largest_repeats} 次。** 因此此前 2896／2897 的边界来自睡眠模块，而非 DexUni 多 arena 的普遍数量上限。这里的最大已测规模不是绝对上限；各次结果与未运行项均保留。

原因为 `rigid_enable_sleep=False` 时，VBD 不构造 `RigidBodySleep`，不分配全局 `int32[n,n]` 邻接表，也不执行其 32 MiB 预算检查。[实际运行源码片段与哈希](../analysis/sleep-construction-source.json)。只修改实验配置，未修改求解器源码，也未扩大预算。
"""
    incomplete = [r for r in rows if r["status"] != "completed"]
    if incomplete:
        text += (
            "\n**仍未完成的规模：**"
            + "；".join(
                f"{n} arena 共 {len(group)} 次尝试，最后阶段为 {','.join(sorted({r['last_stage'] for r in group}))}，状态为 {','.join(sorted({r['status'] for r in group}))}"
                for n in sorted({r["num_envs"] for r in incomplete})
                if (group := [r for r in incomplete if r["num_envs"] == n])
            )
            + "。单进程预算为 180 s（含计时后核验及退出）；本轮共享 GPU，超时不能解释为固定数量上限或与 Isaac Lab 的性能差距，需独占资源后另测。原始日志和超时状态保留。\n"
        )
    text += """

## 1. Time & Memory

以下为 H 组新进程容量测试，原计划每个规模先预载、再重复 3 次。16000 的预载超时后主动停止后续扫描：一条已启动重复取消，另两条未运行，不当作引擎失败。单位为秒和 MiB。计时从 Python 入口到全部环境首步完成并读回 CPU，含导入和初始化，无相机、无渲染。保留 OS／kernel 缓存，不称磁盘冷启动。内存为入口至就绪的外部 20 ms 进程采样峰值；没有测量保留内存增量。本次以 `--allow-busy` 做功能诊断，存在其他 GPU 计算进程的运行只作加载可行性证据，不进入性能统计。原始计时和采样仍全部保留；未独占 GPU 的情况下不报告速度比或内存优势。

"""
    text += table(
        [
            "num_envs",
            "纳入性能统计次数",
            "就绪中位数 [最小,最大] / s",
            "cost_time_ms",
            "RAM 峰值 / MiB",
            "peak_gpu_mb (MiB)",
            "cpu_delta_mb / gpu_delta_mb",
        ],
        [
            [
                s["num_envs"],
                s["repeats"],
                f"{s['ready_s']:.3f} [{s['ready_s_range'][0]:.3f}, {s['ready_s_range'][1]:.3f}]",
                f"{s['ready_s'] * 1000:.1f}",
                f"{s['rss_peak_mib']:.0f}",
                f"{s['vram_peak_mib']:.0f}",
                "未测量 / 未测量",
            ]
            for s in summary
        ]
        + [
            [
                n,
                0,
                "未统计：无独占 GPU 完成样本",
                "不适用",
                "不作性能统计",
                "不作性能统计",
                "未测量 / 未测量",
            ]
            for n in capacity_counts
            if not any(s["num_envs"] == n for s in summary)
        ],
    )
    text += "\n\n## 2. Success & Other Metrics\n\n这里的完成表示真实执行到首步并通过全部环境模型核验，不是人为精度达标线。负载为零重力、50 mm／0.1 kg 立方体、dt=5 ms、VBD 10 次迭代；它验证加载能力，不代表接触稳定性或长期仿真吞吐。\n\n"
    counts = sorted({r["num_envs"] for r in rows})
    text += table(
        [
            "num_envs",
            "用途",
            "完成 / 尝试",
            "success_rate（加载完成）",
            "实际 sleep",
            "最大位置残差 / m",
            "异常",
        ],
        [
            [
                n,
                purpose,
                f"{sum(r['status'] == 'completed' for r in group)}/{len(group)}",
                (
                    "不适用（主动取消）"
                    if all(r["status"] == "cancelled" for r in group)
                    else f"{sum(r['status'] == 'completed' for r in group) / len(group):.1%}"
                ),
                "关闭",
                (
                    f"{max(values):.3g}"
                    if (
                        values := [
                            r["position_error_max_m"]
                            for r in group
                            if r["position_error_max_m"] is not None
                        ]
                    )
                    else "未就绪"
                ),
                "; ".join(
                    sorted(
                        {
                            (
                                "主动停止，非引擎失败"
                                if r["status"] == "cancelled"
                                else r["failure_reason"]
                                or f"{r['status']} at {r['last_stage']}"
                            )
                            for r in group
                            if r["status"] != "completed"
                        }
                    )
                )
                or "无",
            ]
            for n in counts
            for purpose in sorted({r["purpose"] for r in rows if r["num_envs"] == n})
            if (
                group := [
                    r for r in rows if r["num_envs"] == n and r["purpose"] == purpose
                ]
            )
        ],
    )
    text += "\n\n## 3. Leaderboard：容量条件分列，不设综合名次\n\n旧主测引用 [睡眠开启的完整报告](../../multi-arena-v1/reports/L-02.md)。Isaac Lab 保留原 PhysX 配置，本次没有关闭其睡眠重测；下表只对照各自配置下的加载完成记录，不作耗时排名。\n\n"
    text += table(
        [
            "num_envs",
            "EmbodiChain sleep 关",
            "EmbodiChain sleep 开",
            "Isaac Lab 原配置",
        ],
        [
            [
                n,
                (
                    "预载超时；后续主动停止"
                    if all(r["status"] == "cancelled" for r in group)
                    else f"{sum(r['status'] == 'completed' for r in group)}/{len(group)} 完成"
                ),
                (
                    f"{refs[('embodichain', n)]['repeats']} 次均完成"
                    if ("embodichain", n) in refs
                    else "3 次均报睡眠预算错误" if n == 4000 else "未运行"
                ),
                (
                    f"{refs[('isaaclab', n)]['repeats']} 次均完成"
                    if ("isaaclab", n) in refs
                    else "未测"
                ),
            ]
            for n in capacity_counts
            if (
                group := [
                    r for r in rows if r["num_envs"] == n and r["purpose"] == "capacity"
                ]
            )
        ],
    )
    text += f"""

关闭睡眠可作为这份无渲染加载任务的临时配置选择，但没有修复开启睡眠时的容量限制。可睡眠的静止动态刚体不再因睡眠退出求解；实际任务的稳态耗时、接触／堆叠稳定性可能不同，本轮没有测量这些影响，不能把加载结果外推到 P-02 或真实机器人任务。后续实验不自动改用关闭睡眠。

## 数据出处与复现

共 {len(rows)} 次运行，{len(completed)} 次完成。全部完成项已核验 arena／刚体数量、唯一 world ID、质量／惯量、有限状态以及真实 `_rigid_sleep is None`。最大初始位置残差 {validation['position_error_max_m']:.3g} m；小规模与旧边界预检保留 20 步跨 arena 重叠隔离记录。

- [逐次 CSV](../analysis/runs.csv)、[中位数与 run_id](../analysis/summary.json)、[内容核验](../analysis/content-validation.json)、[完整核验](../analysis/validation.json)。
- [运行计划](../plans/)、[原始目录](../runs/)、[SHA-256 清单](../analysis/evidence-manifest.json)。原睡眠开启基线 {len(baseline)} 个原始文件哈希不变。
- [本轮执行条件](../analysis/execution-scope.json)、[停止原因](../analysis/scope-stop.json)、[停止前计划快照](../analysis/capacity-plan-before-scope-stop.json)。16000 未完成不能写成新的硬上限；取消和未运行项不计作引擎报错。
- 继续沿用 EmbodiChain `f179b555`、P-03 冻结 DexSim + MR !1501 parser 副本；实际导入路径、硬件、命令与软件版本分别记录于每次 `run.json`／`launch.json`，没有更换安装环境。采集范围结束后的模型核验不计入就绪时间。
- 本补测没有录制新视频；原 L-02 的 16 arena 视频保持原标签，不冒充本轮大规模或关闭睡眠的录像。

```bash
python -m scripts.benchmark.tech_reports.loading.experiments.l02_multi_batch --suite pilot --platforms embodichain --num-envs 4 2897 --sleep off --output-root outputs/benchmarks/L-02/sleep-off-v1 --tag sleep_off_pilot_v1 --wait-idle
python -m scripts.benchmark.tech_reports.loading.experiments.l02_multi_batch --suite capacity --platforms embodichain --num-envs 1000 4000 8000 16000 --sleep off --output-root outputs/benchmarks/L-02/sleep-off-v1 --tag sleep_off_capacity_v1 --wait-idle
python -m scripts.benchmark.tech_reports.reporting.l02_sleep_review
```

上述 Python 为项目 `embodichain` 环境解释器；前两条复用已有 tag 时跳过完成项，不重跑。第三条仅离线分析。原功能诊断实际使用 `--allow-busy`；若要补独占 GPU 的性能数据，使用上述 `--wait-idle` 和新的 tag，单独统计。

逐次原始记录：

"""
    text += (
        "\n".join(
            f"- [{r['run_id']}](../runs/{r['run_id']}/run.json)：{r['status']}，[退出／停止记录](../runs/{r['run_id']}/exit.json)。"
            for r in rows
        )
        + "\n"
    )
    skipped = [j for j in plans if j["status"].startswith("not_run")]
    if skipped:
        text += (
            "\n未运行的计划项：\n\n"
            + "\n".join(
                f"- N={j['num_envs']}，{j['purpose']}：{j['reason']}。" for j in skipped
            )
            + "\n"
        )
    (reports / "sleep-off.md").write_text(text)
    source = analysis / "source"
    source.mkdir(exist_ok=True)
    for p in [Path(__file__), Path(__file__).with_name("l02_multi_report.py")]:
        shutil.copy2(p, source / p.name)
    manifest = [
        {
            "path": str(p.relative_to(ROOT)),
            "bytes": p.stat().st_size,
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        }
        for parent in [BASE / "runs", BASE / "plans", source]
        for p in sorted(parent.rglob("*"))
        if p.is_file()
    ]
    for name in [
        "sleep-construction-source.json",
        "execution-scope.json",
        "scope-stop.json",
        "capacity-plan-before-scope-stop.json",
    ]:
        p = analysis / name
        manifest.append(
            {
                "path": str(p.relative_to(ROOT)),
                "bytes": p.stat().st_size,
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            }
        )
    (analysis / "evidence-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(validation, indent=2))
    print("Markdown report saved:", reports / "sleep-off.md")


if __name__ == "__main__":
    main()
