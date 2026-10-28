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

"""Build the selected sleep-off L-02 draft from preserved runs; no simulation."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
from statistics import median

from .l02_multi_report import table

__all__ = ["main"]
ROOT = Path(
    os.environ.get("SIM_REPORT_ROOT", Path(__file__).resolve().parents[4])
).resolve()
BASE = ROOT / "outputs/benchmarks/L-02/sleep-off-main-v1"
DIAGNOSTIC = ROOT / "outputs/benchmarks/L-02/sleep-off-v1"
REFERENCE = ROOT / "outputs/benchmarks/L-02/multi-arena-v1"


def main() -> None:
    """Write the three-chapter draft, three-table summary and source index."""
    reports, analysis = BASE / "reports", BASE / "analysis"
    reports.mkdir(exist_ok=True)
    analysis.mkdir(exist_ok=True)
    rows = json.loads((DIAGNOSTIC / "analysis/runs.json").read_text())
    checks = json.loads((DIAGNOSTIC / "analysis/validation.json").read_text())
    isaac = [
        r
        for r in json.loads((REFERENCE / "analysis/summary.json").read_text())
        if r["platform"] == "isaaclab"
    ]
    media = json.loads((BASE / "videos/validation.json").read_text())
    assert len(rows) == 16 and checks["audited_completed_runs"] == 14
    assert all(
        r["rigid_enable_sleep"] is False and r["foreign_compute_pids"] for r in rows
    )
    for row in rows:
        run = DIAGNOSTIC / "runs" / row["run_id"]
        cfg = json.loads((run / "config.json").read_text())
        assert (
            cfg["rigid_enable_sleep"] is False
            and cfg["num_envs"] == row["num_envs"]
            and not cfg["record"]
        )
        if row["status"] == "completed":
            audit = json.loads((run / "model-audit.json").read_text())
            assert (
                audit["rigid_enable_sleep"] is False
                and audit["body_count"] == row["num_envs"]
            )
    for record in media["records"]:
        if record["platform"] == "embodichain":
            assert record["rigid_enable_sleep"] is False

    evidence = []
    for base in [DIAGNOSTIC, REFERENCE]:
        manifest = json.loads((base / "analysis/evidence-manifest.json").read_text())
        for record in manifest:
            path = ROOT / record["path"]
            assert (
                hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
            ), str(path)
        evidence.append(
            dict(
                manifest=str(
                    (base / "analysis/evidence-manifest.json").relative_to(ROOT)
                ),
                verified_files=len(manifest),
            )
        )

    capacity = {
        n: [
            r
            for r in rows
            if r["purpose"] == "capacity"
            and r["num_envs"] == n
            and r["status"] == "completed"
        ]
        for n in [1000, 4000, 8000]
    }
    assert all(len(group) == 3 for group in capacity.values())
    comparison = table(
        [
            "num_envs（每 arena 一个立方体）",
            "EmbodiChain／DexUni，sleep 关闭",
            "Isaac Lab／PhysX，原配置",
        ],
        [
            [1000, "3/3 完成", "10/10 完成"],
            [4000, "3/3 完成", "3/3 完成"],
            [8000, "3/3 完成", "3/3 完成"],
            [16000, "未确认：预载超时，随后停止", "3/3 完成"],
        ],
    )
    observed = table(
        [
            "num_envs",
            "完成次数（预载另计）",
            "原始就绪时间中位数 [最小,最大] / s",
            "原始 RAM 峰值中位数 / MiB",
            "原始 VRAM 峰值中位数 / MiB",
        ],
        [
            [
                n,
                len(group),
                f"{median(r['ready_s'] for r in group):.3f} [{min(r['ready_s'] for r in group):.3f}, {max(r['ready_s'] for r in group):.3f}]",
                f"{median(r['rss_peak_mib'] for r in group):.0f}",
                f"{median(r['vram_peak_mib'] for r in group):.0f}",
            ]
            for n, group in capacity.items()
        ],
    )
    isaac_table = table(
        [
            "形状",
            "num_envs",
            "缓存组",
            "独立次数",
            "就绪时间中位数 [最小,最大] / s",
            "RAM 峰值中位数 / MiB",
            "VRAM 峰值中位数 / MiB",
        ],
        [
            [
                "立方体" if r["shape"] == "cube" else "球体",
                r["num_envs"],
                r["cache"],
                r["repeats"],
                f"{r['ready_s']:.3f} [{r['ready_s_range'][0]:.3f}, {r['ready_s_range'][1]:.3f}]",
                f"{r['rss_peak_mib']:.0f}",
                f"{r['vram_peak_mib']:.0f}",
            ]
            for r in isaac
        ],
    )
    posters = {}
    for record in media["records"]:
        path = ROOT / record["run_path"] / "native_camera/frame_0029.png"
        posters[(record["platform"], record["shape"])] = os.path.relpath(path, reports)

    report = f"""# L-02 基础场景多 arena 加载

2026-10-27 · 初稿 · EmbodiChain 主配置：`rigid_enable_sleep=False` · 部分完成：加载可行性与原生素材已完成，公平性能对比待补测

## 1. 对比与结论

**关闭睡眠后，EmbodiChain／DexUni 已完成 8000 个独立 arena 的加载、首步和全部状态读回，独立重复 3 次均完成。** 每个 arena 仅含一个立方体；规模由 `num_envs` 控制。Isaac Lab 的现有对照已测至 16000 个 arena，独立重复 3 次均完成。两者均未确定绝对容量上限。

{comparison}

**耗时、内存和速度比暂不作跨平台结论。** EmbodiChain 的关闭睡眠数据采自共享 GPU；Isaac Lab 的参考数据采自无其他计算进程的窗口，条件不等价。16000 arena 的一次预载在 180 s 预算内未就绪，随后停止，尚不能判断是否存在新的容量限制。已有记录也不能证明耗时增长全由资源争用造成，需在独占 GPU 下复测。

本版明确选择关闭睡眠作为 L-02 配置。它绕过了睡眠模块的邻接表预算检查，但未修复开启睡眠时的限制，也不代表长期仿真吞吐、接触稳定性或带相机的任务性能。旧配置的 6.403 s 和 1.31 倍不用于本版结论。

## 2. EmbodiChain 详细报告

### 场景与执行方法

使用 `SimulationManagerCfg(num_envs=N)` 原生复制场景，每 arena 一个边长 50 mm、质量 0.1 kg 的动态立方体，环境间距 0.2 m。共享 USD visual 输入，解析 box 碰撞；零重力，初始局部位置 `(0,0,0.1)` m，步长 5 ms，VBD 10 次迭代，`rigid_enable_sleep=False`。每次审计实际求解器的睡眠对象不存在，实际刚体数等于 N；未修改求解器源码。

无窗口、无相机、无渲染。就绪时间从 Python 入口开始，包含模块导入、引擎启动、场景复制、物理准备、首步、GPU 同步及全部状态复制到 CPU。离线输入生成、后续模型审计和录像不计入该时间；180 s 单进程预算另包含审计与退出。各次独立进程；H 组先同配置预载，保留 OS／kernel 缓存，不称为磁盘冷启动。

硬件为 RTX 5090 D v2（24455 MiB）、驱动 580.159.04、Ryzen 9 9950X。EmbodiChain 提交 `f179b5558fa17df06f172f0b1e4e4f81df63be74`；运行依赖为 P-03 冻结 DexSim 副本，包含 MR !1501 的 parser 补丁，不能当作完整 MR head。具体包版本、导入路径、命令及代码快照见[逐次数据索引](data-index.md)。本批次以 `--allow-busy` 做功能验证，全部 16 次尝试都记录到了其他 GPU 计算进程。

### 加载结果与原始观测

N=1000／4000／8000 各一次预载与 3 次重复完成；N=4／2897 各一次预检完成。合计 14 次完成、1 次超时、1 次主动取消，另有 2 次计划项未运行。下表保存原始观测，**不是可比较的性能基线**；内存为入口至就绪的外部 20 ms 采样峰值，没有测量保留内存增量。

{observed}

8000 arena 的就绪时间原始观测为 75–80 s，明显高于 4000 arena；该增长值得排查，但本次资源争用未受控，不能据此确定算法增长率、全部归因于 GPU 竞争，或计算相对 Isaac Lab 的速度差。完整逐次时间与内存样本保留于[原始记录](data-index.md)。

16000 arena 的预载最后完成到场景声明阶段，180 s 内未完成物理准备；随后主动取消一条已启动重复，剩余两条未运行。主动取消和未运行不计为引擎失败，超时也不等同于 OOM 或固定数量上限。原始退出信息和日志均保留。

全部完成运行均核验实际 arena 标识、对象数、质量、惯量和有限状态数组 `(N,1,13)`。最大初始位置残差为 {checks['position_error_max_m']:.3g} m，最大惯量残差为 {checks['inertia_error_max_kg_m2']:.3g} kg·m²。N=4／2897 的跨环境重叠预检运行 20 步，位移和速度响应均为零。这些检查确认复制和隔离，不评价接触精度或长期稳定性。[核验记录](../../sleep-off-v1/analysis/validation.json)。

### 配置选择与保留问题

实际运行代码仅在 `rigid_enable_sleep=True` 时构造 `RigidBodySleep`；该模块按全模型刚体数 n 分配 `int32[n,n]` 邻接表，超过 32 MiB 预算就退出。关闭睡眠后不构造这张表，因此可以越过原 2896／2897 边界。[源码片段与哈希](../../sleep-off-v1/analysis/sleep-construction-source.json)。

开启睡眠的限制仍存在：[历史问题与复测清单](../../multi-arena-v1/reports/release-review.md)。需要睡眠优化的业务仍需修复该问题，并用新版本验证；关闭睡眠对稳态成本及接触任务的影响尚未测量。L-02 的配置选择不自动改变 P-02 等已冻结实验，也不自动改写 L-03～L-05 的配置。

本版待补项是独占 GPU 下的关闭睡眠主矩阵与容量复测：两形状 N=1／100／1000 的 C/H 组，以及立方体 N=2000／4000／8000／16000 的 H 组。当前高容量结论仅覆盖立方体；球体只有本次 N=16 的功能录像。保持同输入、计时边界和新 run_id，保留本批次记录；复测前不做跨平台性能排名。

### 实验画面与数据出处

![EmbodiChain：关闭睡眠，16 个立方体 arena，原生仿真截图]({posters[('embodichain', 'cube')]})

补录了关闭睡眠的立方体、球体原片；与 Isaac Lab 保持相同观察方向、阵列中心偏移、相机内参、分辨率和采样物理时刻。每段为实际仿真 3 s、60 帧、20 fps、1×；零重力静态初态，所以画面中的物体保持不动。两端都创建 16 台原生相机，只导出第一台全景，不纳入无相机加载计时。球体录像不替代其性能主测。[四段配对原片、原始帧、状态与相机核验](../videos/index.md)。

容量数据直接引用原 `sleep-off-v1` 的 16 次尝试，没有复制或重命名为新测量。[逐次数据与复现入口](data-index.md)、[三表 benchmark 摘要](benchmark-summary.md)、[本版证据清单](../analysis/evidence-manifest.json)。旧睡眠开启报告及旧单环境实验均保留，当前主比较不混用其 EmbodiChain 耗时。

## 3. Isaac Lab 详细报告

### 场景与配置

沿用已完成的多 arena 对照，Isaac Lab 提交 `ae37b028ea415c91ea2bc32609efcd759ed2b974`、Isaac Sim 6.1。使用 `InteractiveSceneCfg(num_envs=N, replicate_physics=True, filter_collisions=True)`；GPU PhysX TGS，位置／速度迭代 4/1。睡眠保留原生配置，本轮没有关闭 Isaac 的睡眠重测，不能将两端标成相同睡眠策略。

同输入、零重力、5 ms 步长、每 arena 一个物体、相同的入口至首步全量 CPU 读回边界。H 组先预载，C/H 均保留 OS 和 kernel 缓存。以下仅列没有其他 GPU 计算进程的既有主测／容量样本；预载、脚本调试失败及录像不进入统计。内存使用相同的外部 20 ms 采样方式。

{isaac_table}

范围是独立进程的最小值—最大值，不是置信区间。最高已测规模为 16000 个立方体 arena，3 次完成，就绪时间中位数 8.471 s；不称绝对上限，也不与本版共享 GPU 的 EmbodiChain 耗时计算倍率。所有参考行的 run_id 见[原始汇总](../../multi-arena-v1/analysis/summary.json)与[完整数据索引](../../multi-arena-v1/reports/data-index.md)。

原对照已通过全量模型与状态核验；最大位置残差 1.49e-09 m、惯量残差 6.32e-13 kg·m²，N=4 跨环境重叠检查无响应。早期 Tensor 和相机适配失败留存于历史记录，不当作当前有效加载失败。

![Isaac Lab：16 个立方体 arena，原生仿真截图]({posters[('isaaclab', 'cube')]})

本版复用既有 Isaac 同视角原片，设置未改动。仅录像关闭渲染分区以展示全部 arena，物理碰撞隔离保持；与 EmbodiChain 的光照、材质显示不同，不据此评价渲染质量或效率。[媒体核验及来源](../videos/index.md)。
"""
    (reports / "L-02.md").write_text(report)

    time_rows = [
        [
            "EmbodiChain／sleep 关",
            "cube",
            n,
            "H",
            "0（3 次共享 GPU 观测）",
            "待独占复测",
            "待独占复测",
            "未测量 / 未测量",
        ]
        for n in capacity
    ]
    time_rows += [
        [
            "Isaac Lab／原配置",
            r["shape"],
            r["num_envs"],
            r["cache"],
            r["repeats"],
            f"{r['ready_s'] * 1000:.1f}",
            f"{r['vram_peak_mib']:.0f}",
            "未测量 / 未测量",
        ]
        for r in isaac
    ]
    groups = sorted({(r["num_envs"], r["purpose"]) for r in rows})
    event_rows = []
    for n, purpose in groups:
        group = [r for r in rows if (r["num_envs"], r["purpose"]) == (n, purpose)]
        done = sum(r["status"] == "completed" for r in group)
        cancelled = any(r["status"] == "cancelled" for r in group)
        event_rows.append(
            [
                "EmbodiChain／sleep 关",
                n,
                purpose,
                f"{done}/{len(group)}",
                "不适用：主动取消" if cancelled else f"{done / len(group):.0%}",
                ", ".join(sorted({r["status"] for r in group})),
            ]
        )
    event_rows += [
        [
            "Isaac Lab／原配置",
            r["num_envs"],
            f"{r['shape']} {r['cache']} {r['purpose']}",
            f"{r['repeats']}/{r['repeats']}",
            "100%",
            "completed",
        ]
        for r in isaac
    ]
    summary = "# L-02 运行摘要：关闭睡眠主配置\n\n本版主配置为 EmbodiChain `rigid_enable_sleep=False`；Isaac 沿用原 PhysX 设置。仅加载完成事件可对照，共享 GPU 观测不参与性能排名。旧 sleep 开启配置另见[冻结摘要](../../multi-arena-v1/reports/benchmark-summary.md)。\n\n## 1. Time & Memory\n\n计时从 Python 入口到首步和全部环境 CPU 状态读回。cost_time_ms 为独立进程中位数；峰值以 20 ms 采样，peak_gpu_mb 此处按 MiB。保留内存增量未测量，不填零。完整 RAM 与范围见技术稿。\n\n"
    summary += table(
        [
            "配置",
            "形状",
            "num_envs",
            "组",
            "性能统计次数",
            "cost_time_ms",
            "peak_gpu_mb (MiB)",
            "cpu_delta_mb / gpu_delta_mb",
        ],
        time_rows,
    )
    summary += "\n\n## 2. Success & Other Metrics\n\n完成定义为实际首步与全量状态、模型核验完成，非人为精度门槛。预载和预检单列；16000 主动取消不算引擎失败，另两次未运行不进入分母。旧配置完整失败记录另见历史数据索引。\n\n"
    summary += table(
        ["配置", "num_envs", "用途", "完成 / 尝试", "success_rate", "状态"], event_rows
    )
    summary += (
        "\n\n## 3. Leaderboard\n\n只对照已测加载事件，不设综合名次；各自最大已测完成点不是绝对上限。资源与睡眠策略不同，禁止耗时排名。\n\n"
        + comparison
    )
    summary += "\n\n[逐次数据、限制和复现入口](data-index.md) · [三章技术稿](L-02.md)\n"
    (reports / "benchmark-summary.md").write_text(summary)

    index_rows = []
    for r in rows:
        prefix = f"../../sleep-off-v1/runs/{r['run_id']}"
        index_rows.append(
            [
                f"[{r['run_id']}]({prefix}/)",
                r["num_envs"],
                r["purpose"],
                r["status"],
                "未就绪" if r["ready_s"] is None else f"{r['ready_s']:.6f}",
                f"[配置]({prefix}/config.json) / [计时]({prefix}/run.json) / [退出]({prefix}/exit.json) / [内存]({prefix}/memory.jsonl) / [日志]({prefix}/console.log)",
            ]
        )
    index = "# L-02 数据出处与复现\n\n本版采用关闭睡眠配置，直接引用原始运行，不重命名或冒充新测量。以下时间均为共享 GPU 下的原始观测，不作公平性能比较。主动取消以 exit.json 为准；进程被终止时 run.json 可能保留 running。\n\n"
    index += table(
        ["run_id", "num_envs", "用途", "退出状态", "原始就绪 / s", "原始文件"],
        index_rows,
    )
    index += """

每个完成运行还保存 `first_state.npy`、`model.npz`、`model-audit.json`；小规模隔离预检含重叠前后与 20 步状态。`launch.json` 保存命令、硬件、提交、PYTHONPATH 和起始 GPU 进程；`source/` 是该次运行代码快照。`memory.jsonl` 保留其他计算进程，不能仅按启动时是否空闲判定无竞争。

- [关闭睡眠逐次分析 JSON](../../sleep-off-v1/analysis/runs.json)、[CSV](../../sleep-off-v1/analysis/runs.csv)、[全量模型核验](../../sleep-off-v1/analysis/content-validation.json)、[求解器条件分支与哈希](../../sleep-off-v1/analysis/sleep-construction-source.json)。
- [原执行计划](../../sleep-off-v1/plans/)、[16000 停止原因及未运行项](../../sleep-off-v1/analysis/scope-stop.json)。原计划已经结束，不用旧 tag 自动续跑。
- [Isaac 参考汇总及逐行 run_id](../../multi-arena-v1/analysis/summary.json)、[逐次数据索引](../../multi-arena-v1/reports/data-index.md)。仅选取 platform=isaaclab 的主测／容量行；不选入旧 EmbodiChain 的睡眠开启耗时。
- [本次关闭睡眠视频与复用 Isaac 原片](../videos/index.md)、[相机及编码核验](../videos/validation.json)。媒体不计性能样本。
- [选择范围和来源校验](../analysis/selection.json)、[本版证据哈希](../analysis/evidence-manifest.json)。许可与公开数据集准备仍按项目 SUBMISSION；本版只发布至既有局域网站。

### 复现和补测

在项目根目录使用现有 EmbodiChain 环境。以下报告命令仅离线读取，不启动仿真：

```bash
/home/oem/miniconda3/envs/embodichain/bin/python -m scripts.benchmark.tech_reports.reporting.l02_sleep_off_report
```

待补性能命令显式选择关闭睡眠，并用新的 tag、输出目录；共享 GPU 时等待，不使用 `--allow-busy`。先完成小规模主矩阵，再按规模逐点确认容量，不直接续跑已取消的旧计划：

```bash
/home/oem/miniconda3/envs/embodichain/bin/python -m scripts.benchmark.tech_reports.loading.experiments.l02_multi_batch --suite main --platforms embodichain --sleep off --shapes cube sphere --num-envs 1 100 1000 --cache C H --output-root outputs/benchmarks/L-02/sleep-off-performance-v1 --tag sleep_off_main_v1 --wait-idle
```

原有 Isaac 参考来自相同机器和协议，但非交替同期批次；正式速度比较至少补做同期代表点以检查运行窗口差异。关闭睡眠后未重新做独占 GPU 的大规模性能测量，也没有测接触长期稳定性。
"""
    (reports / "data-index.md").write_text(index)
    selection = dict(
        report_configuration={
            "embodichain": {"rigid_enable_sleep": False},
            "isaaclab": "original PhysX configuration",
        },
        status="partial; functional loading verified, uncontended performance pending",
        capacity_source=str(DIAGNOSTIC.relative_to(ROOT)),
        capacity_run_ids=[r["run_id"] for r in rows],
        capacity_statuses=dict(Counter(r["status"] for r in rows)),
        performance_eligible_embodichain_runs=[],
        isaac_reference_run_ids=[run_id for r in isaac for run_id in r["run_ids"]],
        reference_source=str(REFERENCE.relative_to(ROOT)),
        media=media,
        frozen_sources_verified=evidence,
    )
    (analysis / "selection.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2) + "\n"
    )
    source = analysis / "source"
    source.mkdir(exist_ok=True)
    for path in [
        Path(__file__),
        Path(__file__).with_name("l02_multi_media.py"),
        Path(__file__).with_name("l02_multi_report.py"),
    ]:
        shutil.copy2(path, source / path.name)
    manifest = []
    for path in sorted(BASE.rglob("*")):
        if path.is_file() and path.name not in {
            "evidence-manifest.json",
            "browser-validation.json",
        }:
            manifest.append(
                dict(
                    path=str(path.relative_to(ROOT)),
                    bytes=path.stat().st_size,
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                )
            )
    (analysis / "evidence-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(
        json.dumps(
            dict(
                report=str(reports / "L-02.md"),
                source_checks=evidence,
                selected_isaac_groups=len(isaac),
                media=len(media["records"]),
            ),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
