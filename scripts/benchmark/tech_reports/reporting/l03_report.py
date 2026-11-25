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

"""Rebuild L-03 loading tables and three-chapter draft from saved runs only.

Run: python -m scripts.benchmark.tech_reports.reporting.l03_report
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import shutil
import statistics

__all__ = ["main"]


def main() -> None:
    """Keep functional observations separate from resource-isolated formal timings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("outputs/benchmarks/L-03/three-robots-v1")
    )
    args = parser.parse_args()
    base = args.root
    analysis, reports = base / "analysis", base / "reports"
    analysis.mkdir(exist_ok=True)
    reports.mkdir(exist_ok=True)
    rows, evidence, warnings = [], [], []
    for out in sorted((base / "runs").iterdir()):
        if not (out / "exit.json").exists():
            continue
        cfg = json.loads((out / "config.json").read_text())
        run = (
            json.loads((out / "run.json").read_text())
            if (out / "run.json").exists()
            else {}
        )
        finish = json.loads((out / "exit.json").read_text())
        memory = [
            json.loads(line) for line in (out / "memory.jsonl").read_text().splitlines()
        ]
        ready = run.get("stage_times_s", {}).get("ready")
        until_ready = [
            s
            for s in memory
            if run.get("entry_epoch", float("inf"))
            <= s["epoch"]
            <= run.get("entry_epoch", 0) + (ready or 0)
        ]
        rss = [s["rss_bytes"] for s in until_ready if s["rss_bytes"] is not None]
        gpu = [
            sum(s["gpu_process_bytes"].get(str(pid), 0) for pid in s["tree_pids"])
            for s in until_ready
        ]
        lifetime_rss = [s["rss_bytes"] for s in memory if s["rss_bytes"] is not None]
        lifetime_gpu = [
            sum(s["gpu_process_bytes"].get(str(pid), 0) for pid in s["tree_pids"])
            for s in memory
        ]
        stages = run.get("stage_times_s", {})
        row = {
            k: cfg[k]
            for k in (
                "robot",
                "platform",
                "num_envs",
                "cache_mode",
                "purpose",
                "repeat",
            )
        }
        row.update(
            run_id=out.name,
            status=run.get("status", "no_result"),
            exit_code=finish["code"],
            timeout=finish["timeout"],
            performance_eligible=finish["performance_eligible"],
            external_compute_pids=finish["external_compute_pids"],
            ready_s=ready,
            ready_per_env_ms=(
                1000 * ready / cfg["num_envs"] if ready is not None else None
            ),
            engine_s=stages.get("engine_initialized"),
            declare_s=(
                stages["declared"] - stages["engine_initialized"]
                if "declared" in stages
                else None
            ),
            prepare_s=(
                stages["prepared"] - stages["declared"]
                if "prepared" in stages
                else None
            ),
            initialize_first_readback_s=(
                ready - stages["prepared"] if ready is not None else None
            ),
            peak_rss_mib=max(rss) / 2**20 if rss else None,
            peak_process_gpu_mib=max(gpu) / 2**20 if gpu and max(gpu) > 0 else None,
            peak_rss_per_env_mib=max(rss) / 2**20 / cfg["num_envs"] if rss else None,
            peak_process_gpu_per_env_mib=(
                max(gpu) / 2**20 / cfg["num_envs"] if gpu else None
            ),
            ready_memory_samples=len(until_ready),
            sample_interval_median_ms=(
                1000
                * statistics.median(
                    b["epoch"] - a["epoch"]
                    for a, b in zip(until_ready, until_ready[1:])
                )
                if len(until_ready) > 1
                else None
            ),
            sample_interval_max_ms=(
                1000
                * max(
                    b["epoch"] - a["epoch"]
                    for a, b in zip(until_ready, until_ready[1:])
                )
                if len(until_ready) > 1
                else None
            ),
            lifetime_peak_rss_mib=max(lifetime_rss) / 2**20 if lifetime_rss else None,
            lifetime_peak_process_gpu_mib=(
                max(lifetime_gpu) / 2**20 if lifetime_gpu else None
            ),
            audited_instances=run.get("audited_instances"),
            first_link_change_m=run.get("max_first_link_translation_change_m"),
            last_stage=list(stages)[-1] if stages else None,
        )
        rows.append(row)
        messages = [
            line.replace("\x00", "")
            for line in (out / "console.log").read_text(errors="replace").splitlines()
            if "entity lost balance" in line
        ]
        if messages:
            warnings.append(dict(run=out.name, messages=messages))
        for file in sorted(out.rglob("*")):
            if file.is_file():
                evidence.append(
                    dict(
                        path=str(file),
                        bytes=file.stat().st_size,
                        sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
                    )
                )
    (analysis / "runtime-warnings.json").write_text(
        json.dumps(warnings, indent=2) + "\n"
    )
    (analysis / "runs.json").write_text(json.dumps(rows, indent=2) + "\n")
    (analysis / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    shutil.copy2(__file__, analysis / "analysis-source.py")
    if rows:
        with (analysis / "runs.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    groups = defaultdict(list)
    for row in rows:
        if row["performance_eligible"]:
            groups[
                (row["robot"], row["num_envs"], row["cache_mode"], row["platform"])
            ].append(row)
    summary = []
    for (robot, n, cache, platform), samples in sorted(groups.items()):
        summary.append(
            dict(
                robot=robot,
                num_envs=n,
                cache_mode=cache,
                platform=platform,
                repeats=len(samples),
                ready_median_s=statistics.median(s["ready_s"] for s in samples),
                ready_min_s=min(s["ready_s"] for s in samples),
                ready_max_s=max(s["ready_s"] for s in samples),
                ram_median_mib=statistics.median(s["peak_rss_mib"] for s in samples),
                vram_median_mib=(
                    statistics.median(s["peak_process_gpu_mib"] for s in samples)
                    if all(s["peak_process_gpu_mib"] is not None for s in samples)
                    else None
                ),
                ram_min_mib=min(s["peak_rss_mib"] for s in samples),
                ram_max_mib=max(s["peak_rss_mib"] for s in samples),
                vram_min_mib=min(s["peak_process_gpu_mib"] for s in samples),
                vram_max_mib=max(s["peak_process_gpu_mib"] for s in samples),
                run_ids=[s["run_id"] for s in samples],
                representative_run_id=sorted(samples, key=lambda s: s["ready_s"])[
                    len(samples) // 2
                ]["run_id"],
                phase_median_s={
                    key: statistics.median(s[key] for s in samples)
                    for key in (
                        "engine_s",
                        "declare_s",
                        "prepare_s",
                        "initialize_first_readback_s",
                    )
                },
            )
        )
    (analysis / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    completed = [
        r
        for r in rows
        if r["status"] == "completed" and r["exit_code"] == 0 and not r["timeout"]
    ]
    failures = [r for r in rows if r not in completed]
    formal = [r for r in rows if r["performance_eligible"]]
    main_complete = (
        len(formal) == 108
        and len(summary) == 36
        and all(s["repeats"] == 3 for s in summary)
    )
    media = [r for r in completed if r["purpose"] == "media"]
    labels = {"cartpole": "Cartpole", "franka": "Franka", "go2": "Go2"}
    lookup = {
        (s["robot"], s["num_envs"], s["cache_mode"], s["platform"]): s for s in summary
    }
    by_run = {r["run_id"]: r for r in rows}
    comparison = []
    if main_complete:
        for robot in labels:
            a = lookup[(robot, 1000, "H", "embodichain")]
            b = lookup[(robot, 1000, "H", "isaaclab")]
            comparison.append(
                dict(
                    robot=robot,
                    embodichain=a,
                    isaaclab=b,
                    time_ratio=a["ready_median_s"] / b["ready_median_s"],
                    ram_ratio=a["ram_median_mib"] / b["ram_median_mib"],
                    vram_ratio=a["vram_median_mib"] / b["vram_median_mib"],
                )
            )
    (analysis / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    text = [
        "# L-03：机器人多环境加载",
        "",
        "Cartpole · Franka Panda · Unitree Go2｜2026-10-27｜实验初稿",
        "",
        "## 第一章：对比与结论",
        "",
        "本实验测量从程序入口到全部并行环境可开始仿真的等待时间。每个 arena 放一台完整机器人，通过 num_envs 分别加载 1、100、1000 份；两端读取相同 USD 及依赖。主测关闭渲染和相机，零重力、无关节驱动，步长 5 ms。",
        "",
        f"**正式样本 {len(formal)}/108，另有 18 次 H 预载。** "
        + (
            "三种机器人的全部计划规模均完成加载、首步、全量状态读回和模型核验。已测至 1000 arena，不代表容量上限。"
            if main_complete
            else "正式矩阵尚未完成，不下完整性能排名结论。"
        ),
        "",
    ]
    if comparison:
        ratios = [c["time_ratio"] for c in comparison]
        if min(ratios) > 1:
            text += [
                f"**当前 EmbodiChain 的就绪时间较长。** 在 1000 arena、预载后新进程的代表条件下，耗时为 Isaac Lab 的 {min(ratios):.2f}–{max(ratios):.2f} 倍。这是本次实际配置的端到端加载差距，不能归结为单一 USD 解析器或物理求解器的差距。",
                "",
            ]
        text += [
            "下表每个数值取 3 次独立进程的中位数。RAM／显存为入口至就绪窗口内的进程采样峰值，再取重复运行的中位数。内存两列均按 **EmbodiChain / Isaac Lab** 排列。",
            "",
            "| 机器人 · 1000 arena · H | EmbodiChain 就绪（s） | Isaac Lab 就绪（s） | 时间比 EC / IL | RAM（MiB） | 进程显存（MiB） |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for c in comparison:
            a, b = c["embodichain"], c["isaaclab"]
            text.append(
                f"| {labels[c['robot']]} | {a['ready_median_s']:.3f} | {b['ready_median_s']:.3f} | {c['time_ratio']:.2f}× | {a['ram_median_mib']:.0f} / {b['ram_median_mib']:.0f} | {a['vram_median_mib']:.0f} / {b['vram_median_mib']:.0f} |"
            )
        ram_ratios = [c["ram_ratio"] for c in comparison]
        vram_ratios = [c["vram_ratio"] for c in comparison]
        text += [
            "",
            f"**进程资源占用较低：**本配置下 EmbodiChain 的显存峰值为对照的 {min(vram_ratios):.0%}–{max(vram_ratios):.0%}，RAM 峰值为 {min(ram_ratios):.0%}–{max(ram_ratios):.0%}。这些是整个进程的占用，未拆分机器人模型与运行时内存，不能据此直接推算最大环境数。",
        ]
        text += [
            "",
            "![三种机器人的多环境加载时间；点为中位数，误差线为三次最小至最大值](../figures/loading-time.png)",
            "",
            "图 1｜全部已测规模和 C/H 条件。C、H 均保留系统文件缓存与通用 kernel/shader 缓存；H 额外执行同配置预载，C 不是磁盘冷启动。误差线是 3 次运行范围，不是置信区间。",
            "",
            "**实际使用建议：**评估批量仿真部署时，使用目标环境数的总就绪时间和峰值内存估算启动成本；不能把单环境时间乘以 N，也不能将本结果当作稳态仿真 FPS。两端都保留原生复制、资产共享和碰撞隔离。",
            "",
        ]
    text += [
        "### 发布前需处理的问题",
        "",
        "- **已证实的加载差距：**具体见上表及分阶段数据。下一步定位应区分固定运行时启动、场景声明、物理准备和全量读回，再用同资产、同规模复测；本轮未改引擎源码。",
        "- **已确认的导入兼容性问题：**此前 G1 转换 USD 的 19 个父 Xform 下网格碰撞体被当前冻结 DexSim 导入路径遗漏。G1 已排除本轮，不能用本次三个机器人完成加载来宣称 G1 问题已解决。见 [独立记录与复测要求](../../reports/issues/g1-usd-parent-collider.md)。",
        f"- **待定位的退出提示：**{len(warnings)} 次 EmbodiChain 运行保留 `entity lost balance` 提示，均正常退出。本轮没有证明这是资源泄漏；应另做同进程反复创建／销毁的资源核查。见 [原始日志索引](../analysis/runtime-warnings.json)。",
        "",
        "## 第二章：EmbodiChain 详细报告",
        "",
        "### 共同实验条件与输入",
        "",
        "测试使用同一台工作站、单 GPU 串行执行，并交替两平台的先后顺序。正式运行前和运行中均检查其他 GPU 计算进程；桌面图形进程保留并记录。每个机器人 × N × C/H × 平台重复 3 次，共 108 个正式样本，H 另作 18 次预载。早期共享 GPU 下的 18 次规模试跑、6 次录像均独立保留，不纳入性能统计。",
        "",
    ]
    if formal:
        launch = json.loads(
            (base / "runs" / formal[0]["run_id"] / "launch.json").read_text()
        )
        hw = launch["hardware"]
        text += [
            "| 项目 | 实际配置 |",
            "| --- | --- |",
            f"| GPU | {hw['gpu_name']}；{hw['gpu_total_bytes']/2**30:.2f} GiB；驱动 {hw['driver']} |",
            f"| CPU / RAM | {hw['cpu'].strip().lstrip(':').strip()}；{hw['cpu_count']} 逻辑线程；{hw['ram_total_bytes']/2**30:.2f} GiB |",
            f"| EmbodiChain | `{launch['commits']['embodichain'][:12]}`；公共 SimulationManager 接口 |",
            f"| Isaac Lab | `{launch['commits']['isaaclab'][:12]}`；isaaclab Conda，Isaac Sim 6.1 |",
            "| 数值与场景 | 零重力；dt=0.005 s；无关节驱动；自碰撞关闭；环境间碰撞隔离 |",
            "| 线程环境 | 两端均 OMP_NUM_THREADS=4、OPENBLAS_NUM_THREADS=4；不宣称所有引擎线程池均为 4 线程 |",
            "| 主测 / 录像 | 主测无相机、无渲染；录像 N=4，单独采集，不计入性能 |",
            "",
        ]
    text += [
        "三个机器人均来自本地 NVIDIA Isaac 资产；两平台直接读取同一份冻结 USD 及其引用依赖，逐文件 SHA-256 记录在 [共同输入清单](../../prepared/shared-usd-v1/manifest.json)。本轮三者没有 URDF 转换计时，也没有一端改用另一套简化资产。",
        "",
        "| 机器人 | 每环境活动关节 | 每环境刚体 | 每环境碰撞体 | 根类型 | 共同输入 |",
        "| --- | ---: | ---: | ---: | --- | --- |",
        "| Cartpole | 2 | 3 | 3 | 固定 | Classic/Cartpole/cartpole.usd |",
        "| Franka Panda | 9 | 13 | 12 | 固定 | Factory/franka_mimic.usd |",
        "| Go2 | 12 | 19 | 27 | 浮动 | Unitree/Go2/go2.usd |",
        "",
        "两端按 link／joint 名称对齐质量、惯量、质心、关节初值和限位，并核对所有 arena 的状态与原生碰撞数量。18 次小规模预检还包含 N=4 完全重叠的 100 步隔离检查：重叠与正常间距的最大 link 位置差约 0.00024 mm、关节位置差为 0。原始差异见 [预检分析](../../analysis/three-robots-preflight-v1/model-comparison.json)；正式样本的逐对差异见 [全量核验](../analysis/validation.json)。这不是对接触动力学精度的验证。",
        "",
        "### 计时与内存口径",
        "",
        "T_ready 从 Python 入口开始，包含运行时导入、初始化、场景声明与原生复制、物理准备、完整初态写入、首个物理步、GPU 同步及全部环境的关节／根／link 状态 CPU 读回。输入文件校验在启动前完成；完整模型审计、数组落盘、录像和退出在 ready 之后，不计入主指标。180 s 是整个进程预算，不是单独的加载门槛。",
        "",
        "T_engine 为入口到仿真管理器初始化；T_declare 为随后场景声明／原生克隆；T_prepare 为随后物理准备与同步；最后一段包含初态恢复、首步和全量读回。阶段边界随平台接口划分，不能把其中某段直接当作纯 USD 解析耗时。",
        "",
        "RAM 为进程树 RSS，进程显存合并 NVML 的计算／图形 PID 占用。外部采样目标间隔 20 ms，实际间隔和全生命周期峰值另存 [逐次结果](../analysis/runs.csv)；短于采样间隔的尖峰可能遗漏。GiB/MiB 使用二进制单位，三表摘要的 MB 字段使用十进制单位。",
        "",
        "### 实际执行路径",
        "",
        "用 SimulationManagerCfg(num_envs=N) 原生复制完整机器人，USD 物理属性按 overlay 导入，根类型与资产一致。冻结 DexSim 依赖为 P-03 保存的完整运行副本加 MR !1501 parser 修复，并非最新 DexSim 全版本认证。",
        "",
        "本场景仅含 articulation，实际由 DexUni 的 SolverMuJoCo 分支处理，未构造 VBD。因此配置里的 rigid_enable_sleep=True 对本分支不适用，不能将此处结果解释为 VBD 性能，或套用 L-02 的睡眠容量限制。",
        "",
    ]
    for platform, heading in (
        ("embodichain", None),
        ("isaaclab", "## 第三章：Isaac Lab 详细报告"),
    ):
        if heading:
            text += [
                heading,
                "",
                "使用 isaaclab Conda 中的 Isaac Sim 6.1、GPU PhysX TGS 4/1。InteractiveSceneCfg 保持 replicate_physics=True、filter_collisions=True，每环境加载同一共同 USD。无关节驱动、自碰撞关闭，固定关节未人为合并。5.1 仅曾用于 G1 离线格式转换，不参与本轮对照。",
                "",
                "sim.reset() 包含内部物理预热，计入准备阶段。之后一次性恢复全部环境的根位姿／速度和关节位置／速度，再执行首步；后续不周期清零。当前 6.1 API 使用 xyzw。原 Go2 约 0.415 mm 偏移来自实验脚本遗漏根状态恢复，已单独 [定位与复验](../../reports/go2-initialization-diagnosis.md)，旧偏移不作为引擎精度差异。",
                "",
            ]
        text += [
            "### 正式结果",
            "",
            "每行 3 次。时间以中位数［最小–最大］表示；内存是每次 ready 窗口采样峰值的中位数。完整逐次值、内存范围及 run_id 在 [统计 JSON](../analysis/summary.json) 与 [数据索引](data-index.md)。",
            "",
            "| 机器人 | N | 条件 | 就绪时间（s） | RAM（MiB） | 进程显存（MiB） |",
            "| --- | ---: | --- | ---: | ---: | ---: |",
        ]
        for robot in labels:
            for n in (1, 100, 1000):
                for cache in ("C", "H"):
                    s = lookup.get((robot, n, cache, platform))
                    if s:
                        text.append(
                            f"| {labels[robot]} | {n} | {cache} | {s['ready_median_s']:.3f}［{s['ready_min_s']:.3f}–{s['ready_max_s']:.3f}］ | {s['ram_median_mib']:.1f} | {s['vram_median_mib']:.1f} |"
                        )
        if main_complete:
            growth = []
            for robot in labels:
                one = lookup[(robot, 1, "H", platform)]["ready_median_s"]
                thousand = lookup[(robot, 1000, "H", platform)]["ready_median_s"]
                growth.append(
                    f"{labels[robot]} {one:.3f} → {thousand:.3f} s（增加 {thousand-one:.3f} s）"
                )
            text += [
                "",
                "预载后新进程从 N=1 扩至 N=1000："
                + "；".join(growth)
                + "。这些是端到端时间变化，不能除以关节数后推广为通用的单关节加载成本。",
                "",
            ]
        if platform == "embodichain" and main_complete:
            text += [
                "### 分阶段对比与定位方向",
                "",
                "![1000 arena 的加载阶段分解](../figures/loading-stages.png)",
                "",
                "图 2｜N=1000、H。每根柱取该组就绪时间居中的一次真实运行，因此各段相加等于该次总耗时；各段不是独立中位数的拼接。首步段包含完整初态写入与全量读回，不是单步求解延迟。",
                "",
                "| 机器人 / 平台 | 运行时启动（s） | 声明／复制（s） | 物理准备（s） | 初始化／首步／读回（s） | 原始记录 |",
                "| --- | ---: | ---: | ---: | ---: | --- |",
            ]
            for robot in labels:
                for impl in ("embodichain", "isaaclab"):
                    r = by_run[
                        lookup[(robot, 1000, "H", impl)]["representative_run_id"]
                    ]
                    text.append(
                        f"| {labels[robot]} / {'EmbodiChain' if impl == 'embodichain' else 'Isaac Lab'} | {r['engine_s']:.3f} | {r['declare_s']:.3f} | {r['prepare_s']:.3f} | {r['initialize_first_readback_s']:.3f} | [记录](../runs/{r['run_id']}/run.json) |"
                    )
            starts = [
                by_run[c["embodichain"]["representative_run_id"]]["engine_s"]
                for c in comparison
            ]
            text += [
                "",
                f"本组 EmbodiChain 在场景声明之前已花费 {min(starts):.2f}–{max(starts):.2f} s，说明固定运行时启动成本需要独立检查。图中后续阶段可用于确定下一步 profiler 的范围；本轮只分段计时，没有逐函数归因，不把差距直接解释成某个求解器缺陷。",
                "",
                "### 进程资源占用",
                "",
                "![预载后新进程的 RAM 与显存峰值](../figures/loading-memory.png)",
                "",
                "图 3｜H 条件，3 次峰值的中位数与最小–最大范围。C 的完整数值同样保存在上表和原始数据中。进程内存不包含桌面等其他进程，也不等于仅由机器人模型占用的内存。",
                "",
            ]
    if failures:
        text += ["### 异常尝试", ""]
        for r in failures:
            text.append(
                f"- {r['platform']}／{r['robot']}／N={r['num_envs']}：最后阶段 {r['last_stage']}，exit={r['exit_code']}，timeout={r['timeout']}；[证据](../runs/{r['run_id']}/)。"
            )
    text += [
        "",
        "### 原始数据、视频与复现",
        "",
        f"当前数据目录保存 {len(rows)} 次运行，{len(completed)} 次正常退出、{len(failures)} 次异常。正式样本、H 预载、共享资源试跑和录像的身份各自记录，未重新标记旧样本。另有 18 次初始化／碰撞／隔离预检。",
        "",
        "六段原生视频为两平台三种机器人各一段：N=4、960×540、20 fps、3 s、1×，每场景 4 台相机，仅导出第一台。共同相机相对阵列中心的位姿、内参与采样时刻已核对，保留 360 帧 PNG 和完整状态；前 0.5 s 与无渲染预检对齐。零重力、无驱动，画面展示加载后静态场景；光照和材质呈现不同，不用于画质排名。",
        "",
        "- [全部原片与相机核验](../videos/index.md)，HTML 页面提供在线播放。",
        "- [逐次 CSV](../analysis/runs.csv)、[正式统计与 run_id](../analysis/summary.json)、[全量状态与模型核验](../analysis/validation.json)。",
        "- [数据索引与执行命令](data-index.md)、[三表运行摘要](benchmark-summary.md)、[全部原始运行目录](../runs/)。",
        "- [原始文件 SHA-256](../analysis/evidence.json)、[离线分析源码](../analysis/analysis-source.py)、[图表数据与参数](../figures/plot-data.json)、[矢量图和绘图源码](../figures/)。",
        "- [正式执行计划](../plans/three_robots_main_v1.json)；每次目录均包含 config、launch、run、exit、源码快照、console.log、memory.jsonl、states.npz 和 model.npz。",
        "- [正式测量前保留的旧稿与证据](../history/pre-formal-20261027_185909/)。共享 GPU 下的试跑仅用于功能检查，不参与本稿的速度或内存比较。",
        "",
    ]
    (reports / "L-03.md").write_text("\n".join(text))
    index = [
        "# L-03 数据索引",
        "",
        "每行对应独立进程；配置、源码快照、日志、20 ms 内存采样、初态／首步、模型数组和退出记录均保留。加载耗时与显存比较仅使用 exit.json 的 performance_eligible=true 项。",
        "",
        "| 机器人 | N | 平台 | 阶段／缓存 | 状态 | 原始运行 |",
        "| --- | ---: | --- | --- | --- | --- |",
    ]
    for r in rows:
        index.append(
            f"| {r['robot']} | {r['num_envs']} | {r['platform']} | {r['purpose']}／{r['cache_mode']} | {r['status']}，exit={r['exit_code']} | [{r['run_id']}](../runs/{r['run_id']}/run.json) |"
        )
    index += [
        "",
        "在项目根目录执行以下正式矩阵命令；先确认 GPU 无其他计算进程。既有计划只续做 pending 项，失败项保留；不得删计划来隐藏失败。",
        "",
        "```bash",
        "/home/oem/miniconda3/envs/embodichain/bin/python -m scripts.benchmark.tech_reports.loading.experiments.l03_batch --suite main --num-envs 1 100 1000 --cache C H --tag three_robots_main_v1 --wait-idle",
        "```",
        "",
        "离线重建：",
        "",
        "```bash",
        "/home/oem/miniconda3/envs/embodichain/bin/python -m scripts.benchmark.tech_reports.reporting.l03_report",
        "```",
        "",
    ]
    (reports / "data-index.md").write_text("\n".join(index))
    summary_text = [
        "# L-03 benchmark 运行摘要",
        "",
        "## Time & Memory",
        "",
        "cpu_delta_mb／gpu_delta_mb 未单独测量，填未测；peak_gpu_mb 为入口至 ready 的外部采样进程峰值，按十进制 MB（10⁶ bytes）换算。时间和峰值取三次中位数，只列合规正式样本。",
        "",
        "| robot | num_envs | cache | impl | repeats | cost_time_ms | cpu_delta_mb | gpu_delta_mb | peak_gpu_mb |",
        "| --- | ---: | --- | --- | ---: | ---: | --- | --- | ---: |",
    ]
    for s in summary:
        summary_text.append(
            f"| {s['robot']} | {s['num_envs']} | {s['cache_mode']} | {s['platform']} | {s['repeats']} | {1000*s['ready_median_s']:.2f} | 未测 | 未测 | {s['vram_median_mib']*2**20/1e6:.1f} |"
        )
    summary_text += [
        "",
        "## Success & Other Metrics",
        "",
        "完成率只表示独立进程完成加载、审计且正常退出，不是人为物理精度门槛。预载、功能试跑、媒体与正式样本分开。",
        "",
        "| robot | num_envs | impl | purpose | completed/attempted | success_rate |",
        "| --- | ---: | --- | --- | --- | ---: |",
    ]
    event_groups = defaultdict(list)
    for r in rows:
        event_groups[(r["robot"], r["num_envs"], r["platform"], r["purpose"])].append(r)
    for (robot, n, platform, purpose), samples in sorted(event_groups.items()):
        successes = sum(r in completed for r in samples)
        summary_text.append(
            f"| {robot} | {n} | {platform} | {purpose} | {successes}/{len(samples)} | {successes/len(samples):.4f} |"
        )
    summary_text += [
        "",
        "## Leaderboard",
        "",
        "不合成总分或以功能试跑耗时排名。以下覆盖两平台已尝试的配置；正式性能见第一表。",
        "",
        "| rank | algorithm | overall_success_rate | formal_samples | scope |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for platform in ("embodichain", "isaaclab"):
        samples = [r for r in rows if r["platform"] == platform]
        rate = sum(r in completed for r in samples) / len(samples) if samples else None
        summary_text.append(
            f"| 不设综合名次 | {platform} | {rate:.4f} | {sum(r['performance_eligible'] for r in samples)} | 进程完成率；配置见第二表 |"
            if rate is not None
            else f"| 不设综合名次 | {platform} | 未测 | 0 | 未测 |"
        )
    (reports / "benchmark-summary.md").write_text("\n".join(summary_text) + "\n")
    videos = base / "videos"
    videos.mkdir(exist_ok=True)
    if not (videos / "index.md").exists():
        (videos / "index.md").write_text(
            "# L-03 原生视频\n\n尚未完成原片编码与核验。\n"
        )
    print(
        json.dumps(
            dict(
                attempts=len(rows),
                completed=len(completed),
                failures=len(failures),
                formal_samples=len(formal),
                formal_matrix_complete=main_complete,
                media_runs=len(media),
            ),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
