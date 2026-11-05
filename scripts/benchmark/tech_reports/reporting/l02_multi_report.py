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

"""Rebuild the multi-arena L-02 report from saved runs; never initialize a simulator."""

from __future__ import annotations

import os

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

__all__ = ["main"]
ROOT = Path(
    os.environ.get("SIM_REPORT_ROOT", Path(__file__).resolve().parents[4])
).resolve()
BASE = ROOT / "outputs/benchmarks/L-02/multi-arena-v1"
NAMES = {"embodichain": "EmbodiChain", "isaaclab": "Isaac Lab"}


def table(headers: list[str], rows: list[list[object]]) -> str:
    """Build a small Markdown table."""
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        + ["| " + " | ".join(map(str, row)) + " |" for row in rows]
    )


def main() -> None:
    """Validate all environments and emit traceable tables, plots and three chapters."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--partial",
        action="store_true",
        help="Analyze completed runs while a batch is running",
    )
    parser.add_argument("--input-root", type=Path, default=BASE)
    args = parser.parse_args()
    base = args.input_root.resolve()
    assert base == BASE or args.partial, "Alternate roots support metrics-only analysis"
    analysis, reports = base / "analysis", base / "reports"
    analysis.mkdir(parents=True, exist_ok=True)
    rows, checks = [], []
    for folder in sorted((base / "runs").iterdir()):
        if not (folder / "exit.json").exists():
            continue
        cfg, run, launch, ex = [
            json.loads((folder / name).read_text())
            for name in ["config.json", "run.json", "launch.json", "exit.json"]
        ]
        status = (
            "cancelled"
            if ex.get("cancelled")
            else (
                "timeout"
                if ex["timeout"]
                else (
                    "completed"
                    if ex["code"] == 0 and run["status"] == "completed"
                    else "failed"
                )
            )
        )
        times = run["stage_times_s"]
        row = {
            k: cfg[k]
            for k in [
                "platform",
                "shape",
                "num_envs",
                "objects_per_env",
                "cache_mode",
                "purpose",
                "repeat",
            ]
        }
        row.update(
            rigid_enable_sleep=cfg.get("rigid_enable_sleep", True),
            run_id=folder.name,
            status=status,
            failure_reason=run.get("failure_reason", ex.get("reason", "")),
            last_stage=next(reversed(times), "none"),
            ready_s=times.get("ready"),
            engine_s=times.get("engine_initialized"),
            declaration_s=None,
            prepare_s=None,
            first_step_s=None,
            rss_peak_mib=None,
            vram_peak_mib=None,
            foreign_compute_pids=[],
            position_error_max_m=run.get("position_error_max_m"),
            mass_error_max_kg=run.get("mass_error_max_kg"),
        )
        samples = [
            json.loads(line)
            for line in (folder / "memory.jsonl").read_text().splitlines()
        ]
        cutoff = (
            run["entry_epoch"] + times["ready"] if "ready" in times else ex["end_epoch"]
        )
        samples = [x for x in samples if run["entry_epoch"] <= x["epoch"] <= cutoff]
        row["foreign_compute_pids"] = sorted(
            {
                pid
                for x in samples
                for pid in x["compute_pids"]
                if pid not in x["tree_pids"]
            }
        )
        if "ready" in times:
            row.update(
                declaration_s=times["declared"] - times["engine_initialized"],
                prepare_s=times["prepared"] - times["declared"],
                first_step_s=times["ready"] - times["prepared"],
            )
            assert samples, folder
            row["rss_peak_mib"] = max(x["rss_bytes"] or 0 for x in samples) / 2**20
            vram = [
                sum(
                    v
                    for pid, v in x["gpu_process_bytes"].items()
                    if int(pid) in x["tree_pids"] and v < 2**60
                )
                for x in samples
            ]
            row["vram_peak_mib"] = max(vram) / 2**20
        if status == "completed":
            d = np.load(folder / "model.npz")
            st = np.load(folder / "first_state.npy")
            audit = json.loads((folder / "model-audit.json").read_text())
            n = cfg["num_envs"]
            assert (
                cfg["version"] == 3
                and cfg["scope"] == "multi_arena"
                and cfg["objects_per_env"] == 1
            )
            assert st.shape == (n, 1, 13) and np.isfinite(st).all(), folder
            assert (
                audit["num_envs"]
                == audit["instance_count"]
                == len(set(audit["arena_names"]))
                == n
            )
            assert d["mass"].size == n and len(np.unique(d["env_origins"], axis=0)) == n
            assert len(cfg["objects"]) == 1
            asset = cfg["objects"][0]
            assert (
                hashlib.sha256(Path(asset["path"]).read_bytes()).hexdigest()
                == asset["sha256"]
            )
            inertia = d["inertia"].astype(float).reshape(n, -1)
            if inertia.shape[1] == 9:
                inertia = inertia[:, [0, 4, 8]]
            expected_i = (
                0.1 * 0.05**2 / 6 if cfg["shape"] == "cube" else 0.4 * 0.1 * 0.025**2
            )
            if cfg["platform"] == "embodichain":
                assert sorted(d["body_world"].tolist()) == list(range(n)), folder
                assert (
                    int(np.sum(d["shape_type"] == (7 if cfg["shape"] == "cube" else 3)))
                    == n
                )
            elif "colliders" in audit:
                assert len(audit["colliders"]) == n
                assert {x["type"] for x in audit["colliders"]} == {
                    "Cube" if cfg["shape"] == "cube" else "Sphere"
                }
            check = dict(
                run_id=folder.name,
                num_envs=n,
                status="content_verified",
                canonical_state="first_state.npy: env-local xyz, xyzw quaternion, linear and angular velocity",
                model_pose_convention=(
                    "world xyz + wxyz, native diagnostic"
                    if d["pose"].shape == (n, 7)
                    else "env-local xyz + xyzw"
                ),
                state_shape=list(st.shape),
                inertia_error_max_kg_m2=float(np.max(np.abs(inertia - expected_i))),
                local_position_error_max_m=float(
                    np.linalg.norm(st[..., :3] - cfg["positions"], axis=-1).max()
                ),
                isolation_drift_m=run.get("isolation_max_drift_m"),
                isolation_speed_mps=run.get("isolation_max_speed_mps"),
            )
            checks.append(check)
        rows.append(row)
    assert (
        len({r["rigid_enable_sleep"] for r in rows}) == 1
    ), "Different sleep settings must use separate result roots"
    (analysis / "runs.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n"
    )
    with (analysis / "runs.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    main_attempts = [r for r in rows if r["purpose"] == "main"]
    excluded = [r for r in main_attempts if r["foreign_compute_pids"]]
    main_rows = [r for r in main_attempts if not r["foreign_compute_pids"]]
    if not args.partial:
        assert len(main_rows) == 100 and all(
            r["status"] == "completed" for r in main_rows
        )
        assert all(
            not r["foreign_compute_pids"]
            for r in rows
            if r["purpose"] == "capacity" and r["status"] == "completed"
        )
    groups = defaultdict(list)
    for r in rows:
        if (
            r["purpose"] in ["main", "capacity"]
            and r["status"] == "completed"
            and not r["foreign_compute_pids"]
        ):
            groups[
                (
                    r["platform"],
                    r["shape"],
                    r["num_envs"],
                    r["cache_mode"],
                    r["purpose"],
                )
            ].append(r)
    if not args.partial:
        for (platform, shape, n, cache, purpose), group in groups.items():
            if purpose == "main":
                assert len(group) == (10 if shape == "cube" and n == 1000 else 3), (
                    platform,
                    shape,
                    n,
                    cache,
                )
    summary = []
    for (platform, shape, n, cache, purpose), data in sorted(groups.items()):
        s = dict(
            platform=platform,
            shape=shape,
            num_envs=n,
            cache=cache,
            purpose=purpose,
            rigid_enable_sleep=data[0]["rigid_enable_sleep"],
            repeats=len(data),
            run_ids=[x["run_id"] for x in data],
        )
        for metric in [
            "ready_s",
            "engine_s",
            "declaration_s",
            "prepare_s",
            "first_step_s",
            "rss_peak_mib",
            "vram_peak_mib",
        ]:
            v = [x[metric] for x in data]
            s[metric] = float(np.median(v))
            s[metric + "_range"] = [min(v), max(v)]
        s["ready_p95_s"] = float(np.percentile([x["ready_s"] for x in data], 95))
        summary.append(s)
    (analysis / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (analysis / "content-validation.json").write_text(
        json.dumps(checks, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "runs": len(rows),
                "purposes": dict(Counter(r["purpose"] for r in rows)),
                "statuses": dict(Counter(r["status"] for r in rows)),
                "main_completed": sum(r["status"] == "completed" for r in main_rows),
            },
            ensure_ascii=False,
        )
    )
    if args.partial:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, metric, label in zip(
        axes,
        ["ready_s", "rss_peak_mib", "vram_peak_mib"],
        ["Physics ready (s)", "Peak process RAM (MiB)", "Peak process VRAM (MiB)"],
    ):
        for platform, color in [("embodichain", "#087d78"), ("isaaclab", "#c26428")]:
            data = sorted(
                [
                    s
                    for s in summary
                    if s["platform"] == platform
                    and s["shape"] == "cube"
                    and s["cache"] == "H"
                ],
                key=lambda s: s["num_envs"],
            )
            ns = [s["num_envs"] for s in data]
            values = [s[metric] for s in data]
            errors = np.array(
                [
                    [s[metric] - s[metric + "_range"][0] for s in data],
                    [s[metric + "_range"][1] - s[metric] for s in data],
                ]
            )
            ax.errorbar(
                ns,
                values,
                yerr=errors,
                label=NAMES[platform],
                color=color,
                marker="o",
                capsize=3,
            )
        ax.set_xscale("log")
        ax.set_xlabel("num_envs (one cube per arena)")
        ax.set_ylabel(label)
        ax.grid(alpha=0.15)
    axes[0].legend(frameon=False)
    fig.suptitle(
        "L-02 | Native multi-arena replication | No cameras or rendering | H group"
    )
    fig.tight_layout()
    for ext in ["png", "svg", "pdf"]:
        fig.savefig(analysis / f"loading.{ext}", bbox_inches="tight")
    plt.close(fig)

    key = {
        (s["platform"], s["shape"], s["num_envs"], s["cache"], s["purpose"]): s
        for s in summary
    }
    typical = []
    for shape, n in [("cube", 1), ("cube", 100), ("cube", 1000), ("sphere", 1000)]:
        a = key["embodichain", shape, n, "H", "main"]
        b = key["isaaclab", shape, n, "H", "main"]
        typical.append(
            [
                ("立方体" if shape == "cube" else "球体") + f" × {n} arena",
                f"{a['ready_s']:.3f}",
                f"{b['ready_s']:.3f}",
                f"{a['ready_s']/b['ready_s']:.2f}×",
                f"{a['vram_peak_mib']:.0f} / {b['vram_peak_mib']:.0f}",
            ]
        )
    a = key["embodichain", "cube", 1000, "H", "main"]
    b = key["isaaclab", "cube", 1000, "H", "main"]
    base_a = key["embodichain", "cube", 1, "H", "main"]
    base_b = key["isaaclab", "cube", 1, "H", "main"]
    cap = {
        p: max(
            [
                s["num_envs"]
                for s in summary
                if s["platform"] == p and s["purpose"] == "capacity"
            ],
            default=1000,
        )
        for p in NAMES
    }
    boundary = [r for r in rows if "multi_sleep_boundary_v1" in r["run_id"]]
    assert {r["num_envs"]: r["status"] for r in boundary} == {
        2896: "completed",
        2897: "failed",
    }
    boundary_ok = next(r["run_id"] for r in boundary if r["num_envs"] == 2896)
    boundary_fail = next(r["run_id"] for r in boundary if r["num_envs"] == 2897)
    first_launch = json.loads(
        (BASE / "runs" / main_rows[0]["run_id"] / "launch.json").read_text()
    )
    header = f"""# L-02 基础场景多 arena 加载

2026-10-27 · 协议 v3 · 每个 arena 一个物体 · 规模由 `num_envs` 控制

## 1. 对比与结论

本次测量面向批量并行物理仿真：两端通过公共场景接口复制 N 个独立 arena，每个 arena 加载同一份 50 mm、0.1 kg 的基础物体。主指标是**从 Python 入口到全部环境完成首步且状态可读的时间**，主测无窗口、无相机、无渲染。所有数字来自新运行，不与旧单环境数据混算。

**发布前优先修复：当前 DexUni 睡眠模块对全模型刚体数设置了约 2896 个的硬限制。** 本次每 arena 一个刚体：2896 arena 的边界预检完成，2897 arena 报错；4000 arena 的预载及 3 次正式尝试均报同一错误。Isaac Lab 在 16000 arena 的 3 次尝试均完成。限制来自全局 `int32[n,n]` 睡眠邻接表的 32 MiB 代码预算，**不是 GPU 显存耗尽**；独立 arena 仍被合在同一张表中。本轮保留睡眠开启，未修改求解器。证据和后续复测要求见[问题清单](release-review.md)。

{table(['H 组典型配置','EmbodiChain 就绪 / s','Isaac Lab 就绪 / s','耗时比 E/I','峰值显存 E/I / MiB'],typical)}

表内为每次运行的指标再取中位数；立方体 1000 arena 每端 10 次，其余每端 3 次。显存是计时窗口内进程 NVML 采样峰值，不是整卡占用。

**仍需关注的差距：**1000 个立方体 arena 的完整启动耗时为 Isaac Lab 的 {a['ready_s']/b['ready_s']:.2f} 倍。分阶段数据见后两章。这是本配置的实测差距，不直接证明求解器存在 bug，也不能由旧 Hybrid 诊断解释本次无渲染路径。

**规模表现：**从 1 增至 1000 arena，H 组就绪时间中位数增加 {a['ready_s']-base_a['ready_s']:.3f} / {b['ready_s']-base_b['ready_s']:.3f} s；两端完成 3 次重复的最高已测规模分别为 {cap['embodichain']} / {cap['isaaclab']} arena；EmbodiChain 另做一次 2896 arena 边界预检。它们是本机、当前时间预算内的观测规模，不是绝对上限，也不代表长期物理吞吐或含相机负载的表现。

**实践建议：**批量训练或并行数据生成应直接使用 `num_envs` 与原生场景复制，保持每环境工作量固定；需要图像时另测传感器数量和分辨率。两端 N=4 的跨环境重叠检查均无碰撞响应，正式运行已核验全部环境的实例、状态和惯量。

![多 arena 就绪时间与内存](../analysis/loading.png)

误差线为独立运行最小值—最大值，非置信区间。EmbodiChain 的 4000 arena 准备报错，未作完成耗时绘制；更高规模按方案停止，2896／2897 的单次边界预检另列。图表数据见 [summary.json](../analysis/summary.json) 与 [逐次 CSV](../analysis/runs.csv)。

### 协议修正与适用范围

旧实验使用 `num_envs=1` 放置 1／100／1000 个对象，不能描述本次业务。旧首帧 2.70 倍、2000 个对象超时的结论只适用于那个单场景渲染路径。[旧报告](../../reports/L-02.md)及[旧 profiling](../../reports/loading-profile-20261026.md)保留；本次更换了场景组织方式和渲染配置，**不得将新旧差异写成引擎修复或加速收益**。没有修改引擎或求解器。

本轮只测同模板复制（S），不声称覆盖逐环境不同资产；异构工作站／工作间由 L-04／L-05 另测。共享源文件不等于底层所有内存都共享。C/H 都保留 OS 和 kernel 缓存；H 增加同配置预载，C 不是磁盘／编译器全冷。解析碰撞不涉及 mesh cooking，不能将 C/H 波动解释成明确缓存收益。

### 共同条件与数据出处

GPU：{first_launch['hardware']['gpu_name']}；驱动：{first_launch['hardware']['driver']}。零重力、dt=5 ms、每环境一个动态刚体、arena 间距 0.2 m。visual 输入为相同 USD，物理碰撞为解析 box／sphere；本次不测完整 visual 渲染资源就绪。两端 arena 原生网格排列不同，状态统一到环境局部坐标。计时含模块导入、框架启动、模型准备、GPU 同步与全量状态复制至 CPU；不含输入离线生成、计时后的模型审计或录像。

正式主测共有 {len(main_attempts)} 次尝试，其中 {len(excluded)} 次与其他 GPU 计算进程重叠，原记录保留并排除出性能统计，补测后得到 {len(main_rows)} 次无其他 GPU 计算进程的样本。预载、容量与失败预检见 [数据索引](data-index.md)。原始配置、输入 SHA-256、源码快照、版本、日志、计时、20 ms 内存采样、首步状态和模型数组按 run_id 保存。统一状态取 `first_state.npy`（环境局部坐标、xyzw）；`model.npz` 为诊断读回，其 pose 格式逐运行写入内容核验表；[内容核验](../analysis/content-validation.json)与[证据清单](../analysis/evidence-manifest.json)可下载。分阶段是框架可观测生命周期边界：EmbodiChain 的初始化阶段已创建 N 个 arena，Isaac 的 arena 复制位于声明阶段，因此不能把初始化阶段等同于纯固定开销，也不将各阶段当作同名内部算法的直接性能比较。总就绪时间覆盖二者全部必要步骤。

"""
    details = []
    for i, p in enumerate(NAMES, 2):
        data = [s for s in summary if s["platform"] == p]
        tab = []
        for s in data:
            lo, hi = s["ready_s_range"]
            tab.append(
                [
                    "立方体" if s["shape"] == "cube" else "球体",
                    s["num_envs"],
                    s["cache"],
                    s["repeats"],
                    f"{s['ready_s']:.3f} [{lo:.3f}, {hi:.3f}]",
                    f"{s['rss_peak_mib']:.0f}",
                    f"{s['vram_peak_mib']:.0f}",
                ]
            )
        examples = [r for r in main_rows if r["platform"] == p]
        sample = BASE / "runs" / examples[-1]["run_id"]
        meta = json.loads((sample / "run.json").read_text())
        failures = [
            r for r in rows if r["platform"] == p and r["status"] != "completed"
        ]
        detail = f"## {i}. {NAMES[p]} 详细报告\n\n"
        if p == "embodichain":
            detail += "使用 `SimulationManagerCfg(num_envs=N, arena_space=0.2)`，仅声明一个 `RigidObjectGroup` 成员，交给 SpawnScene 创建 N 个 arena；不是循环创建 N 个独立 world。物理使用 Newton / DexUni VBD 10 次迭代、sleep 开启，主测 `renderer=no-render`。版本为 EmbodiChain `f179b555`，进程加载 P-03 冻结的 DexSim + MR !1501 parser 副本，具体包路径及版本见各 run.json。静态后台节点按 native 数组保留，不计入 N 个动态物体。\n\n"
        else:
            detail += "使用 `InteractiveSceneCfg(num_envs=N, replicate_physics=True, filter_collisions=True)`，一个 `{ENV_REGEX_NS}/Object` 声明由原生机制复制。GPU PhysX TGS 4/1，Isaac Lab `ae37b028` / Isaac Sim 6.1。头部 AppLauncher 禁用相机，保留框架原生启动；reset 内置物理更新计入准备成本。所有根状态减去对应 env_origin 后比较，四元数转为 xyzw 保存。\n\n"
        detail += (
            table(
                [
                    "形状",
                    "num_envs",
                    "缓存组",
                    "独立次数",
                    "T_ready 中位数 [最小,最大] / s",
                    "RAM 峰值中位数 / MiB",
                    "VRAM 峰值中位数 / MiB",
                ],
                tab,
            )
            + "\n\n"
        )
        m = key[p, "cube", 1000, "H", "main"]
        detail += f"1000 立方体 H 组的 p95 为 {m['ready_p95_s']:.3f} s，来自 10 个独立进程，样本量有限。入口至初始化、声明、准备、首步读回的阶段中位数依次为 {m['engine_s']:.3f}、{m['declaration_s']:.3f}、{m['prepare_s']:.3f}、{m['first_step_s']:.3f} s；各阶段中位数之和可能与总中位数不同。\n\n"
        pchecks = [c for c in checks if "_" + p + "_" in c["run_id"]]
        detail += f"全量状态核验中，最大初始位置残差为 {max(c['local_position_error_max_m'] for c in pchecks):.3g} m，最大惯量残差为 {max(c['inertia_error_max_kg_m2'] for c in pchecks):.3g} kg·m²。N=4 跨环境重叠试验保存了写入前、写入后及 20 步状态；位移／速度响应均为零。\n\n"
        if p == "embodichain":
            detail += f"### 睡眠模块容量限制\n\n实际异常为 `ValueError: Rigid sleeping adjacency exceeds the 32 MiB budget`。运行包中 `RigidBodySleep.__init__` 使用全模型 `body_count=n`，在分配 `int32[n,n]` 前检查 `4*n*n > 32*1024*1024`。2896 对应 33,547,264 字节，2897 对应 33,570,436 字节；预算为 33,554,432 字节。\n\n[2896 arena 完成记录](../runs/{boundary_ok}/run.json)、[2897 arena 报错记录](../runs/{boundary_fail}/run.json)、[实际调用栈](../runs/{boundary_fail}/console.log)、[源码位置与 SHA-256](../analysis/sleep-boundary-source.json)。这是当前冻结版本、sleep 开启时的实现限制；对多 link 机器人，更少的 arena 就可能达到同样的总刚体数，L-03～L-05 必须先核对实际 body_count。\n\n后续应评估按 arena／接触岛组织或使用稀疏邻接的睡眠数据结构，修复需另行实测。单纯关闭睡眠或放大预算不作为本轮补救；原配置及全部失败保留。\n\n"
        if failures and p == "isaaclab":
            detail += "早期 Tensor.torch 误用及相机不在 InteractiveScene 复制上下文创建的失败，属于本次适配脚本问题，已修正。它们原样留存，不计入正式主测或作为引擎加载失败的证据。\n\n"
        if failures:
            detail += (
                f"保留 {len(failures)} 次未完成尝试：\n\n"
                + table(
                    ["run_id", "状态", "最后阶段／原因"],
                    [
                        [
                            f"[{r['run_id']}](../runs/{r['run_id']}/run.json)",
                            r["status"]
                            + (
                                "（GPU 竞争，排除计时）"
                                if r["foreign_compute_pids"]
                                else ""
                            ),
                            r["failure_reason"] or r["last_stage"],
                        ]
                        for r in failures
                    ],
                )
                + "\n\n"
            )
        else:
            detail += "该平台本轮已记录运行均完成；不据此推广为任意场景无失败。\n\n"
        detail += f"原始代表运行：[配置](../runs/{sample.name}/config.json)、[计时及版本](../runs/{sample.name}/run.json)、[模型审计](../runs/{sample.name}/model-audit.json)、[模型数组](../runs/{sample.name}/model.npz)、[内存采样](../runs/{sample.name}/memory.jsonl)。完整 run_id 映射见数据索引。\n\n"
        details.append(detail)
    lead, common = header.split("### 协议修正与适用范围", 1)
    details[0] = details[0].replace(
        "## 2. EmbodiChain 详细报告\n\n",
        "## 2. EmbodiChain 详细报告\n\n### 共同方法与数据口径\n\n#### 协议修正与适用范围"
        + common.replace("### 共同条件与数据出处", "#### 共同条件与数据出处")
        + "\n### 平台实现与完整结果\n\n",
    )
    header = lead
    (reports / "L-02.md").write_text(
        header
        + "\n".join(details)
        + "原生媒体见[视频索引](../videos/index.md)。录像配置与无渲染主测分开，视频不表示加载耗时。\n"
    )
    index = "# L-02 多 arena 数据索引\n\n所有目录均是新协议；旧单场景记录不混入统计。`prepare` 为预载，`pilot` 为预检，`main` 为主测，`capacity` 为扩展，`media` 为录像。\n\n"
    index += table(
        ["运行", "平台", "形状", "num_envs", "用途／缓存", "结果"],
        [
            [
                f"[{r['run_id']}](../runs/{r['run_id']}/)",
                NAMES[r["platform"]],
                r["shape"],
                r["num_envs"],
                r["purpose"] + "/" + r["cache_mode"],
                r["status"]
                + ("（GPU 竞争，排除计时）" if r["foreign_compute_pids"] else ""),
            ]
            for r in rows
        ],
    )
    index += "\n\n[逐次表](../analysis/runs.csv) · [统计及 run_id](../analysis/summary.json) · [全量内容核验](../analysis/content-validation.json) · [证据 SHA-256](../analysis/evidence-manifest.json)\n"
    (reports / "data-index.md").write_text(index)
    failures = [r for r in rows if r["status"] != "completed"]
    capacity_failures = [
        r for r in rows if r["purpose"] == "capacity" and r["status"] != "completed"
    ]
    failed_configs = sorted(
        {
            (r["platform"], r["shape"], r["num_envs"], r["cache_mode"])
            for r in capacity_failures
        }
    )
    full_comparison = []
    for shape, n, cache, purpose in sorted(
        {
            (r["shape"], r["num_envs"], r["cache_mode"], r["purpose"])
            for r in rows
            if r["purpose"] in ["main", "capacity"]
        }
    ):
        ea = key.get(("embodichain", shape, n, cache, purpose))
        ib = key.get(("isaaclab", shape, n, cache, purpose))
        full_comparison.append(
            [
                f"{shape}, N={n}, {cache}",
                f"{ea['ready_s']:.3f}" if ea else "failed" if n == 4000 else "not run",
                f"{ib['ready_s']:.3f}" if ib else "not completed",
                f"{ea['ready_s']/ib['ready_s']:.2f}" if ea and ib else "N/A",
                (
                    f"{ea['vram_peak_mib']:.0f} / {ib['vram_peak_mib']:.0f}"
                    if ea and ib
                    else "N/A"
                ),
            ]
        )
    summary_md = "# L-02 multi-arena benchmark summary\n\nNo cameras or rendering; N is num_envs, one dynamic body per arena. C/H retain OS and kernel caches. All raw attempts are in [data-index](data-index.md).\n\n"
    summary_md += (
        "## Time & Memory\n\n"
        + table(
            [
                "Platform",
                "Shape",
                "N",
                "Cache",
                "Runs",
                "Ready median / s",
                "RAM / MiB",
                "VRAM / MiB",
            ],
            [
                [
                    s["platform"],
                    s["shape"],
                    s["num_envs"],
                    s["cache"],
                    s["repeats"],
                    f"{s['ready_s']:.3f}",
                    f"{s['rss_peak_mib']:.0f}",
                    f"{s['vram_peak_mib']:.0f}",
                ]
                for s in summary
            ]
            + [
                [
                    p,
                    shape,
                    n,
                    cache,
                    "0/3",
                    "prepare error: sleep adjacency",
                    "N/A",
                    "N/A",
                ]
                for p, shape, n, cache in failed_configs
            ],
        )
        + "\n\n"
    )
    summary_md += (
        "## Success & Other Metrics\n\n"
        + table(
            ["Scope", "Observed result", "Interpretation"],
            [
                [
                    "Main",
                    f"{len(main_rows)}/100 completed",
                    "Process completion and all-env structural audit; not a physics-accuracy pass threshold",
                ],
                [
                    "All attempts",
                    f"{len(failures)} failed attempts retained",
                    "Includes adapter preflight failures and sleep-capacity errors; see per-run ledger",
                ],
                [
                    "Isolation",
                    "N=4: zero drift and speed after overlap",
                    "No cross-arena collision response in tested case",
                ],
                ["Capacity", str(cap), "Observed scale, not absolute limit"],
            ],
        )
        + "\n\n"
    )
    summary_md += (
        "## Leaderboard\n\n"
        + table(
            ["Case", "EmbodiChain / s", "Isaac Lab / s", "E/I", "Peak VRAM E/I / MiB"],
            full_comparison,
        )
        + "\n\nNo composite rank. Full statistics include all measured configurations; renderer, robot, heterogeneous and sensor workloads are outside this matrix.\n"
    )
    (reports / "benchmark-summary.md").write_text(summary_md)
    release = f"""# L-02 多 arena 发布前问题

**优先级最高：sleep 开启时，多 arena 被全局睡眠邻接表限制。** 2896 arena 边界预检完成，2897 arena 准备报错；4000 arena 预载及三次正式尝试均出现 `Rigid sleeping adjacency exceeds the 32 MiB budget`。Isaac Lab 在 16000 arena 三次完成。

该实现使用全模型 `body_count=n`，分配前检查 `4*n*n > 32*1024*1024`，即 int32 稠密邻接表超过 32 MiB 就退出。2896 为 33,547,264 字节，2897 为 33,570,436 字节。它没有按 arena 分块，因此每个 arena 只有一个、不应跨环境碰撞的物体，也受到全局平方开销约束。这是当前版本的代码限制，不是 GPU OOM。

证据：[2896 运行](../runs/{boundary_ok}/run.json)、[2897 运行](../runs/{boundary_fail}/run.json)、[完整异常](../runs/{boundary_fail}/console.log)、[实际源码位置／哈希／条件](../analysis/sleep-boundary-source.json)。本轮没有更改求解器或关闭睡眠。

后续修复方向是按 arena／接触岛或稀疏结构保存睡眠邻接；需验证隔离、接触唤醒、支撑传播、reset 及多 link 场景。放大预算只是移动边界，不能消除全局平方开销。修复后先复测 2896／2897／4000，再完整扫描 1000～16000，并运行既有睡眠物理回归。L-03～L-05 可能更早达到总刚体数限制，正式启动前检查。

当前同条件差距：1000 立方体 arena H 组 T_ready 为 {a['ready_s']:.3f} / {b['ready_s']:.3f} s（EmbodiChain / Isaac Lab，{a['ready_s']/b['ready_s']:.2f} 倍）。属于已测性能差距，不等同于已确认 bug。阶段耗时提示先区分固定初始化与场景复制成本，再有针对性 profiling；本轮不改引擎。

容量已测至 {cap['embodichain']} / {cap['isaaclab']} arena；完整范围和失败在[逐次数据](data-index.md)。没有实测的上限、含渲染／相机负载、长期运行与机器人场景不得据此宣传。

旧“2.70 倍／2000 对象超时”保留在历史报告，但不能套用到当前业务。新旧协议同时改变 num_envs 组织与渲染需求；任何改善不能归因于引擎修复。

修复后复测：相同共同 USD、num_envs、headless 配置、计时边界、C/H 和重复数；使用新版本、新 run_id，原始结果保留。对渲染路径的修复应另做有相机多 arena 实验。
"""
    (reports / "release-review.md").write_text(release)
    manifest = []
    for parent in [BASE / "runs", BASE / "plans"]:
        for path in sorted(parent.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                manifest.append(
                    {
                        "path": str(path.relative_to(ROOT)),
                        "bytes": path.stat().st_size,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                )
    (analysis / "evidence-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    (analysis / "validation-summary.json").write_text(
        json.dumps(
            {
                "runs": len(rows),
                "content_checked": len(checks),
                "main_runs": len(main_rows),
                "main_attempts": len(main_attempts),
                "excluded_contended_main": [r["run_id"] for r in excluded],
                "main_complete": sum(r["status"] == "completed" for r in main_rows),
                "failed_attempts": len(failures),
                "foreign_compute_measurements": [
                    r["run_id"]
                    for r in rows
                    if r["purpose"] in ["main", "capacity"]
                    and r["foreign_compute_pids"]
                ],
                "raw_files": len(manifest),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
