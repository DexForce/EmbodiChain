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

"""Publish the observed stack boundaries, all exploration points and raw sources.

Run: python -m scripts.benchmark.tech_reports.reporting.p02_stack_report
"""

from __future__ import annotations

import os

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.benchmark.tech_reports.physics.experiments.p02_stack_sweep import PROFILES

__all__ = ["main"]
LABELS = {
    "dexuni10": "DexUni VBD 10；默认微动抑制",
    "physx4": "PhysX TGS 4/1",
    "dexuni100group": "DexUni VBD 100＋分组；默认微动抑制",
    "physx100": "PhysX TGS 100/1",
    "dexuni100group_no_snap": "DexUni VBD 100＋分组；关闭微动抑制",
}


def main() -> None:
    """Build tables and figures without restarting the simulators."""
    project = Path(
        os.environ.get("SIM_REPORT_ROOT", Path(__file__).resolve().parents[4])
    ).resolve()
    root = project / "outputs/benchmarks/P-02/stack-limit-10s"
    report = root / "reports"
    figures = root / "figures"
    report.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)
    boundary = json.loads((root / "boundary-results.json").read_text())
    rows = [
        json.loads(p.read_text())
        for p in sorted((root / "analysis/by-run").glob("*.json"))
    ]
    rows.sort(
        key=lambda r: (
            list(PROFILES).index(r["profile"]),
            r["layers"],
            r["duration_s"],
            r["initial"],
        )
    )
    (root / "analysis/episode-metrics.json").write_text(
        json.dumps(rows, indent=2) + "\n"
    )
    with (root / "analysis/episode-metrics.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    groups = []
    for profile, layers, duration in sorted(
        {(r["profile"], r["layers"], r["duration_s"]) for r in rows},
        key=lambda k: (list(PROFILES).index(k[0]), k[1], k[2]),
    ):
        rr = [
            r
            for r in rows
            if (r["profile"], r["layers"], r["duration_s"])
            == (profile, layers, duration)
        ]
        events = [
            r["first_geometry_event_s"]
            for r in rr
            if r["first_geometry_event_s"] is not None
        ]
        groups.append(
            dict(
                profile=profile,
                layers=layers,
                duration_s=duration,
                trials=len(rr),
                intact=sum(r["intact_full_window"] for r in rr),
                event_first_s=min(events) if events else None,
                event_last_s=max(events) if events else None,
                max_speed_m_s=max(r["max_speed_m_s"] for r in rr),
                max_mechanical_energy_gain_J=max(
                    r["max_mechanical_energy_gain_J"] for r in rr
                ),
                max_bottom_gap_mm=max(r["max_bottom_gap_mm"] for r in rr),
                max_drift_mm=max(r["max_horizontal_drift_mm"] for r in rr),
                final_chain_min=min(r["longest_ordered_chain_final"] for r in rr),
                final_chain_max=max(r["longest_ordered_chain_final"] for r in rr),
                run_ids=[r["run_id"] for r in rr],
            )
        )
    (root / "analysis/group-metrics.json").write_text(
        json.dumps(groups, indent=2) + "\n"
    )
    header = [
        "| 配置（dt=5 ms） | 10 s 内 10/10 完整的最高已测层数 | 再高 1 层保持完整 | 再高 1 层首次事件 / s |",
        "| --- | ---: | ---: | --- |",
    ]
    table = header.copy()
    details = {}
    for key in PROFILES:
        if key not in boundary:
            continue
        n = boundary[key]["highest_verified_layers"]
        if n is None:
            lower = boundary[key]["minimum_supported_layers"]
            count = boundary[key]["minimum_layer_intact"]
            table.append(
                f"| {LABELS[key]} | 在支持范围内未找到 | {lower} 层 {count}/10 | 见完整扫描 |"
            )
            details[key] = (
                f"**{LABELS[key]}：**接口最少支持 {lower} 层；该层数完整 {count}/10，当前已测范围内未找到 10 初态均完整的层数。不能据此声称所有更高层都必然倒塌，或把接口不支持的更低层写成物理失败。"
            )
            continue
        stable = next(
            g
            for g in groups
            if g["profile"] == key and g["layers"] == n and g["duration_s"] == 10
        )
        assert stable["trials"] == stable["intact"] == 10
        higher = next(
            (
                g
                for g in groups
                if g["profile"] == key
                and g["layers"] == n + 1
                and g["duration_s"] == 10
            ),
            None,
        )
        if higher:
            assert higher["trials"] == 10
            count = f"{higher['intact']}/10"
            event = f"{higher['event_first_s']:.3f}–{higher['event_last_s']:.3f}"
        else:
            count, event = "达到搜索上界", "未测更高层"
        table.append(
            f"| {LABELS[key]} | {'≥' if n == boundary[key]['search_ceiling'] else ''}{n} | {count} | {event} |"
        )
        details[key] = (
            f"**{LABELS[key]}：**{n} 层在 10 个初态、10 s 内保持层序与投影关系，全样本最大横漂 {stable['max_drift_mm']:.3f} mm、最大速度 {stable['max_speed_m_s']:.3f} m/s、底层最大离板间隙 {stable['max_bottom_gap_mm']:.3f} mm。"
        )
        if higher:
            details[
                key
            ] += f"{n+1} 层保持完整 {count}，失稳初态的首次事件在 {event} s；全部初态最终最长连续层序为 {higher['final_chain_min']}–{higher['final_chain_max']} 层，最大速度 {higher['max_speed_m_s']:.3f} m/s、相对初始机械能的最大正增量 {higher['max_mechanical_energy_gain_J']:.6f} J。"
    repeat_note = "**重复性限制：DexUni 51 层尚不是可重复的稳定上限。** 原批次 10/10 完整，但同初态录像倒塌；52 层原失稳初态在无录像复跑中也变为完整。PhysX 78／79 层录像均逐点复现原测量。上表是原批次观测边界，详见[重复性核查](../stack-limit-10s/reports/repeatability.md)。"
    details["dexuni100group"] += "\n\n" + repeat_note
    table_text = "\n".join(table)
    summary = (
        "**堆叠上限改为观察实际结构失稳。** 相同 50 mm 方块、质量 0.1 kg、μ=0.8，开启睡眠，独立进程、10 个共同初态，各观察 10 s：\n\n"
        + table_text
        + "\n\n"
        + repeat_note
        + "\n\n这是固定配置和观测窗口下的实测边界；更高层数仍可能有部分初态保持完整。不是引擎的绝对最高层数，VBD/TGS 的迭代次数也不代表等计算预算。两端保留睡眠；DexUni 同时保留默认接触微动抑制。完整扫描、失稳严重程度与录像见[堆叠上限补充报告](../stack-limit-10s/reports/stack-limit.md)。\n"
    )
    (report / "summary-fragment.md").write_text(
        summary.replace("../stack-limit-10s/", str(root) + "/")
    )
    for platform, filename, title in [
        ("embodichain", "dexuni", "### 2.6 10 s 堆叠边界"),
        ("isaaclab", "isaac", "### 3.5 10 s 堆叠边界"),
    ]:
        fragment = (
            title
            + "\n\n"
            + "\n\n".join(
                details[k]
                for k in PROFILES
                if k in details and PROFILES[k][0] == platform
            )
        )
        fragment += "\n\n层间事件是原相邻层水平投影分离或层序反转。它首先标识整柱不再完整，不自动等于全部方块同时倒下；最终连续层数、柱高、速度与录像共同描述实际塌落过程。10 s 是从释放起的完整窗口。详细逐初态数据及原始 NPZ 见[上限补充报告](../stack-limit-10s/reports/stack-limit.md)。\n"
        if (
            platform == "embodichain"
            and (figures / "stable_boundary_native.png").exists()
        ):
            fragment += f"\n![测量边界复录：DexUni 51 层倒塌，PhysX 78 层完整]({figures}/stable_boundary_native.png)\n\n图：原测量边界的同视角复录，层数不同，标题已标注。DexUni 本次录制发生倒塌，与原测量不同。完整 10 s 原片、52／79 层复录及状态核查见[视频索引](../stack-limit-10s/videos/index.md)。\n"
        (report / f"{filename}-fragment.md").write_text(
            fragment.replace("../stack-limit-10s/", str(root) + "/")
        )

    lines = [
        "# P-02 堆叠稳定层数扩展",
        "",
        "2026-10-22 · [返回合并报告](../../reports/P-02.md) · [原生视频](../videos/index.md)",
        "",
        "## 结果",
        "",
        table_text,
        "",
        repeat_note.replace("../stack-limit-10s/reports/", ""),
        "",
        "\n\n".join(details.values()),
        "",
        (report / "repeatability.md").read_text(),
        "",
        "## 方法与解释",
        "",
        METHODS,
        "",
        "## 全部探索点",
        "",
        "粗搜先测最多 3 初态，遇到事件即停止该点；10 s 候选排查也可能提前停止后续初态。每条已经启动的仿真均完成其约定时长。表中分母是实际初态数，不能把 1/1 或 3/3 与最终边界 10/10 混同，也不把自适应选点统计当成无偏倒塌概率。",
        "",
        "| 配置 | 层数 | 窗口 / s | 完整初态数 | 首次事件范围 / s | 全样本最大横漂 / mm | 最大速度 / m·s⁻¹ | 最终连续层数范围 |",
        "| --- | ---: | ---: | ---: | --- | ---: | ---: | --- |",
    ]
    for g in groups:
        event = (
            "未观察到"
            if g["event_first_s"] is None
            else f"{g['event_first_s']:.3f}–{g['event_last_s']:.3f}"
        )
        lines.append(
            f"| {LABELS[g['profile']]} | {g['layers']} | {g['duration_s']:g} | {g['intact']}/{g['trials']} | {event} | {g['max_drift_mm']:.3f} | {g['max_speed_m_s']:.3f} | {g['final_chain_min']}–{g['final_chain_max']} |"
        )
    lines += [
        "",
        "## 分析图与数据",
        "",
        f"![10 s 堆叠边界]({figures}/stack-boundaries.png)",
        "",
        "横轴按已测层数等间距排列，点旁标注完整初态数／实际初态数。不同配置分列；所有新试验均观察 10 s，粗搜和边界核验只改变初态数量。",
        "",
        "[逐初态 CSV](../analysis/episode-metrics.csv) · [逐初态 JSON](../analysis/episode-metrics.json) · [分组数据及 run_id](../analysis/group-metrics.json) · [逐点执行记录](../search-ledger.jsonl) · [最终边界](../boundary-results.json)",
        "",
        "每条 CSV 记录含原始 NPZ 绝对路径与 SHA-256；相邻 `_state.npy` 为未压缩逐步状态。每次运行的 `run.json`、`config/`、标准输出和错误日志位于 `runs/<run_id>/`。10 s 新协议、旧无睡眠长时记录、失败尝试与录制分别保存，不覆盖原 P-02 数据。",
        "",
        "[采样方式校准](../../stack-limit/analysis/state-only-calibration.json)在先前关闭睡眠配置下核对了 PhysX 1 层和 10 层初态 0：不导出接触力的前 12 s 状态与力采样运行逐项一致；这不表示当前开启睡眠的轨迹等同旧轨迹。上限试验两平台均只读状态，不虚构 DexUni 接触力。",
        "",
        "[新旧几何指标交叉核对](../../stack-limit/analysis/metric-crosscheck.json)覆盖旧基线与分组记录。图像另存 [SVG](../figures/stack-boundaries.svg) 和 [PDF](../figures/stack-boundaries.pdf)；绘图数据为上述 group-metrics.json，未用分析图代替原生录像。",
        "",
        "复现：在项目目录使用 embodichain 环境运行 `python -m scripts.benchmark.tech_reports.physics.experiments.p02_stack_sweep`。它逐平台启动对应环境，读取已完成记录后补做缺少项。离线重建使用 `python -m scripts.benchmark.tech_reports.reporting.p02_stack_report`。",
        "",
    ]
    (report / "stack-limit.md").write_text(
        "\n".join(lines).replace("../stack-limit-10s/reports/", "")
    )
    old_index = root.parent / "reports/data-index.md"
    marker = "\n## 10 s 堆叠边界与摩擦解释扩展\n"
    text = old_index.read_text().split(marker)[0]
    text += (
        marker
        + "\n[10 s、保留睡眠的堆叠报告](../stack-limit-10s/reports/stack-limit.md) · [新增运行与文件哈希](../stack-limit-10s/reports/data-index.md) · [边界录像](../stack-limit-10s/videos/index.md) · [摩擦实现分析](friction-review.md)。\n\n原 300 条分析与 124 次尝试、原证据清单及源码快照保留。新源码、输入和原始状态由扩展目录单独归档；旧无睡眠 12／60 s 搜索留在 `../stack-limit/`，不混入新协议。原始文件保留核对见[original-preservation.json](../stack-limit-10s/analysis/original-preservation.json)。\n"
    )
    old_index.write_text(text)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    profiles = [k for k in PROFILES if k in boundary]
    fig, axes = plt.subplots(len(profiles), 1, figsize=(9, 8), constrained_layout=True)
    short = [
        "DexUni VBD 10 / default snap",
        "PhysX TGS 4/1",
        "DexUni VBD 100 / grouping / default snap",
        "PhysX TGS 100/1",
        "DexUni VBD 100 / grouping / no snap",
    ]
    for i, key in enumerate(profiles):
        gg = [g for g in groups if g["profile"] == key and g["duration_s"] == 10]
        ax = axes[i]
        for j, g in enumerate(gg):
            ax.scatter(
                j,
                g["intact"] / g["trials"],
                color="#207a70" if g["intact"] == g["trials"] else "#ba5142",
                s=50,
            )
            ax.annotate(
                f"{g['intact']}/{g['trials']}",
                (j, g["intact"] / g["trials"]),
                xytext=(0, -14),
                textcoords="offset points",
                fontsize=8,
                ha="center",
            )
        ax.set(title=short[i], ylabel="Intact / tested", ylim=(-0.18, 1.15))
        ax.set_yticks([0, 0.5, 1])
        ax.grid(alpha=0.15)
        if gg:
            ax.set_xlim(-0.6, len(gg) - 0.4)
            ax.set_xticks(range(len(gg)), [g["layers"] for g in gg])
    axes[-1].set_xlabel("Tested stack layers (50 mm cubes; categorical spacing)")
    for ext in ["png", "svg", "pdf"]:
        fig.savefig(figures / f"stack-boundaries.{ext}", dpi=180)
    plt.close(fig)
    print(
        json.dumps(
            {
                "trials": len(rows),
                "groups": len(groups),
                "profiles_complete": list(boundary),
                "report": str(report / "stack-limit.md"),
            },
            ensure_ascii=False,
        )
    )


METHODS = """方块边长 50 mm、质量 0.1 kg，所有方块均为动态刚体，μs=μd=0.8；支撑板 3×3 m、顶面 z=0.2 m。dt=5 ms、1 子步，无外力干预。每层初始 xy 扰动 ±0.5 mm、yaw ±1°、层间间隙 0.1 mm。相同初态文件供两平台读取：原前 10 层保持原值，更高层用固定种子扩展到 128 层；低塔取同一数组前缀。

**上限协议只观察释放后的完整 10 s，并保留睡眠。** DexUni 设置 `rigid_enable_sleep=True`，使用内置自由刚体接触岛睡眠；PhysX 不再写入零睡眠阈值，使用 Schema 默认值并读回记录。两者休眠判据不同，不宣称阈值或内部算法相同。DexUni 默认接触微动抑制也保留；没有手工锁定位姿、添加支撑约束或修改求解器。旧的无睡眠 12／60 s 扫描已停止、单独留档，不参与本表统计。

PhysX 场景读回的睡眠阈值约为 `5e-5`（质量归一化动能）；DexUni 的内置规则使用接触岛、等待计时器和累计速度，不能把其代码里的能量阈值直接当成相同定义。原始配置与读回在各运行 `run.json`，相关安装源码与 Schema 已[冻结存档](../analysis/sleep-evidence/manifest.json)。本轮只保留／启用内置机制，不另外调大睡眠阈值。

允许睡眠不等于全部刚体实际入睡。[单层状态观察](../analysis/rest-state-observations.json)保留实际停止变化时刻与末速度；它是状态数组观察，没有冒充后端睡眠标志的直接读回。

先用最多 3 个初态倍增／二分查找候选，再补齐 10 个初态核对边界及相邻更高 1 层；每条均为 10 s，每初态独立进程。候选排查优先执行此前较易失稳的初态以缩短运行时间；最终完整候选与相邻一层仍覆盖全部 10 个初态。同条件粗搜轨迹直接计入该层最终 10 初态，不重复增加样本数。层数搜索上界为 128，到达上界仍完整时只报告下限。自适应搜索按局部随层数变差的假设缩小范围，不证明所有未测层数都更差；结果是固定输入集和时间窗口下的观测边界。

本轮粗搜已使用二分，早期 10 初态复核仍曾逐层回退。当前脚本已将复核也改成二分：10/10 完整则提高下界，出现事件则降低上界，最后测齐相邻两层。修改时本轮四组已测完，因此不重跑既有轨迹、不改写执行历史；实际选点以 `search-ledger.jsonl` 为准。

**不设横漂或倾角合格线。** 原相邻层水平投影完全分离或层序反转时，记录结构事件。轻微晃动本身不记倒塌。事件后保留完整轨迹，记录最长连续层序、塔高、速度、机械能变化与底层离板间隙，并以原生画面区分局部掉块、整体倒塌和异常弹飞；不会仅凭某个速度数值自动叫作“炸飞”。10/10 是该组实际观察数量，不是预先设置的任意精度容差。

默认预算和增加预算配置均保留，VBD 与 TGS 的迭代数字不代表相等计算成本。接触分组接口支持 5–704 个自由刚体，分组搜索从 5 层开始；普通 VBD 与 TGS 从 1 层开始。最初单层分组初始化错误留在旧数据中，不作物理倒塌记录。PhysX 本次为 GPU TGS，不代表 PGS。

两平台统一采样逐步位姿和速度，本次不导出接触力，不填零替代缺失数据。首次研究中“未启用冻结”的描述已经更正：关闭睡眠仍可保留安装版默认接触微动抑制；新协议则明确保留这两项机制。"""


if __name__ == "__main__":
    main()
