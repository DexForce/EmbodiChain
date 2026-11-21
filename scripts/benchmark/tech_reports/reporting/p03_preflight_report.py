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

"""Rebuild the P-03 asset preflight evidence without a simulator.

Run: python -m scripts.benchmark.tech_reports.reporting.p03_preflight_report
"""

from __future__ import annotations

import os
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def main() -> None:
    root = Path(__file__).resolve().parents[4] / "outputs/benchmarks/P-03"
    reports = root / "reports"
    analysis = root / "analysis"
    figures = root / "figures"
    for p in [reports, analysis, figures]:
        p.mkdir(parents=True, exist_ok=True)
    runs = [
        (p.parent, json.loads(p.read_text()))
        for p in sorted((root / "runs").glob("*/run.json"))
    ]
    rows = []
    for folder, meta in runs:
        item = {
            "run_id": folder.name,
            "platform": meta["platform"],
            "asset": meta["asset"],
            "status": meta["status"],
            "failure": meta.get("failure"),
            "raw_state_file": None,
        }
        state = folder / "joint_state.npy"
        if state.exists():
            a = np.load(state)
            assert np.isfinite(a).all()
            names = meta["joint_names"]
            left = next(i for i, n in enumerate(names) if n.endswith("finger_joint1"))
            right = next(i for i, n in enumerate(names) if n.endswith("finger_joint2"))
            item.update(
                raw_state_file=str(state.relative_to(root)),
                samples=len(a),
                dt_s=0.005,
                finger_1_final_m=float(a[-1, 0, left]),
                finger_2_final_m=float(a[-1, 0, right]),
                max_finger_difference_m=float(
                    np.max(np.abs(a[:, 0, left] - a[:, 0, right]))
                ),
                closure_command_applied=len(a) == 301,
            )
        rows.append(item)
    (analysis / "asset-preflight.json").write_text(
        json.dumps(
            {
                "experiment": "P-03",
                "scope": "asset and unloaded joint-drive preflight only; no grasp/contact experiment",
                "joint_state_layout": "[sample, env, 18]; q then qdot in run.json joint_names order; m or rad, m/s or rad/s",
                "closure_command": "In 301-sample probes, leader target changes 0.04 -> 0.015 m before step 100 (t=0.495 s); follower drive stiffness/damping are zero. 101-sample probes only hold the initial target.",
                "runs": rows,
            },
            indent=2,
        )
        + "\n"
    )
    final = {}
    for row in rows:
        if (
            row["status"] == "completed"
            and row.get("closure_command_applied")
            and row["asset"].endswith("franka_mimic.usd")
        ):
            final[row["platform"]] = row
    assert set(final) == {"embodichain", "isaaclab"}
    columns = [
        "time_s",
        "embodichain_leader_m",
        "embodichain_follower_m",
        "isaaclab_leader_m",
        "isaaclab_follower_m",
    ]
    data = [np.arange(301) * 0.005]
    for platform in ["embodichain", "isaaclab"]:
        a = np.load(root / final[platform]["raw_state_file"])
        data.extend([a[:, 0, 7], a[:, 0, 8]])
    with (analysis / "factory-fingers.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(zip(*data))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 2, figsize=(9, 3), sharex=True, sharey=True)
    for ax, platform, label in zip(
        axs,
        ["embodichain", "isaaclab"],
        ["EmbodiChain / DexUni", "Isaac Lab / PhysX TGS"],
    ):
        a = np.load(root / final[platform]["raw_state_file"])
        ax.plot(
            np.arange(len(a)) * 0.005, a[:, 0, 7] * 1000, label="Leader", linewidth=1.8
        )
        ax.plot(
            np.arange(len(a)) * 0.005,
            a[:, 0, 8] * 1000,
            label="Follower",
            linewidth=1.8,
            linestyle="--",
        )
        ax.axvline(0.495, color="gray", linewidth=0.8, linestyle=":")
        ax.set(title=label, xlabel="Time (s)", ylim=(10, 43))
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    axs[0].set_ylabel("Finger displacement (mm)")
    fig.tight_layout()
    for ext in ["png", "svg", "pdf"]:
        fig.savefig(figures / f"factory-finger-closure.{ext}", dpi=160)
    plt.close(fig)
    d = final["embodichain"]
    i = final["isaaclab"]
    report = f"""# P-03 资产与驱动预检

2026-10-22。**已找到本地 Franka 资产，但尚未验证出能按原 Spec 在两端保留同一模型的执行路径，主测暂停，等待用户指示。** 这是资产预检记录，不是 P-03 抓取性能报告。

## 本轮发现

1. `/home/oem/isaacsim_assets/Assets/Isaac/5.1/Isaac/IsaacLab/Factory/franka_mimic.usd`：几何及 USD 依赖存在；两端均能加载并推进。该文件使用 PhysX mimic 扩展连接从动指。当前 EmbodiChain 导入后没有识别该联动；闭合测试确认从动指不随主指闭合。
2. `/home/oem/.cache/embodichain_data/extract/Franka/Panda/PandaWithHand.urdf`：22 个网格引用均存在；文件内模型实际名为 `fr3`，不能仅凭目录名当作 Panda。EmbodiChain 可导入 URDF mimic，空载闭合末端两指为 15.000／15.023 mm。Isaac Lab 的 URDF 导入扩展未能启用，Kit 报缺少 `omni.kit.pip_archive`，依赖链包含 `omni.kit.telemetry-0.5.2`。没有安装或替换依赖。
3. 本地旧版 `Robots/FrankaEmika/panda_instanceable.usd` 使用自动质量／惯量；SkillGen URDF 的部分惯量需要核查。没有把这些版本与 Factory／FR3 混用为共同模型。静态核查见 [资产清单](../analysis/asset-audit.json)。

停止依据是**现有两条导入路径未能满足共同模型要求**，不是资产文件不存在，也不是已证明 P-03 抓取失败。没有改成双指独立驱动、替换求解器或修改求解器源码。

## 闭合测试证据

使用同一 Factory USD、N=1、dt=5 ms，机械臂保持初始目标；前 0.495 s 双指目标为 40 mm，然后只改变主指目标为 15 mm。从动指驱动 Kp／Kd 均为 0，依靠源资产 mimic 联动。主指 Kp=1000 N/m、Kd=20 N·s/m、力上限 40 N；机械臂 Kp=120 N·m/rad、Kd=15 N·m·s/rad。测试持续 1.5 s，未放置抓取物体。两侧求解参数没有作为统一性能预算，本测试不比较速度或接触精度。

| 平台 | 末帧主指位移 | 末帧从动指位移 | 数据出处 |
| --- | ---: | ---: | --- |
| EmbodiChain / DexUni | {d['finger_1_final_m']*1000:.3f} mm | {d['finger_2_final_m']*1000:.3f} mm | [run.json](../runs/{d['run_id']}/run.json)、[原始状态](../{d['raw_state_file']}) |
| Isaac Lab / PhysX TGS | {i['finger_1_final_m']*1000:.3f} mm | {i['finger_2_final_m']*1000:.3f} mm | [run.json](../runs/{i['run_id']}/run.json)、[原始状态](../{i['raw_state_file']}) |

![Factory USD 空载双指闭合状态曲线](../figures/factory-finger-closure.png)

图为原始关节状态的离线曲线，不是仿真截图或视频。绘图数据见 [CSV](../analysis/factory-fingers.csv)，另有 [SVG](../figures/factory-finger-closure.svg)／[PDF](../figures/factory-finger-closure.pdf)。

## 已保留的记录与范围

共 {len(runs)} 次预检尝试，{sum(x['status']=='completed' for x in rows)} 次完成各自的小测试；其余记录包括脚本接入错误和实际扩展依赖错误。已修正的数组／关键字参数问题不能算作平台物理能力缺陷。所有尝试、失败堆栈、命令、源码快照、模型读回及可取得的逐步状态均保留。完整索引：[asset-preflight.json](../analysis/asset-preflight.json)、[原始文件 SHA-256](../analysis/raw-file-manifest.json)。Isaac Kit 关闭时可能返回退出码 0，因此以 `run.json` 的状态及是否有完整数组判断运行完成。

已基于 Factory USD 生成 [10 组候选初态与关节指令](../prepared/factory-v1/commands.npz)，只用于准备验证，尚未成为正式共享输入；不可直接用作主测。其来源／关节链／离线 IK 检查保留在同目录。

未执行：真实抓取、抬升／保持／外力／放回、两形状×10 初态质量矩阵、吞吐矩阵、同视角原生视频。没有形成 P-03 比较结论。

## 后续入口

等待用户指定处理路径：修复／补齐 Isaac URDF 转换环境，或处理 EmbodiChain 对 Factory USD mimic 的导入，再复查相同源资产的实际质量、惯量、关节、碰撞和联动。随后才做真实接触 N=1、外力校准及主测。当前没有运行中的进程。

预检脚本：[p03_asset_probe.py](../../../../scripts/benchmark/tech_reports/physics/experiments/p03_asset_probe.py)。离线重建：[p03_preflight_report.py](../../../../scripts/benchmark/tech_reports/reporting/p03_preflight_report.py)。本次具体命令在各运行的 `launch.json`；复跑必须使用新输出目录。
"""
    (reports / "asset-preflight.md").write_text(report)
    text = [
        "# P-03 资产预检摘要",
        "",
        "仅限资产导入和空载关节测试，P-03 主测未执行。不以预检完成次数作为抓取成功率。",
        "",
        "## Time & Memory",
        "",
        "| platform | cost_time_ms | cpu_delta_mb | gpu_delta_mb | peak_gpu_mb |",
        "| --- | --- | --- | --- | --- |",
        "| EmbodiChain | 未测 | 未测 | 未测 | 未测 |",
        "| Isaac Lab | 未测 | 未测 | 未测 | 未测 |",
        "",
        "## Success & Other Metrics",
        "",
        "| configuration | success_rate | observed_result |",
        "| --- | --- | --- |",
    ]
    for row in rows:
        result = (
            row["failure"]
            if row["status"] == "failed"
            else f"完成 {row['samples']} 个状态样本；非抓取测试"
        )
        text.append(f"| {row['run_id']} | 不适用：预检 | {result} |")
    text += [
        "",
        "## Leaderboard",
        "",
        "| configuration | rank | current_scope |",
        "| --- | --- | --- |",
        "| Factory USD / EmbodiChain DexUni | 不设排名 | 加载及空载推进完成，从动指未跟随闭合 |",
        "| Factory USD / Isaac Lab TGS | 不设排名 | 加载及空载闭合完成 |",
        "| cached FR3 URDF / EmbodiChain DexUni | 不设排名 | 加载及空载闭合完成；未验证有载抓持 |",
        "| cached FR3 URDF / Isaac Lab | 不设排名 | 转换扩展依赖缺失；未进入物理执行 |",
        "",
        "原始状态、错误归因与来源见 [预检记录](asset-preflight.md)。",
    ]
    (reports / "benchmark-summary.md").write_text("\n".join(text) + "\n")
    print(reports / "asset-preflight.md")


if __name__ == "__main__":
    main()
