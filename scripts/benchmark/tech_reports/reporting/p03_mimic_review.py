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

"""Rebuild the frozen P-03 MR 1501 readiness assessment without a simulator.

Run: python -m scripts.benchmark.tech_reports.reporting.p03_mimic_review
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
    out = root / "analysis/mr1501"
    out.mkdir(parents=True, exist_ok=True)
    cases = [
        ("20261022_195026_factory_mimic_probe_embodichain", "修复前 DexUni / 阶跃"),
        ("20261022_194918_factory_mimic_probe_isaaclab", "Isaac TGS / 阶跃"),
        ("20261023_113254_mr1501_factory_mimic_embodichain", "补丁未生效 / 基线复跑"),
        (
            "20261023_113349_mr1501_v2_factory_mimic_embodichain",
            "MR 1501 修复 / DexUni 阶跃",
        ),
        ("20261023_113556_mr1501_p03_smooth_embodichain", "MR 1501 修复 / DexUni 平滑"),
        ("20261023_113610_mr1501_p03_smooth_isaaclab", "Isaac 启动缺环境标志"),
        ("20261023_113641_mr1501_p03_smooth_isaaclab", "Isaac TGS / 平滑"),
    ]
    metrics = []
    hashes = []
    for run_id, label in cases:
        folder = root / "runs" / run_id
        meta = json.loads((folder / "run.json").read_text())
        row = dict(run_id=run_id, label=label, status=meta["status"])
        state_path = folder / "joint_state.npy"
        if state_path.exists():
            state = np.load(state_path)[:, 0]
            assert np.isfinite(state).all()
            names = meta["joint_names"]
            leader = names.index("panda_finger_joint1")
            follower = names.index("panda_finger_joint2")
            t = np.arange(len(state)) * 0.005
            q = state[:, [leader, follower]].astype(np.float64)
            diff_mm = np.abs(q[:, 0] - q[:, 1]) * 1000
            row.update(
                samples=len(state),
                end_s=float(t[-1]),
                final_leader_mm=float(q[-1, 0] * 1000),
                final_follower_mm=float(q[-1, 1] * 1000),
                max_difference_mm=float(diff_mm.max()),
                max_difference_time_s=float(t[diff_mm.argmax()]),
                final_difference_mm=float(diff_mm[-1]),
                maximum_finger_speeds_m_s=np.abs(
                    state[:, [len(names) + leader, len(names) + follower]]
                )
                .max(axis=0)
                .tolist(),
            )
            if len(state) == 2001:
                row["hold_3_to_9_s_max_difference_mm"] = float(
                    diff_mm[(t >= 3) & (t <= 9)].max()
                )
            with (out / f"{run_id}.csv").open("w") as f:
                writer = csv.writer(f)
                writer.writerow(["time_s", "leader_m", "follower_m", "difference_mm"])
                writer.writerows(zip(t, q[:, 0], q[:, 1], diff_mm))
        else:
            row["failure"] = meta.get("failure")
        metrics.append(row)
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                hashes.append(
                    dict(
                        path=str(path.relative_to(root)),
                        bytes=path.stat().st_size,
                        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    )
                )
    result = dict(
        scope="MR 1501 parser backport readiness; unloaded gripper only",
        metrics=metrics,
        raw_files=hashes,
        exclusions={
            cases[2][
                0
            ]: "git apply silently skipped paths; parser SHA equals baseline. This is not a test of the fix.",
            cases[5][
                0
            ]: "No simulation: missing OMNI_KIT_ACCEPT_EULA inherited by the older P-02 launcher. Retried with the existing flag.",
        },
    )
    (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")

    table = []
    for m in metrics:
        if "final_leader_mm" in m:
            table.append(
                f"| {m['label']} | {m['end_s']:.1f} | {m['final_leader_mm']:.3f} / "
                f"{m['final_follower_mm']:.3f} | {m['max_difference_mm']:.3f} | "
                f"{m.get('hold_3_to_9_s_max_difference_mm', float('nan')):.6f} |"
            )
        else:
            table.append(f"| {m['label']} | 未运行 | 未测量 | 未测量 | 未测量 |")
    table_text = "\n".join(table).replace("nan", "不适用")
    source_links = "\n".join(
        f"- {label}：[配置与状态](../runs/{run_id}/run.json)、"
        f"[启动命令](../runs/{run_id}/launch.json)、[日志](../runs/{run_id}/console.log)"
        + (
            f"、[逐步关节数据](../runs/{run_id}/joint_state.npy)。"
            if (root / "runs" / run_id / "joint_state.npy").exists()
            else "。"
        )
        for run_id, label in cases
    )
    text = f"""# P-03：USD mimic 修复复验与续做条件

2026-10-23。**可以继续 P-03 的 N=1 真实抓取试跑。USD mimic 关系丢失这个阻塞已解除；本次未执行带载抓取、扰动、性能矩阵或录像，不表示 P-03 已完成。**

## MR 与验证范围

修复实际为 [MR !1501](http://192.168.3.16/Engine/dexverse/-/merge_requests/1501)，提交 `8b04699adb57ceb0e106e6cfcaee7f0909087176`。用户给出的 !1475 是流体耦合原型，不是本次修复。!1501 唯一生产代码修改是 USD parser：按组合后的 prim 路径解析 mimic，处理符号与单位，再交给既有原生约束创建路径。没有改求解器。读取时 MR 尚未合并，流水线仍在运行，不能当作 CI 已通过。

MR 的真实 Franka 动态测试覆盖 PhysX CPU／Direct GPU，并不覆盖 DexUni。此次通过 EmbodiChain 公共接口补充了 DexUni 实测。实验 Conda 的已安装 parser 仍是旧版，因此将 **MR 的原始解析补丁**应用到进程专用包副本，沿用原实验 native 库、Newton 1.6.0、Warp 1.17.0 与 EmbodiChain `f179b5558`。这是基线加解析补丁，不是完整 MR head 的安装验收，也未替换共享 Conda 包。

身份与复现：[运行依赖及逐文件哈希](../runtime/mr1501-parser-backport-v2/manifest.json)、[原始补丁](../runtime/mr1501-parser-backport-v2/mr1501.patch)、[补丁应用记录](../runtime/mr1501-parser-backport-v2/patch-apply.log)、[MR 元数据快照](../runtime/mr1501-parser-backport-v2/mr1501.json)。后续调用必须保留 `launch.json` 中的进程级 PYTHONPATH；直接启动原 Conda 环境仍会使用未修复的 parser。

## 方法与结果

共同源资产为本机 `Isaac/IsaacLab/Factory/franka_mimic.usd`，SHA-256 `86d0017a093669b222a8c640bf581cc3e3715035aa7c596600f478da011f6c8e`。固定基座、N=1、dt=5 ms；主指 Kp=1000 N/m、Kd=20 N·s/m、力上限 40 N，从指驱动刚度／阻尼均为零。修复后运行元数据已包含 follower 8 → leader 7、倍率 1、偏移 0。

阶跃试验复用原 1.5 s 预检：在 t=0.495 s 后一次性将主指目标从 40 mm 改为 15 mm。平滑试验读取原候选输入初态 0 的夹爪命令，按 20 ms 零阶保持执行：1–2 s 平滑闭合，2–9 s 保持，9–10 s 平滑张开。仅夹爪目标参与重放，机械臂仍保持原预检姿态，没有目标物体或外力。平滑输入来自 [commands.npz](../prepared/factory-v1/commands.npz)，每次运行另存了副本与哈希。

### Time & Memory

| 范围 | cost_time_ms | cpu_delta_mb | gpu_delta_mb | peak_gpu_mb |
| --- | --- | --- | --- | --- |
| 全部预检配置 | 未测量：本轮不测速 | 未测量 | 未测量 | 未测量 |

### Success & Other Metrics

位置与差值均为关节坐标，不是物体尺寸或夹持滑移。无抓取成功率定义。

| 配置／尝试 | 物理窗口 s | 末帧主指 / 从指 mm | 全程最大双指差 mm | 3–9 s 保持期最大差 mm |
| --- | ---: | ---: | ---: | ---: |
{table_text}

修复前从指停在约 40 mm；修复后能闭合并保持约 15 mm，也能重新张开。阶跃输入下 DexUni 最大瞬时差为 21.316 mm，出现在 t=0.515 s；平滑输入下降到 0.235 mm，保持期约 0.023 mm。保留阶跃结果，不能只凭末帧位置宣称所有输入下均紧密同步。

### Leaderboard

本预检不设综合排名，也不能从空载结果推断抓取质量或吞吐。

| 配置 | 已确认 | 比较限制／下一步 |
| --- | --- | --- |
| 修复前 DexUni（含补丁未生效的复跑） | USD mimic 缺失，从指不跟随 | 仅历史缺陷基线 |
| MR 1501 解析补丁＋DexUni | mimic 元数据和空载闭合／张开恢复 | 原生耦合瞬态、实际限速及带载保持待核对 |
| Isaac Lab / PhysX TGS | 同一源资产的空载闭合／张开可运行 | 本轮未执行带载抓取；启动缺环境标志的一次尝试没有物理结果 |

## 主测前需要明确的参数

目前两端实际速度上限不同：EmbodiChain 预检读回每轴约 1e6；Isaac 沿用源资产，机械臂前四轴约 2.175 rad/s、后三轴约 2.610 rad/s，主指 0.05 m/s、从指 0.04 m/s。平滑试验 DexUni 两指最大实测速度约 0.0514／0.0642 m/s，Isaac 约 0.0400／0.0400 m/s。因此上述差值是各自当前配置的观察，**不能作为同限速条件下的求解器精度排名**；阶跃差异也不能全部归因于 mimic 求解。

下一步先通过公共关节驱动配置显式统一源资产的速度上限，记录实际读回及执行效果，再核对碰撞、质量／惯量、TCP 与质心外力，进行 N=1 立方体真实夹持—抬升—保持—扰动—放回。确认后再冻结质量矩阵、性能输入及同视角原生录像。当前不需要改动求解器源码。

首次本地补丁装载时，`git apply` 因目录前缀跳过文件却返回 0；该次 parser 哈希仍与基线相同，已单列为基线复跑。第二次用明确目录应用补丁并核对函数与哈希后才得到修复结果。Isaac 首次平滑尝试缺少旧批次入口已有的启动环境标志；补回同一标志后重跑。两次尝试及日志均保留。

## 原始数据与复算

`joint_state.npy` 布局为 `[sample, 1, 18]`，前 9 项为 q、后 9 项为 qdot，顺序见各运行 `joint_names`；位置单位为 m 或 rad，速度为 m/s 或 rad/s。差值为逐物理步 `abs(q1-q2)`，保持窗为闭区间 3–9 s，没有平滑或删点。运行是否完成与任务成功不同。

{source_links}

[指标、失败分类和原始文件哈希](../analysis/mr1501/results.json)；同目录 CSV 为离线复算曲线。复算入口：`python -m scripts.benchmark.tech_reports.reporting.p03_mimic_review`。原 [2026-10-22 预检](asset-preflight.md)及证据不覆盖。
"""
    path = root / "reports/mimic-mr1501-readiness.md"
    path.write_text(text)
    print(path)


if __name__ == "__main__":
    main()
