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

"""Write the required three-table P-02 benchmark summary without pass/fail gates."""

from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    metrics = json.loads((root / "analysis/main/episode-metrics.json").read_text())
    runs = []
    for p in sorted((root / "runs").glob("*/run.json")):
        m = json.loads(p.read_text())
        if m["phase"] in [
            "main",
            "supplement",
            "independent",
            "independent_supplement",
        ]:
            m["artifact_prefix"] = "../runs"
            runs.append(m)
    stack_metrics = root / "stack-limit-10s/analysis/episode-metrics.json"
    if stack_metrics.exists():
        for p in sorted((root / "stack-limit-10s/runs").glob("*/run.json")):
            m = json.loads(p.read_text())
            if m["phase"] != "capture":
                m["artifact_prefix"] = "../stack-limit-10s/runs"
                runs.append(m)
        for r in json.loads(stack_metrics.read_text()):
            metrics.append(
                dict(
                    run_id=r["run_id"],
                    xy_drift_full_max_mm=r["max_horizontal_drift_mm"],
                    geometric_event_first_s=r["first_geometry_event_s"],
                )
            )
    lines = [
        "# P-02 Benchmark 摘要",
        "",
        "本项测摩擦、接触和堆叠质量，不作为主测速。主测／补测的全部运行列在下表；失败和重跑分别保留。预检、录制和原始逐步数组见 [数据索引](data-index.md)，技术解释见 [合并报告](P-02.md)。",
        "",
        "`cost_time_ms` 仅为含逐步状态读回、接触采样的单条完整轨迹诊断耗时中位数；启动／准备／重置除外，NPZ 压缩在计时外。两平台采样负担不同，不能由此算引擎速度比。没有标准延迟／吞吐或显存增量测量。",
        "",
        "## Time & Memory",
        "",
        "| ID／配置／场景 | 阶段／状态 | cost_time_ms | cpu_delta_mb | gpu_delta_mb | peak_gpu_mb | 进程峰值 RSS / MiB | 完成初态数 |",
        "| --- | --- | ---: | --- | --- | --- | ---: | ---: |",
    ]
    for i, m in enumerate(runs, 1):
        times = m.get("diagnostic_wall_s_per_episode", [])
        timing = f"{np.median(times)*1000:.2f}" if times else "未测"
        label = f"[C{i:02d}]({m['artifact_prefix']}/{m['run_id']}/run.json) {m.get('solver','未初始化')} · {m['scene']['id']} · {m['dt_s']*1000:g} ms · {m['duration_s']:g} s"
        lines.append(
            f"| {label} | {m['phase']} / {m['status']} | {timing} | 未测 | 未测 | 未测 | {m.get('process_peak_rss_mib',float('nan')):.1f} | {m['completed_episodes']} |"
        )
    lines += [
        "",
        "## Success & Other Metrics",
        "",
        "均值来自每条轨迹的指定指标；跨初态范围在合并报告与 CSV 中保留。事件数按原相邻层水平投影分离或层序反转统计，包含 0–2 s；它不是自定精度合格线。",
        "",
        "| ID | success_rate | 加速度相对误差均值 / % | 前 1 s 最大沿坡位置误差均值 / mm | 堆叠全程最大横漂均值 / mm | 几何事件初态数 | 说明 |",
        "| --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for i, m in enumerate(runs, 1):
        rows = [r for r in metrics if r["run_id"] == m["run_id"]]
        values = []
        for field in [
            "acceleration_relative_error_pct",
            "position_max_error_1s_mm",
            "xy_drift_full_max_mm",
        ]:
            x = [r[field] for r in rows if r.get(field) is not None]
            values.append(f"{np.mean(x):.6g}" if x else "不适用／未取得")
        event = (
            f"{sum(r.get('geometric_event_first_s') is not None for r in rows)}/{len(rows)}"
            if rows and m["scene"]["kind"] == "stack"
            else "不适用／未取得"
        )
        note = m.get("failure", "实际误差与事件；无统一质量门槛").replace("|", "/")
        lines.append(
            f"| C{i:02d} | 不适用：连续量实验 | {' | '.join(values)} | {event} | {note} |"
        )
    lines += [
        "",
        "## Leaderboard",
        "",
        "不设综合排名；VBD 迭代与 TGS 位置／速度迭代的工作量不等价。以下只在明确条件下列实际表现，未完成的窗口不作为较快或较优的样本。",
        "",
        "| ID／场景 | 配置 | 窗口内最大穿透均值 / mm | 窗口内穿透 P95 均值 / mm | 名次 |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for i, m in enumerate(runs, 1):
        rows = [r for r in metrics if r["run_id"] == m["run_id"]]
        vals = []
        for field in ["penetration_max_mm", "penetration_p95_mm"]:
            x = [r[field] for r in rows if r.get(field) is not None]
            vals.append(f"{np.mean(x):.6g}" if x else "未取得")
        lines.append(
            f"| C{i:02d} / {m['scene']['id']} | {m.get('solver','未初始化')} / {m['dt_s']*1000:g} ms | {' | '.join(vals)} | 不设综合名次 |"
        )
    lines += [
        "",
        "穿透窗口：理想滑动斜面 0–1 s，理想静止斜面 0–10 s；堆叠这里使用 2–12 s，并在详细稿另保留 0–12 s 最大值。穿透仅指箱体间和测试支撑板的几何交叠，不把接触缓冲距离或离开支撑后的背景地面阶段混入。",
        "",
        "后续堆叠上限扩展在运行行标明 10 s，且两平台保留睡眠，使用状态采样，不导出接触力；该扩展未重算几何穿透，Leaderboard 对应列保持未取得。全部层数、10 初态边界与相邻层数的完整数量见[上限报告](../stack-limit-10s/reports/stack-limit.md)。DexUni 51／52 层在同初态复跑中出现完整／倒塌结果变化，51 层尚不能作为可重复的稳定上限，见[全部复跑](../stack-limit-10s/reports/repeatability.md)。复跑和录像不重复计入 10 个独立初态。扫描过程中提前停止的是后续初态排程，已启动轨迹仍完成其完整窗口。这里的诊断耗时包含各自求解器调用和状态读取，不用作纯物理速度排名。",
    ]
    (root / "reports/benchmark-summary.md").write_text("\n".join(lines) + "\n")
    print(root / "reports/benchmark-summary.md")


if __name__ == "__main__":
    main()
