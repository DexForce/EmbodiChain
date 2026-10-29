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
"""Report C02 completed-step latency with explicit reused and new samples.

Run: .venv/bin/python build_report.py
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["main"]


def _read(path: Path) -> dict[str, Any]:
    """Read retained metadata without changing a source run."""
    return json.loads(path.read_text())


def _stats(rows: list[dict[str, str]], end: float) -> dict[str, Any]:
    """Aggregate every time-matched steady sample without quality filtering."""
    selected = [
        r
        for r in rows
        if r["steady"] == "True" and float(r["actual_time_s"]) <= end + 1e-8
    ]
    assert len(selected) == round(end * 60) - 30
    result: dict[str, Any] = dict(samples=len(selected), window_seconds=end)
    for field in ("physics_s", "outer_s", "readback_s"):
        values = np.array([float(r[field]) * 1000 for r in selected])
        assert np.isfinite(values).all() and (values > 0).all()
        result[field] = dict(
            mean_ms=float(values.mean()),
            median_ms=float(np.median(values)),
            p95_ms=float(np.percentile(values, 95)),
        )
    return result


def main() -> None:
    """Build three tables and machine-readable evidence for all three settings."""
    work = (
        Path(__file__).resolve().parents[5]
        / "outputs/benchmarks/tech_reports/dexuni/c02_bag_frame_packing/performance_comparison"
    )
    root = work.parent
    protocol = _read(work / "protocol.json")
    for rel, expected in protocol["protected_sha256"].items():
        assert hashlib.sha256((root / rel).read_bytes()).hexdigest() == expected, rel
    definitions = [
        ("EmbodiChain 12x6", root / "pilot_performance/embodichain", "steps.csv", True),
        ("Isaac Lab 24x12", work / "isaaclab_24x12", "probe_steps.csv", False),
        ("Isaac Lab 96x24", root / "pilot_performance/isaaclab", "steps.csv", True),
    ]
    result: dict[str, Any] = dict(protocol=protocol, configurations=[])
    for label, path, filename, reused in definitions:
        raw = list(csv.DictReader((path / filename).open()))
        manifest = _read(path / "manifest.json")
        assert manifest["status"] == "completed"
        assert len(raw) == (3840 if reused else 720)
        assert all(int(row["frame"]) == i + 1 for i, row in enumerate(raw))
        assert all(
            abs(float(row["actual_time_s"]) - (i + 1) / 60) < 1e-7
            for i, row in enumerate(raw)
        )
        item = dict(
            label=label,
            reused=reused,
            raw_csv=str(path / filename),
            raw_csv_sha256=hashlib.sha256((path / filename).read_bytes()).hexdigest(),
            common_initial_12s=_stats(raw, 12),
            effective=manifest["effective"],
        )
        if reused:
            item["complete_64s"] = _stats(raw, 64)
            item["measurement_summary"] = _read(path / "summary.json")
        else:
            item["measurement_summary"] = _read(path / "probe_summary.json")
            assert item["measurement_summary"]["error"] is None
            assert (
                abs(
                    item["common_initial_12s"]["physics_s"]["mean_ms"]
                    - item["measurement_summary"]["mean_ms"]
                )
                < 1e-9
            )
        result["configurations"].append(item)
    low = _read(root / "budget_reduction/i24_s12_video/analysis/current_metrics.json")
    result["low_budget_video_metrics"] = low
    result["interpretation"] = (
        "96x24 is a tested working budget for the recorded packing/lift flow, not a proven minimum.24x12 failed the task. Platform controller/solver strategies differ; initial-window latency is not full-task latency or equal-quality performance."
    )
    lines = [
        "# C02 视频与性能对比",
        "",
        "[三档64秒完整对比](../budget_reduction/iteration_comparison.mp4) · [Isaac24×12录像](../budget_reduction/i24_s12_video/preview/video.mp4) · [Isaac96×24成功录像（复用）](../platforms/isaaclab/preview/video.mp4) · [EmbodiChain录像（复用）](../platforms/embodichain/preview/video.mp4)",
        "",
        "Isaac在当前场景和双手轨迹下，实现原视频所示两件装袋与载荷提袋的配置为**96次迭代、24子步**。降为24×12后，完整录到了64秒，但换手后袋子倒下，物品离开场景，未完成装袋和载荷提袋。这说明已测试的低预算不足以保持这段效果；**尚未证明96×24是最低所需配置**。",
        "",
        "## Time & Memory",
        "",
        "| 配置 | 仿真窗口 | 数据来源 | 稳态步数 | cost_time_ms(物理平均) | 中位(ms) | p95(ms) | 含IK/命令整步(ms) | 读回(ms) | cpu_delta_mb | gpu_delta_mb | peak_gpu_mb |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in result["configurations"]:
        s = item["common_initial_12s"]
        p = s["physics_s"]
        lines.append(
            f"| {item['label']} | 0–12s | {'复用全程样本前缀' if item['reused'] else '本次新测'} | {s['samples']} | {p['mean_ms']:.3f} | {p['median_ms']:.3f} | {p['p95_ms']:.3f} | {s['outer_s']['mean_ms']:.3f} | {s['readback_s']['mean_ms']:.3f} | 未测 | 未测 | 未测 |"
        )
    for item in result["configurations"]:
        if "complete_64s" not in item:
            continue
        s = item["complete_64s"]
        p = s["physics_s"]
        lines.append(
            f"| {item['label']} | 0–64s | 复用已有全程摸测 | {s['samples']} | {p['mean_ms']:.3f} | {p['median_ms']:.3f} | {p['p95_ms']:.3f} | {s['outer_s']['mean_ms']:.3f} | {s['readback_s']['mean_ms']:.3f} | 见原报告驻留增量 | 未测 | 未测 |"
        )
    lines += [
        "",
        "每档一条独立进程轨迹；不是五次重复正式测量，步数不等于独立试验次数。前30步预热剔除，其余全保留，包括保形失败的低配样本。物理计时为EC sim.update或Isaac sim.step+scene.update完成耗时，前后同步实际Warp cuda:0设备；IK/命令和状态读回另列。摄像机关闭，源码与输入快照保留。短窗口只覆盖立袋前段，不能当作完整装袋提袋耗时。",
        "",
        "本轮未测瞬时CPU/GPU峰值。原全程样本的进程显存驻留最大采样：EC主进程1496MiB、IK674MiB；Isaac96×24主进程1278MiB、IK674MiB。它们不是分配器峰值或本轮12秒窗口峰值。低配的逐次驻留和外部活动采样见isaaclab_24x12/probe_summary.json。",
        "",
        "## Success & Other Metrics",
        "",
        "| 配置 | 视频动作结果（单次观察） | 性能轨迹几何结果 | 说明 |",
        "| --- | --- | --- | --- |",
        "| EmbodiChain12×6 | 两件装袋、载荷提袋完成 | 原独立64s摸测完成，原严格检查通过 | DexUni DAT/多层校正/Chebyshev；参考轨迹 |",
        "| Isaac24×12 | 倒袋，未完成两件装袋和载荷提袋 | 新计时仅12s；保形失败按用户要求保留 | 普通VBD，双手适配轨迹；录满64s不等于任务成功 |",
        "| Isaac96×24 | 两件装袋、载荷提袋完成 | 原独立64s摸测悬停阶段均装载，终态纸盒X边界超0.191mm | 原视频袋口最大弯曲51.285mm>严格30mm阈值；不能称所有几何检查通过 |",
        "",
        "三档视频均为实际64s、3840步、1921帧30fps，无冻结补帧。24×12视频的原始状态全部有限，但最大TCP目标误差约292mm、物品离开场景，低配退步并非只有外观差别。两档Isaac使用相同双手轨迹配置，实际反馈命令随各自袋体状态变化；EC使用原参考轨迹。不同求解器的一次迭代工作不等价，不能把此表解释为相同控制、相同质量或统一迭代预算的纯求解器对照。",
        "",
        "## Leaderboard",
        "",
        "| 配置 | 排名 | 解释 |",
        "| --- | --- | --- |",
        "| EmbodiChain12×6 | 不作等质量排名 | 列出实测速度及当前录像效果 |",
        "| Isaac24×12 | 不作等质量排名 | 低预算较快，但录像任务失败 |",
        "| Isaac96×24 | 不作等质量排名 | 当前已录成功效果对应的较高预算，最低预算未知 |",
        "",
        "复现：旧原始数据见../pilot_performance；新24×12命令和计时范围见isaaclab_24x12/probe_protocol.json，源文件见其source目录，原始逐步数据见probe_steps.csv。视频来源与SHA256见../budget_reduction/iteration_comparison.json。可通过本目录build_report.py只读重建本报告。",
    ]
    (work / "results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    )
    (work / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                i["label"]: i["common_initial_12s"]["physics_s"]
                for i in result["configurations"]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
