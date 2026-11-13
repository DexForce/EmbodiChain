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

"""Build the three-table P-01 summary from continuous per-trajectory metrics."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import statistics


def main() -> None:
    """Keep all configurations and missing performance fields without pass gates."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    groups = defaultdict(list)
    with args.metrics.open() as stream:
        for row in csv.DictReader(stream):
            groups[row["run_id"]].append(row)
    time_rows, quality_rows, comparison_rows = [], [], []
    records = []
    for run_id, rows in sorted(groups.items()):
        first = rows[0]
        case = first["case"]
        label = f"{first['platform']} / {first['solver']} / {case} / dt={float(first['dt_s'])*1000:g} ms"
        metric = (
            "position_max_0_1s_m" if case == "freefall" else "energy_max_relative_drift"
        )
        scale = 1000 if case == "freefall" else 100
        values = [float(row[metric]) * scale for row in rows]
        mean, low, high = statistics.mean(values), min(values), max(values)
        unit = (
            "mm（前 1 s 最大位置误差）"
            if case == "freefall"
            else "%（10 s 最大相对能量漂移）"
        )
        time_rows.append(f"| {label} | 未测量 | 未测量 | 未测量 | 未测量 |")
        quality_rows.append(
            f"| {label} | {len(rows)} | 不适用 | {mean:.6g} | {low:.6g}–{high:.6g} | {unit} |"
        )
        comparison_rows.append(
            f"| {label} | 不设综合名次 | {mean:.6g} {unit} | `{run_id}` |"
        )
        records.append(
            {
                "run_id": run_id,
                "trials": len(rows),
                "metric": metric,
                "scale": scale,
                "mean": mean,
                "min": low,
                "max": high,
            }
        )
    text = "\n".join(
        [
            "# P-01 benchmark 摘要",
            "",
            "由连续指标表离线生成；不重跑仿真。每行保留一个完整配置，范围为跨初态最小–最大值，不是置信区间。",
            "历史 CPU 读回诊断和进程 RSS 不作为规范性能测量。无二元任务目标，成功率不适用；不设置精度合格线或综合排名。",
            "",
            "## Time & Memory",
            "",
            "| 配置 | cost_time_ms | cpu_delta_mb | gpu_delta_mb | peak_gpu_mb |",
            "| --- | --- | --- | --- | --- |",
            *time_rows,
            "",
            "## Success & Other Metrics",
            "",
            "| 配置 | 独立初态数 | Success | 均值 | 初态范围 | 指标与单位 |",
            "| --- | ---: | --- | ---: | --- | --- |",
            *quality_rows,
            "",
            "## Leaderboard",
            "",
            "同场景、同 dt 才是对应行；不同求解器的迭代数不是等成本预算。完整误差／周期／速度指标仍见原 CSV。",
            "",
            "| 配置 | 排名说明 | 对照指标均值 | 原始 run_id |",
            "| --- | --- | --- | --- |",
            *comparison_rows,
            "",
            f"来源：`{args.metrics.name}`，SHA-256 `{hashlib.sha256(args.metrics.read_bytes()).hexdigest()}`。",
            "",
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)
    args.output.with_suffix(".json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
