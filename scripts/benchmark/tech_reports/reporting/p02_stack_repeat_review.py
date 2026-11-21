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

"""Compare fixed-input headless repeats and native recordings offline.

Run: python -m scripts.benchmark.tech_reports.reporting.p02_stack_repeat_review
"""

from __future__ import annotations

import os

import json
from pathlib import Path
import numpy as np

__all__ = ["main"]


def main() -> None:
    """Preserve event reversals and actual model/initial-state checks."""
    project = Path(
        os.environ.get("SIM_REPORT_ROOT", Path(__file__).resolve().parents[4])
    ).resolve()
    root = project / "outputs/benchmarks/P-02/stack-limit-10s"
    originals = [
        json.loads(p.read_text()) for p in (root / "analysis/by-run").glob("*.json")
    ]
    captures = [
        json.loads(p.read_text()) for p in (root / "analysis/capture").glob("*.json")
    ]
    repeats = [
        json.loads(p.read_text())
        for p in sorted((root / "repeatability/analysis").glob("*.json"))
    ]
    rows = []
    for n, initial in [(51, 0), (52, 4), (78, 0), (79, 0)]:
        original = next(
            r for r in originals if r["layers"] == n and r["initial"] == initial
        )
        ref_path = Path(original["source"])
        ref_meta = json.loads((ref_path.parent / "run.json").read_text())
        reference = np.load(ref_path)["state"]
        samples = [("原测量", original)]
        samples += [
            ("原生录制", r)
            for r in captures
            if r["layers"] == n and r["initial"] == initial
        ]
        samples += [
            (f"无录像复跑 {j+1}", r)
            for j, r in enumerate(
                r for r in repeats if r["layers"] == n and r["initial"] == initial
            )
        ]
        for label, metric in samples:
            path = Path(metric["source"])
            meta = json.loads((path.parent / "run.json").read_text())
            actual = np.load(path)["state"]
            np.testing.assert_array_equal(actual[0], reference[0])
            checked = [
                "scene",
                "dt_s",
                "duration_s",
                "solver",
                "body_readback",
                "support_pose_readback",
                "background_ground",
            ]
            equal_model = {
                k: meta[k] == ref_meta[k]
                for k in checked
                if k in meta and k in ref_meta
            }
            assert all(equal_model.values()), equal_model
            if meta["platform"] == "embodichain":
                assert (
                    meta["requested_sim_config"]["physics_cfg"]
                    == ref_meta["requested_sim_config"]["physics_cfg"]
                )
                equal_model["requested_physics_config"] = True
            changed = np.flatnonzero(np.any(actual != reference, axis=(1, 2)))
            rows.append(
                dict(
                    **metric,
                    sample_kind=label,
                    reference_run=original["run_id"],
                    initial_array_identical=True,
                    checked_model_fields=equal_model,
                    state_array_identical=bool(np.array_equal(actual, reference)),
                    first_differing_step=int(changed[0]) if len(changed) else None,
                    max_position_component_difference_mm=float(
                        np.max(np.abs(actual[..., :3] - reference[..., :3]))
                    )
                    * 1000,
                )
            )
    target = root / "analysis/repeatability-review.json"
    target.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")
    lines = [
        "## 边界重复性与原生录像",
        "",
        "DexUni 分组配置的 51 层在原 10 个初态中均完整，但初态 0 的录像在 1.985 s 出现结构事件，最终最长连续层序只剩 1 层。52 层初态 4 原测量在 1.765 s 失稳，录制及两次不录像复跑均保持完整。51 层初态 0 的两次不录像复跑也完整。",
        "",
        "**51 层只能作为本批次的观测边界，尚不能宣称为可重复的稳定上限。** 已核对初态逐值相同，以及几何、质量、惯量、材料、支撑和实际物理配置一致。52 层无录像复跑也翻转了结果，说明这种差异不只出现在录制场景；具体来源尚未隔离。10 个不同初态与同初态的多次运行是两种不同的检查，不能用前者代替后者。",
        "",
        "PhysX 的 78／79 层录像均逐项复现对应原始状态；79 层该初态在 6.400 s 开始失稳，最终最长连续层序为 2 层。画面确认这两段失稳轨迹发生了整柱倒塌。没有仅凭速度峰值判定数值“炸飞”；这几条轨迹相对初始机械能的最大正增量均为 0 J。",
        "",
        "| 平台／层数／初态 | 样本 | 首次结构事件 / s | 最终连续层数 | 与原测量状态完全相同 | 原始数据 |",
        "| --- | --- | ---: | ---: | --- | --- |",
    ]
    for r in rows:
        event = (
            "无"
            if r["first_geometry_event_s"] is None
            else f"{r['first_geometry_event_s']:.3f}"
        )
        lines.append(
            f"| {r['platform']}／{r['layers']}／{r['initial']} | {r['sample_kind']} | {event} | {r['longest_ordered_chain_final']} | {r['state_array_identical']} | [NPZ]({r['source']}) |"
        )
    lines += [
        "",
        f"[逐项复核与原始文件哈希]({target}) · [无录像复跑计划]({root}/repeatability/plan.json) · [录像及相机核验]({root}/videos/index.md)",
        "",
        "以下均为同一固定相机下的真实运行末帧。左、右层数不同，标签已注明。第一组原本按测量中的完整边界选取，但 DexUni 复录倒塌；第二组原本按更高一层的失稳初态选取，但 DexUni 复录保持完整。全部原片保留，不以重录成功替换异常。",
        "",
        f"![51／78 层复录末帧：DexUni 倒塌，PhysX 完整]({root}/figures/stable_boundary_native.png)",
        "",
        f"![52／79 层复录末帧：DexUni 完整，PhysX 倒塌]({root}/figures/next_layer_native.png)",
        "",
    ]
    (root / "reports/repeatability.md").write_text("\n".join(lines))
    print(
        json.dumps(
            dict(
                compared_runs=len(rows),
                headless_repeats=len(repeats),
                all_initials_identical=True,
                all_checked_model_fields_equal=True,
            )
        )
    )


if __name__ == "__main__":
    main()
