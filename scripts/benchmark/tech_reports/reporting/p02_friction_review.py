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

"""Recompute effective friction and freeze the installed implementation evidence.

Run: python -m scripts.benchmark.tech_reports.reporting.p02_friction_review
"""

from __future__ import annotations

import os

import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

__all__ = ["main"]


def main() -> None:
    """Generate the friction investigation from existing physical trajectories."""
    project = Path(
        os.environ.get("SIM_REPORT_ROOT", Path(__file__).resolve().parents[4])
    ).resolve()
    root = project / "outputs/benchmarks/P-02"
    out = root / "analysis/friction-review"
    (out / "source").mkdir(parents=True, exist_ok=True)
    metrics = json.loads((root / "analysis/main/episode-metrics.json").read_text())
    rows = []
    for r in metrics:
        if r["scene"] != "slope_30_mu54" or r["dt_s"] != 0.005:
            continue
        path = project / r["source"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == r["source_sha256"]
        a = r["fitted_acceleration_m_s2"]
        mu = (9.81 * np.sin(np.pi / 6) - a) / (9.81 * np.cos(np.pi / 6))
        np.testing.assert_allclose(mu, r["inferred_dynamic_mu"], atol=1e-12)
        rows.append(
            dict(
                run_id=r["run_id"],
                platform=r["platform"],
                solver=r["solver"],
                initial=r["initial"],
                acceleration_m_s2=a,
                acceleration_relative_error_pct=r["acceleration_relative_error_pct"],
                fitted_velocity_intercept_mm_s=r["fitted_velocity_intercept_mm_s"],
                configured_mu=0.4,
                inferred_mu=float(mu),
                inferred_mu_relative_error_pct=abs(mu - 0.4) / 0.4 * 100,
                position_max_error_1s_mm=r["position_max_error_1s_mm"],
                source=str(path),
                source_sha256=r["source_sha256"],
            )
        )
    assert len(rows) == 30
    with (out / "effective-friction.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "effective-friction.json").write_text(json.dumps(rows, indent=2) + "\n")

    dex = Path(
        "/home/oem/miniconda3/envs/embodichain/lib/python3.10/site-packages/dexsim"
    )
    kernel = dex / "engine/newton_physics/_jit/modules/mjvbd_v2/vbd"
    evidence = [
        (
            Path("/home/oem/code/EmbodiChain/embodichain/lab/sim/spawn/descriptors.py"),
            [(1619, 1622)],
        ),
        (dex / "scene/batches/rigid_body.py", [(47, 93), (286, 310)]),
        (
            kernel / "rigid_vbd_kernels.py",
            [(287, 297), (454, 550), (1395, 1466), (2216, 2297), (2344, 2413)],
        ),
        (
            kernel / "solver_vbd.py",
            [(110, 175), (842, 900), (1840, 1897), (2201, 2213)],
        ),
    ]
    manifest = []
    for path, ranges in evidence:
        target = out / "source" / path.name
        shutil.copy2(path, target)
        lines = path.read_text().splitlines()
        excerpt = "\n\n".join(
            "\n".join(f"{n}: {lines[n-1]}" for n in range(first, last + 1))
            for first, last in ranges
        )
        target.with_suffix(".excerpts.txt").write_text(excerpt + "\n")
        manifest.append(
            dict(
                original=str(path),
                snapshot=str(target),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                line_ranges=ranges,
            )
        )
    (out / "source-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    table = []
    for solver in [
        "DexUni / VBD 10 iterations",
        "DexUni / VBD 100 iterations",
        "PhysX TGS 4/1",
    ]:
        group = [r for r in rows if r["solver"] == solver]
        assert len(group) == 10
        mean = lambda key: np.mean([r[key] for r in group])
        mus = [r["inferred_mu"] for r in group]
        table.append(
            f"| {solver} | {mean('acceleration_m_s2'):.6f} | {mean('acceleration_relative_error_pct'):.5f} | {mean('inferred_mu'):.6f} [{min(mus):.6f}, {max(mus):.6f}] | {mean('inferred_mu_relative_error_pct'):.5f} | {mean('position_max_error_1s_mm'):.3f} |"
        )
    report = TEXT.replace("{table}", "\n".join(table))
    (root / "reports/friction-review.md").write_text(report)
    print(root / "reports/friction-review.md")


TEXT = """# P-02 摩擦实现与误差解释

补充分析 · 2026-10-22 · [返回三章合并稿](P-02.md)

当前安装版 DexUni 的刚体 VBD 硬接触使用摩擦系数 μ，并将切向力限制在库仑摩擦锥内。已有误差不能解释为“不支持 μ”，也不能直接把加速度误差百分比当成摩擦系数误差百分比。

## 从运动反推实际表现

对持续贴合 30° 斜面的滑块，理想参考为

`a = g(sinθ − μ cosθ)`，因此 `μ_eff = (g sinθ − a_fit)/(g cosθ)`。

以下使用既有关闭睡眠的斜面对照，不混入新堆叠协议。两表面 μd=0.4、dt=5 ms、10 个相同初态；加速度拟合窗口为 0.1–1 s，位置误差为完整 0–1 s。下表为逐初态指标的均值，方括号是初态范围。

| 配置 | 拟合 a / m·s⁻² | a 相对误差 / % | 反推 μ_eff | μ_eff 相对 0.4 的误差 / % | 前 1 s 最大位置误差 / mm |
| --- | ---: | ---: | ---: | ---: | ---: |
{table}

10 次迭代的运动表现相当于 μ≈0.341，100 次接近 μ≈0.399。前者加速度误差约 33%，但反推摩擦系数误差约 15%。原因是加速度由沿坡重力减去摩擦得到，两项相减后参考值较小，相对误差会被放大。100 次迭代改善了持续滑动阶段的加速度；约 25 mm 的位置偏差仍包含释放和接触建立阶段的累积误差。

μ_eff 是在持续贴面、近似平移条件下对运动的等效解释，不是读取求解器接触力得到的材料测量。法向响应、转动、时间离散和状态读回也可能贡献误差；它不证明摩擦内核单独造成了全部偏差。

100 次条件下，0.1–1 s 的速度线性拟合截距均值为 −34.77 mm/s。这是该窗口拟合直线外推到 t=0 的截距，实际设置的初速度仍为零；它说明加速度斜率接近参考时，速度曲线仍带有偏置，因此位置偏差不会同步消失。该观察支持把持续滑动误差与早期接触瞬态分开，不证明释放间隙是偏置的唯一原因。

## 本机实现到底做了什么

以下以运行所用安装源码为准，完整文件、行号与 SHA-256 保存在[证据清单](../analysis/friction-review/source-manifest.json)。不是把当前上游 Newton 文档直接当成本机实现。

1. **参数确实传入。** EmbodiChain `spawn/descriptors.py:1620–1622` 将 `dynamic_friction` 写入 Newton shape 的 `mu`。当前模型只有这一个系数；独立的 `static_friction` 不进入该路径。
2. **接触对的 μ 取几何平均。** `rigid_vbd_kernels.py:287–297` 使用 `sqrt(μ₁μ₂)`。本实验两面参数相同，因此与 PhysX 的 average 规则产生相同 μ，不是本次差异来源。
3. **硬接触使用库仑锥截断。** 同文件 `454–550` 先计算法向反力，再由切向位移和乘子计算切向试探力，最后限制 `‖F_t‖ ≤ μ F_n`。静止时可在锥内提供支撑，滑动时受锥面限制；单一 μ 并不等于“没有静摩擦”。
4. **求解有迭代误差。** 法向和切向乘子在迭代中更新；本配置不启用接触历史，刷新接触时乘子清零。即使 μ 正确，有限迭代的实际运动仍可能偏离理想库仑解。
5. **`friction_epsilon` 不是本项调节旋钮。** 它出现在 soft-contact 分支，当前 rigid hard-contact 分支不使用。默认 `rigid_avbd_beta=0`，本次也不是逐次增加接触罚刚度的配置。

参数与内核入口见[材料映射摘录](../analysis/friction-review/source/descriptors.excerpts.txt)、[刚体 API 摘录](../analysis/friction-review/source/rigid_body.excerpts.txt)、[接触内核摘录](../analysis/friction-review/source/rigid_vbd_kernels.excerpts.txt)、[求解器摘录](../analysis/friction-review/source/solver_vbd.excerpts.txt)。

## 原因判断及证据强弱

**目前最有力的判断是低迭代预算下接触求解不足。** 相同 dt、材料和初态，10→100 次迭代使 μ_eff 显著接近设定值。这支持“对求解预算敏感”，但尚未把误差唯一定位到法向或切向的某一步。

源码给出两个可继续验证的方向：一是法向反力尚未收敛时，`μ F_n` 的摩擦上限也会偏差；二是切向力被截断后，代码仍使用 `K_t=k_t(I−nnᵀ)`。在固定法向力的一维滑动例子中，沿滑动方向的截断力不再随位移线性增长，而这一矩阵仍保留切向刚度，可能使更新偏保守、收敛变慢。后者是代码支持的候选机制，尚未通过隔离实验确认，不能写成已定位的缺陷。

不启用历史意味着每步要重新建立接触乘子，也可能增加有限迭代下的误差。已有初态 0 的历史开关诊断改善了高摩擦静止斜面漂移，但不足以证明对全部滑动条件都有效。

**单系数模型是适用范围差异，但不是此次全部异常的解释。** 例如 30° 高摩擦场景虽然输入 μs/μd=0.8/0.6，实际单系数 0.6 仍大于 `tan30°≈0.577`，理想情况下同样应静止。当前长期滑移不能仅用“没有单独 μs”解释。只有临界角等需要区分静、动摩擦的实验，才必须把模型差异单列。

## 对实验配置的建议

先保持真实的 μ，增加迭代做收敛检查，再检查位置瞬态与长期漂移。当前 100 次配置只在所测滑动条件上表现更接近解析解，不能据此设为所有接触任务的统一默认。不要为补偿求解误差而把材料 μ 人为调大后继续称为同条件比较。

堆叠还受调度和接触微动抑制影响，与单块斜面不完全相同。安装版默认三项 stick 阈值为 `1e-4 m / 1e-4 m / 1e-4 rad`；满足动态—动态粘着条件时会回退极小位姿变化。这独立于睡眠和接触历史，先前“未启用冻结”的描述已经更正。按最新实验要求，新堆叠只观察 10 s，保留睡眠与默认微动抑制，不继续关闭对照；这个仅对特定动态接触触发的机制不能直接解释单块静态斜面的滑动误差。

## 数据与复算

- [30 条逐初态反推结果 CSV](../analysis/friction-review/effective-friction.csv)及 [JSON](../analysis/friction-review/effective-friction.json)含原始 NPZ 路径和 SHA-256。
- 原始 10 次与 100 次对照来自既有主测／补测，没有新增或筛选更有利的斜面轨迹；重置协议相同，独立复放情况见合并稿。
- 复算：在项目目录使用 embodichain 环境运行 `python -m scripts.benchmark.tech_reports.reporting.p02_friction_review`。只读取数组与源码，不启动仿真器，不修改求解器。
- [Newton 官方 VBD 说明](https://newton-physics.github.io/newton/latest/solvers/vbd.html)可用于了解方法背景；上游实现持续演进，本次结论只对应记录的 DexSim 安装版及其内嵌 VBD。
"""


if __name__ == "__main__":
    main()
