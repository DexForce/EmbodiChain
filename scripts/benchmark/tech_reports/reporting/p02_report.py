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

"""Rebuild the P-02 Markdown chapter from traceable per-trajectory metrics."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    metrics = json.loads((root / "analysis/main/episode-metrics.json").read_text())
    primary = [r for r in metrics if r["evaluation_role"] == "primary"]
    assert len(primary) == 160, len(primary)
    scene_labels = {
        "slope_15_mu54": "15°，μs/μd=0.5/0.4",
        "slope_30_mu54": "30°，μs/μd=0.5/0.4",
        "slope_45_mu54": "45°，μs/μd=0.5/0.4",
        "slope_30_mu32": "30°，μs/μd=0.3/0.2",
        "slope_30_mu86": "30°，μs/μd=0.8/0.6",
        "stack_01": "1 层",
        "stack_05": "5 层",
        "stack_10": "10 层",
    }

    def select(platform: str, scene: str, source: list[dict] = primary) -> list[dict]:
        return [r for r in source if r["platform"] == platform and r["scene"] == scene]

    def fmt(
        rows: list[dict], field: str, decimals: int = 3, spread: bool = False
    ) -> str:
        values = np.array(
            [r[field] for r in rows if r.get(field) is not None], dtype=float
        )
        if not len(values):
            return "—"
        mean = f"{values.mean():.{decimals}f}"
        return mean + (
            f" [{values.min():.{decimals}f}, {values.max():.{decimals}f}]"
            if spread
            else ""
        )

    def events(rows: list[dict]) -> str:
        times = [
            r["geometric_event_first_s"]
            for r in rows
            if r.get("geometric_event_first_s") is not None
        ]
        if not times:
            return f"0/{len(rows)}"
        interval = (
            f"{min(times):.3f} s"
            if min(times) == max(times)
            else f"{min(times):.3f}–{max(times):.3f} s"
        )
        return f"{len(times)}/{len(rows)}；{interval}"

    def platform_tables(platform: str) -> str:
        lines = [
            "#### 滑动斜面",
            "",
            "| 条件 | 拟合加速度 / m·s⁻² | 加速度相对误差 / % | 前 1 s 最大沿坡位置误差 / mm | 前 1 s 最大几何穿透 / mm |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for scene in ["slope_30_mu54", "slope_45_mu54", "slope_30_mu32"]:
            rows = select(platform, scene)
            lines.append(
                f"| {scene_labels[scene]} | {fmt(rows,'fitted_acceleration_m_s2',4,True)} | {fmt(rows,'acceleration_relative_error_pct',5)} | {fmt(rows,'position_max_error_1s_mm',3,True)} | {fmt(rows,'penetration_max_mm',3)} |"
            )
        lines += [
            "",
            "参考加速度依次为 1.506716、4.162031、3.205858 m/s²。拟合窗口为 0.1–1 s；位置误差保留释放后的接触建立阶段。",
            "",
            "#### 理想静止斜面",
            "",
            "| 条件 | 前 1 s 最大沿坡位移 / mm | 10 s 最大质心位移 / mm | 测试斜面最大穿透 / mm | 完全离开斜面时刻 / s |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
        for scene in ["slope_15_mu54", "slope_30_mu86"]:
            rows = select(platform, scene)
            times = [
                r["ramp_clear_first_s"]
                for r in rows
                if r["ramp_clear_first_s"] is not None
            ]
            exit_time = (
                f"{min(times):.3f}–{max(times):.3f}（{len(times)}/10）"
                if times
                else "10 s 内未离开"
            )
            lines.append(
                f"| {scene_labels[scene]} | {fmt(rows,'position_max_error_1s_mm',3,True)} | {fmt(rows,'drift_max_mm',3,True)} | {fmt(rows,'penetration_max_mm',3)} | {exit_time} |"
            )
        lines += [
            "",
            "若发生坡面脱离，10 s 位移包含后续下落／落地；它描述整条轨迹，不作为无限斜面公式的误差。",
            "",
            "#### 动态方块堆叠",
            "",
            "| 层数 | 0–12 s 最大横漂 / mm | 0–12 s 最大倾角 / ° | 0–12 s 最大穿透 / mm | 2–12 s 穿透 P95 / mm | 层间几何事件及首次时刻 |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
        for scene in ["stack_01", "stack_05", "stack_10"]:
            rows = select(platform, scene)
            lines.append(
                f"| {scene_labels[scene]} | {fmt(rows,'xy_drift_full_max_mm',3,True)} | {fmt(rows,'tilt_full_max_deg',3,True)} | {fmt(rows,'penetration_full_max_mm',3)} | {fmt(rows,'penetration_p95_mm',3)} | {events(rows)} |"
            )
        lines += [
            "",
            "1 层使用固定初态的逐次重置；5／10 层均为独立进程。表中横漂、倾角先在每条轨迹上取各层／各时刻最大值，再跨 10 初态统计；P95 先在每条轨迹的每步最大穿透序列上计算。所有层的原始位姿及逐初态数值见 [CSV](../analysis/main/episode-metrics.csv)。",
        ]
        return "\n".join(lines)

    def sensitivity_table(platform: str) -> str:
        rows = [
            r
            for r in metrics
            if r["platform"] == platform
            and r["phase"] in ["supplement", "independent_supplement"]
        ]
        keys = sorted({(r["scene"], r["solver"], r["dt_s"], r["phase"]) for r in rows})
        lines = [
            "| 场景 | 求解设置 | dt / ms | 初始化 | 加速度误差 / % | 前 1 s 沿坡误差 / mm | 堆叠全程横漂 / mm | 几何事件 |",
            "| --- | --- | ---: | --- | ---: | ---: | ---: | --- |",
        ]
        for scene, solver, dt, phase in keys:
            group = [
                r
                for r in rows
                if (r["scene"], r["solver"], r["dt_s"], r["phase"])
                == (scene, solver, dt, phase)
            ]
            init = "独立进程" if phase == "independent_supplement" else "重置序列"
            lines.append(
                f"| {scene_labels[scene]} | {solver} | {dt*1000:g} | {init} | {fmt(group,'acceleration_relative_error_pct',5)} | {fmt(group,'position_max_error_1s_mm',3)} | {fmt(group,'xy_drift_full_max_mm',3,True)} | {events(group) if scene.startswith('stack') else '不适用'} |"
            )
        return "\n".join(lines)

    emb = next(
        json.loads(p.read_text())
        for p in (root / "runs").glob("*/run.json")
        if json.loads(p.read_text()).get("phase") == "main"
        and json.loads(p.read_text()).get("platform") == "embodichain"
    )
    isaac = next(
        json.loads(p.read_text())
        for p in (root / "runs").glob("*/run.json")
        if json.loads(p.read_text()).get("phase") == "main"
        and json.loads(p.read_text()).get("platform") == "isaaclab"
    )
    tuned = [
        r
        for r in metrics
        if r["phase"] == "independent_supplement"
        and r["scene"] == "stack_10"
        and "contact grouping" in r["solver"]
    ]
    assert len(tuned) == 10, len(tuned)
    image_root = str(root / "figures")
    comparisons = []
    sliding_100 = [
        r
        for r in metrics
        if r["solver"] == "DexUni / VBD 100 iterations"
        and r["scene"] == "slope_30_mu54"
        and r["dt_s"] == 0.005
    ]
    assert len(sliding_100) == 10
    for label, scene, field, decimals in [
        (
            "30°、μd=0.4：运动反推 μ_eff",
            "slope_30_mu54",
            "inferred_dynamic_mu",
            6,
        ),
        (
            "30°、μd=0.4：加速度相对误差 / %",
            "slope_30_mu54",
            "acceleration_relative_error_pct",
            5,
        ),
        (
            "30°、μd=0.4：1 s 最大沿坡位置误差 / mm",
            "slope_30_mu54",
            "position_max_error_1s_mm",
            3,
        ),
    ]:
        comparisons.append(
            f"| {label} | {fmt(select('embodichain',scene),field,decimals)} | {fmt(sliding_100,field,decimals)} | {fmt(select('isaaclab',scene),field,decimals)} |"
        )
    comparison_table = "\n".join(comparisons)
    media_path = root / "videos/validation.json"
    media = (
        json.loads(media_path.read_text())["captures"] if media_path.exists() else []
    )
    different = [e for e in media if not e.get("state_array_identical", False)]
    media_consistency = f"当前已核验 {len(media)} 段原片，其中 {len(media)-len(different)} 段完整状态数组与对应测量逐项相同；其余差异见视频索引。"
    text = REPORT_TEMPLATE.format(
        comparison_table=comparison_table,
        tuned_drift=fmt(tuned, "xy_drift_full_max_mm", 3, True),
        tuned_tilt=fmt(tuned, "tilt_full_max_deg", 3, True),
        tuned_pen=fmt(tuned, "penetration_full_max_mm", 3, True),
        tuned_events=events(tuned),
        emb_tables=platform_tables("embodichain"),
        isaac_tables=platform_tables("isaaclab"),
        emb_sensitivity=sensitivity_table("embodichain"),
        isaac_sensitivity=sensitivity_table("isaaclab"),
        image_root=image_root,
        emb_commit=emb["source_commit"],
        isaac_commit=isaac["source_commit"],
        gpu=emb["gpu"],
        total_trajectories=len(metrics),
        emb_event=events(select("embodichain", "stack_10")),
        isaac_event=events(select("isaaclab", "stack_10")),
        media_consistency=media_consistency,
        stack_summary=(
            (root / "stack-limit-10s/reports/summary-fragment.md").read_text()
            if (root / "stack-limit-10s/reports/summary-fragment.md").exists()
            else "开启睡眠后的 10 s 堆叠上限正在补测。"
        ),
        stack_emb=(
            (root / "stack-limit-10s/reports/dexuni-fragment.md").read_text()
            if (root / "stack-limit-10s/reports/dexuni-fragment.md").exists()
            else ""
        ),
        stack_isaac=(
            (root / "stack-limit-10s/reports/isaac-fragment.md").read_text()
            if (root / "stack-limit-10s/reports/isaac-fragment.md").exists()
            else ""
        ),
    )
    (root / "reports/P-02.md").write_text(text)
    print(root / "reports/P-02.md")


REPORT_TEMPLATE = """# P-02 摩擦、接触与静态稳定

技术报告初稿 · 2026-10-22 · EmbodiChain / DexUni 与 Isaac Lab / GPU PhysX TGS

[原始数据与复现](data-index.md) · [逐初态 CSV](../analysis/main/episode-metrics.csv) · [同视角原生视频](../videos/index.md) · [三表 benchmark 摘要](benchmark-summary.md)

## 第一章　对比与结论

原始对照覆盖 5 组斜面条件和 1／5／10 层动态方块堆叠，关闭睡眠；扩展堆叠上限采用保留睡眠、完整 10 s 的协议，另表列出。每组使用相同的 10 个初态，主对比共 160 条轨迹，步长 5 ms；DexUni 基线为 VBD 10 次迭代，PhysX 为 TGS 4 次位置／1 次速度迭代。不同求解器的迭代次数不代表相同工作量，本项只比较运动与接触质量。

| 滑动斜面指标（10 初态均值） | DexUni VBD 10 | DexUni VBD 100 | PhysX TGS 4/1 |
| --- | ---: | ---: | ---: |
{comparison_table}

**摩擦使用 μ，低预算误差明显。** 30°、μ=0.4 时，DexUni 10→100 次迭代使运动反推的 μ 从 0.3411 接近至 0.3986；加速度相对误差从 33.22% 降至 0.79%。这支持接触求解对预算敏感，尚未把全部误差定位到单一内核。[摩擦实现与原因分析](friction-review.md)区分已验证事实与候选机制。

{stack_summary}

选择配置时需同时检查运动误差、几何穿透和实际结构事件；本次静态堆叠的边界不能直接代替受扰动、抓取或含关节任务的验证。

以上区间均为初态最小值与最大值，不是置信区间。全部基线、较差配置和重置敏感性记录均保留在后两章及数据索引中；不设置统一 pass/fail 门槛，也不据诊断采样耗时给引擎排名。

## 第二章　EmbodiChain / DexUni 详细报告

### 2.1 共同模型与测量方法

斜面采用 100×100×20 mm、0.1 kg 的薄块和长 12 m 的静态支撑板；初始底面与坡面平行、速度为零，法向间隙 0.1 mm。5 组条件是 15°／30°／45°、μs/μd=0.5/0.4，以及 30°、μs/μd=0.3/0.2 和 0.8/0.6。两接触表面使用相同参数，恢复系数为 0。

堆叠采用 50 mm、0.1 kg 立方体，所有方块均为动态刚体。支撑板顶面在 z=0.2 m；每层 xy 扰动在 ±0.5 mm 内，yaw 在 ±1° 内，初始层间无几何交叠。每条轨迹从释放开始保存 12 s，其中前 2 s 为稳定阶段。稳定阶段发生的位移、穿透与层间分离全部计入全程指标。

重力为 9.81 m/s²，N=1，物理步长 5 ms、内部子步 1；补测取 2.5 ms。背景地面保留在 z=0；离开测试支撑后的背景地面碰撞不计入测试支撑的几何穿透指标。测试几何 margin/rest offset 为 0，接触生成 gap/contact offset 为 1 mm。两个后端的碰撞缓冲语义不完全相同，穿透量统一从保存的箱体位姿计算。

斜面参考为 `tanθ ≤ μs` 时静止，否则 `a=g(sinθ−μd cosθ)`、`s=at²/2`、`v=at`。加速度由 0.1–1 s 的沿坡速度线性拟合；位置／速度误差使用完整 0–1 s，保留接触建立过程。这里的位移参考是理想持续接触轨迹，释放间隙及离散碰撞瞬态也会贡献偏差，不能将所有位置偏差都解释为摩擦系数误差。完整运行保留 10 s，离开有限斜面后的下落和落地单列阶段。

堆叠横漂是各层质心相对初态的水平距离；倾角是各层局部 z 轴与竖直方向的夹角。几何穿透使用方块的 15 个分离轴求最小分离平移，不用 solver 接触缓冲的 separation 代替。层间事件定义为原相邻层水平投影完全分离，或上层中心不再高于下层中心；它是几何事件，不冒充接触力测得的支撑关系。检测分辨率等于物理步长。

### 2.2 实际后端与模型差异

本次通过 EmbodiChain 公共对象、状态和相机接口实现，源码提交为 `{emb_commit}`，底层 DexSim 安装版为 `0.5.1+00106a1db10e330c62cb62d8015b8474e997701b`。GPU 为 {gpu}。入口显式固定当前源码、机器人包和任务包的导入路径；完整软硬件、配置、惯量及物理属性读回见各运行 `run.json`。

当前 Newton/DexUni 材质只消费一个摩擦系数，取配置中的 `dynamic_friction`；单独的 `static_friction` 未进入该模型。本次选点在单系数和双系数理想库仑模型下的静止／滑动分支相同，但不能据此把临界角附近的两种模型视作等价。

基线启用硬接触、10 次 VBD 迭代，未启用接触历史、接触分组和刚体睡眠。运行时读取的质量、惯量、摩擦及碰撞尺寸与输入相符。Newton 路径的 `get_damping()` 返回保留的 Default 后端元数据 0.7/0.7，安装版代码明确将其作为元数据返回，不能把它写成 DexUni 实际施加了该阻尼；本次未添加额外阻尼控制器。

**配置说明更正：**安装版默认 `rigid_contact_stick_motion_eps=1e-4 m`、`rigid_contact_stick_freeze_translation_eps=1e-4 m`、`rigid_contact_stick_freeze_angular_eps=1e-4 rad`。满足动态—动态粘着条件且单步位姿变化很小时，最终更新会回退至上一帧位姿。此逻辑独立于刚体睡眠和接触历史。此前正文“未启用冻结”不准确；原数据、配置快照不变。本项原主测和分组候选均保留了默认值，只有明确关闭三项阈值的诊断才能称为关闭微动抑制。

DexUni 公共接口未提供本项所需的原生接触力／冲量，因此两端统一比较运动与独立几何指标，力相关项标记未提供。安装源码核对记录见 [依赖来源哈希](../analysis/dependency-source-hashes.json)。

### 2.3 主测结果

以下数值为每条轨迹指标的跨初态均值，方括号给出最小值、最大值；每行 10 个初态。

{emb_tables}

![斜面误差对比]({image_root}/slope-errors.png)

图 1：主测的滑动加速度误差与前 1 s 最大沿坡位置误差；误差线是 10 初态范围。

![静止条件的位移]({image_root}/static-drift.png)

图 2：两组理想静止条件的实际质心位移。30° 高摩擦条件下若已经滑出斜面，完整 10 s 位移还包括后续阶段，详见上表离开时刻。

### 2.4 配置敏感性与实践含义

{emb_sensitivity}

主测后的补测先保留了相同的逐次重置协议。发现多层结果受初始化路径影响后，新增独立进程的主测与接触分组补测。表中的两种初始化方式明确分列，不把它们混合计算一个均值。

滑动斜面上，增加迭代数使拟合加速度更接近解析值；前 1 s 位移仍含释放初期累积的偏差。堆叠则不能套用“增加迭代必然改善”的结论：默认调度下 100 次迭代仍可能快速失稳，而接触分组与足够迭代预算组合才明显改善本次 10 层场景。分组控制的是多刚体接触迭代的执行方式，本次证据支持配置相关性，尚未把全部差异归因到某个内核或单一数值机制。

**摩擦误差解释。** 本机硬接触将切向试探力截断至 `μ F_n`，实际使用单一库仑系数，取输入的 `dynamic_friction`。30°、μ=0.4 的等效反推 `μ_eff=(g sinθ−a_fit)/(g cosθ)` 为 10 次 0.341082、100 次 0.398600；它们相对材料值的误差为 14.73% 和 0.35%，并不是 33.22% 和 0.79%。后者是重力与摩擦相减后加速度的相对误差。反推值不是接触力实测值。有限迭代、法向／切向耦合和每步接触乘子初始化是合理研究方向；截断后仍保留切向刚度可能影响收敛，但尚未隔离验证。完整源码证据、30 条原始数据与限制见[摩擦补充分析](friction-review.md)。

原 12 s 接触分组补测的最大横漂为 {tuned_drift} mm，最大倾角为 {tuned_tilt}°，最大几何穿透为 {tuned_pen} mm，层间事件 {tuned_events}；这些是保留默认接触微动抑制的结果。

接触历史另做了初态 0 的小对照：启用 `contact_matching=latest`，关闭三项 stick／freeze 阈值。它改善了该高摩擦斜面的漂移，但没有解决全部滑动误差；样本量不足以作为统一推荐。已核对本次硬接触分支不使用 `friction_epsilon`，因此没有做无效的平滑参数扫描。相关原始运行单列为 `diagnostic`。

![配置敏感性]({image_root}/configuration-sensitivity.png)

图 3：左侧为相同重置协议下的滑动斜面；右侧为独立进程的 10 层堆叠。VBD 与 TGS 的迭代数字不代表相等计算预算，图中不作速度比较。

![接触分组配置的原生运行截图]({image_root}/stack_10_it100_grouped_native.png)

图 4：相同初态、相同镜头的原生仿真画面。左为 DexUni 100 次迭代＋接触分组，右为 PhysX TGS 4/1，均为 5 ms；图像来自运行相机，未用轨迹重建代替。

对后续任务，建议先用低层接触核对材质和真实几何，再对目标层数／接触链做预算扫描。自由刚体堆叠可从本次已测的“100 次迭代＋接触分组”配置继续验证；它不是所有摩擦、关节或软体任务的统一默认值。评估时同时看横漂、倾角、穿透和实际事件，不能只凭视频看起来静止就认定精度足够。

### 2.5 重置敏感性与数据来源

本次独立复放显示，多层场景的长期轨迹并非总能逐点重现。原序列中 PhysX 初态 1 在 11.77 s 出现层间事件，独立进程复放未重现该事件；因此该序列不再进入独立堆叠主测。DexUni 同一检查也发现明显轨迹差异。独立创建场景后的录制也观察到 DexUni 10 层轨迹差异，说明这些变化不能全部归因于 reset；具体机制尚未定位。详见 [重置复放记录](../analysis/reset-replay.json)及视频核验。本次用每个初态独立进程统一多层主测的初始化流程，不声称由此保证确定性。

代表性斜面初态 9 的独立复放中，两平台的滑动轨迹均与序列记录逐项相同；高摩擦条件的最大位置差为 DexUni 0、PhysX 0.00334 mm。斜面保留原 10 初态统计，见 [斜面复放记录](../analysis/slope-reset-replay.json)。未把单条复放当作新增独立初态。当前范围描述 10 个不同初态，尚未量化每个初态重复运行的分布。

每个结果均保留逐物理步状态、原始初态、输入／源码／Spec 快照、日志和运行编号。当前离线统计共收录 {total_trajectories} 条主测及参数／重置研究轨迹，其中正式主对比为 160 条；预检、独立复放诊断和录像另存。`episode-metrics.csv` 每行给出原始 NPZ 路径与 SHA-256，完整运行和文件清单见 [数据索引](data-index.md)。

{stack_emb}

## 第三章　Isaac Lab / GPU PhysX TGS 详细报告

### 3.1 运行设置与校准

Isaac Lab 源码提交为 `{isaac_commit}`，本机安装元数据为 Isaac Lab 17.0.2、Isaac Sim 6.1.0.0、isaaclab-physx 6.0.0。求解器使用 GPU PhysX TGS，位置／速度迭代固定为 4／1，线／角阻尼、睡眠阈值和稳定化均关闭。质量、质心和惯量显式写入并读回；材质分别设置静、动摩擦，组合规则为 average。两接触表面取相同参数。

几何、初态、时间窗口及离线指标与第二章相同。静态接触筛选指向实际碰撞形状，多刚体筛选使用明确的对象路径。原始记录保留有效接触行、法向冲量、平均法向／切向力和接触点；只按有效计数读取，没有把未使用的 buffer 行当作接触。t=0 尚未采集力，保留 NaN。

力数据用于本平台的接触采样校准，不与 DexUni 未提供的力值构造对比。原生 contact separation 与独立几何穿透分开保存。所有 PhysX 结论限定于这里的 TGS 配置；本项未执行 PGS 对照。

### 3.2 主测结果

{isaac_tables}

![堆叠主测统计]({image_root}/stack-baseline.png)

图 5：1／5／10 层的横漂、倾角和几何穿透，均保留从释放开始的完整窗口；5／10 层使用独立进程统计。

![堆叠初态 0 的时序曲线]({image_root}/stack-traces-initial0.png)

图 6：原最高层质心高度与每步最大穿透。竖虚线标记 2 s 稳定阶段结束；在此之前发生的结构变化没有被丢弃。

### 3.3 步长补测与适用范围

{isaac_sensitivity}

斜面步长减半后，位置偏差与静止位移减小；滑动加速度本身已接近理想参考。原重置序列中的 10 层补测也给出了较小漂移，但它与独立进程主测的初始化协议不同，不能把这两者的差异全部归因为步长。

P-01 无接触场景中的 PGS/TGS 结论没有直接套用到本项。本项说明接触问题还需要关注材料规律、接触链长度、求解预算和初始化状态。毫米级差异或一次层间事件的意义，应由后续具体操作任务决定，不人为汇总成一个平台总分。

### 3.4 视频与复现

[视频索引](../videos/index.md)保留 8 组主条件的两平台原片，以及 DexUni 接触分组配置的补充对照。主素材固定初态 0，1280×720、40 fps、正常速度；镜头内参、外参、场景尺寸和采样物理时刻统一。斜面移动镜头按共同解析路径运行，不分别跟随各自物体。30°／45° 展示共同前 1 s，15° 展示 10 s；堆叠展示完整 12 s。另选 10 层初态 3 做两平台事件核查录像，因为该初态在 PhysX 独立主测的 5.595 s 出现层间分离；它是有意选择的事件样本，不代表平均情况。

每个原片目录保存未改动的 PNG 帧、逐帧时间戳／状态、实际相机参数和编码命令。[视频核验](../videos/validation.json)列出录制与测量轨迹的差异、实际镜头读回及解码结果。{media_consistency} 录制不计入独立初态数量；不一致的录制作为独立运行展示，不能当作原主测轨迹的逐帧证明。更换渲染器会改变光照和材质观感，不影响这里统一的几何视角。

复现命令、源码／输入哈希、运行异常及全部未筛选数据见 [数据索引](data-index.md)。分析和绘图均可从原始数组离线重建，无需重新启动仿真器。分析图同时提供 PNG／SVG／PDF 与绘图数据；本稿保持 Markdown，后续再统一编辑排版。

{stack_isaac}
"""

if __name__ == "__main__":
    main()
