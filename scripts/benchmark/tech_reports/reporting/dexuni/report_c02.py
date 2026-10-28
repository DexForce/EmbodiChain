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
"""Rebuild the reference-based C02 preview report without executing simulation.

Run: .venv/bin/python -m scripts.benchmark.tech_reports.reporting.dexuni.report_c02
"""

from __future__ import annotations
import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

__all__ = ["main"]


def main() -> None:
    """Describe the original demo and its contact-only lifting extension."""
    root = (
        Path(__file__).resolve().parents[2].parents[2]
        / "outputs/benchmarks/tech_reports/dexuni/c02_bag_frame_packing"
    )
    a = root / "analysis"
    original = json.loads((a / "original_demo_metrics.json").read_text())
    pilot = json.loads((a / "lift_pilot_metrics.json").read_text())
    current = json.loads((a / "current_checks/current_metrics.json").read_text())
    geometry = json.loads((a / "current_checks/robot_geometry.json").read_text())
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, layout="constrained")
    arrays = {}
    for name, path in (
        ("pilot", a / "lift_pilot_plot_data.npz"),
        ("video", a / "current_checks/plot_data.npz"),
    ):
        with np.load(path) as d:
            for key in d.files:
                arrays[name + "_" + key] = d[key]
            axes[0].plot(
                d["time"], d["bag_bottom_height"] * 1000, label=name + " bag bottom"
            )
            if name == "video":
                for i, label in enumerate(("can bottom", "carton bottom")):
                    axes[0].plot(
                        d["time"], d["payload_bottom_height"][:, i] * 1000, label=label
                    )
            axes[1].plot(d["time"], d["shape_error"] * 1000, label=name)
    for ax in axes:
        ax.axvspan(58, 60, color="green", alpha=0.12)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Clearance above table (mm)")
    axes[1].set_ylabel("Paper rigid-fit RMS (mm)")
    axes[1].set_xlabel("Actual simulation time (s)")
    fig.savefig(a / "checks.png", dpi=150)
    plt.close(fig)
    np.savez_compressed(a / "checks_plot_data.npz", **arrays)
    text = [
        "# C02 重做：原 demo 纸袋扶正、两件装填与带载提袋",
        "",
        "这版直接使用指定 `demo_dexuni_w1_bag_packing.py` 的原纸袋、柔性提手、罐头、纸盒、W1 和桌子。先完整执行原来的50秒扶正/装填流程，再追加14秒袋口换抓、提起、保持和放回。上一版橙色硬框/小方块场景已被用户否定并归档，不作为当前有效结果。",
        "",
        "[完整64秒视频](preview/dexsim.mp4) · [提袋画面](preview/still_58.00s.jpg) · [状态曲线](analysis/checks.png) · [执行清单](manifest.json)。1921帧，30fps、1280×720，包含初始帧，实际推进64秒；没有变速或视频稳定化。当前仅为原生预览，尚未标为用户确认。",
        "",
        "直接调用原文件的 `build_scene`、`solver_config`、`W1BagController` 和 `W1BagChecks`，冻结原源码及完整资产。袋壁/提手1381节点、2712面，原材料、质量、接触和求解设置均保留；不再生成其他纸袋/硬框/软块。原demo没有独立刚性袋框，本版保留其袋口和提手的网格连接，不把自建框混称为原例。",
        "",
        "动作：0–10.5秒靠近并扶正侧躺纸袋；10.5–24秒左手换抓袋口，右手等待；24–36秒罐头装填；36–48秒纸盒装填；48–50秒释放和稳定。随后右手换抓袋口另一侧，56–58秒双手目标上移80mm，58–60秒保持，60–62秒下降，至64秒稳定。新增阶段仍由原控制器解IK，只增加TCP/夹爪目标，没有添加粒子跟随、世界固定、隐藏支撑或外力适配器。",
        "",
        "## Time & Memory",
        "",
        "| 实现 | 时间 / 显存 / RSS | 当前阶段 |",
        "| --- | --- | --- |",
        "| 原生 DexSim / DexUni | N/A | 原demo复现与提袋预览，未测性能 |",
        "| EmbodiChain / Isaac Lab | N/A | 未移植、未测量 |",
        "",
        "## Success & Other Metrics",
        "",
        "| 过程 | 实际秒 / 外步 / 有限状态 | 原demo全部最终检查 | 两件最终留存 | 带载保持期袋底最低离台 mm | 罐头 / 纸盒最低离台 mm |",
        "| --- | --- | --- | --- | ---: | --- |",
        f"| 原demo独立录像 | {original['actual_seconds']:.0f} / {original['actual_steps']} / {original['samples']} | 通过 | 2/2 | N/A | N/A |",
    ]
    for name, m in (("提袋无相机试跑", pilot), ("当前完整录像", current)):
        lift = m["lift"]
        loads = lift["payload_bottom_min_clearance_m"]
        text.append(
            f"| {name} | {m['actual_seconds']:.0f} / {m['actual_steps']} / {m['samples']} | 50秒时通过 | 2/2 | {lift['bag_bottom_min_clearance_m']*1000:.3f} | {loads[0]*1000:.3f} / {loads[1]*1000:.3f} |"
        )
    text += [
        "",
        f"两次提袋运行在58–60秒的全部121个实际外步状态中，两件商品的完整原始圆柱/盒体包围范围均位于纸袋拟合坐标的原判定区。当前袋体拟合位置平均提升{current['lift']['mean_fitted_bag_lift_m']*1000:.3f}mm（相对56秒），试跑{pilot['lift']['mean_fitted_bag_lift_m']*1000:.3f}mm；80mm是机器人目标提升，实物有柔性变形与接触滑移，不能把目标值冒充实测值。两次实际具名qpos回放误差均为0，Graph captured，完整状态有限。",
        "",
        "## Coverage & Status",
        "",
        "| 核查 | 结果 | 范围 |",
        "| --- | --- | --- |",
        "| 原场景与材料 | 资产/参考代码逐文件SHA、初始化质量/半径/材料核验 | 原纸袋、提手、两件动态商品；0世界固定粒子 |",
        "| 原demo逐步检查 | 完整原流程及提袋两次运行通过 | 每步桌面三角相交、关节限位/速度、手部跟踪、袋口弯曲/袋体拟合、物品范围；每10步提手穿纸检查 |",
        f"| 机器人自碰撞及桌面 | 当前{geometry['samples']}个实际姿态无非相邻链节/桌面网格相交，试跑同样通过 | FCL原URDF碰撞三角面，排除同一固定刚体和直接机械邻接，离散外步 |",
        "| 袋壁自碰撞及连接 | 原配置self-contact开启，袋壁/提手共享网格连接 | 原连接32个跨纸壁/提手三角面；不声称穷尽所有自交或连续碰撞 |",
        "| 性能、跨平台和成功率 | N/A | 仅场景重做预览，不外推成功率或质量排名 |",
        "",
        "求解设置沿用原例：1/60秒外步、6子步、12次迭代、one_way/kinematic，原DAT、多层校正和Chebyshev、自碰撞/接触参数均由未改动的原配置函数生成。原生DexSim dfceb00…、Newton1.6.0、Warp1.17.0，详细实际开关见manifest。未改SDK、生产代码、C01正式样本或已有已确认视频。",
        "",
        "复现：仓库根目录，选择已安装的原demo环境并指定不存在的新输出目录。",
        "",
        "```bash",
        "PYTHONPATH=/home/oem/code/repos/dexsimforembodychain/dexsim/modules/dexsim/build_Release/lib/python_package:/home/oem/miniconda3/envs/dexsim1/lib/python3.11/site-packages .venv/bin/python scripts/benchmark/tech_reports/physics/platforms/dexuni/dexsim/c02.py --lift --out /tmp/c02_reference_new",
        "```",
        "",
        "去掉`--lift`即可独立录制未追加动作的50秒原demo。`--no-video`只关闭录像；未提供任何性能计时。每个运行保留输入、冻结原例及记录脚本、具名关节命令、全部状态和最终检查结果。诊断尝试及原版结果归档见analysis/attempts.json；上一版用户否定场景完整归档于analysis/rejected_custom_scene.zip。当前发布路径映射见analysis/publication.json，保护核验见analysis/final_validation.json。",
    ]
    (root / "REPORT.md").write_text("\n".join(text) + "\n")
    (root / "README.md").write_text(
        "# C02 原demo重做版\n\n[完整64秒视频](preview/dexsim.mp4) · [场景与核查](REPORT.md)。原纸袋扶正，罐头和纸盒装入，再由袋口双手带载提起。上一版自建框场景已被用户否定并归档。当前仅原生预览，未移植和测性能。\n"
    )
    print("Reference-based C02 report rebuilt from existing states")


if __name__ == "__main__":
    main()
