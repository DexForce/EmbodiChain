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
"""Report actual C02 recordings, including explicit trajectory adaptations.

Run: .venv/bin/python -m scripts.benchmark.tech_reports.reporting.dexuni.report_c02_ports
Only reads recorded results; never advances simulation or measures performance.
"""

from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any
import numpy as np

__all__ = ["build", "main"]
ROOT = (
    Path(__file__).resolve().parents[2].parents[2]
    / "outputs/benchmarks/tech_reports/dexuni/c02_bag_frame_packing"
)


def read(path: Path) -> dict[str, Any]:
    """Read one retained metadata artifact."""
    return json.loads(path.read_text())


def _mm(value: float | None) -> str:
    """Format an observed distance, preserving missing stages."""
    return "未执行到该阶段" if value is None else f"{value * 1000:.2f}mm"


def build() -> None:
    """Publish the recorded task outcome separately from strict example checks."""
    base = ROOT / "platforms"
    ec, il = base / "embodichain", base / "isaaclab"
    em, im = [read(p / "analysis/current_metrics.json") for p in (ec, il)]
    ef, inf = [read(p / "manifest.json") for p in (ec, il)]
    es, ins = [read(p / "analysis/scene_checks.json") for p in (ec, il)]
    eg, ig = [read(p / "analysis/robot_geometry.json") for p in (ec, il)]
    assert (
        ef["status"] == "completed"
        and em["lift"]["completed"]
        and es["status"] == "passed"
    )
    assert im["samples"] == inf["actual_steps"] + 1
    done = bool(
        inf["status"] == "completed"
        and im["lift"]["completed"]
        and im["lift"]["both_loaded_throughout_hold"]
        and all(im["final_loaded"])
        and ins["checks"]["upright"]
        and ins["checks"]["original_shape_limits"]
    )
    with np.load(il / "preview/states.npz") as states:
        angle = float(
            np.degrees(np.arccos(np.clip(states["bag_rotation"][-1, 2, 2], -1, 1)))
        )
    iterations = inf["effective"]["vbd"]["iterations"]
    substeps = inf["effective"]["num_substeps"]
    cfg = inf.get("controller_trajectory")
    adapted = bool(cfg)
    title = (
        "两平台完成装袋与提袋" if done else "EmbodiChain 完成，Isaac 当前录像未完成任务"
    )
    outcome = (
        "完成两件装袋、提袋、悬停和放回"
        if done
        else f"在{im['actual_seconds']:g}s停止，最后袋体倾斜{angle:.2f}°；完整装袋和提袋未验证完成"
    )
    motion = (
        "Isaac使用原反馈控制器。"
        if not adapted
        else "Isaac使用明确记录的机器人动作适配，参数在其inputs/controller_trajectory.json；两版机器人命令存在差异。"
    )
    if cfg and cfg.get("assist_right_tip"):
        motion += "具体为双手辅助立袋，纸袋夹持每侧2mm，左手在10.5–12.2s原地缓慢松开；右手保持夹持直到左手换抓完成，再退出并于24s开始原两件商品装袋与提袋流程。"
    failed = [name for name, passed in ins["checks"].items() if not passed]
    deliberate = [
        name
        for name in failed
        if cfg
        and cfg.get("assist_right_tip")
        and name in ("right_wait_clear", "right_relaxed_before_packing")
    ]
    residual = [name for name in failed if name not in deliberate]
    strict = "全部通过" if not failed else "未全部通过：" + "、".join(failed)
    explanation = (
        "原例子全部离线判据均通过。" if not failed else f"原例子严格判据{strict}。"
    )
    if deliberate:
        explanation += "其中right_wait_clear/right_relaxed_before_packing属于原单手立袋流程要求，与明确采用的右手辅助流程不同，保留原检查结果以便核对。"
    if residual:
        explanation += (
            "其余未通过项不能按流程差异忽略；实际数值见下表及scene_checks.json。"
        )
    lip_path = ROOT / "analysis/trajectory_repair/lip_bow_diagnostic.json"
    if lip_path.exists():
        lip = read(lip_path)
        if lip.get("states_sha256") == im["states_sha256"]:
            explanation += (
                f"袋口最大弯曲发生在{lip['max_time_s']:.2f}s的松手交接阶段；"
                f"11s之后最大{_mm(lip['max_after11s_m'])}。"
                f"首次抓袋前也已达到{_mm(lip['max_before_grasp_m'])}，"
                "仍超过原30mm限制，不能因后续流程完成而忽略。"
            )
    e, i = em["lift"], im["lift"]
    eclear, iclear = (
        e["payload_bottom_min_clearance_m"],
        i["payload_bottom_min_clearance_m"],
    )
    igoods = "未执行到该阶段" if iclear is None else "/".join(_mm(v) for v in iclear)
    pilot = read(ROOT / "analysis/platform_port/ec_pilot_current_metrics.json")
    media = read(ROOT / "analysis/trajectory_repair/comparison.json")
    pilot_report = ROOT / "pilot_performance/REPORT.md"
    performance_note = (
        "已另行完成每平台一次完整无摄像机[性能摸测](../pilot_performance/REPORT.md)；"
        "本页的录像验证本身没有计时，不据调试墙钟时间排名。"
        if pilot_report.exists()
        else "本轮没有性能摸测或正式测量，不据调试墙钟时间排名。"
    )
    comparison_note = (
        "两版完整原速录像按实际物理时钟并排。"
        if done
        else "对照展示EmbodiChain全部64s；Isaac录像结束后保持最后图像，并持续标明停止及停帧，不代表之后继续模拟。"
    )
    report = f"""# C02 — {title}

EmbodiChain已完成原W1纸袋立袋、放入罐头与纸盒、双手提袋、悬停和放回。原速64s/3840步的录像和独立无摄像机完整复跑均保留。

Isaac Lab / Isaac Sim当前结果：{outcome}。{motion}

[完整对比视频](comparison.mp4) · [EmbodiChain完整视频](embodichain/preview/video.mp4) · [Isaac当前视频](isaaclab/preview/video.mp4) · [复现命令](README.md) · [已接受的原版](../preview/dexsim.mp4)

## Time & Memory

| 平台 | 完成一步耗时 | 进程显存 | 本轮性质 |
|---|---|---|---|
| EmbodiChain | 未测 | 未测 | 完整场景与录像验证 |
| Isaac Lab / Isaac Sim | 未测 | 未测 | 轨迹适配与场景验证 |

{performance_note}

## Success & Other Metrics

| 检查项 | EmbodiChain | Isaac Lab / Isaac Sim |
|---|---|---|
| 实际物理时长/外步/状态 | {em['actual_seconds']:g}s /{ef['actual_steps']} /{em['samples']} | {im['actual_seconds']:g}s /{inf['actual_steps']} /{im['samples']} |
| 连续完成装袋/提袋流程 | 完成 | {'完成' if done else '未完成'} |
| 求解策略 | 原DexUni12迭代×6子步，DAT/多层/Chebyshev | 普通VBD{iterations}迭代×{substeps}子步 |
| 机器人控制 | 原反馈控制+已接受提袋尾段 | {'明确的机器人轨迹适配' if adapted else '原反馈控制+已接受提袋尾段'} |
| 有限状态/实际时钟/命名关节命令 | 通过 | {im['finite']} /时钟误差{im['max_clock_error_s']:.2g}s /关节误差{im['max_joint_replay_error']:.2g} |
| 最大TCP跟踪误差 | {_mm(em['max_tcp_target_error_m'])} | {_mm(im['max_tcp_target_error_m'])} |
| 原袋体形状/直立判据 | 通过 | {ins['checks']['original_shape_limits']} /{ins['checks']['upright']} |
| 袋口最大弯曲（原上限30mm） | {_mm(es['max_lip_bow_m'])} | {_mm(ins['max_lip_bow_m'])} |
| 悬停拟合平均提升 | {_mm(e['mean_fitted_bag_lift_m'])} | {_mm(i['mean_fitted_bag_lift_m'])} |
| 悬停袋底最小离桌 | {_mm(e['bag_bottom_min_clearance_m'])} | {_mm(i['bag_bottom_min_clearance_m'])} |
| 悬停罐头/纸盒底最小离桌 | {'/'.join(_mm(v) for v in eclear)} | {igoods} |
| 两件商品悬停/最终保留 | 全部{e['hold_samples']}悬停状态/是 | {str(i['both_loaded_throughout_hold'])}/{im['final_loaded']} |
| 机器人自身/桌面FCL碰撞样本 | {len(eg['self_collision_samples'])}/{len(eg['table_collision_samples'])}，共{eg['samples']}姿态 | {len(ig['self_collision_samples'])}/{len(ig['table_collision_samples'])}，共{ig['samples']}姿态 |
| 原例子全部离线判据 | 全部通过 | {strict} |
| 独立完整复跑 | 完成，提升{_mm(pilot['lift']['mean_fitted_bag_lift_m'])}，两件保留 | {'本轮为一段完整物理录像，未作成功率测量' if done else '尚无完整通过记录'} |

{explanation}任务目标完成与全部原例子严格判据通过分别报告，不能互相替代。两版都没有调用依赖原Scene的完整运行时检查器；原纸袋/机器人判据由独立离线检查读取实际记录状态执行，详见scene_checks.json。FCL覆盖记录的60Hz姿态；提手与纸面穿越每10帧检查，离散结果不证明连续时间无碰撞或全部袋壁无自交。没有筛选正式性能样本，也不推导成功率。

## Leaderboard

| 项目 | 结论 |
|---|---|
| 任务流程 | EmbodiChain完成；Isaac{'完成装袋与提袋，严格几何分项见上表' if done else '当前录像未完成'} |
| 性能排名 | 未测量，不排名 |

场景保留原1381节点/2712面纸袋、柔性提手、罐头、纸盒、桌面和W1。提手/袋口有32个跨分区连接面，没有独立刚性袋框。两版各自推进自由纸袋及两个动态商品；没有新增耗散力、固定节点、隐藏支撑或袋子/商品位姿跟随。初始化10组纸袋位置、质量、半径、静止张量、拓扑和材料与原版逐位一致。普通Newton VBD没有原DexUni DAT、多层/Galerkin校正和Chebyshev加速；本轮没有改写求解器或调整材料补偿。

原轨迹96×24的Isaac在11s倾斜44.57°失败，旧录像/源代码/输入/原始状态保存在[原轨迹失败归档](../analysis/trajectory_repair/baseline_isaac.zip)。继续调试发现，单纯前推55mm不能解决释放后翻倒；增加30°腕部转角超出所选IK路径的可达范围（约48mm跟踪误差），不作为求解器能力证据；收紧夹持将松手前倾斜降至17.64°，但释放中仍向后翻倒。最新动作适配是否成功以本报告当前真实录像和指标为准，不能由这些短测推断Isaac普遍不支持该任务。新诊断均保留原始状态/输入/源代码和逐文件哈希，见analysis/trajectory_repair的归档及状态记录。

已识别的旧移植问题已修复：恢复实际袋底反馈；按原URDF补回29个碰撞网格、四个1mm手指SDF，仅11个实际碰撞面进入全表面候选对。网格顶点/面和所有记录姿态的固定变换独立核验。Isaac主进程只用安装的Newton/Warp/Isaac运行时；原Python3.11进程仅提供机器人IK。未接触的桌腿导入差异和浮点三角面积舍入继续披露在analysis/platform_port记录中。

当前对照{media['new_frames']}帧、30fps。{comparison_note}两版原始视频不因拼接而修改；完整比较由analysis/trajectory_repair/build_comparison.py从视频重新生成，逐文件哈希及停帧语义见comparison.json。既有原版C02、C01正式数据、M04/M05和平台源码仍由保留校验保护。
"""
    (base / "REPORT.md").write_text(report)
    trajectory_arg = (
        ""
        if not adapted
        else f" --trajectory {il / 'inputs/controller_trajectory.json'}"
    )
    readme = f"""# C02 平台版本与复现

[结果与限制](REPORT.md) · [完整对比](comparison.mp4) · [EmbodiChain完整录像](embodichain/preview/video.mp4) · [Isaac当前录像](isaaclab/preview/video.mp4)

当前Isaac：{outcome}。{motion}{performance_note}须阅读报告中的原例子严格几何分项。

在 `/home/oem/code/repos/EmbodiChain` 执行，`--out`必须是不存在的新目录：

```bash
PYTHONPATH=/home/oem/code/repos/dexsimforembodychain/dexsim/modules/dexsim/build_Release/lib/python_package:/home/oem/miniconda3/envs/dexsim1/lib/python3.11/site-packages .venv/bin/python scripts/benchmark/tech_reports/physics/platforms/dexuni/embodichain/c02.py --out /tmp/c02_ec_new

/home/oem/app/IsaacLab/.venv/bin/python scripts/benchmark/tech_reports/physics/platforms/dexuni/isaaclab/c02.py --visualizer none --iterations {iterations} --substeps {substeps}{trajectory_arg} --out /tmp/c02_il_new
```

Isaac主进程不要设置SDK的PYTHONPATH。`--no-video`关闭摄像机；`--until 2`为短段诊断。未指定`--trajectory`即使用原机器人动作，不会自动采用已适配流程。

只读分析和媒体重建，不会运行物理或覆盖旧正式测量：

```bash
.venv/bin/python -m scripts.benchmark.tech_reports.physics.experiments.dexuni.c02_analysis --run <run>
/home/oem/code/engine/dexsim/.env-3.12/bin/python scripts/benchmark/tech_reports/physics/experiments/dexuni/c02_geometry.py --run <run>
.venv/bin/python scripts/benchmark/tech_reports/reporting/dexuni/verify_c02_scene.py --run <run>
.venv/bin/python scripts/benchmark/tech_reports/reporting/dexuni/build_c02_comparison.py
.venv/bin/python -m scripts.benchmark.tech_reports.reporting.dexuni.report_c02_ports
# Historical validate_ports.py checks pre-cleanup archives; see the cleanup ledger.
```

原轨迹失败视频、原始状态和新调试尝试均保留在analysis/trajectory_repair归档中。几何预检明确标注physical_simulation=false，不作为物理完成或性能结果。完整对比对每个已结束面板持续显示停帧提示。
"""
    (base / "README.md").write_text(readme)
    (base / "report_provenance.json").write_text(
        json.dumps(
            dict(
                generator_sha256=hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                performance_measured=False,
                performance_scope="Current preview recordings only; any independent informal pilot is linked separately",
                independent_pilot_report=(
                    "../pilot_performance/REPORT.md" if pilot_report.exists() else None
                ),
                embodichain_task="completed",
                isaac_task="core_task_completed" if done else "not_completed",
                isaac_original_strict_checks_passed=ins["status"] == "passed",
                isaac_failed_original_checks=failed,
                isaac_deliberate_control_flow_differences=deliberate,
                isaac_controller_trajectory=cfg,
            ),
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    """Build current evidence-based reports without running physics."""
    build()


if __name__ == "__main__":
    main()
