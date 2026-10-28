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
"""Build the P-03 three-chapter draft, figures and three-table benchmark summary."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    root = Path("outputs/benchmarks/P-03")
    analysis = root / "analysis/grasp-v1"
    selection = json.loads((analysis / "selection.json").read_text())
    rows = json.loads((analysis / "metrics.json").read_text())
    runs = json.loads((analysis / "runs.json").read_text())
    by_id = {r["run_id"]: r for r in runs}
    report = root / "reports"
    report.mkdir(exist_ok=True)
    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    labels = {
        "embodichain": "EmbodiChain / DexUni",
        "isaaclab": "Isaac Lab / PhysX TGS",
    }
    shapes = {"cube": "立方体", "cylinder": "圆柱"}
    colors = {"embodichain": "#007A87", "isaaclab": "#CC681B"}
    quality = {}
    for shape, platforms in selection["quality"].items():
        for platform, run_id in platforms.items():
            quality[(shape, platform)] = [r for r in rows if r["run_id"] == run_id]
            assert len(quality[(shape, platform)]) == 10, (
                run_id,
                len(quality[(shape, platform)]),
            )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    plot_manifest = []
    for row, shape in enumerate(["cube", "cylinder"]):
        for platform in ["embodichain", "isaaclab"]:
            run_id = selection["quality"][shape][platform]
            data = [
                dict(np.load(analysis / run_id / f"episode_{i:02d}_traces.npz"))
                for i in range(10)
            ]
            t = data[0]["t"]
            values = [
                np.array([d["lift_m"] for d in data]) * 1000,
                np.array([d["slip_m"] for d in data]) * 1000,
                np.abs(np.array([d["finger_error_m"] for d in data])) * 1000,
            ]
            for c, v in enumerate(values):
                for line in v:
                    axes[row, c].plot(
                        t, line, color=colors[platform], alpha=0.12, linewidth=0.6
                    )
                axes[row, c].plot(
                    t,
                    np.median(v, axis=0),
                    color=colors[platform],
                    label=labels[platform],
                    linewidth=1.6,
                )
            plot_manifest.append(
                dict(
                    run_id=run_id,
                    initials=list(range(10)),
                    traces_directory=str((analysis / run_id).resolve()),
                )
            )
        for c in range(3):
            ax = axes[row, c]
            ax.axvspan(3, 5, color="#c6dadd", alpha=0.2)
            ax.axvspan(5, 7, color="#ead9bd", alpha=0.22)
            ax.set_xlabel("Simulation time (s)")
        axes[row, 0].axhline(100, ls=":", color="#777", lw=1)
        axes[row, 0].set_ylabel(f"{shape.title()} lift (mm)")
        axes[row, 1].set_ylabel("Object / TCP displacement (mm)")
        axes[row, 1].set_xlim(3, 7)
        axes[row, 1].set_ylim(-0.005, 0.15)
        axes[row, 2].set_ylabel("Finger mimic difference (mm)")
        axes[row, 2].set_xlim(1, 9)
    axes[0, 0].legend(loc="lower right", fontsize=8)
    for ext in ["png", "svg", "pdf"]:
        fig.savefig(figures / f"p03-quality.{ext}", dpi=160)
    plt.close(fig)
    (figures / "p03-quality-data.json").write_text(
        json.dumps(
            dict(
                series=plot_manifest,
                aggregation="Thin lines: all 10 initials. Bold: pointwise median. Relative displacement uses t=3 s anchor; it includes settling.",
            ),
            indent=2,
        )
        + "\n"
    )

    def mean_range(values: list[dict], key: str, digits: int = 3) -> str:
        a = np.array([r[key] for r in values])
        return f"{a.mean():.{digits}f} [{a.min():.{digits}f}, {a.max():.{digits}f}]"

    text = [
        "# P-03：接触抓取、保持、扰动与放回\n",
        "> 实验初稿 · 2026-10-23。质量实验和原生素材已完成；正式吞吐矩阵尚未完成，当前不作速度排名。\n",
        "## 第一章　对比与结论\n",
        "应用 USD mimic 解析修复后，EmbodiChain / DexUni 已完成本场景的真实接触抓取：立方体和圆柱各 10 个初态均完成抬升、保持、横向扰动和放回释放；Isaac Lab / PhysX TGS 同样为各 10/10。实验没有在运动中重写实际位姿、清零动力学状态或添加抓持约束。\n",
        "共同条件为 100 g、50 mm 物体，摩擦系数 0.8，抬升命令 100 mm，完整窗口 10 s。结果只覆盖这组资产、控制器和小幅扰动，不代表抓持力极限。下表为 10 个初态的**均值 [最小值, 最大值]**；滑移、穿透列先取每条轨迹的规定窗口最大值，再跨初态汇总。\n",
        "| 物体／平台 | 完成循环 | 3 s 抬升量（mm） | 3–5 s 相对位移最大值（mm） | 5–7 s 新增相对位移（mm） | 手指穿透最大值（mm） |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for shape in ["cube", "cylinder"]:
        for platform in ["embodichain", "isaaclab"]:
            a = quality[(shape, platform)]
            text.append(
                f"| {shapes[shape]}／{'DexUni' if platform=='embodichain' else 'PhysX TGS'} | {sum(r['cycle_completed'] for r in a)}/10 | {mean_range(a,'lift_at_3s_mm',2)} | {mean_range(a,'hold_slip_max_mm')} | {mean_range(a,'perturb_increment_from_5s_max_mm',4)} | {mean_range(a,'finger_penetration_max_mm')} |"
            )
    text += [
        "\nDexUni 的抬升量接近命令值，保持稳定后，规定扰动期间没有出现明显滑移。约 0.12 mm 的相对位移主要形成于抬升结束后的短暂稳定过程；不能把以 3 s 为基准的位移全部归因于 5 s 后的外力。两端扰动窗口新增位移都小于 0.003 mm，接近状态采样的微小波动量级，本轮不据此给摩擦精度排名。\n",
        "接触细节仍有区别：DexUni 的几何穿透约为毫米级，受力保持时两指位置差约 4.6–4.8 mm；PhysX TGS 在本配置下分别约 0.04–0.06 mm 和 0.067 mm。mimic 修复解决了“从指不跟随”的导入问题，受力约束误差仍应单独测量。这些误差没有造成本轮掉落，但对毫米级定位任务需要进一步评估。\n",
        "实用配置要点：使用动态关节和**双向耦合**，让物体接触力反馈给机械臂；先校准机械臂的重力下垂，再测夹持效果。本轮从原建议 Kp=120、Kd=15 调到两端共同的 Kp=2000、Kd=100，保留原力矩上限和全部试跑记录。\n",
        "![两种形状的抬升、相对位移与 mimic 误差](../figures/p03-quality.png)\n",
        "*细线为全部 10 个初态，粗线为逐时刻中位数；阴影分别对应保持和扰动窗口。此图为原始状态计算的分析图。实际仿真画面见下列原片。*\n",
        "[同视角原生视频与帧、轨迹核验](../videos/index.md)。两端固定相机、960×720、20 fps、1× 播放；物理步长仍为 5 ms。\n",
        "## 第二章　EmbodiChain / DexUni 详细报告\n",
        "### 模型与执行\n",
        "使用本地 Factory `franka_mimic.usd`，固定基座，原生两指 mimic；桌面高 0.75 m。目标为 50 mm 立方体、直径和高度各 50 mm 的 64 边近似圆柱。圆柱的最大径向近似误差约 0.030 mm，两端共用同一网格和显式惯量。机器人源文件、依赖文件和输入数组均保存哈希。\n",
        "运行版本为 EmbodiChain `f179b555`，以及原安装基线加 MR !1501 **解析补丁**的冻结 DexSim 副本；没有安装 MR 分支的整套新依赖。Newton 1.6.0、Warp 1.17.0；960 个运行文件通过哈希核验，求解器源码未修改。共享 Conda 的旧 parser 未被替换，复现需用本文启动入口指定的隔离包。\n",
        "DexUni 实际采用关节 MuJoCo 分支与物体 VBD 分支的双向耦合，动态关节、VBD 100 次迭代、每物理步一个子步；双向耦合参数为既有默认值，VBD 睡眠开启。物理周期 5 ms，控制周期 20 ms。两端重放同一组离线关节目标和外力。不同求解器的迭代数不表示相等的计算预算。\n",
        "| 共同参数 | 数值 |\n|---|---|\n| 机械臂驱动 | 力型 PD，Kp=2000 N·m/rad，Kd=100 N·m·s/rad |\n| 主指驱动 | Kp=1000 N/m，Kd=20 N·s/m，上限 40 N |\n| 从指驱动 | Kp=Kd=0，使用原生 mimic |\n| 机械臂力矩上限 | 前四关节 87 N·m，后三关节 12 N·m |\n| 配置限速 | 前四臂关节 2.175 rad/s，后三关节 2.610 rad/s；两指 0.05 / 0.04 m/s |\n| 物体／接触 | 质量 0.1 kg，静／动摩擦 0.8，恢复系数 0；几何 rest offset 0，contact offset 1 mm |\n| 外力 | 5–7 s 质心 +X 方向，0.1962·sin(4π(t−5)) N |\n",
        "动作顺序为 0–1 s 接近、1–2 s 闭合、2–3 s 抬升、3–5 s 保持、5–7 s 扰动、7–9 s 放回、9–10 s 张开。10 个初态包含一个标称位置及九个 xy ±15 mm、yaw ±5° 的固定样本。\n",
        "### 评价定义与数据核验\n",
        "抓持相对位姿为 `inverse(T_TCP) · T_object`，保持位移以 t=3 s 为基准；扰动新增位移另以 t=5 s 为基准，避免混入此前的稳定过程。抖动、转角、速度和每步距离均保留在 CSV 与派生数组中。\n",
        "离桌按物体最低点高于桌面接触邻域识别；邻域 2 mm 来自两接触面的各 1 mm contact offset，不是抬升误差合格线。完整循环要求 3–7 s 连续离桌且双指接触，10 s 回到桌面邻域并与双指分离。中途掉落后恢复仍计为未完成。采样分辨率 5 ms。\n",
        "穿透由原 USD 手指凸包和共同目标凸包的分离轴计算，负间隙对应几何最小分离平移，不含接触 margin；正间隙仅为分离轴投影间隙。没有将未导出的接触力填成零。当前两端不报告原生接触冲量，DexUni 公开 ContactQuery 仍不提供该路径的统一结果；接触 buffer 容量与 overflow 的统一读回也尚未取得。\n",
        "两环境独立施加 0.1 / 0.2 N、持续 0.1 s，理论速度增量 0.1 / 0.2 m/s：DexUni 实测 0.100136 / 0.200272 m/s，约 +0.136%；撤力后 0.1 s 的 x 速度增量为 0。源资产惯量统一到连杆坐标后，两端最大矩阵分量差约 1.10×10⁻⁷ kg·m²，质量差约 2.38×10⁻⁷ kg。\n",
        "### 逐初态结果\n",
    ]
    for platform in ["embodichain", "isaaclab"]:
        if platform == "isaaclab":
            text += [
                "\n## 第三章　Isaac Lab / PhysX TGS 详细报告\n",
                "使用 Isaac Lab `ae37b028`、Isaac Sim 6.1，同一 Factory USD、目标网格、质量／惯量、材料、驱动和命令数组。实际求解器为 GPU PhysX **TGS**，位置迭代 16、速度迭代 1；不是 PGS。配置限速读回与 DexUni 一致，但配置相同不保证约束的实际误差相同。\n",
                "两种形状各 10/10 完成；各初态的抬升量约 99.21–99.28 mm。两环境外力校准得到 0.09999992 / 0.19999984 m/s，撤力后 x 速度保持不变。质量／接触分析、原始保存和循环事件定义与第二章一致。\n",
                "### 逐初态结果\n",
            ]
        text += [
            "| 物体 | 初态 | 完整循环 | 3 s 抬升（mm） | 保持位移 max（mm） | 扰动新增位移 max（mm） | 手指穿透 max（mm） | 放回 xy 误差（mm） |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for shape in ["cube", "cylinder"]:
            for r in quality[(shape, platform)]:
                text.append(
                    f"| {shapes[shape]} | {r['initial']} | {'完成' if r['cycle_completed'] else '未完成'} | {r['lift_at_3s_mm']:.3f} | {r['hold_slip_max_mm']:.4f} | {r['perturb_increment_from_5s_max_mm']:.4f} | {r['finger_penetration_max_mm']:.3f} | {r['return_xy_error_mm']:.3f} |"
                )
        a = sum([quality[(s, platform)] for s in ["cube", "cylinder"]], [])
        speeds = np.array([r["joint_speed_peak"] for r in a])[:, 7:9].max(0) * 1000
        text.append(
            f"\n受力保持时 mimic 最大差的跨初态均值／范围为 {mean_range(a,'mimic_error_hold_max_mm')} mm。测得主指／从指全轨迹速度峰值分别为 {speeds[0]:.1f} / {speeds[1]:.1f} mm/s；这些是实际状态读数，不能用配置的 50 / 40 mm/s 代替。\n"
        )
        if platform == "embodichain":
            text += [
                "### 试跑保留与适用限制\n",
                "低机械臂增益试跑中，两端均明显偏离抓取目标。提高共同增益后，PhysX 已能抓取；DexUni 单向耦合试跑未抬起目标。随后启用已有双向耦合，并显式应用目标碰撞参数，才完成当前抓取。由于后一次同时修正了目标 margin，不能把改善量全部归因于单一选项；单向耦合本身不提供物体对机械臂的反力反馈，不适合本次受力对照。\n",
                "原圆柱资产的 64 边端面导致 DexSim 导入失败，已改为相同顶点的三角面；失败日志及原输入仍保留。立方体 v3 与 v4 的输入 USD 字节一致、物理参数一致，因此沿用 v3 质量数据。\n",
                "本轮显示功能链路已打通。受力 mimic 偏差、速度约束误差和接触穿透的具体来源仍需独立收敛／参数实验；这里不将其归因于 MR !1501 的解析修复，也不据本场景推断所有 DexUni 任务的精度。\n",
            ]
    text += [
        "### 批量运行与尚未完成项\n",
        "两端已运行 N=1 的完整环境步预检，以及两种形状的 N=64 物理循环／完整环境步预检。N=64 使用标称初态的 64 个复制体，不能当作 64 个独立初态。所有复制体的完整状态均保留；具体事件分布见[批次摘要](benchmark-summary.md)。\n",
        "本机 GPU 为 RTX 5090 D v2（24,455 MiB），驱动 580.159.04。正式测速期间需要 GPU 独占。本轮监测到同机其他 pytest／CI GPU 进程，现有耗时只作为诊断，未形成正式吞吐排名。N=1/64/256/1024 的 124 组规定重复、独立同步延迟和第二 GPU 复测尚未完成；本机没有第二张 GPU。P-03 **尚不能按完整 Spec 结项**。\n",
        "### 原始数据与复现\n",
        "全部来源见[数据索引](data-index.md)；正式质量选择固定于 [selection.json](../analysis/grasp-v1/selection.json)。逐物理步 `episode_XX_state.npy` 为 `(2001, N, 52)`，配套 time 数组；列定义位于同目录 `run.json`，输入、执行源码、实际模型读回、版本、命令、日志与退出状态同目录保存。失败运行没有删除。\n",
        "[逐初态／复制体 CSV](../analysis/grasp-v1/metrics.csv) · [指标定义](../analysis/grasp-v1/method.json) · [独立核验](../analysis/grasp-v1/validation.json) · [原始输入](../prepared/grasp-v1/commands.npz) · [运行依赖清单](../runtime/p03-frozen-v1/manifest.json)。\n",
        "在项目根目录用 `embodichain` 环境执行 `python -m scripts.benchmark.tech_reports.reporting.p03_review`、`p03_validate`、`p03_media_review`、`p03_report`（后三者沿用同一 reporting 模块前缀），可离线重算。新增仿真用 `physics.experiments.p03_batch --config scripts/benchmark/tech_reports/assets/physics/P-03/protocol-v4.json`，指定形状、数量、`--mode physics|env` 和新 tag；正式测速加 `--require-idle`。每次自动新建 run_id，不覆盖旧数据。\n",
    ]
    article = "\n".join(text)
    article = article.replace(
        "接近状态采样的微小波动量级，本轮不据此给摩擦精度排名",
        "尚未单独量化微米尺度的数值噪声与重复性，本轮不据此给摩擦精度排名",
    )
    article = article.replace(
        "连续离桌且双指接触", "连续离桌且保持在双指的几何接触邻域"
    )
    if (root / "videos/validation.json").exists():
        media = json.loads((root / "videos/validation.json").read_text())["records"]
        pair = {r["platform"]: r["run_id"] for r in media if r["shape"] == "cube"}
        snapshots = (
            "\n| EmbodiChain / DexUni：t=3 s | Isaac Lab / PhysX TGS：t=3 s |\n|---|---|\n"
            + f"| ![DexUni 立方体抬升](../runs/{pair['embodichain']}/native_camera/frame_0059.png) | ![PhysX 立方体抬升](../runs/{pair['isaaclab']}/native_camera/frame_0059.png) |\n"
        )
        article = article.replace(
            "## 第二章　EmbodiChain / DexUni 详细报告",
            snapshots + "\n## 第二章　EmbodiChain / DexUni 详细报告",
        )
    (report / "P-03.md").write_text(article)

    idx = [
        "# P-03 原始数据索引\n",
        "权威质量数据为下列四组；其他试跑、接口预检、批量预检和录像均单独保留。\n",
    ]
    for shape, platforms in selection["quality"].items():
        for platform, rid in platforms.items():
            idx.append(
                f"- {shapes[shape]}／{labels[platform]}：[运行目录](../runs/{rid}/run.json)，初态 0–9，10 s × 200 Hz。"
            )
    idx += [
        "\n## 全部本轮尝试\n",
        "| run_id | 状态 | 输入版本 | 计时说明 |",
        "|---|---|---|---|",
    ]
    for r in runs:
        scope = (
            "相机采集"
            if r["record"]
            else (
                "外力校准"
                if r["calibrate"]
                else (
                    "共享 GPU 诊断"
                    if r["gpu_competitor_pids"]
                    else "预检／质量运行，非正式性能批次"
                )
            )
        )
        if (
            r["record"]
            and r["platform"] == "isaaclab"
            and "partition_fix" not in r["run_id"]
        ):
            scope = "录像无效：相机未显示环境分区；物理轨迹保留"
        idx.append(
            f"| [{r['run_id']}](../runs/{r['run_id']}/run.json) | {r['status']} | {r['config_version']} | {scope} |"
        )
    idx += [
        "\n原资产预检和 mimic 单独复验不重复归入本轮质量样本，见 [资产预检](asset-preflight.md) 和 [mimic 修复复验](mimic-mr1501-readiness.md)。\n",
        "机器人资产及导出的手指几何遵循原 NVIDIA 资产条款；本轮未上传、未授予再分发许可。实验自建目标几何与实验代码注明 Apache-2.0；结果和原生媒体的公开许可尚未指定。\n",
    ]
    (report / "data-index.md").write_text("\n".join(idx))
    summary = [
        "# P-03 Benchmark 运行摘要\n",
        "本轮质量、校准、功能预检和录制的结果摘要。任务完成事件见[详细报告](P-03.md)。共享 GPU 下不进行速度排序，缺失项不填零。诊断窗口含目标／外力写入、公共物理推进、状态读取及 GPU 轨迹采样；env 行额外保留完整 BaseEnv.step／DirectRLEnv.step。\n",
        "## Time & Memory\n",
        "| 配置／run_id | N／范围 | cost_time_ms（完整 10 s 窗口均值） | cpu_delta_mb（采样窗口） | gpu_delta_mb（采样窗口） | peak_gpu_mb（进程） | 状态 |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for r in runs:
        eps = r["episodes"]
        windows = r.get("timing_windows", [])
        elapsed = f"{np.mean([e['wall_s'] for e in eps])*1000:.1f}" if eps else "未测"
        rd = [
            w["ram_peak_mib"] - w["ram_baseline_mib"]
            for w in windows
            if w["ram_peak_mib"] is not None
        ]
        gd = [
            w["gpu_peak_mib"] - w["gpu_baseline_mib"]
            for w in windows
            if w["gpu_peak_mib"] is not None and w["gpu_baseline_mib"] is not None
        ]
        delta_ram = f"{max(rd):.1f}" if rd else "未测"
        delta_gpu = f"{max(gd):.1f}" if gd else "未测"
        peak = r.get("gpu_peak_process_mib")
        scope = "录制" if r["record"] else "校准" if r["calibrate"] else r["mode"]
        summary.append(
            f"| [{r['run_id']}](../runs/{r['run_id']}/run.json) | {r['num_envs']}／{scope} | {elapsed} | {delta_ram} | {delta_gpu} | {peak if peak is not None else '未测'} | {r['status']}；{'存在 GPU 竞争' if r['gpu_competitor_pids'] else '诊断'} |"
        )
    summary += [
        "\n内存按 MiB 报告，沿用上游字段名 mb；20 ms 周期采样，增量为窗口峰值减起点附近基线，不代表精确瞬时峰值。正式性能 124 组及同步延迟尚未测量。\n",
        "## Success & Other Metrics\n",
        "| 配置／run_id | 独立初态数／每初态复制体 | 完整循环（已保存轨迹） | 抬升均值（mm） | 手指穿透 max（mm） | 状态／说明 |",
        "|---|---|---:|---:|---:|---|",
    ]
    for r in runs:
        a = [x for x in rows if x["run_id"] == r["run_id"]]
        if a:
            summary.append(
                f"| [{r['run_id']}](../runs/{r['run_id']}/run.json) | {len(set(x['initial'] for x in a))}／{r['num_envs']} | {sum(x['cycle_completed'] for x in a)}/{len(a)} | {np.mean([x['lift_at_3s_mm'] for x in a]):.3f} | {max(x['finger_penetration_max_mm'] for x in a):.3f} | {'录像重跑，不增加独立初态' if r['record'] else '全部保存复制体'} |"
            )
        else:
            summary.append(
                f"| [{r['run_id']}](../runs/{r['run_id']}/run.json) | — | 不适用 | — | — | {r['failure'] or '外力校准／尚无完整轨迹'} |"
            )
    summary += [
        "\n## Leaderboard\n",
        "| 同条件质量配置 | 完整循环 | 抬升量均值（mm） | 保持位移最大值的均值（mm） | 速度排名 |",
        "|---|---:|---:|---:|---|",
    ]
    for (shape, platform), a in quality.items():
        summary.append(
            f"| {shapes[shape]}／{labels[platform]} | {sum(x['cycle_completed'] for x in a)}/10 | {np.mean([x['lift_at_3s_mm'] for x in a]):.3f} | {np.mean([x['hold_slip_max_mm'] for x in a]):.4f} | 未形成独占 GPU 数据；不设综合名次 |"
        )
    (report / "benchmark-summary.md").write_text("\n".join(summary) + "\n")
    print(report / "P-03.md")


if __name__ == "__main__":
    main()
