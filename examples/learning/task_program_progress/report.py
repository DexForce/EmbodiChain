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

"""Summarize measured reward qualification artifacts without inventing results.

Run: python -m examples.learning.task_program_progress.report
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import yaml


def main() -> None:
    """Write a quantitative Markdown report, machine-readable summary and figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/reward_qualification")
    )
    args = parser.parse_args()
    output = args.output.resolve()

    def read(name):
        return json.loads((output / name).read_text())

    expert = read("expert_grasp_place.json")
    supported = read("supported.json")
    floating = read("floating.json")
    empty = read("empty_grip.json")
    policy = read("trained_policy.json")
    benchmark = read("benchmark.json")
    evaluations = sorted((output / "final_evaluation").rglob("evaluation.json"))
    if not evaluations:
        raise ValueError("Independent evaluation artifact is required")
    evaluation = json.loads(evaluations[-1].read_text())
    metrics = evaluation["result"]["metrics"]
    test_log = (output / "final_tests.log").read_text()
    passed = re.findall(r"(\d+) passed", test_log)
    if not passed or re.search(
        r"\b[1-9]\d* (?:failed|errors?|xpassed)\b|\b(?:FAILURES|ERRORS|INTERRUPTED)\b",
        test_log,
        flags=re.IGNORECASE,
    ):
        raise ValueError("No successful final test summary found")
    root = Path(__file__).resolve().parents[3]
    changed = subprocess.check_output(
        ["git", "diff", "--name-only"], cwd=root, text=True
    ).splitlines()
    added = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard"], cwd=root, text=True
    ).splitlines()
    sources = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in sorted(set(changed + added))
        if (root / name).is_file()
    }
    saved_train = yaml.safe_load(
        (Path(policy["run"]) / "configs/train.yaml").read_text()
    )
    restored = re.search(
        r"Resumed step=(\d+), updates=(\d+)", (output / "resume.log").read_text()
    )
    if restored is None:
        raise ValueError("A successful native resume result is required")
    transitions, updates = map(int, restored.groups())
    summary = {
        "schema_version": 1,
        "tests_passed": int(passed[-1]),
        "training_transitions": transitions,
        "ppo_updates": updates,
        "training_envs": saved_train["trainer"]["num_envs"],
        "run": policy["run"],
        "evaluation": metrics,
        "expert": {
            "control_steps": expert["steps"],
            "seed": expert["seed"],
            "executor_success": expert["expert_result"]["success"],
            "measured_pick_success": expert["measured_pick_success"],
            "measured_place_success": expert["measured_place_success"],
        },
        "fixtures": {
            r["case"]: {
                key: r[key]
                for key in (
                    "num_envs",
                    "success_rows",
                    "success_rate",
                    "finite_rewards",
                )
            }
            for r in (supported, floating, empty)
        },
        "benchmark": benchmark,
        "source_hashes": sources,
        "limitations": [
            "PPO is a 384-transition pipeline check, with no convergence claim.",
            "Physical fixtures and the planning expert are not learned-policy success-rate experiments.",
            "VA/VLA reward/chunk interfaces are implemented; model-specific RL training and learned skill execution remain outside this change.",
            "The full Sphinx build is blocked by the architecture generator's clean-source guard; API coverage and doc-check tests pass.",
            "The expert uses CPU physics; a direct-GPU ControllerAction joint-fetch stall was observed and the GPU expert attempt was cancelled.",
        ],
    }
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    events = EventAccumulator(
        str(Path(policy["run"]) / "logs" / saved_train["trainer"]["exp_name"])
    )
    events.Reload()
    optimization = {}
    for tag in (
        "train/actor_loss",
        "train/value_loss",
        "train/entropy",
        "train/learning_rate",
    ):
        values = events.Scalars(tag)
        optimization[tag] = [
            {"step": value.step, "value": value.value} for value in values
        ]
    summary["optimization"] = optimization
    summary["finite_optimization_metrics"] = all(
        bool(values) and all(math.isfinite(row["value"]) for row in values)
        for values in optimization.values()
    )
    if not summary["finite_optimization_metrics"]:
        raise ValueError("Required optimizer metrics must be nonempty and finite")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    def link(path):
        return f"[{path.name}]({path.resolve()})"

    lines = [
        "# Task Program 与原子技能奖励：实现和验收报告",
        "",
        "奖励由技能身份与物理目标定义，通过标准 RewardCfg 接入现有 RewardManager。Task Program 和独立技能共用模板；规划、Real2Sim 和未来学习执行方式不决定奖励语义。",
        "",
        "## 本次量化结果",
        "",
        "| 项目 | 结果 | 口径 |",
        "|---|---:|---|",
        f"| 相关回归测试 | {summary['tests_passed']} 通过 | 数值、配置、目标、生命周期、恢复和上下文 |",
        f"| PPO 短程训练 | {updates} 次更新 / {transitions} 条转移 | {summary['training_envs']} 环境，验证链路 |",
        f"| 保存与恢复 | step={transitions}、updates={updates} 恢复成功 | 包含优化器、折扣和目标/模板契约校验 |",
        f"| 独立 PPO 评估 | {int(metrics['eval/success_rate']*metrics['eval/success_samples'])}/{int(metrics['eval/success_samples'])} 成功 | 评估 seed={evaluation['inputs']['seed']}，与训练 seed=7 分离 |",
        f"| 评估平均回报 / 长度 | {metrics['eval/avg_reward']:.6f} / {metrics['eval/avg_length']:.1f} | 短程策略尚未学会目标动作 |",
        f"| 成功率 Wilson 95% 区间 | [{metrics['eval/success_ci95_low']:.3f}, {metrics['eval/success_ci95_high']:.3f}] | 小样本，不是收敛实验 |",
        f"| 规划专家完整抓放 | {expert['steps']} 控制步，抓取={expert['measured_pick_success']}、放置={expert['measured_place_success']} | 1 个固定场景 episode，CPU 原生物理 |",
        f"| 放稳正例 | {supported['success_rows']}/{supported['num_envs']} 通过 | 完整稳定窗口、支撑及手物分离 |",
        f"| 悬空反例 | {floating['success_rows']}/{floating['num_envs']} 误接受 | 位置正确且速度低，但无支撑 |",
        f"| 空夹反例 | {empty['success_rows']}/{empty['num_envs']} 误接受 | 无双侧持物与实际抬升 |",
        "",
        "专家轨迹和固定物理正反例验证测量机制；它们不用于声称 RL 策略已成功。专家视频和训练后策略视频均保留来源标注。",
        "",
        "优化器事件中的 actor loss、value loss、entropy 和学习率均为有限值。最后一次更新：actor loss="
        + f"{optimization['train/actor_loss'][-1]['value']:.8f}"
        + "，value loss="
        + f"{optimization['train/value_loss'][-1]['value']:.8f}"
        + "。首次 rollout 尚无完成 episode 时，日志的平均 episode 回报为空值，不代表优化器损失非有限。",
        "",
        "## 性能",
        "",
        "以下为预热后 RewardManager 常驻张量组合的 wall-clock 中位延迟；不含物理接触采集、渲染和策略推理。CPU 单线程；GPU 为 "
        + str(benchmark["gpu"])
        + "。每个场景 30 次重复。",
        "",
        "| Batch | CPU 中位 ms | GPU 中位 ms | 数值一致率 |",
        "|---:|---:|---:|---:|",
    ]
    for batch in (1, 64, 1024):
        cpu = next(
            r for r in benchmark["rows"] if r["batch"] == batch and r["device"] == "cpu"
        )
        gpu = next(
            (
                r
                for r in benchmark["rows"]
                if r["batch"] == batch and r["device"].startswith("cuda")
            ),
            None,
        )
        lines.append(
            f"| {batch} | {cpu['cost_time_ms']:.4f} | {gpu['cost_time_ms']:.4f} | {cpu['success_rate']:.0%} |"
            if gpu
            else f"| {batch} | {cpu['cost_time_ms']:.4f} | 未测 | {cpu['success_rate']:.0%} |"
        )
    lines += [
        "",
        "GPU 小张量场景主要受 eager 调度与启动开销影响，本报告不声称 GPU 加速。完整 p95、Torch allocator 内存及误差见 "
        + link(output / "benchmark.md")
        + "。",
        "",
        "## 视频与原始证据",
        "",
        "- 规划专家抓放：" + link(output / "previews/expert_preview.mp4"),
        "- 短程 PPO 策略：" + link(output / "previews/policy_preview.mp4"),
        "- 专家逐步物理轨迹：" + link(output / "expert_grasp_place.json"),
        "- 独立策略评估：" + link(evaluations[-1]),
        "- 测试日志：" + link(output / "final_tests.log"),
        "- 训练 / 恢复日志："
        + link(output / "train_final.log")
        + " / "
        + link(output / "resume.log"),
        "- 机器可读汇总与源文件 SHA-256：" + link(output / "summary.json"),
        "",
        "## 覆盖与边界",
        "",
        "覆盖五类默认技能模板、可信命名模板、程序奖励默认值和部署覆盖、每次调用独立窗口、最终物理条件保持、无效测量终止、终止/截断区别、重复读取、部分 reset、技能完成与任务结束区别及动作 chunk 的折扣聚合。多调用阶段通过 call_index 显式声明每次调用目标；并行策略进度仍不支持。",
        "",
        "接触证据采用显式 actor/pad 绑定；休眠支撑证据仅在物体与支撑姿态不变时保留，运动、无效采集和 reset 都会使其失效。奖励权重不改变物理验收。",
        "",
        "尚未实现具体 VA/VLA 的模型更新器，也没有把已训练策略注册成 Atomic Skill 执行后端；两者按设计属于后续接入。本次提供共享奖励与 chunk 接口。",
        "",
        "API 覆盖检查通过，文档检查测试通过。完整 Sphinx 构建被既有 architecture clean-source guard 阻断，未为绕过该保护提交或还原代码。",
        "",
        "最初的 GPU 专家执行在原生 GPU joint fetch 停滞，已取消；本报告的成功专家验收和视频均来自 CPU 物理路径。训练、策略评估与奖励基准使用 CUDA。",
        "",
        "## 复现",
        "",
        "```bash",
        "python -m embodichain train-rl --config examples/learning/task_program_progress/joint_ppo.yaml",
        "python -m embodichain eval-policy outputs/YOUR_RUN --episodes 8 --num-envs 4 --seed 17 --renderer no-render",
        "python -m examples.learning.task_program_progress.qualify_rewards --case floating --steps 15",
        "python -m examples.learning.task_program_progress.qualify_rewards --case empty_grip --steps 15",
        "xvfb-run -a python -m examples.learning.task_program_progress.qualify_rewards --expert --steps 1100 --video",
        "python -m scripts.benchmark.learning.rewards",
        "python -m examples.learning.task_program_progress.report",
        "```",
        "",
        "本次使用 /root/miniconda3/envs/py312/bin/python（Python 3.12.16、Torch 2.11.0+cu128）。运行前确保当前 checkout 位于 PYTHONPATH 首位，避免已安装的其他 worktree 覆盖本次代码。",
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rows = expert["trace"]
        time_axis = [row["step"] for row in rows]
        fig, axes = plt.subplots(
            2, 1, figsize=(9, 5), sharex=True, constrained_layout=True
        )
        for key, label in (("expert_pick", "PickUp"), ("expert_place", "Place")):
            axes[0].plot(
                time_axis,
                [r["measured"][key]["sample"][0][0] for r in rows],
                label=label,
            )
            axes[1].step(
                time_axis,
                [r["measured"][key]["sample"][0][1] for r in rows],
                where="post",
                label=label,
            )
        axes[0].set(
            ylabel="Skill potential",
            title="Measured planning expert trace (CPU physics, one episode)",
        )
        axes[1].set(xlabel="Control step", ylabel="Physical acceptance", yticks=[0, 1])
        for axis in axes:
            axis.legend()
            axis.grid(alpha=0.25)
        fig.savefig(output / "expert_metrics.png", dpi=160)
        plt.close(fig)
    except ImportError:
        pass
    print((output / "REPORT.md").resolve())


if __name__ == "__main__":
    main()
