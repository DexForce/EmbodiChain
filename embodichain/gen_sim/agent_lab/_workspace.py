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

"""Task-independent files that make an experiment discoverable from its cwd."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .catalog import write_json

_GUIDE = """# 实验工作区

你从这个目录独立开始，不需要其他对话提供命令或解法。

## 职责边界

Agent Lab 是启动器与实验规范，不是机器人求解框架。宿主负责环境、执行通道、
预算、录制、用量和交付；你负责阅读 EmbodiChain、选择方法、实现和验证任务。
优先使用项目已有 API 与示例，不必先搭建通用工具库。从可检查的小实验开始，
按实际图片、状态和错误迭代；控制器、规划适配和调参代码放在本轮 workspace。
需要局部兼容时在工作区或实验副本验证并记录补丁，不自动修改主库；确需共享库
修复则单独说明依据。不要把一次任务的临时方案直接升级为项目公共接口。

## 输入与环境

先读 task.json、environment.json、physics.json 和 lab.json。
task.json 是题目与验收，不包含规划路线或答案提示。
environment.json 提供项目导航、解释器和本轮目标。项目代码可以自由查阅；
从 agent_context/MAP.yaml 按需进入相关模块，不要机械扫描整个仓库。
不要读取历史会话或个人记忆。当前工作区内本轮自己产生的代码、观察、失败和
笔记可以继续使用。不要改原项目或原始资产。复用范围按 environment.json 的
reuse_policy：research 允许查阅和复用经过验证的通用适配器及资产校准结果，
无需重新开发机器人基础设施；不等于将历史任务脚本直接当成本轮成功。
isolated 禁止读取其他实验输出及同题旧答案，但项目内通用工具仍可使用。
复用时写 reuse_log.json：文件路径、SHA256、资格验证依据、适用机器人/资产指纹，
以及本轮复核结果。校准必须与当前资产、尺度和坐标系匹配；不匹配时重新测量。
该记录是可审计的复用声明，不是宿主自动认证。不要把未经验证的猜测标为校准。

如果 environment.json 中 task_variant 为 custom_instruction，本次用户指令就是
执行目标和验收依据。不要执行来源任务的旧描述、旧验收或继承其 L3/L4 难度。
结合真实场景，在控制前写 task_interpretation.json，说明任务理解与可测完成条件；
这些条件是对用户要求的展开，不是自行降低要求的授权。它们属于 Codex 提议，
不自动成为独立验收结论。run.json 的 source_task 仅用于溯源，不是额外任务。

## 运行工具

`python3 lab.py --help` 显示工具。它自动使用 environment.json 中的解释器，
无需猜测 Python 路径或 GPU 请求队列的环境变量。

- `python3 lab.py run --script experiment.py --timeout 300`：执行独立 Python
  程序，不要求 solve(lab)。GENSIM_LAB_OUTPUT 指向本次独立产物目录。
- `python3 lab.py run --script probe.py --config lab.json --timeout 300`：
  可选装配和录制预设，执行 `def solve(lab)`，直接访问真实 sim、robot 和对象。
- `python3 lab.py inspect --attempt /absolute/attempt`：解码视频、汇总实际运动。
- `python3 lab.py status`：查看宿主与预算；`python3 lab.py stop`：停止本轮并保留产物。
- `python3 lab.py usage`：读取宿主计量的累计耗时与 token 用量。

需要交互调试时，优先用持久会话，避免每次观察都重建场景或重跑前缀：
- `python3 lab.py session start --config lab.json --timeout 180`：初始化一次。
- `python3 lab.py session exec --script chunk.py --timeout 300`：执行顶层 Python，
  不需要 solve(lab)。变量 lab、state 字典及前序定义保留；result 可返回 JSON/张量。
- `python3 lab.py session observe`：读取当前状态和图片，不推进物理时间。
- `python3 lab.py session close`：关闭并刷新完整视频，然后可启动新场景。

只有显式 step/update 推进物理。普通 Python 异常保留现场并返回 traceback，
不会自动复位、松爪、重试或重新执行动作。修改已 import 的模块需显式 reload，
原生/CUDA 错误可能终止整个进程。宿主超时或退出会停止 worker 并保留已有产物。
每段脚本及同目录依赖会冻结到 attempts/episode_*/chunks/，同一会话视频连续记录。
session 仅是进程与代码生命周期，不接受动作 DSL，也不限制使用任意项目 API。

运行器在宿主执行 GPU 请求；不要在 Codex 的受限 shell 中直接启动 CUDA worker。
若没有正在运行的宿主，工具会立即说明应如何启动，不会无限等待。
你可以选 Atomic Skills、Task Program、IK、现有规划器、自写控制器或其组合。
没有动作 DSL，也没有必须复用的任务解法。Lab 只是便利工具，不是唯一执行形式。
默认 lab.json 是可替换的装配预设，不代表机器人布局天然适合当前资产。
实际坐标系、末端、关节组和控制接口应从项目代码及加载后状态确认，不猜命名。

## 实验闭环

先检查参考图、资产和实际加载画面，记录 scene_review.json：观察到什么、
任务所需对象与关系是否成立、有哪些缺陷、是否适合继续。仅导出成功不是场景合格。
如果场景不足以表达任务，记录 asset_blocked，不要靠修复未获授权的资产继续刷成功。
场景可用时，自行选方法并执行，检查图片与实际状态，再迭代。
完整任务的每个必要阶段都来自 task.json，不能只验证方便完成的子目标。

所有仿真尝试必须保留视频和实际状态，包括失败。使用 Lab 会自动记录；
独立程序需自行接入记录，并在 GENSIM_LAB_OUTPUT 保存可检查产物。
Lab 提供 step、snapshot、event、capture。既有 move_joints / move_tcp 仅保留为
可选兼容便利方法，不是推荐控制方案；后者只是单目标 IK 加关节插值，不保证避障。
FK/IK、夹爪、接触、碰撞与规划直接查阅项目对应 API；宿主不另建机器人调用层。
profile.json 分项记录物理、渲染、视频提交、PNG、状态读取的宿主墙钟时间，
不额外同步 CUDA；不能把这些等待时间直接解释成 GPU kernel 时间。
物体运动必须来自真实接触。控制目标、正常退出、视频存在都不是任务成功证明。

分段调试可用于继续检查当前现场，不必为了下一次观察重跑已经执行的前缀。
完整成功候选另写 solution.py，关闭会话后
用 run 从原始初态连续重跑并保存视频。不得拼接片段、重置物体或读取状态快照
来绕过前序操作。此重跑需预留在本轮预算内；没有重跑就明确写尚未验证。

本轮 objective=cold_start 时：自主检查环境与场景，完成一次你自行选择的
真实机器人控制实验，检查视频和实际 qpos，然后停止。可以只做有意义的
控制验证，不要求完成完整题目；禁止将入口验证声称为完整任务成功。
其他 objective 下按完整题目工作。遇到实质阻塞就记录，不重复无效运行。

结束写 handoff.json，包含 status（control_verified / asset_blocked /
incomplete / candidate）、attempts（产物路径）、evidence（观察依据）、
task_completed（是否完整完成题目）、limitations（尚未证明的内容）。
这是你的声明，仍需外部复核。保存简洁 notes.md；需要完整任务重跑时另写
candidate.json，包含 script、config、rationale。不要等待外部对话指挥下一步。

## 最终交付

在 handoff.json 增加 selected_attempt，值为本轮目录下的 attempts/<尝试ID>，
明确选出最能代表结果的一次完整尝试，不能把不同尝试拼成成功视频。
即使只完成部分目标或失败，也应选择对应录像并写清限制；尚未开始仿真时用 null。
先写 handoff.json，再停止。宿主统一生成 ../final/result.json、report.md 和
可用时的 video.mp4；不要自己编辑 final/，不要把最后一次尝试自动当作最好的一次。
未指定录像时宿主仅选择最近可解码的诊断录像，并标注未选择、未验收。
视频和报告缺失的修复由宿主 finalize 命令完成，不需要重新执行仿真。

## 耗时和 Token 用量

交接必须反映本轮累计全程耗时及 token 用量；先读取 ../usage.json。
使用宿主记录，不根据回答长度、感觉或账户额度变化猜测 token。
说明统计范围、as_of 时间和数据完整性：reported 是已有完整用量事件，
partial 是已观察下界，unavailable 是未知而非零。耗时 reconstructed 是旧记录重建。
本轮所有启动、续跑、失败及重试都应计入，不能只报最后一次成功尝试。
Codex 耗时包含工具与仿真等待，不是纯思考时间；耗时子项有重叠，不能重复相加。
缓存输入、推理输出是子项，不要在输入加输出总量上再加一次。
在 handoff.json 中用 usage_file 指向 ../usage.json，可以用中文 usage_notes
解释主要开销与限制。不要编辑 usage.json 或自行覆盖最终报告中的宿主计量。
你读到的是阶段快照，结束后宿主会补齐最终累计值；不将这些计数称为实际账单金额。
"""

_BRIDGE = '''"""Local entry point; environment selection belongs to the workspace."""
from pathlib import Path
import json
import os
import sys

workspace = Path(__file__).resolve().parent
environment = json.loads((workspace / "environment.json").read_text())
os.chdir(workspace)
os.environ["PYTHONPATH"] = os.pathsep.join(
    [environment["repo"], os.environ.get("PYTHONPATH", "")]
).rstrip(os.pathsep)
os.execv(environment["python"], [
    environment["python"], "-B", "-m", "embodichain.gen_sim.agent_lab",
    "workspace", "--run-dir", str(workspace.parent), *sys.argv[1:],
])
'''


def _write_workspace(root: Path, manifest: dict) -> None:
    workspace = root / "workspace"
    write_json(workspace / "task.json", manifest["task"])
    write_json(
        workspace / "environment.json",
        {
            "repo": manifest["repo"],
            "python": manifest["python"],
            "project_map": str(Path(manifest["repo"]) / "agent_context/MAP.yaml"),
            "objective": manifest.get("objective", "solve"),
            "reuse_policy": manifest.get("reuse_policy", "isolated"),
            "task_variant": manifest.get("task_variant", "default"),
            "acceptance_source": manifest.get("acceptance_source", "task"),
            "robot_preset": "lab.json; replaceable before an episode",
            "budget": "../launch.json records the current deadline",
            "usage_file": "../usage.json",
        },
    )
    write_json(
        workspace / "physics.json",
        {
            "mode": manifest.get("mode", "B"),
            "baseline": "GenSim scene preparation and contact dynamics",
            "preserve": [
                "asset geometry",
                "object mass/friction",
                "gravity",
                "collisions",
                "task-relevant initial states",
            ],
            "free_methods": [
                "planner selection",
                "custom Python control",
                "robot placement before episode",
                "controller tuning",
            ],
            "not_task_solutions": [
                "object teleports",
                "physical attachments",
                "overwriting actual robot state",
                "resets within final episode",
            ],
            "approved_asset_changes": manifest.get("approved_asset_changes", []),
            "further_asset_repairs": "Require explicit user approval; record blockers instead.",
        },
    )
    (workspace / "START.md").write_text(_GUIDE, encoding="utf-8")
    (workspace / "AGENTS.md").write_text(
        "# Agent Lab\n\n先阅读 START.md。当前工作区是独立实验，"
        "不要读取历史会话或个人记忆；按 environment.json 的 reuse_policy 执行复用边界。\n",
        encoding="utf-8",
    )
    (workspace / "lab.py").write_text(_BRIDGE)
    from .usage import refresh_usage

    refresh_usage(root)
    write_json(
        root / "bootstrap_hashes.json",
        {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in workspace.iterdir()
            if path.is_file()
        },
    )
