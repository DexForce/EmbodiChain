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

## 输入与环境

先读 task.json、environment.json、physics.json 和 lab.json。
task.json 是题目与验收，不包含规划路线或答案提示。
environment.json 提供项目导航、解释器和本轮目标。项目代码可以自由查阅；
从 agent_context/MAP.yaml 按需进入相关模块，不要机械扫描整个仓库。
不要读取其他实验输出、历史会话、个人记忆或同题旧解法。当前工作区内
本轮自己产生的代码、观察、失败和笔记可以继续使用。不要改原项目或原始资产。

## 运行工具

`python3 lab.py --help` 显示工具。它自动使用 environment.json 中的解释器，
无需猜测 Python 路径或 GPU 请求队列的环境变量。

- `python3 lab.py run --script probe.py --config lab.json --timeout 300`：
  执行普通 `def solve(lab)`。可直接访问 lab.sim、robot、objects、articulations。
- `python3 lab.py run --script experiment.py --timeout 300`：执行独立 Python
  程序，不要求 solve(lab)。GENSIM_LAB_OUTPUT 指向本次独立产物目录。
- `python3 lab.py inspect --attempt /absolute/attempt`：解码视频、汇总实际运动。
- `python3 lab.py status`：查看宿主与预算；`python3 lab.py stop`：停止本轮并保留产物。
- `python3 lab.py usage`：读取宿主计量的累计耗时与 token 用量。

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
便利方法包括 step、move_joints、move_tcp、snapshot、event、capture。
move_tcp 是单目标 IK 加关节插值，不保证笛卡尔直线或自动避障。
物体运动必须来自真实接触。控制目标、正常退出、视频存在都不是任务成功证明。

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
        "不要读取历史会话、个人记忆、其他实验输出或同题旧解法。\n",
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
