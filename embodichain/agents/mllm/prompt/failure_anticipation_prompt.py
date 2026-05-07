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

from __future__ import annotations

from langchain_core.messages import SystemMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)


class FailureAnticipationPrompt:
    @staticmethod
    def anticipate_potential_failure(**kwargs):
        failure_anticipation_rules = kwargs.get("failure_anticipation_rules", "")
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content=(
                        "You are a robotic manipulation failure-and-recovery data designer. "
                        "Your goal is not to list every theoretical failure. Your goal is to propose only high-value failure cases that should be intentionally injected during execution so the system can collect useful error-recovery demonstrations. "
                        "Given an existing step-by-step task plan, the available runtime error functions, the available monitor functions, and the available atomic actions, you must decide which steps are worth injecting recoverable failures into, and you must provide simple runtime policies for each selected step. "
                        "Every anticipated failure must be physically grounded, common or realistically likely, concise, and tightly tied to the action context of that step. "
                        "The runtime semantics are simple: errors are disturbance sources, while monitor sequences and recovery sequences are paired policies. "
                        "A step may contain multiple errors and multiple monitor-and-recovery policies. "
                        "Different errors may share a policy, and a step may also use multiple distinct policies when one policy is not enough. "
                        "Every recovery must stay inside the current system, use only the provided atomic actions, and restore the task to a state from which execution can plausibly continue. "
                        "Only include failures that satisfy all of the following: they can be caused by one of the listed runtime error functions, they can be detected by one or more of the listed monitor functions, they are common enough to be worth generating as data, and they are recoverable with a short feasible action sequence. "
                        "Do not include catastrophic, irreversible, low-value, redundant, or clearly unrecoverable failures. "
                        "You MUST consider both arms' state and role when relevant. For example, if one arm is moving while the other arm is holding an object, failures involving either arm may matter. "
                        "The available runtime error functions and available monitor functions are hard constraints: only anticipate failures that are directly supported by them.\n"
                    )
                ),
                HumanMessagePromptTemplate.from_template(
                    [
                        {
                            "type": "text",
                            "text": (
                                "Use the context below to design intentional, recoverable failure cases for data generation.\n\n"
                                "**Environment background:** \n{basic_background}\n\n"
                                '**Task goal:** \n"{task_prompt}"\n\n'
                                "**Available atomic actions:** \n{atom_actions}\n"
                                "**Available runtime error functions:** \n{error_functions}\n\n"
                                "**Available monitor functions:** \n{monitor_functions}\n\n"
                                "**Task plan:**\n{task_plan}\n\n"
                                "**REQUIRED OUTPUT**\n"
                                "[ANTICIPATED_FAILURES]:\n"
                                "Step 1:\n"
                                "Errors:\n"
                                "E1.1 [error_type=<exact runtime error type>] <possible injected error for Step 1>\n"
                                "Policies:\n"
                                "M1.1: <one or more exact monitor calls for Step 1>\n"
                                "R1.1: <ordered recovery atomic actions for Step 1>\n"
                                "M1.2: <additional monitor calls for Step 1 if needed>\n"
                                "R1.2: <ordered recovery atomic actions for Step 1>\n"
                                "...\n"
                                "Step 2:\n"
                                "None\n"
                                "...\n"
                                "Step N:\n"
                                "Errors:\n"
                                "EN.1 [error_type=<exact runtime error type>] <possible injected error for Step N>\n"
                                "Policies:\n"
                                "MN.1: <one or more exact monitor calls for Step N>\n"
                                "RN.1: <ordered recovery atomic actions for Step N>\n\n"
                                "Detailed rules:\n"
                                f"{failure_anticipation_rules}"
                            ),
                        },
                    ]
                ),
            ]
        )

        return prompt.invoke(kwargs)
