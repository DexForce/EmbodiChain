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

__all__ = ["RecoveryPrompt"]


class RecoveryPrompt:
    @staticmethod
    def augment_task_graph(**kwargs):
        schema = """{
  "task": "<same task name>",
  "recovery_bindings": [
    {
      "edge_id": "<nominal edge id>",
      "failure_name": "<short failure label>",
      "monitors": [
        {"type": "object_moved", "objects": ["<object name>"], "threshold": 0.02}
      ],
      "recovery": [
        {
          "type": "regrasp",
          "robot_name": "<left_arm or right_arm>",
          "obj_name": "<object name>",
          "pre_grasp_dis": 0.1
        },
        {
          "type": "move_to_safe_pose",
          "robot_name": "<same arm as regrasp>"
        }
      ],
      "merge": "target",
      "repeat_until_success": true
    }
  ]
}"""
        empty_schema = '{"task": "<same task name>", "recovery_bindings": []}'
        kwargs.update(
            {
                "empty_recovery_schema": empty_schema,
                "recovery_schema": schema,
            }
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content=(
                        "You are a robotic manipulation recovery policy designer. "
                        "Given a nominal atomic-action graph, add only compact "
                        "monitor-to-recovery bindings. Do not write graph nodes, "
                        "graph edges, graph branches, Python code, or prose. "
                        "Output only JSON."
                    )
                ),
                HumanMessagePromptTemplate.from_template(
                    [
                        {
                            "type": "text",
                            "text": (
                                "Use the context below to generate a lightweight recovery "
                                "spec for the nominal graph.\n\n"
                                "**Environment background:**\n{basic_background}\n\n"
                                '**Task goal:**\n"{task_prompt}"\n\n'
                                "**Available atomic actions:**\n{atom_actions}\n\n"
                                "**Possible external failure descriptions "
                                "(prompt context only, not executable output):**\n"
                                "{error_functions}\n\n"
                                "**Available monitor functions:**\n{monitor_functions}\n\n"
                                "**Nominal task graph JSON:**\n{task_graph}\n\n"
                                "**Required JSON schema:**\n"
                                "{recovery_schema}\n\n"
                                "Output exactly one JSON object following the schema. "
                                "If no edge needs a recovery branch, output "
                                "`{empty_recovery_schema}`.\n\n"
                                "Follow these recovery rules:\n"
                                "{recovery_rules}"
                            ),
                        },
                    ]
                ),
            ]
        )

        return prompt.invoke(kwargs)

    @staticmethod
    def runtime_recovery_binding(**kwargs):
        schema = """{
  "task": "<same task name>",
  "recovery_bindings": [
    {
      "edge_id": "<current edge id>",
      "failure_name": "<short failure label>",
      "monitors": [
        {"type": "object_fallen", "objects": ["<object name>"], "upright_threshold": 0.65}
      ],
      "recovery": [
        {
          "type": "move_to_safe_pose",
          "arms": ["<left_arm or right_arm>"]
        },
        {
          "type": "regrasp",
          "robot_name": "<left_arm or right_arm>",
          "obj_name": "<object name>",
          "pre_grasp_dis": 0.1
        },
        {
          "type": "retry_failed_edge"
        }
      ],
      "merge": "source",
      "repeat_until_success": true
    }
  ]
}"""
        empty_schema = '{"task": "<same task name>", "recovery_bindings": []}'
        kwargs.update(
            {
                "empty_recovery_schema": empty_schema,
                "recovery_schema": schema,
            }
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content=(
                        "You are a runtime robotic manipulation recovery planner. "
                        "Given the edge that just failed and the current simulator "
                        "state summary, output one compact recovery binding for the "
                        "current edge only. Output only JSON."
                    )
                ),
                HumanMessagePromptTemplate.from_template(
                    [
                        {
                            "type": "text",
                            "text": (
                                "**Environment background:**\n{basic_background}\n\n"
                                '**Task goal:**\n"{task_prompt}"\n\n'
                                "**Available atomic actions:**\n{atom_actions}\n\n"
                                "**Available monitor functions:**\n{monitor_functions}\n\n"
                                "**Recovery rules:**\n{recovery_rules}\n\n"
                                "**Current failed edge:**\n{current_edge}\n\n"
                                "**Triggered monitor:**\n{triggered_monitor}\n\n"
                                "**Runtime state summary:**\n{runtime_state}\n\n"
                                "**Nominal task graph JSON:**\n{task_graph}\n\n"
                                "**Required JSON schema:**\n{recovery_schema}\n\n"
                                "Allowed runtime recovery step types are only "
                                "`move_to_safe_pose`, `regrasp`, "
                                "`retry_failed_edge`, and `replay_edge`. Do not "
                                "output direct `action` steps.\n\n"
                                "Output exactly one JSON object. If the failure is "
                                "not recoverable using supported steps, output "
                                "`{empty_recovery_schema}`."
                            ),
                        },
                    ]
                ),
            ]
        )

        return prompt.invoke(kwargs)
