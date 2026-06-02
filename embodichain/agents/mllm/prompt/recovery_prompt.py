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
