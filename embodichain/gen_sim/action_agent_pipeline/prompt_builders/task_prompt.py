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

from typing import Any

import torch
from langchain_core.messages import SystemMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from embodichain.utils.utility import encode_image

__all__ = ["TaskPrompt"]


class TaskPrompt:
    @staticmethod
    def generate_task_graph(observations: dict[str, Any], **kwargs: Any) -> Any:
        """Build a prompt that asks the task agent for a nominal JSON graph."""
        schema = """{
  "task": "<short task name>",
  "start": "v0_start",
  "goal": "vN_done",
  "nodes": [
    {"id": "v0_start", "semantic": "<initial state>"},
    {"id": "v1_<state>", "semantic": "<state after edge 1>"}
  ],
  "edges": [
    {
      "id": "e01_<action>",
      "source": "v0_start",
      "target": "v1_<state>",
      "left_arm_action": {"fn": "<atomic_action>", "kwargs": {}},
      "right_arm_action": null
    }
  ]
}"""

        observation = (
            observations["rgb"].cpu().numpy()
            if isinstance(observations["rgb"], torch.Tensor)
            else observations["rgb"]
        )
        kwargs.update(
            {
                "graph_schema": schema,
                "observation": encode_image(observation),
            }
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content=(
                        "You are a precise robotic manipulation graph planner. "
                        "Given a camera observation and task description, produce only "
                        "the nominal atomic-action graph. Do not add failure monitors, "
                        "error injection, recovery branches, Python code, or prose. "
                        "All actions must strictly use the provided atomic API functions."
                    )
                ),
                HumanMessagePromptTemplate.from_template(
                    [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64,{observation}",
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                "Use the current camera observation and context below to "
                                "generate a nominal atomic-action graph for the task.\n\n"
                                "**Environment background:**\n{basic_background}\n\n"
                                '**Task goal:**\n"{task_prompt}"\n\n'
                                "**Available atomic actions:**\n{atom_actions}\n\n"
                                "**Required JSON schema:**\n"
                                "{graph_schema}\n\n"
                                "Rules:\n"
                                "- Output exactly one JSON object and nothing else.\n"
                                "- The nominal graph must be one deterministic start-to-goal chain with no branches, cycles, or orphan edges.\n"
                                "- Each edge is one semantic task step from source node to target node.\n"
                                "- Every edge must define at least one non-null arm action.\n"
                                "- Use `null` for an idle arm action.\n"
                                "- Put only JSON primitives inside kwargs: strings, numbers, booleans, null, arrays, or objects.\n"
                                "- Do not include `env`, tensors, comments, validation conditions, monitors, errors, or recovery fields.\n"
                                "- Preserve task order and use both arms on the same edge when they should act simultaneously.\n"
                                "- Use stable ids such as `v0_start`, `v1_grasped`, `e01_grasp_objects`.\n"
                                "- Replace `N` with the concrete final step index; do not literally output `vN_done`.\n"
                                "- The final edge target must equal the `goal` field."
                            ),
                        },
                    ]
                ),
            ]
        )
        return prompt.invoke(kwargs)
