# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
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

from langchain_core.messages import SystemMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from embodichain.utils.utility import encode_image, encode_image_from_path


class CodePrompt:
    @staticmethod
    def one_stage_prompt(**kwargs) -> ChatPromptTemplate:
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content="You are an AI assistant that can generate python code to execute robot arms."
                ),
                HumanMessagePromptTemplate.from_template(
                    [
                        {
                            "type": "text",
                            "text": (
                                "Generate a Python code snippet that accomplishes the following task:\n"
                                "{query}\n\n"
                                "You must strictly follow the rules and available functions described below:\n"
                                "{code_prompt}\n\n"
                                "Here are some reference examples of the expected output code:\n"
                                "{code_example}\n\n"
                            ),
                        }
                    ]
                ),
            ]
        )
        return prompt.invoke(kwargs)

    @staticmethod
    def unified_prompt(observations, **kwargs):
        """
        Unified Vision→Code prompt:
        - Model observes the image
        - Understands the scene and the task goal
        - Generates final executable Python code using atomic robot APIs
        """

        # Encode the image
        observation = observations["rgb"]
        kwargs.update({"observation": encode_image(observation)})

        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content=(
                        "You are a reliable Vision-Language-Code robot assistant. "
                        "You observe an image, understand the scene and the task goal, "
                        "and generate correct Python code using ONLY the allowed atomic robot actions. "
                        "Your final output must be a single Python code block."
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
                                "### Task Goal\n"
                                "{task_prompt}\n\n"
                                "### Environment Background\n"
                                "{basic_background}\n\n"
                                "### Allowed Atomic Actions\n"
                                "{atom_actions}\n\n"
                                "### Code Rules\n"
                                "{code_prompt}\n\n"
                                "### Reference Code Examples\n"
                                "{code_example}\n\n"
                                "### Final Instructions\n"
                                "Understand the scene from the image and generate final executable Python code "
                                "that performs the task using ONLY the allowed atomic actions.\n\n"
                                "Your entire response must be EXACTLY one Python code block:\n"
                                "```python\n"
                                "# your solution code here\n"
                                "```\n"
                            ),
                        },
                    ]
                ),
            ]
        )

        return prompt.invoke(kwargs)

    @staticmethod
    def one_stage_prompt_according_to_task_plan(**kwargs) -> ChatPromptTemplate:
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content=(
                        "You are a reliable robot control code generator.\n"
                        "Your task is to generate Python code that executes robot arm actions.\n\n"
                        "CRITICAL RULES:\n"
                        "- The TASK PLAN defines the available atomic actions, rules, and execution logic.\n"
                        "- You MUST strictly follow the TASK PLAN.\n"
                        "- The CONSTRAINTS section contains additional global constraints you must obey.\n"
                        "- Do NOT invent new actions, functions, parameters, or control flow.\n"
                        "- You MAY include Python comments (# ...) inside the code.\n"
                        "- Your ENTIRE response MUST be a single Python code block.\n"
                        "- The code block MUST be directly executable without modification.\n"
                        "- Do NOT include any text, explanation, or markdown outside the Python code block.\n"
                    )
                ),
                HumanMessagePromptTemplate.from_template(
                    [
                        {
                            "type": "text",
                            "text": (
                                "TASK PLAN (atomic actions, rules, and intended behavior):\n"
                                "{task_plan}\n\n"
                                "GLOBAL CONSTRAINTS (must be satisfied):\n"
                                "{code_prompt}\n\n"
                                "REFERENCE CODE (style and structure only; do NOT copy logic):\n"
                                "{code_example}\n\n"
                                "Generate the corrected Python code now."
                            ),
                        }
                    ]
                ),
            ]
        )
        return prompt.invoke(kwargs)
