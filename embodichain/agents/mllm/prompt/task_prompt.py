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

import torch
from langchain_core.messages import SystemMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from embodichain.utils.utility import encode_image


class TaskPrompt:
    @staticmethod
    def generate_task_plan(observations, **kwargs):
        """
        Hybrid one-pass prompt:
        Step 1: VLM analyzes the image and extracts object IDs.
        Step 2: LLM generates task instructions using only those IDs.
        """
        # Encode image
        observation = (
            observations["rgb"].cpu().numpy()
            if isinstance(observations["rgb"], torch.Tensor)
            else observations["rgb"]
        )
        kwargs.update({"observation": encode_image(observation)})

        # Build hybrid prompt
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content=(
                        "You are a precise and reliable robotic manipulation planner. "
                        "Given a camera observation and a task description, you must generate "
                        "a clear, step-by-step task plan for a robotic arm. "
                        "All actions must strictly use the provided atomic API functions, "
                        "and the plan must be executable without ambiguity."
                        # "After generating the plan, generate the corresponding validation conditions for each step, "
                        # "which can be directly verified from the image observation."
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
                            "text":
                                "Here is the current camera observation.\n"
                                "First, analyze the scene in the image.\n"
                                "Then, given the scene, use the context below to generate an actionable task plan that achieves the task goal:\n\n"
                                "**Environment background:** \n{basic_background}\n\n"
                                '**Task goal:** \n"{task_prompt}"\n\n'
                                "**Available atomic actions:** \n{atom_actions}\n"
                                "**REQUIRED OUTPUT**\n"
                                "[PLANS]:\n"
                                "Step 1: <intent> — <left atomic_action>(...) <right atomic_action>(...)\n"
                                "..."
                                "Step N: <intent> — <left atomic_action>(...) <right atomic_action>(...)\n\n"

                                # "[VALIDATION_CONDITIONS]:\n"
                                # "Step 1: <explicit, image-verifiable post-condition>\n"
                                # "..."
                                # "Step N: <explicit, image-verifiable post-condition>\n\n"
                                #
                                # "Note that the atomic action specified at each step may be None if no action is taken.\n"
                                # "Note that the VALIDATION_CONDITIONS MUST explicitly describe:\n"
                                # "(1) the state of each robot arm and gripper, "
                                # "(2) the state of each relevant object, and "
                                # "(3) the arm–object relationship. Specifically, whether each object should be actively grasped by the gripper or released to be stably supported by the environment."
                            ,
                        },
                    ]
                ),
            ]
        )

        # Return the prompt template and kwargs to be executed by the caller
        return prompt.invoke(kwargs)