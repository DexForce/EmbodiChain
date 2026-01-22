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

import torch
import numpy as np
from langchain_core.messages import SystemMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from embodichain.utils.utility import encode_image, encode_image_from_path


class TaskPrompt:
    @staticmethod
    def one_stage_prompt(observations, **kwargs):
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
                                "Here is the latest camera observation.\n"
                                "First, analyze the scene in the image.\n"
                                "Then, using the context below, produce an actionable task plan.\n\n"
                                "**Environment background:** \n{basic_background}\n\n"
                                '**Task goal:** \n"{task_prompt}"\n\n'
                                "**Available atomic actions:** \n{atom_actions}\n"
                            ),
                        },
                    ]
                ),
            ]
        )

        # Return the prompt template and kwargs to be executed by the caller
        return prompt.invoke(kwargs)

    @staticmethod
    def two_stage_prompt(observations, **kwargs):
        # for VLM generate image descriptions
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content="You are a helpful assistant to operate a robotic arm with a camera to generate task plans according to descriptions."
                ),
                HumanMessagePromptTemplate.from_template(
                    [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/jpg;base64,{observation}",
                            },
                        },
                        {
                            "type": "text",
                            "text": "What is in the image? Return answer with their potential effects.",
                        },
                    ]
                ),
            ]
        )

        observation = (
            observations["rgb"].cpu().numpy()
            if isinstance(observations["rgb"], torch.Tensor)
            else observations["rgb"]
        )
        kwargs.update({"observation": encode_image(observation)})
        # for LLM generate task descriptions
        prompt_query = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content="You are a helpful assistant to operate a robotic arm with a camera to generate task plans according to descriptions."
                ),
                HumanMessagePromptTemplate.from_template(
                    [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/jpg;base64,{observation}",
                            },
                        },
                        {
                            "type": "text",
                            "text": "Here is analysis for this image: {query}.",
                        },
                        {
                            "type": "text",
                            "text": (
                                "Using the context below, produce an actionable task plan.\n\n"
                                "**Environment background:** \n{basic_background}\n\n"
                                '**Task goal:** \n"{task_prompt}"\n\n'
                                "**Available atomic actions:** \n{atom_actions}\n"
                            ),
                        },
                    ]
                ),
            ]
        )

        return [prompt.invoke(kwargs), {"prompt": prompt_query, "kwargs": kwargs}]
