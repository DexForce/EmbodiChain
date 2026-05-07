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

from pathlib import Path
from typing import Dict, List

import numpy as np
from langchain_core.prompts import ChatPromptTemplate

from embodichain.agents.hierarchy.agent_base import AgentBase
from embodichain.agents.mllm.prompt import FailureAnticipationPrompt
from embodichain.data import database_agent_prompt_dir
from embodichain.utils.utility import load_txt

import re
from typing import List

class FailureAnticipationAgent(AgentBase):
    prompt: ChatPromptTemplate
    object_list: List[str]
    target: np.ndarray
    prompt_name: str
    prompt_kwargs: Dict[str, Dict]

    def __init__(self, llm, **kwargs) -> None:
        super().__init__(**kwargs)
        if llm is None:
            raise ValueError(
                "LLM is None. Please set the following environment variables:\n"
                "  - AZURE_OPENAI_ENDPOINT\n"
                "  - AZURE_OPENAI_API_KEY\n"
                "Example:\n"
                "  export AZURE_OPENAI_ENDPOINT='https://your-endpoint.openai.azure.com/'\n"
                "  export AZURE_OPENAI_API_KEY='your-api-key'"
            )
        self.llm = llm

    def generate(self, **kwargs) -> str:
        log_dir = kwargs.get(
            "log_dir", Path(database_agent_prompt_dir) / self.task_name
        )
        file_path = log_dir / "agent_anticipated_failures.txt"

        if not kwargs.get("regenerate", False) and file_path.exists():
            print(
                f"Anticipated failures file already exists at {file_path}, skipping writing."
            )
            return load_txt(file_path)

        prompts_ = getattr(FailureAnticipationPrompt, self.prompt_name)(**kwargs)
        response = self.llm.invoke(prompts_)
        print(
            f"\033[91m\nFailure anticipation agent output:\n{response.content}\n\033[0m"
        )

        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w") as f:
            f.write(response.content)
        print(f"Generated anticipated failures saved to {file_path}")

        return response.content