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
from typing import Any

from embodichain.agents.hierarchy.agent_base import AgentBase
from embodichain.agents.mllm.prompt import TaskPrompt
from embodichain.data import database_agent_prompt_dir
from embodichain.utils.llm_json import normalize_json_content
from embodichain.utils.utility import load_txt

__all__ = ["TaskAgent"]


class TaskAgent(AgentBase):
    """Generate the nominal atomic-action task graph."""

    prompt_name: str
    prompt_kwargs: dict[str, dict[str, Any]]

    def __init__(self, llm, **kwargs) -> None:
        super().__init__(**kwargs)
        if llm is None:
            raise ValueError(
                "LLM is None. Please set the following environment variables:\n"
                "  - OPENAI_API_KEY\n"
                "  - LLM_URL"
            )
        self.llm = llm

    def generate(self, **kwargs) -> str:
        log_dir = kwargs.get(
            "log_dir", Path(database_agent_prompt_dir) / self.task_name
        )
        file_path = Path(log_dir) / "agent_task_graph.json"

        if not kwargs.get("regenerate", False) and file_path.exists():
            print(f"Task graph already exists at {file_path}.")
            return load_txt(file_path)

        prompt = getattr(TaskPrompt, self.prompt_name)(**kwargs)
        response = self.llm.invoke(prompt)
        print(f"\033[92m\nTask agent output:\n{response.content}\n\033[0m")

        content = normalize_json_content(response.content)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        print(f"Generated task graph saved to {file_path}")

        return content

    def act(self, *args, **kwargs):
        return super().act(*args, **kwargs)
