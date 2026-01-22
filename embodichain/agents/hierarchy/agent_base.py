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

from abc import ABCMeta
import os
from pathlib import Path
from embodichain.utils.utility import load_txt
import embodichain.agents.mllm.prompt as mllm_prompt
from embodichain.data import database_2d_dir


def _resolve_prompt_path(file_name: str, config_dir: str = None) -> str:
    # If absolute path, use directly
    if os.path.isabs(file_name):
        if os.path.exists(file_name):
            return file_name
        raise FileNotFoundError(f"Prompt file not found: {file_name}")
    
    # Try config directory first (for task-specific prompts)
    if config_dir:
        config_path = os.path.join(config_dir, file_name)
        if os.path.exists(config_path):
            return config_path
    
    # Try agents/prompts directory (for reusable prompts)
    agents_prompts_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prompts"
    )
    agents_path = os.path.join(agents_prompts_dir, file_name)
    if os.path.exists(agents_path):
        return agents_path
    
    # If still not found, raise error with search paths
    searched_paths = []
    if config_dir:
        searched_paths.append(f"  - {config_dir}/{file_name}")
    searched_paths.append(f"  - {agents_prompts_dir}/{file_name}")
    
    raise FileNotFoundError(
        f"Prompt file not found: {file_name}\n"
        f"Searched in:\n" + "\n".join(searched_paths)
    )


class AgentBase(metaclass=ABCMeta):
    def __init__(self, **kwargs) -> None:

        assert (
            "prompt_kwargs" in kwargs.keys()
        ), "Key prompt_kwargs must exist in config."

        for key, value in kwargs.items():
            setattr(self, key, value)

        # Get config directory if provided
        config_dir = kwargs.get("config_dir", None)
        if config_dir:
            config_dir = os.path.dirname(os.path.abspath(config_dir))

        # Preload and store prompt contents inside self.prompt_kwargs
        for key, val in self.prompt_kwargs.items():
            if val["type"] == "text":
                file_path = _resolve_prompt_path(val["name"], config_dir)
                val["content"] = load_txt(file_path)  # ← store content here
            else:
                raise ValueError(
                    f"Now only support `text` type but {val['type']} is given."
                )

    def generate(self, *args, **kwargs):
        pass

    def act(self, *args, **kwargs):
        pass

    def get_composed_observations(self, **kwargs):
        ret = {"observations": kwargs.get("env").get_obs_for_agent()}
        for key, val in self.prompt_kwargs.items():
            ret[key] = val["content"]
        ret.update(kwargs)
        return ret
