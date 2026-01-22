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

from abc import ABCMeta, abstractmethod
import os
import cv2
from embodichain.utils.utility import load_json, load_txt
from embodichain.agents.mllm.prompt import *
from embodichain.data import database_agent_prompt_dir, database_2d_dir
from embodichain.utils.utility import encode_image


class AgentBase(metaclass=ABCMeta):
    def __init__(self, **kwargs) -> None:

        assert (
            "prompt_kwargs" in kwargs.keys()
        ), "Key prompt_kwargs must exist in config."

        for key, value in kwargs.items():
            setattr(self, key, value)

        # Preload and store prompt contents inside self.prompt_kwargs
        for key, val in self.prompt_kwargs.items():
            if val["type"] == "text":
                file_path = os.path.join(database_agent_prompt_dir, val["name"])
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
