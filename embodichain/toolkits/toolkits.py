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
from embodichain.utils.utility import load_json


class ToolkitsBase(metaclass=ABCMeta):
    @classmethod
    def from_config(cls, path: str):
        assert (
            os.path.basename(path).split(".")[-1] == "json"
        ), "only json file is supported."
        config = load_json(path)
        return config["ToolKits"][cls.__name__]

    @abstractmethod
    def call(self, **kwargs):
        pass
