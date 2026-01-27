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

import os

from .constants import EMBODICHAIN_DEFAULT_DATABASE_ROOT

database_dir = EMBODICHAIN_DEFAULT_DATABASE_ROOT
database_2d_dir = os.path.join(database_dir, "2dasset")
database_agent_prompt_dir = os.path.join(database_dir, "agent_prompt")
database_demo_dir = os.path.join(database_dir, "demostration")

from . import assets
from .dataset import *
