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

from .base_env import *
from .embodied_env import *
from .wrapper import *

# Task environments have moved to the separate ``embodichain_tasks`` package
# (and any third-party package declaring an ``embodichain.tasks`` entry point).
# They are no longer re-exported here so that importing the core envs package
# stays warning-free. Direct imports from ``embodichain.lab.gym.envs.tasks``
# still work via the deprecation shim in ``tasks/__init__.py``.
