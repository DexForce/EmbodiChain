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

"""Environment framework: ``BaseEnv`` / ``EmbodiedEnv`` class hierarchy, task registration, manager wiring, and the step/reset lifecycle."""

from __future__ import annotations

from .base_env import *
from .demo import *
from .embodied_env import *
from .settling import *
from .wrapper import *

# Official task environments live in the bundled ``embodichain_tasks`` package
# and register through the same ``embodichain.tasks`` entry-point mechanism as
# third-party task packages.
