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

"""Environment utilities: the registration system (``register_env``, ``make``), Gymnasium integration helpers, miscellaneous utilities, and ``EnvProfiler`` for step/reset timing."""

from __future__ import annotations

from embodichain.lab.gym.utils.profiler import EnvProfiler, EnvProfilerCfg
from embodichain.lab.gym.utils.trajectory_state import (
    capture_trajectory_state,
    restore_trajectory_state,
)

__all__ = [
    "EnvProfiler",
    "EnvProfilerCfg",
    "capture_trajectory_state",
    "restore_trajectory_state",
]
