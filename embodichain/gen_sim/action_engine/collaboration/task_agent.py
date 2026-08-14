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

"""Deprecated import bridge for the standalone Task Engine."""

from __future__ import annotations

from embodichain.gen_sim.collaboration.coordinator import lower_task_candidate
from embodichain.gen_sim.task_engine.agent import *  # noqa: F401,F403

__all__ = [
    "TaskAgent",
    "TaskGenerationError",
    "derive_scene_request",
    "derive_success_spec",
    "lower_task_candidate",
]
