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

"""Configuration owned by Task Engine orchestration."""

from __future__ import annotations

from embodichain.utils import configclass

__all__ = ["TaskEngineWorkflowCfg"]


@configclass
class TaskEngineWorkflowCfg:
    """Conservative first-version orchestration limits.

    Retry defaults intentionally remain one until remote-service and runtime
    measurements establish safe higher values. The orchestration layer owns
    these limits even though retries are implemented in a later phase.
    """

    max_parallel_workers: int = 2
    max_scene_attempts: int = 1
    max_action_attempts: int = 1

    def __post_init__(self) -> None:
        for field_name in (
            "max_parallel_workers",
            "max_scene_attempts",
            "max_action_attempts",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field_name} must be a positive integer.")
