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

"""Load the task-owned declarative parameter and geometry defaults."""

from __future__ import annotations

from copy import deepcopy
from functools import cache
from typing import Any

from embodichain.utils.utility import load_config
from embodichain_tasks.configs import get_config_path

__all__: list[str] = []


@cache
def _declarations() -> dict[str, Any]:
    return load_config(get_config_path("tasks/manipulation/assembly/defaults.yaml"))


def assembly_defaults(section: str) -> dict[str, Any]:
    """Return owned defaults so per-environment overrides cannot mutate the source."""
    return deepcopy(_declarations()[section])
