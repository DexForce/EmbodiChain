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

"""Load packaged locomotion task data."""

from __future__ import annotations

import json
from importlib.resources import files

__all__ = ["load_task_data"]


def load_task_data(task_name: str) -> dict:
    """Load one velocity task definition from the configuration package."""
    resource = files("embodichain_tasks.configs").joinpath(
        "tasks", "locomotion", "velocity", task_name, "task.json"
    )
    return json.loads(resource.read_text(encoding="utf-8"))
