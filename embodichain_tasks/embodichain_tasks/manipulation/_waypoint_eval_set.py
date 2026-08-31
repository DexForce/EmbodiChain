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
from __future__ import annotations

from pathlib import Path

import torch

from ._waypoint_types import WAYPOINT_TASK_FIELDS


def load_eval_set(path: str | Path) -> dict:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    tasks = payload.get("tasks", payload)
    missing = [field for field in WAYPOINT_TASK_FIELDS if field not in tasks]
    if missing:
        raise ValueError(f"Eval set is missing required fields: {missing}")
    return payload


def save_eval_set(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
