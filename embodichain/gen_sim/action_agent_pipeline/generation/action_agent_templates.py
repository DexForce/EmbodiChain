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

from copy import deepcopy
from functools import lru_cache
import json
from pathlib import Path
from typing import Any

__all__ = [
    "make_dual_ur5_robot_config",
    "make_light_config",
    "make_sensor_config",
]

_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"


def make_dual_ur5_robot_config(*, robot_init_z: float) -> dict[str, Any]:
    """Return a fresh DualUR5 robot config template at the requested z position."""
    config = _load_template("dual_ur5_robot.json")
    config["init_pos"][2] = float(robot_init_z)
    return config


def make_sensor_config() -> list[dict[str, Any]]:
    """Return a fresh default sensor config template."""
    return _load_template("default_sensors.json")


def make_light_config() -> dict[str, Any]:
    """Return a fresh default light config template."""
    return _load_template("default_lights.json")


def _load_template(name: str) -> Any:
    return deepcopy(_read_template(name))


@lru_cache(maxsize=None)
def _read_template(name: str) -> Any:
    path = _TEMPLATE_DIR / name
    return json.loads(path.read_text(encoding="utf-8"))
