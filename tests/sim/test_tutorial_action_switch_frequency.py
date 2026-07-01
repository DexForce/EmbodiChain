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

import ast
from pathlib import Path

EXPECTED_ACTION_SWITCH_INTERVAL = 100
SCRIPTS_ROOT = Path(__file__).resolve().parents[2] / "scripts" / "tutorials" / "sim"


def _parse_module(script_name: str) -> ast.Module:
    script_path = SCRIPTS_ROOT / script_name
    return ast.parse(script_path.read_text(encoding="utf-8"))


def _find_constant_value(module: ast.Module, constant_name: str) -> int | None:
    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == constant_name:
                return ast.literal_eval(node.value)
    return None


def test_create_robot_tutorial_uses_shared_action_switch_interval() -> None:
    module = _parse_module("create_robot.py")

    assert (
        _find_constant_value(module, "ACTION_SWITCH_INTERVAL")
        == EXPECTED_ACTION_SWITCH_INTERVAL
    )


def test_create_sensor_tutorial_uses_shared_action_switch_interval() -> None:
    module = _parse_module("create_sensor.py")

    assert (
        _find_constant_value(module, "ACTION_SWITCH_INTERVAL")
        == EXPECTED_ACTION_SWITCH_INTERVAL
    )
