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

import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

pytestmark = pytest.mark.no_sim

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_TUTORIAL_PATH = (
    _REPOSITORY_ROOT / "scripts/tutorials/sim/ik_manipulability_selection.py"
)


def _load_tutorial_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "ik_manipulability_selection_tutorial", _TUTORIAL_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prepare_variant_robots_registers_every_robot_before_prepare(
    monkeypatch,
) -> None:
    tutorial = _load_tutorial_module()
    events: list[str] = []

    class FakeSimulation:
        def add_robot(self, *, cfg):
            events.append(f"add:{cfg.uid}")
            return SimpleNamespace(uid=cfg.uid)

        def prepare(self) -> None:
            events.append("prepare")

    monkeypatch.setattr(
        tutorial,
        "_robot_cfg",
        lambda uid, _selection, _seed_selection: SimpleNamespace(uid=uid),
    )

    variants = tutorial._prepare_variant_robots(FakeSimulation())

    expected_uids = [
        f"w1_{selection}_{'sel' if seed_selection else 'rand'}"
        for _label, selection, seed_selection in tutorial.VARIANTS
    ]
    assert events == [*(f"add:{uid}" for uid in expected_uids), "prepare"]
    assert [robot.uid for _label, robot in variants] == expected_uids
