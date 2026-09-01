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
_TUTORIAL_PATH = _REPOSITORY_ROOT / "scripts/tutorials/sim/open_drawer.py"


def _load_tutorial_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "open_drawer_tutorial", _TUTORIAL_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_create_scene_fixes_drawer_through_root_properties(monkeypatch) -> None:
    tutorial = _load_tutorial_module()
    robot = object()
    drawer = object()
    captured: dict[str, object] = {}

    class FakeSimulation:
        is_newton_backend = False

        def add_robot(self, cfg):
            return robot

        def add_articulation(self, cfg):
            captured["drawer_cfg"] = cfg
            return drawer

    monkeypatch.setattr(
        tutorial.FrankaPandaCfg,
        "from_dict",
        lambda _config: SimpleNamespace(joint_drive_props=SimpleNamespace(damping={})),
    )
    monkeypatch.setattr(tutorial, "get_data_path", lambda asset: asset)

    tutorial.create_scene(FakeSimulation())

    drawer_cfg = captured["drawer_cfg"]
    assert drawer_cfg.root_props.fixed_base is True
