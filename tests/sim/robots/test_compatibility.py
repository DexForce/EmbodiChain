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

"""Legacy imports share canonical robot classes, enums, and module state."""

from __future__ import annotations

from importlib import import_module
import os
from pathlib import Path
import subprocess
import sys

import pytest

pytestmark = pytest.mark.no_sim

_MODULES = [
    "aloha_mini",
    "tianji_marvin",
    "cobotmagic",
    "franka_panda",
    "ur_robot",
    "dexforce_w1.cfg",
    "dexforce_w1.types",
    "dexforce_w1.specs",
    "dexforce_w1.hand_specs",
    "dexforce_w1.params",
    "dexforce_w1.utils",
]
_CLASSES = [
    "AlohaMiniCfg",
    "TianjiMarvinCfg",
    "CobotMagicCfg",
    "FrankaPandaCfg",
    "URRobotCfg",
    "DexforceW1Cfg",
]


@pytest.mark.parametrize("module_name", _MODULES)
def test_legacy_leaf_module_is_canonical_module(module_name):
    legacy = import_module("embodichain.lab.sim.robots." + module_name)
    canonical = import_module("embodichain_robots." + module_name)

    assert legacy is canonical


@pytest.mark.parametrize("class_name", _CLASSES)
def test_legacy_top_level_export_is_canonical_class(class_name):
    legacy = import_module("embodichain.lab.sim.robots")
    canonical = import_module("embodichain_robots")

    assert getattr(legacy, class_name) is getattr(canonical, class_name)


def test_w1_package_export_and_enum_preserve_identity():
    from embodichain.lab.sim.robots.dexforce_w1 import DexforceW1Cfg
    from embodichain.lab.sim.robots.dexforce_w1.types import DexforceW1Version
    from embodichain_robots.dexforce_w1 import DexforceW1Cfg as CanonicalCfg
    from embodichain_robots.dexforce_w1.types import (
        DexforceW1Version as CanonicalVersion,
    )

    assert DexforceW1Cfg is CanonicalCfg
    assert DexforceW1Version.V025 is CanonicalVersion.V025


def test_monkeypatching_legacy_module_updates_canonical_module(monkeypatch):
    legacy = import_module("embodichain.lab.sim.robots.dexforce_w1.utils")
    canonical = import_module("embodichain_robots.dexforce_w1.utils")
    resolver = lambda path: "/mock/" + path

    monkeypatch.setattr(legacy, "get_data_path", resolver)

    assert canonical.get_data_path("asset.urdf") == "/mock/asset.urdf"


@pytest.mark.parametrize(
    "first_module",
    [
        "embodichain_robots._catalog",
        "embodichain.robot_registry",
        "embodichain_robots.dexforce_w1.types",
        "embodichain.lab.sim.robots",
        "embodichain.lab.sim.robots.dexforce_w1",
        "embodichain.lab.sim.robots.dual_arm",
    ],
)
def test_package_imports_defer_concrete_robot_configs(first_module):
    script = r"""
import importlib
import sys
module = sys.argv[1]
importlib.import_module(module)
if module == "embodichain.robot_registry":
    registry = sys.modules[module]
    assert "FrankaPanda" in registry.list_robot_cfgs()
for name in (
    "aloha_mini", "tianji_marvin", "cobotmagic", "franka_panda", "ur_robot",
    "dexforce_w1.cfg",
):
    assert "embodichain_robots." + name not in sys.modules, name
if module.startswith("embodichain_robots") or module == "embodichain.robot_registry":
    assert "embodichain.lab.sim" not in sys.modules
    assert "torch" not in sys.modules
if module == "embodichain.lab.sim.robots":
    legacy = sys.modules[module]
    assert legacy.ur_robot is importlib.import_module("embodichain_robots.ur_robot")
    assert "embodichain_robots.franka_panda" not in sys.modules
if module == "embodichain.lab.sim.robots.dexforce_w1":
    legacy = sys.modules[module]
    assert legacy.types is importlib.import_module("embodichain_robots.dexforce_w1.types")
    assert "embodichain_robots.dexforce_w1.cfg" not in sys.modules
"""
    env = os.environ.copy()
    # Exercise the installed editable package outside the source checkout.
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "-c", script, first_module],
        cwd=Path.cwd().parent,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stdout + result.stderr
