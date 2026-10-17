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

"""Canonical robot imports are lazy; retired compatibility paths stay absent."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.no_sim

_LEGACY_MODULES = (
    "aloha_mini",
    "tianji_marvin",
    "cobotmagic",
    "franka_panda",
    "ur_robot",
    "dexforce_w1",
    "dexforce_w1.cfg",
    "dexforce_w1.types",
    "dexforce_w1.specs",
    "dexforce_w1.hand_specs",
    "dexforce_w1.params",
    "dexforce_w1.utils",
)
_CONFIG_CLASSES = (
    "AlohaMiniCfg",
    "TianjiMarvinCfg",
    "CobotMagicCfg",
    "FrankaPandaCfg",
    "FrankaPandaPgiCfg",
    "URRobotCfg",
    "DexforceW1Cfg",
)


@pytest.mark.parametrize("module_name", _LEGACY_MODULES)
def test_concrete_robot_legacy_module_is_removed(module_name: str) -> None:
    """Old module imports fail at their retired path, not a missing dependency."""
    name = "embodichain.lab.sim.robots." + module_name
    with pytest.raises(ModuleNotFoundError) as error:
        import_module(name)
    assert error.value.name in {name, "embodichain.lab.sim.robots.dexforce_w1"}


@pytest.mark.parametrize("class_name", _CONFIG_CLASSES)
def test_core_does_not_reexport_concrete_robot_classes(class_name: str) -> None:
    """Core's public robot API contains only generic dual-arm composition."""
    core = import_module("embodichain.lab.sim.robots")
    with pytest.raises(AttributeError, match=class_name):
        getattr(core, class_name)
    assert set(core.__all__) == {"DualArmRobotCfg", "build_dual_arm_cfg"}


@pytest.mark.parametrize("class_name", _CONFIG_CLASSES)
def test_concrete_robot_class_uses_canonical_package(class_name: str) -> None:
    """Every official preset is accessible from the new public import package."""
    canonical = import_module("embodichain_robots")
    cfg_cls = getattr(canonical, class_name)
    assert cfg_cls.__module__.startswith("embodichain_robots.")


@pytest.mark.parametrize(
    "first_module",
    (
        "embodichain_robots._catalog",
        "embodichain.robot_registry",
        "embodichain_robots.dexforce_w1.types",
        "embodichain.lab.sim.robots",
        "embodichain.lab.sim.robots.dual_arm",
    ),
)
def test_package_imports_defer_concrete_robot_configs(first_module: str) -> None:
    """Importing catalogs or generic composition does not load concrete presets."""
    script = """
import importlib
import sys
module = sys.argv[1]
importlib.import_module(module)
for name in (
    "aloha_mini", "tianji_marvin", "cobotmagic", "franka_panda", "ur_robot",
    "dexforce_w1.cfg",
):
    assert "embodichain_robots." + name not in sys.modules, name
if module.startswith("embodichain_robots") or module == "embodichain.robot_registry":
    assert "embodichain.lab.sim" not in sys.modules
    assert "torch" not in sys.modules
"""
    root = Path(__file__).resolve().parents[3]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root)
    result = subprocess.run(
        [sys.executable, "-c", script, first_module],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
