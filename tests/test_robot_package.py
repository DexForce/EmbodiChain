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

"""The main wheel stages official robot presets and core dual-arm composition."""

from __future__ import annotations

from pathlib import Path

from setuptools import Distribution
from setuptools.command.build_py import build_py

from setup import get_package_dir, get_packages


def test_main_distribution_stages_robot_package_without_legacy_modules(tmp_path):
    packages = [
        name
        for name in get_packages()
        if name == "embodichain"
        or name.startswith(("embodichain_robots", "embodichain.lab.sim.robots"))
    ]
    assert "embodichain_robots" in packages
    assert "embodichain_robots.dexforce_w1" in packages
    distribution = Distribution(
        {
            "packages": packages,
            "package_dir": get_package_dir(),
            "script_name": "setup.py",
        }
    )
    command = build_py(distribution)
    command.build_lib = str(tmp_path / "lib")
    command.ensure_finalized()
    command.run()
    output = Path(command.build_lib)

    for relative in (
        "embodichain_robots/__init__.py",
        "embodichain_robots/_catalog.py",
        "embodichain_robots/franka_panda.py",
        "embodichain_robots/ur_robot.py",
        "embodichain_robots/dexforce_w1/cfg.py",
        "embodichain_robots/dexforce_w1/types.py",
        "embodichain/lab/sim/robots/dual_arm.py",
        "embodichain/robot_registry.py",
    ):
        assert (output / relative).is_file(), relative

    legacy = output / "embodichain/lab/sim/robots"
    assert {path.name for path in legacy.glob("*.py")} == {
        "__init__.py",
        "dual_arm.py",
    }
    assert not (legacy / "dexforce_w1").exists()
