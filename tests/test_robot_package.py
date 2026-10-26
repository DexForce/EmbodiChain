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
import subprocess
import sys

from setuptools import Distribution
from setuptools.command.build_py import build_py

from setup import get_package_dir, get_packages


def test_w1_script_resolves_core_and_robots_without_editable_install(
    tmp_path: Path,
) -> None:
    """Standalone startup finds both source trees without editable finders."""
    root = Path(__file__).resolve().parents[1]
    script = """
import importlib.abc
import importlib.machinery
from pathlib import Path
import runpy
import sys

root = Path(sys.argv[1])
sys.meta_path = [finder for finder in sys.meta_path if '__editable__' not in str(finder)]
sys.path = [entry for entry in sys.path if entry and not Path(entry).is_relative_to(root)]

class CoreDiscovered(Exception):
    pass

class CoreProbe(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'embodichain':
            spec = importlib.machinery.PathFinder.find_spec(fullname, sys.path)
            assert spec is not None, 'standalone script cannot find the core package'
            assert Path(spec.origin) == root / 'embodichain/__init__.py'
            robot = sys.modules['embodichain_robots']
            assert Path(robot.__file__) == root / 'extensions/embodichain_robots/__init__.py'
            raise CoreDiscovered

sys.meta_path.insert(0, CoreProbe())
try:
    runpy.run_path(str(root / 'extensions/embodichain_robots/dexforce/w1/cfg.py'), run_name='__main__')
except CoreDiscovered:
    pass
else:
    raise AssertionError('standalone script never attempted to import the core package')
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(root)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr


def test_main_distribution_stages_robot_package_without_legacy_modules(tmp_path):
    packages = [
        name
        for name in get_packages()
        if name == "embodichain"
        or name.startswith(("embodichain_robots", "embodichain.lab.sim.robots"))
    ]
    distribution = Distribution(
        {
            "packages": packages,
            "package_dir": get_package_dir(),
            "package_data": {"": ["profile.json"]},
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
        "embodichain_robots/_asset_catalog.py",
        "embodichain_robots/_locomotion.py",
        "embodichain_robots/locomotion_profiles.py",
        "embodichain_robots/franka/assets.py",
        "embodichain_robots/universal_robots/assets.py",
        "embodichain_robots/dexforce/w1/assets.py",
        "embodichain_robots/the_robot_studio/so101/assets.py",
        "embodichain_robots/franka/panda.py",
        "embodichain_robots/universal_robots/ur_robot.py",
        "embodichain_robots/flexiv/rizon4s/cfg.py",
        "embodichain_robots/the_robot_studio/so101/cfg.py",
        "embodichain_robots/dexforce/w1/cfg.py",
        "embodichain_robots/dexforce/w1/types.py",
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
    for retired in ("franka_panda.py", "ur_robot.py", "so101.py", "dexforce_w1"):
        assert not (output / "embodichain_robots" / retired).exists()

    for owner, retired in (
        ("unitree/go1", "unitree_go1"),
        ("unitree/go2", "unitree_go2"),
        ("unitree/g1", "unitree_g1"),
        ("unitree/h1_2", "unitree_h1_2"),
        ("anybotics/anymal_c", "anymal_c"),
        ("pollen_robotics/microduck", "microduck"),
    ):
        for filename in ("__init__.py", "cfg.py", "assets.py", "profile.json"):
            assert (output / "embodichain_robots" / owner / filename).is_file()
        assert not (output / "embodichain_robots" / f"{retired}.py").exists()

    script = """
import sys
sys.path.insert(0, sys.argv[1])
from embodichain_robots.locomotion_profiles import load_locomotion_robot_profile
for robot, count in (
    ('UnitreeGo1', 12), ('UnitreeGo2', 12), ('UnitreeG1', 29),
    ('UnitreeH1_2', 27), ('ANYmalC', 12), ('MicroDuck', 14),
):
    assert len(load_locomotion_robot_profile(robot)['robot']['joint_names']) == count
profile = load_locomotion_robot_profile('UnitreeGo2')
assert profile['robot']['joint_names'][0] == 'FL_hip_joint'
assert profile['robot']['stiffness'][0] == 20.0
profile['robot']['stiffness'][0] = 999.0
assert load_locomotion_robot_profile('UnitreeGo2')['robot']['stiffness'][0] == 20.0
assert 'torch' not in sys.modules
assert 'embodichain.lab.sim' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(output)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
