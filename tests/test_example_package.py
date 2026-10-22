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

import pytest
from setuptools import Distribution

import setup as package_setup
from setup import get_example_data_files, get_package_dir, get_packages

pytestmark = pytest.mark.no_sim


def test_distribution_preserves_the_ordinary_examples_tree(tmp_path: Path) -> None:
    distribution = Distribution(
        {"data_files": get_example_data_files(), "script_name": "setup.py"}
    )
    command = distribution.get_command_obj("install_data")
    command.install_dir = str(tmp_path)
    command.ensure_finalized()
    command.run()
    output = tmp_path / "share/embodichain/examples"
    for relative in (
        "__init__.py",
        "README.md",
        "sim/atomic_action/__main__.py",
        "sim/atomic_action/host.py",
        "sim/motion/planners/curobo_planner.py",
        "ods/simple_bc_demo/train.py",
        "ods/simple_bc_demo/configs/task.yaml",
        "learning/differentiable_ik/learn_grasp_yaw.py",
        "expansion/asset_generalization/configs/pour_water/task.cobotmagic.yaml",
    ):
        assert (output / relative).is_file(), relative
    assert all(
        name.partition(".")[0]
        in {"embodichain", "embodichain_tasks", "embodichain_robots"}
        for name in get_packages()
    )
    assert "examples" not in get_package_dir()
    assert not (output / "embodichain_examples").exists()


def test_new_example_and_resources_are_packaged_without_registration(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "examples"
    (root / "new_family/configs").mkdir(parents=True)
    (root / "new_family/demo.py").write_text("print('demo')\n")
    (root / "new_family/configs/default.yaml").write_text("enabled: true\n")
    (root / "new_family/__pycache__").mkdir()
    (root / "new_family/__pycache__/demo.pyc").write_bytes(b"cache")
    (root / ".hidden").mkdir()
    (root / ".hidden/secret.py").write_text("hidden")
    monkeypatch.setattr(package_setup, "THIS_DIR", tmp_path)
    monkeypatch.setattr(package_setup, "EXAMPLES_DIR", root)
    files = {destination: sources for destination, sources in get_example_data_files()}
    assert files == {
        "share/embodichain/examples/new_family": ["examples/new_family/demo.py"],
        "share/embodichain/examples/new_family/configs": [
            "examples/new_family/configs/default.yaml"
        ],
    }
