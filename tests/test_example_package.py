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
import subprocess
import sys

from setuptools import Distribution
from setuptools.command.build_py import build_py

from setup import get_package_dir, get_packages


def test_distribution_stages_examples_in_their_own_namespace(tmp_path: Path) -> None:
    packages = [
        name for name in get_packages() if name.startswith("embodichain_examples")
    ]
    distribution = Distribution(
        {
            "packages": packages,
            "package_dir": get_package_dir(),
            "package_data": {
                "embodichain_examples": ["README.md"],
                "embodichain_examples.atomic_action": ["README.md"],
            },
            "script_name": "setup.py",
        }
    )
    command = build_py(distribution)
    command.build_lib = str(tmp_path / "lib")
    command.ensure_finalized()
    command.run()
    output = Path(command.build_lib)
    for relative in (
        "embodichain_examples/__init__.py",
        "embodichain_examples/_catalog.py",
        "embodichain_examples/atomic_action/run.py",
        "embodichain_examples/atomic_action/host.py",
        "embodichain_examples/atomic_action/pickup.py",
        "embodichain_examples/atomic_action/README.md",
    ):
        assert (output / relative).is_file(), relative
    assert not any(
        name.startswith("embodichain.lab.scripts._atomic_action")
        for name in get_packages()
    )
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            (
                "import sys; "
                "from embodichain_examples._catalog import EXAMPLES; "
                "assert 'atomic-action' in EXAMPLES; "
                "assert not {'torch', 'dexsim', 'gymnasium'} & sys.modules.keys()"
            ),
        ],
        cwd=output,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
