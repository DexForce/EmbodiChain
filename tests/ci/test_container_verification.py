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

"""Protect container qualification from source-checkout imports and resources."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import shutil
import sys
import venv

import pytest


@pytest.mark.parametrize("installation", ["wheel", "missing-resource", "editable"])
def test_qualification_requires_installed_code_and_resources(
    installation: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    verifier = Path(__file__).resolve().parents[2] / "docker/verify.py"
    spec = importlib.util.spec_from_file_location("container_verify", verifier)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # A small installed distribution isolates the packaging contract from the
    # heavyweight SDK. Inherited site packages provide pytest, not the payload.
    environment = tmp_path / "runtime"
    venv.EnvBuilder(with_pip=False, system_site_packages=True).create(environment)
    site = (
        environment
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    metadata = site / "embodichain-0.0.0.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text("Name: embodichain\nVersion: 0.0.0\n")
    repo = tmp_path / "checkout"
    for name in ("embodichain", "embodichain_tasks", "embodichain_robots"):
        for base in (site, repo):
            package = base / name
            package.mkdir(parents=True)
            source = (
                "raise RuntimeError('source checkout imported')\n"
                if base == repo and installation == "wheel"
                else ""
            )
            (package / "__init__.py").write_text(source)
    (repo / "embodichain/payload.txt").write_text("shipped")
    if installation == "wheel":
        (site / "embodichain/payload.txt").write_text("shipped")
    if installation == "editable":
        for name in ("embodichain", "embodichain_tasks", "embodichain_robots"):
            shutil.rmtree(site / name)
        (site / "editable.pth").write_text(str(repo) + "\n")

    (repo / "scripts").mkdir()
    (repo / "scripts/support.py").write_text("EXPECTED = 'shipped'\n")
    (repo / "tests").mkdir()
    (repo / "pyproject.toml").write_text("[tool.pytest.ini_options]\n")
    (repo / "tests/test_payload.py").write_text(
        "from importlib.resources import files\n"
        "import subprocess, sys\n"
        "from scripts.support import EXPECTED\n\n"
        "def test_packaged_payload():\n"
        "    assert files('embodichain').joinpath('payload.txt').read_text() == EXPECTED\n"
        "    child = subprocess.check_output([sys.executable, '-c', "
        '"from importlib.resources import files; '
        "print(files('embodichain').joinpath('payload.txt').read_text())\"], text=True)\n"
        "    assert child.strip() == EXPECTED\n"
    )
    monkeypatch.setenv("PYTHONPATH", str(repo))
    monkeypatch.setattr(module.sys, "executable", str(environment / "bin/python"))

    if installation == "wheel":
        result = module._pytest_checks(repo, ["tests/test_payload.py"])
        assert result == {
            "cases": 1,
            "imports": {
                name: str(site / name)
                for name in ("embodichain", "embodichain_tasks", "embodichain_robots")
            },
        }
    else:
        with pytest.raises(RuntimeError, match="Qualification pytest failed"):
            module._pytest_checks(repo, ["tests/test_payload.py"])
        output = capfd.readouterr()
        expected = (
            "FileNotFoundError"
            if installation == "missing-resource"
            else "Qualification requires installed wheel"
        )
        assert expected in output.out + output.err
