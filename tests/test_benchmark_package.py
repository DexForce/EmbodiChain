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

"""Build and install the real wheel before probing benchmark APIs outside git."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import pytest

pytestmark = pytest.mark.no_sim


@pytest.fixture(scope="module")
def installed_benchmark(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """Build from isolated sources and install without changing the Python env."""
    root = Path(__file__).resolve().parents[1]
    workspace = tmp_path_factory.mktemp("installed_benchmark")
    source = workspace / "source"
    source.mkdir()
    for name in (
        "setup.py",
        "pyproject.toml",
        "VERSION",
        "README.md",
        "LICENSE",
        "MANIFEST.in",
    ):
        shutil.copy2(root / name, source / name)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info")
    for name in (
        "embodichain",
        "extensions/embodichain_robots",
        "extensions/embodichain_tasks",
    ):
        shutil.copytree(root / name, source / name, ignore=ignore)
    built = subprocess.run(
        [
            sys.executable,
            "-I",
            "setup.py",
            "bdist_wheel",
            "--dist-dir",
            str(workspace / "dist"),
        ],
        cwd=source,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert built.returncode == 0, built.stdout + built.stderr
    wheel = next((workspace / "dist").glob("*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        entries = archive.namelist()
        assert "embodichain/benchmarks/motion_generation/api.py" in entries
        assert "embodichain/cli/benchmark.py" in entries
        suites = {
            Path(name).name
            for name in entries
            if name.startswith("embodichain/benchmarks/motion_generation/suites/")
            and name.endswith(".yaml")
        }
        assert suites == {
            path.name
            for path in (root / "embodichain/benchmarks/motion_generation/suites").glob(
                "*.yaml"
            )
        }
        assert not any(name.startswith("scripts/") for name in entries)
    target = workspace / "consumer_env"
    created = subprocess.run(
        [sys.executable, "-I", "-m", "venv", "--system-site-packages", str(target)],
        cwd=workspace,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert created.returncode == 0, created.stdout + created.stderr
    # Expose this exact version as an inherited installation. The consumer
    # must install its own copy rather than reuse that distribution.
    inherited = workspace / "inherited"
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(inherited)
    site_packages = (
        target
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    (site_packages / "inherited_benchmark.pth").write_text(str(inherited) + "\n")
    interpreter = target / "bin/python"
    installed = subprocess.run(
        [
            str(interpreter),
            "-I",
            "-m",
            "pip",
            "--isolated",
            "install",
            "--ignore-installed",
            "--no-deps",
            "--no-compile",
            str(wheel),
        ],
        cwd=workspace,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert installed.returncode == 0, installed.stdout + installed.stderr
    return target, source


def test_installed_suites_and_cli_work_without_source_checkout(
    installed_benchmark: tuple[Path, Path], tmp_path: Path
) -> None:
    target, source = installed_benchmark
    # Prove resources are loaded from the wheel even when its build tree is gone.
    shutil.rmtree(source)
    script = r"""
import ast
import importlib.abc
from importlib.resources import files
from pathlib import Path
import sys

target, repository = map(Path, sys.argv[1:3])
sys.meta_path = [finder for finder in sys.meta_path if "__editable__" not in str(finder)]
sys.path = [entry for entry in sys.path if entry and not Path(entry).is_relative_to(repository)]

class RejectScripts(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "scripts" or fullname.startswith("scripts."):
            raise AssertionError("installed benchmark imported source-checkout scripts")

sys.meta_path.insert(0, RejectScripts())
import embodichain.benchmarks.motion_generation as benchmark
from embodichain.benchmarks.motion_generation import load_suite, run_benchmark
assert Path(benchmark.__file__).is_relative_to(target)
assert "embodichain.lab.sim" not in sys.modules
for source in Path(benchmark.__file__).parent.rglob("*.py"):
    for node in ast.walk(ast.parse(source.read_text())):
        names = ([node.module] if isinstance(node, ast.ImportFrom) else
                 [alias.name for alias in node.names] if isinstance(node, ast.Import) else [])
        assert not any(name and name.split(".")[0] == "scripts" for name in names), source
resources = files(benchmark).joinpath("suites")
suite_names = sorted(path.name[:-5] for path in resources.iterdir() if path.name.endswith(".yaml"))
for name in suite_names:
    suite = load_suite(name)
    assert suite.name
assert load_suite().name == "motion_panda_free_space_smoke"
from embodichain.benchmarks.motion_generation import PlannerAdapter
from embodichain.benchmarks.motion_generation.scenarios.atomic_grasps import parallel_jaw_sampler
assert parallel_jaw_sampler(2)
from embodichain.lab.scripts._atomic_action.placement_geometry import build_support_object_target_pose
import torch
assert build_support_object_target_pose(torch.eye(4), torch.device("cpu")).shape == (4, 4)
from embodichain.benchmarks.motion_generation.artifacts import environment_metadata
assert environment_metadata()["git_commit"] is None
from embodichain.cli.main import main
try:
    main(["benchmark", "motion-generation", "--help"])
except SystemExit as error:
    assert error.code == 0
else:
    raise AssertionError("CLI did not dispatch to the installed suite parser")
print("installed benchmark suites and CLI passed")
"""
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [
            str(target / "bin/python"),
            "-I",
            "-c",
            script,
            str(target),
            str(Path(__file__).resolve().parents[1]),
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "--algorithms" in result.stdout
    assert "installed benchmark suites and CLI passed" in result.stdout
    cli = subprocess.run(
        [str(target / "bin/embodichain"), "benchmark", "motion-generation", "--help"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert cli.returncode == 0, cli.stdout + cli.stderr
    assert "--suite" in cli.stdout
