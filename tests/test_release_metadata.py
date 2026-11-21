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

import tomllib
from pathlib import Path
from urllib.parse import urlsplit
from zipfile import ZipFile

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name, parse_wheel_filename
from setuptools import Distribution
from setuptools.command.build_py import build_py

import setup as package_setup
from scripts.validate_wheel_metadata import WheelMetadataError, validate_wheel
from setup import get_package_data, get_packages

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_TEST_VERSION = "0.2.4"


def _read_project_metadata() -> dict:
    with (_REPOSITORY_ROOT / "pyproject.toml").open("rb") as stream:
        return tomllib.load(stream)


def test_docker_locks_match_core_dexsim_requirement() -> None:
    dependencies = _read_project_metadata()["project"]["dependencies"]
    core = next(
        requirement
        for requirement in map(Requirement, dependencies)
        if canonicalize_name(requirement.name) == "dexsim-engine"
    )

    for profile in ("py311", "py312"):
        requirements = []
        for directory in ("profiles", "locks"):
            path = (
                _REPOSITORY_ROOT
                / "docker"
                / directory
                / f"{profile}.{'in' if directory == 'profiles' else 'txt'}"
            )
            lines = [
                line
                for line in path.read_text().splitlines()
                if line.startswith("dexsim-engine @ ")
            ]
            assert len(lines) == 1
            requirements.append(Requirement(lines[0]))
        assert requirements[0].url == requirements[1].url

        wheel_name = Path(urlsplit(requirements[0].url).path).name
        package_name, version, _, tags = parse_wheel_filename(wheel_name)
        assert canonicalize_name(package_name) == "dexsim-engine"
        assert version in core.specifier
        assert any(tag.interpreter == f"cp{profile[2:]}" for tag in tags)


def _write_test_wheel(tmp_path: Path, requirements: list[str]) -> Path:
    metadata_lines = [
        "Metadata-Version: 2.4",
        "Name: embodichain",
        f"Version: {_TEST_VERSION}",
        *(f"Requires-Dist: {requirement}" for requirement in requirements),
        "",
    ]
    wheel_path = tmp_path / f"embodichain-{_TEST_VERSION}-py3-none-any.whl"
    with ZipFile(wheel_path, "w") as wheel:
        wheel.writestr(
            f"embodichain-{_TEST_VERSION}.dist-info/METADATA",
            "\n".join(metadata_lines),
        )
    return wheel_path


def test_official_tasks_share_core_distribution_metadata() -> None:
    task_root = _REPOSITORY_ROOT / "embodichain" / "extensions" / "tasks"
    metadata = _read_project_metadata()
    task_entry_points = metadata["project"]["entry-points"]["embodichain.tasks"]

    assert not (task_root / "VERSION").exists()
    assert not (task_root / "pyproject.toml").exists()
    assert task_entry_points == {"official_tasks": "embodichain.extensions.tasks"}


def test_package_discovery_includes_only_runtime_trees() -> None:
    packages = get_packages()
    runtime_roots = {
        "embodichain",
        "embodichain.extensions",
        "embodichain.extensions.tasks",
        "embodichain.extensions.robots",
        "embodichain.extensions.benchmarks",
        "embodichain.extensions.models",
        "embodichain.extensions.solvers",
        "embodichain.extensions.sensors",
    }
    assert runtime_roots | {
        "embodichain.extensions.tasks.configs",
        "embodichain.extensions.robots.dexforce.w1",
    } <= set(packages)
    assert "embodichain.extensions.tasks.manipulation.tableware" in packages
    assert "embodichain.lab.gym.envs.managers.observations" in packages
    assert "embodichain.extensions.solvers.physics.nerd" in packages
    assert "embodichain.extensions.tasks.tableware" not in packages
    assert all(package.partition(".")[0] == "embodichain" for package in packages)


def test_dry_run_preserves_retired_extension_staging(tmp_path: Path) -> None:
    stale_dir = tmp_path / "embodichain_tasks"
    stale_dir.mkdir()
    payload = stale_dir / "__init__.py"
    payload.touch()
    stale_link = tmp_path / "embodichain_models"
    stale_link.symlink_to(stale_dir, target_is_directory=True)

    distribution = Distribution()
    distribution.dry_run = True
    command = package_setup.BuildPy(distribution)
    command.build_lib = str(tmp_path)
    command.ensure_finalized()
    command.run()

    assert payload.is_file()
    assert stale_link.is_symlink()


def test_new_extension_package_and_resources_are_discovered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = tmp_path / "embodichain" / "extensions" / "example"
    (package / "feature").mkdir(parents=True)
    for module in (package.parent.parent, package.parent, package, package / "feature"):
        (module / "__init__.py").touch()
    resources = {
        "README.md": "Extension usage",
        "LICENSE": "Source license",
        "feature/LICENSE.txt": "Dependency license",
        "feature/THIRD_PARTY_LICENSES": "Third-party attribution",
        "feature/REFERENCE_SOURCES.json": "{}",
        "feature/profile.json": '{"robot": {}}',
        "feature/configs/default.yaml": "enabled: true\n",
        "feature/configs/override.yml": "enabled: false\n",
        "feature/model.urdf": '<robot name="example"/>',
        "feature/preview.png": "Preview image",
    }
    excluded = {
        ".private/secret.json": "Private resource",
        "feature/unrelated.txt": "Unrelated file",
        "feature/output.png": "Generated image",
        "feature/module.pyc": "Cached bytecode",
    }
    for relative, content in {**resources, **excluded}.items():
        path = package / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    monkeypatch.setattr(package_setup, "THIS_DIR", tmp_path)
    monkeypatch.chdir(tmp_path)
    packages = get_packages()
    assert {
        "embodichain.extensions.example",
        "embodichain.extensions.example.feature",
    } <= set(packages)
    distribution = Distribution(
        {
            "packages": packages,
            "package_data": get_package_data(),
            "include_package_data": False,
            "script_name": str(tmp_path / "setup.py"),
        }
    )
    command = build_py(distribution)
    command.build_lib = str(tmp_path / "lib")
    command.ensure_finalized()
    command.run()
    output = Path(command.build_lib) / package.relative_to(tmp_path)
    for relative, content in resources.items():
        assert (output / relative).read_text() == content
    for relative in excluded:
        assert not (output / relative).exists(), relative


def test_project_metadata_has_no_direct_dependencies() -> None:
    project = _read_project_metadata()["project"]
    requirements = list(project["dependencies"])
    for optional_requirements in project["optional-dependencies"].values():
        requirements.extend(optional_requirements)

    direct_dependencies = [
        requirement
        for requirement in requirements
        if Requirement(requirement).url is not None
    ]

    assert direct_dependencies == []


@pytest.mark.parametrize(
    ("version", "supported"),
    [
        ("0.6.0.dev11491", False),
        ("0.6.0rc1", False),
        ("0.6.0rc2", False),
        ("0.6.0", True),
    ],
)
def test_dexsim_dependency_selects_cable_release_wheel(
    version: str, supported: bool
) -> None:
    requirements = [
        Requirement(value)
        for value in _read_project_metadata()["project"]["dependencies"]
    ]
    dexsim_requirement = next(
        requirement
        for requirement in requirements
        if requirement.name.replace("_", "-") == "dexsim-engine"
    )

    assert dexsim_requirement.specifier.contains(version) is supported


def test_wheel_metadata_accepts_index_dependencies(tmp_path: Path) -> None:
    wheel_path = _write_test_wheel(
        tmp_path,
        ["gymnasium>=0.29.1", 'viser==1.0.21; python_version >= "3.11"'],
    )

    validate_wheel(wheel_path)


def test_wheel_metadata_rejects_direct_dependencies(tmp_path: Path) -> None:
    direct_dependency = (
        "nvidia-curobo[cu12] @ "
        'git+https://github.com/NVlabs/curobo.git@v0.8.0; extra == "curobo-cu12"'
    )
    wheel_path = _write_test_wheel(tmp_path, [direct_dependency])

    with pytest.raises(WheelMetadataError, match="nvidia-curobo"):
        validate_wheel(wheel_path)


@pytest.mark.parametrize("version", ["0.2.4", "0.2.4.post1", "0.2.4.post10"])
def test_setup_preserves_full_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, version: str
) -> None:
    (tmp_path / "VERSION").write_text(f"{version}\n", encoding="utf-8")
    monkeypatch.setattr(package_setup, "__file__", str(tmp_path / "setup.py"))

    assert package_setup.get_version() == version
