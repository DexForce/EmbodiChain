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
from zipfile import ZipFile

import pytest
from packaging.requirements import Requirement

from scripts.validate_wheel_metadata import WheelMetadataError, validate_wheel
from setup import get_package_dir, get_packages

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_TEST_VERSION = "0.2.4"


def _read_project_metadata() -> dict:
    with (_REPOSITORY_ROOT / "pyproject.toml").open("rb") as stream:
        return tomllib.load(stream)


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
    task_root = _REPOSITORY_ROOT / "embodichain_tasks"
    metadata = _read_project_metadata()
    task_entry_points = metadata["project"]["entry-points"]["embodichain.tasks"]

    assert not (task_root / "VERSION").exists()
    assert not (task_root / "pyproject.toml").exists()
    assert task_entry_points == {"embodichain_tasks": "embodichain_tasks"}


def test_package_discovery_includes_only_runtime_trees() -> None:
    packages = get_packages()
    package_dir = get_package_dir()

    assert {"embodichain", "embodichain_tasks", "embodichain_tasks.configs"} <= set(
        packages
    )
    assert "embodichain_tasks.manipulation.tableware" in packages
    assert "embodichain_tasks.tableware" not in packages
    assert all(
        package in {"embodichain", "embodichain_tasks"}
        or package.startswith(("embodichain.", "embodichain_tasks."))
        for package in packages
    )
    assert package_dir == {
        "embodichain_tasks": "embodichain_tasks/embodichain_tasks",
        "embodichain_tasks.configs": "embodichain_tasks/configs",
    }


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


def test_wheel_metadata_accepts_index_dependencies(tmp_path: Path) -> None:
    wheel_path = _write_test_wheel(
        tmp_path,
        ["gymnasium>=0.29.1", 'viser==1.0.21; python_version >= "3.10"'],
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
