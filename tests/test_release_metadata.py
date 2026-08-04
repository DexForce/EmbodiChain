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

from setup import get_package_dir, get_packages

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _read_project_metadata() -> dict:
    with (_REPOSITORY_ROOT / "pyproject.toml").open("rb") as stream:
        return tomllib.load(stream)


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
    assert all(
        package in {"embodichain", "embodichain_tasks"}
        or package.startswith(("embodichain.", "embodichain_tasks."))
        for package in packages
    )
    assert package_dir == {
        "embodichain_tasks": "embodichain_tasks/embodichain_tasks",
        "embodichain_tasks.configs": "embodichain_tasks/configs",
    }
