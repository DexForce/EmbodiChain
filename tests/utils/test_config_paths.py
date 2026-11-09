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

"""Tests for stable configuration-path resolution."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from embodichain.utils import resolve_config_path as exported_resolve_config_path
from embodichain.utils.config_paths import (
    resolve_config_path,
    resolve_local_resource_paths,
)


def test_resolve_config_path_preserves_existing_path(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("id: Test-v0\n", encoding="utf-8")

    assert resolve_config_path(config_path) == config_path


def test_resolve_config_path_is_exported_from_utils_package() -> None:
    assert exported_resolve_config_path is resolve_config_path


def test_resolve_config_path_preserves_ordinary_relative_path(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    assert resolve_config_path("local/config.yaml") == Path("local/config.yaml")


@pytest.mark.parametrize(
    "prefix", ["embodichain_tasks/configs", "extensions/embodichain_tasks/configs"]
)
def test_resolve_config_path_redirects_packaged_task_config(
    tmp_path: Path,
    monkeypatch,
    prefix: str,
) -> None:
    monkeypatch.chdir(tmp_path)

    resolved = resolve_config_path(
        f"{prefix}/tasks/manipulation/tableware/" "pour_water/task.cobotmagic.yaml"
    )

    assert resolved.is_file()
    assert resolved.name == "task.cobotmagic.yaml"
    assert resolved.parent.name == "pour_water"


@pytest.mark.parametrize(
    "prefix", ["embodichain_tasks/configs", "extensions/embodichain_tasks/configs"]
)
def test_resolve_config_path_rejects_packaged_path_escape(
    tmp_path: Path,
    monkeypatch,
    prefix: str,
) -> None:
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="stay within the package"):
        resolve_config_path(f"{prefix}/../VERSION")


def test_owner_relative_config_ignores_same_named_cwd_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explicit owner directory controls ordinary relative references."""
    owner = tmp_path / "owner"
    owner.mkdir()
    (owner / "config.yaml").write_text("owner: true\n")
    (tmp_path / "config.yaml").write_text("shadow: true\n")
    monkeypatch.chdir(tmp_path)

    assert resolve_config_path("config.yaml", base_dir=owner) == owner / "config.yaml"
    assert resolve_config_path("config.yaml") == Path("config.yaml")
    assert resolve_config_path(owner / "config.yaml", base_dir=tmp_path) == (
        owner / "config.yaml"
    )


@pytest.mark.parametrize(
    "prefix", ["embodichain_tasks/configs", "extensions/embodichain_tasks/configs"]
)
def test_owned_official_config_ignores_same_named_cwd_shadow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prefix: str
) -> None:
    """Component references bind packaged configs to their imported SDK."""
    reference = Path(f"{prefix}/components/embodiments/franka_panda.yaml")
    shadow = tmp_path / reference
    shadow.parent.mkdir(parents=True)
    shadow.write_text("shadow: true\n")
    monkeypatch.chdir(tmp_path)
    expected = (
        Path(__file__).resolve().parents[2]
        / "extensions/embodichain_tasks/configs/components/embodiments/franka_panda.yaml"
    )

    assert resolve_config_path(reference, base_dir=tmp_path / "deployment") == expected
    assert resolve_config_path(reference) == reference


@pytest.mark.parametrize(
    "prefix", ["embodichain_tasks/configs", "extensions/embodichain_tasks/configs"]
)
def test_owned_official_config_rejects_escape_even_when_cwd_target_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prefix: str
) -> None:
    """An existing CWD file cannot bypass the packaged selector boundary."""
    shadow_root = tmp_path / Path(prefix).parent
    (shadow_root / "configs").mkdir(parents=True)
    (shadow_root / "VERSION").write_text("shadow\n")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="stay within the package"):
        resolve_config_path(f"{prefix}/../VERSION", base_dir=tmp_path / "deployment")


def test_local_resource_paths_keep_their_owner_without_changing_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    owner = tmp_path / "components"
    owner.mkdir()
    asset = tmp_path / "robot.urdf"
    asset.write_text("<robot name='path-test'/>")
    local = owner / "scene.usda"
    local.write_text("#usda 1.0\n")
    declaration = {
        "robot": {"urdf_path": "../robot.urdf"},
        "scene": [{"fpath": "./scene.usda"}],
        "sensor": {"xml_path": "../robot.urdf"},
        "registry": {"fpath": "Robot/model.urdf"},
        "absolute": {"urdf_path": str(asset)},
        "label": "../ordinary-text",
    }
    before = deepcopy(declaration)
    # Resolution must not depend on the caller's CWD or mutate its declaration.
    monkeypatch.chdir(tmp_path)
    resolved = resolve_local_resource_paths(declaration, base_dir=owner)
    assert resolved == {
        "robot": {"urdf_path": str(asset)},
        "scene": [{"fpath": str(local)}],
        "sensor": {"xml_path": str(asset)},
        "registry": {"fpath": "Robot/model.urdf"},
        "absolute": {"urdf_path": str(asset)},
        "label": "../ordinary-text",
    }
    assert declaration == before
    resolved["registry"]["fpath"] = "changed"
    assert declaration == before


def test_local_resource_paths_reject_missing_owned_file(tmp_path: Path) -> None:
    declaration = {"fpath": "./missing.usda"}
    with pytest.raises(FileNotFoundError):
        resolve_local_resource_paths(declaration, base_dir=tmp_path)
    assert declaration == {"fpath": "./missing.usda"}
