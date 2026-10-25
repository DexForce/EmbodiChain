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
"""Integrity and installation isolation for optional assembly dependency patches."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.patches import apply_mujoco_warp_epa_horizon as patcher


@pytest.fixture
def source_package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Prepare a tiny dependency fixture with the patch's source context."""
    source = tmp_path / "source" / "mujoco_warp"
    (source / "_src").mkdir(parents=True)
    (source / "__init__.py").write_text("")
    source_text = (
        "# Original third-party license fixture\n"
        + "\n" * 26
        + "MJ_MINMU = mujoco.mjMINMU  # minimum friction\n"
        + "MJ_MINAWAKE = mujoco.mjMINAWAKE  # minimum number of timesteps before sleeping\n"
        + "# maximum size (by number of edges) of an horizon in EPA algorithm\n"
        + "MJ_MAX_EPAHORIZON = 24\n"
        + "# maximum average number of trianglarfaces EPA can insert at each iteration\n"
        + "MJ_MAX_EPAFACES = 5\n\n"
    )
    sources = {
        "_src/types.py": source_text,
        "_src/collision_convex.py": "# convex allocator fixture\n",
        "_src/collision_flex.py": "# flex allocator fixture\n",
        "_src/collision_gjk.py": "# unchanged EPA algorithm fixture\n",
    }
    for name, text in sources.items():
        (source / name).write_text(text)
    monkeypatch.setattr(
        patcher,
        "_SOURCE_SHA256",
        {
            name: hashlib.sha256(text.encode()).hexdigest()
            for name, text in sources.items()
        },
    )
    monkeypatch.setattr(
        patcher.importlib.metadata,
        "distribution",
        lambda name: SimpleNamespace(version="3.12.0", locate_file=lambda name: source),
    )
    return source


def test_epa_overlay_changes_only_copied_constant(
    tmp_path: Path, source_package: Path
) -> None:
    original = {
        name: (source_package / name).read_bytes() for name in patcher._SOURCE_SHA256
    }
    output = patcher.apply_overlay(tmp_path / "overlay")
    for name, content in original.items():
        assert (source_package / name).read_bytes() == content
        expected = (
            content.replace(b"MJ_MAX_EPAHORIZON = 24\n", b"MJ_MAX_EPAHORIZON = 128\n")
            if name == "_src/types.py"
            else content
        )
        assert (output / "mujoco_warp" / name).read_bytes() == expected
    provenance = json.loads((output / "provenance.json").read_text())
    assert provenance["additional_convex_horizon_bytes_per_naccdmax"] == 416
    assert provenance["epa_horizon_after"] == 128


@pytest.mark.parametrize("mismatch", ["version", "source_hash"])
def test_epa_overlay_rejects_dependency_drift_before_copying(
    tmp_path: Path, source_package: Path, monkeypatch: pytest.MonkeyPatch, mismatch: str
) -> None:
    if mismatch == "version":
        monkeypatch.setattr(
            patcher.importlib.metadata,
            "distribution",
            lambda name: SimpleNamespace(version="3.13.0"),
        )
    else:
        with (source_package / "_src/collision_gjk.py").open("a") as stream:
            stream.write("# a different EPA implementation\n")
    output = tmp_path / "rejected"
    with pytest.raises(RuntimeError, match="Expected mujoco-warp|SHA-256 mismatch"):
        patcher.apply_overlay(output)
    assert not output.exists()


def test_epa_overlay_preserves_existing_output(
    tmp_path: Path, source_package: Path
) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "user-file"
    sentinel.write_text("preserve")
    with pytest.raises(FileExistsError):
        patcher.apply_overlay(output)
    assert sentinel.read_text() == "preserve"
