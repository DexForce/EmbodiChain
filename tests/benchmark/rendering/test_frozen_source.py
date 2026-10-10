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

"""File/subprocess source-binding checks without simulator or GPU imports."""

from __future__ import annotations

from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

import pytest

from scripts.benchmark.rendering import frozen_source as frozen


def test_environment_uses_original_helper_and_lexical_venv_interpreter(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    helper = source / "scripts/benchmark/rendering/formal_runner.py"
    helper.parent.mkdir(parents=True)
    helper.write_text(
        "def _runtime_environment(root, python, isaaclab):\n"
        "    return {'VIRTUAL_ENV': str(python.parent.parent), "
        "'ORIGINAL_HELPER': str(root), 'PYTHON_PATH': str(python), "
        "'ISAACLAB': str(isaaclab)}\n"
    )
    python = tmp_path / "isolated-venv/bin/python"
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    environment = frozen.original_environment(source, python, tmp_path / "isaac")
    assert environment["VIRTUAL_ENV"] == str(python.parent.parent)
    assert environment["PYTHON_PATH"] == str(python)
    assert environment["ORIGINAL_HELPER"] == str(source)
    assert environment["PYTHONDONTWRITEBYTECODE"] == "1"
    assert not list(source.rglob("__pycache__"))


@pytest.mark.parametrize("changed", ["none", "dirty", "commit", "source"])
def test_source_pin_checks_commit_tracked_edits_and_actual_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changed: str
) -> None:
    builder = runpy.run_path(str(Path(__file__).with_name("test_history_contract.py")))[
        "build_valid_history_fixture"
    ]
    fixture = builder(tmp_path)
    original = {"row": fixture["expected_original"], "config": fixture["config"]}
    source = tmp_path / "original/source"

    def git(command: list[str], *, cwd: Path) -> str:
        assert cwd == source
        if command[1] == "status":
            return " M tracked.py" if changed == "dirty" else ""
        return "b" * 40 if changed == "commit" else "a" * 40

    monkeypatch.setattr(frozen, "_run", git)
    if changed == "source":
        (source / "scripts/benchmark/rendering/capture.py").write_text("changed\n")
    if changed == "none":
        assert (
            frozen.verify_source_worktree(original, source, "a" * 40)["commit"]
            == "a" * 40
        )
    else:
        with pytest.raises(ValueError):
            frozen.verify_source_worktree(original, source, "a" * 40)


@pytest.mark.parametrize(
    "module_name",
    ["scripts.benchmark.rendering.capture", "embodichain.lab", "isaaclab.scene"],
)
def test_import_origin_rejects_another_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, module_name: str
) -> None:
    wrong = tmp_path / "other/module.py"
    wrong.parent.mkdir()
    wrong.write_text("# wrong origin\n")
    source = {
        "worktree": str(tmp_path / "source"),
        "isaaclab_root": str(tmp_path / "isaac"),
        "source_files": {},
    }
    monkeypatch.setattr(
        frozen.sys, "modules", {module_name: SimpleNamespace(__file__=str(wrong))}
    )
    with pytest.raises(RuntimeError, match="different checkout|nonoriginal"):
        frozen.verify_imported_sources(source)


def test_runtime_rejects_changed_package_or_loaded_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = {
        "runtime_versions": {
            "python": sys.version,
            "packages": {"fixture": "1"},
            "cuda_runtime": "12.8",
        }
    }
    monkeypatch.setattr(frozen.metadata, "version", lambda _: "2")
    with pytest.raises(RuntimeError, match="packages"):
        frozen.runtime_versions(original)
    monkeypatch.setattr(frozen.metadata, "version", lambda _: "1")
    with pytest.raises(RuntimeError, match="cuda_runtime"):
        frozen.runtime_versions(
            original, SimpleNamespace(version=SimpleNamespace(cuda="12.9"))
        )


@pytest.mark.parametrize("value", ["NaN", "Infinity", "1e999"])
def test_nonfinite_proof_json_is_rejected(tmp_path: Path, value: str) -> None:
    path = tmp_path / "input.json"
    path.write_text('{"value":' + value + "}")
    with pytest.raises(ValueError, match="Non-finite"):
        frozen.read_json(path)
