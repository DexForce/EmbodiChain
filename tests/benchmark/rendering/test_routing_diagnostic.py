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

"""Standard-library launcher preparation tests; no runtime or GPU construction."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import pytest

from scripts.benchmark.rendering import routing_diagnostic as launch


def test_preparation_pins_original_interpreter_sources_and_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "original"
    source.mkdir()
    experiment = source / "outputs/experiment"
    experiment.mkdir(parents=True)
    output = tmp_path / "new-proof"
    identity = {"backend": "embodichain", "run_id": "synthetic-run"}
    original = {
        "row": {
            "command": [sys.executable, "-m", "original.worker", "--gpu-id", "5"],
            "renderer": {"synthetic": True},
        }
    }
    verified = {
        "worktree": str(source),
        "commit": "a" * 40,
        "source_files": {"synthetic": "b" * 64},
        "isaaclab_root": None,
    }
    calls = []
    monkeypatch.setattr(
        launch.frozen,
        "load_original_run",
        lambda root, run: (calls.append(("load", root, run)) or original),
    )
    monkeypatch.setattr(
        launch.frozen,
        "verify_source_worktree",
        lambda row, tree, commit: (calls.append(("verify", tree, commit)) or verified),
    )
    monkeypatch.setattr(
        launch.frozen,
        "freeze_original_run",
        lambda row, frozen, destination: (
            identity,
            {"case": {"synthetic_test_fixture": True}},
        ),
    )
    monkeypatch.setattr(
        launch.frozen,
        "renderer_control_signature",
        lambda metadata, backend: {"synthetic": True},
    )
    monkeypatch.setattr(
        launch.frozen,
        "original_environment",
        lambda tree, python, external: {
            "PYTHONPATH": str(tree),
            "FROZEN_PYTHON": str(python),
        },
    )

    def forbid_process(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Preparation must not launch a renderer process")

    monkeypatch.setattr(launch.subprocess, "run", forbid_process)
    result = launch.prepare_routing_diagnostic(
        source, "a" * 40, experiment, "synthetic-run", output
    )
    assert calls == [
        ("load", experiment, "synthetic-run"),
        ("verify", source, "a" * 40),
    ]
    assert result["command"][0] == str(Path(sys.executable).absolute())
    assert Path(result["command"][1]).name == "routing_worker.py"
    assert Path(result["command"][1]).is_absolute()
    assert result["environment"]["PYTHONPATH"] == str(source)
    payload = launch.frozen.read_json(output / "routing_config.json")
    assert payload["source"] == verified
    assert payload["original_binding"] == identity
    assert payload["gpu_id"] == 5
    assert payload["throughput_claim"] is False
    assert len(payload["proof_sources"]) == 4
    for item in payload["proof_sources"]:
        assert launch.frozen.sha256(output / item["snapshot"]["path"]) == item["sha256"]
    assert launch.frozen.read_json(output / "launch.json")["execution_started"] is False


def test_proof_output_cannot_write_into_original_worktree(tmp_path: Path) -> None:
    source = tmp_path / "original"
    with pytest.raises(ValueError, match="outside the frozen original"):
        launch.prepare_routing_diagnostic(
            source, "a" * 40, source / "outputs", "unused", source / "proof"
        )


def test_missing_original_index_is_not_silently_reassigned() -> None:
    with pytest.raises(ValueError, match="GPU index"):
        launch._gpu_index([sys.executable, "-m", "original.worker"])
