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

"""Exercise immutable reference provenance and paired acceptance failure gates."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.fixture
def acceptance():
    source = (
        Path(__file__).resolve().parents[3]
        / "examples/learning/rigidformer/_support/_acceptance.py"
    )
    spec = importlib.util.spec_from_file_location("rigidformer_acceptance_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def reference_sources(tmp_path, acceptance, monkeypatch):
    checkout = tmp_path / "reference"
    installed = tmp_path / "installed"
    files = {
        "examples/box2d_dataset.py": "# pinned dataset\n",
        "examples/train_box2d.py": "# pinned training recipe\n",
        "rigidformer/__init__.py": "# pinned package exports\n",
        "rigidformer/rigidformer.py": "# pinned model\n",
        "rigidformer/rotary_3d.py": "# pinned model helper\n",
    }
    for name, source in files.items():
        path = checkout / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
        if name.startswith("rigidformer/"):
            destination = installed / Path(name).relative_to("rigidformer")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(source)
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    subprocess.run(["git", "-C", str(checkout), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(checkout),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "Pinned fixture",
        ],
        check=True,
    )
    revision = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    monkeypatch.setattr(acceptance, "RIGIDFORMER_REVISION", revision)
    assert set(acceptance._verify_sources(checkout, installed)) == set(files)
    return checkout, installed


@pytest.mark.parametrize(
    "name",
    [
        "examples/box2d_dataset.py",
        "examples/train_box2d.py",
        "rigidformer/rigidformer.py",
        "rigidformer/rotary_3d.py",
        "rigidformer/__init__.py",
    ],
)
def test_reference_rejects_dirty_sources_even_when_head_and_installed_bytes_match(
    acceptance, reference_sources, name
):
    checkout, installed = reference_sources
    (checkout / name).write_text("# changed after commit\n")
    if name.startswith("rigidformer/"):
        (installed / Path(name).relative_to("rigidformer")).write_text(
            "# changed after commit\n"
        )
    with pytest.raises(
        ValueError, match="Reference source differs from pinned Git blob"
    ):
        acceptance._verify_sources(checkout, installed)


def test_reference_rejects_installed_model_divergence(acceptance, reference_sources):
    checkout, installed = reference_sources
    (installed / "rigidformer.py").write_text("# installed model changed\n")
    with pytest.raises(
        ValueError, match="Installed source differs from pinned Git blob"
    ):
        acceptance._verify_sources(checkout, installed)


def test_loss_curve_gate_checks_every_update_and_complete_finite_history(acceptance):
    assert acceptance._loss_curve_gate([1.0, 2.0, 3.0], [1.0, 2.0001, 3.0], steps=3)[
        "passed"
    ]
    changed = acceptance._loss_curve_gate([1.0, 2.0, 3.0], [1.0, 2.1, 3.0], steps=3)
    assert not changed["passed"]
    assert changed["first_mismatched_step"] == 2
    assert changed["mismatched_steps"] == 1
    assert not acceptance._loss_curve_gate([1.0, 2.0], [1.0, 2.0], steps=3)["passed"]
    assert not acceptance._loss_curve_gate(
        [1.0, float("nan")], [1.0, float("nan")], steps=2
    )["passed"]


@pytest.mark.parametrize(
    "failure",
    [
        RuntimeError("device setup failed"),
        importlib.metadata.PackageNotFoundError("rigidformer"),
        ValueError("Expected reference revision"),
        SystemExit("Acceptance incomplete or failed; inspect summary.json"),
    ],
)
def test_acceptance_records_setup_and_qualification_failures_after_fresh_admission(
    acceptance, monkeypatch, tmp_path, failure
):
    output = tmp_path / "fresh"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "acceptance.py",
            "--reference",
            str(tmp_path / "reference"),
            "--output",
            str(output),
        ],
    )

    def fail(args):
        raise failure

    monkeypatch.setattr(acceptance, "_run", fail)
    with pytest.raises(type(failure)):
        acceptance._main()
    recorded = json.loads((output / "failure.json").read_text())
    assert recorded == {"type": type(failure).__name__, "message": str(failure)}
    (output / "failure.json").write_text("existing result")
    with pytest.raises(FileExistsError):
        acceptance._main()
    assert (output / "failure.json").read_text() == "existing result"
