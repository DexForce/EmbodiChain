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

"""Offline entry points remain usable without the simulation dependency stack."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "arguments", [["--help"], ["dataset", "--help"], ["dataset", "recover", "--help"]]
)
def test_offline_cli_help_without_site_packages(arguments: list[str]) -> None:
    """Dataset command discovery and recovery help need only the standard library."""
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-S", "-m", "embodichain", *arguments],
        env={**os.environ, "PYTHONPATH": str(root)},
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr


def test_offline_packages_do_not_load_simulation_or_training_dependencies() -> None:
    """Importing the offline package cannot start the online engine or GPU stack."""
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            "import sys; import embodichain.data_pipeline.datasets; assert not any(n == 'torch' or n.startswith('embodichain.lab') for n in sys.modules)",
        ],
        env={**os.environ, "PYTHONPATH": str(root)},
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(("operation", "status"), [("validate", 1), ("split", 2)])
def test_dataset_cli_propagates_offline_tool_failure(
    monkeypatch, operation: str, status: int
) -> None:
    """Shell users and CI must observe failed validation or split requests."""
    from types import SimpleNamespace
    from embodichain.cli.dataset import main

    received = []

    def inspect_main(arguments):
        received.extend(arguments)
        return status

    monkeypatch.setitem(
        sys.modules,
        "embodichain.data_pipeline.datasets.inspection",
        SimpleNamespace(main=inspect_main),
    )
    with pytest.raises(SystemExit) as error:
        main([operation, "/dataset"])
    assert error.value.code == status
    assert received == [operation, "/dataset"]
