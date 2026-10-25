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

"""CPU-only CLI preflight for independent-seed acceptance evidence."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

from examples.learning.nerd._support import _acceptance as nerd_acceptance


@pytest.mark.parametrize(
    ("seeds", "message"),
    [
        ("0,0,0,0,0", "seeds must be distinct"),
        ("0,1,2,3,00", "seeds must be distinct"),
        ("0,1,2,3,invalid", "seeds must be comma-separated integers"),
        ("", "seeds must be comma-separated integers"),
    ],
)
def test_invalid_seed_list_fails_before_output_or_workers(
    tmp_path: Path, seeds: str, message: str
) -> None:
    runner = (
        Path(__file__).resolve().parents[4]
        / "examples/learning/nerd/_support/_acceptance.py"
    )
    output = tmp_path / "results"
    # Missing artifacts must never be inspected or passed to a worker.
    result = subprocess.run(
        [
            sys.executable,
            str(runner),
            "--reference-source",
            str(tmp_path / "missing-reference"),
            "--datasets",
            str(tmp_path / "missing-datasets"),
            "--models",
            str(tmp_path / "missing-models"),
            "--output",
            str(output),
            "--seeds",
            seeds,
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 2
    assert message in result.stderr
    assert not output.exists()


@pytest.mark.parametrize("seeds", [[0], [0, 1, 2, 3], [0, 1, 2, 3, 4]])
def test_unique_seed_lists_preserve_short_and_full_runs(seeds: list[int]) -> None:
    # Short runs remain available; the existing qualification gate returns pending.
    assert nerd_acceptance._parse_seeds(",".join(map(str, seeds))) == seeds
