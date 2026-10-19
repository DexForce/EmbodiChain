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

"""Real CLI smoke coverage for single- and dual-arm Cartesian preparation."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

pytestmark = [pytest.mark.gpu, pytest.mark.slow, pytest.mark.subprocess_sim]


@pytest.mark.parametrize("robot", ["ur5", "franka"])
@pytest.mark.parametrize("skill", ["place", "handover"])
def test_atomic_cli_plans_and_replays_nominal_scene(
    tmp_path: Path, robot: str, skill: str
) -> None:
    """Each preset produces finite, timed commands and finishes native cleanup."""
    output = tmp_path / f"{skill}_{robot}"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "embodichain",
            "run-atomic-action",
            "--skill",
            skill,
            "--physics",
            "default",
            "--robot",
            robot,
            "--planner",
            "trapezoidal",
            "--headless",
            "--num-envs",
            "1",
            "--no-expansion",
            "--no-save-video",
            "--output-dir",
            str(output),
        ],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout[-20000:] + result.stderr[-20000:]
    manifest = json.loads((output / "expansion_manifest.json").read_text())
    assert manifest["status"] == "executed"
    assert manifest["recipe_indices"] == [0]
    assert manifest["replays"]
    for replay in manifest["replays"]:
        assert len(replay["rows"]) == 1
        trajectory = torch.load(replay["trajectory_path"], weights_only=True)
        positions, dt = trajectory["positions"], trajectory["dt"]
        assert positions.shape[0] == 1
        assert positions.shape[1] > 1
        assert dt.shape == positions.shape[:2]
        assert torch.isfinite(positions).all()
        assert torch.isfinite(dt).all()
        assert (dt >= 0).all()
        assert (dt.sum(dim=1) > 0).all()
