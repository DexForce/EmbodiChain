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

"""Measured UR5 acceptance and articulated Panda rejection through the real stack."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.requires_sim
@pytest.mark.parametrize("robot_type, accepted", [("ur5", True), ("panda", False)])
def test_real_runner_accepts_only_validated_measured_free_motion(
    tmp_path: Path, robot_type: str, accepted: bool
) -> None:
    pytest.importorskip("curobo")
    pytest.importorskip("lerobot")
    av = pytest.importorskip("av")
    from examples.sim.motion.trajectory_generation.free_motion import run_free_motion

    output = tmp_path / "collection"
    report = run_free_motion(output, robot_type=robot_type, record_video=accepted)
    assert report["counts"]["rollout_attempted"] == 1
    assert report["pending_reserved_bytes"] == 0
    assert not (output / ".writer.lock").exists()
    if not accepted:
        assert not report["target_reached"]
        assert report["counts"]["committed"] == 0
        assert not (output / "manifest.json").exists()
        assert not (output / "preview.mp4").exists()
        diagnostics = next(iter(report["diagnostics"].values()))
        checks = {
            check.check_id: check for check in diagnostics["rollout_validation"].checks
        }
        assert checks["path_collision"].status == "unavailable"
        assert checks["task_success"].metrics["max_gripper_drift"] > 1e-6
        return
    assert report["target_reached"], report["diagnostics"]
    assert report["counts"]["committed"] == report["counts"]["rollout_attempted"] == 1
    assert report["pending_reserved_bytes"] == 0
    assert not (output / ".writer.lock").exists()
    manifest = json.loads((output / "manifest.json").read_text())
    assert len(manifest["episodes"]) == 1
    shard = output / manifest["episodes"][0]["shard"]
    evidence = json.loads((shard / "episode.json").read_text())
    checks = {check["check_id"]: check for check in evidence["validation"]}
    assert {
        "planned_path_collision",
        "path_collision",
        "execution_complete",
        "fixed_collision_world",
        "task_success",
        "actual_motion_limits",
        "motion_quality",
    } <= set(checks)
    assert all(check["status"] == "passed" for check in checks.values())
    assert checks["task_success"]["metrics"]["measured_displacement"] >= 0.07
    assert checks["task_success"]["metrics"]["endpoint_error"] <= 0.01
    assert list((shard / "dataset").rglob("*.parquet"))

    # Rendering must retain every observation, including t=0 and the terminal
    # state, without advancing the one-second physics rollout.
    assert checks["motion_quality"]["metrics"]["duration_s"] == pytest.approx(1.0)
    with np.load(shard / "terminal.npz") as terminal:
        timestamps = terminal["timestamps"]
    with av.open(str(output / "preview.mp4")) as video:
        stream = video.streams.video[0]
        assert stream.codec_context.name == "h264"
        assert (stream.width, stream.height) == (640, 480)
        assert stream.average_rate == evidence["fps"] == 20
        frames = list(video.decode(video=0))
        assert len(frames) == len(timestamps) == evidence["steps"] + 1
        np.testing.assert_allclose(
            [float(frame.pts * frame.time_base) for frame in frames],
            timestamps - timestamps[0],
            rtol=0,
            atol=1e-6,
        )
        first = frames[0].to_ndarray(format="rgb24")
        last = frames[-1].to_ndarray(format="rgb24")
        assert first.std() > 1, "Camera preview must contain a rendered scene"
        assert not np.array_equal(first, last), "Preview must show changing frames"
