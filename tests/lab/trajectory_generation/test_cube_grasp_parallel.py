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

"""Physical cube-pickup augmentation, sustained-grasp checks, and video evidence."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


def test_grasp_results_reject_fallen_slipping_and_distant_objects() -> None:
    from examples.sim.motion.trajectory_generation.cube_grasp_parallel import (
        _grasp_results,
    )

    cubes = np.tile(np.eye(4), (4, 6, 1, 1))
    cubes[:, 0, 2, 3] = 0.325
    cubes[:, 1:, 2, 3] = 0.505
    tcps = cubes.copy()
    cubes[1, -1, 2, 3] = 0.325  # Fell after initially reaching the lift target.
    cubes[2, -1, 0, 3] += 0.02  # Slipped relative to the TCP during the hold.
    cubes[3, 1:, 0, 3] += 0.1  # Elevated, but never near the gripper.
    results = _grasp_results(cubes, tcps, hold_start=3)
    assert [result["success"] for result in results] == [True, False, False, False]


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.requires_sim
def test_parallel_cube_pickups_preserve_contact_commands_and_save_synchronized_video(
    tmp_path: Path,
) -> None:
    av = pytest.importorskip("av")
    root = Path(__file__).resolve().parents[3]
    output = tmp_path / "parallel"
    # Run the real entry point in a child process to exercise native cleanup
    # and the exit status as well as the physics and recorder.
    process = subprocess.run(
        [
            sys.executable,
            "examples/sim/motion/trajectory_generation/cube_grasp_parallel.py",
            "--output",
            str(output),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert process.returncode == 0, (process.stdout + process.stderr)[-8000:]
    report = json.loads((output / "report.json").read_text())
    assert report["success"]
    assert [row["yaw_degrees"] for row in report["results"]] == [0, 0, 90, 90]
    assert all(
        row["success"] and row["min_hold_lift_m"] >= 0.12 for row in report["results"]
    )
    assert min(report["max_path_pair_separation_m"]) > 0.03
    with np.load(output / "rollout.npz") as saved:
        targets = saved["commanded_qpos"]
        reference = saved["reference_qpos"]
        transit_stop = report["transit_stop"]
        np.testing.assert_array_equal(
            targets[:, transit_stop - 1 : reference.shape[1]],
            reference[:, transit_stop - 1 :],
        )
        assert not np.array_equal(
            targets[1, 1 : transit_stop - 1], reference[1, 1 : transit_stop - 1]
        )
        assert targets.shape == saved["measured_qpos"].shape
        np.testing.assert_allclose(
            saved["measured_qpos"][:, 0],
            np.repeat(saved["measured_qpos"][0:1, 0], 4, axis=0),
            atol=1e-6,
            rtol=0,
        )
        np.testing.assert_allclose(
            saved["cube_poses"][:, 0],
            np.repeat(saved["cube_poses"][0:1, 0], 4, axis=0),
            atol=1e-5,
            rtol=0,
        )
        # The quarter-turn changes the closing orientation while preserving
        # the centered grasp position and downward approach direction.
        grasps = saved["grasp_poses"]
        np.testing.assert_allclose(
            grasps[:, :3, 3], np.repeat(grasps[0:1, :3, 3], 4, axis=0), atol=1e-6
        )
        assert abs(np.dot(grasps[0, :3, 0], grasps[2, :3, 0])) < 1e-5
        actual_rotations = saved["tcp_poses"][:, -1, :3, :3]
        assert abs(np.dot(actual_rotations[0, :, 0], actual_rotations[2, :, 0])) < 0.05
        times = saved["timestamps"]
        np.testing.assert_allclose(
            times, np.arange(len(times)) / report["fps"], atol=1e-7
        )
    with av.open(str(output / "preview.mp4")) as video:
        stream = video.streams.video[0]
        assert stream.codec_context.name == "h264"
        assert (stream.width, stream.height) == (1280, 1056)
        assert stream.average_rate == report["fps"] == 20
        frames = list(video.decode(video=0))
        assert len(frames) == report["frame_count"] == len(times)
        np.testing.assert_allclose(
            [float(frame.pts * frame.time_base) for frame in frames], times, atol=1e-7
        )
        first = frames[0].to_ndarray(format="rgb24")
        last = frames[-1].to_ndarray(format="rgb24")
        for row in range(4):
            top = (row // 2) * 528 + 48  # Exclude the changing text header.
            left = (row % 2) * 640
            initial = first[top : top + 480, left : left + 640].astype(float)
            final = last[top : top + 480, left : left + 640].astype(float)
            assert initial.std() > 1
            assert np.abs(initial - final).mean() > 1
