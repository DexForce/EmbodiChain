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

"""End-to-end parallel PickUp collection with physical contacts and sealed data."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.requires_sim
@pytest.mark.subprocess_sim
def test_pickup_collection(tmp_path: Path) -> None:
    av = pytest.importorskip("av")
    import pyarrow.parquet as pq

    root = Path(__file__).resolve().parents[3]
    output = tmp_path / "collection"
    process = subprocess.run(
        [
            sys.executable,
            "examples/sim/motion/trajectory_generation/cube_pickup_collection.py",
            "--output",
            str(output),
            "--episodes",
            "8",
            "--record-video",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=480,
    )
    assert process.returncode == 0, process.stdout[-4000:] + process.stderr[-4000:]
    report = json.loads((output / "generation_report.json").read_text())
    pickup = json.loads((output / "pickup_report.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    assert report["target_reached"] and report["error"] is None
    assert report["counts"]["committed"] == len(manifest["episodes"]) == 8
    assert pickup["prepared_batches"] >= 2
    assert report["configuration"]["source"]["kind"] == "atomic"
    initial_joints, initial_objects, cases, closing_axes, paths = [], [], set(), {}, {}
    per_case_initial = {}
    for record in manifest["episodes"]:
        shard = output / record["shard"]
        evidence = json.loads((shard / "episode.json").read_text())
        case = evidence["identity"]["scene_case_id"]
        cases.add(case)
        checks = {check["check_id"]: check for check in evidence["validation"]}
        assert all(check["status"] == "passed" for check in checks.values())
        assert {
            "planned_path_collision",
            "path_collision",
            "physical_contacts",
            "held_object_stability",
            "actual_motion_limits",
            "task_success",
            "fixed_collision_world",
        } <= checks.keys()
        metrics = checks["held_object_stability"]["metrics"]
        assert metrics["minimum_lift_m"] >= 0.12
        assert (
            min(
                metrics["finger_0_contact_fraction"],
                metrics["finger_1_contact_fraction"],
            )
            >= 0.95
        )
        assert metrics["max_relative_translation_m"] <= 0.01
        assert metrics["hold_seconds"] >= 1.0
        assert checks["physical_contacts"]["metrics"]["physics_samples"] == 2710
        assert evidence["observation_shapes"]["object_pose"] == [4, 4]
        assert len(evidence["metadata"]["commanded_joint_indices"]) == 7
        assert len(evidence["metadata"]["joint_names"]) == 8
        files = sorted((shard / "dataset" / "data").rglob("*.parquet"))
        table = pq.read_table(files)
        assert table.num_rows == 271
        joints = np.array(table["observation.joint_positions"].to_pylist())
        objects = np.array(table["observation.object_pose"].to_pylist()).reshape(
            -1, 4, 4
        )
        tcp = np.array(table["observation.tcp_pose"].to_pylist()).reshape(-1, 4, 4)
        initial_joints.append(joints[0])
        initial_objects.append(objects[0])
        if case in per_case_initial:
            np.testing.assert_allclose(
                objects[0], per_case_initial[case], atol=1e-6, rtol=0
            )
        else:
            per_case_initial[case] = objects[0]
        paths.setdefault(case, []).append(tcp[:81, :3, 3])
        with np.load(shard / "terminal.npz", allow_pickle=False) as terminal:
            np.testing.assert_allclose(
                np.diff(terminal["timestamps"]), 0.05, atol=1e-8, rtol=0
            )
            assert terminal["timestamps"].shape == (272,)
            assert terminal["observation.object_pose"][2, 3] - objects[0, 2, 3] >= 0.12
            closing_axes[case] = terminal["observation.tcp_pose"][:3, 0]
    assert cases == {f"cube_grasp_row_{row}" for row in range(4)}
    np.testing.assert_allclose(
        initial_joints, np.broadcast_to(initial_joints[0], (8, 8)), atol=1e-6, rtol=0
    )
    np.testing.assert_allclose(
        initial_objects,
        np.broadcast_to(initial_objects[0], (8, 4, 4)),
        atol=1e-5,
        rtol=0,
    )
    assert (
        abs(np.dot(closing_axes["cube_grasp_row_0"], closing_axes["cube_grasp_row_2"]))
        < 0.1
    )
    assert any(
        len(values) >= 2 and np.linalg.norm(values[0] - values[1], axis=-1).max() > 0.01
        for values in paths.values()
    )
    with av.open(str(output / "preview.mp4")) as video:
        stream = video.streams.video[0]
        assert stream.codec_context.name == "h264"
        assert (stream.width, stream.height) == (1280, 1056)
        assert float(stream.average_rate) == 20.0
        frames = sum(1 for _ in video.decode(stream))
        assert frames == pickup["video_frames"] == pickup["prepared_batches"] * 272


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.requires_sim
@pytest.mark.subprocess_sim
def test_pickup_cli_releases_native_scene_on_integration_failure(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[3]
    script = """
import sys
from examples.sim.motion.trajectory_generation.cube_pickup_collection import main
from embodichain.lab.trajectory_generation.integrations.contact import PickUpMotionValidator

def refuse(self, *args, **kwargs):
    raise ValueError("injected contact backend failure")

PickUpMotionValidator.__init__ = refuse
sys.argv = ["cube_pickup_collection.py", "--output", sys.argv[1], "--episodes", "1"]
main()
"""
    process = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "failed")],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert process.returncode == 1, process.stdout[-2000:] + process.stderr[-2000:]
    assert "injected contact backend failure" in process.stderr
    assert not (tmp_path / "failed" / "manifest.json").exists()
