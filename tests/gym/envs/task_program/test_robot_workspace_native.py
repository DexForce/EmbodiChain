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

"""Native qualification of reusable scene services and their Task Program consumer."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[4]


@pytest.mark.requires_sim
@pytest.mark.subprocess_sim
@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.parametrize("mode", ["scene", "task"])
def test_native_workspace_modes_use_real_samples_and_measured_state(
    tmp_path: Path, mode: str
) -> None:
    output = tmp_path / mode
    command = (
        "import sys; from embodichain.lab.scripts.robot_workspace import run_demo; "
        "run_demo(mode=sys.argv[1], view='none', seed=7, output=sys.argv[2])"
    )
    child_env = dict(os.environ)
    child_env["PYTHONPATH"] = os.pathsep.join(
        (str(_ROOT), child_env.get("PYTHONPATH", ""))
    )
    child_env["EMBODICHAIN_SIM_EXIT_PROCESS"] = "0"
    completed = subprocess.run(
        [sys.executable, "-c", command, mode, str(output)],
        cwd=_ROOT,
        env=child_env,
        text=True,
        capture_output=True,
        timeout=240,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = json.loads((output / "report.json").read_text())
    assert report["accepted"]
    assert len(report["candidates"]) == 3
    assert report["initialization_order"] == [3, 0, 1, 2]
    assert report["layout"]["table_shape"] == "round"
    assert report["workspace"]["requested_cache_samples"] == 8192
    reachability = report["workspace"]["reachability"]
    with np.load(reachability["cache"], allow_pickle=False) as cloud:
        valid = cloud["reachable"]
        assert valid.any() and (~valid).any()
        assert valid[reachability["candidate_indices"]].all()
        assert not valid[reachability["negative_index"]]
        assert np.all(cloud["colors"][valid] == (40, 220, 100))
        assert np.all(cloud["colors"][~valid] == (245, 65, 65))
        np.testing.assert_allclose(cloud["points"][:, 2], 0.775)
        corners = np.array(
            [[-0.025, -0.025], [-0.025, 0.025], [0.025, -0.025], [0.025, 0.025]]
        )
        offsets = (
            cloud["points"][:, None, :2]
            + corners
            - np.array(report["layout"]["table_center_xy"])
        )
        assert (
            np.linalg.norm(offsets, axis=-1) <= report["layout"]["table_radius_m"]
        ).all()
    with np.load(report["workspace"]["cache"], allow_pickle=False) as cache:
        assert cache["joint_configurations"].shape == (
            report["workspace"]["cache_samples"],
            6,
        )
        points = cache["workspace_points"]
        assert np.isfinite(points).all() and np.ptp(points, axis=0).min() > 0.1
    centers = np.asarray([row["adjusted_pose"] for row in report["candidates"]])[
        :, :3, 3
    ]
    assert np.all(centers[:, 0] >= -0.55) and np.all(centers[:, 0] <= -0.30)
    assert np.all(centers[:, 1] >= -0.18) and np.all(centers[:, 1] <= 0.12)
    for index in range(3):
        for other in range(index):
            assert np.linalg.norm(centers[index, :2] - centers[other, :2]) >= 0.05
    if mode == "scene":
        assert (
            not report["scene_imported_task_program"]
            and not report["scene_imported_gym"]
        )
        assert all(row["initial_validation"]["passed"] for row in report["candidates"])
        assert all(check["passed"] for check in report["restoration"])
        checks = {
            check["name"]: check["status"] for check in report["negative"]["checks"]
        }
    else:
        accepted = [row for row in report["candidates"] if row["scene"]["accepted"]]
        assert len(accepted) == report["successful_trials"] >= 1
        assert report["task_grasp_provider_calls"] >= len(accepted)
        for index, row in enumerate(report["candidates"]):
            if row["execution"] is not None:
                assert row["execution"]["episode_index"] == index
                assert row["execution"]["attempt_id"] == index
        for row in accepted:
            assert row["measured_lift_m"] >= 0.05
            assert row["stable_control_steps"] >= 10
            assert row["first_frame_matches_initial_state"]
            assert (output / row["trajectory"]).stat().st_size > 0
        assert all(row["restoration"]["passed"] for row in report["candidates"])
        assert (
            not report["negative"]["accepted"]
            and report["negative_executed_steps"] == 0
        )
        checks = {
            check["check_id"]: check["status"]
            for check in report["negative"]["initial_validation"]
        }
    assert (
        checks["scene.support"] == "passed" and checks["workspace.pose_ik"] == "failed"
    )
