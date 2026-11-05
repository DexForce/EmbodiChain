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

"""Source-exact distal mesh capture and numeric staging translations."""

from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pytest
from embodichain.lab.sim.scene.robot_distal import capture_robot_distal_meshes
from embodichain.compute.geometry.staging import distal_staging_variants
from ._robot_distal_fixture import distal_fixture


def measured(snapshot):
    return {
        "active": snapshot.active,
        "passive": snapshot.passive,
        "tcp": {k: v.tolist() for k, v in snapshot.tcp.items()},
        "roots": {k: v.tolist() for k, v in snapshot.roots.items()},
        "points": {k: v.tolist() for k, v in snapshot.points.items()},
        "hand_owner_fk_error": dict(snapshot.hand_owner_fk_error),
    }


@pytest.mark.parametrize(
    "mode", ["nominal", "hand_offset", "calibration", "changed", "ambiguous"]
)
def test_distal_capture_and_staging_keep_source_mesh_order_frames_and_failures(
    tmp_path, mode
):
    robot, context, request = distal_fixture(tmp_path, mode)
    expected = json.loads(
        (
            Path(__file__).parent / "data" / f"robot_distal_{mode}_reference.json"
        ).read_text()
    )
    if "unavailable" in expected:
        with pytest.raises(ValueError, match=expected["unavailable"]):
            capture_robot_distal_meshes(
                robot,
                expected_qpos=context.robot.qpos,
                active_control_part="left",
                expected_root_pose=context.robot.root_pose,
            )
        return
    snapshot = capture_robot_distal_meshes(
        robot,
        expected_qpos=context.robot.qpos,
        active_control_part="left",
        expected_root_pose=context.robot.root_pose,
    )
    assert measured(snapshot) == expected["capture"]
    assert all(not points.flags.writeable for points in snapshot.points.values())
    variants = distal_staging_variants(
        request.goal.object_target_pose,
        context.scene.entities["object"].pose,
        context.task.get_held_object("left").object_to_eef[0].numpy(),
        active=snapshot.active,
        passive=snapshot.passive,
        tcp=snapshot.tcp,
        roots=snapshot.roots,
        points=snapshot.points,
        hand_owner_fk_error=snapshot.hand_owner_fk_error,
    )
    assert [
        [label, None if poses is None else poses.tolist(), report]
        for label, poses, report in variants
    ] == expected["variants"]
