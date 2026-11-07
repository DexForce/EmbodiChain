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

"""Control-boundary snapshots avoid cross-row repeated reads and retain owned values."""

from __future__ import annotations
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from embodichain.lab.expansion.visual import _calibration_batch, _calibrated_frames
from embodichain.lab.expansion.collision import _measured_frame
from embodichain.lab.sim.scene.state import SceneState, EntityState
from embodichain.lab.task_program.integrations.feasibility import MeasuredTaskMonitor


@pytest.mark.parametrize("count", [1, 8, 32])
def test_camera_calibration_reads_one_full_batch_and_owns_values(count):
    calls = {"intrinsics": 0, "poses": 0}
    intrinsics = torch.eye(3).repeat(count, 1, 1)
    intrinsics[:, 0, 0] = torch.arange(count) + 100
    poses = torch.eye(4).repeat(count, 1, 1)

    def read_intrinsics():
        calls["intrinsics"] += 1
        return intrinsics

    def read_poses(**kwargs):
        calls["poses"] += 1
        return poses

    camera = SimpleNamespace(
        cfg=SimpleNamespace(width=2, height=2),
        get_intrinsics=read_intrinsics,
        get_arena_pose=read_poses,
    )
    env = SimpleNamespace(
        num_envs=count,
        sim=SimpleNamespace(
            get_sensor_uid_list=lambda: ["camera"], get_sensor=lambda _: camera
        ),
    )
    batch = _calibration_batch(env)
    intrinsics.zero_()
    observation = {
        "sensor": {"camera": {"rgb": torch.zeros(count, 2, 2, 3, dtype=torch.uint8)}}
    }
    for row in range(count):
        frames = _calibrated_frames(env, observation, row, calibration=batch[row])
        assert frames["calibration/camera/intrinsics"][0, 0] == row + 100
    assert calls == {"intrinsics": 1, "poses": 1}


def test_measured_predicates_use_the_frozen_row_snapshot_without_live_reads():
    robot = SimpleNamespace(uid="arm", joint_names=("finger",))
    env = SimpleNamespace(num_envs=2, robot=robot)
    batches = []
    for index in range(2):
        body = EntityState(
            "rigid_object",
            "cube",
            "dynamic",
            (),
            (float(index), 0.0, 0.1, 0.0, 0.0, 0.0, 1.0),
            (0.0,) * 6,
        )
        arm = EntityState(
            "robot",
            "arm",
            "",
            ("finger",),
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            (0.0,) * 6,
            qpos=(float(index),),
            qvel=(0.0,),
            target_qpos=(0.0,),
            target_qvel=(0.0,),
            qf=(0.0,),
        )
        batches.append((body, arm))
    snapshot = SceneState("default", (0, 1), tuple(batches), object(), ())
    frame = _measured_frame(env, snapshot, {"cube": ((), ())})
    checks = (
        {"kind": "joint_range", "joint": "finger", "lower": 0.9, "upper": 1.1},
        {
            "kind": "pose_near",
            "entity": "cube",
            "position": [1.0, 0.0, 0.1],
            "tolerance_m": 0.01,
        },
    )
    results = []
    for row in (0, 1):
        monitor = MeasuredTaskMonitor(checks, (), env_id=row)
        monitor.reset(frame)
        results.append(monitor.validation(frame).accepted)
    assert results == [False, True]
