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


@pytest.mark.parametrize("capture_media", [False, True])
@pytest.mark.parametrize("contact_owner", [None, "environment", "robot"])
def test_contact_acquisition_is_independent_of_media_capture(
    capture_media, contact_owner
):
    from embodichain.lab.expansion.host import _configure_sensor_capture
    from embodichain.lab.sim.sensors import ContactSensorCfg, CameraCfg

    contact = ContactSensorCfg(uid="contacts")
    camera = CameraCfg(uid="camera")
    cfg = SimpleNamespace(
        sensor=[camera] + ([contact] if contact_owner == "environment" else []),
        robot=SimpleNamespace(
            sensors=[camera] + ([contact] if contact_owner == "robot" else [])
        ),
        enable_sensor=True,
        sim_cfg=SimpleNamespace(render_cfg=SimpleNamespace(renderer="no-render")),
    )
    _configure_sensor_capture(cfg, capture_media=capture_media)
    selected = [*cfg.sensor, *cfg.robot.sensors]
    assert (contact in selected) is (contact_owner is not None)
    assert (camera in selected) is capture_media
    assert cfg.enable_sensor is (capture_media or contact_owner is not None)
    assert cfg.sim_cfg.render_cfg.renderer == (
        "hybrid" if capture_media else "no-render"
    )


def test_sampled_collision_rebinds_each_physical_base_after_scene_preparation(
    monkeypatch, tmp_path
):
    from embodichain.lab.expansion import collision
    from embodichain.lab.sim.motion.expansion import real2sim_collision as geometry

    bases = torch.eye(4).repeat(2, 1, 1)
    body = SimpleNamespace(
        uid="box",
        geometry=geometry.BoxCollisionBody("box", (0.1, 0.1, 0.1), np.eye(4)),
        get_local_pose=lambda **kwargs: torch.eye(4).repeat(2, 1, 1),
    )

    class Monitor:
        def __init__(self, *args, **kwargs):
            pass

        def _validate(self, item):
            return item

        def reset(self, env):
            pass

        def predicate(self, env, item):
            return item["entry"]

        def observe(self, env, action):
            pass

        def validation(self, env):
            return None

    monkeypatch.setattr(collision, "MeasuredTaskMonitor", Monitor)
    monkeypatch.setattr(
        geometry.BoxCollisionBody,
        "from_rigid_object",
        classmethod(lambda cls, obj, **kwargs: obj.geometry),
    )
    monkeypatch.setattr(
        collision,
        "load_config",
        lambda path: {
            "kind": "sampled_rigid_v1",
            "initial_checks": [{"entry": True}],
            "final_checks": [{"entry": False}],
            "events": [],
            "allowed_contacts": [],
        },
    )
    urdf = tmp_path / "robot.urdf"
    urdf.write_text(
        '<robot name="base_relocation"><link name="base"/><link name="tool"><collision><geometry><box size="0.2 0.2 0.2"/></geometry></collision></link><joint name="joint" type="prismatic"><parent link="base"/><child link="tool"/><axis xyz="1 0 0"/><limit lower="-2" upper="2" effort="1" velocity="1"/></joint></robot>'
    )
    bases[:, 2, 3] = 1.0
    object_pose = torch.eye(4).repeat(2, 1, 1)
    object_pose[:, 0, 3] = 2.0
    object_pose[:, 2, 3] = 1.0
    body.get_local_pose = lambda **kwargs: object_pose
    env = SimpleNamespace(
        num_envs=2,
        robot=SimpleNamespace(
            cfg=SimpleNamespace(fpath=str(urdf)),
            joint_names=("joint",),
            get_local_pose=lambda **kwargs: bases,
            get_qpos=lambda: torch.zeros(2, 1),
        ),
        sim=SimpleNamespace(
            get_rigid_object_uid_list=lambda: ("box",),
            get_rigid_object=lambda uid: body,
            get_articulation_uid_list=lambda: (),
            get_deformable_object_uid_list=lambda: (),
            get_rigid_object_group_uid_list=lambda: (),
        ),
    )
    adapter = collision._SampledRigidCollision(env, tmp_path / "profile.yaml", tmp_path)
    for positions in ((2.0, 4.0), (4.0, 2.0)):
        bases[:, 0, 3] = torch.tensor(positions)
        adapter.prepare()
        adapter.sample((0, 1), None)
        assert tuple(adapter.results[row].status for row in (0, 1)) == tuple(
            "failed" if x == 2 else "passed" for x in positions
        )
