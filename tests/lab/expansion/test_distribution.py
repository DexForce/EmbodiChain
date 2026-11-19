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

from __future__ import annotations
import numpy as np
import pytest
from embodichain.lab.expansion._distribution import _motion, _summarize


def test_installation_noise_does_not_split_domain_or_modify_raw_poses():
    from embodichain.lab.expansion._distribution import _installation_pose

    measured = [2.3283064365386963e-10, -3.725290298461914e-9, 0, 0, 0, 0, -1]
    raw = list(measured)
    assert _installation_pose(measured) == _installation_pose([0, 0, 0, 0, 0, 0, 1])
    assert measured == raw
    assert _installation_pose([0.01, 0, 0, 0, 0, 0, 1]) != _installation_pose(measured)
    assert _installation_pose([0, 0, 0, 0, 0, -1, -1e-10]) == _installation_pose(
        [0, 0, 0, 0, 0, 1, 1e-10]
    )
    with pytest.raises(ValueError, match="nonzero"):
        _installation_pose([0] * 7)


def test_motion_density_and_time_are_separate():
    a = _motion(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), np.array([0.0, 2.0]))
    b = _motion(
        np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        np.array([0.0, 2.0, 4.0]),
    )
    assert a["path_length"] == b["path_length"] == 1
    assert a["duration_s"] == 2 and b["duration_s"] == 4
    np.testing.assert_allclose(a["path"], b["path"])
    with pytest.raises(ValueError):
        _motion(np.zeros((2, 3)), np.array([1.0, 1.0]))


def test_joint_distribution_and_idempotency():
    rows = [
        dict(
            commit_id=str(i),
            initial_pose=[x, 0, 0],
            final_pose=[x, 0, 0],
            object_motion={},
            geometry_bucket=None,
        )
        for i, x in enumerate([-0.15, 0.15])
    ]
    r = _summarize(rows + rows, 0.1, 1)
    assert r["count"] == 2
    assert r["joint"] == {"-2,0|-2,0": 1, "1,0|1,0": 1}
    assert r["joint_at_minimum"] == 2
    with pytest.raises(ValueError):
        _summarize([rows[0], dict(rows[0], final_pose=[9, 0, 0])], 0.1, 1)


def test_missing_layout_is_not_zero():
    r = _summarize(
        [dict(commit_id="a", object_motion=None, geometry_bucket=None)], 0.1, 5
    )
    assert r["layout_missing"] == 1 and r["layout_valid"] == 0
    assert r["joint"] == {}


@pytest.mark.parametrize(
    ("tasks", "source_uuids", "matches"),
    [
        (("program-a", "program-b"), ("source-a", "source-b"), False),
        (("program-a", "program-a"), ("source-a", "source-b"), True),
        ((None, None), ("source-a", "source-b"), False),
        ((None, None), ("source-a", "source-a"), True),
    ],
)
def test_report_requires_matching_recorded_task_identity(
    tmp_path, monkeypatch, tasks, source_uuids, matches
):
    from types import SimpleNamespace
    import torch
    from embodichain.lab.expansion import report_expansion_distribution
    from embodichain.lab.expansion import persistence

    jobs = [tmp_path / name for name in ("before", "after")]
    episodes = {}
    for job, task_id, source_uuid in zip(jobs, tasks, source_uuids, strict=True):
        job.mkdir()
        (job / "job.json").write_text("{}")
        metadata = {"instruction": "Pick and place", "source_episode_uuid": source_uuid}
        if task_id is not None:
            metadata["scene_case"] = {"task_id": task_id}
        episodes[job.name] = SimpleNamespace(
            episode_id=f"episode-{job.name}",
            commit_id=f"commit-{job.name}",
            metadata=metadata,
            timestamps=torch.tensor([0.0, 1.0]),
            observations={},
            validation=SimpleNamespace(accepted=True),
        )
    monkeypatch.setattr(
        persistence.ArtifactEpisodeSink,
        "inspect",
        lambda self: (SimpleNamespace(confirmed=True, storage_id=self.directory.name),),
    )
    monkeypatch.setattr(persistence, "load_expansion_episode", episodes.__getitem__)
    output = tmp_path / "report"
    if matches:
        report = report_expansion_distribution(jobs[:1], jobs[1:], output, entity="mug")
        assert (report["before"], report["after"]) == (1, 1)
    else:
        with pytest.raises(ValueError, match="matching domain"):
            report_expansion_distribution(jobs[:1], jobs[1:], output, entity="mug")
        assert not output.exists()


def test_orientation_sign_and_out_of_plane_rotation():
    from embodichain.lab.expansion._distribution_pose import orientation

    poses = np.array([[0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0]], dtype=float)
    a = orientation(poses, np.array([0.0, 2.0]))
    poses[:, 3:] *= -1
    b = orientation(poses, np.array([0.0, 2.0]))
    assert a == b
    assert a["angular_path_rad"] == pytest.approx(np.pi)
    assert a["angular_speed_p50_p95"] == pytest.approx([np.pi / 2] * 2)
    assert a["initial_rotvec_rad"] == [0.0, 0.0, 0.0]
    assert abs(a["final_rotvec_rad"][0]) == pytest.approx(np.pi)
    poses[:, 3:] = 0
    with pytest.raises(ValueError):
        orientation(poses, np.array([0.0, 2.0]))


def test_tcp_root_transform_and_calibration(tmp_path):
    import torch
    from types import SimpleNamespace
    from embodichain.lab.expansion._distribution_pose import RecordedTCP

    asset = tmp_path / "test.urdf"
    asset.write_text(
        """<robot name="test"><link name="base"/><link name="tool"/><joint name="slide" type="prismatic"><parent link="base"/><child link="tool"/><axis xyz="1 0 0"/><limit lower="0" upper="2" effort="1" velocity="1"/></joint></robot>"""
    )
    offset = np.eye(4)
    offset[2, 3] = 0.15
    tcp = RecordedTCP(
        {
            "robot": {
                "fpath": str(asset),
                "gripper_bindings": {
                    "eef": {
                        "root_link": "tool",
                        "defaults": {"default_root_from_tcp": offset.tolist()},
                    }
                },
            }
        }
    )
    e = SimpleNamespace(
        metadata={
            "initial_state": [{"kind": "robot", "uid": "r", "joint_names": ["slide"]}],
            "scene_state_fields": [
                {"entity_uid": "r", "field": "pose", "key": "root"},
                {"entity_uid": "r", "field": "qpos", "key": "q"},
            ],
        },
        observations={
            "root": torch.tensor([[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]] * 2),
            "q": torch.tensor([[0.0], [1.0]]),
        },
    )
    p = tcp.poses(e)
    np.testing.assert_allclose(p[:, :3], [[1, 2, 3.15], [2, 2, 3.15]])
    np.testing.assert_allclose(p[:, 3:], [[0, 0, 0, 1]] * 2)


def test_detail_pages_match_agent_records_and_summary(tmp_path, monkeypatch):
    import json
    from embodichain.lab.expansion import _distribution as module

    rows = []
    for i in range(101):
        rows.append(
            dict(
                episode_id=f"e{i}",
                commit_id=f"c{i}",
                initial_pose=[0, 0, 0, 0, 0, 0, 1],
                final_pose=[1, 0, 0, 0, 0, 0, 1],
                object_motion={
                    "path_length": 1.0,
                    "duration_s": 2.0,
                    "path": [[0, 0, 0], [1, 0, 0]],
                },
                joint_motion=None,
                tcp_motion=None,
                orientation={
                    "rotation_bin_initial": "6,6,6",
                    "rotation_bin_final": "6,6,6",
                },
                geometry_bucket=None,
            )
        )
    monkeypatch.setattr(module, "_read", lambda paths, entity: (rows, ["domain"]))
    output = tmp_path / "result"
    module._run([], [], output, "cube", 0.1, 5)
    summary = json.loads((output / "summary.json").read_text())
    detail = [
        json.loads(line)
        for line in (output / "after-episodes.jsonl").read_text().splitlines()
    ]
    pages = [
        json.loads((output / "pages" / f"after-{n}.json").read_text()) for n in range(2)
    ]
    assert [len(p) for p in pages] == [100, 1]
    assert pages[0] + pages[1] == detail
    assert summary["after"]["orientation"]["initial"] == {"6,6,6": 101}
    assert summary["after"]["joint"] == {"0,0|10,0": 101}
    assert summary["after"]["episodes"][100]["page"] == "after-1"
    assert summary["after"]["episodes"][0]["object_motion"] is True
    assert rows[0].get("pick_cell") is None
