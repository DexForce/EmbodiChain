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

"""Result-only preview: provenance, row selection, controls and HTTP lifecycle."""

from __future__ import annotations

import base64
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import threading
import urllib.error
import urllib.request
import numpy as np
import pytest
import torch

from embodichain.lab.expansion.contracts import (
    CandidateIdentity,
    ExpertEpisode,
    ValidationResult,
    ValidationCheck,
)
from embodichain.lab.expansion.persistence import ArtifactEpisodeSink
from embodichain.lab.expansion._preview_data import load_preview, PreviewState
from embodichain.lab.expansion._preview_render import prepare_geometry
from embodichain.lab.expansion._preview_server import PreviewTransport


@pytest.fixture
def recorded_job(tmp_path):
    asset = tmp_path / "robot.urdf"
    asset.write_text(
        '<robot name="test"><link name="base"/><link name="arm"/><link name="tip"/>'
        '<joint name="joint" type="revolute"><parent link="base"/><child link="arm"/><axis xyz="0 0 1"/>'
        '<limit lower="-4" upper="4" effort="1" velocity="1"/></joint>'
        '<joint name="tip_fixed" type="fixed"><parent link="arm"/><child link="tip"/>'
        '<origin xyz="1 0 0"/></joint></robot>'
    )
    output = tmp_path / "result"
    output.mkdir()
    job = {
        "resolved_deployment": {
            "robot": {
                "uid": "robot",
                "fpath": str(asset),
                "solver_cfg": {"arm": {"end_link_name": "tip"}},
            },
            "rigid_object": [
                {
                    "uid": "box",
                    "shape": {"shape_type": "Cube", "size": [0.05, 0.05, 0.05]},
                }
            ],
        }
    }
    (output / "job.json").write_text(json.dumps(job))
    poses = torch.tensor([[0.0, 0, 0, 0, 0, 0, 1]] * 3)
    objects = poses.clone()
    objects[:, :2] = torch.tensor([10.0, 20.0])
    transforms = np.tile(np.eye(4), (2, 1, 1))
    transforms[:, 0, 3] = [100, 200]
    selected = transforms.copy()
    selected[:, 0, 3] += [1, 2]
    sample = {
        "env_ids": [3, 7],
        "success": [True, True],
        "sampling": {"branch_overrides": [0, 1]},
        "candidate_ids": [23, 31],
        "reference_poses": transforms.tolist(),
        "selected_poses": selected.tolist(),
    }
    metadata = {
        "initial_state": [
            {"kind": "robot", "uid": "robot", "joint_names": ["joint"]},
            {"kind": "rigid_object", "uid": "box"},
        ],
        "scene_state_fields": [
            {"entity_uid": "robot", "field": "pose", "key": "robot_pose"}
        ],
        "expansion": [
            {"env_id": 7, "trajectory_requested": 1, "trajectory_selected": 1}
        ],
        "program_segments": [
            {
                "metadata": {
                    "runtime": {
                        "calls": [
                            {
                                "call": {
                                    "call_kind": "pick",
                                    "arguments": {"object": {"entity_id": "box"}},
                                },
                                "plan_attempts": [
                                    {
                                        "planner_diagnostics": {
                                            "metadata": {
                                                "affordance_sample": {"grasp": sample}
                                            }
                                        }
                                    }
                                ],
                                "events": [
                                    {
                                        "kind": "trajectory_segment_entered",
                                        "timestamp": 0.0,
                                        "segment_name": "approach",
                                    },
                                    {
                                        "kind": "trajectory_segment_entered",
                                        "timestamp": 0.4,
                                        "segment_name": "lift",
                                    },
                                ],
                            }
                        ]
                    }
                }
            }
        ],
    }
    episode = ExpertEpisode(
        CandidateIdentity(
            "scene", "initial", "candidate", "family", "source", "v1", "template"
        ),
        {
            "full_joint_positions": torch.tensor([[0.0], [np.pi / 2], [np.pi]]),
            "robot_pose": poses,
            "rigid/box/state": objects,
        },
        torch.zeros((2, 1)),
        torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64),
        "qpos",
        ValidationResult((ValidationCheck("task_success", "passed"),)),
        "episode",
        "commit",
        metadata=metadata,
    )
    ArtifactEpisodeSink(output).submit(episode, submission_id=0)
    return output, job, episode


def test_result_loading_binds_grasp_row_and_fk_without_writing_results(recorded_job):
    path, _, _ = recorded_job
    before = {
        p.relative_to(path): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in path.rglob("*")
        if p.is_file()
    }
    data = load_preview([path])
    clip = data.clips[0]
    assert (
        clip.affordance == "1" and clip.candidate_ids == ("31",) and clip.variant == "1"
    )
    np.testing.assert_allclose(clip.targets[0, :3, 3], [12, 20, 0])
    names, _ = prepare_geometry(data)
    np.testing.assert_allclose(clip.tips, [[1, 0, 0], [0, 1, 0], [-1, 0, 0]], atol=1e-6)
    assert "tip" in names and clip.links.shape[0] == 3
    after = {
        p.relative_to(path): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in path.rglob("*")
        if p.is_file()
    }
    assert before == after


def test_result_loader_rejects_incomplete_scene_and_budget(recorded_job, tmp_path):
    path, job, episode = recorded_job
    with pytest.raises(ValueError, match="aggregate result or native job"):
        load_preview([tmp_path / "missing"])
    assert not (tmp_path / "missing").exists()
    ArtifactEpisodeSink(path).submit(
        replace(episode, episode_id="episode2", commit_id="commit2"), submission_id=1
    )
    with pytest.raises(ValueError, match="budget"):
        load_preview([path], max_episodes=1)
    with pytest.raises(ValueError, match="reference"):
        load_preview([path], reference=path)
    job["resolved_deployment"]["rigid_object"][0]["shape"]["shape_type"] = "Mesh"
    (path / "job.json").write_text(json.dumps(job))
    with pytest.raises(ValueError, match="asset is unavailable"):
        load_preview([path])


def test_result_loader_keeps_real_mesh_spec_and_native_operator_labels(
    recorded_job, tmp_path
):
    path, job, episode = recorded_job
    mesh = tmp_path / "object.ply"
    mesh.write_text(
        "ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nelement face 1\nproperty list uchar int vertex_indices\nend_header\n0 0 0\n1 0 0\n0 1 0\n3 0 1 2\n"
    )
    job["resolved_deployment"]["rigid_object"][0]["shape"] = {
        "shape_type": "Mesh",
        "fpath": str(mesh),
    }
    (path / "job.json").write_text(json.dumps(job))
    data = load_preview([path])
    assert data.boxes["box"] is None
    assert data.object_specs["box"]["shape"]["fpath"] == str(mesh)
    from embodichain.lab.expansion._preview_data import annotations

    result = annotations(
        {
            "source_env_id": 0,
            "expansion": [{"env_id": 0, "operator": "joint_residual", "proposal": 3}],
        },
        data.clips[0].measured,
    )
    assert result["variant"] == "joint_residual"
    assert result["affordance"] == "unrecorded"


def test_filter_empty_intersection_and_phase_alignment(recorded_job):
    path, _, _ = recorded_job
    data = load_preview([path])
    first = data.clips[0]
    data.clips = [
        replace(first, space="A", affordance="0", variant="0"),
        replace(first, space="B", affordance="1", variant="1"),
    ]
    state = PreviewState(data)
    state.apply({"space": "B"})
    assert state.visible() == [1] and state.values["selected"] == 1
    state.apply({"aff": "0", "playing": True})
    assert state.visible() == [] and not state.values["playing"]
    assert state.snapshot()["source_frame"] is None
    state.apply({"space": "all", "aff": "all", "progress": 0.5})
    assert state.frame_at(first) == 0  # phase boundary .4 s -> recorded frame at 0 s
    state.apply({"align": "time"})
    assert state.frame_at(first) == 1  # exactly .5 s
    state.apply({"progress": 1})
    assert state.frame_at(first) == 2
    for update in (
        {"progress": float("nan")},
        {"selected": True},
        {"playing": "true"},
        {"unknown": 0},
        {"aff": "missing"},
    ):
        before = dict(state.values)
        with pytest.raises(ValueError):
            state.apply(update)
        assert state.values == before


def test_http_snapshot_control_and_shutdown(recorded_job):
    path, _, _ = recorded_job
    state = PreviewState(load_preview([path]))
    transport = PreviewTransport(state, 0)
    transport.publish({"frame": 7, "source_frame": 2}, b"jpeg-frame-7")
    thread = threading.Thread(target=transport.server.serve_forever)
    thread.start()
    try:
        snapshot = json.load(urllib.request.urlopen(transport.url + "snapshot"))
        assert snapshot["frame"] == 7 and snapshot["source_frame"] == 2
        assert base64.b64decode(snapshot["image"].split(",")[1]) == b"jpeg-frame-7"
        req = urllib.request.Request(
            transport.url + "control",
            data=b'{"progress":0.5}',
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        assert urllib.request.urlopen(req).status == 204
        assert (
            state.values["progress"] == 0
        )  # only the render owner applies queued commands
        state.apply(transport.commands.get_nowait())
        assert state.values["progress"] == 0.5
        req.add_header("Origin", "https://untrusted.example")
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(req)
        assert error.value.code == 403
    finally:
        transport.server.shutdown()
        thread.join(timeout=5)
        transport.server.server_close()
    assert not thread.is_alive()


def test_preview_cli_dispatch_does_not_collect(monkeypatch, recorded_job):
    from embodichain.cli.expansion import main
    import embodichain.lab.expansion._preview_server as server
    from embodichain.lab.expansion import ExpansionEngine

    path, _, _ = recorded_job
    called = []
    monkeypatch.setattr(
        server, "serve_preview", lambda data, **kw: called.append((len(data.clips), kw))
    )
    monkeypatch.setattr(
        ExpansionEngine,
        "from_job",
        lambda *a, **kw: pytest.fail("preview must not collect"),
    )
    assert main(["preview", "--input", str(path), "--gpu", "2", "--open"]) == 0
    assert (
        called[0][0] == 1 and called[0][1]["gpu"] == 2 and called[0][1]["open_browser"]
    )


def test_viewer_html_is_in_package_data(tmp_path):
    from setup import get_package_data
    from setuptools import Distribution
    from setuptools.command.build_py import build_py
    import shutil

    package = "embodichain.lab.expansion"
    source = tmp_path / "source"
    source.mkdir()
    original = (
        Path(__file__).resolve().parents[3] / "embodichain/lab/expansion/_preview.html"
    )
    shutil.copyfile(original, source / "_preview.html")
    (source / "__init__.py").write_text("")
    distribution = Distribution(
        {
            "packages": [package],
            "package_dir": {package: str(source)},
            "package_data": get_package_data(),
            "include_package_data": False,
        }
    )
    distribution.script_name = "setup.py"
    command = build_py(distribution)
    command.build_lib = str(tmp_path / "build")
    command.ensure_finalized()
    command.run()
    packaged = Path(command.build_lib) / "embodichain/lab/expansion/_preview.html"
    assert packaged.read_bytes() == original.read_bytes()


def test_manifest_keeps_rejected_points_visible_in_coverage(recorded_job):
    path, _, _ = recorded_job
    root = path.parent
    manifest = {
        "records": [
            {"result": path.name, "confirmed": 1, "status": "confirmed"},
            {"result": "unqualified", "confirmed": 0, "status": "rejected"},
        ],
        "coverage_complete": False,
    }
    (root / "preview-plan.json").write_text(json.dumps(manifest))
    data = load_preview([root])
    assert len(data.clips) == 1
    assert (
        PreviewState(data).snapshot()["coverage"]["records"][1]["status"] == "rejected"
    )
    manifest["records"][1]["result"] = "../escape"
    (root / "preview-plan.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="escapes"):
        load_preview([root])


def test_job_preview_collects_then_opens_and_input_alias_stays_read_only(
    monkeypatch, recorded_job
):
    from embodichain.cli.expansion import main
    import embodichain.lab.expansion._preview_job as job
    import embodichain.lab.expansion._preview_server as server
    import torch

    path, _, _ = recorded_job
    calls = []
    monkeypatch.setattr(
        job,
        "plan_preview",
        lambda *a, **kw: {"output": str(path), "records": [{}], "planned_attempts": 1},
    )

    def collect(plan):
        calls.append("collect")
        return {**plan, "confirmed": 1, "coverage_complete": True}

    monkeypatch.setattr(job, "collect_preview", collect)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        server,
        "serve_preview",
        lambda data, **kw: calls.append(("serve", kw["open_browser"])),
    )
    assert main(["preview", "--job", "job.yaml"]) == 0
    assert calls == ["collect", ("serve", True)]
    calls.clear()
    assert main(["preview", "--inpu", str(path)]) == 0
    assert calls == [("serve", False)]


def test_row_identity_without_motion_and_legacy_ambiguity(recorded_job):
    from embodichain.lab.expansion._preview_data import annotations
    from embodichain.lab.gym.envs._json import json_safe_copy

    path, _, episode = recorded_job
    measured = load_preview([path]).clips[0].measured
    metadata = json_safe_copy(episode.metadata, field_name="metadata")
    metadata["expansion"] = []
    assert annotations(metadata, measured)["targets"].shape == (0, 4, 4)
    metadata["source_env_id"] = 7
    result = annotations(metadata, measured)
    np.testing.assert_allclose(result["targets"][0, :3, 3], [12, 20, 0])
    assert result["candidate_ids"] == ("31",)


@pytest.mark.parametrize("second_count", [1, 2])
def test_aggregate_input_keeps_explicit_reference_separate(recorded_job, second_count):
    path, job, episode = recorded_job
    root = path.parent
    directories = []
    for name, count in (("first", 1), ("second", second_count), ("reference", 1)):
        directory = root / name
        directory.mkdir()
        (directory / "job.json").write_text(json.dumps(job))
        for index in range(count):
            ArtifactEpisodeSink(directory).submit(
                replace(
                    episode,
                    episode_id=f"{name}-{index}",
                    commit_id=f"{name}-commit-{index}",
                ),
                submission_id=index,
            )
        directories.append(directory)
    aggregate = root / "aggregate"
    aggregate.mkdir()
    (aggregate / "expansion.json").write_text(
        json.dumps(
            {
                "operators": {
                    str(i): {
                        "directory": str(p),
                        "committed_episodes": 1 if i == 0 else second_count,
                    }
                    for i, p in enumerate(directories[:2])
                }
            }
        )
    )
    data = load_preview(
        [aggregate], reference=directories[2], max_episodes=1 + second_count
    )
    assert len(data.clips) == 1 + second_count
    assert data.reference.space.endswith("reference")
    assert all(not clip.space.endswith("reference") for clip in data.clips)
