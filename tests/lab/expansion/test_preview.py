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
from embodichain.lab.expansion._preview_deviations import (
    deviation_curves,
    recorded_coordinates,
)
from embodichain.lab.expansion._preview_heatmap import object_heatmaps
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
    state = PreviewState(data)
    assert data.solid_clip is clip and state.snapshot()["total"] == 0
    state.apply({"playing": True, "progress": 1})
    assert state.snapshot()["source_frame"] == 2 and state.values["playing"]
    with pytest.raises(ValueError, match="toggle_ghost"):
        state.apply({"toggle_ghost": 0})
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


def test_curves_use_calibrated_tcp_and_per_episode_joint_maxima(recorded_job):
    from scipy.spatial.transform import Rotation
    from xml.etree import ElementTree as ET

    path, _, _ = recorded_job
    data = load_preview([path])
    sample = data.clips[0]
    offset = np.eye(4)
    offset[0, 3] = 1
    robot = {
        "solver_cfg": {"wrist": {"end_link_name": "tool"}},
        "control_parts": {"wrist": ["a|b"]},
        "gripper_bindings": {
            "other": {
                "declaration": {"motion_part": "other_arm"},
                "root_link": "other_tool",
            },
            "hand": {
                "declaration": {"motion_part": "wrist"},
                "root_link": "hand_root",
                "defaults": {"default_root_from_tcp": offset.tolist()},
            },
        },
    }
    model = ET.fromstring(
        '<robot><joint name="a" type="continuous"/><joint name="b" type="revolute"/><joint name="finger" type="prismatic"/></robot>'
    )
    data.arm_joint_groups, tcp_frame = recorded_coordinates(robot, "tool", model)
    assert data.arm_joint_groups == {"deg": ["a", "b"]}

    def clip(times, yaw_deg, joint_names, qpos_deg):
        transforms = np.tile(np.eye(4), (len(times), 1, 1))
        transforms[:, :3, :3] = Rotation.from_euler(
            "z", np.asarray(yaw_deg)[:, None], degrees=True
        ).as_matrix()
        measured = replace(
            sample.measured,
            timestamps=np.array(times),
            joint_names=joint_names,
            qpos=np.deg2rad(qpos_deg),
            root_poses=np.tile([0, 0, 0, 0, 0, 0, 1], (len(times), 1)),
            object_poses={},
        )
        return replace(
            sample, measured=measured, links=transforms[:, None], tcp_frame=tcp_frame
        )

    data.reference = clip(
        [0, 1, 2, 3], [170] * 4, ("a", "b", "finger"), np.zeros((4, 3))
    )
    data.clips = [
        clip([0, 2], [170, -100], ("finger", "b", "a"), [[180, 0, 0], [180, 0, 60]]),
        clip([0, 1], [170, 80], ("a", "finger", "b"), [[0, 180, 0], [0, 180, -60]]),
        clip([0, 1], [170, -160], ("a", "b", "finger"), [[0, 0, 180], [-36, 0, 180]]),
    ]
    report = deviation_curves(data, ["hand_root"])
    assert report["time_s"] == [0, 1, 2, 3] and report["count"] == [3, 3, 1, 0]
    means = report["mean"]
    # At 1 s the three rotations are 45, 90 and 30 degrees. The first
    # position is interpolated halfway along the 90-degree TCP chord.
    chord_mm = 1000 * np.sqrt(2)
    short_chord_mm = 2000 * np.sin(np.deg2rad(15))
    mean_mm = (0.5 * chord_mm + chord_mm + short_chord_mm) / 3
    expected = {
        "translation_mm": (
            [0, short_chord_mm, chord_mm],
            [0, mean_mm, chord_mm],
            [0, chord_mm, chord_mm],
        ),
        "rotation_deg": ([0, 30, 90], [0, 55, 90], [0, 90, 90]),
        # Per-episode worst joint deviations at 1 s are 30, 60 and 36 degrees.
        "qpos_deg": ([0, 30, 60], [0, 42, 60], [0, 60, 60]),
    }
    for key, (low, mean, high) in expected.items():
        band = report["range"][key]
        np.testing.assert_allclose(band["min"][:3], low, atol=1e-9)
        np.testing.assert_allclose(means[key][:3], mean, atol=1e-9)
        np.testing.assert_allclose(band["max"][:3], high, atol=1e-9)
        assert (
            band["min"][-1] is None
            and means[key][-1] is None
            and band["max"][-1] is None
        )
    fallback = replace(data, clips=[data.reference, *data.clips], reference=None)
    implicit = deviation_curves(fallback, ["hand_root"])
    assert implicit["mean"] == means and implicit["range"] == report["range"]
    # Older jobs may lack TCP calibration while retaining valid arm measurements.
    data.clips[0].tcp_frame = None
    no_tcp = deviation_curves(data, ["hand_root"])
    assert no_tcp["mean"] == {"qpos_deg": means["qpos_deg"]}
    assert no_tcp["range"] == {"qpos_deg": report["range"]["qpos_deg"]}


def test_heatmaps_count_each_recording_once_per_object_cell_and_phase(recorded_job):
    path, _, _ = recorded_job
    data = load_preview([path])
    sample = data.solid_clip
    anchor = sample.measured.object_poses["box"]

    def clip(positions):
        poses = anchor.copy()
        poses[:, :2] = positions
        measured = replace(
            sample.measured, object_poses={"box": poses, "anchor": anchor}
        )
        return replace(sample, measured=measured)

    a, b, c, detour = (-0.25, 0.5), (0.125, -0.375), (0.75, 0.25), (0.375, 0.875)
    data.reference = clip([b, detour, a])
    data.clips = [clip([a, a, c]), clip([a, b, b])]
    data.object_specs = {"anchor": {"body_type": "static"}}
    report = object_heatmaps(data)
    assert report["episodes"] == 3 and report["explicit_reference"]
    box, still = report["objects"]
    assert [box["uid"], still["uid"]] == ["box", "anchor"]

    for phase, hits in {
        "all": {a: 3, b: 2, c: 1, detour: 1},
        "start": {a: 2, b: 1},
        "end": {a: 1, b: 1, c: 1},
    }.items():
        view = box["views"][phase]
        expected = np.zeros_like(view["counts"])
        for (x, y), count in hits.items():
            row = np.searchsorted(box["y_edges_m"], y, side="right") - 1
            col = np.searchsorted(box["x_edges_m"], x, side="right") - 1
            expected[row, col] = count
        np.testing.assert_array_equal(view["counts"], expected)
        points = np.array(list(hits))
        np.testing.assert_allclose(
            view["bounds_m"], [points.min(axis=0), points.max(axis=0)]
        )
        # The fixed body's single cell includes all three recordings.
        fixed_counts = np.array(still["views"][phase]["counts"])
        assert fixed_counts.max() == 3 and fixed_counts.sum() == 3

    implicit = object_heatmaps(
        replace(data, clips=[data.reference, *data.clips], reference=None)
    )
    assert implicit == {**report, "explicit_reference": False}


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


def test_independent_ghosts_keep_reference_playing_across_filters(recorded_job):
    path, _, _ = recorded_job
    data = load_preview([path])
    first = data.clips[0]
    data.reference = replace(first, measured=replace(first.measured, name="original"))
    data.clips = [
        replace(first, space="A", affordance="0", variant="0"),
        replace(first, space="B", affordance="1", variant="1"),
    ]
    state = PreviewState(data)
    assert state.snapshot()["visible_ghosts"] == []
    state.apply({"toggle_ghost": 0})
    state.apply({"toggle_ghost": 1})
    assert state.visible_ghosts() == [0, 1]
    state.apply({"toggle_ghost": 1})
    assert state.visible_ghosts() == [0]
    state.apply({"space": "B"})
    assert state.matching() == [1] and state.visible_ghosts() == []
    state.apply({"ghosts": True})
    assert state.visible_ghosts() == [1]
    state.apply({"aff": "0", "playing": True})
    assert state.matching() == [] and state.values["playing"]
    assert state.snapshot()["episode"] == "original"
    assert state.snapshot()["source_frame"] == 0
    state.apply({"space": "all", "aff": "all", "progress": 0.5})
    assert state.visible_ghosts() == [0, 1]
    state.apply({"space": "B", "ghosts": False})
    state.apply({"space": "all"})
    assert state.visible_ghosts() == [0]
    assert state.frame_at(first) == 0  # phase boundary .4 s -> recorded frame at 0 s
    state.apply({"align": "time"})
    assert state.frame_at(first) == 1  # exactly .5 s
    state.apply({"progress": 1})
    assert state.frame_at(first) == 2
    for update in (
        {"progress": float("nan")},
        {"toggle_ghost": True},
        {"toggle_ghost": 2},
        {"playing": "true"},
        {"unknown": 0},
        {"aff": "missing"},
    ):
        before = dict(state.values)
        with pytest.raises(ValueError):
            state.apply(update)
        assert state.values == before


def test_playback_toggles_in_order_and_restarts_from_the_end(recorded_job):
    path, _, _ = recorded_job
    state = PreviewState(load_preview([path]))
    state.apply({"progress": 0.5})
    toggle = {"toggle_playback": True}
    state.apply(toggle)
    assert state.values["playing"] and state.values["progress"] == 0.5
    state.apply(toggle)
    assert not state.values["playing"] and state.values["progress"] == 0.5
    state.apply({"progress": 1})
    state.apply(toggle)
    assert state.values["playing"] and state.values["progress"] == 0


def test_http_snapshot_control_and_shutdown(recorded_job):
    path, _, _ = recorded_job
    state = PreviewState(load_preview([path]))
    transport = PreviewTransport(state, 0)
    transport.publish({"frame": 7, "source_frame": 2}, b"jpeg-frame-7")
    thread = threading.Thread(target=transport.server.serve_forever)
    thread.start()
    try:
        heatmaps = json.load(urllib.request.urlopen(transport.url + "object-heatmaps"))
        assert heatmaps["episodes"] == 1
        assert heatmaps["objects"][0]["views"]["all"]["bounds_m"] == [
            [10, 20],
            [10, 20],
        ]
        assert np.sum(heatmaps["objects"][0]["views"]["all"]["counts"]) == 1
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


def test_viewer_assets_are_in_package_data(tmp_path):
    from setup import get_package_data
    from setuptools import Distribution
    from setuptools.command.build_py import build_py
    import shutil

    package = "embodichain.lab.expansion"
    source = tmp_path / "source"
    source.mkdir()
    originals = Path(__file__).resolve().parents[3] / "embodichain/lab/expansion"
    assets = ("_preview.html", "_preview_charts.js", "_preview_heatmap.js")
    for name in assets:
        shutil.copyfile(originals / name, source / name)
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
    packaged = Path(command.build_lib) / "embodichain/lab/expansion"
    for name in assets:
        assert (packaged / name).read_bytes() == (originals / name).read_bytes()


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
    # Without a separate original, the first input stays solid across group changes.
    fallback = PreviewState(replace(data, reference=None))
    fallback.apply({"space": data.clips[-1].space, "ghosts": True})
    snapshot = fallback.snapshot()
    assert snapshot["episode"] == "first-0"
    assert snapshot["total"] == second_count
    assert {data.clips[i].measured.name for i in snapshot["visible_ghosts"]} == {
        f"second-{index}" for index in range(second_count)
    }
