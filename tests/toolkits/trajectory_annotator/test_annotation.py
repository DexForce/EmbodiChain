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

"""Frame identity, inclusive intervals and durable annotation contracts."""

from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path
import subprocess
import sys
import threading
import time

import pytest

from embodichain.toolkits.trajectory_annotator._models import (
    ConflictError,
    validate_skill_intervals,
)
from embodichain.toolkits.trajectory_annotator._storage import AnnotationStore

pytestmark = pytest.mark.no_sim


def skill_interval(
    start: int = 2, end: int = 4, identifier: str = "flip"
) -> dict[str, object]:
    """A known inclusive span: three frames, independent of storage machinery."""
    return dict(
        id=identifier,
        label=" 翻转包裹 ",
        start_frame=start,
        end_frame=end,
        note="已翻转",
    )


def test_inclusive_ranges_allow_singletons_adjacency_and_gaps() -> None:
    result = validate_skill_intervals(
        [
            skill_interval(8, 9, "c"),
            skill_interval(1, 1, "a"),
            skill_interval(2, 4, "b"),
        ],
        10,
    )
    assert [(item["start_frame"], item["end_frame"]) for item in result] == [
        (1, 1),
        (2, 4),
        (8, 9),
    ]
    assert result[0]["label"] == "翻转包裹"


@pytest.mark.parametrize(
    "skill_intervals",
    [
        [skill_interval(0, 2), skill_interval(2, 4, "other")],
        [skill_interval(-1, 2)],
        [skill_interval(4, 3)],
        [skill_interval(0, 10)],
        [skill_interval(True, 3)],
        [skill_interval(1.5, 3)],
        [skill_interval(0, 1), skill_interval(4, 5)],
        [{**skill_interval(), "label": " "}],
        [{**skill_interval(), "core_interval": {"start_frame": 1, "end_frame": 3}}],
        [{**skill_interval(), "core_interval": {"start_frame": 3, "end_frame": 5}}],
        [{**skill_interval(), "core_interval": {"start_frame": 4, "end_frame": 3}}],
        [{**skill_interval(), "core_interval": {"start_frame": True, "end_frame": 3}}],
    ],
)
def test_invalid_intervals_cannot_be_saved(skill_intervals: object) -> None:
    with pytest.raises(ValueError):
        validate_skill_intervals(skill_intervals, 10)


@pytest.mark.parametrize("variable,start", [(False, 0), (True, 90)])
def test_seeks_match_independent_sequential_decode(
    make_video: Callable[..., Path], variable: bool, start: int
) -> None:
    av = pytest.importorskip("av")
    import cv2
    import numpy as np
    from embodichain.toolkits.trajectory_annotator._cfg import AnnotationCfg
    from embodichain.toolkits.trajectory_annotator._frames import FrameReader
    from embodichain.toolkits.trajectory_annotator.video import VideoFrameSource

    path = make_video(variable=variable, start=start)
    with av.open(str(path)) as source:
        stream = source.streams.video[0]
        baseline = [
            (frame.pts, frame.to_ndarray(format="bgr24"))
            for frame in source.decode(stream)
        ]
        time_base = stream.time_base
        assert stream.codec_context.has_b_frames
    assert len(baseline) == 48  # Includes delayed frames emitted by decoder flush.
    if start:
        assert baseline[0][0] > 0
    cfg = AnnotationCfg(cache_bytes=4000)
    reader = FrameReader(VideoFrameSource(path), cfg)
    try:
        metadata = reader.prepare()
        assert metadata.frame_count == len(baseline)
        for index in [0, 1, 12, 47, 46, 24, 11, 10, 0, 30, 31]:
            preview = reader.preview(index)
            pixels = cv2.imdecode(
                np.frombuffer(preview.jpeg, np.uint8), cv2.IMREAD_COLOR
            )
            assert np.abs(pixels.astype(float) - baseline[index][1]).mean() < 3
            assert preview.index == index
            assert preview.pts == baseline[index][0]
            assert preview.time_seconds == pytest.approx(
                float((baseline[index][0] - baseline[0][0]) * time_base)
            )
            assert reader.cache_bytes <= cfg.cache_bytes
    finally:
        reader.close()


def test_index_budget_and_missing_pts_fail_explicitly(
    make_video: Callable[..., Path],
) -> None:
    pytest.importorskip("av")
    from embodichain.toolkits.trajectory_annotator._cfg import AnnotationCfg
    from embodichain.toolkits.trajectory_annotator._frames import FrameReader
    from embodichain.toolkits.trajectory_annotator.video import VideoFrameSource

    reader = FrameReader(
        VideoFrameSource(make_video(count=4), index_bytes=32), AnnotationCfg()
    )
    try:
        with pytest.raises(ValueError, match="预算"):
            reader.prepare()
    finally:
        reader.close()
    # An elementary H.264 stream has no container presentation timestamps.
    reader = FrameReader(
        VideoFrameSource(make_video(name="raw.h264", count=4)), AnnotationCfg()
    )
    try:
        with pytest.raises(ValueError, match="PTS"):
            reader.prepare()
    finally:
        reader.close()


def test_atomic_save_resume_conflict_and_write_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from embodichain.toolkits.trajectory_annotator import _storage

    video = tmp_path / "video.mp4"
    video.write_bytes(b"video identity fixture")
    details = dict(
        sha256="a" * 64,
        stream_index=0,
        frame_count=10,
        width=160,
        height=96,
        time_base={"numerator": 1, "denominator": 30},
    )
    from dataclasses import replace
    from embodichain.toolkits.trajectory_annotator.frames import FrameSourceInfo

    source = FrameSourceInfo(
        "video", video.name, "a" * 64, 10, 160, 96, 30.0, "pts", details, video
    )
    path = tmp_path / "nested" / "labels.json"
    store = AnnotationStore(path, 1024**2)
    store.initialize(source)
    selected = {**skill_interval(), "core_interval": {"start_frame": 2, "end_frame": 3}}
    saved = store.save([selected], 0)
    original = path.read_bytes()
    assert saved["revision"] == 1
    assert saved["schema_version"] == 3
    assert saved["segments"][0]["core_interval"] == {"start_frame": 2, "end_frame": 3}
    assert json.loads(original)["segments"][0]["label"] == "翻转包裹"
    restored = AnnotationStore(path, 1024**2)
    restored.initialize(source)
    assert restored.snapshot()["document"] == saved
    with pytest.raises(ConflictError):
        store.save([], 0)
    with monkeypatch.context() as context:

        def fail_replace(*args: object) -> None:
            raise PermissionError("read-only destination")

        context.setattr(_storage.os, "replace", fail_replace)
        with pytest.raises(PermissionError):
            store.save([], 1)
    assert path.read_bytes() == original
    assert store.snapshot()["document"] == saved
    assert not list(path.parent.glob(f".{path.name}.*"))
    store.save([], 1)
    with pytest.raises(ConflictError):
        restored.save([], 1)  # Another server changed the file after this load.
    mismatch = AnnotationStore(path, 1024**2)
    with pytest.raises(ValueError, match="不匹配"):
        mismatch.initialize(replace(source, identity="b" * 64))
    assert json.loads(path.read_text())["revision"] == 2
    # Legacy video annotations remain byte-for-byte untouched until the next save.
    legacy = dict(
        schema_version=1,
        frame_index_base=0,
        interval="closed",
        source_video=details,
        revision=3,
        segments=[skill_interval()],
    )
    path.write_text(json.dumps(legacy))
    legacy_bytes = path.read_bytes()
    migrated = AnnotationStore(path, 1024**2)
    migrated.initialize(source)
    assert path.read_bytes() == legacy_bytes
    assert migrated.snapshot()["document"]["schema_version"] == 2
    assert migrated.save([], 3)["revision"] == 4
    assert "source_video" not in json.loads(path.read_text())


def test_http_video_and_persistence_boundaries(
    make_video: Callable[..., Path], tmp_path: Path
) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("av")
    from fastapi.testclient import TestClient
    from embodichain.toolkits.trajectory_annotator._server import create_app

    video = make_video()
    path = tmp_path / "labels.json"
    from embodichain.toolkits.trajectory_annotator.video import VideoFrameSource

    with TestClient(
        create_app(lambda: VideoFrameSource(video), path), base_url="http://127.0.0.1"
    ) as client:
        deadline = time.monotonic() + 10
        while client.get("/api/status").json()["phase"] != "ready":
            assert time.monotonic() < deadline
            time.sleep(0.01)
        project = client.get("/api/project").json()
        headers = {"X-Annotation-Token": project["session_token"]}
        assert client.get("/").status_code == 200
        for asset in ["app.js", "styles.css"]:
            assert client.get(f"/static/{asset}").status_code == 200
        frame = client.get("/api/frames/47?request_id=23")
        assert frame.status_code == 200
        assert frame.headers["X-Frame-Index"] == "47"
        assert frame.headers["X-Request-ID"] == "23"
        assert frame.content.startswith(b"\xff\xd8")
        assert client.get("/api/frames/48").status_code == 422
        payload = dict(segments=[skill_interval()], expected_revision=0)
        assert client.put("/api/annotations", json=payload).status_code == 403
        response = client.put("/api/annotations", json=payload, headers=headers)
        assert response.status_code == 200
        assert json.loads(path.read_text())["segments"][0]["end_frame"] == 4
        assert (
            client.put("/api/annotations", json=payload, headers=headers).status_code
            == 409
        )
        invalid = dict(
            segments=[skill_interval(0, 1), skill_interval(1, 3, "other")],
            expected_revision=1,
        )
        assert (
            client.put("/api/annotations", json=invalid, headers=headers).status_code
            == 422
        )
        assert client.get("/api/project").json()["document"]["revision"] == 1


def test_unified_cli_help_does_not_import_optional_or_simulation_modules() -> None:
    code = """
import sys
from embodichain.cli.main import main
try:
    main(['annotate-trajectory', '--help'])
except SystemExit as error:
    assert error.code == 0
assert not any(name in sys.modules for name in ('av', 'cv2', 'fastapi', 'uvicorn', 'torch'))
assert not any(name.startswith('embodichain.lab') for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "--video" in result.stdout and "--hdf5" in result.stdout


def test_worker_keeps_only_the_latest_pending_seek(
    make_video: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("av")
    from embodichain.toolkits.trajectory_annotator._cfg import AnnotationCfg
    from embodichain.toolkits.trajectory_annotator._frames import (
        SupersededError,
        FrameReader,
        FrameWorker,
    )

    entered, release = threading.Event(), threading.Event()
    original = FrameReader.preview

    def held(self: FrameReader, index: int) -> object:
        if index == 5:
            entered.set()
            assert release.wait(timeout=5)
        return original(self, index)

    monkeypatch.setattr(FrameReader, "preview", held)
    from embodichain.toolkits.trajectory_annotator.video import VideoFrameSource

    path = make_video()
    worker = FrameWorker(
        lambda: VideoFrameSource(path), AnnotationCfg(), lambda _: None
    )
    worker.start()
    try:
        deadline = time.monotonic() + 5
        while worker.status()["phase"] != "ready":
            assert time.monotonic() < deadline
            time.sleep(0.01)
        active = worker.request(5)
        assert entered.wait(timeout=3)
        obsolete = worker.request(6)
        latest = worker.request(7)
        with pytest.raises(SupersededError):
            obsolete.result(timeout=1)
        release.set()
        assert active.result(timeout=3).index == 5
        assert latest.result(timeout=3).index == 7
    finally:
        release.set()
        worker.close()
    assert worker.status()["phase"] == "closed"


@pytest.mark.parametrize("accepted", [True, False])
def test_web_fitting_vfr_clock_concurrency_export_and_stale_revisions(
    make_video: Callable[..., Path],
    tmp_path: Path,
    fitting_oracle: dict[str, object],
    accepted: bool,
) -> None:
    """A fit never owns decoding, blocks edits, exports rejection as success or outlives its annotation."""
    import io
    import numpy as np
    from fastapi.testclient import TestClient
    from embodichain.toolkits.trajectory_annotator._server import create_app
    from embodichain.toolkits.trajectory_annotator.video import VideoFrameSource

    video = make_video(variable=True, start=90)
    fitting_oracle["accepted"] = accepted
    fitting_oracle["release"].clear()
    app = create_app(lambda: VideoFrameSource(video), tmp_path / "annotations.json")
    with TestClient(app, base_url="http://127.0.0.1") as client:
        deadline = time.monotonic() + 10
        while client.get("/api/status").json()["phase"] != "ready":
            assert time.monotonic() < deadline
            time.sleep(0.01)
        headers = {
            "X-Annotation-Token": client.get("/api/project").json()["session_token"]
        }
        selected = {
            **skill_interval(0, 30),
            "label": "pick_up",
            "core_interval": {"start_frame": 5, "end_frame": 20},
        }
        saved = client.put(
            "/api/annotations",
            json=dict(segments=[selected], expected_revision=0),
            headers=headers,
        ).json()
        payload = dict(
            segment_id=selected["id"],
            expected_revision=saved["revision"],
            skill_id="pick_up",
            inputs=dict(
                joints="known.npz",
                robot_profile="robot.json",
                qpos_semantics="measured_state",
                time_offset=0,
            ),
        )
        assert client.post("/api/fitting/jobs", json=payload).status_code == 403
        assert (
            client.post(
                "/api/fitting/jobs",
                json={**payload, "skill_id": "twist"},
                headers=headers,
            ).status_code
            == 422
        )
        assert (
            client.post(
                "/api/fitting/jobs",
                json={**payload, "expected_revision": 0},
                headers=headers,
            ).status_code
            == 409
        )
        assert (
            client.post(
                "/api/fitting/jobs",
                json={**payload, "inputs": {**payload["inputs"], "time_offset": None}},
                headers=headers,
            ).status_code
            == 422
        )
        response = client.post("/api/fitting/jobs", json=payload, headers=headers)
        assert response.status_code == 202
        job = response.json()["job_id"]
        try:
            assert fitting_oracle["entered"].wait(timeout=3)
            assert (
                client.post(
                    "/api/fitting/jobs", json=payload, headers=headers
                ).status_code
                == 409
            )
            assert client.get("/api/frames/47").status_code == 200
            assert client.get("/api/fitting").json()["phase"] == "running"
        finally:
            fitting_oracle["release"].set()
        while (status := client.get("/api/fitting").json())["phase"] == "running":
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert status["phase"] == "done" and status["accepted"] == accepted
        # The source has nonzero PTS and an extra tick every five frames. FPS
        # multiplication would instead yield 5/30 and 20/30, so it must fail here.
        call = fitting_oracle["calls"][0]
        np.testing.assert_allclose(call["frame_times"], [6 / 30, 24 / 30])
        assert call["frame_durations"][-1] == pytest.approx(1 / 30)
        assert call["provenance"]["annotation_video_interval"] == {
            "start_frame": 0,
            "end_frame": 30,
        }
        assert call["provenance"]["annotation_segment"]["core_interval"] == {
            "start_frame": 5,
            "end_frame": 20,
        }
        assert client.get("/api/frame-index?time_seconds=0.2").json() == {"index": 5}
        assert client.get("/api/frame-index?time_seconds=nan").status_code == 422
        preview = client.get(f"/api/fitting/jobs/{job}/preview").json()
        assert preview["report"]["parameters"]["lift_height"] == 0.12
        assert "original_links" not in preview
        assert preview["viewer"]["phase"] == "ready"
        viewer_endpoint = f"/api/fitting/jobs/{job}/viewer"
        assert (
            client.post(viewer_endpoint, json={"time_seconds": 0.5}).status_code == 403
        )
        assert (
            client.post(
                viewer_endpoint, json={"time_seconds": "nan"}, headers=headers
            ).status_code
            == 422
        )
        assert (
            client.post(
                viewer_endpoint, json={"time_seconds": 0.5}, headers=headers
            ).status_code
            == 200
        )
        response = client.get(f"/api/fitting/jobs/{job}/bundle")
        assert response.status_code == 200
        with np.load(io.BytesIO(response.content), allow_pickle=False) as bundle:
            original, patched = bundle["original_qpos"], bundle["patched_qpos"]
            np.testing.assert_array_equal(bundle["timestamps"], np.arange(300) / 100)
            np.testing.assert_array_equal(patched[:23], original[:23])
            np.testing.assert_array_equal(patched[81:], original[81:])
            if accepted:
                assert patched[50, 0] - original[50, 0] == pytest.approx(0.01)
            else:
                np.testing.assert_array_equal(patched, original)
                assert not np.array_equal(bundle["proposed_qpos"], original)
        changed = {**selected, "end_frame": 21}
        assert (
            client.put(
                "/api/annotations",
                json=dict(segments=[changed], expected_revision=1),
                headers=headers,
            ).status_code
            == 200
        )
        assert client.get("/api/fitting").json()["stale"]
        for endpoint in ("bundle", "preview"):
            assert client.get(f"/api/fitting/jobs/{job}/{endpoint}").status_code == 409
        assert (
            client.post(
                viewer_endpoint, json={"time_seconds": 0.5}, headers=headers
            ).status_code
            == 409
        )


def test_cli_fits_saved_core_and_retains_outer_annotation(
    make_video: Callable[..., Path], tmp_path: Path, fitting_oracle: dict[str, object]
) -> None:
    """CLI and web honor the same core, preserving the broad real-motion label."""
    import numpy as np
    from embodichain.toolkits.skill_fitting.cli import main
    from embodichain.toolkits.trajectory_annotator.video import VideoFrameSource
    from embodichain.data_pipeline.skill_fitting import load_skill_fit

    video = make_video(variable=True, start=90)
    annotations = tmp_path / "labels.json"
    source = VideoFrameSource(video)
    try:
        info = source.prepare(threading.Event(), lambda **_: None)
        store = AnnotationStore(annotations, 1024**2)
        store.initialize(info)
        store.save(
            [
                {
                    **skill_interval(0, 30),
                    "label": "Pickup",
                    "core_interval": {"start_frame": 5, "end_frame": 20},
                }
            ],
            0,
        )
    finally:
        source.close()
    output = tmp_path / "fit.npz"
    main(
        [
            "--video",
            str(video),
            "--joints",
            "known.npz",
            "--robot-profile",
            "robot.json",
            "--annotations",
            str(annotations),
            "--segment-id",
            "flip",
            "--skill",
            "pick_up",
            "--qpos-semantics",
            "measured_state",
            "--time-offset",
            "0",
            "--output",
            str(output),
        ]
    )
    call = fitting_oracle["calls"][0]
    np.testing.assert_allclose(call["frame_times"][[0, -1]], [6 / 30, 24 / 30])
    _, report = load_skill_fit(output)
    assert report["annotation_video_interval"] == {"start_frame": 0, "end_frame": 30}
    assert report["annotation_segment"]["core_interval"] == {
        "start_frame": 5,
        "end_frame": 20,
    }


@pytest.mark.parametrize("pickup_factor", [1, 2])
def test_whole_episode_composes_both_fits_and_exports_original_demo_tensors(
    make_video: Callable[..., Path],
    tmp_path: Path,
    fitting_oracle: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    pickup_factor: int,
) -> None:
    """One submission retains both independently fitted cores and all original gaps."""
    import io
    import zipfile
    import numpy as np
    import torch
    import yaml
    from fastapi.testclient import TestClient
    from embodichain.toolkits.trajectory_annotator._server import create_app
    from embodichain.toolkits.trajectory_annotator.video import VideoFrameSource
    from embodichain.lab.gym.utils.gym_utils import config_to_cfg, load_trajectory
    from embodichain.toolkits.skill_fitting import _collision

    fitting_oracle["replay_interpolation_factor"] = pickup_factor
    profile_path = Path(fitting_oracle["profile"])
    profile = json.loads(profile_path.read_text())
    profile["simulation"] = dict(
        root_props=dict(fixed_base=True, min_position_iters=32, min_velocity_iters=8),
        joint_drive_props=dict(
            stiffness={"arm": 70000.0, "hand": 500.0},
            damping={"arm": 1000.0, "hand": 50.0},
        ),
    )
    profile_path.write_text(json.dumps(profile))

    # A supplied object-local convex piece exercises portable collision paths
    # through the production exporter and strict deployment loader.
    parts = tmp_path / "object.obj"
    parts.write_text(
        "o hull\nv 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n"
    )
    scene = tmp_path / "scene.yaml"
    scene.write_text(
        yaml.safe_dump(
            dict(
                objects=dict(
                    target=dict(
                        pose=np.eye(4).tolist(),
                        shape=dict(
                            shape_type="Mesh",
                            fpath="./object.obj",
                            collision=dict(
                                approximation="convex_decomposition",
                                max_hulls=8,
                                decomposition_fpath="./object.obj",
                            ),
                        ),
                    )
                )
            )
        )
    )
    prepare = _collision.prepare_scene_collision
    monkeypatch.setattr(
        _collision,
        "prepare_scene_collision",
        lambda path, **kwargs: prepare(path, cache_root=tmp_path / "cache", **kwargs),
    )

    video = make_video()
    app = create_app(lambda: VideoFrameSource(video), tmp_path / "whole.json")
    with TestClient(app, base_url="http://127.0.0.1") as client:
        deadline = time.monotonic() + 20
        while client.get("/api/status").json()["phase"] != "ready":
            assert time.monotonic() < deadline
            time.sleep(0.01)
        headers = {
            "X-Annotation-Token": client.get("/api/project").json()["session_token"]
        }
        segments = [
            dict(id="second", label="Pickup", start_frame=24, end_frame=29),
            dict(
                id="first",
                label="pick_up",
                start_frame=6,
                end_frame=11,
                core_interval=dict(start_frame=7, end_frame=10),
            ),
        ]
        response = client.put(
            "/api/annotations",
            json=dict(segments=segments, expected_revision=0),
            headers=headers,
        )
        revision = response.json()["revision"]
        payload = dict(
            scope="episode",
            expected_revision=revision,
            inputs=dict(
                joints=fitting_oracle["joints"],
                robot_profile=fitting_oracle["profile"],
                qpos_semantics="measured_state",
                time_offset=0,
                scene=str(scene),
            ),
        )
        response = client.post("/api/fitting/jobs", json=payload, headers=headers)
        assert response.status_code == 202
        job = response.json()["job_id"]
        while (state := client.get("/api/fitting").json())["phase"] == "running":
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert state["phase"] == "done", state
        assert state["accepted"] and state["run_task_available"], state
        assert len(fitting_oracle["calls"]) == 2
        with np.load(
            io.BytesIO(client.get(f"/api/fitting/jobs/{job}/bundle").content)
        ) as result:
            original, patched = result["original_qpos"], result["patched_qpos"]
            expected = original.copy()
            expected[27:34, 0] += 0.01
            expected[83:97, 0] += 0.01
            np.testing.assert_array_equal(patched, expected)
            np.testing.assert_array_equal(original, fitting_oracle["original"])
        response = client.get(f"/api/fitting/jobs/{job}/run-task")
        assert response.status_code == 200
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            manifest = yaml.safe_load(archive.read("skills.yaml"))
            timeline = manifest["timeline"]
            assert [(s["skill_id"], s["start"], s["stop"]) for s in timeline] == [
                ("execute_demo_skill", 0, 20),
                ("pick_up", 20, 40),
                ("execute_demo_skill", 40, 80),
                ("pick_up", 80, 100),
                ("execute_demo_skill", 100, 300),
            ]
            for span in timeline:
                if span["kind"] == "atomic_skill":
                    assert "tensor" not in span
                    assert span["execution"] == "native_atomic_skill"
                    assert span["control_parts"] == {
                        "primary": {"motion": "arm", "grasp": "hand"}
                    }
                    assert span["options"]["lift_height"] == 0.12
                    assert span["duration_scale"] == pickup_factor
                    continue
                data = torch.load(
                    io.BytesIO(archive.read(span["tensor"])), weights_only=True
                )
                np.testing.assert_array_equal(
                    data["qpos"].numpy(), original[span["start"] : span["stop"]].T
                )
                assert data["layout"] == "DOF x n_frame"
            moved = tmp_path / "moved_bundle"
            archive.extractall(moved)
        assert not list(moved.glob("skills/*pick_up.pt"))
        assert not (moved / "trajectory.pt").exists()
        assert not (moved / "qpos.pt").exists()
        task = moved / "task.yaml"
        cfg = config_to_cfg(yaml.safe_load(task.read_text()), source_path=task)
        assert cfg.robot.control_parts == {"arm": ["arm_joint"], "hand": ["hand_joint"]}
        assert cfg.robot.root_props.min_position_iters == 32
        assert cfg.robot.root_props.min_velocity_iters == 8
        assert cfg.robot.joint_drive_props.stiffness == {"arm": 70000.0, "hand": 500.0}
        assert (
            json.loads((moved / "robot_profile.json").read_text())["simulation"]
            == profile["simulation"]
        )
        assert Path(cfg.robot.fpath).is_relative_to(moved)
        assert cfg.rigid_object[0].body_type == "dynamic"
        collider = Path(cfg.rigid_object[0].shape.collision.decomposition_fpath)
        assert collider.is_relative_to(moved) and collider.is_file()
        assert manifest["scene_collision"]["target"]["hull_count"] == 1
        # Changing either annotation invalidates the whole result and both exports.
        segments[0]["end_frame"] = 30
        client.put(
            "/api/annotations",
            json=dict(segments=segments, expected_revision=revision),
            headers=headers,
        )
        for endpoint in ("bundle", "run-task", "preview"):
            assert client.get(f"/api/fitting/jobs/{job}/{endpoint}").status_code == 409


def test_unannotated_episode_preserves_irregular_source_tensor_and_exports_control_clock(
    tmp_path: Path, fitting_oracle: dict[str, object]
) -> None:
    """Demo-only recordings retain raw evidence independently of replay sampling."""
    import numpy as np
    import torch
    from embodichain.toolkits.skill_fitting._episode import fit_episode
    from embodichain.toolkits.skill_fitting._export import export_run_task
    from embodichain.toolkits.skill_fitting._pipeline import FitInputs
    import yaml

    times = np.arange(300, dtype=np.float64) / 100
    times[1] = 0.017
    original = fitting_oracle["original"]
    recording = tmp_path / "irregular.npz"
    np.savez(recording, qpos=original, timestamps=times)
    inputs = FitInputs(
        joints=str(recording),
        robot_profile=fitting_oracle["profile"],
        time_offset=0,
        qpos_semantics="measured_state",
    )
    document = dict(
        source=dict(frame_count=90, identity="known", timing="indexed"),
        revision=0,
        segments=[],
    )

    def no_frame_needed(first: int, last: int):
        pytest.fail("A Demo-only recording does not need a video selection clock.")

    arrays, report = fit_episode(inputs, document, no_frame_needed)
    assert report["accepted"] and report["atomic_skill_count"] == 0
    assert report["demo_skill_count"] == 1
    directory = tmp_path / "demo_only"
    export_run_task(directory, arrays, report, inputs)
    raw = torch.load(directory / "skills/000.execute_demo_skill.pt", weights_only=True)
    np.testing.assert_array_equal(raw["qpos"].numpy(), original.T)
    np.testing.assert_array_equal(raw["timestamps"].numpy(), times)
    manifest = yaml.safe_load((directory / "skills.yaml").read_text())
    assert manifest["schema"] == "native-skill-sequence/v1"
    assert manifest["step_dt"] == pytest.approx(0.01)
    assert not (directory / "trajectory.pt").exists()
    document["segments"] = [dict(id="pour", label="Pour", start_frame=5, end_frame=15)]
    with pytest.raises(ValueError, match="Pour.*尚不支持"):
        fit_episode(inputs, document, no_frame_needed)


@pytest.mark.parametrize("include_video", [False, True])
def test_portable_fitting_inputs_survive_move_without_original_resources(
    tmp_path, fitting_oracle, make_video, monkeypatch, include_video
):
    import shutil
    import numpy as np
    import pytorch_kinematics as pk
    import yaml
    from embodichain.toolkits.skill_fitting._bundle import (
        pack_inputs,
        load_inputs,
        load_annotation_inputs,
    )
    from embodichain.toolkits.skill_fitting._pipeline import FitInputs

    mesh = tmp_path / "local.obj"
    mesh.write_text(
        "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n"
    )
    pose = np.eye(4)
    pose[:3, 3] = [0.3, -0.2, 0.7]
    scene = tmp_path / "scene.yaml"
    scene.write_text(
        yaml.safe_dump(
            dict(
                objects=dict(
                    cube=dict(
                        pose=pose.tolist(),
                        body_type="dynamic",
                        shape=dict(
                            shape_type="Mesh",
                            fpath=mesh.name,
                            collision=dict(approximation="convex_hull"),
                        ),
                    )
                )
            )
        )
    )
    inputs = FitInputs(
        joints=fitting_oracle["joints"],
        robot_profile=fitting_oracle["profile"],
        scene=str(scene),
        time_offset=0.125,
        qpos_semantics="measured_state",
    )
    document = dict(
        source=dict(frame_count=3, identity="vfr", timing="pts"),
        revision=1,
        segments=[],
    )
    clock = np.array([0, 0.03, 0.08])
    durations = np.array([0.03, 0.05, 0.04])
    video = None
    if include_video:
        from embodichain.toolkits.trajectory_annotator.video import VideoFrameSource
        from embodichain.toolkits.trajectory_annotator._cfg import AnnotationCfg

        video = make_video(count=3, variable=True, start=90)
        source = VideoFrameSource(video)
        try:
            info = source.prepare(threading.Event(), lambda **_: None)
            store = AnnotationStore(
                tmp_path / "initial.json", AnnotationCfg().document_bytes
            )
            store.initialize(info)
            document = store.snapshot()["document"]
            frames = [source.get_frame(i) for i in range(3)]
            clock = np.array([f.time_seconds for f in frames])
            durations = np.array([f.delay_seconds for f in frames])
        finally:
            source.close()
    packed = pack_inputs(
        tmp_path / "packed", inputs, document, clock, durations, video=video
    )
    moved = tmp_path / "moved"
    shutil.move(packed, moved)
    scene.unlink()
    mesh.unlink()
    Path(fitting_oracle["profile"]).unlink()
    Path(fitting_oracle["joints"]).unlink()
    if video is not None:
        video.unlink()
    loaded, saved, times, display_durations = load_inputs(moved)
    np.testing.assert_array_equal(times, clock)
    np.testing.assert_array_equal(display_durations, durations)
    assert loaded.time_offset == 0.125
    assert saved["segments"] == document["segments"]
    assert saved["source"]["identity"] == document["source"]["identity"]
    with np.load(loaded.joints) as data:
        np.testing.assert_array_equal(data["qpos"], fitting_oracle["original"])
    scene_data = yaml.safe_load(Path(loaded.scene).read_text())
    np.testing.assert_array_equal(scene_data["objects"]["cube"]["pose"], pose)
    # Exercise the actual URDF string consumer: XML encoding headers broke moved fits.
    pk.build_chain_from_urdf((moved / "robot/robot.urdf").read_text())
    assert not list(moved.rglob("export_report.json"))
    if include_video:
        from fastapi.testclient import TestClient
        from embodichain.toolkits.trajectory_annotator import _cli, app
        from embodichain.toolkits.trajectory_annotator._server import create_app

        launches = []

        def launch(factory, annotations, **kwargs):
            defaults = kwargs["fitting_options"]
            for key in ("joints", "robot_profile", "scene"):
                assert Path(defaults[key]).is_relative_to(moved)
            assert defaults["time_offset"] == 0.125
            with TestClient(
                create_app(factory, annotations, fitting_options=defaults),
                base_url="http://127.0.0.1",
            ) as client:
                deadline = time.monotonic() + 10
                while client.get("/api/status").json()["phase"] != "ready":
                    assert time.monotonic() < deadline
                    time.sleep(0.01)
                project = client.get("/api/project").json()
                state = project["document"]
                if not launches:
                    response = client.put(
                        "/api/annotations",
                        json=dict(
                            expected_revision=state["revision"],
                            segments=[
                                dict(
                                    id="pickup",
                                    label="pick_up",
                                    start_frame=0,
                                    end_frame=2,
                                )
                            ],
                        ),
                        headers={"X-Annotation-Token": project["session_token"]},
                    )
                    assert response.status_code == 200
                else:
                    assert state["segments"][0]["id"] == "pickup"
            launches.append(annotations)

        monkeypatch.setattr(app, "launch_annotation", launch)
        for _ in range(2):
            _cli.cli(["--bundle", str(moved), "--no-browser"])
        assert launches == [moved / "annotations.edited.json"] * 2
        # Editing is persistent, while the immutable seed and inventory still validate.
        assert load_inputs(moved)[1]["segments"] == []
        from embodichain.toolkits.skill_fitting.cli import main as fit_cli

        repacked = tmp_path / "repacked"
        fit_cli(["--bundle", str(moved), "--package-output", str(repacked)])
        assert load_annotation_inputs(repacked)[0].is_relative_to(repacked)
        with pytest.raises(ValueError, match="must not overwrite"):
            load_annotation_inputs(moved, moved / "joints.npz")
        with pytest.raises(SystemExit):
            _cli.cli(["--bundle", str(moved), "--joints", "external.npz"])
        packed_video = next((moved / "video").iterdir())
        packed_video.write_bytes(b"changed video")
        with pytest.raises(ValueError, match="Bundle resource changed: video/"):
            load_annotation_inputs(moved)
    else:
        assert not list(moved.rglob("*.mp4"))
        with pytest.raises(ValueError, match="offline-only bundle has no video"):
            load_annotation_inputs(moved)
    (moved / "joints.npz").write_bytes(b"changed recording")
    with pytest.raises(ValueError, match="Bundle resource changed: joints.npz"):
        load_inputs(moved)


def test_scene_import_keeps_video_requests_live_and_reports_failure(
    make_video, tmp_path, monkeypatch
):
    """A slow scene preparation owns no decoder/request thread; errors are visible."""
    from fastapi.testclient import TestClient
    from embodichain.toolkits.skill_fitting import _collision
    from embodichain.toolkits.trajectory_annotator._server import create_app
    from embodichain.toolkits.trajectory_annotator.video import VideoFrameSource

    source = tmp_path / "scene.yaml"
    source.write_text("objects: {}\n")
    entered, release = threading.Event(), threading.Event()

    def prepare(path, *, progress):
        entered.set()
        assert release.wait(5)
        raise ValueError("invalid collision mesh")

    monkeypatch.setattr(_collision, "prepare_scene_collision", prepare)
    video = make_video()
    app = create_app(lambda: VideoFrameSource(video), tmp_path / "annotations.json")
    with TestClient(app, base_url="http://127.0.0.1") as client:
        deadline = time.monotonic() + 10
        while client.get("/api/status").json()["phase"] != "ready":
            assert time.monotonic() < deadline
            time.sleep(0.01)
        headers = {
            "X-Annotation-Token": client.get("/api/project").json()["session_token"]
        }
        payload = dict(scene=str(source), joints="")
        try:
            assert (
                client.post(
                    "/api/fitting/scene", json=payload, headers=headers
                ).status_code
                == 202
            )
            assert entered.wait(5)
            assert client.get("/api/fitting/scene").json()["phase"] == "running"
            assert client.get("/api/frames/5").status_code == 200
            assert (
                client.post(
                    "/api/fitting/scene",
                    json={**payload, "joints": "other"},
                    headers=headers,
                ).status_code
                == 409
            )
        finally:
            release.set()
        while (state := client.get("/api/fitting/scene").json())["phase"] == "running":
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert state["phase"] == "error" and state["error"] == "invalid collision mesh"
