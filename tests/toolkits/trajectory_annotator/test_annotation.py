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
    saved = store.save([skill_interval()], 0)
    original = path.read_bytes()
    assert saved["revision"] == 1
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
