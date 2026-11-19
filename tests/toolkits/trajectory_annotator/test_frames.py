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

"""Image-source correctness, lazy HDF5 reads and source-independent annotation."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import json
import threading
import time
from typing import Any

import numpy as np
import pytest

from embodichain.toolkits.trajectory_annotator.frames import ArrayFrameSource

pytestmark = pytest.mark.no_sim


@pytest.fixture
def image_hdf5(tmp_path: Path) -> Callable[..., tuple[Path, np.ndarray]]:
    """Write distinctive red/green/blue frames in common recording layouts."""
    h5py = pytest.importorskip("h5py")
    cv2 = pytest.importorskip("cv2")

    def make(
        layout: str = "jpeg", key: str = "observations/images/cam_high"
    ) -> tuple[Path, np.ndarray]:
        path = tmp_path / f"{layout}.hdf5"
        images = np.zeros((4, 24, 32, 3), dtype=np.uint8)
        images[0, :, :, 0] = 230
        images[1, :, :, 1] = 190
        images[2, :, :, 2] = 150
        images[3] = (30, 80, 210)
        with h5py.File(path, "w") as handle:
            handle.attrs["metadata_json"] = json.dumps({"record_fps": 20})
            if layout in {"raw_rgb", "raw_bgr"}:
                data = images if layout == "raw_rgb" else images[..., ::-1]
                handle.create_dataset(key, data=data)
            else:
                encoded = [
                    cv2.imencode(
                        ".jpg" if layout == "jpeg" else ".png", image[:, :, ::-1]
                    )[1]
                    for image in images
                ]
                if layout == "jpeg":
                    dataset = handle.create_dataset(
                        key, (4,), dtype=h5py.vlen_dtype(np.uint8)
                    )
                    for index, payload in enumerate(encoded):
                        dataset[index] = payload
                else:
                    padded = np.zeros((4, max(map(len, encoded)) + 16), dtype=np.uint8)
                    for index, payload in enumerate(encoded):
                        padded[index, : len(payload)] = payload
                    handle.create_dataset(key, data=padded)
        return path, images

    return make


@pytest.mark.parametrize("layout", ["jpeg", "padded_png", "raw_rgb", "raw_bgr"])
def test_hdf5_exact_indices_colors_timing_and_lazy_reads(
    image_hdf5: Callable[..., tuple[Path, np.ndarray]],
    layout: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import h5py
    from embodichain.toolkits.trajectory_annotator.hdf5 import Hdf5FrameSource

    path, expected = image_hdf5(layout)
    reads: list[object] = []
    original = h5py.Dataset.__getitem__

    def tracked(self: h5py.Dataset, index: object) -> Any:
        reads.append(index)
        return original(self, index)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", tracked)
    source = Hdf5FrameSource(path, color_order="bgr" if layout == "raw_bgr" else "rgb")
    try:
        info = source.prepare(threading.Event(), lambda **_: None)
        assert reads == [0]  # Preparation must not decode/materialize the sequence.
        assert (info.frame_count, info.width, info.height, info.fps) == (4, 32, 24, 20)
        assert info.timing == "fixed_fps"
        for index in (3, 0, 2, 1):
            frame = source.get_frame(index)
            assert reads[-1] == index
            assert frame.index == index and frame.pts is None
            assert frame.time_seconds == pytest.approx(index / 20)
            assert frame.delay_seconds == pytest.approx(0.05)
            assert np.abs(frame.image.astype(float) - expected[index]).max() <= (
                2 if layout == "jpeg" else 0
            )
        for index in (-1, 4):
            with pytest.raises(ValueError):
                source.get_frame(index)
    finally:
        source.close()


def test_hdf5_camera_fps_identity_resume_and_missing_metadata(
    image_hdf5: Callable[..., tuple[Path, np.ndarray]],
    tmp_path: Path,
) -> None:
    import h5py
    from embodichain.toolkits.trajectory_annotator.hdf5 import Hdf5FrameSource
    from embodichain.toolkits.trajectory_annotator._storage import AnnotationStore

    # Singular observation is accepted when present, without rewriting an explicit key.
    key = "observation/images/cam_high"
    path, images = image_hdf5("raw_rgb", key)
    with h5py.File(path, "a") as handle:
        handle.create_dataset("observation/images/cam_left_wrist", data=images)

    def prepare(**kwargs: Any) -> Any:
        source = Hdf5FrameSource(path, **kwargs)
        try:
            return source.prepare(threading.Event(), lambda **_: None)
        finally:
            source.close()

    info = prepare()
    assert info.details["image_key"] == key
    annotations = tmp_path / "labels.json"
    store = AnnotationStore(annotations, 1024**2)
    store.initialize(info)
    saved = store.save(
        [dict(id="flip", label="翻转包裹", start_frame=1, end_frame=3)], 0
    )
    restored = AnnotationStore(annotations, 1024**2)
    restored.initialize(prepare())
    assert restored.snapshot()["document"] == saved
    for options in ({"fps": 30}, {"image_key": "observation/images/cam_left_wrist"}):
        with pytest.raises(ValueError, match="不匹配"):
            AnnotationStore(annotations, 1024**2).initialize(prepare(**options))
    with pytest.raises(ValueError, match="找不到"):
        prepare(image_key="observations/images/cam_high")
    with pytest.raises(ValueError, match="相同"):
        AnnotationStore(path, 1024**2).initialize(info)
    with h5py.File(path, "a") as handle:
        del handle.attrs["metadata_json"]
    with pytest.raises(ValueError, match="--fps"):
        prepare()
    assert prepare(fps=25).fps == 25
    with pytest.raises(ValueError, match="不匹配"):
        AnnotationStore(annotations, 1024**2).initialize(prepare(fps=20))


def test_array_input_uses_same_http_save_path_and_worker_lifecycle(
    tmp_path: Path,
) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    import cv2
    from embodichain.toolkits.trajectory_annotator.app import create_annotation_app

    images = np.zeros((3, 24, 32, 3), dtype=np.uint8)
    images[0, :, :, 0] = 240
    images[1, :, :, 1] = 210
    images[2, :, :, 2] = 180
    owners: list[int] = []
    closed = threading.Event()

    class TrackedSource(ArrayFrameSource):
        def prepare(self, *args: Any, **kwargs: Any) -> Any:
            owners.append(threading.get_ident())
            return super().prepare(*args, **kwargs)

        def get_frame(self, index: int) -> Any:
            owners.append(threading.get_ident())
            return super().get_frame(index)

        def close(self) -> None:
            owners.append(threading.get_ident())
            closed.set()

    def factory() -> TrackedSource:
        owners.append(threading.get_ident())
        return TrackedSource(images, fps=10, name="RGB images")

    destination = tmp_path / "images.json"
    app = create_annotation_app(factory, destination)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        deadline = time.monotonic() + 5
        while (status := client.get("/api/status").json())["phase"] != "ready":
            assert status["phase"] != "error", status
            assert time.monotonic() < deadline
            time.sleep(0.01)
        project = client.get("/api/project").json()
        assert project["document"]["source"]["kind"] == "images"
        assert not destination.exists()
        response = client.get("/api/frames/2")
        pixels = cv2.imdecode(
            np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR
        )
        assert np.abs(pixels.astype(float) - images[2, :, :, ::-1]).max() <= 2
        assert response.headers["X-Frame-Time"] == "0.2"
        assert response.headers["X-Frame-PTS"] == ""
        response = client.put(
            "/api/annotations",
            headers={"X-Annotation-Token": project["session_token"]},
            json={
                "segments": [
                    dict(id="one", label="原子技能", start_frame=0, end_frame=2)
                ],
                "expected_revision": 0,
            },
        )
        assert response.status_code == 200
    assert closed.is_set() and len(set(owners)) == 1
    assert owners[0] != threading.get_ident()
    assert json.loads(destination.read_text())["segments"][0]["end_frame"] == 2
    owners.clear()
    closed.clear()
    # A second lifespan must obtain a fresh source, not reuse a cancelled worker.
    with TestClient(app, base_url="http://127.0.0.1") as client:
        deadline = time.monotonic() + 5
        while (status := client.get("/api/status").json())["phase"] != "ready":
            assert status["phase"] != "error", status
            assert time.monotonic() < deadline
            time.sleep(0.01)
        restored = client.get("/api/project").json()
        assert restored["persisted"]
        assert restored["document"]["segments"][0]["end_frame"] == 2
    assert closed.is_set() and len(set(owners)) == 1


def test_owned_previews_survive_reused_frame_buffers() -> None:
    from embodichain.toolkits.trajectory_annotator._cfg import AnnotationCfg
    from embodichain.toolkits.trajectory_annotator._frames import FrameReader
    from embodichain.toolkits.trajectory_annotator.frames import ImageFrame

    # A reusable source buffer is permitted until the next read; previews must own bytes.
    images = np.zeros((2, 24, 32, 3), dtype=np.uint8)
    images[1, :, :, 1] = 220

    class ReusingSource(ArrayFrameSource):
        buffer = np.empty_like(images[0])

        def get_frame(self, index: int) -> ImageFrame:
            self.buffer[:] = images[index]
            return ImageFrame(index, self.buffer, index / 20, 0.05)

    source = ReusingSource(images, fps=20)
    reader = FrameReader(source, AnnotationCfg(cache_bytes=0))
    try:
        reader.prepare()
        first = reader.preview(0)
        previous = first.jpeg
        second = reader.preview(1)
        assert first.jpeg == previous and first.jpeg != second.jpeg
    finally:
        reader.close()
