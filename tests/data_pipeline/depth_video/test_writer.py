# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
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

"""Tests for the depth video writer, sidecar manager, and reader."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import av
import numpy as np
import pytest

from embodichain.data_pipeline.depth_video import (
    DEPTH_METER_UNIT,
    DepthCodecError,
    DepthSidecarManager,
    DepthVideoCfg,
    DepthVideoLibrary,
    detect_depth_encoder,
    load_depth_meta,
    quantize_depth,
)

# These tests encode real MP4s with libx265; skip the whole module if no HEVC
# encoder is available in the bundled FFmpeg build.
pytestmark = pytest.mark.skipif(
    detect_depth_encoder("libx265") is None,
    reason="libx265/hevc encoder not available in this FFmpeg build",
)

H, W = 64, 80
DEPTH_MIN, DEPTH_MAX, SHIFT = 0.1, 3.0, 1.0


def _make_depth(n_frames: int) -> list[np.ndarray]:
    """Return ``n_frames`` distinct float32 depth maps in metres.

    Depths stay strictly inside ``[DEPTH_MIN, DEPTH_MAX]`` so no clamping occurs
    (clamping is exercised separately in ``test_depth_utils``).
    """
    base = np.linspace(
        DEPTH_MIN + 0.05, DEPTH_MAX - 0.05, H * W, dtype=np.float32
    ).reshape(H, W)
    return [base + 0.01 * i for i in range(n_frames)]


def _decode_codes(mp4_path: Path) -> list[np.ndarray]:
    """Decode a sidecar MP4 back to its raw 12-bit uint16 code frames."""
    codes: list[np.ndarray] = []
    with av.open(str(mp4_path), "r") as inp:
        stream = next(s for s in inp.streams if s.type == "video")
        for frame in inp.decode(stream):
            codes.append(frame.to_ndarray(format="gray12le"))
    return codes


class TestDepthVideoWriter:
    """Writer-level encode/decode tests."""

    def test_lossless_codes_bit_exact(self, tmp_path: Path):
        """Lossless libx265 preserves the 12-bit codes exactly."""
        from embodichain.data_pipeline.depth_video import DepthVideoWriter

        cfg = DepthVideoCfg(
            enable=True,
            depth_min=DEPTH_MIN,
            depth_max=DEPTH_MAX,
            shift=SHIFT,
            use_log=True,
            lossless=True,
            input_unit=DEPTH_METER_UNIT,
            output_unit=DEPTH_METER_UNIT,
        )
        depths = _make_depth(3)
        writer = DepthVideoWriter(tmp_path / "ep.mp4", fps=10, cfg=cfg)
        for d in depths:
            writer.add_frame(d)
        out = writer.close()
        assert out.exists()

        decoded = _decode_codes(out)
        assert len(decoded) == 3
        for orig, dec in zip(depths, decoded):
            expected = quantize_depth(
                orig,
                depth_min=DEPTH_MIN,
                depth_max=DEPTH_MAX,
                shift=SHIFT,
                use_log=True,
                video_backend=None,
                input_unit=DEPTH_METER_UNIT,
            )
            assert np.array_equal(dec, expected)

    def test_close_with_no_frames_raises_and_cleans_temp(self, tmp_path: Path):
        from embodichain.data_pipeline.depth_video import DepthVideoWriter

        cfg = DepthVideoCfg(enable=True, input_unit=DEPTH_METER_UNIT)
        writer = DepthVideoWriter(tmp_path / "empty.mp4", fps=10, cfg=cfg)
        with pytest.raises(RuntimeError, match="no frames"):
            writer.close()
        # No partial mp4 left in the directory.
        assert not (tmp_path / "empty.mp4").exists()
        assert not any(p.suffix == ".mp4" for p in tmp_path.iterdir())


class TestDepthSidecarManager:
    """Sidecar manager + metadata + reader integration tests."""

    def _make_manager(self, root: Path) -> DepthSidecarManager:
        cfg = DepthVideoCfg(
            enable=True,
            depth_min=DEPTH_MIN,
            depth_max=DEPTH_MAX,
            shift=SHIFT,
            use_log=True,
            lossless=True,
            input_unit=DEPTH_METER_UNIT,
            output_unit=DEPTH_METER_UNIT,
        )
        mgr = DepthSidecarManager(root, fps=20, cfg=cfg)
        mgr.register_sensor("camera", (H, W))
        mgr.register_sensor("camera_right", (H, W))
        return mgr

    def test_writes_per_sensor_videos_and_metadata(self, tmp_path: Path):
        mgr = self._make_manager(tmp_path)
        depths = _make_depth(4)
        mgr.start_episode(0, ["camera", "camera_right"])
        for d in depths:
            mgr.add_frame("camera", d)
            mgr.add_frame("camera_right", d * 0.9)
        mgr.end_episode(0)
        meta_path = mgr.finalize()
        assert meta_path == tmp_path / "depth_meta.json"
        assert (tmp_path / "depth_videos" / "camera" / "episode_000000.mp4").exists()
        assert (
            tmp_path / "depth_videos" / "camera_right" / "episode_000000.mp4"
        ).exists()

        meta = json.loads(meta_path.read_text())
        assert meta["is_depth_map"] is True
        assert set(meta["sensors"]) == {"camera", "camera_right"}
        cam = meta["sensors"]["camera"]
        assert cam["video.pix_fmt"] == "gray12le"
        assert cam["video.depth_min"] == DEPTH_MIN
        assert cam["video.use_log"] is True
        assert cam["episodes"]["0"]["frame_count"] == 4
        assert cam["episodes"]["0"]["file"].startswith("depth_videos/camera/")

    def test_reader_roundtrip_matches_original(self, tmp_path: Path):
        """The reader dequantizes back to depths close to the originals."""
        mgr = self._make_manager(tmp_path)
        depths = _make_depth(3)
        mgr.start_episode(0, ["camera"])
        for d in depths:
            mgr.add_frame("camera", d)
        mgr.end_episode(0)
        mgr.finalize()

        lib = DepthVideoLibrary(tmp_path, load_depth_meta(tmp_path))
        assert lib.sensors == ["camera", "camera_right"]
        assert lib.frame_count(0, "camera") == 3
        for i, orig in enumerate(depths):
            dec = lib.get(0, "camera", i).squeeze()
            assert dec.shape == (H, W)
            assert float(np.abs(dec - orig).max()) < 5e-3

    def test_abort_episode_removes_partial_files(self, tmp_path: Path):
        mgr = self._make_manager(tmp_path)
        depths = _make_depth(2)
        mgr.start_episode(0, ["camera"])
        for d in depths:
            mgr.add_frame("camera", d)
        # Simulate a failed save_episode(): abort instead of end_episode.
        mgr.abort_episode()
        # No finalised mp4 should exist for the aborted episode.
        assert not (
            tmp_path / "depth_videos" / "camera" / "episode_000000.mp4"
        ).exists()

    def test_multiple_episodes_accumulate_in_metadata(self, tmp_path: Path):
        mgr = self._make_manager(tmp_path)
        for ep in range(3):
            depths = _make_depth(2)
            mgr.start_episode(ep, ["camera"])
            for d in depths:
                mgr.add_frame("camera", d)
            mgr.end_episode(ep)
        mgr.finalize()
        meta = json.loads((tmp_path / "depth_meta.json").read_text())
        assert sorted(meta["sensors"]["camera"]["episodes"]) == ["0", "1", "2"]

    def test_end_episode_surfaces_encoder_close_failure(self, tmp_path: Path):
        """A missing committed depth video is a visible durability failure."""
        mgr = self._make_manager(tmp_path)
        writer = MagicMock()
        writer.close.side_effect = OSError("encoder failed")
        mgr._writers = {"camera": writer}
        mgr._episode_sensors = ["camera"]

        with pytest.raises(RuntimeError, match="encoder failed"):
            mgr.end_episode(0)

        writer.abort.assert_called_once_with()
        assert mgr._writers == {}


class TestCodec:
    """Codec detection tests."""

    def test_detect_returns_available_encoder(self):
        # Skipped on hosts without libx265 via the module-level pytestmark.
        assert detect_depth_encoder("libx265") == "libx265"

    def test_resolve_raises_when_unavailable(self, monkeypatch):
        from embodichain.data_pipeline.depth_video import codec as codec_mod

        monkeypatch.setattr(codec_mod, "get_codec", lambda name: None)
        with pytest.raises(DepthCodecError):
            codec_mod.resolve_depth_vcodec("libx265")
