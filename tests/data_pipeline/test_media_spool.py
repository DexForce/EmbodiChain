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

"""Sensor chunk ownership, bounded buffering and complete episode alignment."""

from __future__ import annotations
import json
import numpy as np
import pytest
from embodichain.data_pipeline.recording.media import MediaSpool, verify_media_manifest


def test_chunks_are_bounded_owned_and_cover_terminal_observation(tmp_path):
    frames = {
        "camera/rgb": np.zeros((2, 3, 3), dtype=np.uint8),
        "camera/depth": np.zeros((2, 3), dtype=np.float32),
    }
    size = sum(value.nbytes for value in frames.values())
    writer = MediaSpool(
        tmp_path / "media",
        max_buffer_bytes=size * 2,
        parent_episode_id="episode",
        calibration={"camera": {"fx": 100}},
    )
    for i in range(5):
        frames["camera/rgb"].fill(i)
        writer.append(i * 0.02, frames)
        assert writer.buffered_bytes <= size * 2
    frames["camera/rgb"].fill(99)
    path = writer.finish()
    manifest = verify_media_manifest(
        path, expected_frames=5, parent_episode_id="episode"
    )
    assert [chunk["count"] for chunk in manifest["chunks"]] == [2, 2, 1]
    with np.load(path.parent / manifest["chunks"][-1]["path"]) as data:
        assert np.all(data["feature_1"] == 4)
    assert writer.finish() == path
    with pytest.raises(RuntimeError):
        writer.append(0.1, frames)


def test_oversized_frame_and_changed_schema_fail_before_buffering(tmp_path):
    writer = MediaSpool(
        tmp_path / "media",
        max_buffer_bytes=8,
        parent_episode_id="episode",
        calibration={},
    )
    with pytest.raises(BufferError):
        writer.append(0, {"rgb": np.zeros((3, 3, 3), dtype=np.uint8)})
    assert writer.buffered_bytes == 0
    writer.append(0, {"depth": np.zeros((1, 1), dtype=np.float32)})
    with pytest.raises(ValueError, match="schema"):
        writer.append(1, {"depth": np.zeros((1, 2), dtype=np.float32)})
    assert writer.buffered_bytes == 4


def test_media_integrity_and_parent_are_verified(tmp_path):
    writer = MediaSpool(
        tmp_path / "media",
        max_buffer_bytes=8,
        parent_episode_id="episode",
        calibration={},
    )
    writer.append(0, {"depth": np.zeros((1, 1), dtype=np.float32)})
    path = writer.finish()
    with pytest.raises(ValueError, match="parent"):
        verify_media_manifest(path, expected_frames=1, parent_episode_id="other")
    chunk = path.parent / "chunk-000000.npz"
    chunk.write_bytes(chunk.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="content"):
        verify_media_manifest(path, expected_frames=1, parent_episode_id="episode")
