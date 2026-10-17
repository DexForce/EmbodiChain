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

"""Presentation evidence and preservation of recorded physics frames."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from embodichain.lab.scripts._robot_workspace_presentation import _write_walkthrough
from embodichain.lab.scripts._robot_workspace_view import _Display

pytestmark = pytest.mark.no_sim


@contextmanager
def _context(value):
    yield value


def _report(tmp_path):
    cache = tmp_path / "cache.npz"
    np.savez(cache, workspace_points=np.array([[-0.5, 0, 0.8], [0.5, 0.5, 1.0]]))
    reachability = tmp_path / "reachability.npz"
    np.savez(
        reachability,
        points=np.array([[-0.5, 0, 0.775], [-1.9, 0, 0.775]]),
        reachable=np.array([True, False]),
        colors=np.array([[40, 220, 100], [245, 65, 65]], dtype=np.uint8),
    )
    pose = np.eye(4)
    pose[:3, 3] = (-0.5, 0, 0.775)
    return {
        "mode": "task",
        "layout": {
            "table_xy_bounds": [[-2.2, -0.4], [0.2, 0.7]],
            "candidate_xy_bounds": [[-0.55, -0.18], [-0.30, 0.12]],
            "robot_base": [0, 0, 0.75],
            "goal": [-0.4, 0.48, 0.775],
            "negative": [-1.9, 0, 0.775],
        },
        "workspace": {
            "cache": str(cache),
            "requested_cache_samples": 8192,
            "cache_samples": 2,
            "returned_samples": 1,
            "reachability": {
                "cache": str(reachability),
                "object_center_height_m": 0.775,
                "reachable_count": 1,
                "rejected_count": 1,
            },
        },
        "candidates": [
            {
                "adjusted_pose": pose.tolist(),
                "cache_index": 0,
                "scene": {
                    "accepted": True,
                    "initial_validation": [
                        {
                            "check_id": "workspace.pose_ik",
                            "status": "passed",
                            "metrics": {},
                        },
                    ],
                },
            }
        ],
        "negative": {
            "initial_validation": [
                {"check_id": "scene.support", "status": "passed"},
                {"check_id": "workspace.pose_ik", "status": "failed"},
            ],
        },
        "negative_executed_steps": 0,
        "video_events": [
            {
                "after_frame": 1,
                "stage": "candidate_0_initial",
                "keyframe": "initial.png",
            },
            {
                "after_frame": 2,
                "stage": "negative_rejected",
                "keyframe": "negative.png",
            },
        ],
    }


def test_explanation_keeps_every_source_frame_and_shows_rejection_without_actions(
    tmp_path, monkeypatch
):
    import imageio.v2 as imageio

    report = _report(tmp_path)
    for name, color in (("initial.png", 30), ("negative.png", 40)):
        Image.fromarray(np.full((480, 640, 3), color, dtype=np.uint8)).save(
            tmp_path / name
        )
    frames = [np.full((480, 640, 3), color, dtype=np.uint8) for color in (10, 20)]

    class Reader:
        def get_meta_data(self):
            return {"fps": 2}

        def __iter__(self):
            return iter(frames)

    written = []
    monkeypatch.setattr(imageio, "get_reader", lambda path: _context(Reader()))
    monkeypatch.setattr(
        imageio,
        "get_writer",
        lambda path, **kwargs: _context(SimpleNamespace(append_data=written.append)),
    )
    before = [frame.copy() for frame in frames]
    result = _write_walkthrough(report, tmp_path)
    # The camera rectangle contains every original frame once, in order. All
    # inserted holds are separately counted; the negative has no action frame.
    source = [
        tuple(frame[120, 30])
        for frame in written
        if tuple(frame[120, 30]) in ((10, 10, 10), (20, 20, 20))
    ]
    assert source == [(10, 10, 10), (20, 20, 20)]
    assert result["source_frames"] == 2
    assert result["inspection_frames"] == 17
    assert len(written) == 19
    assert result["physics_steps_added"] == 0
    assert Image.open(tmp_path / result["rejection"]).size == (1280, 720)
    assert Image.open(tmp_path / result["green_start"]).size == (1280, 720)
    for original, unchanged in zip(frames, before, strict=True):
        np.testing.assert_array_equal(original, unchanged)


def test_capture_records_negative_even_without_another_control_step(tmp_path):
    calls, raw_frames = [], []
    rgb = torch.zeros((1, 480, 640, 4), dtype=torch.uint8)
    display = _Display("offscreen", tmp_path)
    (tmp_path / "frames").mkdir()
    display.camera = SimpleNamespace(
        update=lambda: None, get_data=lambda: {"color": rgb}
    )
    display.writer = SimpleNamespace(
        append_data=lambda frame: raw_frames.append(frame.copy())
    )

    @contextmanager
    def render_frame(**kwargs):
        calls.append("render")
        yield

    sim = SimpleNamespace(
        render_frame=render_frame, update=lambda **kwargs: calls.append("physics")
    )
    display.frame(sim, "candidate_0_initial", force=True)
    rgb[:] = 200
    display.frame(sim, "negative_rejected", force=True)
    assert calls == ["render", "render"]
    assert display.video_frames == len(raw_frames) == 1
    assert [event["after_frame"] for event in display.video_events] == [1, 1]
    negative = np.asarray(Image.open(tmp_path / display.video_events[-1]["keyframe"]))
    assert negative[0, 0, 0] == 200
    assert raw_frames[0][0, 0, 0] == 0


def test_viser_workspace_overlay_uses_real_checked_red_green_points(tmp_path):
    report = _report(tmp_path)
    display = _Display("viser", tmp_path)
    candidate = SimpleNamespace(
        variant=SimpleNamespace(
            pose_changes=[
                SimpleNamespace(arena_pose=report["candidates"][0]["adjusted_pose"])
            ]
        )
    )
    display.candidates((candidate,), torch.eye(4), report["workspace"], sim=None)
    clouds = {cloud.overlay_id: cloud for cloud in display._overlays().point_clouds}
    with np.load(
        report["workspace"]["reachability"]["cache"], allow_pickle=False
    ) as cache:
        np.testing.assert_allclose(
            clouds["workspace_reachability"].points, cache["points"]
        )
        np.testing.assert_array_equal(
            clouds["workspace_reachability"].colors, cache["colors"]
        )


def test_native_cloud_uses_same_checked_points_without_physics_steps(tmp_path):
    report = _report(tmp_path)
    created = []
    handle = object()

    def create(**kwargs):
        created.append(kwargs)
        return handle

    sim = SimpleNamespace(
        visualize_point_cloud=create,
        update=lambda **kwargs: pytest.fail("Visualization must not advance physics"),
    )
    display = _Display("offscreen", tmp_path)
    display.candidates((), torch.eye(4), report["workspace"], sim=sim)
    assert display.native_cloud is handle
    with np.load(
        report["workspace"]["reachability"]["cache"], allow_pickle=False
    ) as data:
        np.testing.assert_array_equal(created[0]["points"], data["points"])
        np.testing.assert_array_equal(created[0]["colors"], data["colors"])
    display.close()
    assert display.native_cloud is None
