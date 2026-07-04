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

import argparse
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from unittest.mock import Mock

from embodichain.lab.sim.utility.demo_utils import (
    DemoRecording,
    add_demo_args,
    format_tensor,
    setup_print_options,
    shutdown_sim,
)


def test_add_demo_args_adds_expected_flags():
    parser = argparse.ArgumentParser()
    parser = add_demo_args(parser)
    args = parser.parse_args(["--headless", "--auto_play", "--record_fps", "60"])
    assert args.headless is True
    assert args.auto_play is True
    assert args.record_fps == 60
    assert args.record_steps is None
    assert args.no_vis_eef_axis is False


def test_format_tensor_rounds_and_moves_to_cpu():
    tensor = torch.tensor([1.23456789, 2.34567891])
    result = format_tensor(tensor)
    assert result == "[1.2346, 2.3457]"


def test_setup_print_options_sets_numpy_and_torch():
    setup_print_options()
    assert np.get_printoptions()["precision"] == 5
    assert np.get_printoptions()["suppress"] is True
    assert torch._tensor_str.PRINT_OPTS.precision == 5
    assert torch._tensor_str.PRINT_OPTS.sci_mode is False


def test_shutdown_sim_calls_destroy():
    sim = Mock(spec=["destroy"])
    shutdown_sim(sim)
    sim.destroy.assert_called_once()


def _make_recording_sim():
    sim = Mock(
        spec=[
            "start_window_record",
            "stop_window_record",
            "wait_window_record_saves",
            "is_window_recording",
            "sim_config",
        ]
    )
    sim.sim_config = SimpleNamespace(width=1920, height=1080)
    sim.start_window_record.return_value = True
    sim.is_window_recording.return_value = False
    return sim


def test_demo_recording_does_nothing_when_record_steps_is_none():
    sim = _make_recording_sim()
    args = SimpleNamespace(
        record_steps=None,
        record_fps=30,
        record_save_path="/tmp",
        auto_play=False,
        headless=True,
    )
    with DemoRecording(sim, args, prefix="demo"):
        pass
    sim.start_window_record.assert_not_called()


def test_demo_recording_starts_and_stops_window_record():
    sim = _make_recording_sim()
    sim.is_window_recording.return_value = True
    args = SimpleNamespace(
        record_steps=10,
        record_fps=30,
        record_save_path="/tmp/recordings",
        auto_play=False,
        headless=True,
    )
    with DemoRecording(sim, args, prefix="demo") as rec:
        assert rec.is_active is True
    sim.start_window_record.assert_called_once()
    call_kwargs = sim.start_window_record.call_args.kwargs
    assert call_kwargs["fps"] == 30
    assert call_kwargs["video_prefix"] == "demo"
    assert "/tmp/recordings" in call_kwargs["save_path"]
    assert call_kwargs["save_path"].endswith(".mp4")
    assert call_kwargs["look_at"] is None
    sim.stop_window_record.assert_called_once()
    sim.wait_window_record_saves.assert_called_once()


def test_demo_recording_passes_look_at():
    sim = _make_recording_sim()
    args = SimpleNamespace(
        record_steps=10,
        record_fps=30,
        record_save_path="/tmp",
        auto_play=False,
        headless=True,
    )
    look_at = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    with DemoRecording(sim, args, prefix="demo", look_at=look_at):
        pass
    call_kwargs = sim.start_window_record.call_args.kwargs
    assert call_kwargs["look_at"] == look_at


def test_demo_recording_warns_and_skips_on_start_failure():
    sim = _make_recording_sim()
    sim.start_window_record.return_value = False
    args = SimpleNamespace(
        record_steps=10,
        record_fps=30,
        record_save_path="/tmp",
        auto_play=False,
        headless=True,
    )
    with pytest.warns(UserWarning, match="Failed to start recording"):
        with DemoRecording(sim, args, prefix="demo"):
            pass
    sim.stop_window_record.assert_not_called()
