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
from unittest.mock import Mock, call, patch

from embodichain.lab.sim.utility.demo_utils import (
    DEFAULT_DEMO_LOOK_AT,
    DemoRecording,
    add_demo_args,
    create_default_sim,
    format_tensor,
    maybe_init_gpu_physics,
    maybe_open_window,
    maybe_wait_for_user,
    replay_trajectory,
    resolve_demo_steps,
    run_simulation_loop,
    setup_print_options,
    shutdown_sim,
)


def test_add_demo_args_adds_expected_flags():
    parser = argparse.ArgumentParser()
    parser = add_demo_args(parser)
    args = parser.parse_args(
        [
            "--headless",
            "--auto-play",
            "--record-fps",
            "60",
            "--num-envs",
            "2",
            "--arena-space",
            "3.0",
            "--gpu-id",
            "1",
        ]
    )
    assert args.headless is True
    assert args.auto_play is True
    assert args.record_fps == 60
    assert args.record_steps is None
    assert args.no_vis_eef_axis is False
    assert args.num_envs == 2
    assert args.arena_space == 3.0
    assert args.gpu_id == 1


def test_add_demo_args_rejects_non_positive_record_steps():
    parser = add_demo_args(argparse.ArgumentParser())
    with pytest.raises(SystemExit):
        parser.parse_args(["--record_steps", "0"])


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


def test_shutdown_sim_finishes_active_recording_before_destroy():
    sim = Mock(
        spec=[
            "destroy",
            "is_window_recording",
            "stop_window_record",
            "wait_window_record_saves",
        ]
    )
    sim.is_window_recording.return_value = True

    shutdown_sim(sim)

    assert sim.method_calls == [
        call.is_window_recording(),
        call.stop_window_record(),
        call.wait_window_record_saves(),
        call.destroy(),
    ]


def test_create_default_sim_forwards_num_envs_and_headless():
    args = SimpleNamespace(headless=True, device="cpu", renderer="auto", gpu_id=2)
    with (
        patch("embodichain.lab.sim.SimulationManager") as mock_sm,
        patch("embodichain.lab.sim.SimulationManagerCfg") as mock_cfg_cls,
    ):
        create_default_sim(args, num_envs=4, add_default_light=False)
    cfg_kwargs = mock_cfg_cls.call_args.kwargs
    assert cfg_kwargs["num_envs"] == 4
    assert cfg_kwargs["headless"] is True
    assert cfg_kwargs["gpu_id"] == 2
    mock_sm.assert_called_once_with(mock_cfg_cls.return_value)


def test_create_default_sim_adds_light_when_requested():
    args = SimpleNamespace(headless=True, device="cpu", renderer="auto")
    with (
        patch("embodichain.lab.sim.SimulationManager") as mock_sm,
        patch("embodichain.lab.sim.SimulationManagerCfg"),
    ):
        sim = create_default_sim(args, num_envs=1, add_default_light=True)
    sim.add_light.assert_called_once()
    mock_sm.return_value.add_light.assert_called_once()


def test_maybe_init_physics_inits_when_enabled():
    sim = Mock(spec=["is_use_gpu_physics", "init_gpu_physics"])
    sim.is_use_gpu_physics = True
    maybe_init_gpu_physics(sim)
    sim.init_gpu_physics.assert_called_once()


def test_maybe_init_physics_skips_when_disabled():
    sim = Mock(spec=["is_use_gpu_physics", "init_gpu_physics"])
    sim.is_use_gpu_physics = False
    maybe_init_gpu_physics(sim)
    sim.init_gpu_physics.assert_not_called()


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
    assert call_kwargs["look_at"] == DEFAULT_DEMO_LOOK_AT
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


def test_demo_recording_uses_exact_mp4_path():
    sim = _make_recording_sim()
    args = SimpleNamespace(
        record_steps=10,
        record_fps=30,
        record_save_path="/tmp/custom_demo.mp4",
        auto_play=False,
        headless=False,
    )
    with DemoRecording(sim, args, prefix="ignored"):
        pass
    call_kwargs = sim.start_window_record.call_args.kwargs
    assert call_kwargs["save_path"] == "/tmp/custom_demo.mp4"
    assert call_kwargs["look_at"] is None


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


def test_maybe_open_window_opens_when_not_headless():
    sim = Mock(spec=["open_window"])
    args = SimpleNamespace(headless=False)
    maybe_open_window(sim, args)
    sim.open_window.assert_called_once()


def test_maybe_open_window_does_nothing_when_headless():
    sim = Mock(spec=["open_window"])
    args = SimpleNamespace(headless=True)
    maybe_open_window(sim, args)
    sim.open_window.assert_not_called()


def test_maybe_wait_for_user_prompts_when_not_auto_play():
    args = SimpleNamespace(auto_play=False)
    with patch("builtins.input", return_value="") as mock_input:
        maybe_wait_for_user(args, "Press enter")
    mock_input.assert_called_once_with("Press enter")


def test_maybe_wait_for_user_skips_when_auto_play():
    args = SimpleNamespace(auto_play=True)
    with patch("builtins.input") as mock_input:
        maybe_wait_for_user(args, "Press enter")
    mock_input.assert_not_called()


def test_resolve_demo_steps_prefers_explicit_record_steps():
    args = SimpleNamespace(auto_play=True, record_steps=12)
    assert resolve_demo_steps(args, auto_play_steps=5) == 12


def test_resolve_demo_steps_makes_auto_play_finite():
    assert (
        resolve_demo_steps(
            SimpleNamespace(auto_play=True, record_steps=None),
            auto_play_steps=5,
        )
        == 5
    )
    assert (
        resolve_demo_steps(SimpleNamespace(auto_play=False, record_steps=None)) is None
    )


def test_run_simulation_loop_updates_until_limit_and_calls_hook():
    sim = Mock(spec=["update", "num_envs"])
    sim.num_envs = 2
    on_step = Mock()

    completed = run_simulation_loop(
        sim,
        max_steps=3,
        steps_per_update=2,
        log_interval=None,
        on_step=on_step,
    )

    assert completed == 3
    assert sim.update.call_args_list == [call(step=2), call(step=2), call(step=2)]
    assert on_step.call_args_list == [call(1), call(2), call(3)]


def test_replay_trajectory_sets_qpos_and_updates_sim():
    robot = Mock(spec=["set_qpos", "get_joint_ids"])
    robot.get_joint_ids.return_value = None
    sim = Mock(spec=["update"])
    # Shape: (batch=1, num_steps=2, num_joints=3)
    traj = torch.tensor(
        [
            [[0.0, 0.1, 0.2], [0.3, 0.4, 0.5]],
        ]
    )
    replay_trajectory(sim, robot, traj, post_steps=1, step_size=4, sleep=0.0)
    assert robot.set_qpos.call_count == 3  # 2 traj + 1 post
    assert sim.update.call_count == 3
    sim.update.assert_has_calls([call(step=4), call(step=4), call(step=2)])
