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

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from embodichain.lab.gym.envs.demo import DemoEpisodeResult
from embodichain.lab.task_program.language.loader import (
    load_task_program as _load_task_program,
)
from embodichain.lab.gym.utils.gym_utils import merge_args_with_gym_config
from embodichain.lab.scripts import run_env
from embodichain.lab.scripts.run_env import (
    _create_parser,
    _run_replay_control_loop,
    generate_function,
)

GYM_CONFIG_PATH = "task.yaml"
GYM_ID = "Dummy-v0"
EPISODE_INDEX = 3
ACTION_LIST_INDEX = 0
REPLAY_NUM_STEPS = 5
REPLAY_TARGET_STEP = 3
VISER_POLL_INTERVAL = 0.05


def _task_program_payload() -> dict[str, object]:
    """Return one minimal strict Task Program payload."""
    return {
        "program_id": "cli_pick",
        "integration": {
            "robot_profile": "default_robot",
            "scene_registry": "default_scene",
            "runtime_preset": "default_runtime",
        },
        "targets": {},
        "program": {
            "kind": "invoke",
            "call": {"kind": "pick", "object": "cube"},
        },
    }


class _LegacyProgressEnv:
    num_envs = 1

    @property
    def unwrapped(self):
        return self

    def get_wrapper_attr(self, name: str):
        if name == "create_demo_action_list":
            return lambda **kwargs: [object()]
        raise AttributeError(name)

    def step(self, action):
        return None, None, False, False, {"success": False}

    def is_task_success(self) -> bool:
        return True


def test_legacy_action_list_displays_episode_and_segment_indices(
    monkeypatch,
) -> None:
    """Progress output distinguishes episodes from their local segments."""
    env = _LegacyProgressEnv()
    progress = MagicMock(side_effect=lambda actions, **kwargs: actions)
    monkeypatch.setattr(run_env.tqdm, "tqdm", progress)

    generated = run_env.generate_and_execute_action_list(
        env,
        ACTION_LIST_INDEX,
        debug_mode=False,
        episode_idx=EPISODE_INDEX,
    )

    assert generated
    assert progress.call_args.kwargs["desc"] == (
        f"Executing episode #{EPISODE_INDEX}, segment #{ACTION_LIST_INDEX}: legacy"
    )


def test_run_env_syncs_viser_images_each_step_by_default() -> None:
    """Run-env uses step-synchronized camera images when no FPS is supplied."""
    args = _create_parser().parse_args(["--gym_config", GYM_CONFIG_PATH, "--viser"])

    merged = merge_args_with_gym_config(args, {"id": GYM_ID, "physics": "default"})

    assert merged["visualization"]["sensor_image_fps"] is None


def test_run_env_accepts_explicit_viser_image_fps() -> None:
    """An explicit image FPS restores wall-clock rate limiting."""
    expected_fps = 6.0
    args = _create_parser().parse_args(
        [
            "--gym_config",
            GYM_CONFIG_PATH,
            "--viser",
            "--viser-image-fps",
            str(expected_fps),
        ]
    )

    merged = merge_args_with_gym_config(args, {"id": GYM_ID, "physics": "default"})

    assert merged["visualization"]["sensor_image_fps"] == expected_fps


def test_run_env_preserves_configured_viser_image_fps() -> None:
    """A file-based rate overrides the run-env step-synchronized default."""
    configured_fps = 4.0
    args = _create_parser().parse_args(["--gym_config", GYM_CONFIG_PATH, "--viser"])

    merged = merge_args_with_gym_config(
        args,
        {
            "id": GYM_ID,
            "physics": "default",
            "visualization": {"sensor_image_fps": configured_fps},
        },
    )

    assert merged["visualization"]["sensor_image_fps"] == configured_fps


def test_run_env_parser_accepts_task_program_path() -> None:
    """The declarative program is an explicit, opt-in CLI input."""
    program_path = "program.yaml"

    args = _create_parser().parse_args(
        ["--gym_config", GYM_CONFIG_PATH, "--task-program", program_path]
    )

    assert args.task_program == program_path


def test_run_env_parser_accepts_debug_trace_mode() -> None:
    """Failed Task Program attempts can expose their structured trace."""
    args = _create_parser().parse_args(
        ["--gym_config", GYM_CONFIG_PATH, "--debug-mode"]
    )

    assert args.debug_mode is True


@pytest.mark.parametrize("suffix", [".json", ".yaml", ".yml"])
def test_load_task_program_safely_decodes_supported_files(
    tmp_path,
    suffix: str,
) -> None:
    """JSON and safe YAML inputs share the same strict schema decoder."""
    path = tmp_path / f"program{suffix}"
    payload = _task_program_payload()
    if suffix == ".json":
        serialized = json.dumps(payload)
    else:
        import yaml

        serialized = yaml.safe_dump(payload)
    path.write_text(serialized, encoding="utf-8")

    program = _load_task_program(path)

    assert program.program_id == "cli_pick"
    assert program.integration.scene_registry == "default_scene"


@pytest.mark.parametrize(
    ("filename", "serialized", "message"),
    [
        (
            "program.json",
            '{"program_id": "one", "program_id": "two"}',
            "Duplicate JSON key",
        ),
        (
            "program.yaml",
            "program_id: one\nprogram_id: two\n",
            "found duplicate key",
        ),
    ],
)
def test_load_task_program_rejects_duplicate_mapping_keys(
    tmp_path,
    filename: str,
    serialized: str,
    message: str,
) -> None:
    """Ambiguous duplicate keys are rejected before schema decoding."""
    path = tmp_path / filename
    path.write_text(serialized, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        _load_task_program(path)


def test_load_task_program_rejects_unsupported_file_extension(tmp_path) -> None:
    """Only explicit JSON and YAML file formats are accepted."""
    path = tmp_path / "program.toml"
    path.write_text('program_id = "cli_pick"', encoding="utf-8")

    with pytest.raises(ValueError, match=".json, .yaml, or .yml"):
        _load_task_program(path)


def test_replay_restores_wrapper_state_without_closing_caller_env(monkeypatch) -> None:
    """Replay leaves the environment close to its CLI owner."""
    env = MagicMock()
    env.unwrapped = env
    replay_env = MagicMock()
    monkeypatch.setattr(
        run_env,
        "load_trajectory",
        lambda path: {"meta": {"num_envs": 1, "num_steps": 1, "lengths": [1]}},
    )
    monkeypatch.setattr(run_env, "ReplayWrapper", lambda *args, **kwargs: replay_env)
    monkeypatch.setattr(run_env, "replay_auto", lambda *args, **kwargs: None)

    run_env.replay(env, "trajectory.pt")

    replay_env.close.assert_not_called()
    replay_env.env.sim.enable_physics.assert_called_once_with(True)
    assert replay_env.env._replay_no_auto_reset is False


def test_preview_quit_returns_without_zero_exit(monkeypatch) -> None:
    """Preview quit lets CLI cleanup failures determine the process status."""
    env = MagicMock()
    env.reset.return_value = (None, {})
    monkeypatch.setattr("builtins.input", lambda: "q")

    run_env.preview(env)

    env.reset.assert_called_once_with()


class _ResetTrackingEnv:
    def __init__(
        self,
        *,
        num_envs: int = 1,
        save_failed_episodes: bool = False,
    ) -> None:
        self.num_envs = num_envs
        self.device = torch.device("cpu")
        self.dataset_manager = SimpleNamespace(
            save_failed_episodes=save_failed_episodes
        )
        self.reset_options = []

    def reset(self, options=None):
        self.reset_options.append(options)
        return None, {}


class _LifecycleTrackingEnv(_ResetTrackingEnv):
    def __init__(self) -> None:
        super().__init__()
        self.events = []

    def reset(self, options=None):
        self.events.append(("reset", options))
        return super().reset(options=options)

    def close(self, *, exit_process=None) -> None:
        self.events.append(("close", exit_process))


def _episode_result(
    *,
    success: bool,
    reason: str,
    length: int = 2,
    num_envs: int = 1,
) -> DemoEpisodeResult:
    return DemoEpisodeResult(
        episode_index=0,
        length=length,
        completed=success,
        success=(success,) * num_envs,
        terminated=(False,) * num_envs,
        truncated=(False,) * num_envs,
        terminal_reason=reason,
        lengths=(length,) * num_envs,
    )


def test_generate_function_discards_retry_then_commits_once(monkeypatch) -> None:
    """Failed attempts are discarded and only the complete episode is saved."""
    env = _ResetTrackingEnv()
    results = iter(
        [
            _episode_result(success=False, reason="empty_plan"),
            _episode_result(success=True, reason="success"),
        ]
    )
    monkeypatch.setattr(
        "embodichain.lab.scripts.run_env.execute_demo_episode",
        lambda *args, **kwargs: next(results),
    )

    generated = generate_function(
        env,
        time_id=3,
        max_attempts=2,
        reset_before=False,
    )

    assert generated
    assert env.reset_options == [{"save_data": False}, None]


def test_generate_function_logs_failed_trace_in_debug_mode(monkeypatch) -> None:
    """Debug retries expose the owned structured episode trace."""
    env = _ResetTrackingEnv()
    result = _episode_result(success=False, reason="segment_validation_failed")
    warnings: list[str] = []
    monkeypatch.setattr(
        "embodichain.lab.scripts.run_env.execute_demo_episode",
        lambda *args, **kwargs: result,
    )
    monkeypatch.setattr(
        "embodichain.lab.scripts.run_env.log_warning",
        warnings.append,
    )

    generated = generate_function(
        env,
        max_attempts=1,
        reset_before=False,
        debug_mode=True,
    )

    assert not generated
    debug_trace = next(
        message for message in warnings if "Failed demo trace" in message
    )
    assert '"terminal_reason":"segment_validation_failed"' in debug_trace


def test_generate_function_commits_failed_episode_when_configured(monkeypatch) -> None:
    """A recorded task failure is a persisted result when explicitly enabled."""
    env = _ResetTrackingEnv(save_failed_episodes=True)
    monkeypatch.setattr(
        "embodichain.lab.scripts.run_env.execute_demo_episode",
        lambda *args, **kwargs: _episode_result(
            success=False,
            reason="task_incomplete",
        ),
    )

    generated = generate_function(
        env,
        time_id=3,
        max_attempts=2,
        reset_before=False,
    )

    assert generated
    assert env.reset_options == [None]


def test_generate_function_retries_empty_failure_even_when_saving_failures(
    monkeypatch,
) -> None:
    """An empty plan cannot count as a saved failed episode."""
    env = _ResetTrackingEnv(save_failed_episodes=True)
    results = iter(
        [
            _episode_result(success=False, reason="empty_plan", length=0),
            _episode_result(success=True, reason="success"),
        ]
    )
    monkeypatch.setattr(
        "embodichain.lab.scripts.run_env.execute_demo_episode",
        lambda *args, **kwargs: next(results),
    )

    generated = generate_function(
        env,
        max_attempts=2,
        reset_before=False,
    )

    assert generated
    assert env.reset_options == [{"save_data": False}, None]


def test_generate_function_commits_only_selected_vector_rows(monkeypatch) -> None:
    """The final partial batch commits selected rows in one full-batch reset."""
    num_envs = 3
    env = _ResetTrackingEnv(num_envs=num_envs)
    monkeypatch.setattr(
        "embodichain.lab.scripts.run_env.execute_demo_episode",
        lambda *args, **kwargs: _episode_result(
            success=True,
            reason="success",
            num_envs=num_envs,
        ),
    )

    generated = generate_function(
        env,
        save_env_ids=(0, 1),
        reset_before=False,
    )

    assert generated
    assert len(env.reset_options) == 1
    assert env.reset_options[0]["commit_env_ids"].tolist() == [0, 1]
    assert env.reset_options[0]["save_data"] is False
    assert "reset_ids" not in env.reset_options[0]


def test_main_counts_max_episodes_as_persisted_env_rows(monkeypatch) -> None:
    """Vector collection persists exactly max_episodes, including a partial batch."""
    env = _ResetTrackingEnv(num_envs=3)
    args = SimpleNamespace(
        replay=False,
        preview=False,
        save_path="",
        save_video=False,
        debug_mode=False,
        regenerate=False,
        record_trajectory=False,
    )
    calls: list[tuple[int, tuple[int, ...]]] = []

    def record_batch(*args, **kwargs) -> bool:
        calls.append((kwargs["time_id"], tuple(kwargs["save_env_ids"])))
        return True

    monkeypatch.setattr(run_env, "generate_function", record_batch)

    run_env.main(
        args,
        env,
        {"max_episodes": 5, "demo_max_attempts": 2},
    )

    assert calls == [(0, (0, 1, 2)), (3, (0, 1))]
    assert sum(len(env_ids) for _, env_ids in calls) == 5


def test_generate_function_rejects_runner_owned_segment_count() -> None:
    """The task, not run-env, defines how many segments an episode contains."""
    with pytest.raises(ValueError, match="create_demo_segments"):
        generate_function(_ResetTrackingEnv(), num_traj=2)


def test_generate_function_discards_partial_episode_on_exception(monkeypatch) -> None:
    """Planner or step exceptions cannot be committed later by finalize()."""
    env = _ResetTrackingEnv()
    monkeypatch.setattr(
        "embodichain.lab.scripts.run_env.execute_demo_episode",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("planner failed")),
    )

    with pytest.raises(RuntimeError, match="planner failed"):
        generate_function(env, reset_before=False)

    assert env.reset_options == [{"save_data": False}]


@pytest.mark.parametrize(
    "exit_type",
    [KeyboardInterrupt, SystemExit, GeneratorExit],
    ids=["keyboard-interrupt", "system-exit", "generator-exit"],
)
def test_generate_function_discards_partial_episode_on_early_exit(
    monkeypatch, exit_type
) -> None:
    """Non-Exception exits abort the pending episode before propagating."""
    env = _ResetTrackingEnv()

    def exit_during_episode(*args, **kwargs):
        raise exit_type()

    monkeypatch.setattr(
        "embodichain.lab.scripts.run_env.execute_demo_episode",
        exit_during_episode,
    )

    with pytest.raises(exit_type):
        generate_function(env, reset_before=False)

    assert env.reset_options == [{"save_data": False}]


def test_cli_aborts_before_closing_environment_once(monkeypatch) -> None:
    """CLI owns finalization and closes only after discarding pending data."""
    env = _LifecycleTrackingEnv()
    args = SimpleNamespace(
        replay=False,
        replay_mode="kinematic",
        preview=False,
        save_path="",
        save_video=False,
        debug_mode=False,
        regenerate=False,
        record_trajectory=False,
    )
    parser = MagicMock()
    parser.parse_args.return_value = args
    gym_config = {
        "id": GYM_ID,
        "max_episodes": 1,
        "demo_max_attempts": 1,
    }

    monkeypatch.setattr(run_env, "_create_parser", lambda: parser)
    monkeypatch.setattr(run_env, "discover_task_packages", lambda: None)
    monkeypatch.setattr(run_env, "execute_init_hooks", lambda: None)
    monkeypatch.setattr(
        run_env,
        "build_env_cfg_from_args",
        lambda parsed_args: (object(), gym_config, {}),
    )
    monkeypatch.setattr(run_env.gymnasium, "make", lambda **kwargs: env)
    monkeypatch.setattr(run_env, "generate_function", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        "embodichain.lab.sim.sim_manager.SimulationManager.flush_cleanup_queue",
        lambda: None,
    )

    run_env.cli([])

    abort_event = ("reset", {"save_data": False})
    assert env.events == [abort_event, abort_event, ("close", None)]


def test_cli_uses_program_already_loaded_by_config_builder(
    monkeypatch,
) -> None:
    """The CLI attaches the strict program config to the environment config."""
    env = _LifecycleTrackingEnv()
    env_cfg = SimpleNamespace(task_program=None)
    decoded_program = object()
    args = SimpleNamespace(
        replay=False,
        replay_mode="kinematic",
        preview=True,
        task_program="program.yaml",
    )
    parser = MagicMock()
    parser.parse_args.return_value = args
    make = MagicMock(return_value=env)

    monkeypatch.setattr(run_env, "_create_parser", lambda: parser)
    monkeypatch.setattr(run_env, "discover_task_packages", lambda: None)
    monkeypatch.setattr(run_env, "execute_init_hooks", lambda: None)

    def build(parsed_args):
        assert parsed_args is args
        env_cfg.task_program = decoded_program
        return env_cfg, {"id": GYM_ID}, {}

    monkeypatch.setattr(run_env, "build_env_cfg_from_args", build)
    monkeypatch.setattr(run_env.gymnasium, "make", make)
    monkeypatch.setattr(run_env, "main", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "embodichain.lab.sim.sim_manager.SimulationManager.flush_cleanup_queue",
        lambda: None,
    )

    run_env.cli([])

    assert env_cfg.task_program is decoded_program
    make.assert_called_once_with(id=GYM_ID, cfg=env_cfg)


def test_close_durability_failure_is_not_swallowed() -> None:
    """A failed recorder barrier makes the runner fail after aborting pending data."""
    env = _LifecycleTrackingEnv()

    def fail_close(*, exit_process=None) -> None:
        env.events.append(("close", exit_process))
        raise OSError("dataset flush failed")

    env.close = fail_close

    with pytest.raises(OSError, match="dataset flush failed"):
        run_env._abort_and_close_env(env)

    assert env.events == [
        ("reset", {"save_data": False}),
        ("close", None),
    ]


def test_cli_preserves_main_error_and_disables_fast_exit(monkeypatch) -> None:
    """Failure cleanup returns normally so the original CLI error reaches the shell."""
    env = _LifecycleTrackingEnv()
    args = SimpleNamespace(
        replay=False,
        replay_mode="kinematic",
        preview=False,
        save_path="",
        save_video=False,
        debug_mode=False,
        regenerate=False,
        record_trajectory=False,
    )
    parser = MagicMock()
    parser.parse_args.return_value = args
    gym_config = {"id": GYM_ID}

    monkeypatch.setattr(run_env, "_create_parser", lambda: parser)
    monkeypatch.setattr(run_env, "discover_task_packages", lambda: None)
    monkeypatch.setattr(run_env, "execute_init_hooks", lambda: None)
    monkeypatch.setattr(
        run_env,
        "build_env_cfg_from_args",
        lambda parsed_args: (object(), gym_config, {}),
    )
    monkeypatch.setattr(run_env.gymnasium, "make", lambda **kwargs: env)
    monkeypatch.setattr(
        run_env,
        "main",
        MagicMock(side_effect=RuntimeError("generation failed")),
    )
    monkeypatch.setattr(
        "embodichain.lab.sim.sim_manager.SimulationManager.flush_cleanup_queue",
        lambda: None,
    )

    with pytest.raises(RuntimeError, match="generation failed"):
        run_env.cli([])

    assert env.events == [
        ("reset", {"save_data": False}),
        ("close", False),
    ]


def test_zero_system_exit_cannot_hide_cleanup_failure(monkeypatch) -> None:
    """A nominal exit code is replaced by a failed durability barrier."""
    env = _LifecycleTrackingEnv()
    args = SimpleNamespace(
        replay=False,
        replay_mode="kinematic",
        preview=False,
        replay_trajectory=None,
    )
    parser = MagicMock()
    parser.parse_args.return_value = args
    monkeypatch.setattr(run_env, "_create_parser", lambda: parser)
    monkeypatch.setattr(run_env, "discover_task_packages", lambda: None)
    monkeypatch.setattr(run_env, "execute_init_hooks", lambda: None)
    monkeypatch.setattr(
        run_env,
        "build_env_cfg_from_args",
        lambda parsed_args: (object(), {"id": GYM_ID}, {}),
    )
    monkeypatch.setattr(run_env.gymnasium, "make", lambda **kwargs: env)
    monkeypatch.setattr(run_env, "main", MagicMock(side_effect=SystemExit(0)))
    monkeypatch.setattr(
        "embodichain.lab.sim.sim_manager.SimulationManager.flush_cleanup_queue",
        lambda: None,
    )

    def fail_close(*, exit_process=None) -> None:
        raise OSError("dataset flush failed")

    env.close = fail_close

    with pytest.raises(OSError, match="dataset flush failed"):
        run_env.cli([])


def test_control_loop_consumes_viser_seek_while_paused() -> None:
    """A browser slider seek is applied without waiting for terminal input."""

    class FakeControlInput:
        single_key = True

        def __init__(self) -> None:
            self.timeouts: list[float | None] = []

        def read_key(self, timeout: float | None = None) -> str:
            self.timeouts.append(timeout)
            return "q"

    class FakeReplayEnv:
        def __init__(self) -> None:
            self._lengths = torch.tensor([REPLAY_NUM_STEPS])
            self.env = SimpleNamespace(
                sim_cfg=SimpleNamespace(physics_dt=0.01),
                cfg=SimpleNamespace(sim_steps_per_control=4),
            )
            self.visited_steps: list[int] = []

        def go_to_step(self, step: int) -> None:
            self.visited_steps.append(step)

    class FakeVisualizationRuntime:
        def __init__(self) -> None:
            self.target_step: int | None = REPLAY_TARGET_STEP
            self.states: list[tuple[int, int, bool]] = []

        def drain_replay_control_command(self) -> int | None:
            target_step, self.target_step = self.target_step, None
            return target_step

        def publish_replay_control(
            self,
            *,
            step: int,
            max_step: int,
            visible: bool = True,
        ) -> None:
            self.states.append((step, max_step, visible))

    replay_env = FakeReplayEnv()
    control_input = FakeControlInput()
    runtime = FakeVisualizationRuntime()
    _run_replay_control_loop(
        replay_env,
        control_input,
        visualization_runtime=runtime,
    )

    assert replay_env.visited_steps == [0, REPLAY_TARGET_STEP]
    assert control_input.timeouts == [VISER_POLL_INTERVAL]
    assert runtime.states[-1] == (
        REPLAY_TARGET_STEP,
        REPLAY_NUM_STEPS - 1,
        False,
    )
