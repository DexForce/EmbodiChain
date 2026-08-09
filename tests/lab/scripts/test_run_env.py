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

from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from embodichain.lab.gym.utils.gym_utils import merge_args_with_gym_config
from embodichain.lab.scripts import run_env
from embodichain.lab.scripts.run_env import (
    _create_parser,
    _run_replay_control_loop,
)

GYM_CONFIG_PATH = "task.yaml"
GYM_ID = "Dummy-v0"
EPISODE_INDEX = 3
ACTION_LIST_INDEX = 0
REPLAY_NUM_STEPS = 5
REPLAY_TARGET_STEP = 3
VISER_POLL_INTERVAL = 0.05


def test_generate_function_displays_episode_and_action_list_indices(
    monkeypatch,
) -> None:
    """Progress output distinguishes episodes from their local action lists."""
    env = MagicMock()
    env.reset.return_value = (None, {})
    env.get_wrapper_attr.return_value.return_value = [object()]
    env.step.return_value = (None, None, None, None, None)
    progress = MagicMock(side_effect=lambda actions, **kwargs: actions)
    monkeypatch.setattr(run_env.tqdm, "tqdm", progress)

    run_env.generate_function(env, num_traj=1, time_id=EPISODE_INDEX)

    assert progress.call_args.kwargs["desc"] == (
        f"Executing episode #{EPISODE_INDEX}, action list #{ACTION_LIST_INDEX}"
    )


def test_run_env_syncs_viser_images_each_step_by_default() -> None:
    """Run-env uses step-synchronized camera images when no FPS is supplied."""
    args = _create_parser().parse_args(["--gym_config", GYM_CONFIG_PATH, "--viser"])

    merged = merge_args_with_gym_config(args, {"id": GYM_ID})

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

    merged = merge_args_with_gym_config(args, {"id": GYM_ID})

    assert merged["visualization"]["sensor_image_fps"] == expected_fps


def test_run_env_preserves_configured_viser_image_fps() -> None:
    """A file-based rate overrides the run-env step-synchronized default."""
    configured_fps = 4.0
    args = _create_parser().parse_args(["--gym_config", GYM_CONFIG_PATH, "--viser"])

    merged = merge_args_with_gym_config(
        args,
        {
            "id": GYM_ID,
            "visualization": {"sensor_image_fps": configured_fps},
        },
    )

    assert merged["visualization"]["sensor_image_fps"] == configured_fps


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
