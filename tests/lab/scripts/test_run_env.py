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

import pytest

from embodichain.lab.gym.envs.demo import DemoEpisodeResult
from embodichain.lab.gym.utils.gym_utils import merge_args_with_gym_config
from embodichain.lab.scripts.run_env import _create_parser, generate_function

GYM_CONFIG_PATH = "task.yaml"
GYM_ID = "Dummy-v0"


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


class _ResetTrackingEnv:
    def __init__(self) -> None:
        self.reset_options = []

    def reset(self, options=None):
        self.reset_options.append(options)
        return None, {}


def _episode_result(*, success: bool, reason: str) -> DemoEpisodeResult:
    return DemoEpisodeResult(
        episode_index=0,
        length=2,
        completed=success,
        success=(success,),
        terminated=(False,),
        truncated=(False,),
        terminal_reason=reason,
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


def test_generate_function_rejects_runner_owned_segment_count() -> None:
    """The task, not run-env, defines how many segments an episode contains."""
    with pytest.raises(ValueError, match="create_demo_segments"):
        generate_function(_ResetTrackingEnv(), num_traj=2)
