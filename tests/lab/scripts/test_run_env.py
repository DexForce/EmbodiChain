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
from unittest.mock import MagicMock, call

from embodichain.lab.gym.utils.gym_utils import merge_args_with_gym_config
from embodichain.lab.scripts import run_env
from embodichain.lab.scripts.run_env import _create_parser

GYM_CONFIG_PATH = "task.yaml"
GYM_ID = "Dummy-v0"
EPISODE_INDEX = 3
ACTION_LIST_INDEX = 0


class _PreviewInput:
    """Deterministic input source for the non-blocking preview loop."""

    def __init__(self, keys: list[str | None]) -> None:
        self._keys = iter(keys)
        self.timeouts: list[float | None] = []

    def read_key(self, timeout: float | None = None) -> str | None:
        self.timeouts.append(timeout)
        return next(self._keys)


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


def test_preview_enables_hidden_ik_gizmos_for_active_solver_parts() -> None:
    """Preview prepares each task-selected arm that has an IK solver."""
    solvers = {"left_arm": object(), "right_arm": object()}
    robot = SimpleNamespace(
        uid="preview_robot",
        control_parts={
            "left_arm": [],
            "left_eef": [],
            "right_arm": [],
        },
        get_solver=MagicMock(side_effect=lambda part: solvers.get(part)),
    )
    sim = MagicMock()
    sim.is_window_opened = True
    sim.has_gizmo.return_value = False
    sim.enable_gizmo.side_effect = [object(), object()]
    env = SimpleNamespace(
        unwrapped=SimpleNamespace(
            sim=sim,
            robot=robot,
            num_envs=1,
            cfg=SimpleNamespace(control_parts=["left_arm", "left_eef", "right_arm"]),
        )
    )

    gizmo_keys = run_env._enable_preview_ik_gizmos(env)

    assert gizmo_keys == (
        ("preview_robot", "left_arm"),
        ("preview_robot", "right_arm"),
    )
    assert sim.enable_gizmo.call_args_list == [
        call(uid="preview_robot", control_part="left_arm", enable_native=True),
        call(uid="preview_robot", control_part="right_arm", enable_native=True),
    ]
    assert sim.set_gizmo_visibility.call_args_list == [
        call("preview_robot", visible=False, control_part="left_arm"),
        call("preview_robot", visible=False, control_part="right_arm"),
    ]


def test_preview_skips_ik_gizmo_for_vectorized_environment() -> None:
    """Native IK Gizmos remain limited to one simulated environment."""
    sim = MagicMock()
    sim.is_window_opened = True
    env = SimpleNamespace(
        unwrapped=SimpleNamespace(
            sim=sim,
            robot=SimpleNamespace(uid="preview_robot"),
            num_envs=2,
        )
    )

    gizmo_keys = run_env._enable_preview_ik_gizmos(env)

    assert gizmo_keys == ()
    sim.enable_gizmo.assert_not_called()


def test_preview_loop_services_native_ik_gizmo_while_waiting() -> None:
    """Each input timeout advances Gizmo processing and one physics step."""
    physics_dt = 0.02
    sim = MagicMock()
    sim.sim_config = SimpleNamespace(physics_dt=physics_dt)
    env = SimpleNamespace(unwrapped=SimpleNamespace(sim=sim))
    control_input = _PreviewInput([None, "q"])

    run_env._run_preview_loop(
        env,
        control_input,
        (("preview_robot", "arm"),),
    )

    sim.update.assert_called_once_with(physics_dt, step=1)
    assert control_input.timeouts == [physics_dt, physics_dt]


def test_preview_terminal_i_toggles_ik_gizmo() -> None:
    """Terminal I mirrors the native-window visibility hotkey."""
    physics_dt = 0.02
    sim = MagicMock()
    sim.sim_config = SimpleNamespace(physics_dt=physics_dt)
    sim.toggle_gizmo_visibility.return_value = True
    env = SimpleNamespace(unwrapped=SimpleNamespace(sim=sim))

    run_env._run_preview_loop(
        env,
        _PreviewInput(["i", "q"]),
        (("preview_robot", "arm"),),
    )

    sim.toggle_gizmo_visibility.assert_called_once_with(
        "preview_robot",
        control_part="arm",
    )


def test_preview_loop_services_viser_without_native_ik_gizmo() -> None:
    """Viser preview keeps processing browser interaction commands."""
    physics_dt = 0.02
    sim = MagicMock()
    sim.sim_config = SimpleNamespace(
        physics_dt=physics_dt,
        visualization=SimpleNamespace(backend="viser"),
    )
    env = SimpleNamespace(unwrapped=SimpleNamespace(sim=sim))

    run_env._run_preview_loop(env, _PreviewInput([None, "q"]), ())

    sim.update.assert_called_once_with(physics_dt, step=1)
