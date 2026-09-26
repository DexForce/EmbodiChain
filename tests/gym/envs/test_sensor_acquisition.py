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

from embodichain.lab.gym.envs import EmbodiedEnv
from embodichain.lab.gym.envs.managers.cfg import (
    EventCfg,
    ObservationCfg,
    SceneEntityCfg,
)


def _sensor_env(*, enable_sensor: bool) -> EmbodiedEnv:
    """Build the smallest environment object needed by ``_setup_sensors``."""
    env = EmbodiedEnv.__new__(EmbodiedEnv)
    env.cfg = SimpleNamespace(
        enable_sensor=enable_sensor,
        sensor=[SimpleNamespace(uid="camera")],
    )
    env.sim = SimpleNamespace(add_sensor=MagicMock(return_value=object()))
    return env


def test_disabled_sensor_acquisition_skips_sensor_creation() -> None:
    """Disabled acquisition does not allocate or fetch configured sensors."""
    env = _sensor_env(enable_sensor=False)

    assert env._setup_sensors() == {}
    env.sim.add_sensor.assert_not_called()


def test_enabled_sensor_acquisition_creates_configured_sensor() -> None:
    """The default keeps the existing configured-sensor behavior."""
    env = _sensor_env(enable_sensor=True)

    sensors = env._setup_sensors()

    assert list(sensors) == ["camera"]
    env.sim.add_sensor.assert_called_once_with(env.cfg.sensor[0])


def test_disabled_sensor_acquisition_filters_sensor_event_and_observation_functors() -> (
    None
):
    """Sensor-dependent manager functors are removed before manager creation."""

    def event_func(_env, _env_ids, entity_cfg):
        del _env, _env_ids, entity_cfg

    def observation_func(_env, _obs, entity_cfg):
        del _env, _obs, entity_cfg

    env = EmbodiedEnv.__new__(EmbodiedEnv)
    env.cfg = SimpleNamespace(
        enable_sensor=False,
        sensor=[SimpleNamespace(uid="camera")],
        filter_visual_rand=False,
        events=SimpleNamespace(
            camera_event=EventCfg(
                func=event_func,
                params={"entity_cfg": SceneEntityCfg(uid="camera")},
            ),
            all_sensor_event=EventCfg(
                func=event_func,
                params={"entity_uids": "all_sensors"},
            ),
            robot_event=EventCfg(
                func=event_func,
                params={"entity_cfg": SceneEntityCfg(uid="robot")},
            ),
        ),
        observations=SimpleNamespace(
            camera_observation=ObservationCfg(
                func=observation_func,
                name="sensor/camera/color",
                params={"entity_cfg": SceneEntityCfg(uid="camera")},
            ),
            robot_observation=ObservationCfg(
                func=observation_func,
                name="robot/qpos",
                params={"entity_cfg": SceneEntityCfg(uid="robot")},
            ),
        ),
    )

    env._apply_functor_filter()

    assert env.cfg.events.camera_event is None
    assert env.cfg.events.all_sensor_event is None
    assert env.cfg.events.robot_event is not None
    assert env.cfg.observations.camera_observation is None
    assert env.cfg.observations.robot_observation is not None
