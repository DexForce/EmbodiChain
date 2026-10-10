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

"""Render demand is known before native World or camera construction."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import embodichain.lab.gym.envs.base_env as base_module
from embodichain.lab.gym.envs import BaseEnv, EmbodiedEnv, EmbodiedEnvCfg
from embodichain.lab.gym.envs.managers.cfg import EventCfg, FunctorCfg, ObservationCfg
from embodichain.lab.gym.envs.managers.record import (
    record_camera_data,
    record_camera_data_async,
    validation_cameras,
)
from embodichain.lab.gym.envs.managers.randomization.visual import (
    randomize_emission_light,
    randomize_visual_material,
    set_rigid_object_visual_material,
)
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import NewtonPhysicsCfg
from embodichain.lab.sim.sensors import (
    Camera,
    CameraCfg,
    ContactSensorCfg,
    StereoCameraCfg,
)
from embodichain.lab.visualization import VisualizationCfg

pytestmark = pytest.mark.no_sim


def _env(**kwargs) -> EmbodiedEnv:
    cfg = EmbodiedEnvCfg(sim_cfg=SimulationManagerCfg(headless=True), **kwargs)
    env = EmbodiedEnv.__new__(EmbodiedEnv)
    env.cfg = cfg
    env.sim_cfg = cfg.sim_cfg
    env._num_envs = cfg.num_envs
    return env


@pytest.mark.parametrize(
    ("sensors", "enabled", "expected"),
    [
        ([], True, False),
        ([ContactSensorCfg(uid="contact")], True, False),
        ([CameraCfg(uid="camera")], True, True),
        ([StereoCameraCfg(uid="stereo")], True, True),
        ([CameraCfg(uid="camera")], False, False),
        ([ContactSensorCfg(uid="contact"), CameraCfg(uid="camera")], True, True),
    ],
)
def test_configured_sensor_render_demand(sensors, enabled, expected) -> None:
    env = _env(sensor=sensors, enable_sensor=enabled)
    assert env._requires_native_renderer() is expected


def test_registered_camera_subclass_requires_rendering(monkeypatch) -> None:
    class CustomCamera(Camera):
        pass

    monkeypatch.setitem(
        SimulationManager.SUPPORTED_SENSOR_TYPES, "CustomCamera", CustomCamera
    )
    env = _env(sensor=[CameraCfg(uid="custom", sensor_type="CustomCamera")])
    assert env._requires_native_renderer()


@pytest.mark.parametrize("consumer", ["window", "viser", "sync"])
def test_non_sensor_consumers_reserve_renderer(consumer) -> None:
    env = _env()
    if consumer == "window":
        env.sim_cfg.headless = False
    elif consumer == "viser":
        env.sim_cfg.visualization = VisualizationCfg(backend="viser")
    else:
        env.sim_cfg.physics_cfg = NewtonPhysicsCfg(sync_to_renderer=True)
    assert env._requires_native_renderer()


def test_custom_demand_hook_reserves_renderer_before_world_creation(
    monkeypatch,
) -> None:
    class CustomEnv(EmbodiedEnv):
        def _requires_native_renderer(self) -> bool:
            return True

    env = CustomEnv.__new__(CustomEnv)
    env.cfg = EmbodiedEnvCfg(sim_cfg=SimulationManagerCfg(headless=True))
    env.cfg.enable_sensor = False
    env.cfg.sensor = [CameraCfg(uid="camera")]
    env.cfg.observations = {
        "camera": ObservationCfg(func=record_camera_data, name="sensor/camera/color")
    }
    env.cfg.filter_visual_rand = True
    env.cfg.events = {"material": EventCfg(func=randomize_visual_material)}
    env.sim_cfg = env.cfg.sim_cfg
    env._num_envs = env.cfg.num_envs
    create_sim = Mock(return_value=SimpleNamespace())
    monkeypatch.setattr(base_module, "SimulationManager", create_sim)
    monkeypatch.setattr(env, "_declare_robot", lambda **kwargs: None)
    monkeypatch.setattr(env, "_prepare_scene", lambda **kwargs: None)

    env._setup_scene()

    assert create_sim.call_args.kwargs["requires_native_renderer"] is True
    assert env.cfg.observations["camera"] is None
    assert env.cfg.events["material"] is None


def test_reusing_authored_configuration_can_enable_cameras(monkeypatch) -> None:
    config = EmbodiedEnvCfg(sim_cfg=SimulationManagerCfg(headless=True))

    def create_sim(cfg, *, requires_native_renderer, **kwargs):
        cfg.render_cfg.renderer = "hybrid" if requires_native_renderer else "no-render"
        return SimpleNamespace(
            profiler=Mock(),
            prepare=Mock(side_effect=RuntimeError("stop after world construction")),
            destroy=Mock(),
        )

    monkeypatch.setattr(base_module, "SimulationManager", create_sim)
    monkeypatch.setattr(EmbodiedEnv, "_declare_robot", lambda self, **kwargs: None)
    monkeypatch.setattr(EmbodiedEnv, "_prepare_scene", lambda self, **kwargs: None)
    for expected in ("no-render", "hybrid"):
        env = EmbodiedEnv.__new__(EmbodiedEnv)
        with pytest.raises(RuntimeError, match="stop after world construction"):
            BaseEnv.__init__(env, config)
        assert env.sim_cfg.render_cfg.renderer == expected
        assert config.sim_cfg.render_cfg.renderer == "auto"
        config.sensor = [CameraCfg(uid="camera")]


@pytest.mark.parametrize("manager", ["events", "observations", "rewards", "dataset"])
@pytest.mark.parametrize("factory", [record_camera_data, record_camera_data_async])
@pytest.mark.parametrize("as_dict", [False, True])
def test_recording_cameras_are_detected_without_constructing_functors(
    monkeypatch, manager, factory, as_dict
) -> None:
    constructor = Mock(side_effect=AssertionError("preflight constructed a recorder"))
    monkeypatch.setattr(factory, "__init__", constructor)
    term = EventCfg(func=factory)
    collection = {"record": term} if as_dict else SimpleNamespace(record=term)
    env = _env(enable_sensor=False, **{manager: collection})
    assert env._requires_native_renderer()
    constructor.assert_not_called()


@pytest.mark.parametrize("cameras", [[], [{"uid": "validation"}]])
@pytest.mark.parametrize("string_func", [False, True])
def test_validation_camera_list_controls_render_demand(cameras, string_func) -> None:
    func = (
        "embodichain.lab.gym.envs.managers.record:validation_cameras"
        if string_func
        else validation_cameras
    )
    env = _env(events={"validation": EventCfg(func=func, params={"cameras": cameras})})
    assert env._requires_native_renderer() is bool(cameras)


def test_disabled_dataset_recorder_is_not_a_consumer() -> None:
    env = _env(
        filter_dataset_saving=True,
        dataset={"record": {"func": record_camera_data, "params": {}}},
    )
    assert not env._requires_native_renderer()
    env.cfg.filter_dataset_saving = False
    assert env._requires_native_renderer()


def test_disabled_sensor_terms_are_filtered_before_preflight() -> None:
    env = _env(
        enable_sensor=False,
        sensor=[CameraCfg(uid="camera")],
        observations={
            "disabled": ObservationCfg(
                func=record_camera_data, name="sensor/camera/color"
            )
        },
    )
    env._apply_functor_filter()
    assert not env._requires_native_renderer()
    assert env.cfg.observations["disabled"] is None


@pytest.mark.parametrize("headless", [False, True])
def test_scene_construction_preserves_original_window_demand(
    monkeypatch, headless
) -> None:
    env = _env()
    env.sim_cfg.headless = headless
    seen = []

    def create_sim(cfg, *, requires_native_renderer, defer_startup_summary):
        seen.append((cfg.headless, requires_native_renderer, defer_startup_summary))
        return SimpleNamespace()

    monkeypatch.setattr(base_module, "SimulationManager", create_sim)
    monkeypatch.setattr(env, "_declare_robot", lambda **kwargs: None)
    monkeypatch.setattr(env, "_prepare_scene", lambda **kwargs: None)
    BaseEnv._setup_scene(env)
    assert seen == [(True, not headless, True)]
    assert env.sim_cfg.headless is headless


def test_failed_world_creation_restores_window_configuration(monkeypatch) -> None:
    env = _env()
    env.sim_cfg.headless = False
    monkeypatch.setattr(
        base_module, "SimulationManager", Mock(side_effect=ValueError("world"))
    )
    with pytest.raises(ValueError, match="world"):
        BaseEnv._setup_scene(env)
    assert env.sim_cfg.headless is False


@pytest.mark.parametrize("as_dict", [False, True])
def test_no_render_filters_visual_effects_and_keeps_physical_events(as_dict) -> None:
    physical = EventCfg(func=lambda env, ids: None)
    terms = {
        "light": EventCfg(func=randomize_emission_light),
        "material": EventCfg(func=randomize_visual_material),
        "fixed_material": EventCfg(func=set_rigid_object_visual_material),
        "string": EventCfg(
            func="embodichain.lab.gym.envs.managers.randomization.visual:randomize_indirect_lighting"
        ),
        "physical": physical,
    }
    collection = terms if as_dict else SimpleNamespace(**terms)
    env = _env(events=collection)
    collection = env.cfg.events
    physical = collection["physical"] if as_dict else collection.physical
    env._filter_native_visual_functors()
    result = collection if as_dict else vars(collection)
    assert result == {
        **dict.fromkeys(["light", "material", "fixed_material", "string"]),
        "physical": physical,
    }


@pytest.mark.parametrize("as_dict", [False, True])
def test_visual_randomization_filter_runs_before_scene_setup(as_dict) -> None:
    terms = {"visual": EventCfg(func=randomize_visual_material)}
    collection = terms if as_dict else SimpleNamespace(**terms)
    env = _env(events=collection, filter_visual_rand=True)
    env._apply_functor_filter()
    assert not env._requires_native_renderer()
    collection = env.cfg.events
    assert (collection if as_dict else vars(collection))["visual"] is None


def test_no_render_skips_configured_lights_without_accessing_assets() -> None:
    env = _env()
    env.sim = SimpleNamespace(has_native_renderer=False, add_light=Mock())
    env.cfg.light.direct = [object()]
    env.cfg.light.indirect = {"env_map": "not-downloaded.hdr"}
    env._setup_lights()
    env.sim.add_light.assert_not_called()
