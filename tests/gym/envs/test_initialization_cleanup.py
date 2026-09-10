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

"""Failed environment initialization releases only its own simulation."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import embodichain.lab.gym.envs.base_env as module

pytestmark = pytest.mark.no_sim


@pytest.mark.parametrize("phase", ["scene", "prepare", "robot"])
def test_initialization_failure_destroys_owned_simulation(monkeypatch, phase):
    error = ValueError("invalid scene declaration")
    sim = SimpleNamespace(profiler=Mock(), prepare=Mock(), destroy=Mock())
    flush = Mock()
    monkeypatch.setattr(module.SimulationManager, "flush_cleanup_queue", flush)

    def setup_scene(env, **kwargs):
        env.sim = sim
        if phase == "scene":
            raise error

    monkeypatch.setattr(module.BaseEnv, "_setup_scene", setup_scene)
    if phase == "prepare":
        sim.prepare.side_effect = error
    monkeypatch.setattr(module.BaseEnv, "_setup_robot", Mock(side_effect=error))
    with pytest.raises(ValueError) as caught:
        module.BaseEnv(module.EnvCfg())
    assert caught.value is error
    sim.destroy.assert_called_once_with(exit_process=False)
    flush.assert_not_called()


def test_initialization_failure_before_simulation_creation_preserves_error(monkeypatch):
    error = ValueError("invalid scene declaration")
    monkeypatch.setattr(module.BaseEnv, "_setup_scene", Mock(side_effect=error))
    with pytest.raises(ValueError) as caught:
        module.BaseEnv(module.EnvCfg())
    assert caught.value is error


def test_cleanup_failure_does_not_mask_initialization_error(monkeypatch):
    original = ValueError("invalid scene")
    sim = SimpleNamespace(destroy=Mock())

    def setup(env, **kwargs):
        env.sim = sim
        raise original

    monkeypatch.setattr(module.BaseEnv, "_setup_scene", setup)
    flush = Mock()
    monkeypatch.setattr(module.SimulationManager, "flush_cleanup_queue", flush)
    sim.destroy.side_effect = RuntimeError("cleanup failed")
    with pytest.raises(ValueError) as caught:
        module.BaseEnv(module.EnvCfg())
    assert caught.value is original


def test_post_base_initialization_failure_releases_simulation(monkeypatch):
    import embodichain.lab.gym.envs.embodied_env as embodied_module

    sim = SimpleNamespace(destroy=Mock())

    def base_init(env, cfg, **kwargs):
        env.cfg = cfg
        env.sim = sim

    monkeypatch.setattr(module.BaseEnv, "__init__", base_init)
    monkeypatch.setattr(module.SimulationManager, "flush_cleanup_queue", Mock())
    original = ValueError("dataset initialization failed")
    monkeypatch.setattr(embodied_module, "DatasetManager", Mock(side_effect=original))
    cfg = embodied_module.EmbodiedEnvCfg(
        dataset={"save": object()}, filter_dataset_saving=False
    )
    with pytest.raises(ValueError) as caught:
        embodied_module.EmbodiedEnv(cfg)
    assert caught.value is original
    sim.destroy.assert_called_once_with(exit_process=False)


def test_simulation_instance_allocation_preserves_live_entries(monkeypatch):
    live = object()
    monkeypatch.setattr(module.SimulationManager, "_instances", {1: live})
    created = module.SimulationManager.__new__(module.SimulationManager)
    assert module.SimulationManager._instances[1] is live
    assert module.SimulationManager._instances[created.instance_id] is created
    assert created.instance_id != 1


def test_native_teardown_waits_until_failed_constructor_unwinds(monkeypatch):
    import gc
    import queue

    events = []
    pending = queue.Queue()
    monkeypatch.setattr(module.SimulationManager, "_cleanup_queue", pending)
    monkeypatch.setattr(module.SimulationManager, "_instances", {})
    monkeypatch.setattr(module.SimulationManager, "wait_scene_destruction", Mock())

    class NativeView:
        def __del__(self):
            events.append("release_view")

    def destroy(*, exit_process):
        assert exit_process is False
        events.append("queue_destroy")
        pending.put(lambda: events.append("destroy_world"))

    def setup(env, **kwargs):
        env.sim = SimpleNamespace(destroy=destroy)
        view = NativeView()
        raise ValueError("camera attachment failed")

    monkeypatch.setattr(module.BaseEnv, "_setup_scene", setup)
    try:
        module.BaseEnv(module.EnvCfg())
    except ValueError:
        assert events == ["queue_destroy"]
    else:
        pytest.fail("Expected failed initialization")
    gc.collect()
    module.SimulationManager.flush_cleanup_queue()
    assert events == ["queue_destroy", "release_view", "destroy_world"]
