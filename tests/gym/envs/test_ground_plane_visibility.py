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

"""Native ground visibility follows explicit settings and active HDR events."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import embodichain.lab.gym.envs.base_env as base_module
from embodichain.lab.gym.envs import BaseEnv, EmbodiedEnv, EmbodiedEnvCfg
from embodichain.lab.gym.envs.managers.cfg import EventCfg
from embodichain.lab.gym.envs.managers.randomization.visual import (
    randomize_indirect_lighting,
)
from embodichain.lab.gym.utils.gym_utils import (
    add_env_launcher_args_to_parser,
    build_env_cfg_from_args,
    config_to_cfg,
)
from embodichain.lab.sim import SimulationManagerCfg
from embodichain.utils.utility import save_config

pytestmark = pytest.mark.no_sim


def _env(cfg: EmbodiedEnvCfg) -> EmbodiedEnv:
    env = EmbodiedEnv.__new__(EmbodiedEnv)
    env.cfg = cfg
    return env


@pytest.mark.parametrize("as_mapping", [False, True])
@pytest.mark.parametrize("string_func", [False, True])
@pytest.mark.parametrize("mode", ["startup", "reset", "interval"])
def test_hdr_event_defaults_to_hidden_without_constructing_functor(
    as_mapping, string_func, mode, monkeypatch
) -> None:
    constructor = Mock(side_effect=AssertionError("HDR preflight loaded assets"))
    monkeypatch.setattr(randomize_indirect_lighting, "__init__", constructor)
    func = (
        "embodichain.lab.gym.envs.managers.randomization.visual:randomize_indirect_lighting"
        if string_func
        else randomize_indirect_lighting
    )
    term = EventCfg(func=func, mode=mode, params={"path": "unused-hdr-folder"})
    events = (
        {"lighting": term, "disabled": None}
        if as_mapping
        else SimpleNamespace(lighting=term, disabled=None)
    )
    env = _env(EmbodiedEnvCfg(events=events))

    assert env._resolve_ground_plane_visibility() is False
    constructor.assert_not_called()


def test_hdr_functor_subclass_defaults_to_hidden() -> None:
    class CustomLighting(randomize_indirect_lighting):
        pass

    cfg = EmbodiedEnvCfg(
        events={
            "lighting": EventCfg(func=CustomLighting, params={"path": "unused-hdr"})
        }
    )
    assert _env(cfg)._resolve_ground_plane_visibility() is False


@pytest.mark.parametrize("with_emissive_event", [False, True])
def test_fixed_map_and_emissive_randomization_keep_ground_visible(
    with_emissive_event,
) -> None:
    cfg = EmbodiedEnvCfg()
    cfg.light.indirect = {"env_map": "unused-fixed-map.hdr"}
    if with_emissive_event:
        cfg.events = {
            "emissive": EventCfg(
                func=randomize_indirect_lighting,
                params={"emissive_intensity_range": [80.0, 150.0]},
            )
        }
    assert _env(cfg)._resolve_ground_plane_visibility() is True


@pytest.mark.parametrize("visible", [False, True])
@pytest.mark.parametrize("with_hdr", [False, True])
def test_explicit_visibility_overrides_hdr_default(visible, with_hdr) -> None:
    cfg = EmbodiedEnvCfg(ground_plane_visible=visible)
    if with_hdr:
        cfg.events = {
            "lighting": EventCfg(
                func=randomize_indirect_lighting, params={"path": "unused-hdr"}
            )
        }
    assert _env(cfg)._resolve_ground_plane_visibility() is visible


@pytest.mark.parametrize("as_mapping", [False, True])
def test_filtered_hdr_event_keeps_ground_visible(as_mapping) -> None:
    term = EventCfg(func=randomize_indirect_lighting, params={"path": "unused-hdr"})
    events = {"lighting": term} if as_mapping else SimpleNamespace(lighting=term)
    env = _env(EmbodiedEnvCfg(events=events, filter_visual_rand=True))

    env._apply_functor_filter()

    assert env._resolve_ground_plane_visibility() is True


@pytest.mark.parametrize("value", ["false", 0, 1, [], {}])
def test_programmatic_visibility_rejects_non_boolean_values(value) -> None:
    with pytest.raises(TypeError, match="ground_plane_visible"):
        _env(
            EmbodiedEnvCfg(ground_plane_visible=value)
        )._resolve_ground_plane_visibility()


@pytest.mark.parametrize(
    "settings",
    [
        {},
        {"ground_plane_visible": None},
        {"ground_plane_visible": False},
        {"ground_plane_visible": True},
    ],
)
def test_task_visibility_decodes_without_resolving_auto(settings) -> None:
    config = {
        "id": "EmbodiedEnv-v1",
        "physics": "default",
        "robot": {"uid": "robot"},
        "env": {},
        **settings,
    }

    cfg = config_to_cfg(config)

    assert cfg.ground_plane_visible is settings.get("ground_plane_visible")


@pytest.mark.parametrize("value", ["false", 0, 1, [], {}])
def test_task_visibility_rejects_non_boolean_values(value) -> None:
    config = {
        "id": "EmbodiedEnv-v1",
        "physics": "default",
        "robot": {"uid": "robot"},
        "env": {},
        "ground_plane_visible": value,
    }
    with pytest.raises(TypeError, match="ground_plane_visible"):
        config_to_cfg(config)


def test_task_visibility_can_accompany_an_environment_component(tmp_path: Path) -> None:
    save_config(
        tmp_path / "environment.yaml",
        {
            "environment_id": "room",
            "physics": "default",
            "simulation": {},
            "env": {},
        },
    )
    cfg = config_to_cfg(
        {
            "id": "EmbodiedEnv-v1",
            "environment": {"component": "environment.yaml"},
            "robot": {"uid": "robot"},
            "ground_plane_visible": False,
        },
        source_path=tmp_path / "task.yaml",
    )

    assert cfg.ground_plane_visible is False


@pytest.mark.parametrize(
    ("task_visible", "runtime_settings", "expected"),
    [
        (None, {}, False),
        (False, {"ground_plane_visible": True}, True),
        (True, {"ground_plane_visible": False}, False),
        (True, {"ground_plane_visible": None}, False),
    ],
)
def test_referenced_expansion_runtime_overrides_task_visibility_and_adds_hdr(
    task_visible, runtime_settings, expected, tmp_path: Path
) -> None:
    save_config(
        tmp_path / "expansion.yaml",
        {
            "runtime": {
                **runtime_settings,
                "env": {
                    "events": {
                        "lighting": {
                            "func": "randomize_indirect_lighting",
                            "mode": "reset",
                            "params": {"path": "unused-hdr"},
                        }
                    }
                },
            }
        },
    )
    task_path = tmp_path / "task.yaml"
    save_config(
        task_path,
        {
            "id": "EmbodiedEnv-v1",
            "physics": "default",
            "robot": {"uid": "robot"},
            "env": {},
            "ground_plane_visible": task_visible,
            "expansion": {"config": "expansion.yaml"},
        },
    )
    parser = argparse.ArgumentParser()
    add_env_launcher_args_to_parser(parser, require_gym_config=True)
    args = parser.parse_args(["--gym_config", str(task_path)])

    cfg, _, _ = build_env_cfg_from_args(args)

    assert cfg.ground_plane_visible is runtime_settings.get(
        "ground_plane_visible", task_visible
    )
    assert _env(cfg)._resolve_ground_plane_visibility() is expected


def test_visibility_is_applied_before_prepare_and_authored_config_is_reusable(
    monkeypatch,
) -> None:
    simulations = []

    def create_sim(*args, **kwargs):
        sim = SimpleNamespace(
            set_ground_plane_visibility=Mock(),
            profiler=Mock(),
            destroy=Mock(),
        )

        def stop_at_prepare():
            sim.set_ground_plane_visibility.assert_called_once()
            raise RuntimeError("stop before materialization")

        sim.prepare = Mock(side_effect=stop_at_prepare)
        simulations.append(sim)
        return sim

    monkeypatch.setattr(base_module, "SimulationManager", create_sim)
    monkeypatch.setattr(EmbodiedEnv, "_declare_robot", lambda *args, **kwargs: None)
    for method in ("_setup_background", "_setup_interactive_objects", "_setup_lights"):
        monkeypatch.setattr(EmbodiedEnv, method, lambda *args, **kwargs: None)
    cfg = EmbodiedEnvCfg(
        sim_cfg=SimulationManagerCfg(headless=True),
        events={
            "lighting": EventCfg(
                func=randomize_indirect_lighting, params={"path": "unused-hdr"}
            )
        },
    )

    for filtered in (False, True):
        cfg.filter_visual_rand = filtered
        env = EmbodiedEnv.__new__(EmbodiedEnv)
        with pytest.raises(RuntimeError, match="stop before materialization"):
            BaseEnv.__init__(env, cfg)
        simulations[-1].set_ground_plane_visibility.assert_called_once_with(filtered)
        assert cfg.ground_plane_visible is None
        assert cfg.events["lighting"] is not None
