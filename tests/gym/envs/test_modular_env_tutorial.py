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

import runpy
from pathlib import Path

import pytest

from embodichain.lab.sim.spawn.descriptors import rigid_desc_from_cfg

pytestmark = pytest.mark.no_sim


def test_tutorial_rigid_objects_compile_to_spawn_descriptors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Resolve paths without downloading assets; descriptor construction is pure.
    monkeypatch.setattr("embodichain.data.get_data_path", lambda path: path)
    namespace = runpy.run_path(
        str(
            Path(__file__).resolve().parents[3] / "scripts/tutorials/gym/modular_env.py"
        )
    )
    cfg = namespace["ExampleCfg"]()

    for object_cfg in [*cfg.background, *cfg.rigid_object]:
        descriptor, _ = rigid_desc_from_cfg(object_cfg)
        assert descriptor.physics is not None
        assert descriptor.renders
        assert descriptor.collisions


@pytest.fixture
def tutorial(monkeypatch):
    monkeypatch.setattr("embodichain.data.get_data_path", lambda path: path)
    return runpy.run_path(
        str(
            Path(__file__).resolve().parents[3] / "scripts/tutorials/gym/modular_env.py"
        )
    )


def test_initial_fork_matches_first_event_draw_without_mutating_config(
    tutorial, monkeypatch
):
    import random
    from embodichain.lab.gym.envs.managers.event_manager import _derive_functor_seed

    paths = ["fork/c.ply", "fork/a.ply", "fork/b.ply"]
    monkeypatch.setitem(
        tutorial["_preselect_initial_fork"].__globals__,
        "get_all_files_in_directory",
        lambda path: paths,
    )
    cfg = tutorial["ExampleCfg"](seed=123)
    original_path = cfg.rigid_object[0].shape.fpath
    state = random.getstate()
    prepared, path = tutorial["_preselect_initial_fork"](cfg)
    expected = random.Random(
        _derive_functor_seed(123, "call", "reset", "replace_obj", 0)
    ).choice(sorted(paths))
    assert path == expected
    assert prepared.rigid_object[0].shape.fpath == expected
    assert cfg.rigid_object[0].shape.fpath == original_path
    assert random.getstate() == state
    assert prepared.events.randomize_fork_mass.mode == "reset"


@pytest.mark.parametrize("matches", [False, True])
def test_only_preselected_first_replacement_is_skipped(tutorial, monkeypatch, matches):
    import random
    from types import SimpleNamespace
    from unittest.mock import Mock
    from embodichain.lab.gym.envs.managers import events

    term = object.__new__(tutorial["_ReplacePreselectedFork"])
    term._asset_group_path = ["fork/a.ply", "fork/b.ply"]
    generator = random.Random(37)
    selected = generator.choice(term._asset_group_path)
    expected_next = generator.random()
    term.asset_cfg = SimpleNamespace(shape=SimpleNamespace(fpath=selected))
    original_cfg = term.asset_cfg
    env = SimpleNamespace(_initial_fork_path=selected if matches else "other.ply")
    calls = []

    def replace(self, *args, **kwargs):
        calls.append(random.choice(self._asset_group_path))

    monkeypatch.setattr(events.replace_assets_from_group, "__call__", replace)
    state = random.getstate()
    try:
        random.seed(37)
        term(env, None, Mock(), "fork/")
        assert len(calls) == (0 if matches else 1)
        if matches:
            assert term.asset_cfg is not original_cfg
        assert random.random() == expected_next
        term(env, None, Mock(), "fork/")
        assert len(calls) == (1 if matches else 2)
    finally:
        random.setstate(state)


def test_unseeded_tutorial_keeps_existing_reset_behavior(tutorial):
    cfg = tutorial["ExampleCfg"](seed=None)
    prepared, path = tutorial["_preselect_initial_fork"](cfg)
    assert prepared is cfg
    assert path is None


def test_preselection_precedes_environment_construction(tutorial, monkeypatch):
    prepare = tutorial["_preselect_initial_fork"]
    monkeypatch.setitem(
        prepare.__globals__,
        "get_all_files_in_directory",
        lambda path: ["fork/first.ply"],
    )
    cfg = tutorial["ExampleCfg"](seed=123)
    observed = []

    def construct(env, prepared, **kwargs):
        observed.append((prepared.rigid_object[0].shape.fpath, env._initial_fork_path))

    monkeypatch.setattr(tutorial["EmbodiedEnv"], "__init__", construct)
    tutorial["ModularEnv"](cfg)
    assert observed == [("fork/first.ply", "fork/first.ply")]


@pytest.mark.parametrize("events_enabled", [False, True])
def test_custom_event_configuration_keeps_normal_construction(tutorial, events_enabled):
    cfg = tutorial["ExampleCfg"](seed=123)
    if events_enabled:
        cfg.events.replace_obj.mode = "startup"
    else:
        cfg.events = None
    prepared, path = tutorial["_preselect_initial_fork"](cfg)
    assert prepared is cfg
    assert path is None
