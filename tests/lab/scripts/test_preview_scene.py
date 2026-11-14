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

from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from embodichain.lab.scripts.preview_scene import preview_scene
from embodichain.lab.scripts._scene_preview import PreviewControls


@pytest.fixture
def preview_world(tmp_path, monkeypatch):
    source = tmp_path / "plain.usda"
    source.write_text('#usda 1.0\n(defaultPrim = "World")\ndef Xform "World" {}\n')
    instances = []
    failure = {"stage": None}

    class Simulation:
        def __init__(self, cfg):
            self.sim_config = cfg
            self.physics = NS(name="test-backend")
            self.simulation_time = 0.0
            self.visualization_runtime = None
            self.events = []
            instances.append(self)

        def add_usd(self, cfg):
            self.events.append("load")
            assert cfg.source.file_path == str(source)
            if failure["stage"] == "load":
                raise RuntimeError("unsupported USD feature")
            return {"/World/Visual": NS(cfg=NS(uid="visual"))}

        def prepare(self):
            self.events.append("prepare")
            if failure["stage"] == "prepare":
                raise RuntimeError("native materialization failed")

        def sync_render_state(self):
            self.events.append("publish")

        def update(self, *, step):
            assert step == 1
            self.events.append("physics")
            self.simulation_time += 0.01

        @contextmanager
        def render_frame(self, **kwargs):
            self.events.append("display")
            yield

        def destroy(self, *, exit_process):
            assert exit_process is False
            self.events.append("destroy")

        @staticmethod
        def flush_cleanup_queue():
            instances[-1].events.append("flush")

    monkeypatch.setattr("embodichain.lab.sim.SimulationManager", Simulation)
    monkeypatch.setattr(
        "embodichain.lab.scripts._scene_preview.camera_view",
        lambda _: ([2.0, -3.0, 2.0], [0.0, 0.0, 0.0]),
    )
    cfg = NS(
        headless=True, visualization=NS(backend="none"), render_cfg=NS(renderer="auto")
    )
    return source, cfg, instances, failure


@pytest.mark.parametrize("steps", [None, 3], ids=["paused", "simulate"])
def test_plain_usd_headless_preserves_explicit_stepping_and_cleanup(
    preview_world, tmp_path, steps
):
    source, cfg, instances, _ = preview_world
    report = tmp_path / "report.json"
    result = preview_scene(
        source, sim_cfg=cfg, simulate=steps is not None, steps=steps, report=report
    )
    assert instances[0].sim_config.default_ground is False
    assert not hasattr(cfg, "default_ground")
    assert cfg.render_cfg.renderer == "auto"
    assert instances[0].sim_config.render_cfg.renderer == "hybrid"
    assert result["physics_steps"] == (0 if steps is None else steps)
    assert result["simulation_time"] == pytest.approx(0.0 if steps is None else 0.03)
    assert result["entities"][0]["prim_path"] == "/World/Visual"
    assert instances[0].events.count("physics") == (0 if steps is None else steps)
    assert instances[0].events[-2:] == ["destroy", "flush"]
    assert json.loads(report.read_text()) == result


@pytest.mark.parametrize("stage", ["load", "prepare"])
def test_failed_scene_import_writes_stage_diagnostic_and_closes_owner(
    preview_world, tmp_path, stage
):
    source, cfg, instances, failure = preview_world
    failure["stage"] = stage
    report = tmp_path / "error.json"
    with pytest.raises(RuntimeError):
        preview_scene(source, sim_cfg=cfg, report=report)
    payload = json.loads(report.read_text())
    assert payload["status"] == "error"
    assert payload["diagnostics"][0]["stage"] == stage
    assert payload["physics_steps"] == 0
    assert instances[0].events[-2:] == ["destroy", "flush"]


@pytest.mark.parametrize("simulate,steps", [(True, None), (False, 1), (True, 0)])
def test_headless_physics_requires_an_explicit_finite_budget(
    preview_world, simulate, steps
):
    source, cfg, instances, _ = preview_world
    with pytest.raises(ValueError):
        preview_scene(source, sim_cfg=cfg, simulate=simulate, steps=steps)
    assert instances == []


def test_report_cannot_overwrite_source(preview_world):
    source, cfg, _, _ = preview_world
    original = source.read_bytes()
    with pytest.raises(ValueError, match="distinct"):
        preview_scene(source, sim_cfg=cfg, report=source)
    assert source.read_bytes() == original


def test_paused_interactive_frames_do_not_integrate_physics(preview_world):
    source, cfg, instances, _ = preview_world
    cfg.visualization.backend = "viser"
    frames = []

    def callback(sim):
        if len(frames) == 2:
            raise KeyboardInterrupt()
        frames.append(sim)
        return True

    result = preview_scene(source, sim_cfg=cfg, on_frame=callback)
    assert result["physics_steps"] == 0
    assert instances[0].events.count("display") == 2
    assert instances[0].events[-2:] == ["destroy", "flush"]


def test_reset_recreates_the_owned_world_instead_of_partial_state_restore(
    preview_world,
):
    from embodichain.lab.scripts._scene_preview import ReloadPreview

    source, cfg, instances, _ = preview_world
    cfg.visualization.backend = "viser"

    def callback(sim):
        if len(instances) == 1:
            raise ReloadPreview()
        assert instances[0].events[-2:] == ["destroy", "flush"]
        raise KeyboardInterrupt()

    result = preview_scene(source, sim_cfg=cfg, on_frame=callback)
    assert result["resets"] == 1
    assert len(instances) == 2
    assert result["physics_steps"] == 0
    assert instances[1].events[-2:] == ["destroy", "flush"]


def test_step_pause_play_commands_have_exact_budgets():
    controls = PreviewControls(False)
    controls.enqueue("step")
    controls.enqueue("step")
    controls.update(None)
    assert [controls.advance() for _ in range(3)] == [True, True, False]
    controls.enqueue("play")
    controls.update(None)
    assert controls.advance() is True
    controls.enqueue("pause")
    controls.update(None)
    assert controls.advance() is False


@pytest.mark.parametrize("argument", ["--output_root", "--scene", "--usd"])
def test_removed_gensim_preview_arguments_are_not_accepted(argument):
    from embodichain.lab.scripts.preview_scene import main

    with pytest.raises(SystemExit) as error:
        main(["--usd-file", "plain.usda", argument, "old"])
    assert error.value.code == 2


def test_stop_is_not_lost_behind_a_full_step_queue():
    controls = PreviewControls(False)
    for _ in range(64):
        controls.enqueue("step")
    controls.enqueue("stop")
    controls.update(None)
    assert controls.stopped
    assert controls.pending_steps == 0
