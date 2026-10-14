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

"""Online fluid deployment and measured physical qualification."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from embodichain.lab.scripts.run_fluid_task_program import (
    _configure,
    _cup_qualification,
)

pytestmark = pytest.mark.no_sim
ROOT = Path(__file__).resolve().parents[3]
TASK = (
    ROOT
    / "embodichain_tasks/configs/tasks/manipulation/tableware/pour_cup/task.franka.fluid.yaml"
)


@pytest.fixture(autouse=True)
def task_mesh_cache(tmp_path, monkeypatch):
    """Resolve task meshes locally so configuration tests never download them."""
    from types import SimpleNamespace
    from embodichain.data import dataset

    directory = tmp_path / "PourCupAssets"
    directory.mkdir()
    for name in (
        "cup.glb",
        "mug_wall.obj",
        "receiver_wall.obj",
        "fluid_table.obj",
        "fluid_floor.obj",
    ):
        (directory / name).touch()
    original = dataset.get_data_class

    def resolve(name, extra_modules=None):
        if name == "PourCupAssets":
            return lambda: SimpleNamespace(extract_dir=str(directory))
        return original(name, extra_modules)

    monkeypatch.setattr(dataset, "get_data_class", resolve)
    return directory


def test_online_task_uses_live_grasp_and_same_body_particle_boundary(tmp_path):
    task_id, cfg = _configure(TASK, num_envs=2, renderer="fast-rt", output=tmp_path)
    assert task_id == "PourCup-Franka-Fluid-v1"
    assert cfg.task_program is not None
    cup = next(c for c in cfg.rigid_object if c.uid == "cup")
    assert cup.body_type == "dynamic"
    assert Path(cup.particle_collision_shape.fpath).is_absolute()
    assert cup.particle_collision_shape.collision.approximation == "sdf"
    assert cup.shape.collision.approximation == "convex_decomposition"
    solver = cfg.sim_cfg.physics_cfg.solver_cfg
    assert solver["rigid_solver_cfg"]["solver_type"] == "mujoco_warp"
    assert solver["rigid_solver_substeps"] == 7
    assert cfg.events.record_camera.params["per_env"] is True
    assert cfg.events.record_camera.params["max_env_num"] == 2
    assert cfg.record_trajectory is False
    assert cfg.sim_steps_per_control * cfg.sim_cfg.physics_dt == pytest.approx(0.025)
    assert not hasattr(cfg.events, "settle_cup")


def test_no_render_coarse_override_resamples_and_keeps_physical_scale(tmp_path):
    _, cfg = _configure(
        TASK, num_envs=1, renderer="no-render", output=tmp_path, particle_spacing=0.003
    )
    fluid = cfg.fluid_object[0]
    assert fluid.render.mode == "none"
    assert fluid.particle_radius == pytest.approx(0.0009)
    assert fluid.render.mesh_surface["kernel_radius"] == pytest.approx(0.0045)
    assert cfg.sim_cfg.physics_cfg.solver_cfg[
        "particle_support_radius"
    ] == pytest.approx(0.00675)
    assert cfg.events.record_camera is None


def test_physical_qualification_does_not_accept_motion_commands_alone():
    initial = np.array([[0, 0, 0.5, 0, 0, 0, 1], [1, 0, 0.5, 0, 0, 0, 1]], dtype=float)
    moving = initial.copy()
    moving[0, 2] += 0.12
    moving[0, 3] = np.sin(np.deg2rad(35))
    moving[0, 6] = np.cos(np.deg2rad(35))
    result = _cup_qualification([initial, moving, initial])
    assert result["accepted"] == [True, False]
    assert result["max_tilt_deg"] == pytest.approx([70, 0])
    assert result["lift_m"] == pytest.approx([0.12, 0])


@pytest.mark.parametrize("renderer", ["fast-rt", "no-render"])
def test_online_runner_commits_last_recording_before_close(
    renderer, tmp_path, monkeypatch
):
    import argparse
    from types import SimpleNamespace
    from unittest.mock import Mock
    import torch
    from embodichain.lab.scripts import run_fluid_task_program as runner

    order = []
    recorder = SimpleNamespace(
        save_and_clear=Mock(side_effect=lambda: order.append("save"))
    )
    env = SimpleNamespace(
        num_envs=2,
        step_dt=0.05,
        control_frequency=20,
        elapsed_steps=torch.zeros(2),
        reset=Mock(),
        event_manager=SimpleNamespace(get_functor=lambda name: recorder),
        sim=SimpleNamespace(
            get_rigid_object=lambda uid: SimpleNamespace(
                get_local_pose=lambda: torch.tensor([[0.0, 0, 0, 0, 0, 0, 1]]).repeat(
                    2, 1
                )
            ),
            get_fluid_object=lambda uid: SimpleNamespace(num_particles=1),
        ),
        close=Mock(side_effect=lambda **kwargs: order.append("close")),
    )
    result = SimpleNamespace(
        completed=False, all_success=False, to_metadata=lambda: {"completed": False}
    )
    monkeypatch.setattr(runner, "make", lambda *args, **kwargs: env)
    monkeypatch.setattr(runner, "discover_task_packages", lambda: None)
    monkeypatch.setattr(runner, "execute_demo_episode", lambda *args, **kwargs: result)
    monkeypatch.setattr(runner, "_sample", lambda env: {"received_ml": [0.0, 0.0]})
    monkeypatch.setattr(runner.SimulationManager, "flush_cleanup_queue", lambda: None)
    args = argparse.Namespace(
        task=str(TASK),
        num_envs=2,
        renderer=renderer,
        particle_spacing=None,
        record_trajectory=False,
        seconds=0.05,
        settle_seconds=0,
    )
    report = runner._run(args, tmp_path)
    assert report["accepted"] is False
    assert order == (["save", "close"] if renderer == "fast-rt" else ["close"])
    assert len(report["videos"]) == (2 if renderer == "fast-rt" else 0)
    assert (tmp_path / "fluid_gym_result.json").is_file()
