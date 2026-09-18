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

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from embodichain.lab.sim.objects.articulation import Articulation


def _articulation() -> Articulation:
    articulation = object.__new__(Articulation)
    articulation.device = torch.device("cpu")
    articulation._entities = [object()] * 3
    articulation._data = SimpleNamespace(
        articulation_view=SimpleNamespace(apply_root_velocity=Mock())
    )
    return articulation


def test_default_selection_writes_all_root_velocities() -> None:
    articulation = _articulation()
    values = torch.arange(18, dtype=torch.float64).reshape(3, 6)
    articulation.set_root_velocity(values)
    actual, ids = (
        articulation._data.articulation_view.apply_root_velocity.call_args.args
    )
    torch.testing.assert_close(actual, values.float())
    assert ids.tolist() == [0, 1, 2]


def test_root_velocity_preserves_requested_row_order() -> None:
    articulation = _articulation()
    values = torch.arange(12, dtype=torch.float32).reshape(2, 6)
    articulation.set_root_velocity(values, env_ids=[2, 0])
    actual, ids = (
        articulation._data.articulation_view.apply_root_velocity.call_args.args
    )
    torch.testing.assert_close(actual, values)
    assert ids.tolist() == [2, 0]


@pytest.mark.parametrize("shape", [(2, 6), (3, 5)])
def test_invalid_root_velocity_shape_does_not_write(shape: tuple[int, int]) -> None:
    articulation = _articulation()
    with pytest.raises(ValueError, match="Root velocity must have shape"):
        articulation.set_root_velocity(torch.zeros(shape))
    articulation._data.articulation_view.apply_root_velocity.assert_not_called()


@pytest.mark.requires_sim
@pytest.mark.parametrize(
    "backend, device",
    [
        ("default", "cpu"),
        pytest.param("default", "cuda:0", marks=pytest.mark.gpu),
        pytest.param("newton", "cuda:0", marks=pytest.mark.gpu),
    ],
)
def test_live_root_velocity_restores_failed_selected_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str, device: str
) -> None:
    from dexsim.scene import ArticulationBatch
    from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
    from embodichain.lab.sim.cfg import ArticulationCfg, physics_cfg_for_backend

    source = tmp_path / "root_velocity.urdf"
    link = """<link name="{name}">
      <inertial><mass value="1"/>
      <inertia ixx=".01" ixy="0" ixz="0" iyy=".01" iyz="0" izz=".01"/></inertial>
      <visual><geometry><box size=".1 .1 .1"/></geometry></visual>
      <collision><geometry><box size=".1 .1 .1"/></geometry></collision>
    </link>"""
    source.write_text(
        '<robot name="root_velocity">'
        + link.format(name="base")
        + link.format(name="tip")
        + '<joint name="hinge" type="revolute"><parent link="base"/><child link="tip"/>'
        '<origin xyz="0 0 .3"/><axis xyz="0 1 0"/>'
        '<limit lower="-1" upper="1" effort="5" velocity="5"/></joint></robot>'
    )
    sim = SimulationManager(
        SimulationManagerCfg(
            headless=True,
            device=device,
            num_envs=3,
            physics_cfg=physics_cfg_for_backend(backend),
        )
    )
    art = None
    try:
        art = sim.add_articulation(
            ArticulationCfg(uid="root_velocity", fpath=str(source), init_pos=(0, 0, 1))
        )
        sim.prepare()
        sim.update()
        initial = (
            torch.arange(18, dtype=torch.float32, device=device).reshape(3, 6) / 10
        )
        art.set_root_velocity(initial)
        torch.testing.assert_close(art.body_data.root_vel, initial)
        original = ArticulationBatch.apply_root_angular_velocity
        failed = False

        def fail_after_write(batch: ArticulationBatch, values: torch.Tensor) -> int:
            nonlocal failed
            result = original(batch, values)
            if not failed:
                failed = True
                raise RuntimeError("injected angular write failure")
            return result

        monkeypatch.setattr(
            ArticulationBatch, "apply_root_angular_velocity", fail_after_write
        )
        selected = [2, 0]
        requested = -initial[selected] - 1.0
        with pytest.raises(RuntimeError, match="injected angular write failure"):
            art.set_root_velocity(requested, env_ids=selected)
        torch.testing.assert_close(art.body_data.root_vel, initial)

        art.set_root_velocity(requested, env_ids=selected)
        expected = initial.clone()
        expected[selected] = requested
        torch.testing.assert_close(art.body_data.root_vel, expected)
        sim.update()
        assert torch.isfinite(art.body_data.root_vel).all()
    finally:
        art = None
        sim.destroy()
        SimulationManager.flush_cleanup_queue()
