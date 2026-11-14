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


"""Contact-grid conservation, authority and real scene acquisition."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import DefaultPhysicsCfg, RigidObjectCfg
from embodichain.lab.sim.sensors import (
    ContactGrid,
    ContactGridCfg,
    SensorCfg,
    SensorContext,
)
from embodichain.lab.sim.shapes import CubeCfg


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_grid_conserves_rows_after_actor_flip_and_marks_partial_data(
    device: str,
) -> None:
    def tensor(value, dtype=None):
        return torch.tensor(value, device=device, dtype=dtype)

    xyz = [[-0.5, 0.0, 0.0], [-0.4, 0.0, 0.0], [0.5, 0.0, 0.0], [0.2, 0.0, 0.3]]
    positions = tensor([xyz, xyz])
    positions[1, :, 0] += 5
    reader = SimpleNamespace(
        normal_impulse_available=True,
        contact_capabilities=SimpleNamespace(impulse=True, friction=True),
        get_actor_ids=lambda *args: tensor([[4], [7]], torch.int32),
        update=lambda: None,
        destroy=lambda: None,
        get_impulse_duration=lambda dt: dt / 2,
        get_latest_contact_loss_mask=lambda: tensor([False, True]),
        reset=lambda env_ids: None,
        get_data=lambda: {
            "position": positions,
            "normal": tensor(
                [[[0.0, 0.0, -1.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [0.0, 0.0, 1.0]]]
                * 2
            ),
            "friction": tensor(
                [
                    [
                        [1.0, 0.0, 0.0],
                        [-2.0, 0.0, 0.0],
                        [float("nan"), 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                    ]
                ]
                * 2
            ),
            "impulse": tensor([[2.0, 3.0, 7.0, 3.0]] * 2),
            "user_ids": tensor(
                [
                    [[4, 10], [11, 4], [4, 12], [4, 10]],
                    [[7, 10], [11, 7], [7, 12], [7, 10]],
                ],
                torch.int32,
            ),
            "is_valid": tensor([[True] * 4] * 2),
            "impulse_valid": tensor([[True] * 4] * 2),
            "friction_valid": tensor([[True, True, False, True]] * 2),
        },
    )
    grid = ContactGrid(
        ContactGridCfg(uid="grid", asset_uid="pad", size=(2.0, 2.0), resolution=(2, 1)),
        context=SensorContext(
            torch.device(device), 2, None, None, contact_factory=lambda _: reader
        ),
    )
    mount = torch.eye(4, device=device).repeat(2, 1, 1)
    mount[1, 0, 3] = 5
    grid._bind(0, lambda: mount)
    try:
        grid._sample(1, 0.02, 1, 0.02)
        data = grid.get_data()
        assert data["count"].tolist() == [[[2, 1]], [[2, 1]]]
        torch.testing.assert_close(
            data["normal_impulse"][..., 2], tensor([[[5.0, 7.0]]] * 2)
        )
        torch.testing.assert_close(
            data["friction_impulse"][..., 0], tensor([[[3.0, 0.0]]] * 2)
        )
        assert data["friction_valid"].tolist() == [[[True, False]], [[False, False]]]
        assert data["normal_valid"].tolist() == [[[True, True]], [[False, False]]]
        torch.testing.assert_close(data["impulse_duration"], tensor([0.01, 0.01]))
        grid.reset([0])
        assert grid.get_sample_metadata()["valid"].tolist() == [False, True]
        assert data["count"].tolist() == [[[0, 0]], [[2, 1]]]
    finally:
        grid._unbind()


@pytest.mark.requires_sim
def test_registered_grid_reads_native_default_cpu_total_impulse() -> None:
    sim = SimulationManager(
        SimulationManagerCfg(
            headless=True, num_envs=2, physics_cfg=DefaultPhysicsCfg(device="cpu")
        )
    )
    try:
        sim.add_rigid_object(
            RigidObjectCfg(
                uid="pad",
                shape=CubeCfg(size=(0.3, 0.3, 0.04)),
                body_type="static",
                init_pos=(0.0, 0.0, 0.05),
            )
        )
        sim.add_rigid_object(
            RigidObjectCfg(
                uid="indenter",
                shape=CubeCfg(size=(0.1, 0.1, 0.08)),
                init_pos=(0.03, 0.0, 0.15),
            )
        )
        sensor = sim.add_sensor(
            SensorCfg.from_dict(
                {
                    "uid": "grid",
                    "sensor_type": "ContactGrid",
                    "asset_uid": "pad",
                    "size": [0.3, 0.3],
                    "resolution": [4, 4],
                    "mount": {"parent": "pad", "pos": [0, 0, 0.02]},
                }
            )
        )
        for _ in range(40):
            sim.update(step=1, render_final_step=False)
            if (sensor.get_data()["count"].sum(dim=(1, 2)) > 0).all():
                break
        data = sensor.get_data()
        assert (data["count"].sum(dim=(1, 2)) > 0).all()
        assert (data["impulse_magnitude"].sum(dim=(1, 2)) > 0).all()
        assert data["magnitude_valid"].all() and not data["normal_valid"].any()
        assert sensor.get_data_descriptors()["impulse_magnitude"]["quantity"] == "total"
    finally:
        sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()
