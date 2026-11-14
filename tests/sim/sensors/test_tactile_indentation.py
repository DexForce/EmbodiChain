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


"""Closed-surface indentation, mesh identity and native pose integration."""

from __future__ import annotations

import math
import pytest
import torch

from embodichain.compute.geometry.raycast import TriangleRayMesh
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import DefaultPhysicsCfg, RigidObjectCfg
from embodichain.lab.sim.sensors import (
    RayScene,
    SensorCfg,
    SensorContext,
    TactileIndentation,
    TactileIndentationCfg,
)
from embodichain.lab.sim.shapes import CubeCfg


def _mesh(mesh, device: str) -> TriangleRayMesh:
    return TriangleRayMesh(
        torch.tensor(mesh.vertices, device=device),
        torch.tensor(mesh.faces, device=device),
    )


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_curved_indentation_matches_dense_and_analytic_solids(device: str) -> None:
    trimesh = pytest.importorskip("trimesh")
    cylinder = _mesh(
        trimesh.creation.cylinder(radius=0.5, height=0.3, sections=64), device
    )
    sphere = _mesh(trimesh.creation.icosphere(subdivisions=3, radius=0.5), device)
    flat = _mesh(trimesh.creation.box(extents=(0.005, 0.2, 0.2)), device)
    poses = torch.eye(4, device=device).repeat(4, 1, 1)
    first = poses.clone()
    first[:, 2, 3] = torch.tensor([0.05, 0.17, 0.15, 2.0], device=device)
    second = poses.clone()
    second[:, 2, 3] = torch.tensor([0.55, 0.45, 0.49, 3.0], device=device)
    third = poses.clone()
    third[:, 0, 3] = torch.tensor([-0.3, 0.25, 0.1, -0.3], device=device)
    third[:, 2, 3] = torch.tensor([0.0, 0.0, 0.2, 0.0], device=device)
    scene = RayScene([(cylinder, first), (sphere, second), (flat, third)])
    cfg = TactileIndentationCfg(
        uid="map",
        target_uids=("cylinder", "sphere", "thin"),
        resolution=(3, 1),
        size=(0.9, 0.15),
        curvature_radius=1.0,
        max_depth=0.75,
    )
    context = SensorContext(
        torch.device(device), 4, None, None, ray_factory=lambda *_: scene
    )
    filtered = TactileIndentation(cfg, context=context)
    dense = TactileIndentation(cfg.replace(use_candidates=False), context=context)
    for sensor in (filtered, dense):
        sensor._bind(1, lambda: poses)
        sensor._sample(1, 0.01, 1, 0.01)
    for name in cfg.get_data_types():
        torch.testing.assert_close(filtered.get_data()[name], dense.get_data()[name])
    actual = filtered.get_data()
    z = math.sqrt(1 - 0.3**2)
    thin_exit = 0.0025 / 0.3
    # The center of row 1 is inside the sphere. Its two closed intersections
    # cannot be confused with the separate cylinder in the same environment.
    expected = torch.tensor(
        [
            [[(0.2 + 1 - z) / z, 0.2, (0.2 + 1 - z) / z]],
            [[0.0, 0.75, 0.0]],
            [[0.0, 0.75, 0.0]],
            [[thin_exit, 0.0, 0.0]],
        ],
        device=device,
    )
    torch.testing.assert_close(actual["depth"], expected, atol=1e-4, rtol=0)
    assert actual["shape_id"].tolist() == [
        [[0, 0, 0]],
        [[-1, 1, -1]],
        [[-1, 1, -1]],
        [[2, -1, -1]],
    ]
    second[:, 2, 3] += 1.0
    filtered._sample(2, 0.02, 2, 0.01)
    assert not filtered.get_data()["hit"][1:3].any()
    filtered.reset([0])
    assert filtered.get_sample_metadata()["valid"].tolist() == [False, True, True, True]
    assert (filtered.get_data()["shape_id"][0] == -1).all()


def test_indentation_rejects_open_target() -> None:
    mesh = TriangleRayMesh(
        torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        torch.tensor([[0, 1, 2]]),
    )
    context = SensorContext(
        torch.device("cpu"),
        1,
        None,
        None,
        ray_factory=lambda *_: RayScene([(mesh, torch.eye(4)[None])]),
    )
    sensor = TactileIndentation(
        TactileIndentationCfg(uid="bad", target_uids=("open",)), context=context
    )
    with pytest.raises(ValueError, match="closed meshes"):
        sensor._bind(0, lambda: torch.eye(4)[None])


@pytest.mark.requires_sim
def test_registered_indentation_observes_motion_in_only_selected_arena() -> None:
    sim = SimulationManager(
        SimulationManagerCfg(
            headless=True, num_envs=2, physics_cfg=DefaultPhysicsCfg(device="cpu")
        )
    )
    try:
        solid = sim.add_rigid_object(
            RigidObjectCfg(
                uid="solid",
                shape=CubeCfg(size=(0.2, 0.2, 0.1)),
                body_type="kinematic",
                init_pos=(0, 0, 0.18),
            )
        )
        sensor = sim.add_sensor(
            SensorCfg.from_dict(
                {
                    "sensor_type": "TactileIndentation",
                    "uid": "indent",
                    "target_uids": ["solid"],
                    "size": [0.08, 0.08],
                    "resolution": [4, 4],
                    "mount": {"pos": [0, 0, 0.15]},
                    "max_depth": 0.02,
                }
            )
        )
        sim.update(step=1, render_final_step=False)
        torch.testing.assert_close(
            sensor.get_data()["depth"], torch.full((2, 4, 4), 0.02)
        )
        solid.set_local_pose(torch.tensor([[0.0, 0.0, 0.3, 0, 0, 0, 1.0]]), env_ids=[1])
        sim.update(step=1, render_final_step=False)
        assert (
            sensor.get_data()["hit"][0].all() and not sensor.get_data()["hit"][1].any()
        )
    finally:
        sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()
