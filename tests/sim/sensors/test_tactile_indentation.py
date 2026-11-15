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
    filtered = TactileIndentation(cfg.replace(use_candidates=True), context=context)
    dense = TactileIndentation(cfg, context=context)
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
                    "shear": {
                        "normal_stiffness": 100,
                        "shear_stiffness": 50,
                        "friction": 0.5,
                        "marker_indices": [10, 0],
                    },
                    "rgb": {
                        "x_response": [0.25, 0, 0],
                        "y_response": [0, 0.25, 0],
                        "background": [0.25, 0.5, 0.75],
                    },
                }
            )
        )
        sim.update(step=1, render_final_step=False)
        torch.testing.assert_close(
            sensor.get_data()["depth"], torch.full((2, 4, 4), 0.02)
        )
        completed_data = sensor.get_data().clone()
        completed_metadata = sensor.get_sample_metadata().clone()
        solid.set_local_pose(torch.tensor([[0.02, 0, 0.18, 0, 0, 0, 1]]), env_ids=[1])
        sensor.update()
        for key, expected in completed_data.items():
            torch.testing.assert_close(sensor.get_data()[key], expected)
        for key, expected in completed_metadata.items():
            torch.testing.assert_close(sensor.get_sample_metadata()[key], expected)

        solid.set_local_pose(torch.tensor([[0.0, 0, 0.18, 0, 0, 0, 1]]), env_ids=[1])
        sim.update(step=1, render_final_step=False)
        torch.testing.assert_close(
            sensor.get_sample_metadata()["physics_step"],
            completed_metadata["physics_step"] + 1,
        )
        torch.testing.assert_close(
            sensor.get_sample_metadata()["sample_id"],
            completed_metadata["sample_id"] + 1,
        )
        torch.testing.assert_close(
            sensor.get_data()["shear_displacement"],
            completed_data["shear_displacement"],
        )

        solid.set_local_pose(torch.tensor([[0.0, 0.0, 0.3, 0, 0, 0, 1.0]]), env_ids=[1])
        sim.update(step=1, render_final_step=False)
        data = sensor.get_data()
        assert data["hit"][0].all() and not data["hit"][1].any()
        torch.testing.assert_close(data["normal_force"][0], torch.full((4, 4), 2.0))
        assert not data["normal_force"][1].any()
        background = torch.tensor([0.25, 0.5, 0.75]).expand(4, 4, 3)
        torch.testing.assert_close(data["rgb"][1], background)
        solid.set_local_pose(torch.tensor([[0.1, 0, 0.18, 0, 0, 0, 1]]), env_ids=[0])
        sim.update(step=1, render_final_step=False)
        data = sensor.get_data()
        assert data["hit"][0].any() and (data["rgb"][0] != background).any()
        assert data["shear_force"][0, 2, 2, 0] > 0.0
        torch.testing.assert_close(
            data["marker_displacement"][0, 0], data["shear_displacement"][0, 2, 2]
        )
        sensor.reset([0])
        torch.testing.assert_close(sensor.get_data()["rgb"][0], background)
        assert not sensor.get_data()["marker_displacement"][0].any()
        assert sensor.get_sample_metadata()["valid"].tolist() == [False, True]
    finally:
        sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()


@pytest.mark.parametrize(
    ("layout", "device"),
    [
        ("flat", "cpu"),
        ("cap", "cpu"),
        pytest.param("explicit", "cuda:0", marks=pytest.mark.gpu),
    ],
)
def test_relative_shear_sticks_slips_and_clears_on_contact_change(
    layout: str, device: str
) -> None:
    trimesh = pytest.importorskip("trimesh")
    box = _mesh(trimesh.creation.box(extents=(0.2, 0.2, 0.04)), device)
    mount = torch.eye(4, device=device).repeat(2, 1, 1)
    first = mount.clone()
    second = mount.clone()
    second[:, 2, 3] = 0.3
    scene = RayScene([(box, first), (box, second)])
    context = SensorContext(
        torch.device(device), 2, None, None, ray_factory=lambda *_: scene
    )
    options = {}
    if layout == "cap":
        options["curvature_radius"] = 0.1
    elif layout == "explicit":
        options["points"] = (
            (-0.01, 0.0, -0.001),
            (0.0, 0.0, 0.0),
            (0.015, 0.0, -0.002),
        )
        options["normals"] = ((-0.1, 0.0, 1.0), (0.0, 0.0, 1.0), (0.1, 0.0, 1.0))

    sensor = TactileIndentation(
        TactileIndentationCfg(
            uid="finger",
            target_uids=("first", "second"),
            size=(0.03, 0.01),
            resolution=(3, 1),
            max_depth=0.02,
            shear=TactileIndentationCfg.ShearCfg(
                normal_stiffness=100.0,
                shear_stiffness=100.0,
                friction=0.25,
                marker_indices=(1,),
            ),
            **options,
        ),
        context=context,
    )
    sensor._bind(1, lambda: mount)

    def check_reading(
        step: int, forces: tuple[float, float], marker_x: tuple[float, float]
    ) -> None:
        sensor._sample(step, step * 0.01, step, 0.01)
        reading = sensor.get_data()
        torch.testing.assert_close(
            reading["normal_force"][:, 0, 1], torch.tensor(forces, device=device)
        )
        torch.testing.assert_close(
            reading["marker_displacement"][:, 0, 0],
            torch.tensor(marker_x, device=device),
        )
        torch.testing.assert_close(
            reading["shear_force"][:, 0, 1],
            reading["marker_displacement"][:, 0] * 100.0,
        )

    check_reading(1, (2.0, 2.0), (0.0, 0.0))
    first[:, 0, 3] += 0.004
    check_reading(2, (2.0, 2.0), (0.004, 0.004))
    first[:, 0, 3] += 0.006
    check_reading(3, (2.0, 2.0), (0.005, 0.005))
    first[0, 0, 3] -= 0.002
    first[1, 2, 3] = 0.3
    second[1, 2, 3] = 0.0
    check_reading(4, (2.0, 2.0), (0.003, 0.0))
    # A common translation has zero relative motion. Reset only the second row.
    sensor.reset([1])
    mount[:, 0, 3] += 0.03
    first[:, 0, 3] += 0.03
    second[1, 2, 3] = 0.3
    check_reading(5, (2.0, 0.0), (0.003, 0.0))


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.gpu)]
)
def test_shear_reads_strided_mount_and_target_poses(device: str) -> None:
    trimesh = pytest.importorskip("trimesh")
    box = _mesh(trimesh.creation.box(extents=(0.2, 0.2, 0.04)), device)
    identity = torch.eye(4, device=device).repeat(2, 1, 1)
    mount = identity.clone().transpose(-1, -2)
    target = identity.clone().transpose(-1, -2)
    assert not mount.is_contiguous() and not target.is_contiguous()
    scene = RayScene([(box, lambda: target)])
    context = SensorContext(
        torch.device(device), 2, None, None, ray_factory=lambda *_: scene
    )
    sensor = TactileIndentation(
        TactileIndentationCfg(
            uid="pad",
            target_uids=("target",),
            resolution=(1, 1),
            max_depth=0.02,
            shear=TactileIndentationCfg.ShearCfg(
                normal_stiffness=100.0,
                shear_stiffness=50.0,
                friction=0.5,
                marker_indices=(0,),
            ),
        ),
        context=context,
    )
    sensor._bind(1, lambda: mount)
    sensor._sample(1, 0.01, 1, 0.01)
    mount[0, 0, 3] = 0.001
    target[:, 0, 3] = torch.tensor([0.003, -0.004], device=device)
    sensor._sample(2, 0.02, 2, 0.01)

    data = sensor.get_data()
    expected = torch.tensor([[0.002, 0.0], [-0.004, 0.0]], device=device)
    assert data["shape_id"][:, 0, 0].tolist() == [0, 0]
    torch.testing.assert_close(
        data["normal_force"][:, 0, 0], torch.full((2,), 2.0, device=device)
    )
    torch.testing.assert_close(data["marker_displacement"][:, 0], expected)
    torch.testing.assert_close(data["shear_force"][:, 0, 0], 50.0 * expected)
