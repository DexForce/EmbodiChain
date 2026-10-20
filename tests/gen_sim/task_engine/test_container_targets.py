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
import numpy as np
import pytest
import trimesh
from shapely.affinity import translate
from shapely.geometry import MultiPoint, box

from embodichain.gen_sim.task_engine._task_program.container_targets import (
    select_container_landing,
)


@pytest.mark.parametrize("side", [-1.0, 1.0])
def test_landing_uses_arm_side_and_erodes_by_full_footprint(side: float) -> None:
    floor = trimesh.creation.box(extents=[0.5, 0.3, 0.01])
    child = np.asarray(trimesh.creation.box(extents=[0.08, 0.08, 0.08]).vertices)
    target = select_container_landing(floor, child, np.array([side, 0.0, 0.3]))

    assert target is not None
    wall_margin = 0.02
    footprint_half_width = float(np.abs(child[:, 0]).max())
    assert target[0] == pytest.approx(
        side * (0.25 - footprint_half_width - wall_margin)
    )
    assert target[1] == pytest.approx(0.0)
    assert target[2] == pytest.approx(0.005 + 0.04 + 0.001)


def test_landing_preserves_holes_in_support_mesh() -> None:
    floor = trimesh.creation.annulus(r_min=0.06, r_max=0.2, height=0.01)
    child = np.asarray(trimesh.creation.box(extents=[0.02, 0.02, 0.02]).vertices)
    target = select_container_landing(floor, child, np.array([0.0, 0.0, 0.3]))

    assert target is not None
    from embodichain.gen_sim.task_engine._task_program.container_targets import (
        _floor_region,
    )

    measured, _ = _floor_region(floor)
    footprint = translate(
        MultiPoint(child[:, :2]).convex_hull, xoff=target[0], yoff=target[1]
    )
    assert measured.covers(footprint)
    assert np.linalg.norm(target[:2]) > 0.06


def test_landing_does_not_invent_room_for_an_oversized_object() -> None:
    floor = trimesh.creation.box(extents=[0.1, 0.1, 0.01])
    child = np.asarray(trimesh.creation.box(extents=[0.2, 0.2, 0.2]).vertices)
    with pytest.raises(ValueError, match="footprint"):
        select_container_landing(floor, child, np.array([1.0, 0.0, 0.3]))


def test_landing_reserves_space_for_prior_object() -> None:
    floor = trimesh.creation.box(extents=[0.5, 0.3, 0.01])
    child = np.asarray(trimesh.creation.box(extents=[0.08, 0.08, 0.08]).vertices)
    arm_root = np.array([1.0, 0.0, 0.3])
    first = select_container_landing(floor, child, arm_root)
    assert first is not None
    radius = float(np.linalg.norm(child[:, :2], axis=1).max())

    second = select_container_landing(
        floor, child, arm_root, occupied=[(first[0], first[1], radius)]
    )

    assert second is not None
    assert first == pytest.approx((0.19, 0.0, 0.046), abs=1e-12, rel=0)
    assert second == pytest.approx((0.09210365148000713, 0.09, 0.046), abs=1e-12, rel=0)
    assert np.linalg.norm(np.asarray(second[:2]) - np.asarray(first[:2])) >= (
        2 * radius + 0.019
    )


def test_landing_rejects_full_floor_without_overlapping() -> None:
    floor = trimesh.creation.box(extents=[0.18, 0.18, 0.01])
    child = np.asarray(trimesh.creation.box(extents=[0.08, 0.08, 0.08]).vertices)
    arm_root = np.array([1.0, 0.0, 0.3])
    first = select_container_landing(floor, child, arm_root)
    assert first is not None
    radius = float(np.linalg.norm(child[:, :2], axis=1).max())

    with pytest.raises(ValueError, match="remaining landing space"):
        select_container_landing(
            floor, child, arm_root, occupied=[(first[0], first[1], radius)]
        )


def test_landing_retries_only_failed_layout_at_five_millimetres() -> None:
    from embodichain.gen_sim.task_engine._task_program.container_targets import (
        _floor_region,
    )

    floor = trimesh.creation.box(extents=[0.22, 0.22, 0.01])
    child = np.asarray(trimesh.creation.box(extents=[0.08] * 3).vertices)
    arm_root = np.array([1.0, 0.0, 0.3])
    first = select_container_landing(floor, child, arm_root)
    assert first == pytest.approx((0.05, 0.0, 0.046))
    radius = float(np.linalg.norm(child[:, :2], axis=1).max())
    second = select_container_landing(
        floor, child, arm_root, occupied=[(first[0], first[1], radius)]
    )
    assert second is not None
    assert (
        np.linalg.norm(np.asarray(second[:2]) - np.asarray(first[:2]))
        >= 2 * radius + 0.005 - 1e-9
    )
    footprint = translate(
        MultiPoint(child[:, :2]).convex_hull, xoff=second[0], yoff=second[1]
    )
    measured, _ = _floor_region(floor)
    assert measured.buffer(-0.005).covers(footprint)


def test_landing_five_millimetre_floor_fallback_keeps_full_footprint() -> None:
    floor = trimesh.creation.box(extents=[0.10, 0.10, 0.01])
    child = np.asarray(trimesh.creation.box(extents=[0.08] * 3).vertices)
    target = select_container_landing(floor, child, np.array([1.0, 0.0, 0.3]))
    assert target is not None
    assert np.max(child[:, 0] + target[0]) <= 0.05 - 0.005 + 1e-9
    assert target[2] == pytest.approx(0.046)


def test_five_millimetre_fallback_checks_actual_concave_boundary(monkeypatch) -> None:
    from embodichain.gen_sim.task_engine._task_program import container_targets

    floor = box(-0.015, -0.015, 0.015, 0.015).difference(
        box(-0.002, -0.002, 0.002, 0.002)
    )
    monkeypatch.setattr(container_targets, "_floor_region", lambda _: (floor, 0.005))
    child = np.asarray(trimesh.creation.box(extents=[0.0002] * 3).vertices)
    target = select_container_landing(
        None, child, np.array([0.00289204362, 0.00379004418, 0.3])
    )
    assert target is not None
    footprint = translate(
        MultiPoint(child[:, :2]).convex_hull, xoff=target[0], yoff=target[1]
    )
    assert floor.covers(footprint)
    assert floor.boundary.distance(footprint) >= 0.005 - 1e-9


def test_landing_requires_a_dominant_near_horizontal_surface() -> None:
    curved = trimesh.creation.icosphere(subdivisions=2, radius=0.2)
    child = np.asarray(trimesh.creation.box(extents=[0.02, 0.02, 0.02]).vertices)
    assert select_container_landing(curved, child, np.array([1.0, 0.0, 0.3])) is None


def test_bundle_landing_is_invariant_under_world_yaw(tmp_path: Path) -> None:
    from embodichain.gen_sim.task_engine.task_program_bundle import (
        _container_target_pose,
    )

    sources = []
    for name, extents in (("tray", [0.5, 0.3, 0.01]), ("cube", [0.08] * 3)):
        mesh = trimesh.creation.box(extents=extents)
        mesh.vertices = np.asarray(mesh.vertices)[:, [0, 2, 1]] * [1, 1, -1]
        path = tmp_path / f"{name}.glb"
        mesh.export(path)
        sources.append(
            {
                "runtime_uid": name,
                "shape": {"fpath": str(path)},
                "init_pos": [0.0, 0.0, 0.0],
                "init_rot": [0.0, 0.0, 0.0],
            }
        )
    mount = np.eye(4)
    mount[0, 3] = 1.0
    simulation = {
        "init_pos": [0.0, 0.0, 0.0],
        "init_rot": [0.0, 0.0, 0.0],
        "urdf_cfg": {
            "components": [{"component_type": "left_arm", "transform": mount.tolist()}]
        },
    }
    baseline = _container_target_pose(
        *sources, resource="left", embodiment={"simulation": simulation}
    )
    assert baseline is not None
    reserved = _container_target_pose(
        *sources,
        resource="left",
        embodiment={"simulation": simulation},
        occupied=[(baseline, sources[1])],
    )
    assert reserved is not None
    assert (
        np.linalg.norm(np.array(reserved)[[3, 7]] - np.array(baseline)[[3, 7]])
        >= 2 * np.sqrt(2) * 0.04 + 0.019
    )
    for source in [*sources, simulation]:
        source["init_rot"] = [0.0, 0.0, 90.0]
    rotated = _container_target_pose(
        *sources, resource="left", embodiment={"simulation": simulation}
    )

    assert baseline[3] > 0.1
    assert rotated == pytest.approx(baseline)

    matrix = np.eye(4)
    matrix[:3, :3] = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    for index, source in enumerate([*sources, simulation]):
        source["init_local_pose"] = matrix.tolist()
        source["init_pos"] = [100.0 * index, 50.0 * index, 0.0]
        source["init_rot"] = [0.0, 0.0, 0.0]
    explicit = _container_target_pose(
        *sources, resource="left", embodiment={"simulation": simulation}
    )
    assert explicit == pytest.approx(baseline)
