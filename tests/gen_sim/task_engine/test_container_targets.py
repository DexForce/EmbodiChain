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

from embodichain.gen_sim.task_engine._task_program.container_targets import (
    select_container_landing,
)


@pytest.mark.parametrize("side", [-1.0, 1.0])
def test_landing_uses_arm_side_and_erodes_by_full_footprint(side: float) -> None:
    floor = trimesh.creation.box(extents=[0.5, 0.3, 0.01])
    child = np.asarray(trimesh.creation.box(extents=[0.08, 0.08, 0.08]).vertices)
    target = select_container_landing(floor, child, np.array([side, 0.0, 0.3]))

    assert target is not None
    radius = np.linalg.norm(child[:, :2], axis=1).max()
    assert target[0] == pytest.approx(side * (0.25 - radius - 0.02))
    assert target[1] == pytest.approx(0.0)
    assert target[2] == pytest.approx(0.005 + 0.04 + 0.001)


def test_landing_preserves_holes_in_support_mesh() -> None:
    floor = trimesh.creation.annulus(r_min=0.06, r_max=0.2, height=0.01)
    child = np.asarray(trimesh.creation.box(extents=[0.02, 0.02, 0.02]).vertices)
    target = select_container_landing(floor, child, np.array([0.0, 0.0, 0.3]))

    assert target is not None
    # The origin is inside the hole, not a valid support region.
    assert np.linalg.norm(target[:2]) > 0.09


def test_landing_does_not_invent_room_for_an_oversized_object() -> None:
    floor = trimesh.creation.box(extents=[0.1, 0.1, 0.01])
    child = np.asarray(trimesh.creation.box(extents=[0.2, 0.2, 0.2]).vertices)
    with pytest.raises(ValueError, match="footprint"):
        select_container_landing(floor, child, np.array([1.0, 0.0, 0.3]))


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
    for source in [*sources, simulation]:
        source["init_rot"] = [0.0, 0.0, 90.0]
    rotated = _container_target_pose(
        *sources, resource="left", embodiment={"simulation": simulation}
    )

    assert baseline is not None
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
