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

"""Generic rigid-object assembly directions and initial-placement checks."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import trimesh
from scipy.spatial.transform import Rotation

from scripts.tools.grasp_assemble._geometry import GraspGeometry, object_waypoints


def _pose(
    translation: tuple[float, float, float],
    rotation: tuple[float, float, float] = (0, 0, 0),
) -> np.ndarray:
    result = np.eye(4)
    result[:3, 3] = translation
    result[:3, :3] = Rotation.from_euler("xyz", rotation, degrees=True).as_matrix()
    return result


def _geometry() -> GraspGeometry:
    geometry = GraspGeometry.__new__(GraspGeometry)
    geometry.assemble_mesh = trimesh.creation.box([0.04, 0.06, 0.08])
    geometry.base_mesh = trimesh.creation.box([0.1, 0.1, 0.3])
    geometry.base_mesh.apply_translation([0, 0, 0.15])
    geometry.assembly = {
        "T_base_assemble": _pose((0.15, 0, 0.2), (0, 90, 0)).tolist(),
        "resolved_validation": {"assemble_axis": [1, 0, 0]},
    }
    geometry.parts = []
    geometry.set_layout(
        {
            "T_world_base": _pose((0.4, 0, 0), (0, 0, 90)).tolist(),
            "T_world_assemble_initial": _pose((-0.3, 0, 0.08), (30, 20, 0)).tolist(),
            "motion": {
                "clearance": 0.08,
                "retract_distance": 0.1,
                "pre_grasp_distance": 0.05,
            },
            "task_plan": {
                "insertion_direction_base": [1, 0, 0],
                "insertion_distance": 0.12,
                "retract_direction_base": [0, -1, 0],
                "retract_distance": 0.08,
            },
        }
    )
    return geometry


def test_sideways_insertion_follows_base_frame_direction() -> None:
    geometry = _geometry()
    lift, rotated, transit, preinsert, target = geometry.waypoints
    # Base +X becomes world +Y after the base yaw.
    np.testing.assert_allclose(
        target[:3, 3] - preinsert[:3, 3], [0, 0.12, 0], atol=1e-12
    )
    np.testing.assert_allclose(transit[:2, 3], preinsert[:2, 3])
    assert transit[2, 3] > preinsert[2, 3]
    np.testing.assert_allclose(lift[:2, 3], geometry.initial[:2, 3])
    np.testing.assert_allclose(rotated[:3, 3], lift[:3, 3])
    np.testing.assert_allclose(rotated[:3, :3], geometry.target[:3, :3])
    np.testing.assert_array_equal(target, geometry.target)
    assert geometry.waypoint_names == (
        "grasp",
        "lift",
        "rotate",
        "transit",
        "hover",
        "place",
    )


def test_sideways_withdrawal_follows_base_frame_direction() -> None:
    geometry = _geometry()
    retract = geometry._retract_pose(geometry.target)
    # Base -Y becomes world +X; withdrawal need not oppose insertion.
    np.testing.assert_allclose(
        retract[:3, 3] - geometry.target[:3, 3], [0.08, 0, 0], atol=1e-12
    )
    np.testing.assert_array_equal(retract[:3, :3], geometry.target[:3, :3])


def test_initial_orientation_is_independent_of_semantic_assemble_axis() -> None:
    geometry = _geometry()
    axis = np.asarray(geometry.assembly["resolved_validation"]["assemble_axis"])
    assert (geometry.initial[:3, :3] @ axis)[2] < 0.99
    assert geometry._object_clearance(geometry.initial) is None


def test_no_rotation_required_for_equal_source_and_target_orientation() -> None:
    initial, target = _pose((-0.3, 0, 0.1)), _pose((0.2, 0, 0.3))
    base = trimesh.creation.box([0.1, 0.1, 0.3])
    assemble = trimesh.creation.box([0.04, 0.06, 0.08])
    route = object_waypoints(
        initial, target, base, assemble, 0.08, np.array([0, 0, -1]), 0.1
    )
    np.testing.assert_allclose(route[0], route[1])
    for item in route:
        np.testing.assert_array_equal(item[:3, :3], initial[:3, :3])


def test_legacy_vertical_route_retains_four_original_keyframes() -> None:
    initial, target = _pose((-0.3, 0, 0.1)), _pose((0.2, 0, 0.43), (180, 0, 0))
    base = trimesh.creation.box([0.1, 0.1, 0.4])
    base.apply_translation([0, 0, 0.2])
    assemble = trimesh.creation.box([0.04, 0.06, 0.08])
    route = object_waypoints(initial, target, base, assemble, 0.12)
    expected_height = (
        max(base.bounds[1, 2], initial[2, 3])
        + np.linalg.norm(assemble.vertices, axis=1).max()
        + 0.12
    )
    assert len(route) == 4
    assert route[0][2, 3] == pytest.approx(expected_height)
    np.testing.assert_array_equal(route[-1], target)
    np.testing.assert_allclose(route[-2][:2, 3], target[:2, 3])


@pytest.mark.parametrize(
    "position,reason", [((0.4, 0, 0.15), "base"), ((-0.3, 0, 0.01), "ground")]
)
def test_invalid_initial_placement_rejected_before_grasp_search(
    position: tuple[float, float, float], reason: str
) -> None:
    geometry = _geometry()
    geometry.initial = _pose(position)
    report = geometry.evaluate(np.eye(4))
    assert not report["accepted"]
    assert "Initial placement" in report["reasons"][0]
    assert reason in report["reasons"][0]


def test_sideways_candidate_reports_the_exact_validated_retract_pose() -> None:
    geometry = _geometry()
    geometry._assemble_collision = lambda index, pose, q: index in (1, 2) and q >= 0.02
    geometry.assemble_checks = [SimpleNamespace(distance=lambda pose: 0.0)] * 3
    geometry._world_clearance = lambda pose, q: None
    # Isolate result construction from physical surfaces; other tests cover overlap.
    geometry._object_clearance = lambda pose: None
    geometry._initial_support = lambda: {"accepted": True}
    result = geometry.evaluate(np.eye(4))
    assert result["accepted"], result["reasons"]
    np.testing.assert_allclose(
        result["T_world_tcp_retract"], geometry._retract_pose(geometry.target)
    )
    assert set(result["waypoints"]) == set(geometry.waypoint_names)


def test_carried_ground_intersection_rejected_by_generic_geometry() -> None:
    geometry = _geometry()
    pose = geometry.target.copy()
    pose[2, 3] = 0
    assert "ground" in geometry._object_clearance(pose)


def test_high_preinsertion_pose_does_not_add_redundant_vertical_clearance() -> None:
    initial, target = _pose((-0.3, 0, 0.1)), _pose((0.2, 0, 0.6))
    base = trimesh.creation.box([0.1, 0.1, 0.3])
    assemble = trimesh.creation.box([0.04, 0.06, 0.08])
    route = object_waypoints(
        initial, target, base, assemble, 0.08, np.array([0, 0, -1]), 0.1
    )
    expected_preinsertion_height = 0.7
    assert route[0][2, 3] == pytest.approx(expected_preinsertion_height)
    assert route[2][2, 3] == pytest.approx(expected_preinsertion_height)
    radius = np.linalg.norm(assemble.vertices, axis=1).max()
    assert route[0][2, 3] - radius >= base.bounds[1, 2] + 0.08


def _support_geometry(
    mesh: trimesh.Trimesh, rpy: tuple[float, float, float]
) -> GraspGeometry:
    geometry = GraspGeometry.__new__(GraspGeometry)
    geometry.assemble_mesh = mesh
    geometry.initial = _pose((0, 0, 0), rpy)
    vertices = mesh.vertices @ geometry.initial[:3, :3].T
    geometry.initial[2, 3] = 0.001 - vertices[:, 2].min()
    return geometry


@pytest.mark.parametrize("rpy,accepted", [((0, 0, 0), False), ((90, 0, 35), True)])
def test_phone_initial_support_prefers_broad_face(
    rpy: tuple[float, float, float], accepted: bool
) -> None:
    phone = trimesh.creation.box([0.072, 0.008, 0.15])
    geometry = _support_geometry(phone, rpy)
    support = geometry._initial_support()
    assert support["accepted"] == accepted
    assert support["minimum_world_z_m"] == pytest.approx(0.001)
    assert support["minimum_edge_margin_m"] > 0
    if accepted:
        assert support["minimum_tipping_angle_degrees"] > 80
    else:
        expected_tipping_angle = np.rad2deg(np.arctan2(0.004, 0.075))
        assert support["minimum_tipping_angle_degrees"] == pytest.approx(
            expected_tipping_angle
        )
        assert "broad stable face" in support["reason"]


def test_narrow_phone_support_rejected_before_contact_search_with_metrics() -> None:
    geometry = _geometry()
    geometry.assemble_mesh = trimesh.creation.box([0.072, 0.008, 0.15])
    geometry.initial = _pose((-0.3, 0, 0.076))
    report = geometry.evaluate(np.eye(4))
    assert not report["accepted"]
    assert "tipping reserve" in report["reasons"][0]
    assert not report["initial_support"]["accepted"]
    assert report["initial_support"]["minimum_edge_margin_m"] == pytest.approx(0.004)


@pytest.mark.parametrize("rpy", [(45, 0, 0), (35, 25, 10)])
def test_point_or_line_initial_support_is_rejected(
    rpy: tuple[float, float, float],
) -> None:
    geometry = _support_geometry(trimesh.creation.box([0.04, 0.06, 0.08]), rpy)
    support = geometry._initial_support()
    assert not support["accepted"]
    assert support["contact_points"] < 3
    assert "point or line" in support["reason"]


def test_projected_com_outside_support_polygon_is_rejected() -> None:
    pedestal = trimesh.creation.box([0.02, 0.02, 0.04])
    pedestal.apply_translation([0, 0, 0.02])
    overhang = trimesh.creation.box([0.1, 0.1, 0.02])
    overhang.apply_translation([0.04, 0, 0.05])
    mesh = trimesh.util.concatenate([pedestal, overhang])
    geometry = _support_geometry(mesh, (0, 0, 0))
    support = geometry._initial_support()
    assert not support["accepted"]
    assert support["minimum_edge_margin_m"] < 0
    assert support["minimum_tipping_angle_degrees"] < 0


def test_com_on_support_edge_has_no_tipping_reserve() -> None:
    geometry = _support_geometry(trimesh.creation.box([0.04, 0.06, 0.08]), (0, 0, 0))
    # An unknown asymmetric density can move COM onto the support boundary;
    # isolate the half-space margin calculation from uniform-density modeling.
    geometry.assemble_mesh = SimpleNamespace(
        vertices=geometry.assemble_mesh.vertices.copy(),
        center_mass=np.array([0.02, 0, 0]),
    )
    support = geometry._initial_support()
    assert not support["accepted"]
    assert support["minimum_edge_margin_m"] == pytest.approx(0)
    assert support["minimum_tipping_angle_degrees"] == pytest.approx(0)


def test_cylindrical_upright_vessel_has_stable_base_support() -> None:
    geometry = _support_geometry(
        trimesh.creation.cylinder(radius=0.043, height=0.105, sections=64), (0, 0, 0)
    )
    support = geometry._initial_support()
    assert support["accepted"]
    assert support["minimum_tipping_angle_degrees"] > 35
