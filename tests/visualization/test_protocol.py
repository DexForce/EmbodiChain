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

import numpy as np
import pytest

from embodichain.lab.visualization.protocol import (
    CameraImage,
    CameraSpec,
    DynamicMeshUpdate,
    GizmoCommand,
    GizmoSpec,
    GizmoState,
    MeshGeometry,
    pose_to_position_wxyz,
)


def test_pose_conversion_preserves_embodichain_wxyz_order() -> None:
    pose = np.array([1.0, 2.0, 3.0, 2.0, 0.0, 0.0, 0.0], dtype=np.float32)

    position, wxyz = pose_to_position_wxyz(pose)

    np.testing.assert_allclose(position, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(wxyz, [1.0, 0.0, 0.0, 0.0])


def test_pose_conversion_accepts_batch_of_four_pose_vectors() -> None:
    poses = np.tile(
        np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        (4, 1),
    )

    positions, wxyz = pose_to_position_wxyz(poses)

    assert positions.shape == (4, 3)
    assert wxyz.shape == (4, 4)


def test_pose_conversion_handles_homogeneous_rotation_matrix() -> None:
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    pose[:3, 3] = [0.5, -0.25, 0.75]
    half_sqrt_two = np.sqrt(0.5)

    position, wxyz = pose_to_position_wxyz(pose)

    np.testing.assert_allclose(position, [0.5, -0.25, 0.75])
    np.testing.assert_allclose(
        wxyz, [half_sqrt_two, 0.0, 0.0, half_sqrt_two], atol=1.0e-6
    )


def test_pose_conversion_rejects_unsupported_shape() -> None:
    with pytest.raises(ValueError, match="Expected pose shape"):
        pose_to_position_wxyz(np.zeros((3, 3), dtype=np.float32))


def test_mesh_geometry_owns_cpu_array_copy() -> None:
    vertices = np.zeros((3, 3), dtype=np.float64)
    faces = np.array([[0, 1, 2]], dtype=np.int64)

    geometry = MeshGeometry("sha256:test", vertices, faces)
    vertices[0, 0] = 42.0

    assert geometry.vertices.dtype == np.float32
    assert geometry.faces.dtype == np.uint32
    assert geometry.vertices[0, 0] == 0.0


def test_camera_spec_accepts_valid_pinhole_parameters() -> None:
    spec = CameraSpec(
        camera_id="env:0/camera:wrist",
        sensor_uid="wrist",
        env_id=0,
        path="/envs/0/cameras/wrist",
        fov_y=0.8,
        aspect=4.0 / 3.0,
        near=0.01,
        far=10.0,
    )

    assert spec.aspect == 4.0 / 3.0


def test_camera_image_owns_rgb_copy() -> None:
    rgb = np.zeros((2, 3, 3), dtype=np.uint8)

    image = CameraImage("env:0/camera:wrist", rgb)
    rgb[0, 0] = 255

    np.testing.assert_array_equal(image.image[0, 0], [0, 0, 0])


def test_dynamic_mesh_update_owns_vertex_copy() -> None:
    vertices = np.zeros((4, 3), dtype=np.float64)

    update = DynamicMeshUpdate("env:0/soft:cow", vertices)
    vertices[0, 0] = 42.0

    assert update.vertices.dtype == np.float32
    assert update.vertices[0, 0] == 0.0


def test_gizmo_protocol_owns_and_normalizes_pose_arrays() -> None:
    position = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    quaternion = np.array([2.0, 0.0, 0.0, 0.0], dtype=np.float64)

    state = GizmoState("cube", position, quaternion)
    command = GizmoCommand(
        run_id="run",
        scene_revision=1,
        sequence=1,
        gizmo_id="cube",
        phase="update",
        client_id="client-a",
        position=position,
        wxyz=quaternion,
    )
    position[0] = 99.0

    np.testing.assert_allclose(state.position, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(command.wxyz, [1.0, 0.0, 0.0, 0.0])


def test_gizmo_spec_rejects_unsupported_target_type() -> None:
    with pytest.raises(ValueError, match="target_type"):
        GizmoSpec(
            gizmo_id="bad",
            target_uid="bad",
            target_type="cloth",
            control_part=None,
            env_id=0,
            path="/interactions/gizmos/bad",
        )
