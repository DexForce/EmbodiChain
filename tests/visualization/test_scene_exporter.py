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

from types import SimpleNamespace

import numpy as np

from embodichain.lab.visualization import (
    FrameOverlay,
    JointControlSpec,
    JointControlState,
    PointCloudOverlay,
    SceneExporter,
    SceneOverlays,
    VisualizationCfg,
)
from embodichain.lab.visualization.scene_exporter import mesh_geometry_id


class _RigidObject:
    def __init__(self) -> None:
        self._vertices = np.array(
            [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]],
            dtype=np.float32,
        )
        self._faces = np.array([[0, 1, 2]], dtype=np.int32)
        self._poses = np.array(
            [
                [0.1, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0],
                [0.2, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )

    def get_vertices(self, env_ids: list[int], scale: bool) -> np.ndarray:
        assert scale
        return np.stack([self._vertices for _ in env_ids])

    def get_triangles(self, env_ids: list[int]) -> np.ndarray:
        return np.stack([self._faces for _ in env_ids])

    def get_local_pose(self, to_matrix: bool = False) -> np.ndarray:
        assert not to_matrix
        return self._poses


def test_mesh_geometry_id_includes_color() -> None:
    """Meshes with identical topology but different colors remain distinct."""
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2]], dtype=np.uint32)

    red_id = mesh_geometry_id(vertices, faces, (255, 0, 0))
    blue_id = mesh_geometry_id(vertices, faces, (0, 0, 255))

    assert red_id != blue_id


class _Articulation:
    link_names = ["base", "tip/link"]

    def __init__(self) -> None:
        self._meshes = {
            "base": (
                np.array(
                    [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.2, 0.0]],
                    dtype=np.float32,
                ),
                np.array([[0, 1, 2]], dtype=np.int32),
            ),
            "tip/link": (
                np.array(
                    [[0.0, 0.0, 0.0], [0.05, 0.0, 0.0], [0.0, 0.05, 0.0]],
                    dtype=np.float32,
                ),
                np.array([[0, 1, 2]], dtype=np.int32),
            ),
        }
        self.body_data = SimpleNamespace(
            body_link_pose=np.array(
                [
                    [
                        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                        [0.4, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0],
                    ],
                    [
                        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                        [0.5, 0.0, 0.6, 1.0, 0.0, 0.0, 0.0],
                    ],
                ],
                dtype=np.float32,
            )
        )

    def get_link_vert_face(self, link_name: str) -> tuple[np.ndarray, np.ndarray]:
        return self._meshes[link_name]


class _RigidObjectGroup:
    def __init__(self) -> None:
        self.cfg = SimpleNamespace(rigid_objects={"left": object(), "right": object()})
        self._vertices = (
            np.array(
                [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]],
                dtype=np.float32,
            ),
            np.array(
                [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.2, 0.0]],
                dtype=np.float32,
            ),
        )
        self._faces = np.array([[0, 1, 2]], dtype=np.int32)
        self._poses = np.array(
            [
                [
                    [0.1, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0],
                    [0.2, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0],
                ],
                [
                    [0.6, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0],
                    [0.7, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0],
                ],
            ],
            dtype=np.float32,
        )

    def get_object_vertices(
        self,
        object_id: int,
        env_ids: list[int],
        scale: bool,
    ) -> np.ndarray:
        assert scale
        return np.stack([self._vertices[object_id] for _ in env_ids])

    def get_object_triangles(
        self,
        object_id: int,
        env_ids: list[int],
    ) -> np.ndarray:
        del object_id
        return np.stack([self._faces for _ in env_ids])

    def get_local_pose(self, to_matrix: bool = False) -> np.ndarray:
        assert not to_matrix
        return self._poses


class _DeformableObject:
    def __init__(self, deformable_type: str) -> None:
        self.deformable_type = deformable_type
        local_vertices = np.array(
            [[0.0, 0.0, 0.0], [0.15, 0.0, 0.0], [0.0, 0.15, 0.0]],
            dtype=np.float32,
        )
        self.vertices = np.stack(
            (
                local_vertices,
                local_vertices + np.array([2.0, 0.0, 0.0], dtype=np.float32),
            )
        )
        self._faces = np.array([[0, 1, 2]], dtype=np.int32)

    def get_surface_vertices(self) -> np.ndarray:
        return self.vertices

    def get_surface_triangles(self, env_ids: list[int]) -> np.ndarray:
        return np.stack([self._faces for _ in env_ids])


class _Simulation:
    num_envs = 2
    arena_offsets = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32)

    def __init__(self) -> None:
        self.rigid = _RigidObject()
        self.robot = _Articulation()

    def get_rigid_object_uid_list(self) -> list[str]:
        return ["cube/name"]

    def get_rigid_object(self, uid: str) -> _RigidObject:
        assert uid == "cube/name"
        return self.rigid

    def get_robot_uid_list(self) -> list[str]:
        return ["robot"]

    def get_robot(self, uid: str) -> _Articulation:
        assert uid == "robot"
        return self.robot

    def get_articulation_uid_list(self) -> list[str]:
        return []

    def get_articulation(self, uid: str) -> None:
        raise AssertionError(f"Unexpected articulation lookup: {uid}")

    def get_rigid_object_group_uid_list(self) -> list[str]:
        return []

    def get_rigid_object_group(self, uid: str) -> None:
        raise AssertionError(f"Unexpected rigid-object-group lookup: {uid}")

    def get_deformable_object_uid_list(self) -> list[str]:
        return []

    def get_deformable_object(self, uid: str) -> None:
        raise AssertionError(f"Unexpected deformable-object lookup: {uid}")

    def get_sensor_uid_list(self) -> list[str]:
        return []

    def get_sensor(self, uid: str) -> None:
        raise AssertionError(f"Unexpected sensor lookup: {uid}")


class _EmptySimulation:
    num_envs = 1
    arena_offsets = np.zeros((1, 3), dtype=np.float32)

    def get_rigid_object_uid_list(self) -> list[str]:
        return []

    def get_rigid_object_group_uid_list(self) -> list[str]:
        return []

    def get_robot_uid_list(self) -> list[str]:
        return []

    def get_robot(self, uid: str) -> None:
        raise AssertionError(f"Unexpected robot lookup: {uid}")

    def get_articulation_uid_list(self) -> list[str]:
        return []

    def get_articulation(self, uid: str) -> None:
        raise AssertionError(f"Unexpected articulation lookup: {uid}")

    def get_deformable_object_uid_list(self) -> list[str]:
        return []

    def get_deformable_object(self, uid: str) -> None:
        raise AssertionError(f"Unexpected deformable-object lookup: {uid}")

    def get_sensor_uid_list(self) -> list[str]:
        return []

    def get_sensor(self, uid: str) -> None:
        raise AssertionError(f"Unexpected sensor lookup: {uid}")


class _JointControlProvider:
    def __init__(self) -> None:
        self.specs = tuple(
            JointControlSpec(
                control_id=f"joint-{env_id}",
                articulation_uid="door",
                env_id=env_id,
                joint_id=0,
                joint_name="hinge",
                joint_type="revolute",
                lower=-1.0,
                upper=1.0,
                step=0.01,
                initial_value=0.0,
            )
            for env_id in range(2)
        )

    def joint_control_specs(self) -> tuple[JointControlSpec, ...]:
        return self.specs

    def joint_control_states(self) -> tuple[JointControlState, ...]:
        return tuple(
            JointControlState(
                control_id=spec.control_id,
                value=0.25 + spec.env_id,
                applied_sequence=spec.env_id + 1,
            )
            for spec in self.specs
        )


class _Gizmo:
    target = SimpleNamespace(cfg=SimpleNamespace(uid="cube"))
    target_type = "rigid_object"
    control_part = None
    cfg = SimpleNamespace(
        axis_length_x=0.2,
        axis_length_y=0.2,
        axis_length_z=0.2,
        axis_size=0.01,
        rings_radius=0.15,
        rings_size=0.01,
    )

    def get_control_pose(self) -> np.ndarray:
        pose = np.eye(4, dtype=np.float32)[None]
        pose[0, :3, 3] = [0.2, 0.3, 0.4]
        return pose

    def is_visible(self) -> bool:
        return True


class _GizmoSimulation(_EmptySimulation):
    def get_gizmo_items(self) -> tuple[tuple[str, _Gizmo], ...]:
        return (("cube", _Gizmo()),)


class _AxisMarkerHandle:
    def __init__(self, pose: np.ndarray, visible: bool = False) -> None:
        self._pose = pose
        self._visible = visible

    def get_world_pose(self) -> np.ndarray:
        return self._pose

    def is_visible(self) -> bool:
        return self._visible


class _AxisMarkerSimulation(_EmptySimulation):
    def __init__(self) -> None:
        pose = np.eye(4, dtype=np.float32)
        pose[:3, 3] = [-0.4, 0.48, 0.1]
        self.marker = _AxisMarkerHandle(pose)

    def get_axis_marker_items(
        self,
    ) -> tuple[tuple[str, tuple[_AxisMarkerHandle, ...], float, float], ...]:
        return (("place_target_axis", (self.marker,), 0.2, 0.01),)


class _Camera:
    def __init__(self, visualization_role: str = "sensor") -> None:
        self.cfg = SimpleNamespace(
            sensor_type="Camera",
            width=4,
            height=2,
            near=0.01,
            far=10.0,
            enable_color=True,
            visualization_role=visualization_role,
        )
        self.update_count = 0
        self._poses = np.tile(np.eye(4, dtype=np.float32), (2, 1, 1))
        self._poses[0, :3, 3] = [0.1, 0.2, 0.3]
        self._poses[1, :3, 3] = [0.4, 0.5, 0.6]
        self._color = np.zeros((2, 2, 4, 4), dtype=np.uint8)
        self._color[0, ..., :3] = [10, 20, 30]
        self._color[1, ..., :3] = [40, 50, 60]
        self._color[..., 3] = 255

    def get_intrinsics(self) -> np.ndarray:
        intrinsic = np.array(
            [[4.0, 0.0, 2.0], [0.0, 4.0, 1.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        return np.stack((intrinsic, intrinsic))

    def get_arena_pose(self, to_matrix: bool) -> np.ndarray:
        assert to_matrix
        return self._poses

    def update(self) -> None:
        self.update_count += 1

    def get_data(self) -> dict[str, np.ndarray]:
        return {"color": self._color}


class _CameraSimulation(_Simulation):
    def __init__(self) -> None:
        super().__init__()
        self.camera = _Camera()

    def get_sensor_uid_list(self) -> list[str]:
        return ["wrist/camera"]

    def get_sensor(self, uid: str) -> _Camera:
        assert uid == "wrist/camera"
        return self.camera


class _StereoCamera(_Camera):
    def __init__(self) -> None:
        super().__init__()
        self.cfg.sensor_type = "StereoCamera"

    def get_intrinsics(self) -> tuple[np.ndarray, np.ndarray]:
        intrinsics = super().get_intrinsics()
        return intrinsics, intrinsics.copy()

    def get_left_right_arena_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return self._poses, self._poses.copy()

    def get_arena_pose(self, to_matrix: bool) -> np.ndarray:
        raise AssertionError("Stereo cameras should use their primary eye pose.")


class _SensorPreviewSimulation(_Simulation):
    def __init__(self) -> None:
        super().__init__()
        self.cameras = {
            "cam_high": _StereoCamera(),
            "record_camera": _Camera(visualization_role="record"),
        }

    def get_sensor_uid_list(self) -> list[str]:
        return list(self.cameras)

    def get_sensor(self, uid: str) -> _Camera:
        return self.cameras[uid]


class _CompleteSimulation(_Simulation):
    def __init__(self) -> None:
        super().__init__()
        self.rigid_group = _RigidObjectGroup()
        self.soft = _DeformableObject("volume")
        self.cloth = _DeformableObject("surface")

    def get_rigid_object_group_uid_list(self) -> list[str]:
        return ["pair"]

    def get_rigid_object_group(self, uid: str) -> _RigidObjectGroup:
        assert uid == "pair"
        return self.rigid_group

    def get_deformable_object_uid_list(self) -> list[str]:
        return ["jelly", "flag"]

    def get_deformable_object(self, uid: str) -> _DeformableObject:
        return {"jelly": self.soft, "flag": self.cloth}[uid]


def test_manifest_deduplicates_geometry_and_escapes_paths() -> None:
    exporter = SceneExporter(
        _Simulation(),
        VisualizationCfg(backend="viser", env_ids=[0, 1]),
        run_id="test-run",
    )

    manifest = exporter.build_manifest()

    assert len(manifest.nodes) == 6
    assert len(manifest.geometries) == 3
    rigid_paths = [node.path for node in manifest.nodes if node.kind == "rigid_object"]
    assert rigid_paths == [
        "/envs/0/rigid_objects/cube%2Fname",
        "/envs/1/rigid_objects/cube%2Fname",
    ]
    assert manifest.nodes[0].geometry_id == manifest.nodes[1].geometry_id


def test_manifest_all_environment_selector_exports_every_arena() -> None:
    exporter = SceneExporter(
        _Simulation(),
        VisualizationCfg(backend="viser", env_ids=None),
        run_id="all-envs",
    )

    manifest = exporter.build_manifest()
    result = exporter.capture(sim_step=1, sim_time=0.01)

    assert {node.env_id for node in manifest.nodes} == {0, 1}
    assert len(manifest.nodes) == 6
    np.testing.assert_allclose(
        result.frame.positions[
            result.frame.node_ids.index("env:1/robot:robot/link:tip%2Flink")
        ],
        [2.5, 0.0, 0.6],
    )


def test_capture_adds_arena_offsets_and_limits_point_clouds() -> None:
    max_points = 4
    exporter = SceneExporter(
        _Simulation(),
        VisualizationCfg(
            backend="viser",
            env_ids=[0, 1],
            point_cloud_max_points=max_points,
        ),
        run_id="test-run",
    )
    exporter.build_manifest()
    points = np.arange(30, dtype=np.float32).reshape(10, 3)
    overlays = SceneOverlays(
        point_clouds=(PointCloudOverlay("cloud", points, colors=(255, 0, 0)),)
    )

    result = exporter.capture(sim_step=12, sim_time=0.12, overlays=overlays)

    env_one_rigid_index = result.frame.node_ids.index("env:1/rigid:cube%2Fname")
    np.testing.assert_allclose(
        result.frame.positions[env_one_rigid_index], [2.2, 0.0, 0.3]
    )
    assert result.frame.overlays.point_clouds[0].points.shape == (max_points, 3)
    assert result.capture_seconds >= 0.0


def test_empty_scene_can_publish_a_current_frame() -> None:
    exporter = SceneExporter(
        _EmptySimulation(),
        VisualizationCfg(backend="viser"),
        run_id="empty-run",
    )

    manifest = exporter.build_manifest()
    result = exporter.capture(sim_step=0, sim_time=0.0)

    assert manifest.scene_revision == 1
    assert manifest.nodes == ()
    assert result.frame.scene_revision == 1
    assert result.frame.positions.shape == (0, 3)


def test_joint_control_provider_is_filtered_and_captured() -> None:
    exporter = SceneExporter(
        _EmptySimulation(),
        VisualizationCfg(backend="viser", env_ids=[0]),
        run_id="joint-run",
    )
    exporter.set_joint_control_provider(_JointControlProvider())

    manifest = exporter.build_manifest()
    result = exporter.capture(sim_step=1, sim_time=0.01)

    assert [control.control_id for control in manifest.joint_controls] == ["joint-0"]
    assert result.frame.joint_controls == (
        JointControlState("joint-0", value=0.25, applied_sequence=1),
    )


def test_gizmo_manifest_and_authoritative_pose_are_exported() -> None:
    exporter = SceneExporter(
        _GizmoSimulation(),
        VisualizationCfg(backend="viser", allow_commands=True),
        run_id="gizmo-run",
    )

    manifest = exporter.build_manifest()
    result = exporter.capture(sim_step=1, sim_time=0.01)

    assert manifest.gizmos[0].gizmo_id == "cube"
    assert manifest.gizmos[0].path == "/interactions/gizmos/cube"
    np.testing.assert_allclose(result.frame.gizmos[0].position, [0.2, 0.3, 0.4])


def test_axis_markers_ignore_headless_native_visibility() -> None:
    exporter = SceneExporter(
        _AxisMarkerSimulation(),
        VisualizationCfg(backend="viser"),
        run_id="axis-marker-run",
    )

    exporter.build_manifest()
    result = exporter.capture(sim_step=1, sim_time=0.01)

    assert len(result.frame.overlays.frames) == 1
    overlay = result.frame.overlays.frames[0]
    assert overlay.overlay_id == "marker:place_target_axis:0"
    np.testing.assert_allclose(overlay.position, [-0.4, 0.48, 0.1])
    np.testing.assert_allclose(overlay.wxyz, [1.0, 0.0, 0.0, 0.0])
    assert overlay.axes_length == 0.2
    assert overlay.axes_radius == 0.01
    assert overlay.visible


def test_axis_marker_id_does_not_collide_with_caller_frame() -> None:
    exporter = SceneExporter(
        _AxisMarkerSimulation(),
        VisualizationCfg(backend="viser"),
        run_id="axis-marker-collision-run",
    )
    caller_frame = FrameOverlay(
        overlay_id="marker:place_target_axis:0",
        position=np.array([0.1, 0.2, 0.3], dtype=np.float32),
        wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )

    exporter.build_manifest()
    result = exporter.capture(
        sim_step=1,
        sim_time=0.01,
        overlays=SceneOverlays(frames=(caller_frame,)),
    )

    frames = result.frame.overlays.frames
    assert [frame.overlay_id for frame in frames] == [
        "marker:place_target_axis:0#1",
        "marker:place_target_axis:0",
    ]
    np.testing.assert_allclose(frames[0].position, [-0.4, 0.48, 0.1])
    np.testing.assert_allclose(frames[1].position, caller_frame.position)


def test_camera_frustum_pose_and_low_frequency_rgb_are_exported() -> None:
    simulation = _CameraSimulation()
    exporter = SceneExporter(
        simulation,
        VisualizationCfg(backend="viser", env_ids=[0, 1]),
        run_id="camera-run",
    )

    manifest = exporter.build_manifest()
    scene_result = exporter.capture(sim_step=3, sim_time=0.03)
    image_result = exporter.capture_camera_images(sim_step=3, sim_time=0.03)

    assert len(manifest.cameras) == 2
    assert manifest.cameras[0].sensor_uid == "wrist/camera"
    np.testing.assert_allclose(
        manifest.cameras[0].fov_y,
        2.0 * np.arctan(2.0 / 8.0),
    )
    np.testing.assert_allclose(scene_result.frame.camera_positions[1], [2.4, 0.5, 0.6])
    np.testing.assert_allclose(
        np.abs(scene_result.frame.camera_wxyz[0]),
        [0.0, 1.0, 0.0, 0.0],
        atol=1.0e-6,
    )
    assert simulation.camera.update_count == 1
    assert len(image_result.frame.images) == 2
    assert image_result.frame.images[0].image.shape == (2, 4, 3)
    np.testing.assert_array_equal(
        image_result.frame.images[1].image[0, 0],
        [40, 50, 60],
    )


def test_stereo_sensor_and_record_camera_primary_rgb_are_exported() -> None:
    simulation = _SensorPreviewSimulation()
    exporter = SceneExporter(
        simulation,
        VisualizationCfg(backend="viser", env_ids=[0]),
        run_id="sensor-preview-run",
    )

    manifest = exporter.build_manifest()
    scene_result = exporter.capture(sim_step=1, sim_time=0.01)
    image_result = exporter.capture_camera_images(sim_step=1, sim_time=0.01)

    assert [camera.sensor_uid for camera in manifest.cameras] == [
        "cam_high",
        "record_camera",
    ]
    assert [camera.role for camera in manifest.cameras] == ["sensor", "record"]
    assert [image.camera_id for image in image_result.frame.images] == [
        "env:0/camera:cam_high",
        "env:0/camera:record_camera",
    ]
    np.testing.assert_allclose(scene_result.frame.camera_positions[0], [0.1, 0.2, 0.3])
    np.testing.assert_array_equal(
        image_result.frame.images[0].image[0, 0],
        [10, 20, 30],
    )
    assert simulation.cameras["cam_high"].update_count == 1
    assert simulation.cameras["record_camera"].update_count == 1


def test_rigid_groups_and_deformable_meshes_are_exported() -> None:
    exporter = SceneExporter(
        _CompleteSimulation(),
        VisualizationCfg(backend="viser", env_ids=[0, 1]),
        run_id="complete-run",
    )

    manifest = exporter.build_manifest()
    pose_only = exporter.capture(
        sim_step=1,
        sim_time=0.01,
        capture_dynamic_geometry=False,
    )
    with_deformables = exporter.capture(
        sim_step=2,
        sim_time=0.02,
        capture_dynamic_geometry=True,
    )

    group_nodes = [node for node in manifest.nodes if node.kind == "rigid_group_object"]
    deformable_nodes = [node for node in manifest.nodes if node.dynamic_geometry]
    assert len(group_nodes) == 4
    assert {node.kind for node in deformable_nodes} == {
        "soft_object",
        "cloth_object",
    }
    assert len(deformable_nodes) == 4
    assert exporter.has_deformables
    assert pose_only.frame.dynamic_meshes == ()
    assert len(with_deformables.frame.dynamic_meshes) == 4

    group_node_id = "env:1/rigid_group:pair/object:right"
    group_index = with_deformables.frame.node_ids.index(group_node_id)
    np.testing.assert_allclose(
        with_deformables.frame.positions[group_index],
        [2.7, 0.0, 0.2],
    )
    soft_update = next(
        mesh
        for mesh in with_deformables.frame.dynamic_meshes
        if mesh.node_id == "env:1/soft:jelly"
    )
    np.testing.assert_allclose(
        soft_update.vertices,
        np.array(
            [[0.0, 0.0, 0.0], [0.15, 0.0, 0.0], [0.0, 0.15, 0.0]],
            dtype=np.float32,
        ),
        atol=1.0e-6,
    )
