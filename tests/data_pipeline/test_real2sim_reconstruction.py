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
"""Independent contracts for source preservation and observed scene recovery."""

from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
from embodichain.data_pipeline.real2sim import RobotDemonstration
from embodichain.data_pipeline.real2sim.reconstruction.inputs import (
    CameraObservation,
    RobotModel,
    ReconstructionInput,
    RegisteredScene,
)
from embodichain.data_pipeline.real2sim.reconstruction.video_fusion import (
    RecoveryCfg,
    recover_scene_image,
)
from embodichain.data_pipeline.real2sim.reconstruction.refinement import (
    refine_trajectory,
    detect_gripper_events,
)


def _inputs():
    names = tuple(f"j{i}" for i in range(8))
    demo = RobotDemonstration(
        source_id="source",
        source_revision="1",
        unit_id="unit",
        joint_names=names,
        joint_units=("rad",) * 6 + ("m",) * 2,
        qpos=np.zeros((100, 8)),
        timestamps=np.arange(100) / 30,
        time_unit="s",
        robot_geometry_sha256="a" * 64,
        recording_sha256="b" * 64,
    )
    robot = RobotModel(
        Path("not_loaded.urdf"), names, ("arm",), ("tcp",), np.eye(4), (), -0.035
    )
    camera = CameraObservation(Path("video.mp4"), Path("first.png"), np.arange(100))
    return ReconstructionInput(
        demo, robot, {"front": camera}, "front", "Move the object"
    )


def test_obstacle_broad_phase_preserves_penalties_and_checks_nearby_geometry():
    import open3d as o3d
    import trimesh
    from embodichain.data_pipeline.real2sim.reconstruction._contact_fit import (
        _body,
        _obstacle_distances,
    )

    body = _body("obstacle", trimesh.creation.box(), np.eye(4)[None], 16)
    points = np.array([[0.0, 0, 0], [0.5004, 0, 0], [0.6, 0, 0]])
    exact = body.bvh.compute_signed_distance(
        o3d.core.Tensor(points.astype(np.float32)), nthreads=2
    ).numpy()
    np.testing.assert_allclose(_obstacle_distances(body, points), exact)

    class NoQuery:
        def __getattr__(self, name):
            raise AssertionError("Separated bodies must not run mesh queries")

    far = points + [3.0, 0.0, 0.0]
    expected = body.bvh.compute_signed_distance(
        o3d.core.Tensor(far.astype(np.float32)), nthreads=2
    ).numpy()
    body.bvh = NoQuery()
    conservative = _obstacle_distances(body, far)
    assert np.all(conservative <= expected)
    np.testing.assert_array_equal(
        np.minimum(conservative - 0.001, 0), np.minimum(expected - 0.001, 0)
    )


def test_visual_only_robot_gets_real_collision_inputs_without_editing_source(tmp_path):
    import xml.etree.ElementTree as ET
    from embodichain.data_pipeline.real2sim.reconstruction._robot_geometry import (
        prepare_robot_collisions,
    )

    meshes = tmp_path / "meshes"
    meshes.mkdir()
    (meshes / "finger.obj").write_text("v 0 0 0\nv 0 1 0\nv 1 0 0\nf 1 2 3\n")
    source = tmp_path / "robot.urdf"
    original = (
        '<robot name="test"><link name="finger"><visual>'
        '<origin xyz="0.01 0.02 0.03" rpy="0 0 0.2"/>'
        '<geometry><mesh filename="meshes/finger.obj" scale="1 2 3"/></geometry>'
        '</visual><inertial><mass value="0.1"/></inertial></link></robot>'
    )
    source.write_text(original)
    target, report = prepare_robot_collisions(
        source, tmp_path / "prepared", ("finger",)
    )
    root = ET.parse(target).getroot()
    link = root.find("link")
    assert report["derived_from_visuals"]
    assert report["generated_links"] == ["finger"]
    assert ET.tostring(link.find("collision/geometry")) == ET.tostring(
        link.find("visual/geometry")
    )
    assert link.find("collision/origin").attrib == link.find("visual/origin").attrib
    assert link.find("inertial/mass").get("value") == "0.1"
    assert (target.parent / "meshes/finger.obj").resolve() == meshes / "finger.obj"
    assert source.read_text() == original
    kept, info = prepare_robot_collisions(target, tmp_path / "unused", ("finger",))
    assert kept == target and not info["derived_from_visuals"]
    assert not (tmp_path / "unused").exists()
    ET.SubElement(root, "link", name="missing_finger")
    ET.ElementTree(root).write(target)
    with pytest.raises(ValueError, match="lacks collision geometry"):
        prepare_robot_collisions(target, tmp_path / "invalid", ("missing_finger",))


def test_settling_updates_geometry_without_mutating_source_and_rejects_interaction():
    from embodichain.data_pipeline.real2sim.reconstruction.settling import (
        apply_settled_poses,
    )

    matrix = np.diag([2.0, 2.0, 2.0, 1.0])
    matrix[2, 3] = 0.03
    points = np.array([[0, 0, 0.03], [0.02, 0, 0.03], [0, 0.02, 0.05]])
    scene = RegisteredScene(
        {"objects": [{"id": "cup", "transform": matrix.tolist()}]},
        {"cup": points.copy()},
    )
    final = np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0.02], [0, 0, 0, 1]])
    poses = np.broadcast_to(final, (301, 1, 4, 4)).copy()
    contacts = np.zeros(301, bool)
    times = np.arange(301) / 30
    times[-1] = times[-2]  # Native final checkpoints can repeat the last sample.
    result, _ = apply_settled_poses(scene, times, ("cup",), poses, contacts)
    expected = np.array([[0, 0, 0.02], [0, 0.02, 0.02], [-0.02, 0, 0.04]])
    np.testing.assert_allclose(result.vertices["cup"], expected, atol=1e-12)
    np.testing.assert_allclose(
        np.linalg.norm(
            np.asarray(result.description["objects"][0]["transform"])[:3, :3], axis=0
        ),
        [2, 2, 2],
    )
    np.testing.assert_array_equal(scene.vertices["cup"], points)
    np.testing.assert_array_equal(scene.description["objects"][0]["transform"], matrix)
    contacts[10] = True
    with pytest.raises(ValueError, match="without robot contact"):
        apply_settled_poses(scene, times, ("cup",), poses, contacts)
    contacts[10] = False
    poses[-1, 0, 2, 3] += 0.01
    with pytest.raises(ValueError, match="did not settle"):
        apply_settled_poses(scene, times, ("cup",), poses, contacts)


def test_initial_support_separation_handles_nested_fits_without_changing_source():
    from embodichain.data_pipeline.real2sim.reconstruction.settling import (
        prepare_settling_scene,
    )

    objects, vertices = [], {}
    # Child-first input reproduces independently registered, overlapping meshes.
    for uid, low, high in [
        ("block", -0.003, 0.06),
        ("paper", -0.006, 0.004),
        ("table", -0.04, 0),
    ]:
        points = np.array([[0.0, 0.0, low], [0.1, 0.2, high]])
        matrix = np.diag([2.0, 3.0, 4.0, 1.0])
        matrix[:3, 3] = [0.05, 0.1, low]
        objects.append(dict(id=uid, bounds=points.tolist(), transform=matrix.tolist()))
        vertices[uid] = points
    scene = RegisteredScene({"objects": objects}, vertices)
    nodes = [
        dict(object_id="block", parent_id="paper", parent_relation="on"),
        dict(object_id="paper", parent_id="table", parent_relation="on"),
        dict(object_id="table", parent_id=None),
    ]
    result, report = prepare_settling_scene(scene, nodes)
    np.testing.assert_allclose(result.vertices["paper"][:, 2], [0.02, 0.03])
    np.testing.assert_allclose(result.vertices["block"][:, 2], [0.05, 0.113])
    assert report["vertical_lift_m"]["block"] == pytest.approx(0.053)
    for before, after in zip(objects, result.description["objects"]):
        uid = before["id"]
        np.testing.assert_array_equal(result.vertices[uid][:, :2], vertices[uid][:, :2])
        np.testing.assert_array_equal(
            np.asarray(after["transform"])[:3, :3], np.diag([2, 3, 4])
        )
    np.testing.assert_array_equal(scene.vertices["paper"][:, 2], [-0.006, 0.004])
    assert scene.description["objects"][0]["transform"][2][3] == -0.003
    nodes[2]["parent_id"] = "block"
    with pytest.raises(ValueError, match="Cyclic"):
        prepare_settling_scene(scene, nodes)


def test_frame_mapping_and_metric_calibration_are_explicit():
    with pytest.raises(ValueError, match="frame IDs"):
        CameraObservation(Path("v"), Path("i"), np.array([0.0, 1.0]))
    with pytest.raises(ValueError, match="frame IDs"):
        CameraObservation(Path("v"), Path("i"), np.array([1, 0], dtype=np.uint64))
    with pytest.raises(ValueError, match="Depth requires"):
        CameraObservation(Path("v"), Path("i"), np.arange(2), depth_m=np.ones((4, 4)))
    source = _inputs()
    with pytest.raises(ValueError, match="first_image"):
        replace(
            source,
            cameras={"front": replace(source.cameras["front"], first_image=None)},
        )
    short = replace(source.cameras["front"], sample_frames=np.arange(3))
    with pytest.raises(ValueError, match="one video-frame mapping"):
        replace(source, cameras={"front": short})


def test_settling_uses_local_surface_on_a_sloped_table(tmp_path):
    import trimesh
    from embodichain.data_pipeline.real2sim.reconstruction.settling import (
        prepare_settling_scene,
    )

    table = trimesh.creation.box(extents=[2, 1, 0.04])
    table.apply_translation([0, 0, -0.02])
    shear = np.eye(4)
    shear[2, 0] = 0.4
    table.apply_transform(shear)
    path = tmp_path / "table.glb"
    table.export(path)
    block = trimesh.creation.box(extents=[0.1, 0.1, 0.1])
    block.apply_translation([-0.8, 0, -0.27])
    objects = [
        dict(id=uid, bounds=m.bounds.tolist(), transform=np.eye(4).tolist())
        for uid, m in [("table", table), ("block", block)]
    ]
    objects[0]["path"] = str(path)
    scene = RegisteredScene(
        {"objects": objects},
        {"table": table.vertices.copy(), "block": block.vertices.copy()},
    )
    nodes = [
        dict(object_id="table", parent_id=None),
        dict(object_id="block", parent_id="table", parent_relation="on"),
    ]
    result, report = prepare_settling_scene(scene, nodes)
    assert report["vertical_lift_m"]["block"] == pytest.approx(0.04, abs=1e-6)
    np.testing.assert_array_equal(
        result.vertices["block"][:, :2], block.vertices[:, :2]
    )
    scene.vertices["block"][:, 0] += 4
    with pytest.raises(ValueError, match="No upward table surface"):
        prepare_settling_scene(scene, nodes)


def test_recovery_uses_visible_donors_keeps_unknown_and_preserves_other_pixels():
    rng = np.random.default_rng(17)
    background = rng.integers(0, 256, (96, 128, 3), dtype=np.uint8)
    masks = {i: np.zeros((96, 128), bool) for i in range(4)}
    masks[0][30:60, 40:80] = True
    masks[1][30:40, 40:50] = True
    masks[2][30:40, 40:50] = True
    images = {i: background.copy() for i in masks}
    for i, mask in masks.items():
        images[i][mask] = 0
    source = images[0].copy()
    result = recover_scene_image(images, masks, [1, 2], cfg=RecoveryCfg(mask_margin=0))
    expected = masks[0] & ~masks[1]
    np.testing.assert_array_equal(result.rgb[expected], background[expected])
    np.testing.assert_array_equal(result.rgb[~masks[0]], source[~masks[0]])
    np.testing.assert_array_equal(result.unresolved, masks[1])
    assert np.all(result.donor_frames[masks[1]] == -1)
    assert set(np.unique(result.donor_frames[expected])) <= {1, 2}
    np.testing.assert_array_equal(images[0], source)
    with pytest.raises(ValueError, match="memory budget"):
        recover_scene_image(images, masks, [1, 2], cfg=RecoveryCfg(max_working_bytes=1))


def test_non_30hz_source_is_rejected_before_robot_or_scene_loading():
    source = _inputs()
    changed = replace(
        source,
        demonstration=replace(source.demonstration, timestamps=np.arange(100) / 25),
    )
    with pytest.raises(ValueError, match="30 Hz"):
        refine_trajectory(changed, RegisteredScene({}, {}))
    np.testing.assert_array_equal(source.demonstration.timestamps, np.arange(100) / 30)


def test_agibot_missing_observation_keeps_original_clock_and_frame_indices():
    from embodichain.data_pipeline.real2sim.reconstruction.refinement import _clock_gaps

    source = _inputs()
    timestamps = np.arange(100) * 0.033339
    timestamps[41:] += 0.033336
    timestamps[76:] += 0.033336
    inputs = replace(
        source,
        demonstration=replace(source.demonstration, timestamps=timestamps),
        robot=replace(source.robot, gripper_calibrations=({},)),
    )
    assert _clock_gaps(inputs) == [40, 75]
    np.testing.assert_array_equal(inputs.demonstration.timestamps, timestamps)
    np.testing.assert_array_equal(inputs.cameras["front"].sample_frames, np.arange(100))
    interrupted = timestamps.copy()
    interrupted[50:] += 0.1
    with pytest.raises(ValueError, match="one-sample gaps"):
        _clock_gaps(
            replace(
                inputs,
                demonstration=replace(inputs.demonstration, timestamps=interrupted),
            )
        )


@pytest.mark.parametrize("offset", [0.0, 0.12])
@pytest.mark.parametrize("native_table_scale", [(1.0, 1.0, 1.0), (0.25, 0.5, 1.0)])
def test_register_single_object_from_calibrated_table_depth(
    tmp_path, offset, native_table_scale
):
    import open3d as o3d
    import trimesh
    from PIL import Image
    from scipy.spatial.transform import Rotation
    from embodichain.data_pipeline.real2sim.reconstruction.registration import (
        register_scene,
    )

    table = trimesh.creation.box(extents=[0.8, 0.8, 0.04])
    table.apply_translation([0, 0, -0.02])
    block = trimesh.creation.box(extents=[0.06, 0.06, 0.06])
    block.apply_translation([offset, 0, 0.03])
    bvh = o3d.t.geometry.RaycastingScene(nthreads=2)
    assets = []
    for uid, mesh in [("table", table), ("block", block)]:
        bvh.add_triangles(
            o3d.t.geometry.TriangleMesh(
                o3d.core.Tensor(np.asarray(mesh.vertices, dtype=np.float32)),
                o3d.core.Tensor(np.asarray(mesh.faces, dtype=np.int64)),
            )
        )
        y_up = mesh.copy()
        y_up.apply_transform(
            trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
        )
        path = tmp_path / f"{uid}.glb"
        y_up.export(path)
        assets.append(
            dict(
                uid=uid,
                name=uid,
                category=uid,
                mesh_path=str(path),
                init_pos=[0, 0, 0],
                init_rot=[0, 0, 0],
                body_scale=list(native_table_scale) if uid == "table" else [1, 1, 1],
            )
        )
    k = np.array([[100, 0, 64], [0, 100, 64], [0, 0, 1]])
    camera = np.eye(4)
    camera[:3, :3] = Rotation.from_euler("x", np.pi).as_matrix()
    camera[2, 3] = 1
    yy, xx = np.indices((128, 128))
    rays = np.zeros((128, 128, 6), dtype=np.float32)
    rays[..., :3] = camera[:3, 3]
    rays[..., 3:] = (
        np.stack([(xx - 64) / 100, (yy - 64) / 100, np.ones_like(xx)], -1)
        @ camera[:3, :3].T
    )
    hit = bvh.cast_rays(o3d.core.Tensor(rays), nthreads=2)
    depth = hit["t_hit"].numpy()
    depth[~np.isfinite(depth)] = 0
    labels = hit["geometry_ids"].numpy()
    masks = {"table": labels == 0, "block": labels == 1}
    image = tmp_path / "image.png"
    Image.new("RGB", (128, 128), (100, 100, 100)).save(image)
    observation = CameraObservation(
        Path("unused.mp4"), image, np.arange(2), k, camera, depth
    )
    result = register_scene(observation, assets, masks, {"block": "table"}, 0.0)
    actual = next(obj for obj in result.description["objects"] if obj["id"] == "block")
    np.testing.assert_allclose(actual["bounds"], block.bounds, atol=0.012)
    actual_table = next(
        obj for obj in result.description["objects"] if obj["id"] == "table"
    )
    # A native table/foreground scale mismatch must not leave targets outside
    # their support after independent metric object fitting.
    bounds = np.asarray(actual_table["bounds"])
    np.testing.assert_allclose(bounds[:, :2], table.bounds[:, :2], atol=0.02)
    assert bounds[1, 2] == pytest.approx(0, abs=0.002)
    # The underside is unobserved and retains the generated thickness.
    assert result.description["source_table_plane"][2] == pytest.approx(0, abs=0.002)


def test_container_release_clears_fingers_and_does_not_lower_a_safe_demonstration():
    from embodichain.data_pipeline.real2sim.reconstruction.refinement import (
        _container_release_lift,
    )

    object_points = np.array([[0.0, 0.0, 0.20], [0.02, 0.01, 0.25]])
    finger_points = np.array([[0.0, 0.0, 0.18], [0.01, 0.01, 0.26]])
    lift = _container_release_lift(object_points, finger_points, 0.20)
    assert lift == pytest.approx(0.023)
    assert (
        _container_release_lift(
            object_points + [0, 0, 0.1], finger_points + [0, 0, 0.1], 0.20
        )
        == 0
    )


def test_observed_release_relation_overrides_generated_asset_category():
    from embodichain.data_pipeline.real2sim.reconstruction.refinement import (
        _inside_receiver,
    )

    assert _inside_receiver("mesh_cup", {"receiver_relation": "inside"})
    assert _inside_receiver("unknown_asset", {"receiver_relation": "inside"})
    assert not _inside_receiver("cup", {"receiver_relation": "on"})
    assert not _inside_receiver("cup", {"receiver_relation": None})
    assert _inside_receiver("cup", None)  # Existing ungrounded API behavior.


def test_release_uses_sheet_support_without_category_whitelist(tmp_path):
    import trimesh
    from embodichain.data_pipeline.real2sim.reconstruction.refinement import (
        _support_at_release,
    )

    sheet = trimesh.creation.box(extents=[0.5, 0.4, 0.02])
    sheet.apply_translation([0, 0, 0.01])
    sheet.export(tmp_path / "sheet.glb")
    carried = trimesh.creation.box(extents=[0.05, 0.05, 0.05])
    carried.apply_translation([0, 0, 0.2])
    carried.export(tmp_path / "carried.glb")
    objects = {
        name: dict(path=str(tmp_path / f"{name}.glb"), transform=np.eye(4).tolist())
        for name in ["sheet", "carried"]
    }
    deltas = {name: np.eye(4) for name in objects}
    height, uid = _support_at_release(
        objects, deltas, "carried", np.array([[0.0, 0.0]]), 0.0
    )
    assert uid == "sheet"
    assert height == pytest.approx(0.02, abs=1e-6)
    assert _support_at_release(
        objects, deltas, "carried", np.array([[0.8, 0.8]]), 0.0
    ) == (0.0, None)


def test_approach_opens_before_contact_and_preserves_other_phases():
    from embodichain.data_pipeline.real2sim.reconstruction.refinement import (
        _approach_opening_profile,
    )

    original = np.full(160, 0.0035)
    original[115:131] = np.linspace(0.05, 0.0035, 16)
    raw = original.copy()
    fitted = 0.031
    result, report = _approach_opening_profile(
        original, original, 115, 130, fitted, 0.055
    )
    assert (result[55:116] >= fitted + 0.01).all()
    assert result[130] == pytest.approx(fitted)
    assert (np.diff(result[115:131]) <= 1e-12).all()
    assert result.max() <= 0.055
    np.testing.assert_array_equal(result[:25], original[:25])
    np.testing.assert_array_equal(result[131:], original[131:])
    np.testing.assert_array_equal(original, raw)


def test_opening_profile_bounds_release_residual_without_changing_source():
    from embodichain.data_pipeline.real2sim.reconstruction.refinement import (
        _approach_opening_profile,
    )

    original = np.full(180, 0.0035)
    original[115:131] = np.linspace(0.0535, 0.0035, 16)
    original[145:156] = np.linspace(0.0035, 0.0535, 11)
    original[156:] = 0.0535
    adjusted = original.copy()
    adjusted[130:151] = 0.032
    adjusted[151:161] += np.linspace(0.0285, 0, 10)
    source_copy, adjusted_copy = original.copy(), adjusted.copy()
    result, report = _approach_opening_profile(
        original, adjusted, 115, 130, 0.032, 0.055
    )
    assert result.max() <= 0.055
    np.testing.assert_allclose(result[131:], np.minimum(adjusted[131:], 0.055))
    assert report["limit_clamped_frames"] == np.flatnonzero(adjusted > 0.055).tolist()
    np.testing.assert_array_equal(original, source_copy)
    np.testing.assert_array_equal(adjusted, adjusted_copy)


def test_gripper_event_detection_preserves_complete_input():
    q = np.zeros((300, 8))
    aperture = np.interp(
        np.arange(300),
        [0, 30, 50, 100, 130, 210, 240, 299],
        [0.01, 0.01, 0.05, 0.05, 0.015, 0.015, 0.05, 0.05],
    )
    q[:, 6] = q[:, 7] = aperture
    original = q.copy()
    events = detect_gripper_events(q)
    assert len(events) == 1
    event = events[0]
    assert 90 < event["close_start"] < 115
    assert 115 < event["grasp"] < 140
    assert 210 < event["release"] < 245
    assert event["side"] == 0
    np.testing.assert_array_equal(q, original)


@pytest.mark.parametrize("closing_transition", [False, True])
def test_approach_clearance_lifts_finger_sweep_without_changing_other_motion(
    tmp_path, closing_transition
):
    import trimesh
    from embodichain.data_pipeline.real2sim.reconstruction._approach import (
        clear_approach,
    )

    # An analytic Cartesian stage isolates clearance geometry from IK accuracy.
    class Stage:
        def finger_surfaces(self, side):
            return ["finger"], [np.array([1, 0, 0])]

        def tcp(self, q, side):
            transforms = np.tile(np.eye(4), (len(q), 1, 1))
            transforms[:, :3, 3] = q[:, :3]
            return transforms

        def links(self, q):
            return {"finger": self.tcp(q, 0)}

    def solve_ik(robot, source, side, target):
        result = source.copy()
        result[:3] = target[:3, 3]
        return result, {"tcp_error_m": 0.0}

    finger = trimesh.creation.box(extents=[0.015, 0.015, 0.08])
    finger.export(tmp_path / "finger.glb")
    block = trimesh.creation.box(extents=[0.06, 0.06, 0.06])
    block.apply_translation([0, 0, 0.03])
    block.export(tmp_path / "block.glb")
    scene = {
        "objects": [
            dict(
                id="block",
                path=str(tmp_path / "block.glb"),
                transform=np.eye(4).tolist(),
            )
        ]
    }
    visuals = (
        dict(link="finger", path=str(tmp_path / "finger.glb"), origin=np.eye(4)),
    )
    q = np.zeros((50, 4))
    q[:, 0] = np.linspace(-0.1, 0.0 if closing_transition else 0.1, 50)
    q[:, 2] = 0.05
    q[:, 3] = np.arange(50)  # unrelated joint must remain untouched
    original = q.copy()
    close_start = 35 if closing_transition else 49
    corrected, report = clear_approach(
        Stage(),
        q,
        0,
        scene,
        {},
        visuals,
        0,
        close_start,
        solve_ik,
        grasp=49 if closing_transition else None,
    )
    assert report["max_lift_m"] > 0.05
    # While crossing the block footprint, the actual finger bottom clears it.
    crossing = (abs(q[:, 0]) < 0.03) & (np.arange(len(q)) <= close_start)
    assert np.min(corrected[crossing, 2] - 0.04) >= 0.061
    np.testing.assert_array_equal(corrected[:, [0, 1, 3]], q[:, [0, 1, 3]])
    np.testing.assert_array_equal(corrected[[0, -1]], q[[0, -1]])
    np.testing.assert_array_equal(q, original)


def test_idle_clearance_windows_do_not_cross_successful_grasp_anchors():
    from embodichain.data_pipeline.real2sim.reconstruction._approach import (
        _idle_windows,
    )

    phases = [
        dict(side=0, close_start=20, grasp=30, open_end=50),
        dict(side=0, close_start=70, grasp=80, open_end=100),
        dict(side=0, close_start=130, grasp=140, open_end=160),
        dict(side=0, close_start=190, grasp=200, open_end=220),
    ]
    grounding = {(0, g): {"object_id": None} for g in [30, 80, 200]}
    grounding[(0, 140)] = {"object_id": "block"}
    result = _idle_windows(phases, grounding, 250, 120)
    assert result == [
        dict(side=0, start=0, checked_end=100, fade_end=140, grasps=[30, 80]),
        dict(side=0, start=160, checked_end=220, fade_end=249, grasps=[200]),
    ]


def test_recording_starting_with_open_jaws_keeps_first_grasp():
    q = np.zeros((400, 8))
    opening = np.interp(
        np.arange(400),
        [0, 80, 120, 260, 300, 399],
        [0.05, 0.05, 0.015, 0.015, 0.05, 0.05],
    )
    q[:, 6] = q[:, 7] = opening
    events = detect_gripper_events(q, require_open_transition=False)
    assert len(events) == 1
    assert 100 < events[0]["grasp"] < 140
    assert 260 < events[0]["release"] < 300


def test_explicit_seven_axis_and_coupled_gripper_keep_other_joints(tmp_path):
    from embodichain.data_pipeline.real2sim.reconstruction._kinematics import Robot

    names = ("torso", *[f"arm{i}" for i in range(7)], "inner", "outer")
    links = '<link name="base"/>' + "".join(
        f'<link name="link{i}"/>' for i in range(10)
    )
    joints = "".join(
        f'<joint name="{name}" type="revolute"><parent link="{("base" if i == 0 else "link" + str(i-1))}"/>'
        f'<child link="link{i}"/><axis xyz="0 0 1"/><limit lower="-3" upper="3" effort="10" velocity="2"/></joint>'
        for i, name in enumerate(names)
    )
    urdf = tmp_path / "robot.urdf"
    urdf.write_text(f'<robot name="coupled">{links}{joints}</robot>')
    calibration = dict(
        joint_names=["inner", "outer"],
        width_m=[0, 0.1],
        qpos=[[0, 0], [1, -1]],
        pad_links=["link8", "link9"],
        outward_axes=[[1, 0, 0], [1, 0, 0]],
    )
    model = RobotModel(
        urdf,
        names,
        ("arm",),
        ("link7",),
        np.eye(4),
        (),
        0.0,
        arm_joint_groups=(names[1:8],),
        gripper_calibrations=(calibration,),
    )
    robot = Robot(model)
    source = np.full((3, 10), 0.2)
    result = robot.set_half_open(source, 0, np.array([0, 0.025, 0.05]))
    np.testing.assert_array_equal(result[:, :8], source[:, :8])
    np.testing.assert_allclose(result[:, 8:], [[0, 0], [0.5, -0.5], [1, -1]])
    np.testing.assert_allclose(robot.half_open(result, 0), [0, 0.025, 0.05])
    assert len(robot.arm_columns[0]) == 7
    np.testing.assert_array_equal(source, np.full((3, 10), 0.2))
