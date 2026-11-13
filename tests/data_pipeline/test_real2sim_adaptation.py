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
from embodichain.data_pipeline.real2sim.adaptation.inputs import (
    CameraObservation,
    RobotModel,
    ReconstructionInput,
    RegisteredScene,
)
from embodichain.data_pipeline.real2sim.adaptation.video_fusion import (
    RecoveryCfg,
    recover_scene_image,
)
from embodichain.data_pipeline.real2sim.adaptation.refinement import (
    refine_trajectory,
    detect_gripper_events,
)


def test_comparison_export_preserves_explicit_source_frame_mapping(tmp_path):
    import json
    import av
    from embodichain.data_pipeline.real2sim.adaptation.media import (
        create_video_stream,
        encode_video_frame,
        export_comparison,
    )

    source, output = tmp_path / "source", tmp_path / "replay"
    source.mkdir()
    (output / "videos").mkdir(parents=True)
    (output / "diagnostics").mkdir()
    np.savez(
        source / "trajectory_refined.npz",
        timestamps=[0, 0.13, 0.26],
        source_frame_indices=[0, 2, 4],
    )
    (output / "diagnostics/render_times.json").write_text(
        json.dumps({"timestamps": [0, 0.1, 0.2]})
    )
    for path, colors in (
        (source / "original_front.mp4", [(r, 0, 0) for r in (40, 80, 120, 160, 200)]),
        (output / "videos/simulation.mp4", [(0, 0, 180)] * 3),
    ):
        with av.open(str(path), "w") as video:
            stream = create_video_stream(video, 640, 480)
            for index, color in enumerate(colors):
                image = np.full((480, 640, 3), color, dtype=np.uint8)
                encode_video_frame(video, stream, image, index)
            for packet in stream.encode():
                video.mux(packet)

    target = export_comparison(source, output)
    metadata = json.loads((output / "diagnostics/comparison_full.json").read_text())
    assert metadata["source_frame_indices"] == [0, 2, 4]
    assert metadata["trajectory_sample_indices"] == [0, 1, 2]
    assert metadata["frames"] == 3
    with av.open(str(target)) as video:
        frames = [frame.to_ndarray(format="rgb24") for frame in video.decode(video=0)]
    assert len(frames) == 3
    for frame, red in zip(frames, (40, 120, 200), strict=True):
        assert frame.shape == (480, 1280, 3)
        np.testing.assert_allclose(frame[240, 320], [red, 0, 0], atol=8)
        np.testing.assert_allclose(frame[240, 960], [0, 0, 180], atol=8)


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


@pytest.mark.parametrize("capture_yaw", [0.0, np.pi / 2])
def test_handoff_conditions_receiver_without_reapplying_captured_object_motion(
    capture_yaw,
):
    from scipy.spatial.transform import Rotation
    from embodichain.data_pipeline.real2sim.adaptation.refinement import (
        _handoff_receiver_target,
    )

    def pose(yaw, xyz):
        result = np.eye(4)
        result[:3, :3] = Rotation.from_euler("z", yaw).as_matrix()
        result[:3, 3] = xyz
        return result

    # The donor's recorded motion is a quarter turn around the origin plus
    # one metre in X. The receiver is one metre along its resulting local X.
    donor_start = pose(0, [0.5, 0, 0])
    donor_end = pose(np.pi / 2, [1, 0.5, 0])
    receiver = pose(np.pi / 2, [1, 1, 0])
    captured = pose(capture_yaw, [0, 0, 4])
    held = pose(np.pi + capture_yaw, [3, 2, 4])
    target, nominal = _handoff_receiver_target(
        donor_start, donor_end, receiver, captured, held
    )
    np.testing.assert_allclose(target, pose(np.pi, [2, 2, 0]), atol=1e-12)
    np.testing.assert_allclose(
        nominal, pose(np.pi / 2 + capture_yaw, [1, 0, 4]), atol=1e-12
    )
    unchanged, _ = _handoff_receiver_target(
        donor_start, donor_end, receiver, captured, nominal
    )
    np.testing.assert_allclose(unchanged, receiver, atol=1e-12)


def test_obstacle_broad_phase_preserves_penalties_and_checks_nearby_geometry():
    import open3d as o3d
    import trimesh
    from embodichain.data_pipeline.real2sim.adaptation._contact_fit import (
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
    from embodichain.data_pipeline.real2sim.adaptation._robot_geometry import (
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
    from embodichain.data_pipeline.real2sim.adaptation.refinement import _clock_gaps

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
def test_registration_preserves_native_scene_under_one_similarity(
    tmp_path, offset, native_table_scale
):
    import open3d as o3d
    import trimesh
    from PIL import Image
    from scipy.spatial.transform import Rotation
    from embodichain.data_pipeline.real2sim.adaptation.registration import (
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
    before = {a["uid"]: Path(a["mesh_path"]).read_bytes() for a in assets}
    result = register_scene(observation, assets, masks, 0.0)
    shared = np.asarray(result.description["global_similarity"])
    rotation_scale = shared[:3, :3]
    scale = np.cbrt(np.linalg.det(rotation_scale))
    assert scale > 0
    np.testing.assert_allclose(
        rotation_scale.T @ rotation_scale, np.eye(3) * scale**2, atol=1e-10
    )
    zup = np.eye(4)
    zup[:3, :3] = Rotation.from_euler("x", 90, degrees=True).as_matrix()
    objects = {o["id"]: o for o in result.description["objects"]}
    native_transforms = {}
    for asset in assets:
        uid = asset["uid"]
        native = np.diag([*asset["body_scale"], 1.0]) @ zup
        native_transforms[uid] = native
        actual = np.asarray(objects[uid]["transform"])
        np.testing.assert_allclose(actual, shared @ native, atol=1e-10)
        assert Path(asset["mesh_path"]).read_bytes() == before[uid]
    # Even when source and reconstruction differ, registration cannot move the
    # block relative to its table or stretch either asset to match the source.
    np.testing.assert_allclose(
        np.linalg.inv(objects["table"]["transform"]) @ objects["block"]["transform"],
        np.linalg.inv(native_transforms["table"]) @ native_transforms["block"],
        atol=1e-10,
    )
    assert "table_color_srgb" not in result.description
    assert result.description["source_table_plane"][2] == pytest.approx(0, abs=0.002)
    assert np.max(result.vertices["table"][:, 2]) == pytest.approx(0, abs=0.002)


def test_container_release_clears_fingers_and_does_not_lower_a_safe_demonstration():
    from embodichain.data_pipeline.real2sim.adaptation.refinement import (
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
    from embodichain.data_pipeline.real2sim.adaptation.refinement import (
        _inside_receiver,
    )

    assert _inside_receiver("mesh_cup", {"receiver_relation": "inside"})
    assert _inside_receiver("unknown_asset", {"receiver_relation": "inside"})
    assert not _inside_receiver("cup", {"receiver_relation": "on"})
    assert not _inside_receiver("cup", {"receiver_relation": None})
    assert _inside_receiver("cup", None)  # Existing ungrounded API behavior.


def test_release_uses_sheet_support_without_category_whitelist(tmp_path):
    import trimesh
    from embodichain.data_pipeline.real2sim.adaptation.refinement import (
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
    from embodichain.data_pipeline.real2sim.adaptation.refinement import (
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
    from embodichain.data_pipeline.real2sim.adaptation.refinement import (
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


@pytest.mark.parametrize(
    "closing_transition, entry_collision, finger_gain",
    [
        (False, False, 1.0),
        (True, False, 1.0),
        (False, True, 1.0),
        (False, False, 0.3),
        (False, False, 0.0),
    ],
)
def test_approach_clearance_lifts_finger_sweep_without_changing_other_motion(
    tmp_path, closing_transition, entry_collision, finger_gain
):
    import trimesh
    from embodichain.data_pipeline.real2sim.adaptation._approach import (
        clear_approach,
    )

    # An analytic Cartesian stage isolates clearance geometry from IK accuracy.
    class Stage:
        path = tmp_path / "hand.urdf"
        names = ["jaw", "other_jaw"]
        jaw_columns = [[0, 1]]

        def finger_surfaces(self, side):
            return ["finger", "other"], [np.array([1, 0, 0]), np.array([-1, 0, 0])]

        def tcp(self, q, side):
            transforms = np.tile(np.eye(4), (len(q), 1, 1))
            transforms[:, :3, 3] = q[:, :3]
            return transforms

        def links(self, q):
            transforms = self.tcp(q, 0)
            if finger_gain != 1:
                # A coupled stage changes finger clearance more slowly than
                # TCP height, requiring all eight corrections and a final check.
                transforms[:, 2, 3] = 0.09 + finger_gain * (q[:, 2] - 0.05)
            other = transforms.copy()
            other[:, 0, 3] += 1.0
            return {"finger": transforms, "other": other}

    def solve_ik(robot, source, side, target):
        result = source.copy()
        result[:3] = target[:3, 3]
        return result, {"tcp_error_m": 0.0}

    Stage.path.write_text(
        '<robot name="stage"><link name="palm"/>'
        + "".join(
            f'<link name="{link}"><collision><geometry><box size="0.015 0.015 0.08"/></geometry></collision></link>'
            f'<joint name="{joint}" type="prismatic"><parent link="palm"/><child link="{link}"/><axis xyz="1 0 0"/></joint>'
            for link, joint in [("finger", "jaw"), ("other", "other_jaw")]
        )
        + "</robot>"
    )
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
    q[:, 0] = np.linspace(
        -0.02 if entry_collision else -0.1, 0.0 if closing_transition else 0.1, 50
    )
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
    assert report["max_lift_m"] > 0
    if finger_gain == 0:
        # An unreachable preferred clearance remains a reviewable candidate.
        # It cannot certify task failure before a physical rollout exists.
        assert report["status"] == "review_needed"
        assert report["minimum_checked_gap_m"] < 0.003
        assert not report["physical_success_verified"]
        assert np.isfinite(corrected).all()
        np.testing.assert_array_equal(corrected[:, [0, 1, 3]], q[:, [0, 1, 3]])
        np.testing.assert_array_equal(q, original)
        return
    # While crossing the block footprint, the actual finger bottom clears it.
    crossing = (abs(q[:, 0]) < 0.03) & (np.arange(len(q)) <= close_start)
    bottom = Stage().links(corrected)["finger"][:, 2, 3] - 0.04
    assert np.min(bottom[crossing]) >= 0.063
    if finger_gain != 1:
        assert report["iterations"] == 8
    np.testing.assert_array_equal(corrected[:, [0, 1, 3]], q[:, [0, 1, 3]])
    np.testing.assert_array_equal(corrected[-1], q[-1])
    if entry_collision:
        assert corrected[0, 2] - 0.04 >= 0.061
    else:
        np.testing.assert_array_equal(corrected[0], q[0])
    np.testing.assert_array_equal(q, original)


def test_idle_clearance_windows_do_not_cross_successful_grasp_anchors():
    from embodichain.data_pipeline.real2sim.adaptation._approach import (
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
        dict(side=0, start=0, checked_end=130, fade_end=140, grasps=[30, 80]),
        dict(side=0, start=160, checked_end=249, fade_end=249, grasps=[200]),
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
    from embodichain.data_pipeline.real2sim.adaptation._kinematics import Robot

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


def test_gripper_commands_use_urdf_limit_before_closure_is_refined(tmp_path):
    from embodichain.data_pipeline.real2sim.adaptation._kinematics import Robot
    from embodichain.data_pipeline.real2sim.adaptation.refinement import (
        _approach_opening_profile,
    )

    names = tuple(f"j{i}" for i in range(8))
    links = '<link name="base"/>' + "".join(f'<link name="link{i}"/>' for i in range(8))
    joints = "".join(
        f'<joint name="{name}" type="{("revolute" if i < 6 else "prismatic")}">'
        f'<parent link="{("base" if i == 0 else "link" + str(i-1))}"/>'
        f'<child link="link{i}"/><axis xyz="0 0 1"/>'
        f'<limit lower="{(-3 if i < 6 else 0)}" upper="{(3 if i < 6 else 0.05)}" effort="10" velocity="2"/></joint>'
        for i, name in enumerate(names)
    )
    urdf = tmp_path / "robot.urdf"
    urdf.write_text(f'<robot name="parallel">{links}{joints}</robot>')
    robot = Robot(RobotModel(urdf, names, ("arm",), ("link5",), np.eye(4), (), 0.0))
    # The generic 65 mm setting cannot override this robot's 50 mm jaw travel.
    assert robot.opening_limits == [(0.0, 0.05)]
    source = np.zeros((100, 8))
    source[:, 6:] = np.interp(
        np.arange(100), [0, 50, 70, 99], [0.055, 0.055, 0.013, 0.013]
    )[:, None]
    original = source.copy()
    fitted = robot.opening_limits[0][1] - 0.002
    opening, _ = _approach_opening_profile(
        source[:, 6], source[:, 6], 50, 70, fitted, robot.opening_limits[0][1], 0
    )
    command = robot.set_half_open(source, 0, opening)
    assert command[50, 6] == pytest.approx(0.05)
    assert command[70, 6] == pytest.approx(0.048)
    assert np.all(command[:, 6:] <= 0.05)
    np.testing.assert_array_equal(command[:, :6], source[:, :6])
    np.testing.assert_array_equal(source, original)


def test_cad_visual_winding_repair_preserves_source_and_closed_surface(tmp_path):
    import trimesh
    from embodichain.data_pipeline.real2sim.adaptation._robot_geometry import (
        prepare_robot_visual_winding,
    )

    box = trimesh.creation.box()
    triangles = box.vertices[box.faces].copy()
    triangles[:3] = triangles[:3, ::-1]
    # CAD normal seams duplicate vertices, concealing adjacency from a raw mesh.
    broken = trimesh.Trimesh(
        vertices=triangles.reshape(-1, 3),
        faces=np.arange(triangles.size // 3).reshape(-1, 3),
        process=False,
    )
    source = tmp_path / "source.glb"
    broken.export(source)
    before = source.read_bytes()
    output, report = prepare_robot_visual_winding(source, tmp_path / "fixed.glb")
    fixed = trimesh.load(output, force="mesh", process=True)
    assert fixed.is_watertight and fixed.is_winding_consistent
    assert fixed.volume == pytest.approx(1)
    assert len(fixed.faces) == len(box.faces)
    np.testing.assert_allclose(fixed.bounds, box.bounds)
    assert source.read_bytes() == before
    assert sum(report["reversed_faces"].values()) == 3


def test_contact_fit_rejects_same_side_pads_despite_split_inward_faces(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace
    import trimesh
    from embodichain.data_pipeline.real2sim.adaptation import _contact_fit

    class CartesianJaws:
        arm_columns = [np.arange(3)]
        arm_limits = [(np.full(3, -1.0), np.full(3, 1.0))]
        names = ["left_jaw", "right_jaw"]
        jaw_columns = [[0, 1]]
        path = tmp_path / "hand.urdf"
        config = {"tcp_links": ["tcp"]}
        prefixes = ["arm"]
        offset = np.eye(4)
        opening_limits = [(0.001, 0.06)]

        def tcp(self, q, side):
            pose = np.repeat(np.eye(4)[None], len(q), axis=0)
            pose[:, :3, 3] = q[:, :3]
            return pose

        def ik(self, source, side, target):
            result = source.copy()
            result[:3] = target[:3, 3]
            return result, {}

        def links(self, q):
            tcp = self.tcp(q, 0)
            left, right = tcp.copy(), tcp.copy()
            left[:, 0, 3] += q[:, 3]
            right[:, 0, 3] -= q[:, 3]
            return dict(tcp=tcp, left=left, right=right)

        def half_open(self, q, side):
            return q[..., 3]

        def set_half_open(self, q, side, opening):
            result = q.copy()
            result[..., 3] = opening
            return result

        def finger_surfaces(self, side):
            return ["left", "right"], [np.array([1, 0, 0]), np.array([-1, 0, 0])]

    # Freeze optimization at a valid bilateral candidate, while examining the
    # source before-diagnostics. This isolates surface orientation from solver
    # convergence without bypassing the full-hand feasibility check.
    monkeypatch.setattr(
        _contact_fit,
        "least_squares",
        lambda fun, x0, **kwargs: SimpleNamespace(
            x=np.array([0.0, 0.0, 0.0, 0.051]) if len(x0) == 4 else np.array(x0),
            cost=0.0,
            success=True,
            nfev=0,
        ),
    )
    box = trimesh.creation.box(extents=[0.1, 0.04, 0.04])
    triangles = box.triangles.copy()
    flipped = np.flatnonzero(box.face_normals[:, 0] < -0.9)[0]
    triangles[flipped] = triangles[flipped, ::-1]
    broken = trimesh.Trimesh(
        vertices=triangles.reshape(-1, 3),
        faces=np.arange(triangles.size // 3).reshape(-1, 3),
        process=False,
    )
    asset = tmp_path / "block.glb"
    broken.export(asset)
    original_bytes = asset.read_bytes()
    finger = tmp_path / "finger.glb"
    trimesh.creation.box(extents=[0.002, 0.04, 0.04]).export(finger)
    table = tmp_path / "table.glb"
    trimesh.creation.box(extents=[1, 1, 0.02]).export(table)
    table_pose = np.eye(4)
    table_pose[2, 3] = -0.2
    target = dict(id="block", path=str(asset), transform=np.eye(4))
    scene = dict(
        objects=[dict(id="table", path=str(table), transform=table_pose), target]
    )
    visuals = tuple(
        dict(link=link, path=str(finger), origin=np.eye(4))
        for link in ("left", "right")
    )
    CartesianJaws.path.write_text(
        '<robot name="cartesian"><link name="tcp"/>'
        + "".join(
            f'<link name="{link}"><collision><geometry><box size="0.002 0.04 0.04"/></geometry></collision></link>'
            f'<joint name="{link}_jaw" type="prismatic"><parent link="tcp"/><child link="{link}"/><axis xyz="1 0 0"/></joint>'
            for link in ("left", "right")
        )
        + "</robot>"
    )
    # Both pads straddle the negative-X face, 100 mm from the opposite face.
    source = np.array([-0.05, 0.0, 0.0, 0.003])
    _, _, report = _contact_fit.fit_grasp(
        scene, visuals, CartesianJaws(), source, 0, target, np.eye(4)
    )
    left, right = report["before"]["fingers"]
    assert left["opposing_pad_gap_m"] > 0.09
    assert right["opposing_pad_gap_m"] < 0.005
    assert asset.read_bytes() == original_bytes


def test_grounded_target_and_handoff_override_source_distance():
    from embodichain.data_pipeline.real2sim.adaptation.refinement import (
        _select_grasp_object,
        _is_handoff,
    )

    objects = {"banana": {}, "bowl": {}}
    distances = {"banana": 0.9, "bowl": 0.01}
    grounding = {"object_id": "banana"}
    assert _select_grasp_object(objects, distances, grounding, []) == "banana"
    assert _is_handoff("banana", grounding, 0.9, [0.01])
    assert not _is_handoff("banana", {"object_id": "bowl"}, 0.001, [0.2])
    assert not _is_handoff("banana", {"object_id": None}, 0.001, [0.2])
    # Missing observations retain the existing geometric association rules.
    assert _select_grasp_object(objects, distances, None, []) == "bowl"
    assert _select_grasp_object(objects, {"banana": 0.9}, None, []) is None
    assert not _is_handoff("banana", None, 0.9, [0.01])
    assert _is_handoff("banana", None, 0.01, [0.2])


def test_grounded_target_still_rejects_missing_or_unsupported_assets():
    from embodichain.data_pipeline.real2sim.adaptation.refinement import (
        _select_grasp_object,
    )

    with pytest.raises(ValueError, match="absent"):
        _select_grasp_object({}, {}, {"object_id": "missing"}, [])
    objects = {"hinge": {"articulated": True}}
    with pytest.raises(ValueError, match="articulated motion is unsupported"):
        _select_grasp_object(
            objects,
            {},
            {"object_id": "hinge", "internal_articulation_motion_visible": True},
            [],
        )
    assert (
        _select_grasp_object(
            objects,
            {"hinge": 0.9},
            {"object_id": "hinge", "internal_articulation_motion_visible": False},
            [],
        )
        == "hinge"
    )


@pytest.mark.parametrize("with_observation", [False, True])
def test_observed_event_delta_reaches_fit_without_changing_source_or_scene(
    monkeypatch, with_observation
):
    from copy import deepcopy
    from types import SimpleNamespace
    from scipy.spatial.transform import Rotation
    from embodichain.data_pipeline.real2sim.adaptation import refinement

    inputs = _inputs()
    scene = {
        "objects": [dict(id="banana", category="banana", transform=np.eye(4).tolist())],
        "source_table_plane": [0, 0, 0],
    }
    vertices = {"banana": np.array([[0.0, 0.0, 0.0], [0.02, 0.0, 0.0]])}
    registered = RegisteredScene(scene, vertices)
    scene_before, q_before, vertices_before = (
        deepcopy(scene),
        inputs.demonstration.qpos.copy(),
        vertices["banana"].copy(),
    )
    event = dict(side=0, close_start=30, grasp=40, release=70, open_end=80)
    robot = SimpleNamespace(
        config={"robot_type": "cobotmagic", "pad_center_tcp_z_m": 0.0},
        tcp=lambda q, side: np.repeat(np.eye(4)[None], len(q), axis=0),
        half_open=lambda q, side: np.zeros(len(q)),
    )
    monkeypatch.setattr(refinement, "Robot", lambda model: robot)
    monkeypatch.setattr(
        refinement, "_demonstration_events", lambda inputs, robot: [event]
    )
    delta = np.eye(4)
    delta[:3, :3] = Rotation.from_euler("x", 25, degrees=True).as_matrix()
    delta[1, 3] = 0.009
    expected = delta if with_observation else np.eye(4)

    class ReachedFit(Exception):
        pass

    def capture_fit(*args, **kwargs):
        np.testing.assert_allclose(args[6], expected)
        np.testing.assert_allclose(args[7]["banana"], expected)
        assert args[0] == scene_before
        raise ReachedFit

    monkeypatch.setattr(refinement, "fit_grasp", capture_fit)
    options = (
        {"grasp_object_deltas": {(0, 40): {"banana": delta}}}
        if with_observation
        else {}
    )
    with pytest.raises(ReachedFit):
        refinement.refine_trajectory(
            inputs,
            registered,
            grounding=({"side": 0, "grasp": 40, "object_id": "banana"},),
            **options,
        )
    np.testing.assert_array_equal(inputs.demonstration.qpos, q_before)
    np.testing.assert_array_equal(vertices["banana"], vertices_before)
    assert registered.description == scene_before
    if with_observation:
        np.testing.assert_allclose(event["observed_object_deltas"]["banana"], delta)
    else:
        assert "observed_object_deltas" not in event


@pytest.mark.parametrize(
    "bad",
    [
        "event",
        "object",
        "scale",
        "reflection",
        "nonfinite",
        "empty_grasp",
        "articulation",
    ],
)
def test_observed_event_delta_rejects_invalid_observation_contract(bad):
    from embodichain.data_pipeline.real2sim.adaptation.refinement import (
        _validated_grasp_object_deltas,
    )

    phases = [dict(side=0, grasp=40)]
    objects = {"banana": {}}
    grounding = {(0, 40): {"object_id": "banana"}}
    delta = np.eye(4)
    key, uid = (0, 40), "banana"
    if bad == "event":
        key = (1, 40)
    elif bad == "object":
        uid = "missing"
    elif bad == "scale":
        delta[0, 0] = 2
    elif bad == "reflection":
        delta[0, 0] = -1
    elif bad == "nonfinite":
        delta[0, 3] = np.nan
    elif bad == "empty_grasp":
        grounding[(0, 40)]["object_id"] = None
    elif bad == "articulation":
        objects[uid]["articulated"] = True
    with pytest.raises(ValueError, match="Object pose observation"):
        _validated_grasp_object_deltas({key: {uid: delta}}, phases, objects, grounding)
