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

"""Actual native hand asset regressions; CPU geometry only, no simulator."""

from __future__ import annotations

_GRIPPER_IDS = {
    "franka_panda_hand": "FrankaPandaHand",
    "dh_pgi_140_80": "DHPGI14080",
    "cobotmagic_v100_gripper": "CobotMagicV100Gripper",
    "pika_gripper": "PikaGripper",
    "tianji_marvin_gripper": "TianjiMarvinGripper",
    "robotiq_2f85": "RobotiqArg2F85",
    "robotiq_arg2f_140": "RobotiqArg2F140",
}

from embodichain.robot_registry import resolve_gripper_defaults


from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
import torch
import trimesh

from embodichain.data import get_data_path

from embodichain.toolkits.graspkit.pg_grasp import (
    GripperCollisionCfg,
    GripperCollisionChecker,
)
from embodichain_benchmarks.motion_generation._atomic_fixtures import (
    create_parallel_jaw_grasp_pose_generator,
)


def test_native_open_hand_mesh_rejects_real_collision_without_false_palm_collision() -> (
    None
):
    native = create_parallel_jaw_grasp_pose_generator(
        n_sample=32,
        force_refresh=False,
        gripper_data=resolve_gripper_defaults("FrankaPandaHand"),
    )
    geometry = native.collision_geometry
    assert geometry.provenance["joint_positions"] == {
        "fr3_finger_joint1": 0.04,
        "fr3_finger_joint2": 0.04,
    }
    assert geometry.provenance["point_spacing_m"] == 0.012
    bounds = {
        part["link"]: np.array(part["bounds_m"])
        for part in geometry.provenance["parts"]
    }
    # Independently read native asset bounds and URDF offsets: TCP z=.1034,
    # finger origins z=.0584, opening +/- .04, and native +Y -> canonical +X.
    np.testing.assert_allclose(
        bounds["fr3_hand"][:, 2],
        [-0.12932480182647704, -0.03743784930706025],
        atol=1e-9,
    )
    np.testing.assert_allclose(
        bounds["fr3_leftfinger"][:, 0],
        [0.03991623569803778, 0.06634548954665662],
        atol=1e-9,
    )
    np.testing.assert_allclose(
        bounds["fr3_rightfinger"][:, 0],
        [-0.06634548954665662, -0.03991623569803778],
        atol=1e-9,
    )
    np.testing.assert_allclose(
        bounds["fr3_leftfinger"][:, 2],
        [-0.04486666800396052, 0.008899880498647685],
        atol=1e-9,
    )
    # Complete palm width must remain represented, not shrink to the old 80x22mm box.
    assert np.diff(bounds["fr3_hand"], axis=0)[0, 0] > 0.2
    assert np.diff(bounds["fr3_hand"], axis=0)[0, 1] > 0.06

    can = trimesh.load(
        get_data_path("SodaCan/simple_cola_can.obj"), force="mesh", process=False
    )
    vertices = np.asarray(can.vertices) * 0.56
    planes = torch.tensor(ConvexHull(vertices).equations, dtype=torch.float32)
    # The can is separated from each complete native mesh by coordinate bounds.
    # This proves triangle separation, independently of finite point sampling.
    assert vertices[:, 2].min() > bounds["fr3_hand"][1, 2]
    assert vertices[:, 0].max() < bounds["fr3_leftfinger"][0, 0]
    assert vertices[:, 0].min() > bounds["fr3_rightfinger"][1, 0]

    def query(batch_points, collision_threshold, **_):
        distance = (batch_points @ planes[:, :3].T + planes[:, 3]).amax(-1)
        return distance <= collision_threshold, distance

    def make_checker(fixed):
        checker = object.__new__(GripperCollisionChecker)
        checker.device = torch.device("cpu")
        checker.obj_mesh_verts = torch.tensor(vertices, dtype=torch.float32)
        checker.cfg = GripperCollisionCfg()
        checker.surface_points = fixed.points
        checker._checker = SimpleNamespace(query_batch_points=query)
        return checker

    poses = torch.eye(4).repeat(2, 1, 1)
    poses[1, 2, 3] = 0.05  # Place the actual hand surface into the can.
    full_contact_width = torch.full((2,), float(np.ptp(vertices[:, 0])))
    new_flags, new_distances = make_checker(geometry).query(
        torch.eye(4), poses, is_filter_ground_collision=False
    )
    assert new_flags.tolist() == [False, True]
    assert new_distances[0] > 0 and new_distances[1] < 0

    # A non-identity common world transform must not apply the TCP conversion twice.
    world = torch.eye(4)
    world[:3, :3] = torch.tensor(
        Rotation.from_euler("xyz", [0.3, -0.4, 0.2]).as_matrix(), dtype=torch.float32
    )
    world[:3, 3] = torch.tensor([0.31, -0.27, 0.64])
    flags, distances = make_checker(geometry).query(
        world, world @ poses, is_filter_ground_collision=False
    )
    torch.testing.assert_close(flags, new_flags)
    torch.testing.assert_close(distances, new_distances, rtol=1e-4, atol=1e-7)

    legacy = create_parallel_jaw_grasp_pose_generator(n_sample=32, force_refresh=False)
    assert legacy.collision_geometry.provenance["parts"]


@pytest.mark.parametrize(
    "model_id, expected_tcp_z, expected_offset",
    [("dh_pgi_140_80", 0.17, 0.02), ("robotiq_arg2f_140", 0.20, 0.0)],
)
def test_benchmark_mounted_hand_uses_bound_solver_tcp(
    model_id: str, expected_tcp_z: float, expected_offset: float
) -> None:
    from embodichain_benchmarks.motion_generation._atomic_fixtures import (
        create_ur5_gripper_robot_cfg,
    )
    from embodichain_benchmarks.motion_generation.scenarios.atomic_skill import (
        AtomicSkillScenario,
    )

    scenario = AtomicSkillScenario()
    if model_id == "dh_pgi_140_80":
        cfg = create_ur5_gripper_robot_cfg()
    else:
        from embodichain_benchmarks.motion_generation import load_suite
        from embodichain_benchmarks.motion_generation.registry import (
            create_robot_provider,
        )

        import embodichain_benchmarks.motion_generation.robots.dual_arm

        cfg = create_robot_provider(
            load_suite("atomic_dual_franka_robotiq_hand_over").robot
        ).build_cfg()
        scenario.control_part = "left_arm"
    scenario.robot = SimpleNamespace(cfg=cfg)
    scenario.grasp_model = lambda: resolve_gripper_defaults(
        _GRIPPER_IDS.get(model_id, model_id),
        variant="left" if model_id == "tianji_marvin_gripper" else None,
    )
    actual_tcp = scenario.grasp_root_from_tcp()
    assert actual_tcp[2][3] == expected_tcp_z
    sampler = create_parallel_jaw_grasp_pose_generator(
        n_sample=32,
        force_refresh=False,
        gripper_data=scenario.grasp_model(),
        root_from_tcp=actual_tcp,
    )
    offset = np.array(sampler.collision_geometry.provenance["grasp_from_tcp"])
    np.testing.assert_allclose(offset[:3, 3], [0, 0, expected_offset], atol=1e-9)
    scenario.grasp_model = lambda: resolve_gripper_defaults("FrankaPandaHand")
    assert scenario.grasp_root_from_tcp() is None  # Native uses its named hand TCP.


def test_grasp_depth_moves_contact_toward_hand_and_rebases_collision_surfaces() -> None:
    """Depth changes physical contact, rather than only relabelling the TCP."""
    tcp = np.eye(4)
    tcp[:3, :3] = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    tcp[2, 3] = 0.20
    arguments = dict(
        n_sample=32,
        force_refresh=False,
        gripper_data=resolve_gripper_defaults("RobotiqArg2F140"),
        root_from_tcp=tcp,
    )
    original = create_parallel_jaw_grasp_pose_generator(**arguments).collision_geometry
    deeper = create_parallel_jaw_grasp_pose_generator(
        **arguments, grasp_depth_offset_m=0.01
    ).collision_geometry
    # Moving the contact origin 1 cm toward the root moves all hand surfaces
    # 1 cm forward in contact coordinates, before candidate collision checks.
    torch.testing.assert_close(
        deeper.points,
        original.points + original.points.new_tensor([0, 0, 0.01]),
        atol=1e-7,
        rtol=0,
    )
    world_contact = torch.eye(4, dtype=torch.float64)
    world_contact[:3, :3] = torch.tensor(
        Rotation.from_euler("xyz", [0.3, -0.4, 0.2]).as_matrix()
    )
    world_contact[:3, 3] = torch.tensor([0.1, -0.2, 0.6])
    for geometry, expected_depth in ((original, 0.20), (deeper, 0.19)):
        world_tcp = geometry.canonical_to_tcp_poses(world_contact)
        world_hand = world_tcp @ torch.tensor(np.linalg.inv(tcp))
        hand_contact = torch.linalg.inv(world_hand) @ world_contact
        torch.testing.assert_close(
            hand_contact[:3, 3],
            torch.tensor([0, 0, expected_depth], dtype=torch.float64),
            atol=1e-9,
            rtol=0,
        )
        np.testing.assert_allclose(
            geometry.provenance["root_from_contact_frame"][2][3], expected_depth
        )


@pytest.mark.parametrize(
    "suite_name",
    [
        "atomic_franka_robotiq_door_pour_align_push",
        "atomic_franka_robotiq_spatial_generalization",
    ],
)
def test_robotiq_push_targets_clear_ground_with_closed_native_geometry(suite_name):
    """Check actual closed meshes, including the smallest authored cube."""
    from copy import deepcopy

    from embodichain.lab.sim.utility.gripper_geometry import (
        build_gripper_collision_geometry,
    )
    from embodichain_benchmarks.motion_generation.config import load_suite
    from embodichain_benchmarks.motion_generation.registry import create_robot_provider
    from embodichain_benchmarks.motion_generation import robots as _robots
    from embodichain_benchmarks.motion_generation.scenarios.atomic_skill_providers import (
        _push_poses,
    )

    suite = load_suite(suite_name)
    track = next(
        track
        for track in suite.tracks
        if any(skill["id"] == "push_object" for skill in track.config["skills"])
    )
    skill = next(
        skill for skill in track.config["skills"] if skill["id"] == "push_object"
    )
    cfg = create_robot_provider(suite.robot).build_cfg()
    model = deepcopy(resolve_gripper_defaults(track.config["grasp_model"]))
    model["joint_commands"]["open"] = dict(
        zip(
            cfg.control_parts["eef"], track.config["gripper"]["grasp_qpos"], strict=True
        )
    )
    geometry = build_gripper_collision_geometry(
        model, root_from_tcp=cfg.solver_cfg["arm"].tcp, point_spacing=0.005
    )
    contact_from_tcp = torch.tensor(
        geometry.provenance["grasp_from_tcp"], dtype=torch.float64
    )
    points = geometry.points.to(torch.float64)
    for size in (0.04, 0.05, 0.06):
        # Conservative support at z=0: no extra native rest-offset clearance.
        object_pose = torch.eye(4)[None]
        object_pose[:, 2, 3] = size / 2
        for case in skill["cases"]:
            target = object_pose.clone()
            target[:, :3, 3] += torch.tensor(case["target_object_offset_m"])
            poses = _push_poses(
                object_pose,
                target,
                contact_distance=case["contact_distance_m"],
                approach_height=skill["approach_height_m"],
                retract_height=skill["retract_height_m"],
                push_overshoot=skill["push_overshoot_m"],
                contact_frame_to_eef=torch.tensor(
                    skill.get("contact_frame_to_eef", torch.eye(4).tolist())
                ),
                object_contact_offset=torch.tensor(skill["object_contact_offset_m"]),
            )
            world_from_contact = poses[0, 1].to(torch.float64) @ torch.linalg.inv(
                contact_from_tcp
            )
            world_points = (
                points @ world_from_contact[:3, :3].T + world_from_contact[:3, 3]
            )
            assert float(world_points[:, 2].min()) >= 0.001
            if size == 0.05:
                # Nominal cube starts about 2 mm ahead of the closed finger face.
                direction = (target[0, :3, 3] - object_pose[0, :3, 3]).to(torch.float64)
                direction = direction / torch.linalg.vector_norm(direction)
                slab = world_points[
                    (world_points[:, 2] >= 0) & (world_points[:, 2] <= size)
                ]
                cube_face = (
                    object_pose[0, :3, 3].to(torch.float64) @ direction - size / 2
                )
                clearance = cube_face - (slab @ direction).amax()
                assert 0.0015 <= float(clearance) <= 0.003


@pytest.mark.parametrize(
    "suite_name",
    [
        "atomic_franka_robotiq_smoke",
        "atomic_franka_robotiq_pose_batch",
        "atomic_franka_robotiq_randomized",
        "atomic_franka_robotiq_spatial_generalization",
    ],
)
def test_robotiq_press_calibrates_closed_pads_to_button_surface(suite_name):
    """Surface contact and the native button stroke must clear the fixed panel."""
    from copy import deepcopy
    from xml.etree import ElementTree

    from embodichain.lab.sim.utility.gripper_geometry import (
        build_gripper_collision_geometry,
    )
    from embodichain_benchmarks.motion_generation.config import load_suite
    from embodichain_benchmarks.motion_generation.registry import create_robot_provider
    from embodichain_benchmarks.motion_generation import robots as _robots

    suite = load_suite(suite_name)
    track = next(
        track
        for track in suite.tracks
        if any(skill["id"] == "press" for skill in track.config["skills"])
    )
    skill = next(skill for skill in track.config["skills"] if skill["id"] == "press")
    cfg = create_robot_provider(suite.robot).build_cfg()
    model = deepcopy(resolve_gripper_defaults(track.config["grasp_model"]))
    model["joint_commands"]["open"] = dict(
        zip(
            cfg.control_parts["eef"], track.config["gripper"]["grasp_qpos"], strict=True
        )
    )
    geometry = build_gripper_collision_geometry(
        model, root_from_tcp=cfg.solver_cfg["arm"].tcp, point_spacing=0.003
    )
    # Meshes are in canonical contact coordinates; read their actual forward
    # extent in solver TCP coordinates, independently of the declared offset.
    contact_from_tcp = np.asarray(geometry.provenance["grasp_from_tcp"])
    tcp_points = (geometry.points.numpy() - contact_from_tcp[:3, 3]) @ contact_from_tcp[
        :3, :3
    ]
    forward_extent = float(tcp_points[:, 2].max())
    assert forward_extent + skill["press_tcp_offset_m"] == pytest.approx(0.0, abs=1e-6)

    microwave = ElementTree.parse(get_data_path("Microwave/microwave.urdf")).getroot()
    button_joint = microwave.find("joint[@name='button_joint']")
    stroke = float(button_joint.find("limit").get("upper"))
    assert skill["press_distance_m"] == pytest.approx(stroke)

    button_collision = microwave.find("link[@name='button_link']/collision")
    button_center = np.fromstring(button_collision.find("origin").get("xyz"), sep=" ")
    button_length = float(button_collision.find("geometry/cylinder").get("length"))
    button_origin = np.fromstring(button_joint.find("origin").get("xyz"), sep=" ")
    button_front = button_origin[1] + button_center[1] - button_length / 2
    panel = microwave.find(
        "link[@name='body_link']/collision[@name='right_electronics']"
    )
    panel_center = np.fromstring(panel.find("origin").get("xyz"), sep=" ")
    panel_size = np.fromstring(panel.find("geometry/box").get("size"), sep=" ")
    panel_front = panel_center[1] - panel_size[1] / 2
    # TCP +Z follows microwave-local +Y. Every closed-hand surface remains
    # outside the fixed panel even at the full six-millimetre button stroke.
    terminal_forward_surface = (
        button_front
        + skill["press_tcp_offset_m"]
        + skill["press_distance_m"]
        + forward_extent
    )
    assert panel_front - terminal_forward_surface > 0.03
