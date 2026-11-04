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

from types import SimpleNamespace

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
import torch
import trimesh

from embodichain.data import get_data_path
from embodichain.toolkits.graspkit import get_parallel_jaw_gripper_model
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
        gripper_model=get_parallel_jaw_gripper_model("franka_panda_hand"),
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


def test_benchmark_pgi_uses_bound_solver_tcp_without_offsetting_native_panda() -> None:
    from embodichain_benchmarks.motion_generation._atomic_fixtures import (
        create_ur5_gripper_robot_cfg,
    )
    from embodichain_benchmarks.motion_generation.scenarios.atomic_skill import (
        AtomicSkillScenario,
    )

    scenario = AtomicSkillScenario()
    scenario.robot = SimpleNamespace(cfg=create_ur5_gripper_robot_cfg())
    scenario.grasp_model = lambda: get_parallel_jaw_gripper_model("dh_pgi_140_80")
    actual_tcp = scenario.grasp_root_from_tcp()
    assert actual_tcp[2][3] == 0.17  # Actual UR5-PGI default, not contact centre .15.
    sampler = create_parallel_jaw_grasp_pose_generator(
        n_sample=32,
        force_refresh=False,
        gripper_model=scenario.grasp_model(),
        root_from_tcp=actual_tcp,
    )
    offset = np.array(sampler.collision_geometry.provenance["grasp_from_tcp"])
    np.testing.assert_allclose(offset[:3, 3], [0, 0, 0.02], atol=1e-9)
    scenario.grasp_model = lambda: get_parallel_jaw_gripper_model("franka_panda_hand")
    assert scenario.grasp_root_from_tcp() is None  # Native uses its named hand TCP.
