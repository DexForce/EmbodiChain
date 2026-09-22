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
"""FEP geometric branch accuracy on the registered Franka arm, without simulation."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import warp as wp
from scipy.spatial.transform import Rotation

from embodichain.data import get_data_path
from embodichain.lab.sim.motion.solvers import FEPSolver, FEPSolverCfg
from embodichain.compute.kinematics._warp.fep import _matches_urdf_fk, _vec7


@wp.kernel
def _verify_fk_residuals(
    qpos: wp.array(dtype=wp.float32),
    targets: wp.array(dtype=wp.mat44f),
    frames: wp.array(dtype=wp.mat44d),
    axes: wp.array(dtype=wp.vec3d),
    valid: wp.array(dtype=wp.bool),
):
    row = wp.tid()
    joints = _vec7()
    for i in range(7):
        joints[i] = wp.float64(qpos[i])
    valid[row] = _matches_urdf_fk(
        joints,
        wp.mat44d(targets[row]),
        frames,
        axes,
        wp.float64(1e-5),
        wp.float64(1e-5),
    )


@pytest.fixture(
    params=[
        "cpu",
        pytest.param(
            "cuda",
            marks=[
                pytest.mark.gpu,
                pytest.mark.skipif(
                    not torch.cuda.is_available(), reason="CUDA unavailable"
                ),
            ],
        ),
    ]
)
def solver(request: pytest.FixtureRequest) -> FEPSolver:
    return FEPSolverCfg(
        urdf_path=get_data_path("Franka/Panda/PandaWithHand.urdf"),
        root_link_name="base",
        end_link_name="fr3_hand_tcp",
        batch_size=16,
    ).init_solver(device=request.param)


def _assert_pose_accuracy(target: torch.Tensor, actual: torch.Tensor) -> None:
    expected = target.cpu().numpy().astype(np.float64)
    result = actual.cpu().numpy().astype(np.float64)
    translation = np.linalg.norm(expected[:, :3, 3] - result[:, :3, 3], axis=-1)
    rotation = (
        Rotation.from_matrix(expected[:, :3, :3]).inv()
        * Rotation.from_matrix(result[:, :3, :3])
    ).magnitude()
    assert np.all(translation <= 1e-5)
    assert np.all(rotation <= 1e-5)


def _assert_limits(solver: FEPSolver, joints: torch.Tensor) -> None:
    assert bool(torch.isfinite(joints).all())
    assert bool((joints >= solver.lower_qpos_limits).all())
    assert bool((joints <= solver.upper_qpos_limits).all())


@pytest.mark.parametrize("with_tcp", [False, True])
def test_franka_distant_seed_roundtrip(solver: FEPSolver, with_tcp: bool) -> None:
    generator = torch.Generator().manual_seed(7)
    lower, upper = solver.lower_qpos_limits, solver.upper_qpos_limits
    qpos = lower + (upper - lower) * (
        0.25 + 0.5 * torch.rand(64, 7, generator=generator).to(solver.device)
    )
    seed = lower + (upper - lower) * torch.rand(64, 7, generator=generator).to(
        solver.device
    )
    # Only the redundant coordinate is shared; the first six joints are far
    # from the solution and must not need a local numerical correction.
    seed[:, 6] = qpos[:, 6]
    if with_tcp:
        tcp = np.eye(4)
        tcp[:3, :3] = Rotation.from_euler("xyz", [0.2, -0.1, 0.3]).as_matrix()
        tcp[:3, 3] = [0.05, -0.03, 0.1]
        solver.set_tcp(tcp)
    target = solver.get_fk(qpos)
    valid, joints = solver.get_ik(target, seed)
    assert bool(valid.all())
    _assert_pose_accuracy(target, solver.get_fk(joints))
    _assert_limits(solver, joints)
    torch.testing.assert_close(joints[:, 6], seed[:, 6], atol=0, rtol=0)
    all_valid, candidates = solver.get_ik(target, seed, return_all_solutions=True)
    assert all_valid.shape == (64, 8)
    assert bool(all_valid.any(dim=-1).all())
    assert int(all_valid.sum(dim=-1).max()) > 1
    _assert_pose_accuracy(
        target[:, None].expand(-1, 8, -1, -1)[all_valid],
        solver.get_fk(candidates[all_valid]),
    )
    torch.testing.assert_close(candidates[:, 0], joints)
    # Every non-singular input configuration must be represented, regardless
    # of which branch is nearest to the unrelated seed.
    distances = (
        (candidates - qpos[:, None])
        .abs()
        .amax(dim=-1)
        .masked_fill(~all_valid, torch.inf)
    )
    # Near singularities float32 target rounding amplifies joint-angle error;
    # pose acceptance above remains 1e-5 m/rad.
    assert bool((distances.min(dim=-1).values < 5e-3).all())


def test_fixed_q7_all_solutions_preserves_branches_when_seed_is_duplicate(
    solver: FEPSolver,
) -> None:
    # Widen the limits for this algorithm fixture so all eight geometric
    # branches are admissible, including both elbow configurations.
    limits = torch.full((7,), torch.pi, device=solver.device)
    solver.set_qpos_limits(-limits, limits)
    seed = limits.new_tensor([[0.2, -0.4, 0.3, -1.5, 0.2, 1.2, 0.5]])
    target = solver.get_fk(seed)
    valid, joints = solver.get_ik(target, seed, return_all_solutions=True)

    assert valid.shape == (1, 8) and bool(valid.all())
    torch.testing.assert_close(joints[:, 0], seed, atol=0, rtol=0)
    _assert_pose_accuracy(target.expand(8, -1, -1), solver.get_fk(joints[0]))
    _assert_limits(solver, joints)
    torch.testing.assert_close(joints[..., 6], seed[:, 6:7].expand(-1, 8))
    distances = (joints[0, :, None] - joints[0, None, :]).abs().amax(-1)
    distances.fill_diagonal_(torch.inf)
    assert float(distances.min()) > 1e-6
    costs = (joints - seed[:, None]).square().sum(-1)
    assert bool((costs[:, 1:] >= costs[:, :-1]).all())


def test_franka_previous_solution_tracks_motion(solver: FEPSolver) -> None:
    midpoint = solver.get_default_qpos_seed()
    midpoint[1] = 0.5  # A continuous branch away from the shoulder singularity.
    phase = torch.linspace(0, 2 * torch.pi, 65, device=solver.device)[:, None]
    reference = midpoint + 0.15 * torch.sin(
        phase + torch.arange(7, device=solver.device) * 0.35
    )
    reference[:, 6] = midpoint[6]  # Exercise a fixed-redundancy path.
    targets = solver.get_fk(reference)
    previous = reference[:1]
    for target in targets:
        valid, joints = solver.get_ik(target, previous)
        assert bool(valid.all())
        _assert_pose_accuracy(target[None], solver.get_fk(joints))
        _assert_limits(solver, joints)
        assert float((joints - previous).abs().max()) < 0.1
        previous = joints


@pytest.mark.parametrize("search", [False, True])
def test_manipulability_ranks_valid_retained_candidates(
    solver: FEPSolver, search: bool
) -> None:
    from embodichain.compute.kinematics import yoshikawa_manipulability

    solver.cfg.redundancy_search = search
    solver.cfg.batch_size = 2
    solver.cfg.max_joint_step = 0.1 if search else None
    # Admit both elbow configurations and start on a less manipulable branch.
    limits = torch.full((7,), torch.pi, device=solver.device)
    solver.set_qpos_limits(-limits, limits)
    seed = limits.new_tensor([[0.9, 0.75, 0.25, 0.6, -0.5, 0.3, 0.5]]).repeat(3, 1)
    target = solver.get_fk(seed)
    target[-1, 0, 3] += 10
    # A live nonuniform weight remains relevant to search-pool construction.
    solver.set_ik_nearest_weight(np.array([2.0, 0.2, 1.0, 0.5, 3.0, 0.7, 1.0]))
    valid, candidates = solver.get_ik(target, seed, return_all_solutions=True)
    scores = candidates.new_full(valid.shape, -torch.inf)
    scores[valid] = yoshikawa_manipulability(solver.get_jacobian(candidates[valid]))
    assert bool((scores[:2].amax(1) > scores[:2, 0] + 1e-6).all())

    solver.cfg.ik_solution_selection = "manipulability"
    success, result = solver.get_ik(target, seed)
    assert success.tolist() == [True, True, False]
    assert result.shape == (3, 7)
    selected_scores = yoshikawa_manipulability(solver.get_jacobian(result[success]))
    maximum = scores[success].amax(1)
    assert bool((selected_scores >= maximum - (1e-8 + 1e-5 * maximum.abs())).all())
    tied = valid & (
        scores
        >= scores.amax(1, keepdim=True)
        - (1e-8 + 1e-5 * scores.amax(1, keepdim=True).abs())
    )
    distances = (
        (candidates - seed[:, None]).square()
        * torch.as_tensor(solver.ik_nearest_weight, device=solver.device)[None, None]
    ).sum(-1)
    expected = candidates[
        torch.arange(len(seed), device=solver.device),
        distances.masked_fill(~tied, torch.inf).argmin(1),
    ]
    torch.testing.assert_close(result, expected)
    _assert_pose_accuracy(target[success], solver.get_fk(result[success]))
    _assert_limits(solver, result)
    if search:
        assert float((result[success] - seed[success]).abs().max()) <= 0.1
    torch.testing.assert_close(result[-1], seed[-1], atol=0, rtol=0)
    all_valid, all_joints = solver.get_ik(target, seed, return_all_solutions=True)
    torch.testing.assert_close(all_valid, valid)
    torch.testing.assert_close(all_joints, candidates)
    empty_valid, empty_joints = solver.get_ik(target[:0], seed[:0])
    assert empty_valid.shape == (0,) and empty_joints.shape == (0, 7)


def test_manipulability_ties_preserve_sequential_circle_continuity(
    solver: FEPSolver,
) -> None:
    seed = solver.get_default_qpos_seed()[None]
    seed[:, 1] = -0.4
    home = solver.get_fk(seed)
    angle = torch.linspace(0, 0.7, 40, device=solver.device)
    targets = home.repeat(len(angle), 1, 1)
    targets[:, 0, 3] += 0.12 * (angle.cos() - 1)
    targets[:, 1, 3] += 0.12 * angle.sin()
    solver.cfg.ik_solution_selection = "manipulability"

    maximum_step = 0.0
    for target in targets:
        valid, joints = solver.get_ik(target[None], seed)
        assert bool(valid.all())
        maximum_step = max(maximum_step, float((joints - seed).abs().max()))
        seed = joints

    assert maximum_step < 0.05


def test_fep_rejects_numerical_sampling_and_unknown_selection(
    solver: FEPSolver,
) -> None:
    from embodichain.lab.sim.motion.solvers import SolverCfg

    for fields, match in (
        ({"num_samples": 30}, "num_samples.*redundancy_search"),
        ({"ik_solution_selection": "unknown"}, "ik_solution_selection"),
    ):
        with pytest.raises(ValueError, match=match):
            FEPSolverCfg(**fields)
        with pytest.raises(ValueError, match=match):
            SolverCfg.from_dict({"class_type": "FEPSolver", **fields})
    seed = solver.get_default_qpos_seed()[None]
    with pytest.raises(ValueError, match="num_samples.*redundancy_search"):
        solver.get_ik(solver.get_fk(seed), seed, num_samples=30)


@pytest.mark.parametrize("selection", ["nearest", "manipulability"])
def test_franka_robot_ik_shapes_frames_and_limit_sync(
    solver: FEPSolver, selection: str
) -> None:
    from types import SimpleNamespace

    from embodichain.lab.sim.objects.robot import Robot
    from embodichain.lab.sim.robots import FrankaPandaCfg

    cfg = FrankaPandaCfg.from_dict({})
    assert cfg.solver_cfg["arm"].class_type == "PytorchSolver"
    assert cfg.solver_cfg["arm"].num_samples == 30
    cfg.solver_cfg["arm"] = FEPSolverCfg(
        urdf_path=solver.urdf_path,
        root_link_name="base",
        end_link_name="fr3_hand_tcp",
        redundancy_search=True,
        ik_solution_selection=selection,
    )
    # Exercise Robot's real binding, frame conversion and limit synchronization;
    # only the physics-owned poses/limits are supplied without a live simulator.
    robot = object.__new__(Robot)
    robot.cfg, robot.device = cfg, solver.device
    robot._all_indices = list(range(8))
    robot._joint_ids = {"arm": list(range(7))}
    robot._solvers = {}
    limits = torch.stack((solver.lower_qpos_limits, solver.upper_qpos_limits), -1)
    robot._data = SimpleNamespace(qpos_limits=limits[None].clone())
    robot.init_solver(cfg.solver_cfg)
    solver = robot.get_solver("arm")
    generator = torch.Generator().manual_seed(23)
    source = solver.lower_qpos_limits + (
        solver.upper_qpos_limits - solver.lower_qpos_limits
    ) * (0.3 + 0.4 * torch.rand(8, 7, generator=generator).to(solver.device))
    source[-3:-1] = solver.get_default_qpos_seed()
    source[-3, 1] = 0  # Shoulder singularity.
    seed = source.clone()
    seed[-2, 6] = 2  # Target has no solution at this seed q7.
    target = solver.get_fk(source)
    target[-1, 0, 3] += 10
    seed[-1, 0] = solver.upper_qpos_limits[0] + 1
    base = torch.eye(4, device=solver.device).repeat(8, 1, 1)
    base[:, :3, :3] = base.new_tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    base[:, :3, 3] = base.new_tensor([0.3, -0.2, 0.1])
    robot.get_link_pose = lambda **kwargs: base[kwargs["env_ids"]]
    success, joints = robot.compute_ik(base @ target, seed, name="arm")
    assert success.tolist() == [True] * 7 + [False]
    assert joints.shape == (8, 7)
    _assert_pose_accuracy(target[success], solver.get_fk(joints[success]))
    _assert_limits(solver, joints)
    torch.testing.assert_close(joints[-1], seed[-1].clamp(limits[:, 0], limits[:, 1]))
    batch_valid, batch_joints = robot.compute_batch_ik(
        (base @ target)[:, None], seed[:, None], name="arm"
    )
    assert batch_valid.shape == (8, 1) and batch_joints.shape == (8, 1, 7)
    torch.testing.assert_close(batch_valid[:, 0], success)
    torch.testing.assert_close(batch_joints[:, 0], joints)
    robot._data.qpos_limits[0] = source[0, :, None].expand(-1, 2)
    robot._sync_solver_limits("arm")
    torch.testing.assert_close(solver.lower_qpos_limits, source[0])
    torch.testing.assert_close(solver.upper_qpos_limits, source[0])
    success, joints = robot.compute_ik(
        (base @ target)[:1], seed[:1], name="arm", env_ids=[0]
    )
    assert success.shape == (1,) and joints.shape == (1, 7) and bool(success.all())
    torch.testing.assert_close(joints, source[:1], atol=0, rtol=0)


def test_franka_seedless_solve_starts_at_feasible_midpoint(solver: FEPSolver) -> None:
    midpoint = solver.get_default_qpos_seed()[None]
    valid, result = solver.get_ik(solver.get_fk(midpoint), return_all_solutions=True)
    assert valid.shape == (1, 8) and bool(valid[:, 0].all())
    torch.testing.assert_close(result[:, 0], midpoint, atol=1e-5, rtol=0)
    _assert_pose_accuracy(solver.get_fk(midpoint), solver.get_fk(result[:, 0]))
    _assert_limits(solver, result[:, 0])


def test_franka_unreachable_targets_and_joint_limits(solver: FEPSolver) -> None:
    seed = solver.get_default_qpos_seed()[None].repeat(2, 1)
    target = solver.get_fk(seed)
    target[1, 0, 3] += 10.0
    valid, joints = solver.get_ik(target, seed)
    assert valid.tolist() == [True, False]
    _assert_pose_accuracy(target[:1], solver.get_fk(joints[:1]))
    torch.testing.assert_close(joints[1], seed[1], atol=0, rtol=0)

    target = solver.get_fk(seed + 0.1)
    solver.set_qpos_limits(seed[0], seed[0])
    valid, joints = solver.get_ik(target, seed + 0.1)
    assert not bool(valid.any())
    torch.testing.assert_close(joints, seed, atol=0, rtol=0)


@pytest.mark.parametrize("q2", [0.0, 1e-7, -1e-7])
def test_shoulder_singularity_uses_seed_representative(
    solver: FEPSolver, q2: float
) -> None:
    qpos = solver.get_default_qpos_seed()[None]
    qpos[:, 0], qpos[:, 1], qpos[:, 2] = 0.7, q2, -0.3
    seed = qpos.clone()
    seed[:, 0] += 0.4
    valid, joints = solver.get_ik(solver.get_fk(qpos), seed)
    assert bool(valid.all())
    _assert_pose_accuracy(solver.get_fk(qpos), solver.get_fk(joints))
    assert abs(float(joints[0, 0] - seed[0, 0])) < 1e-5


@pytest.mark.parametrize("search", [False, True])
@pytest.mark.parametrize("all_solutions", [False, True])
def test_near_extension_retains_a_feasible_seed(
    solver: FEPSolver, search: bool, all_solutions: bool
) -> None:
    # CPU regressions from near-extension FK: the rounded triangle cosine
    # and plane/circle ratio exceeded the old dimensionless rejection bounds.
    seed = torch.tensor(
        [
            [
                1.49994659,
                0.0351877213,
                0.727172375,
                -0.467002422,
                1.36514664,
                2.10143089,
                -0.356151581,
            ],
            [
                1.49269152,
                -0.410849571,
                -0.692257404,
                -0.468002439,
                1.52222824,
                1.86559343,
                1.05568433,
            ],
        ],
        device=solver.device,
    )
    target = solver.get_fk(seed)
    solver.cfg.redundancy_search = search
    solver.cfg.max_joint_step = 0.04 if search else None
    solver.cfg.arm_angle_weight = solver.cfg.joint_limit_weight = 0.0
    valid, joints = solver.get_ik(target, seed, all_solutions)
    if all_solutions:
        _assert_pose_accuracy(
            target[:, None].expand(-1, 8, -1, -1)[valid],
            solver.get_fk(joints[valid]),
        )
        valid, joints = valid[:, 0], joints[:, 0]
    assert bool(valid.all())
    torch.testing.assert_close(joints, seed, atol=0, rtol=0)
    _assert_pose_accuracy(target, solver.get_fk(joints))
    _assert_limits(solver, joints)

    # The triangle-rounding case must also reconstruct geometric branches
    # when the seed itself does not meet the target pose.
    perturbed = seed[:1].clone()
    perturbed[:, 0] += 0.01
    valid, joints = solver.get_ik(target[:1], perturbed)
    assert bool(valid.all())
    assert float((joints - perturbed).abs().max()) > 1e-3
    _assert_pose_accuracy(target[:1], solver.get_fk(joints))
    _assert_limits(solver, joints)


@pytest.mark.parametrize("search", [False, True])
def test_solver_is_independent_of_default_float_dtype(
    solver: FEPSolver, search: bool
) -> None:
    solver.cfg.redundancy_search = search
    seed = solver.get_default_qpos_seed()[None]
    seed[:, 1] = -0.4
    target = solver.get_fk(seed + 0.1)
    expected_angles = solver.get_arm_angle(seed)
    expected_valid, expected_joints = solver.get_ik(target, seed, True)
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        angles = solver.get_arm_angle(seed)
        valid, joints = solver.get_ik(target, seed, True)
        if search:
            # Exercise both the implicit seed angle and an explicit reference.
            valid2, joints2 = solver.get_ik(target, seed, True, arm_angle=angles)
            torch.testing.assert_close(valid2, valid, atol=0, rtol=0)
            torch.testing.assert_close(joints2, joints, atol=0, rtol=0)
    finally:
        torch.set_default_dtype(previous_dtype)
    assert joints.dtype == angles.dtype == torch.float32
    torch.testing.assert_close(angles, expected_angles, atol=0, rtol=0)
    torch.testing.assert_close(valid, expected_valid, atol=0, rtol=0)
    torch.testing.assert_close(joints, expected_joints, atol=0, rtol=0)


def test_geometry_rejects_incompatible_seven_axis_chain(solver: FEPSolver) -> None:
    import copy

    chain = copy.deepcopy(solver.pk_serial_chain)
    frames = [f for f in chain._serial_frames if f.joint.joint_type == "revolute"]
    frames[3].joint.axis = frames[3].joint.axis.new_tensor([1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="Franka-compatible"):
        solver.cfg.init_solver(device=solver.device, pk_serial_chain=chain)


def test_geometry_is_extracted_from_modified_urdf(tmp_path) -> None:
    import xml.etree.ElementTree as ET

    tree = ET.parse(get_data_path("Franka/Panda/PandaWithHand.urdf"))
    root = tree.getroot()
    root.find("joint[@name='fr3_joint1']/origin").set("rpy", "0.2 -0.3 0.4")
    root.find("joint[@name='fr3_joint1']/origin").set("xyz", "0.1 -0.2 0.4")
    root.find("joint[@name='fr3_joint4']/origin").set("xyz", "0.09 0 0")
    root.find("joint[@name='fr3_joint3']/axis").set("xyz", "0 0 -1")
    path = tmp_path / "modified_franka.urdf"
    tree.write(path)
    solver = FEPSolverCfg(
        urdf_path=str(path), root_link_name="base", end_link_name="fr3_hand_tcp"
    ).init_solver("cpu")
    qpos = solver.get_default_qpos_seed()[None].repeat(3, 1)
    qpos[:, :3] += torch.tensor([[0.2, 0.3, -0.2], [-0.5, -0.4, 0.6], [1.0, 0.8, -0.7]])
    seed = solver.get_default_qpos_seed()[None].expand(3, -1)
    valid, joints = solver.get_ik(solver.get_fk(qpos), seed)
    assert bool(valid.all())
    _assert_pose_accuracy(solver.get_fk(qpos), solver.get_fk(joints))


def test_nonrigid_targets_are_rejected(solver: FEPSolver) -> None:
    target = torch.eye(4, device=solver.device)
    target[0, 0] = -1
    with pytest.raises(ValueError, match="rigid"):
        solver.get_ik(target)


def test_failure_at_fixed_q7_does_not_change_redundancy(solver: FEPSolver) -> None:
    feasible_seed = solver.get_default_qpos_seed()[None]
    target = solver.get_fk(feasible_seed)
    seed = feasible_seed.clone()
    seed[:, 6] = 2.0
    valid, joints = solver.get_ik(target, seed)
    assert not bool(valid.any())
    torch.testing.assert_close(joints, seed, atol=0, rtol=0)
    valid, joints = solver.get_ik(target, feasible_seed)
    assert bool(valid.all())
    _assert_pose_accuracy(target, solver.get_fk(joints))


def test_arm_angle_matches_urdf_planes_and_ignores_tcp(solver: FEPSolver) -> None:
    generator = torch.Generator().manual_seed(19)
    qpos = solver.lower_qpos_limits + (
        solver.upper_qpos_limits - solver.lower_qpos_limits
    ) * (0.2 + 0.6 * torch.rand(24, 7, generator=generator).to(solver.device))
    transforms = solver.pk_serial_chain.forward_kinematics(qpos, end_only=False)
    shoulder, elbow, wrist = [
        transforms[name].get_matrix().cpu().numpy().astype(float)
        for name in ("fr3_link1", "fr3_link4", "fr3_link7")
    ]
    # Independent measurement from link origins and the joint-4 axis in URDF FK.
    base_rotation = solver._model_frames[0, :3, :3]
    p = (wrist[:, :3, 3] - shoulder[:, :3, 3]) @ base_rotation.T
    e = (elbow[:, :3, 3] - shoulder[:, :3, 3]) @ base_rotation.T
    normal = np.cross(p, e)
    axis4 = (elbow[:, :3, :3] @ solver._local_axes[3].cpu().numpy()) @ base_rotation.T
    axis4 *= float(solver._scales[3])
    normal *= np.where(np.sum(normal * axis4, axis=-1) < 0, -1, 1)[:, None]
    reference = np.column_stack((p[:, 1], -p[:, 0], np.zeros(len(p))))
    expected = np.arctan2(
        np.sum(
            p / np.linalg.norm(p, axis=-1)[:, None] * np.cross(reference, normal),
            axis=-1,
        ),
        np.sum(reference * normal, axis=-1),
    )
    actual = solver.get_arm_angle(qpos)
    np.testing.assert_allclose(actual.cpu(), expected, atol=2e-6, rtol=0)
    qpos[:, 6] += 0.5  # q7 rotates the flange, not the arm plane.
    tcp = np.eye(4)
    tcp[:3, 3] = [0.1, -0.05, 0.2]
    solver.set_tcp(tcp)
    torch.testing.assert_close(solver.get_arm_angle(qpos), actual, atol=0, rtol=0)


def test_redundancy_search_recovers_unavailable_seed_q7(solver: FEPSolver) -> None:
    seed = solver.get_default_qpos_seed()[None]
    target = solver.get_fk(seed)
    seed[:, 6] = 2.0
    assert not bool(solver.get_ik(target, seed)[0].any())
    solver.cfg.redundancy_search = True
    valid, joints = solver.get_ik(target, seed)
    assert bool(valid.all())
    assert abs(float(joints[0, 6] - seed[0, 6])) > 0.2
    _assert_pose_accuracy(target, solver.get_fk(joints))
    _assert_limits(solver, joints)


def test_dynamic_arm_preference_moves_redundancy_with_bounded_steps(
    solver: FEPSolver,
) -> None:
    seed = solver.get_default_qpos_seed()[None]
    seed[:, 1] = -0.4
    target = solver.get_fk(seed)
    other_seed = seed.clone()
    other_seed[:, 6] += 0.3
    valid, other = solver.get_ik(target, other_seed)
    assert bool(valid.all())
    first_angle = solver.get_arm_angle(seed)
    last_angle = solver.get_arm_angle(other)
    difference = torch.atan2(
        (last_angle - first_angle).sin(), (last_angle - first_angle).cos()
    )
    assert float(difference.abs()) > 0.05
    solver.cfg.redundancy_search = True
    solver.cfg.max_joint_step = 0.04
    solver.cfg.arm_angle_weight = 5.0
    original = seed.clone()
    for fraction in torch.linspace(0, 1, 25, device=solver.device):
        preferred = first_angle + fraction * difference
        valid, joints = solver.get_ik(target, seed, arm_angle=preferred)
        assert bool(valid.all())
        assert float((joints - seed).abs().max()) <= solver.cfg.max_joint_step
        _assert_pose_accuracy(target, solver.get_fk(joints))
        _assert_limits(solver, joints)
        seed = joints
    residual = solver.get_arm_angle(seed) - last_angle
    residual = torch.atan2(residual.sin(), residual.cos()).abs()
    assert float(residual) < 0.25 * float(difference.abs())
    assert float((seed[:, 6] - original[:, 6]).abs()) > 0.05


def test_search_never_relaxes_joint_step_to_recover_a_target(solver: FEPSolver) -> None:
    seed = solver.get_default_qpos_seed()[None]
    seed[:, 1] = -0.4
    target = solver.get_fk(seed + 0.3)
    solver.cfg.redundancy_search = True
    solver.cfg.max_joint_step = 0.001
    valid, joints = solver.get_ik(target, seed)
    assert not bool(valid.any())
    torch.testing.assert_close(joints, seed, atol=0, rtol=0)


@pytest.mark.parametrize(
    "source,seed,max_step",
    [
        pytest.param(
            [
                0.33872294,
                -0.37048233,
                -0.73394728,
                -0.43634129,
                -2.79588604,
                1.81237090,
                -1.55357599,
            ],
            [
                0.21888685,
                -0.83114201,
                -1.20652080,
                -0.73283648,
                2.25695515,
                2.39923143,
                -0.78844237,
            ],
            None,
            id="between_global_probes",
        ),
        pytest.param(
            [
                -0.14697027,
                0.31347203,
                -2.89489865,
                -0.49433589,
                -0.11459351,
                1.74433935,
                -1.92992818,
            ],
            [
                1.86871433,
                0.78295660,
                -2.13244963,
                -2.87968063,
                0.70322752,
                0.97309530,
                1.66412330,
            ],
            None,
            id="near_joint_limit",
        ),
        pytest.param(
            [
                1.64923573,
                1.63284898,
                -2.70391393,
                -0.44597244,
                -0.00169277,
                2.04031634,
                -2.79520035,
            ],
            [
                -0.21390224,
                -0.66291332,
                0.56566668,
                -1.45624304,
                -2.62791348,
                2.98212957,
                2.14443421,
            ],
            None,
            id="no_geometric_guide_on_coarse_grid",
        ),
        pytest.param(
            [
                -1.70282781,
                1.78013253,
                -0.07427406,
                -0.48466778,
                2.68795776,
                1.40875340,
                0.10276794,
            ],
            [
                -1.36928284,
                -0.53461707,
                2.22791147,
                -1.90276706,
                1.44888592,
                3.96429300,
                -2.06968570,
            ],
            None,
            id="thin_boundary_interval_without_coarse_guide",
        ),
        pytest.param(
            [
                2.51740360,
                -0.0009034872,
                0.23464584,
                -2.98413491,
                1.48044109,
                4.15530872,
                1.99430156,
            ],
            [
                2.48794341,
                0.01642403,
                0.25525361,
                -2.98405194,
                1.50051355,
                4.13571644,
                1.99149823,
            ],
            0.04,
            id="thin_step_interval_near_shoulder_singularity",
        ),
    ],
)
def test_adaptive_search_recovers_thin_feasible_intervals(
    solver: FEPSolver, source: list[float], seed: list[float], max_step: float | None
) -> None:
    # Reachable regressions from full-range sampling (generator seed 2026).
    # The source q7 is deliberately withheld from the solver.
    reference = torch.tensor([source], device=solver.device)
    previous = torch.tensor([seed], device=solver.device)
    target = solver.get_fk(reference)
    solver.cfg.redundancy_search = True
    solver.cfg.max_joint_step = max_step
    valid, joints = solver.get_ik(target, previous)
    assert bool(valid.all())
    _assert_pose_accuracy(target, solver.get_fk(joints))
    _assert_limits(solver, joints)
    if max_step is not None:
        assert float((joints - previous).abs().max()) <= max_step


@pytest.mark.parametrize("all_solutions", [False, True])
@pytest.mark.parametrize("max_step", [None, 0.04])
def test_search_pruning_preserves_ranked_solutions(
    solver: FEPSolver,
    monkeypatch: pytest.MonkeyPatch,
    all_solutions: bool,
    max_step: float | None,
) -> None:
    generator = torch.Generator().manual_seed(19)
    lower, upper = solver.lower_qpos_limits, solver.upper_qpos_limits
    source = lower + (upper - lower) * (
        0.2 + 0.6 * torch.rand(16, 7, generator=generator).to(solver.device)
    )
    seed = lower + (upper - lower) * torch.rand(16, 7, generator=generator).to(
        solver.device
    )
    if max_step is not None:
        seed = source + 0.02 * (
            torch.rand(16, 7, generator=generator).to(solver.device) - 0.5
        )
    solver.cfg.redundancy_search = True
    solver.cfg.max_joint_step = max_step
    # Include zero-weight coordinates: raw distance is not a valid bound.
    solver.set_ik_nearest_weight(np.array([0.0, 0.1, 5.0, 0.0, 8.0, 0.3, 2.0]))
    target = solver.get_fk(source)
    valid, result = solver.get_ik(target, seed, all_solutions)
    generate = solver._generate_branches

    def unpruned(*args, **kwargs):
        kwargs["score_bounds"] = None
        return generate(*args, **kwargs)

    monkeypatch.setattr(solver, "_generate_branches", unpruned)
    expected_valid, expected = solver.get_ik(target, seed, all_solutions)
    assert bool(expected_valid.any())
    torch.testing.assert_close(valid, expected_valid, atol=0, rtol=0)
    torch.testing.assert_close(result, expected, atol=0, rtol=0)


def test_near_limit_guidance_is_never_returned_as_a_feasible_solution(
    solver: FEPSolver,
) -> None:
    source = solver.get_default_qpos_seed()[None]
    source[:, 1] = -0.4
    target = solver.get_fk(source)
    lower, upper = solver.lower_qpos_limits.clone(), solver.upper_qpos_limits.clone()
    # Keep q7 fixed and place the target's unique in-range shoulder branch
    # 0.01 rad beyond joint 2's upper bound, within the guidance allowance.
    lower[6] = upper[6] = source[0, 6]
    upper[1] = -0.41
    solver.set_qpos_limits(lower, upper)
    seed = source.clone()
    seed[:, 1] = -0.45
    solver.cfg.redundancy_search = True
    valid, joints = solver.get_ik(target, seed, return_all_solutions=True)
    assert not bool(valid.any())
    torch.testing.assert_close(joints, seed[:, None].expand(-1, 8, -1), atol=0, rtol=0)


@pytest.mark.parametrize("count", [1, 8])
def test_search_merges_candidates_with_stable_ties_and_guidance(
    solver: FEPSolver, count: int
) -> None:
    from embodichain.compute.kinematics._warp.fep import select_redundancy

    device = solver.device
    lower, upper = solver.lower_qpos_limits, solver.upper_qpos_limits
    seed = solver.get_default_qpos_seed()[None].repeat(4, 1)
    seed[3, 0] = upper[0] + 0.1  # All-invalid rows must return the clamped seed.
    rows = torch.tensor([3, 1], device=device)
    old_joints = (
        seed[:, None] + 0.01 * torch.arange(count, device=device)[None, :, None]
    )
    old_costs = (
        2.0 + 2 * torch.arange(count, device=device, dtype=torch.float64)
    ).repeat(4, 1)
    old_costs[3] = torch.inf
    # Duplicates within new candidates and across the old/new sets; cost 4
    # ties an old solution, which must retain priority over new candidates.
    candidates = (
        seed[rows, None] + 0.01 * seed.new_tensor([-2, 0, -2, 3, 4, 5])[None, :, None]
    )
    costs = torch.tensor(
        [[torch.inf] * 6, [3, 2, 1, torch.inf, 4, 4]],
        device=device,
        dtype=torch.float64,
    )
    guide_costs = costs.new_tensor([[0.01, 2, 1, 3, 4, 5], [2, 0.1, 0.1, 3, 4, 5]])
    expected_valid = torch.isfinite(old_costs)
    expected_joints, expected_costs = old_joints.clone(), old_costs.clone()
    # Explicit ranking: the cheaper -2 duplicate wins, then old solutions
    # precede equal-cost new ones; rows 0 and 2 must remain untouched.
    expected_joints[1] = (
        seed[1] + 0.01 * seed.new_tensor([-2, 0, 1, 4, 5, 2, 3, 6][:count])[:, None]
    )
    expected_costs[1] = costs.new_tensor([1, 2, 4, 4, 4, 6, 8, 14][:count])
    expected_joints[3] = seed[3].clamp(lower, upper)
    stream = (
        wp.stream_from_torch(torch.cuda.current_stream(device))
        if device.type == "cuda"
        else None
    )
    for guidance in (False, True):
        joints, scores = old_joints.clone(), old_costs.clone()
        valid = torch.isfinite(scores)
        centers = seed[:, 6].clone()
        guide_scores = torch.full((4,), 0.2, device=device, dtype=torch.float64)
        expected_centers, expected_guides = centers.clone(), guide_scores.clone()
        if guidance:
            expected_centers[rows] = candidates[[0, 1], [0, 1], 6]
            expected_guides[rows] = guide_scores.new_tensor([0.01, 0.1])
        wp.launch(
            select_redundancy,
            dim=len(rows),
            inputs=[
                wp.from_torch(rows),
                wp.from_torch(seed),
                wp.from_torch(lower),
                wp.from_torch(upper),
                wp.from_torch(candidates),
                wp.from_torch(costs.clone()),
                wp.from_torch(guide_costs),
                guidance,
            ],
            outputs=[
                wp.from_torch(valid),
                wp.from_torch(joints),
                wp.from_torch(scores),
                wp.from_torch(centers),
                wp.from_torch(guide_scores),
            ],
            device=str(device),
            stream=stream,
        )
        for result, expected in (
            (valid, expected_valid),
            (joints, expected_joints),
            (scores, expected_costs),
            (centers, expected_centers),
            (guide_scores, expected_guides),
        ):
            torch.testing.assert_close(result, expected, atol=0, rtol=0)


def test_limit_preference_moves_away_from_a_joint_boundary(solver: FEPSolver) -> None:
    seed = solver.get_default_qpos_seed()[None]
    seed[:, 1] = -0.4
    seed[:, 6] = solver.upper_qpos_limits[6]
    target = solver.get_fk(seed)
    solver.cfg.redundancy_search = True
    solver.cfg.arm_angle_weight = 0
    solver.cfg.joint_limit_weight = 0
    valid, unchanged = solver.get_ik(target, seed)
    assert bool(valid.all())
    torch.testing.assert_close(unchanged, seed, atol=1e-5, rtol=0)
    solver.cfg.joint_limit_weight = 0.01
    valid, improved = solver.get_ik(target, seed)
    assert bool(valid.all())
    width = solver.upper_qpos_limits - solver.lower_qpos_limits
    margin = (
        torch.minimum(
            improved - solver.lower_qpos_limits, solver.upper_qpos_limits - improved
        )
        / width
    )
    assert float(margin.min()) > 0.01
    _assert_pose_accuracy(target, solver.get_fk(improved))


def test_search_chunking_preserves_results_and_prior_outputs(solver: FEPSolver) -> None:
    solver.cfg.redundancy_search = True
    seed = solver.get_default_qpos_seed()[None].repeat(5, 1)
    seed[:, 1] = -0.4
    target = solver.get_fk(seed + 0.1)
    valid, joints = solver.get_ik(target, seed)
    saved = joints.clone()
    solver.cfg.batch_size = 2
    valid2, joints2 = solver.get_ik(target, seed)
    torch.testing.assert_close(valid, valid2)
    torch.testing.assert_close(joints, joints2)
    solver.get_ik(target, seed, arm_angle=0.8)
    torch.testing.assert_close(joints, saved, atol=0, rtol=0)


@pytest.mark.parametrize("all_solutions", [False, True])
def test_search_with_locked_q7_matches_fixed_geometric_solutions(
    solver: FEPSolver, all_solutions: bool
) -> None:
    generator = torch.Generator().manual_seed(23)
    lower, upper = solver.lower_qpos_limits.clone(), solver.upper_qpos_limits.clone()
    source = lower + (upper - lower) * (
        0.25 + 0.5 * torch.rand(12, 7, generator=generator).to(solver.device)
    )
    seed = lower + (upper - lower) * torch.rand(12, 7, generator=generator).to(
        solver.device
    )
    lower[6] = upper[6] = source[:, 6] = seed[:, 6] = 0.0
    solver.set_qpos_limits(lower, upper)
    target = solver.get_fk(source)
    target[-1, 0, 3] += 10  # Include a failed row and its clamped-seed fallback.
    expected_valid, expected = solver.get_ik(target, seed, all_solutions)
    assert bool(expected_valid[:-1].any()) and not bool(expected_valid[-1].any())
    # All clipped q7 probes coincide. With the same distance-only score,
    # searching must retain exactly the fixed-q7 branches and their order.
    solver.cfg.redundancy_search = True
    solver.cfg.arm_angle_weight = solver.cfg.joint_limit_weight = 0.0
    valid, result = solver.get_ik(target, seed, all_solutions)
    torch.testing.assert_close(valid, expected_valid, atol=0, rtol=0)
    torch.testing.assert_close(result, expected, atol=0, rtol=0)


def test_arm_angle_reports_an_undefined_reference_plane(solver: FEPSolver) -> None:
    joints = torch.zeros(1, 7, device=solver.device)
    # Rotate the zero-pose shoulder-to-joint-7 vector onto the vertical axis.
    d3, d5, _, _, a7 = solver._dimensions[:5].tolist()
    joints[:, 1] = -np.arctan2(a7, d3 + d5)
    assert bool(torch.isnan(solver.get_arm_angle(joints)).all())


@pytest.mark.parametrize(
    "field,value",
    [
        ("arm_angle_weight", -1.0),
        ("joint_limit_weight", float("nan")),
        ("max_joint_step", 0.0),
    ],
)
def test_search_configuration_rejects_invalid_scores_and_bounds(
    field: str, value: float
) -> None:
    with pytest.raises(ValueError, match=field):
        FEPSolverCfg(redundancy_search=True, **{field: value})


def test_search_candidates_are_distinct_and_periodic_preference_is_equivalent(
    solver: FEPSolver,
) -> None:
    seed = solver.get_default_qpos_seed()[None]
    seed[:, 1] = -0.4
    target = solver.get_fk(seed)
    solver.cfg.redundancy_search = True
    preferred = solver.get_arm_angle(seed) + 0.2
    valid, joints = solver.get_ik(
        target, seed, return_all_solutions=True, arm_angle=preferred
    )
    assert bool(valid.all())
    _assert_pose_accuracy(target.expand(8, -1, -1), solver.get_fk(joints[0]))
    distances = (joints[0, :, None] - joints[0, None, :]).abs().amax(-1)
    distances.fill_diagonal_(torch.inf)
    assert float(distances.min()) > 1e-6
    valid2, joints2 = solver.get_ik(target, seed, arm_angle=preferred + 2 * torch.pi)
    assert bool(valid2.all())
    torch.testing.assert_close(joints[:, 0], joints2, atol=1e-5, rtol=0)


def test_search_handles_mixed_failure_empty_batches_and_collapsed_limits(
    solver: FEPSolver,
) -> None:
    seed = solver.get_default_qpos_seed()[None].repeat(2, 1)
    seed[:, 1] = -0.4
    target = solver.get_fk(seed)
    target[1, 0, 3] += 10
    solver.cfg.redundancy_search = True
    valid, joints = solver.get_ik(target, seed)
    assert valid.tolist() == [True, False]
    torch.testing.assert_close(joints[1], seed[1], atol=0, rtol=0)
    valid, joints = solver.get_ik(target[:0], seed[:0])
    assert valid.shape == (0,) and joints.shape == (0, 7)
    solver.set_qpos_limits(seed[0], seed[0])
    valid, joints = solver.get_ik(target[:1], seed[:1], return_all_solutions=True)
    assert int(valid.sum()) == 1
    torch.testing.assert_close(joints[0, 0], seed[0], atol=1e-6, rtol=0)


@pytest.mark.parametrize("preferred", [float("nan"), [0.0, 1.0]])
def test_search_rejects_invalid_arm_angle(solver: FEPSolver, preferred) -> None:
    solver.cfg.redundancy_search = True
    seed = solver.get_default_qpos_seed()[None]
    with pytest.raises(ValueError, match="arm_angle"):
        solver.get_ik(solver.get_fk(seed), seed, arm_angle=preferred)


def test_solutions_on_joint_limit_boundaries(solver: FEPSolver) -> None:
    qpos = solver.get_default_qpos_seed()[None].repeat(14, 1)
    indices = torch.arange(7, device=solver.device)
    qpos[indices, indices] = solver.lower_qpos_limits
    qpos[indices + 7, indices] = solver.upper_qpos_limits
    target = solver.get_fk(qpos)
    valid, joints = solver.get_ik(target, qpos)
    assert bool(valid.all())
    _assert_pose_accuracy(target, solver.get_fk(joints))
    _assert_limits(solver, joints)


def test_nearest_branch_uses_live_joint_weights(solver: FEPSolver) -> None:
    generator = torch.Generator().manual_seed(7)
    lower, upper = solver.lower_qpos_limits, solver.upper_qpos_limits
    reference = lower + (upper - lower) * (
        0.25 + 0.5 * torch.rand(16, 7, generator=generator).to(solver.device)
    )
    seed = lower + (upper - lower) * torch.rand(16, 7, generator=generator).to(
        solver.device
    )
    seed[:, 6] = reference[:, 6]
    target = solver.get_fk(reference)
    flags, candidates = solver.get_ik(target, seed, return_all_solutions=True)
    saved = candidates.clone()
    weights = np.array([10.0, 0.1, 8.0, 0.5, 5.0, 0.7, 2.0])
    solver.set_ik_nearest_weight(weights)
    # Compute the nearest candidate independently of the fused selector.
    costs = (
        (candidates.double() - seed[:, None].double()).square()
        * torch.tensor(weights, device=solver.device)
    ).sum(dim=-1)
    expected = candidates[
        torch.arange(len(seed), device=solver.device),
        costs.masked_fill(~flags, torch.inf).argmin(dim=-1),
    ]
    valid, result = solver.get_ik(target, seed)
    assert bool(valid.all())
    torch.testing.assert_close(result, expected, atol=0, rtol=0)
    torch.testing.assert_close(candidates, saved, atol=0, rtol=0)


def test_tcp_updates_refresh_both_geometry_and_fk(solver: FEPSolver) -> None:
    qpos = solver.get_default_qpos_seed()[None]
    qpos[:, :3] += qpos.new_tensor([0.3, 0.5, -0.2])
    for translation in ([0.1, -0.03, 0.07], [-0.05, 0.02, 0.03], [0, 0, 0]):
        tcp = np.eye(4)
        tcp[:3, :3] = Rotation.from_euler("xyz", translation).as_matrix()
        tcp[:3, 3] = translation
        solver.set_tcp(tcp)
        target = solver.get_fk(qpos)
        valid, result = solver.get_ik(target, qpos)
        assert bool(valid.all())
        _assert_pose_accuracy(target, solver.get_fk(result))


@pytest.mark.parametrize(
    "invalid", ["target", "seed", "limits", "rotation", "last_row"]
)
def test_fused_validation_rejects_invalid_inputs(
    solver: FEPSolver, invalid: str
) -> None:
    seed = solver.get_default_qpos_seed()[None].repeat(2, 1)
    target = solver.get_fk(seed)
    if invalid == "target":
        target[1, 0, 3] = torch.nan
    elif invalid == "seed":
        seed[1, 0] = torch.inf
    elif invalid == "limits":
        solver.lower_qpos_limits[0] = solver.upper_qpos_limits[0] + 1
    elif invalid == "rotation":
        target[1, :3, :3] *= 2
    else:
        target[1, 3, 0] = 1
    with pytest.raises(ValueError):
        solver.get_ik(target, seed)


def test_empty_batch_has_empty_results(solver: FEPSolver) -> None:
    target = torch.empty(0, 4, 4, device=solver.device)
    valid, result = solver.get_ik(target)
    assert valid.shape == (0,) and result.shape == (0, 7)
    valid, result = solver.get_ik(target, return_all_solutions=True)
    assert valid.shape == (0, 8) and result.shape == (0, 8, 7)


def test_native_fk_verifier_rejects_translation_and_rotation_errors(
    solver: FEPSolver,
) -> None:
    qpos = solver.get_default_qpos_seed()
    targets = solver.get_fk(qpos).repeat(4, 1, 1)
    targets[1, 0, 3] += 2e-5
    targets[2, :3, :3] = targets[2, :3, :3] @ torch.diag(qpos.new_tensor([-1, -1, 1]))
    targets[3, :3, :3] = targets[3, :3, :3] @ torch.as_tensor(
        Rotation.from_rotvec([0, 0, 1e-4]).as_matrix(),
        device=solver.device,
        dtype=torch.float32,
    )
    valid = torch.empty(4, device=solver.device, dtype=torch.bool)
    stream = (
        wp.stream_from_torch(torch.cuda.current_stream(solver.device))
        if solver.device.type == "cuda"
        else None
    )
    wp.launch(
        _verify_fk_residuals,
        dim=4,
        inputs=[
            wp.from_torch(qpos),
            wp.from_torch(targets, dtype=wp.mat44f),
            wp.from_torch(solver._frames, dtype=wp.mat44d),
            wp.from_torch(solver._local_axes, dtype=wp.vec3d),
        ],
        outputs=[wp.from_torch(valid)],
        device=str(solver.device),
        stream=stream,
    )
    assert valid.tolist() == [True, False, False, False]


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("search", [False, True])
def test_cached_geometry_across_cuda_streams(search: bool) -> None:
    creator, consumer = torch.cuda.Stream(), torch.cuda.Stream()
    with torch.cuda.stream(creator):
        solver = FEPSolverCfg(
            urdf_path=get_data_path("Franka/Panda/PandaWithHand.urdf"),
            root_link_name="base",
            end_link_name="fr3_hand_tcp",
            redundancy_search=search,
        ).init_solver("cuda")
        seed = solver.get_default_qpos_seed()[None]
        target = solver.get_fk(seed)
    consumer.wait_stream(creator)
    with torch.cuda.stream(consumer):
        valid, result = solver.get_ik(target, seed)
        saved = result.clone()
        tcp = np.eye(4)
        tcp[:3, 3] = [0.03, -0.05, 0.1]
        solver.set_tcp(tcp)
        target2 = solver.get_fk(seed)
    creator.wait_stream(consumer)
    with torch.cuda.stream(creator):
        valid2, result2 = solver.get_ik(target2, seed)
        assert bool(valid.all()) and bool(valid2.all())
        _assert_pose_accuracy(target2, solver.get_fk(result2))
        torch.testing.assert_close(result, saved, atol=0, rtol=0)
