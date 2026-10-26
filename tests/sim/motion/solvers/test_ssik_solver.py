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

"""SSIK's original-model, batching and continuous-control contracts."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pytest
import pytorch_kinematics as pk
import torch
import warp as wp
from scipy.spatial.transform import Rotation

from embodichain.compute.kinematics._warp.ssik_7r import (
    mat87d,
    solve_srs,
    vec7d,
    vec8i,
)
from embodichain.lab.sim.motion.solvers import SSIKSolver, SSIKSolverCfg, SolverCfg
from embodichain_robots import AlohaMiniCfg, TianjiMarvinCfg

_DEVICE_CASES = [
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
_ARM_CASES = [
    ("aloha", "left", False),
    ("aloha", "right", False),
    ("marvin", "left", False),
    ("marvin", "right", False),
    ("marvin", "left", True),
    ("marvin", "right", True),
]


@wp.kernel
def _fixed_swivel_candidates(
    axes: wp.array(dtype=wp.vec3d),
    offsets: wp.array(dtype=wp.vec3d),
    home: wp.mat33d,
    shoulder: wp.vec3d,
    wrist: wp.vec3d,
    targets: wp.array(dtype=wp.mat44d),
    swivels: wp.array(dtype=wp.float64),
    seeds: wp.array(dtype=vec7d),
    candidates: wp.array2d(dtype=mat87d),
    validity: wp.array2d(dtype=vec8i),
):
    target, swivel = wp.tid()
    qpos, valid = solve_srs(
        axes,
        offsets,
        home,
        shoulder,
        wrist,
        targets[target],
        swivels[swivel],
        seeds[target],
    )
    candidates[target, swivel] = qpos
    validity[target, swivel] = valid


def _make_solver(
    device: str = "cpu",
    robot: str = "aloha",
    side: str = "left",
    with_gripper: bool = False,
    **options: Any,
) -> SSIKSolver:
    preset = (
        AlohaMiniCfg.from_dict({})
        if robot == "aloha"
        else TianjiMarvinCfg.from_dict({"with_gripper": with_gripper})
    )
    part = f"{side}_arm"
    chain = preset.build_pk_serial_chain(torch.device(device))[part]
    original = preset.solver_cfg[part]
    cfg = SSIKSolverCfg(
        urdf_path=preset.fpath,
        root_link_name=original.root_link_name,
        end_link_name=original.end_link_name,
        joint_names=chain.get_joint_parameter_names(),
        tcp=original.tcp,
        **options,
    )
    return cfg.init_solver(device=device, pk_serial_chain=chain)


@pytest.fixture(params=_DEVICE_CASES)
def solver(request: pytest.FixtureRequest) -> SSIKSolver:
    # A small chunk forces the same production path as a large batched call.
    return _make_solver(request.param, chunk_size=3)


def _sample(solver: SSIKSolver, count: int = 8) -> torch.Tensor:
    generator = torch.Generator().manual_seed(20261025)
    fraction = 0.15 + 0.7 * torch.rand(count, solver.dof, generator=generator)
    return solver.lower_qpos_limits + fraction.to(solver.device) * (
        solver.upper_qpos_limits - solver.lower_qpos_limits
    )


def _pk_fk(solver: SSIKSolver, joints: torch.Tensor) -> torch.Tensor:
    """Compose the original asset chain, without the extracted Warp model."""
    flange = solver.pk_serial_chain.forward_kinematics(joints).get_matrix()
    tcp = torch.as_tensor(solver.tcp_xpos, dtype=flange.dtype, device=flange.device)
    return flange @ tcp


def _assert_pose(target: torch.Tensor, actual: torch.Tensor) -> None:
    expected = target.detach().cpu().numpy().astype(np.float64)
    result = actual.detach().cpu().numpy().astype(np.float64)
    translation = np.linalg.norm(expected[..., :3, 3] - result[..., :3, 3], axis=-1)
    expected_rotation = Rotation.from_matrix(expected[..., :3, :3])
    result_rotation = Rotation.from_matrix(result[..., :3, :3])
    rotation = (expected_rotation.inv() * result_rotation).magnitude()
    assert np.all(translation <= 1e-5)
    assert np.all(rotation <= 1e-5)


def _assert_limits(solver: SSIKSolver, joints: torch.Tensor) -> None:
    assert bool(torch.isfinite(joints).all())
    assert bool((joints >= solver.lower_qpos_limits).all())
    assert bool((joints <= solver.upper_qpos_limits).all())


@pytest.mark.parametrize("device", _DEVICE_CASES)
@pytest.mark.parametrize("robot,side,with_gripper", _ARM_CASES)
def test_actual_arm_roundtrip_and_sampled_branches(
    device: str, robot: str, side: str, with_gripper: bool
) -> None:
    solver = _make_solver(device, robot, side, with_gripper)
    qpos = _sample(solver)
    targets = _pk_fk(solver, qpos)
    # An unrelated midpoint seed must suffice; the generating posture is not
    # supplied as a candidate or used as a numerical-refinement warm start.
    seed = solver.get_default_qpos_seed()[None].expand(len(qpos), -1)
    valid, result = solver.get_ik(targets, seed)
    assert valid.shape == (len(qpos),) and bool(valid.all())
    assert result.shape == qpos.shape
    _assert_pose(targets, _pk_fk(solver, result))
    _assert_limits(solver, result)

    mask, branches = solver.get_ik(targets[:1], seed[:1], return_all_solutions=True)
    expected_count = 8 if solver.dof == 6 else 8 * (solver.cfg.num_samples + 1)
    assert mask.shape == (1, expected_count)
    assert branches.shape == (1, expected_count, solver.dof)
    assert bool(mask.any())
    _assert_limits(solver, branches)
    _assert_pose(
        targets[:1].expand(int(mask.sum()), -1, -1), _pk_fk(solver, branches[mask])
    )


def test_mixed_failure_returns_projected_seed_and_preserves_old_results(
    solver: SSIKSolver,
) -> None:
    qpos = _sample(solver, 11)
    targets = _pk_fk(solver, qpos)
    targets[1, :3, 3] = 100  # Beyond the arm's sub-metre reach.
    seed = solver.get_default_qpos_seed()[None].expand(11, -1).clone()
    seed[1] = solver.upper_qpos_limits + 10
    valid, joints = solver.get_ik(targets, seed)
    assert valid.tolist() == [True, False] + [True] * 9
    torch.testing.assert_close(joints[1], solver.upper_qpos_limits, atol=0, rtol=0)
    _assert_pose(targets[valid], _pk_fk(solver, joints[valid]))
    snapshot = valid.clone(), joints.clone()
    mask, candidates = solver.get_ik(targets[:1], seed[:1], return_all_solutions=True)
    all_snapshot = mask.clone(), candidates.clone()
    for count in (2, 9, 1):
        solver.get_ik(targets[:count], seed[:count])
    torch.testing.assert_close(valid, snapshot[0], atol=0, rtol=0)
    torch.testing.assert_close(joints, snapshot[1], atol=0, rtol=0)
    torch.testing.assert_close(mask, all_snapshot[0], atol=0, rtol=0)
    torch.testing.assert_close(candidates, all_snapshot[1], atol=0, rtol=0)


@pytest.mark.parametrize("all_solutions", [False, True])
def test_nonfinite_targets_and_seeds_fail_with_finite_outputs(
    solver: SSIKSolver, all_solutions: bool
) -> None:
    targets = _pk_fk(solver, _sample(solver, 4))
    seed = solver.get_default_qpos_seed()[None].expand(4, -1).clone()
    targets[1, 0, 3] = torch.nan
    seed[2, 0] = torch.nan
    targets[3, 1, 3] = torch.inf
    valid, result = solver.get_ik(targets, seed, return_all_solutions=all_solutions)
    row_valid = valid.any(-1) if all_solutions else valid
    assert row_valid.tolist() == [True, False, False, False]
    _assert_limits(solver, result)
    fallback = solver.get_default_qpos_seed()
    torch.testing.assert_close(
        result[1:], fallback.expand_as(result[1:]), atol=0, rtol=0
    )


def test_malformed_reflection_affine_and_scaled_targets_are_not_solutions(
    solver: SSIKSolver,
) -> None:
    targets = _pk_fk(solver, _sample(solver, 4))
    targets[1, :3, 0] *= -1  # Orthogonal, but a reflection rather than SO(3).
    targets[2, 3, 0] = 0.2  # A nonrigid homogeneous bottom row.
    targets[3, :3, :3] *= 1.1  # Scaling cannot be accepted as an orientation.
    valid, result = solver.get_ik(targets)
    assert valid.tolist() == [True, False, False, False]
    _assert_limits(solver, result)


@pytest.mark.parametrize(
    "overshoot,position_tolerance,expected",
    [(4e-7, 1e-5, True), (4e-7, 1e-8, False), (1e-3, 1e-5, False)],
)
def test_bound_roundoff_projection_still_requires_original_fk_acceptance(
    solver: SSIKSolver,
    overshoot: float,
    position_tolerance: float,
    expected: bool,
) -> None:
    seed = _sample(solver, 1)
    tcp = np.eye(4)
    tcp[0, 3] = 5  # Make sub-microradian joint errors visible in position.
    solver.set_tcp(tcp)
    outside = seed.clone()
    outside[:, 0] += overshoot
    target = _pk_fk(solver, outside)
    solver.set_qpos_limits(seed[0], seed[0])
    solver.cfg.position_tolerance = position_tolerance
    valid, result = solver.get_ik(target, seed)
    assert valid.tolist() == [expected]
    torch.testing.assert_close(result, seed, atol=0, rtol=0)
    _assert_limits(solver, result)


def test_weight_updates_select_the_weighted_minimum_candidate(
    solver: SSIKSolver,
) -> None:
    # The hardware's narrow box admits only one branch for these targets.
    # Broaden this selection fixture so joint weighting has an observable
    # choice while the original-chain pose contract remains unchanged.
    limit = torch.full((solver.dof,), torch.pi, device=solver.device)
    solver.set_qpos_limits(-limit, limit)
    qpos = _sample(solver, 5)
    targets = _pk_fk(solver, qpos)
    seed = solver.get_default_qpos_seed()[None].expand(len(qpos), -1)
    mask, candidates = solver.get_ik(targets, seed, return_all_solutions=True)
    assert bool((mask.sum(-1) > 1).any())
    selections = []
    for emphasis in (0, solver.dof - 1):
        weights = np.ones(solver.dof)
        weights[emphasis] = 50
        assert solver.set_ik_nearest_weight(weights)
        cost = (
            ((candidates - seed[:, None]) * candidates.new_tensor(weights))
            .square()
            .sum(-1)
        )
        expected = candidates[
            torch.arange(len(qpos), device=solver.device),
            cost.masked_fill(~mask, torch.inf).argmin(-1),
        ]
        valid, result = solver.get_ik(targets, seed)
        assert bool(valid.all())
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=0)
        selections.append(expected)
    assert bool((selections[0] - selections[1]).abs().amax(-1).gt(0.1).any())


def test_tcp_and_effective_limit_updates_use_the_original_chain(
    solver: SSIKSolver,
) -> None:
    qpos = _sample(solver, 1)
    tcp = np.eye(4)
    tcp[:3, :3] = Rotation.from_euler("xyz", [0.2, -0.3, 0.4]).as_matrix()
    tcp[:3, 3] = [0.08, -0.03, 0.12]
    solver.set_tcp(tcp)
    target = _pk_fk(solver, qpos)
    valid, result = solver.get_ik(target, qpos)
    assert bool(valid.all())
    _assert_pose(target, _pk_fk(solver, result))
    # A narrow feasible box retains this posture; changing it far away must
    # invalidate every branch and return the newly projected seed.
    solver.set_qpos_limits(qpos[0] - 0.01, qpos[0] + 0.01)
    valid, result = solver.get_ik(target, qpos)
    assert bool(valid.all())
    _assert_limits(solver, result)
    locked = qpos[0] + 0.2
    solver.set_qpos_limits(locked, locked)
    valid, result = solver.get_ik(target, qpos)
    assert valid.tolist() == [False]
    torch.testing.assert_close(result[0], locked, atol=0, rtol=0)


def test_seedless_noncontiguous_and_multiaxis_batches(solver: SSIKSolver) -> None:
    qpos = _sample(solver, 12)
    targets = _pk_fk(solver, qpos)
    valid, result = solver.get_ik(targets[::2])
    assert not targets[::2].is_contiguous()
    assert valid.shape == (6,) and bool(valid.all())
    _assert_pose(targets[::2], _pk_fk(solver, result))
    poses = targets.reshape(2, 6, 4, 4).transpose(0, 1)
    seeds = qpos.reshape(2, 6, solver.dof).transpose(0, 1)
    assert not poses.is_contiguous() and not seeds.is_contiguous()
    mask, result = solver.get_ik_batch(poses, seeds)
    assert mask.shape == (6, 2) and result.shape == (6, 2, solver.dof)
    assert bool(mask.all())
    _assert_pose(
        poses.reshape(-1, 4, 4), _pk_fk(solver, result.reshape(-1, solver.dof))
    )
    for all_solutions in (False, True):
        mask, result = solver.get_ik(targets[:0], return_all_solutions=all_solutions)
        assert mask.shape[0] == result.shape[0] == 0
        assert result.shape[-1] == solver.dof


def test_continuous_selection_enforces_previous_step_and_holds_failure(
    solver: SSIKSolver,
) -> None:
    solver.cfg.max_joint_step = 0.05
    initial = solver.get_default_qpos_seed()[None]
    # The second posture is >0.05 from the original seed, but only 0.04
    # from the preceding frame. The last candidate is an excessive jump.
    candidates = initial[:, None, None].expand(1, 4, 1, solver.dof).clone()
    candidates[0, :, 0, 0] += candidates.new_tensor([0.04, 0.08, 0.12, 0.30])
    validity = torch.ones((1, 4, 1), dtype=torch.bool, device=solver.device)
    validity[:, 2] = False
    valid, result = solver._select_continuous_ik_path(candidates, validity, initial)
    assert valid.tolist() == [[True, True, False, False]]
    expected = initial[:, None].expand(1, 4, solver.dof).clone()
    expected[0, :, 0] += result.new_tensor([0.04, 0.08, 0.08, 0.08])
    torch.testing.assert_close(result, expected)


def test_continuous_invalid_initial_seed_has_a_finite_failure_hold() -> None:
    solver = _make_solver()
    initial = solver.get_default_qpos_seed()[None]
    initial[0, 0] = torch.nan
    candidates = torch.zeros((1, 2, 1, solver.dof))
    validity = torch.zeros((1, 2, 1), dtype=torch.bool)
    valid, result = solver._select_continuous_ik_path(candidates, validity, initial)
    assert valid.tolist() == [[False, False]]
    expected = solver.get_default_qpos_seed()[None, None].expand_as(result)
    torch.testing.assert_close(result, expected, atol=0, rtol=0)


def test_continuous_candidates_defer_step_bounds_until_previous_frame(
    solver: SSIKSolver,
) -> None:
    solver.cfg.max_joint_step = 0.05
    initial = _sample(solver, 1)
    path = initial.expand(4, -1).clone()
    path[:, 0] += path.new_tensor([0.0, 0.035, 0.07, 0.105])
    targets = _pk_fk(solver, path)
    seeds = initial.expand(len(path), -1)
    mask, candidates = solver._get_continuous_ik_candidates(targets, seeds)
    assert bool(mask.any(-1).all())
    valid, result = solver._select_continuous_ik_path(
        candidates[None], mask[None], initial
    )
    assert bool(valid.all())
    _assert_pose(targets, _pk_fk(solver, result[0]))
    steps = torch.diff(torch.cat((initial, result[0])), dim=0)
    assert bool((steps.abs() <= solver.cfg.max_joint_step).all())
    assert float((result[0, -1] - initial[0]).abs().max()) > 0.05


def _synthetic_srs(
    device: str,
    axes: np.ndarray,
    *,
    offsets: np.ndarray | None = None,
    srs_search: str = "uniform",
) -> SSIKSolver:
    # All shoulder axes intersect at the origin, all wrist axes at the
    # elbow-plus-forearm point. The elbow origin need not lie on an outer
    # shoulder axis, nor be straight at home.
    if offsets is None:
        offsets = np.zeros((7, 3))
        offsets[3] = [0.25, 0.06, -0.02]
        offsets[4] = [0.19, -0.03, 0.07]
    links = "".join(f'<link name="l{i}"/>' for i in range(8))
    joints = "".join(
        f'<joint name="j{i}" type="revolute"><parent link="l{i}"/>'
        f'<child link="l{i+1}"/><origin xyz="{" ".join(map(str, offsets[i]))}"/>'
        f'<axis xyz="{" ".join(map(str, axes[i]))}"/>'
        '<limit lower="-3.14159265" upper="3.14159265" velocity="1" effort="1"/></joint>'
        for i in range(7)
    )
    chain = pk.build_serial_chain_from_urdf(
        f'<robot name="synthetic_srs">{links}{joints}</robot>', "l7"
    ).to(device=device)
    return SSIKSolverCfg(srs_search=srs_search).init_solver(
        device=device, pk_serial_chain=chain
    )


@pytest.mark.parametrize("device", _DEVICE_CASES)
def test_srs_supports_nonorthogonal_concurrent_axis_triples(device: str) -> None:
    rng = np.random.default_rng(42917)
    axes = rng.normal(size=(7, 3))
    axes /= np.linalg.norm(axes, axis=1)[:, None]
    solver = _synthetic_srs(device, axes)
    seed = torch.as_tensor(
        rng.uniform(-1, 1, (4, 7)), dtype=torch.float32, device=device
    )
    target = _pk_fk(solver, seed)
    valid, result = solver.get_ik(target, seed)
    assert bool(valid.all())
    _assert_pose(target, _pk_fk(solver, result))
    torch.testing.assert_close(result, seed, atol=1e-4, rtol=0)


@pytest.mark.parametrize("device", _DEVICE_CASES)
def test_srs_seed_local_skips_an_elbow_without_a_shoulder_decomposition(
    device: str,
) -> None:
    # A narrow five-degree shoulder triple can represent the second elbow
    # sheet at the seeded swivel while the first sheet has no decomposition.
    tilt = np.deg2rad(5)
    axes = np.array(
        [
            [0, 0, 1],
            [0, np.sin(tilt), np.cos(tilt)],
            [0, 0, 1],
            [0, 1, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 0, 1],
        ]
    )
    offsets = np.zeros((7, 3))
    offsets[3:5, 0] = 0.3
    solver = _synthetic_srs(device, axes, offsets=offsets, srs_search="seed_local")
    generating = torch.tensor([[0.1, 0.2, 0.3, -1.0, 0.2, 0.4, 0.1]], device=device)
    target = _pk_fk(solver, generating).expand(2, -1, -1)
    seed = generating.expand(2, -1).clone()
    seed[1, 0] += 0.03
    valid, candidates = solver.get_ik(target, seed, return_all_solutions=True)
    assert valid.shape == (2, 1040)
    assert not bool(valid[:, 1024:1028].any())
    assert bool(valid[:, 1028:1032].any(-1).all())
    assert bool(valid[:, -8:].any(-1).all())
    _assert_pose(
        target[:, None].expand(-1, 1040, -1, -1)[valid],
        _pk_fk(solver, candidates[valid]),
    )
    _assert_limits(solver, candidates[valid])
    nearest_valid, nearest = solver.get_ik(target, seed)
    assert bool(nearest_valid.all())
    distances = ((candidates - seed[:, None]) ** 2).sum(-1)
    distances = torch.where(valid, distances, torch.inf)
    expected = candidates[torch.arange(2, device=device), distances.argmin(-1)]
    torch.testing.assert_close(nearest, expected, atol=1e-6, rtol=0)
    if device == "cuda":
        reference = _synthetic_srs(
            "cpu", axes, offsets=offsets, srs_search="seed_local"
        )
        expected_valid, expected = reference.get_ik(
            target.cpu(), seed.cpu(), return_all_solutions=True
        )
        torch.testing.assert_close(valid.cpu(), expected_valid)
        delta = (candidates.cpu().double() - expected.double() + torch.pi) % (
            2 * torch.pi
        ) - torch.pi
        assert bool((delta[expected_valid].abs() < 1e-4).all())


@pytest.mark.parametrize("device", _DEVICE_CASES)
def test_srs_seeded_euler_gimbal_preserves_a_finite_valid_posture(device: str) -> None:
    axes = np.array(
        [[0, 0, 1], [0, 1, 0], [0, 0, 1], [0, 1, 0], [0, 0, 1], [0, 1, 0], [0, 0, 1]]
    )
    solver = _synthetic_srs(device, axes)
    # Both shoulder and wrist Euler decompositions have a zero middle angle.
    # Keeping the caller's outer angle avoids introducing an arbitrary jump.
    seed = torch.tensor([[0.3, 0, -0.5, -0.8, 0.7, 0, -0.2]], device=device)
    target = _pk_fk(solver, seed)
    valid, result = solver.get_ik(target, seed)
    assert bool(valid.all())
    _assert_pose(target, _pk_fk(solver, result))
    _assert_limits(solver, result)
    torch.testing.assert_close(result, seed, atol=2e-3, rtol=0)


def test_final_position_acceptance_is_at_tcp_after_joint_projection() -> None:
    axes = np.array(
        [[0, 0, 1], [0, 1, 0], [0, 0, 1], [0, 1, 0], [0, 0, 1], [0, 1, 0], [0, 0, 1]]
    )
    solver = _synthetic_srs("cpu", axes)
    # Exact original-chain targets remove input rounding from this tolerance
    # test. A long tool magnifies the accepted flange orientation residual
    # into a TCP position error exceeding the independent position threshold.
    solver.pk_serial_chain.to(dtype=torch.float64)
    tcp = np.eye(4)
    tcp[0, 3] = 20
    solver.set_tcp(tcp)
    seed = torch.tensor([[0.2, 0.4, -0.3, -0.8, 0.5, 0.7, -0.2]])
    outside = seed.double()
    outside[0, -1] += 8e-7
    target = _pk_fk(solver, outside)
    solver.set_qpos_limits(seed[0], seed[0])
    actual = _pk_fk(solver, seed.double())
    position_error = torch.linalg.vector_norm(
        actual[:, :3, 3] - target[:, :3, 3], dim=-1
    )
    assert bool((position_error > solver.cfg.position_tolerance).all())
    valid, result = solver.get_ik(target, seed)
    assert valid.tolist() == [False]
    torch.testing.assert_close(result, seed, atol=0, rtol=0)


@pytest.mark.parametrize("robot_name", ["aloha", "marvin"])
def test_robot_facade_converts_roots_and_selects_actual_continuous_ik(
    robot_name: str,
) -> None:
    from embodichain.lab.sim.objects.robot import Robot

    solver = _make_solver(robot=robot_name, max_joint_step=0.05)
    initial = _sample(solver, 2)
    path = initial[:, None].expand(2, 4, solver.dof).clone()
    path[:, :, 0] += path.new_tensor([0, 0.02, 0.04, 0.06])
    local_targets = _pk_fk(solver, path.reshape(-1, solver.dof)).reshape(2, 4, 4, 4)
    roots = torch.eye(4).repeat(2, 1, 1)
    roots[:, :3, :3] = torch.from_numpy(
        Rotation.from_euler("xyz", [[0.2, -0.1, 0.4], [-0.3, 0.2, -0.5]]).as_matrix()
    ).float()
    roots[:, :3, 3] = torch.tensor([[0.2, -0.3, 0.4], [-0.4, 0.1, 0.5]])
    arena_targets = roots[:, None] @ local_targets
    robot = object.__new__(Robot)
    robot.device = torch.device("cpu")
    robot._all_indices = [0, 1]
    robot._solvers = {"arm": solver}

    def root_pose(link_name: str, env_ids: list[int], to_matrix: bool) -> torch.Tensor:
        assert link_name == solver.root_link_name and to_matrix
        return roots[env_ids]

    robot.get_link_pose = root_pose
    for continuous in (False, True):
        valid, result = robot.compute_batch_ik(
            arena_targets, initial if continuous else path, "arm", continuous=continuous
        )
        assert valid.shape == (2, 4) and bool(valid.all())
        assert result.shape == path.shape
        _assert_pose(
            local_targets.reshape(-1, 4, 4),
            _pk_fk(solver, result.reshape(-1, solver.dof)),
        )
        if continuous:
            delta = torch.diff(torch.cat((initial[:, None], result), dim=1), dim=1)
            assert bool((delta.abs() <= solver.cfg.max_joint_step).all())


@pytest.mark.gpu
@pytest.mark.parametrize("robot", ["aloha", "marvin"])
def test_cpu_cuda_parity_and_stream_safe_reuse(robot: str) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    cpu = _make_solver("cpu", robot=robot, chunk_size=3)
    cuda = _make_solver("cuda", robot=robot, chunk_size=3)
    qpos = _sample(cpu, 7)
    targets = _pk_fk(cpu, qpos)
    seed = qpos + 0.02
    expected_mask, expected = cpu.get_ik(targets, seed)
    torch.cuda.synchronize()
    saved = []
    for count in (1, 7, 2):
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            saved.append(
                (count, cuda.get_ik(targets[:count].cuda(), seed[:count].cuda()))
            )
    torch.cuda.synchronize()
    for count, (valid, result) in saved:
        torch.testing.assert_close(valid.cpu(), expected_mask[:count])
        torch.testing.assert_close(result.cpu(), expected[:count], atol=2e-4, rtol=0)
        _assert_pose(targets[:count], _pk_fk(cuda, result))


@pytest.mark.gpu
def test_cuda_weight_updates_are_visible_to_another_stream() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    solver = _make_solver("cuda")
    limit = torch.full((solver.dof,), torch.pi, device=solver.device)
    solver.set_qpos_limits(-limit, limit)
    qpos = _sample(solver, 5)
    targets = _pk_fk(solver, qpos)
    seed = solver.get_default_qpos_seed()[None].expand(len(qpos), -1)
    old_weights, new_weights = np.ones(solver.dof), np.ones(solver.dof)
    old_weights[0], new_weights[-1] = 50, 50
    solver.set_ik_nearest_weight(old_weights)
    mask, candidates = solver.get_ik(targets, seed, return_all_solutions=True)
    candidates_cpu, mask_cpu, seed_cpu = candidates.cpu(), mask.cpu(), seed.cpu()
    selections = []
    for weights in (old_weights, new_weights):
        cost = (
            ((candidates_cpu - seed_cpu[:, None]) * torch.as_tensor(weights))
            .square()
            .sum(-1)
        )
        index = cost.masked_fill(~mask_cpu, torch.inf).argmin(-1)
        selections.append(candidates_cpu[torch.arange(len(qpos)), index])
    assert not torch.allclose(selections[0], selections[1])
    torch.cuda.synchronize()
    writer, reader = torch.cuda.Stream(), torch.cuda.Stream()
    with torch.cuda.stream(writer):
        # Leave the old in-place tensor values visible long enough that an
        # unfenced read on another stream would choose the old branch.
        torch.cuda._sleep(20_000_000)
        solver.set_ik_nearest_weight(new_weights)
    with torch.cuda.stream(reader):
        valid, result = solver.get_ik(targets, seed)
    torch.cuda.synchronize()
    assert bool(valid.all())
    torch.testing.assert_close(result.cpu(), selections[1], atol=1e-6, rtol=0)


@pytest.mark.parametrize(
    "options",
    [
        {"backend": "unknown"},
        {"num_samples": 0},
        {"num_samples": True},
        {"chunk_size": 1.5},
        {"position_tolerance": 0},
        {"position_tolerance": None},
        {"rotation_tolerance": None},
        {"rotation_tolerance": float("nan")},
        {"max_joint_step": -0.1},
        {"reference_artifact": "reference.json"},
    ],
)
def test_configuration_rejects_invalid_search_and_acceptance_settings(
    options: dict[str, Any],
) -> None:
    with pytest.raises(ValueError):
        SSIKSolverCfg(**options)


def test_configuration_decodes_through_solver_registry() -> None:
    cfg = SolverCfg.from_dict(
        {"class_type": "SSIKSolver", "num_samples": 32, "chunk_size": 7}
    )
    assert isinstance(cfg, SSIKSolverCfg)
    assert cfg.num_samples == 32 and cfg.chunk_size == 7


@pytest.mark.parametrize(
    "dof,prismatic", [(5, False), (6, False), (7, False), (6, True)]
)
def test_unsupported_geometry_is_rejected_before_solving(
    dof: int, prismatic: bool
) -> None:
    # Nonconcurrent alternating axes do not fit either supported family.
    axes = ("1 0 0", "0 1 0", "0 0 1")
    links = "".join(f'<link name="l{i}"/>' for i in range(dof + 1))
    joints = "".join(
        f'<joint name="j{i}" type="{"prismatic" if prismatic and i == 0 else "revolute"}">'
        f'<parent link="l{i}"/><child link="l{i+1}"/>'
        f'<origin xyz="0.1 0.03 0.02"/><axis xyz="{axes[i % 3]}"/>'
        '<limit lower="-2" upper="2" velocity="1" effort="1"/></joint>'
        for i in range(dof)
    )
    chain = pk.build_serial_chain_from_urdf(
        f'<robot name="unsupported">{links}{joints}</robot>', f"l{dof}"
    )
    with pytest.raises(ValueError, match="SSIK"):
        SSIKSolverCfg().init_solver(pk_serial_chain=chain)


def test_reference_backend_rejects_cuda_without_loading_a_model() -> None:
    with pytest.raises(ValueError, match="CPU"):
        SSIKSolverCfg(backend="reference").init_solver(device="cuda")


@pytest.fixture
def pinned_reference() -> None:
    pytest.importorskip("ssik", reason="Optional pinned SSIK reference unavailable")
    from embodichain.lab.sim.motion.solvers._ssik_reference import _require_ssik

    try:
        _require_ssik()
    except ImportError:
        pytest.skip("SSIK is installed but is not the pinned reference revision")


@pytest.mark.parametrize("robot,side,with_gripper", _ARM_CASES[:4])
def test_optional_reference_actual_arm_roundtrip(
    pinned_reference: None, robot: str, side: str, with_gripper: bool
) -> None:
    solver = _make_solver(
        robot=robot, side=side, with_gripper=with_gripper, backend="reference"
    )
    qpos = _sample(solver, 3)
    targets = _pk_fk(solver, qpos)
    valid, result = solver.get_ik(targets)
    assert valid.shape == (3,) and bool(valid.all())
    _assert_limits(solver, result)
    _assert_pose(targets, _pk_fk(solver, result))
    mask, branches = solver.get_ik(targets[:1], return_all_solutions=True)
    assert mask.ndim == 2 and mask.shape[0] == 1
    assert branches.shape == (*mask.shape, solver.dof)
    assert bool(mask.any())
    _assert_pose(
        targets[:1].expand(int(mask.sum()), -1, -1), _pk_fk(solver, branches[mask])
    )


def test_reference_artifact_checks_model_before_executing_source(
    pinned_reference: None, tmp_path: Path
) -> None:
    artifact = tmp_path / "wrong_model.py"
    artifact.write_text("raise AssertionError('incompatible artifact executed')\n")
    artifact.with_suffix(".json").write_text(
        json.dumps({"model_fingerprint": "a different robot"})
    )
    with pytest.raises(ValueError, match="does not match"):
        _make_solver(backend="reference", reference_artifact=str(artifact))


def test_reference_artifact_cli_build_and_load_roundtrip(
    pinned_reference: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from embodichain.lab.sim.motion.solvers._ssik_reference import _main

    direct = _make_solver(backend="reference")
    artifact = tmp_path / "aloha_reference.py"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ssik_reference",
            "--urdf",
            direct.cfg.urdf_path,
            "--root",
            direct.cfg.root_link_name,
            "--end",
            direct.cfg.end_link_name,
            "--output",
            str(artifact),
        ],
    )
    _main()
    assert artifact.is_file() and artifact.with_suffix(".json").is_file()
    loaded = _make_solver(backend="reference", reference_artifact=str(artifact))
    qpos = _sample(direct, 3)
    targets = _pk_fk(direct, qpos)
    expected_valid, expected = direct.get_ik(targets)
    valid, result = loaded.get_ik(targets)
    assert bool(valid.all())
    torch.testing.assert_close(valid, expected_valid)
    torch.testing.assert_close(result, expected, atol=1e-5, rtol=0)
    _assert_pose(targets, _pk_fk(loaded, result))
    # Source bytes are part of the manifest contract as well as robot frames.
    artifact.write_text(artifact.read_text() + "\n# changed after build\n")
    with pytest.raises(ValueError, match="source fingerprint"):
        _make_solver(backend="reference", reference_artifact=str(artifact))


@pytest.mark.parametrize("geometry", ["marvin_left", "marvin_right", "nonorthogonal"])
@pytest.mark.parametrize("cached", [False, True])
def test_srs_fixed_swivel_branch_sets_match_pinned_reference(
    pinned_reference: None, geometry: str, cached: bool
) -> None:
    from ssik.kinematics.poe_fk import poe_forward_kinematics
    from ssik.kinematics.predicates import _classify_srs_7r_geometric
    from ssik.solvers.seven_r.srs import solve
    from embodichain.lab.sim.motion.solvers._ssik_reference import _as_kinbody, _policy

    if geometry == "nonorthogonal":
        axes = np.random.default_rng(42917).normal(size=(7, 3))
        axes /= np.linalg.norm(axes, axis=1)[:, None]
        solver = _synthetic_srs("cpu", axes)
    else:
        solver = _make_solver(robot="marvin", side=geometry.split("_")[1])
    model = solver._model
    kb = _as_kinbody(model, solver.cfg)
    policy = _policy()
    classification = _classify_srs_7r_geometric(kb, policy)
    assert classification is not None
    qpos = _sample(solver, 10).numpy().astype(np.float64)
    targets = np.stack([poe_forward_kinematics(kb, q) for q in qpos])
    swivels = np.array([-2.3, -0.2, 1.4])
    candidates = wp.zeros((len(qpos), len(swivels)), dtype=mat87d, device="cpu")
    validity = wp.zeros((len(qpos), len(swivels)), dtype=vec8i, device="cpu")
    wp.launch(
        _fixed_swivel_candidates,
        dim=(len(qpos), len(swivels)),
        inputs=[
            wp.array(model.axes, dtype=wp.vec3d, device="cpu"),
            wp.array(model.offsets, dtype=wp.vec3d, device="cpu"),
            wp.mat33d(kb.joints[-1].T_right[:3, :3]),
            wp.vec3d(classification.shoulder_pivot),
            wp.vec3d(classification.wrist_pivot),
            wp.array(targets, dtype=wp.mat44d, device="cpu"),
            wp.array(swivels, dtype=wp.float64, device="cpu"),
            wp.array(qpos, dtype=vec7d, device="cpu"),
        ],
        outputs=[candidates, validity],
        device="cpu",
    )
    actual, valid = candidates.numpy(), validity.numpy().astype(bool)
    if cached:
        from embodichain.compute.kinematics._warp.ssik_srs_geometry import (
            CONTEXT_SIZE,
            prepare_srs_context,
            reconstruct_srs_context,
        )

        count = len(qpos) * len(swivels)
        context = wp.empty((count, CONTEXT_SIZE), dtype=wp.float64, device="cpu")
        raw = wp.empty((count, 8, 7), dtype=wp.float64, device="cpu")
        raw_valid = wp.empty((count, 8), dtype=wp.int32, device="cpu")
        axes = wp.array(model.axes, dtype=wp.vec3d, device="cpu")
        wp.launch(
            prepare_srs_context,
            count,
            inputs=[
                axes,
                wp.array(model.offsets, dtype=wp.vec3d, device="cpu"),
                wp.mat33d(kb.joints[-1].T_right[:3, :3]),
                wp.vec3d(classification.shoulder_pivot),
                wp.vec3d(classification.wrist_pivot),
                wp.array(
                    np.repeat(targets, len(swivels), axis=0),
                    dtype=wp.mat44d,
                    device="cpu",
                ),
                wp.mat44d(np.eye(4)),
                wp.array(
                    np.repeat(qpos, len(swivels), axis=0),
                    dtype=wp.float64,
                    device="cpu",
                ),
                wp.array(
                    solver.lower_qpos_limits.numpy().astype(np.float64),
                    dtype=wp.float64,
                    device="cpu",
                ),
                wp.array(
                    solver.upper_qpos_limits.numpy().astype(np.float64),
                    dtype=wp.float64,
                    device="cpu",
                ),
                1e-5,
                0,
            ],
            outputs=[context],
            device="cpu",
        )
        wp.launch(
            reconstruct_srs_context,
            (count, 8),
            inputs=[
                context,
                axes,
                wp.array(np.tile(swivels, len(qpos)), dtype=wp.float64, device="cpu"),
                wp.zeros(count, dtype=wp.float64, device="cpu"),
            ],
            outputs=[raw_valid, raw],
            device="cpu",
        )
        actual = raw.numpy().reshape(len(qpos), len(swivels), 8, 7)
        valid = raw_valid.numpy().astype(bool).reshape(len(qpos), len(swivels), 8)

    compared = 0
    raw_compared = 0
    for target_index, target in enumerate(targets):
        for swivel_index, swivel in enumerate(swivels):
            expected, _ = solve(
                kb,
                target,
                policy,
                swivel_samples=np.array([swivel]),
                fk_atol=1e-5,
            )
            expected_qpos = np.array([row.q for row in expected]).reshape(-1, 7)
            actual_qpos = actual[target_index, swivel_index][
                valid[target_index, swivel_index]
            ]
            # Apply the same original-chain closure and limit criteria to both
            # sets. Masks remain authoritative; branch ordering is irrelevant.
            filtered, closed = [], []
            for rows in (actual_qpos, expected_qpos):
                keep, closed_rows = [], []
                for q in rows:
                    if np.linalg.norm(poe_forward_kinematics(kb, q) - target) > 1e-5:
                        continue
                    closed_rows.append(q)
                    lo, hi = (
                        solver.lower_qpos_limits.numpy(),
                        solver.upper_qpos_limits.numpy(),
                    )
                    first, last = np.ceil((lo - q) / (2 * np.pi)), np.floor(
                        (hi - q) / (2 * np.pi)
                    )
                    if np.all(first <= last):
                        keep.append(q)
                filtered.append(np.array(keep).reshape(-1, 7))
                closed.append(np.array(closed_rows).reshape(-1, 7))
            # Compare raw geometric branches as well as the limit-filtered
            # representatives. An invalid mask cannot silently hide a missing
            # branch merely because that branch violates this preset's limits.
            for result, reference in (closed, filtered):
                assert (len(result) == 0) == (len(reference) == 0)
                if len(result) == 0:
                    continue
                distances = np.max(
                    np.abs(
                        (result[:, None] - reference[None] + np.pi) % (2 * np.pi)
                        - np.pi
                    ),
                    axis=-1,
                )
                assert np.all(distances.min(axis=0) < 1e-4)
                assert np.all(distances.min(axis=1) < 1e-4)
            raw_compared += len(closed[1])
            compared += len(filtered[1])
    assert raw_compared > 0 and compared > 0


@pytest.mark.parametrize(
    ("side", "witness"),
    [
        pytest.param(
            "left",
            [
                0.8403108876911829,
                -1.697616436034471,
                2.8483732277158538,
                -1.2659621229034035,
                0.9528700639925121,
                -2.1388794901772377,
            ],
            id="quantized-elbow-tangent",
        ),
        pytest.param(
            "left",
            [
                -0.4848797951963461,
                -3.187832105932237,
                0.9903254496394207,
                0.26256075330340434,
                1.5293779021570368,
                2.9713360160970597,
            ],
            id="quantized-shoulder-tangent-at-wrist-limit",
        ),
        pytest.param(
            "right",
            [
                0.3510048395622123,
                -3.4821357947416436,
                1.3787724891674582,
                -0.1338148162966517,
                -0.8942663428230884,
                0.7207185667499196,
            ],
            id="quantized-shoulder-limit",
        ),
    ],
)
def test_float32_target_roundoff_at_geometric_boundaries(
    side: str, witness: list[float]
) -> None:
    """Reachable PK poses remain solvable after the caller rounds to float32.

    These frozen formal-benchmark witnesses previously lost every feasible
    branch through a squared-distance tangent or an amplified limit excess.
    The expected pose comes from the original PK chain at float64 precision,
    independently of the extracted Warp geometry.
    """
    solver = _make_solver(side=side)
    original_chain = copy.deepcopy(solver.pk_serial_chain).to(dtype=torch.float64)
    known_joints = torch.tensor([witness], dtype=torch.float64)
    target = original_chain.forward_kinematics(known_joints).get_matrix().float()
    target = target @ torch.as_tensor(solver.tcp_xpos, dtype=torch.float32)
    seed = solver.get_default_qpos_seed()[None]
    success, joints = solver.get_ik(target, qpos_seed=seed)
    assert bool(success.all())
    _assert_limits(solver, joints)
    _assert_pose(target, _pk_fk(solver, joints))


@pytest.mark.parametrize("device", _DEVICE_CASES)
@pytest.mark.parametrize(
    "robot,side",
    [("aloha", "left"), ("aloha", "right"), ("marvin", "left"), ("marvin", "right")],
)
def test_isolated_geometric_provider_preserves_complete_accepted_slots(
    device: str, robot: str, side: str
) -> None:
    """Isolated kernels retain CPU legacy slots and independent PK closure."""
    from embodichain.compute.kinematics._ssik_geometric import _GeometricCandidates
    from embodichain.compute.kinematics._warp.ssik import accept_raw

    solver = _make_solver(device, robot=robot, side=side, srs_search="seed_local")
    reference = (
        solver
        if device == "cpu"
        else _make_solver(robot=robot, side=side, srs_search="seed_local")
    )
    solver.set_ik_nearest_weight(np.arange(1, solver.dof + 1, dtype=np.float32) / 3)
    qpos = _sample(reference)
    targets = _pk_fk(reference, qpos).double().to(device)
    seed = qpos.flip(0).double().to(device)
    lower, upper = solver.lower_qpos_limits.double(), solver.upper_qpos_limits.double()
    weights = solver.ik_nearest_weight.float()
    reference.set_ik_nearest_weight(weights.cpu().numpy())
    provider = _GeometricCandidates(solver._model, device, srs_search="seed_local")
    flange = targets @ torch.as_tensor(
        np.linalg.inv(solver.tcp_xpos), device=targets.device
    )
    raw_valid, raw = provider.generate(flange, seed, lower, upper, weights, 1e-5)
    valid = torch.empty_like(raw_valid)
    result = torch.empty_like(raw, dtype=torch.float32)
    wp.launch(
        accept_raw,
        dim=raw_valid.shape,
        inputs=[
            solver._warp_model,
            wp.from_torch(raw),
            wp.from_torch(raw_valid),
            wp.from_torch(targets, dtype=wp.mat44d),
            wp.mat44d(solver.tcp_xpos),
            wp.from_torch(seed),
            wp.from_torch(lower),
            wp.from_torch(upper),
            1e-5,
            1e-5,
            float("inf"),
        ],
        outputs=[wp.from_torch(valid), wp.from_torch(result)],
        device=device,
    )
    expected_valid, expected = reference.get_ik(
        targets.cpu(), seed.cpu(), return_all_solutions=True
    )
    mask = valid.bool()
    torch.testing.assert_close(mask.cpu(), expected_valid)
    delta = (result.double() - expected.to(device).double() + torch.pi) % (
        2 * torch.pi
    ) - torch.pi
    assert bool((delta[mask].abs() < 1e-4).all())
    _assert_pose(
        targets[:, None].expand(-1, mask.shape[1], -1, -1)[mask],
        _pk_fk(reference, result[mask].cpu()).to(device),
    )
