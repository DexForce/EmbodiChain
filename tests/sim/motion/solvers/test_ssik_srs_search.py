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

"""Seed-local geometric SRS search tracks whole paths without seed resets."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from embodichain.lab.sim.motion.solvers import SSIKSolver, SSIKSolverCfg, SolverCfg
from embodichain.lab.sim.objects.robot import Robot
from embodichain_robots import TianjiMarvinCfg

_DEVICES = [
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

# Original joint-space endpoints of frozen formal paths 0, 15 and 18, seed2531.
# These protect two shoulder-pole crossings and the old q0-limit drift; FK is
# recomputed from the asset rather than depending on external benchmark files.
_START = [
    [
        -0.1733906899570985,
        0.0839036767182777,
        -0.18495447334142057,
        -0.8055006175897184,
        -0.7753526987484558,
        -0.2878093994186187,
        0.4175159683587777,
    ],
    [
        0.35968383890943345,
        0.15033376983715208,
        0.637817900032839,
        -0.7609173323441571,
        0.43802710973597625,
        0.013720240190801647,
        -0.11875655564529586,
    ],
    [
        0.49043056623811454,
        0.3250342845787313,
        -0.024168048371214557,
        -0.6014826898167507,
        -0.623785825235132,
        -0.14325939351614858,
        -0.0381515080676087,
    ],
]
_FINISH = [
    [
        -0.48084259243618366,
        -0.25353807664737804,
        -0.2333236354188224,
        -0.8094306373706381,
        -0.40341133853590305,
        -0.03531414579042691,
        0.22552691454097906,
    ],
    [
        -0.8087837494221226,
        -0.6200704591489479,
        0.8386930614850949,
        -0.6128757423644904,
        -0.071906927819255,
        0.060931192981366955,
        -0.34134341899084886,
    ],
    [
        0.42961584435067035,
        -0.40568608688789864,
        -0.07540798055112541,
        -1.160543089966636,
        -0.4343279907739541,
        0.24697568792529156,
        0.41876068036927894,
    ],
]


def _solver(device: str, side: str = "left") -> SSIKSolver:
    preset = TianjiMarvinCfg.from_dict({"with_gripper": False})
    part = f"{side}_arm"
    original = preset.solver_cfg[part]
    chain = preset.build_pk_serial_chain(torch.device(device))[part]
    return SSIKSolverCfg(
        urdf_path=preset.fpath,
        root_link_name=original.root_link_name,
        end_link_name=original.end_link_name,
        tcp=original.tcp,
        srs_search="seed_local",
        max_joint_step=0.05,
    ).init_solver(device=device, pk_serial_chain=chain)


def _fk(solver: SSIKSolver, joints: torch.Tensor) -> torch.Tensor:
    chain = deepcopy(solver.pk_serial_chain).to(
        device=torch.device("cpu"), dtype=torch.float64
    )
    return chain.forward_kinematics(
        joints.detach().cpu().double()
    ).get_matrix() @ torch.as_tensor(solver.tcp_xpos, dtype=torch.float64)


def _assert_pose(target: torch.Tensor, actual: torch.Tensor) -> None:
    expected, result = (
        target.detach().cpu().double().numpy(),
        actual.detach().cpu().double().numpy(),
    )
    assert np.all(
        np.linalg.norm(expected[:, :3, 3] - result[:, :3, 3], axis=-1) <= 1e-5
    )
    angles = (
        Rotation.from_matrix(expected[:, :3, :3]).inv()
        * Rotation.from_matrix(result[:, :3, :3])
    ).magnitude()
    assert np.all(angles <= 1e-5)


def _paths(frames: int) -> torch.Tensor:
    start, finish = torch.tensor(_START, dtype=torch.float64), torch.tensor(
        _FINISH, dtype=torch.float64
    )
    t = torch.linspace(0, 1, frames, dtype=torch.float64)
    smooth = 3 * t.square() - 2 * t.pow(3)
    return start[:, None] + (finish - start)[:, None] * smooth[None, :, None]


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("side", ["left", "right"])
def test_whole_thousand_frame_robot_paths_cross_poles_and_avoid_limit_drift(
    device: str, side: str
) -> None:
    solver = _solver(device, side)
    qpos = _paths(1000)
    local_goals = _fk(solver, qpos.reshape(-1, 7)).reshape(3, 1000, 4, 4)
    roots = torch.eye(4, dtype=torch.float64).repeat(3, 1, 1)
    roots[:, :3, :3] = torch.from_numpy(
        Rotation.from_euler(
            "xyz", [[0, 0, 0], [0.1, -0.2, 0.3], [-0.1, 0.2, -0.4]]
        ).as_matrix()
    )
    roots[:, :3, 3] = torch.tensor(
        [[0.2, -0.1, 0.4], [-0.3, 0.2, 0.1], [0.1, 0.3, -0.2]]
    )
    arena_goals = (roots[:, None] @ local_goals).to(solver.device)
    robot = object.__new__(Robot)
    robot.device, robot._all_indices, robot._solvers = (
        solver.device,
        [0, 1, 2],
        {"arm": solver},
    )
    roots = roots.float().to(solver.device)
    robot.get_link_pose = lambda **kwargs: roots[kwargs["env_ids"]]
    initial = qpos[:, 0].float().to(solver.device)
    valid, result = robot.compute_batch_ik(arena_goals, initial, "arm", continuous=True)
    assert valid.shape == (3, 1000) and bool(valid.all())
    assert result.shape == (3, 1000, 7)
    _assert_pose(local_goals.reshape(-1, 4, 4), _fk(solver, result.reshape(-1, 7)))
    delta = torch.diff(torch.cat((initial[:, None], result), dim=1), dim=1)
    assert bool((delta.abs() <= 0.05).all())
    assert bool((result >= solver.lower_qpos_limits).all())
    assert bool((result <= solver.upper_qpos_limits).all())


@pytest.mark.parametrize("device", _DEVICES)
def test_seed_local_nearest_and_all_expose_the_same_sampled_collection(
    device: str,
) -> None:
    solver = _solver(device)
    solver.cfg.max_joint_step = None
    qpos = _paths(5)[2]
    targets = _fk(solver, qpos).float().to(solver.device)
    seed = qpos[:1].float().to(solver.device).expand(5, -1)
    mask, candidates = solver.get_ik(targets, seed, return_all_solutions=True)
    assert mask.shape == (5, 8 * (solver.cfg.num_samples + 2))
    assert candidates.shape == (*mask.shape, 7)
    valid, nearest = solver.get_ik(targets, seed)
    assert bool(valid.all())
    torch.testing.assert_close(valid, mask.any(-1))
    cost = (
        (
            (candidates.double() - seed[:, None].double())
            * solver.ik_nearest_weight.double()
        )
        .square()
        .sum(-1)
        .masked_fill(~mask, torch.inf)
    )
    expected = candidates[torch.arange(5, device=solver.device), cost.argmin(-1)]
    torch.testing.assert_close(nearest, expected, atol=1e-6, rtol=0)
    _assert_pose(
        targets[:, None].expand(-1, mask.shape[1], -1, -1)[mask],
        _fk(solver, candidates[mask]),
    )


@pytest.mark.parametrize("device", _DEVICES)
def test_target_aware_path_equals_preceding_seed_streaming(device: str) -> None:
    solver = _solver(device)
    qpos = _paths(24)[2]
    targets = _fk(solver, qpos).float().to(solver.device)
    initial = qpos[:1].float().to(solver.device)
    valid, result = solver._get_continuous_ik_path(targets[None], initial)
    previous = initial
    flags, streaming = [], []
    for target in targets:
        flag, previous = solver.get_ik(target[None], previous)
        flags.append(flag[0])
        streaming.append(previous[0])
    torch.testing.assert_close(valid[0], torch.stack(flags))
    torch.testing.assert_close(result[0], torch.stack(streaming), atol=2e-4, rtol=0)
    assert bool(valid.all())


@pytest.mark.parametrize("device", _DEVICES)
def test_path_failures_hold_posture_and_later_valid_frames_recover(device: str) -> None:
    solver = _solver(device)
    initial = torch.tensor(_START[2], dtype=torch.float32)[None].to(solver.device)
    joints = initial.expand(6, -1).clone()
    joints[:, 0] += joints.new_tensor([0, 0.005, 0.01, 0.015, 0.02, 0.025])
    targets = _fk(solver, joints).float().to(solver.device)
    targets[2, :3, 3] = 100
    targets[3, :3, 0] *= -1
    valid, result = solver._get_continuous_ik_path(targets[None], initial)
    assert valid.tolist() == [[True, True, False, False, True, True]]
    torch.testing.assert_close(result[:, 2], result[:, 1], atol=0, rtol=0)
    torch.testing.assert_close(result[:, 3], result[:, 1], atol=0, rtol=0)
    _assert_pose(targets[valid[0]], _fk(solver, result[0, valid[0]]))
    saved = result.clone()
    solver._get_continuous_ik_path(targets[:1][None], initial)
    torch.testing.assert_close(result, saved, atol=0, rtol=0)


@pytest.mark.parametrize("device", _DEVICES)
def test_tolerance_boundary_uses_continuous_free_roll_and_original_tcp_goal(
    device: str,
) -> None:
    solver = _solver(device)
    # Frozen Cartesian-line frame664 lies about1 micrometre beyond the exact
    # radial shell. The former SO(3)-nearest free roll jumped q0 by0.083rad.
    seed = torch.tensor(
        [
            [
                -0.8551024794578552,
                0.010084564797580242,
                0.3812989294528961,
                -0.12346228957176208,
                -1.0922772884368896,
                -0.2205537110567093,
                -0.30408281087875366,
            ]
        ],
        device=solver.device,
    )
    target = torch.tensor(
        [
            [
                [
                    -0.8839234892560275,
                    -0.38512482809809523,
                    -0.26525107336987275,
                    0.872151336492351,
                ],
                [
                    0.3941366367101602,
                    -0.9188201413046051,
                    0.020636364403664856,
                    -0.0652852568785355,
                ],
                [
                    -0.2516656050084364,
                    -0.08630419871251163,
                    0.96395850976084,
                    1.2833552669996569,
                ],
                [0, 0, 0, 1],
            ]
        ],
        device=solver.device,
    )
    valid, result = solver.get_ik(target, seed)
    assert valid.tolist() == [True]
    assert bool(((result - seed).abs() <= 0.05).all())
    _assert_pose(target, _fk(solver, result))
    solver.cfg.position_tolerance = 1e-7
    valid, result = solver.get_ik(target, seed)
    assert valid.tolist() == [False]
    torch.testing.assert_close(result, seed, atol=0, rtol=0)


def test_seed_local_mode_is_explicit_and_roundtrippable() -> None:
    assert SSIKSolverCfg().srs_search == "uniform"
    cfg = SolverCfg.from_dict({"class_type": "SSIKSolver", "srs_search": "seed_local"})
    assert isinstance(cfg, SSIKSolverCfg) and cfg.srs_search == "seed_local"
    assert SolverCfg.from_dict(cfg.to_dict()).srs_search == "seed_local"
    with pytest.raises(ValueError, match="srs_search"):
        SSIKSolverCfg(srs_search="unknown")
