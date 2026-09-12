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

import torch
import pytest
import numpy as np
import warp as wp

from embodichain.data import get_data_path
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.objects import Robot
from embodichain.lab.sim.motion.solvers import URSolver, URSolverCfg
from embodichain.lab.sim.cfg import (
    RenderCfg,
    JointDrivePropertiesCfg,
    RobotCfg,
    LightCfg,
    RigidBodyAttributesCfg,
    RigidObjectCfg,
    URDFCfg,
)


def _make_analytic_ur_solver(
    device: str = "cpu",
    weights: list[float] | None = None,
    limits: torch.Tensor | None = None,
) -> URSolver:
    """Build the analytical solver without loading robot assets or a simulator."""
    wp.init()
    if limits is None:
        limits = torch.tensor([[-2.0 * torch.pi, 2.0 * torch.pi]] * 6)
    cfg = URSolverCfg(
        ur_type="ur5",
        joint_names=[f"joint_{index}" for index in range(6)],
        ik_nearest_weight=weights,
        user_qpos_limits=limits.tolist(),
    )
    # The analytical IK path never accesses the serial chain; explicit limits
    # bypass the only chain-dependent setup needed by these tests.
    return cfg.init_solver(device=torch.device(device), pk_serial_chain=object())


def _ur_dh_poses(solver: URSolver, qpos: torch.Tensor) -> torch.Tensor:
    """Generate independent DH targets, including the configured TCP."""
    cfg = solver.cfg
    dh_parameters = [
        (cfg.d1, 0.0, cfg.alpha1),
        (0.0, cfg.a2, 0.0),
        (0.0, cfg.a3, 0.0),
        (cfg.d4, 0.0, cfg.alpha4),
        (cfg.d5, 0.0, cfg.alpha5),
        (cfg.d6, 0.0, 0.0),
    ]
    poses = []
    for joints in qpos.cpu():
        pose = torch.eye(4)
        for theta, (d, a, alpha) in zip(joints, dh_parameters):
            pose = pose @ URSolver.dh_matrix(theta, d, a, alpha)
        poses.append(pose)
    tcp = torch.as_tensor(solver.tcp_xpos, dtype=torch.float32)
    return (torch.stack(poses) @ tcp).to(solver.device)


def _sample_ur_joints(count: int = 12) -> torch.Tensor:
    generator = torch.Generator().manual_seed(712)
    joints = torch.rand((count, 6), generator=generator) * 5.0 - 2.5
    # Keep the wrist away from a singularity while covering both wrist branches.
    joints[:, 4] = joints[:, 4].sign() * (0.4 + joints[:, 4].abs() * 0.7)
    return joints


def _assert_nearest_matches_all_solutions(
    solver: URSolver,
    poses: torch.Tensor,
    seed: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    all_valid, all_qpos = solver.get_ik(poses, seed, return_all_solutions=True)
    batch_size = 1 if poses.ndim == 2 else len(poses)
    assert all_valid.shape == (batch_size, 512)
    assert all_qpos.shape == (batch_size, 512, 6)
    reference_seed = solver.get_default_qpos_seed()[None] if seed is None else seed
    distances = torch.norm(
        solver.ik_nearest_weight * (all_qpos - reference_seed[:, None, :]), dim=-1
    )
    distances[~all_valid] = float("inf")
    indices = distances.argmin(dim=1)
    rows = torch.arange(batch_size, device=solver.device)
    valid, qpos = solver.get_ik(poses, seed, return_all_solutions=False)
    assert valid.shape == (batch_size,)
    assert qpos.shape == (batch_size, 6)
    assert valid.dtype == torch.bool
    assert qpos.dtype == torch.float32
    assert valid.device == solver.device
    assert qpos.device == solver.device
    torch.testing.assert_close(valid, all_valid[rows, indices])
    torch.testing.assert_close(
        qpos, all_qpos[rows, indices], rtol=0.0, atol=2e-6, equal_nan=True
    )
    return valid, qpos, all_valid, all_qpos


@pytest.mark.no_sim
@pytest.mark.parametrize("seed_dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "weights",
    [None, [0.0, 0.25, -2.0, 4.0, 0.5, 1.5]],
    ids=["uniform", "nonuniform-with-zero-and-negative"],
)
def test_ur_nearest_matches_weighted_full_candidates(
    weights: list[float] | None, seed_dtype: torch.dtype
) -> None:
    solver = _make_analytic_ur_solver(weights=weights)
    joints = _sample_ur_joints()
    poses = _ur_dh_poses(solver, joints)
    seed = joints.roll(1, dims=0).to(seed_dtype)
    seed[::2] += 2.0 * torch.pi
    seed[1::2] -= 2.0 * torch.pi
    valid, _, _, _ = _assert_nearest_matches_all_solutions(solver, poses, seed)
    assert valid.all()


@pytest.mark.no_sim
def test_ur_nearest_preserves_periodic_representatives() -> None:
    solver = _make_analytic_ur_solver()
    joints = torch.tensor([[0.3, -1.1, 1.4, -0.5, 0.9, 0.4]])
    shifted = joints - 2.0 * torch.pi * joints.sign()
    mixed = joints.clone()
    mixed[:, ::2] = shifted[:, ::2]
    seed = torch.cat([joints, shifted, mixed])
    poses = _ur_dh_poses(solver, joints).repeat(3, 1, 1)
    valid, qpos, _, _ = _assert_nearest_matches_all_solutions(solver, poses, seed)
    assert valid.all()
    torch.testing.assert_close(qpos, seed, rtol=0.0, atol=5e-5)


@pytest.mark.no_sim
def test_ur_nearest_respects_limits_requiring_periodic_shifts() -> None:
    joints = torch.tensor([[0.4, -1.1, 1.3, -0.7, 0.8, 0.2]])
    shifted = joints.clone()
    shifted[0, 0] += 2.0 * torch.pi
    shifted[0, 3] -= 2.0 * torch.pi
    limits = torch.stack([shifted[0] - 0.08, shifted[0] + 0.08], dim=1)
    solver = _make_analytic_ur_solver(limits=limits)
    poses = _ur_dh_poses(solver, joints)
    valid, qpos, _, _ = _assert_nearest_matches_all_solutions(solver, poses, joints)
    assert valid.all()
    assert (qpos >= limits[:, 0]).all()
    assert (qpos <= limits[:, 1]).all()
    torch.testing.assert_close(qpos, shifted, rtol=0.0, atol=5e-5)


@pytest.mark.no_sim
def test_ur_nearest_zero_weights_keep_first_valid_candidate_on_ties() -> None:
    solver = _make_analytic_ur_solver(weights=[0.0] * 6)
    poses = _ur_dh_poses(solver, _sample_ur_joints(3))
    seed = torch.full((1, 6), 2.0)  # A single seed broadcasts across targets.
    valid, qpos, all_valid, all_qpos = _assert_nearest_matches_all_solutions(
        solver, poses, seed
    )
    assert valid.all()
    assert (all_valid.sum(dim=1) > 1).all()
    first_indices = all_valid.to(torch.int32).argmax(dim=1)
    torch.testing.assert_close(qpos, all_qpos[torch.arange(3), first_indices])


@pytest.mark.no_sim
@pytest.mark.parametrize("seed_value", [float("nan"), float("inf"), 1e30])
def test_ur_nearest_preserves_nonfinite_distance_selection(seed_value: float) -> None:
    joints = torch.tensor([[0.4, -1.1, 1.3, -0.7, 0.8, 0.2]])
    # Exclude the unshifted first candidate to exercise validity together with
    # argmin's first-NaN/first-infinity behavior, including finite overflow.
    shifted = joints[0].clone()
    shifted[0] += 2.0 * torch.pi
    limits = torch.stack([shifted - 0.08, shifted + 0.08], dim=1)
    solver = _make_analytic_ur_solver(limits=limits)
    poses = _ur_dh_poses(solver, joints)
    seed = torch.full_like(joints, seed_value)
    _assert_nearest_matches_all_solutions(solver, poses, seed)


@pytest.mark.no_sim
def test_ur_nearest_broadcasts_singleton_seed_across_targets() -> None:
    solver = _make_analytic_ur_solver(weights=[1.0, 0.5, 2.0, 4.0, 0.25, 1.5])
    poses = _ur_dh_poses(solver, _sample_ur_joints(3))
    seed = torch.tensor([[3.0, -4.0, 5.0, 1.0, -2.0, 4.0]], dtype=torch.float64)
    valid, _, _, _ = _assert_nearest_matches_all_solutions(solver, poses, seed)
    assert valid.all()


@pytest.mark.no_sim
def test_ur_nearest_default_seed_unbatched_pose_and_tcp() -> None:
    solver = _make_analytic_ur_solver()
    solver.set_tcp(
        np.array(
            [
                [0.0, -1.0, 0.0, 0.04],
                [1.0, 0.0, 0.0, -0.02],
                [0.0, 0.0, 1.0, 0.12],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
    )
    pose = _ur_dh_poses(solver, _sample_ur_joints(1))[0]
    valid, _, _, _ = _assert_nearest_matches_all_solutions(solver, pose, None)
    assert valid.all()


@pytest.mark.no_sim
@pytest.mark.parametrize("failure", ["unreachable", "excluded-by-limits"])
def test_ur_nearest_all_invalid_preserves_first_candidate(failure: str) -> None:
    solver = _make_analytic_ur_solver()
    poses = _ur_dh_poses(solver, _sample_ur_joints(2))
    if failure == "unreachable":
        poses[:, :3, 3] = 10.0
    else:
        solver.set_qpos_limits(torch.zeros(6), torch.full((6,), 0.01))
    valid, qpos, all_valid, all_qpos = _assert_nearest_matches_all_solutions(
        solver, poses, None
    )
    assert not valid.any()
    assert not all_valid.any()
    torch.testing.assert_close(qpos, all_qpos[:, 0], equal_nan=True)


@pytest.mark.no_sim
def test_ur_nearest_avoids_full_candidate_buffers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    solver = _make_analytic_ur_solver()
    joints = _sample_ur_joints(3)
    poses = _ur_dh_poses(solver, joints)
    # Warm up before observing the per-call allocations.
    solver.get_ik(poses, joints)
    allocation_sizes = []

    def record_allocations(allocate):
        def wrapped(*args, **kwargs):
            array = allocate(*args, **kwargs)
            allocation_sizes.append(array.size)
            return array

        return wrapped

    for allocator in ("zeros", "empty"):
        monkeypatch.setattr(wp, allocator, record_allocations(getattr(wp, allocator)))
    valid, qpos = solver.get_ik(poses, joints, return_all_solutions=False)
    assert valid.all()
    assert qpos.shape == joints.shape
    # Even the smallest full-candidate buffer contains N * 512 scalar values.
    assert all(size < len(joints) * 512 for size in allocation_sizes)


@pytest.mark.no_sim
@pytest.mark.gpu
@pytest.mark.parametrize("seed_dtype", [torch.float32, torch.float64])
def test_ur_nearest_cuda_matches_full_candidates(seed_dtype: torch.dtype) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    solver = _make_analytic_ur_solver(
        device="cuda:0", weights=[0.0, 0.25, -2.0, 4.0, 0.5, 1.5]
    )
    joints = _sample_ur_joints(4).to(solver.device)
    poses = _ur_dh_poses(solver, joints)
    seed = (joints - 2.0 * torch.pi * joints.sign()).to(seed_dtype)
    valid, _, _, _ = _assert_nearest_matches_all_solutions(solver, poses, seed)
    assert valid.all()


def grid_sample_qpos_from_limits(
    qpos_limits: torch.Tensor,
    steps_per_joint: int = 4,
    device=None,
    max_samples: int = 4096,
    safe_margin: float = 5 / 180 * np.pi,  # 5 degrees in radians
) -> torch.Tensor:
    """Generate grid samples for qpos from qpos_limits.

    Args:
        qpos_limits: tensor of shape (1, n, 2) or (n, 2) where each row is [low, high].
        steps_per_joint: number of values per joint (defaults to 2: low and high).
        device: torch device to place the samples on.
        max_samples: cap the number of returned samples (take first N if grid is larger).

    Returns:
        Tensor of shape (N, n) where N <= max_samples.
    """
    if device is None:
        device = qpos_limits.device

    limits = qpos_limits.squeeze(0) if qpos_limits.dim() == 3 else qpos_limits
    lows = limits[:, 0].to(device) + safe_margin * 1.01
    highs = limits[:, 1].to(device) - safe_margin * 1.01

    # create per-joint linspaces
    grids = [
        torch.linspace(l.item(), h.item(), steps_per_joint, device=device)
        for l, h in zip(lows, highs)
    ]

    # meshgrid and stack
    mesh = torch.meshgrid(*grids, indexing="ij")
    stacked = torch.stack([m.reshape(-1) for m in mesh], dim=1)

    if stacked.shape[0] > max_samples:
        return stacked[:max_samples]
    return stacked


# Base test class for OPWSolver
class BaseSolverTest:
    sim = None  # Define as a class attribute

    def setup_simulation(self, sim_device):
        config = SimulationManagerCfg(headless=True, sim_device=sim_device)
        self.sim = SimulationManager(config)

        ur10_urdf_path = get_data_path("UniversalRobots/UR10/UR10.urdf")
        gripper_urdf_path = get_data_path("DH_PGC_140_50_M/DH_PGC_140_50_M.urdf")
        # Configure the robot with its components and control properties
        cfg = RobotCfg(
            uid="UR10",
            urdf_cfg=URDFCfg(
                components=[
                    {"component_type": "arm", "urdf_path": ur10_urdf_path},
                    {"component_type": "hand", "urdf_path": gripper_urdf_path},
                ]
            ),
            drive_pros=JointDrivePropertiesCfg(
                stiffness={"Joint[0-9]": 1e4, "FINGER[1-2]": 1e3},
                damping={"Joint[0-9]": 1e3, "FINGER[1-2]": 1e2},
                max_effort={"Joint[0-9]": 1e5, "FINGER[1-2]": 1e4},
                drive_type="force",
            ),
            control_parts={
                "arm": ["Joint[0-9]"],
                "hand": ["FINGER[1-2]"],
            },
            solver_cfg={
                "arm": URSolverCfg(
                    ur_type="ur10",
                    tcp=[
                        [0.0, 1.0, 0.0, 0.0],
                        [-1.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.12],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                )
            },
            init_qpos=[
                0.0,
                -np.pi / 2,
                -np.pi / 2,
                np.pi / 2,
                -np.pi / 2,
                0.0,
                0.0,
                0.0,
            ],
            init_pos=(0, 0, 0),
        )
        self.robot: Robot = self.sim.add_robot(cfg=cfg)

    def test_ik(self):
        # Test inverse kinematics (IK) with a 1x4x4 homogeneous matrix pose and a joint_seed
        arm_name = "arm"
        # generate a small grid of qpos samples from the joint limits (low/high)
        qpos_limit = self.robot.get_qpos_limits(name=arm_name)
        sample_qpos = grid_sample_qpos_from_limits(
            qpos_limit, steps_per_joint=8, device=self.robot.device, max_samples=65536
        )
        sample_qpos = sample_qpos[None, :, :]
        fk_xpos = self.robot.compute_batch_fk(
            qpos=sample_qpos, name=arm_name, to_matrix=True
        )
        fk_xpos_xyzquat = self.robot.compute_batch_fk(
            qpos=sample_qpos, name=arm_name, to_matrix=False
        )

        res, ik_qpos = self.robot.compute_batch_ik(
            pose=fk_xpos, joint_seed=sample_qpos, name=arm_name
        )

        res, ik_qpos_xyzquat = self.robot.compute_batch_ik(
            pose=fk_xpos_xyzquat, joint_seed=sample_qpos, name=arm_name
        )

        assert torch.allclose(
            ik_qpos, ik_qpos_xyzquat, atol=5e-3, rtol=5e-3
        ), "IK results do not match for different pose formats"

        ik_xpos = self.robot.compute_batch_fk(
            qpos=ik_qpos_xyzquat, name=arm_name, to_matrix=True
        )
        assert torch.allclose(
            sample_qpos, ik_qpos, atol=5e-3, rtol=5e-3
        ), f"FK and IK qpos do not match for {arm_name}"

        assert torch.allclose(
            fk_xpos, ik_xpos, atol=5e-3, rtol=5e-3
        ), f"FK and IK xpos do not match for {arm_name}"
        # test for failed xpos
        invalid_pose = torch.tensor(
            [
                [
                    [1.0, 0.0, 0.0, 10.0],
                    [0.0, 1.0, 0.0, 10.0],
                    [0.0, 0.0, 1.0, 10.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ],
            dtype=torch.float32,
            device=self.robot.device,
        )
        res, ik_qpos = self.robot.compute_ik(
            pose=invalid_pose, joint_seed=ik_qpos[:, 0, :], name=arm_name
        )
        dof = ik_qpos.shape[-1]
        assert res[0] == False
        assert ik_qpos.shape == (1, dof)

    def teardown_method(self):
        """Clean up resources after each test method."""
        self.sim.destroy()
        SimulationManager.flush_cleanup_queue()


class TestURSolverCUDA(BaseSolverTest):
    def setup_method(self):
        self.setup_simulation("cuda")


class TestURSolver(BaseSolverTest):
    def setup_method(self):
        self.setup_simulation("cpu")


if __name__ == "__main__":
    np.set_printoptions(precision=5, suppress=True)
    pytest_args = ["-v", "-s", __file__]
    pytest.main(pytest_args)
