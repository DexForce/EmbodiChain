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

import os
from pathlib import Path
import torch
import pytest
import numpy as np

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.objects import Robot
from embodichain.lab.sim.cfg import RobotCfg, RenderCfg
from embodichain.lab.sim.motion.solvers import PytorchSolver, PytorchSolverCfg
from embodichain.data import get_data_path
from embodichain.utils.utility import reset_all_seeds


@pytest.mark.no_sim
@pytest.mark.parametrize(
    "tcp_kind", ["identity", "translation", "rotation_translation"]
)
def test_ik_removes_tcp_before_solving(monkeypatch, tcp_kind):
    """The numerical solver must receive flange poses, with the TCP removed."""
    solver = object.__new__(PytorchSolver)
    solver.device = torch.device("cpu")
    solver.dof = 1
    solver._num_samples = 1
    solver._is_only_position_constraint = False
    solver._seed_sampler = None
    solver.lower_qpos_limits = torch.tensor([-2.0])
    solver.upper_qpos_limits = torch.tensor([2.0])
    tcp = np.eye(4)
    if tcp_kind != "identity":
        tcp[0, 3] = 0.13  # Marvin's 13 cm tool offset.
    if tcp_kind == "rotation_translation":
        # +90 degrees about Y: unlike identity, this rotation is not symmetric.
        tcp[:3, :3] = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
    solver.set_tcp(tcp)

    flange_poses = torch.tensor(
        [
            [[1, 0, 0, 0.5], [0, 1, 0, 0.2], [0, 0, 1, 0.3], [0, 0, 0, 1]],
            [[0, -1, 0, -0.2], [1, 0, 0, 0.4], [0, 0, 1, 0.6], [0, 0, 0, 1]],
        ],
        dtype=torch.float32,
    )
    target = flange_poses @ torch.as_tensor(tcp, dtype=torch.float32)
    original_target = target.clone()
    received = []

    def capture_target(target_pose, joint_seed):
        received.append(target_pose.clone())
        # Stop at the numerical solver boundary; convergence is tested separately.
        return torch.zeros(len(joint_seed), dtype=torch.bool), joint_seed.clone()

    monkeypatch.setattr(solver, "_compute_inverse_kinematics", capture_target)
    for _ in range(2):
        solver.get_ik(target, qpos_seed=torch.zeros((2, 1)))
        torch.testing.assert_close(received[-1], flange_poses, atol=1e-7, rtol=0)
        torch.testing.assert_close(target, original_target, atol=0, rtol=0)
        np.testing.assert_array_equal(solver.get_tcp(), tcp)

    valid, candidates = solver.get_ik(
        target,
        qpos_seed=torch.zeros((2, 1)),
        return_all_solutions=True,
    )
    assert valid.shape == (2, 1)
    assert candidates.shape == (2, 1, 1)


@pytest.mark.no_sim
def test_continuous_path_selection_tracks_previous_solution() -> None:
    """Manipulability ordering cannot make a continuous path jump branches."""
    solver = object.__new__(PytorchSolver)
    solver.device = torch.device("cpu")
    solver.dof = 1
    solver.ik_nearest_weight = torch.ones(1)
    candidates = torch.tensor(
        [[[[0.1], [2.8]], [[2.9], [0.2]], [[0.3], [3.0]]]],
        dtype=torch.float32,
    )
    valid = torch.ones(1, 3, 2, dtype=torch.bool)

    path_valid, path = solver._select_continuous_ik_path(
        candidates, valid, torch.zeros(1, 1)
    )

    assert path_valid.all()
    torch.testing.assert_close(path, torch.tensor([[[0.1], [0.2], [0.3]]]))


@pytest.mark.no_sim
def test_return_all_solutions_preserves_candidate_validity(monkeypatch) -> None:
    solver = object.__new__(PytorchSolver)
    solver.device = torch.device("cpu")
    solver.dof = 1
    solver._num_samples = 2
    solver._is_only_position_constraint = False
    solver._seed_sampler = None
    solver.lower_qpos_limits = torch.tensor([-2.0])
    solver.upper_qpos_limits = torch.tensor([2.0])
    solver.set_tcp(np.eye(4))
    convergence = torch.tensor([True, False, False, True])
    solutions = torch.tensor([[0.1], [0.2], [0.3], [0.4]])
    monkeypatch.setattr(
        solver,
        "_compute_inverse_kinematics",
        lambda target_pose, joint_seed: (convergence, solutions),
    )

    valid, candidates = solver.get_ik(
        torch.eye(4).repeat(2, 1, 1),
        qpos_seed=torch.zeros(2, 1),
        return_all_solutions=True,
    )

    assert valid.tolist() == [[True, False], [False, True]]
    torch.testing.assert_close(candidates, solutions.reshape(2, 2, 1))


@pytest.mark.no_sim
def test_continuous_path_selection_holds_last_solution_on_failure() -> None:
    solver = object.__new__(PytorchSolver)
    solver.device = torch.device("cpu")
    solver.dof = 1
    solver.ik_nearest_weight = torch.ones(1)
    candidates = torch.tensor([[[[0.1]], [[float("nan")]], [[0.3]]]])
    valid = torch.ones(1, 3, 1, dtype=torch.bool)

    path_valid, path = solver._select_continuous_ik_path(
        candidates, valid, torch.zeros(1, 1)
    )

    assert path_valid.tolist() == [[True, False, True]]
    torch.testing.assert_close(path, torch.tensor([[[0.1], [0.1], [0.3]]]))


@pytest.fixture
def planar_cfg(tmp_path: Path) -> PytorchSolverCfg:
    """A two-link planar arm cannot satisfy a rotation about X."""
    source = tmp_path / "planar.urdf"
    source.write_text(
        '<robot name="planar"><link name="base"/><link name="first"/>'
        '<link name="second"/><link name="tip"/>'
        '<joint name="a" type="revolute"><parent link="base"/>'
        '<child link="first"/><axis xyz="0 0 1"/>'
        '<limit lower="-3" upper="3" effort="1" velocity="1"/></joint>'
        '<joint name="b" type="revolute"><parent link="first"/>'
        '<child link="second"/><origin xyz="1 0 0"/><axis xyz="0 0 1"/>'
        '<limit lower="-3" upper="3" effort="1" velocity="1"/></joint>'
        '<joint name="tool" type="fixed"><parent link="second"/>'
        '<child link="tip"/><origin xyz="1 0 0"/></joint></robot>'
    )
    return PytorchSolverCfg(
        urdf_path=str(source),
        root_link_name="base",
        end_link_name="tip",
        joint_names=["a", "b"],
        is_only_position_constraint=True,
        pos_eps=1e-4,
        rot_eps=1e-4,
        max_iterations=120,
        dt=0.5,
        num_samples=1,
    )


def _planar_fk(qpos: torch.Tensor, tcp: np.ndarray) -> torch.Tensor:
    """Independent closed-form FK for two unit-length links and a fixed tool."""
    first, total = qpos[:, 0], qpos.sum(dim=-1)
    pose = torch.eye(4, device=qpos.device).repeat(len(qpos), 1, 1)
    pose[:, 0, 0] = pose[:, 1, 1] = total.cos()
    pose[:, 1, 0] = total.sin()
    pose[:, 0, 1] = -total.sin()
    pose[:, 0, 3] = first.cos() + total.cos()
    pose[:, 1, 3] = first.sin() + total.sin()
    return pose @ torch.as_tensor(tcp, dtype=qpos.dtype, device=qpos.device)


@pytest.mark.no_sim
@pytest.mark.parametrize(
    "with_tcp,seed_selection,device",
    [
        (False, False, "cpu"),
        (True, False, "cpu"),
        (True, True, "cpu"),
        pytest.param(True, False, "cuda", marks=pytest.mark.gpu),
    ],
)
def test_position_only_dls_ignores_orientation_and_rejects_unreachable_position(
    planar_cfg: PytorchSolverCfg, with_tcp: bool, seed_selection: bool, device: str
) -> None:
    tcp = np.eye(4)
    if with_tcp:
        tcp[:3, 3] = [0.25, 0.08, 0.04]
        tcp[:3, :3] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
    planar_cfg.tcp = tcp
    planar_cfg.enable_seed_selection = seed_selection
    planar_cfg.seed_db_size = 128
    planar_cfg.num_samples = 3 if seed_selection else 1
    solver = planar_cfg.init_solver(device=device)
    seeds = torch.tensor([[0.2, -0.4], [-0.2, 0.5], [0.2, -0.4]], device=device)
    target = _planar_fk(
        torch.tensor([[0.6, -0.7], [-0.4, 0.9], [0.6, -0.7]], device=device), tcp
    )
    target[:, :3, :3] = torch.diag(torch.tensor([1.0, -1.0, -1.0]))
    # Every configuration has the same TCP z; this third position is unreachable.
    target[2, 2, 3] += 1.0
    original = target.clone()
    valid, candidates = solver.get_ik(target, seeds, return_all_solutions=True)
    assert valid.shape == (3, planar_cfg.num_samples)
    assert candidates.shape == (3, planar_cfg.num_samples, 2)
    assert valid.any(dim=1).tolist() == [True, True, False]
    actual = _planar_fk(candidates.reshape(-1, 2), tcp).reshape(3, -1, 4, 4)
    error = torch.linalg.vector_norm(
        actual[..., :3, 3] - target[:, None, :3, 3], dim=-1
    )
    assert (error[valid] < planar_cfg.pos_eps).all()
    assert (candidates[valid].abs() <= 3.0).all()
    torch.testing.assert_close(target, original, atol=0, rtol=0)

    # Identical positions and seeds must give identical answers for another rotation.
    target[:, :3, :3] = torch.eye(3)
    other_valid, other_candidates = solver.get_ik(
        target, seeds, return_all_solutions=True
    )
    assert torch.equal(valid, other_valid)
    torch.testing.assert_close(candidates, other_candidates, atol=1e-6, rtol=0)


@pytest.mark.no_sim
def test_position_only_dls_updates_tcp_and_runtime_constraint_mode(
    planar_cfg: PytorchSolverCfg,
) -> None:
    solver = planar_cfg.init_solver(device="cpu")
    seed = torch.tensor([0.2, -0.4])
    goal = torch.tensor([[0.6, -0.7]])
    tcp = np.eye(4)
    for offset in ([0.25, 0.08, 0.04], [-0.1, 0.15, -0.02]):
        tcp[:3, 3] = offset
        solver.set_tcp(tcp)
        target = _planar_fk(goal, tcp)[0]
        target[:3, :3] = torch.diag(torch.tensor([1.0, -1.0, -1.0]))
        valid, qpos = solver.get_ik(target, seed)
        assert valid.tolist() == [True]
        assert qpos.shape == (1, 1, 2)
        actual = _planar_fk(qpos.reshape(1, 2), tcp)[0]
        assert (actual[:3, 3] - target[:3, 3]).norm() < planar_cfg.pos_eps

    params = solver.get_iteration_params()
    assert params["is_only_position_constraint"] is True
    assert solver.set_iteration_params(**params)
    assert solver.get_iteration_params() == params
    params["is_only_position_constraint"] = False
    assert solver.set_iteration_params(**params)
    # Full-pose mode must still reject the same unreachable orientation.
    valid, _ = solver.get_ik(target, seed)
    assert not valid.any()
    full_target = _planar_fk(goal, tcp)
    valid, qpos = solver.get_ik(full_target, seed)
    assert valid.all(), (qpos, solver.pik.err_all, solver.get_iteration_params())
    torch.testing.assert_close(
        _planar_fk(qpos.reshape(1, 2), tcp), full_target, atol=3e-4, rtol=0
    )
    params["is_only_position_constraint"] = True
    assert solver.set_iteration_params(**params)
    valid, _ = solver.get_ik(target, seed)
    assert valid.all()


def grid_sample_qpos_from_limits(
    qpos_limits: torch.Tensor,
    steps_per_joint: int = 4,
    device=None,
    max_samples: int = 4096,
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
    lows = limits[:, 0].to(device) + 1e-2
    highs = limits[:, 1].to(device) - 1e-2

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


# Base test class for CPU and CUDA
class BaseSolverTest:
    sim = None  # Define as a class attribute

    def setup_simulation(self, solver_type: str):
        # Set up simulation with specified device (CPU or CUDA)
        config = SimulationManagerCfg(headless=True, device="cpu")
        self.sim = SimulationManager(config)

        # Load robot URDF file
        urdf = get_data_path("DexforceW1V021/DexforceW1_v02_1.urdf")
        assert os.path.isfile(urdf)

        cfg_dict = {
            "fpath": urdf,
            "control_parts": {
                "left_arm": [f"LEFT_J{i+1}" for i in range(7)],
                "right_arm": [f"RIGHT_J{i+1}" for i in range(7)],
            },
            "solver_cfg": {
                "left_arm": {
                    "class_type": solver_type,
                    "end_link_name": "left_ee",
                    "root_link_name": "left_arm_base",
                    "ik_nearest_weight": [1.0, 1.0, 1.0, 0.9, 0.9, 0.1, 0.1],
                    "num_samples": 30,
                },
                "right_arm": {
                    "class_type": solver_type,
                    "end_link_name": "right_ee",
                    "root_link_name": "right_arm_base",
                    "num_samples": 30,
                },
            },
        }

        self.robot: Robot = self.sim.add_robot(cfg=RobotCfg.from_dict(cfg_dict))
        self.sim.prepare()

        # Wait for robot to stabilize.
        self.sim.update(step=100)

    @pytest.mark.parametrize("arm_name", ["left_arm", "right_arm"])
    def test_ik(self, arm_name: str):
        reset_all_seeds(0)
        qpos_limit = torch.tensor(
            [
                [0.2, 0.8],
                [0.2, 0.8],
                [0.2, 0.8],
                [0.2, 0.8],
                [0.2, 0.8],
                [0.2, 0.8],
                [0.2, 0.8],
            ]
        )
        # generate a small grid of qpos samples from the joint limits (low/high)
        sample_qpos = grid_sample_qpos_from_limits(
            qpos_limit, steps_per_joint=3, device=self.robot.device, max_samples=200
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

        ik_xpos = self.robot.compute_batch_fk(
            qpos=ik_qpos_xyzquat, name=arm_name, to_matrix=True
        )

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
        assert res[0].item() == False
        assert ik_qpos.shape == (1, dof)

    def teardown_method(self):
        """Clean up resources after each test method."""
        self.sim.destroy()
        SimulationManager.flush_cleanup_queue()


class TestPytorchSolver(BaseSolverTest):
    def setup_method(self):
        self.setup_simulation(solver_type="PytorchSolver")


if __name__ == "__main__":
    np.set_printoptions(precision=5, suppress=True)
    test_solver = TestPytorchSolver()
    test_solver.setup_method()
