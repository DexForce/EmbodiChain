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

import pytest
import torch
import warp as wp

from embodichain.lab.sim.robots import CobotMagicCfg, URRobotCfg
from embodichain.lab.sim.objects.robot import Robot
from embodichain.lab.sim.motion.solvers._buffers import _IKBuffers


@pytest.fixture(params=["ur5", "cobotmagic"])
def solver(request):
    """Construct real analytic solvers and asset chains without a renderer."""
    wp.init()
    preset = (
        URRobotCfg.from_dict({"robot_type": "ur5"})
        if request.param == "ur5"
        else CobotMagicCfg.from_dict({})
    )
    part = "arm" if request.param == "ur5" else "left_arm"
    device = torch.device(getattr(request, "param_device", "cpu"))
    chain = preset.build_pk_serial_chain(device)[part]
    cfg = preset.solver_cfg[part]
    cfg.joint_names = chain.get_joint_parameter_names()
    result = cfg.init_solver(device=device, pk_serial_chain=chain)
    result.compiled_fk = chain.forward_kinematics_tensor
    return result


def _analytic_fk(solver, qpos):
    """Evaluate the analytic solver's geometry, independently of asset FK."""
    if hasattr(solver, "get_fk_warp"):
        return solver.get_fk_warp(qpos)
    cfg = solver.cfg
    result = torch.eye(4, device=qpos.device).expand(len(qpos), 4, 4).clone()
    params = [
        (cfg.d1, 0.0, torch.pi / 2),
        (0.0, cfg.a2, 0.0),
        (0.0, cfg.a3, 0.0),
        (cfg.d4, 0.0, torch.pi / 2),
        (cfg.d5, 0.0, -torch.pi / 2),
        (cfg.d6, 0.0, 0.0),
    ]
    for joint, (d, a, alpha) in enumerate(params):
        theta = qpos[:, joint]
        ct, st = torch.cos(theta), torch.sin(theta)
        ca, sa = torch.cos(theta.new_tensor(alpha)), torch.sin(theta.new_tensor(alpha))
        transform = torch.eye(4, device=qpos.device).expand(len(qpos), 4, 4).clone()
        transform[:, 0, :] = torch.stack((ct, -st * ca, st * sa, a * ct), 1)
        transform[:, 1, :] = torch.stack((st, ct * ca, -ct * sa, a * st), 1)
        transform[:, 2, :] = theta.new_tensor([0.0, sa, ca, d])
        result = result @ transform
    return result @ torch.as_tensor(
        solver.tcp_xpos, dtype=qpos.dtype, device=qpos.device
    )


def _targets(solver, count):
    generator = torch.Generator(device=solver.device).manual_seed(13)
    lower, upper = solver.lower_qpos_limits, solver.upper_qpos_limits
    qpos = lower + (
        0.1 + 0.8 * torch.rand(count, 6, generator=generator, device=solver.device)
    ) * (upper - lower)
    # OPW's analytic geometry has a pre-existing difference from the packaged
    # URDF chain; use its own FK to test analytic round trips, independently of sim.
    fk = lambda q: _analytic_fk(solver, q)
    return qpos, fk(qpos)


def test_candidate_reuse_and_old_results_remain_valid(solver, monkeypatch):
    qpos, poses = _targets(solver, 17)
    solver.prepare_buffers(32)
    original_zeros = solver._ik_buffers.zeros
    allocations = []
    for count in [9, 3, 17, 1]:
        valid, result = solver.get_ik(
            poses[:count], qpos[:count], return_all_solutions=True
        )
        old_valid, old_result = valid.clone(), result.clone()
        ptr = solver._ik_buffers.arrays["qpos"].data_ptr()
        allocations.append(ptr)
        bad_pose = poses[:1].clone()
        bad_pose[:, :3, 3] = 100
        invalid, _ = solver.get_ik(bad_pose, qpos[:1])
        assert not invalid.bool().any()
        torch.testing.assert_close(valid, old_valid)
        torch.testing.assert_close(result, old_result)

        # Allocation-only reference: fresh zeroed buffers, identical unchanged kernels.
        def fresh(name, batch, width, dtype):
            return wp.from_torch(
                torch.zeros(batch * width, dtype=dtype, device=solver.device)
            )

        with monkeypatch.context() as patch:
            patch.setattr(solver._ik_buffers, "zeros", fresh)
            expected_valid, expected_result = solver.get_ik(
                poses[:count], qpos[:count], return_all_solutions=True
            )
        torch.testing.assert_close(valid, expected_valid)
        torch.testing.assert_close(result, expected_result)
    assert len(set(allocations)) == 1
    solver._ik_buffers.zeros = original_zeros


def test_selected_results_remain_valid_after_scratch_reuse(solver):
    """Selected IK outputs must not alias reusable candidate storage."""
    qpos, poses = _targets(solver, 7)
    valid, result = solver.get_ik(poses, qpos)
    expected_valid, expected_result = valid.clone(), result.clone()

    solver.get_ik(poses.flip(0), qpos.flip(0))

    torch.testing.assert_close(valid, expected_valid)
    torch.testing.assert_close(result, expected_result)


def test_best_solution_round_trip_and_limits(solver):
    qpos, poses = _targets(solver, 13)
    success, result = solver.get_ik_batch(
        poses.reshape(1, 13, 4, 4), qpos.reshape(1, 13, 6)
    )
    assert success.dtype == torch.bool
    assert success.shape == (1, 13)
    assert result.shape == (1, 13, 6)
    assert success.all()
    old = result.clone()
    solver.get_ik(poses[:2], qpos[:2])
    torch.testing.assert_close(result, old)
    fk = lambda q: _analytic_fk(solver, q)
    torch.testing.assert_close(fk(result.reshape(-1, 6)), poses, atol=3e-5, rtol=3e-5)
    assert (result >= solver.lower_qpos_limits - 1e-6).all()
    assert (result <= solver.upper_qpos_limits + 1e-6).all()
    solver.set_qpos_limits(qpos[0] - 0.01, qpos[0] + 0.01)
    valid, out = solver.get_ik(poses[:1], qpos[:1])
    assert valid.bool().all()
    assert (out >= solver.lower_qpos_limits - 1e-6).all()
    assert (out <= solver.upper_qpos_limits + 1e-6).all()


def test_robot_batch_transforms_broadcast_and_preserve_seed_shape(solver):
    qpos, _ = _targets(solver, 6)
    qpos = qpos.reshape(2, 3, 6)
    bases = torch.eye(4).repeat(2, 1, 1)
    bases[0, :3, 3] = torch.tensor([0.2, -0.3, 0.4])
    bases[1, :3, :3] = torch.tensor([[0.0, -1, 0], [1.0, 0, 0], [0, 0, 1]])
    bases[1, :3, 3] = torch.tensor([-0.4, 0.1, 0.5])
    robot = SimpleNamespace(
        _all_indices=[0, 1],
        _solvers={"arm": solver},
        device=solver.device,
        get_link_pose=lambda **kwargs: bases[kwargs["env_ids"]],
    )
    actual = Robot.compute_batch_fk(robot, qpos, "arm", to_matrix=True)
    expected = torch.stack([bases[i] @ solver.get_fk(qpos[i]) for i in range(2)])
    torch.testing.assert_close(actual, expected)
    # IK input uses the analytic FK, including the same per-environment roots.
    fk = lambda q: _analytic_fk(solver, q)
    targets = torch.stack([bases[i] @ fk(qpos[i]) for i in range(2)])
    valid, result = Robot.compute_batch_ik(robot, targets, qpos, "arm")
    assert valid.shape == (2, 3) and valid.all()
    torch.testing.assert_close(
        fk(result.reshape(-1, 6)), fk(qpos.reshape(-1, 6)), atol=3e-5, rtol=3e-5
    )
    with pytest.raises(ValueError, match="batch axes"):
        solver.get_ik_batch(targets, qpos[:, :1])


def test_scratch_growth_zeroes_active_tail():
    wp.init()
    buffers = _IKBuffers(torch.device("cpu"))
    with buffers.borrow():
        first = buffers.zeros("q", 4, 6, torch.float32)
        wp.to_torch(first).fill_(7)
    with buffers.borrow():
        assert not wp.to_torch(buffers.zeros("q", 2, 6, torch.float32)).any()
    with buffers.borrow():
        assert not wp.to_torch(buffers.zeros("q", 7, 6, torch.float32)).any()


@pytest.mark.gpu
@pytest.mark.parametrize("kind", ["ur5", "cobotmagic"])
def test_cuda_stream_reuse(kind):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    wp.init()
    preset = (
        URRobotCfg.from_dict({"robot_type": "ur5"})
        if kind == "ur5"
        else CobotMagicCfg.from_dict({})
    )
    part = "arm" if kind == "ur5" else "left_arm"
    device = torch.device("cuda:0")
    chain = preset.build_pk_serial_chain(device)[part]
    cfg = preset.solver_cfg[part]
    cfg.joint_names = chain.get_joint_parameter_names()
    solver = cfg.init_solver(device=device, pk_serial_chain=chain)
    solver.compiled_fk = chain.forward_kinematics_tensor
    qpos, poses = _targets(solver, 19)
    torch.cuda.synchronize()
    outputs = []
    for count in [7, 19, 3, 11]:
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            outputs.append(
                (
                    count,
                    solver.get_ik(
                        poses[:count], qpos[:count], return_all_solutions=True
                    ),
                )
            )
    torch.cuda.synchronize()
    for count, (valid, result) in outputs:
        expected_valid, expected_result = solver.get_ik(
            poses[:count], qpos[:count], return_all_solutions=True
        )
        torch.testing.assert_close(valid, expected_valid)
        torch.testing.assert_close(result, expected_result)
