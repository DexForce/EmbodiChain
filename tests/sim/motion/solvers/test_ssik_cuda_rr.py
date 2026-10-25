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

"""Public CUDA RR behavior on the unchanged original Rizon4S URDF."""

from __future__ import annotations

import sys

import numpy as np
import pytest
import torch

from embodichain.lab.sim.motion.solvers import SSIKSolverCfg
from embodichain_robots import FlexivRizon4SCfg


def _make_solver(**kwargs):
    preset = FlexivRizon4SCfg.from_dict({})
    original = preset.solver_cfg["arm"]
    return SSIKSolverCfg(
        backend="cuda_rr",
        urdf_path=preset.fpath,
        root_link_name=original.root_link_name,
        end_link_name=original.end_link_name,
        tcp=original.tcp,
        **kwargs,
    ).init_solver(
        "cuda", pk_serial_chain=preset.build_pk_serial_chain()["arm"].to(device="cuda")
    )


def _fk(solver, q):
    chain = solver.pk_serial_chain
    return chain.forward_kinematics(q).get_matrix() @ torch.as_tensor(
        solver.tcp_xpos, dtype=q.dtype, device=q.device
    )


def _pose_matches(target, actual):
    position = (actual[..., :3, 3] - target[..., :3, 3]).norm(dim=-1)
    difference = actual[..., :3, :3].transpose(-1, -2) @ target[..., :3, :3]
    skew = torch.stack(
        [
            difference[..., 2, 1] - difference[..., 1, 2],
            difference[..., 0, 2] - difference[..., 2, 0],
            difference[..., 1, 0] - difference[..., 0, 1],
        ],
        dim=-1,
    )
    angle = torch.atan2(
        skew.norm(dim=-1), difference.diagonal(dim1=-2, dim2=-1).sum(-1) - 1
    )
    return (position <= 1e-5) & (angle <= 1e-5)


def test_cuda_rr_rejects_cpu_device_before_loading_optional_reference() -> None:
    with pytest.raises(ValueError, match="CUDA devices only"):
        SSIKSolverCfg(backend="cuda_rr").init_solver("cpu")


@pytest.mark.gpu
def test_cuda_rr_original_urdf_tcp_all_candidates_and_failure_hold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # CUDA preparation and solve must work without the optional reference.
    monkeypatch.setitem(sys.modules, "ssik", None)
    solver = _make_solver()
    source = torch.tensor(
        [[0.2, 0.4, -0.7, 1.2, 0.6, 0.8, -1.0], [-0.5, -0.3, 1.1, -0.6, 0.7, 1.0, 0.2]],
        device="cuda",
    )
    target = _fk(solver, source)
    valid, result = solver.get_ik(target, source.flip(0))
    assert valid.shape == (2,) and bool(valid.all())
    assert _pose_matches(target, _fk(solver, result)).all()
    assert (
        (result >= solver.lower_qpos_limits) & (result <= solver.upper_qpos_limits)
    ).all()
    snapshot = result.clone()
    mask, candidates = solver.get_ik(target[:1], source[:1], return_all_solutions=True)
    assert mask.shape == (1, 384) and candidates.shape == (1, 384, 7)
    assert bool(mask.any())
    assert _pose_matches(
        target[:1].expand(int(mask.sum()), -1, -1), _fk(solver, candidates[mask])
    ).all()
    far = target[:1].clone()
    far[:, :3, 3] += 5
    failed, held = solver.get_ik(far, source[:1])
    assert not failed.any()
    torch.testing.assert_close(held, source[:1], atol=0, rtol=0)
    torch.testing.assert_close(result, snapshot, atol=0, rtol=0)
    tcp = np.eye(4)
    tcp[:3, 3] = [0.2, -0.1, 0.3]
    solver.set_tcp(tcp)
    target = _fk(solver, source)
    valid, result = solver.get_ik(target, source)
    assert bool(valid.all())
    assert _pose_matches(target, _fk(solver, result)).all()


@pytest.mark.gpu
def test_cuda_rr_restores_leading_batch_axes_and_rejects_invalid_targets() -> None:
    solver = _make_solver()
    source = torch.tensor([[0.2, 0.4, 0.0, 1.2, 0.6, 0.8, -1.0]], device="cuda")
    target = _fk(solver, source).expand(4, -1, -1).clone()
    target[1, :3, 3] += 5
    target[2, 0, 0] = float("nan")
    target[3, 3, 3] = 2
    valid, result = solver.get_ik_batch(
        target.reshape(2, 2, 4, 4), source.expand(2, 2, -1)
    )
    assert valid.shape == (2, 2) and result.shape == (2, 2, 7)
    assert valid.tolist() == [[True, False], [False, False]]
    assert torch.isfinite(result).all()
    torch.testing.assert_close(
        result.reshape(4, 7)[1:], source.expand(3, -1), atol=0, rtol=0
    )


@pytest.mark.gpu
def test_cuda_rr_effective_limits_and_step_constraint_are_hard() -> None:
    solver = _make_solver(max_joint_step=0.05)
    seed = torch.tensor([[0.2, 0.4, 0.0, 1.2, 0.6, 0.8, -1.0]], device="cuda")
    target = _fk(solver, seed)
    assert solver.set_qpos_limits(
        seed[0].cpu().numpy() - 0.01, seed[0].cpu().numpy() + 0.01
    )
    valid, result = solver.get_ik(target, seed)
    assert valid.all()
    assert ((result - seed).abs() <= 0.05).all()
    shifted = seed.clone()
    shifted[0, 0] += 0.2
    valid, result = solver.get_ik(_fk(solver, shifted), seed)
    assert not valid.any()
    torch.testing.assert_close(result, seed, atol=0, rtol=0)
