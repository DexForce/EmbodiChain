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

"""Public Warp CPU/CUDA RR behavior on the original Rizon4S URDF."""

from __future__ import annotations

import sys
from typing import Any

import numpy as np
import pytest
import torch

from embodichain.lab.sim.motion.solvers import SSIKSolverCfg, SolverCfg
from embodichain_robots import FlexivRizon4SCfg


def _make_solver(device: str = "cuda", *, legacy: bool = False, **kwargs: Any):
    preset = FlexivRizon4SCfg.from_dict({})
    original = preset.solver_cfg["arm"]
    return SSIKSolverCfg(
        backend="cuda_rr" if legacy else "warp",
        geometry_mode="strict" if legacy else "rr",
        urdf_path=preset.fpath,
        root_link_name=original.root_link_name,
        end_link_name=original.end_link_name,
        tcp=original.tcp,
        **kwargs,
    ).init_solver(
        device, pk_serial_chain=preset.build_pk_serial_chain()["arm"].to(device=device)
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


def test_legacy_rr_rejects_cpu_device_before_loading_optional_reference() -> None:
    with pytest.raises(ValueError, match="CUDA devices only"):
        SSIKSolverCfg(backend="cuda_rr").init_solver("cpu")


@pytest.mark.parametrize("search", ["fixed", "seed_local"])
def test_legacy_rr_search_configuration_round_trips(
    search: str,
) -> None:
    cfg = SSIKSolverCfg(backend="cuda_rr", rr_search=search)
    restored = SolverCfg.from_dict(cfg.to_dict())
    assert isinstance(restored, SSIKSolverCfg)
    assert restored.backend == "cuda_rr" and restored.rr_search == search
    assert SSIKSolverCfg().rr_search == "fixed"


@pytest.mark.parametrize("backend", ["warp", "reference"])
def test_rr_seed_local_requires_explicit_algebraic_mode(backend: str) -> None:
    with pytest.raises(ValueError, match="rr_search"):
        SSIKSolverCfg(backend=backend, rr_search="seed_local")


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("search", ["fixed", "seed_local"])
def test_rr_original_urdf_tcp_all_candidates_and_failure_hold(
    monkeypatch: pytest.MonkeyPatch,
    search: str,
    device: str,
) -> None:
    # Warp preparation and solve must work without the optional reference.
    monkeypatch.setitem(sys.modules, "ssik", None)
    solver = _make_solver(device, rr_search=search)
    source = torch.tensor(
        [[0.2, 0.4, -0.7, 1.2, 0.6, 0.8, -1.0], [-0.5, -0.3, 1.1, -0.6, 0.7, 1.0, 0.2]],
        device=device,
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
    slots = 384 if search == "fixed" else 600
    assert mask.shape == (1, slots) and candidates.shape == (1, slots, 7)
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


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_rr_seed_local_nearest_and_all_use_the_same_current_sample_pool(
    device: str,
) -> None:
    solver = _make_solver(device, rr_search="seed_local", max_joint_step=0.05)
    seed = torch.tensor([[0.2, 0.4, 0.137, 1.2, 0.6, 0.8, -1]], device=device)
    moving = seed.clone()
    moving[0, 0] += 0.001
    moving[0, 2] += 0.0005
    target = _fk(solver, moving)
    valid, result = solver.get_ik(target, seed)
    mask, candidates = solver.get_ik(target, seed, return_all_solutions=True)
    assert valid.all() and mask.shape == (1, 600)
    score = ((candidates - seed[:, None]) * solver.ik_nearest_weight).square().sum(-1)
    best = score.masked_fill(~mask, float("inf")).argmin(-1)
    torch.testing.assert_close(
        result, candidates[torch.arange(1, device=device), best], atol=1e-6, rtol=0
    )
    assert _pose_matches(target, _fk(solver, result)).all()
    assert ((result - seed).abs() <= 0.05).all()


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_rr_restores_leading_batch_axes_and_rejects_invalid_targets(
    device: str,
) -> None:
    solver = _make_solver(device)
    source = torch.tensor([[0.2, 0.4, 0.0, 1.2, 0.6, 0.8, -1.0]], device=device)
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


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_rr_effective_limits_and_step_constraint_are_hard(device: str) -> None:
    solver = _make_solver(device, max_joint_step=0.05)
    seed = torch.tensor([[0.2, 0.4, 0.0, 1.2, 0.6, 0.8, -1.0]], device=device)
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


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_warp_rr_continuous_path_regenerates_from_previous_result(device: str) -> None:
    """CPU RR paths use their algebraic provider rather than SRS reconstruction."""
    solver = _make_solver(device, rr_search="seed_local", max_joint_step=0.05)
    seed = torch.tensor([[0.2, 0.4, 0.137, 1.2, 0.6, 0.8, -1.0]], device=device)
    source = seed.expand(3, -1).clone()
    source[:, 0] += torch.tensor([0.001, 0.002, 0.003], device=device)
    source[:, 2] += torch.tensor([0.0005, 0.001, 0.0015], device=device)
    target = _fk(solver, source)
    far = target[1:2].clone()
    far[:, :3, 3] += 5.0
    sequence = torch.cat((target[:1], far, target[1:]), dim=0)[None]
    valid, result = solver._get_continuous_ik_path(sequence, seed)
    assert valid.tolist() == [[True, False, True, True]]
    assert _pose_matches(sequence[valid], _fk(solver, result[valid])).all()
    previous = torch.cat((seed[:, None], result[:, :-1]), dim=1)
    assert ((result - previous).abs() <= 0.05).all()
    torch.testing.assert_close(result[:, 1], result[:, 0], atol=0, rtol=0)


@pytest.mark.gpu
def test_legacy_cuda_rr_and_unified_warp_rr_return_the_same_sampled_pool() -> None:
    """Existing CUDA-only configurations retain their candidate/selection behavior."""
    modern = _make_solver(rr_search="seed_local", max_joint_step=0.05)
    legacy = _make_solver(legacy=True, rr_search="seed_local", max_joint_step=0.05)
    seed = torch.tensor([[0.2, 0.4, 0.137, 1.2, 0.6, 0.8, -1.0]], device="cuda")
    source = seed.clone()
    source[:, 0] += 0.001
    target = _fk(modern, source)
    for all_solutions in (False, True):
        actual = modern.get_ik(target, seed, return_all_solutions=all_solutions)
        expected = legacy.get_ik(target, seed, return_all_solutions=all_solutions)
        torch.testing.assert_close(actual[0], expected[0], atol=0, rtol=0)
        torch.testing.assert_close(actual[1], expected[1], atol=0, rtol=0)
