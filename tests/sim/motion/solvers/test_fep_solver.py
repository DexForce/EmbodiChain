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

from pathlib import Path

import numpy as np
import pytest
import torch
import pytorch_kinematics as pk

from embodichain.compute.kinematics import _fep
from embodichain.lab.sim.motion.solvers import FEPSolver, FEPSolverCfg, SolverCfg


def _urdf(joint_type: str = "revolute", count: int = 7) -> str:
    links = "".join(f'<link name="link{i}"/>' for i in range(count + 1))
    axes = ("0 0 1", "0 1 0", "1 0 0", "0 1 0", "1 0 0", "0 1 0", "1 0 0")
    joints = "".join(f"""<joint name="joint{i}" type="{joint_type}">
        <parent link="link{i}"/><child link="link{i+1}"/>
        <origin xyz="{0.02 if i % 2 else 0} 0 {0.1 if i else 0}"/>
        <axis xyz="{axes[i]}"/>
        <limit lower="-2" upper="2" velocity="1" effort="1"/>
        </joint>""" for i in range(count))
    return f'<robot name="offset_7r">{links}{joints}</robot>'


@pytest.fixture
def solver() -> FEPSolver:
    chain = pk.build_serial_chain_from_urdf(_urdf(), "link7")
    return FEPSolverCfg(batch_size=2).init_solver(pk_serial_chain=chain)


def _reference(device: torch.device | str = "cpu") -> torch.Tensor:
    return torch.tensor([[0.2, -0.35, 0.25, -0.6, 0.3, 0.4, -0.2]], device=device)


@pytest.mark.parametrize(
    "method",
    [
        "seeded_numerical",
        "compatibility",
        "configuration",
        "all_configurations",
        "nearest_redundancy",
    ],
)
def test_perturbed_seed_roundtrip_and_live_tcp(solver: FEPSolver, method: str) -> None:
    expected = _reference().repeat(3, 1)
    expected[:, 0] += torch.tensor([0.0, 0.03, -0.04])
    seed = expected + torch.tensor([0.03, -0.02, 0.01, 0.02, -0.01, 0.02, 0.0])
    tool = np.array(
        [
            [0.0, -1.0, 0.0, 0.1],
            [1.0, 0.0, 0.0, -0.2],
            [0.0, 0.0, 1.0, 0.05],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    for tcp in (tool, np.eye(4)):
        solver.set_tcp(tcp)
        target = solver.get_fk(expected)
        mask, joints = solver.get_ik(
            target, seed, solve_method=method, return_all_solutions=True
        )
        assert mask.shape == (3, 8 if method == "all_configurations" else 1)
        assert joints.shape == (*mask.shape, 7)
        assert bool(mask.any(dim=1).all())
        poses = target[:, None].expand(-1, joints.shape[1], -1, -1)[mask]
        error = _fep.pose_error(poses, solver.get_fk(joints[mask]))
        assert bool((error[:, :3].norm(dim=-1) <= solver.cfg.position_tolerance).all())
        assert bool((error[:, 3:].norm(dim=-1) <= solver.cfg.rotation_tolerance).all())
        if method == "configuration":
            assert torch.equal(
                solver.get_configuration(joints[:, 0])[:, :3],
                solver.get_configuration(seed)[:, :3],
            )
        nearest_mask, nearest = solver.get_ik(target, seed, solve_method=method)
        assert torch.equal(nearest_mask, mask[:, 0])
        torch.testing.assert_close(nearest, joints[:, 0])


def test_unreachable_mixed_batch_and_limits(solver: FEPSolver) -> None:
    seed = _reference().repeat(2, 1)
    target = solver.get_fk(seed)
    target[1, 0, 3] = 10
    success, joints = solver.get_ik(target, seed)
    assert success.tolist() == [True, False]
    torch.testing.assert_close(joints, seed)
    solver.set_qpos_limits([0.0] * 7, [0.0] * 7)
    success, joints = solver.get_ik(target, seed, return_all_solutions=True)
    assert not bool(success.any())
    assert torch.equal(joints, torch.zeros(2, 1, 7))


def test_batch_adapter_broadcast_empty_and_noncontiguous(solver: FEPSolver) -> None:
    qpos = _reference().repeat(6, 1).reshape(2, 3, 7)
    poses = solver.get_fk_batch(qpos)
    valid, solved = solver.get_ik_batch(poses, qpos)
    assert valid.shape == (2, 3) and bool(valid.all())
    torch.testing.assert_close(solved, qpos)
    valid, solved = solver.get_ik(poses.reshape(6, 4, 4)[::2], qpos[0, 0])
    assert valid.shape == (3,) and bool(valid.all())
    for all_solutions in (False, True):
        valid, solved = solver.get_ik(
            torch.empty(0, 4, 4), return_all_solutions=all_solutions
        )
        assert valid.shape == ((0, 1) if all_solutions else (0,))
        assert solved.shape == ((0, 1, 7) if all_solutions else (0, 7))


def test_config_factory_and_urdf_initialization(tmp_path: Path) -> None:
    path = tmp_path / "seven.urdf"
    path.write_text(_urdf())
    cfg = SolverCfg.from_dict(
        {
            "class_type": "FEPSolver",
            "urdf_path": str(path),
            "root_link_name": "link0",
            "end_link_name": "link7",
        }
    )
    solver = cfg.init_solver()
    assert isinstance(solver, FEPSolver)
    assert solver.dof == 7
    success, solved = solver.get_ik(solver.get_fk(_reference()), _reference())
    assert bool(success.all())
    torch.testing.assert_close(solved, _reference())


@pytest.mark.parametrize(
    "settings",
    [
        {"solve_method": "analytic"},
        {"damping": 0.0},
        {"max_iterations": 0},
        {"batch_size": 1.5},
        {"rotation_tolerance": float("nan")},
        {"redundancy_step": 0.0},
    ],
)
def test_invalid_config(settings: dict) -> None:
    with pytest.raises(ValueError):
        FEPSolverCfg(**settings)


@pytest.mark.parametrize("joint_type,count", [("prismatic", 7), ("revolute", 6)])
def test_reject_incompatible_chain(joint_type: str, count: int) -> None:
    chain = pk.build_serial_chain_from_urdf(_urdf(joint_type, count), f"link{count}")
    with pytest.raises(ValueError, match="seven revolute"):
        FEPSolverCfg().init_solver(pk_serial_chain=chain)


def test_reject_wrong_joint_order(solver: FEPSolver) -> None:
    with pytest.raises(ValueError, match="chain order"):
        FEPSolverCfg(joint_names=solver.joint_names[::-1]).init_solver(
            pk_serial_chain=solver.pk_serial_chain
        )


def test_invalid_input_and_configuration(solver: FEPSolver) -> None:
    pose = solver.get_fk(_reference())
    with pytest.raises(ValueError, match="finite target"):
        solver.get_ik(pose * torch.nan)
    with pytest.raises(ValueError, match="finite seeds"):
        solver.get_ik(pose, torch.zeros(6))
    with pytest.raises(ValueError, match="Unknown"):
        solver.get_ik(pose, solve_method="analytic")
    with pytest.raises(ValueError, match="configuration mode"):
        solver.get_ik(pose, configuration=torch.tensor([1.0, 1.0, 1.0, 0.0]))
    config = solver.get_configuration(_reference())[0]
    success, joints = solver.get_ik(
        pose, _reference(), solve_method="configuration", configuration=config
    )
    assert bool(success.all())
    assert torch.equal(solver.get_configuration(joints)[:, :3], config[None, :3])
    config[0] = 0
    with pytest.raises(ValueError, match="signs"):
        solver.get_ik(pose, solve_method="configuration", configuration=config)


def test_configuration_zero_and_wrapped_weighted_ranking(solver: FEPSolver) -> None:
    assert solver.get_configuration(torch.zeros(7)).tolist() == [1, 1, 1, 0]
    candidates = torch.zeros(1, 3, 7)
    candidates[0, 0, 0] = 2 * torch.pi - 0.01
    candidates[0, 1, 0] = 0.1
    candidates[0, 2, 0] = 0
    mask, ranked = solver._rank(
        torch.tensor([[True, True, False]]), candidates, torch.zeros(1, 7)
    )
    assert mask.tolist() == [[True, True, False]]
    torch.testing.assert_close(ranked, candidates)
    solver.set_ik_nearest_weight(np.array([100.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))
    candidates[0, 0, 0] = 0.01
    candidates[0, 1, 0] = 0
    candidates[0, 1, 1] = 0.1
    _, ranked = solver._rank(mask, candidates, torch.zeros(1, 7))
    torch.testing.assert_close(ranked[:, 0], candidates[:, 1])


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize(
    "method",
    [
        "seeded_numerical",
        "compatibility",
        "configuration",
        "all_configurations",
        "nearest_redundancy",
    ],
)
def test_cuda_perturbed_seed_roundtrip(method: str) -> None:
    chain = pk.build_serial_chain_from_urdf(_urdf(), "link7").to(device="cuda")
    solver = FEPSolverCfg().init_solver(device="cuda", pk_serial_chain=chain)
    tool = np.eye(4)
    tool[:3, 3] = [0.1, -0.2, 0.05]
    solver.set_tcp(tool)
    qpos = _reference("cuda").repeat(3, 1)
    target = solver.get_fk(qpos)
    valid, result = solver.get_ik(
        target, qpos + 0.02, solve_method=method, return_all_solutions=True
    )
    assert bool(valid.any(dim=1).all())
    expected = target[:, None].expand(-1, result.shape[1], -1, -1)[valid]
    torch.testing.assert_close(
        solver.get_fk(result[valid]), expected, atol=1e-5, rtol=0
    )


def test_redundancy_search_stops_per_target_and_compares_both_directions(
    solver: FEPSolver, monkeypatch: pytest.MonkeyPatch
) -> None:
    solver.cfg.redundancy_step = 0.1
    solver.cfg.redundancy_range = 0.5
    seed = torch.zeros(2, 7)
    targets = torch.eye(4).repeat(2, 1, 1)
    targets[1, 0, 3] = 1
    calls = []

    def correction(
        target: torch.Tensor, trial: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        calls.append(trial.clone())
        solved = trial.clone()
        threshold = torch.where(target[:, 0, 3] == 0, 0.1, 0.2)
        valid = trial[:, 6].abs() >= threshold - 1e-6
        # The positive direction gives a closer final solution at the same radius.
        solved[:, 0] = torch.where(trial[:, 6] < 0, 1.0, 0.0)
        return valid, solved

    monkeypatch.setattr(solver, "_solve_seed", correction)
    valid, joints = solver.get_ik(targets, seed, solve_method="nearest_redundancy")
    assert valid.tolist() == [True, True]
    assert [len(call) for call in calls] == [2, 4, 2]
    torch.testing.assert_close(joints[:, 6], torch.tensor([0.1, 0.2]))
    assert torch.equal(joints[:, 0], torch.zeros(2))


def test_redundancy_search_skips_repeated_clamped_seeds(
    solver: FEPSolver, monkeypatch: pytest.MonkeyPatch
) -> None:
    solver.set_qpos_limits([-2.0] * 6 + [0.0], [2.0] * 6 + [0.0])
    calls = []

    def correction(
        target: torch.Tensor, seed: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        calls.append(seed.clone())
        return torch.zeros(len(seed), dtype=torch.bool), seed.clone()

    monkeypatch.setattr(solver, "_solve_seed", correction)
    valid, joints = solver.get_ik(
        torch.eye(4), torch.zeros(7), solve_method="nearest_redundancy"
    )
    assert not valid.item()
    assert len(calls) == 1
    assert torch.equal(joints, torch.zeros(1, 7))


def test_configuration_rejects_converged_wrong_branch(
    solver: FEPSolver, monkeypatch: pytest.MonkeyPatch
) -> None:
    def correction(
        target: torch.Tensor, seed: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        wrong = seed.clone()
        wrong[:, [1, 3, 5]] = 0.5
        return torch.ones(len(seed), dtype=torch.bool), wrong

    monkeypatch.setattr(solver, "_solve_seed", correction)
    valid, _ = solver.get_ik(torch.eye(4), -torch.ones(7), solve_method="configuration")
    assert not valid.item()
    valid, _ = solver.get_ik(
        torch.eye(4),
        -torch.ones(7),
        solve_method="all_configurations",
        return_all_solutions=True,
    )
    assert valid.sum().item() == 1


def test_pose_error_does_not_accept_half_turn_as_zero() -> None:
    from scipy.spatial.transform import Rotation

    target = torch.eye(4).repeat(2, 1, 1)
    target[0, :3, :3] = torch.tensor(
        Rotation.from_rotvec([0.0, np.pi, 0.0]).as_matrix(), dtype=torch.float32
    )
    target[1, :3, :3] = torch.tensor(
        Rotation.from_rotvec([1e-6, 0.0, 0.0]).as_matrix(), dtype=torch.float32
    )
    error = _fep.pose_error(target, torch.eye(4).repeat(2, 1, 1))
    torch.testing.assert_close(error[:, 3:].norm(dim=-1), torch.tensor([np.pi, 1e-6]))


def test_fixed_terminal_offsets_and_independent_fk() -> None:
    from scipy.spatial.transform import Rotation

    fixed = """<link name="tool"/><joint name="fixed_tool" type="fixed">
    <parent link="link7"/><child link="tool"/>
    <origin xyz="0.05 0.02 0.03" rpy="0.1 0.2 0.3"/></joint>"""
    chain = pk.build_serial_chain_from_urdf(
        _urdf().replace("</robot>", fixed + "</robot>"), "tool"
    )
    solver = FEPSolverCfg().init_solver(pk_serial_chain=chain)
    tcp = np.eye(4)
    tcp[:3, 3] = [0.1, -0.2, 0.05]
    solver.set_tcp(tcp)
    qpos = _reference()[0]
    axes = np.array(
        [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0], [1, 0, 0]]
    )
    expected = np.eye(4)
    for i, (angle, axis) in enumerate(zip(qpos.numpy(), axes)):
        transform = np.eye(4)
        transform[:3, 3] = [0.02 if i % 2 else 0, 0, 0.1 if i else 0]
        transform[:3, :3] = Rotation.from_rotvec(angle * axis).as_matrix()
        expected = expected @ transform
    terminal = np.eye(4)
    terminal[:3, 3] = [0.05, 0.02, 0.03]
    terminal[:3, :3] = Rotation.from_euler("xyz", [0.1, 0.2, 0.3]).as_matrix()
    expected = expected @ terminal @ tcp
    np.testing.assert_allclose(solver.get_fk(qpos)[0].numpy(), expected, atol=2e-7)
    target = torch.tensor(expected, dtype=torch.float32)
    valid, joints = solver.get_ik(target, qpos + 0.02)
    assert valid.item()
    actual = solver.get_fk(joints)[0].numpy().astype(np.float64)
    assert np.linalg.norm(actual[:3, 3] - expected[:3, 3]) <= 1e-5
    angle = (
        Rotation.from_matrix(actual[:3, :3]).inv()
        * Rotation.from_matrix(expected[:3, :3])
    ).magnitude()
    assert angle <= 1e-5


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize(
    "method,width",
    [("seeded_numerical", 1), ("all_configurations", 8), ("nearest_redundancy", 2)],
)
def test_cuda_auto_chunking_bounds_candidate_budget(
    method: str, width: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    chain = pk.build_serial_chain_from_urdf(_urdf(), "link7").to(device="cuda")
    solver = FEPSolverCfg().init_solver(device="cuda", pk_serial_chain=chain)
    chunks = []
    candidates = 8 if method == "all_configurations" else 1

    def search(
        target: torch.Tensor,
        seed: torch.Tensor,
        method: str,
        configuration: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        chunks.append(len(seed))
        return torch.ones(
            len(seed), candidates, device=seed.device, dtype=torch.bool
        ), seed[:, None].expand(-1, candidates, -1)

    monkeypatch.setattr(solver, "_search", search)
    count = 17000
    valid, result = solver.get_ik(
        torch.eye(4, device="cuda").expand(count, -1, -1), solve_method=method
    )
    assert valid.shape == (count,) and bool(valid.all())
    assert result.shape == (count, 7)
    assert sum(chunks) == count
    assert max(chunks) * width <= 16384
    chunks.clear()
    solver.cfg.batch_size = 3
    solver.get_ik(torch.eye(4, device="cuda").expand(7, -1, -1), solve_method=method)
    assert chunks == [3, 3, 1]


def _oblique_chain_urdf() -> str:
    """Include oblique axes, rotated joint origins and an internal fixed link."""
    model = _urdf().replace('axis xyz="0 1 0"', 'axis xyz="0.6 0 0.8"')
    model = model.replace("<origin xyz=", '<origin rpy="0.1 -0.2 0.15" xyz=')
    model = model.replace('<parent link="link3"/>', '<parent link="bend"/>')
    fixed = """<link name="bend"/><joint name="elbow_fixed" type="fixed">
    <parent link="link3"/><child link="bend"/>
    <origin xyz="0.03 -0.01 0.02" rpy="-0.2 0.1 0.3"/></joint>"""
    return model.replace("</robot>", fixed + "</robot>")


@pytest.mark.parametrize(
    "device",
    [
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
    ],
)
def test_fused_kernel_without_fallback_on_oblique_chain(device: str) -> None:
    from scipy.spatial.transform import Rotation

    chain = pk.build_serial_chain_from_urdf(_oblique_chain_urdf(), "link7").to(
        device=device
    )
    solver = FEPSolverCfg(backend="warp").init_solver(
        device=device, pk_serial_chain=chain
    )
    generator = torch.Generator().manual_seed(117)
    qpos = (torch.rand(32, 7, generator=generator) * 2 - 1).to(device)
    tool = np.eye(4)
    tool[:3, :3] = Rotation.from_euler("xyz", [0.2, -0.1, 0.3]).as_matrix()
    tool[:3, 3] = [0.1, -0.2, 0.05]
    for tcp in (tool, np.eye(4)):
        solver.set_tcp(tcp)
        target = solver.get_fk(qpos)
        valid, result = solver._warp_model.solve(
            target,
            qpos + 0.01,
            solver.lower_qpos_limits,
            solver.upper_qpos_limits,
            solver.tcp_xpos,
            solver.cfg,
        )
        assert bool(valid.all())
        actual = solver.get_fk(result).cpu().numpy().astype(np.float64)
        expected = target.cpu().numpy().astype(np.float64)
        error = np.linalg.norm(actual[:, :3, 3] - expected[:, :3, 3], axis=-1)
        angles = (
            Rotation.from_matrix(actual[:, :3, :3]).inv()
            * Rotation.from_matrix(expected[:, :3, :3])
        ).magnitude()
        assert bool((error <= 1e-5).all())
        assert bool((angles <= 1e-5).all())
        assert bool(
            (
                (result >= solver.lower_qpos_limits)
                & (result <= solver.upper_qpos_limits)
            ).all()
        )
    # Exact seeds and current limits must be observed without Torch fallback.
    valid, result = solver._warp_model.solve(
        target,
        qpos,
        solver.lower_qpos_limits,
        solver.upper_qpos_limits,
        solver.tcp_xpos,
        solver.cfg,
    )
    assert bool(valid.all())
    torch.testing.assert_close(result, qpos, atol=0, rtol=0)
    solver.set_qpos_limits([0.0] * 7, [0.0] * 7)
    valid, result = solver._warp_model.solve(
        target,
        qpos,
        solver.lower_qpos_limits,
        solver.upper_qpos_limits,
        solver.tcp_xpos,
        solver.cfg,
    )
    assert not bool(valid.any())
    assert torch.equal(result, torch.zeros_like(result))
    valid, result = solver.get_ik(target[:2], qpos[:2])
    assert not bool(valid.any())
    assert torch.equal(result, torch.zeros_like(result))


def test_native_verifier_retries_original_seed(
    solver: FEPSolver, monkeypatch: pytest.MonkeyPatch
) -> None:
    class InvalidNative:
        def solve(self, target, seed, *args):
            # Simulate a backend that reports success with the wrong FK pose.
            return torch.ones(len(seed), dtype=torch.bool), torch.zeros_like(seed)

    monkeypatch.setattr(solver, "_warp_model", InvalidNative())
    seed = _reference()
    target = solver.get_fk(seed)
    valid, result = solver.get_ik(target, seed)
    assert valid.item()
    torch.testing.assert_close(result, seed, atol=0, rtol=0)


def test_unknown_fep_backend_rejected() -> None:
    with pytest.raises(ValueError, match="backend"):
        FEPSolverCfg(backend="unavailable")


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_fused_results_on_secondary_stream_are_owned() -> None:
    chain = pk.build_serial_chain_from_urdf(_urdf(), "link7").to(device="cuda")
    solver = FEPSolverCfg().init_solver(device="cuda", pk_serial_chain=chain)
    qpos = _reference("cuda")
    target = solver.get_fk(qpos)
    other = torch.cuda.Stream()
    other.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(other):
        valid, result = solver.get_ik(target, qpos + 0.02)
        saved = result.clone()
        solver.get_ik(target, qpos - 0.01)
    torch.cuda.current_stream().wait_stream(other)
    assert valid.item()
    torch.testing.assert_close(result, saved, atol=0, rtol=0)
    torch.testing.assert_close(solver.get_fk(result), target, atol=1e-5, rtol=0)


def test_native_candidate_meeting_public_tolerance_avoids_retry(
    solver: FEPSolver, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed = _reference()

    class StrictNative:
        def solve(self, target, initial, *args):
            # Native tightening can report failure at its budget even though
            # the public FK tolerance is already met.
            return torch.zeros(len(initial), dtype=torch.bool), seed.clone()

    def unexpected_retry(*args):
        raise AssertionError("A verified candidate must not repeat numerical IK")

    monkeypatch.setattr(solver, "_warp_model", StrictNative())
    monkeypatch.setattr(solver, "_solve_seed_torch", unexpected_retry)
    valid, result = solver.get_ik(solver.get_fk(seed), seed + 0.01)
    assert valid.item()
    torch.testing.assert_close(result, seed)
