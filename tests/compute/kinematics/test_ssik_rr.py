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

"""Independent PK checks of prepared algebraic candidates and CUDA QR roots."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import warnings

import numpy as np
import pytest
import pytorch_kinematics as pk
from scipy.spatial.transform import Rotation
import torch

from embodichain.compute.kinematics._ssik_model import _extract_model
from embodichain.compute.kinematics._ssik_rr import _PreparedRRCandidates
from embodichain.compute.kinematics._ssik_rr_parametric import _condition_target


@pytest.fixture(scope="module")
def rr_chain() -> pk.SerialChain:
    # Reconstruct an independent URDF chain from the licensed physical constants.
    # A 43 mm fixed terminal extension exercises the actual/prebuilt frame bridge.
    import embodichain.compute.kinematics._ssik_rr as module

    sidecar = Path(module.__file__).parent / "_warp/data/ssik_rr_rizon4.npz"
    with np.load(sidecar, allow_pickle=False) as constants:
        xml = ['<robot name="rr_fixture"><link name="base"/>']
        for j in range(7):
            parent = "base" if j == 0 else f"link{j - 1}"
            xyz = " ".join(map(str, constants["t_left"][j, :3, 3]))
            axis = " ".join(map(str, constants["axes"][j]))
            xml.append(
                f'<link name="link{j}"/><joint name="q{j}" type="revolute">'
                f'<parent link="{parent}"/><child link="link{j}"/>'
                f'<origin xyz="{xyz}"/><axis xyz="{axis}"/>'
                f'<limit lower="{constants["lo"][j]}" upper="{constants["hi"][j]}" effort="1" velocity="1"/></joint>'
            )
        terminal = constants["t_right"][-1].copy()
        terminal[:3, 3] += terminal[:3, :3] @ np.array([0.0, 0.0, 0.043])
        xyz = " ".join(map(str, terminal[:3, 3]))
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Gimbal lock detected", category=UserWarning
            )
            rpy = " ".join(
                map(str, Rotation.from_matrix(terminal[:3, :3]).as_euler("xyz"))
            )
        xml.append(
            f'<link name="tip"/><joint name="terminal" type="fixed">'
            f'<parent link="link6"/><child link="tip"/><origin xyz="{xyz}" rpy="{rpy}"/></joint></robot>'
        )
    return pk.build_serial_chain_from_urdf("".join(xml), "tip", "base")


def _check_candidates(
    chain: pk.SerialChain,
    target: torch.Tensor,
    flags: torch.Tensor,
    candidates: torch.Tensor,
) -> torch.Tensor:
    actual = chain.forward_kinematics(
        candidates.float().double().reshape(-1, 7)
    ).get_matrix()
    actual = actual.reshape(len(target), candidates.shape[1], 4, 4)
    error = (actual - target[:, None]).norm(dim=(-1, -2))
    return flags.bool() & (error <= 1e-5)


def _world_roundtrip_case() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Actual Rizon4S seed2531 trajectory3/frame337. Its native world/root F32
    # conversion exposed the exact-SO(3) reduction's input-conditioning error.
    root = np.array(
        [
            [1, -8.184881039596803e-8, 2.3640840396410567e-8, 0.13700000941753387],
            [8.184881039596803e-8, 1, -2.435027823821656e-8, -0.25099998712539673],
            [-2.364083684369689e-8, 2.435028179093024e-8, 1, 0.11899998784065247],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )
    world = np.array(
        [
            [
                -0.26144683361053467,
                0.6695428490638733,
                0.6952394843101501,
                0.37192708253860474,
            ],
            [
                -0.8762074112892151,
                0.13748784363269806,
                -0.4619065225124359,
                -0.40724971890449524,
            ],
            [
                -0.4048531651496887,
                -0.7299379706382751,
                0.5507126450538635,
                1.3425434827804565,
            ],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )
    inverse = np.eye(4, dtype=np.float32)
    inverse[:3, :3] = root[:3, :3].T
    inverse[:3, 3] = -inverse[:3, :3] @ root[:3, 3]
    target = torch.from_numpy(inverse @ world).double().unsqueeze(0)
    seed = torch.tensor(
        [
            [
                -0.7131089568138123,
                0.09755972027778625,
                0.21346667408943176,
                0.9629471302032471,
                -0.6111317276954651,
                1.4228380918502808,
                -0.4900585114955902,
            ]
        ],
        dtype=torch.float64,
    )
    witness = torch.tensor(
        [
            [
                -0.7125424146652222,
                0.09760601818561554,
                0.21356834471225739,
                0.9624706506729126,
                -0.609802782535553,
                1.4245902299880981,
                -0.49003639817237854,
            ]
        ],
        dtype=torch.float64,
    )
    return target, seed, witness


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_rr_candidates_preserve_all_slices_and_original_fixed_end_frame(
    rr_chain: pk.SerialChain, device: str
) -> None:
    model = _extract_model(rr_chain)
    chain = rr_chain.to(dtype=torch.float64, device=device)
    engine = _PreparedRRCandidates(model, device)
    q = torch.tensor(
        [[0.2, 0.4, -0.7, 1.2, 0.6, 0.8, -1.0], [-0.5, -0.3, 1.1, -0.6, 0.7, 1.0, 0.2]],
        dtype=torch.float64,
        device=device,
    )
    target = chain.forward_kinematics(q).get_matrix()
    flags, candidates = engine.generate(target)
    assert flags.shape == (2, 384)
    assert candidates.shape == (2, 384, 7)
    assert _check_candidates(chain, target, flags, candidates).any(dim=-1).all()
    # Every geometric branch has its slice's locked joint coordinate.
    locked = candidates[:, :, 2].reshape(2, 16, 24)
    expected = torch.as_tensor(engine._data.lock_angles.numpy(), device=device)
    assert torch.allclose(
        locked[flags.reshape(2, 16, 24).bool()],
        expected[None, :, None].expand_as(locked)[flags.reshape(2, 16, 24).bool()],
    )
    original = candidates.clone()
    engine.generate(target[:1])
    assert torch.equal(candidates, original)


def test_rr_rejects_a_changed_joint_prefix(rr_chain: pk.SerialChain) -> None:
    model = _extract_model(rr_chain)
    offsets = model.offsets.copy()
    offsets[3, 0] += 1e-4
    with pytest.raises(ValueError, match="joint prefix"):
        _PreparedRRCandidates(replace(model, offsets=offsets), "cpu")


@pytest.mark.gpu
@pytest.mark.parametrize("search", ["fixed", "seed_local"])
def test_rr_cuda_chunking_noncontiguous_targets_and_stream_order(
    rr_chain: pk.SerialChain,
    search: str,
) -> None:
    model = _extract_model(rr_chain)
    chain = rr_chain.to(dtype=torch.float64, device="cuda")
    engine = _PreparedRRCandidates(model, "cuda", search=search)
    q = torch.tensor(
        [[0.2, 0.4, -0.7, 1.2, 0.6, 0.8, -1.0]], device="cuda", dtype=torch.float64
    ).expand(33, 7)
    target = chain.forward_kinematics(q).get_matrix()
    lower = torch.tensor(model.limits[:, 0], device="cuda")
    upper = torch.tensor(model.limits[:, 1], device="cuda")
    storage = torch.empty((33, 4, 8), device="cuda", dtype=torch.float64)
    storage[:, :, ::2] = target
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        flags, candidates = engine.generate(storage[:, :, ::2], q, lower, upper)
        good = _check_candidates(chain, target, flags, candidates)
    torch.cuda.current_stream().wait_stream(stream)
    assert good.any(dim=-1).all()
    assert engine._capacity == engine.max_chunk_size
    assert sum(t.nbytes for t in engine._workspace.values()) < 32 * 1024**2


@pytest.mark.parametrize("search", ["fixed", "seed_local"])
def test_rr_empty_batch(rr_chain: pk.SerialChain, search: str) -> None:
    model = _extract_model(rr_chain)
    engine = _PreparedRRCandidates(model, "cpu", search=search)
    valid, candidates = engine.generate(
        torch.empty((0, 4, 4), dtype=torch.float64),
        torch.empty((0, 7), dtype=torch.float64),
        torch.tensor(model.limits[:, 0]),
        torch.tensor(model.limits[:, 1]),
    )
    assert valid.shape == (0, engine.candidate_count)
    assert candidates.shape == (0, engine.candidate_count, 7)


def test_parameterized_rr_follows_arbitrary_seed_lock_on_cpu(
    rr_chain: pk.SerialChain,
) -> None:
    model = _extract_model(rr_chain)
    chain = rr_chain.to(dtype=torch.float64, device="cpu")
    engine = _PreparedRRCandidates(model, "cpu", search="seed_local")
    q = torch.tensor(
        [
            [0.2, 0.4, -0.000000001, 1.2, 0.6, 0.8, -1],
            [0.2, 0.4, 0.137, 1.2, 0.6, 0.8, -1],
        ],
        dtype=torch.float64,
    )
    targets = chain.forward_kinematics(q).get_matrix().float().double()
    flags, candidates = engine.generate(
        targets,
        q,
        torch.tensor(model.limits[:, 0]),
        torch.tensor(model.limits[:, 1]),
        0.05,
    )
    assert flags.shape == (2, 600) and candidates.shape == (2, 600, 7)
    good = _check_candidates(chain, targets, flags, candidates)
    # The exact seed lock is an algebraically evaluated parameter, rather than
    # a renamed old coefficient chart. Both points are absent from the16-grid.
    seed_slots = slice(16 * 24, 17 * 24)
    assert good[:, seed_slots].any(-1).all()
    difference = (candidates[:, seed_slots, 2] - q[:, None, 2]).abs()
    assert (difference[flags[:, seed_slots].bool()] < 1e-14).all()


def test_parameterized_rr_conditions_f32_pose_without_accepting_invalid_inputs(
    rr_chain: pk.SerialChain,
) -> None:
    chain = rr_chain.to(dtype=torch.float64, device="cpu")
    model = _extract_model(chain)
    engine = _PreparedRRCandidates(model, "cpu", search="seed_local")
    target, seed, witness = _world_roundtrip_case()
    assert _check_candidates(chain, target, torch.ones((1, 1)), witness[:, None]).all()
    assert (witness - seed).abs().max() < 0.05
    original = target.numpy().copy()
    prepared = _condition_target(original[0])
    assert prepared is not None
    assert np.linalg.norm(prepared[:3, :3].T @ prepared[:3, :3] - np.eye(3)) < 1e-14
    assert np.array_equal(prepared[:, 3], original[0, :, 3])
    targets = target.expand(5, -1, -1).clone()
    targets[1, :3, 0] *= -1  # Reflection cannot be repaired into an accepted pose.
    targets[2, :3, :3] *= 2
    targets[3, 3, 3] = 0.9
    targets[4, 0, 3] = torch.nan
    untouched = targets.clone()
    flags, candidates = engine.generate(
        targets,
        seed.expand(5, -1),
        torch.tensor(model.limits[:, 0]),
        torch.tensor(model.limits[:, 1]),
        0.05,
    )
    assert flags.shape == (5, 600)
    assert _check_candidates(chain, target, flags[:1], candidates[:1]).any()
    assert not flags[1:].any()
    torch.testing.assert_close(targets, untouched, equal_nan=True)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_parameterized_rr_matches_explicit_qz_conditional_branches(
    rr_chain: pk.SerialChain,
    device: str,
) -> None:
    model = _extract_model(rr_chain)
    chain = rr_chain.to(dtype=torch.float64, device="cpu")
    q = torch.tensor(
        [
            [0.2, 0.4, angle, 1.2, 0.6, 0.8, -1]
            for angle in (0, -1e-9, 1e-9, -0.137, 0.137, 2.9)
        ],
        dtype=torch.float64,
    )
    target = chain.forward_kinematics(q).get_matrix().float().double()
    roundtrip, roundtrip_seed, _ = _world_roundtrip_case()
    reflected = roundtrip.clone()
    reflected[:, :3, 0] *= -1
    target = torch.cat((target, roundtrip, reflected))
    q = torch.cat((q, roundtrip_seed, roundtrip_seed))
    lower, upper = torch.tensor(model.limits[:, 0]), torch.tensor(model.limits[:, 1])
    cpu = _PreparedRRCandidates(model, "cpu", search="seed_local")
    tested = _PreparedRRCandidates(model, device, search="seed_local")
    # Independent QZ/SVD is explicitly invoked as a test oracle. CPU runtime
    # dispatch must now exercise Warp rather than silently selecting this path.
    reference_flags, reference = cpu._provider._cpu_qz(
        target, q, lower, upper, float("inf")
    )
    flags, actual = tested.generate(
        target.to(device), q.to(device), lower.to(device), upper.to(device)
    )
    flags, actual = flags.cpu(), actual.cpu()
    assert not flags[-1].any() and not reference_flags[-1].any()
    reference_good = _check_candidates(chain, target, reference_flags, reference)
    actual_good = _check_candidates(chain, target, flags, actual)
    for pose in range(len(q)):
        for sample in range(25):
            block = slice(sample * 24, (sample + 1) * 24)
            expected_rows = reference[pose, block][reference_good[pose, block]]
            actual_rows = actual[pose, block][actual_good[pose, block]]
            assert bool(len(expected_rows)) == bool(len(actual_rows))
            if len(expected_rows):
                error = (
                    (
                        (expected_rows[:, None] - actual_rows[None] + torch.pi)
                        % (2 * torch.pi)
                        - torch.pi
                    )
                    .abs()
                    .amax(-1)
                )
                assert (error.amin(0) < 1e-3).all() and (error.amin(1) < 1e-3).all()


def test_parameterized_cpu_runtime_is_warp_and_preserves_output_ownership(
    rr_chain: pk.SerialChain,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scipy.linalg

    model = _extract_model(rr_chain)
    chain = rr_chain.to(dtype=torch.float64, device="cpu")
    engine = _PreparedRRCandidates(model, "cpu", search="seed_local")
    q = torch.tensor(
        [[0.2, 0.4, 0.137, 1.2, 0.6, 0.8, -1]], dtype=torch.float64, device="cpu"
    )
    targets = chain.forward_kinematics(q.expand(2, -1)).get_matrix().float().double()
    storage = torch.empty((2, 4, 8), dtype=torch.float64, device="cpu")
    storage[:, :, ::2] = targets
    lower = torch.tensor(model.limits[:, 0], device="cpu")
    upper = torch.tensor(model.limits[:, 1], device="cpu")

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError(
            "Production CPU RR called the independent algebraic oracle"
        )

    monkeypatch.setattr(engine._provider, "_cpu_qz", forbidden)
    monkeypatch.setattr(scipy.linalg, "eig", forbidden)
    monkeypatch.setattr(scipy.linalg, "qz", forbidden)
    with torch.device("meta"):
        flags, candidates = engine.generate(storage[:1, :, ::2], q, lower, upper, 0.05)
        original_flags, original_candidates = flags.clone(), candidates.clone()
        newer_flags, newer = engine.generate(
            storage[:, :, ::2], q.expand(2, -1), lower, upper, 0.05
        )
        empty_flags, empty_candidates = engine.generate(
            storage[:0, :, ::2], q[:0], lower, upper, 0.05
        )
    for output in (
        flags,
        candidates,
        newer_flags,
        newer,
        empty_flags,
        empty_candidates,
    ):
        assert output.device.type == "cpu"
    assert empty_flags.shape == (0, 600) and empty_candidates.shape == (0, 600, 7)
    torch.testing.assert_close(flags, original_flags)
    torch.testing.assert_close(candidates, original_candidates)
    good = _check_candidates(chain, targets, newer_flags, newer)
    within_step = (newer.float().double() - q[:, None]).abs().amax(-1) <= 0.05
    assert (good & within_step).any(-1).all()
    assert torch.isfinite(newer).all()


@pytest.mark.gpu
@pytest.mark.parametrize("search", ["fixed", "seed_local"])
def test_rr_cuda_workspace_growth_preserves_inflight_results_on_another_stream(
    rr_chain: pk.SerialChain,
    search: str,
) -> None:
    model = _extract_model(rr_chain)
    chain = rr_chain.to(dtype=torch.float64, device="cuda")
    engine = _PreparedRRCandidates(model, "cuda", search=search)
    q = torch.tensor(
        [[0.2, 0.4, -0.7, 1.2, 0.6, 0.8, -1.0]], device="cuda", dtype=torch.float64
    )
    target = chain.forward_kinematics(q).get_matrix()
    lower = torch.tensor(model.limits[:, 0], device="cuda")
    upper = torch.tensor(model.limits[:, 1], device="cuda")
    engine.prepare_buffers(1)
    first, second = torch.cuda.Stream(), torch.cuda.Stream()
    first.wait_stream(torch.cuda.current_stream())
    second.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(first):
        flags, candidates = engine.generate(target, q, lower, upper)
    # Growth frees the original workspace while its first-stream kernels may
    # still be running. Allocator reuse must respect the recorded usage stream.
    with torch.cuda.stream(second):
        newer_flags, newer = engine.generate(
            target.expand(3, -1, -1), q.expand(3, -1), lower, upper
        )
    torch.cuda.current_stream().wait_stream(first)
    torch.cuda.current_stream().wait_stream(second)
    assert _check_candidates(chain, target, flags, candidates).any()
    assert (
        _check_candidates(chain, target.expand(3, -1, -1), newer_flags, newer)
        .any(dim=-1)
        .all()
    )
