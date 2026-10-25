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
    actual = chain.forward_kinematics(candidates.reshape(-1, 7)).get_matrix()
    actual = actual.reshape(len(target), 384, 4, 4)
    error = (actual - target[:, None]).norm(dim=(-1, -2))
    return flags.bool() & (error <= 1e-5)


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
def test_rr_cuda_chunking_noncontiguous_targets_and_stream_order(
    rr_chain: pk.SerialChain,
) -> None:
    model = _extract_model(rr_chain)
    chain = rr_chain.to(dtype=torch.float64, device="cuda")
    engine = _PreparedRRCandidates(model, "cuda")
    q = torch.tensor(
        [[0.2, 0.4, -0.7, 1.2, 0.6, 0.8, -1.0]], device="cuda", dtype=torch.float64
    ).expand(33, 7)
    target = chain.forward_kinematics(q).get_matrix()
    storage = torch.empty((33, 4, 8), device="cuda", dtype=torch.float64)
    storage[:, :, ::2] = target
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        flags, candidates = engine.generate(storage[:, :, ::2])
        good = _check_candidates(chain, target, flags, candidates)
    torch.cuda.current_stream().wait_stream(stream)
    assert good.any(dim=-1).all()
    assert engine._capacity == engine.max_chunk_size
    assert sum(t.nbytes for t in engine._workspace.values()) < 32 * 1024**2


def test_rr_empty_batch(rr_chain: pk.SerialChain) -> None:
    engine = _PreparedRRCandidates(_extract_model(rr_chain), "cpu")
    valid, candidates = engine.generate(torch.empty((0, 4, 4), dtype=torch.float64))
    assert valid.shape == (0, 384)
    assert candidates.shape == (0, 384, 7)


@pytest.mark.gpu
def test_rr_cuda_workspace_growth_preserves_inflight_results_on_another_stream(
    rr_chain: pk.SerialChain,
) -> None:
    model = _extract_model(rr_chain)
    chain = rr_chain.to(dtype=torch.float64, device="cuda")
    engine = _PreparedRRCandidates(model, "cuda")
    q = torch.tensor(
        [[0.2, 0.4, -0.7, 1.2, 0.6, 0.8, -1.0]], device="cuda", dtype=torch.float64
    )
    target = chain.forward_kinematics(q).get_matrix()
    engine.prepare_buffers(1)
    first, second = torch.cuda.Stream(), torch.cuda.Stream()
    first.wait_stream(torch.cuda.current_stream())
    second.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(first):
        flags, candidates = engine.generate(target)
    # Growth frees the original workspace while its first-stream kernels may
    # still be running. Allocator reuse must respect the recorded usage stream.
    with torch.cuda.stream(second):
        newer_flags, newer = engine.generate(target.expand(3, -1, -1))
    torch.cuda.current_stream().wait_stream(first)
    torch.cuda.current_stream().wait_stream(second)
    assert _check_candidates(chain, target, flags, candidates).any()
    assert (
        _check_candidates(chain, target.expand(3, -1, -1), newer_flags, newer)
        .any(dim=-1)
        .all()
    )
