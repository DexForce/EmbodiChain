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

"""Original X5A geometry and bounded CUDA polishing regression witnesses."""

from __future__ import annotations

import os
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from embodichain.lab.sim.motion.solvers import SSIKSolver, SSIKSolverCfg
from embodichain_robots import ARX5Cfg, FlexivRizon4SCfg

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


def _solver(device: str = "cpu", **options: object) -> SSIKSolver:
    preset = ARX5Cfg.from_dict({})
    original = preset.solver_cfg["arm"]
    chain = preset.build_pk_serial_chain(torch.device(device))["arm"]
    return SSIKSolverCfg(
        urdf_path=preset.fpath,
        root_link_name=original.root_link_name,
        end_link_name=original.end_link_name,
        joint_names=chain.get_joint_parameter_names(),
        tcp=original.tcp,
        **options,
    ).init_solver(device=device, pk_serial_chain=chain)


def _assert_source_pose(
    solver: SSIKSolver, target: torch.Tensor, qpos: torch.Tensor
) -> None:
    tcp = torch.as_tensor(solver.tcp_xpos, device=solver.device, dtype=torch.float64)
    chain = solver.pk_serial_chain.to(dtype=torch.float64)
    actual = chain.forward_kinematics(qpos.double()).get_matrix() @ tcp
    _assert_transform(target, actual)


def _assert_transform(target: torch.Tensor, actual: torch.Tensor) -> None:
    actual = actual.cpu().numpy()
    expected = target.cpu().double().numpy()
    translation = np.linalg.norm(actual[:, :3, 3] - expected[:, :3, 3], axis=-1)
    rotation = (
        Rotation.from_matrix(actual[:, :3, :3]).inv()
        * Rotation.from_matrix(expected[:, :3, :3])
    ).magnitude()
    assert np.all(translation <= 1e-5)
    assert np.all(rotation <= 1e-5)


def test_actual_arx5_requires_explicit_near_parallel_geometry() -> None:
    # The authored 7.24e-6-radian axis mismatch is larger than PK roundoff.
    with pytest.raises(ValueError, match="geometry"):
        _solver()
    with pytest.raises(ValueError, match="geometry"):
        _solver(geometry_mode="near_parallel", axis_alignment_tolerance=1e-6)


@pytest.mark.parametrize("device", _DEVICES)
def test_actual_arx5_original_model_tangent_and_singular_branches(device: str) -> None:
    solver = _solver(device, geometry_mode="near_parallel", chunk_size=2)
    # The first two poses are independently constructed raw-PK Jacobian
    # singular witnesses. The latter two need tangent projection in the
    # parallel surrogate; the exact source model remains authoritative.
    joints = torch.tensor(
        [
            [
                8.51198858,
                -4.94784783,
                -3.87848850,
                3.23189706,
                -7.77026919,
                -1.86587036,
            ],
            [-2.22905342, -6.86660366, -0.49254067, 8.36887464, 9.45439372, 9.46502063],
            [2.73665891, -8.53988670, 2.91669473, -7.51980882, -0.25815424, 8.68778765],
            [
                -2.96355401,
                -9.98779149,
                2.91586633,
                -7.57319299,
                4.79461378,
                -3.67113247,
            ],
        ],
        dtype=torch.float64,
        device=device,
    )
    chain = solver.pk_serial_chain.to(dtype=torch.float64)
    tcp = torch.as_tensor(solver.tcp_xpos, device=device, dtype=torch.float64)
    targets = (chain.forward_kinematics(joints).get_matrix() @ tcp).float()
    seed = solver.get_default_qpos_seed()[None].expand(len(joints), -1)
    success, result = solver.get_ik(targets, seed)
    assert success.tolist() == [True] * len(joints)
    assert result.dtype == torch.float32 and result.device.type == device
    _assert_source_pose(solver, targets, result)

    mask, candidates = solver.get_ik(targets[:1], seed[:1], return_all_solutions=True)
    assert mask.shape == (1, 9) and candidates.shape == (1, 9, 6)
    assert bool(mask.any())
    _assert_source_pose(
        solver, targets[:1].expand(int(mask.sum()), -1, -1), candidates[mask]
    )


@pytest.mark.parametrize("device", _DEVICES)
def test_seed_initialization_is_always_a_ninth_candidate(device: str) -> None:
    solver = _solver(device, geometry_mode="near_parallel", max_joint_step=0.05)
    seed = torch.tensor([[0.4, -0.7, 0.6, -0.4, 0.5, 0.8]], device=device)
    tcp = torch.as_tensor(solver.tcp_xpos, device=device, dtype=torch.float32)
    target = solver.pk_serial_chain.forward_kinematics(seed).get_matrix() @ tcp
    mask, candidates = solver.get_ik(target, seed, return_all_solutions=True)
    # The seed slot participates even when analytical slots already work.
    # This protects the explicit hybrid candidate set, rather than a hidden
    # retry entered only after another candidate-generation path fails.
    assert mask.shape == (1, 9) and bool(mask[0, :8].any())
    assert bool(mask[0, 8])
    assert bool(((candidates[mask] - seed).abs() <= 0.05).all())
    _assert_source_pose(solver, target, candidates[:, 8])


@pytest.mark.parametrize("device", _DEVICES)
def test_continuous_batch_reseeds_across_the_near_parallel_wrist(device: str) -> None:
    solver = _solver(device, geometry_mode="near_parallel", max_joint_step=0.05)
    # Seed2531's reachable smooth-joint path3 crossed this wrist configuration.
    # Keeping one initial candidate seed lost73 of192 frames; refreshing from
    # each prior result must preserve the feasible bounded path in either direction.
    endpoints = torch.tensor(
        [
            [
                -2.922868307,
                2.841086898,
                1.203681351,
                -0.194751795,
                -1.817375245,
                2.267730447,
            ],
            [
                0.753565957,
                2.927347550,
                -2.142941306,
                -1.357887040,
                2.805696481,
                1.769882563,
            ],
        ],
        dtype=torch.float64,
        device=device,
    )
    phase = torch.linspace(0, 1, 192, dtype=torch.float64, device=device)
    joints = (
        endpoints[0]
        + (endpoints[1] - endpoints[0])
        * (3 * phase.square() - 2 * phase.pow(3))[:, None]
    )
    source = torch.stack((joints, joints.flip(0)))
    assert bool((source.diff(dim=1).abs() < 0.05).all())
    chain = solver.pk_serial_chain.to(dtype=torch.float64)
    tcp = torch.as_tensor(solver.tcp_xpos, device=device, dtype=torch.float64)
    targets = (
        (chain.forward_kinematics(source.reshape(-1, 6)).get_matrix() @ tcp)
        .reshape(2, 192, 4, 4)
        .float()
    )
    seeds = source[:, 0].float()
    valid, path = solver._get_continuous_ik_path(targets, seeds)
    assert valid.shape == (2, 192) and bool(valid.all())
    assert bool(
        (torch.cat((seeds[:, None], path), dim=1).diff(dim=1).abs() <= 0.05).all()
    )
    _assert_source_pose(solver, targets.reshape(-1, 4, 4), path.reshape(-1, 6))


@pytest.mark.parametrize("device", _DEVICES)
def test_tangent_control_uses_actual_tcp_and_separate_pose_tolerances(
    device: str,
) -> None:
    solver = _solver(device, geometry_mode="near_parallel", max_joint_step=0.05)
    # Seed2531 line7 frames43/44 admit these independently verified float32
    # poses inside both acceptance balls. Flange-space L2 polishing rejected
    # frame43 with position1.01e-5m even though rotation retained ample slack.
    seed = torch.tensor(
        [
            [
                2.2403149605,
                -0.4634959400,
                2.9168815613,
                2.9063024521,
                -1.3069185019,
                1.3950436115,
            ]
        ],
        device=device,
    )
    target = torch.tensor(
        [
            [0.0711401105, -0.3881445825, -0.9188485742, 0.0721772239],
            [-0.1633595377, 0.9042108655, -0.3946090341, -0.3609723747],
            [0.9839983582, 0.1781752110, 0.0009185353, -0.1294952929],
            [0.0, 0.0, 0.0, 1.0],
        ],
        device=device,
    ).repeat(2, 1, 1)
    target[1, 0, 3] = 0.0721791536
    witness = torch.tensor(
        [
            [
                2.2403213978,
                -0.4634903371,
                2.9168827534,
                2.9062840939,
                -1.3069145679,
                1.3950216770,
            ],
            [
                2.2403254509,
                -0.4634888470,
                2.9168827534,
                2.9062829018,
                -1.3069109917,
                1.3950191736,
            ],
        ],
        device=device,
    )
    _assert_source_pose(solver, target, witness)
    assert bool((torch.cat((seed, witness)).diff(dim=0).abs() < 0.05).all())
    valid, result = solver._get_continuous_ik_path(target[None], seed)
    assert bool(valid.all())
    _assert_source_pose(solver, target, result[0])
    assert bool((torch.cat((seed, result[0])).diff(dim=0).abs() <= 0.05).all())


@pytest.mark.parametrize("device", _DEVICES)
def test_near_parallel_hot_path_has_no_host_copy_and_unreachable_failure(
    device: str,
) -> None:
    solver = _solver(device, geometry_mode="near_parallel")
    seed = solver.get_default_qpos_seed()[None]
    tcp = torch.as_tensor(solver.tcp_xpos, device=device, dtype=torch.float32)
    target = solver.pk_serial_chain.forward_kinematics(seed).get_matrix() @ tcp
    target = target.repeat(2, 1, 1)
    target[1, :3, 3] = 100
    # Initialization may inspect the source chain. The actual solve cannot
    # silently transfer to the CPU reference or a Python numerical solver.
    with patch.object(torch.Tensor, "cpu", side_effect=AssertionError("host transfer")):
        with patch.object(
            torch.Tensor, "numpy", side_effect=AssertionError("host transfer")
        ):
            success, result = solver.get_ik(target, seed.expand(2, -1))
    assert success.tolist() == [True, False]
    torch.testing.assert_close(result[1], seed[0], atol=0, rtol=0)
    _assert_source_pose(solver, target[:1], result[:1])


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("backend", ["near", "rr"])
def test_owned_outputs_survive_solver_destruction_between_cuda_streams(
    backend: str,
) -> None:
    if backend == "near":
        solver = _solver("cuda", geometry_mode="near_parallel")
        seed = torch.tensor([[0.4, -0.7, 0.6, -0.4, 0.5, 0.8]], device="cuda")
    else:
        preset = FlexivRizon4SCfg.from_dict({})
        original = preset.solver_cfg["arm"]
        solver = SSIKSolverCfg(
            backend="cuda_rr",
            urdf_path=preset.fpath,
            root_link_name=original.root_link_name,
            end_link_name=original.end_link_name,
            tcp=original.tcp,
        ).init_solver(
            "cuda",
            pk_serial_chain=preset.build_pk_serial_chain(torch.device("cuda"))["arm"],
        )
        seed = torch.tensor([[0.2, 0.4, 0.0, 1.2, 0.6, 0.8, -1.0]], device="cuda")
    chain = solver.pk_serial_chain
    tcp = np.array(solver.tcp_xpos, copy=True)
    target = chain.forward_kinematics(seed).get_matrix() @ torch.as_tensor(
        tcp, device="cuda", dtype=torch.float32
    )
    valid, _ = solver.get_ik(target, seed)
    assert bool(valid.all())
    # Keep the matrix allocated on A; only its address is needed to exercise
    # allocator reuse, not to assert any private implementation detail.
    pointer = solver._tcp_inverse_tensor.data_ptr()
    torch.cuda.synchronize()
    compute, consumer = torch.cuda.Stream(), torch.cuda.Stream()
    finished = torch.cuda.Event()
    with torch.cuda.stream(compute):
        torch.cuda._sleep(300_000_000)
        valid, result = solver.get_ik(target, seed)
        finished.record(compute)
    with torch.cuda.stream(consumer):
        consumer.wait_event(finished)
        observed_valid, observed_result = valid.clone(), result.clone()
    del solver
    allocations = []
    for _ in range(8192):
        if finished.query():
            break
        block = torch.empty((4, 4), dtype=torch.float64, device="cuda")
        allocations.append(block)
        if block.data_ptr() == pointer:
            block.fill_(float("nan"))
            break
    torch.cuda.synchronize()
    assert bool(observed_valid.all())
    actual = chain.to(dtype=torch.float64).forward_kinematics(
        observed_result.double()
    ).get_matrix() @ torch.as_tensor(tcp, device="cuda", dtype=torch.float64)
    _assert_transform(target, actual)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_first_solver_in_a_fresh_process_can_initialize_cuda() -> None:
    # A preceding CPU solver initializes Warp globally and used to hide the
    # constructor's stream access before its runtime existed. Exercise the
    # actual first-CUDA-constructor contract in an independent interpreter.
    program = """
import faulthandler
import json
import os
import sys
import time

started = time.monotonic()
def stage(phase, **details):
    print(json.dumps({"phase": phase, "elapsed_seconds": time.monotonic() - started,
                      "pid": os.getpid(), **details}), flush=True)

# PK temporarily redirects fd2 while constructing a chain. Keep the crash
# and timeout stack destination independent of that suppression context.
trace_output = os.fdopen(os.dup(sys.stderr.fileno()), "w", buffering=1)
faulthandler.enable(file=trace_output)
faulthandler.dump_traceback_later(60, repeat=True, file=trace_output)
stage("before_torch_import", executable=sys.executable, cwd=os.getcwd(),
      native_threads={name: os.environ.get(name) for name in
                      ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                       "NUMEXPR_NUM_THREADS")},
      warp_cache_override=os.environ.get("WARP_CACHE_PATH"))
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
stage("torch_imported", version=torch.__version__,
      intraop_threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads())
from embodichain.lab.sim.motion.solvers import SSIKSolverCfg
from embodichain_robots import ARX5Cfg
stage("solver_imported", source=sys.modules[SSIKSolverCfg.__module__].__file__)
preset = ARX5Cfg.from_dict({})
original = preset.solver_cfg["arm"]
stage("preset_resolved", urdf=preset.fpath)
stage("before_cuda_chain")
chain = preset.build_pk_serial_chain(torch.device("cuda"))["arm"]
stage("cuda_chain_ready")
stage("before_first_solver_constructor")
solver = SSIKSolverCfg(
    urdf_path=preset.fpath,
    root_link_name=original.root_link_name,
    end_link_name=original.end_link_name,
    joint_names=chain.get_joint_parameter_names(),
    tcp=original.tcp,
    geometry_mode="near_parallel",
).init_solver(device="cuda", pk_serial_chain=chain)
import warp as wp
stage("solver_ready", warp_version=wp.__version__, warp_cache=wp.config.kernel_cache_dir)
seed = solver.get_default_qpos_seed()[None]
stage("before_target_fk")
target = chain.forward_kinematics(seed).get_matrix() @ torch.as_tensor(
    original.tcp, device="cuda", dtype=torch.float32
)
stage("target_ready")
stage("before_first_ik")
success, joints = solver.get_ik(target, seed)
stage("ik_returned")
torch.cuda.synchronize()
stage("cuda_completed")
assert success.shape == (1,) and success.all()
assert joints.shape == (1, 6) and joints.is_cuda
faulthandler.cancel_dump_traceback_later()
stage("passed")
"""
    # The GPU lane does not use the fast lane's native-thread defaults.
    # Bound this correctness subprocess independently, preserve its inherited
    # Warp cache, and leave the first solver responsible for Warp initialization.
    environment = os.environ.copy()
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        environment[name] = "1"
    try:
        completed = subprocess.run(
            [sys.executable, "-u", "-c", program],
            capture_output=True,
            text=True,
            timeout=240,
            check=False,
            env=environment,
        )
    except subprocess.TimeoutExpired as error:

        def decoded(value: str | bytes | None) -> str:
            return (
                value.decode(errors="replace")
                if isinstance(value, bytes)
                else value or ""
            )

        pytest.fail(
            "Fresh CUDA constructor/IK exceeded240seconds. Last subprocess stages and stacks:\n"
            + decoded(error.stdout)
            + "\n"
            + decoded(error.stderr),
            pytrace=False,
        )
    assert completed.returncode == 0, completed.stdout + completed.stderr
