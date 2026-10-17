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

"""CPU TOPPRA projection and inference scheduling contracts."""

from __future__ import annotations

import numpy as np
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import os
import subprocess
import sys
import textwrap
from threading import Event
import pytest
import torch
import warp as wp
from scipy.optimize import linprog

from embodichain.compute.trajectory._toppra_warp import _retime_toppra_warp
from embodichain.compute.trajectory import _toppra_warp as adapter
from embodichain.compute.trajectory._warp import toppra as kernels


def _project(planes: np.ndarray) -> np.ndarray:
    """Compute the largest feasible x for rest-feasible half-plane batches."""
    wp.init()
    batch = len(planes)
    arrays = [
        wp.from_numpy(
            np.ascontiguousarray(planes[:, None, :, index]),
            dtype=wp.float64,
            device="cpu",
        )
        for index in range(3)
    ]
    counts = wp.from_numpy(np.ones(batch, dtype=np.int32), device="cpu")
    caps = wp.empty((batch, 1), dtype=wp.float64, device="cpu")
    wp.launch(
        kernels._project_caps_cpu,
        dim=(batch, 1),
        inputs=[*arrays, counts, caps],
        device="cpu",
    )
    return caps.numpy()[:, 0]


@pytest.mark.parametrize("scale", [1e-100, 1.0, 1e100])
def test_cpu_projection_matches_analytic_polygons(scale: float) -> None:
    # Every inequality has the form a*x + b*y <= c, with x,y >= 0.
    planes = (
        np.array(
            [
                [
                    [1.0, 1.0, 1.0],
                    [1.0, -1.0, 0.0],
                    [1.0, 0.0, 2.0],
                    [0.0, 1.0, 2.0],
                ],  # x <= y and x+y <= 1
                [
                    [1.0, 0.0, 2.0],
                    [0.0, 1.0, 3.0],
                    [0.0, -1.0, 0.0],
                    [0.0, 0.0, 0.0],
                ],  # independent box
                [
                    [1.0, 1.0, 1.0],
                    [1.0, -1.0, 1.0],
                    [1.0, 0.0, 2.0],
                    [0.0, 1.0, 2.0],
                ],  # optimum at y=0
                [
                    [1.0, -1.0, 0.0],
                    [-1.0, 1.0, 0.0],
                    [1.0, 0.0, 3.0],
                    [0.0, 1.0, 2.0],
                ],  # line x=y
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 2.0],
                    [0.0, -1.0, 0.0],
                    [0.0, 0.0, 0.0],
                ],  # locked x
            ]
        )
        * scale
    )
    np.testing.assert_allclose(
        _project(planes), [0.5, 2.0, 1.0, 2.0, 0.0], rtol=1e-12, atol=1e-12
    )


def test_cpu_projection_matches_independent_linear_programs() -> None:
    rng = np.random.default_rng(712)
    # This includes 14-DoF-sized plane sets, zero one-sided bounds and parallel
    # planes. SciPy's HiGHS LP is independent of the TOPPRA projection algorithm.
    planes = rng.normal(size=(96, 86, 3))
    planes[:, :, 2] = rng.uniform(0, 2, (96, 86))
    planes[::3, :4, 2] = 0
    planes[:, -2, :] = [1.0, 0.0, 3.0]
    planes[:, -1, :] = [0.0, 1.0, 4.0]
    expected = []
    for row in planes:
        solution = linprog(
            [-1.0, 0.0],
            A_ub=row[:, :2],
            b_ub=row[:, 2],
            bounds=[(0, None), (0, None)],
            method="highs",
        )
        assert solution.success
        expected.append(solution.x[0])
    np.testing.assert_allclose(_project(planes), expected, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("sampling", [{"sample_count": 129}, {"sample_dt": 0.07}])
def test_cpu_optimized_forward_matches_differentiable_solver(sampling: dict) -> None:
    rng = np.random.default_rng(731)
    points = torch.tensor(rng.normal(size=(8, 16, 7)), dtype=torch.float64)
    velocity = np.column_stack((-rng.uniform(0.4, 1.2, 7), rng.uniform(0.5, 1.4, 7)))
    acceleration = np.column_stack(
        (-rng.uniform(0.8, 1.6, 7), rng.uniform(0.9, 1.8, 7))
    )
    optimized = _retime_toppra_warp(points, velocity, acceleration, **sampling)
    tracked = _retime_toppra_warp(
        points.requires_grad_(), velocity, acceleration, **sampling
    )
    assert torch.equal(optimized["success"], tracked["success"])
    for key in ("positions", "velocities", "accelerations", "dt"):
        torch.testing.assert_close(optimized[key], tracked[key], rtol=1e-10, atol=1e-10)
    tracked["dt"].sum().backward()
    assert torch.isfinite(points.grad).all()


@pytest.mark.parametrize("sampling", [{"sample_count": 33}, {"sample_dt": 0.07}])
@pytest.mark.parametrize("mode", ["grad_enabled", "no_grad", "inference"])
def test_cpu_chunks_preserve_padding_failure_and_torch_modes(
    monkeypatch: pytest.MonkeyPatch, sampling: dict, mode: str
) -> None:
    # Force two chunks on small inputs. The first chunk is entirely stationary
    # or failed, so its local length is 2 even for quantity=33. The second
    # includes different durations and ignored NaN tail padding.
    points = torch.tensor(
        [
            [[2.0, 3.0], [2.0, 3.0], [2.0, 3.0], [2.0, 3.0]],
            [[0.0, 0.0], [float("nan"), 0.0], [0.2, 0.1], [0.5, 0.2]],
            [[0.0, 0.0], [0.2, 0.1], [0.5, 0.2], [float("nan"), 0.0]],
            [[0.0, 0.0], [0.3, 0.1], [0.6, 0.2], [1.0, 0.3]],
        ],
        dtype=torch.float64,
    )
    lengths = torch.tensor([4, 4, 3, 4])
    if mode != "grad_enabled":
        points.requires_grad_()
    context = (
        torch.inference_mode()
        if mode == "inference"
        else torch.no_grad() if mode == "no_grad" else nullcontext()
    )
    observed_modes = []
    original = adapter._cpu_chunk

    def record(*args, **kwargs):
        result = original(*args, **kwargs)
        observed_modes.append(
            (result["positions"].requires_grad, result["positions"].is_inference())
        )
        return result

    monkeypatch.setattr(adapter, "_cpu_chunk", record)
    with context:
        with monkeypatch.context() as patch:
            patch.setattr(adapter, "_cpu_workers", lambda *args: 1)
            expected = _retime_toppra_warp(
                points, 1.0, 2.0, path_lengths=lengths, **sampling
            )
        monkeypatch.setattr(adapter, "_cpu_workers", lambda *args: 2)
        actual = _retime_toppra_warp(points, 1.0, 2.0, path_lengths=lengths, **sampling)
    assert actual["success"].tolist() == [True, False, True, True]
    for key in actual:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    assert observed_modes == [(False, mode == "inference")] * 2


def test_cpu_parallel_noncontiguous_rows_are_independent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    points = torch.randn(
        12, 3, 7, generator=torch.Generator().manual_seed(491), dtype=torch.float64
    ).transpose(1, 2)
    assert not points.is_contiguous()
    monkeypatch.setattr(adapter, "_cpu_workers", lambda *args: 3)
    batched = _retime_toppra_warp(points, 1.0, 2.0, sample_dt=0.09)
    monkeypatch.setattr(adapter, "_cpu_workers", lambda *args: 1)
    for row in range(len(points)):
        expected = _retime_toppra_warp(points[row : row + 1], 1.0, 2.0, sample_dt=0.09)
        valid = expected["positions"].shape[1]
        for key in ("positions", "velocities", "accelerations", "dt"):
            torch.testing.assert_close(
                batched[key][row : row + 1, :valid], expected[key], rtol=0, atol=0
            )
        assert batched["dt"][row, valid:].count_nonzero() == 0


def test_cpu_inference_does_not_record_into_foreign_warp_tape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    points = torch.linspace(0, 1, 24, dtype=torch.float64).reshape(4, 3, 2)
    monkeypatch.setattr(adapter, "_cpu_workers", lambda *args: 2)
    # Compile/init before capture. Warp's tape is process-global; ordinary
    # inference must not record launches into an unrelated caller's tape.
    _retime_toppra_warp(points, 1.0, 2.0, sample_count=11)
    with wp.Tape() as tape:
        actual = _retime_toppra_warp(points, 1.0, 2.0, sample_count=11)
    assert actual["success"].all()
    assert not tape.launches


def test_cpu_concurrent_requests_use_independent_buffers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    points = torch.randn(
        8, 4, 3, generator=torch.Generator().manual_seed(292), dtype=torch.float64
    )
    monkeypatch.setattr(adapter, "_cpu_workers", lambda *args: 1)
    expected = [
        _retime_toppra_warp(points * scale, 1.0, 2.0, sample_count=27)
        for scale in (1.0, 2.0, 3.0)
    ]
    monkeypatch.setattr(adapter, "_cpu_workers", lambda *args: 2)
    with ThreadPoolExecutor(max_workers=3) as requests:
        futures = [
            requests.submit(
                _retime_toppra_warp, points * scale, 1.0, 2.0, sample_count=27
            )
            for scale in (1.0, 2.0, 3.0)
        ]
        actual = [future.result() for future in futures]
    for result, reference in zip(actual, expected):
        for key in result:
            torch.testing.assert_close(result[key], reference[key], rtol=0, atol=0)


def test_cpu_worker_error_drains_other_tasks_and_pool_remains_usable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    points = torch.zeros(4, 3, 2, dtype=torch.float64)
    points[:, :, 0] = torch.linspace(0.0, 1.0, 3)
    points[2:, :, 1] = 1.0
    monkeypatch.setattr(adapter, "_cpu_workers", lambda *args: 2)
    # Load before injecting an error so this tests request cleanup, not JIT.
    _retime_toppra_warp(points, 1.0, 2.0, sample_count=11)
    second_started, second_finished = Event(), Event()
    original = adapter._cpu_chunk

    def faulty(chunk, *args, **kwargs):
        if chunk[0, 0, 1] == 0:
            assert second_started.wait(5)
            raise RuntimeError("injected CPU worker failure")
        second_started.set()
        result = original(chunk, *args, **kwargs)
        second_finished.set()
        return result

    with monkeypatch.context() as patch:
        patch.setattr(adapter, "_cpu_chunk", faulty)
        with pytest.raises(RuntimeError, match="injected CPU worker failure"):
            _retime_toppra_warp(points, 1.0, 2.0, sample_count=11)
        assert second_finished.is_set()
    assert _retime_toppra_warp(points, 1.0, 2.0, sample_count=11)["success"].all()


@pytest.mark.skipif(not hasattr(os, "fork"), reason="fork is unavailable")
def test_cpu_pool_is_recreated_after_fork() -> None:
    # Isolate fork from any CUDA context or other native worker pools in pytest.
    code = textwrap.dedent("""
        import os, signal, time
        import torch
        from embodichain.compute.trajectory import _toppra_warp as m
        m._cpu_workers=lambda *args:2
        points=torch.linspace(0,1,24,dtype=torch.float64).reshape(4,3,2)
        m._retime_toppra_warp(points,1.,2.,sample_count=11)
        child=os.fork()
        if child==0:
            try:
                result=m._retime_toppra_warp(points,1.,2.,sample_count=11)
                os._exit(0 if result['success'].all() else 2)
            except BaseException:
                os._exit(3)
        deadline=time.monotonic()+8
        while time.monotonic()<deadline:
            finished,status=os.waitpid(child,os.WNOHANG)
            if finished:
                assert status==0,status
                break
            time.sleep(.02)
        else:
            os.kill(child,signal.SIGKILL)
            os.waitpid(child,0)
            raise AssertionError('inherited CPU pool deadlocked')
    """)
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        },
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
