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

import gc
import math

import numpy as np
import pytest
import torch

from embodichain.lab.sim.motion.planners.toppra_planner import _toppra_solve_one_env
from embodichain.lab.sim.motion.planners.utils import TrajectorySampleMethod


# Numerical contracts are also exercised without the simulation adapter in
# tests/compute/test_toppra.py.
class TestToppraWorker:
    def test_solve_one_env_quantity(self):
        # 2-waypoint, 6-DOF
        wp = np.array([[0.0] * 6, [0.5] * 6])
        out = _toppra_solve_one_env(
            waypoints=wp,
            vel_constraint=1.0,
            acc_constraint=2.0,
            sample_method=TrajectorySampleMethod.QUANTITY,
            sample_interval=20,
        )
        assert out["success"] is True
        assert out["positions"].shape == (20, 6)
        assert out["velocities"].shape == (20, 6)
        assert out["dt"].shape == (20,)

    def test_solve_one_env_infeasible_exception(self):
        # A path needs at least two waypoints; the worker isolates this failure.
        wp = np.array([[0.0] * 6])
        out = _toppra_solve_one_env(
            waypoints=wp,
            vel_constraint=1.0,
            acc_constraint=2.0,
            sample_method=TrajectorySampleMethod.QUANTITY,
            sample_interval=10,
        )
        assert out["success"] is False

    def test_solve_one_env_time_sampling(self):
        wp = np.array([[0.0] * 6, [0.5] * 6])
        out = _toppra_solve_one_env(
            waypoints=wp,
            vel_constraint=1.0,
            acc_constraint=2.0,
            sample_method=TrajectorySampleMethod.TIME,
            sample_interval=0.05,
        )
        assert out["success"] is True
        assert out["positions"].shape[0] == out["n"]
        assert out["n"] >= 2

    def test_solve_one_env_same_waypoint_shortcut(self):
        wp = np.array([[0.3] * 6, [0.3] * 6])  # identical
        out = _toppra_solve_one_env(
            waypoints=wp,
            vel_constraint=1.0,
            acc_constraint=2.0,
            sample_method=TrajectorySampleMethod.QUANTITY,
            sample_interval=20,
        )
        assert out["success"] is True
        assert out["n"] == 2
        assert out["dt"].sum() == 0.0

    def test_solve_one_env_duplicate_plateau(self):
        # Long plateaus of identical waypoints (e.g. from interpolating a
        # segment where start_qpos equals the first target) must not make
        # TOPPRA's controllable-set computation fail.
        wp = np.array([[0.3] * 6] * 12 + [[0.5] * 6] * 12)
        out = _toppra_solve_one_env(
            waypoints=wp,
            vel_constraint=1.0,
            acc_constraint=2.0,
            sample_method=TrajectorySampleMethod.QUANTITY,
            sample_interval=20,
        )
        assert out["success"] is True
        assert out["positions"].shape == (20, 6)


class TestToppraCfgFields:
    def test_cfg_defaults(self):
        from embodichain.lab.sim.motion.planners.toppra_planner import ToppraPlannerCfg

        cfg = ToppraPlannerCfg(robot_uid="x")
        assert cfg.max_workers is None
        assert cfg.mp_context is None  # None => auto-select by device

    def test_mp_context_auto_resolution(self):
        # _resolve_mp_context is a pure function of (cfg value, device) — no sim
        # needed. None auto-selects: fork on CPU, spawn on GPU; explicit values
        # are honored as-is.
        import torch

        from embodichain.lab.sim.motion.planners.toppra_planner import ToppraPlanner

        resolve = ToppraPlanner._resolve_mp_context
        # Auto: CPU -> fork, CUDA -> spawn (creating a torch.device does NOT
        # initialize CUDA, so this is safe to assert without a GPU).
        assert resolve(None, torch.device("cpu")) == "fork"
        assert resolve(None, torch.device("cuda", 0)) == "spawn"
        assert resolve(None, torch.device("cuda")) == "spawn"
        # Explicit override wins regardless of device.
        assert resolve("spawn", torch.device("cpu")) == "spawn"
        assert resolve("fork", torch.device("cuda", 0)) == "fork"


class TestToppraPlanBatched:
    def _make_planner(self):
        from embodichain.lab.sim.motion.planners.toppra_planner import (
            ToppraPlanner,
            ToppraPlannerCfg,
        )
        from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
        from embodichain.lab.sim.robots import CobotMagicCfg

        sim = SimulationManager(
            SimulationManagerCfg(headless=True, device="cpu", num_envs=2)
        )
        robot = sim.add_robot(
            cfg=CobotMagicCfg.from_dict(
                {"uid": "t", "init_pos": [0, 0, 0.7775], "init_qpos": [0.0] * 16}
            )
        )
        sim.prepare()
        planner = ToppraPlanner(ToppraPlannerCfg(robot_uid="t", max_workers=1))
        return planner, sim

    def test_plan_batched_quantity_uniform_N(self):
        from embodichain.lab.sim.motion.planners.utils import (
            PlanState,
            TrajectorySampleMethod,
        )
        from embodichain.lab.sim.motion.planners.toppra_planner import ToppraPlanOptions

        planner, sim = self._make_planner()
        try:
            B, dofs = 2, 6
            wp = torch.zeros(B, dofs)
            wp[:, 0] = torch.linspace(0.0, 0.4, B)
            states = [
                PlanState.from_qpos(torch.zeros(B, dofs)),
                PlanState.from_qpos(wp),
            ]
            opts = ToppraPlanOptions(
                sample_method=TrajectorySampleMethod.QUANTITY,
                sample_interval=15,
                constraints={"velocity": 1.0, "acceleration": 2.0},
            )
            r = planner.plan(states, opts)
            assert r.success.shape == (B,)
            assert r.success.all().item()
            assert r.positions.shape == (B, 15, dofs)
        finally:
            del planner
            gc.collect()
            sim.destroy()
            import embodichain.lab.sim as om

            om.SimulationManager.flush_cleanup_queue()

    def test_plan_batched_time_tailpads(self):
        from embodichain.lab.sim.motion.planners.utils import (
            PlanState,
            TrajectorySampleMethod,
        )
        from embodichain.lab.sim.motion.planners.toppra_planner import ToppraPlanOptions

        planner, sim = self._make_planner()
        try:
            B, dofs = 2, 6
            wp = torch.zeros(B, dofs)
            wp[:, 0] = torch.tensor([0.1, 0.9])  # different durations
            states = [
                PlanState.from_qpos(torch.zeros(B, dofs)),
                PlanState.from_qpos(wp),
            ]
            opts = ToppraPlanOptions(
                sample_method=TrajectorySampleMethod.TIME,
                sample_interval=0.05,
                constraints={"velocity": 1.0, "acceleration": 2.0},
            )
            r = planner.plan(states, opts)
            assert r.success.shape == (B,)
            assert r.positions.shape[0] == B
            assert r.duration.shape == (B,)

            # The env with the SHORTEST duration got tail-padded; its trailing
            # padded rows must equal its last real waypoint (held pose) and the
            # padded tail must be constant.
            shortest_env = int(r.duration.argmin().item())
            longest_env = int(r.duration.argmax().item())
            tail = r.positions[shortest_env]  # (max_n, DOF)
            max_n = tail.shape[0]
            # Reconstruct n_real for the shortest env the same way the worker does.
            n_real = max(2, int(math.ceil(r.duration[shortest_env].item() / 0.05)) + 1)
            # The longest env should not be padded (its n_real == max_n).
            assert r.positions[longest_env].shape[0] == max_n
            # Shortest env must actually have been padded (otherwise the test
            # isn't exercising the tail-pad branch).
            assert n_real < max_n, "expected shortest env to be tail-padded"
            held = tail[n_real - 1]  # last real waypoint
            padded = tail[n_real:]  # all padded rows
            assert torch.allclose(padded, held.expand(padded.shape))
        finally:
            del planner
            gc.collect()
            sim.destroy()
            import embodichain.lab.sim as om

            om.SimulationManager.flush_cleanup_queue()

    @pytest.mark.slow
    @pytest.mark.parametrize("mp_context", ["fork", "spawn"])
    def test_plan_batched_pool_path(self, mp_context):
        # Exercise the real ProcessPoolExecutor branch (max_workers=2, B=3)
        # under both start methods: 'fork' is the CPU default, 'spawn' is the
        # GPU fallback. Both must plan correctly and reap workers on GC.
        from embodichain.lab.sim.motion.planners.utils import (
            PlanState,
            TrajectorySampleMethod,
        )
        from embodichain.lab.sim.motion.planners.toppra_planner import (
            ToppraPlanner,
            ToppraPlannerCfg,
            ToppraPlanOptions,
        )
        from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
        from embodichain.lab.sim.robots import CobotMagicCfg

        sim = SimulationManager(
            SimulationManagerCfg(headless=True, device="cpu", num_envs=3)
        )
        sim.add_robot(
            cfg=CobotMagicCfg.from_dict(
                {"uid": "p", "init_pos": [0, 0, 0.7775], "init_qpos": [0.0] * 16}
            )
        )
        sim.prepare()
        planner = ToppraPlanner(
            ToppraPlannerCfg(robot_uid="p", max_workers=2, mp_context=mp_context)
        )
        assert planner._mp_context == mp_context
        try:
            B, dofs = 3, 6
            wp = torch.zeros(B, dofs)
            wp[:, 0] = torch.linspace(0.1, 0.5, B)
            states = [
                PlanState.from_qpos(torch.zeros(B, dofs)),
                PlanState.from_qpos(wp),
            ]
            opts = ToppraPlanOptions(
                sample_method=TrajectorySampleMethod.QUANTITY,
                sample_interval=12,
                constraints={"velocity": 1.0, "acceleration": 2.0},
            )
            r = planner.plan(states, opts)
            assert r.success.shape == (B,)
            assert r.success.all().item()
            assert r.positions.shape == (B, 12, dofs)
        finally:
            del planner
            gc.collect()
            sim.destroy()
            import embodichain.lab.sim as om

            om.SimulationManager.flush_cleanup_queue()

    @pytest.mark.slow
    @pytest.mark.parametrize("mp_context", ["fork", "spawn"])
    def test_workers_reaped_on_gc(self, mp_context):
        # Regression: abandoning the planner must reap its worker processes,
        # under both 'fork' (CPU default) and 'spawn' (GPU fallback).
        #
        # Workers install prctl(PR_SET_PDEATHSIG) in _worker_init, so the kernel
        # kills them the moment the parent process dies — including the
        # os._exit(0) path taken by SimulationManager.destroy(), which skips
        # every Python finalizer. That covers process exit. For in-process GC
        # of an abandoned planner (e.g. between tests in a long-running
        # session), __del__ -> _shutdown_pool() must terminate workers
        # synchronously so none leak.
        from embodichain.lab.sim.motion.planners.utils import (
            PlanState,
            TrajectorySampleMethod,
        )
        from embodichain.lab.sim.motion.planners.toppra_planner import (
            ToppraPlanner,
            ToppraPlannerCfg,
            ToppraPlanOptions,
        )
        from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
        from embodichain.lab.sim.robots import CobotMagicCfg

        sim = SimulationManager(
            SimulationManagerCfg(headless=True, device="cpu", num_envs=3)
        )
        sim.add_robot(
            cfg=CobotMagicCfg.from_dict(
                {
                    "uid": "close_reap",
                    "init_pos": [0, 0, 0.7775],
                    "init_qpos": [0.0] * 16,
                }
            )
        )
        sim.prepare()
        planner = ToppraPlanner(
            ToppraPlannerCfg(
                robot_uid="close_reap", max_workers=2, mp_context=mp_context
            )
        )
        assert planner._mp_context == mp_context
        worker_processes = []
        try:
            B, dofs = 3, 6
            wp = torch.zeros(B, dofs)
            wp[:, 0] = torch.linspace(0.1, 0.5, B)
            states = [
                PlanState.from_qpos(torch.zeros(B, dofs)),
                PlanState.from_qpos(wp),
            ]
            opts = ToppraPlanOptions(
                sample_method=TrajectorySampleMethod.QUANTITY,
                sample_interval=12,
                constraints={"velocity": 1.0, "acceleration": 2.0},
            )
            r = planner.plan(states, opts)
            assert r.success.all().item()

            # Capture the worker processes created by the executor.
            worker_processes = list(planner._pool._processes.values())
            assert len(worker_processes) == 2
        finally:
            # Abandon the planner; __del__ -> _shutdown_pool must reap workers.
            del planner
            gc.collect()
            for proc in worker_processes:
                assert not proc.is_alive(), (
                    f"TOPPRA worker process {proc.pid} survived planner GC; "
                    "worker processes must be reaped when the planner is collected."
                )
            sim.destroy()
            import embodichain.lab.sim as om

            om.SimulationManager.flush_cleanup_queue()


@pytest.mark.slow
class TestToppraNumericalRegression:
    def test_batched_equals_inline_single(self):
        from embodichain.lab.sim.motion.planners.toppra_planner import (
            _toppra_solve_one_env,
            ToppraPlanner,
            ToppraPlannerCfg,
            ToppraPlanOptions,
        )
        from embodichain.lab.sim.motion.planners.utils import (
            PlanState,
            TrajectorySampleMethod,
        )
        from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
        from embodichain.lab.sim.robots import CobotMagicCfg

        sim = SimulationManager(
            SimulationManagerCfg(headless=True, device="cpu", num_envs=4)
        )
        sim.add_robot(
            cfg=CobotMagicCfg.from_dict(
                {"uid": "r", "init_pos": [0, 0, 0.7775], "init_qpos": [0.0] * 16}
            )
        )
        sim.prepare()
        planner = ToppraPlanner(ToppraPlannerCfg(robot_uid="r", max_workers=1))
        try:
            B, dofs = 4, 6
            wp = torch.zeros(B, dofs)
            wp[:, 0] = torch.linspace(0.1, 0.6, B)
            states = [
                PlanState.from_qpos(torch.zeros(B, dofs)),
                PlanState.from_qpos(wp),
            ]
            opts = ToppraPlanOptions(
                sample_method=TrajectorySampleMethod.QUANTITY,
                sample_interval=20,
                constraints={"velocity": 1.0, "acceleration": 2.0},
            )
            r = planner.plan(states, opts)
            # Compare each env to the inline single-env solve
            for b in range(B):
                single = _toppra_solve_one_env(
                    np.stack([np.zeros(dofs), wp[b].numpy()]),
                    1.0,
                    2.0,
                    TrajectorySampleMethod.QUANTITY,
                    20,
                )
                assert np.allclose(
                    r.positions[b].cpu().numpy(), single["positions"], atol=1e-5
                )
        finally:
            del planner
            gc.collect()
            sim.destroy()
            import embodichain.lab.sim as om

            om.SimulationManager.flush_cleanup_queue()


@pytest.mark.no_sim
@pytest.mark.parametrize("backend", ["numpy", "warp", "auto"])
def test_in_tree_planner_preserves_batch_timing_and_failures(backend: str) -> None:
    from embodichain.lab.sim.motion.planners.toppra_planner import (
        ToppraPlanner,
        ToppraPlannerCfg,
        ToppraPlanOptions,
    )
    from embodichain.lab.sim.motion.planners.utils import PlanState

    planner = object.__new__(ToppraPlanner)
    planner.cfg = ToppraPlannerCfg(robot_uid="unused", backend=backend, max_workers=1)
    planner.device = torch.device("cpu")
    planner._pool = None
    states = [
        PlanState.from_qpos(torch.tensor([[0.0], [0.0], [2.0]])),
        PlanState.from_qpos(torch.tensor([[1.0], [float("nan")], [2.0]])),
    ]
    result = planner.plan(
        states,
        ToppraPlanOptions(
            constraints={"velocity": [[-0.5, 1.0]], "acceleration": [[-2.0, 1.0]]},
            sample_method=TrajectorySampleMethod.TIME,
            sample_interval=0.03,
        ),
    )
    assert result.success.tolist() == [True, False, True]
    assert planner._pool is None
    assert result.duration[0] > 0
    assert result.duration[1:].count_nonzero() == 0
    assert result.positions[2].eq(2.0).all()
    assert result.velocities[2].count_nonzero() == 0
    assert result.dt.max() <= 0.03
    reference = _toppra_solve_one_env(
        np.array([[0.0], [1.0]]),
        [[-0.5, 1.0]],
        [[-2.0, 1.0]],
        TrajectorySampleMethod.TIME,
        0.03,
    )
    np.testing.assert_allclose(result.positions[0], reference["positions"], atol=1e-6)
    np.testing.assert_allclose(result.dt[0], reference["dt"], atol=1e-6)


@pytest.mark.no_sim
@pytest.mark.parametrize("backend", ["numpy", "warp"])
def test_in_tree_planner_invalid_limits_return_failure(backend: str) -> None:
    from embodichain.lab.sim.motion.planners.toppra_planner import (
        ToppraPlanner,
        ToppraPlannerCfg,
        ToppraPlanOptions,
    )
    from embodichain.lab.sim.motion.planners.utils import PlanState

    planner = object.__new__(ToppraPlanner)
    planner.cfg = ToppraPlannerCfg(robot_uid="unused", backend=backend, max_workers=1)
    planner.device = torch.device("cpu")
    planner._pool = None
    result = planner.plan(
        [PlanState.from_qpos(torch.zeros(1, 1)), PlanState.from_qpos(torch.ones(1, 1))],
        ToppraPlanOptions(
            constraints={"velocity": -1.0, "acceleration": 2.0}, sample_interval=10
        ),
    )
    assert not result.success.any()
    assert result.dt.count_nonzero() == 0


@pytest.mark.no_sim
@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.float64])
def test_cuda_auto_backend_keeps_plan_on_device(
    monkeypatch: pytest.MonkeyPatch,
    dtype: torch.dtype,
) -> None:
    from embodichain.lab.sim.motion.planners.toppra_planner import (
        ToppraPlanner,
        ToppraPlannerCfg,
        ToppraPlanOptions,
    )
    from embodichain.lab.sim.motion.planners.utils import PlanState

    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    planner = object.__new__(ToppraPlanner)
    planner.cfg = ToppraPlannerCfg(robot_uid="unused")
    planner.device = torch.device("cuda")
    planner._pool = None
    points = torch.tensor(
        [
            [[0.0], [0.4]],
            [[1.0], [1.0]],
            [[0.0], [float("nan")]],
        ],
        device="cuda",
        dtype=dtype,
    )
    states = [PlanState.from_qpos(points[:, i]) for i in range(2)]
    with monkeypatch.context() as patch:

        def forbidden(*args, **kwargs):
            raise AssertionError("auto CUDA planning must not copy waypoints to CPU")

        patch.setattr(torch.Tensor, "cpu", forbidden)
        result = planner.plan(states, ToppraPlanOptions(sample_interval=128.9))
    assert result.success.tolist() == [True, True, False]
    assert result.positions.is_cuda and result.dt.is_cuda
    assert result.positions.shape == (3, 128, 1)
    assert result.positions.dtype == result.dt.dtype == torch.float32
    assert planner._pool is None
    assert result.duration[0] > 0
    assert result.duration[1:].count_nonzero() == 0
    torch.testing.assert_close(result.positions[0, -1], points[0, -1].float())


@pytest.mark.no_sim
@pytest.mark.parametrize("backend", ["numpy", "warp"])
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.float32, torch.float64, torch.int64]
)
@pytest.mark.parametrize("quantity", [20, 20.0, 20.9])
def test_legacy_planner_output_dtype_and_quantity_coercion(
    backend: str, dtype: torch.dtype, quantity: float | int
) -> None:
    from embodichain.lab.sim.motion.planners.toppra_planner import (
        ToppraPlanner,
        ToppraPlannerCfg,
        ToppraPlanOptions,
    )
    from embodichain.lab.sim.motion.planners.utils import PlanState

    planner = object.__new__(ToppraPlanner)
    planner.cfg = ToppraPlannerCfg(robot_uid="unused", backend=backend, max_workers=1)
    planner.device = torch.device("cpu")
    planner._pool = None
    result = planner.plan(
        [
            PlanState.from_qpos(torch.zeros(1, 1, dtype=dtype)),
            PlanState.from_qpos(torch.ones(1, 1, dtype=dtype)),
        ],
        ToppraPlanOptions(sample_interval=quantity),
    )
    assert result.success.tolist() == [True]
    assert result.positions.shape == (1, 20, 1)
    for values in (
        result.positions,
        result.velocities,
        result.accelerations,
        result.dt,
    ):
        assert values.dtype == torch.float32


@pytest.mark.no_sim
@pytest.mark.parametrize("backend", ["auto", "warp"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_planner_preserves_waypoint_gradients_and_duration_contract(
    backend: str,
    dtype: torch.dtype,
) -> None:
    from embodichain.lab.sim.motion.planners.toppra_planner import (
        ToppraPlanner,
        ToppraPlannerCfg,
        ToppraPlanOptions,
    )
    from embodichain.lab.sim.motion.planners.utils import PlanState

    planner = object.__new__(ToppraPlanner)
    planner.cfg = ToppraPlannerCfg(robot_uid="unused", backend=backend)
    planner.device = torch.device("cpu")
    planner._pool = None
    # Use a non-leaf input to verify gradients reach the caller's computation.
    parameter = torch.tensor([[0.365]], dtype=dtype, requires_grad=True)
    distance = 2 * parameter
    result = planner.plan(
        [
            PlanState.from_qpos(torch.zeros_like(distance)),
            PlanState.from_qpos(distance),
        ],
        ToppraPlanOptions(
            constraints={"velocity": 100.0, "acceleration": 2.0},
            sample_interval=17,
        ),
    )
    assert result.success.all()
    assert planner._pool is None
    assert result.positions.dtype == torch.float32
    assert result.positions.requires_grad and result.duration.requires_grad
    result.duration.sum().backward()
    expected = result.duration.detach().to(dtype) / (2 * parameter.detach().flatten())
    torch.testing.assert_close(parameter.grad.flatten(), expected)


@pytest.mark.no_sim
def test_numpy_planner_rejects_autograd_instead_of_detaching() -> None:
    from embodichain.lab.sim.motion.planners.toppra_planner import (
        ToppraPlanner,
        ToppraPlannerCfg,
        ToppraPlanOptions,
    )
    from embodichain.lab.sim.motion.planners.utils import PlanState

    planner = object.__new__(ToppraPlanner)
    planner.cfg = ToppraPlannerCfg(robot_uid="unused", backend="numpy")
    planner.device = torch.device("cpu")
    planner._pool = None
    states = [
        PlanState.from_qpos(torch.zeros(1, 1)),
        PlanState.from_qpos(torch.ones(1, 1, requires_grad=True)),
    ]
    options = ToppraPlanOptions(
        sample_method=TrajectorySampleMethod.TIME,
        sample_interval=0.1,
    )
    with pytest.raises(NotImplementedError, match="Warp backend"):
        planner.plan(states, options)


@pytest.mark.no_sim
@pytest.mark.parametrize("backend", ["auto", "warp"])
@pytest.mark.parametrize(
    "sampling", [TrajectorySampleMethod.TIME, TrajectorySampleMethod.QUANTITY]
)
def test_planner_autograd_routes_trainable_limits_without_trainable_waypoints(
    backend: str,
    sampling: TrajectorySampleMethod,
) -> None:
    from embodichain.lab.sim.motion.planners.toppra_planner import (
        ToppraPlanner,
        ToppraPlannerCfg,
        ToppraPlanOptions,
    )
    from embodichain.lab.sim.motion.planners.utils import PlanState

    planner = object.__new__(ToppraPlanner)
    planner.cfg = ToppraPlannerCfg(robot_uid="unused", backend=backend)
    planner.device = torch.device("cpu")
    planner._pool = None
    parameter = torch.tensor(0.6, dtype=torch.float64, requires_grad=True)
    acceleration = parameter.exp()  # Verify the caller's graph is preserved.
    result = planner.plan(
        [
            PlanState.from_qpos(torch.zeros(1, 1)),
            PlanState.from_qpos(torch.full((1, 1), 0.73)),
        ],
        ToppraPlanOptions(
            constraints={"velocity": 100.0, "acceleration": acceleration},
            sample_method=sampling,
            sample_interval=0.13 if sampling == TrajectorySampleMethod.TIME else 17,
        ),
    )
    assert result.success.all() and result.duration.requires_grad
    assert result.positions.dtype == torch.float32 and planner._pool is None
    result.duration.sum().backward()
    torch.testing.assert_close(
        parameter.grad, -result.duration.sum().detach().double() / 2
    )


@pytest.mark.no_sim
@pytest.mark.parametrize("backend", ["numpy", "warp", "auto"])
def test_no_grad_planning_accepts_trainable_inputs_as_constants(backend: str) -> None:
    from embodichain.lab.sim.motion.planners.toppra_planner import (
        ToppraPlanner,
        ToppraPlannerCfg,
        ToppraPlanOptions,
    )
    from embodichain.lab.sim.motion.planners.utils import PlanState

    planner = object.__new__(ToppraPlanner)
    planner.cfg = ToppraPlannerCfg(robot_uid="unused", backend=backend)
    planner.device = torch.device("cpu")
    planner._pool = None
    end = torch.ones(1, 1, requires_grad=True)
    limit = torch.tensor(1.0, requires_grad=True)
    with torch.no_grad():
        result = planner.plan(
            [PlanState.from_qpos(torch.zeros_like(end)), PlanState.from_qpos(end)],
            ToppraPlanOptions(
                constraints={"velocity": limit, "acceleration": 2 * limit},
                sample_interval=17,
            ),
        )
    assert result.success.all() and not result.duration.requires_grad


@pytest.mark.no_sim
def test_toppra_option_copies_preserve_tensor_graphs_and_isolate_containers() -> None:
    from embodichain.lab.sim.motion.planners.toppra_planner import ToppraPlanOptions

    parameter = torch.tensor(0.6, requires_grad=True)
    limit = parameter.exp()
    options = ToppraPlanOptions(constraints={"velocity": [1.0], "acceleration": limit})
    copied = options.copy()
    assert copied.constraints is not options.constraints
    assert copied.constraints["acceleration"] is limit
    copied.constraints["velocity"][0] = 2.0
    assert options.constraints["velocity"] == [1.0]
    assert ToppraPlanOptions().constraints == {"velocity": 0.2, "acceleration": 0.5}


@pytest.mark.no_sim
def test_motion_generator_keeps_toppra_gradients_through_option_copy_and_resampling() -> (
    None
):
    from embodichain.lab.sim.motion.motion_generator import (
        MotionGenerator,
        MotionGenOptions,
    )
    from embodichain.lab.sim.motion.planners.toppra_planner import (
        ToppraPlanner,
        ToppraPlannerCfg,
        ToppraPlanOptions,
    )
    from embodichain.lab.sim.motion.planners.utils import PlanState

    planner = object.__new__(ToppraPlanner)
    planner.cfg = ToppraPlannerCfg(robot_uid="unused")
    planner.device = torch.device("cpu")
    planner._pool = None
    generator = object.__new__(MotionGenerator)
    generator.planner = planner
    generator.device = planner.device
    distance = torch.tensor([[0.73]], dtype=torch.float64, requires_grad=True)
    parameter = torch.tensor(0.6, dtype=torch.float64, requires_grad=True)
    result = generator.generate(
        [
            PlanState.from_qpos(torch.zeros_like(distance)),
            PlanState.from_qpos(distance),
        ],
        MotionGenOptions(
            sample_count=29,
            plan_opts=ToppraPlanOptions(
                constraints={"velocity": 100.0, "acceleration": parameter.exp()},
                sample_interval=17,
            ),
        ),
    )
    assert result.success.all() and result.positions.shape == (1, 29, 1)
    duration = result.duration.sum()
    dq, da = torch.autograd.grad(duration, (distance, parameter))
    torch.testing.assert_close(dq, duration.detach().double() / (2 * distance.detach()))
    torch.testing.assert_close(da, -duration.detach().double() / 2)


@pytest.mark.no_sim
@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_cuda_time_planner_backward_preserves_input_gradients(
    dtype: torch.dtype,
) -> None:
    from embodichain.lab.sim.motion.planners.toppra_planner import (
        ToppraPlanner,
        ToppraPlannerCfg,
        ToppraPlanOptions,
    )
    from embodichain.lab.sim.motion.planners.utils import PlanState

    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    planner = object.__new__(ToppraPlanner)
    planner.cfg = ToppraPlannerCfg(robot_uid="unused")
    planner.device = torch.device("cuda")
    planner._pool = None
    distance = torch.tensor([[0.73]], device="cuda", dtype=dtype, requires_grad=True)
    acceleration = torch.tensor(1.7, device="cuda", requires_grad=True)
    options = ToppraPlanOptions(
        constraints={"velocity": 100.0, "acceleration": acceleration},
        sample_method=TrajectorySampleMethod.TIME,
        sample_interval=0.13,
    )
    result = planner.plan(
        [
            PlanState.from_qpos(torch.zeros_like(distance)),
            PlanState.from_qpos(distance),
        ],
        options,
    )
    duration = result.duration.sum()
    dq, da = torch.autograd.grad(duration, (distance, acceleration))
    assert result.success.all() and dq.is_cuda and da.is_cuda
    expected_dq = duration.detach().double() / (2 * distance.detach().double())
    torch.testing.assert_close(dq, expected_dq.to(dtype))
    torch.testing.assert_close(da, -duration.detach() / (2 * acceleration.detach()))
