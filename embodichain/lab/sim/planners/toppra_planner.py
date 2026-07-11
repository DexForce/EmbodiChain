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

import os

import torch
import numpy as np
from concurrent.futures.process import BrokenProcessPool

from embodichain.utils import logger, configclass
from embodichain.lab.sim.planners.utils import TrajectorySampleMethod
from embodichain.lab.sim.planners.base_planner import (
    validate_plan_options,
    BasePlanner,
    BasePlannerCfg,
    PlanOptions,
    _infer_batch_size,
)
from .utils import PlanState, PlanResult

try:
    import toppra as ta
    import toppra.constraint as constraint
except ImportError:
    logger.log_error(
        "toppra not installed. Install with `pip install toppra==0.6.3`", ImportError
    )

ta.setup_logging(level="WARN")


def _build_constraint_arrays(value, acc, dofs: int) -> tuple[np.ndarray, np.ndarray]:
    """Expand scalar limits to (dofs, 2) arrays; pass through arrays as-is."""
    if isinstance(value, (float, int)):
        vlims = np.array([[-value, value] for _ in range(dofs)])
    else:
        vlims = np.array(value)
    if isinstance(acc, (float, int)):
        alims = np.array([[-acc, acc] for _ in range(dofs)])
    else:
        alims = np.array(acc)
    return vlims, alims


def _toppra_solve_one_env(
    waypoints: np.ndarray,
    vel_constraint,
    acc_constraint,
    sample_method: "TrajectorySampleMethod",
    sample_interval: float | int,
) -> dict:
    """Solve a single-env TOPPRA trajectory. Pure numpy/scipy — picklable, no torch/robot.

    Args:
        waypoints: ``(N, DOF)`` numpy array of joint waypoints.
        vel_constraint / acc_constraint: scalar or per-DoF array limits.
        sample_method: TIME or QUANTITY.
        sample_interval: seconds (TIME) or sample count (QUANTITY).

    Returns:
        dict with ``positions`` ``(N_b, DOF)``, ``velocities``, ``accelerations``,
        ``dt`` ``(N_b,)``, ``success`` bool, ``n`` int, ``duration`` float.
    """
    dofs = waypoints.shape[1]
    vlims, alims = _build_constraint_arrays(vel_constraint, acc_constraint, dofs)

    if sample_method == TrajectorySampleMethod.TIME and sample_interval <= 0:
        return _empty_failure(dofs)
    if sample_method == TrajectorySampleMethod.QUANTITY and sample_interval < 2:
        return _empty_failure(dofs)

    # Remove consecutive duplicate waypoints. Long plateaus of identical points
    # (e.g. when start_qpos equals the first target and joint-space interpolation
    # is enabled) can make TOPPRA's controllable-set computation numerically
    # ill-conditioned and fail with "Instance is not controllable".
    dup_tol = 1e-6
    keep = [0]
    for i in range(1, len(waypoints)):
        if np.max(np.abs(waypoints[i] - waypoints[keep[-1]])) >= dup_tol:
            keep.append(i)
    if keep[-1] != len(waypoints) - 1:
        keep.append(len(waypoints) - 1)
    waypoints = waypoints[keep]

    # Trivial same-waypoint shortcut
    if len(waypoints) == 2 and np.sum(np.abs(waypoints[1] - waypoints[0])) < 1e-3:
        pos = np.stack([waypoints[0], waypoints[1]])
        return {
            "positions": pos,
            "velocities": np.zeros_like(pos),
            "accelerations": np.zeros_like(pos),
            "dt": np.array([0.0, 0.0], dtype=np.float32),
            "success": True,
            "n": 2,
            "duration": 0.0,
        }

    ss = np.linspace(0.0, 1.0, len(waypoints))
    try:
        path = ta.SplineInterpolator(ss, waypoints)
        pc_vel = constraint.JointVelocityConstraint(vlims)
        pc_acc = constraint.JointAccelerationConstraint(alims)
        instance = ta.algorithm.TOPPRA(
            [pc_vel, pc_acc],
            path,
            parametrizer="ParametrizeConstAccel",
            gridpt_min_nb_points=max(100, 10 * len(waypoints)),
        )
        jnt_traj = instance.compute_trajectory()
    except Exception:
        return _empty_failure(dofs)

    if jnt_traj is None:
        return _empty_failure(dofs)

    duration = float(jnt_traj.duration)
    if duration <= 0:
        return _empty_failure(dofs)

    if sample_method == TrajectorySampleMethod.TIME:
        n_points = max(2, int(np.ceil(duration / sample_interval)) + 1)
        ts = np.linspace(0.0, duration, n_points)
    else:
        ts = np.linspace(0.0, duration, num=int(sample_interval))

    positions = np.array([jnt_traj.eval(t) for t in ts])
    velocities = np.array([jnt_traj.evald(t) for t in ts])
    accelerations = np.array([jnt_traj.evaldd(t) for t in ts])
    dt = np.diff(ts, prepend=0.0).astype(np.float32)
    return {
        "positions": positions,
        "velocities": velocities,
        "accelerations": accelerations,
        "dt": dt,
        "success": True,
        "n": len(ts),
        "duration": duration,
    }


def _empty_failure(dofs: int) -> dict:
    z = np.zeros((2, dofs), dtype=np.float32)
    return {
        "positions": z,
        "velocities": np.zeros_like(z),
        "accelerations": np.zeros_like(z),
        "dt": np.array([0.0, 0.0], dtype=np.float32),
        "success": False,
        "n": 2,
        "duration": 0.0,
    }


def _set_parent_death_signal() -> None:
    r"""Best-effort: ask the kernel to SIGKILL this worker when its parent dies.

    This is the **only** cleanup mechanism that survives ``os._exit(0)`` in the
    parent process, which is what :meth:`SimulationManager.destroy` does by
    default (gated by ``EMBODICHAIN_SIM_EXIT_PROCESS``).  ``os._exit`` skips
    every Python-level finalizer — ``atexit`` handlers, ``__del__`` methods,
    and ``concurrent.futures``' internal ``_python_exit`` that would otherwise
    join/terminate daemon workers — so the worker processes would be orphaned
    and reparented to init, leaving residual python processes behind.

    A kernel parent-death signal fires regardless of *how* the parent exits
    (``os._exit``, a crash, or a normal return), so workers are reaped even
    when no Python cleanup runs.  No-op on non-Linux or if ``prctl`` is
    unavailable; in that case workers still rely on ``__del__`` and the
    daemon-process reaping that runs on a normal interpreter shutdown.
    """
    try:
        import ctypes
        import signal

        libc = ctypes.CDLL(None)
        PR_SET_PDEATHSIG = 1
        libc.prctl.argtypes = [
            ctypes.c_int,
            ctypes.c_ulong,
            ctypes.c_ulong,
            ctypes.c_ulong,
            ctypes.c_ulong,
        ]
        libc.prctl.restype = ctypes.c_int
        libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0)
    except Exception:
        pass


def _worker_init() -> None:
    r"""Initializer run in each worker when the pool starts.

    Two responsibilities:

    1. **Parent-death signal (Linux).**  Install ``prctl(PR_SET_PDEATHSIG,
       SIGKILL)`` so the kernel kills this worker the instant its parent
       dies.  This is what guarantees no residual worker processes when the
       parent exits via ``os._exit(0)`` (see :func:`_set_parent_death_signal`).
       Because of this, callers never need to explicitly shut the planner
       down — workers self-terminate with the parent.

    2. **atexit clearing (fork only).**  When ``fork`` is used, the parent
       has usually already initialized CUDA/GPU by the time the pool is
       created, and ``fork`` copies the parent's atexit registry into the
       child.  Without clearing it the worker would try to run the parent's
       cleanup routines on exit, which can deadlock or corrupt state.  With
       ``spawn`` this is not necessary, but the initializer still runs so
       that ``fork`` remains usable for advanced callers who create the pool
       before CUDA is initialized.
    """
    import atexit
    import multiprocessing as mp

    if mp.get_start_method() == "fork":
        atexit._clear()

    _set_parent_death_signal()


__all__ = ["ToppraPlanner", "ToppraPlannerCfg", "ToppraPlanOptions"]


@configclass
class ToppraPlannerCfg(BasePlannerCfg):

    planner_type: str = "toppra"
    max_workers: int | None = None
    """Worker process count for the batched fan-out. None => min(cpu_count()//2, B)."""
    mp_context: str | None = None
    """Multiprocessing start method for the batched fan-out.

    ``None`` (default) auto-selects based on the simulation device:
    ``'fork'`` on CPU and ``'spawn'`` on GPU. ``'fork'`` is faster — workers
    inherit the parent's already-loaded modules, so pool startup is
    near-instant — and is safe here because the TOPPRA worker
    (:func:`_toppra_solve_one_env`) is pure numpy/scipy and never touches the
    parent's Vulkan/Warp/CUDA context or render threads; ``_worker_init``
    clears the inherited atexit registry and installs ``prctl(PR_SET_PDEATHSIG)``
    so workers are reaped when the parent dies (incl. the ``os._exit`` path).
    ``'spawn'`` is the safer choice when the parent has initialized CUDA
    physics (``sim_device='cuda'``) — fork-after-CUDA-init is the officially
    unsupported case — or if fork deadlocks are observed, at the cost of
    re-importing modules per worker.
    """


@configclass
class ToppraPlanOptions(PlanOptions):

    constraints: dict = {
        "velocity": 0.2,
        "acceleration": 0.5,
    }
    """Constraints for the planner, including velocity and acceleration limits.

    Should be a dictionary with keys 'velocity' and 'acceleration', each containing a value or a list of limits for each joint.
    """

    sample_method: TrajectorySampleMethod = TrajectorySampleMethod.QUANTITY
    """Method for sampling the trajectory.

    Options are 'time' for uniform time intervals or 'quantity' for a fixed number of samples.
    """

    sample_interval: float | int = 0.01
    """Interval for sampling the trajectory.

    If sample_method is 'time', this is the time interval in seconds.
    If sample_method is 'quantity', this is the total number of samples.
    """


class ToppraPlanner(BasePlanner):
    def __init__(self, cfg: ToppraPlannerCfg):
        r"""Initialize the TOPPRA trajectory planner.

        References:
            - TOPPRA: Time-Optimal Path Parameterization for Robotic Systems (https://github.com/hungpham2511/toppra)

        Args:
            cfg: Configuration object containing ToppraPlanner settings
        """
        super().__init__(cfg)

        self._pool = None
        # Resolve the multiprocessing start method once, now that self.device
        # (from BasePlanner) is known. None => auto: fork on CPU, spawn on GPU.
        self._mp_context = self._resolve_mp_context(cfg.mp_context, self.device)
        # No atexit / __del__-based shutdown is registered here: workers install
        # prctl(PR_SET_PDEATHSIG) in _worker_init, so the kernel reaps them the
        # moment the parent process dies — including the os._exit(0) path taken
        # by SimulationManager.destroy(), which skips every Python finalizer.
        # __del__ below only handles in-process GC of an abandoned planner.

    def default_plan_options(self) -> ToppraPlanOptions:
        """Return backend-default planning options."""
        return ToppraPlanOptions()

    @staticmethod
    def _resolve_mp_context(mp_context: str | None, device: torch.device) -> str:
        """Return the multiprocessing start method to use for the worker pool.

        An explicit ``mp_context`` is honored as-is. ``None`` auto-selects:
        ``'fork'`` on CPU (fast — workers inherit loaded modules; safe because
        the worker is pure numpy/scipy), ``'spawn'`` everywhere else (safer
        under an initialized CUDA context).

        Args:
            mp_context: The cfg value; ``None`` means auto-select.
            device: The physics device the planner's robot runs on.

        Returns:
            One of ``'fork'`` / ``'spawn'``.
        """
        if mp_context is not None:
            return mp_context
        return "fork" if device.type == "cpu" else "spawn"

    def _get_pool(self, batch_size: int):
        if self._pool is not None:
            return self._pool
        import multiprocessing as mp

        max_workers = self.cfg.max_workers
        if max_workers is None:
            max_workers = max(1, min((os.cpu_count() or 2) // 2, batch_size))
        ctx = mp.get_context(self._mp_context)
        from concurrent.futures import ProcessPoolExecutor

        self._pool = ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=ctx,
            initializer=_worker_init,
        )
        return self._pool

    def _shutdown_pool(self) -> None:
        r"""Shut down the TOPPRA worker process pool (internal).

        We do **not** use ``shutdown(wait=True)``: with ``fork`` workers can
        inherit the parent's CUDA/GPU context, causing them to deadlock inside
        the driver at exit.  ``wait=True`` would then hang the main process,
        and if the user kills it the workers are left behind as residual
        python processes.

        Instead we cancel pending work and then forcibly terminate/join/kill
        every worker process so that this method returns only after all
        workers are actually gone.

        This is **not** part of the public API.  It exists for two internal
        callers: the ``BrokenProcessPool`` recovery path in :meth:`plan`, and
        ``__del__`` (so an abandoned planner in a long-running process does
        not leak workers).  Normal process exit does not rely on it — workers
        install ``prctl(PR_SET_PDEATHSIG)`` in :func:`_worker_init` and are
        reaped by the kernel when the parent dies, even under ``os._exit``.
        """
        if self._pool is None:
            return

        # Capture the worker process objects before shutdown clears them.
        worker_processes = list(getattr(self._pool, "_processes", {}).values())

        # Stop accepting new work and cancel any futures that have not started.
        self._pool.shutdown(wait=False, cancel_futures=True)

        # Forcibly reap every worker.  This is required for fork-based pools
        # when the parent has initialized CUDA: the children inherit the
        # context and may not exit cleanly on their own.
        for proc in worker_processes:
            if not proc.is_alive():
                continue
            proc.terminate()
            proc.join(timeout=5.0)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=1.0)

        self._pool = None

    def __del__(self):
        # Only matters for in-process GC of an abandoned planner (and as a
        # non-Linux fallback).  Process-exit cleanup is handled by the kernel
        # via PR_SET_PDEATHSIG installed in each worker, which survives the
        # os._exit(0) path that SimulationManager.destroy() takes.
        try:
            self._shutdown_pool()
        except Exception:
            pass

    @validate_plan_options(options_cls=ToppraPlanOptions)
    def plan(
        self,
        target_states: list[PlanState],
        options: ToppraPlanOptions = ToppraPlanOptions(),
    ) -> PlanResult:
        r"""Execute trajectory planning.

        Args:
            target_states: list of :class:`PlanState` waypoints. Tensor fields
                carry a leading batch dim ``B``: ``qpos`` is ``(B, DOF)``.
            options: :class:`ToppraPlanOptions` with constraints and sampling.

        Returns:
            PlanResult containing the planned trajectory details. All tensor
            fields are env-batched with leading dim ``B``: ``success`` ``(B,)``,
            ``positions``/``velocities``/``accelerations`` ``(B, N, DOF)``,
            ``dt`` ``(B, N)``, ``duration`` ``(B,)``.
        """
        for i, t in enumerate(target_states):
            if t.qpos is None:
                logger.log_error(f"Target state at index {i} missing qpos", ValueError)

        b = _infer_batch_size(target_states) or 1
        dofs = target_states[0].qpos.shape[-1]

        # Build (B, N, DOF) numpy waypoints
        waypoints = np.stack(
            [s.qpos.detach().cpu().numpy() for s in target_states], axis=1
        )  # (B, N, DOF)

        vc = options.constraints["velocity"]
        ac = options.constraints["acceleration"]
        args_per_env = [
            (waypoints[i], vc, ac, options.sample_method, options.sample_interval)
            for i in range(b)
        ]

        # Single-env planning never needs a process pool.
        if b == 1:
            results = [_toppra_solve_one_env(*a) for a in args_per_env]
        else:
            # Inline fallback for max_workers==1 or a single-core machine.
            max_workers = self.cfg.max_workers
            use_inline = (max_workers == 1) or (
                max_workers is None and ((os.cpu_count() or 2) // 2) <= 1
            )
            if use_inline:
                results = [_toppra_solve_one_env(*a) for a in args_per_env]
            else:
                pool = self._get_pool(b)
                results = [None] * b
                try:
                    futures = [
                        pool.submit(_toppra_solve_one_env, *a) for a in args_per_env
                    ]
                    broken = False
                    for i, fut in enumerate(futures):
                        try:
                            results[i] = fut.result()
                        except BrokenProcessPool:
                            logger.log_warning(
                                "TOPPRA process pool broke; returning failure."
                            )
                            self._shutdown_pool()
                            broken = True
                            break
                        except Exception:
                            results[i] = _empty_failure(dofs)
                    if broken:
                        for i in range(b):
                            if results[i] is None:
                                results[i] = _empty_failure(dofs)
                except BrokenProcessPool:
                    # pool was already broken at submit time
                    logger.log_warning("TOPPRA process pool broke; returning failure.")
                    self._shutdown_pool()
                    for i in range(b):
                        if results[i] is None:
                            results[i] = _empty_failure(dofs)

        return self._assemble_batched_result(results, dofs)

    def _assemble_batched_result(self, results: list[dict], dofs: int) -> PlanResult:
        """Stack per-env TOPPRA results into a batched :class:`PlanResult`.

        Each entry of ``results`` is the dict returned by
        :func:`_toppra_solve_one_env`. Env trajectories may have different
        lengths (``n``); this method pads shorter trajectories out to the
        longest by repeating their final waypoint (held pose) with zero
        velocity and acceleration, so every output tensor shares the same
        ``(B, N, DOF)`` / ``(B, N)`` shape.

        Args:
            results: list of per-env result dicts (length ``B``).
            dofs: per-env degrees of freedom.

        Returns:
            PlanResult with env-batched tensors (``success`` ``(B,)``,
            ``positions``/``velocities``/``accelerations`` ``(B, N, DOF)``,
            ``dt`` ``(B, N)``, ``duration`` ``(B,)``).
        """
        b = len(results)
        max_n = max(r["n"] for r in results)
        positions = np.zeros((b, max_n, dofs), dtype=np.float32)
        velocities = np.zeros((b, max_n, dofs), dtype=np.float32)
        accelerations = np.zeros((b, max_n, dofs), dtype=np.float32)
        dt = np.zeros((b, max_n), dtype=np.float32)
        duration = np.zeros((b,), dtype=np.float32)
        success = np.zeros((b,), dtype=bool)
        for i, r in enumerate(results):
            n = r["n"]
            positions[i, :n] = r["positions"]
            velocities[i, :n] = r["velocities"]
            accelerations[i, :n] = r["accelerations"]
            dt[i, :n] = r["dt"]
            duration[i] = r["duration"]
            success[i] = r["success"]
            # tail-pad: repeat final waypoint for held-pose rows
            if n < max_n:
                positions[i, n:] = r["positions"][-1]
                velocities[i, n:] = 0.0
                accelerations[i, n:] = 0.0
        return PlanResult(
            success=torch.as_tensor(success, device=self.device),
            positions=torch.as_tensor(positions, device=self.device),
            velocities=torch.as_tensor(velocities, device=self.device),
            accelerations=torch.as_tensor(accelerations, device=self.device),
            dt=torch.as_tensor(dt, device=self.device),
            duration=torch.as_tensor(duration, device=self.device),
        )
