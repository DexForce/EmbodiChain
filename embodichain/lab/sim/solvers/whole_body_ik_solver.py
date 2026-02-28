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
"""WholeBodyIKSolver — whole‑body IK solver.

This solver inherits from :class:`~embodichain.lab.sim.solvers.PinocchioSolver` at the API level but completely re‑implements ``__init__``, ``get_ik`` and ``get_fk`` to handle redundant whole‑body IK.

On top of the standard configuration interface used by other solvers (``end_link_name`` + ``tcp`` + ``root_link_name``), it adds:

- a **reduced whole‑body model** built with ``pin.buildReducedRobot`` that keeps all active joints (single arm + torso);
- **null‑space secondary objectives** (regularization, smoothness, leg / torso stability) to prevent redundant DOFs from drifting;
- an optional **CasADi / IPOPT backend** for global nonlinear optimisation.

Two backends are available via ``WholeBodyIKSolverCfg.backend``:

- ``"casadi"`` *(default)* — nonlinear programme solved by IPOPT via CasADi's symbolic FK (``pinocchio.casadi``); more robust for difficult configurations.
- ``"dls"`` — iterative Damped Least‑Squares with null‑space projection; fast and suitable for real‑time use.

The configuration interface is fully aligned with :class:`PinocchioSolver`::

    WholeBodyIKSolverCfg(
        end_link_name="left_ee",
        root_link_name="base_link",
        tcp=left_arm_tcp,        # 4x4 TCP matrix
        joint_names=None,        # auto‑filled from control_parts by Robot.init_solver
        backend="casadi",        # or "dls"
        ...
    )

Typical usage via :meth:`DexforceW1Cfg._build_default_solver_cfg`::

    robot.compute_ik(pose, name="left_arm_body")    # left arm whole‑body IK
    robot.compute_ik(pose, name="right_arm_body")   # right arm whole‑body IK

DLS per‑iteration algorithm:

1. ``pin.framesForwardKinematics(q)`` → current end‑effector pose
2. apply TCP inverse to map target into the end‑link frame: ``target_ee = target @ inv(tcp)``
3. ``pin.log6(oMf.actInv(target_ee))`` → 6D pose error in LOCAL frame
4. ``pin.computeFrameJacobian(q, fid, LOCAL)`` → 6 x dof Jacobian
5. DLS pseudo‑inverse ``J^T (JJ^T + λI)^{-1}`` → primary task increment
6. null‑space term ``(I - J^† J) * (−∇E_sec)`` → secondary objectives increment
7. ``pin.integrate`` + clipping joint limits

CasADi backend NLP formulation (uses ``casadi.Opti`` + IPOPT)::

    min_{q}  w_pos   * ||p(q) - p*||^2
           + w_rot   * ||log3(R(q) @ R*.T)||^2   ← SO(3) log‑map, same as DLS
           + w_reg   * sum(reg_weights[i] * q[i]^2)
           + w_smooth * ||q - q_seed||^2
           + sum_k  leg_cost_k(q)
    s.t.   lbq <= q <= ubq
"""

import os
from dataclasses import field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from embodichain.utils import configclass, logger
from embodichain.lab.sim.solvers.base_solver import BaseSolver
from embodichain.lab.sim.solvers.pinocchio_solver import (
    PinocchioSolver,
    PinocchioSolverCfg,
)
from embodichain.lab.sim.utility.import_utils import lazy_import_pinocchio


# ---------------------------------------------------------------------------
# Leg / torso stability secondary objectives
# ---------------------------------------------------------------------------


@configclass
class LegCostCfg:
    """Configuration of a single leg / torso stability secondary cost.

    The cost has the form ``weight * (sum_i coefficients[i] * q[joint_names[i]])^2``.

    Its gradient is used inside the null‑space projection as a secondary objective to guide joint angles.

    Example::

        # Constrain WAIST and KNEE to move with equal magnitude but opposite sign
        LegCostCfg(joint_names=["WAIST", "KNEE"], coefficients=[1.0, -1.0], weight=10.0)
    """

    joint_names: List[str] = field(default_factory=list)
    coefficients: List[float] = field(default_factory=list)
    weight: float = 10.0


# ---------------------------------------------------------------------------
# Solver configuration
# ---------------------------------------------------------------------------


@configclass
class WholeBodyIKSolverCfg(PinocchioSolverCfg):
    """Whole‑body IK solver configuration (inherits from :class:`PinocchioSolverCfg`).

    The configuration interface is fully aligned with :class:`PinocchioSolverCfg`:
    end‑effectors are defined via ``end_link_name`` (an existing URDF link),
    ``tcp`` (a 4x4 tool offset matrix), and ``root_link_name``, with no extra
    end‑effector configuration type.

    The semantics of ``joint_names`` differ slightly from :class:`PinocchioSolverCfg`:
    here it represents the set of joints that should remain **unlocked** in
    the reduced model, rather than a single serial chain. It is recommended to
    leave it as ``None`` and let :meth:`Robot.init_solver` populate it from
    the corresponding ``control_parts`` entry.

    Attributes:
        urdf_dir:              Directory for URDF meshes; if None, falls back to mesh_path or the URDF directory.
        backend:               IK backend: ``"casadi"`` (default, nonlinear optimisation via CasADi / IPOPT) or ``"dls"`` (iterative Damped Least‑Squares). The CasADi backend requires ``casadi`` and ``pinocchio.casadi`` to be installed.
        w_pos:                 Weight for end‑effector position error.
        w_rot:                 Weight for end‑effector orientation error. Both backends use a log‑map rotation error: ``"dls"`` uses the rotation part of ``pin.log6``; ``"casadi"`` uses ``cpin.log3(R(q) @ R_target.T)``. The two are geometrically consistent.
        max_iterations:        Maximum iterations per IK call (default 200). For the DLS backend this is the iteration cap; for the CasADi backend it is passed to IPOPT as ``ipopt.max_iter``.
        dt:                    Integration time step for the DLS backend (default 0.5, ignored by the CasADi backend).
        damp:                  DLS damping factor (default 1e‑3, ignored by the CasADi backend).
        pos_eps:               Convergence threshold on translation (meters).
        rot_eps:               Convergence threshold on rotation (radians).
        w_regularization:      Weight for the **joint regularization** term in the secondary objective (both backends).
            The cost term is ``w_regularization * Σᵢ (reg_weight[i] * q[i]²)``, where ``reg_weight[i]`` is 1 by default
            and can be increased per joint via ``joint_reg_extra``. This penalizes large joint angles and pushes the
            solution toward the zero configuration, which helps avoid singularities and extreme poses. Increase to
            favor more “neutral” postures; decrease (e.g. 0.01 or smaller) when the solver should prioritize reaching
            the target over posture comfort.
        w_smooth:              Weight for the **smoothness** term in the secondary objective (both backends).
            The cost term is ``w_smooth * ‖q − q_seed‖²``. This penalizes deviation from the seed configuration
            ``q_seed`` (typically the previous solution or current robot state). Larger values make the solution
            stick closer to the seed, giving smoother temporal behavior in sequences (e.g. teleoperation or keyframe
            interpolation) but may prevent the end‑effector from reaching the target if the target is far. Smaller
            values allow larger jumps toward the target per solve. Tune together with ``w_pos``: high ``w_smooth``
            + high ``w_pos`` can still reach the target over multiple steps with a good seed each time.
        joint_reg_extra:       Extra regularization weights for specific joints, as {"joint_name": extra_weight}.
        leg_costs_mode2:       List of :class:`LegCostCfg` to use when ``leg_mode == 2``.
        leg_costs_mode3:       List of :class:`LegCostCfg` to use when ``leg_mode == 3``.
        leg_mode:              Leg mode selector: 2 = standing, 3 = walking.
        casadi_ipopt_options:  Additional IPOPT option dict passed to ``casadi.nlpsol`` when using the CasADi backend (e.g. ``{"ipopt.tol": 1e-8}``). Overrides the solver's default IPOPT settings.
    """

    class_type: str = "WholeBodyIKSolver"

    # Override PinocchioSolverCfg defaults to better suit whole‑body IK
    max_iterations: int = 500
    dt: float = 0.5
    damp: float = 1e-3
    pos_eps: float = 1e-3
    rot_eps: float = 1e-3

    # Whole‑body specific fields
    urdf_dir: Optional[str] = None

    # Backend selection: "casadi" (default) or "dls"
    backend: str = "casadi"

    w_pos: float = 100.0
    w_rot: float = 10.0

    # Secondary objective: E_reg = w_regularization * Σ reg_weight[i]*q[i]²  → prefer joints near zero
    w_regularization: float = 0.01
    # Secondary objective: E_smooth = w_smooth * ‖q − q_seed‖²  → prefer staying close to seed config
    w_smooth: float = 0.05

    joint_reg_extra: Optional[Dict[str, float]] = None

    leg_costs_mode2: Optional[List[LegCostCfg]] = None
    leg_costs_mode3: Optional[List[LegCostCfg]] = None
    leg_mode: int = 2

    # CasADi backend options
    casadi_ipopt_options: Optional[Dict[str, Any]] = None

    def init_solver(self, **kwargs) -> "WholeBodyIKSolver":
        solver = WholeBodyIKSolver(cfg=self, **kwargs)
        solver.set_tcp(self._get_tcp_as_numpy())
        return solver


# ---------------------------------------------------------------------------
# Solver implementation
# ---------------------------------------------------------------------------


class WholeBodyIKSolver(PinocchioSolver):
    """Whole‑body IK solver (DLS iterative IK + null‑space secondary objectives).

    This class inherits from :class:`PinocchioSolver` but completely overrides
    ``__init__``, ``get_ik`` and ``get_fk``. The configuration interface remains
    compatible (via ``end_link_name``, ``root_link_name``, ``tcp``, and
    ``joint_names``), while the internals are tailored for redundant whole‑body IK.

    Per‑iteration algorithm:

    1. Forward kinematics: ``pin.framesForwardKinematics(q)`` to get the current end‑effector pose.
    2. TCP inverse: ``target_ee = target @ inv(tcp)`` to express the target in the end‑link frame.
    3. Pose error: ``pin.log6(oMf.actInv(target_ee))`` for a 6D LOCAL‑frame error.
    4. Jacobian: ``pin.computeFrameJacobian(q, frame_id, LOCAL)``.
    5. Primary task: DLS pseudo‑inverse ``J^T (JJ^T + λI)^{-1} e`` → ``dq_primary``.
    6. Secondary task: regularization + smoothness + leg stability, projected through ``(I - J^† J)`` → ``dq_secondary``.
    7. Integration and limits: ``q ← pin.integrate(q, (dq_primary + dq_secondary) * dt)`` and clipping to joint limits.
    """

    def __init__(self, cfg: WholeBodyIKSolverCfg, **kwargs):
        # Skip PinocchioSolver.__init__ (which assumes a serial chain).
        # Call BaseSolver.__init__ directly to set cfg, device, joint_names, etc.
        kwargs["pk_serial_chain"] = "_skip_"
        BaseSolver.__init__(self, cfg=cfg, **kwargs)
        self.pk_serial_chain = None

        # Keep attribute naming consistent with PinocchioSolver
        self.pin = lazy_import_pinocchio()
        pin = self.pin

        urdf_dir = cfg.urdf_dir or cfg.mesh_path or os.path.dirname(cfg.urdf_path)

        # 1. Load full robot and lock inactive joints
        full_robot = pin.RobotWrapper.BuildFromURDF(cfg.urdf_path, urdf_dir)

        active_set = set(cfg.joint_names or [])
        joints_to_lock = [
            name
            for name in full_robot.model.names
            if name not in active_set and name != "universe"
        ]
        self._reduced = full_robot.buildReducedRobot(
            list_of_joints_to_lock=joints_to_lock,
            reference_configuration=np.zeros(full_robot.model.nq),
        )

        self.dof = self._reduced.model.nq

        # Pinocchio uses its own joint order according to the URDF tree, which may
        # differ from cfg.joint_names. Compute permutation arrays so that the
        # solver’s external interface strictly follows cfg.joint_names.
        pin_order = [
            self._reduced.model.names[i] for i in range(1, self._reduced.model.njoints)
        ]
        cfg_order = list(cfg.joint_names or pin_order)

        # cfg_order[i] index in pinocchio order → _cfg_to_pin[i]
        self._cfg_to_pin: np.ndarray = np.array(
            [pin_order.index(j) for j in cfg_order], dtype=int
        )
        # pin_order[i] index in cfg_order → _pin_to_cfg[i]
        self._pin_to_cfg: np.ndarray = np.array(
            [cfg_order.index(j) for j in pin_order], dtype=int
        )

        # Expose joint_names and limits in cfg order
        self.joint_names = cfg_order

        # 2. Look up end‑effector frame ID using end_link_name (same as other solvers)
        if cfg.end_link_name is None:
            raise ValueError(
                "[WholeBodyIKSolver] end_link_name must be specified in cfg."
            )
        self._ee_frame_id: int = self._reduced.model.getFrameId(cfg.end_link_name)
        if self._ee_frame_id >= self._reduced.model.nframes:
            raise ValueError(
                f"[WholeBodyIKSolver] end_link_name='{cfg.end_link_name}' "
                "not found in reduced model. Check URDF or joint_names."
            )

        # 3. root_base_xpos: transform from Pinocchio universe to root_link_name.
        #    This mirrors the logic in PinocchioSolver: get_ik receives targets
        #    in the root_link_name frame and we need to map them into the universe
        #    frame used by Pinocchio FK. When root_link_name is the robot root
        #    (e.g. "base_link"), this becomes the identity.
        self._root_base_xpos = np.eye(4)
        if cfg.root_link_name is not None:
            fid_root = self._reduced.model.getFrameId(cfg.root_link_name)
            if fid_root < self._reduced.model.nframes:
                placement = self._reduced.model.frames[fid_root].placement
                self._root_base_xpos[:3, :3] = placement.rotation
                self._root_base_xpos[:3, 3] = placement.translation
            else:
                logger.log_warning(
                    f"[WholeBodyIKSolver] root_link_name='{cfg.root_link_name}' "
                    "not found in reduced model; using identity for root_base_xpos."
                )

        # 4. Map joint names to pinocchio internal q indices
        self._joint_name_to_q_idx: Dict[str, int] = {}
        for i in range(1, self._reduced.model.njoints):
            self._joint_name_to_q_idx[self._reduced.model.names[i]] = i - 1

        # 5. Regularization weight vector
        self._reg_weights = np.ones(self.dof)
        if cfg.joint_reg_extra:
            for jname, w in cfg.joint_reg_extra.items():
                idx = self._joint_name_to_q_idx.get(jname)
                if idx is not None:
                    self._reg_weights[idx] = 1.0 + float(w)
                else:
                    logger.log_warning(
                        f"[WholeBodyIKSolver] joint_reg_extra: joint '{jname}' "
                        "not found in reduced model, skipped."
                    )

        # 6. Pre‑compute leg / torso secondary cost terms
        leg_cfgs = (
            cfg.leg_costs_mode3 if int(cfg.leg_mode) == 3 else cfg.leg_costs_mode2
        ) or []
        self._leg_cost_terms: List[Tuple[np.ndarray, np.ndarray, float]] = []
        for lc in leg_cfgs:
            if not lc.joint_names or not lc.coefficients:
                continue
            if len(lc.joint_names) != len(lc.coefficients):
                logger.log_warning(
                    "[WholeBodyIKSolver] LegCostCfg joint_names / coefficients "
                    "length mismatch, skipped."
                )
                continue
            indices = []
            coeffs = []
            for jname, coeff in zip(lc.joint_names, lc.coefficients):
                idx = self._joint_name_to_q_idx.get(jname)
                if idx is None:
                    logger.log_warning(
                        f"[WholeBodyIKSolver] LegCostCfg: joint '{jname}' "
                        "not found, skipped."
                    )
                    continue
                indices.append(idx)
                coeffs.append(float(coeff))
            if indices:
                self._leg_cost_terms.append(
                    (np.array(indices, dtype=int), np.array(coeffs), float(lc.weight))
                )

        # Expose joint limits in cfg order (same attribute names as PinocchioSolver)
        self.lower_position_limits = self._reduced.model.lowerPositionLimit[
            self._pin_to_cfg
        ]
        self.upper_position_limits = self._reduced.model.upperPositionLimit[
            self._pin_to_cfg
        ]

        self._last_q = np.zeros(self.dof)

        #: Diagnostics from the last get_ik call, useful to assess solution quality.
        #: Fields: success(bool), iterations(int), pos_err(float,m), rot_err(float,rad)
        self.last_solve_info: dict = {
            "success": False,
            "iterations": 0,
            "pos_err": float("inf"),
            "rot_err": float("inf"),
        }

        # Build CasADi NLP at init time if requested, so the first solve is fast.
        if cfg.backend == "casadi":
            self._build_casadi_solver()

    # -----------------------------------------------------------------------
    # Internal: secondary objective gradient
    # -----------------------------------------------------------------------

    def _secondary_gradient(self, q: np.ndarray, q_seed: np.ndarray) -> np.ndarray:
        """Compute the joint‑space gradient of secondary objectives (regularization + smoothness + leg/torso stability)."""
        cfg: WholeBodyIKSolverCfg = self.cfg
        g = np.zeros(self.dof)
        g += 2.0 * cfg.w_regularization * self._reg_weights * q
        g += 2.0 * cfg.w_smooth * (q - q_seed)
        for indices, coeffs, weight in self._leg_cost_terms:
            val = float(np.dot(coeffs, q[indices]))
            g[indices] += 2.0 * weight * val * coeffs
        return g

    # -----------------------------------------------------------------------
    # Internal: DLS backend
    # -----------------------------------------------------------------------

    def _get_ik_dls(
        self,
        target_se3,
        target_t: np.ndarray,
        q_seed: np.ndarray,
    ) -> Tuple[np.ndarray, bool, float, float, int]:
        """Run DLS IK iterations.

        Args:
            target_se3: Pinocchio SE3 target (in universe frame, TCP removed).
            target_t:   Target translation (``target_se3.translation``), pre‑extracted for speed.
            q_seed:     Seed in Pinocchio joint order.

        Returns:
            ``(q_sol, success, pos_err, rot_err, n_iter)`` all in Pinocchio joint order.
        """
        pin = self.pin
        cfg: WholeBodyIKSolverCfg = self.cfg
        fid = self._ee_frame_id
        w_diag = np.array([cfg.w_pos] * 3 + [cfg.w_rot] * 3)

        q = q_seed.copy().astype(float)
        success = False
        _iter = 0
        _pos_err = float("inf")
        _rot_err = float("inf")

        for _iter in range(cfg.max_iterations):
            pin.framesForwardKinematics(self._reduced.model, self._reduced.data, q)
            oMf = self._reduced.data.oMf[fid]

            err6 = pin.log6(oMf.actInv(target_se3)).vector
            _pos_err = float(np.linalg.norm(oMf.translation - target_t))
            _rot_err = float(np.linalg.norm(err6[3:]))

            if _pos_err < cfg.pos_eps and _rot_err < cfg.rot_eps:
                success = True
                break

            err = err6 * w_diag
            J_full = pin.computeFrameJacobian(
                self._reduced.model, self._reduced.data, q, fid, pin.LOCAL
            )
            J = J_full * w_diag[:, None]

            JJt = J @ J.T
            JJt[np.diag_indices_from(JJt)] += cfg.damp
            J_pinv = J.T @ np.linalg.solve(JJt, np.eye(6))

            dq_primary = J_pinv @ err
            N = np.eye(self.dof) - J_pinv @ J
            g_sec = self._secondary_gradient(q, q_seed)
            dq_secondary = -N @ g_sec

            q = pin.integrate(
                self._reduced.model, q, (dq_primary + dq_secondary) * cfg.dt
            )
            q = np.clip(
                q,
                self._reduced.model.lowerPositionLimit,
                self._reduced.model.upperPositionLimit,
            )

        return q, success, _pos_err, _rot_err, _iter + 1

    # -----------------------------------------------------------------------
    # Internal: CasADi backend
    # -----------------------------------------------------------------------

    def _build_casadi_solver(self) -> None:
        """Build the CasADi / IPOPT optimisation problem for IK using ``casadi.Opti``.

        The NLP formulation matches the style of the reference ``dex_w1_ArmIK``
        implementation (single end‑effector variant)::

            min_{q}  w_pos  * ||p(q) - p*||^2
                   + w_rot  * ||log3(R(q) @ R*.T)||^2    ← SO(3) log‑map error
                   + w_reg  * sum(reg_w[i] * q[i]^2)
                   + w_smooth * ||q - q_seed||^2
                   + sum_k leg_cost_k(q)
            s.t.   lbq <= q <= ubq

        Key design points:

        - Uses ``casadi.Opti`` (high‑level stack) so that solver state is
          preserved between calls — only ``set_value`` / ``set_initial`` are
          called per solve, which avoids redundant JIT compilation overhead.
        - Rotation error via ``cpin.log3(R_ee @ R_target.T)`` is geometrically
          consistent with the DLS backend's ``pin.log6`` error.
        - The target 4×4 pose matrix is passed as a single ``opti.parameter(4,4)``,
          mirroring the reference implementation.

        Requires ``casadi`` and ``pinocchio.casadi`` (available with Pinocchio ≥ 2.9).
        """
        try:
            import casadi as ca
            import pinocchio.casadi as cpin
        except ImportError as exc:
            raise ImportError(
                "[WholeBodyIKSolver] The 'casadi' backend requires both the 'casadi' package "
                "and pinocchio's CasADi bindings (pinocchio.casadi). "
                "Install casadi via: pip install casadi\n"
                f"Original error: {exc}"
            ) from exc

        cfg: WholeBodyIKSolverCfg = self.cfg
        nq = self.dof

        # ── Phase 1: build cost Function using SX ─────────────────────────────
        # pinocchio.casadi requires SX (symbolic scalar) variables.
        # opti.variable() returns MX, which is incompatible with cpin FK calls.
        # Solution (mirrors dex_w1_ArmIK): build a ca.Function from SX, then
        # call that Function on the MX optimization variables in Phase 2.

        cmodel = cpin.Model(self._reduced.model)
        cdata = cmodel.createData()

        cq_sx = ca.SX.sym("q", nq)  # joint config  (SX)
        cTf_sx = ca.SX.sym("tf", 4, 4)  # target 4×4   (SX)
        cqs_sx = ca.SX.sym("qs", nq)  # seed config   (SX)

        cpin.framesForwardKinematics(cmodel, cdata, cq_sx)
        oMf_rot = cdata.oMf[self._ee_frame_id].rotation  # (3, 3) SX
        oMf_tr = cdata.oMf[self._ee_frame_id].translation  # (3,)   SX

        pos_err = oMf_tr - cTf_sx[:3, 3]
        rot_err = cpin.log3(oMf_rot @ cTf_sx[:3, :3].T)  # SO(3) log‑map

        rw = ca.DM(self._reg_weights)
        leg_total_sx = ca.SX(0)
        for indices, coeffs, weight in self._leg_cost_terms:
            leg_val = ca.SX(0)
            for j in range(len(indices)):
                leg_val = leg_val + float(coeffs[j]) * cq_sx[int(indices[j])]
            leg_total_sx = leg_total_sx + weight * leg_val * leg_val

        cost_sx = (
            cfg.w_pos * ca.dot(pos_err, pos_err)
            + cfg.w_rot * ca.sumsqr(rot_err)
            + cfg.w_regularization * ca.dot(rw * cq_sx, cq_sx)
            + cfg.w_smooth * ca.sumsqr(cq_sx - cqs_sx)
            + leg_total_sx
        )

        # Wrap as a reusable Function (SX inputs → scalar SX output)
        cost_fn = ca.Function("wbik_cost", [cq_sx, cTf_sx, cqs_sx], [cost_sx])

        # ── Phase 2: Opti problem with MX variables ───────────────────────────
        # opti.variable() / opti.parameter() return MX.
        # Calling a ca.Function on MX inputs is fully supported by CasADi.
        opti = ca.Opti()
        var_q = opti.variable(nq)  # decision variable (MX)
        var_q_seed = opti.parameter(nq)  # smoothness anchor  (MX)
        param_tf = opti.parameter(4, 4)  # target pose        (MX)

        opti.minimize(cost_fn(var_q, param_tf, var_q_seed))
        opti.subject_to(
            opti.bounded(
                self._reduced.model.lowerPositionLimit,
                var_q,
                self._reduced.model.upperPositionLimit,
            )
        )

        # ── IPOPT options ─────────────────────────────────────────────────────
        # Pass any of these under casadi_ipopt_options["ipopt"] to override defaults.
        # Full list: https://coin-or.github.io/Ipopt/OPTIONS.html
        #
        # Termination:
        #   tol (float, default 1e-8)     — Desired relative convergence tolerance; smaller = stricter.
        #   max_iter (int, default 3000)   — Max iterations before stop (we use cfg.max_iterations).
        #   acceptable_tol (float, 1e-6)  — Looser “acceptable” tolerance; if met for acceptable_iter
        #                                    steps in a row, IPOPT may stop early.
        #   acceptable_iter (int, 15)     — Number of “acceptable” iterates in a row to trigger early stop.
        #   dual_inf_tol (float)           — Absolute tolerance on dual infeasibility.
        #   constr_viol_tol (float)        — Absolute tolerance on constraint/bound violation.
        #   compl_inf_tol (float)         — Absolute tolerance on complementarity.
        # Output:
        #   print_level (0–12, default 5)  — Console verbosity; 0 = silent.
        # Time / divergence:
        #   max_cpu_time, max_wall_time   — Hard limits (seconds).
        #   diverging_iterates_tol        — Abort if primal iterates exceed this (absolute).
        # Line search (step size is controlled internally; no single “step length” option):
        #   alpha_for_y, alpha_pr, alpha_du — Tuned by line search; expert options exist but no simple
        #                                      “max step” knob. Use w_smooth in the cost to limit motion.
        solver_opts: Dict[str, Any] = {
            "ipopt": {
                "print_level": 0,
                "max_iter": cfg.max_iterations,
                "tol": 1e-8,
            },
            "print_time": False,
            "calc_lam_p": False,
        }
        if cfg.casadi_ipopt_options:
            user = cfg.casadi_ipopt_options
            if "ipopt" in user and isinstance(user["ipopt"], dict):
                solver_opts["ipopt"].update(user["ipopt"])
                user = {k: v for k, v in user.items() if k != "ipopt"}
            solver_opts.update(user)

        opti.solver("ipopt", solver_opts)

        # Store for reuse across all _get_ik_casadi calls
        self._casadi_opti = opti
        self._casadi_var_q = var_q
        self._casadi_var_q_seed = var_q_seed
        self._casadi_param_tf = param_tf
        self._ca = ca

    def _get_ik_casadi(
        self,
        target_se3,
        target_t: np.ndarray,
        q_seed: np.ndarray,
    ) -> Tuple[np.ndarray, bool, float, float, int]:
        """Solve IK using the pre‑built CasADi / IPOPT Opti problem.

        Follows the same call pattern as ``dex_w1_ArmIK.solve_ik``:
        update parameters with ``set_value`` / ``set_initial``, call ``solve``,
        and fall back to ``opti.debug.value`` on solver failure.

        Args:
            target_se3: Pinocchio SE3 target (in universe frame, TCP removed).
            target_t:   Target translation, pre‑extracted for speed.
            q_seed:     Seed in Pinocchio joint order.

        Returns:
            ``(q_sol, success, pos_err, rot_err, n_iter)`` all in Pinocchio joint order.
        """
        pin = self.pin
        cfg: WholeBodyIKSolverCfg = self.cfg

        # Build the 4×4 target transform for the parameter
        tf = np.eye(4)
        tf[:3, :3] = target_se3.rotation
        tf[:3, 3] = target_se3.translation

        self._casadi_opti.set_initial(self._casadi_var_q, q_seed)
        self._casadi_opti.set_value(self._casadi_var_q_seed, q_seed)
        self._casadi_opti.set_value(self._casadi_param_tf, tf)

        try:
            self._casadi_opti.solve()
            q_sol = np.array(self._casadi_opti.value(self._casadi_var_q)).flatten()
        except Exception:
            # IPOPT may return a sub‑optimal solution on failure; use debug values
            q_sol = np.array(
                self._casadi_opti.debug.value(self._casadi_var_q)
            ).flatten()

        # Evaluate actual errors with Pinocchio (consistent with DLS backend)
        pin.framesForwardKinematics(self._reduced.model, self._reduced.data, q_sol)
        oMf = self._reduced.data.oMf[self._ee_frame_id]
        pos_err = float(np.linalg.norm(oMf.translation - target_t))
        err6 = pin.log6(oMf.actInv(target_se3)).vector
        rot_err = float(np.linalg.norm(err6[3:]))
        success = pos_err < cfg.pos_eps and rot_err < cfg.rot_eps

        # Fetch IPOPT iteration count from Opti stats (if available in stats dict).
        stats = self._casadi_opti.stats()
        n_iter = int(stats.get("iter_count", 0)) if isinstance(stats, dict) else 0

        return q_sol, success, pos_err, rot_err, n_iter

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def describe_chain(self) -> str:
        """Return a human‑readable description of the kinematic chain from root to end‑effector."""
        model = self._reduced.model
        active_set = set(self.joint_names)
        cfg: WholeBodyIKSolverCfg = self.cfg

        fid = self._ee_frame_id
        chain = []
        jid = model.frames[fid].parentJoint
        while jid > 0:
            chain.append(model.names[jid])
            jid = model.parents[jid]
        chain.reverse()

        lines = [f"EE '{cfg.end_link_name}'  (root='{cfg.root_link_name}')"]
        lines.append("  universe (base)")
        for depth, jname in enumerate(chain):
            tag = "[active]" if jname in active_set else "[locked]"
            indent = "  " + "    " * depth + "└─ "
            lines.append(f"{indent}{jname}  {tag}")
        indent = "  " + "    " * len(chain) + "└─ "
        lines.append(f"{indent}(frame: {cfg.end_link_name})")
        return "\n".join(lines)

    def get_ik(
        self,
        target_xpos: "torch.Tensor | np.ndarray",
        qpos_seed: "torch.Tensor | np.ndarray | None" = None,
        qvel_seed: "np.ndarray | None" = None,
        return_all_solutions: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Solve whole‑body inverse kinematics.

        The public interface is compatible with :class:`PinocchioSolver`; the
        active end‑effector and TCP are taken from the configuration (there is
        no extra parameter for end‑effector selection here).

        Args:
            target_xpos:          Target TCP pose as a homogeneous transform, shape ``(4, 4)`` in world/root coordinates, or a batch ``(1, 4, 4)`` (the first element is used).
            qpos_seed:            Seed configuration in cfg joint order, used as initialization and smoothness reference. If None, the last solution (or zeros) is used.
            qvel_seed:            Reserved for signature compatibility with :class:`PinocchioSolver` (currently unused).
            return_all_solutions: Reserved for signature compatibility (currently unused, only the best solution is returned).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - ``success`` — shape ``(1,)`` bool tensor, True if convergence criteria are met.
                - ``joints``  — shape ``(nq,)`` float32 tensor, IK solution in cfg joint order.
        """
        pin = self.pin

        # ── Pre‑process inputs ────────────────────────────────────────────────
        if isinstance(target_xpos, torch.Tensor):
            target_xpos = target_xpos.detach().cpu().numpy()
        target_xpos = np.asarray(target_xpos, dtype=float)
        if target_xpos.ndim == 3:
            target_xpos = target_xpos[0]

        # root_base_xpos: root_link frame → Pinocchio universe frame (same convention as PinocchioSolver)
        # TCP inverse: map target into the end_link frame
        compute_xpos = self._root_base_xpos @ target_xpos @ np.linalg.inv(self.tcp_xpos)
        target_R = compute_xpos[:3, :3]
        target_t = compute_xpos[:3, 3]
        target_se3 = pin.SE3(target_R.copy(), target_t.copy())

        if qpos_seed is not None:
            if isinstance(qpos_seed, torch.Tensor):
                qpos_seed = qpos_seed.detach().cpu().numpy()
            q_seed_cfg = np.asarray(qpos_seed, dtype=float).flatten()
            q_seed = q_seed_cfg[self._cfg_to_pin]
        else:
            q_seed = self._last_q.copy()

        # ── Dispatch to selected backend ──────────────────────────────────────
        cfg: WholeBodyIKSolverCfg = self.cfg
        if cfg.backend == "casadi":
            q, success, _pos_err, _rot_err, _n_iter = self._get_ik_casadi(
                target_se3, target_t, q_seed
            )
        else:
            q, success, _pos_err, _rot_err, _n_iter = self._get_ik_dls(
                target_se3, target_t, q_seed
            )

        self._last_q = q.copy()
        self.last_solve_info = {
            "success": success,
            "iterations": _n_iter,
            "pos_err": _pos_err,
            "rot_err": _rot_err,
        }

        q_cfg = q[self._pin_to_cfg]
        return (
            torch.tensor([success], dtype=torch.bool),
            torch.from_numpy(q_cfg).to(dtype=torch.float32),
        )

    def get_fk(
        self,
        qpos: "torch.Tensor | np.ndarray",
        **kwargs,
    ) -> torch.Tensor:
        """Compute forward kinematics for the end‑effector (including TCP offset).

        The interface is fully compatible with :class:`PinocchioSolver`.

        Args:
            qpos: Joint angles in cfg joint order, shape ``(nq,)`` or ``(B, nq)``.

        Returns:
            torch.Tensor: Tool‑centre‑point pose as a homogeneous transform.
                - Unbatched input ``(nq,)``  → ``(4, 4)``
                - Batched input   ``(B, nq)`` → ``(B, 4, 4)``
        """
        pin = self.pin

        if isinstance(qpos, torch.Tensor):
            qpos = qpos.detach().cpu().numpy()
        qpos = np.asarray(qpos, dtype=float)

        batched = qpos.ndim == 2
        if not batched:
            qpos = qpos[np.newaxis]  # (1, nq)

        results = np.empty((len(qpos), 4, 4))
        for i, q_cfg in enumerate(qpos):
            q_pin = q_cfg[self._cfg_to_pin]
            pin.framesForwardKinematics(self._reduced.model, self._reduced.data, q_pin)
            oMf = self._reduced.data.oMf[self._ee_frame_id]
            mat = np.eye(4)
            mat[:3, :3] = oMf.rotation
            mat[:3, 3] = oMf.translation
            results[i] = mat @ self.tcp_xpos

        out = torch.from_numpy(results).to(dtype=torch.float32)
        return out if batched else out[0]
