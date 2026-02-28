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
- **null‑space secondary objectives** (regularization, smoothness, leg / torso stability) to prevent redundant DOFs from drifting.

The configuration interface is fully aligned with :class:`PinocchioSolver`::

    WholeBodyIKSolverCfg(
        end_link_name="left_ee",
        root_link_name="base_link",
        tcp=left_arm_tcp,        # 4x4 TCP matrix
        joint_names=None,        # auto‑filled from control_parts by Robot.init_solver
        ...
    )

Typical usage via :meth:`DexforceW1Cfg._build_default_solver_cfg`::

    robot.compute_ik(pose, name="left_arm_body")    # left arm whole‑body IK
    robot.compute_ik(pose, name="right_arm_body")   # right arm whole‑body IK

Per‑iteration algorithm:

1. ``pin.framesForwardKinematics(q)`` → current end‑effector pose
2. apply TCP inverse to map target into the end‑link frame: ``target_ee = target @ inv(tcp)``
3. ``pin.log6(oMf.actInv(target_ee))`` → 6D pose error in LOCAL frame
4. ``pin.computeFrameJacobian(q, fid, LOCAL)`` → 6 x dof Jacobian
5. DLS pseudo‑inverse ``J^T (JJ^T + λI)^{-1}`` → primary task increment
6. null‑space term ``(I - J^† J) * (−∇E_sec)`` → secondary objectives increment
7. ``pin.integrate`` + clipping joint limits
"""

import os
from dataclasses import field
from typing import Dict, List, Optional, Tuple

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
        urdf_dir:          Directory for URDF meshes; if None, falls back to mesh_path or the URDF directory.
        w_pos:             Weight for end‑effector position error.
        w_rot:             Weight for end‑effector orientation error.
        max_iterations:    Maximum DLS iterations per IK call (default 200, overriding the parent default).
        dt:                Integration time step (default 0.5, overriding the parent default).
        damp:              DLS damping factor (default 1e‑3, overriding the parent default).
        pos_eps:           Convergence threshold on translation (meters).
        rot_eps:           Convergence threshold on rotation (radians).
        w_regularization:  Weight for joint regularization in the secondary objective.
        w_smooth:          Weight for smoothness (stay close to q_seed) in the secondary objective.
        joint_reg_extra:   Extra regularization weights for specific joints, as {"joint_name": extra_weight}.
        leg_costs_mode2:   List of :class:`LegCostCfg` to use when ``leg_mode == 2``.
        leg_costs_mode3:   List of :class:`LegCostCfg` to use when ``leg_mode == 3``.
        leg_mode:          Leg mode selector: 2 = standing, 3 = walking.
    """

    class_type: str = "WholeBodyIKSolver"

    # Override PinocchioSolverCfg defaults to better suit whole‑body IK
    max_iterations: int = 200
    dt: float = 0.5
    damp: float = 1e-3
    pos_eps: float = 1e-3
    rot_eps: float = 1e-3

    # Whole‑body specific fields
    urdf_dir: Optional[str] = None

    w_pos: float = 1.0
    w_rot: float = 0.1

    w_regularization: float = 0.02
    w_smooth: float = 0.1

    joint_reg_extra: Optional[Dict[str, float]] = None

    leg_costs_mode2: Optional[List[LegCostCfg]] = None
    leg_costs_mode3: Optional[List[LegCostCfg]] = None
    leg_mode: int = 2

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

        # Handle input
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

        cfg: WholeBodyIKSolverCfg = self.cfg
        fid = self._ee_frame_id
        w_diag = np.array([cfg.w_pos] * 3 + [cfg.w_rot] * 3)

        # DLS IK iterations (all internal vectors in pinocchio joint order)
        q = q_seed.copy().astype(float)
        success = False
        _iter = 0
        _pos_err = float("inf")
        _rot_err = float("inf")

        for _iter in range(cfg.max_iterations):
            pin.framesForwardKinematics(self._reduced.model, self._reduced.data, q)
            oMf = self._reduced.data.oMf[fid]

            # 6D error in LOCAL frame (same frame as log6, numerically stable even near 180°)
            err6 = pin.log6(oMf.actInv(target_se3)).vector

            _pos_err = float(np.linalg.norm(oMf.translation - target_t))
            _rot_err = float(np.linalg.norm(err6[3:]))

            if _pos_err < cfg.pos_eps and _rot_err < cfg.rot_eps:
                success = True
                break

            err = err6 * w_diag

            J_full = pin.computeFrameJacobian(
                self._reduced.model,
                self._reduced.data,
                q,
                fid,
                pin.LOCAL,
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

        self._last_q = q.copy()
        self.last_solve_info = {
            "success": success,
            "iterations": _iter + 1,
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
        """计算末端执行器的正运动学（含 TCP 偏移）。

        接口与 :class:`PinocchioSolver` 完全一致。

        Args:
            qpos: 关节角，shape ``(nq,)`` 或 ``(batch, nq)``（cfg 顺序）。

        Returns:
            torch.Tensor: 工具末端位姿（含 TCP）。
                - 单配置输入 ``(nq,)``  → ``(4, 4)``
                - 批次输入   ``(B, nq)`` → ``(B, 4, 4)``
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
