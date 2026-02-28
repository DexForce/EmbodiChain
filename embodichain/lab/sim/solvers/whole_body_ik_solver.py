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
"""WholeBodyIKSolver — 全身IK求解器。

继承自 :class:`~embodichain.lab.sim.solvers.PinocchioSolver`，完全重写
``__init__``、``get_ik``、``get_fk``，在与项目其他 solver 保持相同配置接口
（``end_link_name`` + ``tcp`` + ``root_link_name``）的基础上，额外支持：

- **全身冗余模型**：``pin.buildReducedRobot`` 保留所有活动关节（单侧手臂 + 躯干）
- **零空间次级任务**：正则化、平滑、腿部稳定性约束，防止冗余自由度随机漂移

配置接口与 :class:`PinocchioSolver` 完全一致::

    WholeBodyIKSolverCfg(
        end_link_name="left_ee",        # URDF 已有的末端 link，与其他 solver 相同
        root_link_name="base_link",     # 根坐标系，用于 robot.compute_ik 中的坐标变换
        tcp=left_arm_tcp,               # 4×4 工具中心点矩阵，与其他 solver 相同
        joint_names=None,               # 由 robot.init_solver 从 control_parts 自动填充
        ...
    )

推荐通过 :meth:`DexforceW1Cfg._build_default_whole_body_solver_cfg` 使用::

    robot.compute_ik(pose, name="left_arm_body")   # 左臂全身IK
    robot.compute_ik(pose, name="right_arm_body")  # 右臂全身IK

算法（每次迭代）：

1. ``pin.framesForwardKinematics(q)`` → 当前末端位姿
2. 应用 TCP 逆变换将目标转换到 end_link 坐标：``target_ee = target @ inv(tcp)``
3. ``pin.log6(oMf.actInv(target_ee))`` → 6D 误差（LOCAL 帧，数值稳定）
4. ``pin.computeFrameJacobian(q, fid, LOCAL)`` → 6×nq 雅可比
5. DLS 伪逆 ``J^T (JJ^T + λI)^{-1}`` → 主任务增量
6. 零空间 ``(I - J^† J) * (−∇E_sec)`` → 次级任务增量
7. ``pin.integrate`` + clip 关节限位
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
# 腿部 / 躯干稳定性次级目标
# ---------------------------------------------------------------------------


@configclass
class LegCostCfg:
    """一条腿部 / 躯干稳定性次级代价项（线性组合的平方）。

    代价形式：``weight * (sum_i coefficients[i] * q[joint_names[i]])²``

    梯度在零空间投影中作为次级目标引导关节角。

    Examples::

        # 约束 WAIST 与 KNEE 等幅反向（保持腿部对称）
        LegCostCfg(joint_names=["WAIST", "KNEE"], coefficients=[1.0, -1.0], weight=10.0)
    """

    joint_names: List[str] = field(default_factory=list)
    coefficients: List[float] = field(default_factory=list)
    weight: float = 10.0


# ---------------------------------------------------------------------------
# 求解器总配置
# ---------------------------------------------------------------------------


@configclass
class WholeBodyIKSolverCfg(PinocchioSolverCfg):
    """全身IK求解器配置（继承自 :class:`PinocchioSolverCfg`）。

    配置接口与 :class:`PinocchioSolverCfg` 完全对齐：
    通过 ``end_link_name``（URDF 已有 link）、``tcp``（工具偏移矩阵）、
    ``root_link_name`` 定义末端执行器，无需额外的 ``EndEffectorCfg``。

    ``joint_names`` 的语义与 :class:`PinocchioSolverCfg` 不同：
    这里表示"需要保持活动（未锁定）的关节集合"，而非串联运动链路径。
    推荐将其设为 ``None`` 并通过 ``robot.init_solver`` 从 ``control_parts`` 自动填充。

    Attributes:
        urdf_dir:          URDF 网格文件搜索目录；None 时依次尝试 mesh_path 或 urdf_path 所在目录。
        w_pos:             末端位置误差权重（越大位置跟踪越精确）。
        w_rot:             末端姿态误差权重。
        max_iterations:    每次 IK 调用的最大迭代步数（默认 200，覆盖父类 1000）。
        dt:                积分步长（默认 0.5，覆盖父类 0.1）。
        damp:              DLS 阻尼系数（默认 1e-3，覆盖父类 1e-6）。
        pos_eps:           位置收敛阈值，米（默认 1e-3，覆盖父类 5e-4）。
        rot_eps:           姿态收敛阈值，弧度（默认 1e-3，覆盖父类 5e-4）。
        w_regularization:  次级任务：关节正则化权重。
        w_smooth:          次级任务：平滑（靠近 q_seed）权重。
        joint_reg_extra:   对特定关节追加正则化系数，格式 {"joint_name": extra_weight}。
        leg_costs_mode2:   leg_mode=2 时的稳定性次级代价列表。
        leg_costs_mode3:   leg_mode=3 时的稳定性次级代价列表。
        leg_mode:          腿部模式：2=标准站立，3=行走。
    """

    class_type: str = "WholeBodyIKSolver"

    # ── 覆盖 PinocchioSolverCfg 的默认值以适配全身IK ────────────────────────
    max_iterations: int = 200
    dt: float = 0.5
    damp: float = 1e-3
    pos_eps: float = 1e-3
    rot_eps: float = 1e-3

    # ── 全身IK独有字段 ────────────────────────────────────────────────────────
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
# 求解器实现
# ---------------------------------------------------------------------------


class WholeBodyIKSolver(PinocchioSolver):
    """全身IK求解器（DLS 迭代 IK + 零空间次级任务）。

    继承自 :class:`PinocchioSolver`，完全重写 ``__init__``、``get_ik``、``get_fk``。
    配置接口与 :class:`PinocchioSolver` 完全一致，通过继承的 ``end_link_name``
    和 ``tcp_xpos`` 处理末端执行器，无额外自定义配置类。

    算法步骤（每次迭代）：

    1. **正运动学**：``pin.framesForwardKinematics(q)`` 得到当前末端位姿
    2. **TCP 逆变换**：``target_ee = target @ inv(tcp)``，将工具目标换算到 end_link
    3. **误差计算**：``pin.log6(oMf.actInv(target_ee))``，LOCAL 帧，数值稳定
    4. **雅可比**：``pin.computeFrameJacobian(q, frame_id, LOCAL)``
    5. **主任务 dq**：DLS 伪逆 ``J^T (JJ^T + λI)^{-1} e``
    6. **次级任务**：正则化 + 平滑 + 腿部稳定性的梯度，零空间 ``(I - J^† J)`` 投影
    7. **积分 & 限位**：``q ← pin.integrate(q, dq * dt)``，然后 clip 到关节限位
    """

    def __init__(self, cfg: WholeBodyIKSolverCfg, **kwargs):
        # 跳过 PinocchioSolver.__init__（串联链专用初始化）。
        # 直接调用 BaseSolver.__init__ 完成 cfg、device、joint_names 等基础属性设置。
        kwargs["pk_serial_chain"] = "_skip_"
        BaseSolver.__init__(self, cfg=cfg, **kwargs)
        self.pk_serial_chain = None

        # 与 PinocchioSolver 保持一致的属性名
        self.pin = lazy_import_pinocchio()
        pin = self.pin

        urdf_dir = cfg.urdf_dir or cfg.mesh_path or os.path.dirname(cfg.urdf_path)

        # ── 1. 加载完整机器人，锁定非活动关节 ────────────────────────────────
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

        # pinocchio 按 URDF 运动树决定关节顺序，与 cfg.joint_names 的顺序无关。
        # 这里计算两个排列数组，让 solver 的对外接口严格遵循 cfg.joint_names 的顺序。
        pin_order = [
            self._reduced.model.names[i] for i in range(1, self._reduced.model.njoints)
        ]
        cfg_order = list(cfg.joint_names or pin_order)

        # cfg_order[i] 在 pinocchio 中的索引 → _cfg_to_pin[i]
        self._cfg_to_pin: np.ndarray = np.array(
            [pin_order.index(j) for j in cfg_order], dtype=int
        )
        # pin_order[i] 在 cfg_order 中的索引 → _pin_to_cfg[i]
        self._pin_to_cfg: np.ndarray = np.array(
            [cfg_order.index(j) for j in pin_order], dtype=int
        )

        # 对外暴露的 joint_names 和限位均以 cfg 顺序为准
        self.joint_names = cfg_order

        # ── 2. 查找末端帧 ID（使用 end_link_name，与其他 solver 一致）──────────
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

        # ── 3. root_base_xpos：从 Pinocchio universe 到 root_link_name 的偏移 ──
        # 与 PinocchioSolver 逻辑一致：get_ik 收到的目标在 root_link_name 坐标系下，
        # 需要再乘以此矩阵才能得到 Pinocchio FK 使用的 universe 坐标系下的目标。
        # 当 root_link_name 是机器人的根节点（如 "base_link"）时，该矩阵为单位阵。
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

        # ── 4. 关节名 → pinocchio 内部 q 索引映射 ──────────────────────────
        self._joint_name_to_q_idx: Dict[str, int] = {}
        for i in range(1, self._reduced.model.njoints):
            self._joint_name_to_q_idx[self._reduced.model.names[i]] = i - 1

        # ── 5. 正则化权重向量 ─────────────────────────────────────────────────
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

        # ── 6. 预编译腿部次级代价梯度函数 ─────────────────────────────────────
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

        # 关节限位以 cfg 顺序对外暴露（与 PinocchioSolver 的属性名一致）
        self.lower_position_limits = self._reduced.model.lowerPositionLimit[
            self._pin_to_cfg
        ]
        self.upper_position_limits = self._reduced.model.upperPositionLimit[
            self._pin_to_cfg
        ]

        self._last_q = np.zeros(self.dof)

        #: 上一次 get_ik 调用后的诊断信息，可用于判断解的质量。
        #: 字段：success(bool), iterations(int), pos_err(float,m), rot_err(float,rad)
        self.last_solve_info: dict = {
            "success": False,
            "iterations": 0,
            "pos_err": float("inf"),
            "rot_err": float("inf"),
        }

    # -----------------------------------------------------------------------
    # 内部：次级任务梯度
    # -----------------------------------------------------------------------

    def _secondary_gradient(self, q: np.ndarray, q_seed: np.ndarray) -> np.ndarray:
        """计算次级目标（正则化 + 平滑 + 腿部稳定性）的关节空间梯度。"""
        cfg: WholeBodyIKSolverCfg = self.cfg
        g = np.zeros(self.dof)
        g += 2.0 * cfg.w_regularization * self._reg_weights * q
        g += 2.0 * cfg.w_smooth * (q - q_seed)
        for indices, coeffs, weight in self._leg_cost_terms:
            val = float(np.dot(coeffs, q[indices]))
            g[indices] += 2.0 * weight * val * coeffs
        return g

    # -----------------------------------------------------------------------
    # 公开 API
    # -----------------------------------------------------------------------

    def describe_chain(self) -> str:
        """返回从根节点到末端执行器的运动链描述字符串。

        Returns:
            格式化后的链路描述字符串，可直接 print。
        """
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
        """求解全身逆运动学。

        接口与 :class:`PinocchioSolver` 完全一致，通过继承的 ``end_link_name`` 和
        ``tcp_xpos`` 定义目标末端，无需额外参数。

        Args:
            target_xpos:          目标工具末端位姿，``(4, 4)`` 齐次变换矩阵（世界/根坐标系）。
                                  也接受 ``(1, 4, 4)`` 批量张量（取第 0 个）。
            qpos_seed:            当前关节角，作为迭代初始值和平滑基准（cfg 顺序）。
                                  若为 None，使用上次求解结果（或全零）。
            qvel_seed:            预留参数（暂未使用），与 :class:`PinocchioSolver` 签名对齐。
            return_all_solutions: 预留参数（暂未使用），与 :class:`PinocchioSolver` 签名对齐。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - ``success`` — shape ``(1,)`` bool 张量，True 表示收敛。
                - ``joints``  — shape ``(nq,)`` float32 张量，求解后关节角（cfg 顺序）。
        """
        pin = self.pin

        # ── 处理输入 ──────────────────────────────────────────────────────────
        if isinstance(target_xpos, torch.Tensor):
            target_xpos = target_xpos.detach().cpu().numpy()
        target_xpos = np.asarray(target_xpos, dtype=float)
        if target_xpos.ndim == 3:
            target_xpos = target_xpos[0]

        # root_base_xpos：root_link 坐标系 → Pinocchio universe 坐标系（与 PinocchioSolver 一致）
        # TCP 逆变换：将工具目标换算到 end_link 坐标
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

        # ── 迭代 DLS IK（内部全程使用 pinocchio 顺序）───────────────────────
        q = q_seed.copy().astype(float)
        success = False
        _iter = 0
        _pos_err = float("inf")
        _rot_err = float("inf")

        for _iter in range(cfg.max_iterations):
            pin.framesForwardKinematics(self._reduced.model, self._reduced.data, q)
            oMf = self._reduced.data.oMf[fid]

            # 6D 误差（LOCAL 帧，与 log6 保持同一坐标系，对 180° 奇点数值稳定）
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
