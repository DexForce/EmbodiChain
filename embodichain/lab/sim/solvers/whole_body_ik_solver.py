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
"""WholeBodyIKSolver — 全身IK求解器（纯 Pinocchio，无 CasADi 依赖）。

算法：带零空间次级任务的阻尼最小二乘（DLS）迭代 IK。

- **主任务**：末端执行器位置 + 姿态跟踪（6D 误差，加权）
- **次级任务**（零空间投影）：关节正则化 + 平滑 + 腿部稳定性约束
- **关节限位**：每步积分后直接 clip（硬约束）

仅依赖 ``pinocchio``（pip install pin 即可），无需 pinocchio-casadi / CasADi / IPOPT。

用法示例::

    cfg = WholeBodyIKSolverCfg(
        urdf_path="/path/to/robot.urdf",
        urdf_dir="/path/to/meshes",
        joint_names=["LEFT_J1", ..., "RIGHT_J7", "WAIST", ...],
        end_effectors={
            "left":  EndEffectorCfg(parent_joint="LEFT_J7",  ...),
            "right": EndEffectorCfg(parent_joint="RIGHT_J7", ...),
            "head":  EndEffectorCfg(parent_joint="NECK2",    ...),
        },
        active_ee="left",
    )
    solver = cfg.init_solver()

    success, q = solver.get_ik(target_pose_4x4, qpos_seed=current_q, active_ee="right")
"""

import os
from dataclasses import field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from embodichain.utils import configclass, logger
from embodichain.lab.sim.solvers.base_solver import BaseSolver, SolverCfg
from embodichain.lab.sim.utility.import_utils import lazy_import_pinocchio


# ---------------------------------------------------------------------------
# 末端执行器帧配置
# ---------------------------------------------------------------------------


@configclass
class EndEffectorCfg:
    """单个末端执行器帧的配置。

    Attributes:
        frame_name:   注册到 Pinocchio 模型中的帧名称，调用时也用此名称索引。
        parent_joint: 该帧挂载的父关节名（必须存在于 reduced model 中）。
        rotation:     父关节坐标系到末端帧的旋转矩阵（3×3，行主序嵌套列表）。
        translation:  父关节坐标系到末端帧的平移向量 [x, y, z]（米）。
        w_pos:        位置误差权重（越大位置跟踪越精确）。
        w_rot:        姿态误差权重。
    """

    frame_name: str = ""
    parent_joint: str = ""
    rotation: List[List[float]] = field(
        default_factory=lambda: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    )
    translation: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    w_pos: float = 1.0
    w_rot: float = 0.1


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
class WholeBodyIKSolverCfg(SolverCfg):
    """全身IK求解器配置（纯 Pinocchio，DLS + 零空间投影）。

    Attributes:
        joint_names:        保留为活动关节的关节名列表（其余关节将被锁定为 0）。
        urdf_dir:           URDF 网格文件搜索目录。None 时自动取 urdf_path 所在目录。
        end_effectors:      末端执行器字典，键为自定义名称，值为 :class:`EndEffectorCfg`。
        active_ee:          默认激活的末端执行器名称。
        max_iterations:     每次 IK 调用的最大迭代步数。
        dt:                 积分步长（越小越稳，越大收敛越快）。
        damp:               DLS 阻尼系数（防止奇异）。
        pos_eps:            位置收敛阈值（米）。
        rot_eps:            姿态收敛阈值（弧度）。
        w_regularization:   次级任务：关节正则化权重。
        w_smooth:           次级任务：平滑（靠近 q_seed）权重。
        joint_reg_extra:    对特定关节追加正则化系数，格式 {"joint_name": extra_weight}。
        leg_costs_mode2:    leg_mode=2 时的稳定性次级代价列表。
        leg_costs_mode3:    leg_mode=3 时的稳定性次级代价列表。
        leg_mode:           腿部模式：2=标准站立，3=行走。
    """

    class_type: str = "WholeBodyIKSolver"

    urdf_dir: Optional[str] = None

    end_effectors: Optional[Dict[str, EndEffectorCfg]] = None

    active_ee: Optional[str] = None

    max_iterations: int = 200
    dt: float = 0.5
    damp: float = 1e-3

    pos_eps: float = 1e-3
    rot_eps: float = 1e-3

    w_regularization: float = 0.02
    w_smooth: float = 0.1

    joint_reg_extra: Optional[Dict[str, float]] = None

    leg_costs_mode2: Optional[List[LegCostCfg]] = None
    leg_costs_mode3: Optional[List[LegCostCfg]] = None
    leg_mode: int = 2

    def init_solver(self, **kwargs) -> "WholeBodyIKSolver":
        return WholeBodyIKSolver(cfg=self, **kwargs)


# ---------------------------------------------------------------------------
# 求解器实现
# ---------------------------------------------------------------------------


class WholeBodyIKSolver(BaseSolver):
    """全身IK求解器（DLS 迭代 IK + 零空间次级任务）。

    算法步骤（每次迭代）：

    1. **正运动学**：`pin.framesForwardKinematics(q)` 得到当前末端位姿
    2. **误差计算**：位置误差 + 姿态误差（`pin.log3`），加权成 6D 误差向量
    3. **雅可比**：`pin.computeFrameJacobian(q, frame_id)` 得到 6×nq 雅可比
    4. **主任务 dq**：DLS 伪逆 ``J^T (JJ^T + λI)^{-1} e``
    5. **次级任务**：正则化 + 平滑 + 腿部稳定性的梯度，通过零空间
       ``(I - J^† J)`` 投影后叠加到 dq
    6. **积分 & 限位**：``q ← pin.integrate(q, dq * dt)``，然后 clip 到关节限位
    """

    def __init__(self, cfg: WholeBodyIKSolverCfg, **kwargs):
        # 注入哨兵跳过 BaseSolver 中的 pk_serial_chain 构建
        kwargs["pk_serial_chain"] = "_skip_"
        super().__init__(cfg=cfg, **kwargs)
        self.pk_serial_chain = None

        self._pin = lazy_import_pinocchio()
        pin = self._pin

        urdf_dir = cfg.urdf_dir or os.path.dirname(cfg.urdf_path)

        # ── 1. 加载完整机器人，锁定非活动关节 ────────────────────────────
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

        # ── 2. 注册末端执行器帧 ───────────────────────────────────────────
        ee_cfgs: Dict[str, EndEffectorCfg] = cfg.end_effectors or {}
        for ee_name, ee_cfg in ee_cfgs.items():
            frame_name = ee_cfg.frame_name or ee_name
            parent_id = self._reduced.model.getJointId(ee_cfg.parent_joint)
            if parent_id >= self._reduced.model.njoints:
                raise ValueError(
                    f"[WholeBodyIKSolver] parent_joint='{ee_cfg.parent_joint}' "
                    f"for EE '{ee_name}' not found in reduced model."
                )
            rot = np.array(ee_cfg.rotation, dtype=float)
            trans = np.array(ee_cfg.translation, dtype=float)
            self._reduced.model.addFrame(
                pin.Frame(
                    frame_name,
                    parent_id,
                    pin.SE3(rot.T, trans),
                    pin.FrameType.OP_FRAME,
                )
            )
        # 注册后需要重新创建 data
        self._reduced.data = self._reduced.model.createData()

        # ── 3. 缓存末端帧 ID ─────────────────────────────────────────────
        self._ee_frame_ids: Dict[str, int] = {}
        self._ee_cfgs: Dict[str, EndEffectorCfg] = {}
        for ee_name, ee_cfg in ee_cfgs.items():
            frame_name = ee_cfg.frame_name or ee_name
            self._ee_frame_ids[ee_name] = self._reduced.model.getFrameId(frame_name)
            self._ee_cfgs[ee_name] = ee_cfg

        # ── 4. 关节名 → pinocchio 内部 q 索引映射（内部计算始终用 pin 顺序）
        self._joint_name_to_q_idx: Dict[str, int] = {}
        for i in range(1, self._reduced.model.njoints):
            self._joint_name_to_q_idx[self._reduced.model.names[i]] = i - 1

        # ── 5. 正则化权重向量 ─────────────────────────────────────────────
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

        # ── 6. 预编译腿部次级代价梯度函数 ────────────────────────────────
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

        # 关节限位以 cfg 顺序对外暴露
        self.lower_position_limits = self._reduced.model.lowerPositionLimit[
            self._pin_to_cfg
        ]
        self.upper_position_limits = self._reduced.model.upperPositionLimit[
            self._pin_to_cfg
        ]

        self._active_ee: Optional[str] = cfg.active_ee
        self._last_q = np.zeros(self.dof)

        #: 上一次 get_ik 调用后的诊断信息，可用于判断解的质量。
        #: 字段说明：
        #:   success     (bool)  — 迭代是否在阈值内收敛
        #:   iterations  (int)   — 实际迭代步数
        #:   pos_err     (float) — 收敛时的位置误差（米）
        #:   rot_err     (float) — 收敛时的旋转误差（弧度）
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
        """计算次级目标（正则化 + 平滑 + 腿部稳定性）的关节空间梯度。

        次级代价：
            E_sec = w_reg  * Σ reg_w[i] * q[i]²
                  + w_smo  * ||q - q_seed||²
                  + Σ_k  weight_k * (Σ_j coeff_j * q[idx_j])²

        对 q 的梯度（乘以 2）：
            g_reg[i]  = 2 * w_reg  * reg_w[i] * q[i]
            g_smo[i]  = 2 * w_smo  * (q[i] - q_seed[i])
            g_leg_k   = 2 * weight_k * val_k * coeff_j  （在各参与关节处）
        """
        cfg: WholeBodyIKSolverCfg = self.cfg
        g = np.zeros(self.dof)

        # 正则化
        g += 2.0 * cfg.w_regularization * self._reg_weights * q

        # 平滑（靠近 q_seed）
        g += 2.0 * cfg.w_smooth * (q - q_seed)

        # 腿部稳定性
        for indices, coeffs, weight in self._leg_cost_terms:
            val = float(np.dot(coeffs, q[indices]))
            g[indices] += 2.0 * weight * val * coeffs

        return g

    # -----------------------------------------------------------------------
    # 公开 API
    # -----------------------------------------------------------------------

    def describe_chain(self, ee_name: str | None = None) -> str:
        """返回从根节点到指定末端执行器的完整运动链描述字符串。

        以树状格式打印每一层的关节名称，同时标注该关节是否属于活动关节
        （参与 IK 求解）或已被锁定（零自由度）。

        Args:
            ee_name: 末端执行器名称；为 None 时打印所有已注册末端的链路。

        Returns:
            格式化后的链路描述字符串，可直接 print。

        Example::

            solver = cfg.init_solver()
            print(solver.describe_chain("left"))
            # universe
            #   └─ ANKLE        [active]
            #       └─ KNEE     [active]
            #           └─ BUTTOCK [active]
            #               └─ WAIST  [active]
            #                   └─ LEFT_J1 [active]
            #                       └─ ...
            #                           └─ LEFT_J7 [active]
            #                               └─ (EE frame: left)
        """
        model = self._reduced.model
        active_set = set(self.joint_names)

        def _joint_chain(frame_id: int) -> list[str]:
            """从 frame 所在关节向根回溯，返回 [root→EE] 的关节名列表。"""
            chain = []
            jid = model.frames[frame_id].parentJoint
            while jid > 0:
                chain.append(model.names[jid])
                jid = model.parents[jid]
            chain.reverse()
            return chain

        targets = [ee_name] if ee_name is not None else list(self._ee_frame_ids.keys())
        lines = []
        for name in targets:
            if name not in self._ee_frame_ids:
                lines.append(f"[EE '{name}' not found]")
                continue
            chain = _joint_chain(self._ee_frame_ids[name])
            lines.append(
                f"EE '{name}'  (parent_joint={self._ee_cfgs[name].parent_joint})"
            )
            lines.append("  universe (base)")
            for depth, jname in enumerate(chain):
                tag = "[active]" if jname in active_set else "[locked]"
                indent = "  " + "    " * depth + "└─ "
                lines.append(f"{indent}{jname}  {tag}")
            ee_cfg = self._ee_cfgs[name]
            indent = "  " + "    " * len(chain) + "└─ "
            lines.append(f"{indent}(EE frame offset  t={ee_cfg.translation})")
            lines.append("")
        return "\n".join(lines)

    def set_active_ee(self, ee_name: str) -> None:
        """切换默认激活的末端执行器。

        Args:
            ee_name: 末端执行器名称，必须已在 ``end_effectors`` 中配置。
        """
        if ee_name not in self._ee_frame_ids:
            raise KeyError(
                f"[WholeBodyIKSolver] EE '{ee_name}' not found. "
                f"Available: {list(self._ee_frame_ids.keys())}"
            )
        self._active_ee = ee_name

    def get_ik(
        self,
        target_xpos: "torch.Tensor | np.ndarray",
        qpos_seed: "torch.Tensor | np.ndarray | None" = None,
        active_ee: "str | None" = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """求解全身逆运动学（单末端输入）。

        算法：带零空间次级任务的阻尼最小二乘迭代 IK。

        Args:
            target_xpos:  目标末端位姿，``(4, 4)`` 齐次变换矩阵。
                          也接受 ``(1, 4, 4)`` 批量张量（取第 0 个）。
            qpos_seed:    当前关节角，作为迭代初始值和平滑基准。
                          若为 None，使用上次求解结果（或全零）。
            active_ee:    指定本次求解的末端执行器名称。
                          若为 None，使用 :attr:`_active_ee` 默认值。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - ``success`` — shape ``(1,)`` bool 张量，True 表示收敛。
                - ``joints``  — shape ``(nq,)`` float32 张量，求解后关节角。
        """
        pin = self._pin

        # ── 确定激活末端 ─────────────────────────────────────────────────
        ee = active_ee or self._active_ee
        if ee is None:
            raise ValueError(
                "[WholeBodyIKSolver] active_ee not specified and no default set. "
                "Pass active_ee= or set cfg.active_ee."
            )
        if ee not in self._ee_frame_ids:
            raise KeyError(
                f"[WholeBodyIKSolver] EE '{ee}' not found. "
                f"Available: {list(self._ee_frame_ids.keys())}"
            )

        # ── 处理输入 ─────────────────────────────────────────────────────
        if isinstance(target_xpos, torch.Tensor):
            target_xpos = target_xpos.detach().cpu().numpy()
        target_xpos = np.asarray(target_xpos, dtype=float)
        if target_xpos.ndim == 3:
            target_xpos = target_xpos[0]

        if qpos_seed is not None:
            if isinstance(qpos_seed, torch.Tensor):
                qpos_seed = qpos_seed.detach().cpu().numpy()
            # 输入为 cfg 顺序 → 转换为 pinocchio 内部顺序
            q_seed_cfg = np.asarray(qpos_seed, dtype=float).flatten()
            q_seed = q_seed_cfg[self._cfg_to_pin]
        else:
            q_seed = self._last_q.copy()  # 内部始终存储 pin 顺序

        cfg: WholeBodyIKSolverCfg = self.cfg
        fid = self._ee_frame_ids[ee]
        ee_cfg = self._ee_cfgs[ee]

        # 目标 SE3
        target_R = target_xpos[:3, :3]
        target_t = target_xpos[:3, 3]
        target_se3 = pin.SE3(target_R.copy(), target_t.copy())

        # 权重对角（位置 x3，姿态 x3）
        w_diag = np.array([ee_cfg.w_pos] * 3 + [ee_cfg.w_rot] * 3)

        # ── 迭代 DLS IK（内部全程使用 pinocchio 顺序）────────────────────
        q = q_seed.copy().astype(float)
        success = False
        _iter = 0
        _pos_err = float("inf")
        _rot_err = float("inf")

        for _iter in range(cfg.max_iterations):
            # FK
            pin.framesForwardKinematics(self._reduced.model, self._reduced.data, q)
            oMf = self._reduced.data.oMf[fid]

            # 6D 误差：log6(oMf^{-1} * target)，在 LOCAL 帧下计算。
            # 相比 log3(R_cur @ R_tgt^T)，此方式严格与 LOCAL 帧雅可比自洽，
            # 且对 180° 附近的大角度旋转在数值上更稳定（无 sin(θ)→0 奇点问题）。
            err6 = pin.log6(oMf.actInv(target_se3)).vector  # shape (6,)

            # 收敛检查：位置误差用世界坐标（直觉上更清晰），旋转误差用 log6 角速度模长
            _pos_err = float(np.linalg.norm(oMf.translation - target_t))
            _rot_err = float(np.linalg.norm(err6[3:]))

            if _pos_err < cfg.pos_eps and _rot_err < cfg.rot_eps:
                success = True
                break

            err = err6 * w_diag

            # 雅可比（LOCAL 帧，与 log6 保持同一坐标系）
            J_full = pin.computeFrameJacobian(
                self._reduced.model,
                self._reduced.data,
                q,
                fid,
                pin.LOCAL,
            )
            J = J_full * w_diag[:, None]  # 按误差权重缩放雅可比行

            # DLS 伪逆：J^T (JJ^T + λI)^{-1}
            JJt = J @ J.T
            JJt[np.diag_indices_from(JJt)] += cfg.damp
            J_pinv = J.T @ np.linalg.solve(JJt, np.eye(6))  # (nq, 6)

            # 主任务增量
            dq_primary = J_pinv @ err

            # 零空间投影矩阵：I - J^† J
            N = np.eye(self.dof) - J_pinv @ J

            # 次级任务梯度（沿梯度下降方向取反）
            g_sec = self._secondary_gradient(q, q_seed)
            dq_secondary = -N @ g_sec

            # 积分
            dq = dq_primary + dq_secondary
            q = pin.integrate(self._reduced.model, q, dq * cfg.dt)

            # 硬关节限位（pin 顺序的限位数组）
            q = np.clip(
                q,
                self._reduced.model.lowerPositionLimit,
                self._reduced.model.upperPositionLimit,
            )

        self._last_q = q.copy()  # 内部以 pin 顺序存储

        self.last_solve_info = {
            "success": success,
            "iterations": _iter + 1,
            "pos_err": _pos_err,
            "rot_err": _rot_err,
        }

        # 输出转换为 cfg 顺序
        q_cfg = q[self._pin_to_cfg]
        return (
            torch.tensor([success], dtype=torch.bool),
            torch.from_numpy(q_cfg).to(dtype=torch.float32),
        )

    def get_fk(
        self,
        qpos: "torch.Tensor | np.ndarray",
        ee_name: "str | None" = None,
        **kwargs,
    ) -> torch.Tensor:
        """计算指定末端执行器的正运动学。

        Args:
            qpos:    关节角，shape ``(nq,)``。
            ee_name: 末端执行器名称，若为 None 则使用 ``active_ee``。

        Returns:
            torch.Tensor: 末端位姿，shape ``(4, 4)``。
        """
        pin = self._pin
        ee = ee_name or self._active_ee
        if ee is None:
            raise ValueError(
                "[WholeBodyIKSolver] ee_name not specified and no active_ee set."
            )

        if isinstance(qpos, torch.Tensor):
            qpos = qpos.detach().cpu().numpy()
        qpos_cfg = np.asarray(qpos, dtype=float).flatten()
        # 输入为 cfg 顺序 → 转换为 pinocchio 内部顺序
        qpos = qpos_cfg[self._cfg_to_pin]

        pin.framesForwardKinematics(self._reduced.model, self._reduced.data, qpos)
        fid = self._ee_frame_ids[ee]
        oMf = self._reduced.data.oMf[fid]

        mat = np.eye(4)
        mat[:3, :3] = oMf.rotation
        mat[:3, 3] = oMf.translation
        return torch.from_numpy(mat).to(dtype=torch.float32)

    def get_all_ee_names(self) -> List[str]:
        """返回所有已配置的末端执行器名称列表。"""
        return list(self._ee_frame_ids.keys())
