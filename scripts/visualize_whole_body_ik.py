"""
全身IK可视化脚本：在仿真窗口中展示 WholeBodyIKSolver 的全身联动效果。

三个演示场景，目标点均在合理工作域内，机器人只做温和的躯干配合，不出现极端蹲姿或大幅前倾：
  1. 正常手臂运动（基线）   ── 前伸 +0.18 m，高度略降 0.05 m，仅手臂动
  2. 低位目标（轻度屈膝）   ── 高度降低约 0.30 m，膝/髋温和弯曲
  3. 前方目标（腰部温和前倾）── 在手臂伸展极限基础上再向前 ~0.10 m，腰略微前倾

用法：
  python scripts/visualize_whole_body_ik.py
"""

import time
import numpy as np
import torch
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import MarkerCfg
from embodichain.lab.sim.robots.dexforce_w1 import DexforceW1Cfg
from embodichain.lab.sim.solvers import WholeBodyIKSolverCfg, EndEffectorCfg

# ── 路径 ──────────────────────────────────────────────────────────────────────
URDF_PATH = (
    "/home/dex/.cache/embodichain_data/extract/DexforceW1V021/DexforceW1_v02_1.urdf"
)
URDF_DIR = "/home/dex/.cache/embodichain_data/extract/DexforceW1V021"

# ── IK 求解器配置 ─────────────────────────────────────────────────────────────
# 关键：躯干关节的正则化权重大幅降低（joint_reg_extra 为负值），
# 否则 solver 会把躯干拉回零位，无法展示全身联动。
ik_cfg = WholeBodyIKSolverCfg(
    urdf_path=URDF_PATH,
    urdf_dir=URDF_DIR,
    # 顺序与 sim "full_body" 控制部件完全对齐
    joint_names=[
        "ANKLE",
        "KNEE",
        "BUTTOCK",
        "WAIST",  # [0:4]  torso
        "LEFT_J1",
        "LEFT_J2",
        "LEFT_J3",
        "LEFT_J4",
        "LEFT_J5",
        "LEFT_J6",
        "LEFT_J7",  # [4:11]
        "RIGHT_J1",
        "RIGHT_J2",
        "RIGHT_J3",
        "RIGHT_J4",
        "RIGHT_J5",
        "RIGHT_J6",
        "RIGHT_J7",  # [11:18]
    ],
    end_effectors={
        "left": EndEffectorCfg(
            parent_joint="LEFT_J7",
            rotation=[[0, 1, 0], [0.707, 0, 0.707], [0.707, 0, -0.707]],
            translation=[-0.15, 0, 0],
            w_pos=1.0,
            w_rot=0.1,
        ),
        "right": EndEffectorCfg(
            parent_joint="RIGHT_J7",
            rotation=[[0, 1, 0], [0.707, 0, 0.707], [0.707, 0, -0.707]],
            translation=[-0.15, 0, 0],
            w_pos=1.0,
            w_rot=0.1,
        ),
    },
    active_ee="left",
    max_iterations=500,
    dt=0.3,
    pos_eps=3e-3,
    rot_eps=0.05,
    leg_costs_mode2=[],
)


def check_ik_quality(ik_solver) -> bool:
    """打印上一次 get_ik 调用的质量报告（直接读 solver.last_solve_info）。

    返回是否通过（位置误差 < 10 mm 且旋转误差 < 5°）。
    """
    info = ik_solver.last_solve_info
    converged = info["success"]
    pos_err_mm = info["pos_err"] * 1000
    rot_err_deg = np.degrees(info["rot_err"])
    iters = info["iterations"]

    pos_ok = pos_err_mm < 10.0
    rot_ok = rot_err_deg < 5.0
    overall = pos_ok and rot_ok  # 以实际误差为准，不依赖 solver 内部收敛标志

    status = "✓ 合格" if overall else "✗ 不合格"
    print(
        f"  [{status}]  收敛={converged}  迭代={iters}"
        f"  位置误差={pos_err_mm:.1f} mm  旋转误差={rot_err_deg:.1f}°"
    )
    if not converged:
        # success=False 仅说明 solver 在 max_iterations 内未达到 pos_eps/rot_eps 阈值，
        # 不代表解不可用——若实际误差已经满足使用要求则可正常使用。
        print(
            "    ↳ solver 提前停止（未达到 pos_eps/rot_eps），但实际误差可能已满足需求"
        )
    if not pos_ok:
        print(f"    ↳ 位置误差过大（{pos_err_mm:.1f} mm > 10 mm），目标可能超出工作域")
    if not rot_ok:
        print(
            f"    ↳ 旋转误差过大（{rot_err_deg:.1f}° > 5°），可增大 w_rot 或放宽 rot_eps"
        )
    return overall


def print_joint_changes(
    q_before: np.ndarray,
    q_after: np.ndarray,
    joint_names: list,
    threshold: float = 0.05,
):
    """打印变化量超过阈值的关节，区分躯干和手臂。"""
    diff = np.abs(q_after - q_before)
    moved = [
        (joint_names[i], diff[i], q_before[i], q_after[i])
        for i in range(len(joint_names))
        if diff[i] > threshold
    ]

    body_joints = {"ANKLE", "KNEE", "BUTTOCK", "WAIST"}
    body_moved = [x for x in moved if x[0] in body_joints]
    arm_moved = [x for x in moved if x[0] not in body_joints]

    if body_moved:
        print("  🦿 躯干关节（全身IK联动）：")
        for name, d, b, a in body_moved:
            print(f"     {name:10s}  {b:+.3f} → {a:+.3f}  (Δ={d:.3f} rad)")
    else:
        print("  ── 躯干关节未明显变化（目标在手臂工作域内）")

    if arm_moved:
        print("  💪 手臂关节：")
        for name, d, b, a in arm_moved:
            print(f"     {name:10s}  {b:+.3f} → {a:+.3f}  (Δ={d:.3f} rad)")


def set_robot_full_body(robot, q: np.ndarray, device: str) -> None:
    """将 IK 输出（18 维，无 NECK）扩展为 full_body 的 20 维后写入仿真。

    full_body 顺序：torso(0:4) + head(4:6) + left_arm(6:13) + right_arm(13:20)
    IK 顺序：      torso(0:4)             + left_arm(4:11)  + right_arm(11:18)

    同时设置 target=True 和 target=False，防止高刚度躯干关节被 PD 控制器拉回零位。
    """
    q_full = np.zeros(20)
    q_full[0:4] = q[0:4]  # ANKLE KNEE BUTTOCK WAIST
    q_full[4:6] = 0.0  # NECK1 NECK2 锁定为 0
    q_full[6:13] = q[4:11]  # LEFT_J1..J7
    q_full[13:20] = q[11:18]  # RIGHT_J1..J7
    qpos = torch.tensor(q_full, dtype=torch.float32, device=device).unsqueeze(0)
    robot.set_qpos(qpos, name="full_body", target=True)
    robot.set_qpos(qpos, name="full_body", target=False)


def smooth_move(
    robot,
    sim,
    q_from: np.ndarray,
    q_to: np.ndarray,
    n_frames: int = 80,
    fps: float = 30.0,
) -> None:
    device = str(sim.device)
    for i in range(n_frames + 1):
        alpha = i / n_frames
        alpha = alpha * alpha * (3 - 2 * alpha)  # smoothstep
        q = q_from + alpha * (q_to - q_from)
        set_robot_full_body(robot, q, device)
        sim.update(step=1)
        time.sleep(1.0 / fps)


def main():
    print("初始化 IK solver...")
    ik_solver = ik_cfg.init_solver()

    # 打印每个末端执行器的运动链，直观查看 IK 从哪条路径求解
    print("\n── 运动链（URDF 拓扑结构）────────────────────────────────────────")
    print(ik_solver.describe_chain())
    print("────────────────────────────────────────────────────────────────\n")

    # ── 用给定关节角配置计算 FK 目标（与 visualize_origin.py 相同的方式）────
    # 关节顺序与 cfg.joint_names 对齐：
    #   [0:4]  ANKLE KNEE BUTTOCK WAIST  (torso)
    #   [4:6]  NECK1 NECK2               (head)
    #   [6:13] LEFT_J1 ~ LEFT_J7         (left_arm)
    #   [13:20] RIGHT_J1 ~ RIGHT_J7      (right_arm)
    TEST_CASES = [
        {
            "title": "场景1 · 左手（手臂弯曲）",
            "ee": "left",
            "q": np.array(
                [
                    0,
                    0,
                    0,
                    0,  # torso
                    0,
                    0,  # head
                    0.3,
                    -0.5,
                    0.2,
                    -0.8,
                    0.1,
                    0.0,
                    0.3,  # left_arm
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                dtype=float,
            ),  # right_arm
        },
        {
            "title": "场景2 · 右手（手臂弯曲）",
            "ee": "right",
            "q": np.array(
                [
                    0,
                    0,
                    0,
                    0,  # torso
                    0,
                    0,  # head
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,  # left_arm
                    0.3,
                    0.5,
                    -0.2,
                    0.8,
                    -0.1,
                    0.0,
                    -0.3,
                ],
                dtype=float,
            ),  # right_arm
        },
    ]

    # FK 计算每个场景的目标位姿（目标由 q 决定，seed 在运行时取 q_current）
    SCENARIOS = []
    for tc in TEST_CASES:
        target_pose = ik_solver.get_fk(
            torch.from_numpy(tc["q"]).float(), ee_name=tc["ee"]
        ).numpy()
        SCENARIOS.append(
            {
                "title": tc["title"],
                "ee": tc["ee"],
                "target": target_pose,
            }
        )
        print(f"  {tc['title']:20s}  EE 位置: {target_pose[:3, 3]}")

    # ── 初始化仿真 ───────────────────────────────────────────────────────────
    print("\n启动仿真窗口...")
    sim = SimulationManager(
        SimulationManagerCfg(headless=False, sim_device="cpu", num_envs=1)
    )
    sim.set_manual_update(True)
    robot_cfg = DexforceW1Cfg.from_dict(
        {"version": "v021", "arm_kind": "anthropomorphic"}
    )
    robot = sim.add_robot(cfg=robot_cfg)
    for _ in range(20):
        sim.update(step=1)

    q_current = np.zeros(ik_solver.dof)

    # ── 逐场景运行 ───────────────────────────────────────────────────────────
    MARKER_NAME = "target_ee"
    for idx, sc in enumerate(SCENARIOS):
        print(f"\n{'='*55}")
        print(f"  {sc['title']}")
        print(f"  目标位置: {sc['target'][:3, 3]}")
        print(f"{'='*55}")

        # 显示目标坐标系 marker（先清除上一个）
        sim.remove_marker(MARKER_NAME)
        sim.draw_marker(
            MarkerCfg(
                name=MARKER_NAME,
                marker_type="axis",
                axis_xpos=sc["target"],
                axis_size=0.008,
                axis_len=0.12,
            )
        )
        sim.update(step=1)

        # IK 求解（以当前机器人状态为 seed）
        success, q_ik = ik_solver.get_ik(
            sc["target"],
            qpos_seed=torch.from_numpy(q_current).float(),
            active_ee=sc["ee"],
        )
        q_ik_np = q_ik.numpy()

        # IK 质量检查（直接读 solver 内部诊断信息）
        check_ik_quality(ik_solver)

        # 打印关节变化（全身联动核心展示）
        print_joint_changes(q_current, q_ik_np, ik_solver.joint_names)

        # 仿真中平滑运动
        print("  → 运动中...", end="", flush=True)
        smooth_move(robot, sim, q_from=q_current, q_to=q_ik_np)
        print(" 完成")

        q_current = q_ik_np.copy()
        input("  [按 Enter 继续下一场景]")

    # ── 回到零位 ─────────────────────────────────────────────────────────────
    sim.remove_marker(MARKER_NAME)
    print("\n回到零位...")
    smooth_move(robot, sim, q_from=q_current, q_to=np.zeros(ik_solver.dof))

    print("\n所有场景完成。按 Ctrl+C 退出。")
    q_zero = np.zeros(ik_solver.dof)
    try:
        while True:
            set_robot_full_body(robot, q_zero, str(sim.device))
            sim.update(step=1)
            time.sleep(1 / 30)
    except KeyboardInterrupt:
        pass

    sim.destroy()


if __name__ == "__main__":
    main()
