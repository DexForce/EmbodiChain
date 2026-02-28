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
from embodichain.lab.sim.solvers import WholeBodyIKSolverCfg

# ── 路径 ──────────────────────────────────────────────────────────────────────
URDF_PATH = (
    "/home/dex/.cache/embodichain_data/extract/DexforceW1V021/DexforceW1_v02_1.urdf"
)
URDF_DIR = "/home/dex/.cache/embodichain_data/extract/DexforceW1V021"

# ── TCP（与 _build_default_solver_cfg 中 SRSSolver 使用的 TCP 保持同步）────────
_LEFT_TCP = np.array(
    [
        [-1.0, 0.0, 0.0, 0.012],
        [0.0, 0.0, 1.0, 0.0675],
        [0.0, 1.0, 0.0, 0.127],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
_RIGHT_TCP = np.array(
    [
        [1.0, 0.0, 0.0, 0.012],
        [0.0, 0.0, -1.0, -0.0675],
        [0.0, 1.0, 0.0, 0.127],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

# ── 躯干关节 ──────────────────────────────────────────────────────────────────
TORSO_JOINTS = ["ANKLE", "KNEE", "BUTTOCK", "WAIST"]
LEFT_ARM_JOINTS = [f"LEFT_J{i+1}" for i in range(7)]
RIGHT_ARM_JOINTS = [f"RIGHT_J{i+1}" for i in range(7)]

# 两个独立求解器，各管一侧手臂（torso 在前，与 control_parts 顺序对齐）
_COMMON_CFG = dict(
    urdf_path=URDF_PATH,
    urdf_dir=URDF_DIR,
    max_iterations=500,
    dt=0.3,
    pos_eps=3e-3,
    rot_eps=0.05,
    leg_costs_mode2=[],
)

left_solver_cfg = WholeBodyIKSolverCfg(
    **_COMMON_CFG,
    joint_names=TORSO_JOINTS + LEFT_ARM_JOINTS,
    end_link_name="left_ee",
    tcp=_LEFT_TCP,
)
right_solver_cfg = WholeBodyIKSolverCfg(
    **_COMMON_CFG,
    joint_names=TORSO_JOINTS + RIGHT_ARM_JOINTS,
    end_link_name="right_ee",
    tcp=_RIGHT_TCP,
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
    overall = pos_ok and rot_ok

    status = "✓ 合格" if overall else "✗ 不合格"
    print(
        f"  [{status}]  收敛={converged}  迭代={iters}"
        f"  位置误差={pos_err_mm:.1f} mm  旋转误差={rot_err_deg:.1f}°"
    )
    if not converged:
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

    body_joints = set(TORSO_JOINTS)
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


def set_robot_full_body(
    robot, q_torso: np.ndarray, q_left: np.ndarray, q_right: np.ndarray, device: str
) -> None:
    """将分散的关节角合并为 full_body 的 20 维后写入仿真。

    full_body 顺序：torso(0:4) + head(4:6) + left_arm(6:13) + right_arm(13:20)
    同时设置 target=True 和 target=False，防止 PD 控制器拉回零位。
    """
    q_full = np.zeros(20)
    q_full[0:4] = q_torso
    q_full[4:6] = 0.0  # NECK1 NECK2 锁定为 0
    q_full[6:13] = q_left
    q_full[13:20] = q_right
    qpos = torch.tensor(q_full, dtype=torch.float32, device=device).unsqueeze(0)
    robot.set_qpos(qpos, name="full_body", target=True)
    robot.set_qpos(qpos, name="full_body", target=False)


def smooth_move(
    robot,
    sim,
    q_torso_from: np.ndarray,
    q_torso_to: np.ndarray,
    q_left_from: np.ndarray,
    q_left_to: np.ndarray,
    q_right_from: np.ndarray,
    q_right_to: np.ndarray,
    n_frames: int = 80,
    fps: float = 30.0,
) -> None:
    device = str(sim.device)
    for i in range(n_frames + 1):
        alpha = i / n_frames
        alpha = alpha * alpha * (3 - 2 * alpha)  # smoothstep
        t = q_torso_from + alpha * (q_torso_to - q_torso_from)
        l = q_left_from + alpha * (q_left_to - q_left_from)
        r = q_right_from + alpha * (q_right_to - q_right_from)
        set_robot_full_body(robot, t, l, r, device)
        sim.update(step=1)
        time.sleep(1.0 / fps)


def main():
    print("初始化 IK solver...")
    left_solver = left_solver_cfg.init_solver()
    right_solver = right_solver_cfg.init_solver()

    print("\n── 左臂运动链（URDF 拓扑结构）────────────────────────────────────────")
    print(left_solver.describe_chain())
    print("\n── 右臂运动链 ──────────────────────────────────────────────────────────")
    print(right_solver.describe_chain())
    print("────────────────────────────────────────────────────────────────\n")

    # ── FK 配置（q 以 full_body 20-DOF 顺序提供，提取对应子集喂给各 solver）──
    # full_body 顺序：torso(0:4) + head(4:6) + left_arm(6:13) + right_arm(13:20)
    # left_solver  顺序：torso(0:4) + left_arm(4:11)
    # right_solver 顺序：torso(0:4) + right_arm(4:11)

    def q20_to_left(q20: np.ndarray) -> np.ndarray:
        return np.concatenate([q20[0:4], q20[6:13]])

    def q20_to_right(q20: np.ndarray) -> np.ndarray:
        return np.concatenate([q20[0:4], q20[13:20]])

    TEST_CASES = [
        {
            "title": "场景1 · 左手（手臂弯曲）",
            "side": "left",
            "q20": np.array(
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
                ],  # right_arm
                dtype=float,
            ),
        },
        {
            "title": "场景2 · 右手（手臂弯曲）",
            "side": "right",
            "q20": np.array(
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
                ],  # right_arm
                dtype=float,
            ),
        },
    ]

    SCENARIOS = []
    for tc in TEST_CASES:
        if tc["side"] == "left":
            q_sub = q20_to_left(tc["q20"])
            target_pose = left_solver.get_fk(torch.from_numpy(q_sub).float()).numpy()
        else:
            q_sub = q20_to_right(tc["q20"])
            target_pose = right_solver.get_fk(torch.from_numpy(q_sub).float()).numpy()
        SCENARIOS.append(
            {"title": tc["title"], "side": tc["side"], "target": target_pose}
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

    # 全局关节状态（分量存储，便于各 solver 独立更新）
    q_torso = np.zeros(4)
    q_left = np.zeros(7)
    q_right = np.zeros(7)

    # ── 逐场景运行 ───────────────────────────────────────────────────────────
    MARKER_NAME = "target_ee"
    for idx, sc in enumerate(SCENARIOS):
        print(f"\n{'='*55}")
        print(f"  {sc['title']}")
        print(f"  目标位置: {sc['target'][:3, 3]}")
        print(f"{'='*55}")

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

        if sc["side"] == "left":
            q_seed = np.concatenate([q_torso, q_left])
            success, q_ik = left_solver.get_ik(
                sc["target"],
                qpos_seed=torch.from_numpy(q_seed).float(),
            )
            q_ik_np = q_ik.numpy()
            check_ik_quality(left_solver)
            print_joint_changes(q_seed, q_ik_np, left_solver.joint_names)

            q_torso_new = q_ik_np[:4]
            q_left_new = q_ik_np[4:]
            q_right_new = q_right.copy()
        else:
            q_seed = np.concatenate([q_torso, q_right])
            success, q_ik = right_solver.get_ik(
                sc["target"],
                qpos_seed=torch.from_numpy(q_seed).float(),
            )
            q_ik_np = q_ik.numpy()
            check_ik_quality(right_solver)
            print_joint_changes(q_seed, q_ik_np, right_solver.joint_names)

            q_torso_new = q_ik_np[:4]
            q_left_new = q_left.copy()
            q_right_new = q_ik_np[4:]

        print("  → 运动中...", end="", flush=True)
        smooth_move(
            robot,
            sim,
            q_torso,
            q_torso_new,
            q_left,
            q_left_new,
            q_right,
            q_right_new,
        )
        print(" 完成")

        q_torso = q_torso_new
        q_left = q_left_new
        q_right = q_right_new
        input("  [按 Enter 继续下一场景]")

    # ── 回到零位 ─────────────────────────────────────────────────────────────
    sim.remove_marker(MARKER_NAME)
    print("\n回到零位...")
    smooth_move(
        robot,
        sim,
        q_torso,
        np.zeros(4),
        q_left,
        np.zeros(7),
        q_right,
        np.zeros(7),
    )

    print("\n所有场景完成。按 Ctrl+C 退出。")
    try:
        while True:
            set_robot_full_body(
                robot, np.zeros(4), np.zeros(7), np.zeros(7), str(sim.device)
            )
            sim.update(step=1)
            time.sleep(1 / 30)
    except KeyboardInterrupt:
        pass

    sim.destroy()


if __name__ == "__main__":
    main()
