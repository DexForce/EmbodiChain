"""
全身IK可视化脚本：在仿真窗口中直观展示 IK 求解结果。

每个 test case 流程：
  1. 根据给定关节角 q_test 用 FK 求出目标末端位姿
  2. 用 IK 反求关节角 q_ik
  3. 在仿真窗口中平滑地把机器人从零配置插值运动到 q_ik

用法：
  python scripts/visualize_whole_body_ik.py
"""

import time
import numpy as np
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.robots.dexforce_w1 import DexforceW1Cfg
from embodichain.lab.sim.solvers import WholeBodyIKSolverCfg, EndEffectorCfg, LegCostCfg

# ── IK 求解器配置（与 test_whole_body_ik.py 完全相同）──────────────────────
URDF_PATH = (
    "/home/dex/.cache/embodichain_data/extract/DexforceW1V021/DexforceW1_v02_1.urdf"
)
URDF_DIR = "/home/dex/.cache/embodichain_data/extract/DexforceW1V021"

ik_cfg = WholeBodyIKSolverCfg(
    urdf_path=URDF_PATH,
    urdf_dir=URDF_DIR,
    joint_names=[
        "ANKLE",
        "KNEE",
        "BUTTOCK",
        "WAIST",
        "NECK1",
        "NECK2",
        "LEFT_J1",
        "LEFT_J2",
        "LEFT_J3",
        "LEFT_J4",
        "LEFT_J5",
        "LEFT_J6",
        "LEFT_J7",
        "RIGHT_J1",
        "RIGHT_J2",
        "RIGHT_J3",
        "RIGHT_J4",
        "RIGHT_J5",
        "RIGHT_J6",
        "RIGHT_J7",
    ],
    end_effectors={
        "left": EndEffectorCfg(
            parent_joint="LEFT_J7",
            rotation=[[0, 1, 0], [0.707, 0, 0.707], [0.707, 0, -0.707]],
            translation=[-0.15, 0, 0],
        ),
        "right": EndEffectorCfg(
            parent_joint="RIGHT_J7",
            rotation=[[0, 1, 0], [0.707, 0, 0.707], [0.707, 0, -0.707]],
            translation=[-0.15, 0, 0],
        ),
        "head": EndEffectorCfg(
            parent_joint="NECK2",
            rotation=[[0, 0, -1], [0, -1, 0], [-1, 0, 0]],
            translation=[0, -0.1, 0],
        ),
    },
    active_ee="left",
    leg_costs_mode2=[
        LegCostCfg(
            joint_names=["WAIST", "KNEE"], coefficients=[1.0, -1.0], weight=10.0
        ),
        LegCostCfg(
            joint_names=["WAIST", "BUTTOCK", "KNEE"],
            coefficients=[1.0, 1.0, 1.0],
            weight=10.0,
        ),
    ],
)

# ── IK 测试用例（与 test_whole_body_ik.py 相同）────────────────────────────
# IK solver 关节顺序：
# [0]ANKLE [1]KNEE [2]BUTTOCK [3]WAIST
# [4]LEFT_J1 ... [10]LEFT_J7
# [11]NECK1 [12]NECK2
# [13]RIGHT_J1 ... [19]RIGHT_J7

TEST_CASES = [
    {
        "name": "左手（零配置）",
        "ee": "left",
        "q": np.zeros(20),
    },
    {
        "name": "左手（手臂弯曲）",
        "ee": "left",
        "q": np.array(
            [
                0,
                0,
                0,
                0,
                0.3,
                -0.5,
                0.2,
                -0.8,
                0.1,
                0.0,
                0.3,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            dtype=float,
        ),
    },
    {
        "name": "右手（手臂弯曲）",
        "ee": "right",
        "q": np.array(
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0.3,
                0.5,
                -0.2,
                0.8,
                -0.1,
                0.0,
                -0.3,
            ],
            dtype=float,
        ),
    },
    {
        "name": "头部（颈部偏转）",
        "ee": "head",
        "q": np.array(
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.2, 0.1, 0, 0, 0, 0, 0, 0, 0],
            dtype=float,
        ),
    },
]


def set_robot_full_body(robot, q_ik: np.ndarray, device: str) -> None:
    """将 IK 解（20维）分发到仿真机器人的各控制部件。

    IK solver 关节顺序 → 控制部件映射：
      q[0:4]   → "torso"     (ANKLE, KNEE, BUTTOCK, WAIST)
      q[4:11]  → "left_arm"  (LEFT_J1 ~ LEFT_J7)
      q[11:13] → "head"      (NECK1, NECK2)
      q[13:20] → "right_arm" (RIGHT_J1 ~ RIGHT_J7)
    """

    def to_tensor(arr):
        return torch.tensor(arr, dtype=torch.float32, device=device).unsqueeze(0)

    robot.set_qpos(to_tensor(q_ik[0:4]), name="torso", target=False)
    robot.set_qpos(to_tensor(q_ik[4:11]), name="left_arm", target=False)
    robot.set_qpos(to_tensor(q_ik[11:13]), name="head", target=False)
    robot.set_qpos(to_tensor(q_ik[13:20]), name="right_arm", target=False)


def smooth_move(
    robot,
    sim,
    q_from: np.ndarray,
    q_to: np.ndarray,
    n_frames: int = 60,
    fps: float = 30.0,
) -> None:
    """在 n_frames 步内把机器人从 q_from 平滑插值运动到 q_to。"""
    device = str(sim.device)
    dt = 1.0 / fps
    for i in range(n_frames + 1):
        alpha = i / n_frames
        # 使用 smoothstep 曲线让运动更自然
        alpha = alpha * alpha * (3 - 2 * alpha)
        q_interp = q_from + alpha * (q_to - q_from)
        set_robot_full_body(robot, q_interp, device)
        sim.update(step=1)
        time.sleep(dt)


def main():
    print("=" * 55)
    print("  WholeBodyIK 可视化（关闭窗口退出）")
    print("=" * 55)

    # ── 初始化仿真（有窗口模式）──────────────────────────────────────────
    print("\n[1/3] 初始化仿真场景...")
    sim_cfg = SimulationManagerCfg(headless=False, sim_device="cpu", num_envs=1)
    sim = SimulationManager(sim_cfg)
    sim.set_manual_update(True)

    # ── 加载机器人 ────────────────────────────────────────────────────────
    print("[2/3] 加载 DexforceW1 机器人（anthropomorphic 手臂）...")
    robot_cfg = DexforceW1Cfg.from_dict(
        {
            "version": "v021",
            "arm_kind": "anthropomorphic",
        }
    )
    robot = sim.add_robot(cfg=robot_cfg)

    # 先让仿真稳定几帧
    for _ in range(20):
        sim.update(step=1)

    # ── 初始化 IK 求解器 ──────────────────────────────────────────────────
    print("[3/3] 初始化 WholeBodyIKSolver...")
    ik_solver = ik_cfg.init_solver()
    print(f"      关节数 nq={ik_solver.dof}\n")

    # ── 当前机器人关节角（全零） ───────────────────────────────────────────
    q_current = np.zeros(20)

    # ── 逐个运行测试用例 ───────────────────────────────────────────────────
    for idx, tc in enumerate(TEST_CASES):
        q_test = tc["q"]
        ee = tc["ee"]
        name = tc["name"]

        # Step 1: FK 求目标位姿
        target_pose = ik_solver.get_fk(torch.from_numpy(q_test).float(), ee_name=ee)
        target_pos = target_pose.numpy()[:3, 3]

        # Step 2: IK 求解
        success, q_ik = ik_solver.get_ik(
            target_pose,
            qpos_seed=torch.from_numpy(q_test).float(),
            active_ee=ee,
        )
        q_ik_np = q_ik.numpy()

        # Step 3: FK 验证误差
        fk_check = ik_solver.get_fk(q_ik, ee_name=ee)
        pos_err = np.linalg.norm(fk_check.numpy()[:3, 3] - target_pos) * 1000

        status = "PASS ✓" if pos_err < 20 else "FAIL ✗"
        print(f"[{idx+1}/{len(TEST_CASES)}] {name}")
        print(
            f"  末端:    {ee}   IK收敛: {success.item()}   位置误差: {pos_err:.2f} mm  [{status}]"
        )
        print(f"  目标位置: {target_pos}")

        # Step 4: 仿真中平滑运动到 IK 解
        print("  → 运动中...", end="", flush=True)
        smooth_move(robot, sim, q_from=q_current, q_to=q_ik_np, n_frames=80)
        print(" 完成")

        # 下一个 case 从当前 IK 解出发
        q_current = q_ik_np.copy()
        print()

    # ── 循环结束，回到零位 ─────────────────────────────────────────────────
    print("所有 test case 完成，回到零位...")
    smooth_move(robot, sim, q_from=q_current, q_to=np.zeros(20), n_frames=80)

    print("\n按 Ctrl+C 退出，或直接关闭仿真窗口。")
    try:
        while True:
            sim.update(step=1)
            time.sleep(1 / 30)
    except KeyboardInterrupt:
        pass

    sim.destroy()
    print("退出。")


if __name__ == "__main__":
    main()
