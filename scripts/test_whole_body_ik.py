"""
简单验证脚本：测试 WholeBodyIKSolver 的 IK 计算是否正确。

原理：
  1. 给定一组关节角 q_test
  2. 用 FK 算出末端位姿 target_pose
  3. 用 IK 从 target_pose 反求关节角 q_ik
  4. 再对 q_ik 做 FK，检查末端位置误差是否足够小

用法：
  python scripts/test_whole_body_ik.py
"""

import numpy as np
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from embodichain.lab.sim.solvers import WholeBodyIKSolverCfg, EndEffectorCfg, LegCostCfg

# ── 配置 ──────────────────────────────────────────────────────────────────────
URDF_PATH = (
    "/home/dex/.cache/embodichain_data/extract/DexforceW1V021/DexforceW1_v02_1.urdf"
)
URDF_DIR = "/home/dex/.cache/embodichain_data/extract/DexforceW1V021"

cfg = WholeBodyIKSolverCfg(
    urdf_path=URDF_PATH,
    urdf_dir=URDF_DIR,
    # 顺序与 sim 的 "full_body" 控制部件完全对齐：torso + head + left_arm + right_arm
    joint_names=[
        "ANKLE",
        "KNEE",
        "BUTTOCK",
        "WAIST",  # [0:4]  torso
        "NECK1",
        "NECK2",  # [4:6]  head
        "LEFT_J1",
        "LEFT_J2",
        "LEFT_J3",
        "LEFT_J4",
        "LEFT_J5",
        "LEFT_J6",
        "LEFT_J7",  # [6:13]
        "RIGHT_J1",
        "RIGHT_J2",
        "RIGHT_J3",
        "RIGHT_J4",
        "RIGHT_J5",
        "RIGHT_J6",
        "RIGHT_J7",  # [13:20]
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

print("正在初始化 WholeBodyIKSolver（首次运行需编译 CasADi 计算图，约需几秒）...")
solver = cfg.init_solver()
print(f"初始化完成，关节数 nq={solver.dof}\n")

# ── 测试用例 ──────────────────────────────────────────────────────────────────
# 关节顺序与 sim "full_body" 对齐，也与 cfg.joint_names 一致：
# [0:4]  ANKLE, KNEE, BUTTOCK, WAIST   (torso)
# [4:6]  NECK1, NECK2                  (head)
# [6:13] LEFT_J1 ~ LEFT_J7             (left_arm)
# [13:20] RIGHT_J1 ~ RIGHT_J7          (right_arm)

test_cases = [
    {
        "name": "左手（零配置）",
        "ee": "left",
        "q": np.zeros(solver.dof),
    },
    {
        "name": "左手（手臂弯曲）",
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
        "name": "右手（手臂弯曲）",
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
    {
        "name": "头部（颈部偏转）",
        "ee": "head",
        "q": np.array(
            [
                0,
                0,
                0,
                0,  # torso
                0.2,
                0.1,  # head (NECK1, NECK2)
                0,
                0,
                0,
                0,
                0,
                0,
                0,  # left_arm
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
]

# ── 运行测试 ──────────────────────────────────────────────────────────────────
all_passed = True
for tc in test_cases:
    q_test = tc["q"]
    ee = tc["ee"]
    name = tc["name"]

    # Step 1: FK
    target_pose = solver.get_fk(torch.from_numpy(q_test).float(), ee_name=ee)
    target_pos = target_pose.numpy()[:3, 3]

    # Step 2: IK
    success, q_ik = solver.get_ik(
        target_pose,
        qpos_seed=torch.from_numpy(q_test).float(),
        active_ee=ee,
    )

    # Step 3: FK(IK解) 验证
    fk_check = solver.get_fk(q_ik, ee_name=ee)
    pos_err = np.linalg.norm(fk_check.numpy()[:3, 3] - target_pos)

    passed = pos_err < 0.02
    all_passed = all_passed and passed
    status = "PASS ✓" if passed else "FAIL ✗"

    print(f"[{status}] {name}")
    print(f"         目标位置:   {target_pos}")
    print(f"         FK(IK解):   {fk_check.numpy()[:3, 3]}")
    print(f"         位置误差:   {pos_err*1000:.2f} mm   |   IK收敛: {success.item()}")
    print()

print("─" * 50)
print("全部通过 ✓" if all_passed else "存在失败项，请检查上方输出")
