"""
全身IK可视化脚本：在仿真窗口中展示 WholeBodyIKSolver 的全身联动效果。

用 robot.compute_fk / robot.compute_ik 接口，与其他 solver 示例完全一致。
求解器由 DexforceW1Cfg._build_default_solver_cfg 自动配置，control part 名
"left_arm_body" / "right_arm_body" 即对应全身IK实例。

用法：
  python scripts/visualize_whole_body_ik_robot.py
"""

import time
import numpy as np
import torch
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import MarkerCfg
from embodichain.lab.sim.robots.dexforce_w1 import DexforceW1Cfg

TORSO_JOINTS = ["ANKLE", "KNEE", "BUTTOCK", "WAIST"]


def check_ik_quality(solver) -> bool:
    info = solver.last_solve_info
    pos_err_mm = info["pos_err"] * 1000
    rot_err_deg = np.degrees(info["rot_err"])
    iters = info["iterations"]
    overall = pos_err_mm < 10.0 and rot_err_deg < 5.0

    status = "✓ 合格" if overall else "✗ 不合格"
    print(
        f"  [{status}]  收敛={info['success']}  迭代={iters}"
        f"  位置误差={pos_err_mm:.1f} mm  旋转误差={rot_err_deg:.1f}°"
    )
    return overall


def print_joint_changes(
    q_before: np.ndarray,
    q_after: np.ndarray,
    joint_names: list,
    threshold: float = 0.05,
):
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


def smooth_move(
    robot,
    sim,
    part: str,
    q_from: torch.Tensor,
    q_to: torch.Tensor,
    n_frames: int = 80,
    fps: float = 30.0,
) -> None:
    for i in range(n_frames + 1):
        alpha = i / n_frames
        alpha = alpha * alpha * (3 - 2 * alpha)  # smoothstep
        q = q_from + alpha * (q_to - q_from)
        robot.set_qpos(q.unsqueeze(0), name=part, target=True)
        robot.set_qpos(q.unsqueeze(0), name=part, target=False)
        sim.update(step=1)
        time.sleep(1.0 / fps)


def main():
    # ── 初始化仿真（先于FK计算，因为坐标变换依赖 get_link_pose）──────────────
    print("启动仿真窗口...")
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

    # ── 打印运动链 ────────────────────────────────────────────────────────────
    print("\n── 左臂运动链 ──────────────────────────────────────────────────────────")
    print(robot._solvers["left_arm_body"].describe_chain())
    print("\n── 右臂运动链 ──────────────────────────────────────────────────────────")
    print(robot._solvers["right_arm_body"].describe_chain())
    print("────────────────────────────────────────────────────────────────\n")

    # ── FK 目标（q 以 left/right_arm_body 顺序：torso(0:4) + arm(4:11)）────────
    # robot.compute_fk 输入 (1, dof) 张量，输出世界坐标系下的 (1, 4, 4)
    TEST_CASES = [
        {
            "title": "场景1 · 左手（手臂弯曲）",
            "part": "left_arm_body",
            "q": torch.tensor(
                [
                    [0, 0, 0, 0, 0.3, -0.5, 0.2, -0.8, 0.1, 0.0, 0.3]  # torso
                ],  # left_arm
                dtype=torch.float32,
            ),
        },
        {
            "title": "场景2 · 右手（手臂弯曲）",
            "part": "right_arm_body",
            "q": torch.tensor(
                [
                    [0, 0, 0, 0, 0.3, 0.5, -0.2, 0.8, -0.1, 0.0, -0.3]  # torso
                ],  # right_arm
                dtype=torch.float32,
            ),
        },
    ]

    SCENARIOS = []
    for tc in TEST_CASES:
        target_pose = robot.compute_fk(tc["q"], name=tc["part"], to_matrix=True)
        SCENARIOS.append(
            {"title": tc["title"], "part": tc["part"], "target": target_pose}
        )
        print(f"  {tc['title']:20s}  EE 位置: {target_pose[0, :3, 3].numpy()}")

    # ── 逐场景运行 ───────────────────────────────────────────────────────────
    MARKER_NAME = "target_ee"
    for sc in SCENARIOS:
        part = sc["part"]
        solver = robot._solvers[part]

        print(f"\n{'='*55}")
        print(f"  {sc['title']}")
        print(f"  目标位置: {sc['target'][0, :3, 3].numpy()}")
        print(f"{'='*55}")

        sim.remove_marker(MARKER_NAME)
        sim.draw_marker(
            MarkerCfg(
                name=MARKER_NAME,
                marker_type="axis",
                axis_xpos=sc["target"][0].numpy(),
                axis_size=0.008,
                axis_len=0.12,
            )
        )
        sim.update(step=1)

        # 以当前关节角为 seed（robot.get_qpos 直接按 control_part 顺序返回）
        q_seed = robot.get_qpos(name=part)  # (1, dof)

        success, q_ik = robot.compute_ik(
            pose=sc["target"], name=part, joint_seed=q_seed
        )
        check_ik_quality(solver)
        print_joint_changes(q_seed[0].numpy(), q_ik[0].numpy(), solver.joint_names)

        print("  → 运动中...", end="", flush=True)
        smooth_move(robot, sim, part, q_seed[0], q_ik[0])
        print(" 完成")

        input("  [按 Enter 继续下一场景]")

    # ── 回到零位 ─────────────────────────────────────────────────────────────
    sim.remove_marker(MARKER_NAME)
    print("\n回到零位...")
    for part in ["left_arm_body", "right_arm_body"]:
        q_cur = robot.get_qpos(name=part)[0]
        smooth_move(robot, sim, part, q_cur, torch.zeros_like(q_cur), n_frames=60)

    print("\n所有场景完成。按 Ctrl+C 退出。")
    try:
        while True:
            sim.update(step=1)
            time.sleep(1 / 30)
    except KeyboardInterrupt:
        pass

    sim.destroy()


if __name__ == "__main__":
    main()
