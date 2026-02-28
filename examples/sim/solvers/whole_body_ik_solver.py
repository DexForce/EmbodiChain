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
import time
import numpy as np
import torch

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import MarkerCfg
from embodichain.lab.sim.robots.dexforce_w1 import DexforceW1Cfg


def main():
    np.set_printoptions(precision=5, suppress=True)
    torch.set_printoptions(precision=5, sci_mode=False)

    # ── Initialize simulation ───────────────────────────────────────────────────
    # FK targets must be computed after sim is ready (robot.compute_fk uses get_link_pose)
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

    # ── Inspect kinematic chain ─────────────────────────────────────────────────
    # left_arm_body / right_arm_body each: torso (4 DOF) + arm (7 DOF)
    print(robot._solvers["left_arm_body"].describe_chain())

    # ── Generate IK targets via forward kinematics ─────────────────────────────
    # joint order for left_arm_body:  torso(0:4) + left_arm(4:11)
    # joint order for right_arm_body: torso(0:4) + right_arm(4:11)
    qpos_fk_left = torch.tensor(
        [[0, 0, 0, 0, 0.3, -0.5, 0.2, -0.8, 0.1, 0.0, 0.3]], dtype=torch.float32
    )
    qpos_fk_right = torch.tensor(
        [[0, 0, 0, 0, 0.3, 0.5, -0.2, 0.8, -0.1, 0.0, -0.3]], dtype=torch.float32
    )

    # robot.compute_fk returns (1, 4, 4) homogeneous transform in world frame
    t0 = time.perf_counter()
    target_left = robot.compute_fk(qpos_fk_left, name="left_arm_body", to_matrix=True)
    target_right = robot.compute_fk(
        qpos_fk_right, name="right_arm_body", to_matrix=True
    )
    fk_time_ms = (time.perf_counter() - t0) * 1000
    print(f"[Timing] FK (left+right targets): {fk_time_ms:.3f} ms")

    print(f"Left hand target position: {target_left[0, :3, 3].numpy()}")
    print(f"Right hand target position: {target_right[0, :3, 3].numpy()}")

    # ── Solve IK ──────────────────────────────────────────────────────────────
    # robot.compute_ik transforms target from world to solver.root_link_name frame
    qpos_seed_left = robot.get_qpos(name="left_arm_body")  # (1, 11)
    t0 = time.perf_counter()
    res_left, ik_left = robot.compute_ik(
        pose=target_left, name="left_arm_body", joint_seed=qpos_seed_left
    )
    ik_left_time_ms = (time.perf_counter() - t0) * 1000
    solver_left = robot._solvers["left_arm_body"]
    info = solver_left.last_solve_info
    print(
        f"Left hand IK: success={res_left[0].item()}  "
        f"pos_err={info['pos_err']*1000:.2f} mm  "
        f"rot_err={np.degrees(info['rot_err']):.1f} deg  "
        f"iters={info['iterations']}"
    )

    qpos_seed_right = robot.get_qpos(name="right_arm_body")  # (1, 11)
    t0 = time.perf_counter()
    res_right, ik_right = robot.compute_ik(
        pose=target_right, name="right_arm_body", joint_seed=qpos_seed_right
    )
    ik_right_time_ms = (time.perf_counter() - t0) * 1000
    solver_right = robot._solvers["right_arm_body"]
    info = solver_right.last_solve_info
    print(
        f"Right hand IK: success={res_right[0].item()}  "
        f"pos_err={info['pos_err']*1000:.2f} mm  "
        f"rot_err={np.degrees(info['rot_err']):.1f} deg  "
        f"iters={info['iterations']}"
    )

    # ── Verify: FK(IK result) matches target ───────────────────────────────────
    t0 = time.perf_counter()
    fk_check_left = robot.compute_fk(ik_left, name="left_arm_body", to_matrix=True)
    fk_check_right = robot.compute_fk(ik_right, name="right_arm_body", to_matrix=True)
    fk_check_time_ms = (time.perf_counter() - t0) * 1000
    print(
        f"[Timing] IK left: {ik_left_time_ms:.3f} ms  IK right: {ik_right_time_ms:.3f} ms"
    )
    print(f"[Timing] FK (verify left+right): {fk_check_time_ms:.3f} ms")
    print(f"Left hand FK(IK result): {fk_check_left[0, :3, 3].numpy()}")
    print(f"Right hand FK(IK result): {fk_check_right[0, :3, 3].numpy()}")

    # ── Draw markers ───────────────────────────────────────────────────────────
    sim.draw_marker(
        MarkerCfg(
            name="target_left",
            marker_type="axis",
            axis_xpos=target_left[0].numpy(),
            axis_size=0.005,
            axis_len=0.1,
        )
    )
    sim.draw_marker(
        MarkerCfg(
            name="target_right",
            marker_type="axis",
            axis_xpos=target_right[0].numpy(),
            axis_size=0.005,
            axis_len=0.1,
        )
    )

    # ── Move to left hand target first, then right hand ────────────────────────
    def smooth_move(q_start, q_end, part, n_frames=120):
        for i in range(n_frames + 1):
            alpha = i / n_frames
            alpha = alpha * alpha * (3 - 2 * alpha)  # smoothstep ease-in-out
            q = q_start + alpha * (q_end - q_start)
            robot.set_qpos(q.unsqueeze(0), name=part)
            sim.update(step=1)
            time.sleep(1 / 30)

    print("Moving left hand...")
    smooth_move(robot.get_qpos(name="left_arm_body")[0], ik_left[0], "left_arm_body")

    print("Moving right hand...")
    smooth_move(robot.get_qpos(name="right_arm_body")[0], ik_right[0], "right_arm_body")

    print("Done. Press Ctrl+C to exit.")
    try:
        while True:
            sim.update(step=1)
            time.sleep(1 / 30)
    except KeyboardInterrupt:
        pass

    sim.destroy()


if __name__ == "__main__":
    main()
