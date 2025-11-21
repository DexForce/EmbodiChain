# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
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

import torch
import numpy as np
import warp as wp
from scipy.spatial.transform import Rotation
from embodichain.lab.sim.solvers.opw_solver import OPWSolver, OPWSolverCfg
from typing import Tuple, List
import time


def get_pose_err(matrix_a: np.ndarray, matrix_b: np.ndarray) -> Tuple[float, float]:
    t_err = np.linalg.norm(matrix_a[:3, 3] - matrix_b[:3, 3])
    relative_rot = matrix_a[:3, :3].T @ matrix_b[:3, :3]
    cos_angle = (np.trace(relative_rot) - 1) / 2.0
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    r_err = np.arccos(cos_angle)
    return t_err, r_err


def get_poses_err(
    matrix_a_list: List[np.ndarray], matrix_b_list: List[np.ndarray]
) -> Tuple[float, float]:
    t_errs = []
    r_errs = []
    for mat_a, mat_b in zip(matrix_a_list, matrix_b_list):
        t_err, r_err = get_pose_err(mat_a, mat_b)
        t_errs.append(t_err)
        r_errs.append(r_err)
    return np.mean(t_errs), np.mean(r_errs)


def check_opw_solver(solver_warp, solver_py_opw, n_samples=1000):
    DOF = 6
    qpos_np = np.random.uniform(low=-np.pi, high=np.pi, size=(n_samples, DOF)).astype(
        float
    )
    qpos = torch.tensor(qpos_np, device=torch.device("cuda"), dtype=torch.float32)
    xpos = solver_warp.get_fk(qpos)
    qpos_seed = torch.tensor(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        device=torch.device("cuda"),
        dtype=torch.float32,
    )

    warp_ik_start_time = time.time()
    warp_ik_success, warp_ik_qpos = solver_warp.get_ik(
        xpos,
        qpos_seed=qpos_seed,
        initial_guess=qpos,
        # return_all_solutions=True,
    )
    warp_cost_time = time.time() - warp_ik_start_time

    # TODO: debug code
    # warp_ik_success_np = warp_ik_success.cpu().numpy()
    # warp_ik_failure_indices = np.where(warp_ik_success_np == False)[0]
    # if len(warp_ik_failure_indices) > 0:
    #     failure_qpos = qpos_np[warp_ik_failure_indices]
    #     failure_xpos = xpos.cpu().numpy()[warp_ik_failure_indices]
    #     print("=====warp_ik_failure_qpos:\n", repr(failure_qpos))
    #     print("=====warp_ik_failure_xpos:\n", repr(failure_xpos))

    #     print("=====xpos:\n", repr(xpos.cpu().numpy()))
    #     print("=====warp_ik_qpos:\n", repr(warp_ik_qpos.cpu().numpy()))
    #     print("=====warp_ik_success:\n", repr(warp_ik_success.cpu().numpy()))

    check_xpos = solver_warp.get_fk(warp_ik_qpos)
    warp_t_mean_err, warp_r_mean_err = get_poses_err(
        [x.cpu().numpy() for x in xpos],
        [x.cpu().numpy() for x in check_xpos],
    )

    py_opw_ik_start_time = time.time()
    py_opw_ik_success, py_opw_ik_qpos = solver_py_opw.get_ik(
        xpos, qpos_seed=qpos_seed, initial_guess=qpos
    )
    py_opw_cost_time = time.time() - py_opw_ik_start_time

    check_xpos = solver_warp.get_fk(py_opw_ik_qpos.to(torch.device("cuda")))
    py_opw_t_mean_err, py_opw_r_mean_err = get_poses_err(
        [x.cpu().numpy() for x in xpos],
        [x.cpu().numpy() for x in check_xpos],
    )

    return (
        warp_cost_time,
        warp_t_mean_err,
        warp_r_mean_err,
        py_opw_cost_time,
        py_opw_t_mean_err,
        py_opw_r_mean_err,
    )


def benchmark_opw_solver():
    cfg = OPWSolverCfg()
    cfg.a1 = 400.333
    cfg.a2 = -251.449
    cfg.b = 0.0
    cfg.c1 = 830
    cfg.c2 = 1177.556
    cfg.c3 = 1443.593
    cfg.c4 = 230
    cfg.offsets = (
        0.0,
        82.21350356417211 * np.pi / 180.0,
        -167.21710113148163 * np.pi / 180.0,
        0.0,
        0.0,
        0.0,
    )
    cfg.flip_axes = (True, False, True, True, False, True)
    cfg.has_parallelogram = False

    # TODO: ignore pk_serial_chain for OPW
    solver_warp = cfg.init_solver(device=torch.device("cuda"), pk_serial_chain="")
    solver_py_opw = cfg.init_solver(device=torch.device("cpu"), pk_serial_chain="")
    n_samples = [100, 1000, 10000, 100000]
    # n_samples = [100]
    for n_sample in n_samples:
        # check_opw_solver(solver_warp, solver_py_opw, device=device, n_samples=n_sample)
        (
            warp_cost_time,
            warp_t_mean_err,
            warp_r_mean_err,
            py_opw_cost_time,
            py_opw_t_mean_err,
            py_opw_r_mean_err,
        ) = check_opw_solver(solver_warp, solver_py_opw, n_samples=n_sample)
        print(f"===warp OPW Solver FK/IK test over {n_sample} samples:")
        print(f"  Warp IK time: {warp_cost_time * 1000:.6f} ms")
        print(f"Translation mean error: {warp_t_mean_err*1000:.6f} mm")
        print(f"Rotation mean error: {warp_r_mean_err*180/np.pi:.6f} degrees")
        print(f"===Py OPW IK time: {py_opw_cost_time * 1000:.6f} ms")
        print(f"Translation mean error: {py_opw_t_mean_err*1000:.6f} mm")
        print(f"Rotation mean error: {py_opw_r_mean_err*180/np.pi:.6f} degrees")


if __name__ == "__main__":
    benchmark_opw_solver()
