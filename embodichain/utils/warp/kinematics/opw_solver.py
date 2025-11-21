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

import warp as wp
import numpy as np
from typing import Tuple


wp_vec48f = wp.types.vector(length=48, dtype=float)
wp_vec6f = wp.types.vector(length=6, dtype=float)


@wp.func
def normalize_to_pi(angle: float) -> float:
    angle = (angle + wp.pi) % (2.0 * wp.pi) - wp.pi
    return angle


@wp.func
def safe_acos(x: float) -> float:
    return wp.acos(wp.clamp(x, -1.0, 1.0))


@wp.func
def th4_th6_for_branch(
    i: int,
    r_: wp.mat33f,
    sin1: wp.vec4f,
    cos1: wp.vec4f,
    s23: wp.vec4f,
    c23: wp.vec4f,
) -> Tuple[float, float]:
    th4_y = r_[1, 2] * cos1[i] - r_[0, 2] * sin1[i]
    th4_x = (
        r_[0, 2] * c23[i] * cos1[i] + r_[1, 2] * c23[i] * sin1[i] - r_[2, 2] * s23[i]
    )
    th4 = wp.atan2(th4_y, th4_x)

    th6_y = (
        r_[0, 1] * s23[i] * cos1[i] + r_[1, 1] * s23[i] * sin1[i] + r_[2, 1] * c23[i]
    )
    th6_x = (
        -r_[0, 0] * s23[i] * cos1[i] - r_[1, 0] * s23[i] * sin1[i] - r_[2, 0] * c23[i]
    )
    th6 = wp.atan2(th6_y, th6_x)
    return th4, th6


@wp.struct
class OPWparam:
    a1: float
    a2: float
    b: float
    c1: float
    c2: float
    c3: float
    c4: float


@wp.func
def get_transform_err(
    transform1: wp.mat44f, transform2: wp.mat44f
) -> Tuple[float, float]:
    t_diff = wp.vec3f(
        transform1[0, 3] - transform2[0, 3],
        transform1[1, 3] - transform2[1, 3],
        transform1[2, 3] - transform2[2, 3],
    )
    t_err = wp.length(t_diff)
    r1 = wp.mat33f(
        transform1[0, 0],
        transform1[0, 1],
        transform1[0, 2],
        transform1[1, 0],
        transform1[1, 1],
        transform1[1, 2],
        transform1[2, 0],
        transform1[2, 1],
        transform1[2, 2],
    )
    r2 = wp.mat33f(
        transform2[0, 0],
        transform2[0, 1],
        transform2[0, 2],
        transform2[1, 0],
        transform2[1, 1],
        transform2[1, 2],
        transform2[2, 0],
        transform2[2, 1],
        transform2[2, 2],
    )
    r_diff = wp.transpose(r1) * r2
    cos_value = 0.5 * (wp.trace(r_diff) - 1.0)
    r_err = wp.abs(safe_acos(cos_value))
    return t_err, r_err


@wp.func
def opw_single_fk(
    q1: float, q2: float, q3: float, q4: float, q5: float, q6: float, params: OPWparam
):
    psi3 = wp.atan2(params.a2, params.c3)
    k = wp.sqrt(params.a2 * params.a2 + params.c3 * params.c3)

    # Precompute q23_psi3 for better readability and reuse
    q23_psi3 = q2 + q3 + psi3
    sin_q23_psi3 = wp.sin(q23_psi3)
    cos_q23_psi3 = wp.cos(q23_psi3)

    cx1 = params.c2 * wp.sin(q2) + k * sin_q23_psi3 + params.a1
    cy1 = params.b
    cz1 = params.c2 * wp.cos(q2) + k * cos_q23_psi3

    cx0 = cx1 * wp.cos(q1) - cy1 * wp.sin(q1)
    cy0 = cx1 * wp.sin(q1) + cy1 * wp.cos(q1)
    cz0 = cz1 + params.c1

    s1, c1 = wp.sin(q1), wp.cos(q1)
    s2, c2 = wp.sin(q2), wp.cos(q2)
    s3, c3 = wp.sin(q3), wp.cos(q3)
    s4, c4 = wp.sin(q4), wp.cos(q4)
    s5, c5 = wp.sin(q5), wp.cos(q5)
    s6, c6 = wp.sin(q6), wp.cos(q6)

    r_0c = wp.mat33f(
        c1 * c2 * c3 - c1 * s2 * s3,
        -s1,
        c1 * c2 * s3 + c1 * s2 * c3,
        s1 * c2 * c3 - s1 * s2 * s3,
        c1,
        s1 * c2 * s3 + s1 * s2 * c3,
        -s2 * c3 - c2 * s3,
        0.0,
        -s2 * s3 + c2 * c3,
    )
    r_ce = wp.mat33f(
        c4 * c5 * c6 - s4 * s6,
        -c4 * c5 * s6 - s4 * c6,
        c4 * s5,
        s4 * c5 * c6 + c4 * s6,
        -s4 * c5 * s6 + c4 * c6,
        s4 * s5,
        -s5 * c6,
        s5 * s6,
        c5,
    )

    r_0e = r_0c * r_ce
    t_0e = wp.vec3f(
        cx0 + params.c4 * r_0e[0, 2],
        cy0 + params.c4 * r_0e[1, 2],
        cz0 + params.c4 * r_0e[2, 2],
    )

    return wp.mat44f(
        r_0e[0, 0],
        r_0e[0, 1],
        r_0e[0, 2],
        t_0e[0],
        r_0e[1, 0],
        r_0e[1, 1],
        r_0e[1, 2],
        t_0e[1],
        r_0e[2, 0],
        r_0e[2, 1],
        r_0e[2, 2],
        t_0e[2],
        0.0,
        0.0,
        0.0,
        1.0,
    )


@wp.kernel
def opw_fk_kernel(
    qpos: wp.array(dtype=float),
    ee_pose: wp.mat44f,
    params: OPWparam,
    offsets: wp.array(dtype=float),
    sign_corrections: wp.array(dtype=float),
    xpos: wp.array(dtype=float),
):
    i = wp.tid()
    dof = 6
    q1 = qpos[0 + i * dof] * sign_corrections[0] - offsets[0]
    q2 = qpos[1 + i * dof] * sign_corrections[1] - offsets[1]
    q3 = qpos[2 + i * dof] * sign_corrections[2] - offsets[2]
    q4 = qpos[3 + i * dof] * sign_corrections[3] - offsets[3]
    q5 = qpos[4 + i * dof] * sign_corrections[4] - offsets[4]
    q6 = qpos[5 + i * dof] * sign_corrections[5] - offsets[5]

    p_0e = opw_single_fk(q1, q2, q3, q4, q5, q6, params)
    result = p_0e * ee_pose

    # assign to result
    for t in range(16):
        xpos[t + i * 16] = result[t // 4, t % 4]


@wp.kernel
def opw_ik_kernel(
    xpos: wp.array(dtype=float),
    ee_pose_inv: wp.mat44f,
    params: OPWparam,
    offsets: wp.array(dtype=float),
    sign_corrections: wp.array(dtype=float),
    qpos: wp.array(dtype=float),
    ik_valid: wp.array(dtype=int),
):
    i = wp.tid()
    # TODO: warp slice ?
    ee_pose = (
        wp.mat44f(
            xpos[i * 16 + 0],
            xpos[i * 16 + 1],
            xpos[i * 16 + 2],
            xpos[i * 16 + 3],
            xpos[i * 16 + 4],
            xpos[i * 16 + 5],
            xpos[i * 16 + 6],
            xpos[i * 16 + 7],
            xpos[i * 16 + 8],
            xpos[i * 16 + 9],
            xpos[i * 16 + 10],
            xpos[i * 16 + 11],
            xpos[i * 16 + 12],
            xpos[i * 16 + 13],
            xpos[i * 16 + 14],
            xpos[i * 16 + 15],
        )
        * ee_pose_inv
    )
    r_ = wp.mat33f(
        ee_pose[0, 0],
        ee_pose[0, 1],
        ee_pose[0, 2],
        ee_pose[1, 0],
        ee_pose[1, 1],
        ee_pose[1, 2],
        ee_pose[2, 0],
        ee_pose[2, 1],
        ee_pose[2, 2],
    )
    rz_ = wp.vec3f(ee_pose[0, 2], ee_pose[1, 2], ee_pose[2, 2])
    t_ = wp.vec3f(ee_pose[0, 3], ee_pose[1, 3], ee_pose[2, 3])

    # to wrist center position
    c = t_ - params.c4 * rz_

    r_xy2 = c[0] * c[0] + c[1] * c[1]
    nx1_sqrt_arg = r_xy2 - params.b * params.b
    nx1 = wp.sqrt(nx1_sqrt_arg) - params.a1

    tmp1 = wp.atan2(c[1], c[0])
    tmp2 = wp.atan2(params.b, nx1 + params.a1)
    theta1_i = tmp1 - tmp2
    theta1_ii = tmp1 + tmp2 - wp.pi

    tmp3 = c[2] - params.c1
    s1_2 = nx1 * nx1 + tmp3 * tmp3

    tmp4 = nx1 + 2.0 * params.a1
    s2_2 = tmp4 * tmp4 + tmp3 * tmp3
    kappa_2 = params.a2 * params.a2 + params.c3 * params.c3

    c2_2 = params.c2 * params.c2

    tmp5 = s1_2 + c2_2 - kappa_2
    s1 = wp.sqrt(s1_2)
    s2 = wp.sqrt(s2_2)

    # theta2
    tmp13 = safe_acos(tmp5 / (2.0 * s1 * params.c2))
    tmp14 = wp.atan2(nx1, c[2] - params.c1)
    theta2_i = -tmp13 + tmp14
    theta2_ii = tmp13 + tmp14

    tmp6 = s2_2 + c2_2 - kappa_2
    tmp15 = safe_acos(tmp6 / (2.0 * s2 * params.c2))
    tmp16 = wp.atan2(nx1 + 2.0 * params.a1, c[2] - params.c1)
    theta2_iii = -tmp15 - tmp16
    theta2_iv = tmp15 - tmp16

    # theta3
    tmp7 = s1_2 - c2_2 - kappa_2
    tmp8 = s2_2 - c2_2 - kappa_2
    tmp9 = 2.0 * params.c2 * wp.sqrt(kappa_2)
    tmp10 = wp.atan2(params.a2, params.c3)

    tmp11 = safe_acos(tmp7 / tmp9)
    theta3_i = tmp11 - tmp10
    theta3_ii = -tmp11 - tmp10

    tmp12 = safe_acos(tmp8 / tmp9)
    theta3_iii = tmp12 - tmp10
    theta3_iv = -tmp12 - tmp10

    # precompute sin/cos(theta1)
    theta1_i_sin = wp.sin(theta1_i)
    theta1_i_cos = wp.cos(theta1_i)
    theta1_ii_sin = wp.sin(theta1_ii)
    theta1_ii_cos = wp.cos(theta1_ii)

    sin1 = wp.vec4f(theta1_i_sin, theta1_i_sin, theta1_ii_sin, theta1_ii_sin)
    cos1 = wp.vec4f(theta1_i_cos, theta1_i_cos, theta1_ii_cos, theta1_ii_cos)
    s23 = wp.vec4f(
        wp.sin(theta2_i + theta3_i),
        wp.sin(theta2_ii + theta3_ii),
        wp.sin(theta2_iii + theta3_iii),
        wp.sin(theta2_iv + theta3_iv),
    )
    c23 = wp.vec4f(
        wp.cos(theta2_i + theta3_i),
        wp.cos(theta2_ii + theta3_ii),
        wp.cos(theta2_iii + theta3_iii),
        wp.cos(theta2_iv + theta3_iv),
    )

    # m for theta5
    m = wp.vec4f(
        r_[0, 2] * s23[0] * cos1[0] + r_[1, 2] * s23[0] * sin1[0] + r_[2, 2] * c23[0],
        r_[0, 2] * s23[1] * cos1[1] + r_[1, 2] * s23[1] * sin1[1] + r_[2, 2] * c23[1],
        r_[0, 2] * s23[2] * cos1[2] + r_[1, 2] * s23[2] * sin1[2] + r_[2, 2] * c23[2],
        r_[0, 2] * s23[3] * cos1[3] + r_[1, 2] * s23[3] * sin1[3] + r_[2, 2] * c23[3],
    )
    theta5 = wp.vec4f(
        wp.atan2(wp.sqrt(wp.clamp(1.0 - m[0] * m[0], 0.0, 1.0)), m[0]),
        wp.atan2(wp.sqrt(wp.clamp(1.0 - m[1] * m[1], 0.0, 1.0)), m[1]),
        wp.atan2(wp.sqrt(wp.clamp(1.0 - m[2] * m[2], 0.0, 1.0)), m[2]),
        wp.atan2(wp.sqrt(wp.clamp(1.0 - m[3] * m[3], 0.0, 1.0)), m[3]),
    )

    theta4_i, theta6_i = th4_th6_for_branch(0, r_, sin1, cos1, s23, c23)
    theta4_ii, theta6_ii = th4_th6_for_branch(1, r_, sin1, cos1, s23, c23)
    theta4_iii, theta6_iii = th4_th6_for_branch(2, r_, sin1, cos1, s23, c23)
    theta4_iv, theta6_iv = th4_th6_for_branch(3, r_, sin1, cos1, s23, c23)
    theta5_i, theta5_ii, theta5_iii, theta5_iv = (
        theta5[0],
        theta5[1],
        theta5[2],
        theta5[3],
    )
    theta5_v, theta5_vi, theta5_vii, theta5_viii = (
        -theta5_i,
        -theta5_ii,
        -theta5_iii,
        -theta5_iv,
    )

    theta4_v, theta4_vi, theta4_vii, theta4_viii = (
        theta4_i + wp.pi,
        theta4_ii + wp.pi,
        theta4_iii + wp.pi,
        theta4_iv + wp.pi,
    )
    theta6_v, theta6_vi, theta6_vii, theta6_viii = (
        theta6_i - wp.pi,
        theta6_ii - wp.pi,
        theta6_iii - wp.pi,
        theta6_iv - wp.pi,
    )
    # combine all 8 solutions
    theta = wp_vec48f(
        theta1_i,
        theta2_i,
        theta3_i,
        theta4_i,
        theta5_i,
        theta6_i,
        theta1_i,
        theta2_ii,
        theta3_ii,
        theta4_ii,
        theta5_ii,
        theta6_ii,
        theta1_ii,
        theta2_iii,
        theta3_iii,
        theta4_iii,
        theta5_iii,
        theta6_iii,
        theta1_ii,
        theta2_iv,
        theta3_iv,
        theta4_iv,
        theta5_iv,
        theta6_iv,
        theta1_i,
        theta2_i,
        theta3_i,
        theta4_v,
        theta5_v,
        theta6_v,
        theta1_i,
        theta2_ii,
        theta3_ii,
        theta4_vi,
        theta5_vi,
        theta6_vi,
        theta1_ii,
        theta2_iii,
        theta3_iii,
        theta4_vii,
        theta5_vii,
        theta6_vii,
        theta1_ii,
        theta2_iv,
        theta3_iv,
        theta4_viii,
        theta5_viii,
        theta6_viii,
    )
    DOF = 6
    N_SOL = 8
    # apply sign correction and offsets, and write to qpos
    for j in range(N_SOL):
        qpos_start = i * DOF * N_SOL + j * DOF

        for k in range(DOF):
            idx = j * DOF + k
            qpos[qpos_start + k] = normalize_to_pi(
                (theta[idx] + offsets[k]) * sign_corrections[k]
            )

        # filter invalid solutions
        check_ee_pose = opw_single_fk(
            theta[j * DOF + 0],
            theta[j * DOF + 1],
            theta[j * DOF + 2],
            theta[j * DOF + 3],
            theta[j * DOF + 4],
            theta[j * DOF + 5],
            params,
        )
        t_err, r_err = get_transform_err(check_ee_pose, ee_pose)
        # mark invalid solutions (cannot pass ik check)
        if t_err > 1e-2 or r_err > 1e-1:
            ik_valid[i * N_SOL + j] = 0
        else:
            ik_valid[i * N_SOL + j] = 1


@wp.kernel
def opw_best_ik_kernel(
    full_ik_result: wp.array(dtype=float),
    full_ik_valid: wp.array(dtype=int),
    qpos_seed: wp.array(dtype=float),
    joint_weights: wp_vec6f,
    best_ik_result: wp.array(dtype=float),
    best_ik_valid: wp.array(dtype=int),
):
    i = wp.tid()
    DOF = 6
    N_SOL = 8

    best_weighted_dis = float(1e10)
    best_ids = int(-1)
    for j in range(N_SOL):
        is_full_valid = full_ik_valid[i * N_SOL + j]
        if is_full_valid == 0:
            # invalid ik result
            continue
        weighted_dis = 0.0
        for t in range(DOF):
            weighted_dis += (
                (full_ik_result[i * N_SOL * DOF + j * DOF + t] - qpos_seed[i * DOF + t])
                * joint_weights[0]
                * (
                    full_ik_result[i * N_SOL * DOF + j * DOF + t]
                    - qpos_seed[i * DOF + t]
                )
                * joint_weights[0]
            )
        if weighted_dis < best_weighted_dis:
            best_weighted_dis = weighted_dis
            best_ids = j
    if best_ids != -1:
        # found best solution
        best_ik_valid[i] = 1
        for k in range(DOF):
            best_ik_result[i * DOF + k] = full_ik_result[
                i * N_SOL * DOF + best_ids * DOF + k
            ]
    else:
        # no valid solution
        best_ik_valid[i] = 0
