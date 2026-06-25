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

from __future__ import annotations

from typing import Tuple

import warp as wp

wp_vec6f = wp.types.vector(length=6, dtype=float)
wp_vec48f = wp.types.vector(length=48, dtype=float)


@wp.func
def normalize_to_pi(angle: float) -> float:
    # TODO: Cannot work in warp.
    # return (angle + wp.pi) % (2.0 * wp.pi) - wp.pi
    return wp.atan2(wp.sin(angle), wp.cos(angle))


@wp.struct
class URParam:
    d1: float
    a2: float
    a3: float
    d4: float
    d5: float
    d6: float


@wp.func
def _safe_sqrt_ur(x: float) -> float:
    return wp.sqrt(wp.max(x, float(0.0)))


@wp.func
def _ur_dh(theta: float, d: float, a: float, alpha: float) -> wp.mat44f:
    """Compute a standard Denavit-Hartenberg transformation matrix."""
    sa = wp.sin(alpha)
    ca = wp.cos(alpha)
    st = wp.sin(theta)
    ct = wp.cos(theta)
    return wp.mat44f(
        ct,
        -st * ca,
        st * sa,
        a * ct,
        st,
        ct * ca,
        -ct * sa,
        a * st,
        float(0.0),
        sa,
        ca,
        d,
        float(0.0),
        float(0.0),
        float(0.0),
        float(1.0),
    )


@wp.func
def ur_single_fk(
    q1: float,
    q2: float,
    q3: float,
    q4: float,
    q5: float,
    q6: float,
    params: URParam,
) -> wp.mat44f:
    """Compute UR robot forward kinematics via sequential DH matrix products.

    The DH parameter convention follows the standard UR kinematic description:
    - Joint 1: (theta1, d1,  0,   pi/2)
    - Joint 2: (theta2,  0,  a2,  0   )
    - Joint 3: (theta3,  0,  a3,  0   )
    - Joint 4: (theta4, d4,  0,   pi/2)
    - Joint 5: (theta5, d5,  0,  -pi/2)
    - Joint 6: (theta6, d6,  0,   0   )
    """
    half_pi = wp.pi * float(0.5)
    T01 = _ur_dh(q1, params.d1, float(0.0), half_pi)
    T12 = _ur_dh(q2, float(0.0), params.a2, float(0.0))
    T23 = _ur_dh(q3, float(0.0), params.a3, float(0.0))
    T34 = _ur_dh(q4, params.d4, float(0.0), half_pi)
    T45 = _ur_dh(q5, params.d5, float(0.0), -half_pi)
    T56 = _ur_dh(q6, params.d6, float(0.0), float(0.0))
    T02 = T01 * T12
    T03 = T02 * T23
    T04 = T03 * T34
    T05 = T04 * T45
    return T05 * T56


@wp.func
def _ur_transform_err(t1: wp.mat44f, t2: wp.mat44f) -> Tuple[float, float]:
    """Return (translation_error, rotation_error) between two homogeneous transforms."""
    dx = t1[0, 3] - t2[0, 3]
    dy = t1[1, 3] - t2[1, 3]
    dz = t1[2, 3] - t2[2, 3]
    t_err = wp.sqrt(dx * dx + dy * dy + dz * dz)

    r1 = wp.mat33f(
        t1[0, 0],
        t1[0, 1],
        t1[0, 2],
        t1[1, 0],
        t1[1, 1],
        t1[1, 2],
        t1[2, 0],
        t1[2, 1],
        t1[2, 2],
    )
    r2 = wp.mat33f(
        t2[0, 0],
        t2[0, 1],
        t2[0, 2],
        t2[1, 0],
        t2[1, 1],
        t2[1, 2],
        t2[2, 0],
        t2[2, 1],
        t2[2, 2],
    )
    r_diff = wp.transpose(r1) * r2
    cos_val = wp.clamp(
        float(0.5) * (wp.trace(r_diff) - float(1.0)), float(-1.0), float(1.0)
    )
    r_err = wp.abs(wp.acos(cos_val))
    return t_err, r_err


@wp.func
def _ur_solve_234(
    c1: float,
    s1: float,
    theta5: float,
    theta6: float,
    r11: float,
    r21: float,
    r31: float,
    px: float,
    py: float,
    pz: float,
    d1: float,
    a2: float,
    a3: float,
    d5: float,
    d6: float,
) -> Tuple[float, float, float, float, float, float]:
    """Compute joint angles 2/3/4 given joint angles 1/5/6.

    Returns:
        (theta2a, theta3a, theta4a, theta2b, theta3b, theta4b)
        where a/b correspond to the two elbow-up/down branches from theta3.
    """
    c5 = wp.cos(theta5)
    s5 = wp.sin(theta5)
    c6 = wp.cos(theta6)
    s6 = wp.sin(theta6)

    A234 = c1 * r11 + s1 * r21
    H1_234 = c5 * c6 * r31 - s6 * A234
    H2_234 = c5 * c6 * A234 + s6 * r31
    theta234 = wp.atan2(H1_234, H2_234)
    c234 = wp.cos(theta234)
    s234 = wp.sin(theta234)

    KC = c1 * px + s1 * py - s234 * d5 + c234 * s5 * d6
    KS = pz - d1 + c234 * d5 + s234 * s5 * d6

    H3_cos = (KS * KS + KC * KC - a2 * a2 - a3 * a3) / (float(2.0) * a2 * a3)
    H3_sin = _safe_sqrt_ur(float(1.0) - H3_cos * H3_cos)

    theta3a = wp.atan2(H3_sin, H3_cos)
    theta3b = -theta3a

    base_t2 = wp.atan2(KS, KC)
    theta2a = base_t2 - wp.atan2(a3 * wp.sin(theta3a), a3 * wp.cos(theta3a) + a2)
    theta2b = base_t2 - wp.atan2(a3 * wp.sin(theta3b), a3 * wp.cos(theta3b) + a2)

    theta4a = theta234 - theta2a - theta3a
    theta4b = theta234 - theta2b - theta3b

    return theta2a, theta3a, theta4a, theta2b, theta3b, theta4b


@wp.func
def _shift_to_limit(q: float, lo: float, hi: float) -> float:
    """Return an FK-equivalent value of ``q`` shifted by +/- 2*pi inside [lo, hi].

    UR joints are 2*pi-periodic, so ``q`` and ``q +/- 2*pi`` yield identical
    forward kinematics. When a 2*pi-shifted representative falls inside the
    joint limits it is returned; otherwise ``q`` itself is returned as a
    repeated fallback (the validity check later flags out-of-limit values).

    Args:
        q: Joint angle normalized to [-pi, pi].
        lo: Lower joint limit.
        hi: Upper joint limit.

    Returns:
        A 2*pi-shifted copy of ``q`` inside [lo, hi] when one exists, else ``q``.
    """
    two_pi = 2.0 * wp.pi
    q_plus = q + two_pi
    if q_plus >= lo and q_plus <= hi:
        return q_plus
    q_minus = q - two_pi
    if q_minus >= lo and q_minus <= hi:
        return q_minus
    return q


@wp.kernel
def ur_ik_kernel(
    xpos: wp.array(dtype=float),  # [n_sample * 16]  row-major 4x4 target poses
    params: URParam,
    lower_qpos_limits_wp: wp.array(dtype=float),  # [6]  lower joint limits
    upper_qpos_limits_wp: wp.array(dtype=float),  # [6]  upper joint limits
    qpos: wp.array(dtype=float),  # [n_sample * 512 * DOF]  output joint solutions
    ik_valid: wp.array(dtype=int),  # [n_sample * 512]         output validity flags
):
    """Compute expanded analytical IK solutions for a batch of UR poses.

    Each thread handles one target pose. The 8 base analytical solutions are
    expanded to ``8 * 2**6 = 512`` candidates: for every base solution and every
    joint, a second FK-equivalent value shifted by +/- 2*pi is generated when it
    falls inside the joint limits (UR joints are 2*pi-periodic, so this preserves
    the end-effector pose). When no shifted value fits the limits, the joint's own
    value is repeated. Each candidate is flagged valid only if the base FK matches
    the target *and* every joint lies within its limits.
    """
    i = wp.tid()
    DOF = int(6)
    N_SOL = int(8)
    N_SHIFT = int(64)  # 2**6 per-joint +/- 2*pi shift combinations
    base = i * 16

    # Load rotation and translation from the row-major 4x4 target pose.
    r11 = xpos[base + 0]
    r12 = xpos[base + 1]
    r13 = xpos[base + 2]
    px = xpos[base + 3]
    r21 = xpos[base + 4]
    r22 = xpos[base + 5]
    r23 = xpos[base + 6]
    py = xpos[base + 7]
    r31 = xpos[base + 8]
    r32 = xpos[base + 9]
    r33 = xpos[base + 10]
    pz = xpos[base + 11]

    # Reconstruct target pose matrix for FK error check.
    target_pose = wp.mat44f(
        r11,
        r12,
        r13,
        px,
        r21,
        r22,
        r23,
        py,
        r31,
        r32,
        r33,
        pz,
        float(0.0),
        float(0.0),
        float(0.0),
        float(1.0),
    )

    d1 = params.d1
    a2 = params.a2
    a3 = params.a3
    d4 = params.d4
    d5 = params.d5
    d6 = params.d6

    # ---- theta1: two solutions ----
    A1 = px - d6 * r13
    B1 = d6 * r23 - py
    H1_sq = A1 * A1 + B1 * B1 - d4 * d4
    H2 = wp.atan2(_safe_sqrt_ur(H1_sq), d4)
    base_ang1 = wp.atan2(A1, B1)
    theta1a = base_ang1 + H2
    theta1b = base_ang1 - H2

    # ---- theta5 & theta6 for theta1a ----
    c1a = wp.cos(theta1a)
    s1a = wp.sin(theta1a)
    c5_val_a = s1a * r13 - c1a * r23
    s5_sq_a = (s1a * r11 - c1a * r21) * (s1a * r11 - c1a * r21) + (
        s1a * r12 - c1a * r22
    ) * (s1a * r12 - c1a * r22)
    s5p_a = _safe_sqrt_ur(s5_sq_a)
    t5_pos_a = wp.atan2(s5p_a, c5_val_a)  # theta5 branch with positive s5
    t5_neg_a = wp.atan2(-s5p_a, c5_val_a)  # theta5 branch with negative s5
    H61_a = c1a * r22 - s1a * r12
    H62_a = s1a * r11 - c1a * r21
    sin5p_a = wp.sin(t5_pos_a)
    sin5n_a = wp.sin(t5_neg_a)
    sgn5p_a = float(1.0)
    sgn5n_a = float(-1.0)
    if wp.abs(sin5p_a) > float(1e-12):
        if sin5p_a < float(0.0):
            sgn5p_a = float(-1.0)
        if sin5n_a < float(0.0):
            sgn5n_a = float(-1.0)
        else:
            sgn5n_a = float(1.0)
    t6_5pa = wp.atan2(sgn5p_a * H61_a, sgn5p_a * H62_a)  # theta6 for (1a, 5pos)
    t6_5na = wp.atan2(sgn5n_a * H61_a, sgn5n_a * H62_a)  # theta6 for (1a, 5neg)

    # ---- theta5 & theta6 for theta1b ----
    c1b = wp.cos(theta1b)
    s1b = wp.sin(theta1b)
    c5_val_b = s1b * r13 - c1b * r23
    s5_sq_b = (s1b * r11 - c1b * r21) * (s1b * r11 - c1b * r21) + (
        s1b * r12 - c1b * r22
    ) * (s1b * r12 - c1b * r22)
    s5p_b = _safe_sqrt_ur(s5_sq_b)
    t5_pos_b = wp.atan2(s5p_b, c5_val_b)
    t5_neg_b = wp.atan2(-s5p_b, c5_val_b)
    H61_b = c1b * r22 - s1b * r12
    H62_b = s1b * r11 - c1b * r21
    sin5p_b = wp.sin(t5_pos_b)
    sin5n_b = wp.sin(t5_neg_b)
    sgn5p_b = float(1.0)
    sgn5n_b = float(-1.0)
    if wp.abs(sin5p_b) > float(1e-12):
        if sin5p_b < float(0.0):
            sgn5p_b = float(-1.0)
        if sin5n_b < float(0.0):
            sgn5n_b = float(-1.0)
        else:
            sgn5n_b = float(1.0)
    t6_5pb = wp.atan2(sgn5p_b * H61_b, sgn5p_b * H62_b)
    t6_5nb = wp.atan2(sgn5n_b * H61_b, sgn5n_b * H62_b)

    # ---- theta2/3/4 for each (theta1, theta5, theta6) group ----
    # Group 0: theta1a + positive theta5 -> solutions 0, 1
    t2a_g0, t3a_g0, t4a_g0, t2b_g0, t3b_g0, t4b_g0 = _ur_solve_234(
        c1a, s1a, t5_pos_a, t6_5pa, r11, r21, r31, px, py, pz, d1, a2, a3, d5, d6
    )
    # Group 1: theta1a + negative theta5 -> solutions 2, 3
    t2a_g1, t3a_g1, t4a_g1, t2b_g1, t3b_g1, t4b_g1 = _ur_solve_234(
        c1a, s1a, t5_neg_a, t6_5na, r11, r21, r31, px, py, pz, d1, a2, a3, d5, d6
    )
    # Group 2: theta1b + positive theta5 -> solutions 4, 5
    t2a_g2, t3a_g2, t4a_g2, t2b_g2, t3b_g2, t4b_g2 = _ur_solve_234(
        c1b, s1b, t5_pos_b, t6_5pb, r11, r21, r31, px, py, pz, d1, a2, a3, d5, d6
    )
    # Group 3: theta1b + negative theta5 -> solutions 6, 7
    t2a_g3, t3a_g3, t4a_g3, t2b_g3, t3b_g3, t4b_g3 = _ur_solve_234(
        c1b, s1b, t5_neg_b, t6_5nb, r11, r21, r31, px, py, pz, d1, a2, a3, d5, d6
    )

    # Assemble all 8 solutions as a flat 48-element vector [t1,t2,t3,t4,t5,t6, ...]
    theta = wp_vec48f(
        theta1a,
        t2a_g0,
        t3a_g0,
        t4a_g0,
        t5_pos_a,
        t6_5pa,  # sol 0
        theta1a,
        t2b_g0,
        t3b_g0,
        t4b_g0,
        t5_pos_a,
        t6_5pa,  # sol 1
        theta1a,
        t2a_g1,
        t3a_g1,
        t4a_g1,
        t5_neg_a,
        t6_5na,  # sol 2
        theta1a,
        t2b_g1,
        t3b_g1,
        t4b_g1,
        t5_neg_a,
        t6_5na,  # sol 3
        theta1b,
        t2a_g2,
        t3a_g2,
        t4a_g2,
        t5_pos_b,
        t6_5pb,  # sol 4
        theta1b,
        t2b_g2,
        t3b_g2,
        t4b_g2,
        t5_pos_b,
        t6_5pb,  # sol 5
        theta1b,
        t2a_g3,
        t3a_g3,
        t4a_g3,
        t5_neg_b,
        t6_5nb,  # sol 6
        theta1b,
        t2b_g3,
        t3b_g3,
        t4b_g3,
        t5_neg_b,
        t6_5nb,  # sol 7
    )

    # Expand each of the 8 base solutions into 2**6 = 64 per-joint +/- 2*pi shift
    # variants, yielding 8 * 64 = 512 candidates total. Shifting is FK-equivalent
    # and only applied when it lands inside the joint limits; otherwise the joint's
    # own value is repeated. Validity requires both FK match and joint-limit fit.
    for j in range(N_SOL):
        q1 = normalize_to_pi(theta[j * DOF + 0])
        q2 = normalize_to_pi(theta[j * DOF + 1])
        q3 = normalize_to_pi(theta[j * DOF + 2])
        q4 = normalize_to_pi(theta[j * DOF + 3])
        q5 = normalize_to_pi(theta[j * DOF + 4])
        q6 = normalize_to_pi(theta[j * DOF + 5])

        # Precompute the per-joint shifted representative inside [lo, hi].
        q1_shift = _shift_to_limit(q1, lower_qpos_limits_wp[0], upper_qpos_limits_wp[0])
        q2_shift = _shift_to_limit(q2, lower_qpos_limits_wp[1], upper_qpos_limits_wp[1])
        q3_shift = _shift_to_limit(q3, lower_qpos_limits_wp[2], upper_qpos_limits_wp[2])
        q4_shift = _shift_to_limit(q4, lower_qpos_limits_wp[3], upper_qpos_limits_wp[3])
        q5_shift = _shift_to_limit(q5, lower_qpos_limits_wp[4], upper_qpos_limits_wp[4])
        q6_shift = _shift_to_limit(q6, lower_qpos_limits_wp[5], upper_qpos_limits_wp[5])

        fk_result = ur_single_fk(q1, q2, q3, q4, q5, q6, params)
        t_err, r_err = _ur_transform_err(fk_result, target_pose)
        fk_ok = int(1)
        if t_err > float(1e-2) or r_err > float(1e-1):
            fk_ok = int(0)

        for k in range(N_SHIFT):
            out_start = i * DOF * N_SOL * N_SHIFT + (j * N_SHIFT + k) * DOF
            oq1 = q1 if (k & 1) == 0 else q1_shift
            oq2 = q2 if (k & 2) == 0 else q2_shift
            oq3 = q3 if (k & 4) == 0 else q3_shift
            oq4 = q4 if (k & 8) == 0 else q4_shift
            oq5 = q5 if (k & 16) == 0 else q5_shift
            oq6 = q6 if (k & 32) == 0 else q6_shift
            qpos[out_start + 0] = oq1
            qpos[out_start + 1] = oq2
            qpos[out_start + 2] = oq3
            qpos[out_start + 3] = oq4
            qpos[out_start + 4] = oq5
            qpos[out_start + 5] = oq6

            valid = fk_ok
            tol = float(1e-9)
            if (
                oq1 < lower_qpos_limits_wp[0] - tol
                or oq1 > upper_qpos_limits_wp[0] + tol
            ):
                valid = int(0)
            if (
                oq2 < lower_qpos_limits_wp[1] - tol
                or oq2 > upper_qpos_limits_wp[1] + tol
            ):
                valid = int(0)
            if (
                oq3 < lower_qpos_limits_wp[2] - tol
                or oq3 > upper_qpos_limits_wp[2] + tol
            ):
                valid = int(0)
            if (
                oq4 < lower_qpos_limits_wp[3] - tol
                or oq4 > upper_qpos_limits_wp[3] + tol
            ):
                valid = int(0)
            if (
                oq5 < lower_qpos_limits_wp[4] - tol
                or oq5 > upper_qpos_limits_wp[4] + tol
            ):
                valid = int(0)
            if (
                oq6 < lower_qpos_limits_wp[5] - tol
                or oq6 > upper_qpos_limits_wp[5] + tol
            ):
                valid = int(0)
            ik_valid[i * N_SOL * N_SHIFT + j * N_SHIFT + k] = valid
