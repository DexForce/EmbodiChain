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


@wp.func
def identity_mat44() -> wp.mat44:
    # fmt: off
    return wp.mat44(
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0
    )
    # fmt: on


@wp.func
def identity_mat33() -> wp.mat33:
    # fmt: off
    return wp.mat33(
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0
    )
    # fmt: on


@wp.func
def safe_acos(x: float) -> float:
    return wp.acos(wp.clamp(x, -0.999999, 0.999999))


@wp.func
def safe_division(numerator: float, denominator: float, eps: float = 1e-10) -> float:
    if wp.abs(denominator) < eps:
        return 0.0
    return numerator / denominator


@wp.func
def skew(vec: wp.vec3) -> wp.mat33:
    """
    Calculate the skew-symmetric matrix of a vector.

    Args:
        vec (wp.vec3): Input vector.

    Returns:
        wp.mat33: Skew-symmetric matrix.
    """
    # fmt: off
    return wp.mat33(
        0.0, -vec[2], vec[1],
        vec[2], 0.0, -vec[0],
        -vec[1], vec[0], 0.0,
    )
    # fmt: on


@wp.func
def dh_transform(d: float, alpha: float, a: float, theta: float) -> wp.mat44:
    """
    Compute the Denavit-Hartenberg transformation matrix.

    Args:
        d (float): Link offset.
        alpha (float): Link twist.
        a (float): Link length.
        theta (float): Joint angle.

    Returns:
        wp.mat44: The resulting transformation matrix.
    """
    ct, st = wp.cos(theta), wp.sin(theta)
    ca, sa = wp.cos(alpha), wp.sin(alpha)
    # fmt: off
    return wp.mat44(
        ct,     -st * ca,  st * sa,    a * ct,
        st,     ct * ca,   -ct * sa,   a * st,
        0.0,    sa,        ca,         d,
        0.0,    0.0,       0.0,        1.0
    )
    # fmt: on


@wp.func
def transform_pose(
    target_xpos: wp.mat44,
    T_b_ob_inv: wp.mat44,
    T_e_oe_inv: wp.mat44,
    tcp_inv: wp.mat44,
) -> wp.mat44:
    """
    Transform the target pose to the TCP frame.
    Args:
        target_xpos (wp.mat44): The target pose matrix.
        T_b_ob_inv (wp.mat44): Inverse base-to-object transform.
        tcp_inv (wp.mat44): Inverse TCP transform.
        T_e_oe_inv (wp.mat44): Inverse end-effector transform.
    Returns:
        wp.mat44: Transformed pose in TCP frame.
    """
    return T_b_ob_inv @ target_xpos @ tcp_inv @ T_e_oe_inv


@wp.kernel
def transform_pose_kernel(
    target_xpos: wp.array(dtype=wp.mat44),
    T_b_ob_inv: wp.mat44,
    T_e_oe_inv: wp.mat44,
    tcp_inv: wp.mat44,
    output: wp.array(dtype=wp.mat44),
):
    """
    Transform a batch of target poses to the TCP frame.

    Args:
        target_xpos (wp.array): Batch of target pose matrices.
        T_b_ob_inv (wp.mat44): Inverse base-to-object transform.
        tcp_inv (wp.mat44): Inverse TCP transform.
        T_e_oe_inv (wp.mat44): Inverse end-effector transform.
        output (wp.array): Output array for transformed poses.
    """
    tid = wp.tid()
    output[tid] = T_b_ob_inv @ target_xpos[tid] @ tcp_inv @ T_e_oe_inv


@wp.func
def calculate_arm_joint_angles(
    P_s_to_w: wp.vec3,
    elbow_GC4: float,
    link_lengths: wp.array(dtype=float),
    res: wp.array(dtype=int),
    joints: wp.array(dtype=wp.vec4),
    tid: int,
):
    """
    Compute joint angles for a 3-DOF arm given the shoulder-to-wrist vector.

    Args:
        P_s_to_w (wp.vec3): Shoulder-to-wrist vector.
        elbow_GC4 (float): Elbow configuration, typically ±1.
        link_lengths (wp.array): [d_bs, d_se, d_ew] for each segment length.
        res (wp.array): Output success flag.
        joints (wp.array): Output joint angles.
        tid (int): Thread index.
    """
    d_bs = link_lengths[0]
    d_se = link_lengths[1]
    d_ew = link_lengths[2]

    # Extract components
    x, y, z = P_s_to_w.x, P_s_to_w.y, P_s_to_w.z
    horizontal_distance = wp.length(wp.vec2(x, y))
    shoulder_to_wrist_length = wp.length(P_s_to_w)

    # Initialize joint values
    joints_val = wp.vec4()

    # Check reachability
    if shoulder_to_wrist_length < wp.abs(d_bs + d_ew):
        res[tid] = 0
        joints[tid] = joints_val
        return

    # Compute elbow angle
    elbow_cos_angle = (
        wp.pow(shoulder_to_wrist_length, 2.0) - wp.pow(d_se, 2.0) - wp.pow(d_ew, 2.0)
    ) / (2.0 * d_se * d_ew)
    if wp.abs(elbow_cos_angle) > 1.0:
        res[tid] = 0
        joints[tid] = joints_val
        return

    joints_val[3] = elbow_GC4 * safe_acos(elbow_cos_angle)

    # Compute shoulder angle
    joints_val[0] = wp.atan2(y, x) if wp.abs(z) > 1e-6 else 0.0

    # Compute joint 2 angle
    angle_phi = safe_acos(
        (wp.pow(d_se, 2.0) + wp.pow(shoulder_to_wrist_length, 2.0) - wp.pow(d_ew, 2.0))
        / (2.0 * d_se * shoulder_to_wrist_length)
    )
    joints_val[1] = wp.atan2(horizontal_distance, z) + elbow_GC4 * angle_phi

    # Set success flag and output joint values
    res[tid] = 1
    joints[tid] = joints_val


@wp.func
def compute_reference_plane(
    pose: wp.mat44,
    elbow_GC4: float,
    link_lengths: wp.array(dtype=float),
    dh_params: wp.array(dtype=float),
    res: wp.array(dtype=int),
    plane_normal: wp.array(dtype=wp.vec3),
    base_to_elbow_rotation: wp.array(dtype=wp.mat33),
    joints: wp.array(dtype=wp.vec4),
    tid: int,
):
    """
    Compute the reference plane normal, base-to-elbow rotation, and joint angles.

    Args:
        pose (wp.mat44): Target pose matrix (4x4).
        elbow_GC4 (float): Elbow configuration, typically ±1.
        link_lengths (wp.array): Link lengths, at least [d_bs, d_se, d_ew, d_hand].
        dh_params (wp.array): DH parameters, shape [num_joints * 4].
        res (wp.array): Output success flag.
        plane_normal (wp.array): Output plane normal vector.
        base_to_elbow_rotation (wp.array): Output base-to-elbow rotation matrix.
        joints (wp.array): Output joint angles.
        tid (int): Thread index.
    """
    # Extract position and rotation
    P_target = wp.vec3(pose[0, 3], pose[1, 3], pose[2, 3])
    # fmt: off
    R_target = wp.mat33(
        pose[0, 0], pose[0, 1], pose[0, 2],
        pose[1, 0], pose[1, 1], pose[1, 2],
        pose[2, 0], pose[2, 1], pose[2, 2],
    )
    # fmt: on

    # Base to shoulder
    P02 = wp.vec3(0.0, 0.0, link_lengths[0])
    P67 = wp.vec3(0.0, 0.0, dh_params[6 * 4 + 0])

    # Wrist position
    P06 = P_target - R_target @ P67
    # Shoulder to wrist
    P26 = P06 - P02

    # Calculate joint angles
    calculate_arm_joint_angles(P26, elbow_GC4, link_lengths, res, joints, tid)
    if res[tid] == 0:
        plane_normal[tid] = wp.vec3()
        base_to_elbow_rotation[tid] = identity_mat33()
        joints[tid] = wp.vec4()
        return

    # Lower arm transformation (joint 4)
    T34 = dh_transform(
        dh_params[3 * 4 + 0], dh_params[3 * 4 + 1], dh_params[3 * 4 + 2], 0.0
    )
    P34 = wp.vec3(T34[0, 3], T34[1, 3], T34[2, 3])

    # Reference plane normal
    v1 = wp.normalize(P34 - P02)
    v2 = wp.normalize(P06 - P02)
    plane_normal[tid] = wp.cross(v1, v2)

    # Compute base-to-elbow rotation
    base_to_elbow_rotation[tid] = identity_mat33()
    for i in range(3):
        base_idx = i * 4
        T = dh_transform(
            dh_params[base_idx + 0],
            dh_params[base_idx + 1],
            dh_params[base_idx + 2],
            joints[tid][i],
        )
        # fmt: off
        base_to_elbow_rotation[tid] = base_to_elbow_rotation[tid] @ wp.mat33(
            T[0, 0], T[0, 1], T[0, 2],
            T[1, 0], T[1, 1], T[1, 2],
            T[2, 0], T[2, 1], T[2, 2],
        )
        # fmt: on

    res[tid] = 1


@wp.kernel
def compute_fk_kernel(
    joint_angles: wp.array(dtype=float),
    dh_params: wp.array(dtype=float),
    rotation_directions: wp.array(dtype=float),
    T_b_ob: wp.mat44,
    T_oe_e: wp.mat44,
    tcp_transform: wp.mat44,
    pose_out: wp.array(dtype=wp.mat44),
    success: wp.array(dtype=int),
):
    """
    Compute forward kinematics (FK) for a batch of joint states.

    Args:
        joint_angles (wp.array): Array of joint angles for each target ([N * num_joints]).
        dh_params (wp.array): Denavit-Hartenberg parameters for the robot
            ([num_joints * 4], where each joint has [d, alpha, a, theta]).
        rotation_directions (wp.array): Array of rotation direction multipliers for each joint ([num_joints]).
        T_b_ob (wp.mat44): Base-to-object transformation matrix.
        T_oe_e (wp.mat44): End-effector-to-object transformation matrix.
        tcp_transform (wp.mat44): Tool center point (TCP) transformation matrix.
        pose_out (wp.array): Output array for computed poses ([N, 4x4]).
        success (wp.array): Output array indicating whether FK computation was successful ([N]).
    """
    tid = wp.tid()
    num_joints = rotation_directions.shape[0]

    # Initialize pose as identity matrix
    pose = identity_mat44()

    # Loop through each joint and apply DH transformation
    for i in range(num_joints):
        base_idx = i * 4
        d = dh_params[base_idx + 0]
        alpha = dh_params[base_idx + 1]
        a = dh_params[base_idx + 2]
        theta = dh_params[base_idx + 3]
        theta += joint_angles[tid * num_joints + i] * rotation_directions[i]
        T = dh_transform(d, alpha, d, theta)
        pose = pose @ T

    # Apply additional transforms: base, end-effector, TCP
    pose = T_b_ob @ pose @ T_oe_e @ tcp_transform

    # Output pose and set success flag
    pose_out[tid] = pose
    success[tid] = 1


@wp.func
def frobenius_norm(mat: wp.mat44) -> float:
    """
    Compute the Frobenius norm of a 4x4 matrix.

    Args:
        mat (wp.mat44): Input matrix.

    Returns:
        float: Frobenius norm of the matrix.
    """
    norm = 0.0
    for i in range(4):
        for j in range(4):
            norm += wp.pow(mat[i, j], 2.0)
    return wp.sqrt(norm)


@wp.func
def validate_fk_with_target(
    q1: float,
    q2: float,
    q3: float,
    q4: float,
    q5: float,
    q6: float,
    q7: float,
    dh_params: wp.array(dtype=float),
    rotation_directions: wp.array(dtype=float),
    target_xpos: wp.mat44,
    tolerance: float,
) -> int:
    """
    Validate if the FK result matches the target pose within a given tolerance.

    Args:
        joint_angles (wp.array): Joint angles for FK computation.
        dh_params (wp.array): Denavit-Hartenberg parameters.
        rotation_directions (wp.array): Rotation direction multipliers for each joint.
        target_xpos (wp.mat44): Target pose matrix.
        tolerance (float): Allowed error tolerance for validation.

    Returns:
        int: 1 if FK result matches the target pose within tolerance, 0 otherwise.
    """
    num_joints = wp.int32(rotation_directions.shape[0])

    # Initialize pose as identity matrix
    pose = identity_mat44()

    # Compute FK
    for i in range(num_joints):
        d = dh_params[i * 4 + 0]
        alpha = dh_params[i * 4 + 1]
        a = dh_params[i * 4 + 2]
        theta = dh_params[i * 4 + 3]
        # Apply joint angle with rotation direction
        if i == 0:
            joint_angle = q1
        elif i == 1:
            joint_angle = q2
        elif i == 2:
            joint_angle = q3
        elif i == 3:
            joint_angle = q4
        elif i == 4:
            joint_angle = q5
        elif i == 5:
            joint_angle = q6
        elif i == 6:
            joint_angle = q7

        theta += joint_angle * rotation_directions[i]
        T = dh_transform(d, alpha, a, theta)
        pose = pose @ T

    # Compute the Frobenius norm of the difference
    pose_diff = pose - target_xpos
    pose_error = frobenius_norm(pose_diff)

    # Validate against tolerance
    return 1 if pose_error <= tolerance else 0


# TODO: automatic gradient support
@wp.kernel
def compute_ik_kernel(
    combinations: wp.array(dtype=wp.vec3),
    target_xpos_list: wp.array(dtype=wp.mat44),
    angles_list: wp.array(dtype=float),
    qpos_limits: wp.array(dtype=wp.vec2),
    configs: wp.array(dtype=wp.vec3),
    dh_params: wp.array(dtype=float),
    link_lengths: wp.array(dtype=float),
    rotation_directions: wp.array(dtype=float),
    res_arm_angles: wp.array(dtype=int),
    joints_arm: wp.array(dtype=wp.vec4),
    res_plane_normal: wp.array(dtype=int),
    plane_normal: wp.array(dtype=wp.vec3),
    base_to_elbow_rotation: wp.array(dtype=wp.mat33),
    joints_plane: wp.array(dtype=wp.vec4),
    success: wp.array(dtype=int),
    qpos_out: wp.array(dtype=float),
):
    """
    Compute inverse kinematics (IK) in parallel for multiple target poses.

    Args:
        combinations (wp.array): Array of combinations, where each entry specifies
            the indices of the target pose, configuration, and reference angle.
        target_xpos_list (wp.array): Array of target poses (4x4 transformation matrices).
        angles_list (wp.array): Array of reference angles for IK computation.
        qpos_limits (wp.array): Array of joint position limits (min, max) for each joint.
        configs (wp.array): Array of configuration vectors (shoulder, elbow, wrist).
        dh_params (wp.array): Denavit-Hartenberg parameters for the robot.
        link_lengths (wp.array): Array of link lengths for the robot arm.
        rotation_directions (wp.array): Array of rotation direction multipliers for each joint.
        res_arm_angles (wp.array): Output array for arm joint angle computation results.
        joints_arm (wp.array): Output array for computed arm joint angles.
        res_plane_normal (wp.array): Output array for plane normal computation results.
        plane_normal (wp.array): Output array for computed plane normal vectors.
        base_to_elbow_rotation (wp.array): Output array for base-to-elbow rotation matrices.
        joints_plane (wp.array): Output array for computed joint angles in the plane.
        success (wp.array): Output array indicating whether IK computation was successful.
        qpos_out (wp.array): Output array for computed joint positions.

    Notes:
        This kernel computes the inverse kinematics for a batch of target poses in parallel.
        It validates the computed joint positions against joint limits and the target pose.
        Successful solutions are stored in the output arrays.
    """
    tid = wp.tid()  # Thread ID (for batch processing, if needed)

    # Extract indices
    target_idx = int(combinations[tid][0])
    config_idx = int(combinations[tid][1])
    angle_idx = int(combinations[tid][2])

    # Load inputs
    target_xpos = target_xpos_list[target_idx]
    config = configs[config_idx]
    angle_ref = angles_list[angle_idx]

    # Extract shoulder, elbow, wrist configurations
    shoulder_config, elbow_config, wrist_config = config.x, config.y, config.z

    # Transform target pose (xpos_ = target_xpos @ tcp_inv @ T_e_oe_inv)
    # fmt: off
    P_target = wp.vec3(target_xpos[0, 3], target_xpos[1, 3], target_xpos[2, 3])
    R_target = wp.mat33(
        target_xpos[0, 0], target_xpos[0, 1], target_xpos[0, 2],
        target_xpos[1, 0], target_xpos[1, 1], target_xpos[1, 2],
        target_xpos[2, 0], target_xpos[2, 1], target_xpos[2, 2],
    )
    # fmt: on

    # Compute shoulder-to-wrist vector
    P02 = wp.vec3(0.0, 0.0, link_lengths[0])
    P67 = wp.vec3(0.0, 0.0, dh_params[12])
    P06 = P_target - R_target @ P67
    P26 = P06 - P02

    calculate_arm_joint_angles(
        P26, elbow_config, link_lengths, res_arm_angles, joints_arm, tid
    )
    if res_arm_angles[tid] == 0:
        success[tid] = 0
        return
    joints_v = joints_arm[tid]

    # fmt: off
    # Calculate transformations
    T34 = dh_transform(
        dh_params[12],
        dh_params[13],
        dh_params[14],
        joints_v[3],
    )
    R34 = wp.mat33(
        T34[0, 0], T34[0, 1], T34[0, 2],
        T34[1, 0], T34[1, 1], T34[1, 2],
        T34[2, 0], T34[2, 1], T34[2, 2],
    )
    # fmt: on

    # Calculate reference joint angles
    compute_reference_plane(
        target_xpos,
        elbow_config,
        link_lengths,
        dh_params,
        res_plane_normal,
        plane_normal,
        base_to_elbow_rotation,
        joints_plane,
        tid,
    )
    if res_plane_normal[tid] == 0:
        success[tid] = 0
        return

    R03_o = base_to_elbow_rotation[tid]

    usw = wp.normalize(P26)
    skew_usw = skew(usw)
    s_psi = wp.sin(angle_ref)
    c_psi = wp.cos(angle_ref)

    # Calculate shoulder joint angles (q1, q2, q3)
    As = skew_usw @ R03_o
    Bs = -skew_usw @ skew_usw @ R03_o
    Cs = wp.outer(usw, usw) @ R03_o
    R03 = (
        (skew_usw @ R03_o) * s_psi
        + (-skew_usw @ skew_usw @ R03_o) * c_psi
        + (wp.outer(usw, usw) @ R03_o)
    )

    # TODO: judgment shoulder singularity
    q1 = wp.atan2(R03[1, 1] * shoulder_config, R03[0, 1] * shoulder_config)
    q2 = safe_acos(R03[2, 1]) * shoulder_config
    q3 = wp.atan2(-R03[2, 2] * shoulder_config, -R03[2, 0] * shoulder_config)

    # Calculate wrist joint angles (q5, q6, q7)
    Aw = wp.transpose(R34) @ wp.transpose(As) @ R_target
    Bw = wp.transpose(R34) @ wp.transpose(Bs) @ R_target
    Cw = wp.transpose(R34) @ wp.transpose(Cs) @ R_target
    R47 = Aw * s_psi + Bw * c_psi + Cw

    q4 = joints_v[3]
    # TODO: judgment wrist singularity
    q5 = wp.atan2(R47[1, 2] * wrist_config, R47[0, 2] * wrist_config)
    q6 = safe_acos(R47[2, 2]) * wrist_config
    q7 = wp.atan2(R47[2, 1] * wrist_config, -R47[2, 0] * wrist_config)

    out_of_limits = int(0)

    q1_val = (q1 - dh_params[3]) * rotation_directions[0]
    q2_val = (q2 - dh_params[7]) * rotation_directions[1]
    q3_val = (q3 - dh_params[11]) * rotation_directions[2]
    q4_val = (q4 - dh_params[15]) * rotation_directions[3]
    q5_val = (q5 - dh_params[19]) * rotation_directions[4]
    q6_val = (q6 - dh_params[23]) * rotation_directions[5]
    q7_val = (q7 - dh_params[27]) * rotation_directions[6]

    out_of_limits = int(0)
    out_of_limits = out_of_limits | (
        1 if (q1_val < qpos_limits[0][0] or q1_val > qpos_limits[0][1]) else 0
    )
    out_of_limits = out_of_limits | (
        1 if (q2_val < qpos_limits[1][0] or q2_val > qpos_limits[1][1]) else 0
    )
    out_of_limits = out_of_limits | (
        1 if (q3_val < qpos_limits[2][0] or q3_val > qpos_limits[2][1]) else 0
    )
    out_of_limits = out_of_limits | (
        1 if (q4_val < qpos_limits[3][0] or q4_val > qpos_limits[3][1]) else 0
    )
    out_of_limits = out_of_limits | (
        1 if (q5_val < qpos_limits[4][0] or q5_val > qpos_limits[4][1]) else 0
    )
    out_of_limits = out_of_limits | (
        1 if (q6_val < qpos_limits[5][0] or q6_val > qpos_limits[5][1]) else 0
    )
    out_of_limits = out_of_limits | (
        1 if (q7_val < qpos_limits[6][0] or q7_val > qpos_limits[6][1]) else 0
    )

    # Check joint limits
    if out_of_limits == 1:
        success[tid] = 0
        return

    is_valid = validate_fk_with_target(
        q1=q1_val,
        q2=q2_val,
        q3=q3_val,
        q4=q4_val,
        q5=q5_val,
        q6=q6_val,
        q7=q7_val,
        dh_params=dh_params,
        rotation_directions=rotation_directions,
        target_xpos=target_xpos,
        tolerance=1e-4,
    )

    # Save joint angles only if valid
    if is_valid:
        qpos_out[tid * 7] = q1_val
        qpos_out[tid * 7 + 1] = q2_val
        qpos_out[tid * 7 + 2] = q3_val
        qpos_out[tid * 7 + 3] = q4_val
        qpos_out[tid * 7 + 4] = q5_val
        qpos_out[tid * 7 + 5] = q6_val
        qpos_out[tid * 7 + 6] = q7_val
        success[tid] = 1  # Mark as successful
    else:
        success[tid] = 0  # Mark as failed


@wp.kernel
def sort_ik_kernel(
    qpos_out: wp.array(dtype=float),  # [N * N_SOL, 7]
    success: wp.array(dtype=int),  # [N * N_SOL]
    qpos_seed: wp.array(dtype=float),  # [N, 7]
    ik_weight: wp.array(dtype=float),  # [7]
    distances: wp.array(dtype=float),  # [N, N_SOL]
    indices: wp.array(dtype=int),  # [N, N_SOL]
    N_SOL: int,
    sorted_qpos: wp.array(dtype=float),  # [N, N_SOL, 7]
    sorted_valid: wp.array(dtype=int),  # [N, N_SOL]
):
    """
    Sort inverse kinematics (IK) solutions for multiple targets based on their distances
    to a seed configuration.

    Args:
        qpos_out (wp.array): Array of computed joint positions for all solutions
            ([N * N_SOL, 7]).
        success (wp.array): Array indicating whether each solution is valid ([N * N_SOL]).
        qpos_seed (wp.array): Array of seed joint positions for each target ([N, 7]).
        ik_weight (wp.array): Array of weights for each joint to compute distance ([7]).
        distances (wp.array): Output array to store computed distances ([N, N_SOL]).
        indices (wp.array): Output array to store sorted indices ([N, N_SOL]).
        N_SOL (int): Number of solutions per target.
        sorted_qpos (wp.array): Output array for sorted joint positions ([N, N_SOL, 7]).
        sorted_valid (wp.array): Output array for sorted validity flags ([N, N_SOL]).
    """
    tid = wp.tid()  # target index

    # 1. compute distances
    for i in range(N_SOL):
        idx = tid * N_SOL + i
        valid = success[idx]
        dist = 0.0
        if valid:
            for j in range(7):
                diff = qpos_out[idx * 7 + j] - qpos_seed[tid * 7 + j]
                dist += ik_weight[j] * diff * diff
        else:
            dist = 1e10

        distances[idx] = dist
        indices[idx] = i

    # 2. bubble sort (only sort the N_SOL solutions for the current target)
    for i in range(N_SOL):
        min_idx = i
        for j in range(i + 1, N_SOL):
            idx_a = tid * N_SOL + min_idx
            idx_b = tid * N_SOL + j
            if distances[idx_b] < distances[idx_a]:
                min_idx = j
        # Swap
        if min_idx != i:
            idx_i = tid * N_SOL + i
            idx_min = tid * N_SOL + min_idx
            tmp_dist = distances[idx_i]
            distances[idx_i] = distances[idx_min]
            distances[idx_min] = tmp_dist
            tmp_idx = indices[idx_i]
            indices[idx_i] = indices[idx_min]
            indices[idx_min] = tmp_idx

    # 3. reorder qpos_out and success according to sorted indices
    for i in range(N_SOL):
        src_idx = tid * N_SOL + indices[tid * N_SOL + i]
        for j in range(7):
            sorted_qpos[(tid * N_SOL + i) * 7 + j] = qpos_out[src_idx * 7 + j]
        sorted_valid[tid * N_SOL + i] = success[src_idx]


@wp.kernel
def nearest_ik_kernel(
    qpos_out: wp.array(dtype=float),  # [N * N_SOL * 7]
    success: wp.array(dtype=int),  # [N * N_SOL]
    qpos_seed: wp.array(dtype=float),  # [N * 7]
    ik_weight: wp.array(dtype=float),  # [7]
    N_SOL: int,
    nearest_qpos: wp.array(dtype=float),  # [N * 7]
    nearest_valid: wp.array(dtype=int),  # [N]
):
    """
    Find the nearest valid inverse kinematics (IK) solution for each target.

    Args:
        qpos_out (wp.array): Array of computed joint positions for all solutions
            ([N * N_SOL, 7]).
        success (wp.array): Array indicating whether each solution is valid ([N * N_SOL]).
        qpos_seed (wp.array): Array of seed joint positions for each target ([N, 7]).
        ik_weight (wp.array): Array of weights for each joint to compute distance ([7]).
        N_SOL (int): Number of solutions per target.
        nearest_qpos (wp.array): Output array for the nearest joint positions ([N, 7]).
        nearest_valid (wp.array): Output array indicating whether a valid solution was found ([N]).
    """

    tid = wp.tid()  # target index

    min_dist = float(1e20)
    nearest_idx = int(-1)

    for i in range(N_SOL):
        idx = tid * N_SOL + i
        if success[idx]:
            dist = 0.0
            for j in range(7):
                diff = qpos_out[idx * 7 + j] - qpos_seed[tid * 7 + j]
                dist += ik_weight[j] * diff * diff
            if dist < min_dist:
                min_dist = dist
                nearest_idx = idx

    if nearest_idx >= 0:
        for j in range(7):
            nearest_qpos[tid * 7 + j] = qpos_out[nearest_idx * 7 + j]
        nearest_valid[tid] = 1
    else:
        for j in range(7):
            nearest_qpos[tid * 7 + j] = 0.0
        nearest_valid[tid] = 0


@wp.kernel
def check_success_kernel(
    success_wp: wp.array(dtype=int),
    num_solutions: int,
    success_counts: wp.array(dtype=int),
):
    """
    Count the number of successful inverse kinematics (IK) solutions for each target.

    Args:
        success_wp (wp.array): Array indicating whether each solution is valid
            ([N * num_solutions], where N is the number of targets).
        num_solutions (int): Number of solutions per target.
        success_counts (wp.array): Output array to store the count of valid solutions
            for each target ([N]).
    """
    tid = wp.tid()  # target index
    count = int(0)

    for i in range(num_solutions):
        idx = tid * num_solutions + i
        if success_wp[idx]:
            count += 1

    success_counts[tid] = count
