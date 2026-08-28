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
    return wp.acos(wp.clamp(x, -1.0, 1.0))


@wp.func
def wrap_to_limit(value: float, lower: float, upper: float, seed: float) -> wp.vec2:
    """Return ``(success, equivalent_value)`` nearest the seed within limits."""
    two_pi = 2.0 * wp.pi
    k_min = wp.ceil((lower - value) / two_pi)
    k_max = wp.floor((upper - value) / two_pi)
    if k_min > k_max:
        return wp.vec2(0.0, value)
    nearest_k = wp.floor((seed - value) / two_pi + 0.5)
    nearest_k = wp.clamp(nearest_k, k_min, k_max)
    return wp.vec2(1.0, value + nearest_k * two_pi)


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
    joints_val[0] = wp.atan2(y, x) if horizontal_distance > 1e-6 else 0.0

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

    # Compute the reference shoulder-to-elbow pose in the base frame.
    base_to_elbow_pose = identity_mat44()
    for i in range(3):
        base_idx = i * 4
        T = dh_transform(
            dh_params[base_idx + 0],
            dh_params[base_idx + 1],
            dh_params[base_idx + 2],
            joints[tid][i],
        )
        base_to_elbow_pose = base_to_elbow_pose @ T

    reference_elbow = wp.vec3(
        base_to_elbow_pose[0, 3],
        base_to_elbow_pose[1, 3],
        base_to_elbow_pose[2, 3],
    )
    reference_upper = reference_elbow - P02
    shoulder_to_wrist = P06 - P02
    upper_norm = wp.length(reference_upper)
    wrist_norm = wp.length(shoulder_to_wrist)
    if upper_norm < 1e-10 or wrist_norm < 1e-10:
        res[tid] = 0
        plane_normal[tid] = wp.vec3()
        base_to_elbow_rotation[tid] = identity_mat33()
        return

    normal = wp.cross(reference_upper / upper_norm, shoulder_to_wrist / wrist_norm)
    normal_norm = wp.length(normal)
    if normal_norm < 1e-10:
        res[tid] = 0
        plane_normal[tid] = wp.vec3()
        base_to_elbow_rotation[tid] = identity_mat33()
        return

    plane_normal[tid] = normal / normal_norm
    base_to_elbow_rotation[tid] = wp.mat33(
        base_to_elbow_pose[0, 0],
        base_to_elbow_pose[0, 1],
        base_to_elbow_pose[0, 2],
        base_to_elbow_pose[1, 0],
        base_to_elbow_pose[1, 1],
        base_to_elbow_pose[1, 2],
        base_to_elbow_pose[2, 0],
        base_to_elbow_pose[2, 1],
        base_to_elbow_pose[2, 2],
    )

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
        T = dh_transform(d, alpha, a, theta)
        pose = pose @ T

    # Apply additional transforms: base, end-effector, TCP
    pose = T_b_ob @ pose @ T_oe_e @ tcp_transform

    # Output pose and set success flag
    pose_out[tid] = pose
    success[tid] = 1


@wp.kernel
def compute_arm_angle_kernel(
    qpos: wp.array(dtype=float),
    dh_params: wp.array(dtype=float),
    link_lengths: wp.array(dtype=float),
    rotation_directions: wp.array(dtype=float),
    arm_angles: wp.array(dtype=float),
    success: wp.array(dtype=int),
):
    """Compute the geometric SRS arm angle for each joint configuration."""
    tid = wp.tid()
    actual_pose = identity_mat44()
    actual_elbow = wp.vec3()

    for i in range(7):
        base_idx = i * 4
        theta = dh_params[base_idx + 3] + qpos[tid * 7 + i] * rotation_directions[i]
        actual_pose = actual_pose @ dh_transform(
            dh_params[base_idx + 0],
            dh_params[base_idx + 1],
            dh_params[base_idx + 2],
            theta,
        )
        if i == 2:
            actual_elbow = wp.vec3(
                actual_pose[0, 3], actual_pose[1, 3], actual_pose[2, 3]
            )

    target_position = wp.vec3(actual_pose[0, 3], actual_pose[1, 3], actual_pose[2, 3])
    target_rotation = wp.mat33(
        actual_pose[0, 0],
        actual_pose[0, 1],
        actual_pose[0, 2],
        actual_pose[1, 0],
        actual_pose[1, 1],
        actual_pose[1, 2],
        actual_pose[2, 0],
        actual_pose[2, 1],
        actual_pose[2, 2],
    )
    shoulder = wp.vec3(0.0, 0.0, link_lengths[0])
    wrist_offset = wp.vec3(0.0, 0.0, dh_params[6 * 4 + 0])
    wrist = target_position - target_rotation @ wrist_offset
    shoulder_to_wrist = wrist - shoulder
    distance = wp.length(shoulder_to_wrist)
    if distance < 1e-10:
        arm_angles[tid] = 0.0
        success[tid] = 0
        return

    elbow_model = dh_params[3 * 4 + 3] + qpos[tid * 7 + 3] * rotation_directions[3]
    elbow_config = -1.0 if elbow_model < 0.0 else 1.0
    shoulder_cosine = (
        wp.pow(link_lengths[1], 2.0)
        + wp.pow(distance, 2.0)
        - wp.pow(link_lengths[2], 2.0)
    ) / (2.0 * link_lengths[1] * distance)
    horizontal_distance = wp.length(wp.vec2(shoulder_to_wrist[0], shoulder_to_wrist[1]))
    q1_reference = (
        wp.atan2(shoulder_to_wrist[1], shoulder_to_wrist[0])
        if horizontal_distance > 1e-6
        else 0.0
    )
    q2_reference = wp.atan2(
        horizontal_distance,
        shoulder_to_wrist[2],
    ) + elbow_config * safe_acos(shoulder_cosine)

    reference_pose = identity_mat44()
    for i in range(3):
        base_idx = i * 4
        theta = 0.0
        if i == 0:
            theta = q1_reference
        elif i == 1:
            theta = q2_reference
        reference_pose = reference_pose @ dh_transform(
            dh_params[base_idx + 0],
            dh_params[base_idx + 1],
            dh_params[base_idx + 2],
            theta,
        )

    reference_elbow = wp.vec3(
        reference_pose[0, 3], reference_pose[1, 3], reference_pose[2, 3]
    )
    axis = shoulder_to_wrist / distance
    reference_upper = reference_elbow - shoulder
    actual_upper = actual_elbow - shoulder
    reference_radial = reference_upper - axis * wp.dot(reference_upper, axis)
    actual_radial = actual_upper - axis * wp.dot(actual_upper, axis)
    reference_norm = wp.length(reference_radial)
    actual_norm = wp.length(actual_radial)
    if reference_norm < 1e-10 or actual_norm < 1e-10:
        arm_angles[tid] = 0.0
        success[tid] = 0
        return

    reference_radial = reference_radial / reference_norm
    actual_radial = actual_radial / actual_norm
    arm_angles[tid] = wp.atan2(
        wp.dot(axis, wp.cross(reference_radial, actual_radial)),
        wp.dot(reference_radial, actual_radial),
    )
    success[tid] = 1


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

    # Match NumPy's element-wise ``allclose(..., rtol=0)`` semantics used by
    # the CPU backend. A Frobenius threshold would become stricter as the
    # number of matrix elements grows.
    for row in range(4):
        for column in range(4):
            if wp.abs(pose[row, column] - target_xpos[row, column]) > tolerance:
                return 0
    return 1


# TODO: automatic gradient support
@wp.kernel
def compute_ik_kernel(
    target_xpos_list: wp.array(dtype=wp.mat44),
    angles_list: wp.array(dtype=float),
    qpos_seed: wp.array(dtype=float),
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
    num_configs: int,
    num_angles: int,
    success: wp.array(dtype=int),
    qpos_out: wp.array(dtype=float),
):
    """
    Compute inverse kinematics (IK) in parallel for multiple target poses.

    Args:
        target_xpos_list (wp.array): Array of target poses (4x4 transformation matrices).
        angles_list (wp.array): Array of reference angles for IK computation.
        qpos_seed (wp.array): Seed joint positions used for periodic limit wrapping.
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
        num_configs (int): Number of shoulder/elbow/wrist configurations.
        num_angles (int): Number of redundancy-angle samples per target.
        success (wp.array): Output array indicating whether IK computation was successful.
        qpos_out (wp.array): Output array for computed joint positions.

    Notes:
        This kernel computes the inverse kinematics for a batch of target poses in parallel.
        It validates the computed joint positions against joint limits and the target pose.
        Successful solutions are stored in the output arrays.
    """
    tid = wp.tid()  # Thread ID (for batch processing, if needed)

    # Extract indices
    angle_idx = tid % num_angles
    config_idx = (tid // num_angles) % num_configs
    target_idx = tid // (num_angles * num_configs)

    # Load inputs
    target_xpos = target_xpos_list[target_idx]
    config = configs[config_idx]
    angle_ref = angles_list[target_idx * num_angles + angle_idx]

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
    P67 = wp.vec3(0.0, 0.0, dh_params[6 * 4 + 0])
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

    q2 = safe_acos(R03[2, 1]) * shoulder_config
    q1 = float(0.0)
    q3 = float(0.0)
    if wp.abs(wp.sin(q2)) <= 1e-6:
        q1 = qpos_seed[target_idx * 7] * rotation_directions[0] + dh_params[3]
        q3 = wp.atan2(R03[1, 0], R03[0, 0]) - q1
    else:
        q1 = wp.atan2(R03[1, 1] * shoulder_config, R03[0, 1] * shoulder_config)
        q3 = wp.atan2(-R03[2, 2] * shoulder_config, -R03[2, 0] * shoulder_config)

    # Calculate wrist joint angles (q5, q6, q7)
    Aw = wp.transpose(R34) @ wp.transpose(As) @ R_target
    Bw = wp.transpose(R34) @ wp.transpose(Bs) @ R_target
    Cw = wp.transpose(R34) @ wp.transpose(Cs) @ R_target
    R47 = Aw * s_psi + Bw * c_psi + Cw

    q4 = joints_v[3]
    q6 = safe_acos(R47[2, 2]) * wrist_config
    q5 = float(0.0)
    q7 = float(0.0)
    if wp.abs(wp.sin(q6)) <= 1e-6:
        q5 = qpos_seed[target_idx * 7 + 4] * rotation_directions[4] + dh_params[19]
        q7 = wp.atan2(-R47[2, 0], R47[0, 0]) - q5
    else:
        q5 = wp.atan2(R47[1, 2] * wrist_config, R47[0, 2] * wrist_config)
        q7 = wp.atan2(R47[2, 1] * wrist_config, -R47[2, 0] * wrist_config)

    out_of_limits = int(0)

    q1_val = (q1 - dh_params[3]) * rotation_directions[0]
    q2_val = (q2 - dh_params[7]) * rotation_directions[1]
    q3_val = (q3 - dh_params[11]) * rotation_directions[2]
    q4_val = (q4 - dh_params[15]) * rotation_directions[3]
    q5_val = (q5 - dh_params[19]) * rotation_directions[4]
    q6_val = (q6 - dh_params[23]) * rotation_directions[5]
    q7_val = (q7 - dh_params[27]) * rotation_directions[6]

    wrapped_q1 = wrap_to_limit(
        q1_val, qpos_limits[0][0], qpos_limits[0][1], qpos_seed[target_idx * 7]
    )
    wrapped_q2 = wrap_to_limit(
        q2_val, qpos_limits[1][0], qpos_limits[1][1], qpos_seed[target_idx * 7 + 1]
    )
    wrapped_q3 = wrap_to_limit(
        q3_val, qpos_limits[2][0], qpos_limits[2][1], qpos_seed[target_idx * 7 + 2]
    )
    wrapped_q4 = wrap_to_limit(
        q4_val, qpos_limits[3][0], qpos_limits[3][1], qpos_seed[target_idx * 7 + 3]
    )
    wrapped_q5 = wrap_to_limit(
        q5_val, qpos_limits[4][0], qpos_limits[4][1], qpos_seed[target_idx * 7 + 4]
    )
    wrapped_q6 = wrap_to_limit(
        q6_val, qpos_limits[5][0], qpos_limits[5][1], qpos_seed[target_idx * 7 + 5]
    )
    wrapped_q7 = wrap_to_limit(
        q7_val, qpos_limits[6][0], qpos_limits[6][1], qpos_seed[target_idx * 7 + 6]
    )
    q1_val = wrapped_q1[1]
    q2_val = wrapped_q2[1]
    q3_val = wrapped_q3[1]
    q4_val = wrapped_q4[1]
    q5_val = wrapped_q5[1]
    q6_val = wrapped_q6[1]
    q7_val = wrapped_q7[1]

    out_of_limits = int(
        wrapped_q1[0]
        * wrapped_q2[0]
        * wrapped_q3[0]
        * wrapped_q4[0]
        * wrapped_q5[0]
        * wrapped_q6[0]
        * wrapped_q7[0]
        < 0.5
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
                raw_diff = qpos_out[idx * 7 + j] - qpos_seed[tid * 7 + j]
                diff = wp.atan2(wp.sin(raw_diff), wp.cos(raw_diff))
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
