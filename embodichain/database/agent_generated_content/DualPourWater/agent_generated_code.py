from functools import partial
from embodichain.lab.sim.agent.atom_actions import *

# Step 1 — Grasp the bottle with the right arm and the cup with the left arm simultaneously
drive(
    left_arm_action=grasp(
        robot_name="left_arm",
        obj_name="cup",
        pre_grasp_dis=0.10,
    ),
    right_arm_action=grasp(
        robot_name="right_arm",
        obj_name="bottle",
        pre_grasp_dis=0.10,
    ),
    # error_functions=[
    #     partial(
    #         inject_object_error,
    #         error_type="misplaced_object",
    #         error_obj="cup",
    #     ),
    #     partial(
    #         inject_object_error,
    #         error_type="misplaced_object",
    #         error_obj="bottle",
    #     ),
    #     partial(
    #         inject_action_error,
    #         error_type="wrong_affordance",
    #         error_arm="left_arm",
    #     ),
    #     partial(
    #         inject_action_error,
    #         error_type="wrong_affordance",
    #         error_arm="right_arm",
    #     ),
    # ],
    # monitor_sequences=[
    #     [
    #         partial(
    #             monitor_object_moved,
    #             obj_name="cup",
    #             threshold=0.02,
    #         )
    #     ],
    #     [
    #         partial(
    #             monitor_object_moved,
    #             obj_name="bottle",
    #             threshold=0.02,
    #         )
    #     ],
    # ],
    # recovery_sequences=[
    #     [
    #         partial(
    #             drive,
    #             left_arm_action=partial(
    #                 grasp,
    #                 robot_name="left_arm",
    #                 obj_name="cup",
    #                 pre_grasp_dis=0.10,
    #             ),
    #             right_arm_action=None,
    #         ),
    #     ],
    #     [
    #         partial(
    #             drive,
    #             left_arm_action=None,
    #             right_arm_action=partial(
    #                 grasp,
    #                 robot_name="right_arm",
    #                 obj_name="bottle",
    #                 pre_grasp_dis=0.10,
    #             ),
    #         ),
    #     ],
    # ],
)

# Step 2 — Lift the cup by 0.10 m to prepare for pouring
drive(
    left_arm_action=move_by_relative_offset(
        robot_name="left_arm",
        dx=0.0,
        dy=0.0,
        dz=0.10,
        mode="extrinsic",
    ),
    right_arm_action=None,
    # error_functions=[
    #     partial(
    #         inject_action_error,
    #         error_type="wrong_affordance",
    #         error_arm="left_arm",
    #     ),
    # ],
    # monitor_sequences=[
    #     [
    #         partial(
    #             monitor_object_held,
    #             robot_name="left_arm",
    #             obj_name="cup",
    #             threshold=0.05,
    #         )
    #     ],
    # ],
    # recovery_sequences=[
    #     [
    #         partial(
    #             drive,
    #             left_arm_action=partial(
    #                 grasp,
    #                 robot_name="left_arm",
    #                 obj_name="cup",
    #                 pre_grasp_dis=0.10,
    #             ),
    #             right_arm_action=None,
    #         ),
    #         partial(
    #             drive,
    #             left_arm_action=partial(
    #                 move_by_relative_offset,
    #                 robot_name="left_arm",
    #                 dx=0.0,
    #                 dy=0.0,
    #                 dz=0.10,
    #                 mode="extrinsic",
    #             ),
    #             right_arm_action=None,
    #         ),
    #     ],
    # ],
)

# Step 3 — Move the cup to the pouring position at [0.55, 0.05]
drive(
    left_arm_action=move_to_absolute_position(
        robot_name="left_arm",
        x=0.55,
        y=0.05,
        z=None,
    ),
    right_arm_action=None,
    error_functions=[
        partial(
            inject_object_error,
            error_type="misplaced_object",
            error_obj="cup",
        ),
    ],
    monitor_sequences=[
        [
            partial(
                monitor_object_held,
                robot_name="left_arm",
                obj_name="cup",
                threshold=0.05,
            )
        ],
    ],
    recovery_sequences=[
        [
            partial(
                drive,
                left_arm_action=partial(
                    grasp,
                    robot_name="left_arm",
                    obj_name="cup",
                    pre_grasp_dis=0.10,
                ),
                right_arm_action=None,
            ),
            partial(
                drive,
                left_arm_action=partial(
                    move_by_relative_offset,
                    robot_name="left_arm",
                    dx=0.0,
                    dy=0.0,
                    dz=0.10,
                    mode="extrinsic",
                ),
                right_arm_action=None,
            ),
            partial(
                drive,
                left_arm_action=partial(
                    move_to_absolute_position,
                    robot_name="left_arm",
                    x=0.55,
                    y=0.05,
                    z=None,
                ),
                right_arm_action=None,
            ),
        ],
    ],
)

# Step 4 — Position the bottle relative to the cup for pouring
drive(
    left_arm_action=None,
    right_arm_action=move_relative_to_object(
        robot_name="right_arm",
        obj_name="cup",
        x_offset=0.05,
        y_offset=-0.10,
        z_offset=0.125,
    ),
    # error_functions=[
    #     partial(
    #         inject_action_error,
    #         error_type="wrong_affordance",
    #         error_arm="right_arm",
    #     ),
    # ],
    # monitor_sequences=[
    #     [
    #         partial(
    #             monitor_object_held,
    #             robot_name="right_arm",
    #             obj_name="bottle",
    #             threshold=0.05,
    #         )
    #     ],
    # ],
    # recovery_sequences=[
    #     [
    #         partial(
    #             drive,
    #             left_arm_action=None,
    #             right_arm_action=partial(
    #                 grasp,
    #                 robot_name="right_arm",
    #                 obj_name="bottle",
    #                 pre_grasp_dis=0.10,
    #             ),
    #         ),
    #         partial(
    #             drive,
    #             left_arm_action=None,
    #             right_arm_action=partial(
    #                 move_relative_to_object,
    #                 robot_name="right_arm",
    #                 obj_name="cup",
    #                 x_offset=0.05,
    #                 y_offset=-0.10,
    #                 z_offset=0.125,
    #             ),
    #         ),
    #     ],
    # ],
)

# Step 5 — Tilt the bottle to pour water into the cup
drive(
    left_arm_action=None,
    right_arm_action=rotate_eef(
        robot_name="right_arm",
        degree=-45,
    ),
    error_functions=[
        partial(
            inject_action_error,
            error_type="wrong_affordance",
            error_arm="right_arm",
        ),
    ],
    monitor_sequences=[
        [
            partial(
                monitor_object_held,
                robot_name="right_arm",
                obj_name="bottle",
                threshold=0.05,
            )
        ],
    ],
    recovery_sequences=[
        [
            partial(
                drive,
                left_arm_action=None,
                right_arm_action=partial(
                    grasp,
                    robot_name="right_arm",
                    obj_name="bottle",
                    pre_grasp_dis=0.10,
                ),
            ),
            partial(
                drive,
                left_arm_action=None,
                right_arm_action=partial(
                    move_relative_to_object,
                    robot_name="right_arm",
                    obj_name="cup",
                    x_offset=0.05,
                    y_offset=-0.10,
                    z_offset=0.125,
                ),
            ),
            partial(
                drive,
                left_arm_action=None,
                right_arm_action=partial(
                    rotate_eef,
                    robot_name="right_arm",
                    degree=-45,
                ),
            ),
        ],
    ],
)

# Step 6 — Return the bottle to its upright position after pouring
drive(
    left_arm_action=None,
    right_arm_action=rotate_eef(
        robot_name="right_arm",
        degree=45,
    ),
    error_functions=[
        partial(
            inject_action_error,
            error_type="wrong_affordance",
            error_arm="right_arm",
        ),
    ],
    monitor_sequences=[
        [
            partial(
                monitor_object_held,
                robot_name="right_arm",
                obj_name="bottle",
                threshold=0.05,
            )
        ],
    ],
    recovery_sequences=[
        [
            partial(
                drive,
                left_arm_action=None,
                right_arm_action=partial(
                    grasp,
                    robot_name="right_arm",
                    obj_name="bottle",
                    pre_grasp_dis=0.10,
                ),
            ),
            partial(
                drive,
                left_arm_action=None,
                right_arm_action=partial(
                    rotate_eef,
                    robot_name="right_arm",
                    degree=45,
                ),
            ),
        ],
    ],
)

# Step 7 — Place the bottle at [0.7, −0.1] and the cup at [0.6, 0.1] simultaneously
drive(
    left_arm_action=place_on_table(
        robot_name="left_arm",
        obj_name="cup",
        x=0.6,
        y=0.1,
        pre_place_dis=0.08,
    ),
    right_arm_action=place_on_table(
        robot_name="right_arm",
        obj_name="bottle",
        x=0.7,
        y=-0.1,
        pre_place_dis=0.08,
    ),
    error_functions=[
        partial(
            inject_action_error,
            error_type="wrong_affordance",
            error_arm="left_arm",
        ),
        partial(
            inject_action_error,
            error_type="wrong_affordance",
            error_arm="right_arm",
        ),
    ],
    monitor_sequences=[
        [
            partial(
                monitor_object_held,
                robot_name="left_arm",
                obj_name="cup",
                threshold=0.05,
            )
        ],
        [
            partial(
                monitor_object_held,
                robot_name="right_arm",
                obj_name="bottle",
                threshold=0.05,
            )
        ],
    ],
    recovery_sequences=[
        [
            partial(
                drive,
                left_arm_action=partial(
                    grasp,
                    robot_name="left_arm",
                    obj_name="cup",
                    pre_grasp_dis=0.10,
                ),
                right_arm_action=None,
            ),
            partial(
                drive,
                left_arm_action=partial(
                    place_on_table,
                    robot_name="left_arm",
                    obj_name="cup",
                    x=0.6,
                    y=0.1,
                    pre_place_dis=0.08,
                ),
                right_arm_action=None,
            ),
        ],
        [
            partial(
                drive,
                left_arm_action=None,
                right_arm_action=partial(
                    grasp,
                    robot_name="right_arm",
                    obj_name="bottle",
                    pre_grasp_dis=0.10,
                ),
            ),
            partial(
                drive,
                left_arm_action=None,
                right_arm_action=partial(
                    place_on_table,
                    robot_name="right_arm",
                    obj_name="bottle",
                    x=0.7,
                    y=-0.1,
                    pre_place_dis=0.08,
                ),
            ),
        ],
    ],
)

# Step 8 — Return both arms to their initial poses
drive(
    left_arm_action=back_to_initial_pose(
        robot_name="left_arm",
    ),
    right_arm_action=back_to_initial_pose(
        robot_name="right_arm",
    ),
)