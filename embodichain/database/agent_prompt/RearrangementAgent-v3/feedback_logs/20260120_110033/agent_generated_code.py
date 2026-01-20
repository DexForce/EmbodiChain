# Step 1 — Grasp the fork and spoon
drive(
    left_arm_action=grasp(
        robot_name='left_arm',
        obj_name='fork',
        pre_grasp_dis=0.10
    ),
    right_arm_action=grasp(
        robot_name='right_arm',
        obj_name='spoon',
        pre_grasp_dis=0.10
    )
)

# Step 2 — Reorient end-effectors to downward-facing pose
drive(
    left_arm_action=orient_eef(
        robot_name='left_arm',
        direction='down'
    ),
    right_arm_action=orient_eef(
        robot_name='right_arm',
        direction='down'
    )
)

# Step 3 — Place fork and spoon on opposite sides of the plate
drive(
    left_arm_action=move_relative_to_object(
        robot_name='left_arm',
        obj_name='plate',
        y_offset=0.16
    ),
    right_arm_action=move_relative_to_object(
        robot_name='right_arm',
        obj_name='plate',
        y_offset=-0.16
    )
)

drive(
    left_arm_action=open_gripper(
        robot_name='left_arm'
    ),
    right_arm_action=open_gripper(
        robot_name='right_arm'
    )
)

# Step 4 — Return arms to initial pose
drive(
    left_arm_action=back_to_initial_pose(
        robot_name='left_arm'
    ),
    right_arm_action=back_to_initial_pose(
        robot_name='right_arm'
    )
)