# Step 1 — Grasp the Fork and Spoon Simultaneously
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

# Step 2 — Reorient End-Effectors to Downward-Facing Pose
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

# Step 3 — Place the Fork and Spoon on Opposite Sides of the Plate
drive(
    left_arm_action=move_relative_to_object(
        robot_name='left_arm',
        obj_name='plate',
        x_offset=0,
        y_offset=0.16,
        z_offset=0
    ),
    right_arm_action=move_relative_to_object(
        robot_name='right_arm',
        obj_name='plate',
        x_offset=0,
        y_offset=-0.16,
        z_offset=0
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

# Step 4 — Return Both Arms to Initial Pose
drive(
    left_arm_action=back_to_initial_pose(
        robot_name='left_arm'
    ),
    right_arm_action=back_to_initial_pose(
        robot_name='right_arm'
    )
)