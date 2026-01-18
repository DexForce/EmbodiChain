# Step 1: Grasp the bottle
drive(
    right_arm_action=grasp(
        robot_name='right_arm',
        obj_name='bottle',
        pre_grasp_dis=0.10
    ),
    left_arm_action=None
)

# Step 2: Move the bottle to the pouring position relative to the cup
drive(
    right_arm_action=move_relative_to_object(
        robot_name='right_arm',
        obj_name='cup',
        x_offset=0.05,
        y_offset=-0.10,
        z_offset=0.125
    ),
    left_arm_action=None
)

# Step 3: Pour water into the cup
drive(
    right_arm_action=rotate_eef(
        robot_name='right_arm',
        degree=-90
    ),
    left_arm_action=None
)

# Step 4: Return the bottle to its upright position
drive(
    right_arm_action=rotate_eef(
        robot_name='right_arm',
        degree=90
    ),
    left_arm_action=None
)

# Step 5: Place the bottle at the specified location
drive(
    right_arm_action=place_on_table(
        robot_name='right_arm',
        obj_name='bottle',
        x=0.7,
        y=-0.1,
        pre_place_dis=0.08
    ),
    left_arm_action=None
)

# Step 6: Return the right arm to its initial pose
drive(
    right_arm_action=back_to_initial_pose(
        robot_name='right_arm'
    ),
    left_arm_action=None
)