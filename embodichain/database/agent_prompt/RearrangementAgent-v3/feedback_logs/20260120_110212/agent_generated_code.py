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

# Step 2 — Reorient Both End-Effectors to a Downward-Facing Pose
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

# Step 3 — Place the Fork and Spoon at Specified Positions
drive(
    left_arm_action=place_on_table(
        robot_name='left_arm',
        obj_name='fork',
        x=None,
        y=0.16,
        pre_place_dis=0.08
    ),
    right_arm_action=place_on_table(
        robot_name='right_arm',
        obj_name='spoon',
        x=None,
        y=-0.16,
        pre_place_dis=0.08
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