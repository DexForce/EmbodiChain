# Step 1: Grasp the fork with the left arm and the spoon with the right arm
drive(
    left_arm_action=grasp(robot_name="left_arm", obj_name="fork", pre_grasp_dis=0.10),
    right_arm_action=grasp(
        robot_name="right_arm", obj_name="spoon", pre_grasp_dis=0.10
    ),
)

# Step 2: Reorient both end-effectors to a downward-facing pose
drive(
    left_arm_action=orient_eef(robot_name="left_arm", direction="down"),
    right_arm_action=orient_eef(robot_name="right_arm", direction="down"),
)

# Step 3: Place the fork at y = +0.16 and the spoon at y = −0.16 relative to the plate’s center
drive(
    left_arm_action=place_on_table(
        robot_name="left_arm", obj_name="fork", x=0.0, y=0.16, pre_place_dis=0.08
    ),
    right_arm_action=place_on_table(
        robot_name="right_arm", obj_name="spoon", x=0.0, y=-0.16, pre_place_dis=0.08
    ),
)

# Step 4: Return both arms to their initial poses
drive(
    left_arm_action=back_to_initial_pose(robot_name="left_arm"),
    right_arm_action=back_to_initial_pose(robot_name="right_arm"),
)
