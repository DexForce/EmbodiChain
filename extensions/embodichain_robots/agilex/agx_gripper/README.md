# AgileX AGX two-finger gripper

Resolve the standalone URDF as `AgxGripper/agx_gripper.urdf` and mount its
`gripper_base` link. The `gripper` prismatic joint commands a jaw opening of
0–0.1 m. The source `gripper_joint1` and `gripper_joint2` mimic it at +0.5
and −0.5 respectively.

The asset is pinned to the MIT-licensed
[AgileX standalone description](https://github.com/agilexrobotics/agx_arm_urdf/tree/680fa925cd00db7e2628563d2edc5540963045de/agx_gripper)
and retains the DAE visual meshes, STL collision meshes and source inertials.
The portable URDF rewrites ROS package references to relative paths. Tool
mounting transforms and drive settings belong to the robot embodiment.
